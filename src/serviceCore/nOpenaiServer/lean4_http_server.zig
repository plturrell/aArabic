const std = @import("std");

const json = std.json;
const mem = std.mem;
const net = std.net;
const Child = std.process.Child;

const default_host = "0.0.0.0";
const default_port: u16 = 8002;
const max_request_bytes: usize = 2 * 1024 * 1024;
const max_output_bytes_default: usize = 1024 * 1024;
const max_output_bytes_cap: usize = 8 * 1024 * 1024;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var log_enabled: ?bool = null;
var lean_version: []const u8 = "unknown";
var lean_version_storage: ?[]u8 = null;

var file_counter = std.atomic.Value(u64).init(0);

const LeanCheckRequest = struct {
    code: []const u8,
    filename: ?[]const u8 = null,
    max_output_bytes: ?usize = null,
    max_memory_mb: ?u32 = null,
    trust_level: ?u8 = null,
    root: ?[]const u8 = null,
};

const LeanRunRequest = struct {
    code: []const u8,
    filename: ?[]const u8 = null,
    args: ?[]const []const u8 = null,
    stdin: ?[]const u8 = null,
    max_output_bytes: ?usize = null,
    max_memory_mb: ?u32 = null,
    trust_level: ?u8 = null,
    root: ?[]const u8 = null,
};

const LeanDiagnostic = struct {
    severity: []const u8,
    message: []const u8,
    file: ?[]const u8 = null,
    line: ?u32 = null,
    column: ?u32 = null,
    end_line: ?u32 = null,
    end_column: ?u32 = null,
    kind: ?[]const u8 = null,
};

const LeanCheckResponse = struct {
    success: bool,
    exit_code: i32,
    elapsed_ms: i64,
    diagnostics: []const LeanDiagnostic,
    stdout: []const u8,
    stderr: []const u8,
    lean_version: []const u8,
};

const LeanRunResponse = struct {
    success: bool,
    exit_code: i32,
    elapsed_ms: i64,
    stdout: []const u8,
    stderr: []const u8,
    lean_version: []const u8,
};

const ErrorResponse = struct {
    @"error": []const u8,
};

const Response = struct {
    status: u16,
    body: []u8,
    content_type: []const u8 = "application/json",
};

const RunOutput = struct {
    stdout: []u8,
    stderr: []u8,
    exit_code: i32,
};

const LeanTempFile = struct {
    work_dir: []const u8,
    file_name: []u8,
    file_path: []u8,
};

fn logEnabled() bool {
    if (log_enabled) |enabled| {
        return enabled;
    }
    log_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
    return log_enabled.?;
}

fn log(comptime fmt: []const u8, args: anytype) void {
    if (logEnabled()) {
        std.debug.print(fmt, args);
    }
}

fn getenv(name: []const u8) ?[]const u8 {
    if (std.posix.getenv(name)) |value| {
        return value[0..value.len];
    }
    return null;
}

fn parseEnvU16(name: []const u8, default_value: u16) u16 {
    if (getenv(name)) |value| {
        return std.fmt.parseInt(u16, value, 10) catch default_value;
    }
    return default_value;
}

fn parseEnvUsize(name: []const u8, default_value: usize) usize {
    if (getenv(name)) |value| {
        return std.fmt.parseInt(usize, value, 10) catch default_value;
    }
    return default_value;
}

fn resolveLeanBin() []const u8 {
    return getenv("LEAN4_BIN") orelse getenv("LEAN_BIN") orelse "lean";
}

fn resolveLeanRoot() ?[]const u8 {
    return getenv("LEAN4_ROOT") orelse getenv("SHIMMY_LEAN4_ROOT");
}

fn resolveWorkDir() []const u8 {
    return getenv("SHIMMY_LEAN4_WORK_DIR") orelse "tmp/lean4";
}

fn resolveMaxOutputBytes(request_value: ?usize) usize {
    var limit = parseEnvUsize("SHIMMY_LEAN4_MAX_OUTPUT_BYTES", max_output_bytes_default);
    if (request_value) |value| {
        limit = value;
    }
    if (limit == 0) {
        limit = max_output_bytes_default;
    }
    if (limit > max_output_bytes_cap) {
        limit = max_output_bytes_cap;
    }
    return limit;
}

fn encodeJson(value: anytype) ![]u8 {
    return json.Stringify.valueAlloc(allocator, value, .{});
}

fn errorBody(message: []const u8) ![]u8 {
    return encodeJson(ErrorResponse{ .@"error" = message });
}

fn sendResponse(stream: net.Stream, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        413 => "Payload Too Large",
        500 => "Internal Server Error",
        else => "OK",
    };

    var header_buf: [512]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
            "Content-Type: {s}\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
            "Server: Shimmy-lean4/1.0 (Zig)\r\n" ++
            "\r\n",
        .{ status, reason, content_type, body.len },
    );

    _ = try stream.writeAll(header);
    if (body.len > 0) {
        _ = try stream.writeAll(body);
    }
}

fn parseContentLength(header: []const u8) ?usize {
    var lines = mem.splitSequence(u8, header, "\r\n");
    while (lines.next()) |line| {
        if (std.ascii.startsWithIgnoreCase(line, "Content-Length:")) {
            const value = mem.trim(u8, line["Content-Length:".len..], " \t");
            return std.fmt.parseInt(usize, value, 10) catch null;
        }
    }
    return null;
}

fn readRequest(stream: net.Stream) ![]u8 {
    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);

    var temp: [4096]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: usize = 0;

    while (true) {
        const n = try stream.read(&temp);
        if (n == 0) break;
        try buffer.appendSlice(allocator, temp[0..n]);
        if (buffer.items.len > max_request_bytes) {
            return error.RequestTooLarge;
        }

        if (header_end == null) {
            if (mem.indexOf(u8, buffer.items, "\r\n\r\n")) |idx| {
                header_end = idx + 4;
                const header = buffer.items[0..idx];
                content_length = parseContentLength(header) orelse 0;
            }
        }

        if (header_end != null and buffer.items.len >= header_end.? + content_length) {
            break;
        }
    }

    return buffer.toOwnedSlice(allocator);
}

fn sanitizeFilename(input: []const u8) []const u8 {
    if (input.len == 0) return "Main.lean";
    const base = std.fs.path.basename(input);
    if (base.len == 0) return "Main.lean";
    for (base) |c| {
        if (!(std.ascii.isAlphanumeric(c) or c == '-' or c == '_' or c == '.')) {
            return "Main.lean";
        }
    }
    return base;
}

fn buildFileName(base: []const u8, id: u64) ![]u8 {
    var stem = base;
    if (mem.endsWith(u8, base, ".lean")) {
        stem = base[0 .. base.len - 5];
    }
    if (stem.len == 0) {
        stem = "Main";
    }
    return std.fmt.allocPrint(allocator, "{s}-{d}.lean", .{ stem, id });
}

fn createLeanTempFile(code: []const u8, filename_hint: ?[]const u8) !LeanTempFile {
    const work_dir = resolveWorkDir();
    try std.fs.cwd().makePath(work_dir);

    const base = sanitizeFilename(filename_hint orelse "Main.lean");
    const id = file_counter.fetchAdd(1, .seq_cst);
    const file_name = try buildFileName(base, id);
    errdefer allocator.free(file_name);

    const file_path = try std.fs.path.join(allocator, &.{ work_dir, file_name });
    errdefer allocator.free(file_path);

    try std.fs.cwd().writeFile(.{ .sub_path = file_path, .data = code });

    return LeanTempFile{
        .work_dir = work_dir,
        .file_name = file_name,
        .file_path = file_path,
    };
}

fn cleanupLeanTempFile(temp: LeanTempFile) void {
    std.fs.cwd().deleteFile(temp.file_path) catch {};
    allocator.free(temp.file_name);
    allocator.free(temp.file_path);
}

fn termToExitCode(term: Child.Term) i32 {
    return switch (term) {
        .Exited => |code| @as(i32, @intCast(code)),
        .Signal => |sig| 128 + @as(i32, @intCast(sig)),
        .Stopped => |sig| 128 + @as(i32, @intCast(sig)),
        else => -1,
    };
}

fn runLean(argv: []const []const u8, stdin_payload: ?[]const u8, cwd: ?[]const u8, max_output_bytes: usize) !RunOutput {
    var child = Child.init(argv, allocator);
    child.stdin_behavior = if (stdin_payload != null) .Pipe else .Ignore;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;
    child.cwd = cwd;

    var stdout = std.ArrayList(u8).empty;
    defer stdout.deinit(allocator);
    var stderr = std.ArrayList(u8).empty;
    defer stderr.deinit(allocator);

    try child.spawn();
    errdefer {
        _ = child.kill() catch {};
    }

    if (stdin_payload) |input| {
        if (child.stdin) |stdin_stream| {
            try stdin_stream.writeAll(input);
            stdin_stream.close();
        }
    }

    try child.collectOutput(allocator, &stdout, &stderr, max_output_bytes);
    const term = try child.wait();

    return RunOutput{
        .stdout = try stdout.toOwnedSlice(allocator),
        .stderr = try stderr.toOwnedSlice(allocator),
        .exit_code = termToExitCode(term),
    };
}

fn getStringField(obj: json.ObjectMap, key: []const u8) ?[]const u8 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .string => |s| s,
            else => null,
        };
    }
    return null;
}

fn getU32Field(obj: json.ObjectMap, key: []const u8) ?u32 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |i| if (i >= 0) @as(u32, @intCast(i)) else null,
            else => null,
        };
    }
    return null;
}

fn parseDiagnostics(stdout: []const u8) ![]LeanDiagnostic {
    var list = std.ArrayList(LeanDiagnostic).empty;
    errdefer list.deinit(allocator);

    var lines = mem.splitSequence(u8, stdout, "\n");
    while (lines.next()) |line| {
        const trimmed = mem.trim(u8, line, " \r\t");
        if (trimmed.len == 0) continue;

        const parsed = json.parseFromSlice(json.Value, allocator, trimmed, .{}) catch continue;
        defer parsed.deinit();

        const obj = switch (parsed.value) {
            .object => |value| value,
            else => continue,
        };

        const severity = getStringField(obj, "severity") orelse continue;
        const message = getStringField(obj, "data") orelse getStringField(obj, "message") orelse "";
        const file_name = getStringField(obj, "fileName");
        const kind = getStringField(obj, "kind");

        var line_num: ?u32 = null;
        var column_num: ?u32 = null;
        var end_line: ?u32 = null;
        var end_column: ?u32 = null;

        if (obj.get("pos")) |pos_val| {
            if (pos_val == .object) {
                const pos_obj = pos_val.object;
                line_num = getU32Field(pos_obj, "line");
                column_num = getU32Field(pos_obj, "column");
            }
        }

        if (obj.get("endPos")) |pos_val| {
            if (pos_val == .object) {
                const pos_obj = pos_val.object;
                end_line = getU32Field(pos_obj, "line");
                end_column = getU32Field(pos_obj, "column");
            }
        }

        const diag = LeanDiagnostic{
            .severity = try allocator.dupe(u8, severity),
            .message = try allocator.dupe(u8, message),
            .file = if (file_name) |f| try allocator.dupe(u8, f) else null,
            .line = line_num,
            .column = column_num,
            .end_line = end_line,
            .end_column = end_column,
            .kind = if (kind) |k| try allocator.dupe(u8, k) else null,
        };

        try list.append(allocator, diag);
    }

    return list.toOwnedSlice(allocator);
}

fn freeDiagnostics(diagnostics: []LeanDiagnostic) void {
    for (diagnostics) |diag| {
        allocator.free(diag.severity);
        allocator.free(diag.message);
        if (diag.file) |file_name| {
            allocator.free(file_name);
        }
        if (diag.kind) |kind| {
            allocator.free(kind);
        }
    }
    allocator.free(diagnostics);
}

fn resolveRootOverride(request_root: ?[]const u8) ?[]const u8 {
    if (request_root) |root| {
        if (root.len > 0) {
            return root;
        }
    }
    return resolveLeanRoot();
}

fn handleCheck(body: []const u8) !Response {
    const parsed = json.parseFromSlice(LeanCheckRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        return Response{ .status = 400, .body = try errorBody("Invalid JSON payload") };
    };
    defer parsed.deinit();

    const req = parsed.value;
    if (req.code.len == 0) {
        return Response{ .status = 400, .body = try errorBody("code is required") };
    }

    const temp = try createLeanTempFile(req.code, req.filename);
    defer cleanupLeanTempFile(temp);

    var args = std.ArrayList([]const u8).empty;
    defer args.deinit(allocator);
    var owned_args = std.ArrayList([]u8).empty;
    defer {
        for (owned_args.items) |item| {
            allocator.free(item);
        }
        owned_args.deinit(allocator);
    }

    try args.append(allocator, resolveLeanBin());
    try args.append(allocator, "--json");

    if (req.trust_level) |trust| {
        try args.append(allocator, "-t");
        const trust_arg = try std.fmt.allocPrint(allocator, "{d}", .{trust});
        try owned_args.append(allocator, trust_arg);
        try args.append(allocator, trust_arg);
    }

    if (req.max_memory_mb) |mem_mb| {
        try args.append(allocator, "-M");
        const mem_arg = try std.fmt.allocPrint(allocator, "{d}", .{mem_mb});
        try owned_args.append(allocator, mem_arg);
        try args.append(allocator, mem_arg);
    }

    if (resolveRootOverride(req.root)) |root| {
        try args.append(allocator, "--root");
        try args.append(allocator, root);
    }

    try args.append(allocator, temp.file_name);

    const max_output = resolveMaxOutputBytes(req.max_output_bytes);
    const start = std.time.milliTimestamp();

    const result = runLean(args.items, null, temp.work_dir, max_output) catch {
        return Response{ .status = 500, .body = try errorBody("Lean execution failed") };
    };
    const elapsed = std.time.milliTimestamp() - start;

    const diagnostics = parseDiagnostics(result.stdout) catch {
        allocator.free(result.stdout);
        allocator.free(result.stderr);
        return Response{ .status = 500, .body = try errorBody("Failed to parse Lean output") };
    };
    defer freeDiagnostics(diagnostics);
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const response = LeanCheckResponse{
        .success = result.exit_code == 0,
        .exit_code = result.exit_code,
        .elapsed_ms = elapsed,
        .diagnostics = diagnostics,
        .stdout = result.stdout,
        .stderr = result.stderr,
        .lean_version = lean_version,
    };

    return Response{ .status = 200, .body = try encodeJson(response) };
}

fn handleRun(body: []const u8) !Response {
    const parsed = json.parseFromSlice(LeanRunRequest, allocator, body, .{
        .ignore_unknown_fields = true,
    }) catch {
        return Response{ .status = 400, .body = try errorBody("Invalid JSON payload") };
    };
    defer parsed.deinit();

    const req = parsed.value;
    if (req.code.len == 0) {
        return Response{ .status = 400, .body = try errorBody("code is required") };
    }

    const temp = try createLeanTempFile(req.code, req.filename);
    defer cleanupLeanTempFile(temp);

    var args = std.ArrayList([]const u8).empty;
    defer args.deinit(allocator);
    var owned_args = std.ArrayList([]u8).empty;
    defer {
        for (owned_args.items) |item| {
            allocator.free(item);
        }
        owned_args.deinit(allocator);
    }

    try args.append(allocator, resolveLeanBin());
    try args.append(allocator, "--run");

    if (req.trust_level) |trust| {
        try args.append(allocator, "-t");
        const trust_arg = try std.fmt.allocPrint(allocator, "{d}", .{trust});
        try owned_args.append(allocator, trust_arg);
        try args.append(allocator, trust_arg);
    }

    if (req.max_memory_mb) |mem_mb| {
        try args.append(allocator, "-M");
        const mem_arg = try std.fmt.allocPrint(allocator, "{d}", .{mem_mb});
        try owned_args.append(allocator, mem_arg);
        try args.append(allocator, mem_arg);
    }

    if (resolveRootOverride(req.root)) |root| {
        try args.append(allocator, "--root");
        try args.append(allocator, root);
    }

    try args.append(allocator, temp.file_name);

    if (req.args) |extra_args| {
        for (extra_args) |arg| {
            try args.append(allocator, arg);
        }
    }

    const max_output = resolveMaxOutputBytes(req.max_output_bytes);
    const start = std.time.milliTimestamp();

    const result = runLean(args.items, req.stdin, temp.work_dir, max_output) catch {
        return Response{ .status = 500, .body = try errorBody("Lean execution failed") };
    };
    const elapsed = std.time.milliTimestamp() - start;

    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const response = LeanRunResponse{
        .success = result.exit_code == 0,
        .exit_code = result.exit_code,
        .elapsed_ms = elapsed,
        .stdout = result.stdout,
        .stderr = result.stderr,
        .lean_version = lean_version,
    };

    return Response{ .status = 200, .body = try encodeJson(response) };
}

fn handleInfo() !Response {
    const payload = std.fmt.allocPrint(
        allocator,
        "{{\"service\":\"lean4-runtime\",\"version\":\"1.0.0\",\"lean_version\":\"{s}\",\"endpoints\":[\"/health\",\"/v1/lean4/check\",\"/v1/lean4/run\"]}}",
        .{lean_version},
    ) catch return Response{ .status = 500, .body = try errorBody("Failed to build response") };

    return Response{ .status = 200, .body = payload };
}

fn handleHealth() !Response {
    const payload = std.fmt.allocPrint(
        allocator,
        "{{\"status\":\"ok\",\"service\":\"lean4-runtime\",\"lean_version\":\"{s}\"}}",
        .{lean_version},
    ) catch return Response{ .status = 500, .body = try errorBody("Health check failed") };

    return Response{ .status = 200, .body = payload };
}

fn loadLeanVersion() void {
    const version = getLeanVersion() orelse return;
    lean_version_storage = version;
    lean_version = version;
}

fn getLeanVersion() ?[]u8 {
    var args = [_][]const u8{ resolveLeanBin(), "--version" };
    const result = std.process.Child.run(.{
        .allocator = allocator,
        .argv = &args,
        .max_output_bytes = 1024,
    }) catch return null;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    const trimmed = mem.trim(u8, result.stdout, " \r\n");
    if (trimmed.len == 0) return null;
    return allocator.dupe(u8, trimmed) catch null;
}

fn handleConnection(connection: net.Server.Connection) !void {
    defer connection.stream.close();

    const request_data = readRequest(connection.stream) catch |err| {
        if (err == error.RequestTooLarge) {
            const body = try errorBody("Request too large");
            defer allocator.free(body);
            try sendResponse(connection.stream, 413, "application/json", body);
            return;
        }
        log("request read error: {any}\n", .{err});
        return;
    };
    defer allocator.free(request_data);

    const first_line_end = mem.indexOf(u8, request_data, "\r\n") orelse return;
    const request_line = request_data[0..first_line_end];

    var parts = mem.splitSequence(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    const clean_path = if (mem.indexOf(u8, path, "?")) |idx| path[0..idx] else path;

    log("{s} {s}\n", .{ method, clean_path });

    if (mem.eql(u8, method, "OPTIONS")) {
        try sendResponse(connection.stream, 204, "application/json", "");
        return;
    }

    var body: []const u8 = "";
    if (mem.indexOf(u8, request_data, "\r\n\r\n")) |idx| {
        body = request_data[idx + 4 ..];
    }

    var response: Response = undefined;
    if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/")) {
        response = try handleInfo();
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/health")) {
        response = try handleHealth();
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/lean4/check")) {
        response = try handleCheck(body);
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/lean4/run")) {
        response = try handleRun(body);
    } else {
        response = Response{ .status = 404, .body = try errorBody("Not found") };
    }

    defer allocator.free(response.body);
    try sendResponse(connection.stream, response.status, response.content_type, response.body);
}

pub fn main() !void {
    defer _ = gpa.deinit();
    defer if (lean_version_storage) |version| allocator.free(version);

    loadLeanVersion();

    const host = getenv("SHIMMY_LEAN4_HOST") orelse default_host;
    const port = parseEnvU16("SHIMMY_LEAN4_PORT", default_port);

    const addr = try net.Address.parseIp(host, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("Lean4 service listening on {s}:{d}\n", .{ host, port });

    while (true) {
        const conn = try server.accept();
        handleConnection(conn) catch |err| {
            log("connection error: {any}\n", .{err});
        };
    }
}
