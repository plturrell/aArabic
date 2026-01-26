const std = @import("std");

const http = std.http;
const json = std.json;
const mem = std.mem;
const Allocator = mem.Allocator;

const empty_json: [:0]const u8 = "{}";

pub const LeanDiagnostic = struct {
    severity: []const u8,
    message: []const u8,
    file: ?[]const u8 = null,
    line: ?u32 = null,
    column: ?u32 = null,
    end_line: ?u32 = null,
    end_column: ?u32 = null,
    kind: ?[]const u8 = null,
};

pub const CheckRequest = struct {
    code: []const u8,
    filename: ?[]const u8 = null,
    max_output_bytes: ?usize = null,
    max_memory_mb: ?u32 = null,
    trust_level: ?u8 = null,
    root: ?[]const u8 = null,
};

pub const RunRequest = struct {
    code: []const u8,
    filename: ?[]const u8 = null,
    args: ?[]const []const u8 = null,
    stdin: ?[]const u8 = null,
    max_output_bytes: ?usize = null,
    max_memory_mb: ?u32 = null,
    trust_level: ?u8 = null,
    root: ?[]const u8 = null,
};

pub const CheckResponse = struct {
    success: bool,
    exit_code: i32,
    elapsed_ms: i64,
    diagnostics: []LeanDiagnostic,
    stdout: []const u8,
    stderr: []const u8,
    lean_version: []const u8,
};

pub const RunResponse = struct {
    success: bool,
    exit_code: i32,
    elapsed_ms: i64,
    stdout: []const u8,
    stderr: []const u8,
    lean_version: []const u8,
};

pub const Lean4Client = struct {
    allocator: Allocator,
    base_url: []const u8,
    client: http.Client,

    pub fn init(allocator: Allocator, host: []const u8, port: u16) !*Lean4Client {
        const client = try allocator.create(Lean4Client);
        const base_url = try std.fmt.allocPrint(allocator, "http://{s}:{d}", .{ host, port });

        client.* = Lean4Client{
            .allocator = allocator,
            .base_url = base_url,
            .client = http.Client{ .allocator = allocator },
        };

        return client;
    }

    pub fn deinit(self: *Lean4Client) void {
        self.allocator.free(self.base_url);
        self.client.deinit();
        self.allocator.destroy(self);
    }

    pub fn checkRaw(self: *Lean4Client, request: CheckRequest) ![]u8 {
        const payload = try json.Stringify.valueAlloc(self.allocator, request, .{});
        defer self.allocator.free(payload);
        return self.postJson("/v1/lean4/check", payload);
    }

    pub fn runRaw(self: *Lean4Client, request: RunRequest) ![]u8 {
        const payload = try json.Stringify.valueAlloc(self.allocator, request, .{});
        defer self.allocator.free(payload);
        return self.postJson("/v1/lean4/run", payload);
    }

    pub fn check(self: *Lean4Client, request: CheckRequest) !CheckResponse {
        const raw = try self.checkRaw(request);
        defer self.allocator.free(raw);
        return parseCheckResponse(self.allocator, raw);
    }

    pub fn run(self: *Lean4Client, request: RunRequest) !RunResponse {
        const raw = try self.runRaw(request);
        defer self.allocator.free(raw);
        return parseRunResponse(self.allocator, raw);
    }

    pub fn freeCheckResponse(self: *Lean4Client, response: *CheckResponse) void {
        self.allocator.free(response.stdout);
        self.allocator.free(response.stderr);
        self.allocator.free(response.lean_version);
        freeDiagnostics(self.allocator, response.diagnostics);
    }

    pub fn freeRunResponse(self: *Lean4Client, response: *RunResponse) void {
        self.allocator.free(response.stdout);
        self.allocator.free(response.stderr);
        self.allocator.free(response.lean_version);
    }

    pub fn postJsonRaw(self: *Lean4Client, endpoint: []const u8, payload_json: []const u8) ![]u8 {
        return self.postJson(endpoint, payload_json);
    }

    fn postJson(self: *Lean4Client, endpoint: []const u8, payload_json: []const u8) ![]u8 {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, endpoint });
        defer self.allocator.free(url);

        const uri = try std.Uri.parse(url);

        var response_body: std.io.Writer.Allocating = .init(self.allocator);
        defer response_body.deinit();

        const req = try self.client.fetch(.{
            .location = .{ .uri = uri },
            .method = .POST,
            .payload = payload_json,
            .response_writer = &response_body.writer,
            .headers = .{ .content_type = .{ .override = "application/json" } },
        });

        if (req.status != .ok) {
            return error.HttpRequestFailed;
        }

        return self.allocator.dupe(u8, response_body.written());
    }
};

fn parseCheckResponse(allocator: Allocator, body: []const u8) !CheckResponse {
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidResponse,
    };

    const success = getBoolField(obj, "success") orelse false;
    const exit_code = getI32Field(obj, "exit_code") orelse -1;
    const elapsed_ms = getI64Field(obj, "elapsed_ms") orelse 0;

    const stdout_value = getStringField(obj, "stdout") orelse "";
    const stderr_value = getStringField(obj, "stderr") orelse "";
    const lean_version_value = getStringField(obj, "lean_version") orelse "";

    const stdout = try allocator.dupe(u8, stdout_value);
    errdefer allocator.free(stdout);
    const stderr = try allocator.dupe(u8, stderr_value);
    errdefer allocator.free(stderr);
    const lean_version = try allocator.dupe(u8, lean_version_value);
    errdefer allocator.free(lean_version);

    var diagnostics: []LeanDiagnostic = &.{};
    if (obj.get("diagnostics")) |diag_value| {
        diagnostics = try parseDiagnostics(allocator, diag_value);
    } else {
        diagnostics = try allocator.alloc(LeanDiagnostic, 0);
    }
    errdefer freeDiagnostics(allocator, diagnostics);

    return CheckResponse{
        .success = success,
        .exit_code = exit_code,
        .elapsed_ms = elapsed_ms,
        .diagnostics = diagnostics,
        .stdout = stdout,
        .stderr = stderr,
        .lean_version = lean_version,
    };
}

fn parseRunResponse(allocator: Allocator, body: []const u8) !RunResponse {
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    defer parsed.deinit();

    const obj = switch (parsed.value) {
        .object => |value| value,
        else => return error.InvalidResponse,
    };

    const success = getBoolField(obj, "success") orelse false;
    const exit_code = getI32Field(obj, "exit_code") orelse -1;
    const elapsed_ms = getI64Field(obj, "elapsed_ms") orelse 0;

    const stdout_value = getStringField(obj, "stdout") orelse "";
    const stderr_value = getStringField(obj, "stderr") orelse "";
    const lean_version_value = getStringField(obj, "lean_version") orelse "";

    const stdout = try allocator.dupe(u8, stdout_value);
    errdefer allocator.free(stdout);
    const stderr = try allocator.dupe(u8, stderr_value);
    errdefer allocator.free(stderr);
    const lean_version = try allocator.dupe(u8, lean_version_value);
    errdefer allocator.free(lean_version);

    return RunResponse{
        .success = success,
        .exit_code = exit_code,
        .elapsed_ms = elapsed_ms,
        .stdout = stdout,
        .stderr = stderr,
        .lean_version = lean_version,
    };
}

fn parseDiagnostics(allocator: Allocator, value: json.Value) ![]LeanDiagnostic {
    const array = switch (value) {
        .array => |arr| arr,
        else => return allocator.alloc(LeanDiagnostic, 0),
    };

    var list = std.ArrayList(LeanDiagnostic){};
    errdefer {
        for (list.items) |diag| {
            freeDiagnosticFields(allocator, diag);
        }
        list.deinit();
    }

    for (array.items) |item| {
        const obj = switch (item) {
            .object => |value_obj| value_obj,
            else => continue,
        };

        const severity = getStringField(obj, "severity") orelse "unknown";
        const message = getStringField(obj, "message") orelse "";
        const file_value = getStringField(obj, "file");
        const kind_value = getStringField(obj, "kind");

        const diag = LeanDiagnostic{
            .severity = try allocator.dupe(u8, severity),
            .message = try allocator.dupe(u8, message),
            .file = try dupOptionalString(allocator, file_value),
            .line = getU32Field(obj, "line"),
            .column = getU32Field(obj, "column"),
            .end_line = getU32Field(obj, "end_line"),
            .end_column = getU32Field(obj, "end_column"),
            .kind = try dupOptionalString(allocator, kind_value),
        };

        try list.append(diag);
    }

    return list.toOwnedSlice();
}

fn freeDiagnostics(allocator: Allocator, diagnostics: []LeanDiagnostic) void {
    for (diagnostics) |diag| {
        freeDiagnosticFields(allocator, diag);
    }
    allocator.free(diagnostics);
}

fn freeDiagnosticFields(allocator: Allocator, diag: LeanDiagnostic) void {
    allocator.free(diag.severity);
    allocator.free(diag.message);
    if (diag.file) |file_name| {
        allocator.free(file_name);
    }
    if (diag.kind) |kind| {
        allocator.free(kind);
    }
}

fn dupOptionalString(allocator: Allocator, value: ?[]const u8) !?[]u8 {
    if (value) |string_value| {
        return try allocator.dupe(u8, string_value);
    }
    return null;
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

fn getBoolField(obj: json.ObjectMap, key: []const u8) ?bool {
    if (obj.get(key)) |value| {
        return switch (value) {
            .bool => |b| b,
            else => null,
        };
    }
    return null;
}

fn getI32Field(obj: json.ObjectMap, key: []const u8) ?i32 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |i| @as(i32, @intCast(i)),
            .float => |f| @as(i32, @intFromFloat(f)),
            else => null,
        };
    }
    return null;
}

fn getI64Field(obj: json.ObjectMap, key: []const u8) ?i64 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |i| i,
            .float => |f| @as(i64, @intFromFloat(f)),
            else => null,
        };
    }
    return null;
}

fn getU32Field(obj: json.ObjectMap, key: []const u8) ?u32 {
    if (obj.get(key)) |value| {
        return switch (value) {
            .integer => |i| if (i >= 0) @as(u32, @intCast(i)) else null,
            .float => |f| if (f >= 0) @as(u32, @intFromFloat(f)) else null,
            else => null,
        };
    }
    return null;
}

const CClient = opaque {};

export fn lean4_client_create(host: [*:0]const u8, port: u16) callconv(.c) ?*CClient {
    const host_slice = mem.span(host);
    const client = Lean4Client.init(std.heap.page_allocator, host_slice, port) catch return null;
    return @ptrCast(client);
}

export fn lean4_client_destroy(client: *CClient) callconv(.c) void {
    const real_client: *Lean4Client = @ptrCast(@alignCast(client));
    real_client.deinit();
}

export fn lean4_check_json(client: *CClient, code: [*:0]const u8) callconv(.c) [*:0]const u8 {
    const real_client: *Lean4Client = @ptrCast(@alignCast(client));
    const request = CheckRequest{ .code = mem.span(code) };
    const raw = real_client.checkRaw(request) catch return empty_json.ptr;
    const cstr = toSentinel(raw) catch {
        real_client.allocator.free(raw);
        return empty_json.ptr;
    };
    real_client.allocator.free(raw);
    return cstr.ptr;
}

export fn lean4_run_json(
    client: *CClient,
    code: [*:0]const u8,
    stdin_ptr: ?[*:0]const u8,
) callconv(.c) [*:0]const u8 {
    const real_client: *Lean4Client = @ptrCast(@alignCast(client));
    const stdin_value = if (stdin_ptr) |ptr| mem.span(ptr) else null;
    const request = RunRequest{ .code = mem.span(code), .stdin = stdin_value };
    const raw = real_client.runRaw(request) catch return empty_json.ptr;
    const cstr = toSentinel(raw) catch {
        real_client.allocator.free(raw);
        return empty_json.ptr;
    };
    real_client.allocator.free(raw);
    return cstr.ptr;
}

export fn lean4_request_json(
    client: *CClient,
    endpoint: [*:0]const u8,
    payload_json: [*:0]const u8,
) callconv(.c) [*:0]const u8 {
    const real_client: *Lean4Client = @ptrCast(@alignCast(client));
    const raw = real_client.postJsonRaw(mem.span(endpoint), mem.span(payload_json)) catch return empty_json.ptr;
    const cstr = toSentinel(raw) catch {
        real_client.allocator.free(raw);
        return empty_json.ptr;
    };
    real_client.allocator.free(raw);
    return cstr.ptr;
}

export fn lean4_free_json(ptr: [*:0]const u8) callconv(.c) void {
    const slice = mem.span(ptr);
    std.heap.page_allocator.free(@constCast(slice));
}

fn toSentinel(raw: []u8) ![:0]u8 {
    var out = try std.heap.page_allocator.allocSentinel(u8, raw.len, 0);
    @memcpy(out[0..raw.len], raw);
    return out;
}
