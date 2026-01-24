const std = @import("std");
const net = std.net;
const mem = std.mem;
const json = std.json;
const HanaAgent = @import("orchestration/agents/hana_agent.zig");
const jwt = @import("shared/auth/jwt_validator.zig");

fn constantTimeEquals(a: []const u8, b: []const u8) bool {
    if (a.len != b.len) return false;
    var diff: u8 = 0;
    for (a, 0..) |ch, i| diff |= ch ^ b[i];
    return diff == 0;
}

fn findHeader(req: []const u8, name: []const u8) ?[]const u8 {
    var it = mem.splitSequence(u8, req, "\r\n");
    _ = it.next(); // request line
    while (it.next()) |line| {
        if (line.len == 0) break;
        if (mem.startsWith(u8, line, name)) {
            const idx = mem.indexOf(u8, line, ":") orelse return null;
            var start: usize = idx + 1;
            if (start < line.len and line[start] == ' ') start += 1;
            return line[start..];
        }
    }
    return null;
}

const Response = struct {
    status: u16,
    body: []const u8,
    content_type: []const u8 = "application/json",
};

fn badRequest(stream: net.Stream, msg: []const u8) !void {
    try sendResponse(stream, .{ .status = 400, .body = msg, .content_type = "text/plain" });
}

fn forbidden(stream: net.Stream, msg: []const u8) !void {
    try sendResponse(stream, .{ .status = 403, .body = msg, .content_type = "text/plain" });
}

fn unauthorized(stream: net.Stream, msg: []const u8) !void {
    try sendResponse(stream, .{ .status = 401, .body = msg, .content_type = "text/plain" });
}

fn ensureNonEmpty(val: ?[]const u8, field: []const u8, stream: net.Stream) ![]const u8 {
    if (val == null or val.?.len == 0) {
        try badRequest(stream, field);
        return error.Invalid;
    }
    return val.?;
}

fn sendResponse(stream: net.Stream, resp: Response) !void {
    var buf: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&buf,
        "HTTP/1.1 {d} OK\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n",
        .{ resp.status, resp.content_type, resp.body.len },
    );
    _ = try stream.writeAll(header);
    _ = try stream.writeAll(resp.body);
}

fn parseJson(allocator: std.mem.Allocator, body: []const u8) !json.Value {
    const tree = try json.parseFromSlice(json.Value, allocator, body, .{});
    return tree.value;
}

fn jwtAuthorized(header_value: []const u8) bool {
    if (jwt.findBearerToken(header_value)) |token| {
        return jwt.validateWithKey(token, "");
    }
    return false;
}

fn isAuthorized(req: []const u8) bool {
    const auth_header = findHeader(req, "Authorization") orelse return false;

    // Basic-Auth check
    const admin_user = std.process.getEnvVarOwned(std.heap.page_allocator, "ADMIN_USER") catch null;
    defer if (admin_user) |u| std.heap.page_allocator.free(u);
    const admin_pass = std.process.getEnvVarOwned(std.heap.page_allocator, "ADMIN_PASS") catch null;
    defer if (admin_pass) |p| std.heap.page_allocator.free(p);

    if (admin_user != null and admin_pass != null) {
        const creds = std.fmt.allocPrint(std.heap.page_allocator, "{s}:{s}", .{ admin_user.?, admin_pass.? }) catch return false;
        defer std.heap.page_allocator.free(creds);
        var encoded_buf: [512]u8 = undefined;
        const encoded_slice = std.base64.standard.Encoder.encode(&encoded_buf, creds);
        const expected = std.fmt.allocPrint(std.heap.page_allocator, "Basic {s}", .{encoded_slice}) catch return false;
        defer std.heap.page_allocator.free(expected);
        if (constantTimeEquals(auth_header, expected)) return true;
    }

    // JWT check (exp only; signature not verified)
    if (jwtAuthorized(auth_header)) return true;

    return false;
}

fn getField(v: json.Value, key: []const u8) ?[]const u8 {
    if (v == .object) {
        if (v.object.get(key)) |val| {
            return switch (val) {
                .string => |s| s,
                else => null,
            };
        }
    }
    return null;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const address = try net.Address.parseIp("0.0.0.0", 8090);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("🛡️  Agents Admin Server on 8090\n", .{});

    const hana_cfg = HanaAgent.Config{
        .bridge_url = std.process.getEnvVarOwned(allocator, "HANA_BRIDGE_URL") catch "http://localhost:3001/sql",
        .schema = std.process.getEnvVarOwned(allocator, "HANA_SCHEMA") catch "DBADMIN",
    };
    var agent = HanaAgent.init(allocator, hana_cfg);

    while (true) {
        const conn = server.accept() catch |err| {
            std.debug.print("accept err: {}\n", .{err});
            continue;
        };
        defer conn.stream.close();

        var buf: [8192]u8 = undefined;
        const n = conn.stream.read(&buf) catch continue;
        if (n == 0) continue;
        const req = buf[0..n];

        const first_line_end = mem.indexOf(u8, req, "\r\n") orelse continue;
        const line = req[0..first_line_end];
        var parts = mem.splitSequence(u8, line, " ");
        const method = parts.next() orelse continue;
        const path = parts.next() orelse continue;

        const body_start = mem.indexOf(u8, req, "\r\n\r\n") orelse req.len;
        const body = if (body_start + 4 <= req.len) req[body_start + 4 ..] else "";

        if (!isAuthorized(req)) {
            try sendResponse(conn.stream, .{ .status = 401, .body = "unauthorized", .content_type = "text/plain" });
            continue;
        }

        // Parse query params for limit/offset
        var limit: usize = 50;
        var offset: usize = 0;
        if (mem.indexOf(u8, path, "?")) |q_idx| {
            const qs = path[q_idx + 1 ..];
            var params = mem.splitSequence(u8, qs, "&");
            while (params.next()) |p| {
                if (mem.indexOf(u8, p, "=")) |eq| {
                    const key = p[0..eq];
                    const val = p[eq + 1 ..];
                    if (mem.eql(u8, key, "limit")) {
                        limit = std.fmt.parseInt(usize, val, 10) catch limit;
                        if (limit > 500) limit = 500;
                    } else if (mem.eql(u8, key, "offset")) {
                        offset = std.fmt.parseInt(usize, val, 10) catch offset;
                    }
                }
            }
        }

        if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/admin/prompts")) {
            const val = parseJson(allocator, body) catch {
                try sendResponse(conn.stream, .{ .status = 400, .body = "invalid json", .content_type = "text/plain" });
                continue;
            };
            const text = ensureNonEmpty(getField(val, "prompt_text"), "prompt_text required", conn.stream) catch continue;
            const model = ensureNonEmpty(getField(val, "model_name"), "model_name required", conn.stream) catch continue;
            const user = ensureNonEmpty(getField(val, "user_id"), "user_id required", conn.stream) catch continue;
            const tags = getField(val, "tags") orelse "";
            _ = agent.createPrompt(text, model, user, 1, tags) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "create failed", .content_type = "text/plain" });
                continue;
            };
            agent.logAudit("CREATE_PROMPT", "PROMPTS", text);
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"status\":\"ok\"}" });
            continue;
        }

        if (mem.eql(u8, method, "GET") and mem.startsWith(u8, path, "/admin/prompts")) {
            const data = agent.listPrompts(limit, offset) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "query failed", .content_type = "text/plain" });
                continue;
            };
            defer allocator.free(data);
            try sendResponse(conn.stream, .{ .status = 200, .body = data, .content_type = "application/json" });
            continue;
        }

        if (mem.eql(u8, method, "DELETE") and mem.startsWith(u8, path, "/admin/prompts/")) {
            const id_str = path["/admin/prompts/".len..];
            const id = std.fmt.parseInt(i32, id_str, 10) catch 0;
            _ = agent.deletePrompt(id) catch {};
            agent.logAudit("DELETE_PROMPT", "PROMPTS", id_str);
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"deleted\":true}" });
            continue;
        }

        if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/admin/comparisons")) {
            const val = parseJson(allocator, body) catch {
                try sendResponse(conn.stream, .{ .status = 400, .body = "invalid json", .content_type = "text/plain" });
                continue;
            };
            const prompt = ensureNonEmpty(getField(val, "prompt"), "prompt required", conn.stream) catch continue;
            const ma = ensureNonEmpty(getField(val, "modelA"), "modelA required", conn.stream) catch continue;
            const mb = ensureNonEmpty(getField(val, "modelB"), "modelB required", conn.stream) catch continue;
            const winner = ensureNonEmpty(getField(val, "winner"), "winner required", conn.stream) catch continue;
            const respA = getField(val, "responseA") orelse "";
            const respB = getField(val, "responseB") orelse "";
            _ = agent.createComparison(prompt, ma, mb, winner, respA, respB, 0, 0, 0, 0) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "create failed", .content_type = "text/plain" });
                continue;
            };
            agent.logAudit("CREATE_COMPARISON", "PROMPT_COMPARISONS", prompt);
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"status\":\"ok\"}" });
            continue;
        }

        if (mem.eql(u8, method, "GET") and mem.startsWith(u8, path, "/admin/comparisons")) {
            const data = agent.listComparisons(limit, offset) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "query failed", .content_type = "text/plain" });
                continue;
            };
            defer allocator.free(data);
            try sendResponse(conn.stream, .{ .status = 200, .body = data, .content_type = "application/json" });
            continue;
        }

        if (mem.eql(u8, method, "DELETE") and mem.startsWith(u8, path, "/admin/comparisons/")) {
            const id_str = path["/admin/comparisons/".len..];
            const id = std.fmt.parseInt(i32, id_str, 10) catch 0;
            _ = agent.deleteComparison(id) catch {};
            agent.logAudit("DELETE_COMPARISON", "PROMPT_COMPARISONS", id_str);
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"deleted\":true}" });
            continue;
        }

        try sendResponse(conn.stream, .{ .status = 404, .body = "not found", .content_type = "text/plain" });
    }
}
