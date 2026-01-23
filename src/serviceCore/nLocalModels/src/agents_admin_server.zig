const std = @import("std");
const net = std.net;
const mem = std.mem;
const json = std.json;
const HanaAgent = @import("orchestration/agents/hana_agent.zig");

const Response = struct {
    status: u16,
    body: []const u8,
    content_type: []const u8 = "application/json",
};

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

fn isAuthorized(req: []const u8) bool {
    // Very basic Basic-Auth check; replace with JWT if needed
    const admin_user = std.process.getEnvVarOwned(std.heap.page_allocator, "ADMIN_USER") catch null;
    defer if (admin_user) |u| std.heap.page_allocator.free(u);
    const admin_pass = std.process.getEnvVarOwned(std.heap.page_allocator, "ADMIN_PASS") catch null;
    defer if (admin_pass) |p| std.heap.page_allocator.free(p);

    if (admin_user == null or admin_pass == null) return true; // dev mode

    const creds = std.fmt.allocPrint(std.heap.page_allocator, "{s}:{s}", .{ admin_user.?, admin_pass.? }) catch return false;
    defer std.heap.page_allocator.free(creds);

    var encoded_buf: [512]u8 = undefined;
    const encoded_slice = std.base64.standard.Encoder.encode(&encoded_buf, creds);

    const header = std.fmt.allocPrint(std.heap.page_allocator, "Authorization: Basic {s}", .{encoded_slice}) catch return false;
    defer std.heap.page_allocator.free(header);
    return mem.indexOf(u8, req, header) != null;
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

        if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/admin/prompts")) {
            const val = parseJson(allocator, body) catch {
                try sendResponse(conn.stream, .{ .status = 400, .body = "invalid json", .content_type = "text/plain" });
                continue;
            };
            const text = getField(val, "prompt_text") orelse "";
            const model = getField(val, "model_name") orelse "";
            const user = getField(val, "user_id") orelse "admin";
            const tags = getField(val, "tags") orelse "";
            _ = agent.createPrompt(text, model, user, 1, tags) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "create failed", .content_type = "text/plain" });
                continue;
            };
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"status\":\"ok\"}" });
            continue;
        }

        if (mem.eql(u8, method, "GET") and mem.startsWith(u8, path, "/admin/prompts")) {
            const data = agent.listPrompts(50, 0) catch {
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
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"deleted\":true}" });
            continue;
        }

        if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/admin/comparisons")) {
            const val = parseJson(allocator, body) catch {
                try sendResponse(conn.stream, .{ .status = 400, .body = "invalid json", .content_type = "text/plain" });
                continue;
            };
            const prompt = getField(val, "prompt") orelse "";
            const ma = getField(val, "modelA") orelse "";
            const mb = getField(val, "modelB") orelse "";
            const winner = getField(val, "winner") orelse "";
            const respA = getField(val, "responseA") orelse "";
            const respB = getField(val, "responseB") orelse "";
            _ = agent.createComparison(prompt, ma, mb, winner, respA, respB, 0, 0, 0, 0) catch {
                try sendResponse(conn.stream, .{ .status = 500, .body = "create failed", .content_type = "text/plain" });
                continue;
            };
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"status\":\"ok\"}" });
            continue;
        }

        if (mem.eql(u8, method, "GET") and mem.startsWith(u8, path, "/admin/comparisons")) {
            const data = agent.listComparisons(50, 0) catch {
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
            try sendResponse(conn.stream, .{ .status = 200, .body = "{\"deleted\":true}" });
            continue;
        }

        try sendResponse(conn.stream, .{ .status = 404, .body = "not found", .content_type = "text/plain" });
    }
}
