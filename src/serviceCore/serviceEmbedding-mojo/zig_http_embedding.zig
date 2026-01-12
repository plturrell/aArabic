// Zig HTTP Server for Mojo Embedding Service
// Provides HTTP capabilities for Mojo handlers

const std = @import("std");
const net = std.net;
const mem = std.mem;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

const RequestCallback = *const fn (
    method: [*:0]const u8,
    path: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize,
) callconv(.c) [*:0]const u8;

pub const ServerConfig = extern struct {
    port: u16,
    host: [*:0]const u8,
    callback: RequestCallback,
};

export fn zig_embedding_serve(config: *const ServerConfig) callconv(.c) c_int {
    std.debug.print("Embedding HTTP server starting on port {d}\n", .{config.port});

    startServer(config) catch |err| {
        std.debug.print("Server error: {any}\n", .{err});
        return -1;
    };

    return 0;
}

fn startServer(config: *const ServerConfig) !void {
    const host = mem.span(config.host);
    const addr = try net.Address.parseIp(host, config.port);

    var server = try addr.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();

    std.debug.print("Listening on {s}:{d}\n", .{ host, config.port });

    while (true) {
        const conn = try server.accept();
        handleConnection(conn, config.callback) catch |err| {
            std.debug.print("Connection error: {any}\n", .{err});
        };
    }
}

fn handleConnection(conn: net.Server.Connection, callback: RequestCallback) !void {
    defer conn.stream.close();

    var buffer: [16384]u8 = undefined;
    const bytes_read = try conn.stream.read(&buffer);
    if (bytes_read == 0) return;

    const request_data = buffer[0..bytes_read];

    var method_buf: [16]u8 = undefined;
    var path_buf: [1024]u8 = undefined;
    var body_start: usize = 0;

    var lines = mem.splitSequence(u8, request_data, "\r\n");
    const first_line = lines.next() orelse return error.InvalidRequest;

    var parts = mem.splitSequence(u8, first_line, " ");
    const method = parts.next() orelse return error.InvalidMethod;
    const path = parts.next() orelse return error.InvalidPath;

    @memcpy(method_buf[0..method.len], method);
    method_buf[method.len] = 0;

    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;

    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, request_data, body_marker)) |idx| {
        body_start = idx + body_marker.len;
    }

    const body = if (body_start < bytes_read)
        request_data[body_start..]
    else
        &[_]u8{};

    var body_buf: [8192]u8 = undefined;
    @memcpy(body_buf[0..body.len], body);
    body_buf[body.len] = 0;

    const response_ptr = callback(
        @ptrCast(&method_buf),
        @ptrCast(&path_buf),
        @ptrCast(&body_buf),
        body.len,
    );

    const response = mem.span(response_ptr);

    const http_response = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 200 OK\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
            "\r\n" ++
            "{s}",
        .{ response.len, response },
    );
    defer allocator.free(http_response);

    _ = try conn.stream.writeAll(http_response);
}

export fn zig_embedding_get(url: [*:0]const u8) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);

    const result = httpGet(url_str) catch |err| {
        std.debug.print("GET error: {any}\n", .{err});
        return "{}";
    };

    return result.ptr;
}

fn httpGet(url: []const u8) ![:0]const u8 {
    const uri = try std.Uri.parse(url);

    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80,
    );

    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();

    const request = try std.fmt.allocPrint(
        allocator,
        "GET {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "Connection: close\r\n" ++
            "\r\n",
        .{ uri.path.percent_encoded, uri.host.?.percent_encoded },
    );
    defer allocator.free(request);

    _ = try conn.writeAll(request);

    var buffer: [16384]u8 = undefined;
    const bytes_read = try conn.read(&buffer);

    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const body = buffer[body_start..bytes_read];

        const result = try allocator.allocSentinel(u8, body.len, 0);
        @memcpy(result[0..body.len], body);
        return result;
    }

    return "{}";
}
