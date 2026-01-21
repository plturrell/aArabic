const std = @import("std");
const net = std.net;
const fs = std.fs;
const mem = std.mem;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: {s} <port> <directory>\n", .{args[0]});
        return;
    }

    const port = try std.fmt.parseInt(u16, args[1], 10);
    const dir_path = args[2];

    // Verify directory exists
    var dir = fs.cwd().openDir(dir_path, .{}) catch |err| {
        std.debug.print("Error: Could not open directory '{s}': {any}\n", .{ dir_path, err });
        return;
    };
    dir.close();

    const addr = try net.Address.parseIp("0.0.0.0", port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("⚡ nWebServe running on http://localhost:{d}\n", .{port});
    std.debug.print("📂 Serving: {s}\n", .{dir_path});

    while (true) {
        const conn = try server.accept();
        handleConnection(conn, allocator, dir_path) catch |err| {
            std.debug.print("Connection error: {any}\n", .{err});
        };
    }
}

fn handleConnection(conn: net.Server.Connection, allocator: mem.Allocator, root_dir: []const u8) !void {
    defer conn.stream.close();
    
    // Buffer for request
    var buf: [4096]u8 = undefined;
    const bytes_read = try conn.stream.read(&buf);
    if (bytes_read == 0) return;
    
    const request = buf[0..bytes_read];
    
    // Parse Request Line (GET /path HTTP/1.1)
    var lines = mem.splitSequence(u8, request, "\r\n");
    const first_line = lines.next() orelse return;
    var parts = mem.splitSequence(u8, first_line, " ");
    
    const method = parts.next() orelse return;
    var path = parts.next() orelse return;
    
    if (!mem.eql(u8, method, "GET")) return;

    // Sanitize path
    if (path.len == 0 or mem.eql(u8, path, "/")) {
        path = "/index.html";
    }
    
    // Remove query parameters
    if (mem.indexOf(u8, path, "?")) |idx| {
        path = path[0..idx];
    }

    // Security: Prevent directory traversal
    if (mem.indexOf(u8, path, "..")) |idx| {
         _ = idx;
        try send404(conn.stream);
        return;
    }

    // Construct full path
    const safe_path = if (path[0] == '/') path[1..] else path;
    
    var root = fs.cwd().openDir(root_dir, .{}) catch |err| {
        std.debug.print("Error opening directory: {any}\n", .{err});
        try send404(conn.stream);
        return;
    };
    defer root.close();

    const file = root.openFile(safe_path, .{}) catch |err| {
        std.debug.print("Error opening file '{s}': {any}\n", .{safe_path, err});
        try send404(conn.stream);
        return;
    };
    defer file.close();

    const stat = try file.stat();
    const file_size = stat.size;
    const mime_type = getMimeType(path);

    // Header
    const header = try std.fmt.allocPrint(allocator, 
        "HTTP/1.1 200 OK\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nConnection: close\r\nAccess-Control-Allow-Origin: *\r\n\r\n", 
        .{ mime_type, file_size }
    );
    defer allocator.free(header);
    
    try conn.stream.writeAll(header);

    // Body (Chunked Send)
    var file_buf: [8192]u8 = undefined;
    while (true) {
        const n = try file.read(&file_buf);
        if (n == 0) break;
        try conn.stream.writeAll(file_buf[0..n]);
    }
}

fn send404(stream: net.Stream) !void {
    const response = "HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\nContent-Length: 13\r\nConnection: close\r\n\r\n404 Not Found";
    _ = stream.writeAll(response) catch {};
}

fn getMimeType(path: []const u8) []const u8 {
    if (mem.endsWith(u8, path, ".html")) return "text/html";
    if (mem.endsWith(u8, path, ".css")) return "text/css";
    if (mem.endsWith(u8, path, ".js")) return "application/javascript";
    if (mem.endsWith(u8, path, ".json")) return "application/json";
    if (mem.endsWith(u8, path, ".png")) return "image/png";
    if (mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (mem.endsWith(u8, path, ".xml")) return "application/xml";
    return "text/plain";
}
