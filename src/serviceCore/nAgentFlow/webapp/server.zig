const std = @import("std");

const PORT = 8082;
const WEBAPP_DIR = ".";

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const address = try std.net.Address.parseIp("127.0.0.1", PORT);
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();

    std.debug.print("⚙️  nAgentFlow webapp serving at http://localhost:{d}\n", .{PORT});
    std.debug.print("Press Ctrl+C to stop\n\n", .{});

    while (true) {
        const connection = try server.accept();
        const thread = try std.Thread.spawn(.{}, handleConnection, .{ allocator, connection });
        thread.detach();
    }
}

fn handleConnection(allocator: std.mem.Allocator, connection: std.net.Server.Connection) void {
    defer connection.stream.close();
    
    handleRequest(allocator, connection) catch |err| {
        std.debug.print("Error handling request: {}\n", .{err});
    };
}

fn handleRequest(allocator: std.mem.Allocator, connection: std.net.Server.Connection) !void {
    var buffer: [4096]u8 = undefined;
    
    const bytes_read = try connection.stream.read(&buffer);
    if (bytes_read == 0) return;
    
    const request = buffer[0..bytes_read];
    
    var lines = std.mem.splitScalar(u8, request, '\n');
    const first_line = lines.next() orelse return;
    
    var parts = std.mem.splitScalar(u8, first_line, ' ');
    _ = parts.next();
    var path = parts.next() orelse "/";
    
    if (std.mem.indexOfScalar(u8, path, '?')) |idx| {
        path = path[0..idx];
    }
    if (std.mem.endsWith(u8, path, "\r")) {
        path = path[0 .. path.len - 1];
    }
    
    if (std.mem.eql(u8, path, "/")) {
        path = "/index.html";
    }
    
    std.debug.print("GET {s}\n", .{path});
    
    var path_buffer: [512]u8 = undefined;
    const file_path = try std.fmt.bufPrint(&path_buffer, "{s}{s}", .{ WEBAPP_DIR, path });
    
    const file = std.fs.cwd().openFile(file_path, .{}) catch {
        try sendResponse(connection.stream, 404, "Not Found", "text/plain");
        return;
    };
    defer file.close();
    
    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
    defer allocator.free(content);
    
    const content_type = getContentType(path);
    try sendResponse(connection.stream, 200, content, content_type);
}

fn sendResponse(stream: std.net.Stream, status: u16, body: []const u8, content_type: []const u8) !void {
    const status_text = if (status == 200) "OK" else if (status == 404) "Not Found" else "Error";
    
    var response_buffer: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&response_buffer, 
        "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Cache-Control: no-store, no-cache, must-revalidate\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{ status, status_text, content_type, body.len }
    );
    
    _ = try stream.writeAll(header);
    _ = try stream.writeAll(body);
}

fn getContentType(path: []const u8) []const u8 {
    if (std.mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (std.mem.endsWith(u8, path, ".css")) return "text/css";
    if (std.mem.endsWith(u8, path, ".js")) return "application/javascript";
    if (std.mem.endsWith(u8, path, ".json")) return "application/json";
    if (std.mem.endsWith(u8, path, ".xml")) return "application/xml";
    if (std.mem.endsWith(u8, path, ".png")) return "image/png";
    if (std.mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (std.mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (std.mem.endsWith(u8, path, ".wasm")) return "application/wasm";
    if (std.mem.endsWith(u8, path, ".properties")) return "text/plain";
    return "application/octet-stream";
}