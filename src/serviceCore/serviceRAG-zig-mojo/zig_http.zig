// Zig HTTP Server for Mojo
// Provides HTTP capabilities that Mojo stdlib doesn't have yet
// Exports C ABI functions that Mojo can call via FFI

const std = @import("std");
const http = std.http;
const net = std.net;
const mem = std.mem;

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Callback type for Mojo to handle requests
const RequestCallback = *const fn (
    method: [*:0]const u8,
    path: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize
) callconv(.c) [*:0]const u8;

// HTTP Server configuration
pub const ServerConfig = extern struct {
    port: u16,
    host: [*:0]const u8,
    callback: RequestCallback,
};

/// Initialize and start HTTP server
/// This will be called from Mojo
export fn zig_http_serve(config: *const ServerConfig) callconv(.c) c_int {
    std.debug.print("🚀 Zig HTTP Server starting on port {d}\n", .{config.port});
    
    // Start server
    startServer(config) catch |err| {
        std.debug.print("❌ Server error: {any}\n", .{err});
        return -1;
    };
    
    return 0;
}

fn startServer(config: *const ServerConfig) !void {
    // Parse address
    const host = mem.span(config.host);
    const addr = try net.Address.parseIp(host, config.port);
    
    // Create server
    var server = try addr.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();
    
    std.debug.print("✅ Server listening on {any}:{d}\n", .{addr, config.port});
    std.debug.print("📡 Ready to handle requests...\n", .{});
    
    // Accept connections loop
    while (true) {
        // Accept connection
        const conn = try server.accept();
        
        // Handle in separate thread (basic version - single threaded for now)
        handleConnection(conn, config.callback) catch |err| {
            std.debug.print("⚠️  Connection error: {any}\n", .{err});
        };
    }
}

fn handleConnection(conn: net.Server.Connection, callback: RequestCallback) !void {
    defer conn.stream.close();
    
    var buffer: [8192]u8 = undefined;
    
    // Read request
    const bytes_read = try conn.stream.read(&buffer);
    if (bytes_read == 0) return;
    
    const request_data = buffer[0..bytes_read];
    
    // Parse HTTP request (simplified)
    var method_buf: [16]u8 = undefined;
    var path_buf: [1024]u8 = undefined;
    var body_start: usize = 0;
    
    // Find method and path
    var lines = mem.splitSequence(u8, request_data, "\r\n");
    const first_line = lines.next() orelse return error.InvalidRequest;
    
    var parts = mem.splitSequence(u8, first_line, " ");
    const method = parts.next() orelse return error.InvalidMethod;
    const path = parts.next() orelse return error.InvalidPath;
    
    // Copy to null-terminated buffers
    @memcpy(method_buf[0..method.len], method);
    method_buf[method.len] = 0;
    
    @memcpy(path_buf[0..path.len], path);
    path_buf[path.len] = 0;
    
    // Find body (after \r\n\r\n)
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, request_data, body_marker)) |idx| {
        body_start = idx + body_marker.len;
    }
    
    const body = if (body_start < bytes_read) 
        request_data[body_start..] 
    else 
        &[_]u8{};
    
    // Ensure body is null-terminated for C ABI
    var body_buf: [4096]u8 = undefined;
    @memcpy(body_buf[0..body.len], body);
    body_buf[body.len] = 0;
    
    std.debug.print("📥 {s} {s} ({d} bytes)\n", .{method, path, body.len});
    
    // Call Mojo callback
    const response_ptr = callback(
        @ptrCast(&method_buf),
        @ptrCast(&path_buf),
        @ptrCast(&body_buf),
        body.len
    );
    
    const response = mem.span(response_ptr);
    
    // Send HTTP response
    const http_response = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "\r\n" ++
        "{s}",
        .{response.len, response}
    );
    defer allocator.free(http_response);
    
    _ = try conn.stream.writeAll(http_response);
    
    std.debug.print("📤 Response sent ({d} bytes)\n", .{response.len});
}

/// Simple HTTP GET request (for calling other services)
export fn zig_http_get(url: [*:0]const u8) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    
    const result = httpGet(url_str) catch |err| {
        std.debug.print("❌ GET error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn httpGet(url: []const u8) ![:0]const u8 {
    // Parse URL (simplified - assumes http://host:port/path)
    const uri = try std.Uri.parse(url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send GET request
    const request = try std.fmt.allocPrint(
        allocator,
        "GET {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded}
    );
    defer allocator.free(request);
    
    _ = try conn.writeAll(request);
    
    // Read response
    var buffer: [16384]u8 = undefined;
    const bytes_read = try conn.read(&buffer);
    
    // Find body
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const body = buffer[body_start..bytes_read];
        
        // Allocate and copy
        const result = try allocator.allocSentinel(u8, body.len, 0);
        @memcpy(result[0..body.len], body);
        
        return result;
    }
    
    return "{}";
}

/// Simple HTTP POST request (for calling other services)
export fn zig_http_post(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize
) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpPost(url_str, body_data) catch |err| {
        std.debug.print("❌ POST error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn httpPost(url: []const u8, body: []const u8) ![:0]const u8 {
    const uri = try std.Uri.parse(url);
    
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send POST request
    const request = try std.fmt.allocPrint(
        allocator,
        "POST {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
    );
    defer allocator.free(request);
    
    _ = try conn.writeAll(request);
    
    // Read response
    var buffer: [16384]u8 = undefined;
    const bytes_read = try conn.read(&buffer);
    
    // Find body
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const response_body = buffer[body_start..bytes_read];
        
        const result = try allocator.allocSentinel(u8, response_body.len, 0);
        @memcpy(result[0..response_body.len], response_body);
        
        return result;
    }
    
    return "{}";
}

// For testing
pub fn main() !void {
    std.debug.print("🧪 Zig HTTP Library Test\n", .{});
    std.debug.print("Build with: zig build-lib zig_http.zig -dynamic\n", .{});
    std.debug.print("Use from Mojo via FFI\n", .{});
}
