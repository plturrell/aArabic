// Zig HTTP Server for Shimmy-Mojo LLM Inference
// OpenAI-compatible API with zero Python dependencies
// Exports C ABI functions that Mojo can call via FFI

const std = @import("std");
const http = std.http;
const net = std.net;
const mem = std.mem;

const header_line = "================================================================================";

var log_enabled: ?bool = null;

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

/// Initialize and start HTTP server for LLM inference
export fn zig_shimmy_serve(config: *const ServerConfig) callconv(.c) c_int {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("🦙 Shimmy-Mojo HTTP Server (Zig + Mojo)\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("🚀 Starting on port {d}\n", .{config.port});
    std.debug.print("🔥 Pure Mojo LLM inference backend\n", .{});
    std.debug.print("⚡ Zero Python dependencies\n\n", .{});
    
    // Start server
    startServer(config) catch |err| {
        std.debug.print("❌ Server error: {any}\n", .{err});
        return -1;
    };
    
    return 0;
}

/// Convenience entry point for Mojo callers (avoids struct construction).
export fn zig_shimmy_serve_simple(
    port: u16,
    host: [*:0]const u8,
    callback: RequestCallback,
) callconv(.c) c_int {
    const config = ServerConfig{
        .port = port,
        .host = host,
        .callback = callback,
    };
    return zig_shimmy_serve(&config);
}

fn startServer(config: *const ServerConfig) !void {
    // Parse address
    const host = mem.span(config.host);
    const addr = try net.Address.parseIp(host, config.port);
    
    // Create server with options
    var server = try addr.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();
    
    std.debug.print("✅ Server listening on {s}:{d}\n", .{host, config.port});
    std.debug.print("\n", .{});
    std.debug.print("📡 OpenAI-compatible endpoints:\n", .{});
    std.debug.print("   GET  /                        - Server info\n", .{});
    std.debug.print("   GET  /health                  - Health check\n", .{});
    std.debug.print("   GET  /v1/models               - List models\n", .{});
    std.debug.print("   POST /v1/chat/completions     - Chat API\n", .{});
    std.debug.print("   POST /v1/completions          - Completions API\n", .{});
    std.debug.print("   GET  /api/tags                - Ollama-compatible\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎯 Ready to handle requests...\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    
    // Accept connections loop
    while (true) {
        // Accept connection
        const conn = try server.accept();
        
        // Handle request (single-threaded for now)
        handleConnection(conn, config.callback) catch |err| {
            std.debug.print("⚠️  Connection error: {any}\n", .{err});
        };
    }
}

fn handleConnection(conn: net.Server.Connection, callback: RequestCallback) !void {
    defer conn.stream.close();
    
    // Larger buffer for LLM prompts
    var buffer: [16384]u8 = undefined;
    
    // Read request
    const bytes_read = try conn.stream.read(&buffer);
    if (bytes_read == 0) return;
    
    const request_data = buffer[0..bytes_read];
    
    // Parse HTTP request
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
    var body_buf: [8192]u8 = undefined;
    @memcpy(body_buf[0..body.len], body);
    body_buf[body.len] = 0;
    
    log("{s} {s} ({d} bytes)\n", .{method, path, body.len});
    
    // Call Mojo callback for inference
    const response_ptr = callback(
        @ptrCast(&method_buf),
        @ptrCast(&path_buf),
        @ptrCast(&body_buf),
        body.len
    );
    
    const response = mem.span(response_ptr);
    
    // Determine content type
    const content_type = if (mem.indexOf(u8, path, "/v1/") != null or 
                             mem.indexOf(u8, path, "/api/") != null)
        "application/json"
    else
        "application/json";
    
    // Send HTTP response with CORS
    const http_response = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "Server: Shimmy-Mojo/1.0 (Zig+Mojo)\r\n" ++
        "\r\n" ++
        "{s}",
        .{content_type, response.len, response}
    );
    defer allocator.free(http_response);
    
    _ = try conn.stream.writeAll(http_response);
    
    log("Response sent ({d} bytes)\n\n", .{response.len});
}

/// HTTP GET for calling external services (embeddings, etc)
export fn zig_shimmy_get(url: [*:0]const u8) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    
    const result = httpGet(url_str) catch |err| {
        std.debug.print("❌ GET error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn httpGet(url: []const u8) ![:0]const u8 {
    const uri = try std.Uri.parse(url);
    
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    const request = try std.fmt.allocPrint(
        allocator,
        "GET {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "User-Agent: Shimmy-Mojo/1.0\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded}
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

/// HTTP POST for calling external services (vector DB, etc)
export fn zig_shimmy_post(
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

/// ✅ P2-14 FIXED: HTTP PUT for updating resources
export fn zig_shimmy_put(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize
) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpPut(url_str, body_data) catch |err| {
        std.debug.print("❌ PUT error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

/// ✅ P2-14 FIXED: HTTP DELETE for removing resources
export fn zig_shimmy_delete(url: [*:0]const u8) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    
    const result = httpDelete(url_str) catch |err| {
        std.debug.print("❌ DELETE error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

/// ✅ P2-14 FIXED: HTTP PATCH for partial updates
export fn zig_shimmy_patch(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize
) callconv(.c) [*:0]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpPatch(url_str, body_data) catch |err| {
        std.debug.print("❌ PATCH error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn httpPost(url: []const u8, body: []const u8) ![:0]const u8 {
    return httpRequest("POST", url, body);
}

/// ✅ P2-14: HTTP PUT implementation
fn httpPut(url: []const u8, body: []const u8) ![:0]const u8 {
    return httpRequest("PUT", url, body);
}

/// ✅ P2-14: HTTP DELETE implementation
fn httpDelete(url: []const u8) ![:0]const u8 {
    return httpRequest("DELETE", url, "");
}

/// ✅ P2-14: HTTP PATCH implementation
fn httpPatch(url: []const u8, body: []const u8) ![:0]const u8 {
    return httpRequest("PATCH", url, body);
}

/// ✅ P2-14: Unified HTTP request function
fn httpRequest(method: []const u8, url: []const u8, body: []const u8) ![:0]const u8 {
    const uri = try std.Uri.parse(url);
    
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Build request based on whether there's a body
    const request = if (body.len > 0)
        try std.fmt.allocPrint(
            allocator,
            "{s} {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "User-Agent: Shimmy-Mojo/1.0\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Connection: close\r\n" ++
            "\r\n" ++
            "{s}",
            .{method, uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
        )
    else
        try std.fmt.allocPrint(
            allocator,
            "{s} {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "User-Agent: Shimmy-Mojo/1.0\r\n" ++
            "Connection: close\r\n" ++
            "\r\n",
            .{method, uri.path.percent_encoded, uri.host.?.percent_encoded}
        );
    defer allocator.free(request);
    
    _ = try conn.writeAll(request);
    
    var buffer: [16384]u8 = undefined;
    const bytes_read = try conn.read(&buffer);
    
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

// ============================================================================
// ✅ P3-20: Binary-Safe HTTP Functions
// ============================================================================

/// Binary-safe HTTP GET (returns pointer + length, handles embedded nulls)
export fn zig_shimmy_get_binary(
    url: [*:0]const u8,
    out_len: *c_int
) callconv(.c) [*]const u8 {
    const url_str = mem.span(url);
    
    const result = httpRequestBinary("GET", url_str, "") catch |err| {
        std.debug.print("❌ GET error: {any}\n", .{err});
        out_len.* = 2;
        return "{}".ptr;
    };
    
    out_len.* = @intCast(result.len);
    return result.ptr;
}

/// Binary-safe HTTP POST
export fn zig_shimmy_post_binary(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize,
    out_len: *c_int
) callconv(.c) [*]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpRequestBinary("POST", url_str, body_data) catch |err| {
        std.debug.print("❌ POST error: {any}\n", .{err});
        out_len.* = 2;
        return "{}".ptr;
    };
    
    out_len.* = @intCast(result.len);
    return result.ptr;
}

/// Binary-safe HTTP PUT
export fn zig_shimmy_put_binary(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize,
    out_len: *c_int
) callconv(.c) [*]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpRequestBinary("PUT", url_str, body_data) catch |err| {
        std.debug.print("❌ PUT error: {any}\n", .{err});
        out_len.* = 2;
        return "{}".ptr;
    };
    
    out_len.* = @intCast(result.len);
    return result.ptr;
}

/// Binary-safe HTTP DELETE
export fn zig_shimmy_delete_binary(
    url: [*:0]const u8,
    out_len: *c_int
) callconv(.c) [*]const u8 {
    const url_str = mem.span(url);
    
    const result = httpRequestBinary("DELETE", url_str, "") catch |err| {
        std.debug.print("❌ DELETE error: {any}\n", .{err});
        out_len.* = 2;
        return "{}".ptr;
    };
    
    out_len.* = @intCast(result.len);
    return result.ptr;
}

/// Binary-safe HTTP PATCH
export fn zig_shimmy_patch_binary(
    url: [*:0]const u8,
    body: [*:0]const u8,
    body_len: usize,
    out_len: *c_int
) callconv(.c) [*]const u8 {
    const url_str = mem.span(url);
    const body_data = body[0..body_len];
    
    const result = httpRequestBinary("PATCH", url_str, body_data) catch |err| {
        std.debug.print("❌ PATCH error: {any}\n", .{err});
        out_len.* = 2;
        return "{}".ptr;
    };
    
    out_len.* = @intCast(result.len);
    return result.ptr;
}

/// Binary-safe HTTP request (returns slice, not null-terminated)
fn httpRequestBinary(method: []const u8, url: []const u8, body: []const u8) ![]const u8 {
    const uri = try std.Uri.parse(url);
    
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 80
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Build request based on whether there's a body
    const request = if (body.len > 0)
        try std.fmt.allocPrint(
            allocator,
            "{s} {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "User-Agent: Shimmy-Mojo/1.0\r\n" ++
            "Content-Type: application/json\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Connection: close\r\n" ++
            "\r\n" ++
            "{s}",
            .{method, uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
        )
    else
        try std.fmt.allocPrint(
            allocator,
            "{s} {s} HTTP/1.1\r\n" ++
            "Host: {s}\r\n" ++
            "User-Agent: Shimmy-Mojo/1.0\r\n" ++
            "Connection: close\r\n" ++
            "\r\n",
            .{method, uri.path.percent_encoded, uri.host.?.percent_encoded}
        );
    defer allocator.free(request);
    
    _ = try conn.writeAll(request);
    
    // Read response (potentially larger for binary data)
    var buffer: [65536]u8 = undefined;
    const bytes_read = try conn.read(&buffer);
    
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const response_body = buffer[body_start..bytes_read];
        
        // Allocate exact size (not null-terminated for binary safety)
        const result = try allocator.alloc(u8, response_body.len);
        @memcpy(result, response_body);
        
        return result;
    }
    
    // Return empty JSON as fallback
    const fallback = try allocator.alloc(u8, 2);
    @memcpy(fallback, "{}");
    return fallback;
}

// Test/demo entry point
pub fn main() !void {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("🦙 Shimmy-Mojo HTTP Server (Zig Library)\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Build Instructions:\n", .{});
    std.debug.print("  macOS:  zig build-lib zig_http_shimmy.zig -dynamic -OReleaseFast\n", .{});
    std.debug.print("  Linux:  zig build-lib zig_http_shimmy.zig -dynamic -OReleaseFast\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Output:\n", .{});
    std.debug.print("  • libzig_http_shimmy.dylib (macOS)\n", .{});
    std.debug.print("  • libzig_http_shimmy.so (Linux)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Usage from Mojo:\n", .{});
    std.debug.print("  from sys.ffi import OwnedDLHandle\n", .{});
    std.debug.print("  var lib = OwnedDLHandle(\"./libzig_http_shimmy.dylib\")\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Features:\n", .{});
    std.debug.print("  ✅ OpenAI-compatible API\n", .{});
    std.debug.print("  ✅ Zero Python dependencies\n", .{});
    std.debug.print("  ✅ High-performance Zig HTTP\n", .{});
    std.debug.print("  ✅ Pure Mojo inference\n", .{});
    std.debug.print("  ✅ FFI callback architecture\n", .{});
    std.debug.print("  ✅ CORS support\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("{s}\n", .{header_line});
}
