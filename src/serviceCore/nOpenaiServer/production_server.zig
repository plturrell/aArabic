// Production Zig Reverse Proxy + Static File Server
// Serves webapp files + proxies API to openai_http_server:11434
// Pure Zig solution - no NGINX/Caddy dependencies

const std = @import("std");
const net = std.net;
const fs = std.fs;
const mem = std.mem;

fn getMimeType(path: []const u8) []const u8 {
    if (mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (mem.endsWith(u8, path, ".css")) return "text/css";
    if (mem.endsWith(u8, path, ".js")) return "application/javascript";
    if (mem.endsWith(u8, path, ".json")) return "application/json";
    if (mem.endsWith(u8, path, ".xml")) return "application/xml";
    if (mem.endsWith(u8, path, ".png")) return "image/png";
    if (mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (mem.endsWith(u8, path, ".ico")) return "image/x-icon";
    if (mem.endsWith(u8, path, ".woff")) return "font/woff";
    if (mem.endsWith(u8, path, ".woff2")) return "font/woff2";
    return "application/octet-stream";
}

fn sendResponse(stream: net.Stream, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        404 => "Not Found",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        else => "OK",
    };
    
    var header_buf: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{ status, reason, content_type, body.len },
    );
    
    _ = try stream.writeAll(header);
    _ = try stream.writeAll(body);
}

fn proxyToOpenAI(client_stream: net.Stream, allocator: std.mem.Allocator, method: []const u8, path: []const u8, body: []const u8) !void {
    // Connect to openai_http_server on localhost:11434
    const api_addr = try net.Address.parseIp("127.0.0.1", 11434);
    const api_stream = net.tcpConnectToAddress(api_addr) catch |err| {
        std.debug.print("❌ Failed to connect to OpenAI API: {}\n", .{err});
        try sendResponse(client_stream, 502, "application/json", "{\"error\":\"API server unavailable\"}");
        return;
    };
    defer api_stream.close();
    
    // Build proxy request
    const proxy_request = try std.fmt.allocPrint(allocator,
        "{s} {s} HTTP/1.1\r\n" ++
        "Host: localhost:11434\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ method, path, body.len, body },
    );
    defer allocator.free(proxy_request);
    
    // Send to OpenAI API
    _ = try api_stream.writeAll(proxy_request);
    
    // Read full response from API
    var response_list = std.ArrayList(u8).empty;
    defer response_list.deinit(allocator);
    
    var temp_buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = api_stream.read(&temp_buf) catch |err| {
            if (err == error.ConnectionResetByPeer or err == error.BrokenPipe) break;
            return err;
        };
        if (bytes_read == 0) break;
        try response_list.appendSlice(allocator, temp_buf[0..bytes_read]);
    }
    
    const full_response = response_list.items;
    
    if (full_response.len > 0) {
        // Forward complete response to client
        _ = try client_stream.writeAll(full_response);
        std.debug.print("✅ Proxied {s} {s} → {d} bytes\n", .{ method, path, full_response.len });
    }
}

fn serveStaticFile(stream: net.Stream, allocator: std.mem.Allocator, requested_path: []const u8) !void {
    // Security: block directory traversal
    if (mem.indexOf(u8, requested_path, "..") != null) {
        try sendResponse(stream, 404, "text/plain", "Not Found");
        return;
    }
    
    // Map URL path to file path
    var file_path: []const u8 = undefined;
    if (mem.eql(u8, requested_path, "/")) {
        file_path = "webapp/index.html";
    } else {
        const clean_path = if (mem.startsWith(u8, requested_path, "/")) 
            requested_path[1..] else requested_path;
        file_path = try std.fmt.allocPrint(allocator, "webapp/{s}", .{clean_path});
        defer allocator.free(file_path);
        
        return serveFile(stream, allocator, file_path);
    }
    
    try serveFile(stream, allocator, file_path);
}

fn serveFile(stream: net.Stream, allocator: std.mem.Allocator, file_path: []const u8) !void {
    const file = fs.cwd().openFile(file_path, .{}) catch {
        try sendResponse(stream, 404, "text/plain", "File Not Found");
        return;
    };
    defer file.close();
    
    const file_size = try file.getEndPos();
    const content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);
    
    _ = try file.readAll(content);
    
    const mime_type = getMimeType(file_path);
    try sendResponse(stream, 200, mime_type, content);
}

fn handleConnection(stream: net.Stream, allocator: std.mem.Allocator) !void {
    defer stream.close();
    
    // Read HTTP request
    var buffer: [8192]u8 = undefined;
    const n = stream.read(&buffer) catch |err| {
        std.debug.print("❌ Read error: {}\n", .{err});
        return;
    };
    
    if (n == 0) return;
    
    const request = buffer[0..n];
    
    // Parse request line
    const first_line_end = mem.indexOf(u8, request, "\r\n") orelse return;
    const request_line = request[0..first_line_end];
    
    var parts = mem.splitSequence(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    
    std.debug.print("📥 {s} {s}\n", .{ method, path });
    
    // Handle OPTIONS for CORS
    if (mem.eql(u8, method, "OPTIONS")) {
        try sendResponse(stream, 200, "text/plain", "");
        return;
    }
    
    // Extract body if present
    var body: []const u8 = "";
    if (mem.indexOf(u8, request, "\r\n\r\n")) |idx| {
        body = request[idx + 4 ..];
    }
    
    // Route based on path
    // API proxy - forward to openai_http_server:11434
    // Strip /api prefix if present
    var proxy_path = path;
    if (mem.startsWith(u8, path, "/api/v1/")) {
        proxy_path = path[4..]; // Remove "/api" prefix
    }
    
    if (mem.startsWith(u8, path, "/api/v1/") or 
        mem.startsWith(u8, path, "/v1/") or
        mem.eql(u8, path, "/health") or
        mem.eql(u8, path, "/metrics")) {
        try proxyToOpenAI(stream, allocator, method, proxy_path, body);
        return;
    }
    
    // WebSocket upgrade (proxy to OpenAI server)
    if (mem.eql(u8, path, "/ws")) {
        if (mem.indexOf(u8, request, "Upgrade:") != null) {
            try proxyToOpenAI(stream, allocator, method, path, body);
            return;
        }
    }
    
    // Serve static files for everything else
    if (mem.eql(u8, method, "GET")) {
        try serveStaticFile(stream, allocator, path);
        return;
    }
    
    try sendResponse(stream, 404, "text/plain", "Not Found");
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const address = try net.Address.parseIp("0.0.0.0", 8080);
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();
    
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("🚀 Production Zig Server - Static Files + API Proxy\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("🌐 Frontend: http://localhost:8080\n", .{});
    std.debug.print("📁 Static: webapp/*\n", .{});
    std.debug.print("🔌 Proxy: /api/v1/* → localhost:11434\n", .{});
    std.debug.print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n", .{});
    std.debug.print("✅ Ready! Listening on port 8080...\n\n", .{});
    
    while (true) {
        const connection = server.accept() catch |err| {
            std.debug.print("❌ Accept error: {}\n", .{err});
            continue;
        };
        
        handleConnection(connection.stream, allocator) catch |err| {
            std.debug.print("❌ Handler error: {}\n", .{err});
        };
    }
}
