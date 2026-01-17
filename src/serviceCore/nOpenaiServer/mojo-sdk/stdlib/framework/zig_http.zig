// Zig HTTP Server for Mojo Service Framework
// Days 105-109: Zig networking layer for Shimmy-Mojo
//
// This provides a high-performance HTTP server in Zig that bridges
// to Mojo handlers via FFI callbacks.

const std = @import("std");
const net = std.net;
const mem = std.mem;
const Allocator = std.mem.Allocator;

// ============================================================================
// Server Configuration
// ============================================================================

pub const ServerConfig = struct {
    host: []const u8 = "0.0.0.0",
    port: u16 = 8080,
    read_timeout_ms: u32 = 30000,
    write_timeout_ms: u32 = 30000,
    max_body_size: usize = 1024 * 1024, // 1MB
    max_header_size: usize = 8192,
    max_connections: usize = 1024,
};

// ============================================================================
// HTTP Request
// ============================================================================

pub const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    PATCH,
    OPTIONS,
    HEAD,

    pub fn fromString(s: []const u8) HttpMethod {
        if (mem.eql(u8, s, "GET")) return .GET;
        if (mem.eql(u8, s, "POST")) return .POST;
        if (mem.eql(u8, s, "PUT")) return .PUT;
        if (mem.eql(u8, s, "DELETE")) return .DELETE;
        if (mem.eql(u8, s, "PATCH")) return .PATCH;
        if (mem.eql(u8, s, "OPTIONS")) return .OPTIONS;
        if (mem.eql(u8, s, "HEAD")) return .HEAD;
        return .GET;
    }

    pub fn toString(self: HttpMethod) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .DELETE => "DELETE",
            .PATCH => "PATCH",
            .OPTIONS => "OPTIONS",
            .HEAD => "HEAD",
        };
    }
};

pub const HttpHeader = struct {
    name: []const u8,
    value: []const u8,
};

pub const HttpRequest = struct {
    method: HttpMethod,
    path: []const u8,
    query: ?[]const u8,
    headers: std.ArrayList(HttpHeader),
    body: []const u8,

    allocator: Allocator,

    pub fn init(allocator: Allocator) HttpRequest {
        return HttpRequest{
            .method = .GET,
            .path = "/",
            .query = null,
            .headers = std.ArrayList(HttpHeader).init(allocator),
            .body = "",
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HttpRequest) void {
        self.headers.deinit();
    }

    pub fn getHeader(self: *const HttpRequest, name: []const u8) ?[]const u8 {
        for (self.headers.items) |h| {
            if (std.ascii.eqlIgnoreCase(h.name, name)) {
                return h.value;
            }
        }
        return null;
    }

    pub fn contentType(self: *const HttpRequest) ?[]const u8 {
        return self.getHeader("Content-Type");
    }

    pub fn contentLength(self: *const HttpRequest) ?usize {
        if (self.getHeader("Content-Length")) |len_str| {
            return std.fmt.parseInt(usize, len_str, 10) catch null;
        }
        return null;
    }
};

// ============================================================================
// HTTP Response
// ============================================================================

pub const HttpResponse = struct {
    status_code: u16,
    status_text: []const u8,
    headers: std.ArrayList(HttpHeader),
    body: []const u8,

    allocator: Allocator,

    pub fn init(allocator: Allocator) HttpResponse {
        var resp = HttpResponse{
            .status_code = 200,
            .status_text = "OK",
            .headers = std.ArrayList(HttpHeader).init(allocator),
            .body = "",
            .allocator = allocator,
        };
        // Add default headers
        resp.addHeader("Content-Type", "application/json") catch {};
        return resp;
    }

    pub fn deinit(self: *HttpResponse) void {
        self.headers.deinit();
    }

    pub fn addHeader(self: *HttpResponse, name: []const u8, value: []const u8) !void {
        try self.headers.append(.{ .name = name, .value = value });
    }

    pub fn setStatus(self: *HttpResponse, code: u16, text: []const u8) void {
        self.status_code = code;
        self.status_text = text;
    }

    pub fn setBody(self: *HttpResponse, body: []const u8) void {
        self.body = body;
    }

    pub fn toBytes(self: *const HttpResponse, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        var writer = buffer.writer();

        // Status line
        try writer.print("HTTP/1.1 {d} {s}\r\n", .{ self.status_code, self.status_text });

        // Headers
        for (self.headers.items) |h| {
            try writer.print("{s}: {s}\r\n", .{ h.name, h.value });
        }

        // Content-Length
        try writer.print("Content-Length: {d}\r\n", .{self.body.len});

        // Empty line and body
        try writer.print("\r\n{s}", .{self.body});

        return buffer.toOwnedSlice();
    }
};

// ============================================================================
// Mojo Callback Interface
// ============================================================================

/// Callback function type for Mojo request handler
pub const MojoHandler = *const fn (
    method: [*:0]const u8,
    path: [*:0]const u8,
    body: [*]const u8,
    body_len: usize,
) callconv(.C) [*:0]u8;

/// Global handler set by Mojo
var mojo_handler: ?MojoHandler = null;

/// Register the Mojo request handler
export fn zig_http_set_handler(handler: MojoHandler) void {
    mojo_handler = handler;
}

// ============================================================================
// HTTP Parser
// ============================================================================

pub const HttpParser = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) HttpParser {
        return .{ .allocator = allocator };
    }

    pub fn parseRequest(self: *HttpParser, data: []const u8) !HttpRequest {
        var request = HttpRequest.init(self.allocator);

        // Find end of headers
        const header_end = mem.indexOf(u8, data, "\r\n\r\n") orelse return error.InvalidRequest;
        const headers_data = data[0..header_end];
        const body_start = header_end + 4;

        // Parse request line
        var lines = mem.splitSequence(u8, headers_data, "\r\n");
        const request_line = lines.first();

        var parts = mem.splitScalar(u8, request_line, ' ');
        const method_str = parts.next() orelse return error.InvalidRequest;
        const path_full = parts.next() orelse return error.InvalidRequest;

        request.method = HttpMethod.fromString(method_str);

        // Parse path and query
        if (mem.indexOf(u8, path_full, "?")) |query_start| {
            request.path = path_full[0..query_start];
            request.query = path_full[query_start + 1 ..];
        } else {
            request.path = path_full;
        }

        // Parse headers
        while (lines.next()) |line| {
            if (line.len == 0) continue;

            if (mem.indexOf(u8, line, ": ")) |sep| {
                const name = line[0..sep];
                const value = line[sep + 2 ..];
                try request.headers.append(.{ .name = name, .value = value });
            }
        }

        // Body
        if (body_start < data.len) {
            request.body = data[body_start..];
        }

        return request;
    }
};

// ============================================================================
// HTTP Server
// ============================================================================

pub const HttpServer = struct {
    config: ServerConfig,
    allocator: Allocator,
    listener: ?net.Server,
    running: bool,

    pub fn init(allocator: Allocator, config: ServerConfig) HttpServer {
        return HttpServer{
            .config = config,
            .allocator = allocator,
            .listener = null,
            .running = false,
        };
    }

    pub fn deinit(self: *HttpServer) void {
        if (self.listener) |*l| {
            l.deinit();
        }
    }

    pub fn start(self: *HttpServer) !void {
        const address = try net.Address.parseIp4(self.config.host, self.config.port);

        self.listener = try net.Address.listen(address, .{
            .reuse_address = true,
        });

        self.running = true;

        std.debug.print("🚀 Zig HTTP server listening on {s}:{d}\n", .{
            self.config.host,
            self.config.port,
        });

        while (self.running) {
            if (self.listener.?.accept()) |conn| {
                self.handleConnection(conn) catch |err| {
                    std.debug.print("Connection error: {}\n", .{err});
                };
            } else |err| {
                if (err == error.WouldBlock) continue;
                std.debug.print("Accept error: {}\n", .{err});
            }
        }
    }

    pub fn stop(self: *HttpServer) void {
        self.running = false;
    }

    fn handleConnection(self: *HttpServer, conn: net.Server.Connection) !void {
        defer conn.stream.close();

        var buffer: [8192]u8 = undefined;
        const bytes_read = try conn.stream.read(&buffer);

        if (bytes_read == 0) return;

        const data = buffer[0..bytes_read];

        // Parse request
        var parser = HttpParser.init(self.allocator);
        var request = try parser.parseRequest(data);
        defer request.deinit();

        // Create response
        var response = HttpResponse.init(self.allocator);
        defer response.deinit();

        // Call Mojo handler if registered
        if (mojo_handler) |handler| {
            // Create null-terminated strings for Mojo
            var method_buf: [16]u8 = undefined;
            const method_str = request.method.toString();
            @memcpy(method_buf[0..method_str.len], method_str);
            method_buf[method_str.len] = 0;

            var path_buf: [4096]u8 = undefined;
            @memcpy(path_buf[0..request.path.len], request.path);
            path_buf[request.path.len] = 0;

            // Call Mojo
            const response_ptr = handler(
                @ptrCast(&method_buf),
                @ptrCast(&path_buf),
                request.body.ptr,
                request.body.len,
            );

            // Use response from Mojo
            const response_len = mem.len(response_ptr);
            response.setBody(response_ptr[0..response_len]);
        } else {
            // Default response
            response.setBody("{\"error\":\"No handler registered\"}");
            response.setStatus(500, "Internal Server Error");
        }

        // Send response
        const response_bytes = try response.toBytes(self.allocator);
        defer self.allocator.free(response_bytes);

        _ = try conn.stream.write(response_bytes);
    }
};

// ============================================================================
// Exported Functions for Mojo
// ============================================================================

var global_server: ?*HttpServer = null;
var global_allocator: Allocator = undefined;

/// Initialize the HTTP server
export fn zig_http_init(
    host: [*:0]const u8,
    port: u16,
    read_timeout_ms: u32,
    write_timeout_ms: u32,
    max_body_size: usize,
) bool {
    global_allocator = std.heap.page_allocator;

    const config = ServerConfig{
        .host = mem.sliceTo(host, 0),
        .port = port,
        .read_timeout_ms = read_timeout_ms,
        .write_timeout_ms = write_timeout_ms,
        .max_body_size = max_body_size,
    };

    global_server = global_allocator.create(HttpServer) catch return false;
    global_server.?.* = HttpServer.init(global_allocator, config);

    return true;
}

/// Start the HTTP server (blocking)
export fn zig_http_start() bool {
    if (global_server) |server| {
        server.start() catch return false;
        return true;
    }
    return false;
}

/// Stop the HTTP server
export fn zig_http_stop() void {
    if (global_server) |server| {
        server.stop();
    }
}

/// Cleanup
export fn zig_http_deinit() void {
    if (global_server) |server| {
        server.deinit();
        global_allocator.destroy(server);
        global_server = null;
    }
}

// ============================================================================
// Response Builder for Mojo
// ============================================================================

/// Create a response string (JSON format)
export fn zig_http_create_response(
    status: u16,
    content_type: [*:0]const u8,
    body: [*]const u8,
    body_len: usize,
) [*:0]u8 {
    const allocator = std.heap.page_allocator;

    // For now, just return the body as-is
    // In a full implementation, we'd build proper HTTP response
    const result = allocator.allocSentinel(u8, body_len, 0) catch return @ptrCast("");
    @memcpy(result[0..body_len], body[0..body_len]);

    return result;
}

/// Free a response string
export fn zig_http_free_response(ptr: [*:0]u8) void {
    const allocator = std.heap.page_allocator;
    const len = mem.len(ptr);
    allocator.free(ptr[0 .. len + 1]);
}

// ============================================================================
// Tests
// ============================================================================

test "parse simple GET request" {
    const allocator = std.testing.allocator;
    var parser = HttpParser.init(allocator);

    const data = "GET /api/users HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\r\n";
    var request = try parser.parseRequest(data);
    defer request.deinit();

    try std.testing.expectEqual(HttpMethod.GET, request.method);
    try std.testing.expectEqualStrings("/api/users", request.path);
}

test "parse POST request with body" {
    const allocator = std.testing.allocator;
    var parser = HttpParser.init(allocator);

    const data = "POST /api/users HTTP/1.1\r\nContent-Type: application/json\r\n\r\n{\"name\":\"test\"}";
    var request = try parser.parseRequest(data);
    defer request.deinit();

    try std.testing.expectEqual(HttpMethod.POST, request.method);
    try std.testing.expectEqualStrings("/api/users", request.path);
    try std.testing.expectEqualStrings("{\"name\":\"test\"}", request.body);
}

test "response to bytes" {
    const allocator = std.testing.allocator;

    var response = HttpResponse.init(allocator);
    defer response.deinit();

    response.setBody("{\"status\":\"ok\"}");

    const bytes = try response.toBytes(allocator);
    defer allocator.free(bytes);

    try std.testing.expect(mem.indexOf(u8, bytes, "HTTP/1.1 200 OK") != null);
    try std.testing.expect(mem.indexOf(u8, bytes, "{\"status\":\"ok\"}") != null);
}
