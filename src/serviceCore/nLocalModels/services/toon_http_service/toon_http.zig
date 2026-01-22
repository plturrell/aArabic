// TOON HTTP Server - Pure Zig Implementation
// High-performance HTTP server for TOON encoding/decoding
// Zero external dependencies, integrates with existing zig_toon.zig

const std = @import("std");
const net = std.net;
const mem = std.mem;
const json = std.json;
const toon_encoder = @import("toon_encoder");

// ============================================================================
// Configuration
// ============================================================================

const Config = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = 8085,
    max_connections: usize = 100,
    buffer_size: usize = 1024 * 1024, // 1MB
    timeout_ms: u64 = 30000, // 30 seconds
};

// ============================================================================
// HTTP Request Parser
// ============================================================================

const HttpMethod = enum {
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
    OPTIONS,
    PATCH,
    UNKNOWN,

    pub fn fromString(s: []const u8) HttpMethod {
        if (mem.eql(u8, s, "GET")) return .GET;
        if (mem.eql(u8, s, "POST")) return .POST;
        if (mem.eql(u8, s, "PUT")) return .PUT;
        if (mem.eql(u8, s, "DELETE")) return .DELETE;
        if (mem.eql(u8, s, "HEAD")) return .HEAD;
        if (mem.eql(u8, s, "OPTIONS")) return .OPTIONS;
        if (mem.eql(u8, s, "PATCH")) return .PATCH;
        return .UNKNOWN;
    }
};

const HttpRequest = struct {
    method: HttpMethod,
    path: []const u8,
    version: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    allocator: mem.Allocator,

    pub fn deinit(self: *HttpRequest) void {
        self.headers.deinit();
    }
};

fn parseHttpRequest(allocator: mem.Allocator, data: []const u8) !HttpRequest {
    var lines_iter = mem.splitSequence(u8, data, "\r\n");
    
    // Parse request line: METHOD /path HTTP/1.1
    const request_line = lines_iter.next() orelse return error.InvalidRequest;
    var parts = mem.splitSequence(u8, request_line, " ");
    
    const method_str = parts.next() orelse return error.InvalidRequest;
    const path = parts.next() orelse return error.InvalidRequest;
    const version = parts.next() orelse return error.InvalidRequest;
    
    var headers = std.StringHashMap([]const u8).init(allocator);
    errdefer headers.deinit();
    
    // Parse headers
    while (lines_iter.next()) |line| {
        if (line.len == 0) break; // Empty line marks end of headers
        
        if (mem.indexOf(u8, line, ":")) |colon_pos| {
            const key = mem.trim(u8, line[0..colon_pos], " \t");
            const value = mem.trim(u8, line[colon_pos + 1..], " \t");
            try headers.put(key, value);
        }
    }
    
    // Rest is body
    const body_start = mem.indexOf(u8, data, "\r\n\r\n");
    const body = if (body_start) |pos| data[pos + 4..] else "";
    
    return HttpRequest{
        .method = HttpMethod.fromString(method_str),
        .path = path,
        .version = version,
        .headers = headers,
        .body = body,
        .allocator = allocator,
    };
}

// ============================================================================
// HTTP Response Builder
// ============================================================================

const HttpResponse = struct {
    status_code: u16,
    status_text: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    allocator: mem.Allocator,

    pub fn init(allocator: mem.Allocator) HttpResponse {
        return .{
            .status_code = 200,
            .status_text = "OK",
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = "",
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HttpResponse) void {
        self.headers.deinit();
    }

    pub fn setStatus(self: *HttpResponse, code: u16, text: []const u8) void {
        self.status_code = code;
        self.status_text = text;
    }

    pub fn setHeader(self: *HttpResponse, key: []const u8, value: []const u8) !void {
        try self.headers.put(key, value);
    }

    pub fn setBody(self: *HttpResponse, body: []const u8) void {
        self.body = body;
    }

    pub fn build(self: *HttpResponse) ![]u8 {
        var response = std.ArrayList(u8){};
        defer response.deinit(self.allocator);
        const writer = response.writer(self.allocator);

        // Status line
        try writer.print("HTTP/1.1 {d} {s}\r\n", .{ self.status_code, self.status_text });

        // Headers
        var headers_iter = self.headers.iterator();
        while (headers_iter.next()) |entry| {
            try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }

        // Content-Length if not already set
        if (!self.headers.contains("Content-Length")) {
            try writer.print("Content-Length: {d}\r\n", .{self.body.len});
        }

        // End of headers
        try writer.writeAll("\r\n");

        // Body
        if (self.body.len > 0) {
            try writer.writeAll(self.body);
        }

        return try response.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// JSON Helper Functions
// ============================================================================

fn parseJsonBody(allocator: mem.Allocator, body: []const u8) !json.Parsed(json.Value) {
    return json.parseFromSlice(json.Value, allocator, body, .{});
}

fn createJsonResponse(allocator: mem.Allocator, value: anytype) ![]u8 {
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit(allocator);
    try std.fmt.format(buffer.writer(allocator), "{f}", .{json.fmt(value, .{})});
    return try buffer.toOwnedSlice(allocator);
}

// ============================================================================
// Statistics Calculation
// ============================================================================

fn calculateTokenStats(allocator: mem.Allocator, json_str: []const u8, toon_str: []const u8) ![]u8 {
    // Simple token counting (approximate)
    const json_tokens = countTokens(json_str);
    const toon_tokens = countTokens(toon_str);
    const reduction = if (json_tokens > 0)
        @as(f64, @floatFromInt(json_tokens - toon_tokens)) / @as(f64, @floatFromInt(json_tokens)) * 100.0
    else
        0.0;

    const stats = .{
        .original_tokens = json_tokens,
        .toon_tokens = toon_tokens,
        .reduction_percent = reduction,
        .original_bytes = json_str.len,
        .toon_bytes = toon_str.len,
    };

    return createJsonResponse(allocator, stats);
}

fn countTokens(text: []const u8) usize {
    // Approximate token count: split by whitespace, punctuation, etc.
    var count: usize = 0;
    var in_token = false;

    for (text) |c| {
        const is_separator = c == ' ' or c == '\n' or c == '\t' or 
                           c == ',' or c == ':' or c == '{' or c == '}' or 
                           c == '[' or c == ']' or c == '"';
        
        if (is_separator) {
            if (in_token) {
                count += 1;
                in_token = false;
            }
            if (c != ' ' and c != '\n' and c != '\t') {
                count += 1; // Punctuation counts as token
            }
        } else {
            in_token = true;
        }
    }

    if (in_token) count += 1;
    return count;
}

// ============================================================================
// Request Handlers
// ============================================================================

fn handleEncode(allocator: mem.Allocator, request: *HttpRequest) ![]u8 {
    // Parse JSON body
    const parsed = try parseJsonBody(allocator, request.body);
    defer parsed.deinit();

    // Extract text field
    const text_value = parsed.value.object.get("text") orelse return error.MissingTextField;
    const text = text_value.string;

    // Call TOON encoder
    const json_z = try allocator.dupeZ(u8, text);
    defer allocator.free(json_z);

    const toon_ptr = toon_encoder.zig_toon_encode(json_z.ptr, text.len);
    const toon_str = mem.span(toon_ptr);

    // Build response
    const response_data = .{
        .toon = toon_str,
        .original_length = text.len,
        .toon_length = toon_str.len,
    };

    return createJsonResponse(allocator, response_data);
}

fn handleDecode(allocator: mem.Allocator, request: *HttpRequest) ![]u8 {
    // Parse JSON body
    const parsed = try parseJsonBody(allocator, request.body);
    defer parsed.deinit();

    // Extract text field
    const text_value = parsed.value.object.get("text") orelse return error.MissingTextField;
    const text = text_value.string;

    // Call TOON decoder
    const toon_z = try allocator.dupeZ(u8, text);
    defer allocator.free(toon_z);

    const json_ptr = toon_encoder.zig_toon_decode(toon_z.ptr, text.len);
    const json_str = mem.span(json_ptr);

    // Build response
    const response_data = .{
        .json = json_str,
        .original_length = text.len,
        .json_length = json_str.len,
    };

    return createJsonResponse(allocator, response_data);
}

fn handleEncodeWithStats(allocator: mem.Allocator, request: *HttpRequest) ![]u8 {
    // Parse JSON body
    const parsed = try parseJsonBody(allocator, request.body);
    defer parsed.deinit();

    // Extract text field
    const text_value = parsed.value.object.get("text") orelse return error.MissingTextField;
    const text = text_value.string;

    // Call TOON encoder
    const json_z = try allocator.dupeZ(u8, text);
    defer allocator.free(json_z);

    const toon_ptr = toon_encoder.zig_toon_encode(json_z.ptr, text.len);
    const toon_str = mem.span(toon_ptr);

    // Calculate statistics
    const stats_json = try calculateTokenStats(allocator, text, toon_str);
    defer allocator.free(stats_json);

    const stats_parsed = try parseJsonBody(allocator, stats_json);
    defer stats_parsed.deinit();

    // Build response with TOON and stats
    const response_data = .{
        .toon = toon_str,
        .statistics = stats_parsed.value,
    };

    return createJsonResponse(allocator, response_data);
}

fn handleHealth(allocator: mem.Allocator) ![]u8 {
    const response_data = .{
        .status = "healthy",
        .service = "toon-http-server",
        .version = "1.0.0",
        .uptime_seconds = getUptimeSeconds(), // ✅ FIXED: P1 Issue #10
    };

    return createJsonResponse(allocator, response_data);
}

fn handleRoot(allocator: mem.Allocator) ![]u8 {
    const response_data = .{
        .service = "TOON HTTP Server",
        .version = "1.0.0",
        .endpoints = .{
            .encode = "POST /encode - Encode JSON to TOON",
            .decode = "POST /decode - Decode TOON to JSON",
            .encode_with_stats = "POST /encode-with-stats - Encode with statistics",
            .health = "GET /health - Health check",
        },
    };

    return createJsonResponse(allocator, response_data);
}

// ============================================================================
// Router
// ============================================================================

fn routeRequest(allocator: mem.Allocator, request: *HttpRequest) !HttpResponse {
    var response = HttpResponse.init(allocator);
    errdefer response.deinit();

    // Set common headers
    try response.setHeader("Content-Type", "application/json");
    try response.setHeader("Server", "TOON-HTTP-Zig/1.0");
    try response.setHeader("Access-Control-Allow-Origin", "*");

    // Handle OPTIONS for CORS
    if (request.method == .OPTIONS) {
        try response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        try response.setHeader("Access-Control-Allow-Headers", "Content-Type");
        response.setStatus(204, "No Content");
        return response;
    }

    // Route to handlers
    if (request.method == .POST and mem.eql(u8, request.path, "/encode")) {
        const body = handleEncode(allocator, request) catch |err| {
            response.setStatus(400, "Bad Request");
            const error_msg = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            response.setBody(error_msg);
            return response;
        };
        response.setBody(body);
        return response;
    }

    if (request.method == .POST and mem.eql(u8, request.path, "/decode")) {
        const body = handleDecode(allocator, request) catch |err| {
            response.setStatus(400, "Bad Request");
            const error_msg = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            response.setBody(error_msg);
            return response;
        };
        response.setBody(body);
        return response;
    }

    if (request.method == .POST and mem.eql(u8, request.path, "/encode-with-stats")) {
        const body = handleEncodeWithStats(allocator, request) catch |err| {
            response.setStatus(400, "Bad Request");
            const error_msg = try std.fmt.allocPrint(allocator, "{{\"error\":\"{s}\"}}", .{@errorName(err)});
            response.setBody(error_msg);
            return response;
        };
        response.setBody(body);
        return response;
    }

    if (request.method == .GET and mem.eql(u8, request.path, "/health")) {
        const body = try handleHealth(allocator);
        response.setBody(body);
        return response;
    }

    if (request.method == .GET and mem.eql(u8, request.path, "/")) {
        const body = try handleRoot(allocator);
        response.setBody(body);
        return response;
    }

    // 404 Not Found
    response.setStatus(404, "Not Found");
    const error_msg = try std.fmt.allocPrint(allocator, "{{\"error\":\"Not Found\",\"path\":\"{s}\"}}", .{request.path});
    response.setBody(error_msg);
    return response;
}

// ============================================================================
// Connection Handler
// ============================================================================

fn handleConnection(allocator: mem.Allocator, stream: net.Stream, config: Config) !void {
    defer stream.close();

    var buffer = try allocator.alloc(u8, config.buffer_size);
    defer allocator.free(buffer);

    // Read request
    const bytes_read = try stream.read(buffer);
    if (bytes_read == 0) return;

    const request_data = buffer[0..bytes_read];

    // Parse HTTP request
    var request = parseHttpRequest(allocator, request_data) catch |err| {
        std.debug.print("Failed to parse request: {any}\n", .{err});
        
        // Send 400 Bad Request
        const error_response = "HTTP/1.1 400 Bad Request\r\nContent-Type: text/plain\r\nContent-Length: 11\r\n\r\nBad Request";
        _ = try stream.write(error_response);
        return;
    };
    defer request.deinit();

    std.debug.print("{s} {s}\n", .{ @tagName(request.method), request.path });

    // Route and handle request
    var response = try routeRequest(allocator, &request);
    defer response.deinit();

    // Build and send response
    const response_bytes = try response.build();
    defer allocator.free(response_bytes);

    _ = try stream.write(response_bytes);
}

// ============================================================================
// Server Uptime Tracking
// ============================================================================

var server_start_time: i64 = 0;

fn getUptimeSeconds() i64 {
    if (server_start_time == 0) return 0;
    const current_time = std.time.timestamp();
    return current_time - server_start_time;
}

// ============================================================================
// Server
// ============================================================================

pub fn startServer(config: Config) !void {
    // ✅ FIXED: P1 Issue #10 - Initialize server start time
    server_start_time = std.time.timestamp();
    
    const allocator = std.heap.page_allocator;

    const address = try net.Address.parseIp(config.host, config.port);
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();

    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("🚀 TOON HTTP Server Starting\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("   Host: {s}\n", .{config.host});
    std.debug.print("   Port: {d}\n", .{config.port});
    std.debug.print("   URL:  http://{s}:{d}\n", .{ config.host, config.port });
    std.debug.print("=" ** 80 ++ "\n\n", .{});
    std.debug.print("📋 Available Endpoints:\n", .{});
    std.debug.print("   GET  /              - Service information\n", .{});
    std.debug.print("   GET  /health        - Health check\n", .{});
    std.debug.print("   POST /encode        - Encode JSON to TOON\n", .{});
    std.debug.print("   POST /decode        - Decode TOON to JSON\n", .{});
    std.debug.print("   POST /encode-with-stats - Encode with statistics\n\n", .{});
    std.debug.print("✨ Server ready! Press Ctrl+C to stop.\n\n", .{});

    while (true) {
        const connection = server.accept() catch |err| {
            std.debug.print("Failed to accept connection: {any}\n", .{err});
            continue;
        };

        // Handle connection (synchronous for now, can be made async later)
        handleConnection(allocator, connection.stream, config) catch |err| {
            std.debug.print("Error handling connection: {any}\n", .{err});
        };
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

pub fn main() !void {
    var config = Config{};

    // Parse command line arguments
    var args = try std.process.argsWithAllocator(std.heap.page_allocator);
    defer args.deinit();

    _ = args.skip(); // Skip program name

    while (args.next()) |arg| {
        if (mem.eql(u8, arg, "--port")) {
            if (args.next()) |port_str| {
                config.port = try std.fmt.parseInt(u16, port_str, 10);
            }
        } else if (mem.eql(u8, arg, "--host")) {
            if (args.next()) |host| {
                config.host = host;
            }
        } else if (mem.eql(u8, arg, "--help")) {
            std.debug.print("TOON HTTP Server\n\n", .{});
            std.debug.print("Usage: toon_http [options]\n\n", .{});
            std.debug.print("Options:\n", .{});
            std.debug.print("  --host <host>    Server host (default: 127.0.0.1)\n", .{});
            std.debug.print("  --port <port>    Server port (default: 8085)\n", .{});
            std.debug.print("  --help           Show this help message\n", .{});
            return;
        }
    }

    try startServer(config);
}

// ============================================================================
// Tests
// ============================================================================

test "HTTP method parsing" {
    try std.testing.expectEqual(HttpMethod.GET, HttpMethod.fromString("GET"));
    try std.testing.expectEqual(HttpMethod.POST, HttpMethod.fromString("POST"));
    try std.testing.expectEqual(HttpMethod.UNKNOWN, HttpMethod.fromString("INVALID"));
}

test "Token counting" {
    const text = "hello world, test";
    const count = countTokens(text);
    try std.testing.expect(count > 0);
}
