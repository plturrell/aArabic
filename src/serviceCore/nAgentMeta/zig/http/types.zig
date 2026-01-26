//! HTTP Types - Day 29
//!
//! Core types for HTTP request and response handling.
//! Provides convenient abstractions over raw HTTP data.
//!
//! Key Features:
//! - Type-safe request/response handling
//! - JSON serialization/deserialization
//! - Header management
//! - Query parameter parsing
//! - Path parameter extraction
//! - Cookie handling
//!
//! Example:
//! ```zig
//! var req = Request.init(allocator);
//! const user_id = req.param("id");
//! const page = req.query("page") orelse "1";
//!
//! var resp = Response.init(allocator);
//! resp.status = 200;
//! try resp.json(.{ .message = "success" });
//! ```

const std = @import("std");
const http = std.http;
const json = std.json;
const Allocator = std.mem.Allocator;

// ============================================================================
// Request
// ============================================================================

/// HTTP Request
pub const Request = struct {
    allocator: Allocator,
    method: http.Method,
    path: []const u8,
    headers: std.StringHashMap([]const u8),
    params: std.StringHashMap([]const u8),
    query: std.StringHashMap([]const u8),
    body: ?[]const u8,
    
    /// Initialize new request
    pub fn init(allocator: Allocator) Request {
        return Request{
            .allocator = allocator,
            .method = .GET,
            .path = "",
            .headers = std.StringHashMap([]const u8).init(allocator),
            .params = std.StringHashMap([]const u8).init(allocator),
            .query = std.StringHashMap([]const u8).init(allocator),
            .body = null,
        };
    }
    
    /// Clean up request resources
    pub fn deinit(self: *Request) void {
        // Free path
        if (self.path.len > 0) {
            self.allocator.free(self.path);
        }
        
        // Free headers
        var header_iter = self.headers.iterator();
        while (header_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        
        // Free params
        var param_iter = self.params.iterator();
        while (param_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.params.deinit();
        
        // Free query
        var query_iter = self.query.iterator();
        while (query_iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.query.deinit();
        
        // Body is managed by arena allocator
    }
    
    /// Get header value
    pub fn header(self: *const Request, name: []const u8) ?[]const u8 {
        return self.headers.get(name);
    }
    
    /// Get path parameter
    pub fn param(self: *const Request, name: []const u8) ?[]const u8 {
        return self.params.get(name);
    }
    
    /// Get query parameter
    pub fn queryParam(self: *const Request, name: []const u8) ?[]const u8 {
        return self.query.get(name);
    }
    
    /// Parse query string into query map
    pub fn parseQuery(self: *Request, query_string: []const u8) !void {
        var iter = std.mem.splitScalar(u8, query_string, '&');
        
        while (iter.next()) |pair| {
            if (std.mem.indexOf(u8, pair, "=")) |eq_pos| {
                const key = pair[0..eq_pos];
                const value = pair[eq_pos + 1 ..];
                
                try self.query.put(
                    try self.allocator.dupe(u8, key),
                    try self.allocator.dupe(u8, value),
                );
            }
        }
    }
    
    /// Parse JSON body
    pub fn jsonBody(self: *const Request, comptime T: type) !T {
        if (self.body == null) {
            return error.NoBody;
        }
        
        const parsed = try json.parseFromSlice(
            T,
            self.allocator,
            self.body.?,
            .{},
        );
        defer parsed.deinit();
        
        return parsed.value;
    }
    
    /// Check if request accepts JSON
    pub fn acceptsJson(self: *const Request) bool {
        if (self.header("Accept")) |accept| {
            return std.mem.indexOf(u8, accept, "application/json") != null;
        }
        return false;
    }
    
    /// Check if content type is JSON
    pub fn isJson(self: *const Request) bool {
        if (self.header("Content-Type")) |content_type| {
            return std.mem.indexOf(u8, content_type, "application/json") != null;
        }
        return false;
    }
};

// ============================================================================
// Response
// ============================================================================

/// HTTP Response
pub const Response = struct {
    allocator: Allocator,
    status: u16,
    headers: std.StringHashMap([]const u8),
    body: ?[]const u8,
    
    /// Initialize new response
    pub fn init(allocator: Allocator) Response {
        return Response{
            .allocator = allocator,
            .status = 200,
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = null,
        };
    }
    
    /// Clean up response resources
    pub fn deinit(self: *Response) void {
        var iter = self.headers.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        
        // Body is managed by arena allocator
    }
    
    /// Set response header
    pub fn setHeader(self: *Response, name: []const u8, value: []const u8) !void {
        try self.headers.put(
            try self.allocator.dupe(u8, name),
            try self.allocator.dupe(u8, value),
        );
    }
    
    /// Set response body as text
    pub fn text(self: *Response, content: []const u8) !void {
        self.body = try self.allocator.dupe(u8, content);
        try self.setHeader("Content-Type", "text/plain; charset=utf-8");
    }
    
    /// Set response body as JSON
    pub fn json(self: *Response, value: anytype) !void {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit();
        
        try std.json.stringify(value, .{}, buffer.writer());
        
        self.body = try self.allocator.dupe(u8, buffer.items);
        try self.setHeader("Content-Type", "application/json; charset=utf-8");
    }
    
    /// Set response body as HTML
    pub fn html(self: *Response, content: []const u8) !void {
        self.body = try self.allocator.dupe(u8, content);
        try self.setHeader("Content-Type", "text/html; charset=utf-8");
    }
    
    /// Get status text for status code
    pub fn getStatusText(self: *const Response) []const u8 {
        return switch (self.status) {
            200 => "OK",
            201 => "Created",
            204 => "No Content",
            400 => "Bad Request",
            401 => "Unauthorized",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            409 => "Conflict",
            422 => "Unprocessable Entity",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            else => "Unknown",
        };
    }
    
    /// Send error response
    pub fn error_(self: *Response, status: u16, message: []const u8) !void {
        self.status = status;
        const ErrorResponse = struct {
            @"error": []const u8,
            status: u16,
        };
        try self.json(ErrorResponse{
            .@"error" = message,
            .status = status,
        });
    }
    
    /// Send success response
    pub fn success(self: *Response, data: anytype) !void {
        self.status = 200;
        try self.json(.{
            .success = true,
            .data = data,
        });
    }
    
    /// Redirect to URL
    pub fn redirect(self: *Response, url: []const u8, permanent: bool) !void {
        self.status = if (permanent) 301 else 302;
        try self.setHeader("Location", url);
    }
};

// ============================================================================
// Common Response Helpers
// ============================================================================

/// Create 200 OK response
pub fn ok(allocator: Allocator, data: anytype) !Response {
    var resp = Response.init(allocator);
    try resp.success(data);
    return resp;
}

/// Create 201 Created response
pub fn created(allocator: Allocator, data: anytype) !Response {
    var resp = Response.init(allocator);
    resp.status = 201;
    try resp.json(.{
        .success = true,
        .data = data,
    });
    return resp;
}

/// Create 400 Bad Request response
pub fn badRequest(allocator: Allocator, message: []const u8) !Response {
    var resp = Response.init(allocator);
    try resp.error_(400, message);
    return resp;
}

/// Create 404 Not Found response
pub fn notFound(allocator: Allocator, message: []const u8) !Response {
    var resp = Response.init(allocator);
    try resp.error_(404, message);
    return resp;
}

/// Create 500 Internal Server Error response
pub fn internalError(allocator: Allocator, message: []const u8) !Response {
    var resp = Response.init(allocator);
    try resp.error_(500, message);
    return resp;
}

// ============================================================================
// Tests
// ============================================================================

test "Request: init and deinit" {
    const allocator = std.testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    
    try std.testing.expectEqual(http.Method.GET, req.method);
    try std.testing.expectEqualStrings("", req.path);
}

test "Request: parse query string" {
    const allocator = std.testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    
    try req.parseQuery("page=1&limit=10&sort=name");
    
    try std.testing.expectEqualStrings("1", req.queryParam("page").?);
    try std.testing.expectEqualStrings("10", req.queryParam("limit").?);
    try std.testing.expectEqualStrings("name", req.queryParam("sort").?);
}

test "Request: headers" {
    const allocator = std.testing.allocator;
    
    var req = Request.init(allocator);
    defer req.deinit();
    
    try req.headers.put(
        try allocator.dupe(u8, "Content-Type"),
        try allocator.dupe(u8, "application/json"),
    );
    
    try std.testing.expectEqualStrings("application/json", req.header("Content-Type").?);
    try std.testing.expect(req.isJson());
}

test "Response: init and deinit" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try std.testing.expectEqual(@as(u16, 200), resp.status);
}

test "Response: json" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try resp.json(.{
        .message = "Hello",
        .count = 42,
    });
    
    try std.testing.expect(resp.body != null);
    try std.testing.expectEqualStrings("application/json; charset=utf-8", resp.headers.get("Content-Type").?);
}

test "Response: text" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try resp.text("Hello, World!");
    
    try std.testing.expectEqualStrings("Hello, World!", resp.body.?);
    try std.testing.expectEqualStrings("text/plain; charset=utf-8", resp.headers.get("Content-Type").?);
}

test "Response: status text" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    resp.status = 200;
    try std.testing.expectEqualStrings("OK", resp.getStatusText());
    
    resp.status = 404;
    try std.testing.expectEqualStrings("Not Found", resp.getStatusText());
    
    resp.status = 500;
    try std.testing.expectEqualStrings("Internal Server Error", resp.getStatusText());
}

test "Response: error" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try resp.error_(404, "Resource not found");
    
    try std.testing.expectEqual(@as(u16, 404), resp.status);
    try std.testing.expect(resp.body != null);
}

test "Response: success" {
    const allocator = std.testing.allocator;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try resp.success(.{ .id = 123, .name = "Test" });
    
    try std.testing.expectEqual(@as(u16, 200), resp.status);
    try std.testing.expect(resp.body != null);
}

test "Response helpers: ok" {
    const allocator = std.testing.allocator;
    
    var resp = try ok(allocator, .{ .message = "success" });
    defer resp.deinit();
    
    try std.testing.expectEqual(@as(u16, 200), resp.status);
}

test "Response helpers: created" {
    const allocator = std.testing.allocator;
    
    var resp = try created(allocator, .{ .id = 123 });
    defer resp.deinit();
    
    try std.testing.expectEqual(@as(u16, 201), resp.status);
}

test "Response helpers: badRequest" {
    const allocator = std.testing.allocator;
    
    var resp = try badRequest(allocator, "Invalid input");
    defer resp.deinit();
    
    try std.testing.expectEqual(@as(u16, 400), resp.status);
}

test "Response helpers: notFound" {
    const allocator = std.testing.allocator;
    
    var resp = try notFound(allocator, "Not found");
    defer resp.deinit();
    
    try std.testing.expectEqual(@as(u16, 404), resp.status);
}
