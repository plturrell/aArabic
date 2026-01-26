//! HTTP Client Wrapper - Day 34
//!
//! Provides a simplified wrapper around std.http.Client for Keycloak integration.
//! Handles request/response lifecycle, headers, and error handling.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// HTTP response with owned data
pub const HttpResponse = struct {
    status_code: u16,
    body: []const u8,
    headers: std.StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator) HttpResponse {
        return HttpResponse{
            .status_code = 0,
            .body = &[_]u8{},
            .headers = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *HttpResponse, allocator: Allocator) void {
        if (self.body.len > 0) {
            allocator.free(self.body);
        }
        
        var iter = self.headers.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
    }
};

/// Simple HTTP client wrapper
pub const HttpClient = struct {
    allocator: Allocator,
    client: std.http.Client,
    
    pub fn init(allocator: Allocator) HttpClient {
        return HttpClient{
            .allocator = allocator,
            .client = std.http.Client{ .allocator = allocator },
        };
    }
    
    pub fn deinit(self: *HttpClient) void {
        self.client.deinit();
    }
    
    /// Make HTTP request
    pub fn request(
        self: *HttpClient,
        method: std.http.Method,
        url: []const u8,
        headers: ?std.StringHashMap([]const u8),
        body: ?[]const u8,
    ) !HttpResponse {
        // Parse URL
        const uri = try std.Uri.parse(url);
        
        // Prepare request headers
        var header_buffer: [4096]u8 = undefined;
        var req = try self.client.open(method, uri, .{
            .server_header_buffer = &header_buffer,
        });
        defer req.deinit();
        
        // Set headers
        if (headers) |hdrs| {
            var iter = hdrs.iterator();
            while (iter.next()) |entry| {
                try req.headers.append(entry.key_ptr.*, entry.value_ptr.*);
            }
        }
        
        // Send request
        try req.send();
        
        // Write body if present
        if (body) |b| {
            try req.writeAll(b);
        }
        
        try req.finish();
        
        // Wait for response
        try req.wait();
        
        // Read response body
        var response = HttpResponse.init(self.allocator);
        errdefer response.deinit(self.allocator);
        
        response.status_code = @intFromEnum(req.response.status);
        
        // Read body into buffer
        var body_buffer = std.ArrayList(u8){};
        defer body_buffer.deinit();
        
        var read_buffer: [4096]u8 = undefined;
        while (true) {
            const bytes_read = try req.read(&read_buffer);
            if (bytes_read == 0) break;
            try body_buffer.appendSlice(read_buffer[0..bytes_read]);
        }
        
        response.body = try body_buffer.toOwnedSlice();
        
        // Copy response headers
        var header_iter = req.response.iterateHeaders();
        while (header_iter.next()) |header| {
            const name_copy = try self.allocator.dupe(u8, header.name);
            errdefer self.allocator.free(name_copy);
            const value_copy = try self.allocator.dupe(u8, header.value);
            errdefer self.allocator.free(value_copy);
            
            try response.headers.put(name_copy, value_copy);
        }
        
        return response;
    }
    
    /// Helper for GET requests
    pub fn get(
        self: *HttpClient,
        url: []const u8,
        headers: ?std.StringHashMap([]const u8),
    ) !HttpResponse {
        return try self.request(.GET, url, headers, null);
    }
    
    /// Helper for POST requests
    pub fn post(
        self: *HttpClient,
        url: []const u8,
        headers: ?std.StringHashMap([]const u8),
        body: []const u8,
    ) !HttpResponse {
        return try self.request(.POST, url, headers, body);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "HttpClient initialization" {
    const allocator = std.testing.allocator;
    
    var client = HttpClient.init(allocator);
    defer client.deinit();
    
    try std.testing.expect(true);
}

test "HttpResponse init and deinit" {
    const allocator = std.testing.allocator;
    
    var response = HttpResponse.init(allocator);
    defer response.deinit(allocator);
    
    try std.testing.expectEqual(@as(u16, 0), response.status_code);
    try std.testing.expectEqual(@as(usize, 0), response.body.len);
}

test "HttpResponse with data" {
    const allocator = std.testing.allocator;
    
    var response = HttpResponse.init(allocator);
    defer response.deinit(allocator);
    
    response.status_code = 200;
    response.body = try allocator.dupe(u8, "test body");
    
    const name = try allocator.dupe(u8, "Content-Type");
    const value = try allocator.dupe(u8, "application/json");
    try response.headers.put(name, value);
    
    try std.testing.expectEqual(@as(u16, 200), response.status_code);
    try std.testing.expectEqualStrings("test body", response.body);
    try std.testing.expect(response.headers.count() > 0);
}
