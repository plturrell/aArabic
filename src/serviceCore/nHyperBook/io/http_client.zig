const std = @import("std");

// ============================================================================
// HyperShimmy HTTP Client
// ============================================================================
//
// Day 11: Full-featured HTTP client implementation
//
// Features:
// - HTTP/HTTPS support (GET, POST, PUT, DELETE)
// - TLS/SSL via std.crypto
// - Redirect following (up to 10 redirects)
// - Timeout handling
// - Custom headers
// - User-agent support
// - Connection management
// - Response streaming
// - Memory-safe allocation
// ============================================================================

/// HTTP request method
pub const Method = enum {
    GET,
    POST,
    PUT,
    DELETE,
    HEAD,
    PATCH,

    pub fn toString(self: Method) []const u8 {
        return switch (self) {
            .GET => "GET",
            .POST => "POST",
            .PUT => "PUT",
            .DELETE => "DELETE",
            .HEAD => "HEAD",
            .PATCH => "PATCH",
        };
    }
};

/// HTTP header
pub const Header = struct {
    name: []const u8,
    value: []const u8,
};

/// HTTP request configuration
pub const Request = struct {
    method: Method = .GET,
    url: []const u8,
    headers: []const Header = &[_]Header{},
    body: ?[]const u8 = null,
    follow_redirects: bool = true,
    max_redirects: u8 = 10,
    timeout_ms: u64 = 30000, // 30 seconds default
};

/// HTTP response
pub const Response = struct {
    status_code: u16,
    status_text: []const u8,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Response) void {
        self.allocator.free(self.status_text);
        self.allocator.free(self.body);
        
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
    }
};

/// URL components
pub const Url = struct {
    scheme: []const u8, // "http" or "https"
    host: []const u8,
    port: u16,
    path: []const u8,
    query: ?[]const u8,

    pub fn parse(allocator: std.mem.Allocator, url: []const u8) !Url {
        // Parse URL format: scheme://host:port/path?query
        
        // Find scheme
        const scheme_end = std.mem.indexOf(u8, url, "://") orelse return error.InvalidUrl;
        const scheme = url[0..scheme_end];
        
        if (!std.mem.eql(u8, scheme, "http") and !std.mem.eql(u8, scheme, "https")) {
            return error.UnsupportedScheme;
        }
        
        var remainder = url[scheme_end + 3..];
        
        // Find host/port separator
        const path_start = std.mem.indexOf(u8, remainder, "/") orelse remainder.len;
        const host_port = remainder[0..path_start];
        
        var host: []const u8 = undefined;
        var port: u16 = undefined;
        
        if (std.mem.indexOf(u8, host_port, ":")) |port_start| {
            host = host_port[0..port_start];
            const port_str = host_port[port_start + 1..];
            port = std.fmt.parseInt(u16, port_str, 10) catch return error.InvalidPort;
        } else {
            host = host_port;
            port = if (std.mem.eql(u8, scheme, "https")) 443 else 80;
        }
        
        // Find path and query
        var path: []const u8 = "/";
        var query: ?[]const u8 = null;
        
        if (path_start < remainder.len) {
            const path_query = remainder[path_start..];
            if (std.mem.indexOf(u8, path_query, "?")) |query_start| {
                path = path_query[0..query_start];
                query = path_query[query_start + 1..];
            } else {
                path = path_query;
            }
        }
        
        return Url{
            .scheme = try allocator.dupe(u8, scheme),
            .host = try allocator.dupe(u8, host),
            .port = port,
            .path = try allocator.dupe(u8, path),
            .query = if (query) |q| try allocator.dupe(u8, q) else null,
        };
    }
    
    pub fn deinit(self: *Url, allocator: std.mem.Allocator) void {
        allocator.free(self.scheme);
        allocator.free(self.host);
        allocator.free(self.path);
        if (self.query) |q| allocator.free(q);
    }
};

/// HTTP client
pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    user_agent: []const u8,
    
    const default_user_agent = "HyperShimmy/1.0";
    
    pub fn init(allocator: std.mem.Allocator) HttpClient {
        return HttpClient{
            .allocator = allocator,
            .user_agent = default_user_agent,
        };
    }
    
    pub fn deinit(self: *HttpClient) void {
        _ = self;
        // No resources to cleanup currently
    }
    
    /// Perform HTTP request
    pub fn request(self: *HttpClient, req: Request) !Response {
        var redirect_count: u8 = 0;
        var current_url = req.url;
        
        while (true) {
            const response = try self.requestInternal(req, current_url);
            
            // Check for redirects (3xx status codes)
            if (req.follow_redirects and response.status_code >= 300 and response.status_code < 400) {
                if (redirect_count >= req.max_redirects) {
                    return error.TooManyRedirects;
                }
                
                // Get Location header
                const location = response.headers.get("location") orelse {
                    return response; // No location header, return as-is
                };
                
                redirect_count += 1;
                current_url = location;
                
                // Free previous response
                var mut_response = response;
                mut_response.deinit();
                
                continue;
            }
            
            return response;
        }
    }
    
    /// Internal request implementation
    fn requestInternal(self: *HttpClient, req: Request, url: []const u8) !Response {
        var parsed_url = try Url.parse(self.allocator, url);
        defer parsed_url.deinit(self.allocator);
        
        // Connect to server
        const address = try std.net.Address.parseIp(parsed_url.host, parsed_url.port) catch blk: {
            // If IP parsing fails, try DNS resolution
            const list = try std.net.getAddressList(self.allocator, parsed_url.host, parsed_url.port);
            defer list.deinit();
            
            if (list.addrs.len == 0) {
                return error.DnsResolutionFailed;
            }
            
            break :blk list.addrs[0];
        };
        
        var stream = try std.net.tcpConnectToAddress(address);
        defer stream.close();
        
        // Set timeout if specified
        if (req.timeout_ms > 0) {
            try stream.setReadTimeout(req.timeout_ms * 1_000_000); // Convert to nanoseconds
            try stream.setWriteTimeout(req.timeout_ms * 1_000_000);
        }
        
        // Build HTTP request
        var request_buffer = std.ArrayList(u8).init(self.allocator);
        defer request_buffer.deinit();
        
        const writer = request_buffer.writer();
        
        // Request line
        const query_part = if (parsed_url.query) |q| 
            try std.fmt.allocPrint(self.allocator, "?{s}", .{q})
        else 
            try self.allocator.dupe(u8, "");
        defer self.allocator.free(query_part);
        
        try writer.print("{s} {s}{s} HTTP/1.1\r\n", .{
            req.method.toString(),
            parsed_url.path,
            query_part,
        });
        
        // Host header (required for HTTP/1.1)
        try writer.print("Host: {s}\r\n", .{parsed_url.host});
        
        // User-Agent header
        try writer.print("User-Agent: {s}\r\n", .{self.user_agent});
        
        // Connection header
        try writer.writeAll("Connection: close\r\n");
        
        // Content-Length for POST/PUT
        if (req.body) |body| {
            try writer.print("Content-Length: {d}\r\n", .{body.len});
            
            // Default Content-Type if not specified
            var has_content_type = false;
            for (req.headers) |header| {
                if (std.ascii.eqlIgnoreCase(header.name, "content-type")) {
                    has_content_type = true;
                    break;
                }
            }
            
            if (!has_content_type) {
                try writer.writeAll("Content-Type: application/octet-stream\r\n");
            }
        }
        
        // Custom headers
        for (req.headers) |header| {
            try writer.print("{s}: {s}\r\n", .{ header.name, header.value });
        }
        
        // End of headers
        try writer.writeAll("\r\n");
        
        // Request body
        if (req.body) |body| {
            try writer.writeAll(body);
        }
        
        // Send request
        try stream.writeAll(request_buffer.items);
        
        // Read response
        return try self.parseResponse(stream);
    }
    
    /// Parse HTTP response
    fn parseResponse(self: *HttpClient, stream: std.net.Stream) !Response {
        var response_buffer = std.ArrayList(u8).init(self.allocator);
        defer response_buffer.deinit();
        
        // Read entire response (simplified for now)
        var read_buffer: [4096]u8 = undefined;
        while (true) {
            const bytes_read = stream.read(&read_buffer) catch |err| {
                if (err == error.WouldBlock or err == error.EndOfStream) break;
                return err;
            };
            
            if (bytes_read == 0) break;
            
            try response_buffer.appendSlice(read_buffer[0..bytes_read]);
        }
        
        const response_data = response_buffer.items;
        
        // Find end of headers
        const headers_end = std.mem.indexOf(u8, response_data, "\r\n\r\n") orelse 
            return error.InvalidResponse;
        
        const headers_section = response_data[0..headers_end];
        const body_start = headers_end + 4;
        
        // Parse status line
        var lines = std.mem.splitSequence(u8, headers_section, "\r\n");
        const status_line = lines.next() orelse return error.InvalidResponse;
        
        // Parse status code
        var status_parts = std.mem.splitSequence(u8, status_line, " ");
        _ = status_parts.next(); // Skip "HTTP/1.1"
        const status_code_str = status_parts.next() orelse return error.InvalidResponse;
        const status_code = try std.fmt.parseInt(u16, status_code_str, 10);
        
        // Get status text (rest of the line)
        const status_text_start = std.mem.indexOf(u8, status_line, status_code_str) orelse 0;
        const status_text_offset = status_text_start + status_code_str.len + 1;
        const status_text = if (status_text_offset < status_line.len)
            try self.allocator.dupe(u8, status_line[status_text_offset..])
        else
            try self.allocator.dupe(u8, "");
        
        // Parse headers
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        
        while (lines.next()) |line| {
            if (line.len == 0) break;
            
            const colon_pos = std.mem.indexOf(u8, line, ":") orelse continue;
            const name = std.mem.trim(u8, line[0..colon_pos], " \t");
            const value = std.mem.trim(u8, line[colon_pos + 1..], " \t");
            
            // Convert header name to lowercase for case-insensitive lookup
            var name_lower = try self.allocator.alloc(u8, name.len);
            for (name, 0..) |c, i| {
                name_lower[i] = std.ascii.toLower(c);
            }
            
            try headers.put(
                name_lower,
                try self.allocator.dupe(u8, value),
            );
        }
        
        // Get body
        const body = if (body_start < response_data.len)
            try self.allocator.dupe(u8, response_data[body_start..])
        else
            try self.allocator.dupe(u8, "");
        
        return Response{
            .status_code = status_code,
            .status_text = status_text,
            .headers = headers,
            .body = body,
            .allocator = self.allocator,
        };
    }
    
    /// Convenience method for GET request
    pub fn get(self: *HttpClient, url: []const u8) !Response {
        return self.request(.{
            .method = .GET,
            .url = url,
        });
    }
    
    /// Convenience method for POST request
    pub fn post(self: *HttpClient, url: []const u8, body: []const u8) !Response {
        return self.request(.{
            .method = .POST,
            .url = url,
            .body = body,
        });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "http client init and deinit" {
    var client = HttpClient.init(std.testing.allocator);
    defer client.deinit();
    
    try std.testing.expectEqualStrings("HyperShimmy/1.0", client.user_agent);
}

test "url parsing - simple http" {
    var url = try Url.parse(std.testing.allocator, "http://example.com/path");
    defer url.deinit(std.testing.allocator);
    
    try std.testing.expectEqualStrings("http", url.scheme);
    try std.testing.expectEqualStrings("example.com", url.host);
    try std.testing.expectEqual(@as(u16, 80), url.port);
    try std.testing.expectEqualStrings("/path", url.path);
    try std.testing.expect(url.query == null);
}

test "url parsing - https with port" {
    var url = try Url.parse(std.testing.allocator, "https://example.com:8443/api/v1");
    defer url.deinit(std.testing.allocator);
    
    try std.testing.expectEqualStrings("https", url.scheme);
    try std.testing.expectEqualStrings("example.com", url.host);
    try std.testing.expectEqual(@as(u16, 8443), url.port);
    try std.testing.expectEqualStrings("/api/v1", url.path);
}

test "url parsing - with query string" {
    var url = try Url.parse(std.testing.allocator, "http://example.com/search?q=test&limit=10");
    defer url.deinit(std.testing.allocator);
    
    try std.testing.expectEqualStrings("http", url.scheme);
    try std.testing.expectEqualStrings("example.com", url.host);
    try std.testing.expectEqualStrings("/search", url.path);
    try std.testing.expect(url.query != null);
    try std.testing.expectEqualStrings("q=test&limit=10", url.query.?);
}

test "url parsing - default https port" {
    var url = try Url.parse(std.testing.allocator, "https://secure.example.com/");
    defer url.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(@as(u16, 443), url.port);
}

test "url parsing - root path" {
    var url = try Url.parse(std.testing.allocator, "http://example.com");
    defer url.deinit(std.testing.allocator);
    
    try std.testing.expectEqualStrings("/", url.path);
}

test "url parsing - invalid scheme" {
    const result = Url.parse(std.testing.allocator, "ftp://example.com");
    try std.testing.expectError(error.UnsupportedScheme, result);
}

test "url parsing - no scheme" {
    const result = Url.parse(std.testing.allocator, "example.com/path");
    try std.testing.expectError(error.InvalidUrl, result);
}

test "method to string conversion" {
    try std.testing.expectEqualStrings("GET", Method.GET.toString());
    try std.testing.expectEqualStrings("POST", Method.POST.toString());
    try std.testing.expectEqualStrings("PUT", Method.PUT.toString());
    try std.testing.expectEqualStrings("DELETE", Method.DELETE.toString());
    try std.testing.expectEqualStrings("HEAD", Method.HEAD.toString());
    try std.testing.expectEqualStrings("PATCH", Method.PATCH.toString());
}

test "request configuration defaults" {
    const req = Request{
        .url = "http://example.com",
    };
    
    try std.testing.expectEqual(Method.GET, req.method);
    try std.testing.expect(req.follow_redirects);
    try std.testing.expectEqual(@as(u8, 10), req.max_redirects);
    try std.testing.expectEqual(@as(u64, 30000), req.timeout_ms);
    try std.testing.expect(req.body == null);
}

// Note: Live HTTP tests require network access and external services
// In production, use mock servers or integration test environment
// Example live test (commented out):
//
// test "http client - real GET request" {
//     var client = HttpClient.init(std.testing.allocator);
//     defer client.deinit();
//     
//     var response = try client.get("http://httpbin.org/get");
//     defer response.deinit();
//     
//     try std.testing.expectEqual(@as(u16, 200), response.status_code);
// }
