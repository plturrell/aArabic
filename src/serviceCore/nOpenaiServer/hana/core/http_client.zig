const std = @import("std");

/// Enhanced HTTP client for HANA Cloud OData operations
/// Supports CSRF tokens, Basic Auth, and custom headers

pub const HttpMethod = enum {
    GET,
    POST,
    PATCH,
    DELETE,
    HEAD,
};

pub const HttpRequest = struct {
    method: HttpMethod,
    url: []const u8,
    body: ?[]const u8 = null,
    headers: std.StringHashMap([]const u8),
    
    pub fn init(allocator: std.mem.Allocator, method: HttpMethod, url: []const u8) HttpRequest {
        return .{
            .method = method,
            .url = url,
            .headers = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *HttpRequest) void {
        self.headers.deinit();
    }
    
    pub fn addHeader(self: *HttpRequest, key: []const u8, value: []const u8) !void {
        try self.headers.put(key, value);
    }
    
    pub fn setBody(self: *HttpRequest, body: []const u8) void {
        self.body = body;
    }
};

pub const HttpResponse = struct {
    status_code: u16,
    headers: std.StringHashMap([]const u8),
    body: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) HttpResponse {
        return .{
            .status_code = 0,
            .headers = std.StringHashMap([]const u8).init(allocator),
            .body = &[_]u8{},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *HttpResponse) void {
        var it = self.headers.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.headers.deinit();
        if (self.body.len > 0) {
            self.allocator.free(self.body);
        }
    }
    
    pub fn getHeader(self: *const HttpResponse, key: []const u8) ?[]const u8 {
        return self.headers.get(key);
    }
};

/// HTTP Client with HANA Cloud support
pub const HttpClient = struct {
    allocator: std.mem.Allocator,
    username: ?[]const u8 = null,
    password: ?[]const u8 = null,
    csrf_token: ?[]const u8 = null,
    
    pub fn init(allocator: std.mem.Allocator) HttpClient {
        return .{
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *HttpClient) void {
        if (self.csrf_token) |token| {
            self.allocator.free(token);
        }
    }
    
    pub fn setBasicAuth(self: *HttpClient, username: []const u8, password: []const u8) void {
        self.username = username;
        self.password = password;
    }
    
    /// Fetch CSRF token from HANA Cloud
    pub fn fetchCsrfToken(self: *HttpClient, base_url: []const u8) !void {
        var request = HttpRequest.init(self.allocator, .HEAD, base_url);
        defer request.deinit();
        
        // Add CSRF token fetch header
        try request.addHeader("X-CSRF-Token", "Fetch");
        
        // Add Basic Auth if configured
        if (self.username) |username| {
            if (self.password) |password| {
                const auth_header = try self.buildBasicAuthHeader(username, password);
                defer self.allocator.free(auth_header);
                try request.addHeader("Authorization", auth_header);
            }
        }
        
        // Make request
        var response = try self.execute(&request);
        defer response.deinit();
        
        // Extract CSRF token from response headers
        if (response.getHeader("X-CSRF-Token")) |token| {
            if (self.csrf_token) |old_token| {
                self.allocator.free(old_token);
            }
            self.csrf_token = try self.allocator.dupe(u8, token);
            std.log.info("CSRF token fetched: {s}", .{self.csrf_token.?});
        } else {
            // Fallback: generate timestamp-based token
            const timestamp = std.time.milliTimestamp();
            if (self.csrf_token) |old_token| {
                self.allocator.free(old_token);
            }
            self.csrf_token = try std.fmt.allocPrint(
                self.allocator,
                "csrf-{d}",
                .{timestamp},
            );
            std.log.warn("No CSRF token in response, using generated: {s}", .{self.csrf_token.?});
        }
    }
    
    /// Execute HTTP request
    pub fn execute(self: *HttpClient, request: *HttpRequest) !HttpResponse {
        // Parse URL
        const uri = try std.Uri.parse(request.url);
        
        // Determine port
        const port = uri.port orelse if (std.mem.startsWith(u8, request.url, "https://")) @as(u16, 443) else @as(u16, 80);
        
        // Connect
        const host = uri.host.?.percent_encoded;
        const addr = try std.net.Address.parseIp(host, port);
        const conn = try std.net.tcpConnectToAddress(addr);
        defer conn.close();
        
        // Build HTTP request
        const method_str = switch (request.method) {
            .GET => "GET",
            .POST => "POST",
            .PATCH => "PATCH",
            .DELETE => "DELETE",
            .HEAD => "HEAD",
        };
        
        // Start building request
        var request_builder = std.ArrayList(u8).init(self.allocator);
        defer request_builder.deinit();
        const writer = request_builder.writer();
        
        // Request line
        try writer.print("{s} {s} HTTP/1.1\r\n", .{ method_str, uri.path.percent_encoded });
        try writer.print("Host: {s}\r\n", .{host});
        try writer.writeAll("User-Agent: HANA-Router/1.0\r\n");
        try writer.writeAll("Accept: application/json\r\n");
        
        // Add custom headers
        var it = request.headers.iterator();
        while (it.next()) |entry| {
            try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
        
        // Add Basic Auth if configured and not already in headers
        if (self.username) |username| {
            if (self.password) |password| {
                if (!request.headers.contains("Authorization")) {
                    const auth_header = try self.buildBasicAuthHeader(username, password);
                    defer self.allocator.free(auth_header);
                    try writer.print("Authorization: {s}\r\n", .{auth_header});
                }
            }
        }
        
        // Add CSRF token for write operations
        if (request.method == .POST or request.method == .PATCH or request.method == .DELETE) {
            if (self.csrf_token) |token| {
                if (!request.headers.contains("X-CSRF-Token")) {
                    try writer.print("X-CSRF-Token: {s}\r\n", .{token});
                }
            }
        }
        
        // Add body if present
        if (request.body) |body| {
            try writer.print("Content-Type: application/json\r\n", .{});
            try writer.print("Content-Length: {d}\r\n", .{body.len});
            try writer.writeAll("\r\n");
            try writer.writeAll(body);
        } else {
            try writer.writeAll("\r\n");
        }
        
        // Send request
        const request_data = try request_builder.toOwnedSlice();
        defer self.allocator.free(request_data);
        _ = try conn.writeAll(request_data);
        
        // Read response
        var buffer: [16384]u8 = undefined;
        const bytes_read = try conn.read(&buffer);
        
        // Parse response
        return try self.parseResponse(buffer[0..bytes_read]);
    }
    
    /// Parse HTTP response
    fn parseResponse(self: *HttpClient, data: []const u8) !HttpResponse {
        var response = HttpResponse.init(self.allocator);
        
        // Split headers and body
        const header_body_split = "\r\n\r\n";
        const split_idx = std.mem.indexOf(u8, data, header_body_split) orelse return error.InvalidResponse;
        
        const headers_section = data[0..split_idx];
        const body_section = data[split_idx + header_body_split.len ..];
        
        // Parse status line
        var lines = std.mem.splitSequence(u8, headers_section, "\r\n");
        const status_line = lines.next() orelse return error.InvalidResponse;
        
        // Extract status code
        var status_parts = std.mem.splitSequence(u8, status_line, " ");
        _ = status_parts.next(); // HTTP/1.1
        const status_code_str = status_parts.next() orelse return error.InvalidResponse;
        response.status_code = try std.fmt.parseInt(u16, status_code_str, 10);
        
        // Parse headers
        while (lines.next()) |line| {
            if (line.len == 0) break;
            
            const colon_idx = std.mem.indexOf(u8, line, ":") orelse continue;
            const key = std.mem.trim(u8, line[0..colon_idx], " \t");
            const value = std.mem.trim(u8, line[colon_idx + 1 ..], " \t");
            
            const key_owned = try self.allocator.dupe(u8, key);
            const value_owned = try self.allocator.dupe(u8, value);
            try response.headers.put(key_owned, value_owned);
        }
        
        // Copy body
        if (body_section.len > 0) {
            response.body = try self.allocator.dupe(u8, body_section);
        }
        
        return response;
    }
    
    /// Build Basic Auth header
    fn buildBasicAuthHeader(self: *HttpClient, username: []const u8, password: []const u8) ![]const u8 {
        // Concatenate username:password
        const credentials = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ username, password },
        );
        defer self.allocator.free(credentials);
        
        // Base64 encode
        const encoded_len = std.base64.standard.Encoder.calcSize(credentials.len);
        const encoded = try self.allocator.alloc(u8, encoded_len);
        defer self.allocator.free(encoded);
        
        const encoded_slice = std.base64.standard.Encoder.encode(encoded, credentials);
        
        // Build header value
        return try std.fmt.allocPrint(
            self.allocator,
            "Basic {s}",
            .{encoded_slice},
        );
    }
};

// Tests
test "HttpClient initialization" {
    const allocator = std.testing.allocator;
    
    var client = HttpClient.init(allocator);
    defer client.deinit();
    
    try std.testing.expect(client.csrf_token == null);
}

test "HttpRequest creation" {
    const allocator = std.testing.allocator;
    
    var request = HttpRequest.init(allocator, .GET, "https://example.com/api");
    defer request.deinit();
    
    try request.addHeader("Accept", "application/json");
    try std.testing.expect(request.headers.get("Accept") != null);
}

test "Basic Auth header" {
    const allocator = std.testing.allocator;
    
    var client = HttpClient.init(allocator);
    defer client.deinit();
    
    const auth_header = try client.buildBasicAuthHeader("user", "pass");
    defer allocator.free(auth_header);
    
    try std.testing.expect(std.mem.startsWith(u8, auth_header, "Basic "));
}
