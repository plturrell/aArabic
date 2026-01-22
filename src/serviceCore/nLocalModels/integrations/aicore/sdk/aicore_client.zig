//! SAP AI Core SDK HTTP Client
//!
//! HTTP client for SAP AI Core REST API with OAuth 2.0 authentication.
//! Provides token caching, automatic refresh, and authenticated API requests.
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-21

const std = @import("std");
const AICoreConfig = @import("../aicore_config.zig").AICoreConfig;

/// OAuth 2.0 token information
pub const TokenInfo = struct {
    /// The access token
    access_token: []const u8,
    /// Token type (usually "bearer")
    token_type: []const u8,
    /// Expiration timestamp (seconds since epoch)
    expires_at: i64,

    /// Check if token is expired (with 60 second buffer)
    pub fn isExpired(self: TokenInfo) bool {
        const now = std.time.timestamp();
        return now >= (self.expires_at - 60);
    }
};

/// API error response
pub const ApiError = struct {
    /// HTTP status code
    status_code: u16,
    /// Error message
    message: []const u8,
    /// Error code (if provided)
    error_code: ?[]const u8,
    /// Request ID for debugging
    request_id: ?[]const u8,

    pub fn format(self: ApiError, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "AI Core API Error {d}: {s}", .{
            self.status_code,
            self.message,
        });
    }
};

/// HTTP response wrapper
pub const Response = struct {
    status_code: u16,
    body: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Response) void {
        self.allocator.free(self.body);
    }

    /// Check if response indicates success (2xx)
    pub fn isSuccess(self: Response) bool {
        return self.status_code >= 200 and self.status_code < 300;
    }

    /// Parse response body as JSON
    pub fn parseJson(self: Response, comptime T: type) !T {
        return std.json.parseFromSlice(T, self.allocator, self.body, .{});
    }
};

/// AI Core HTTP Client with OAuth authentication
pub const AICoreClient = struct {
    allocator: std.mem.Allocator,
    config: AICoreConfig,
    http_client: std.http.Client,
    token: ?TokenInfo,
    /// Enable request/response logging
    logging_enabled: bool,

    /// Initialize the AI Core client
    pub fn init(allocator: std.mem.Allocator, config: AICoreConfig) AICoreClient {
        return AICoreClient{
            .allocator = allocator,
            .config = config,
            .http_client = std.http.Client{ .allocator = allocator },
            .token = null,
            .logging_enabled = true,
        };
    }

    /// Cleanup resources
    pub fn deinit(self: *AICoreClient) void {
        if (self.token) |token| {
            self.allocator.free(token.access_token);
            self.allocator.free(token.token_type);
        }
        self.http_client.deinit();
    }

    /// Authenticate with OAuth 2.0 client credentials flow
    pub fn authenticate(self: *AICoreClient) !void {
        const token_url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/oauth/token",
            .{self.config.auth_url},
        );
        defer self.allocator.free(token_url);

        // Build request body (form-urlencoded)
        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=client_credentials&client_id={s}&client_secret={s}",
            .{ self.config.client_id, self.config.client_secret },
        );
        defer self.allocator.free(body);

        if (self.logging_enabled) {
            std.log.info("[AICoreClient] Authenticating to {s}", .{self.config.auth_url});
        }

        // Make token request
        const uri = try std.Uri.parse(token_url);
        var header_buffer: [4096]u8 = undefined;

        var req = try self.http_client.open(.POST, uri, .{
            .server_header_buffer = &header_buffer,
            .extra_headers = &[_]std.http.Header{
                .{ .name = "Content-Type", .value = "application/x-www-form-urlencoded" },
            },
        });
        defer req.deinit();

        req.transfer_encoding = .{ .content_length = body.len };
        try req.send();
        try req.writeAll(body);
        try req.finish();
        try req.wait();

        // Read response
        const response_body = try req.reader().readAllAlloc(self.allocator, 1024 * 1024);
        defer self.allocator.free(response_body);

        if (req.status != .ok) {
            std.log.err("[AICoreClient] Authentication failed: {d}", .{@intFromEnum(req.status)});
            return error.AuthenticationFailed;
        }

        // Parse token response
        try self.parseTokenResponse(response_body);

        if (self.logging_enabled) {
            std.log.info("[AICoreClient] Authentication successful, token expires at {d}", .{
                self.token.?.expires_at,
            });
        }
    }

    /// Parse OAuth token response JSON
    fn parseTokenResponse(self: *AICoreClient, response_body: []const u8) !void {
        const parsed = try std.json.parseFromSlice(
            struct {
                access_token: []const u8,
                token_type: []const u8,
                expires_in: i64,
            },
            self.allocator,
            response_body,
            .{},
        );
        defer parsed.deinit();

        // Free old token if exists
        if (self.token) |old_token| {
            self.allocator.free(old_token.access_token);
            self.allocator.free(old_token.token_type);
        }

        // Store new token with calculated expiration
        const now = std.time.timestamp();
        self.token = TokenInfo{
            .access_token = try self.allocator.dupe(u8, parsed.value.access_token),
            .token_type = try self.allocator.dupe(u8, parsed.value.token_type),
            .expires_at = now + parsed.value.expires_in,
        };
    }

    /// Get current valid token, refreshing if necessary
    pub fn getToken(self: *AICoreClient) ![]const u8 {
        if (self.token) |token| {
            if (!token.isExpired()) {
                return token.access_token;
            }
            // Token expired, need to refresh
            if (self.logging_enabled) {
                std.log.info("[AICoreClient] Token expired, refreshing...", .{});
            }
        }

        try self.authenticate();
        return self.token.?.access_token;
    }

    /// Make an authenticated API request
    pub fn request(
        self: *AICoreClient,
        method: std.http.Method,
        path: []const u8,
        body: ?[]const u8,
    ) !Response {
        // Ensure we have a valid token
        const token = try self.getToken();

        // Build full URL
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2{s}",
            .{ self.config.api_url, path },
        );
        defer self.allocator.free(url);

        if (self.logging_enabled) {
            std.log.info("[AICoreClient] {s} {s}", .{ @tagName(method), url });
        }

        // Build authorization header
        const auth_header = try std.fmt.allocPrint(
            self.allocator,
            "Bearer {s}",
            .{token},
        );
        defer self.allocator.free(auth_header);

        // Make request
        const response = try self.doRequest(method, url, auth_header, body);

        // Handle 401 with token refresh
        if (response.status_code == 401) {
            if (self.logging_enabled) {
                std.log.warn("[AICoreClient] Got 401, refreshing token and retrying...", .{});
            }

            // Free the failed response
            var resp = response;
            resp.deinit();

            // Clear token and re-authenticate
            if (self.token) |old_token| {
                self.allocator.free(old_token.access_token);
                self.allocator.free(old_token.token_type);
                self.token = null;
            }

            const new_token = try self.getToken();
            const new_auth = try std.fmt.allocPrint(
                self.allocator,
                "Bearer {s}",
                .{new_token},
            );
            defer self.allocator.free(new_auth);

            return try self.doRequest(method, url, new_auth, body);
        }

        return response;
    }

    /// Internal HTTP request implementation
    fn doRequest(
        self: *AICoreClient,
        method: std.http.Method,
        url: []const u8,
        auth_header: []const u8,
        body: ?[]const u8,
    ) !Response {
        const uri = try std.Uri.parse(url);
        var header_buffer: [8192]u8 = undefined;

        var extra_headers = [_]std.http.Header{
            .{ .name = "Authorization", .value = auth_header },
            .{ .name = "AI-Resource-Group", .value = self.config.resource_group },
            .{ .name = "Content-Type", .value = "application/json" },
        };

        var req = try self.http_client.open(method, uri, .{
            .server_header_buffer = &header_buffer,
            .extra_headers = &extra_headers,
        });
        defer req.deinit();

        if (body) |b| {
            req.transfer_encoding = .{ .content_length = b.len };
        }

        try req.send();

        if (body) |b| {
            try req.writeAll(b);
        }

        try req.finish();
        try req.wait();

        const response_body = try req.reader().readAllAlloc(self.allocator, 10 * 1024 * 1024);

        if (self.logging_enabled) {
            std.log.info("[AICoreClient] Response: {d} ({d} bytes)", .{
                @intFromEnum(req.status),
                response_body.len,
            });
        }

        return Response{
            .status_code = @intFromEnum(req.status),
            .body = response_body,
            .allocator = self.allocator,
        };
    }

    /// GET request convenience method
    pub fn get(self: *AICoreClient, path: []const u8) !Response {
        return self.request(.GET, path, null);
    }

    /// POST request convenience method
    pub fn post(self: *AICoreClient, path: []const u8, body: []const u8) !Response {
        return self.request(.POST, path, body);
    }

    /// DELETE request convenience method
    pub fn delete(self: *AICoreClient, path: []const u8) !Response {
        return self.request(.DELETE, path, null);
    }

    /// PATCH request convenience method
    pub fn patch(self: *AICoreClient, path: []const u8, body: []const u8) !Response {
        return self.request(.PATCH, path, body);
    }

    /// Build base URL for AI Core API
    pub fn getBaseUrl(self: *AICoreClient) []const u8 {
        return self.config.api_url;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "token expiry check" {
    const testing = std.testing;

    // Token that expires in the future
    const valid_token = TokenInfo{
        .access_token = "test",
        .token_type = "bearer",
        .expires_at = std.time.timestamp() + 3600,
    };
    try testing.expect(!valid_token.isExpired());

    // Token that is already expired
    const expired_token = TokenInfo{
        .access_token = "test",
        .token_type = "bearer",
        .expires_at = std.time.timestamp() - 100,
    };
    try testing.expect(expired_token.isExpired());

    // Token expiring within buffer (60 seconds)
    const expiring_soon = TokenInfo{
        .access_token = "test",
        .token_type = "bearer",
        .expires_at = std.time.timestamp() + 30,
    };
    try testing.expect(expiring_soon.isExpired());
}

test "response success check" {
    const testing = std.testing;

    const success = Response{ .status_code = 200, .body = "", .allocator = testing.allocator };
    try testing.expect(success.isSuccess());

    const created = Response{ .status_code = 201, .body = "", .allocator = testing.allocator };
    try testing.expect(created.isSuccess());

    const error_resp = Response{ .status_code = 400, .body = "", .allocator = testing.allocator };
    try testing.expect(!error_resp.isSuccess());

    const server_error = Response{ .status_code = 500, .body = "", .allocator = testing.allocator };
    try testing.expect(!server_error.isSuccess());
}

test "api error format" {
    const testing = std.testing;

    const api_error = ApiError{
        .status_code = 404,
        .message = "Deployment not found",
        .error_code = "DEPLOYMENT_NOT_FOUND",
        .request_id = "abc-123",
    };

    const formatted = try api_error.format(testing.allocator);
    defer testing.allocator.free(formatted);

    try testing.expectEqualStrings("AI Core API Error 404: Deployment not found", formatted);
}

test "client init and deinit" {
    const testing = std.testing;

    const config = AICoreConfig{
        .auth_url = "https://auth.example.com",
        .api_url = "https://api.example.com",
        .client_id = "test-client",
        .client_secret = "test-secret",
        .resource_group = "default",
    };

    var client = AICoreClient.init(testing.allocator, config);
    defer client.deinit();

    try testing.expect(client.token == null);
    try testing.expect(client.logging_enabled);
}

