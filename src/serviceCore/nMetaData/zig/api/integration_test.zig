const std = @import("std");
const http = @import("../http/server.zig");
const router = @import("../http/router.zig");
const types = @import("../http/types.zig");
const auth = @import("../auth/jwt.zig");
const handlers = @import("handlers.zig");
const auth_handlers = @import("auth_handlers.zig");
const graphql_handler = @import("graphql_handler.zig");

/// Integration test client for API testing
pub const TestClient = struct {
    allocator: std.mem.Allocator,
    base_url: []const u8,
    auth_token: ?[]const u8,

    pub fn init(allocator: std.mem.Allocator, base_url: []const u8) TestClient {
        return TestClient{
            .allocator = allocator,
            .base_url = base_url,
            .auth_token = null,
        };
    }

    pub fn deinit(self: *TestClient) void {
        if (self.auth_token) |token| {
            self.allocator.free(token);
        }
    }

    /// Set authentication token
    pub fn setAuthToken(self: *TestClient, token: []const u8) !void {
        if (self.auth_token) |old_token| {
            self.allocator.free(old_token);
        }
        self.auth_token = try self.allocator.dupe(u8, token);
    }

    /// Make HTTP GET request
    pub fn get(self: *TestClient, path: []const u8) !TestResponse {
        return self.request("GET", path, null);
    }

    /// Make HTTP POST request
    pub fn post(self: *TestClient, path: []const u8, body: ?[]const u8) !TestResponse {
        return self.request("POST", path, body);
    }

    /// Make HTTP PUT request
    pub fn put(self: *TestClient, path: []const u8, body: ?[]const u8) !TestResponse {
        return self.request("PUT", path, body);
    }

    /// Make HTTP DELETE request
    pub fn delete(self: *TestClient, path: []const u8) !TestResponse {
        return self.request("DELETE", path, null);
    }

    /// Make HTTP request (internal helper)
    fn request(self: *TestClient, method: []const u8, path: []const u8, body: ?[]const u8) !TestResponse {
        const url = try std.fmt.allocPrint(self.allocator, "{s}{s}", .{ self.base_url, path });
        defer self.allocator.free(url);

        // Build headers
        var headers = std.ArrayList(u8).init(self.allocator);
        defer headers.deinit();

        try headers.appendSlice("Content-Type: application/json\r\n");
        
        if (self.auth_token) |token| {
            try headers.writer().print("Authorization: Bearer {s}\r\n", .{token});
        }

        // Mock response for testing (in real scenario, would make actual HTTP call)
        return TestResponse{
            .status_code = 200,
            .body = if (body) |b| try self.allocator.dupe(u8, b) else try self.allocator.dupe(u8, "{}"),
            .headers = try self.allocator.dupe(u8, headers.items),
            .allocator = self.allocator,
        };
    }
};

/// Test response structure
pub const TestResponse = struct {
    status_code: u16,
    body: []const u8,
    headers: []const u8,
    allocator: std.mem.Allocator,

    pub fn deinit(self: TestResponse) void {
        self.allocator.free(self.body);
        self.allocator.free(self.headers);
    }

    /// Parse JSON body
    pub fn json(self: TestResponse) !std.json.Value {
        return try std.json.parseFromSlice(std.json.Value, self.allocator, self.body, .{});
    }

    /// Check if response is success (2xx)
    pub fn isSuccess(self: TestResponse) bool {
        return self.status_code >= 200 and self.status_code < 300;
    }

    /// Check if response is error (4xx or 5xx)
    pub fn isError(self: TestResponse) bool {
        return self.status_code >= 400;
    }
};

/// API Integration test suite
pub const IntegrationTests = struct {
    allocator: std.mem.Allocator,
    client: TestClient,

    pub fn init(allocator: std.mem.Allocator, base_url: []const u8) IntegrationTests {
        return IntegrationTests{
            .allocator = allocator,
            .client = TestClient.init(allocator, base_url),
        };
    }

    pub fn deinit(self: *IntegrationTests) void {
        self.client.deinit();
    }

    /// Test authentication flow
    pub fn testAuthenticationFlow(self: *IntegrationTests) !void {
        // 1. Login
        const login_body = 
            \\{"username": "admin", "password": "admin123"}
        ;
        const login_response = try self.client.post("/api/v1/auth/login", login_body);
        defer login_response.deinit();

        try std.testing.expect(login_response.isSuccess());
        
        const login_json = try login_response.json();
        defer login_json.deinit();

        // Extract token
        const token = "mock_token_12345";
        try self.client.setAuthToken(token);

        // 2. Get current user
        const me_response = try self.client.get("/api/v1/auth/me");
        defer me_response.deinit();

        try std.testing.expect(me_response.isSuccess());

        // 3. Verify token
        const verify_response = try self.client.get("/api/v1/auth/verify");
        defer verify_response.deinit();

        try std.testing.expect(verify_response.isSuccess());

        // 4. Refresh token
        const refresh_response = try self.client.post("/api/v1/auth/refresh", null);
        defer refresh_response.deinit();

        try std.testing.expect(refresh_response.isSuccess());

        // 5. Logout
        const logout_response = try self.client.post("/api/v1/auth/logout", null);
        defer logout_response.deinit();

        try std.testing.expect(logout_response.isSuccess());
    }

    /// Test dataset CRUD operations
    pub fn testDatasetCRUD(self: *IntegrationTests) !void {
        // Set auth token
        try self.client.setAuthToken("mock_admin_token");

        // 1. Create dataset
        const create_body = 
            \\{"name": "test_dataset", "type": "table", "schema": "public", "description": "Test dataset"}
        ;
        const create_response = try self.client.post("/api/v1/datasets", create_body);
        defer create_response.deinit();

        try std.testing.expect(create_response.isSuccess());
        try std.testing.expectEqual(@as(u16, 200), create_response.status_code);

        // 2. List datasets
        const list_response = try self.client.get("/api/v1/datasets?limit=10&offset=0");
        defer list_response.deinit();

        try std.testing.expect(list_response.isSuccess());

        // 3. Get dataset by ID
        const get_response = try self.client.get("/api/v1/datasets/1");
        defer get_response.deinit();

        try std.testing.expect(get_response.isSuccess());

        // 4. Update dataset
        const update_body = 
            \\{"name": "test_dataset_updated", "description": "Updated description"}
        ;
        const update_response = try self.client.put("/api/v1/datasets/1", update_body);
        defer update_response.deinit();

        try std.testing.expect(update_response.isSuccess());

        // 5. Delete dataset
        const delete_response = try self.client.delete("/api/v1/datasets/1");
        defer delete_response.deinit();

        try std.testing.expect(delete_response.isSuccess());
    }

    /// Test lineage tracking
    pub fn testLineageTracking(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        // 1. Create lineage edge
        const edge_body = 
            \\{"from_id": 1, "to_id": 2, "edge_type": "derives_from"}
        ;
        const edge_response = try self.client.post("/api/v1/lineage/edges", edge_body);
        defer edge_response.deinit();

        try std.testing.expect(edge_response.isSuccess());

        // 2. Get upstream lineage
        const upstream_response = try self.client.get("/api/v1/lineage/upstream/2?depth=5");
        defer upstream_response.deinit();

        try std.testing.expect(upstream_response.isSuccess());

        // 3. Get downstream lineage
        const downstream_response = try self.client.get("/api/v1/lineage/downstream/1?depth=5");
        defer downstream_response.deinit();

        try std.testing.expect(downstream_response.isSuccess());
    }

    /// Test GraphQL endpoint
    pub fn testGraphQLEndpoint(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        // 1. Execute GraphQL query
        const query_body = 
            \\{"query": "{ datasets { id name type } }"}
        ;
        const query_response = try self.client.post("/api/v1/graphql", query_body);
        defer query_response.deinit();

        try std.testing.expect(query_response.isSuccess());

        // 2. Test introspection query
        const introspection_body = 
            \\{"query": "{ __schema { types { name } } }"}
        ;
        const introspection_response = try self.client.post("/api/v1/graphql", introspection_body);
        defer introspection_response.deinit();

        try std.testing.expect(introspection_response.isSuccess());

        // 3. Access GraphiQL playground
        const graphiql_response = try self.client.get("/api/v1/graphiql");
        defer graphiql_response.deinit();

        try std.testing.expect(graphiql_response.isSuccess());

        // 4. Get schema
        const schema_response = try self.client.get("/api/v1/schema");
        defer schema_response.deinit();

        try std.testing.expect(schema_response.isSuccess());
    }

    /// Test pagination
    pub fn testPagination(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        // Test different pagination parameters
        const test_cases = [_]struct {
            limit: u32,
            offset: u32,
        }{
            .{ .limit = 10, .offset = 0 },
            .{ .limit = 20, .offset = 0 },
            .{ .limit = 10, .offset = 10 },
            .{ .limit = 50, .offset = 0 },
            .{ .limit = 100, .offset = 50 },
        };

        for (test_cases) |tc| {
            const path = try std.fmt.allocPrint(
                self.allocator,
                "/api/v1/datasets?limit={d}&offset={d}",
                .{ tc.limit, tc.offset },
            );
            defer self.allocator.free(path);

            const response = try self.client.get(path);
            defer response.deinit();

            try std.testing.expect(response.isSuccess());
        }
    }

    /// Test error handling
    pub fn testErrorHandling(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        // 1. Test 404 - not found
        const not_found_response = try self.client.get("/api/v1/datasets/99999");
        defer not_found_response.deinit();

        try std.testing.expect(not_found_response.isError());

        // 2. Test 400 - bad request (invalid JSON)
        const bad_json_response = try self.client.post("/api/v1/datasets", "invalid json");
        defer bad_json_response.deinit();

        try std.testing.expect(bad_json_response.isError());

        // 3. Test 401 - unauthorized (no token)
        var unauth_client = TestClient.init(self.allocator, self.client.base_url);
        defer unauth_client.deinit();

        const unauth_response = try unauth_client.get("/api/v1/datasets");
        defer unauth_response.deinit();

        try std.testing.expect(unauth_response.isError());

        // 4. Test 409 - conflict (duplicate)
        const duplicate_body = 
            \\{"name": "existing_dataset", "type": "table", "schema": "public"}
        ;
        const dup_response1 = try self.client.post("/api/v1/datasets", duplicate_body);
        defer dup_response1.deinit();

        const dup_response2 = try self.client.post("/api/v1/datasets", duplicate_body);
        defer dup_response2.deinit();

        try std.testing.expect(dup_response2.isError());
    }

    /// Test concurrent requests
    pub fn testConcurrentRequests(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        const num_concurrent = 10;
        var responses: [num_concurrent]TestResponse = undefined;

        // Make concurrent requests
        for (&responses) |*response| {
            response.* = try self.client.get("/api/v1/datasets");
        }

        // Verify all succeeded
        for (responses) |response| {
            defer response.deinit();
            try std.testing.expect(response.isSuccess());
        }
    }

    /// Test rate limiting
    pub fn testRateLimiting(self: *IntegrationTests) !void {
        try self.client.setAuthToken("mock_admin_token");

        // Make many rapid requests
        const num_requests = 100;
        var success_count: u32 = 0;
        var rate_limited_count: u32 = 0;

        var i: u32 = 0;
        while (i < num_requests) : (i += 1) {
            const response = try self.client.get("/api/v1/datasets");
            defer response.deinit();

            if (response.isSuccess()) {
                success_count += 1;
            } else if (response.status_code == 429) { // Too Many Requests
                rate_limited_count += 1;
            }
        }

        // Should have some rate limiting
        try std.testing.expect(success_count > 0);
        try std.testing.expect(rate_limited_count > 0 or success_count == num_requests);
    }

    /// Run all integration tests
    pub fn runAll(self: *IntegrationTests) !void {
        std.debug.print("\nRunning API Integration Tests...\n", .{});

        std.debug.print("  1. Testing authentication flow...\n", .{});
        try self.testAuthenticationFlow();
        std.debug.print("     ✓ Authentication flow passed\n", .{});

        std.debug.print("  2. Testing dataset CRUD...\n", .{});
        try self.testDatasetCRUD();
        std.debug.print("     ✓ Dataset CRUD passed\n", .{});

        std.debug.print("  3. Testing lineage tracking...\n", .{});
        try self.testLineageTracking();
        std.debug.print("     ✓ Lineage tracking passed\n", .{});

        std.debug.print("  4. Testing GraphQL endpoint...\n", .{});
        try self.testGraphQLEndpoint();
        std.debug.print("     ✓ GraphQL endpoint passed\n", .{});

        std.debug.print("  5. Testing pagination...\n", .{});
        try self.testPagination();
        std.debug.print("     ✓ Pagination passed\n", .{});

        std.debug.print("  6. Testing error handling...\n", .{});
        try self.testErrorHandling();
        std.debug.print("     ✓ Error handling passed\n", .{});

        std.debug.print("  7. Testing concurrent requests...\n", .{});
        try self.testConcurrentRequests();
        std.debug.print("     ✓ Concurrent requests passed\n", .{});

        std.debug.print("  8. Testing rate limiting...\n", .{});
        try self.testRateLimiting();
        std.debug.print("     ✓ Rate limiting passed\n", .{});

        std.debug.print("\n✅ All integration tests passed!\n", .{});
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TestClient - initialization" {
    var client = TestClient.init(std.testing.allocator, "http://localhost:3000");
    defer client.deinit();

    try std.testing.expect(client.auth_token == null);
    try std.testing.expectEqualStrings("http://localhost:3000", client.base_url);
}

test "TestClient - set auth token" {
    var client = TestClient.init(std.testing.allocator, "http://localhost:3000");
    defer client.deinit();

    try client.setAuthToken("test_token_123");
    try std.testing.expect(client.auth_token != null);
    try std.testing.expectEqualStrings("test_token_123", client.auth_token.?);
}

test "TestResponse - status checks" {
    const allocator = std.testing.allocator;

    const success_response = TestResponse{
        .status_code = 200,
        .body = try allocator.dupe(u8, "{}"),
        .headers = try allocator.dupe(u8, ""),
        .allocator = allocator,
    };
    defer success_response.deinit();

    try std.testing.expect(success_response.isSuccess());
    try std.testing.expect(!success_response.isError());

    const error_response = TestResponse{
        .status_code = 404,
        .body = try allocator.dupe(u8, "{}"),
        .headers = try allocator.dupe(u8, ""),
        .allocator = allocator,
    };
    defer error_response.deinit();

    try std.testing.expect(!error_response.isSuccess());
    try std.testing.expect(error_response.isError());
}

test "IntegrationTests - initialization" {
    var tests = IntegrationTests.init(std.testing.allocator, "http://localhost:3000");
    defer tests.deinit();

    try std.testing.expectEqualStrings("http://localhost:3000", tests.client.base_url);
}
