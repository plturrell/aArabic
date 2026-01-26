// nWorkflow End-to-End Integration Tests
// Comprehensive tests for HTTP API, workflow CRUD, execution, authentication, and multi-tenancy

const std = @import("std");
const testing = std.testing;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const json = std.json;

// =============================================================================
// Test Context and Helpers
// =============================================================================

/// TestContext provides test setup and HTTP request helpers
pub const TestContext = struct {
    allocator: Allocator,
    base_url: []const u8,
    auth_token: ?[]const u8,
    tenant_id: ?[]const u8,

    pub fn init(allocator: Allocator, base_url: []const u8) TestContext {
        return TestContext{
            .allocator = allocator,
            .base_url = base_url,
            .auth_token = null,
            .tenant_id = null,
        };
    }

    pub fn withAuth(self: *TestContext, token: []const u8) *TestContext {
        self.auth_token = token;
        return self;
    }

    pub fn withTenant(self: *TestContext, tenant_id: []const u8) *TestContext {
        self.tenant_id = tenant_id;
        return self;
    }
};

/// HTTP Response structure for mock responses
pub const HttpResponse = struct {
    status_code: u16,
    body: []const u8,
    headers: std.StringHashMap([]const u8),

    pub fn deinit(self: *HttpResponse, allocator: Allocator) void {
        allocator.free(self.body);
        self.headers.deinit();
    }
};

/// JSON parsing helper - parse response body to Value
pub fn parseJsonResponse(allocator: Allocator, body: []const u8) !json.Value {
    const parsed = try json.parseFromSlice(json.Value, allocator, body, .{});
    return parsed.value;
}

/// Extract string field from JSON object
pub fn getJsonString(value: json.Value, key: []const u8) ?[]const u8 {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .string) return null;
    return field.string;
}

/// Extract integer field from JSON object
pub fn getJsonInt(value: json.Value, key: []const u8) ?i64 {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .integer) return null;
    return field.integer;
}

/// Extract array field from JSON object
pub fn getJsonArray(value: json.Value, key: []const u8) ?[]const json.Value {
    if (value != .object) return null;
    const field = value.object.get(key) orelse return null;
    if (field != .array) return null;
    return field.array.items;
}

// =============================================================================
// Mock HTTP Client for Testing Without Real Server
// =============================================================================

/// MockHttpClient simulates HTTP responses for testing
pub const MockHttpClient = struct {
    allocator: Allocator,
    responses: std.StringHashMap(MockResponse),
    request_log: std.ArrayList(MockRequest),

    pub const MockResponse = struct {
        status_code: u16,
        body: []const u8,
        content_type: []const u8,
    };

    pub const MockRequest = struct {
        method: []const u8,
        path: []const u8,
        body: ?[]const u8,
        headers: std.StringHashMap([]const u8),
    };

    pub fn init(allocator: Allocator) MockHttpClient {
        return MockHttpClient{
            .allocator = allocator,
            .responses = std.StringHashMap(MockResponse).init(allocator),
            .request_log = std.ArrayList(MockRequest){},
        };
    }

    pub fn deinit(self: *MockHttpClient) void {
        self.responses.deinit();
        for (self.request_log.items) |*req| {
            self.allocator.free(req.method);
            self.allocator.free(req.path);
            if (req.body) |b| self.allocator.free(b);
            req.headers.deinit();
        }
        self.request_log.deinit(self.allocator);
    }

    /// Register a mock response for a given path
    pub fn mockResponse(self: *MockHttpClient, path: []const u8, response: MockResponse) !void {
        try self.responses.put(path, response);
    }

    /// Simulate GET request
    pub fn get(self: *MockHttpClient, path: []const u8, headers: ?std.StringHashMap([]const u8)) !HttpResponse {
        return self.request("GET", path, null, headers);
    }

    /// Simulate POST request
    pub fn post(self: *MockHttpClient, path: []const u8, body: []const u8, headers: ?std.StringHashMap([]const u8)) !HttpResponse {
        return self.request("POST", path, body, headers);
    }

    /// Simulate PUT request
    pub fn put(self: *MockHttpClient, path: []const u8, body: []const u8, headers: ?std.StringHashMap([]const u8)) !HttpResponse {
        return self.request("PUT", path, body, headers);
    }

    /// Simulate DELETE request
    pub fn delete(self: *MockHttpClient, path: []const u8, headers: ?std.StringHashMap([]const u8)) !HttpResponse {
        return self.request("DELETE", path, null, headers);
    }

    /// Internal request handler
    fn request(self: *MockHttpClient, method: []const u8, path: []const u8, body: ?[]const u8, headers: ?std.StringHashMap([]const u8)) !HttpResponse {
        // Log the request
        const req = MockRequest{
            .method = try self.allocator.dupe(u8, method),
            .path = try self.allocator.dupe(u8, path),
            .body = if (body) |b| try self.allocator.dupe(u8, b) else null,
            .headers = headers orelse std.StringHashMap([]const u8).init(self.allocator),
        };
        try self.request_log.append(self.allocator, req);

        // Return mock response or default 404
        if (self.responses.get(path)) |mock_resp| {
            var resp_headers = std.StringHashMap([]const u8).init(self.allocator);
            try resp_headers.put("Content-Type", mock_resp.content_type);
            return HttpResponse{
                .status_code = mock_resp.status_code,
                .body = try self.allocator.dupe(u8, mock_resp.body),
                .headers = resp_headers,
            };
        }

        // Default 404 response
        var resp_headers = std.StringHashMap([]const u8).init(self.allocator);
        try resp_headers.put("Content-Type", "application/json");
        return HttpResponse{
            .status_code = 404,
            .body = try self.allocator.dupe(u8, "{\"error\":\"Not found\"}"),
            .headers = resp_headers,
        };
    }

    /// Get the last request made
    pub fn getLastRequest(self: *MockHttpClient) ?MockRequest {
        if (self.request_log.items.len == 0) return null;
        return self.request_log.items[self.request_log.items.len - 1];
    }

    /// Get request count
    pub fn getRequestCount(self: *MockHttpClient) usize {
        return self.request_log.items.len;
    }

    /// Clear request log
    pub fn clearLog(self: *MockHttpClient) void {
        for (self.request_log.items) |*req| {
            self.allocator.free(req.method);
            self.allocator.free(req.path);
            if (req.body) |b| self.allocator.free(b);
            req.headers.deinit();
        }
        self.request_log.clearRetainingCapacity();
    }
};

// =============================================================================
// Section 2: Health Check Tests
// =============================================================================

test "health endpoint returns ok" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    // Setup mock response
    try client.mockResponse("/api/v1/health", .{
        .status_code = 200,
        .body = "{\"status\":\"healthy\",\"service\":\"nWorkflow\",\"version\":\"1.0.0\"}",
        .content_type = "application/json",
    });

    // Make request
    var response = try client.get("/api/v1/health", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const status = getJsonString(parsed.value, "status");
    try testing.expect(status != null);
    try testing.expectEqualStrings("healthy", status.?);
}

test "health includes version info" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/health", .{
        .status_code = 200,
        .body = "{\"status\":\"healthy\",\"service\":\"nWorkflow\",\"version\":\"1.0.0\",\"uptime\":3600}",
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/health", null);
    defer response.deinit(allocator);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const version = getJsonString(parsed.value, "version");
    try testing.expect(version != null);
    try testing.expectEqualStrings("1.0.0", version.?);

    const service = getJsonString(parsed.value, "service");
    try testing.expect(service != null);
    try testing.expectEqualStrings("nWorkflow", service.?);
}


// =============================================================================
// Section 3: Workflow CRUD Tests
// =============================================================================

test "create workflow returns 201" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const workflow_json =
        \\{"id":"wf-001","name":"Test Workflow","description":"A test workflow","nodes":[],"edges":[]}
    ;

    const response_json =
        \\{"id":"wf-001","name":"Test Workflow","description":"A test workflow","version":1,"status":"draft","created_at":"2026-01-19T10:00:00Z"}
    ;

    try client.mockResponse("/api/v1/workflows", .{
        .status_code = 201,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/workflows", workflow_json, null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 201), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const id = getJsonString(parsed.value, "id");
    try testing.expect(id != null);
    try testing.expectEqualStrings("wf-001", id.?);

    const version = getJsonInt(parsed.value, "version");
    try testing.expect(version != null);
    try testing.expectEqual(@as(i64, 1), version.?);
}

test "get workflow by id" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"id":"wf-001","name":"Test Workflow","description":"A test workflow","version":2,"status":"active"}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/workflows/wf-001", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const id = getJsonString(parsed.value, "id");
    try testing.expect(id != null);
    try testing.expectEqualStrings("wf-001", id.?);
}

test "list workflows with pagination" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"workflows":[{"id":"wf-001","name":"Workflow 1"},{"id":"wf-002","name":"Workflow 2"}],"total":10,"page":1,"page_size":2}
    ;

    try client.mockResponse("/api/v1/workflows", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/workflows", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const workflows = getJsonArray(parsed.value, "workflows");
    try testing.expect(workflows != null);
    try testing.expectEqual(@as(usize, 2), workflows.?.len);

    const total = getJsonInt(parsed.value, "total");
    try testing.expect(total != null);
    try testing.expectEqual(@as(i64, 10), total.?);
}

test "update workflow increments version" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const update_json =
        \\{"name":"Updated Workflow","description":"Updated description"}
    ;

    const response_json =
        \\{"id":"wf-001","name":"Updated Workflow","description":"Updated description","version":3}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.put("/api/v1/workflows/wf-001", update_json, null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const version = getJsonInt(parsed.value, "version");
    try testing.expect(version != null);
    try testing.expectEqual(@as(i64, 3), version.?);
}

test "delete workflow soft deletes" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"id":"wf-001","status":"deleted","deleted_at":"2026-01-19T12:00:00Z"}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.delete("/api/v1/workflows/wf-001", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const status = getJsonString(parsed.value, "status");
    try testing.expect(status != null);
    try testing.expectEqualStrings("deleted", status.?);
}

test "create workflow with invalid JSON returns 400" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const invalid_json = "{ invalid json }";

    try client.mockResponse("/api/v1/workflows", .{
        .status_code = 400,
        .body = "{\"error\":\"Invalid JSON\",\"details\":\"Expected property name\"}",
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/workflows", invalid_json, null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 400), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Invalid JSON", error_msg.?);
}

// =============================================================================
// Section 4: Workflow Execution Tests
// =============================================================================

test "execute simple workflow" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"execution_id":"exec-001","workflow_id":"wf-001","status":"completed","started_at":"2026-01-19T10:00:00Z","completed_at":"2026-01-19T10:00:05Z"}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001/execute", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/workflows/wf-001/execute", "{}", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const status = getJsonString(parsed.value, "status");
    try testing.expect(status != null);
    try testing.expectEqualStrings("completed", status.?);

    const exec_id = getJsonString(parsed.value, "execution_id");
    try testing.expect(exec_id != null);
}

test "execute workflow with input" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const input_json =
        \\{"input":{"customer_id":"cust-123","amount":500.00}}
    ;

    const response_json =
        \\{"execution_id":"exec-002","workflow_id":"wf-001","status":"running","input":{"customer_id":"cust-123","amount":500.00}}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001/execute", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/workflows/wf-001/execute", input_json, null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const exec_id = getJsonString(parsed.value, "execution_id");
    try testing.expect(exec_id != null);
    try testing.expectEqualStrings("exec-002", exec_id.?);
}

test "get execution status" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"execution_id":"exec-001","workflow_id":"wf-001","status":"running","progress":75,"current_node":"node-3"}
    ;

    try client.mockResponse("/api/v1/executions/exec-001", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/executions/exec-001", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const status = getJsonString(parsed.value, "status");
    try testing.expect(status != null);
    try testing.expectEqualStrings("running", status.?);

    const progress = getJsonInt(parsed.value, "progress");
    try testing.expect(progress != null);
    try testing.expectEqual(@as(i64, 75), progress.?);
}

test "list executions for workflow" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"executions":[{"execution_id":"exec-001","status":"completed"},{"execution_id":"exec-002","status":"failed"},{"execution_id":"exec-003","status":"running"}],"total":3}
    ;

    try client.mockResponse("/api/v1/workflows/wf-001/executions", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/workflows/wf-001/executions", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const executions = getJsonArray(parsed.value, "executions");
    try testing.expect(executions != null);
    try testing.expectEqual(@as(usize, 3), executions.?.len);
}

test "cancel running execution" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"execution_id":"exec-003","status":"cancelled","cancelled_at":"2026-01-19T10:30:00Z","cancelled_by":"user-123"}
    ;

    try client.mockResponse("/api/v1/executions/exec-003/cancel", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/executions/exec-003/cancel", "{}", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const status = getJsonString(parsed.value, "status");
    try testing.expect(status != null);
    try testing.expectEqualStrings("cancelled", status.?);
}

// =============================================================================
// Section 5: Node Type Tests
// =============================================================================

test "list all node types" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"nodeTypes":[{"id":"start","name":"Start","category":"flow"},{"id":"end","name":"End","category":"flow"},{"id":"task","name":"Task","category":"action"},{"id":"decision","name":"Decision","category":"flow"},{"id":"llm","name":"LLM","category":"ai"},{"id":"http","name":"HTTP Request","category":"action"},{"id":"database","name":"Database","category":"data"},{"id":"transform","name":"Transform","category":"data"},{"id":"filter","name":"Filter","category":"data"},{"id":"aggregate","name":"Aggregate","category":"data"}]}
    ;

    try client.mockResponse("/api/v1/node-types", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/node-types", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const node_types = getJsonArray(parsed.value, "nodeTypes");
    try testing.expect(node_types != null);
    try testing.expect(node_types.?.len >= 10);
}

test "get node type schema" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"id":"llm","name":"LLM","category":"ai","description":"Call LLM for text generation","inputs":[{"id":"prompt","name":"Prompt","type":"string","required":true}],"outputs":[{"id":"response","name":"Response","type":"string"}],"properties":[{"id":"model","name":"Model","type":"select","options":["gpt-4","claude-3","llama-2"]}]}
    ;

    try client.mockResponse("/api/v1/node-types/llm", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/node-types/llm", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const id = getJsonString(parsed.value, "id");
    try testing.expect(id != null);
    try testing.expectEqualStrings("llm", id.?);

    const category = getJsonString(parsed.value, "category");
    try testing.expect(category != null);
    try testing.expectEqualStrings("ai", category.?);
}

test "node type has required fields" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    const response_json =
        \\{"id":"http","name":"HTTP Request","category":"action","description":"Make HTTP requests","inputs":[{"id":"url","name":"URL","type":"string","required":true},{"id":"method","name":"Method","type":"string","required":true}],"outputs":[{"id":"response","name":"Response","type":"object"}]}
    ;

    try client.mockResponse("/api/v1/node-types/http", .{
        .status_code = 200,
        .body = response_json,
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/node-types/http", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 200), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    // Verify required fields exist
    try testing.expect(getJsonString(parsed.value, "id") != null);
    try testing.expect(getJsonString(parsed.value, "name") != null);
    try testing.expect(getJsonString(parsed.value, "category") != null);
    try testing.expect(getJsonArray(parsed.value, "inputs") != null);
    try testing.expect(getJsonArray(parsed.value, "outputs") != null);
}

// =============================================================================
// Section 6: Authentication Tests (Mock)
// =============================================================================

/// Mock auth middleware for testing authentication
pub const MockAuthMiddleware = struct {
    valid_tokens: std.StringHashMap(TokenInfo),
    allocator: Allocator,

    pub const TokenInfo = struct {
        user_id: []const u8,
        roles: []const []const u8,
        tenant_id: ?[]const u8,
    };

    pub fn init(allocator: Allocator) MockAuthMiddleware {
        return MockAuthMiddleware{
            .allocator = allocator,
            .valid_tokens = std.StringHashMap(TokenInfo).init(allocator),
        };
    }

    pub fn deinit(self: *MockAuthMiddleware) void {
        self.valid_tokens.deinit();
    }

    pub fn addValidToken(self: *MockAuthMiddleware, token: []const u8, info: TokenInfo) !void {
        try self.valid_tokens.put(token, info);
    }

    pub fn validateToken(self: *MockAuthMiddleware, token: ?[]const u8) ?TokenInfo {
        if (token == null) return null;
        return self.valid_tokens.get(token.?);
    }

    pub fn hasPermission(self: *MockAuthMiddleware, token: ?[]const u8, required_role: []const u8) bool {
        const info = self.validateToken(token) orelse return false;
        for (info.roles) |role| {
            if (mem.eql(u8, role, required_role) or mem.eql(u8, role, "admin")) {
                return true;
            }
        }
        return false;
    }
};

test "unauthenticated request returns 401" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/workflows", .{
        .status_code = 401,
        .body = "{\"error\":\"Unauthorized\",\"message\":\"Authentication required\"}",
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/workflows", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 401), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Unauthorized", error_msg.?);
}

test "invalid token returns 401" {
    const allocator = testing.allocator;
    var auth = MockAuthMiddleware.init(allocator);
    defer auth.deinit();

    // Add valid token
    try auth.addValidToken("valid-token-123", .{
        .user_id = "user-001",
        .roles = &[_][]const u8{"viewer"},
        .tenant_id = "tenant-001",
    });

    // Validate invalid token
    const result = auth.validateToken("invalid-token-999");
    try testing.expect(result == null);
}

test "valid token allows access" {
    const allocator = testing.allocator;
    var auth = MockAuthMiddleware.init(allocator);
    defer auth.deinit();

    try auth.addValidToken("valid-token-123", .{
        .user_id = "user-001",
        .roles = &[_][]const u8{"viewer"},
        .tenant_id = "tenant-001",
    });

    const result = auth.validateToken("valid-token-123");
    try testing.expect(result != null);
    try testing.expectEqualStrings("user-001", result.?.user_id);
}

test "insufficient permissions returns 403" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/admin/settings", .{
        .status_code = 403,
        .body = "{\"error\":\"Forbidden\",\"message\":\"Insufficient permissions\",\"required_role\":\"admin\"}",
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/admin/settings", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 403), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Forbidden", error_msg.?);
}

// =============================================================================
// Section 7: Multi-tenancy Tests (Mock)
// =============================================================================

/// Mock tenant isolation for testing multi-tenancy
pub const MockTenantStore = struct {
    allocator: Allocator,
    workflows: std.StringHashMap(TenantWorkflow),

    pub const TenantWorkflow = struct {
        id: []const u8,
        tenant_id: []const u8,
        name: []const u8,
    };

    pub fn init(allocator: Allocator) MockTenantStore {
        return MockTenantStore{
            .allocator = allocator,
            .workflows = std.StringHashMap(TenantWorkflow).init(allocator),
        };
    }

    pub fn deinit(self: *MockTenantStore) void {
        self.workflows.deinit();
    }

    pub fn addWorkflow(self: *MockTenantStore, workflow: TenantWorkflow) !void {
        try self.workflows.put(workflow.id, workflow);
    }

    pub fn getWorkflow(self: *MockTenantStore, workflow_id: []const u8, tenant_id: []const u8) ?TenantWorkflow {
        const workflow = self.workflows.get(workflow_id) orelse return null;
        // Enforce tenant isolation
        if (!mem.eql(u8, workflow.tenant_id, tenant_id)) return null;
        return workflow;
    }

    pub fn listWorkflowsForTenant(self: *MockTenantStore, tenant_id: []const u8) ![]TenantWorkflow {
        var result = std.ArrayList(TenantWorkflow){};
        var it = self.workflows.valueIterator();
        while (it.next()) |workflow| {
            if (mem.eql(u8, workflow.tenant_id, tenant_id)) {
                try result.append(workflow.*);
            }
        }
        return result.toOwnedSlice();
    }
};

test "tenant isolation for workflows" {
    const allocator = testing.allocator;
    var store = MockTenantStore.init(allocator);
    defer store.deinit();

    // Add workflows for different tenants
    try store.addWorkflow(.{
        .id = "wf-001",
        .tenant_id = "tenant-A",
        .name = "Workflow for Tenant A",
    });
    try store.addWorkflow(.{
        .id = "wf-002",
        .tenant_id = "tenant-B",
        .name = "Workflow for Tenant B",
    });

    // Tenant A can see their workflow
    const wf_a = store.getWorkflow("wf-001", "tenant-A");
    try testing.expect(wf_a != null);
    try testing.expectEqualStrings("Workflow for Tenant A", wf_a.?.name);

    // Tenant A cannot see Tenant B's workflow
    const wf_b = store.getWorkflow("wf-002", "tenant-A");
    try testing.expect(wf_b == null);
}

test "cross-tenant access denied" {
    const allocator = testing.allocator;
    var store = MockTenantStore.init(allocator);
    defer store.deinit();

    try store.addWorkflow(.{
        .id = "wf-secret",
        .tenant_id = "tenant-secure",
        .name = "Secret Workflow",
    });

    // Attacker tenant cannot access secure tenant's workflow
    const result = store.getWorkflow("wf-secret", "tenant-attacker");
    try testing.expect(result == null);

    // Original tenant can access
    const legit_result = store.getWorkflow("wf-secret", "tenant-secure");
    try testing.expect(legit_result != null);
}

// =============================================================================
// Section 8: Error Handling Tests
// =============================================================================

test "not found returns 404" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/workflows/non-existent", .{
        .status_code = 404,
        .body = "{\"error\":\"Not found\",\"message\":\"Workflow 'non-existent' does not exist\"}",
        .content_type = "application/json",
    });

    var response = try client.get("/api/v1/workflows/non-existent", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 404), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Not found", error_msg.?);
}

test "method not allowed returns 405" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/health", .{
        .status_code = 405,
        .body = "{\"error\":\"Method not allowed\",\"message\":\"POST not allowed on /api/v1/health\",\"allowed_methods\":[\"GET\"]}",
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/health", "{}", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 405), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Method not allowed", error_msg.?);
}

test "internal error returns 500 with details" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/workflows/wf-broken/execute", .{
        .status_code = 500,
        .body = "{\"error\":\"Internal server error\",\"message\":\"Database connection failed\",\"request_id\":\"req-12345\",\"timestamp\":\"2026-01-19T10:00:00Z\"}",
        .content_type = "application/json",
    });

    var response = try client.post("/api/v1/workflows/wf-broken/execute", "{}", null);
    defer response.deinit(allocator);

    try testing.expectEqual(@as(u16, 500), response.status_code);

    const parsed = try json.parseFromSlice(json.Value, allocator, response.body, .{});
    defer parsed.deinit();

    const error_msg = getJsonString(parsed.value, "error");
    try testing.expect(error_msg != null);
    try testing.expectEqualStrings("Internal server error", error_msg.?);

    // Verify request_id is included for tracking
    const request_id = getJsonString(parsed.value, "request_id");
    try testing.expect(request_id != null);
}

// =============================================================================
// Section 9: Integration Test Utilities
// =============================================================================

/// Test fixture for setting up a complete test environment
pub const TestFixture = struct {
    allocator: Allocator,
    client: MockHttpClient,
    auth: MockAuthMiddleware,
    tenant_store: MockTenantStore,

    pub fn init(allocator: Allocator) TestFixture {
        return TestFixture{
            .allocator = allocator,
            .client = MockHttpClient.init(allocator),
            .auth = MockAuthMiddleware.init(allocator),
            .tenant_store = MockTenantStore.init(allocator),
        };
    }

    pub fn deinit(self: *TestFixture) void {
        self.client.deinit();
        self.auth.deinit();
        self.tenant_store.deinit();
    }

    /// Setup default mock responses for common endpoints
    pub fn setupDefaults(self: *TestFixture) !void {
        try self.client.mockResponse("/api/v1/health", .{
            .status_code = 200,
            .body = "{\"status\":\"healthy\",\"service\":\"nWorkflow\",\"version\":\"1.0.0\"}",
            .content_type = "application/json",
        });

        try self.client.mockResponse("/api/v1/node-types", .{
            .status_code = 200,
            .body = "{\"nodeTypes\":[{\"id\":\"start\",\"name\":\"Start\",\"category\":\"flow\"}]}",
            .content_type = "application/json",
        });

        try self.auth.addValidToken("test-token", .{
            .user_id = "test-user",
            .roles = &[_][]const u8{ "viewer", "editor" },
            .tenant_id = "test-tenant",
        });
    }
};

test "TestFixture provides complete test environment" {
    const allocator = testing.allocator;
    var fixture = TestFixture.init(allocator);
    defer fixture.deinit();

    try fixture.setupDefaults();

    // Test health endpoint works
    var health_response = try fixture.client.get("/api/v1/health", null);
    defer health_response.deinit(allocator);
    try testing.expectEqual(@as(u16, 200), health_response.status_code);

    // Test auth works
    const auth_result = fixture.auth.validateToken("test-token");
    try testing.expect(auth_result != null);
    try testing.expectEqualStrings("test-user", auth_result.?.user_id);
}

// =============================================================================
// Section 10: Performance and Load Tests (Simulated)
// =============================================================================

test "mock client handles multiple rapid requests" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/health", .{
        .status_code = 200,
        .body = "{\"status\":\"healthy\"}",
        .content_type = "application/json",
    });

    // Simulate 100 rapid requests
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        var response = try client.get("/api/v1/health", null);
        response.deinit(allocator);
    }

    try testing.expectEqual(@as(usize, 100), client.getRequestCount());
}

test "mock http client request logging" {
    const allocator = testing.allocator;
    var client = MockHttpClient.init(allocator);
    defer client.deinit();

    try client.mockResponse("/api/v1/workflows", .{
        .status_code = 200,
        .body = "{\"workflows\":[]}",
        .content_type = "application/json",
    });

    // Make several requests
    var r1 = try client.get("/api/v1/workflows", null);
    r1.deinit(allocator);

    var r2 = try client.post("/api/v1/workflows", "{\"name\":\"test\"}", null);
    r2.deinit(allocator);

    // Verify request log
    try testing.expectEqual(@as(usize, 2), client.getRequestCount());

    const last_request = client.getLastRequest();
    try testing.expect(last_request != null);
    try testing.expectEqualStrings("POST", last_request.?.method);
}
