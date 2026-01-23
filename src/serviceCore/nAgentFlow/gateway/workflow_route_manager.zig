const std = @import("std");
const Allocator = std.mem.Allocator;
const ApisixClient = @import("apisix_client.zig").ApisixClient;
const ApisixConfig = @import("apisix_client.zig").ApisixConfig;
const RouteConfig = @import("apisix_client.zig").RouteConfig;
const PluginConfig = @import("apisix_client.zig").PluginConfig;

/// Manages APISIX routes for nWorkflow workflows
/// Automatically creates, updates, and deletes routes when workflows are modified
pub const WorkflowRouteManager = struct {
    allocator: Allocator,
    apisix_client: *ApisixClient,
    workflow_routes: std.StringHashMap(WorkflowRoute),
    base_upstream_url: []const u8,

    pub fn init(allocator: Allocator, apisix_config: ApisixConfig, base_upstream_url: []const u8) !*WorkflowRouteManager {
        const manager = try allocator.create(WorkflowRouteManager);
        errdefer allocator.destroy(manager);

        const client = try ApisixClient.init(allocator, apisix_config);
        errdefer client.deinit();

        manager.* = .{
            .allocator = allocator,
            .apisix_client = client,
            .workflow_routes = std.StringHashMap(WorkflowRoute).init(allocator),
            .base_upstream_url = try allocator.dupe(u8, base_upstream_url),
        };

        return manager;
    }

    pub fn deinit(self: *WorkflowRouteManager) void {
        // Clean up all workflow routes
        var iterator = self.workflow_routes.iterator();
        while (iterator.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            entry.value_ptr.deinit(self.allocator);
        }
        self.workflow_routes.deinit();

        self.allocator.free(self.base_upstream_url);
        self.apisix_client.deinit();
        self.allocator.destroy(self);
    }

    /// Register a workflow with APISIX gateway
    /// Creates routes for: execute, status, logs
    pub fn registerWorkflow(self: *WorkflowRouteManager, workflow_id: []const u8, config: WorkflowRouteConfig) !void {
        // Create main execution route
        const execute_route_id = try self.createExecutionRoute(workflow_id, config);
        errdefer self.apisix_client.deleteRoute(execute_route_id) catch {};

        // Create status route
        const status_route_id = try self.createStatusRoute(workflow_id, config);
        errdefer self.apisix_client.deleteRoute(status_route_id) catch {};

        // Create logs route
        const logs_route_id = try self.createLogsRoute(workflow_id, config);
        errdefer self.apisix_client.deleteRoute(logs_route_id) catch {};

        // Create WebSocket route if enabled
        var ws_route_id: ?[]const u8 = null;
        if (config.enable_websocket) {
            ws_route_id = try self.createWebSocketRoute(workflow_id, config);
        }

        // Store workflow route information
        const workflow_route = WorkflowRoute{
            .workflow_id = try self.allocator.dupe(u8, workflow_id),
            .execute_route_id = execute_route_id,
            .status_route_id = status_route_id,
            .logs_route_id = logs_route_id,
            .websocket_route_id = ws_route_id,
        };

        try self.workflow_routes.put(try self.allocator.dupe(u8, workflow_id), workflow_route);
    }

    /// Unregister a workflow from APISIX gateway
    pub fn unregisterWorkflow(self: *WorkflowRouteManager, workflow_id: []const u8) !void {
        const workflow_route = self.workflow_routes.get(workflow_id) orelse return error.WorkflowNotFound;

        // Delete all routes
        try self.apisix_client.deleteRoute(workflow_route.execute_route_id);
        try self.apisix_client.deleteRoute(workflow_route.status_route_id);
        try self.apisix_client.deleteRoute(workflow_route.logs_route_id);

        if (workflow_route.websocket_route_id) |ws_id| {
            try self.apisix_client.deleteRoute(ws_id);
        }

        // Remove from registry
        if (self.workflow_routes.fetchRemove(workflow_id)) |entry| {
            self.allocator.free(entry.key);
            var value_copy = entry.value;
            value_copy.deinit(self.allocator);
        }
    }

    /// Update rate limiting for a workflow
    pub fn updateRateLimit(self: *WorkflowRouteManager, workflow_id: []const u8, count: u32, time_window: u32) !void {
        const workflow_route = self.workflow_routes.get(workflow_id) orelse return error.WorkflowNotFound;

        const rate_limit_plugin = PluginConfig{ .rate_limit = .{
            .count = count,
            .time_window = time_window,
            .key_type = "consumer",
        } };

        // Apply to all routes
        try self.apisix_client.enablePlugin(workflow_route.execute_route_id, rate_limit_plugin);
        try self.apisix_client.enablePlugin(workflow_route.status_route_id, rate_limit_plugin);
        try self.apisix_client.enablePlugin(workflow_route.logs_route_id, rate_limit_plugin);
    }

    /// Enable API key authentication for a workflow
    pub fn enableApiKeyAuth(self: *WorkflowRouteManager, workflow_id: []const u8) !void {
        const workflow_route = self.workflow_routes.get(workflow_id) orelse return error.WorkflowNotFound;

        const key_auth_plugin = PluginConfig{ .key_auth = .{
            .header = "X-API-Key",
        } };

        try self.apisix_client.enablePlugin(workflow_route.execute_route_id, key_auth_plugin);
        try self.apisix_client.enablePlugin(workflow_route.status_route_id, key_auth_plugin);
        try self.apisix_client.enablePlugin(workflow_route.logs_route_id, key_auth_plugin);
    }

    /// Enable JWT authentication for a workflow
    pub fn enableJwtAuth(self: *WorkflowRouteManager, workflow_id: []const u8, secret: []const u8, claims: []const []const u8) !void {
        const workflow_route = self.workflow_routes.get(workflow_id) orelse return error.WorkflowNotFound;

        const jwt_plugin = PluginConfig{ .jwt_auth = .{
            .secret = secret,
            .claims_to_verify = claims,
        } };

        try self.apisix_client.enablePlugin(workflow_route.execute_route_id, jwt_plugin);
        try self.apisix_client.enablePlugin(workflow_route.status_route_id, jwt_plugin);
        try self.apisix_client.enablePlugin(workflow_route.logs_route_id, jwt_plugin);
    }

    /// Enable CORS for a workflow
    pub fn enableCors(self: *WorkflowRouteManager, workflow_id: []const u8, allow_origins: []const u8, allow_methods: []const u8) !void {
        const workflow_route = self.workflow_routes.get(workflow_id) orelse return error.WorkflowNotFound;

        const cors_plugin = PluginConfig{ .cors = .{
            .allow_origins = allow_origins,
            .allow_methods = allow_methods,
        } };

        try self.apisix_client.enablePlugin(workflow_route.execute_route_id, cors_plugin);
        try self.apisix_client.enablePlugin(workflow_route.status_route_id, cors_plugin);
        try self.apisix_client.enablePlugin(workflow_route.logs_route_id, cors_plugin);
    }

    /// List all registered workflows
    pub fn listWorkflows(self: *const WorkflowRouteManager) ![][]const u8 {
        var list: std.ArrayListUnmanaged([]const u8) = .{};
        errdefer list.deinit(self.allocator);

        var iterator = self.workflow_routes.keyIterator();
        while (iterator.next()) |key| {
            try list.append(self.allocator, try self.allocator.dupe(u8, key.*));
        }

        return list.toOwnedSlice(self.allocator);
    }

    /// Get route information for a workflow
    pub fn getWorkflowRoute(self: *const WorkflowRouteManager, workflow_id: []const u8) ?WorkflowRoute {
        return self.workflow_routes.get(workflow_id);
    }

    // Private helper methods

    fn createExecutionRoute(self: *WorkflowRouteManager, workflow_id: []const u8, config: WorkflowRouteConfig) ![]const u8 {
        const uri = try std.fmt.allocPrint(self.allocator, "/api/v1/workflows/{s}/execute", .{workflow_id});
        defer self.allocator.free(uri);

        const methods = [_][]const u8{"POST"};

        var plugins: std.ArrayListUnmanaged(PluginConfig) = .{};
        defer plugins.deinit(self.allocator);

        // Add default rate limiting
        if (config.rate_limit) |rl| {
            try plugins.append(self.allocator, PluginConfig{ .rate_limit = .{
                .count = rl.count,
                .time_window = rl.time_window,
                .key_type = rl.key_type,
            } });
        }

        // Add CORS if specified
        if (config.cors) |cors| {
            try plugins.append(self.allocator, PluginConfig{ .cors = .{
                .allow_origins = cors.allow_origins,
                .allow_methods = cors.allow_methods,
            } });
        }

        const route = RouteConfig{
            .uri = uri,
            .methods = &methods,
            .upstream_url = self.base_upstream_url,
            .plugins = plugins.items,
            .priority = config.priority,
        };

        return try self.apisix_client.createRoute(route);
    }

    fn createStatusRoute(self: *WorkflowRouteManager, workflow_id: []const u8, config: WorkflowRouteConfig) ![]const u8 {
        const uri = try std.fmt.allocPrint(self.allocator, "/api/v1/workflows/{s}/status", .{workflow_id});
        defer self.allocator.free(uri);

        const methods = [_][]const u8{"GET"};

        const route = RouteConfig{
            .uri = uri,
            .methods = &methods,
            .upstream_url = self.base_upstream_url,
            .priority = config.priority,
        };

        return try self.apisix_client.createRoute(route);
    }

    fn createLogsRoute(self: *WorkflowRouteManager, workflow_id: []const u8, config: WorkflowRouteConfig) ![]const u8 {
        const uri = try std.fmt.allocPrint(self.allocator, "/api/v1/workflows/{s}/logs", .{workflow_id});
        defer self.allocator.free(uri);

        const methods = [_][]const u8{"GET"};

        const route = RouteConfig{
            .uri = uri,
            .methods = &methods,
            .upstream_url = self.base_upstream_url,
            .priority = config.priority,
        };

        return try self.apisix_client.createRoute(route);
    }

    fn createWebSocketRoute(self: *WorkflowRouteManager, workflow_id: []const u8, config: WorkflowRouteConfig) ![]const u8 {
        const uri = try std.fmt.allocPrint(self.allocator, "/ws/workflows/{s}", .{workflow_id});
        defer self.allocator.free(uri);

        const methods = [_][]const u8{"GET"};

        const route = RouteConfig{
            .uri = uri,
            .methods = &methods,
            .upstream_url = self.base_upstream_url,
            .priority = config.priority,
            .enable_websocket = true,
        };

        return try self.apisix_client.createRoute(route);
    }
};

/// Configuration for workflow routes
pub const WorkflowRouteConfig = struct {
    rate_limit: ?struct {
        count: u32,
        time_window: u32,
        key_type: []const u8,
    } = null,
    cors: ?struct {
        allow_origins: []const u8,
        allow_methods: []const u8,
    } = null,
    enable_websocket: bool = false,
    priority: i32 = 0,
};

/// Information about a workflow's routes
pub const WorkflowRoute = struct {
    workflow_id: []const u8,
    execute_route_id: []const u8,
    status_route_id: []const u8,
    logs_route_id: []const u8,
    websocket_route_id: ?[]const u8,

    pub fn deinit(self: *WorkflowRoute, allocator: Allocator) void {
        allocator.free(self.workflow_id);
        allocator.free(self.execute_route_id);
        allocator.free(self.status_route_id);
        allocator.free(self.logs_route_id);
        if (self.websocket_route_id) |ws_id| {
            allocator.free(ws_id);
        }
    }
};

// Tests
test "WorkflowRouteManager init and deinit" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    try std.testing.expectEqualStrings("http://localhost:8090", manager.base_upstream_url);
}

test "WorkflowRouteManager register workflow" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{
        .rate_limit = .{
            .count = 100,
            .time_window = 60,
            .key_type = "consumer",
        },
        .enable_websocket = true,
    };

    try manager.registerWorkflow("workflow-123", config);

    const route = manager.getWorkflowRoute("workflow-123");
    try std.testing.expect(route != null);
    try std.testing.expectEqualStrings("workflow-123", route.?.workflow_id);
    try std.testing.expect(route.?.websocket_route_id != null);
}

test "WorkflowRouteManager unregister workflow" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{};
    try manager.registerWorkflow("workflow-123", config);

    try manager.unregisterWorkflow("workflow-123");

    const route = manager.getWorkflowRoute("workflow-123");
    try std.testing.expect(route == null);
}

test "WorkflowRouteManager list workflows" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{};
    try manager.registerWorkflow("workflow-1", config);
    try manager.registerWorkflow("workflow-2", config);
    try manager.registerWorkflow("workflow-3", config);

    const workflows = try manager.listWorkflows();
    defer {
        for (workflows) |wf| {
            allocator.free(wf);
        }
        allocator.free(workflows);
    }

    try std.testing.expectEqual(@as(usize, 3), workflows.len);
}

test "WorkflowRouteManager update rate limit" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{};
    try manager.registerWorkflow("workflow-123", config);

    // Should not error
    try manager.updateRateLimit("workflow-123", 200, 120);
}

test "WorkflowRouteManager enable authentication" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{};
    try manager.registerWorkflow("workflow-123", config);

    // Enable API key auth
    try manager.enableApiKeyAuth("workflow-123");

    // Enable JWT auth
    const claims = [_][]const u8{ "sub", "exp" };
    try manager.enableJwtAuth("workflow-123", "secret-key", &claims);
}

test "WorkflowRouteManager enable CORS" {
    const allocator = std.testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    const manager = try WorkflowRouteManager.init(allocator, apisix_config, "http://localhost:8090");
    defer manager.deinit();

    const config = WorkflowRouteConfig{};
    try manager.registerWorkflow("workflow-123", config);

    try manager.enableCors("workflow-123", "https://example.com", "GET,POST");
}
