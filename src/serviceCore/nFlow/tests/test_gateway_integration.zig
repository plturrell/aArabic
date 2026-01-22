const std = @import("std");
const testing = std.testing;

const ApisixClient = @import("../gateway/apisix_client.zig").ApisixClient;
const ApisixConfig = @import("../gateway/apisix_client.zig").ApisixConfig;
const RouteConfig = @import("../gateway/apisix_client.zig").RouteConfig;
const PluginConfig = @import("../gateway/apisix_client.zig").PluginConfig;
const WorkflowRouteManager = @import("../gateway/workflow_route_manager.zig").WorkflowRouteManager;
const WorkflowRouteConfig = @import("../gateway/workflow_route_manager.zig").WorkflowRouteConfig;
const ApiKeyManager = @import("../gateway/api_key_manager.zig").ApiKeyManager;
const ApiKeyScope = @import("../gateway/api_key_manager.zig").ApiKeyScope;

// Integration Tests for APISIX Gateway

test "APISIX Gateway Integration: Full workflow lifecycle" {
    const allocator = testing.allocator;

    // Setup APISIX client
    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    // Create route manager
    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    // Register a workflow with rate limiting
    const workflow_config = WorkflowRouteConfig{
        .rate_limit = .{
            .count = 100,
            .time_window = 60,
            .key_type = "consumer",
        },
        .enable_websocket = true,
    };

    try route_manager.registerWorkflow("test-workflow-001", workflow_config);

    // Verify workflow was registered
    const route = route_manager.getWorkflowRoute("test-workflow-001");
    try testing.expect(route != null);
    try testing.expectEqualStrings("test-workflow-001", route.?.workflow_id);

    // Unregister workflow
    try route_manager.unregisterWorkflow("test-workflow-001");

    // Verify workflow was unregistered
    const removed_route = route_manager.getWorkflowRoute("test-workflow-001");
    try testing.expect(removed_route == null);
}

test "APISIX Gateway Integration: Multiple workflows with different configurations" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    // Workflow 1: Basic config
    const config1 = WorkflowRouteConfig{};
    try route_manager.registerWorkflow("workflow-basic", config1);

    // Workflow 2: With rate limiting
    const config2 = WorkflowRouteConfig{
        .rate_limit = .{
            .count = 50,
            .time_window = 30,
            .key_type = "route",
        },
    };
    try route_manager.registerWorkflow("workflow-limited", config2);

    // Workflow 3: With CORS
    const config3 = WorkflowRouteConfig{
        .cors = .{
            .allow_origins = "*",
            .allow_methods = "GET,POST,PUT,DELETE",
        },
    };
    try route_manager.registerWorkflow("workflow-cors", config3);

    // Workflow 4: Full featured
    const config4 = WorkflowRouteConfig{
        .rate_limit = .{
            .count = 200,
            .time_window = 120,
            .key_type = "consumer",
        },
        .cors = .{
            .allow_origins = "https://example.com",
            .allow_methods = "GET,POST",
        },
        .enable_websocket = true,
        .priority = 10,
    };
    try route_manager.registerWorkflow("workflow-full", config4);

    // Verify all workflows are registered
    const workflows = try route_manager.listWorkflows();
    defer {
        for (workflows) |wf| {
            allocator.free(wf);
        }
        allocator.free(workflows);
    }

    try testing.expectEqual(@as(usize, 4), workflows.len);
}

test "API Key Integration: Complete key management lifecycle" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const key_manager = try ApiKeyManager.init(allocator, apisix_client);
    defer key_manager.deinit();

    // Generate global key
    const global_key = try key_manager.generateKey(.global, "Global access key");
    defer allocator.free(global_key);

    // Generate workflow-scoped key
    const workflow_scope = ApiKeyScope{ .workflow = "workflow-123" };
    const workflow_key = try key_manager.generateKey(workflow_scope, "Workflow-specific key");
    defer allocator.free(workflow_key);

    // Generate user-scoped key
    const user_scope = ApiKeyScope{ .user = "user-456" };
    const user_key = try key_manager.generateKey(user_scope, "User access key");
    defer allocator.free(user_key);

    // Validate keys
    try testing.expect(try key_manager.validateKey(global_key, null));
    try testing.expect(try key_manager.validateKey(workflow_key, "workflow-123"));
    try testing.expect(!try key_manager.validateKey(workflow_key, "workflow-999"));

    // List all keys
    const keys = try key_manager.listKeys(null);
    defer {
        for (keys) |*key_info| {
            key_info.deinit(allocator);
        }
        allocator.free(keys);
    }

    try testing.expectEqual(@as(usize, 3), keys.len);

    // Revoke a key
    try key_manager.revokeKey(workflow_key);
    try testing.expect(!try key_manager.validateKey(workflow_key, "workflow-123"));
}

test "API Key Integration: Key rotation and expiration" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const key_manager = try ApiKeyManager.init(allocator, apisix_client);
    defer key_manager.deinit();

    // Generate key
    const original_key = try key_manager.generateKey(.global, "Original key");
    defer allocator.free(original_key);

    // Rotate key
    const rotated_key = try key_manager.rotateKey(original_key);
    defer allocator.free(rotated_key);

    // Original key should be invalid
    try testing.expect(!try key_manager.validateKey(original_key, null));

    // Rotated key should be valid
    try testing.expect(try key_manager.validateKey(rotated_key, null));

    // Test expiring key
    const expiring_key = try key_manager.generateKeyWithExpiration(
        .global,
        "Expiring key",
        2, // 2 seconds
    );
    defer allocator.free(expiring_key);

    // Should be valid immediately
    try testing.expect(try key_manager.validateKey(expiring_key, null));

    // Wait for expiration
    std.time.sleep(2_500_000_000); // 2.5 seconds

    // Should be invalid after expiration
    try testing.expect(!try key_manager.validateKey(expiring_key, null));

    // Cleanup expired keys
    const cleaned = try key_manager.cleanupExpiredKeys();
    try testing.expect(cleaned > 0);
}

test "Workflow Route Integration: Dynamic plugin management" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    // Register workflow
    const config = WorkflowRouteConfig{};
    try route_manager.registerWorkflow("dynamic-workflow", config);

    // Enable API key authentication
    try route_manager.enableApiKeyAuth("dynamic-workflow");

    // Update rate limiting
    try route_manager.updateRateLimit("dynamic-workflow", 500, 300);

    // Enable CORS
    try route_manager.enableCors("dynamic-workflow", "https://app.example.com", "GET,POST,PUT");

    // Enable JWT authentication
    const jwt_claims = [_][]const u8{ "sub", "exp", "iat" };
    try route_manager.enableJwtAuth("dynamic-workflow", "jwt-secret-key", &jwt_claims);

    // Verify workflow still exists
    const route = route_manager.getWorkflowRoute("dynamic-workflow");
    try testing.expect(route != null);
}

test "Complete Integration: Workflow with API Key Protection" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    // Setup components
    const apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    const key_manager = try ApiKeyManager.init(allocator, apisix_client);
    defer key_manager.deinit();

    // 1. Register workflow with rate limiting and CORS
    const workflow_config = WorkflowRouteConfig{
        .rate_limit = .{
            .count = 100,
            .time_window = 60,
            .key_type = "consumer",
        },
        .cors = .{
            .allow_origins = "*",
            .allow_methods = "POST",
        },
        .enable_websocket = false,
    };

    try route_manager.registerWorkflow("protected-workflow", workflow_config);

    // 2. Enable API key authentication
    try route_manager.enableApiKeyAuth("protected-workflow");

    // 3. Generate workflow-specific API key
    const workflow_scope = ApiKeyScope{ .workflow = "protected-workflow" };
    const api_key = try key_manager.generateKey(workflow_scope, "Workflow access key");
    defer allocator.free(api_key);

    // 4. Validate key for correct workflow
    try testing.expect(try key_manager.validateKey(api_key, "protected-workflow"));

    // 5. Validate key fails for different workflow
    try testing.expect(!try key_manager.validateKey(api_key, "other-workflow"));

    // 6. Get key info and verify usage tracking
    const key_info = key_manager.getKeyInfo(api_key).?;
    defer {
        var mutable_info = key_info;
        mutable_info.deinit(allocator);
    }

    try testing.expect(key_info.usage_count > 0);
    try testing.expect(key_info.last_used_at != null);

    // 7. Unregister workflow
    try route_manager.unregisterWorkflow("protected-workflow");
}

test "Performance: Register and manage 100 workflows" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    const start_time = std.time.milliTimestamp();

    // Register 100 workflows
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const workflow_id = try std.fmt.allocPrint(allocator, "perf-workflow-{d}", .{i});
        defer allocator.free(workflow_id);

        const config = WorkflowRouteConfig{
            .rate_limit = .{
                .count = 100,
                .time_window = 60,
                .key_type = "consumer",
            },
        };

        try route_manager.registerWorkflow(workflow_id, config);
    }

    const register_time = std.time.milliTimestamp() - start_time;

    // List all workflows
    const list_start = std.time.milliTimestamp();
    const workflows = try route_manager.listWorkflows();
    defer {
        for (workflows) |wf| {
            allocator.free(wf);
        }
        allocator.free(workflows);
    }
    const list_time = std.time.milliTimestamp() - list_start;

    // Verify count
    try testing.expectEqual(@as(usize, 100), workflows.len);

    // Performance assertions (adjust based on requirements)
    try testing.expect(register_time < 5000); // < 5 seconds for 100 workflows
    try testing.expect(list_time < 100); // < 100ms to list

    std.debug.print("\nPerformance Results:\n", .{});
    std.debug.print("  - Register 100 workflows: {d}ms\n", .{register_time});
    std.debug.print("  - List 100 workflows: {d}ms\n", .{list_time});
    std.debug.print("  - Avg per workflow: {d}ms\n", .{register_time / 100});
}

test "Error Handling: Invalid operations" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "integration-test-key",
    };

    const route_manager = try WorkflowRouteManager.init(
        allocator,
        apisix_config,
        "http://localhost:8090",
    );
    defer route_manager.deinit();

    const apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    const key_manager = try ApiKeyManager.init(allocator, apisix_client);
    defer key_manager.deinit();

    // Test unregistering non-existent workflow
    const unregister_result = route_manager.unregisterWorkflow("non-existent");
    try testing.expectError(error.WorkflowNotFound, unregister_result);

    // Test updating rate limit for non-existent workflow
    const update_result = route_manager.updateRateLimit("non-existent", 100, 60);
    try testing.expectError(error.WorkflowNotFound, update_result);

    // Test revoking non-existent key
    const revoke_result = key_manager.revokeKey("non-existent-key");
    try testing.expectError(error.KeyNotFound, revoke_result);

    // Test rotating non-existent key
    const rotate_result = key_manager.rotateKey("non-existent-key");
    try testing.expectError(error.KeyNotFound, rotate_result);
}
