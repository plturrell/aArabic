// Day 32 Gateway Integration Tests
// Tests for Load Balancer, Health Checks, and Transformations

const std = @import("std");
const testing = std.testing;
const ApisixClient = @import("../gateway/apisix_client.zig").ApisixClient;
const ApisixConfig = @import("../gateway/apisix_client.zig").ApisixConfig;
const LoadBalancerManager = @import("../gateway/load_balancer.zig").LoadBalancerManager;
const LoadBalancerType = @import("../gateway/load_balancer.zig").LoadBalancerType;
const UpstreamNode = @import("../gateway/load_balancer.zig").UpstreamNode;
const UpstreamConfig = @import("../gateway/load_balancer.zig").UpstreamConfig;
const HealthCheckConfig = @import("../gateway/load_balancer.zig").HealthCheckConfig;
const ActiveHealthCheck = @import("../gateway/load_balancer.zig").ActiveHealthCheck;
const TransformerManager = @import("../gateway/transformer.zig").TransformerManager;
const TransformConfig = @import("../gateway/transformer.zig").TransformConfig;
const HeaderTransform = @import("../gateway/transformer.zig").HeaderTransform;
const QueryTransform = @import("../gateway/transformer.zig").QueryTransform;
const UriRewrite = @import("../gateway/transformer.zig").UriRewrite;
const TemplateEngine = @import("../gateway/transformer.zig").TemplateEngine;
const JsonPathEvaluator = @import("../gateway/transformer.zig").JsonPathEvaluator;

test "LoadBalancer: roundrobin with multiple nodes" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const nodes = [_]UpstreamNode{
        .{ .host = "backend1.local", .port = 8080, .weight = 1 },
        .{ .host = "backend2.local", .port = 8080, .weight = 1 },
        .{ .host = "backend3.local", .port = 8080, .weight = 2 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = &nodes,
        .retries = 3,
    };

    const upstream_id = try lb_manager.createUpstream("web-service", config);
    defer allocator.free(upstream_id);

    const info = try lb_manager.getUpstream("web-service");
    try testing.expectEqual(@as(usize, 3), info.config.nodes.len);
    try testing.expectEqual(LoadBalancerType.roundrobin, info.config.type);
}

test "LoadBalancer: consistent hashing" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const nodes = [_]UpstreamNode{
        .{ .host = "cache1.local", .port = 6379, .weight = 1 },
        .{ .host = "cache2.local", .port = 6379, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .chash,
        .nodes = &nodes,
        .hash_on = "vars",
    };

    _ = try lb_manager.createUpstream("cache-service", config);

    const info = try lb_manager.getUpstream("cache-service");
    try testing.expectEqual(LoadBalancerType.chash, info.config.type);
}

test "LoadBalancer: least connections algorithm" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const nodes = [_]UpstreamNode{
        .{ .host = "worker1.local", .port = 3000, .weight = 1 },
        .{ .host = "worker2.local", .port = 3000, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .least_conn,
        .nodes = &nodes,
    };

    _ = try lb_manager.createUpstream("worker-pool", config);

    const info = try lb_manager.getUpstream("worker-pool");
    try testing.expectEqual(LoadBalancerType.least_conn, info.config.type);
}

test "HealthCheck: active health monitoring" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const nodes = [_]UpstreamNode{
        .{ .host = "api.local", .port = 8080, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = &nodes,
    };

    _ = try lb_manager.createUpstream("api-service", config);

    // Enable health checks
    const health_checks = HealthCheckConfig{
        .active = ActiveHealthCheck{
            .type = "http",
            .timeout = 1,
            .http_path = "/health",
            .healthy = .{
                .interval = 2,
                .successes = 2,
            },
            .unhealthy = .{
                .interval = 5,
                .http_failures = 3,
                .timeouts = 3,
            },
        },
    };

    try lb_manager.enableHealthChecks("api-service", health_checks);

    const info = try lb_manager.getUpstream("api-service");
    try testing.expect(info.config.checks != null);
    try testing.expect(info.config.checks.?.active != null);
}

test "HealthCheck: custom health endpoint" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const nodes = [_]UpstreamNode{
        .{ .host = "service.local", .port = 9000, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = &nodes,
    };

    _ = try lb_manager.createUpstream("custom-service", config);

    const health_checks = HealthCheckConfig{
        .active = ActiveHealthCheck{
            .http_path = "/api/v1/healthz",
            .healthy = .{ .interval = 1, .successes = 1 },
            .unhealthy = .{ .interval = 3, .http_failures = 2, .timeouts = 2 },
        },
    };

    try lb_manager.enableHealthChecks("custom-service", health_checks);

    const info = try lb_manager.getUpstream("custom-service");
    const active = info.config.checks.?.active.?;
    try testing.expectEqualStrings("/api/v1/healthz", active.http_path);
}

test "Transformation: add custom headers" {
    const allocator = testing.allocator;

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    const headers = [_]HeaderTransform{
        .{ .action = .add, .header_name = "X-Request-ID", .header_value = "12345" },
        .{ .action = .add, .header_name = "X-Tenant-ID", .header_value = "tenant-a" },
    };

    const config = TransformConfig{
        .headers = &headers,
    };

    try transformer.registerTransformation("route-1", config);

    const json = try transformer.serializeRequestTransformer(config);
    try testing.expect(std.mem.indexOf(u8, json, "X-Request-ID") != null);
    try testing.expect(std.mem.indexOf(u8, json, "X-Tenant-ID") != null);
}

test "Transformation: remove headers" {
    const allocator = testing.allocator;

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    const headers = [_]HeaderTransform{
        .{ .action = .remove, .header_name = "X-Internal-Token" },
        .{ .action = .remove, .header_name = "X-Debug-Info" },
    };

    const config = TransformConfig{
        .headers = &headers,
    };

    const json = try transformer.serializeRequestTransformer(config);
    try testing.expect(std.mem.indexOf(u8, json, "X-Internal-Token") != null);
}

test "Transformation: add query parameters" {
    const allocator = testing.allocator;

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    const query_params = [_]QueryTransform{
        .{ .action = .add, .param_name = "api_key", .param_value = "secret123" },
        .{ .action = .add, .param_name = "version", .param_value = "v2" },
    };

    const config = TransformConfig{
        .query_params = &query_params,
    };

    const json = try transformer.serializeRequestTransformer(config);
    try testing.expect(std.mem.indexOf(u8, json, "api_key") != null);
    try testing.expect(std.mem.indexOf(u8, json, "version") != null);
}

test "Transformation: URI rewrite patterns" {
    const allocator = testing.allocator;

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    const rewrite = UriRewrite{
        .regex = "^/api/v1/(.*)",
        .replacement = "/api/v2/$1",
    };

    const json = try transformer.serializeUriRewrite(rewrite);
    try testing.expect(std.mem.indexOf(u8, json, "/api/v1/") != null);
    try testing.expect(std.mem.indexOf(u8, json, "/api/v2/") != null);
}

test "Template: complex variable substitution" {
    const allocator = testing.allocator;

    var engine = TemplateEngine.init(allocator);

    var vars = std.StringHashMap([]const u8).init(allocator);
    defer vars.deinit();

    try vars.put("user_id", "12345");
    try vars.put("action", "purchase");
    try vars.put("amount", "99.99");

    const template = 
        \\{
        \\  "user": "{{user_id}}",
        \\  "action": "{{action}}",
        \\  "amount": {{amount}}
        \\}
    ;

    const result = try engine.render(template, vars);
    defer allocator.free(result);

    try testing.expect(std.mem.indexOf(u8, result, "\"user\": \"12345\"") != null);
    try testing.expect(std.mem.indexOf(u8, result, "\"action\": \"purchase\"") != null);
}

test "JsonPath: nested object navigation" {
    const allocator = testing.allocator;

    var evaluator = JsonPathEvaluator.init(allocator);

    const json = 
        \\{
        \\  "user": {
        \\    "profile": {
        \\      "name": "Alice",
        \\      "email": "alice@example.com"
        \\    }
        \\  }
        \\}
    ;

    const result = try evaluator.evaluate(json, "$.user.profile.email");

    if (result) |value| {
        defer allocator.free(value);
        try testing.expect(std.mem.indexOf(u8, value, "alice@example.com") != null);
    } else {
        try testing.expect(false);
    }
}

test "JsonPath: array filtering" {
    const allocator = testing.allocator;

    var evaluator = JsonPathEvaluator.init(allocator);

    const json = 
        \\{
        \\  "orders": [
        \\    {"id": 1, "status": "completed"},
        \\    {"id": 2, "status": "pending"},
        \\    {"id": 3, "status": "completed"}
        \\  ]
        \\}
    ;

    const result = try evaluator.evaluate(json, "$.orders[1].status");

    if (result) |value| {
        defer allocator.free(value);
        try testing.expect(std.mem.indexOf(u8, value, "pending") != null);
    } else {
        try testing.expect(false);
    }
}

test "Integration: complete workflow route with transformations" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    // 1. Create upstream with load balancing
    const nodes = [_]UpstreamNode{
        .{ .host = "workflow1.local", .port = 8090, .weight = 2 },
        .{ .host = "workflow2.local", .port = 8090, .weight = 1 },
    };

    const lb_config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = &nodes,
        .retries = 2,
    };

    const upstream_id = try lb_manager.createUpstream("workflow-backend", lb_config);
    defer allocator.free(upstream_id);

    // 2. Enable health checks
    const health_checks = HealthCheckConfig{
        .active = ActiveHealthCheck{
            .http_path = "/health",
            .healthy = .{ .interval = 5, .successes = 2 },
            .unhealthy = .{ .interval = 10, .http_failures = 3, .timeouts = 3 },
        },
    };

    try lb_manager.enableHealthChecks("workflow-backend", health_checks);

    // 3. Add transformations
    const headers = [_]HeaderTransform{
        .{ .action = .add, .header_name = "X-Workflow-Version", .header_value = "v2" },
        .{ .action = .remove, .header_name = "X-Internal-Debug" },
    };

    const transform_config = TransformConfig{
        .headers = &headers,
    };

    try transformer.registerTransformation("workflow-route-1", transform_config);

    // Verify everything was set up correctly
    const info = try lb_manager.getUpstream("workflow-backend");
    try testing.expectEqual(@as(usize, 2), info.config.nodes.len);
    try testing.expect(info.config.checks != null);

    const transform = transformer.getTransformation("workflow-route-1");
    try testing.expect(transform != null);
}

test "Performance: bulk upstream operations" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    const start_time = std.time.milliTimestamp();

    // Create 50 upstreams
    var i: usize = 0;
    while (i < 50) : (i += 1) {
        const nodes = [_]UpstreamNode{
            .{ .host = "backend.local", .port = 8080, .weight = 1 },
        };

        const config = UpstreamConfig{
            .type = .roundrobin,
            .nodes = &nodes,
        };

        const name = try std.fmt.allocPrint(allocator, "service-{d}", .{i});
        defer allocator.free(name);

        const upstream_id = try lb_manager.createUpstream(name, config);
        allocator.free(upstream_id);
    }

    const end_time = std.time.milliTimestamp();
    const duration = end_time - start_time;

    // Should create 50 upstreams in less than 1 second
    try testing.expect(duration < 1000);

    const upstreams = try lb_manager.listUpstreams();
    try testing.expectEqual(@as(usize, 50), upstreams.len);
}

test "Error: invalid upstream operations" {
    const allocator = testing.allocator;

    const apisix_config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var lb_manager = try LoadBalancerManager.init(allocator, &apisix_client);
    defer lb_manager.deinit();

    // Try to get non-existent upstream
    const result = lb_manager.getUpstream("non-existent");
    try testing.expectError(error.UpstreamNotFound, result);

    // Try to remove node from non-existent upstream
    const remove_result = lb_manager.removeNode("non-existent", "host", 8080);
    try testing.expectError(error.UpstreamNotFound, remove_result);
}

test "Error: invalid transformation operations" {
    const allocator = testing.allocator;

    var transformer = try TransformerManager.init(allocator);
    defer transformer.deinit();

    // Try to remove non-existent transformation
    const result = transformer.removeTransformation("non-existent");
    try testing.expectError(error.TransformationNotFound, result);

    // Get non-existent transformation returns null
    const transform = transformer.getTransformation("non-existent");
    try testing.expect(transform == null);
}
