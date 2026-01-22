const std = @import("std");
const HanaClient = @import("../core/client.zig").HanaClient;
const queries = @import("../core/queries.zig");

/// Example: Using HANA for Router Persistence
/// 
/// This example demonstrates how to use the unified HANA module
/// to persist Router decisions and query analytics.

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Load configuration from environment
    const config = try HanaClient.HanaConfig{
        .host = std.os.getenv("HANA_HOST") orelse "localhost",
        .port = 30015,
        .database = std.os.getenv("HANA_DATABASE") orelse "NOPENAI_DB",
        .user = std.os.getenv("HANA_USER") orelse "NUCLEUS_APP",
        .password = std.os.getenv("HANA_PASSWORD") orelse "",
        .pool_min = 5,
        .pool_max = 10,
    };

    // Initialize HANA client
    std.log.info("Connecting to HANA at {s}:{d}...", .{ config.host, config.port });
    const client = try HanaClient.init(allocator, config);
    defer client.deinit();

    std.log.info("Connection pool initialized with {d}-{d} connections", .{ 
        config.pool_min, config.pool_max 
    });

    // Example 1: Save a routing decision
    try example1_saveDecision(client, allocator);

    // Example 2: Query routing stats
    try example2_queryStats(client, allocator);

    // Example 3: Get model performance
    try example3_modelPerformance(client, allocator);

    // Example 4: Save assignment
    try example4_saveAssignment(client, allocator);

    // Print final metrics
    const metrics = client.getMetrics();
    std.log.info("\nFinal Metrics:", .{});
    std.log.info("  Total Queries: {d}", .{metrics.total_queries});
    std.log.info("  Failed Queries: {d}", .{metrics.failed_queries});
    std.log.info("  Active Connections: {d}", .{metrics.active_connections});
    std.log.info("  Idle Connections: {d}", .{metrics.idle_connections});
}

fn example1_saveDecision(client: *HanaClient, allocator: std.mem.Allocator) !void {
    std.log.info("\n=== Example 1: Save Routing Decision ===", .{});

    const decision = queries.RoutingDecision{
        .id = try queries.generateDecisionId(allocator),
        .request_id = "req_example_001",
        .task_type = "coding",
        .agent_id = "gpu_agent_1",
        .model_id = "llama-70b-q4",
        .capability_score = 0.95,
        .performance_score = 0.88,
        .composite_score = 0.92,
        .strategy_used = "balanced",
        .latency_ms = 45,
        .success = true,
        .fallback_used = false,
        .timestamp = std.time.milliTimestamp(),
    };
    defer allocator.free(decision.id);

    std.log.info("Saving decision {s}...", .{decision.id});
    try queries.saveRoutingDecision(client, decision);
    std.log.info("✓ Decision saved successfully", .{});
}

fn example2_queryStats(client: *HanaClient, allocator: std.mem.Allocator) !void {
    std.log.info("\n=== Example 2: Query Routing Stats (Last 24h) ===", .{});

    const stats = try queries.getRoutingStats(client, 24, allocator);
    defer {
        stats.decisions_by_task.deinit();
        stats.decisions_by_strategy.deinit();
    }

    std.log.info("Total Decisions: {d}", .{stats.total_decisions});
    std.log.info("Successful: {d}", .{stats.successful_decisions});
    std.log.info("Average Latency: {d:.2}ms", .{stats.avg_latency_ms});
    std.log.info("Fallback Rate: {d:.1}%", .{stats.fallback_rate});
}

fn example3_modelPerformance(client: *HanaClient, allocator: std.mem.Allocator) !void {
    std.log.info("\n=== Example 3: Model Performance ===", .{});

    const perf = try queries.getModelPerformance(client, "llama-70b-q4", allocator);

    std.log.info("Model: {s}", .{perf.model_id});
    std.log.info("Total Requests: {d}", .{perf.total_requests});
    std.log.info("Success Rate: {d:.1}%", .{perf.success_rate * 100});
    std.log.info("Avg Latency: {d:.2}ms", .{perf.avg_latency_ms});
    std.log.info("P95 Latency: {d:.2}ms", .{perf.p95_latency_ms});
    std.log.info("P99 Latency: {d:.2}ms", .{perf.p99_latency_ms});
}

fn example4_saveAssignment(client: *HanaClient, allocator: std.mem.Allocator) !void {
    std.log.info("\n=== Example 4: Save Agent-Model Assignment ===", .{});

    const assignment = queries.Assignment{
        .id = try queries.generateAssignmentId(allocator),
        .agent_id = "gpu_agent_1",
        .model_id = "llama-70b-q4",
        .match_score = 0.95,
        .status = "ACTIVE",
        .assignment_method = "AUTO",
        .capabilities_json = "{\"coding\": 0.95, \"math\": 0.88}",
        .created_at = std.time.milliTimestamp(),
        .updated_at = std.time.milliTimestamp(),
    };
    defer allocator.free(assignment.id);

    std.log.info("Saving assignment {s}...", .{assignment.id});
    try queries.saveAssignment(client, assignment);
    std.log.info("✓ Assignment saved successfully", .{});
}
