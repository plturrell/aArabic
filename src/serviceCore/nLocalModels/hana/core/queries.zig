const std = @import("std");
const client_module = @import("client.zig");
const HanaClient = client_module.HanaClient;
const Parameter = client_module.Parameter;
const QueryResult = client_module.QueryResult;
const Allocator = std.mem.Allocator;

/// Router data structures for HANA persistence

pub const Assignment = struct {
    id: []const u8,
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32,
    status: []const u8, // ACTIVE, INACTIVE, TESTING, OVERRIDDEN
    assignment_method: []const u8, // AUTO, MANUAL, FALLBACK
    capabilities_json: []const u8,
    created_at: i64,
    updated_at: i64,
};

pub const RoutingDecision = struct {
    id: []const u8,
    request_id: []const u8,
    task_type: []const u8, // coding, math, reasoning, arabic, general
    agent_id: []const u8,
    model_id: []const u8,
    capability_score: f32,
    performance_score: f32,
    composite_score: f32,
    strategy_used: []const u8, // balanced, speed, quality, cost
    latency_ms: i32,
    success: bool,
    fallback_used: bool,
    timestamp: i64,
};

pub const InferenceMetrics = struct {
    id: []const u8,
    model_id: []const u8,
    latency_ms: i32,
    ttft_ms: i32, // Time to first token
    tokens_generated: u32,
    cache_hit: bool,
    timestamp: i64,
};

/// Assignment operations

pub fn saveAssignment(hana_client: *HanaClient, assignment: Assignment) !void {
    const sql = 
        \\INSERT INTO AGENT_MODEL_ASSIGNMENTS (
        \\  ASSIGNMENT_ID, AGENT_ID, MODEL_ID, MATCH_SCORE, 
        \\  STATUS, ASSIGNMENT_METHOD, CAPABILITIES, 
        \\  ASSIGNED_AT, LAST_UPDATED
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query to prevent SQL injection
    const params = [_]Parameter{
        .{ .string = assignment.id },
        .{ .string = assignment.agent_id },
        .{ .string = assignment.model_id },
        .{ .float = assignment.match_score },
        .{ .string = assignment.status },
        .{ .string = assignment.assignment_method },
        .{ .string = assignment.capabilities_json },
    };
    
    try hana_client.executeParameterized(sql, &params);
}

pub fn getActiveAssignments(hana_client: *HanaClient, allocator: Allocator) ![]Assignment {
    const sql = 
        \\SELECT 
        \\  ASSIGNMENT_ID, AGENT_ID, MODEL_ID, MATCH_SCORE,
        \\  STATUS, ASSIGNMENT_METHOD, CAPABILITIES,
        \\  ASSIGNED_AT, LAST_UPDATED
        \\FROM AGENT_MODEL_ASSIGNMENTS
        \\WHERE STATUS = ?
        \\ORDER BY MATCH_SCORE DESC
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .string = "ACTIVE" },
    };
    
    const result = try hana_client.queryParameterized(sql, &params, allocator);
    defer result.deinit();
    
    // ✅ P1-9 FIXED: Parse result rows into Assignment structs
    var assignments = std.ArrayList(Assignment){};
    errdefer assignments.deinit();
    
    for (result.rows) |row| {
        if (row.values.len >= 8) {
            const assignment = Assignment{
                .id = row.values[0].asString() orelse "",
                .agent_id = row.values[1].asString() orelse "",
                .model_id = row.values[2].asString() orelse "",
                .match_score = @floatCast(row.values[3].asFloat() orelse 0.0),
                .status = row.values[4].asString() orelse "INACTIVE",
                .assignment_method = row.values[5].asString() orelse "UNKNOWN",
                .capabilities_json = row.values[6].asString() orelse "{}",
                .created_at = row.values[7].asInt() orelse 0,
                .updated_at = if (row.values.len > 8) row.values[8].asInt() orelse 0 else 0,
            };
            try assignments.append(assignment);
        }
    }
    
    return assignments.toOwnedSlice();
}

pub fn updateAssignmentMetrics(
    hana_client: *HanaClient,
    assignment_id: []const u8,
    success: bool,
    latency_ms: i32
) !void {
    const sql = 
        \\CALL SP_UPDATE_ASSIGNMENT_METRICS(?, ?, ?)
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .string = assignment_id },
        .{ .bool_value = success },
        .{ .int = latency_ms },
    };
    
    try hana_client.executeParameterized(sql, &params);
}

/// Routing decision operations

pub fn saveRoutingDecision(hana_client: *HanaClient, decision: RoutingDecision) !void {
    const sql = 
        \\INSERT INTO ROUTING_DECISIONS (
        \\  DECISION_ID, REQUEST_ID, TASK_TYPE, AGENT_ID, MODEL_ID,
        \\  CAPABILITY_SCORE, PERFORMANCE_SCORE, COMPOSITE_SCORE,
        \\  STRATEGY_USED, LATENCY_MS, SUCCESS, FALLBACK_USED,
        \\  DECISION_TIMESTAMP
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query to prevent SQL injection
    const params = [_]Parameter{
        .{ .string = decision.id },
        .{ .string = decision.request_id },
        .{ .string = decision.task_type },
        .{ .string = decision.agent_id },
        .{ .string = decision.model_id },
        .{ .float = decision.capability_score },
        .{ .float = decision.performance_score },
        .{ .float = decision.composite_score },
        .{ .string = decision.strategy_used },
        .{ .int = decision.latency_ms },
        .{ .bool_value = decision.success },
        .{ .bool_value = decision.fallback_used },
    };
    
    try hana_client.executeParameterized(sql, &params);
}

pub const RoutingStats = struct {
    total_decisions: u64,
    successful_decisions: u64,
    avg_latency_ms: f32,
    fallback_rate: f32,
    decisions_by_task: std.StringHashMap(u64),
    decisions_by_strategy: std.StringHashMap(u64),
};

pub fn getRoutingStats(hana_client: *HanaClient, hours: u32, allocator: Allocator) !RoutingStats {
    const sql = 
        \\SELECT 
        \\  COUNT(*) as total,
        \\  SUM(CASE WHEN SUCCESS = true THEN 1 ELSE 0 END) as successful,
        \\  AVG(LATENCY_MS) as avg_latency,
        \\  SUM(CASE WHEN FALLBACK_USED = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as fallback_rate
        \\FROM ROUTING_DECISIONS
        \\WHERE DECISION_TIMESTAMP >= ADD_HOURS(CURRENT_TIMESTAMP, -?)
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .int = hours },
    };
    
    const result = try hana_client.queryParameterized(sql, &params, allocator);
    defer result.deinit();
    
    // ✅ P1-9 FIXED: Parse result into RoutingStats
    if (result.rows.len > 0) {
        const row = result.rows[0];
        return RoutingStats{
            .total_decisions = @intCast(row.getInt("total")),
            .successful_decisions = @intCast(row.getInt("successful")),
            .avg_latency_ms = @floatCast(row.getFloat("avg_latency")),
            .fallback_rate = @floatCast(row.getFloat("fallback_rate")),
            .decisions_by_task = std.StringHashMap(u64).init(allocator),
            .decisions_by_strategy = std.StringHashMap(u64).init(allocator),
        };
    }
    
    return RoutingStats{
        .total_decisions = 0,
        .successful_decisions = 0,
        .avg_latency_ms = 0.0,
        .fallback_rate = 0.0,
        .decisions_by_task = std.StringHashMap(u64).init(allocator),
        .decisions_by_strategy = std.StringHashMap(u64).init(allocator),
    };
}

/// Model performance operations

pub const ModelPerformance = struct {
    model_id: []const u8,
    total_requests: u64,
    successful_requests: u64,
    success_rate: f32,
    avg_latency_ms: f32,
    p95_latency_ms: f32,
    p99_latency_ms: f32,
    total_agents: u32,
};

pub fn getModelPerformance(hana_client: *HanaClient, model_id: []const u8, allocator: Allocator) !ModelPerformance {
    const sql = 
        \\SELECT 
        \\  MODEL_ID,
        \\  COUNT(*) as total,
        \\  SUM(CASE WHEN SUCCESS = true THEN 1 ELSE 0 END) as successful,
        \\  AVG(LATENCY_MS) as avg_latency
        \\FROM ROUTING_DECISIONS
        \\WHERE MODEL_ID = ?
        \\  AND DECISION_TIMESTAMP >= ADD_HOURS(CURRENT_TIMESTAMP, -24)
        \\GROUP BY MODEL_ID
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .string = model_id },
    };
    
    const result = try hana_client.queryParameterized(sql, &params, allocator);
    defer result.deinit();
    
    // ✅ P1-9 FIXED: Parse result into ModelPerformance
    if (result.rows.len > 0) {
        const row = result.rows[0];
        const total = row.getInt("total");
        const successful = row.getInt("successful");
        
        return ModelPerformance{
            .model_id = model_id,
            .total_requests = @intCast(total),
            .successful_requests = @intCast(successful),
            .success_rate = if (total > 0) @as(f32, @floatFromInt(successful)) / @as(f32, @floatFromInt(total)) else 0.0,
            .avg_latency_ms = @floatCast(row.getFloat("avg_latency")),
            .p95_latency_ms = 0.0, // Would need separate query for percentiles
            .p99_latency_ms = 0.0, // Would need separate query for percentiles
            .total_agents = 0, // Would need separate query for agent count
        };
    }
    
    return ModelPerformance{
        .model_id = model_id,
        .total_requests = 0,
        .successful_requests = 0,
        .success_rate = 0.0,
        .avg_latency_ms = 0.0,
        .p95_latency_ms = 0.0,
        .p99_latency_ms = 0.0,
        .total_agents = 0,
    };
}

/// Analytics operations

pub const AgentModelPair = struct {
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32,
    success_rate: f32,
    avg_latency_ms: f32,
    recent_requests: u64,
};

pub fn getTopAgentModelPairs(hana_client: *HanaClient, limit: u32, allocator: Allocator) ![]AgentModelPair {
    const sql = 
        \\SELECT * FROM V_TOP_AGENT_MODEL_PAIRS
        \\LIMIT ?
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .int = limit },
    };
    
    const result = try hana_client.queryParameterized(sql, &params, allocator);
    defer result.deinit();
    
    // ✅ P1-9 FIXED: Parse result into AgentModelPair array
    var pairs = std.ArrayList(AgentModelPair){};
    errdefer pairs.deinit();
    
    for (result.rows) |row| {
        if (row.values.len >= 6) {
            const pair = AgentModelPair{
                .agent_id = row.values[0].asString() orelse "",
                .model_id = row.values[1].asString() orelse "",
                .match_score = @floatCast(row.values[2].asFloat() orelse 0.0),
                .success_rate = @floatCast(row.values[3].asFloat() orelse 0.0),
                .avg_latency_ms = @floatCast(row.values[4].asFloat() orelse 0.0),
                .recent_requests = @intCast(row.values[5].asInt() orelse 0),
            };
            try pairs.append(pair);
        }
    }
    
    return pairs.toOwnedSlice();
}

pub const AnalyticsSummary = struct {
    total_decisions: u64,
    successful_rate: f32,
    avg_latency_ms: f32,
    fallback_rate: f32,
    top_task_type: []const u8,
    top_strategy: []const u8,
};

pub fn getRoutingAnalytics24H(hana_client: *HanaClient, allocator: Allocator) !AnalyticsSummary {
    const sql = 
        \\SELECT * FROM V_ROUTING_ANALYTICS_24H
    ;
    
    const result = try hana_client.query(sql, allocator);
    defer result.deinit();
    
    // ✅ P1-9 FIXED: Parse result into AnalyticsSummary
    if (result.rows.len > 0) {
        const row = result.rows[0];
        return AnalyticsSummary{
            .total_decisions = @intCast(row.getInt("total_decisions")),
            .successful_rate = @floatCast(row.getFloat("successful_rate")),
            .avg_latency_ms = @floatCast(row.getFloat("avg_latency_ms")),
            .fallback_rate = @floatCast(row.getFloat("fallback_rate")),
            .top_task_type = row.getString("top_task_type") orelse "general",
            .top_strategy = row.getString("top_strategy") orelse "balanced",
        };
    }
    
    return AnalyticsSummary{
        .total_decisions = 0,
        .successful_rate = 0.0,
        .avg_latency_ms = 0.0,
        .fallback_rate = 0.0,
        .top_task_type = "general",
        .top_strategy = "balanced",
    };
}

/// Inference metrics operations

pub fn saveMetrics(hana_client: *HanaClient, metrics: InferenceMetrics) !void {
    const sql = 
        \\INSERT INTO INFERENCE_METRICS (
        \\  METRIC_ID, MODEL_ID, LATENCY_MS, TTFT_MS,
        \\  TOKENS_GENERATED, CACHE_HIT, TIMESTAMP
        \\) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ;
    
    // ✅ P1-9 FIXED: Use parameterized query
    const params = [_]Parameter{
        .{ .string = metrics.id },
        .{ .string = metrics.model_id },
        .{ .int = metrics.latency_ms },
        .{ .int = metrics.ttft_ms },
        .{ .int = metrics.tokens_generated },
        .{ .bool_value = metrics.cache_hit },
    };
    
    try hana_client.executeParameterized(sql, &params);
}

pub fn saveMetricsBatch(hana_client: *HanaClient, metrics_list: []const InferenceMetrics) !void {
    // ✅ P1-9 FIXED: Batch insert with transaction
    if (metrics_list.len == 0) return;
    
    // Begin transaction
    try hana_client.execute("BEGIN TRANSACTION");
    
    // Insert each metric
    for (metrics_list) |metrics| {
        saveMetrics(hana_client, metrics) catch |err| {
            // Rollback on error
            _ = hana_client.execute("ROLLBACK");
            return err;
        };
    }
    
    // Commit transaction
    try hana_client.execute("COMMIT");
}

/// Helper functions for ID generation

pub fn generateId(allocator: Allocator, prefix: []const u8) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    const random = std.crypto.random.int(u32);
    
    return try std.fmt.allocPrint(allocator, "{s}_{d}_{d}", .{ prefix, timestamp, random });
}

pub fn generateAssignmentId(allocator: Allocator) ![]const u8 {
    return generateId(allocator, "assign");
}

pub fn generateDecisionId(allocator: Allocator) ![]const u8 {
    return generateId(allocator, "decision");
}

pub fn generateMetricsId(allocator: Allocator) ![]const u8 {
    return generateId(allocator, "metric");
}

// Tests

test "generateId creates unique IDs" {
    const allocator = std.testing.allocator;
    
    const id1 = try generateId(allocator, "test");
    defer allocator.free(id1);
    
    const id2 = try generateId(allocator, "test");
    defer allocator.free(id2);
    
    try std.testing.expect(!std.mem.eql(u8, id1, id2));
}

test "Assignment structure size" {
    // Ensure Assignment struct has reasonable size
    const size = @sizeOf(Assignment);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 1024); // Should be less than 1KB
}

test "RoutingDecision structure size" {
    // Ensure RoutingDecision struct has reasonable size
    const size = @sizeOf(RoutingDecision);
    try std.testing.expect(size > 0);
    try std.testing.expect(size < 1024); // Should be less than 1KB
}
