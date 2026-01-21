const std = @import("std");
const HanaClient = @import("hana_client.zig").HanaClient;
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

pub fn saveAssignment(client: *HanaClient, assignment: Assignment) !void {
    const sql = 
        \\INSERT INTO AGENT_MODEL_ASSIGNMENTS (
        \\  ASSIGNMENT_ID, AGENT_ID, MODEL_ID, MATCH_SCORE, 
        \\  STATUS, ASSIGNMENT_METHOD, CAPABILITIES, 
        \\  ASSIGNED_AT, LAST_UPDATED
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
    ;
    
    // TODO: Use prepared statement with parameters
    _ = assignment;
    
    try client.execute(sql);
}

pub fn getActiveAssignments(client: *HanaClient, allocator: Allocator) ![]Assignment {
    const sql = 
        \\SELECT 
        \\  ASSIGNMENT_ID, AGENT_ID, MODEL_ID, MATCH_SCORE,
        \\  STATUS, ASSIGNMENT_METHOD, CAPABILITIES,
        \\  ASSIGNED_AT, LAST_UPDATED
        \\FROM AGENT_MODEL_ASSIGNMENTS
        \\WHERE STATUS = 'ACTIVE'
        \\ORDER BY MATCH_SCORE DESC
    ;
    
    const result = try client.query(sql, allocator);
    defer allocator.free(result);
    
    // TODO: Parse result rows into Assignment structs
    return try allocator.alloc(Assignment, 0);
}

pub fn updateAssignmentMetrics(
    client: *HanaClient,
    assignment_id: []const u8,
    success: bool,
    latency_ms: i32
) !void {
    // Call stored procedure SP_UPDATE_ASSIGNMENT_METRICS
    const sql = 
        \\CALL SP_UPDATE_ASSIGNMENT_METRICS(?, ?, ?)
    ;
    
    // TODO: Use prepared statement with parameters
    _ = assignment_id;
    _ = success;
    _ = latency_ms;
    
    try client.execute(sql);
}

/// Routing decision operations

pub fn saveRoutingDecision(client: *HanaClient, decision: RoutingDecision) !void {
    const sql = 
        \\INSERT INTO ROUTING_DECISIONS (
        \\  DECISION_ID, REQUEST_ID, TASK_TYPE, AGENT_ID, MODEL_ID,
        \\  CAPABILITY_SCORE, PERFORMANCE_SCORE, COMPOSITE_SCORE,
        \\  STRATEGY_USED, LATENCY_MS, SUCCESS, FALLBACK_USED,
        \\  DECISION_TIMESTAMP
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ;
    
    // TODO: Use prepared statement with parameters
    _ = decision;
    
    try client.execute(sql);
}

pub const RoutingStats = struct {
    total_decisions: u64,
    successful_decisions: u64,
    avg_latency_ms: f32,
    fallback_rate: f32,
    decisions_by_task: std.StringHashMap(u64),
    decisions_by_strategy: std.StringHashMap(u64),
};

pub fn getRoutingStats(client: *HanaClient, hours: u32, allocator: Allocator) !RoutingStats {
    const sql = 
        \\SELECT 
        \\  COUNT(*) as total,
        \\  SUM(CASE WHEN SUCCESS = true THEN 1 ELSE 0 END) as successful,
        \\  AVG(LATENCY_MS) as avg_latency,
        \\  SUM(CASE WHEN FALLBACK_USED = true THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as fallback_rate
        \\FROM ROUTING_DECISIONS
        \\WHERE DECISION_TIMESTAMP >= ADD_HOURS(CURRENT_TIMESTAMP, -?)
    ;
    
    _ = hours;
    const result = try client.query(sql, allocator);
    defer allocator.free(result);
    
    // TODO: Parse result into RoutingStats
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

pub fn getModelPerformance(client: *HanaClient, model_id: []const u8, allocator: Allocator) !ModelPerformance {
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
    
    _ = model_id;
    const result = try client.query(sql, allocator);
    defer allocator.free(result);
    
    // TODO: Parse result into ModelPerformance
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

pub fn getTopAgentModelPairs(client: *HanaClient, limit: u32, allocator: Allocator) ![]AgentModelPair {
    const sql = 
        \\SELECT * FROM V_TOP_AGENT_MODEL_PAIRS
        \\LIMIT ?
    ;
    
    _ = limit;
    const result = try client.query(sql, allocator);
    defer allocator.free(result);
    
    // TODO: Parse result into AgentModelPair array
    return try allocator.alloc(AgentModelPair, 0);
}

pub const AnalyticsSummary = struct {
    total_decisions: u64,
    successful_rate: f32,
    avg_latency_ms: f32,
    fallback_rate: f32,
    top_task_type: []const u8,
    top_strategy: []const u8,
};

pub fn getRoutingAnalytics24H(client: *HanaClient, allocator: Allocator) !AnalyticsSummary {
    const sql = 
        \\SELECT * FROM V_ROUTING_ANALYTICS_24H
    ;
    
    const result = try client.query(sql, allocator);
    defer allocator.free(result);
    
    // TODO: Parse result into AnalyticsSummary
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

pub fn saveMetrics(client: *HanaClient, metrics: InferenceMetrics) !void {
    const sql = 
        \\INSERT INTO INFERENCE_METRICS (
        \\  METRIC_ID, MODEL_ID, LATENCY_MS, TTFT_MS,
        \\  TOKENS_GENERATED, CACHE_HIT, TIMESTAMP
        \\) VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    ;
    
    // TODO: Use prepared statement with parameters
    _ = metrics;
    
    try client.execute(sql);
}

pub fn saveMetricsBatch(client: *HanaClient, metrics_list: []const InferenceMetrics) !void {
    // Batch insert for better performance
    if (metrics_list.len == 0) return;
    
    // TODO: Use batch INSERT statement
    for (metrics_list) |metrics| {
        try saveMetrics(client, metrics);
    }
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
