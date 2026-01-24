// ============================================================================
// Router Queries - Day 52 Implementation
// ============================================================================
// Purpose: HANA database queries for model router persistence using OData
// Week: Week 11 (Days 51-55) - HANA Backend Integration
// Phase: Month 4 - HANA Integration & Scalability
// ============================================================================

const std = @import("std");
const ODataPersistence = @import("../../../../nLang/n-c-sdk/lib/hana/odata_persistence.zig").ODataPersistence;
const ODataConfig = @import("../../../../nLang/n-c-sdk/lib/hana/odata_persistence.zig").ODataConfig;
const AssignmentEntity = @import("../../../../nLang/n-c-sdk/lib/hana/odata_persistence.zig").AssignmentEntity;
const RoutingDecisionEntity = @import("../../../../nLang/n-c-sdk/lib/hana/odata_persistence.zig").RoutingDecisionEntity;
const MetricsEntity = @import("../../../../nLang/n-c-sdk/lib/hana/odata_persistence.zig").MetricsEntity;

// ============================================================================
// DATABASE TYPES
// ============================================================================

/// Assignment record for AGENT_MODEL_ASSIGNMENTS table
pub const Assignment = struct {
    id: []const u8,
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32,
    status: []const u8, // 'ACTIVE', 'INACTIVE'
    assignment_method: []const u8, // 'GREEDY', 'BALANCED', 'OPTIMAL', 'MANUAL'
    capabilities_json: []const u8,
    assigned_by: []const u8,
    created_at: i64,
    updated_at: i64,
    
    pub fn init(
        allocator: std.mem.Allocator,
        agent_id: []const u8,
        model_id: []const u8,
        match_score: f32,
        assignment_method: []const u8,
    ) !Assignment {
        const id = try generateAssignmentId(allocator);
        const timestamp = std.time.milliTimestamp();
        
        return .{
            .id = id,
            .agent_id = try allocator.dupe(u8, agent_id),
            .model_id = try allocator.dupe(u8, model_id),
            .match_score = match_score,
            .status = try allocator.dupe(u8, "ACTIVE"),
            .assignment_method = try allocator.dupe(u8, assignment_method),
            .capabilities_json = try allocator.dupe(u8, "{}"),
            .assigned_by = try allocator.dupe(u8, "system"),
            .created_at = timestamp,
            .updated_at = timestamp,
        };
    }
    
    pub fn deinit(self: *Assignment, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.agent_id);
        allocator.free(self.model_id);
        allocator.free(self.status);
        allocator.free(self.assignment_method);
        allocator.free(self.capabilities_json);
        allocator.free(self.assigned_by);
    }
};

/// Routing decision record for ROUTING_DECISIONS table
pub const RoutingDecision = struct {
    id: []const u8,
    request_id: []const u8,
    agent_id: []const u8,
    model_id: []const u8,
    assignment_id: []const u8,
    strategy: []const u8,
    match_score: f32,
    decision_time_ms: f32,
    created_at: i64,
    
    pub fn init(
        allocator: std.mem.Allocator,
        request_id: []const u8,
        agent_id: []const u8,
        model_id: []const u8,
        assignment_id: []const u8,
        strategy: []const u8,
        match_score: f32,
        decision_time_ms: f32,
    ) !RoutingDecision {
        const id = try generateRoutingId(allocator);
        
        return .{
            .id = id,
            .request_id = try allocator.dupe(u8, request_id),
            .agent_id = try allocator.dupe(u8, agent_id),
            .model_id = try allocator.dupe(u8, model_id),
            .assignment_id = try allocator.dupe(u8, assignment_id),
            .strategy = try allocator.dupe(u8, strategy),
            .match_score = match_score,
            .decision_time_ms = decision_time_ms,
            .created_at = std.time.milliTimestamp(),
        };
    }
    
    pub fn deinit(self: *RoutingDecision, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.request_id);
        allocator.free(self.agent_id);
        allocator.free(self.model_id);
        allocator.free(self.assignment_id);
        allocator.free(self.strategy);
    }
};

/// Inference metrics record for INFERENCE_METRICS table
pub const InferenceMetrics = struct {
    id: []const u8,
    request_id: []const u8,
    model_id: []const u8,
    agent_id: []const u8,
    latency_ms: f32,
    tokens_processed: u32,
    success: bool,
    error_message: ?[]const u8,
    created_at: i64,
    
    pub fn init(
        allocator: std.mem.Allocator,
        request_id: []const u8,
        model_id: []const u8,
        agent_id: []const u8,
        latency_ms: f32,
        tokens_processed: u32,
        success: bool,
    ) !InferenceMetrics {
        const id = try generateMetricsId(allocator);
        
        return .{
            .id = id,
            .request_id = try allocator.dupe(u8, request_id),
            .model_id = try allocator.dupe(u8, model_id),
            .agent_id = try allocator.dupe(u8, agent_id),
            .latency_ms = latency_ms,
            .tokens_processed = tokens_processed,
            .success = success,
            .error_message = null,
            .created_at = std.time.milliTimestamp(),
        };
    }
    
    pub fn deinit(self: *InferenceMetrics, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.request_id);
        allocator.free(self.model_id);
        allocator.free(self.agent_id);
        if (self.error_message) |msg| {
            allocator.free(msg);
        }
    }
};

// ============================================================================
// ID GENERATION
// ============================================================================

/// Generate unique assignment ID
pub fn generateAssignmentId(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    var random_bytes: [8]u8 = undefined;
    std.crypto.random.bytes(&random_bytes);
    
    return std.fmt.allocPrint(
        allocator,
        "asn_{d}_{x}",
        .{ timestamp, std.fmt.fmtSliceHexLower(&random_bytes) },
    );
}

/// Generate unique routing decision ID
pub fn generateRoutingId(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    var random_bytes: [8]u8 = undefined;
    std.crypto.random.bytes(&random_bytes);
    
    return std.fmt.allocPrint(
        allocator,
        "rtd_{d}_{x}",
        .{ timestamp, std.fmt.fmtSliceHexLower(&random_bytes) },
    );
}

/// Generate unique metrics ID
pub fn generateMetricsId(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    var random_bytes: [8]u8 = undefined;
    std.crypto.random.bytes(&random_bytes);
    
    return std.fmt.allocPrint(
        allocator,
        "met_{d}_{x}",
        .{ timestamp, std.fmt.fmtSliceHexLower(&random_bytes) },
    );
}

// ============================================================================
// DATABASE OPERATIONS (Real OData via HANA)
// ============================================================================

/// Save assignment to AGENT_MODEL_ASSIGNMENTS table via OData
pub fn saveAssignment(
    client: *ODataPersistence,
    assignment: *const Assignment,
) !void {
    // Validate data
    if (assignment.agent_id.len == 0) return error.InvalidAgentId;
    if (assignment.model_id.len == 0) return error.InvalidModelId;
    if (assignment.match_score < 0.0 or assignment.match_score > 100.0) {
        return error.InvalidMatchScore;
    }
    
    // Convert to OData entity
    const entity = AssignmentEntity{
        .id = assignment.id,
        .agent_id = assignment.agent_id,
        .model_id = assignment.model_id,
        .match_score = assignment.match_score,
        .status = assignment.status,
        .assignment_method = assignment.assignment_method,
        .capabilities_json = assignment.capabilities_json,
    };
    
    // Execute real OData POST
    try client.createAssignment(entity);
    
    std.log.debug("Saved assignment via OData: {s} -> {s} (score: {d:.2})", .{
        assignment.agent_id,
        assignment.model_id,
        assignment.match_score,
    });
}

/// Save routing decision to ROUTING_DECISIONS table via OData
pub fn saveRoutingDecision(
    client: *ODataPersistence,
    decision: *const RoutingDecision,
) !void {
    // Validate data
    if (decision.request_id.len == 0) return error.InvalidRequestId;
    if (decision.agent_id.len == 0) return error.InvalidAgentId;
    if (decision.model_id.len == 0) return error.InvalidModelId;
    
    // Convert to OData entity (mapping fields)
    const entity = RoutingDecisionEntity{
        .id = decision.id,
        .request_id = decision.request_id,
        .task_type = decision.strategy, // Map strategy to task_type
        .agent_id = decision.agent_id,
        .model_id = decision.model_id,
        .capability_score = decision.match_score,
        .performance_score = decision.match_score,
        .composite_score = decision.match_score,
        .strategy_used = decision.strategy,
        .latency_ms = @intFromFloat(decision.decision_time_ms),
        .success = true,
        .fallback_used = false,
    };
    
    // Execute real OData POST
    try client.createRoutingDecision(entity);
    
    std.log.debug("Saved routing decision via OData: request={s}, agent={s}, model={s}", .{
        decision.request_id,
        decision.agent_id,
        decision.model_id,
    });
}

/// Save inference metrics to INFERENCE_METRICS table via OData
pub fn saveMetrics(
    client: *ODataPersistence,
    metrics: *const InferenceMetrics,
) !void {
    // Validate data
    if (metrics.request_id.len == 0) return error.InvalidRequestId;
    if (metrics.latency_ms < 0.0) return error.InvalidLatency;
    
    // Convert to OData entity
    const entity = MetricsEntity{
        .id = metrics.id,
        .model_id = metrics.model_id,
        .latency_ms = @intFromFloat(metrics.latency_ms),
        .ttft_ms = 0, // Not tracked yet
        .tokens_generated = metrics.tokens_processed,
        .cache_hit = false, // Not tracked yet
    };
    
    // Execute real OData POST
    const entities = [_]MetricsEntity{entity};
    try client.createMetricsBatch(&entities);
    
    std.log.debug("Saved metrics via OData: request={s}, latency={d:.2}ms, success={}", .{
        metrics.request_id,
        metrics.latency_ms,
        metrics.success,
    });
}

/// Batch save multiple assignments via OData
pub fn saveAssignmentBatch(
    client: *ODataPersistence,
    assignments: []const Assignment,
) !void {
    // Save each assignment (OData batch support would be better)
    for (assignments) |*assignment| {
        try saveAssignment(client, assignment);
    }
    
    std.log.debug("Saved {d} assignments via OData batch", .{assignments.len});
}

/// Get active assignments count via OData
pub fn getActiveAssignmentsCount(client: *ODataPersistence) !u32 {
    const assignments = try client.getActiveAssignments();
    defer client.allocator.free(assignments);
    
    return @intCast(assignments.len);
}

// ============================================================================
// QUERY HELPERS
// ============================================================================

/// Build SQL for inserting assignment (for reference)
pub fn buildInsertAssignmentSQL(allocator: std.mem.Allocator) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\INSERT INTO AGENT_MODEL_ASSIGNMENTS (
        \\  ID, AGENT_ID, MODEL_ID, MATCH_SCORE, STATUS,
        \\  ASSIGNMENT_METHOD, CAPABILITIES_JSON, ASSIGNED_BY,
        \\  CREATED_AT, UPDATED_AT
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ,
        .{},
    );
}

/// Build SQL for inserting routing decision (for reference)
pub fn buildInsertRoutingDecisionSQL(allocator: std.mem.Allocator) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\INSERT INTO ROUTING_DECISIONS (
        \\  ID, REQUEST_ID, AGENT_ID, MODEL_ID, ASSIGNMENT_ID,
        \\  STRATEGY, MATCH_SCORE, DECISION_TIME_MS, CREATED_AT
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ,
        .{},
    );
}

/// Build SQL for inserting metrics (for reference)
pub fn buildInsertMetricsSQL(allocator: std.mem.Allocator) ![]const u8 {
    return try std.fmt.allocPrint(
        allocator,
        \\INSERT INTO INFERENCE_METRICS (
        \\  ID, REQUEST_ID, MODEL_ID, AGENT_ID, LATENCY_MS,
        \\  TOKENS_PROCESSED, SUCCESS, ERROR_MESSAGE, CREATED_AT
        \\) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ,
        .{},
    );
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "Assignment: creation and validation" {
    const allocator = std.testing.allocator;
    
    var assignment = try Assignment.init(
        allocator,
        "agent-1",
        "model-gpt4",
        85.5,
        "GREEDY",
    );
    defer assignment.deinit(allocator);
    
    try std.testing.expect(assignment.id.len > 0);
    try std.testing.expectEqualStrings("agent-1", assignment.agent_id);
    try std.testing.expectEqualStrings("model-gpt4", assignment.model_id);
    try std.testing.expectApproxEqAbs(@as(f32, 85.5), assignment.match_score, 0.01);
    try std.testing.expectEqualStrings("ACTIVE", assignment.status);
}

test "RoutingDecision: creation and validation" {
    const allocator = std.testing.allocator;
    
    var decision = try RoutingDecision.init(
        allocator,
        "req-123",
        "agent-1",
        "model-gpt4",
        "asn-456",
        "GREEDY",
        85.5,
        12.3,
    );
    defer decision.deinit(allocator);
    
    try std.testing.expect(decision.id.len > 0);
    try std.testing.expectEqualStrings("req-123", decision.request_id);
    try std.testing.expectApproxEqAbs(@as(f32, 12.3), decision.decision_time_ms, 0.01);
}

test "InferenceMetrics: creation and validation" {
    const allocator = std.testing.allocator;
    
    var metrics = try InferenceMetrics.init(
        allocator,
        "req-123",
        "model-gpt4",
        "agent-1",
        45.2,
        150,
        true,
    );
    defer metrics.deinit(allocator);
    
    try std.testing.expect(metrics.id.len > 0);
    try std.testing.expectEqual(@as(u32, 150), metrics.tokens_processed);
    try std.testing.expect(metrics.success);
}

test "ID generation: uniqueness" {
    const allocator = std.testing.allocator;
    
    const id1 = try generateAssignmentId(allocator);
    defer allocator.free(id1);
    
    const id2 = try generateAssignmentId(allocator);
    defer allocator.free(id2);
    
    // IDs should be different
    try std.testing.expect(!std.mem.eql(u8, id1, id2));
    
    // IDs should have correct prefix
    try std.testing.expect(std.mem.startsWith(u8, id1, "asn_"));
    try std.testing.expect(std.mem.startsWith(u8, id2, "asn_"));
}


test "SQL builders" {
    const allocator = std.testing.allocator;
    
    const sql1 = try buildInsertAssignmentSQL(allocator);
    defer allocator.free(sql1);
    try std.testing.expect(std.mem.indexOf(u8, sql1, "INSERT INTO") != null);
    
    const sql2 = try buildInsertRoutingDecisionSQL(allocator);
    defer allocator.free(sql2);
    try std.testing.expect(std.mem.indexOf(u8, sql2, "ROUTING_DECISIONS") != null);
    
    const sql3 = try buildInsertMetricsSQL(allocator);
    defer allocator.free(sql3);
    try std.testing.expect(std.mem.indexOf(u8, sql3, "INFERENCE_METRICS") != null);
}
