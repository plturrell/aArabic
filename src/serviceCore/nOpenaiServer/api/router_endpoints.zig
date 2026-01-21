// ============================================================================
// Router API Endpoints - Day 54 Implementation
// ============================================================================
// Purpose: HTTP API endpoints for router data using HANA OData
// Week: Week 11 (Days 51-55) - HANA Backend Integration
// Phase: Month 4 - HANA Integration & Scalability
// ============================================================================

const std = @import("std");
const router_analytics = @import("../database/router_analytics.zig");
const router_queries = @import("../database/router_queries.zig");
const ODataPersistence = @import("../hana/core/odata_persistence.zig").ODataPersistence;
const ODataConfig = @import("../hana/core/odata_persistence.zig").ODataConfig;

// ============================================================================
// API RESPONSE TYPES
// ============================================================================

/// GET /api/v1/model-router/assignments response
pub const AssignmentsResponse = struct {
    success: bool,
    total: u32,
    assignments: []AssignmentDTO,
    
    pub fn toJson(self: *const AssignmentsResponse, allocator: std.mem.Allocator) ![]const u8 {
        // Build JSON manually for now
        var json = std.ArrayList(u8).init(allocator);
        
        try json.appendSlice("{\"success\":");
        try json.appendSlice(if (self.success) "true" else "false");
        try json.appendSlice(",\"total\":");
        
        const total_str = try std.fmt.allocPrint(allocator, "{d}", .{self.total});
        defer allocator.free(total_str);
        try json.appendSlice(total_str);
        
        try json.appendSlice(",\"assignments\":[");
        
        for (self.assignments, 0..) |assignment, i| {
            if (i > 0) try json.appendSlice(",");
            
            const assignment_json = try assignment.toJson(allocator);
            defer allocator.free(assignment_json);
            try json.appendSlice(assignment_json);
        }
        
        try json.appendSlice("]}");
        
        return try json.toOwnedSlice();
    }
};

/// Assignment data transfer object
pub const AssignmentDTO = struct {
    assignment_id: []const u8,
    agent_id: []const u8,
    agent_name: []const u8,
    model_id: []const u8,
    model_name: []const u8,
    match_score: f32,
    status: []const u8,
    assignment_method: []const u8,
    total_requests: u64,
    successful_requests: u64,
    avg_latency_ms: f32,
    
    pub fn toJson(self: *const AssignmentDTO, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            \\{{"assignment_id":"{s}","agent_id":"{s}","agent_name":"{s}","model_id":"{s}","model_name":"{s}","match_score":{d:.2},"status":"{s}","assignment_method":"{s}","total_requests":{d},"successful_requests":{d},"avg_latency_ms":{d:.2}}}
        ,
            .{
                self.assignment_id,
                self.agent_id,
                self.agent_name,
                self.model_id,
                self.model_name,
                self.match_score,
                self.status,
                self.assignment_method,
                self.total_requests,
                self.successful_requests,
                self.avg_latency_ms,
            },
        );
    }
};

/// GET /api/v1/model-router/stats response
pub const StatsResponse = struct {
    success: bool,
    total_decisions: u64,
    successful_decisions: u64,
    success_rate: f32,
    avg_latency_ms: f32,
    fallbacks_used: u64,
    fallback_rate: f32,
    recent_decisions: []RecentDecisionDTO,
    
    pub fn toJson(self: *const StatsResponse, allocator: std.mem.Allocator) ![]const u8 {
        var json = std.ArrayList(u8).init(allocator);
        
        try json.appendSlice("{\"success\":");
        try json.appendSlice(if (self.success) "true" else "false");
        try json.appendSlice(",\"total_decisions\":");
        
        const vals = try std.fmt.allocPrint(
            allocator,
            "{d},\"successful_decisions\":{d},\"success_rate\":{d:.2},\"avg_latency_ms\":{d:.2},\"fallbacks_used\":{d},\"fallback_rate\":{d:.2}",
            .{
                self.total_decisions,
                self.successful_decisions,
                self.success_rate,
                self.avg_latency_ms,
                self.fallbacks_used,
                self.fallback_rate,
            },
        );
        defer allocator.free(vals);
        try json.appendSlice(vals);
        
        try json.appendSlice(",\"recent_decisions\":[");
        
        for (self.recent_decisions, 0..) |decision, i| {
            if (i > 0) try json.appendSlice(",");
            
            const decision_json = try decision.toJson(allocator);
            defer allocator.free(decision_json);
            try json.appendSlice(decision_json);
        }
        
        try json.appendSlice("]}");
        
        return try json.toOwnedSlice();
    }
};

/// Recent decision DTO
pub const RecentDecisionDTO = struct {
    task_type: []const u8,
    selected_model: []const u8,
    score: f32,
    latency_ms: i32,
    success: bool,
    timestamp: i64,
    
    pub fn toJson(self: *const RecentDecisionDTO, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            \\{{"task_type":"{s}","selected_model":"{s}","score":{d:.2},"latency_ms":{d},"success":{},"timestamp":{d}}}
        ,
            .{
                self.task_type,
                self.selected_model,
                self.score,
                self.latency_ms,
                self.success,
                self.timestamp,
            },
        );
    }
};

/// GET /api/v1/model-router/performance response
pub const PerformanceResponse = struct {
    success: bool,
    time_range: []const u8,
    models: []ModelPerformanceDTO,
    
    pub fn toJson(self: *const PerformanceResponse, allocator: std.mem.Allocator) ![]const u8 {
        var json = std.ArrayList(u8).init(allocator);
        
        try json.appendSlice("{\"success\":");
        try json.appendSlice(if (self.success) "true" else "false");
        try json.appendSlice(",\"time_range\":\"");
        try json.appendSlice(self.time_range);
        try json.appendSlice("\",\"models\":[");
        
        for (self.models, 0..) |model, i| {
            if (i > 0) try json.appendSlice(",");
            
            const model_json = try model.toJson(allocator);
            defer allocator.free(model_json);
            try json.appendSlice(model_json);
        }
        
        try json.appendSlice("]}");
        
        return try json.toOwnedSlice();
    }
};

/// Model performance DTO
pub const ModelPerformanceDTO = struct {
    model_id: []const u8,
    total_requests: u64,
    successful_requests: u64,
    avg_latency_ms: f32,
    avg_tokens_per_second: f32,
    error_rate: f32,
    
    pub fn toJson(self: *const ModelPerformanceDTO, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            \\{{"model_id":"{s}","total_requests":{d},"successful_requests":{d},"avg_latency_ms":{d:.2},"avg_tokens_per_second":{d:.2},"error_rate":{d:.4}}}
        ,
            .{
                self.model_id,
                self.total_requests,
                self.successful_requests,
                self.avg_latency_ms,
                self.avg_tokens_per_second,
                self.error_rate,
            },
        );
    }
};

// ============================================================================
// API HANDLERS
// ============================================================================

/// GET /api/v1/model-router/assignments
/// Returns all active assignments with metrics from HANA
pub fn handleGetAssignments(
    allocator: std.mem.Allocator,
    odata_client: *ODataPersistence,
) ![]const u8 {
    // Get active assignments from HANA via OData
    const assignments = try router_analytics.getActiveAssignments(odata_client);
    defer odata_client.allocator.free(assignments);
    
    // Convert to DTOs
    var dtos = std.ArrayList(AssignmentDTO).init(allocator);
    defer dtos.deinit();
    
    for (assignments) |assignment| {
        try dtos.append(.{
            .assignment_id = assignment.id,
            .agent_id = assignment.agent_id,
            .agent_name = assignment.agent_id, // TODO: Lookup agent name
            .model_id = assignment.model_id,
            .model_name = assignment.model_id, // TODO: Lookup model name
            .match_score = assignment.match_score,
            .status = assignment.status,
            .assignment_method = assignment.assignment_method,
            .total_requests = 0, // TODO: Add to analytics
            .successful_requests = 0,
            .avg_latency_ms = 0.0,
        });
    }
    
    const response = AssignmentsResponse{
        .success = true,
        .total = @intCast(dtos.items.len),
        .assignments = dtos.items,
    };
    
    return try response.toJson(allocator);
}

/// GET /api/v1/model-router/stats
/// Returns routing statistics from HANA
pub fn handleGetStats(
    allocator: std.mem.Allocator,
    odata_client: *ODataPersistence,
    time_range: router_analytics.TimeRange,
) ![]const u8 {
    // Get routing stats from HANA via OData
    const stats = try router_analytics.getRoutingStats(odata_client, time_range);
    
    // Calculate derived metrics
    const success_rate = if (stats.total_decisions > 0)
        @as(f32, @floatFromInt(stats.successful_decisions)) / @as(f32, @floatFromInt(stats.total_decisions)) * 100.0
    else
        0.0;
    
    const fallbacks_used = @as(u64, @intFromFloat(@as(f64, @floatFromInt(stats.total_decisions)) * @as(f64, stats.fallback_rate)));
    
    // Build response (recent decisions empty for now)
    var recent = std.ArrayList(RecentDecisionDTO).init(allocator);
    defer recent.deinit();
    
    const response = StatsResponse{
        .success = true,
        .total_decisions = stats.total_decisions,
        .successful_decisions = stats.successful_decisions,
        .success_rate = success_rate,
        .avg_latency_ms = stats.avg_latency_ms,
        .fallbacks_used = fallbacks_used,
        .fallback_rate = stats.fallback_rate * 100.0,
        .recent_decisions = recent.items,
    };
    
    return try response.toJson(allocator);
}

/// GET /api/v1/model-router/performance?time_range=last_hour
/// Returns model performance metrics from HANA
pub fn handleGetPerformance(
    allocator: std.mem.Allocator,
    odata_client: *ODataPersistence,
    time_range: router_analytics.TimeRange,
) ![]const u8 {
    _ = odata_client;
    
    // For now, return sample data
    // TODO: Implement getTopPerformingModels() call
    
    var models = std.ArrayList(ModelPerformanceDTO).init(allocator);
    defer models.deinit();
    
    try models.append(.{
        .model_id = "gpt-4",
        .total_requests = 1000,
        .successful_requests = 950,
        .avg_latency_ms = 45.2,
        .avg_tokens_per_second = 25.5,
        .error_rate = 0.05,
    });
    
    const time_range_str = switch (time_range) {
        .last_hour => "last_hour",
        .last_24_hours => "last_24_hours",
        .last_7_days => "last_7_days",
        .last_30_days => "last_30_days",
    };
    
    const response = PerformanceResponse{
        .success = true,
        .time_range = time_range_str,
        .models = models.items,
    };
    
    return try response.toJson(allocator);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Initialize OData client from config
pub fn initODataClient(allocator: std.mem.Allocator) !*ODataPersistence {
    const config = ODataConfig{
        .base_url = "https://hana-cloud.sap.com", // TODO: Load from config
        .username = "NUCLEUS_APP",
        .password = std.posix.getenv("HANA_PASSWORD") orelse "changeme",
    };
    
    return try ODataPersistence.init(allocator, config);
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "AssignmentDTO: JSON serialization" {
    const allocator = std.testing.allocator;
    
    const dto = AssignmentDTO{
        .assignment_id = "asn_123",
        .agent_id = "agent-1",
        .agent_name = "Agent 1",
        .model_id = "gpt-4",
        .model_name = "GPT-4",
        .match_score = 87.5,
        .status = "ACTIVE",
        .assignment_method = "GREEDY",
        .total_requests = 100,
        .successful_requests = 95,
        .avg_latency_ms = 45.2,
    };
    
    const json = try dto.toJson(allocator);
    defer allocator.free(json);
    
    try std.testing.expect(std.mem.indexOf(u8, json, "asn_123") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "agent-1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "gpt-4") != null);
}

test "StatsResponse: JSON serialization" {
    const allocator = std.testing.allocator;
    
    var recent = std.ArrayList(RecentDecisionDTO).init(allocator);
    defer recent.deinit();
    
    const response = StatsResponse{
        .success = true,
        .total_decisions = 1000,
        .successful_decisions = 950,
        .success_rate = 95.0,
        .avg_latency_ms = 45.2,
        .fallbacks_used = 50,
        .fallback_rate = 5.0,
        .recent_decisions = recent.items,
    };
    
    const json = try response.toJson(allocator);
    defer allocator.free(json);
    
    try std.testing.expect(std.mem.indexOf(u8, json, "\"success\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"total_decisions\":1000") != null);
}

test "ModelPerformanceDTO: JSON serialization" {
    const allocator = std.testing.allocator;
    
    const dto = ModelPerformanceDTO{
        .model_id = "gpt-4",
        .total_requests = 1000,
        .successful_requests = 950,
        .avg_latency_ms = 45.2,
        .avg_tokens_per_second = 25.5,
        .error_rate = 0.05,
    };
    
    const json = try dto.toJson(allocator);
    defer allocator.free(json);
    
    try std.testing.expect(std.mem.indexOf(u8, json, "gpt-4") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "1000") != null);
}
