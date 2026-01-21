// ============================================================================
// Router API - Day 24 Implementation
// ============================================================================
// Purpose: REST API endpoints for model router functionality
// Week: Week 5 (Days 21-25) - Model Router Foundation
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");
const auto_assign = @import("auto_assign.zig");
const capability_scorer = @import("capability_scorer.zig");

// Type aliases
const AgentRegistry = auto_assign.AgentRegistry;
const ModelRegistry = auto_assign.ModelRegistry;
const AutoAssigner = auto_assign.AutoAssigner;
const AssignmentDecision = auto_assign.AssignmentDecision;
const AssignmentStrategy = auto_assign.AssignmentStrategy;

// ============================================================================
// API REQUEST/RESPONSE TYPES
// ============================================================================

/// Request body for auto-assign-all endpoint
pub const AutoAssignRequest = struct {
    strategy: []const u8, // "greedy", "optimal", "balanced"
    
    pub fn parseStrategy(self: *const AutoAssignRequest) !AssignmentStrategy {
        if (std.mem.eql(u8, self.strategy, "greedy")) return .greedy;
        if (std.mem.eql(u8, self.strategy, "optimal")) return .optimal;
        if (std.mem.eql(u8, self.strategy, "balanced")) return .balanced;
        return error.InvalidStrategy;
    }
};

/// Response for auto-assign-all endpoint
pub const AutoAssignResponse = struct {
    success: bool,
    assignments: []AssignmentDecision,
    strategy: []const u8,
    total_assignments: usize,
    avg_match_score: f32,
    timestamp: []const u8,
};

/// Query parameters for get assignments endpoint
pub const GetAssignmentsQuery = struct {
    status: ?[]const u8, // "ACTIVE", "INACTIVE", null = all
    agent_id: ?[]const u8,
    model_id: ?[]const u8,
    page: u32,
    page_size: u32,
    
    pub fn init() GetAssignmentsQuery {
        return .{
            .status = null,
            .agent_id = null,
            .model_id = null,
            .page = 1,
            .page_size = 50,
        };
    }
};

/// Single assignment record
pub const AssignmentRecord = struct {
    assignment_id: []const u8,
    agent_id: []const u8,
    agent_name: []const u8,
    model_id: []const u8,
    model_name: []const u8,
    match_score: f32,
    status: []const u8,
    assignment_method: []const u8,
    assigned_by: []const u8,
    assigned_at: []const u8,
    last_updated: []const u8,
    total_requests: u32,
    successful_requests: u32,
    avg_latency_ms: ?f32,
};

/// Response for get assignments endpoint
pub const GetAssignmentsResponse = struct {
    success: bool,
    assignments: []AssignmentRecord,
    total_count: u32,
    page: u32,
    page_size: u32,
    total_pages: u32,
};

/// Request body for update assignment endpoint
pub const UpdateAssignmentRequest = struct {
    model_id: []const u8,
    status: ?[]const u8, // "ACTIVE", "INACTIVE"
    notes: ?[]const u8,
};

/// Response for update assignment endpoint
pub const UpdateAssignmentResponse = struct {
    success: bool,
    assignment: AssignmentRecord,
    message: []const u8,
};

// ============================================================================
// ROUTER API HANDLER
// ============================================================================

pub const RouterApiHandler = struct {
    allocator: std.mem.Allocator,
    agent_registry: *AgentRegistry,
    model_registry: *ModelRegistry,
    
    pub fn init(
        allocator: std.mem.Allocator,
        agent_registry: *AgentRegistry,
        model_registry: *ModelRegistry,
    ) RouterApiHandler {
        return .{
            .allocator = allocator,
            .agent_registry = agent_registry,
            .model_registry = model_registry,
        };
    }
    
    /// POST /api/v1/model-router/auto-assign-all
    pub fn handleAutoAssignAll(
        self: *RouterApiHandler,
        request: AutoAssignRequest,
    ) !AutoAssignResponse {
        const strategy = try request.parseStrategy();
        
        // Create auto-assigner
        var assigner = AutoAssigner.init(
            self.allocator,
            self.agent_registry,
            self.model_registry,
        );
        
        // Perform assignment
        var decisions = try assigner.assignAll(strategy);
        defer decisions.deinit();
        
        // Calculate average match score
        var total_score: f32 = 0.0;
        for (decisions.items) |decision| {
            total_score += decision.match_score;
        }
        const avg_score = if (decisions.items.len > 0)
            total_score / @as(f32, @floatFromInt(decisions.items.len))
        else
            0.0;
        
        // TODO: Store assignments in HANA database
        // This would be implemented with actual database connection
        // For now, we return the decisions
        
        // Get current timestamp
        const timestamp = try self.getCurrentTimestamp();
        
        return AutoAssignResponse{
            .success = true,
            .assignments = decisions.items,
            .strategy = request.strategy,
            .total_assignments = decisions.items.len,
            .avg_match_score = avg_score,
            .timestamp = timestamp,
        };
    }
    
    /// GET /api/v1/model-router/assignments
    pub fn handleGetAssignments(
        self: *RouterApiHandler,
        query: GetAssignmentsQuery,
    ) !GetAssignmentsResponse {
        _ = self;
        
        // TODO: Query assignments from HANA database
        // For now, return empty list
        
        const total_count: u32 = 0;
        const total_pages: u32 = if (query.page_size > 0)
            (total_count + query.page_size - 1) / query.page_size
        else
            0;
        
        return GetAssignmentsResponse{
            .success = true,
            .assignments = &[_]AssignmentRecord{},
            .total_count = total_count,
            .page = query.page,
            .page_size = query.page_size,
            .total_pages = total_pages,
        };
    }
    
    /// PUT /api/v1/model-router/assignments/:id
    pub fn handleUpdateAssignment(
        self: *RouterApiHandler,
        assignment_id: []const u8,
        request: UpdateAssignmentRequest,
    ) !UpdateAssignmentResponse {
        _ = assignment_id;
        _ = request;
        
        // TODO: Update assignment in HANA database
        // For now, return placeholder response
        
        const timestamp = try self.getCurrentTimestamp();
        
        return UpdateAssignmentResponse{
            .success = false,
            .assignment = AssignmentRecord{
                .assignment_id = "",
                .agent_id = "",
                .agent_name = "",
                .model_id = "",
                .model_name = "",
                .match_score = 0.0,
                .status = "INACTIVE",
                .assignment_method = "manual",
                .assigned_by = "system",
                .assigned_at = timestamp,
                .last_updated = timestamp,
                .total_requests = 0,
                .successful_requests = 0,
                .avg_latency_ms = null,
            },
            .message = "Database integration pending - Day 24",
        };
    }
    
    /// DELETE /api/v1/model-router/assignments/:id
    pub fn handleDeleteAssignment(
        self: *RouterApiHandler,
        assignment_id: []const u8,
    ) !struct { success: bool, message: []const u8 } {
        _ = self;
        _ = assignment_id;
        
        // TODO: Delete assignment from HANA database
        
        return .{
            .success = false,
            .message = "Database integration pending - Day 24",
        };
    }
    
    /// GET /api/v1/model-router/stats
    pub fn handleGetStats(self: *RouterApiHandler) !struct {
        total_agents: u32,
        online_agents: u32,
        total_models: u32,
        total_assignments: u32,
        avg_match_score: f32,
    } {
        const total_agents = @as(u32, @intCast(self.agent_registry.agents.items.len));
        
        var online_count: u32 = 0;
        for (self.agent_registry.agents.items) |agent| {
            if (agent.status == .online) {
                online_count += 1;
            }
        }
        
        const total_models = @as(u32, @intCast(self.model_registry.models.items.len));
        
        return .{
            .total_agents = total_agents,
            .online_agents = online_count,
            .total_models = total_models,
            .total_assignments = 0, // TODO: Query from database
            .avg_match_score = 0.0, // TODO: Query from database
        };
    }
    
    // Helper function to get current timestamp
    fn getCurrentTimestamp(self: *RouterApiHandler) ![]const u8 {
        _ = self;
        // Placeholder - would use actual timestamp in production
        return "2026-01-21T19:40:00Z";
    }
};

// ============================================================================
// JSON SERIALIZATION HELPERS
// ============================================================================

/// Serialize AutoAssignResponse to JSON
pub fn serializeAutoAssignResponse(
    allocator: std.mem.Allocator,
    response: AutoAssignResponse,
) ![]const u8 {
    var json = std.ArrayList(u8).init(allocator);
    const writer = json.writer();
    
    try writer.writeAll("{");
    try writer.print("\"success\":{},", .{response.success});
    try writer.print("\"strategy\":\"{s}\",", .{response.strategy});
    try writer.print("\"total_assignments\":{},", .{response.total_assignments});
    try writer.print("\"avg_match_score\":{d:.2},", .{response.avg_match_score});
    try writer.print("\"timestamp\":\"{s}\",", .{response.timestamp});
    
    try writer.writeAll("\"assignments\":[");
    for (response.assignments, 0..) |decision, i| {
        if (i > 0) try writer.writeAll(",");
        try writer.writeAll("{");
        try writer.print("\"agent_id\":\"{s}\",", .{decision.agent_id});
        try writer.print("\"agent_name\":\"{s}\",", .{decision.agent_name});
        try writer.print("\"model_id\":\"{s}\",", .{decision.model_id});
        try writer.print("\"model_name\":\"{s}\",", .{decision.model_name});
        try writer.print("\"match_score\":{d:.2},", .{decision.match_score});
        try writer.print("\"assignment_method\":\"{s}\"", .{decision.assignment_method.toString()});
        try writer.writeAll("}");
    }
    try writer.writeAll("]");
    
    try writer.writeAll("}");
    
    return json.toOwnedSlice();
}

/// Serialize GetAssignmentsResponse to JSON
pub fn serializeGetAssignmentsResponse(
    allocator: std.mem.Allocator,
    response: GetAssignmentsResponse,
) ![]const u8 {
    var json = std.ArrayList(u8).init(allocator);
    const writer = json.writer();
    
    try writer.writeAll("{");
    try writer.print("\"success\":{},", .{response.success});
    try writer.print("\"total_count\":{},", .{response.total_count});
    try writer.print("\"page\":{},", .{response.page});
    try writer.print("\"page_size\":{},", .{response.page_size});
    try writer.print("\"total_pages\":{},", .{response.total_pages});
    try writer.writeAll("\"assignments\":[]");
    try writer.writeAll("}");
    
    return json.toOwnedSlice();
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "RouterApiHandler: auto-assign-all" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try auto_assign.createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try auto_assign.createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var handler = RouterApiHandler.init(allocator, &agent_registry, &model_registry);
    
    const request = AutoAssignRequest{
        .strategy = "greedy",
    };
    
    const response = try handler.handleAutoAssignAll(request);
    
    try std.testing.expect(response.success);
    try std.testing.expectEqual(@as(usize, 3), response.total_assignments);
    try std.testing.expect(response.avg_match_score > 0.0);
}

test "RouterApiHandler: get assignments (empty)" {
    const allocator = std.testing.allocator;
    
    var agent_registry = AgentRegistry.init(allocator);
    defer agent_registry.deinit();
    
    var model_registry = ModelRegistry.init(allocator);
    defer model_registry.deinit();
    
    var handler = RouterApiHandler.init(allocator, &agent_registry, &model_registry);
    
    const query = GetAssignmentsQuery.init();
    const response = try handler.handleGetAssignments(query);
    
    try std.testing.expect(response.success);
    try std.testing.expectEqual(@as(u32, 0), response.total_count);
    try std.testing.expectEqual(@as(usize, 0), response.assignments.len);
}

test "RouterApiHandler: get stats" {
    const allocator = std.testing.allocator;
    
    var agent_registry = try auto_assign.createSampleAgentRegistry(allocator);
    defer agent_registry.deinit();
    
    var model_registry = try auto_assign.createSampleModelRegistry(allocator);
    defer model_registry.deinit();
    
    var handler = RouterApiHandler.init(allocator, &agent_registry, &model_registry);
    
    const stats = try handler.handleGetStats();
    
    try std.testing.expectEqual(@as(u32, 3), stats.total_agents);
    try std.testing.expectEqual(@as(u32, 3), stats.online_agents);
    try std.testing.expectEqual(@as(u32, 3), stats.total_models);
}

test "JSON serialization: AutoAssignResponse" {
    const allocator = std.testing.allocator;
    
    const response = AutoAssignResponse{
        .success = true,
        .assignments = &[_]AssignmentDecision{},
        .strategy = "greedy",
        .total_assignments = 0,
        .avg_match_score = 85.5,
        .timestamp = "2026-01-21T00:00:00Z",
    };
    
    const json = try serializeAutoAssignResponse(allocator, response);
    defer allocator.free(json);
    
    try std.testing.expect(std.mem.indexOf(u8, json, "\"success\":true") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"strategy\":\"greedy\"") != null);
}
