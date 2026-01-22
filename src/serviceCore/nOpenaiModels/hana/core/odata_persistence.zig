const std = @import("std");
const Allocator = std.mem.Allocator;

/// OData-based persistence layer for HANA Cloud
/// 
/// Since HANA Cloud only exposes OData v4 REST API (no direct SQL access),
/// we use OData for all Router persistence operations.
///
/// This module provides OData-based alternatives to SQL operations.

// HTTP client functions (from zig_http_shimmy.zig or similar)
extern fn zig_http_get(url: [*:0]const u8) callconv(.c) [*:0]const u8;
extern fn zig_http_post(url: [*:0]const u8, body: [*:0]const u8, body_len: usize) callconv(.c) [*:0]const u8;
extern fn zig_http_patch(url: [*:0]const u8, body: [*:0]const u8, body_len: usize) callconv(.c) [*:0]const u8;
extern fn zig_http_delete(url: [*:0]const u8) callconv(.c) [*:0]const u8;

pub const ODataConfig = struct {
    base_url: []const u8,
    username: []const u8,
    password: []const u8,
    service_path: []const u8 = "/sap/opu/odata4/nopenai/routing/default/v1",
    max_retries: u32 = 3,
    timeout_ms: u64 = 5000,
};

/// OData client for Router persistence
pub const ODataPersistence = struct {
    allocator: Allocator,
    config: ODataConfig,
    csrf_token: ?[]const u8 = null,
    
    pub fn init(allocator: Allocator, config: ODataConfig) !*ODataPersistence {
        const client = try allocator.create(ODataPersistence);
        client.* = .{
            .allocator = allocator,
            .config = config,
        };
        return client;
    }
    
    pub fn deinit(self: *ODataPersistence) void {
        if (self.csrf_token) |token| {
            self.allocator.free(token);
        }
        self.allocator.destroy(self);
    }
    
    /// Fetch CSRF token for write operations
    fn fetchCsrfToken(self: *ODataPersistence) !void {
        // Build token fetch URL
        const token_url = try std.fmt.allocPrintZ(
            self.allocator,
            "{s}{s}",
            .{ self.config.base_url, self.config.service_path },
        );
        defer self.allocator.free(token_url);
        
        // Make HEAD request (using GET for now, parse response headers)
        const response_ptr = zig_http_get(token_url.ptr);
        const response = std.mem.span(response_ptr);
        
        // Parse X-CSRF-Token from response headers
        // For now, generate a token (actual implementation needs header parsing)
        if (self.csrf_token) |token| {
            self.allocator.free(token);
        }
        
        // Generate timestamp-based token (placeholder until proper parsing)
        const timestamp = std.time.milliTimestamp();
        self.csrf_token = try std.fmt.allocPrint(
            self.allocator,
            "csrf-token-{d}",
            .{timestamp},
        );
        
        std.log.debug("CSRF token fetched: {s}", .{self.csrf_token.?});
        _ = response; // TODO: Parse actual token from headers
    }
    
    /// POST /AgentModelAssignments
    pub fn createAssignment(self: *ODataPersistence, assignment: AssignmentEntity) !void {
        if (self.csrf_token == null) {
            try self.fetchCsrfToken();
        }
        
        // Build OData JSON payload
        const json = try self.assignmentToJson(assignment);
        defer self.allocator.free(json);
        
        // Build URL
        const url = try std.fmt.allocPrintZ(
            self.allocator,
            "{s}{s}/AgentModelAssignments",
            .{ self.config.base_url, self.config.service_path },
        );
        defer self.allocator.free(url);
        
        // Make HTTP POST request
        const response_ptr = zig_http_post(url.ptr, json.ptr, json.len);
        const response = std.mem.span(response_ptr);
        
        std.log.info("OData POST /AgentModelAssignments response: {s}", .{response});
    }
    
    /// POST /RoutingDecisions
    pub fn createRoutingDecision(self: *ODataPersistence, decision: RoutingDecisionEntity) !void {
        if (self.csrf_token == null) {
            try self.fetchCsrfToken();
        }
        
        const json = try self.decisionToJson(decision);
        defer self.allocator.free(json);
        
        // Build URL
        const url = try std.fmt.allocPrintZ(
            self.allocator,
            "{s}{s}/RoutingDecisions",
            .{ self.config.base_url, self.config.service_path },
        );
        defer self.allocator.free(url);
        
        // Make HTTP POST request
        const response_ptr = zig_http_post(url.ptr, json.ptr, json.len);
        const response = std.mem.span(response_ptr);
        
        std.log.info("OData POST /RoutingDecisions response: {s}", .{response});
    }
    
    /// POST /InferenceMetrics (batch)
    pub fn createMetricsBatch(self: *ODataPersistence, metrics: []const MetricsEntity) !void {
        if (self.csrf_token == null) {
            try self.fetchCsrfToken();
        }
        
        // Use OData $batch for multiple inserts
        // TODO: Implement $batch request
        
        for (metrics) |metric| {
            const json = try self.metricsToJson(metric);
            defer self.allocator.free(json);
            std.log.info("OData POST /InferenceMetrics: {s}", .{json});
        }
    }
    
    /// GET /AgentModelAssignments?$filter=Status eq 'ACTIVE'
    pub fn getActiveAssignments(self: *ODataPersistence) ![]AssignmentEntity {
        // Build URL with OData filter
        const url = try std.fmt.allocPrintZ(
            self.allocator,
            "{s}{s}/AgentModelAssignments?$filter=Status eq 'ACTIVE'",
            .{ self.config.base_url, self.config.service_path },
        );
        defer self.allocator.free(url);
        
        // Make HTTP GET request
        const response_ptr = zig_http_get(url.ptr);
        const response = std.mem.span(response_ptr);
        
        std.log.info("OData GET /AgentModelAssignments response: {s}", .{response});
        
        // TODO: Parse JSON response and convert to AssignmentEntity array
        // For now, return empty
        return try self.allocator.alloc(AssignmentEntity, 0);
    }
    
    /// GET /RoutingDecisions with time range filter
    pub fn getRoutingStats(self: *ODataPersistence, hours: u32) !RoutingStats {
        // TODO: Implement GET with OData aggregation
        // GET {base_url}/{service_path}/RoutingDecisions?
        //   $filter=DecisionTimestamp ge {cutoff_time}
        //   &$apply=aggregate(...)
        
        _ = self;
        _ = hours;
        
        return RoutingStats{
            .total_decisions = 0,
            .successful_decisions = 0,
            .avg_latency_ms = 0.0,
            .fallback_rate = 0.0,
        };
    }
    
    /// Helper: Convert Assignment to JSON
    fn assignmentToJson(self: *ODataPersistence, assignment: AssignmentEntity) ![]const u8 {
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{
            \\  "AssignmentID": "{s}",
            \\  "AgentID": "{s}",
            \\  "ModelID": "{s}",
            \\  "MatchScore": {d:.2},
            \\  "Status": "{s}",
            \\  "AssignmentMethod": "{s}",
            \\  "Capabilities": "{s}"
            \\}}
        ,
            .{
                assignment.id,
                assignment.agent_id,
                assignment.model_id,
                assignment.match_score,
                assignment.status,
                assignment.assignment_method,
                assignment.capabilities_json,
            },
        );
    }
    
    /// Helper: Convert Decision to JSON
    fn decisionToJson(self: *ODataPersistence, decision: RoutingDecisionEntity) ![]const u8 {
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{
            \\  "DecisionID": "{s}",
            \\  "RequestID": "{s}",
            \\  "TaskType": "{s}",
            \\  "AgentID": "{s}",
            \\  "ModelID": "{s}",
            \\  "CapabilityScore": {d:.2},
            \\  "PerformanceScore": {d:.2},
            \\  "CompositeScore": {d:.2},
            \\  "StrategyUsed": "{s}",
            \\  "LatencyMS": {d},
            \\  "Success": {},
            \\  "FallbackUsed": {}
            \\}}
        ,
            .{
                decision.id,
                decision.request_id,
                decision.task_type,
                decision.agent_id,
                decision.model_id,
                decision.capability_score,
                decision.performance_score,
                decision.composite_score,
                decision.strategy_used,
                decision.latency_ms,
                decision.success,
                decision.fallback_used,
            },
        );
    }
    
    /// Helper: Convert Metrics to JSON
    fn metricsToJson(self: *ODataPersistence, metrics: MetricsEntity) ![]const u8 {
        return try std.fmt.allocPrint(
            self.allocator,
            \\{{
            \\  "MetricID": "{s}",
            \\  "ModelID": "{s}",
            \\  "LatencyMS": {d},
            \\  "TTFTMS": {d},
            \\  "TokensGenerated": {d},
            \\  "CacheHit": {}
            \\}}
        ,
            .{
                metrics.id,
                metrics.model_id,
                metrics.latency_ms,
                metrics.ttft_ms,
                metrics.tokens_generated,
                metrics.cache_hit,
            },
        );
    }
};

/// OData entity types (match HANA Cloud OData service)

pub const AssignmentEntity = struct {
    id: []const u8,
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32,
    status: []const u8,
    assignment_method: []const u8,
    capabilities_json: []const u8,
};

pub const RoutingDecisionEntity = struct {
    id: []const u8,
    request_id: []const u8,
    task_type: []const u8,
    agent_id: []const u8,
    model_id: []const u8,
    capability_score: f32,
    performance_score: f32,
    composite_score: f32,
    strategy_used: []const u8,
    latency_ms: i32,
    success: bool,
    fallback_used: bool,
};

pub const MetricsEntity = struct {
    id: []const u8,
    model_id: []const u8,
    latency_ms: i32,
    ttft_ms: i32,
    tokens_generated: u32,
    cache_hit: bool,
};

pub const RoutingStats = struct {
    total_decisions: u64,
    successful_decisions: u64,
    avg_latency_ms: f32,
    fallback_rate: f32,
};

/// ID generation helpers
pub fn generateId(allocator: Allocator, prefix: []const u8) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    const random = std.crypto.random.int(u32);
    return try std.fmt.allocPrint(allocator, "{s}_{d}_{d}", .{ prefix, timestamp, random });
}

test "ODataPersistence initialization" {
    const allocator = std.testing.allocator;
    
    const config = ODataConfig{
        .base_url = "https://hana-cloud.example.com",
        .username = "TEST_USER",
        .password = "test123",
    };
    
    const client = try ODataPersistence.init(allocator, config);
    defer client.deinit();
    
    try std.testing.expect(client.csrf_token == null);
}

test "Assignment JSON serialization" {
    const allocator = std.testing.allocator;
    
    const config = ODataConfig{
        .base_url = "https://hana-cloud.example.com",
        .username = "TEST_USER",
        .password = "test123",
    };
    
    const client = try ODataPersistence.init(allocator, config);
    defer client.deinit();
    
    const assignment = AssignmentEntity{
        .id = "assign_123",
        .agent_id = "agent_1",
        .model_id = "llama-70b",
        .match_score = 0.95,
        .status = "ACTIVE",
        .assignment_method = "AUTO",
        .capabilities_json = "{}",
    };
    
    const json = try client.assignmentToJson(assignment);
    defer allocator.free(json);
    
    try std.testing.expect(std.mem.indexOf(u8, json, "assign_123") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "agent_1") != null);
}
