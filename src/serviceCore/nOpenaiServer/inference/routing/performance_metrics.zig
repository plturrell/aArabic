// ============================================================================
// Performance Metrics Collection - Day 26 Implementation
// ============================================================================
// Purpose: Track and analyze model router performance metrics
// Week: Week 6 (Days 26-30) - Performance Monitoring & Feedback Loop
// Phase: Month 2 - Model Router & Orchestration
// ============================================================================

const std = @import("std");

// ============================================================================
// METRIC TYPES
// ============================================================================

/// Latency measurement in milliseconds
pub const LatencyMetric = struct {
    min: f32,
    max: f32,
    avg: f32,
    p50: f32,
    p95: f32,
    p99: f32,
    count: u64,
};

/// Success rate metric
pub const SuccessRateMetric = struct {
    total_requests: u64,
    successful_requests: u64,
    failed_requests: u64,
    success_rate: f32, // 0.0 - 1.0
};

/// Routing decision record
pub const RoutingDecision = struct {
    decision_id: []const u8,
    timestamp: i64,
    agent_id: []const u8,
    model_id: []const u8,
    match_score: f32,
    latency_ms: f32,
    success: bool,
    error_msg: ?[]const u8,
    
    pub fn init(
        allocator: std.mem.Allocator,
        decision_id: []const u8,
        agent_id: []const u8,
        model_id: []const u8,
        match_score: f32,
    ) !RoutingDecision {
        return .{
            .decision_id = try allocator.dupe(u8, decision_id),
            .timestamp = std.time.milliTimestamp(),
            .agent_id = try allocator.dupe(u8, agent_id),
            .model_id = try allocator.dupe(u8, model_id),
            .match_score = match_score,
            .latency_ms = 0.0,
            .success = false,
            .error_msg = null,
        };
    }
    
    pub fn deinit(self: *RoutingDecision, allocator: std.mem.Allocator) void {
        allocator.free(self.decision_id);
        allocator.free(self.agent_id);
        allocator.free(self.model_id);
        if (self.error_msg) |msg| {
            allocator.free(msg);
        }
    }
};

// ============================================================================
// PERFORMANCE TRACKER
// ============================================================================

pub const PerformanceTracker = struct {
    allocator: std.mem.Allocator,
    
    // Recent decisions (ring buffer)
    recent_decisions: std.ArrayList(RoutingDecision),
    max_recent: usize,
    
    // Aggregated metrics by model
    model_metrics: std.StringHashMap(ModelMetrics),
    
    // Aggregated metrics by agent
    agent_metrics: std.StringHashMap(AgentMetrics),
    
    // Global metrics
    global_latencies: std.ArrayList(f32),
    global_success_count: u64,
    global_failure_count: u64,
    
    pub const ModelMetrics = struct {
        total_requests: u64,
        successful_requests: u64,
        failed_requests: u64,
        total_latency_ms: f64,
        min_latency_ms: f32,
        max_latency_ms: f32,
        avg_match_score: f64,
        
        pub fn init() ModelMetrics {
            return .{
                .total_requests = 0,
                .successful_requests = 0,
                .failed_requests = 0,
                .total_latency_ms = 0.0,
                .min_latency_ms = std.math.floatMax(f32),
                .max_latency_ms = 0.0,
                .avg_match_score = 0.0,
            };
        }
        
        pub fn update(self: *ModelMetrics, latency_ms: f32, match_score: f32, success: bool) void {
            self.total_requests += 1;
            if (success) {
                self.successful_requests += 1;
            } else {
                self.failed_requests += 1;
            }
            
            self.total_latency_ms += latency_ms;
            self.min_latency_ms = @min(self.min_latency_ms, latency_ms);
            self.max_latency_ms = @max(self.max_latency_ms, latency_ms);
            
            // Update running average of match score
            const old_count = @as(f64, @floatFromInt(self.total_requests - 1));
            const new_count = @as(f64, @floatFromInt(self.total_requests));
            self.avg_match_score = (self.avg_match_score * old_count + match_score) / new_count;
        }
        
        pub fn getSuccessRate(self: *const ModelMetrics) f32 {
            if (self.total_requests == 0) return 0.0;
            return @as(f32, @floatFromInt(self.successful_requests)) / 
                   @as(f32, @floatFromInt(self.total_requests));
        }
        
        pub fn getAvgLatency(self: *const ModelMetrics) f32 {
            if (self.total_requests == 0) return 0.0;
            return @as(f32, @floatCast(self.total_latency_ms / @as(f64, @floatFromInt(self.total_requests))));
        }
    };
    
    pub const AgentMetrics = struct {
        total_assignments: u64,
        successful_assignments: u64,
        failed_assignments: u64,
        avg_match_score: f64,
        current_model_id: ?[]const u8,
        
        pub fn init() AgentMetrics {
            return .{
                .total_assignments = 0,
                .successful_assignments = 0,
                .failed_assignments = 0,
                .avg_match_score = 0.0,
                .current_model_id = null,
            };
        }
        
        pub fn update(self: *AgentMetrics, match_score: f32, success: bool, model_id: []const u8) void {
            self.total_assignments += 1;
            if (success) {
                self.successful_assignments += 1;
            } else {
                self.failed_assignments += 1;
            }
            
            // Update running average
            const old_count = @as(f64, @floatFromInt(self.total_assignments - 1));
            const new_count = @as(f64, @floatFromInt(self.total_assignments));
            self.avg_match_score = (self.avg_match_score * old_count + match_score) / new_count;
            
            self.current_model_id = model_id;
        }
    };
    
    pub fn init(allocator: std.mem.Allocator, max_recent: usize) PerformanceTracker {
        return .{
            .allocator = allocator,
            .recent_decisions = std.ArrayList(RoutingDecision).init(allocator),
            .max_recent = max_recent,
            .model_metrics = std.StringHashMap(ModelMetrics).init(allocator),
            .agent_metrics = std.StringHashMap(AgentMetrics).init(allocator),
            .global_latencies = std.ArrayList(f32).init(allocator),
            .global_success_count = 0,
            .global_failure_count = 0,
        };
    }
    
    pub fn deinit(self: *PerformanceTracker) void {
        for (self.recent_decisions.items) |*decision| {
            decision.deinit(self.allocator);
        }
        self.recent_decisions.deinit();
        self.model_metrics.deinit();
        self.agent_metrics.deinit();
        self.global_latencies.deinit();
    }
    
    /// Record a routing decision
    pub fn recordDecision(
        self: *PerformanceTracker,
        decision: RoutingDecision,
    ) !void {
        // Add to recent decisions (ring buffer)
        if (self.recent_decisions.items.len >= self.max_recent) {
            var old = self.recent_decisions.orderedRemove(0);
            old.deinit(self.allocator);
        }
        try self.recent_decisions.append(decision);
        
        // Update model metrics
        var model_entry = try self.model_metrics.getOrPut(decision.model_id);
        if (!model_entry.found_existing) {
            model_entry.value_ptr.* = ModelMetrics.init();
        }
        model_entry.value_ptr.update(decision.latency_ms, decision.match_score, decision.success);
        
        // Update agent metrics
        var agent_entry = try self.agent_metrics.getOrPut(decision.agent_id);
        if (!agent_entry.found_existing) {
            agent_entry.value_ptr.* = AgentMetrics.init();
        }
        agent_entry.value_ptr.update(decision.match_score, decision.success, decision.model_id);
        
        // Update global metrics
        try self.global_latencies.append(decision.latency_ms);
        if (decision.success) {
            self.global_success_count += 1;
        } else {
            self.global_failure_count += 1;
        }
    }
    
    /// Get latency statistics
    pub fn getLatencyMetrics(self: *const PerformanceTracker, allocator: std.mem.Allocator) !LatencyMetric {
        if (self.global_latencies.items.len == 0) {
            return LatencyMetric{
                .min = 0.0,
                .max = 0.0,
                .avg = 0.0,
                .p50 = 0.0,
                .p95 = 0.0,
                .p99 = 0.0,
                .count = 0,
            };
        }
        
        // Sort latencies for percentile calculation
        const sorted = try allocator.dupe(f32, self.global_latencies.items);
        defer allocator.free(sorted);
        
        std.mem.sort(f32, sorted, {}, comptime std.sort.asc(f32));
        
        const min: f32 = sorted[0];
        const max: f32 = sorted[sorted.len - 1];
        
        var sum: f64 = 0.0;
        for (sorted) |lat| {
            sum += lat;
        }
        
        const avg = @as(f32, @floatCast(sum / @as(f64, @floatFromInt(sorted.len))));
        const p50_idx = sorted.len / 2;
        const p95_idx = (sorted.len * 95) / 100;
        const p99_idx = (sorted.len * 99) / 100;
        
        return LatencyMetric{
            .min = min,
            .max = max,
            .avg = avg,
            .p50 = sorted[p50_idx],
            .p95 = sorted[@min(p95_idx, sorted.len - 1)],
            .p99 = sorted[@min(p99_idx, sorted.len - 1)],
            .count = sorted.len,
        };
    }
    
    /// Get success rate
    pub fn getSuccessRate(self: *const PerformanceTracker) SuccessRateMetric {
        const total = self.global_success_count + self.global_failure_count;
        const rate = if (total > 0)
            @as(f32, @floatFromInt(self.global_success_count)) / @as(f32, @floatFromInt(total))
        else
            0.0;
        
        return SuccessRateMetric{
            .total_requests = total,
            .successful_requests = self.global_success_count,
            .failed_requests = self.global_failure_count,
            .success_rate = rate,
        };
    }
    
    /// Get top performing models
    pub fn getTopModels(
        self: *const PerformanceTracker,
        allocator: std.mem.Allocator,
        limit: usize,
    ) ![]ModelPerformance {
        var performances = std.ArrayList(ModelPerformance).init(allocator);
        
        var iter = self.model_metrics.iterator();
        while (iter.next()) |entry| {
            try performances.append(.{
                .model_id = entry.key_ptr.*,
                .total_requests = entry.value_ptr.total_requests,
                .success_rate = entry.value_ptr.getSuccessRate(),
                .avg_latency_ms = entry.value_ptr.getAvgLatency(),
                .avg_match_score = @as(f32, @floatCast(entry.value_ptr.avg_match_score)),
            });
        }
        
        // Sort by success rate * avg_match_score (quality metric)
        std.mem.sort(ModelPerformance, performances.items, {}, struct {
            fn lessThan(_: void, a: ModelPerformance, b: ModelPerformance) bool {
                const score_a = a.success_rate * a.avg_match_score;
                const score_b = b.success_rate * b.avg_match_score;
                return score_a > score_b;
            }
        }.lessThan);
        
        // Return top N
        const result_len = @min(limit, performances.items.len);
        const owned = try performances.toOwnedSlice();
        return owned[0..result_len];
    }
    
    pub const ModelPerformance = struct {
        model_id: []const u8,
        total_requests: u64,
        success_rate: f32,
        avg_latency_ms: f32,
        avg_match_score: f32,
    };
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/// Generate a unique decision ID
pub fn generateDecisionId(allocator: std.mem.Allocator) ![]const u8 {
    const timestamp = std.time.milliTimestamp();
    const random = std.crypto.random.int(u32);
    return std.fmt.allocPrint(allocator, "decision_{d}_{x}", .{ timestamp, random });
}

// ============================================================================
// UNIT TESTS
// ============================================================================

test "PerformanceTracker: basic recording" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    var decision = try RoutingDecision.init(allocator, "test-1", "agent-1", "model-1", 85.5);
    decision.latency_ms = 150.0;
    decision.success = true;
    
    try tracker.recordDecision(decision);
    
    try std.testing.expectEqual(@as(usize, 1), tracker.recent_decisions.items.len);
    try std.testing.expectEqual(@as(u64, 1), tracker.global_success_count);
}

test "PerformanceTracker: latency metrics" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // Record several decisions with varying latencies
    const latencies = [_]f32{ 100.0, 150.0, 200.0, 250.0, 300.0 };
    
    for (latencies, 0..) |lat, i| {
        const id = try std.fmt.allocPrint(allocator, "test-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try RoutingDecision.init(allocator, id, "agent-1", "model-1", 85.0);
        decision.latency_ms = lat;
        decision.success = true;
        try tracker.recordDecision(decision);
    }
    
    const metrics = try tracker.getLatencyMetrics(allocator);
    
    try std.testing.expectEqual(@as(f32, 100.0), metrics.min);
    try std.testing.expectEqual(@as(f32, 300.0), metrics.max);
    try std.testing.expectEqual(@as(u64, 5), metrics.count);
    try std.testing.expect(metrics.avg >= 190.0 and metrics.avg <= 210.0);
}

test "PerformanceTracker: success rate" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // 7 successful, 3 failed
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "test-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try RoutingDecision.init(allocator, id, "agent-1", "model-1", 85.0);
        decision.latency_ms = 150.0;
        decision.success = (i < 7);
        try tracker.recordDecision(decision);
    }
    
    const rate_metric = tracker.getSuccessRate();
    
    try std.testing.expectEqual(@as(u64, 10), rate_metric.total_requests);
    try std.testing.expectEqual(@as(u64, 7), rate_metric.successful_requests);
    try std.testing.expectEqual(@as(u64, 3), rate_metric.failed_requests);
    try std.testing.expect(rate_metric.success_rate >= 0.69 and rate_metric.success_rate <= 0.71);
}

test "PerformanceTracker: model metrics" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // Record decisions for model-1
    var i: u32 = 0;
    while (i < 5) : (i += 1) {
        const id = try std.fmt.allocPrint(allocator, "test-{d}", .{i});
        defer allocator.free(id);
        
        var decision = try RoutingDecision.init(allocator, id, "agent-1", "model-1", 80.0 + @as(f32, @floatFromInt(i)));
        decision.latency_ms = 100.0 + @as(f32, @floatFromInt(i * 10));
        decision.success = true;
        try tracker.recordDecision(decision);
    }
    
    const model_metrics = tracker.model_metrics.get("model-1").?;
    
    try std.testing.expectEqual(@as(u64, 5), model_metrics.total_requests);
    try std.testing.expectEqual(@as(u64, 5), model_metrics.successful_requests);
    try std.testing.expect(model_metrics.getAvgLatency() >= 120.0 and model_metrics.getAvgLatency() <= 140.0);
}

test "PerformanceTracker: top models" {
    const allocator = std.testing.allocator;
    
    var tracker = PerformanceTracker.init(allocator, 100);
    defer tracker.deinit();
    
    // Record decisions for multiple models
    const models = [_][]const u8{ "model-1", "model-2", "model-3" };
    const scores = [_]f32{ 90.0, 85.0, 80.0 };
    
    for (models, 0..) |model, idx| {
        var i: u32 = 0;
        while (i < 3) : (i += 1) {
            const id = try std.fmt.allocPrint(allocator, "{s}-{d}", .{ model, i });
            defer allocator.free(id);
            
            var decision = try RoutingDecision.init(allocator, id, "agent-1", model, scores[idx]);
            decision.latency_ms = 150.0;
            decision.success = true;
            try tracker.recordDecision(decision);
        }
    }
    
    const top_models = try tracker.getTopModels(allocator, 2);
    defer allocator.free(top_models);
    
    try std.testing.expectEqual(@as(usize, 2), top_models.len);
    try std.testing.expect(top_models[0].avg_match_score > top_models[1].avg_match_score);
}
