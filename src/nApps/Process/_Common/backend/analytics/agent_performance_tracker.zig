const std = @import("std");

/// Agent Performance Tracker
/// Monitors and analyzes agent performance over time
/// Provides insights for optimization and capacity planning

pub const AgentPerformanceTracker = struct {
    allocator: std.mem.Allocator,
    performance_records: std.ArrayList(PerformanceRecord),
    agent_metrics: std.StringHashMap(AgentMetrics),
    
    pub fn init(allocator: std.mem.Allocator) AgentPerformanceTracker {
        return AgentPerformanceTracker{
            .allocator = allocator,
            .performance_records = std.ArrayList(PerformanceRecord).init(allocator),
            .agent_metrics = std.StringHashMap(AgentMetrics).init(allocator),
        };
    }
    
    pub fn deinit(self: *AgentPerformanceTracker) void {
        self.performance_records.deinit();
        self.agent_metrics.deinit();
    }
    
    // =========================================================================
    // Record Performance Events
    // =========================================================================
    
    /// Record task assignment
    pub fn recordAssignment(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
        task_id: []const u8,
        task_complexity: TaskComplexity,
        timestamp: i64,
    ) !void {
        const record = PerformanceRecord{
            .event_type = .assignment,
            .agent_id = try self.allocator.dupe(u8, agent_id),
            .task_id = try self.allocator.dupe(u8, task_id),
            .timestamp = timestamp,
            .duration_ms = null,
            .success = null,
            .task_complexity = task_complexity,
        };
        
        try self.performance_records.append(record);
        try self.updateAgentMetrics(agent_id, .assignment);
    }
    
    /// Record task completion
    pub fn recordCompletion(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
        task_id: []const u8,
        duration_ms: u64,
        success: bool,
        timestamp: i64,
    ) !void {
        const record = PerformanceRecord{
            .event_type = .completion,
            .agent_id = try self.allocator.dupe(u8, agent_id),
            .task_id = try self.allocator.dupe(u8, task_id),
            .timestamp = timestamp,
            .duration_ms = duration_ms,
            .success = success,
            .task_complexity = .medium, // Could be looked up from assignment
        };
        
        try self.performance_records.append(record);
        try self.updateAgentMetrics(agent_id, if (success) .success else .failure);
    }
    
    fn updateAgentMetrics(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
        event: enum { assignment, success, failure },
    ) !void {
        const result = try self.agent_metrics.getOrPut(agent_id);
        
        if (!result.found_existing) {
            result.value_ptr.* = AgentMetrics{
                .total_assignments = 0,
                .total_completions = 0,
                .total_failures = 0,
                .total_duration_ms = 0,
                .avg_duration_ms = 0,
                .success_rate = 0.0,
                .tasks_by_complexity = .{
                    .low = 0,
                    .medium = 0,
                    .high = 0,
                },
            };
        }
        
        switch (event) {
            .assignment => result.value_ptr.total_assignments += 1,
            .success => {
                result.value_ptr.total_completions += 1;
                // Recalculate success rate
                const total_finished = result.value_ptr.total_completions + result.value_ptr.total_failures;
                if (total_finished > 0) {
                    result.value_ptr.success_rate = @as(f32, @floatFromInt(result.value_ptr.total_completions)) / 
                                                    @as(f32, @floatFromInt(total_finished));
                }
            },
            .failure => result.value_ptr.total_failures += 1,
        }
    }
    
    // =========================================================================
    // Analytics Queries
    // =========================================================================
    
    /// Get agent performance summary
    pub fn getAgentSummary(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
    ) ?AgentMetrics {
        return self.agent_metrics.get(agent_id);
    }
    
    /// Get top performing agents
    pub fn getTopPerformers(
        self: *AgentPerformanceTracker,
        limit: usize,
    ) ![]AgentPerformanceSummary {
        var summaries = std.ArrayList(AgentPerformanceSummary).init(self.allocator);
        defer summaries.deinit();
        
        var it = self.agent_metrics.iterator();
        while (it.next()) |entry| {
            try summaries.append(.{
                .agent_id = entry.key_ptr.*,
                .metrics = entry.value_ptr.*,
                .score = entry.value_ptr.success_rate * 100.0,
            });
        }
        
        // Sort by score descending
        const items = summaries.items;
        std.sort.insertion(AgentPerformanceSummary, items, {}, comparePerformance);
        
        const result_limit = @min(limit, items.len);
        const result = try self.allocator.alloc(AgentPerformanceSummary, result_limit);
        @memcpy(result, items[0..result_limit]);
        
        return result;
    }
    
    fn comparePerformance(_: void, a: AgentPerformanceSummary, b: AgentPerformanceSummary) bool {
        return a.score > b.score;
    }
    
    /// Calculate average task duration for agent
    pub fn getAverageTaskDuration(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
    ) !u64 {
        var total_duration: u64 = 0;
        var count: u64 = 0;
        
        for (self.performance_records.items) |record| {
            if (std.mem.eql(u8, record.agent_id, agent_id) and 
                record.event_type == .completion and 
                record.duration_ms != null) 
            {
                total_duration += record.duration_ms.?;
                count += 1;
            }
        }
        
        if (count == 0) return 0;
        return total_duration / count;
    }
    
    /// Get workload distribution
    pub fn getWorkloadDistribution(self: *AgentPerformanceTracker) !WorkloadDistribution {
        var distribution = WorkloadDistribution{
            .total_tasks = 0,
            .by_complexity = .{
                .low = 0,
                .medium = 0,
                .high = 0,
            },
            .by_agent = std.StringHashMap(u32).init(self.allocator),
        };
        
        for (self.performance_records.items) |record| {
            if (record.event_type == .assignment) {
                distribution.total_tasks += 1;
                
                switch (record.task_complexity) {
                    .low => distribution.by_complexity.low += 1,
                    .medium => distribution.by_complexity.medium += 1,
                    .high => distribution.by_complexity.high += 1,
                }
                
                const result = try distribution.by_agent.getOrPut(record.agent_id);
                if (!result.found_existing) {
                    result.value_ptr.* = 0;
                }
                result.value_ptr.* += 1;
            }
        }
        
        return distribution;
    }
    
    // =========================================================================
    // Trend Analysis
    // =========================================================================
    
    /// Analyze performance trends over time
    pub fn analyzeTrends(
        self: *AgentPerformanceTracker,
        agent_id: []const u8,
        time_window_days: u32,
    ) !PerformanceTrend {
        const now = std.time.timestamp();
        const window_start = now - (@as(i64, time_window_days) * 86400);
        
        var trend = PerformanceTrend{
            .direction = .stable,
            .success_rate_change = 0.0,
            .avg_duration_change = 0.0,
            .task_count = 0,
        };
        
        // Calculate metrics for first and second half of window
        const mid_point = window_start + ((now - window_start) / 2);
        
        var first_half_successes: u32 = 0;
        var first_half_total: u32 = 0;
        var second_half_successes: u32 = 0;
        var second_half_total: u32 = 0;
        
        for (self.performance_records.items) |record| {
            if (!std.mem.eql(u8, record.agent_id, agent_id)) continue;
            if (record.timestamp < window_start) continue;
            
            if (record.event_type == .completion) {
                trend.task_count += 1;
                
                if (record.timestamp < mid_point) {
                    first_half_total += 1;
                    if (record.success == true) first_half_successes += 1;
                } else {
                    second_half_total += 1;
                    if (record.success == true) second_half_successes += 1;
                }
            }
        }
        
        // Calculate trend direction
        if (first_half_total > 0 and second_half_total > 0) {
            const first_rate = @as(f32, @floatFromInt(first_half_successes)) / @as(f32, @floatFromInt(first_half_total));
            const second_rate = @as(f32, @floatFromInt(second_half_successes)) / @as(f32, @floatFromInt(second_half_total));
            
            trend.success_rate_change = second_rate - first_rate;
            
            if (trend.success_rate_change > 0.05) {
                trend.direction = .improving;
            } else if (trend.success_rate_change < -0.05) {
                trend.direction = .declining;
            }
        }
        
        return trend;
    }
    
    // =========================================================================
    // Export & Reporting
    // =========================================================================
    
    /// Export performance data as JSON
    pub fn exportJSON(self: *AgentPerformanceTracker) ![]u8 {
        var json = std.ArrayList(u8).init(self.allocator);
        const writer = json.writer();
        
        try writer.writeAll("{\n");
        try writer.print("  \"total_records\": {d},\n", .{self.performance_records.items.len});
        try writer.writeAll("  \"agents\": [\n");
        
        var it = self.agent_metrics.iterator();
        var first = true;
        while (it.next()) |entry| {
            if (!first) try writer.writeAll(",\n");
            first = false;
            
            try writer.writeAll("    {\n");
            try writer.print("      \"agent_id\": \"{s}\",\n", .{entry.key_ptr.*});
            try writer.print("      \"assignments\": {d},\n", .{entry.value_ptr.total_assignments});
            try writer.print("      \"completions\": {d},\n", .{entry.value_ptr.total_completions});
            try writer.print("      \"success_rate\": {d:.2}\n", .{entry.value_ptr.success_rate});
            try writer.writeAll("    }");
        }
        
        try writer.writeAll("\n  ]\n}\n");
        
        return json.toOwnedSlice();
    }
};

// =========================================================================
// Supporting Types
// =========================================================================

pub const PerformanceRecord = struct {
    event_type: enum { assignment, completion, escalation },
    agent_id: []const u8,
    task_id: []const u8,
    timestamp: i64,
    duration_ms: ?u64,
    success: ?bool,
    task_complexity: TaskComplexity,
};

pub const TaskComplexity = enum {
    low,
    medium,
    high,
};

pub const AgentMetrics = struct {
    total_assignments: u32,
    total_completions: u32,
    total_failures: u32,
    total_duration_ms: u64,
    avg_duration_ms: u32,
    success_rate: f32,
    tasks_by_complexity: struct {
        low: u32,
        medium: u32,
        high: u32,
    },
};

pub const AgentPerformanceSummary = struct {
    agent_id: []const u8,
    metrics: AgentMetrics,
    score: f32,
};

pub const WorkloadDistribution = struct {
    total_tasks: u32,
    by_complexity: struct {
        low: u32,
        medium: u32,
        high: u32,
    },
    by_agent: std.StringHashMap(u32),
};

pub const PerformanceTrend = struct {
    direction: enum { improving, stable, declining },
    success_rate_change: f32,
    avg_duration_change: f32,
    task_count: u32,
};

// =========================================================================
// Tests
// =========================================================================

test "AgentPerformanceTracker: record and query" {
    const allocator = std.testing.allocator;
    
    var tracker = AgentPerformanceTracker.init(allocator);
    defer tracker.deinit();
    
    // Record assignment
    try tracker.recordAssignment(
        "agent-001",
        "task-001",
        .medium,
        std.time.timestamp(),
    );
    
    // Record completion
    try tracker.recordCompletion(
        "agent-001",
        "task-001",
        1800000, // 30 minutes
        true,
        std.time.timestamp(),
    );
    
    const metrics = tracker.getAgentSummary("agent-001");
    try std.testing.expect(metrics != null);
    try std.testing.expectEqual(@as(u32, 1), metrics.?.total_assignments);
    try std.testing.expectEqual(@as(u32, 1), metrics.?.total_completions);
}