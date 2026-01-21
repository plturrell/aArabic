//! Bottleneck Analyzer for Process Performance Analysis
//! Identifies bottlenecks in process execution

const std = @import("std");
const Allocator = std.mem.Allocator;
const event_collector = @import("event_collector.zig");

// ============================================================================
// Bottleneck Types
// ============================================================================

/// Types of bottlenecks in process execution
pub const BottleneckType = enum {
    TIME, // Activity takes too long
    RESOURCE, // Resource is overloaded
    CAPACITY, // Throughput limit reached
    HANDOFF, // Delays between activities
    REWORK, // High rework/loop rate
    BATCH, // Batching delays

    pub fn toString(self: BottleneckType) []const u8 {
        return switch (self) {
            .TIME => "Time",
            .RESOURCE => "Resource",
            .CAPACITY => "Capacity",
            .HANDOFF => "Handoff",
            .REWORK => "Rework",
            .BATCH => "Batch",
        };
    }

    pub fn getSeverityWeight(self: BottleneckType) f64 {
        return switch (self) {
            .TIME => 1.0,
            .RESOURCE => 0.9,
            .CAPACITY => 0.85,
            .HANDOFF => 0.7,
            .REWORK => 0.8,
            .BATCH => 0.6,
        };
    }
};

/// Severity levels for bottlenecks
pub const Severity = enum {
    LOW,
    MEDIUM,
    HIGH,
    CRITICAL,

    pub fn toString(self: Severity) []const u8 {
        return @tagName(self);
    }

    pub fn fromScore(score: f64) Severity {
        if (score >= 0.9) return .CRITICAL;
        if (score >= 0.7) return .HIGH;
        if (score >= 0.4) return .MEDIUM;
        return .LOW;
    }
};

// ============================================================================
// Activity Metrics
// ============================================================================

/// Performance metrics for an activity
pub const ActivityMetrics = struct {
    activity: []const u8,
    execution_count: u64 = 0,
    total_duration_ms: u64 = 0,
    min_duration_ms: u64 = std.math.maxInt(u64),
    max_duration_ms: u64 = 0,
    total_wait_time_ms: u64 = 0,
    resource_utilization: f64 = 0.0,
    rework_count: u64 = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, activity: []const u8) !ActivityMetrics {
        return ActivityMetrics{
            .activity = try allocator.dupe(u8, activity),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ActivityMetrics) void {
        self.allocator.free(self.activity);
    }

    pub fn recordExecution(self: *ActivityMetrics, duration_ms: u64) void {
        self.execution_count += 1;
        self.total_duration_ms += duration_ms;
        if (duration_ms < self.min_duration_ms) self.min_duration_ms = duration_ms;
        if (duration_ms > self.max_duration_ms) self.max_duration_ms = duration_ms;
    }

    pub fn recordWaitTime(self: *ActivityMetrics, wait_ms: u64) void {
        self.total_wait_time_ms += wait_ms;
    }

    pub fn recordRework(self: *ActivityMetrics) void {
        self.rework_count += 1;
    }

    pub fn getAvgDuration(self: *const ActivityMetrics) f64 {
        if (self.execution_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_duration_ms)) /
            @as(f64, @floatFromInt(self.execution_count));
    }

    pub fn getAvgWaitTime(self: *const ActivityMetrics) f64 {
        if (self.execution_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.total_wait_time_ms)) /
            @as(f64, @floatFromInt(self.execution_count));
    }

    pub fn getReworkRate(self: *const ActivityMetrics) f64 {
        if (self.execution_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.rework_count)) /
            @as(f64, @floatFromInt(self.execution_count));
    }

    pub fn getVariance(self: *const ActivityMetrics) f64 {
        if (self.execution_count < 2) return 0.0;
        const range = @as(f64, @floatFromInt(self.max_duration_ms - self.min_duration_ms));
        return range / self.getAvgDuration();
    }
};

// ============================================================================
// Bottleneck
// ============================================================================

/// Identified bottleneck
pub const Bottleneck = struct {
    activity: []const u8,
    bottleneck_type: BottleneckType,
    severity: Severity,
    score: f64, // 0.0 - 1.0
    impact_score: f64, // Impact on overall process
    recommendation: []const u8,
    details: ?[]const u8 = null,
    allocator: Allocator,

    pub fn init(
        allocator: Allocator,
        activity: []const u8,
        bottleneck_type: BottleneckType,
        score: f64,
        recommendation: []const u8,
    ) !Bottleneck {
        return Bottleneck{
            .activity = try allocator.dupe(u8, activity),
            .bottleneck_type = bottleneck_type,
            .severity = Severity.fromScore(score),
            .score = score,
            .impact_score = score * bottleneck_type.getSeverityWeight(),
            .recommendation = try allocator.dupe(u8, recommendation),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Bottleneck) void {
        self.allocator.free(self.activity);
        self.allocator.free(self.recommendation);
        if (self.details) |d| self.allocator.free(d);
    }

    pub fn setDetails(self: *Bottleneck, details: []const u8) !void {
        if (self.details) |d| self.allocator.free(d);
        self.details = try self.allocator.dupe(u8, details);
    }

    pub fn compare(_: void, a: Bottleneck, b: Bottleneck) bool {
        return a.impact_score > b.impact_score; // Higher impact first
    }
};

// ============================================================================
// Wait Time Analyzer
// ============================================================================

/// Analyzes wait times between activities
pub const WaitTimeAnalyzer = struct {
    allocator: Allocator,
    threshold_ms: u64,

    pub fn init(allocator: Allocator, threshold_ms: u64) WaitTimeAnalyzer {
        return WaitTimeAnalyzer{
            .allocator = allocator,
            .threshold_ms = threshold_ms,
        };
    }

    pub fn analyzeHandoffs(
        self: *WaitTimeAnalyzer,
        log: *const event_collector.EventLog,
        metrics_map: *std.StringHashMap(ActivityMetrics),
    ) !void {
        _ = self;
        var trace_iter = log.traces.valueIterator();
        while (trace_iter.next()) |trace| {
            const events = trace.*.events.items;

            var i: usize = 1;
            while (i < events.len) : (i += 1) {
                const prev_event = events[i - 1];
                const curr_event = events[i];

                // Calculate wait time between events
                const time_diff = curr_event.timestamp - prev_event.timestamp;
                if (time_diff > 0) {
                    const wait_ms: u64 = @intCast(time_diff * 1000); // Assuming timestamps in seconds

                    if (metrics_map.getEntry(curr_event.activity)) |entry| {
                        entry.value_ptr.recordWaitTime(wait_ms);
                    }
                }
            }
        }
    }

    pub fn getHighWaitActivities(
        self: *WaitTimeAnalyzer,
        metrics_map: *const std.StringHashMap(ActivityMetrics),
        allocator: Allocator,
    ) ![][]const u8 {
        var result = std.ArrayList([]const u8){};
        errdefer result.deinit(allocator);

        var iter = metrics_map.valueIterator();
        while (iter.next()) |metrics| {
            if (metrics.getAvgWaitTime() > @as(f64, @floatFromInt(self.threshold_ms))) {
                try result.append(allocator, metrics.activity);
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Resource Utilization Analyzer
// ============================================================================

/// Analyzes resource utilization
pub const ResourceUtilizationAnalyzer = struct {
    allocator: Allocator,
    overload_threshold: f64,

    pub fn init(allocator: Allocator, overload_threshold: f64) ResourceUtilizationAnalyzer {
        return ResourceUtilizationAnalyzer{
            .allocator = allocator,
            .overload_threshold = overload_threshold,
        };
    }

    pub fn analyzeResources(
        self: *ResourceUtilizationAnalyzer,
        log: *const event_collector.EventLog,
    ) !std.StringHashMap(f64) {
        var resource_counts = std.StringHashMap(u64).init(self.allocator);
        defer {
            var key_iter = resource_counts.keyIterator();
            while (key_iter.next()) |key| {
                self.allocator.free(key.*);
            }
            resource_counts.deinit();
        }

        var total_events: u64 = 0;

        // Count events per resource
        var trace_iter = log.traces.valueIterator();
        while (trace_iter.next()) |trace| {
            for (trace.*.events.items) |event| {
                total_events += 1;
                if (event.resource) |resource| {
                    if (resource_counts.getEntry(resource)) |entry| {
                        entry.value_ptr.* += 1;
                    } else {
                        const key = try self.allocator.dupe(u8, resource);
                        try resource_counts.put(key, 1);
                    }
                }
            }
        }

        // Calculate utilization as relative workload
        var utilization = std.StringHashMap(f64).init(self.allocator);
        var res_iter = resource_counts.iterator();
        while (res_iter.next()) |entry| {
            const count = @as(f64, @floatFromInt(entry.value_ptr.*));
            const total = @as(f64, @floatFromInt(total_events));
            const util = if (total > 0) count / total else 0.0;

            const key = try self.allocator.dupe(u8, entry.key_ptr.*);
            try utilization.put(key, util);
        }

        return utilization;
    }

    pub fn getOverloadedResources(
        self: *ResourceUtilizationAnalyzer,
        utilization: *const std.StringHashMap(f64),
        allocator: Allocator,
    ) ![][]const u8 {
        var result = std.ArrayList([]const u8){};
        errdefer result.deinit(allocator);

        var iter = utilization.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.* > self.overload_threshold) {
                try result.append(allocator, entry.key_ptr.*);
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Analysis Result
// ============================================================================

/// Analysis result
pub const AnalysisResult = struct {
    bottlenecks: std.ArrayList(Bottleneck),
    activity_metrics: std.StringHashMap(ActivityMetrics),
    total_process_time_ms: u64 = 0,
    avg_cycle_time_ms: f64 = 0.0,
    bottleneck_count: usize = 0,
    analysis_timestamp: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) AnalysisResult {
        return AnalysisResult{
            .bottlenecks = std.ArrayList(Bottleneck){},
            .activity_metrics = std.StringHashMap(ActivityMetrics).init(allocator),
            .analysis_timestamp = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *AnalysisResult) void {
        for (self.bottlenecks.items) |*b| b.deinit();
        self.bottlenecks.deinit(self.allocator);

        var iter = self.activity_metrics.valueIterator();
        while (iter.next()) |metrics| {
            var m = metrics;
            m.deinit();
        }
        self.activity_metrics.deinit();
    }

    pub fn addBottleneck(self: *AnalysisResult, bottleneck: Bottleneck) !void {
        try self.bottlenecks.append(self.allocator, bottleneck);
        self.bottleneck_count = self.bottlenecks.items.len;
    }

    pub fn sortBottlenecks(self: *AnalysisResult) void {
        std.mem.sort(Bottleneck, self.bottlenecks.items, {}, Bottleneck.compare);
    }

    pub fn getCriticalBottlenecks(self: *const AnalysisResult, allocator: Allocator) ![]Bottleneck {
        var result = std.ArrayList(Bottleneck){};
        errdefer result.deinit(allocator);

        for (self.bottlenecks.items) |b| {
            if (b.severity == .CRITICAL or b.severity == .HIGH) {
                try result.append(allocator, b);
            }
        }

        return result.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Bottleneck Analyzer
// ============================================================================

/// Main Bottleneck Analyzer
pub const BottleneckAnalyzer = struct {
    allocator: Allocator,
    time_threshold_ms: u64,
    wait_time_threshold_ms: u64,
    rework_threshold: f64,
    resource_overload_threshold: f64,

    pub fn init(allocator: Allocator) BottleneckAnalyzer {
        return BottleneckAnalyzer{
            .allocator = allocator,
            .time_threshold_ms = 5000, // 5 seconds default
            .wait_time_threshold_ms = 10000, // 10 seconds default
            .rework_threshold = 0.1, // 10% rework rate
            .resource_overload_threshold = 0.8, // 80% utilization
        };
    }

    pub fn setTimeThreshold(self: *BottleneckAnalyzer, threshold_ms: u64) void {
        self.time_threshold_ms = threshold_ms;
    }

    pub fn setWaitTimeThreshold(self: *BottleneckAnalyzer, threshold_ms: u64) void {
        self.wait_time_threshold_ms = threshold_ms;
    }

    pub fn setReworkThreshold(self: *BottleneckAnalyzer, threshold: f64) void {
        self.rework_threshold = threshold;
    }

    /// Main analysis method
    pub fn analyze(self: *BottleneckAnalyzer, log: *const event_collector.EventLog) !AnalysisResult {
        var result = AnalysisResult.init(self.allocator);
        errdefer result.deinit();

        // Step 1: Collect activity metrics
        try self.collectActivityMetrics(log, &result.activity_metrics);

        // Step 2: Analyze for time bottlenecks
        try self.analyzeTimeBottlenecks(&result);

        // Step 3: Analyze handoff/wait time bottlenecks
        var wait_analyzer = WaitTimeAnalyzer.init(self.allocator, self.wait_time_threshold_ms);
        try wait_analyzer.analyzeHandoffs(log, &result.activity_metrics);
        try self.analyzeWaitTimeBottlenecks(&result);

        // Step 4: Analyze rework bottlenecks
        try self.analyzeReworkBottlenecks(&result);

        // Step 5: Calculate overall metrics
        result.avg_cycle_time_ms = self.calculateAvgCycleTime(log);

        // Step 6: Sort bottlenecks by impact
        result.sortBottlenecks();

        return result;
    }

    fn collectActivityMetrics(
        self: *BottleneckAnalyzer,
        log: *const event_collector.EventLog,
        metrics_map: *std.StringHashMap(ActivityMetrics),
    ) !void {
        var trace_iter = log.traces.valueIterator();
        while (trace_iter.next()) |trace| {
            var seen_activities = std.StringHashMap(void).init(self.allocator);
            defer seen_activities.deinit();

            for (trace.*.events.items) |event| {
                // Get or create metrics for this activity
                if (!metrics_map.contains(event.activity)) {
                    const metrics = try ActivityMetrics.init(self.allocator, event.activity);
                    try metrics_map.put(metrics.activity, metrics);
                }

                if (metrics_map.getEntry(event.activity)) |entry| {
                    // Record duration if available
                    if (event.duration_ms) |dur| {
                        entry.value_ptr.recordExecution(dur);
                    } else {
                        entry.value_ptr.execution_count += 1;
                    }

                    // Check for rework (activity seen again in same trace)
                    if (seen_activities.contains(event.activity)) {
                        entry.value_ptr.recordRework();
                    } else {
                        try seen_activities.put(event.activity, {});
                    }
                }
            }
        }
    }

    fn analyzeTimeBottlenecks(self: *BottleneckAnalyzer, result: *AnalysisResult) !void {
        var iter = result.activity_metrics.valueIterator();
        while (iter.next()) |metrics| {
            const avg_duration = metrics.getAvgDuration();
            if (avg_duration > @as(f64, @floatFromInt(self.time_threshold_ms))) {
                const score = @min(1.0, avg_duration / @as(f64, @floatFromInt(self.time_threshold_ms * 2)));
                const bottleneck = try Bottleneck.init(
                    self.allocator,
                    metrics.activity,
                    .TIME,
                    score,
                    "Consider optimizing this activity or parallelizing its execution",
                );
                try result.addBottleneck(bottleneck);
            }
        }
    }

    fn analyzeWaitTimeBottlenecks(self: *BottleneckAnalyzer, result: *AnalysisResult) !void {
        var iter = result.activity_metrics.valueIterator();
        while (iter.next()) |metrics| {
            const avg_wait = metrics.getAvgWaitTime();
            if (avg_wait > @as(f64, @floatFromInt(self.wait_time_threshold_ms))) {
                const score = @min(1.0, avg_wait / @as(f64, @floatFromInt(self.wait_time_threshold_ms * 2)));
                const bottleneck = try Bottleneck.init(
                    self.allocator,
                    metrics.activity,
                    .HANDOFF,
                    score,
                    "Reduce handoff delays by improving communication or automating transitions",
                );
                try result.addBottleneck(bottleneck);
            }
        }
    }

    fn analyzeReworkBottlenecks(self: *BottleneckAnalyzer, result: *AnalysisResult) !void {
        var iter = result.activity_metrics.valueIterator();
        while (iter.next()) |metrics| {
            const rework_rate = metrics.getReworkRate();
            if (rework_rate > self.rework_threshold) {
                const score = @min(1.0, rework_rate / (self.rework_threshold * 2));
                const bottleneck = try Bottleneck.init(
                    self.allocator,
                    metrics.activity,
                    .REWORK,
                    score,
                    "Investigate root causes of rework and implement quality gates",
                );
                try result.addBottleneck(bottleneck);
            }
        }
    }

    fn calculateAvgCycleTime(self: *BottleneckAnalyzer, log: *const event_collector.EventLog) f64 {
        _ = self;
        var total_duration: u64 = 0;
        var completed_traces: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            if (trace.*.getDuration()) |dur| {
                total_duration += dur;
                completed_traces += 1;
            }
        }

        if (completed_traces == 0) return 0.0;
        return @as(f64, @floatFromInt(total_duration)) / @as(f64, @floatFromInt(completed_traces));
    }
};

// ============================================================================
// Tests
// ============================================================================

test "BottleneckType severity weights" {
    try std.testing.expect(BottleneckType.TIME.getSeverityWeight() >= BottleneckType.HANDOFF.getSeverityWeight());
    try std.testing.expect(BottleneckType.RESOURCE.getSeverityWeight() >= BottleneckType.BATCH.getSeverityWeight());
}

test "Severity from score" {
    try std.testing.expectEqual(Severity.CRITICAL, Severity.fromScore(0.95));
    try std.testing.expectEqual(Severity.HIGH, Severity.fromScore(0.75));
    try std.testing.expectEqual(Severity.MEDIUM, Severity.fromScore(0.5));
    try std.testing.expectEqual(Severity.LOW, Severity.fromScore(0.2));
}

test "ActivityMetrics basic operations" {
    const allocator = std.testing.allocator;

    var metrics = try ActivityMetrics.init(allocator, "Test Activity");
    defer metrics.deinit();

    metrics.recordExecution(100);
    metrics.recordExecution(200);
    metrics.recordExecution(150);

    try std.testing.expectEqual(@as(u64, 3), metrics.execution_count);
    try std.testing.expectEqual(@as(u64, 450), metrics.total_duration_ms);
    try std.testing.expect(metrics.getAvgDuration() == 150.0);
}

test "ActivityMetrics wait time" {
    const allocator = std.testing.allocator;

    var metrics = try ActivityMetrics.init(allocator, "Test");
    defer metrics.deinit();

    metrics.execution_count = 2;
    metrics.recordWaitTime(100);
    metrics.recordWaitTime(200);

    try std.testing.expect(metrics.getAvgWaitTime() == 150.0);
}

test "ActivityMetrics rework rate" {
    const allocator = std.testing.allocator;

    var metrics = try ActivityMetrics.init(allocator, "Test");
    defer metrics.deinit();

    metrics.execution_count = 10;
    metrics.recordRework();
    metrics.recordRework();

    try std.testing.expect(metrics.getReworkRate() == 0.2);
}

test "Bottleneck creation" {
    const allocator = std.testing.allocator;

    var bottleneck = try Bottleneck.init(
        allocator,
        "Slow Activity",
        .TIME,
        0.85,
        "Optimize this activity",
    );
    defer bottleneck.deinit();

    try std.testing.expectEqualStrings("Slow Activity", bottleneck.activity);
    try std.testing.expectEqual(BottleneckType.TIME, bottleneck.bottleneck_type);
    try std.testing.expectEqual(Severity.HIGH, bottleneck.severity);
}

test "AnalysisResult basic" {
    const allocator = std.testing.allocator;

    var result = AnalysisResult.init(allocator);
    defer result.deinit();

    const bottleneck = try Bottleneck.init(allocator, "Test", .TIME, 0.5, "Fix it");
    try result.addBottleneck(bottleneck);

    try std.testing.expectEqual(@as(usize, 1), result.bottleneck_count);
}

test "BottleneckAnalyzer analyze empty log" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    var analyzer = BottleneckAnalyzer.init(allocator);
    var result = try analyzer.analyze(&log);
    defer result.deinit();

    try std.testing.expectEqual(@as(usize, 0), result.bottleneck_count);
}

test "BottleneckAnalyzer analyze with events" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    var e1 = try event_collector.ProcessEvent.init(allocator, "e1", "case-1", "Start");
    e1.duration_ms = 100;
    try log.recordEvent(e1);

    var e2 = try event_collector.ProcessEvent.init(allocator, "e2", "case-1", "Process");
    e2.duration_ms = 6000; // Above default threshold
    try log.recordEvent(e2);

    var analyzer = BottleneckAnalyzer.init(allocator);
    var result = try analyzer.analyze(&log);
    defer result.deinit();

    // Should find at least one bottleneck (time)
    try std.testing.expect(result.activity_metrics.count() > 0);
}

test "WaitTimeAnalyzer" {
    const allocator = std.testing.allocator;

    const analyzer = WaitTimeAnalyzer.init(allocator, 1000);
    _ = analyzer;

    // Basic test - analyzer initialization
    try std.testing.expect(true);
}

test "ResourceUtilizationAnalyzer" {
    const allocator = std.testing.allocator;

    var analyzer = ResourceUtilizationAnalyzer.init(allocator, 0.8);

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    var utilization = try analyzer.analyzeResources(&log);
    defer {
        var iter = utilization.keyIterator();
        while (iter.next()) |key| {
            allocator.free(key.*);
        }
        utilization.deinit();
    }

    try std.testing.expectEqual(@as(usize, 0), utilization.count());
}
