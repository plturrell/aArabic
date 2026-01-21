//! KPI Calculator for Process Performance Metrics
//! Calculates key performance indicators for process analysis

const std = @import("std");
const Allocator = std.mem.Allocator;
const event_collector = @import("event_collector.zig");

// ============================================================================
// KPI Types
// ============================================================================

/// Types of KPIs
pub const KPIType = enum {
    CYCLE_TIME, // Total time from start to end
    THROUGHPUT, // Cases completed per time unit
    EFFICIENCY, // Value-added time / total time
    REWORK_RATE, // Percentage of rework
    SLA_COMPLIANCE, // Percentage meeting SLA
    FIRST_TIME_RIGHT, // Percentage completed without rework
    WAIT_TIME, // Average wait time
    PROCESSING_TIME, // Average processing time
    RESOURCE_UTILIZATION, // Resource usage percentage
    CASE_VOLUME, // Number of cases

    pub fn toString(self: KPIType) []const u8 {
        return switch (self) {
            .CYCLE_TIME => "Cycle Time",
            .THROUGHPUT => "Throughput",
            .EFFICIENCY => "Efficiency",
            .REWORK_RATE => "Rework Rate",
            .SLA_COMPLIANCE => "SLA Compliance",
            .FIRST_TIME_RIGHT => "First Time Right",
            .WAIT_TIME => "Wait Time",
            .PROCESSING_TIME => "Processing Time",
            .RESOURCE_UTILIZATION => "Resource Utilization",
            .CASE_VOLUME => "Case Volume",
        };
    }

    pub fn getUnit(self: KPIType) []const u8 {
        return switch (self) {
            .CYCLE_TIME, .WAIT_TIME, .PROCESSING_TIME => "ms",
            .THROUGHPUT => "cases/hour",
            .EFFICIENCY, .REWORK_RATE, .SLA_COMPLIANCE, .FIRST_TIME_RIGHT, .RESOURCE_UTILIZATION => "%",
            .CASE_VOLUME => "cases",
        };
    }
};

/// Trend direction
pub const Trend = enum {
    UP,
    DOWN,
    STABLE,

    pub fn toString(self: Trend) []const u8 {
        return switch (self) {
            .UP => "↑",
            .DOWN => "↓",
            .STABLE => "→",
        };
    }
};

// ============================================================================
// KPI Value
// ============================================================================

/// KPI value with metadata
pub const KPIValue = struct {
    kpi_type: KPIType,
    value: f64,
    unit: []const u8,
    trend: Trend = .STABLE,
    previous_value: ?f64 = null,
    target_value: ?f64 = null,
    timestamp: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, kpi_type: KPIType, value: f64) !KPIValue {
        return KPIValue{
            .kpi_type = kpi_type,
            .value = value,
            .unit = kpi_type.getUnit(),
            .timestamp = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KPIValue) void {
        _ = self;
        // No dynamic allocations to free
    }

    pub fn setTarget(self: *KPIValue, target: f64) void {
        self.target_value = target;
    }

    pub fn setPreviousValue(self: *KPIValue, prev: f64) void {
        self.previous_value = prev;
        // Calculate trend
        const diff = self.value - prev;
        const threshold = prev * 0.05; // 5% threshold
        if (diff > threshold) {
            self.trend = .UP;
        } else if (diff < -threshold) {
            self.trend = .DOWN;
        } else {
            self.trend = .STABLE;
        }
    }

    pub fn isOnTarget(self: *const KPIValue) bool {
        if (self.target_value) |target| {
            return switch (self.kpi_type) {
                // Lower is better
                .CYCLE_TIME, .WAIT_TIME, .REWORK_RATE => self.value <= target,
                // Higher is better
                else => self.value >= target,
            };
        }
        return true;
    }

    pub fn getPercentageOfTarget(self: *const KPIValue) ?f64 {
        if (self.target_value) |target| {
            if (target == 0) return null;
            return (self.value / target) * 100.0;
        }
        return null;
    }
};

// ============================================================================
// KPI Definition
// ============================================================================

/// KPI definition with calculation parameters
pub const KPIDefinition = struct {
    kpi_type: KPIType,
    name: []const u8,
    description: []const u8,
    target_value: ?f64 = null,
    warning_threshold: ?f64 = null,
    critical_threshold: ?f64 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, kpi_type: KPIType, name: []const u8, description: []const u8) !KPIDefinition {
        return KPIDefinition{
            .kpi_type = kpi_type,
            .name = try allocator.dupe(u8, name),
            .description = try allocator.dupe(u8, description),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KPIDefinition) void {
        self.allocator.free(self.name);
        self.allocator.free(self.description);
    }

    pub fn setTarget(self: *KPIDefinition, target: f64) void {
        self.target_value = target;
    }

    pub fn setThresholds(self: *KPIDefinition, warning: f64, critical: f64) void {
        self.warning_threshold = warning;
        self.critical_threshold = critical;
    }
};

// ============================================================================
// Process KPIs
// ============================================================================

/// Aggregated process KPIs
pub const ProcessKPIs = struct {
    cycle_time: KPIValue,
    throughput: KPIValue,
    efficiency: KPIValue,
    rework_rate: KPIValue,
    sla_compliance: KPIValue,
    first_time_right: KPIValue,
    avg_wait_time: KPIValue,
    avg_processing_time: KPIValue,
    case_volume: KPIValue,
    calculation_timestamp: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !ProcessKPIs {
        return ProcessKPIs{
            .cycle_time = try KPIValue.init(allocator, .CYCLE_TIME, 0),
            .throughput = try KPIValue.init(allocator, .THROUGHPUT, 0),
            .efficiency = try KPIValue.init(allocator, .EFFICIENCY, 0),
            .rework_rate = try KPIValue.init(allocator, .REWORK_RATE, 0),
            .sla_compliance = try KPIValue.init(allocator, .SLA_COMPLIANCE, 0),
            .first_time_right = try KPIValue.init(allocator, .FIRST_TIME_RIGHT, 0),
            .avg_wait_time = try KPIValue.init(allocator, .WAIT_TIME, 0),
            .avg_processing_time = try KPIValue.init(allocator, .PROCESSING_TIME, 0),
            .case_volume = try KPIValue.init(allocator, .CASE_VOLUME, 0),
            .calculation_timestamp = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessKPIs) void {
        self.cycle_time.deinit();
        self.throughput.deinit();
        self.efficiency.deinit();
        self.rework_rate.deinit();
        self.sla_compliance.deinit();
        self.first_time_right.deinit();
        self.avg_wait_time.deinit();
        self.avg_processing_time.deinit();
        self.case_volume.deinit();
    }

    pub fn getKPI(self: *const ProcessKPIs, kpi_type: KPIType) KPIValue {
        return switch (kpi_type) {
            .CYCLE_TIME => self.cycle_time,
            .THROUGHPUT => self.throughput,
            .EFFICIENCY => self.efficiency,
            .REWORK_RATE => self.rework_rate,
            .SLA_COMPLIANCE => self.sla_compliance,
            .FIRST_TIME_RIGHT => self.first_time_right,
            .WAIT_TIME => self.avg_wait_time,
            .PROCESSING_TIME => self.avg_processing_time,
            .CASE_VOLUME => self.case_volume,
            .RESOURCE_UTILIZATION => self.efficiency, // Fallback
        };
    }

    pub fn getAllKPIs(self: *const ProcessKPIs, allocator: Allocator) ![]KPIValue {
        var result = std.ArrayList(KPIValue){};
        errdefer result.deinit(allocator);

        try result.append(allocator, self.cycle_time);
        try result.append(allocator, self.throughput);
        try result.append(allocator, self.efficiency);
        try result.append(allocator, self.rework_rate);
        try result.append(allocator, self.sla_compliance);
        try result.append(allocator, self.first_time_right);
        try result.append(allocator, self.avg_wait_time);
        try result.append(allocator, self.avg_processing_time);
        try result.append(allocator, self.case_volume);

        return result.toOwnedSlice(allocator);
    }
};

// ============================================================================
// KPI Calculator
// ============================================================================

/// KPI Calculator
pub const KPICalculator = struct {
    allocator: Allocator,
    sla_threshold_ms: u64,

    pub fn init(allocator: Allocator) KPICalculator {
        return KPICalculator{
            .allocator = allocator,
            .sla_threshold_ms = 86400000, // 24 hours default
        };
    }

    pub fn setSLAThreshold(self: *KPICalculator, threshold_ms: u64) void {
        self.sla_threshold_ms = threshold_ms;
    }

    /// Calculate a specific KPI
    pub fn calculate(self: *KPICalculator, kpi_type: KPIType, log: *const event_collector.EventLog) !KPIValue {
        const value = switch (kpi_type) {
            .CYCLE_TIME => self.calculateCycleTime(log),
            .THROUGHPUT => self.calculateThroughput(log),
            .EFFICIENCY => self.calculateEfficiency(log),
            .REWORK_RATE => self.calculateReworkRate(log),
            .SLA_COMPLIANCE => self.calculateSLACompliance(log),
            .FIRST_TIME_RIGHT => self.calculateFirstTimeRight(log),
            .WAIT_TIME => self.calculateAvgWaitTime(log),
            .PROCESSING_TIME => self.calculateAvgProcessingTime(log),
            .CASE_VOLUME => self.calculateCaseVolume(log),
            .RESOURCE_UTILIZATION => self.calculateResourceUtilization(log),
        };

        return KPIValue.init(self.allocator, kpi_type, value);
    }

    /// Calculate all KPIs
    pub fn calculateAll(self: *KPICalculator, log: *const event_collector.EventLog) !ProcessKPIs {
        var kpis = try ProcessKPIs.init(self.allocator);

        kpis.cycle_time.value = self.calculateCycleTime(log);
        kpis.throughput.value = self.calculateThroughput(log);
        kpis.efficiency.value = self.calculateEfficiency(log);
        kpis.rework_rate.value = self.calculateReworkRate(log);
        kpis.sla_compliance.value = self.calculateSLACompliance(log);
        kpis.first_time_right.value = self.calculateFirstTimeRight(log);
        kpis.avg_wait_time.value = self.calculateAvgWaitTime(log);
        kpis.avg_processing_time.value = self.calculateAvgProcessingTime(log);
        kpis.case_volume.value = self.calculateCaseVolume(log);

        return kpis;
    }

    fn calculateCycleTime(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var total_duration: u64 = 0;
        var completed_count: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            if (trace.*.getDuration()) |dur| {
                total_duration += dur;
                completed_count += 1;
            }
        }

        if (completed_count == 0) return 0.0;
        return @as(f64, @floatFromInt(total_duration)) / @as(f64, @floatFromInt(completed_count));
    }

    fn calculateThroughput(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var completed_count: u64 = 0;
        var min_time: i64 = std.math.maxInt(i64);
        var max_time: i64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            if (trace.*.end_time) |end| {
                completed_count += 1;
                const start = trace.*.start_time;
                if (start < min_time) min_time = start;
                if (end > max_time) max_time = end;
            }
        }

        if (completed_count == 0 or max_time <= min_time) return 0.0;

        const duration_hours = @as(f64, @floatFromInt(max_time - min_time)) / 3600.0;
        if (duration_hours <= 0) return 0.0;

        return @as(f64, @floatFromInt(completed_count)) / duration_hours;
    }

    fn calculateEfficiency(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var total_processing: u64 = 0;
        var total_duration: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            // Sum up processing time from events
            for (trace.*.events.items) |event| {
                if (event.duration_ms) |dur| {
                    total_processing += dur;
                }
            }

            if (trace.*.getDuration()) |dur| {
                total_duration += dur;
            }
        }

        if (total_duration == 0) return 0.0;
        return (@as(f64, @floatFromInt(total_processing)) / @as(f64, @floatFromInt(total_duration))) * 100.0;
    }

    fn calculateReworkRate(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        var total_activities: u64 = 0;
        var rework_count: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            var seen = std.StringHashMap(void).init(self.allocator);
            defer seen.deinit();

            for (trace.*.events.items) |event| {
                total_activities += 1;
                if (seen.contains(event.activity)) {
                    rework_count += 1;
                } else {
                    seen.put(event.activity, {}) catch {};
                }
            }
        }

        if (total_activities == 0) return 0.0;
        return (@as(f64, @floatFromInt(rework_count)) / @as(f64, @floatFromInt(total_activities))) * 100.0;
    }

    fn calculateSLACompliance(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        var total_cases: u64 = 0;
        var compliant_cases: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            if (trace.*.getDuration()) |dur| {
                total_cases += 1;
                if (dur <= self.sla_threshold_ms) {
                    compliant_cases += 1;
                }
            }
        }

        if (total_cases == 0) return 100.0; // No cases = 100% compliant
        return (@as(f64, @floatFromInt(compliant_cases)) / @as(f64, @floatFromInt(total_cases))) * 100.0;
    }

    fn calculateFirstTimeRight(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        var total_cases: u64 = 0;
        var ftr_cases: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            total_cases += 1;

            var seen = std.StringHashMap(void).init(self.allocator);
            defer seen.deinit();

            var has_rework = false;
            for (trace.*.events.items) |event| {
                if (seen.contains(event.activity)) {
                    has_rework = true;
                    break;
                }
                seen.put(event.activity, {}) catch {};
            }

            if (!has_rework) {
                ftr_cases += 1;
            }
        }

        if (total_cases == 0) return 100.0;
        return (@as(f64, @floatFromInt(ftr_cases)) / @as(f64, @floatFromInt(total_cases))) * 100.0;
    }

    fn calculateAvgWaitTime(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var total_wait: u64 = 0;
        var wait_count: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            const events = trace.*.events.items;
            var i: usize = 1;
            while (i < events.len) : (i += 1) {
                const time_diff = events[i].timestamp - events[i - 1].timestamp;
                if (time_diff > 0) {
                    total_wait += @intCast(time_diff * 1000);
                    wait_count += 1;
                }
            }
        }

        if (wait_count == 0) return 0.0;
        return @as(f64, @floatFromInt(total_wait)) / @as(f64, @floatFromInt(wait_count));
    }

    fn calculateAvgProcessingTime(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var total_processing: u64 = 0;
        var event_count: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            for (trace.*.events.items) |event| {
                if (event.duration_ms) |dur| {
                    total_processing += dur;
                    event_count += 1;
                }
            }
        }

        if (event_count == 0) return 0.0;
        return @as(f64, @floatFromInt(total_processing)) / @as(f64, @floatFromInt(event_count));
    }

    fn calculateCaseVolume(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        return @as(f64, @floatFromInt(log.traces.count()));
    }

    fn calculateResourceUtilization(self: *KPICalculator, log: *const event_collector.EventLog) f64 {
        _ = self;
        var resource_events: u64 = 0;
        var total_events: u64 = 0;

        var iter = log.traces.valueIterator();
        while (iter.next()) |trace| {
            for (trace.*.events.items) |event| {
                total_events += 1;
                if (event.resource != null) {
                    resource_events += 1;
                }
            }
        }

        if (total_events == 0) return 0.0;
        return (@as(f64, @floatFromInt(resource_events)) / @as(f64, @floatFromInt(total_events))) * 100.0;
    }
};

// ============================================================================
// KPI Dashboard
// ============================================================================

/// KPI Dashboard for aggregated views
pub const KPIDashboard = struct {
    kpis: ProcessKPIs,
    definitions: std.ArrayList(KPIDefinition),
    history: std.ArrayList(ProcessKPIs),
    max_history: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !KPIDashboard {
        return KPIDashboard{
            .kpis = try ProcessKPIs.init(allocator),
            .definitions = std.ArrayList(KPIDefinition){},
            .history = std.ArrayList(ProcessKPIs){},
            .max_history = 100,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *KPIDashboard) void {
        self.kpis.deinit();

        for (self.definitions.items) |*def| def.deinit();
        self.definitions.deinit(self.allocator);

        for (self.history.items) |*h| h.deinit();
        self.history.deinit(self.allocator);
    }

    pub fn addDefinition(self: *KPIDashboard, definition: KPIDefinition) !void {
        try self.definitions.append(self.allocator, definition);
    }

    pub fn update(self: *KPIDashboard, log: *const event_collector.EventLog) !void {
        var calculator = KPICalculator.init(self.allocator);

        // Store previous KPIs in history
        if (self.history.items.len >= self.max_history) {
            var old = self.history.orderedRemove(0);
            old.deinit();
        }

        // Calculate new KPIs
        var new_kpis = try calculator.calculateAll(log);

        // Set previous values for trend calculation
        new_kpis.cycle_time.setPreviousValue(self.kpis.cycle_time.value);
        new_kpis.throughput.setPreviousValue(self.kpis.throughput.value);
        new_kpis.efficiency.setPreviousValue(self.kpis.efficiency.value);
        new_kpis.rework_rate.setPreviousValue(self.kpis.rework_rate.value);
        new_kpis.sla_compliance.setPreviousValue(self.kpis.sla_compliance.value);

        // Store old KPIs in history
        try self.history.append(self.allocator, self.kpis);

        // Update current KPIs
        self.kpis = new_kpis;
    }

    pub fn getKPI(self: *const KPIDashboard, kpi_type: KPIType) KPIValue {
        return self.kpis.getKPI(kpi_type);
    }

    pub fn getHealthScore(self: *const KPIDashboard) f64 {
        var score: f64 = 0.0;
        var count: f64 = 0.0;

        // Weight different KPIs
        score += self.kpis.sla_compliance.value * 0.3;
        count += 0.3;

        score += self.kpis.first_time_right.value * 0.2;
        count += 0.2;

        score += self.kpis.efficiency.value * 0.2;
        count += 0.2;

        score += (100.0 - self.kpis.rework_rate.value) * 0.15;
        count += 0.15;

        // Throughput contribution (normalized)
        if (self.kpis.throughput.value > 0) {
            score += 15.0; // Bonus for having throughput
            count += 0.15;
        }

        if (count == 0) return 0.0;
        return score / count;
    }

    pub fn getSummary(self: *const KPIDashboard, allocator: Allocator) ![]u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);

        const writer = buffer.writer(allocator);

        try writer.print("=== KPI Dashboard Summary ===\n", .{});
        try writer.print("Cycle Time: {d:.2} ms\n", .{self.kpis.cycle_time.value});
        try writer.print("Throughput: {d:.2} cases/hour\n", .{self.kpis.throughput.value});
        try writer.print("Efficiency: {d:.2}%\n", .{self.kpis.efficiency.value});
        try writer.print("SLA Compliance: {d:.2}%\n", .{self.kpis.sla_compliance.value});
        try writer.print("First Time Right: {d:.2}%\n", .{self.kpis.first_time_right.value});
        try writer.print("Rework Rate: {d:.2}%\n", .{self.kpis.rework_rate.value});
        try writer.print("Health Score: {d:.2}%\n", .{self.getHealthScore()});

        return buffer.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "KPIType toString and getUnit" {
    try std.testing.expectEqualStrings("Cycle Time", KPIType.CYCLE_TIME.toString());
    try std.testing.expectEqualStrings("ms", KPIType.CYCLE_TIME.getUnit());
    try std.testing.expectEqualStrings("%", KPIType.EFFICIENCY.getUnit());
}

test "Trend toString" {
    try std.testing.expectEqualStrings("↑", Trend.UP.toString());
    try std.testing.expectEqualStrings("↓", Trend.DOWN.toString());
    try std.testing.expectEqualStrings("→", Trend.STABLE.toString());
}

test "KPIValue basic" {
    const allocator = std.testing.allocator;

    var kpi = try KPIValue.init(allocator, .CYCLE_TIME, 1000.0);
    defer kpi.deinit();

    try std.testing.expectEqual(KPIType.CYCLE_TIME, kpi.kpi_type);
    try std.testing.expect(kpi.value == 1000.0);
}

test "KPIValue target" {
    const allocator = std.testing.allocator;

    var kpi = try KPIValue.init(allocator, .CYCLE_TIME, 800.0);
    defer kpi.deinit();

    kpi.setTarget(1000.0);
    try std.testing.expect(kpi.isOnTarget()); // 800 <= 1000 for cycle time
}

test "KPIValue trend calculation" {
    const allocator = std.testing.allocator;

    var kpi = try KPIValue.init(allocator, .THROUGHPUT, 100.0);
    defer kpi.deinit();

    kpi.setPreviousValue(80.0);
    try std.testing.expectEqual(Trend.UP, kpi.trend);

    kpi.setPreviousValue(120.0);
    try std.testing.expectEqual(Trend.DOWN, kpi.trend);
}

test "KPIDefinition" {
    const allocator = std.testing.allocator;

    var def = try KPIDefinition.init(allocator, .CYCLE_TIME, "Avg Cycle Time", "Average time to complete a case");
    defer def.deinit();

    def.setTarget(5000.0);
    try std.testing.expect(def.target_value.? == 5000.0);
}

test "ProcessKPIs init" {
    const allocator = std.testing.allocator;

    var kpis = try ProcessKPIs.init(allocator);
    defer kpis.deinit();

    try std.testing.expect(kpis.cycle_time.value == 0.0);
    try std.testing.expect(kpis.throughput.value == 0.0);
}

test "KPICalculator empty log" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    var calculator = KPICalculator.init(allocator);
    var kpis = try calculator.calculateAll(&log);
    defer kpis.deinit();

    try std.testing.expect(kpis.case_volume.value == 0.0);
}

test "KPICalculator with events" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    var e1 = try event_collector.ProcessEvent.init(allocator, "e1", "case-1", "Start");
    e1.duration_ms = 100;
    try log.recordEvent(e1);

    var e2 = try event_collector.ProcessEvent.init(allocator, "e2", "case-1", "End");
    e2.duration_ms = 200;
    try log.recordEvent(e2);

    var calculator = KPICalculator.init(allocator);
    var kpis = try calculator.calculateAll(&log);
    defer kpis.deinit();

    try std.testing.expect(kpis.case_volume.value == 1.0);
}

test "KPIDashboard init" {
    const allocator = std.testing.allocator;

    var dashboard = try KPIDashboard.init(allocator);
    defer dashboard.deinit();

    try std.testing.expect(dashboard.kpis.cycle_time.value == 0.0);
}

test "KPIDashboard update" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    const e1 = try event_collector.ProcessEvent.init(allocator, "e1", "case-1", "Activity");
    try log.recordEvent(e1);

    var dashboard = try KPIDashboard.init(allocator);
    defer dashboard.deinit();

    try dashboard.update(&log);

    try std.testing.expect(dashboard.kpis.case_volume.value == 1.0);
}

test "KPIDashboard health score" {
    const allocator = std.testing.allocator;

    var dashboard = try KPIDashboard.init(allocator);
    defer dashboard.deinit();

    // Set some values
    dashboard.kpis.sla_compliance.value = 95.0;
    dashboard.kpis.first_time_right.value = 90.0;
    dashboard.kpis.efficiency.value = 80.0;
    dashboard.kpis.rework_rate.value = 5.0;
    dashboard.kpis.throughput.value = 10.0;

    const health = dashboard.getHealthScore();
    try std.testing.expect(health > 0.0);
}

