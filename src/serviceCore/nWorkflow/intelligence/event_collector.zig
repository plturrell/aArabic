//! Event Log Collector for Process Mining
//! Collects XES-format event logs for process analysis

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Event Lifecycle States (XES Standard)
// ============================================================================

/// Event lifecycle states following XES standard
pub const EventLifecycle = enum {
    START,
    COMPLETE,
    ASSIGN,
    REASSIGN,
    SUSPEND,
    RESUME,
    ABORT,

    pub fn toString(self: EventLifecycle) []const u8 {
        return @tagName(self);
    }
};

// ============================================================================
// Process Event
// ============================================================================

/// Process event for mining
pub const ProcessEvent = struct {
    event_id: []const u8,
    case_id: []const u8, // Process instance ID
    activity: []const u8, // Activity/task name
    timestamp: i64,
    lifecycle: EventLifecycle = .COMPLETE,
    resource: ?[]const u8 = null, // User/system that performed
    cost: ?f64 = null,
    duration_ms: ?u64 = null,
    attributes: ?[]const u8 = null, // JSON additional attributes
    allocator: Allocator,

    pub fn init(allocator: Allocator, event_id: []const u8, case_id: []const u8, activity: []const u8) !ProcessEvent {
        return ProcessEvent{
            .event_id = try allocator.dupe(u8, event_id),
            .case_id = try allocator.dupe(u8, case_id),
            .activity = try allocator.dupe(u8, activity),
            .timestamp = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn initWithTimestamp(allocator: Allocator, event_id: []const u8, case_id: []const u8, activity: []const u8, timestamp: i64) !ProcessEvent {
        return ProcessEvent{
            .event_id = try allocator.dupe(u8, event_id),
            .case_id = try allocator.dupe(u8, case_id),
            .activity = try allocator.dupe(u8, activity),
            .timestamp = timestamp,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessEvent) void {
        self.allocator.free(self.event_id);
        self.allocator.free(self.case_id);
        self.allocator.free(self.activity);
        if (self.resource) |r| self.allocator.free(r);
        if (self.attributes) |a| self.allocator.free(a);
    }

    pub fn setResource(self: *ProcessEvent, resource: []const u8) !void {
        if (self.resource) |r| self.allocator.free(r);
        self.resource = try self.allocator.dupe(u8, resource);
    }

    pub fn setAttributes(self: *ProcessEvent, attributes: []const u8) !void {
        if (self.attributes) |a| self.allocator.free(a);
        self.attributes = try self.allocator.dupe(u8, attributes);
    }

    pub fn toXes(self: *const ProcessEvent, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.print(
            \\<event>
            \\  <string key="concept:name" value="{s}"/>
            \\  <string key="lifecycle:transition" value="{s}"/>
            \\  <date key="time:timestamp" value="{d}"/>
        , .{
            self.activity,
            self.lifecycle.toString(),
            self.timestamp,
        });

        if (self.resource) |res| {
            try writer.print(
                \\  <string key="org:resource" value="{s}"/>
            , .{res});
        }

        if (self.cost) |c| {
            try writer.print(
                \\  <float key="cost:total" value="{d:.2}"/>
            , .{c});
        }

        try writer.writeAll("</event>\n");
        return buffer.toOwnedSlice(allocator);
    }

    pub fn clone(self: *const ProcessEvent, allocator: Allocator) !ProcessEvent {
        var cloned = ProcessEvent{
            .event_id = try allocator.dupe(u8, self.event_id),
            .case_id = try allocator.dupe(u8, self.case_id),
            .activity = try allocator.dupe(u8, self.activity),
            .timestamp = self.timestamp,
            .lifecycle = self.lifecycle,
            .resource = null,
            .cost = self.cost,
            .duration_ms = self.duration_ms,
            .attributes = null,
            .allocator = allocator,
        };
        if (self.resource) |r| cloned.resource = try allocator.dupe(u8, r);
        if (self.attributes) |a| cloned.attributes = try allocator.dupe(u8, a);
        return cloned;
    }
};

// ============================================================================
// Process Trace
// ============================================================================

/// Process trace (sequence of events for one case)
pub const ProcessTrace = struct {
    case_id: []const u8,
    events: std.ArrayList(ProcessEvent),
    start_time: i64,
    end_time: ?i64 = null,
    is_complete: bool = false,
    variant_id: ?u64 = null,
    attributes: ?[]const u8 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, case_id: []const u8) !ProcessTrace {
        return ProcessTrace{
            .case_id = try allocator.dupe(u8, case_id),
            .events = std.ArrayList(ProcessEvent){},
            .start_time = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn initWithTimestamp(allocator: Allocator, case_id: []const u8, start_time: i64) !ProcessTrace {
        return ProcessTrace{
            .case_id = try allocator.dupe(u8, case_id),
            .events = std.ArrayList(ProcessEvent){},
            .start_time = start_time,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessTrace) void {
        for (self.events.items) |*event| {
            event.deinit();
        }
        self.events.deinit(self.allocator);
        self.allocator.free(self.case_id);
        if (self.attributes) |a| self.allocator.free(a);
    }

    pub fn addEvent(self: *ProcessTrace, event: ProcessEvent) !void {
        try self.events.append(self.allocator, event);
    }

    pub fn complete(self: *ProcessTrace) void {
        self.end_time = std.time.timestamp();
        self.is_complete = true;
    }

    pub fn completeWithTimestamp(self: *ProcessTrace, end_time: i64) void {
        self.end_time = end_time;
        self.is_complete = true;
    }

    pub fn getDuration(self: *const ProcessTrace) ?u64 {
        if (self.end_time) |end| {
            if (end >= self.start_time) {
                return @intCast(end - self.start_time);
            }
        }
        return null;
    }

    pub fn getActivitySequence(self: *const ProcessTrace, allocator: Allocator) ![][]const u8 {
        var result = std.ArrayList([]const u8){};
        errdefer result.deinit(allocator);

        for (self.events.items) |event| {
            try result.append(allocator, event.activity);
        }
        return result.toOwnedSlice(allocator);
    }

    pub fn getEventCount(self: *const ProcessTrace) usize {
        return self.events.items.len;
    }

    pub fn hasActivity(self: *const ProcessTrace, activity: []const u8) bool {
        for (self.events.items) |event| {
            if (std.mem.eql(u8, event.activity, activity)) return true;
        }
        return false;
    }

    pub fn getVariantHash(self: *const ProcessTrace) u64 {
        var hasher = std.hash.Wyhash.init(0);
        for (self.events.items) |event| {
            hasher.update(event.activity);
            hasher.update("|");
        }
        return hasher.final();
    }
};

// ============================================================================
// Event Log
// ============================================================================

/// Event Log (collection of traces)
pub const EventLog = struct {
    name: []const u8,
    traces: std.StringHashMap(*ProcessTrace),
    total_events: u64 = 0,
    activities: std.StringHashMap(u64), // Activity -> count
    resources: std.StringHashMap(u64), // Resource -> count
    start_time: ?i64 = null,
    end_time: ?i64 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8) !EventLog {
        return EventLog{
            .name = try allocator.dupe(u8, name),
            .traces = std.StringHashMap(*ProcessTrace).init(allocator),
            .activities = std.StringHashMap(u64).init(allocator),
            .resources = std.StringHashMap(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *EventLog) void {
        var iter = self.traces.valueIterator();
        while (iter.next()) |trace_ptr| {
            trace_ptr.*.deinit();
            self.allocator.destroy(trace_ptr.*);
        }
        self.traces.deinit();

        // Free activity keys
        var act_iter = self.activities.keyIterator();
        while (act_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.activities.deinit();

        // Free resource keys
        var res_iter = self.resources.keyIterator();
        while (res_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.resources.deinit();
        self.allocator.free(self.name);
    }

    pub fn recordEvent(self: *EventLog, event: ProcessEvent) !void {
        // Get or create trace
        var trace = self.traces.get(event.case_id) orelse blk: {
            const new_trace = try self.allocator.create(ProcessTrace);
            new_trace.* = try ProcessTrace.init(self.allocator, event.case_id);
            try self.traces.put(new_trace.case_id, new_trace);
            break :blk new_trace;
        };

        try trace.addEvent(event);
        self.total_events += 1;

        // Track activity counts - use existing key if found
        if (self.activities.getEntry(event.activity)) |entry| {
            entry.value_ptr.* += 1;
        } else {
            const act_key = try self.allocator.dupe(u8, event.activity);
            try self.activities.put(act_key, 1);
        }

        // Track resource counts
        if (event.resource) |res| {
            if (self.resources.getEntry(res)) |entry| {
                entry.value_ptr.* += 1;
            } else {
                const res_key = try self.allocator.dupe(u8, res);
                try self.resources.put(res_key, 1);
            }
        }

        // Update time bounds
        if (self.start_time == null or event.timestamp < self.start_time.?) {
            self.start_time = event.timestamp;
        }
        if (self.end_time == null or event.timestamp > self.end_time.?) {
            self.end_time = event.timestamp;
        }
    }

    pub fn getTraceCount(self: *const EventLog) usize {
        return self.traces.count();
    }

    pub fn getActivityCount(self: *const EventLog) usize {
        return self.activities.count();
    }

    pub fn getResourceCount(self: *const EventLog) usize {
        return self.resources.count();
    }

    pub fn getTrace(self: *const EventLog, case_id: []const u8) ?*ProcessTrace {
        return self.traces.get(case_id);
    }

    pub fn getActivityFrequency(self: *const EventLog, activity: []const u8) u64 {
        return self.activities.get(activity) orelse 0;
    }

    pub fn getVariantCount(self: *EventLog) usize {
        var variants = std.AutoHashMap(u64, void).init(self.allocator);
        defer variants.deinit();

        var iter = self.traces.valueIterator();
        while (iter.next()) |trace| {
            const hash = trace.*.getVariantHash();
            variants.put(hash, {}) catch {};
        }
        return variants.count();
    }

    pub fn toXes(self: *const EventLog, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        errdefer buffer.deinit(allocator);
        const writer = buffer.writer(allocator);

        try writer.print(
            \\<?xml version="1.0" encoding="UTF-8"?>
            \\<log xes.version="1.0" xes.features="nested-attributes">
            \\  <string key="concept:name" value="{s}"/>
            \\
        , .{self.name});

        var trace_iter = self.traces.valueIterator();
        while (trace_iter.next()) |trace| {
            try writer.print(
                \\  <trace>
                \\    <string key="concept:name" value="{s}"/>
                \\
            , .{trace.*.case_id});

            for (trace.*.events.items) |event| {
                const event_xes = try event.toXes(allocator);
                defer allocator.free(event_xes);
                try writer.print("    {s}", .{event_xes});
            }

            try writer.writeAll("  </trace>\n");
        }

        try writer.writeAll("</log>");
        return buffer.toOwnedSlice(allocator);
    }

    pub fn getStatistics(self: *const EventLog) LogStatistics {
        var total_trace_duration: u64 = 0;
        var completed_traces: u64 = 0;

        var iter = self.traces.valueIterator();
        while (iter.next()) |trace| {
            if (trace.*.getDuration()) |dur| {
                total_trace_duration += dur;
                completed_traces += 1;
            }
        }

        const avg_duration = if (completed_traces > 0)
            total_trace_duration / completed_traces
        else
            0;

        return LogStatistics{
            .trace_count = self.traces.count(),
            .event_count = self.total_events,
            .activity_count = self.activities.count(),
            .resource_count = self.resources.count(),
            .avg_trace_duration = avg_duration,
            .completed_traces = completed_traces,
        };
    }
};

/// Log statistics summary
pub const LogStatistics = struct {
    trace_count: usize,
    event_count: u64,
    activity_count: usize,
    resource_count: usize,
    avg_trace_duration: u64,
    completed_traces: u64,
};

// ============================================================================
// Event Collector
// ============================================================================

/// Event Collector - manages multiple event logs
pub const EventCollector = struct {
    logs: std.StringHashMap(*EventLog),
    default_log_name: []const u8,
    is_collecting: bool = true,
    events_collected: u64 = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, default_log_name: []const u8) !EventCollector {
        var collector = EventCollector{
            .logs = std.StringHashMap(*EventLog).init(allocator),
            .default_log_name = try allocator.dupe(u8, default_log_name),
            .allocator = allocator,
        };

        // Create default log
        _ = try collector.createLog(default_log_name);

        return collector;
    }

    pub fn deinit(self: *EventCollector) void {
        var iter = self.logs.valueIterator();
        while (iter.next()) |log| {
            log.*.deinit();
            self.allocator.destroy(log.*);
        }
        self.logs.deinit();
        self.allocator.free(self.default_log_name);
    }

    pub fn createLog(self: *EventCollector, name: []const u8) !*EventLog {
        if (self.logs.get(name)) |existing| return existing;

        const log = try self.allocator.create(EventLog);
        log.* = try EventLog.init(self.allocator, name);
        try self.logs.put(log.name, log);
        return log;
    }

    pub fn getLog(self: *EventCollector, name: []const u8) ?*EventLog {
        return self.logs.get(name);
    }

    pub fn getDefaultLog(self: *EventCollector) ?*EventLog {
        return self.logs.get(self.default_log_name);
    }

    pub fn record(self: *EventCollector, log_name: []const u8, event: ProcessEvent) !void {
        if (!self.is_collecting) return error.CollectionStopped;
        const log = self.logs.get(log_name) orelse return error.LogNotFound;
        try log.recordEvent(event);
        self.events_collected += 1;
    }

    pub fn recordToDefault(self: *EventCollector, event: ProcessEvent) !void {
        try self.record(self.default_log_name, event);
    }

    pub fn pause(self: *EventCollector) void {
        self.is_collecting = false;
    }

    pub fn resume_collection(self: *EventCollector) void {
        self.is_collecting = true;
    }

    pub fn getLogCount(self: *const EventCollector) usize {
        return self.logs.count();
    }

    pub fn getTotalEventsCollected(self: *const EventCollector) u64 {
        return self.events_collected;
    }

    pub fn exportLog(self: *EventCollector, log_name: []const u8, allocator: Allocator) ![]const u8 {
        const log = self.logs.get(log_name) orelse return error.LogNotFound;
        return log.toXes(allocator);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ProcessEvent creation" {
    const allocator = std.testing.allocator;

    var event = try ProcessEvent.init(allocator, "evt-1", "case-1", "Submit Order");
    defer event.deinit();

    try std.testing.expectEqualStrings("case-1", event.case_id);
    try std.testing.expectEqualStrings("Submit Order", event.activity);
    try std.testing.expectEqual(EventLifecycle.COMPLETE, event.lifecycle);
}

test "ProcessEvent with resource" {
    const allocator = std.testing.allocator;

    var event = try ProcessEvent.init(allocator, "evt-1", "case-1", "Review");
    defer event.deinit();

    try event.setResource("user@example.com");
    try std.testing.expectEqualStrings("user@example.com", event.resource.?);
}

test "ProcessTrace operations" {
    const allocator = std.testing.allocator;

    var trace = try ProcessTrace.init(allocator, "case-1");
    defer trace.deinit();

    const event1 = try ProcessEvent.init(allocator, "e1", "case-1", "Start");
    try trace.addEvent(event1);

    const event2 = try ProcessEvent.init(allocator, "e2", "case-1", "Process");
    try trace.addEvent(event2);

    try std.testing.expectEqual(@as(usize, 2), trace.events.items.len);
    try std.testing.expect(trace.hasActivity("Start"));
    try std.testing.expect(trace.hasActivity("Process"));
    try std.testing.expect(!trace.hasActivity("End"));
}

test "ProcessTrace completion" {
    const allocator = std.testing.allocator;

    var trace = try ProcessTrace.initWithTimestamp(allocator, "case-1", 1000);
    defer trace.deinit();

    try std.testing.expect(!trace.is_complete);
    trace.completeWithTimestamp(2000);
    try std.testing.expect(trace.is_complete);
    try std.testing.expectEqual(@as(?u64, 1000), trace.getDuration());
}

test "ProcessTrace variant hash" {
    const allocator = std.testing.allocator;

    var trace1 = try ProcessTrace.init(allocator, "case-1");
    defer trace1.deinit();

    const event1a = try ProcessEvent.init(allocator, "e1", "case-1", "A");
    const event1b = try ProcessEvent.init(allocator, "e2", "case-1", "B");
    try trace1.addEvent(event1a);
    try trace1.addEvent(event1b);

    var trace2 = try ProcessTrace.init(allocator, "case-2");
    defer trace2.deinit();

    const event2a = try ProcessEvent.init(allocator, "e3", "case-2", "A");
    const event2b = try ProcessEvent.init(allocator, "e4", "case-2", "B");
    try trace2.addEvent(event2a);
    try trace2.addEvent(event2b);

    // Same activity sequence should have same hash
    try std.testing.expectEqual(trace1.getVariantHash(), trace2.getVariantHash());
}

test "EventLog recording" {
    const allocator = std.testing.allocator;

    var log = try EventLog.init(allocator, "test-log");
    defer log.deinit();

    const event = try ProcessEvent.init(allocator, "e1", "case-1", "Activity A");
    try log.recordEvent(event);

    try std.testing.expectEqual(@as(usize, 1), log.getTraceCount());
    try std.testing.expectEqual(@as(u64, 1), log.total_events);
}

test "EventLog multiple cases" {
    const allocator = std.testing.allocator;

    var log = try EventLog.init(allocator, "test-log");
    defer log.deinit();

    const e1 = try ProcessEvent.init(allocator, "e1", "case-1", "Start");
    try log.recordEvent(e1);

    const e2 = try ProcessEvent.init(allocator, "e2", "case-2", "Start");
    try log.recordEvent(e2);

    const e3 = try ProcessEvent.init(allocator, "e3", "case-1", "End");
    try log.recordEvent(e3);

    try std.testing.expectEqual(@as(usize, 2), log.getTraceCount());
    try std.testing.expectEqual(@as(u64, 3), log.total_events);
}

test "EventLog statistics" {
    const allocator = std.testing.allocator;

    var log = try EventLog.init(allocator, "test-log");
    defer log.deinit();

    const e1 = try ProcessEvent.init(allocator, "e1", "case-1", "A");
    const e2 = try ProcessEvent.init(allocator, "e2", "case-1", "B");
    try log.recordEvent(e1);
    try log.recordEvent(e2);

    const stats = log.getStatistics();
    try std.testing.expectEqual(@as(usize, 1), stats.trace_count);
    try std.testing.expectEqual(@as(u64, 2), stats.event_count);
}

test "EventCollector multi-log" {
    const allocator = std.testing.allocator;

    var collector = try EventCollector.init(allocator, "default");
    defer collector.deinit();

    _ = try collector.createLog("process-1");
    _ = try collector.createLog("process-2");

    try std.testing.expectEqual(@as(usize, 3), collector.logs.count()); // default + 2 created
}

test "EventCollector record to default" {
    const allocator = std.testing.allocator;

    var collector = try EventCollector.init(allocator, "main-log");
    defer collector.deinit();

    const event = try ProcessEvent.init(allocator, "e1", "case-1", "Test Activity");
    try collector.recordToDefault(event);

    try std.testing.expectEqual(@as(u64, 1), collector.events_collected);

    const log = collector.getDefaultLog().?;
    try std.testing.expectEqual(@as(usize, 1), log.getTraceCount());
}

test "EventCollector pause and resume" {
    const allocator = std.testing.allocator;

    var collector = try EventCollector.init(allocator, "log");
    defer collector.deinit();

    try std.testing.expect(collector.is_collecting);

    collector.pause();
    try std.testing.expect(!collector.is_collecting);

    var event = try ProcessEvent.init(allocator, "e1", "case-1", "Test");
    defer event.deinit();

    const result = collector.recordToDefault(event);
    try std.testing.expectError(error.CollectionStopped, result);

    collector.resume_collection();
    try std.testing.expect(collector.is_collecting);
}

test "EventLog XES export" {
    const allocator = std.testing.allocator;

    var log = try EventLog.init(allocator, "test-export");
    defer log.deinit();

    const event = try ProcessEvent.init(allocator, "e1", "case-1", "Submit");
    try log.recordEvent(event);

    const xes = try log.toXes(allocator);
    defer allocator.free(xes);

    try std.testing.expect(std.mem.indexOf(u8, xes, "xes.version") != null);
    try std.testing.expect(std.mem.indexOf(u8, xes, "concept:name") != null);
    try std.testing.expect(std.mem.indexOf(u8, xes, "Submit") != null);
}
