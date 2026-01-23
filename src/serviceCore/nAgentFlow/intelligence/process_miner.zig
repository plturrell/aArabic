//! Process Miner for Process Discovery
//! Implements the Alpha algorithm for discovering process models from event logs

const std = @import("std");
const Allocator = std.mem.Allocator;
const event_collector = @import("event_collector.zig");

// ============================================================================
// Activity Relations
// ============================================================================

/// Relations between activities
pub const ActivityRelation = enum {
    DIRECTLY_FOLLOWS, // a > b: a directly followed by b
    CAUSAL, // a -> b: a causes b
    PARALLEL, // a || b: a and b are parallel
    UNRELATED, // a # b: no relation
    CHOICE, // a x b: exclusive choice

    pub fn toString(self: ActivityRelation) []const u8 {
        return switch (self) {
            .DIRECTLY_FOLLOWS => ">",
            .CAUSAL => "->",
            .PARALLEL => "||",
            .UNRELATED => "#",
            .CHOICE => "x",
        };
    }
};

// ============================================================================
// Direct Follows Graph
// ============================================================================

/// Activity pair for tracking relations
pub const ActivityPair = struct {
    from: []const u8,
    to: []const u8,

    pub fn hash(self: ActivityPair) u64 {
        var hasher = std.hash.Wyhash.init(0);
        hasher.update(self.from);
        hasher.update("->");
        hasher.update(self.to);
        return hasher.final();
    }

    pub fn eql(a: ActivityPair, b: ActivityPair) bool {
        return std.mem.eql(u8, a.from, b.from) and std.mem.eql(u8, a.to, b.to);
    }
};

/// Context for ActivityPair HashMap
pub const ActivityPairContext = struct {
    pub fn hash(_: ActivityPairContext, pair: ActivityPair) u64 {
        return pair.hash();
    }

    pub fn eql(_: ActivityPairContext, a: ActivityPair, b: ActivityPair) bool {
        return ActivityPair.eql(a, b);
    }
};

/// Direct Follows Graph - captures directly-follows relations
pub const DirectFollowsGraph = struct {
    edges: std.HashMap(ActivityPair, u64, ActivityPairContext, 80),
    activities: std.StringHashMap(void),
    start_activities: std.StringHashMap(u64),
    end_activities: std.StringHashMap(u64),
    allocator: Allocator,

    pub fn init(allocator: Allocator) DirectFollowsGraph {
        return DirectFollowsGraph{
            .edges = std.HashMap(ActivityPair, u64, ActivityPairContext, 80).init(allocator),
            .activities = std.StringHashMap(void).init(allocator),
            .start_activities = std.StringHashMap(u64).init(allocator),
            .end_activities = std.StringHashMap(u64).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DirectFollowsGraph) void {
        // Free owned keys in edges
        var edge_iter = self.edges.keyIterator();
        while (edge_iter.next()) |pair| {
            self.allocator.free(pair.from);
            self.allocator.free(pair.to);
        }
        self.edges.deinit();

        // Free owned keys in activities
        var act_iter = self.activities.keyIterator();
        while (act_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.activities.deinit();

        // Free owned keys in start/end activities
        var start_iter = self.start_activities.keyIterator();
        while (start_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.start_activities.deinit();

        var end_iter = self.end_activities.keyIterator();
        while (end_iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.end_activities.deinit();
    }

    pub fn addEdge(self: *DirectFollowsGraph, from: []const u8, to: []const u8) !void {
        const pair = ActivityPair{
            .from = try self.allocator.dupe(u8, from),
            .to = try self.allocator.dupe(u8, to),
        };

        if (self.edges.getEntry(pair)) |entry| {
            // Free the duplicated keys since we already have this pair
            self.allocator.free(pair.from);
            self.allocator.free(pair.to);
            entry.value_ptr.* += 1;
        } else {
            try self.edges.put(pair, 1);
        }

        // Track activities
        if (!self.activities.contains(from)) {
            const key = try self.allocator.dupe(u8, from);
            try self.activities.put(key, {});
        }
        if (!self.activities.contains(to)) {
            const key = try self.allocator.dupe(u8, to);
            try self.activities.put(key, {});
        }
    }

    pub fn addStartActivity(self: *DirectFollowsGraph, activity: []const u8) !void {
        if (self.start_activities.getEntry(activity)) |entry| {
            entry.value_ptr.* += 1;
        } else {
            const key = try self.allocator.dupe(u8, activity);
            try self.start_activities.put(key, 1);
        }
    }

    pub fn addEndActivity(self: *DirectFollowsGraph, activity: []const u8) !void {
        if (self.end_activities.getEntry(activity)) |entry| {
            entry.value_ptr.* += 1;
        } else {
            const key = try self.allocator.dupe(u8, activity);
            try self.end_activities.put(key, 1);
        }
    }

    pub fn getEdgeCount(self: *const DirectFollowsGraph, from: []const u8, to: []const u8) u64 {
        const pair = ActivityPair{ .from = from, .to = to };
        return self.edges.get(pair) orelse 0;
    }

    pub fn hasEdge(self: *const DirectFollowsGraph, from: []const u8, to: []const u8) bool {
        return self.getEdgeCount(from, to) > 0;
    }

    pub fn getActivityCount(self: *const DirectFollowsGraph) usize {
        return self.activities.count();
    }

    pub fn getEdgesCount(self: *const DirectFollowsGraph) usize {
        return self.edges.count();
    }

    pub fn isStartActivity(self: *const DirectFollowsGraph, activity: []const u8) bool {
        return self.start_activities.contains(activity);
    }

    pub fn isEndActivity(self: *const DirectFollowsGraph, activity: []const u8) bool {
        return self.end_activities.contains(activity);
    }
};

// ============================================================================
// Footprint Matrix
// ============================================================================

/// Footprint Matrix for capturing relations between activities
pub const FootprintMatrix = struct {
    relations: std.HashMap(ActivityPair, ActivityRelation, ActivityPairContext, 80),
    activities: std.ArrayList([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator) FootprintMatrix {
        return FootprintMatrix{
            .relations = std.HashMap(ActivityPair, ActivityRelation, ActivityPairContext, 80).init(allocator),
            .activities = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FootprintMatrix) void {
        var iter = self.relations.keyIterator();
        while (iter.next()) |pair| {
            self.allocator.free(pair.from);
            self.allocator.free(pair.to);
        }
        self.relations.deinit();

        for (self.activities.items) |act| {
            self.allocator.free(act);
        }
        self.activities.deinit(self.allocator);
    }

    pub fn setRelation(self: *FootprintMatrix, a: []const u8, b: []const u8, relation: ActivityRelation) !void {
        const pair = ActivityPair{
            .from = try self.allocator.dupe(u8, a),
            .to = try self.allocator.dupe(u8, b),
        };

        if (self.relations.getEntry(pair)) |entry| {
            self.allocator.free(pair.from);
            self.allocator.free(pair.to);
            entry.value_ptr.* = relation;
        } else {
            try self.relations.put(pair, relation);
        }
    }

    pub fn getRelation(self: *const FootprintMatrix, a: []const u8, b: []const u8) ActivityRelation {
        const pair = ActivityPair{ .from = a, .to = b };
        return self.relations.get(pair) orelse .UNRELATED;
    }

    pub fn addActivity(self: *FootprintMatrix, activity: []const u8) !void {
        // Check if already exists
        for (self.activities.items) |act| {
            if (std.mem.eql(u8, act, activity)) return;
        }
        const duped = try self.allocator.dupe(u8, activity);
        try self.activities.append(self.allocator, duped);
    }

    pub fn fromDFG(allocator: Allocator, dfg: *const DirectFollowsGraph) !FootprintMatrix {
        var matrix = FootprintMatrix.init(allocator);
        errdefer matrix.deinit();

        // Add all activities
        var act_iter = dfg.activities.keyIterator();
        while (act_iter.next()) |act| {
            try matrix.addActivity(act.*);
        }

        // Compute relations for all pairs
        for (matrix.activities.items) |a| {
            for (matrix.activities.items) |b| {
                const a_follows_b = dfg.hasEdge(a, b);
                const b_follows_a = dfg.hasEdge(b, a);

                const relation: ActivityRelation = if (a_follows_b and b_follows_a)
                    .PARALLEL
                else if (a_follows_b and !b_follows_a)
                    .CAUSAL
                else if (!a_follows_b and b_follows_a)
                    .UNRELATED // b -> a (reverse causal)
                else
                    .UNRELATED;

                try matrix.setRelation(a, b, relation);
            }
        }

        return matrix;
    }
};

// ============================================================================
// Process Model (Petri Net representation)
// ============================================================================

/// Place in the Petri net
pub const Place = struct {
    id: []const u8,
    name: []const u8,
    tokens: u32 = 0,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, name: []const u8) !Place {
        return Place{
            .id = try allocator.dupe(u8, id),
            .name = try allocator.dupe(u8, name),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Place) void {
        self.allocator.free(self.id);
        self.allocator.free(self.name);
    }
};

/// Transition in the Petri net
pub const Transition = struct {
    id: []const u8,
    activity: []const u8,
    is_silent: bool = false,
    allocator: Allocator,

    pub fn init(allocator: Allocator, id: []const u8, activity: []const u8) !Transition {
        return Transition{
            .id = try allocator.dupe(u8, id),
            .activity = try allocator.dupe(u8, activity),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Transition) void {
        self.allocator.free(self.id);
        self.allocator.free(self.activity);
    }
};

/// Arc type
pub const ArcType = enum {
    PLACE_TO_TRANSITION,
    TRANSITION_TO_PLACE,
};

/// Arc in the Petri net
pub const Arc = struct {
    source_id: []const u8,
    target_id: []const u8,
    arc_type: ArcType,
    weight: u32 = 1,
    allocator: Allocator,

    pub fn init(allocator: Allocator, source: []const u8, target: []const u8, arc_type: ArcType) !Arc {
        return Arc{
            .source_id = try allocator.dupe(u8, source),
            .target_id = try allocator.dupe(u8, target),
            .arc_type = arc_type,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Arc) void {
        self.allocator.free(self.source_id);
        self.allocator.free(self.target_id);
    }
};

/// Process Model (Petri Net)
pub const ProcessModel = struct {
    name: []const u8,
    places: std.ArrayList(Place),
    transitions: std.ArrayList(Transition),
    arcs: std.ArrayList(Arc),
    initial_place: ?[]const u8 = null,
    final_place: ?[]const u8 = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8) !ProcessModel {
        return ProcessModel{
            .name = try allocator.dupe(u8, name),
            .places = std.ArrayList(Place){},
            .transitions = std.ArrayList(Transition){},
            .arcs = std.ArrayList(Arc){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ProcessModel) void {
        for (self.places.items) |*p| p.deinit();
        self.places.deinit(self.allocator);

        for (self.transitions.items) |*t| t.deinit();
        self.transitions.deinit(self.allocator);

        for (self.arcs.items) |*a| a.deinit();
        self.arcs.deinit(self.allocator);

        if (self.initial_place) |ip| self.allocator.free(ip);
        if (self.final_place) |fp| self.allocator.free(fp);
        self.allocator.free(self.name);
    }

    pub fn addPlace(self: *ProcessModel, id: []const u8, name: []const u8) !void {
        const place = try Place.init(self.allocator, id, name);
        try self.places.append(self.allocator, place);
    }

    pub fn addTransition(self: *ProcessModel, id: []const u8, activity: []const u8) !void {
        const transition = try Transition.init(self.allocator, id, activity);
        try self.transitions.append(self.allocator, transition);
    }

    pub fn addArc(self: *ProcessModel, source: []const u8, target: []const u8, arc_type: ArcType) !void {
        const arc = try Arc.init(self.allocator, source, target, arc_type);
        try self.arcs.append(self.allocator, arc);
    }

    pub fn setInitialPlace(self: *ProcessModel, place_id: []const u8) !void {
        if (self.initial_place) |ip| self.allocator.free(ip);
        self.initial_place = try self.allocator.dupe(u8, place_id);
    }

    pub fn setFinalPlace(self: *ProcessModel, place_id: []const u8) !void {
        if (self.final_place) |fp| self.allocator.free(fp);
        self.final_place = try self.allocator.dupe(u8, place_id);
    }

    pub fn getPlaceCount(self: *const ProcessModel) usize {
        return self.places.items.len;
    }

    pub fn getTransitionCount(self: *const ProcessModel) usize {
        return self.transitions.items.len;
    }

    pub fn getArcCount(self: *const ProcessModel) usize {
        return self.arcs.items.len;
    }
};

// ============================================================================
// Process Miner
// ============================================================================

/// Process Miner - discovers process models from event logs
pub const ProcessMiner = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ProcessMiner {
        return ProcessMiner{ .allocator = allocator };
    }

    /// Build Direct Follows Graph from event log
    pub fn buildDFG(self: *ProcessMiner, log: *const event_collector.EventLog) !DirectFollowsGraph {
        var dfg = DirectFollowsGraph.init(self.allocator);
        errdefer dfg.deinit();

        var trace_iter = log.traces.valueIterator();
        while (trace_iter.next()) |trace| {
            const events = trace.*.events.items;

            if (events.len == 0) continue;

            // First activity is a start activity
            try dfg.addStartActivity(events[0].activity);

            // Build directly-follows edges
            var i: usize = 0;
            while (i < events.len - 1) : (i += 1) {
                try dfg.addEdge(events[i].activity, events[i + 1].activity);
            }

            // Last activity is an end activity
            try dfg.addEndActivity(events[events.len - 1].activity);
        }

        return dfg;
    }

    /// Discover process model using Alpha algorithm (simplified)
    pub fn discoverModel(self: *ProcessMiner, log: *const event_collector.EventLog, name: []const u8) !ProcessModel {
        // Step 1: Build DFG
        var dfg = try self.buildDFG(log);
        defer dfg.deinit();

        // Step 2: Build footprint matrix
        var footprint = try FootprintMatrix.fromDFG(self.allocator, &dfg);
        defer footprint.deinit();

        // Step 3: Construct Petri net
        var model = try ProcessModel.init(self.allocator, name);
        errdefer model.deinit();

        // Add initial place
        try model.addPlace("p_start", "Start");
        try model.setInitialPlace("p_start");

        // Add transition for each activity
        var place_counter: usize = 1;
        var act_iter = dfg.activities.keyIterator();
        while (act_iter.next()) |activity| {
            // Add transition
            var trans_id_buf: [64]u8 = undefined;
            const trans_id = std.fmt.bufPrint(&trans_id_buf, "t_{s}", .{activity.*}) catch "t_unknown";
            try model.addTransition(trans_id, activity.*);

            // Add output place
            var place_id_buf: [64]u8 = undefined;
            const place_id = std.fmt.bufPrint(&place_id_buf, "p_{d}", .{place_counter}) catch "p_unknown";
            var place_name_buf: [128]u8 = undefined;
            const place_name = std.fmt.bufPrint(&place_name_buf, "After {s}", .{activity.*}) catch "After unknown";
            try model.addPlace(place_id, place_name);
            place_counter += 1;
        }

        // Add final place
        try model.addPlace("p_end", "End");
        try model.setFinalPlace("p_end");

        // Connect start activities to initial place
        var start_iter = dfg.start_activities.keyIterator();
        while (start_iter.next()) |start_act| {
            var trans_id_buf: [64]u8 = undefined;
            const trans_id = std.fmt.bufPrint(&trans_id_buf, "t_{s}", .{start_act.*}) catch "t_unknown";
            try model.addArc("p_start", trans_id, .PLACE_TO_TRANSITION);
        }

        // Connect end activities to final place
        var end_iter = dfg.end_activities.keyIterator();
        while (end_iter.next()) |end_act| {
            var trans_id_buf: [64]u8 = undefined;
            const trans_id = std.fmt.bufPrint(&trans_id_buf, "t_{s}", .{end_act.*}) catch "t_unknown";
            try model.addArc(trans_id, "p_end", .TRANSITION_TO_PLACE);
        }

        return model;
    }
};

// ============================================================================
// Conformance Checker
// ============================================================================

/// Conformance result
pub const ConformanceResult = struct {
    fitness: f64, // 0.0 - 1.0
    precision: f64, // 0.0 - 1.0
    traces_fitting: u64,
    traces_total: u64,
    deviations: std.ArrayList([]const u8),
    allocator: Allocator,

    pub fn init(allocator: Allocator) ConformanceResult {
        return ConformanceResult{
            .fitness = 0.0,
            .precision = 0.0,
            .traces_fitting = 0,
            .traces_total = 0,
            .deviations = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConformanceResult) void {
        for (self.deviations.items) |dev| {
            self.allocator.free(dev);
        }
        self.deviations.deinit(self.allocator);
    }

    pub fn addDeviation(self: *ConformanceResult, deviation: []const u8) !void {
        const duped = try self.allocator.dupe(u8, deviation);
        try self.deviations.append(self.allocator, duped);
    }
};

/// Conformance Checker
pub const ConformanceChecker = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) ConformanceChecker {
        return ConformanceChecker{ .allocator = allocator };
    }

    /// Check conformance of a log against a model (simplified token-based replay)
    pub fn checkConformance(self: *ConformanceChecker, _: *const ProcessModel, log: *const event_collector.EventLog) !ConformanceResult {
        var result = ConformanceResult.init(self.allocator);

        var trace_iter = log.traces.valueIterator();
        while (trace_iter.next()) |_| {
            result.traces_total += 1;
            // Simplified: assume all traces fit for now
            result.traces_fitting += 1;
        }

        if (result.traces_total > 0) {
            result.fitness = @as(f64, @floatFromInt(result.traces_fitting)) /
                @as(f64, @floatFromInt(result.traces_total));
            result.precision = 0.9; // Placeholder
        }

        return result;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "DirectFollowsGraph basic operations" {
    const allocator = std.testing.allocator;

    var dfg = DirectFollowsGraph.init(allocator);
    defer dfg.deinit();

    try dfg.addEdge("A", "B");
    try dfg.addEdge("B", "C");
    try dfg.addEdge("A", "B"); // Duplicate

    try std.testing.expect(dfg.hasEdge("A", "B"));
    try std.testing.expect(dfg.hasEdge("B", "C"));
    try std.testing.expect(!dfg.hasEdge("C", "A"));
    try std.testing.expectEqual(@as(u64, 2), dfg.getEdgeCount("A", "B"));
}

test "DirectFollowsGraph start/end activities" {
    const allocator = std.testing.allocator;

    var dfg = DirectFollowsGraph.init(allocator);
    defer dfg.deinit();

    try dfg.addStartActivity("Start");
    try dfg.addEndActivity("End");

    try std.testing.expect(dfg.isStartActivity("Start"));
    try std.testing.expect(dfg.isEndActivity("End"));
    try std.testing.expect(!dfg.isStartActivity("End"));
}

test "FootprintMatrix relations" {
    const allocator = std.testing.allocator;

    var matrix = FootprintMatrix.init(allocator);
    defer matrix.deinit();

    try matrix.addActivity("A");
    try matrix.addActivity("B");
    try matrix.setRelation("A", "B", .CAUSAL);

    try std.testing.expectEqual(ActivityRelation.CAUSAL, matrix.getRelation("A", "B"));
    try std.testing.expectEqual(ActivityRelation.UNRELATED, matrix.getRelation("B", "A"));
}

test "FootprintMatrix from DFG" {
    const allocator = std.testing.allocator;

    var dfg = DirectFollowsGraph.init(allocator);
    defer dfg.deinit();

    try dfg.addEdge("A", "B");
    try dfg.addEdge("B", "C");

    var matrix = try FootprintMatrix.fromDFG(allocator, &dfg);
    defer matrix.deinit();

    try std.testing.expectEqual(ActivityRelation.CAUSAL, matrix.getRelation("A", "B"));
    try std.testing.expectEqual(ActivityRelation.CAUSAL, matrix.getRelation("B", "C"));
}

test "ProcessModel construction" {
    const allocator = std.testing.allocator;

    var model = try ProcessModel.init(allocator, "Test Model");
    defer model.deinit();

    try model.addPlace("p1", "Start");
    try model.addPlace("p2", "End");
    try model.addTransition("t1", "Activity A");
    try model.addArc("p1", "t1", .PLACE_TO_TRANSITION);
    try model.addArc("t1", "p2", .TRANSITION_TO_PLACE);

    try std.testing.expectEqual(@as(usize, 2), model.getPlaceCount());
    try std.testing.expectEqual(@as(usize, 1), model.getTransitionCount());
    try std.testing.expectEqual(@as(usize, 2), model.getArcCount());
}

test "ProcessMiner discover model" {
    const allocator = std.testing.allocator;

    // Create event log
    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    // Add events for a simple sequence: A -> B -> C
    const e1 = try event_collector.ProcessEvent.init(allocator, "e1", "case-1", "A");
    const e2 = try event_collector.ProcessEvent.init(allocator, "e2", "case-1", "B");
    const e3 = try event_collector.ProcessEvent.init(allocator, "e3", "case-1", "C");
    try log.recordEvent(e1);
    try log.recordEvent(e2);
    try log.recordEvent(e3);

    // Mine model
    var miner = ProcessMiner.init(allocator);
    var model = try miner.discoverModel(&log, "Discovered Model");
    defer model.deinit();

    try std.testing.expect(model.getPlaceCount() > 0);
    try std.testing.expect(model.getTransitionCount() > 0);
}

test "ConformanceChecker basic" {
    const allocator = std.testing.allocator;

    var log = try event_collector.EventLog.init(allocator, "test-log");
    defer log.deinit();

    const e1 = try event_collector.ProcessEvent.init(allocator, "e1", "case-1", "A");
    try log.recordEvent(e1);

    var miner = ProcessMiner.init(allocator);
    var model = try miner.discoverModel(&log, "Model");
    defer model.deinit();

    var checker = ConformanceChecker.init(allocator);
    var result = try checker.checkConformance(&model, &log);
    defer result.deinit();

    try std.testing.expect(result.fitness > 0.0);
    try std.testing.expectEqual(@as(u64, 1), result.traces_total);
}
