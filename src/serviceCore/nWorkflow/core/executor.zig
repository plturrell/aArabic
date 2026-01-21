// Petri Net Execution Engine
// Part of serviceCore nWorkflow
// Day 4-5: Execution Strategies, Conflict Resolution, State Persistence, Events
//
// The Executor manages how transitions fire in a Petri Net:
// - Sequential: One transition at a time (deterministic)
// - Concurrent: Multiple transitions in parallel
// - Priority-based: Highest priority transitions first
// - Custom: User-defined strategies
// - Event system for monitoring and integration
// - Performance metrics and optimization
// - Event filtering and replay capabilities

const std = @import("std");
const petri_net = @import("petri_net");
const Allocator = std.mem.Allocator;
const PetriNet = petri_net.PetriNet;
const Marking = petri_net.Marking;

/// ExecutionStrategy defines how transitions are selected and fired
pub const ExecutionStrategy = enum {
    sequential,      // Fire one enabled transition at a time
    concurrent,      // Fire all enabled transitions in parallel
    priority_based,  // Fire highest priority transition first
    custom,          // User-defined strategy
    
    pub fn description(self: ExecutionStrategy) []const u8 {
        return switch (self) {
            .sequential => "Sequential: One transition at a time (deterministic)",
            .concurrent => "Concurrent: All enabled transitions in parallel",
            .priority_based => "Priority-based: Highest priority first",
            .custom => "Custom: User-defined strategy",
        };
    }
};

/// ConflictResolution defines how to choose between multiple enabled transitions
pub const ConflictResolution = enum {
    priority,        // Use transition priority
    random,          // Random selection (fairness)
    round_robin,     // Rotate through transitions
    weighted_random, // Weighted random based on priority
};

/// ExecutionEvent represents things that happen during execution
pub const ExecutionEvent = union(enum) {
    transition_fired: struct {
        transition_id: []const u8,
        timestamp: i64,
        duration_ns: u64,
    },
    token_moved: struct {
        from_place: []const u8,
        to_place: []const u8,
        token_id: u64,
        timestamp: i64,
    },
    deadlock_detected: struct {
        timestamp: i64,
    },
    state_changed: struct {
        timestamp: i64,
    },
    execution_started: struct {
        timestamp: i64,
        strategy: []const u8,
    },
    execution_completed: struct {
        timestamp: i64,
        total_steps: usize,
        total_duration_ns: u64,
    },
    execution_failed: struct {
        timestamp: i64,
        error_message: []const u8,
    },
    
    pub fn getType(self: ExecutionEvent) []const u8 {
        return switch (self) {
            .transition_fired => "transition_fired",
            .token_moved => "token_moved",
            .deadlock_detected => "deadlock_detected",
            .state_changed => "state_changed",
            .execution_started => "execution_started",
            .execution_completed => "execution_completed",
            .execution_failed => "execution_failed",
        };
    }
};

/// Snapshot represents a saved state of the Petri Net
pub const Snapshot = struct {
    allocator: Allocator,
    marking: Marking,
    timestamp: i64,
    metadata: std.StringHashMap([]const u8),
    
    pub fn init(allocator: Allocator, marking: Marking) !Snapshot {
        return Snapshot{
            .allocator = allocator,
            .marking = marking,
            .timestamp = std.time.timestamp(),
            .metadata = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *Snapshot) void {
        self.marking.deinit();
        var it = self.metadata.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }
    
    pub fn setMetadata(self: *Snapshot, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        const value_copy = try self.allocator.dupe(u8, value);
        try self.metadata.put(key_copy, value_copy);
    }
    
    pub fn getMetadata(self: *const Snapshot, key: []const u8) ?[]const u8 {
        return self.metadata.get(key);
    }
};

/// EventListener is a callback function for execution events
pub const EventListener = *const fn (ExecutionEvent) void;

/// CustomStrategyFn is a user-defined strategy function
pub const CustomStrategyFn = *const fn ([][]const u8) []const u8;

/// EventFilter determines which events to process
pub const EventFilter = struct {
    allow_transition_fired: bool = true,
    allow_token_moved: bool = true,
    allow_deadlock_detected: bool = true,
    allow_state_changed: bool = true,
    allow_execution_started: bool = true,
    allow_execution_completed: bool = true,
    allow_execution_failed: bool = true,
    
    pub fn matches(self: *const EventFilter, event: ExecutionEvent) bool {
        return switch (event) {
            .transition_fired => self.allow_transition_fired,
            .token_moved => self.allow_token_moved,
            .deadlock_detected => self.allow_deadlock_detected,
            .state_changed => self.allow_state_changed,
            .execution_started => self.allow_execution_started,
            .execution_completed => self.allow_execution_completed,
            .execution_failed => self.allow_execution_failed,
        };
    }
    
    pub fn onlyErrors() EventFilter {
        return EventFilter{
            .allow_transition_fired = false,
            .allow_token_moved = false,
            .allow_deadlock_detected = true,
            .allow_state_changed = false,
            .allow_execution_started = false,
            .allow_execution_completed = false,
            .allow_execution_failed = true,
        };
    }
    
    pub fn onlyImportant() EventFilter {
        return EventFilter{
            .allow_transition_fired = true,
            .allow_token_moved = false,
            .allow_deadlock_detected = true,
            .allow_state_changed = false,
            .allow_execution_started = true,
            .allow_execution_completed = true,
            .allow_execution_failed = true,
        };
    }
};

/// PetriNetExecutor manages the execution of a Petri Net
pub const PetriNetExecutor = struct {
    allocator: Allocator,
    net: *PetriNet,
    strategy: ExecutionStrategy,
    conflict_resolution: ConflictResolution,
    event_listeners: std.ArrayList(EventListener),
    execution_history: std.ArrayList(ExecutionEvent),
    max_history_size: usize,
    step_count: usize,
    round_robin_index: usize,
    custom_strategy_fn: ?CustomStrategyFn,
    event_filter: EventFilter,
    start_time_ns: u64,
    total_fire_time_ns: u64,
    rng: std.Random.DefaultPrng, // Proper PRNG for random selection

    pub fn init(allocator: Allocator, net: *PetriNet, strategy: ExecutionStrategy) !PetriNetExecutor {
        // Initialize PRNG with cryptographic seed for proper randomness
        var seed: u64 = undefined;
        std.posix.getrandom(std.mem.asBytes(&seed)) catch {
            // Fallback to timestamp if getrandom fails
            seed = @as(u64, @intCast(std.time.nanoTimestamp()));
        };

        return PetriNetExecutor{
            .allocator = allocator,
            .net = net,
            .strategy = strategy,
            .conflict_resolution = .priority,
            .event_listeners = .{},
            .execution_history = .{},
            .max_history_size = 1000,
            .step_count = 0,
            .round_robin_index = 0,
            .custom_strategy_fn = null,
            .event_filter = EventFilter{},
            .start_time_ns = 0,
            .total_fire_time_ns = 0,
            .rng = std.Random.DefaultPrng.init(seed),
        };
    }
    
    pub fn deinit(self: *PetriNetExecutor) void {
        self.event_listeners.deinit(self.allocator);
        // Free event history
        for (self.execution_history.items) |event| {
            switch (event) {
                .execution_failed => |failed| {
                    self.allocator.free(failed.error_message);
                },
                else => {},
            }
        }
        self.execution_history.deinit(self.allocator);
    }
    
    /// Set the conflict resolution strategy
    pub fn setConflictResolution(self: *PetriNetExecutor, resolution: ConflictResolution) void {
        self.conflict_resolution = resolution;
    }
    
    /// Set a custom strategy function
    pub fn setCustomStrategy(self: *PetriNetExecutor, strategy_fn: CustomStrategyFn) void {
        self.custom_strategy_fn = strategy_fn;
        self.strategy = .custom;
    }
    
    /// Set event filter for selective event processing
    pub fn setEventFilter(self: *PetriNetExecutor, filter: EventFilter) void {
        self.event_filter = filter;
    }
    
    /// Execute one step (fire one or more transitions based on strategy)
    pub fn step(self: *PetriNetExecutor) !bool {
        var enabled = try self.net.getEnabledTransitions();
        defer enabled.deinit(self.allocator);
        
        if (enabled.items.len == 0) {
            try self.emitEvent(.{ .deadlock_detected = .{ .timestamp = std.time.timestamp() } });
            return false;
        }
        
        switch (self.strategy) {
            .sequential => {
                const transition_id = try self.selectTransition(enabled.items);
                try self.fireTransitionWithEvent(transition_id);
                self.step_count += 1;
                return true;
            },
            .concurrent => {
                // Fire all enabled transitions
                for (enabled.items) |transition_id| {
                    try self.fireTransitionWithEvent(transition_id);
                }
                self.step_count += 1;
                return true;
            },
            .priority_based => {
                const transition_id = try self.selectTransitionByPriority(enabled.items);
                try self.fireTransitionWithEvent(transition_id);
                self.step_count += 1;
                return true;
            },
            .custom => {
                if (self.custom_strategy_fn) |strategy_fn| {
                    const transition_id = strategy_fn(enabled.items);
                    if (transition_id.len > 0) {
                        try self.fireTransitionWithEvent(transition_id);
                        self.step_count += 1;
                        return true;
                    }
                }
                // Fallback to first transition
                const transition_id = enabled.items[0];
                try self.fireTransitionWithEvent(transition_id);
                self.step_count += 1;
                return true;
            },
        }
    }
    
    /// Run for a maximum number of steps or until deadlock
    pub fn run(self: *PetriNetExecutor, max_steps: usize) !void {
        self.start_time_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
        const strategy_name = @tagName(self.strategy);
        try self.emitEvent(.{ .execution_started = .{ 
            .timestamp = std.time.timestamp(),
            .strategy = strategy_name,
        } });
        
        var steps: usize = 0;
        while (steps < max_steps) : (steps += 1) {
            const continued = try self.step();
            if (!continued) break;
        }
        
        const end_time_ns = std.time.nanoTimestamp();
        const duration = @as(u64, @intCast(end_time_ns - @as(i128, @intCast(self.start_time_ns))));
        
        try self.emitEvent(.{ .execution_completed = .{
            .timestamp = std.time.timestamp(),
            .total_steps = steps,
            .total_duration_ns = duration,
        } });
    }
    
    /// Run until no transitions are enabled
    pub fn runUntilComplete(self: *PetriNetExecutor) !void {
        self.start_time_ns = @as(u64, @intCast(std.time.nanoTimestamp()));
        const strategy_name = @tagName(self.strategy);
        try self.emitEvent(.{ .execution_started = .{ 
            .timestamp = std.time.timestamp(),
            .strategy = strategy_name,
        } });
        
        var steps: usize = 0;
        while (true) {
            const continued = try self.step();
            if (!continued) break;
            steps += 1;
            
            // Safety: prevent infinite loops
            if (steps > 100000) {
                const msg = try self.allocator.dupe(u8, "Maximum step limit exceeded");
                try self.emitEvent(.{ .execution_failed = .{
                    .timestamp = std.time.timestamp(),
                    .error_message = msg,
                } });
                return error.MaxStepsExceeded;
            }
        }
        
        const end_time_ns = std.time.nanoTimestamp();
        const duration = @as(u64, @intCast(end_time_ns - @as(i128, @intCast(self.start_time_ns))));
        
        try self.emitEvent(.{ .execution_completed = .{
            .timestamp = std.time.timestamp(),
            .total_steps = steps,
            .total_duration_ns = duration,
        } });
    }
    
    /// Create a snapshot of the current state
    pub fn createSnapshot(self: *const PetriNetExecutor) !Snapshot {
        const marking = try self.net.getCurrentMarking();
        var snapshot = try Snapshot.init(self.allocator, marking);
        
        // Add metadata
        const step_str = try std.fmt.allocPrint(self.allocator, "{d}", .{self.step_count});
        try snapshot.setMetadata("step_count", step_str);
        self.allocator.free(step_str);
        
        return snapshot;
    }
    
    /// Restore from a snapshot
    pub fn restoreSnapshot(self: *PetriNetExecutor, snapshot: *const Snapshot) !void {
        // Collect all place IDs first to avoid iterator invalidation
        var place_ids = try std.ArrayList([]const u8).initCapacity(self.allocator, self.net.places.count());
        defer place_ids.deinit(self.allocator);
        
        var place_it = self.net.places.iterator();
        while (place_it.next()) |entry| {
            try place_ids.append(self.allocator, entry.key_ptr.*);
        }
        
        // Clear current state - safely clear tokens and reinitialize ArrayList
        for (place_ids.items) |place_id| {
            if (self.net.places.get(place_id)) |place_ptr| {
                // Deinit tokens and reinitialize the ArrayList properly
                for (place_ptr.tokens.items) |*token| {
                    token.deinit(self.allocator);
                }
                place_ptr.tokens.deinit(self.allocator);
                place_ptr.tokens = try std.ArrayList(petri_net.Token).initCapacity(self.allocator, 4);
            }
        }
        
        // Restore from marking
        for (place_ids.items) |place_id| {
            const count = snapshot.marking.get(place_id);
            var i: usize = 0;
            while (i < count) : (i += 1) {
                try self.net.addTokenToPlace(place_id, "{}");
            }
        }
        
        // Restore metadata if present
        if (snapshot.getMetadata("step_count")) |step_str| {
            self.step_count = try std.fmt.parseInt(usize, step_str, 10);
        }
        
        try self.emitEvent(.{ .state_changed = .{ .timestamp = std.time.timestamp() } });
    }
    
    /// Add an event listener
    pub fn addEventListener(self: *PetriNetExecutor, listener: EventListener) !void {
        try self.event_listeners.append(self.allocator, listener);
    }
    
    /// Remove an event listener
    pub fn removeEventListener(self: *PetriNetExecutor, listener: EventListener) !void {
        for (self.event_listeners.items, 0..) |l, i| {
            if (l == listener) {
                _ = self.event_listeners.orderedRemove(i);
                return;
            }
        }
    }
    
    /// Emit an event to all listeners
    pub fn emitEvent(self: *PetriNetExecutor, event: ExecutionEvent) !void {
        // Check filter
        if (!self.event_filter.matches(event)) {
            return;
        }
        
        // Add to history
        try self.execution_history.append(self.allocator, event);
        
        // Limit history size
        if (self.execution_history.items.len > self.max_history_size) {
            _ = self.execution_history.orderedRemove(0);
        }
        
        // Notify listeners
        for (self.event_listeners.items) |listener| {
            listener(event);
        }
    }
    
    /// Get execution statistics
    pub fn getStats(self: *const PetriNetExecutor) ExecutionStats {
        var transition_count: usize = 0;
        var deadlock_count: usize = 0;
        var total_fire_duration_ns: u64 = 0;
        
        for (self.execution_history.items) |event| {
            switch (event) {
                .transition_fired => |fired| {
                    transition_count += 1;
                    total_fire_duration_ns += fired.duration_ns;
                },
                .deadlock_detected => deadlock_count += 1,
                else => {},
            }
        }
        
        const avg_fire_time_ns = if (transition_count > 0) 
            total_fire_duration_ns / transition_count 
        else 
            0;
        
        return ExecutionStats{
            .total_steps = self.step_count,
            .transitions_fired = transition_count,
            .deadlocks_detected = deadlock_count,
            .events_recorded = self.execution_history.items.len,
            .avg_transition_fire_time_ns = avg_fire_time_ns,
            .total_fire_time_ns = total_fire_duration_ns,
        };
    }
    
    /// Replay execution from history (for debugging/analysis)
    pub fn replayHistory(self: *const PetriNetExecutor, allocator: Allocator) ![]ExecutionEvent {
        const replay = try allocator.alloc(ExecutionEvent, self.execution_history.items.len);
        @memcpy(replay, self.execution_history.items);
        return replay;
    }
    
    /// Export metrics in JSON format
    pub fn exportMetrics(self: *const PetriNetExecutor, allocator: Allocator) ![]const u8 {
        const stats = self.getStats();
        
        // Build JSON manually for simplicity
        const json = try std.fmt.allocPrint(allocator,
            \\{{
            \\  "total_steps": {d},
            \\  "transitions_fired": {d},
            \\  "deadlocks_detected": {d},
            \\  "events_recorded": {d},
            \\  "avg_transition_fire_time_ns": {d},
            \\  "total_fire_time_ns": {d},
            \\  "strategy": "{s}",
            \\  "conflict_resolution": "{s}"
            \\}}
        , .{
            stats.total_steps,
            stats.transitions_fired,
            stats.deadlocks_detected,
            stats.events_recorded,
            stats.avg_transition_fire_time_ns,
            stats.total_fire_time_ns,
            @tagName(self.strategy),
            @tagName(self.conflict_resolution),
        });
        
        return json;
    }
    
    /// Clear execution history
    pub fn clearHistory(self: *PetriNetExecutor) void {
        for (self.execution_history.items) |event| {
            switch (event) {
                .execution_failed => |failed| {
                    self.allocator.free(failed.error_message);
                },
                else => {},
            }
        }
        self.execution_history.clearRetainingCapacity();
    }
    
    // ========================================================================
    // Private helper methods
    // ========================================================================
    
    fn fireTransitionWithEvent(self: *PetriNetExecutor, transition_id: []const u8) !void {
        const start_ns = std.time.nanoTimestamp();
        try self.net.fireTransition(transition_id);
        const end_ns = std.time.nanoTimestamp();
        const duration = @as(u64, @intCast(end_ns - start_ns));
        
        self.total_fire_time_ns += duration;
        
        try self.emitEvent(.{ .transition_fired = .{
            .transition_id = transition_id,
            .timestamp = std.time.timestamp(),
            .duration_ns = duration,
        } });
        try self.emitEvent(.{ .state_changed = .{ .timestamp = std.time.timestamp() } });
    }
    
    fn selectTransition(self: *PetriNetExecutor, enabled: [][]const u8) ![]const u8 {
        if (enabled.len == 0) return error.NoEnabledTransitions;
        
        return switch (self.conflict_resolution) {
            .priority => try self.selectTransitionByPriority(enabled),
            .random => self.selectTransitionRandom(enabled),
            .round_robin => self.selectTransitionRoundRobin(enabled),
            .weighted_random => try self.selectTransitionWeightedRandom(enabled),
        };
    }
    
    fn selectTransitionByPriority(self: *PetriNetExecutor, enabled: [][]const u8) ![]const u8 {
        if (enabled.len == 0) return error.NoEnabledTransitions;
        
        var highest_priority: i32 = std.math.minInt(i32);
        var selected: []const u8 = enabled[0];
        
        for (enabled) |trans_id| {
            if (self.net.transitions.get(trans_id)) |trans| {
                if (trans.priority > highest_priority) {
                    highest_priority = trans.priority;
                    selected = trans_id;
                }
            }
        }
        
        return selected;
    }
    
    fn selectTransitionRandom(self: *PetriNetExecutor, enabled: [][]const u8) []const u8 {
        if (enabled.len == 0) return "";

        // Use proper PRNG for uniform random selection
        const index = self.rng.random().uintLessThan(usize, enabled.len);
        return enabled[index];
    }
    
    fn selectTransitionRoundRobin(self: *PetriNetExecutor, enabled: [][]const u8) []const u8 {
        if (enabled.len == 0) return "";
        
        const selected = enabled[self.round_robin_index % enabled.len];
        self.round_robin_index += 1;
        return selected;
    }
    
    fn selectTransitionWeightedRandom(self: *PetriNetExecutor, enabled: [][]const u8) ![]const u8 {
        if (enabled.len == 0) return error.NoEnabledTransitions;
        
        // Calculate total weight (priority values)
        var total_weight: i32 = 0;
        for (enabled) |trans_id| {
            if (self.net.transitions.get(trans_id)) |trans| {
                total_weight += @max(1, trans.priority);
            }
        }
        
        // Select based on weight
        const rand_value = @rem(std.time.timestamp(), @as(i64, @intCast(total_weight)));
        var cumulative: i32 = 0;
        
        for (enabled) |trans_id| {
            if (self.net.transitions.get(trans_id)) |trans| {
                cumulative += @max(1, trans.priority);
                if (rand_value < cumulative) {
                    return trans_id;
                }
            }
        }
        
        return enabled[0];
    }
};

pub const ExecutionStats = struct {
    total_steps: usize,
    transitions_fired: usize,
    deadlocks_detected: usize,
    events_recorded: usize,
    avg_transition_fire_time_ns: u64,
    total_fire_time_ns: u64,
    
    pub fn format(self: ExecutionStats, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator,
            \\Execution Statistics:
            \\  Total Steps: {d}
            \\  Transitions Fired: {d}
            \\  Deadlocks: {d}
            \\  Events: {d}
            \\  Avg Fire Time: {d} ns
            \\  Total Fire Time: {d} ns
        , .{
            self.total_steps,
            self.transitions_fired,
            self.deadlocks_detected,
            self.events_recorded,
            self.avg_transition_fire_time_ns,
            self.total_fire_time_ns,
        });
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "Sequential execution strategy" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Sequential Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    const result = try executor.step();
    try std.testing.expect(result);
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 0), marking.get("p1"));
    try std.testing.expectEqual(@as(usize, 1), marking.get("p2"));
}

test "Concurrent execution strategy" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Concurrent Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input 1", null);
    _ = try net.addPlace("p2", "Input 2", null);
    _ = try net.addPlace("p3", "Output 1", null);
    _ = try net.addPlace("p4", "Output 2", null);
    
    _ = try net.addTransition("t1", "Process 1", 0);
    _ = try net.addTransition("t2", "Process 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p3");
    _ = try net.addArc("a3", .input, 1, "p2", "t2");
    _ = try net.addArc("a4", .output, 1, "t2", "p4");
    
    try net.addTokenToPlace("p1", "{}");
    try net.addTokenToPlace("p2", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .concurrent);
    defer executor.deinit();
    
    const result = try executor.step();
    try std.testing.expect(result);
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 1), marking.get("p3"));
    try std.testing.expectEqual(@as(usize, 1), marking.get("p4"));
}

test "Priority-based execution" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Priority Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Output Low", null);
    _ = try net.addPlace("p3", "Output High", null);
    
    _ = try net.addTransition("t_low", "Low Priority", 1);
    _ = try net.addTransition("t_high", "High Priority", 10);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t_low");
    _ = try net.addArc("a2", .output, 1, "t_low", "p2");
    _ = try net.addArc("a3", .input, 1, "p1", "t_high");
    _ = try net.addArc("a4", .output, 1, "t_high", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .priority_based);
    defer executor.deinit();
    
    const result = try executor.step();
    try std.testing.expect(result);
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    // High priority transition should have fired
    try std.testing.expectEqual(@as(usize, 1), marking.get("p3"));
    try std.testing.expectEqual(@as(usize, 0), marking.get("p2"));
}

test "Snapshot creation and restoration" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Snapshot Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "State", null);
    _ = try net.addPlace("p2", "Result", null);
    _ = try net.addTransition("t1", "Action", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    // Create snapshot before execution
    var snapshot = try executor.createSnapshot();
    defer snapshot.deinit();
    
    // Execute
    _ = try executor.step();
    
    var marking_after = try net.getCurrentMarking();
    defer marking_after.deinit();
    try std.testing.expectEqual(@as(usize, 0), marking_after.get("p1"));
    try std.testing.expectEqual(@as(usize, 1), marking_after.get("p2"));
    
    // Restore
    try executor.restoreSnapshot(&snapshot);
    
    var marking_restored = try net.getCurrentMarking();
    defer marking_restored.deinit();
    try std.testing.expectEqual(@as(usize, 1), marking_restored.get("p1"));
    try std.testing.expectEqual(@as(usize, 0), marking_restored.get("p2"));
}

test "Event emission and listening" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Event Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    const TestListener = struct {
        var fired_count: usize = 0;
        
        fn listener(event: ExecutionEvent) void {
            switch (event) {
                .transition_fired => fired_count += 1,
                else => {},
            }
        }
    };
    
    TestListener.fired_count = 0;
    try executor.addEventListener(TestListener.listener);
    
    _ = try executor.step();
    
    try std.testing.expectEqual(@as(usize, 1), TestListener.fired_count);
}

test "Run with max steps" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Max Steps Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "Middle", null);
    _ = try net.addPlace("p3", "End", null);
    
    _ = try net.addTransition("t1", "Step 1", 0);
    _ = try net.addTransition("t2", "Step 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    _ = try net.addArc("a3", .input, 1, "p2", "t2");
    _ = try net.addArc("a4", .output, 1, "t2", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.run(10);
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 1), marking.get("p3"));
}

test "Deadlock detection" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Deadlock Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Empty", null);
    _ = try net.addTransition("t1", "Blocked", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    const result = try executor.step();
    try std.testing.expect(!result);
    
    // Check event history for deadlock
    var found_deadlock = false;
    for (executor.execution_history.items) |event| {
        if (event == .deadlock_detected) {
            found_deadlock = true;
            break;
        }
    }
    try std.testing.expect(found_deadlock);
}

test "Execution statistics" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Stats Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    _ = try executor.step();
    
    const stats = executor.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.total_steps);
    try std.testing.expectEqual(@as(usize, 1), stats.transitions_fired);
}

test "Conflict resolution - round robin" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Round Robin Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Output A", null);
    _ = try net.addPlace("p3", "Output B", null);
    
    _ = try net.addTransition("t1", "Option A", 0);
    _ = try net.addTransition("t2", "Option B", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    _ = try net.addArc("a3", .input, 1, "p1", "t2");
    _ = try net.addArc("a4", .output, 1, "t2", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    executor.setConflictResolution(.round_robin);
    
    _ = try executor.step();
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    // One of the outputs should have a token
    const total = marking.get("p2") + marking.get("p3");
    try std.testing.expectEqual(@as(usize, 1), total);
}

test "Run until complete" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Complete Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "Middle", null);
    _ = try net.addPlace("p3", "End", null);
    
    _ = try net.addTransition("t1", "Step 1", 0);
    _ = try net.addTransition("t2", "Step 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    _ = try net.addArc("a3", .input, 1, "p2", "t2");
    _ = try net.addArc("a4", .output, 1, "t2", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.runUntilComplete();
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 0), marking.get("p1"));
    try std.testing.expectEqual(@as(usize, 0), marking.get("p2"));
    try std.testing.expectEqual(@as(usize, 1), marking.get("p3"));
    
    const stats = executor.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.transitions_fired);
}

test "Memory leak check" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Memory Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Test", null);
    _ = try net.addTransition("t1", "Test", 0);
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    // Create and destroy snapshots
    var snapshot = try executor.createSnapshot();
    defer snapshot.deinit();
    
    try snapshot.setMetadata("test", "value");
    
    // All memory should be cleaned up by defers
}

// ============================================================================
// DAY 5 TESTS: Performance, Custom Strategies, Filtering, Replay
// ============================================================================

test "Custom execution strategy" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Custom Strategy Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Input", null);
    _ = try net.addPlace("p2", "Output A", null);
    _ = try net.addPlace("p3", "Output B", null);
    
    _ = try net.addTransition("t_a", "Option A", 0);
    _ = try net.addTransition("t_b", "Option B", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t_a");
    _ = try net.addArc("a2", .output, 1, "t_a", "p2");
    _ = try net.addArc("a3", .input, 1, "p1", "t_b");
    _ = try net.addArc("a4", .output, 1, "t_b", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    // Custom strategy: always select first transition that starts with "t_b"
    const customStrategy = struct {
        fn selectTransition(enabled: [][]const u8) []const u8 {
            for (enabled) |trans_id| {
                if (trans_id.len >= 3 and trans_id[0] == 't' and trans_id[1] == '_' and trans_id[2] == 'b') {
                    return trans_id;
                }
            }
            return enabled[0];
        }
    }.selectTransition;
    
    executor.setCustomStrategy(customStrategy);
    
    _ = try executor.step();
    
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 0), marking.get("p2"));
    try std.testing.expectEqual(@as(usize, 1), marking.get("p3"));
}

test "Event filtering - only errors" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Filter Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Empty", null);
    _ = try net.addTransition("t1", "Blocked", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    // Set filter to only allow errors
    executor.setEventFilter(EventFilter.onlyErrors());
    
    _ = try executor.step();
    
    // Only deadlock event should be in history
    try std.testing.expectEqual(@as(usize, 1), executor.execution_history.items.len);
    try std.testing.expect(executor.execution_history.items[0] == .deadlock_detected);
}

test "Event filtering - only important" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Filter Important Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    executor.setEventFilter(EventFilter.onlyImportant());
    
    try executor.run(10);
    
    // Should have: execution_started, transition_fired, execution_completed
    // Should NOT have: state_changed events
    var has_state_changed = false;
    for (executor.execution_history.items) |event| {
        if (event == .state_changed) {
            has_state_changed = true;
            break;
        }
    }
    try std.testing.expect(!has_state_changed);
}

test "Performance metrics collection" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Perf Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "Middle", null);
    _ = try net.addPlace("p3", "End", null);
    
    _ = try net.addTransition("t1", "Step 1", 0);
    _ = try net.addTransition("t2", "Step 2", 0);
    
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    _ = try net.addArc("a3", .input, 1, "p2", "t2");
    _ = try net.addArc("a4", .output, 1, "t2", "p3");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.runUntilComplete();
    
    const stats = executor.getStats();
    
    // Should have timing metrics
    try std.testing.expect(stats.avg_transition_fire_time_ns > 0);
    try std.testing.expect(stats.total_fire_time_ns > 0);
    try std.testing.expectEqual(@as(usize, 2), stats.transitions_fired);
}

test "Execution replay" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Replay Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.run(10);
    
    const original_count = executor.execution_history.items.len;
    
    // Replay history
    const replay = try executor.replayHistory(allocator);
    defer allocator.free(replay);
    
    try std.testing.expectEqual(original_count, replay.len);
    
    // Verify events are copied correctly
    for (replay, 0..) |event, i| {
        const original_type = executor.execution_history.items[i].getType();
        const replay_type = event.getType();
        try std.testing.expect(std.mem.eql(u8, original_type, replay_type));
    }
}

test "Metrics export to JSON" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "JSON Export Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.run(10);
    
    const json = try executor.exportMetrics(allocator);
    defer allocator.free(json);
    
    // Verify JSON contains expected fields
    try std.testing.expect(std.mem.indexOf(u8, json, "total_steps") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "transitions_fired") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "avg_transition_fire_time_ns") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "strategy") != null);
}

test "Clear execution history" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Clear History Test");
    defer net.deinit();
    
    _ = try net.addPlace("p1", "Start", null);
    _ = try net.addPlace("p2", "End", null);
    _ = try net.addTransition("t1", "Process", 0);
    _ = try net.addArc("a1", .input, 1, "p1", "t1");
    _ = try net.addArc("a2", .output, 1, "t1", "p2");
    
    try net.addTokenToPlace("p1", "{}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .sequential);
    defer executor.deinit();
    
    try executor.run(10);
    
    try std.testing.expect(executor.execution_history.items.len > 0);
    
    executor.clearHistory();
    
    try std.testing.expectEqual(@as(usize, 0), executor.execution_history.items.len);
}

test "Execution strategy descriptions" {
    try std.testing.expect(ExecutionStrategy.sequential.description().len > 0);
    try std.testing.expect(ExecutionStrategy.concurrent.description().len > 0);
    try std.testing.expect(ExecutionStrategy.priority_based.description().len > 0);
    try std.testing.expect(ExecutionStrategy.custom.description().len > 0);
}

test "Event type identification" {
    const event1: ExecutionEvent = .{ .transition_fired = .{
        .transition_id = "t1",
        .timestamp = 12345,
        .duration_ns = 1000,
    } };
    
    const event2: ExecutionEvent = .{ .deadlock_detected = .{
        .timestamp = 12345,
    } };
    
    try std.testing.expect(std.mem.eql(u8, event1.getType(), "transition_fired"));
    try std.testing.expect(std.mem.eql(u8, event2.getType(), "deadlock_detected"));
}

test "Stats formatting" {
    const allocator = std.testing.allocator;
    
    const stats = ExecutionStats{
        .total_steps = 10,
        .transitions_fired = 8,
        .deadlocks_detected = 0,
        .events_recorded = 20,
        .avg_transition_fire_time_ns = 1500,
        .total_fire_time_ns = 12000,
    };
    
    const formatted = try stats.format(allocator);
    defer allocator.free(formatted);
    
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Total Steps: 10") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Transitions Fired: 8") != null);
}

test "Performance benchmark - sequential vs concurrent" {
    const allocator = std.testing.allocator;
    
    // Test sequential execution - simple linear workflow
    {
        var net_seq = try PetriNet.init(allocator, "Benchmark Sequential");
        defer net_seq.deinit();
        
        _ = try net_seq.addPlace("p1", "Start", null);
        _ = try net_seq.addPlace("p2", "Middle", null);
        _ = try net_seq.addPlace("p3", "End", null);
        _ = try net_seq.addTransition("t1", "Step1", 0);
        _ = try net_seq.addTransition("t2", "Step2", 0);
        _ = try net_seq.addArc("a1", .input, 1, "p1", "t1");
        _ = try net_seq.addArc("a2", .output, 1, "t1", "p2");
        _ = try net_seq.addArc("a3", .input, 1, "p2", "t2");
        _ = try net_seq.addArc("a4", .output, 1, "t2", "p3");
        
        try net_seq.addTokenToPlace("p1", "{}");
        
        var executor_seq = try PetriNetExecutor.init(allocator, &net_seq, .sequential);
        defer executor_seq.deinit();
        
        try executor_seq.runUntilComplete();
        const stats_seq = executor_seq.getStats();
        
        // Sequential should execute 2 transitions in 2 steps
        try std.testing.expectEqual(@as(usize, 2), stats_seq.total_steps);
        try std.testing.expectEqual(@as(usize, 2), stats_seq.transitions_fired);
    }
    
    // Test concurrent execution - parallel branches
    {
        var net_con = try PetriNet.init(allocator, "Benchmark Concurrent");
        defer net_con.deinit();
        
        _ = try net_con.addPlace("p1", "Input1", null);
        _ = try net_con.addPlace("p2", "Input2", null);
        _ = try net_con.addPlace("p3", "Output1", null);
        _ = try net_con.addPlace("p4", "Output2", null);
        _ = try net_con.addTransition("t1", "Proc1", 0);
        _ = try net_con.addTransition("t2", "Proc2", 0);
        _ = try net_con.addArc("a1", .input, 1, "p1", "t1");
        _ = try net_con.addArc("a2", .output, 1, "t1", "p3");
        _ = try net_con.addArc("a3", .input, 1, "p2", "t2");
        _ = try net_con.addArc("a4", .output, 1, "t2", "p4");
        
        try net_con.addTokenToPlace("p1", "{}");
        try net_con.addTokenToPlace("p2", "{}");
        
        var executor_con = try PetriNetExecutor.init(allocator, &net_con, .concurrent);
        defer executor_con.deinit();
        
        try executor_con.runUntilComplete();
        const stats_con = executor_con.getStats();
        
        // Concurrent should execute 2 transitions in 1 step
        try std.testing.expectEqual(@as(usize, 1), stats_con.total_steps);
        try std.testing.expectEqual(@as(usize, 2), stats_con.transitions_fired);
    }
}

test "Integration test - complex workflow with all features" {
    const allocator = std.testing.allocator;
    
    var net = try PetriNet.init(allocator, "Complex Integration");
    defer net.deinit();
    
    // Build a workflow: start -> process_a -> process_b -> end
    _ = try net.addPlace("start", "Start", null);
    _ = try net.addPlace("mid_a", "After A", null);
    _ = try net.addPlace("mid_b", "After B", null);
    _ = try net.addPlace("end", "End", null);
    
    _ = try net.addTransition("t_a", "Process A", 10);
    _ = try net.addTransition("t_b", "Process B", 5);
    _ = try net.addTransition("t_c", "Process C", 1);
    
    _ = try net.addArc("a1", .input, 1, "start", "t_a");
    _ = try net.addArc("a2", .output, 1, "t_a", "mid_a");
    _ = try net.addArc("a3", .input, 1, "mid_a", "t_b");
    _ = try net.addArc("a4", .output, 1, "t_b", "mid_b");
    _ = try net.addArc("a5", .input, 1, "mid_b", "t_c");
    _ = try net.addArc("a6", .output, 1, "t_c", "end");
    
    try net.addTokenToPlace("start", "{\"data\": \"test\"}");
    
    var executor = try PetriNetExecutor.init(allocator, &net, .priority_based);
    defer executor.deinit();
    
    // Set filter to only important events
    executor.setEventFilter(EventFilter.onlyImportant());
    
    // Track events
    const TestListener = struct {
        var event_count: usize = 0;
        
        fn listener(event: ExecutionEvent) void {
            _ = event;
            event_count += 1;
        }
    };
    
    TestListener.event_count = 0;
    try executor.addEventListener(TestListener.listener);
    
    // Create snapshot before execution
    var snapshot = try executor.createSnapshot();
    defer snapshot.deinit();
    
    // Execute
    try executor.runUntilComplete();
    
    // Verify completion
    var marking = try net.getCurrentMarking();
    defer marking.deinit();
    try std.testing.expectEqual(@as(usize, 1), marking.get("end"));
    
    // Verify stats
    const stats = executor.getStats();
    try std.testing.expectEqual(@as(usize, 3), stats.transitions_fired);
    try std.testing.expect(stats.avg_transition_fire_time_ns > 0);
    
    // Verify events were received
    try std.testing.expect(TestListener.event_count > 0);
    
    // Export metrics
    const json = try executor.exportMetrics(allocator);
    defer allocator.free(json);
    try std.testing.expect(json.len > 0);
    
    // Test replay
    const replay = try executor.replayHistory(allocator);
    defer allocator.free(replay);
    try std.testing.expect(replay.len > 0);
}
