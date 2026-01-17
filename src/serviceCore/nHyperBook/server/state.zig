// ============================================================================
// HyperShimmy State Management Module
// ============================================================================
// State machines, persistence, validation, and transitions
// Day 53: State Management
// ============================================================================

const std = @import("std");
const mem = std.mem;
const json = std.json;

// ============================================================================
// State Machine
// ============================================================================

/// Generic state machine implementation
pub fn StateMachine(comptime StateEnum: type, comptime EventEnum: type) type {
    return struct {
        const Self = @This();
        
        pub const TransitionFn = *const fn (StateEnum, EventEnum) ?StateEnum;
        pub const ValidationFn = *const fn (StateEnum, EventEnum) bool;
        pub const HookFn = *const fn (StateEnum, StateEnum, EventEnum) void;
        
        current_state: StateEnum,
        transition_fn: TransitionFn,
        validation_fn: ?ValidationFn = null,
        on_transition: ?HookFn = null,
        history: std.ArrayList(StateTransition),
        allocator: mem.Allocator,
        
        pub const StateTransition = struct {
            from_state: StateEnum,
            to_state: StateEnum,
            event: EventEnum,
            timestamp: i64,
        };
        
        pub fn init(
            allocator: mem.Allocator,
            initial_state: StateEnum,
            transition_fn: TransitionFn,
        ) Self {
            return .{
                .allocator = allocator,
                .current_state = initial_state,
                .transition_fn = transition_fn,
                .history = std.ArrayList(StateTransition){},
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.history.deinit(self.allocator);
        }
        
        /// Trigger an event and transition to new state if valid
        pub fn trigger(self: *Self, event: EventEnum) !bool {
            // Validate transition if validator exists
            if (self.validation_fn) |validator| {
                if (!validator(self.current_state, event)) {
                    return false;
                }
            }
            
            // Get next state
            const next_state = self.transition_fn(self.current_state, event) orelse {
                return false;
            };
            
            // Record transition
            const transition = StateTransition{
                .from_state = self.current_state,
                .to_state = next_state,
                .event = event,
                .timestamp = std.time.timestamp(),
            };
            try self.history.append(self.allocator, transition);
            
            // Call hook if exists
            if (self.on_transition) |hook| {
                hook(self.current_state, next_state, event);
            }
            
            // Update state
            self.current_state = next_state;
            return true;
        }
        
        /// Get current state
        pub fn getState(self: *const Self) StateEnum {
            return self.current_state;
        }
        
        /// Check if state is current
        pub fn isState(self: *const Self, state: StateEnum) bool {
            return self.current_state == state;
        }
        
        /// Get transition history
        pub fn getHistory(self: *const Self) []const StateTransition {
            return self.history.items;
        }
        
        /// Clear transition history
        pub fn clearHistory(self: *Self) void {
            self.history.clearRetainingCapacity();
        }
    };
}

// ============================================================================
// State Store
// ============================================================================

/// Persistent state storage
pub const StateStore = struct {
    allocator: mem.Allocator,
    states: std.StringHashMap([]const u8),
    
    pub fn init(allocator: mem.Allocator) StateStore {
        return .{
            .allocator = allocator,
            .states = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *StateStore) void {
        var it = self.states.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.states.deinit();
    }
    
    /// Save state data
    pub fn save(self: *StateStore, key: []const u8, value: []const u8) !void {
        // Check if key exists and free old key/value
        if (self.states.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value);
        }
        
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.states.put(key_copy, value_copy);
    }
    
    /// Load state data
    pub fn load(self: *const StateStore, key: []const u8) ?[]const u8 {
        return self.states.get(key);
    }
    
    /// Delete state data
    pub fn delete(self: *StateStore, key: []const u8) bool {
        if (self.states.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value);
            return true;
        }
        return false;
    }
    
    /// Check if key exists
    pub fn exists(self: *const StateStore, key: []const u8) bool {
        return self.states.contains(key);
    }
    
    /// Get all keys
    pub fn keys(self: *const StateStore, allocator: mem.Allocator) ![][]const u8 {
        var key_list = std.ArrayList([]const u8){};
        errdefer key_list.deinit(allocator);
        
        var it = self.states.keyIterator();
        while (it.next()) |key| {
            try key_list.append(allocator, key.*);
        }
        
        return try key_list.toOwnedSlice(allocator);
    }
    
    /// Clear all state data
    pub fn clear(self: *StateStore) void {
        var it = self.states.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.states.clearRetainingCapacity();
    }
};

// ============================================================================
// State Validator
// ============================================================================

/// State validation utilities
pub const StateValidator = struct {
    pub const ValidationError = error{
        InvalidState,
        InvalidTransition,
        RequiredFieldMissing,
        InvalidFieldValue,
    };
    
    /// Validate state transition
    pub fn validateTransition(
        comptime StateEnum: type,
        current: StateEnum,
        next: StateEnum,
        allowed_transitions: []const struct { from: StateEnum, to: StateEnum },
    ) bool {
        for (allowed_transitions) |transition| {
            if (transition.from == current and transition.to == next) {
                return true;
            }
        }
        return false;
    }
    
    /// Validate state data structure
    pub fn validateStruct(comptime T: type, data: T) !void {
        const fields = @typeInfo(T).Struct.fields;
        inline for (fields) |field| {
            const value = @field(data, field.name);
            
            // Check for null in optional fields that shouldn't be null
            if (@typeInfo(field.type) == .Optional) {
                // Optional fields are allowed to be null
                continue;
            }
            
            // For slices, check they're not empty if required
            if (@typeInfo(field.type) == .Pointer) {
                if (value.len == 0) {
                    return ValidationError.RequiredFieldMissing;
                }
            }
        }
    }
};

// ============================================================================
// State Snapshot
// ============================================================================

/// State snapshot for backup and restore
pub const StateSnapshot = struct {
    allocator: mem.Allocator,
    data: std.StringHashMap([]const u8),
    timestamp: i64,
    label: []const u8,
    
    pub fn init(allocator: mem.Allocator, label: []const u8) !StateSnapshot {
        return .{
            .allocator = allocator,
            .data = std.StringHashMap([]const u8).init(allocator),
            .timestamp = std.time.timestamp(),
            .label = try allocator.dupe(u8, label),
        };
    }
    
    pub fn deinit(self: *StateSnapshot) void {
        var it = self.data.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.data.deinit();
        self.allocator.free(self.label);
    }
    
    /// Add state to snapshot
    pub fn addState(self: *StateSnapshot, key: []const u8, value: []const u8) !void {
        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);
        
        try self.data.put(key_copy, value_copy);
    }
    
    /// Restore snapshot to state store
    pub fn restore(self: *const StateSnapshot, store: *StateStore) !void {
        var it = self.data.iterator();
        while (it.next()) |entry| {
            try store.save(entry.key_ptr.*, entry.value_ptr.*);
        }
    }
};

// ============================================================================
// State Manager
// ============================================================================

/// High-level state management
pub const StateManager = struct {
    allocator: mem.Allocator,
    store: StateStore,
    snapshots: std.ArrayList(StateSnapshot),
    max_snapshots: usize,
    
    pub fn init(allocator: mem.Allocator, max_snapshots: usize) StateManager {
        return .{
            .allocator = allocator,
            .store = StateStore.init(allocator),
            .snapshots = std.ArrayList(StateSnapshot){},
            .max_snapshots = max_snapshots,
        };
    }
    
    pub fn deinit(self: *StateManager) void {
        for (self.snapshots.items) |*snapshot| {
            snapshot.deinit();
        }
        self.snapshots.deinit(self.allocator);
        self.store.deinit();
    }
    
    /// Create a snapshot of current state
    pub fn createSnapshot(self: *StateManager, label: []const u8) !void {
        var snapshot = try StateSnapshot.init(self.allocator, label);
        errdefer snapshot.deinit();
        
        // Copy all current state
        const all_keys = try self.store.keys(self.allocator);
        defer self.allocator.free(all_keys);
        
        for (all_keys) |key| {
            if (self.store.load(key)) |value| {
                try snapshot.addState(key, value);
            }
        }
        
        try self.snapshots.append(self.allocator, snapshot);
        
        // Remove oldest snapshot if limit exceeded
        if (self.snapshots.items.len > self.max_snapshots) {
            var old = self.snapshots.orderedRemove(0);
            old.deinit();
        }
    }
    
    /// Restore from latest snapshot
    pub fn restoreLatest(self: *StateManager) !bool {
        if (self.snapshots.items.len == 0) return false;
        
        const latest = &self.snapshots.items[self.snapshots.items.len - 1];
        try latest.restore(&self.store);
        return true;
    }
    
    /// Restore from named snapshot
    pub fn restoreSnapshot(self: *StateManager, label: []const u8) !bool {
        for (self.snapshots.items) |*snapshot| {
            if (mem.eql(u8, snapshot.label, label)) {
                try snapshot.restore(&self.store);
                return true;
            }
        }
        return false;
    }
    
    /// Get snapshot count
    pub fn snapshotCount(self: *const StateManager) usize {
        return self.snapshots.items.len;
    }
};

// ============================================================================
// Tests
// ============================================================================

// Example state and event enums for testing
const TestState = enum {
    idle,
    processing,
    completed,
    failed,
};

const TestEvent = enum {
    start,
    finish,
    fail,
    reset,
};

fn testTransitionFn(current: TestState, event: TestEvent) ?TestState {
    return switch (current) {
        .idle => switch (event) {
            .start => .processing,
            else => null,
        },
        .processing => switch (event) {
            .finish => .completed,
            .fail => .failed,
            else => null,
        },
        .completed, .failed => switch (event) {
            .reset => .idle,
            else => null,
        },
    };
}

test "state machine transitions" {
    const allocator = std.testing.allocator;
    var sm = StateMachine(TestState, TestEvent).init(allocator, .idle, testTransitionFn);
    defer sm.deinit();
    
    try std.testing.expect(sm.isState(.idle));
    
    _ = try sm.trigger(.start);
    try std.testing.expect(sm.isState(.processing));
    
    _ = try sm.trigger(.finish);
    try std.testing.expect(sm.isState(.completed));
    
    try std.testing.expect(sm.getHistory().len == 2);
}

test "state store" {
    const allocator = std.testing.allocator;
    var store = StateStore.init(allocator);
    defer store.deinit();
    
    try store.save("key1", "value1");
    try store.save("key2", "value2");
    
    try std.testing.expect(store.exists("key1"));
    try std.testing.expectEqualStrings("value1", store.load("key1").?);
    
    _ = store.delete("key1");
    try std.testing.expect(!store.exists("key1"));
}

test "state snapshot" {
    const allocator = std.testing.allocator;
    var store = StateStore.init(allocator);
    defer store.deinit();
    
    try store.save("key1", "value1");
    
    var snapshot = try StateSnapshot.init(allocator, "backup1");
    defer snapshot.deinit();
    
    try snapshot.addState("key1", "value1");
    
    store.clear();
    try std.testing.expect(!store.exists("key1"));
    
    try snapshot.restore(&store);
    try std.testing.expect(store.exists("key1"));
}

test "state manager" {
    const allocator = std.testing.allocator;
    var manager = StateManager.init(allocator, 5);
    defer manager.deinit();
    
    try manager.store.save("key1", "value1");
    try manager.createSnapshot("snapshot1");
    
    try std.testing.expectEqual(@as(usize, 1), manager.snapshotCount());
    
    // After restore, key1 should have snapshot value
    try manager.store.save("key1", "value2");
    _ = try manager.restoreLatest();
    try std.testing.expectEqualStrings("value1", manager.store.load("key1").?);
}
