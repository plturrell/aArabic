//! HANA-based Memory & State Management for nWorkflow
//!
//! Unified state persistence and caching using SAP HANA.
//! Replaces the previous PostgreSQL + DragonflyDB architecture.

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// Import HANA cache
const HanaCache = @import("hana_cache").HanaCache;
const HanaCacheConfig = @import("hana_cache").HanaCacheConfig;

// Import HANA persistence
const hana_store = @import("hana_store");
const HanaWorkflowStore = hana_store.HanaWorkflowStore;
const hana = @import("hana_sdk");
const HanaConfig = hana.Config;

/// HANA state manager configuration
pub const HanaStateConfig = struct {
    host: []const u8,
    port: u16 = 443,
    user: []const u8,
    password: []const u8,
    database: []const u8 = "NWORKFLOW",
    cache_database: []const u8 = "NWORKFLOW_CACHE",
    table_prefix: []const u8 = "nworkflow",
    cache_table_prefix: []const u8 = "cache",
};

/// Variable scope for state management
pub const VariableScope = union(enum) {
    global: void,
    workflow: []const u8, // workflow_id
    session: []const u8, // session_id
    user: []const u8, // user_id from Keycloak

    pub fn toKey(self: VariableScope, allocator: Allocator, var_key: []const u8) ![]const u8 {
        return switch (self) {
            .global => try std.fmt.allocPrint(allocator, "global:{s}", .{var_key}),
            .workflow => |wf_id| try std.fmt.allocPrint(allocator, "workflow:{s}:{s}", .{ wf_id, var_key }),
            .session => |sess_id| try std.fmt.allocPrint(allocator, "session:{s}:{s}", .{ sess_id, var_key }),
            .user => |user_id| try std.fmt.allocPrint(allocator, "user:{s}:{s}", .{ user_id, var_key }),
        };
    }
};

/// Workflow state snapshot for persistence
pub const WorkflowState = struct {
    workflow_id: []const u8,
    execution_id: []const u8,
    marking: Marking,
    timestamp: i64,
    metadata: StringHashMap([]const u8),

    pub fn init(allocator: Allocator, workflow_id: []const u8, execution_id: []const u8, marking: Marking) !WorkflowState {
        return .{
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .execution_id = try allocator.dupe(u8, execution_id),
            .marking = marking,
            .timestamp = std.time.timestamp(),
            .metadata = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *WorkflowState, allocator: Allocator) void {
        allocator.free(self.workflow_id);
        allocator.free(self.execution_id);
        var iter = self.metadata.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        self.metadata.deinit();
    }

    pub fn serialize(self: *const WorkflowState, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{{\"workflow_id\":\"{s}\",\"execution_id\":\"{s}\",\"timestamp\":{d}}}", .{
            self.workflow_id,
            self.execution_id,
            self.timestamp,
        });
    }

    pub fn deserialize(allocator: Allocator, data: []const u8) !WorkflowState {
        _ = allocator;
        _ = data;
        return error.NotImplemented;
    }
};

/// Mock Marking structure (would import from petri_net.zig)
pub const Marking = struct {
    places: StringHashMap(usize),

    pub fn init(allocator: Allocator) Marking {
        return .{
            .places = StringHashMap(usize).init(allocator),
        };
    }

    pub fn deinit(self: *Marking) void {
        var iter = self.places.iterator();
        while (iter.next()) |entry| {
            self.places.allocator.free(entry.key_ptr.*);
        }
        self.places.deinit();
    }
};

/// State recovery information
pub const RecoveryPoint = struct {
    workflow_id: []const u8,
    execution_id: []const u8,
    checkpoint_id: []const u8,
    timestamp: i64,
    state_data: []const u8,

    pub fn init(allocator: Allocator, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8, state_data: []const u8) !RecoveryPoint {
        return .{
            .workflow_id = try allocator.dupe(u8, workflow_id),
            .execution_id = try allocator.dupe(u8, execution_id),
            .checkpoint_id = try allocator.dupe(u8, checkpoint_id),
            .timestamp = std.time.timestamp(),
            .state_data = try allocator.dupe(u8, state_data),
        };
    }

    pub fn deinit(self: *RecoveryPoint, allocator: Allocator) void {
        allocator.free(self.workflow_id);
        allocator.free(self.execution_id);
        allocator.free(self.checkpoint_id);
        allocator.free(self.state_data);
    }
};

/// HANA-based state manager for nWorkflow
pub const StateManager = struct {
    allocator: Allocator,
    cache: HanaCache,
    store: *HanaWorkflowStore,
    local_variables: StringHashMap([]const u8),

    pub fn init(allocator: Allocator, config: HanaStateConfig) !StateManager {
        // Initialize HANA cache
        const cache_config = HanaCacheConfig{
            .host = config.host,
            .port = config.port,
            .user = config.user,
            .password = config.password,
            .database = config.cache_database,
            .table_prefix = config.cache_table_prefix,
        };

        var cache = try HanaCache.init(allocator, cache_config);
        try cache.connect();

        // Initialize HANA persistence store
        const hana_config = HanaConfig{
            .host = config.host,
            .port = config.port,
            .user = config.user,
            .password = config.password,
            .database = config.database,
        };

        const store = try HanaWorkflowStore.init(allocator, hana_config, config.table_prefix);

        return .{
            .allocator = allocator,
            .cache = cache,
            .store = store,
            .local_variables = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *StateManager) void {
        self.cache.deinit();
        self.store.deinit();
        
        var iter = self.local_variables.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.local_variables.deinit();
    }

    // === Workflow State Persistence (HANA) ===

    pub fn saveWorkflowState(self: *StateManager, state: *const WorkflowState) !void {
        const serialized = try state.serialize(self.allocator);
        defer self.allocator.free(serialized);

        // Cache the state for fast access
        try self.cache.cacheWorkflowState(state.workflow_id, serialized, 3600);
    }

    pub fn loadWorkflowState(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !WorkflowState {
        _ = execution_id;
        
        // Try cache first
        if (try self.cache.getWorkflowState(workflow_id)) |data| {
            return WorkflowState.deserialize(self.allocator, data);
        }

        return error.StateNotFound;
    }

    pub fn deleteWorkflowState(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !void {
        _ = execution_id;
        try self.cache.deleteWorkflowState(workflow_id);
    }

    // === Session Management (HANA Cache) ===

    pub fn saveSession(self: *StateManager, session_id: []const u8, data: []const u8, ttl: u32) !void {
        try self.cache.storeSession(session_id, data, ttl);
    }

    pub fn loadSession(self: *StateManager, session_id: []const u8) !?[]const u8 {
        return try self.cache.getSession(session_id);
    }

    pub fn deleteSession(self: *StateManager, session_id: []const u8) !void {
        try self.cache.deleteSession(session_id);
    }

    pub fn extendSession(self: *StateManager, session_id: []const u8, additional_ttl: u32) !void {
        if (try self.loadSession(session_id)) |data| {
            try self.saveSession(session_id, data, additional_ttl);
        } else {
            return error.SessionNotFound;
        }
    }

    // === Variable Storage (Scoped, HANA Cache) ===

    pub fn setVariable(self: *StateManager, scope: VariableScope, key: []const u8, value: []const u8) !void {
        const scoped_key = try scope.toKey(self.allocator, key);
        defer self.allocator.free(scoped_key);

        // Store in HANA cache
        try self.cache.set(scoped_key, value, null);

        // Also keep in local cache
        const local_key = try self.allocator.dupe(u8, scoped_key);
        errdefer self.allocator.free(local_key);

        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        // Remove old value if exists
        if (self.local_variables.get(scoped_key)) |old_value| {
            self.allocator.free(old_value);
        }

        try self.local_variables.put(local_key, value_copy);
    }

    pub fn getVariable(self: *StateManager, scope: VariableScope, key: []const u8) !?[]const u8 {
        const scoped_key = try scope.toKey(self.allocator, key);
        defer self.allocator.free(scoped_key);

        // Check local cache first
        if (self.local_variables.get(scoped_key)) |value| {
            return value;
        }

        // Fall back to HANA cache
        if (try self.cache.get(scoped_key)) |value| {
            // Cache locally
            const local_key = try self.allocator.dupe(u8, scoped_key);
            const local_value = try self.allocator.dupe(u8, value);
            try self.local_variables.put(local_key, local_value);
            return local_value;
        }

        return null;
    }

    pub fn deleteVariable(self: *StateManager, scope: VariableScope, key: []const u8) !void {
        const scoped_key = try scope.toKey(self.allocator, key);
        defer self.allocator.free(scoped_key);

        // Remove from HANA cache
        _ = try self.cache.del(scoped_key);

        // Remove from local cache
        if (self.local_variables.fetchRemove(scoped_key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value);
        }
    }

    pub fn listVariables(self: *StateManager, scope: VariableScope) !std.ArrayList([]const u8) {
        const StrList = std.ArrayList([]const u8);
        var result = StrList.init(self.allocator);
        errdefer result.deinit();

        const prefix = try scope.toKey(self.allocator, "");
        defer self.allocator.free(prefix);

        // List from local cache
        var iter = self.local_variables.iterator();
        while (iter.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                if (std.mem.lastIndexOf(u8, entry.key_ptr.*, ":")) |last_colon| {
                    const var_name = entry.key_ptr.*[last_colon + 1 ..];
                    const name_copy = try self.allocator.dupe(u8, var_name);
                    try result.append(name_copy);
                }
            }
        }

        return result;
    }

    // === State Recovery (HANA Cache) ===

    pub fn createCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8, state: *const WorkflowState) !void {
        const serialized = try state.serialize(self.allocator);
        defer self.allocator.free(serialized);

        const key = try std.fmt.allocPrint(
            self.allocator,
            "checkpoint:{s}:{s}:{s}",
            .{ workflow_id, execution_id, checkpoint_id },
        );
        defer self.allocator.free(key);

        // Store checkpoint in cache with 24 hour TTL
        try self.cache.set(key, serialized, 86400);
    }

    pub fn loadCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8) !WorkflowState {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "checkpoint:{s}:{s}:{s}",
            .{ workflow_id, execution_id, checkpoint_id },
        );
        defer self.allocator.free(key);

        if (try self.cache.get(key)) |data| {
            return WorkflowState.deserialize(self.allocator, data);
        }

        return error.CheckpointNotFound;
    }

    pub fn listCheckpoints(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !std.ArrayList([]const u8) {
        _ = workflow_id;
        _ = execution_id;
        
        const StrList = std.ArrayList([]const u8);
        return StrList.init(self.allocator);
    }

    pub fn deleteCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8) !void {
        const key = try std.fmt.allocPrint(
            self.allocator,
            "checkpoint:{s}:{s}:{s}",
            .{ workflow_id, execution_id, checkpoint_id },
        );
        defer self.allocator.free(key);

        _ = try self.cache.del(key);
    }

    // === Utility Methods ===

    pub fn clearCache(self: *StateManager) void {
        var iter = self.local_variables.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.local_variables.clearRetainingCapacity();
    }

    pub fn getStats(self: *const StateManager) StateStats {
        return .{
            .hana_connected = self.cache.is_connected,
            .local_variable_count = self.local_variables.count(),
        };
    }

    /// Cleanup expired cache entries
    pub fn cleanupExpired(self: *StateManager) !void {
        try self.cache.cleanupExpired();
    }
};

pub const StateStats = struct {
    hana_connected: bool,
    local_variable_count: usize,
};

// === Tests ===

test "StateManager initialization" {
    const allocator = std.testing.allocator;

    const config = HanaStateConfig{
        .host = "localhost",
        .port = 39017,
        .user = "SYSTEM",
        .password = "Password123",
        .database = "STATE",
        .cache_database = "STATE_CACHE",
    };

    var manager = try StateManager.init(allocator, config);
    defer manager.deinit();

    const stats = manager.getStats();
    try std.testing.expect(stats.hana_connected);
    try std.testing.expectEqual(@as(usize, 0), stats.local_variable_count);
}

test "Variable scopes" {
    const allocator = std.testing.allocator;

    const global = VariableScope{ .global = {} };
    const global_key = try global.toKey(allocator, "config");
    defer allocator.free(global_key);
    try std.testing.expectEqualStrings("global:config", global_key);

    const workflow = VariableScope{ .workflow = "wf123" };
    const wf_key = try workflow.toKey(allocator, "status");
    defer allocator.free(wf_key);
    try std.testing.expectEqualStrings("workflow:wf123:status", wf_key);
}

test "WorkflowState serialization" {
    const allocator = std.testing.allocator;

    var marking = Marking.init(allocator);
    defer marking.deinit();

    var state = try WorkflowState.init(allocator, "wf123", "exec456", marking);
    defer state.deinit(allocator);

    const serialized = try state.serialize(allocator);
    defer allocator.free(serialized);

    try std.testing.expect(serialized.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "wf123") != null);
}