// Day 25: Memory & State Management
// State persistence, session cache, and variable storage

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// Mock database connections (will be replaced with real implementations in Phase 3)
pub const PostgresConnection = struct {
    allocator: Allocator,
    connected: bool,

    pub fn init(allocator: Allocator, _: PostgresConfig) !PostgresConnection {
        return .{
            .allocator = allocator,
            .connected = true,
        };
    }

    pub fn deinit(self: *PostgresConnection) void {
        self.connected = false;
    }

    pub fn execute(self: *PostgresConnection, sql: []const u8, params: []const []const u8) !void {
        if (!self.connected) return error.NotConnected;
        _ = sql;
        _ = params;
        // Mock implementation
    }

    pub fn query(self: *PostgresConnection, sql: []const u8, params: []const []const u8) ![]const u8 {
        if (!self.connected) return error.NotConnected;
        _ = sql;
        _ = params;
        // Mock implementation - return empty JSON array
        return "[]";
    }
};

pub const DragonflyConnection = struct {
    allocator: Allocator,
    connected: bool,
    cache: StringHashMap(CacheEntry),

    const CacheEntry = struct {
        value: []const u8,
        expires_at: i64,
    };

    pub fn init(allocator: Allocator, _: DragonflyConfig) !DragonflyConnection {
        return .{
            .allocator = allocator,
            .connected = true,
            .cache = StringHashMap(CacheEntry).init(allocator),
        };
    }

    pub fn deinit(self: *DragonflyConnection) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.value);
        }
        self.cache.deinit();
        self.connected = false;
    }

    pub fn set(self: *DragonflyConnection, key: []const u8, value: []const u8, ttl: u32) !void {
        if (!self.connected) return error.NotConnected;

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);

        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        const now = std.time.timestamp();
        const expires_at = if (ttl > 0) now + @as(i64, ttl) else std.math.maxInt(i64);

        // Remove old entry if exists
        if (self.cache.get(key)) |old_entry| {
            self.allocator.free(old_entry.value);
        }

        try self.cache.put(key_copy, .{
            .value = value_copy,
            .expires_at = expires_at,
        });
    }

    pub fn get(self: *DragonflyConnection, key: []const u8) !?[]const u8 {
        if (!self.connected) return error.NotConnected;

        if (self.cache.get(key)) |entry| {
            const now = std.time.timestamp();
            if (now >= entry.expires_at) {
                // Expired
                return null;
            }
            return entry.value;
        }
        return null;
    }

    pub fn delete(self: *DragonflyConnection, key: []const u8) !void {
        if (!self.connected) return error.NotConnected;

        if (self.cache.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value.value);
        }
    }
};

pub const PostgresConfig = struct {
    host: []const u8,
    port: u16,
    database: []const u8,
    username: []const u8,
    password: []const u8,

    pub fn fromConnectionString(allocator: Allocator, conn_str: []const u8) !PostgresConfig {
        _ = allocator;
        _ = conn_str;
        // Mock implementation
        return .{
            .host = "localhost",
            .port = 5432,
            .database = "nworkflow",
            .username = "nworkflow",
            .password = "secret",
        };
    }
};

pub const DragonflyConfig = struct {
    host: []const u8,
    port: u16,
    password: ?[]const u8,
    db: u32,

    pub fn default() DragonflyConfig {
        return .{
            .host = "localhost",
            .port = 6379,
            .password = null,
            .db = 0,
        };
    }
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
        // Simple JSON serialization
        return std.fmt.allocPrint(allocator, "{{\"workflow_id\":\"{s}\",\"execution_id\":\"{s}\",\"timestamp\":{d}}}", .{
            self.workflow_id,
            self.execution_id,
            self.timestamp,
        });
    }

    pub fn deserialize(allocator: Allocator, data: []const u8) !WorkflowState {
        _ = allocator;
        _ = data;
        // Mock implementation - would parse JSON
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

/// Main state manager for nWorkflow
pub const StateManager = struct {
    allocator: Allocator,
    postgres_conn: PostgresConnection,
    dragonfly_conn: DragonflyConnection,
    local_variables: StringHashMap([]const u8),

    pub fn init(allocator: Allocator, postgres_config: PostgresConfig, dragonfly_config: DragonflyConfig) !StateManager {
        return .{
            .allocator = allocator,
            .postgres_conn = try PostgresConnection.init(allocator, postgres_config),
            .dragonfly_conn = try DragonflyConnection.init(allocator, dragonfly_config),
            .local_variables = StringHashMap([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *StateManager) void {
        self.postgres_conn.deinit();
        self.dragonfly_conn.deinit();
        var iter = self.local_variables.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.local_variables.deinit();
    }

    // === Workflow State Persistence (PostgreSQL) ===

    pub fn saveWorkflowState(self: *StateManager, state: *const WorkflowState) !void {
        const serialized = try state.serialize(self.allocator);
        defer self.allocator.free(serialized);

        const query = "INSERT INTO workflow_states (workflow_id, execution_id, state_data, timestamp) VALUES ($1, $2, $3, $4)";
        const timestamp_str = try std.fmt.allocPrint(self.allocator, "{d}", .{state.timestamp});
        defer self.allocator.free(timestamp_str);

        const params = [_][]const u8{ state.workflow_id, state.execution_id, serialized, timestamp_str };
        try self.postgres_conn.execute(query, &params);
    }

    pub fn loadWorkflowState(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !WorkflowState {
        const query = "SELECT state_data FROM workflow_states WHERE workflow_id = $1 AND execution_id = $2 ORDER BY timestamp DESC LIMIT 1";
        const params = [_][]const u8{ workflow_id, execution_id };

        const result = try self.postgres_conn.query(query, &params);
        defer self.allocator.free(result);

        if (result.len == 2) { // Empty array "[]"
            return error.StateNotFound;
        }

        return WorkflowState.deserialize(self.allocator, result);
    }

    pub fn deleteWorkflowState(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !void {
        const query = "DELETE FROM workflow_states WHERE workflow_id = $1 AND execution_id = $2";
        const params = [_][]const u8{ workflow_id, execution_id };
        try self.postgres_conn.execute(query, &params);
    }

    // === Session Management (DragonflyDB) ===

    pub fn saveSession(self: *StateManager, session_id: []const u8, data: []const u8, ttl: u32) !void {
        const key = try std.fmt.allocPrint(self.allocator, "session:{s}", .{session_id});
        defer self.allocator.free(key);

        try self.dragonfly_conn.set(key, data, ttl);
    }

    pub fn loadSession(self: *StateManager, session_id: []const u8) !?[]const u8 {
        const key = try std.fmt.allocPrint(self.allocator, "session:{s}", .{session_id});
        defer self.allocator.free(key);

        return try self.dragonfly_conn.get(key);
    }

    pub fn deleteSession(self: *StateManager, session_id: []const u8) !void {
        const key = try std.fmt.allocPrint(self.allocator, "session:{s}", .{session_id});
        defer self.allocator.free(key);

        try self.dragonfly_conn.delete(key);
    }

    pub fn extendSession(self: *StateManager, session_id: []const u8, additional_ttl: u32) !void {
        // Load current session data
        if (try self.loadSession(session_id)) |data| {
            // Re-save with new TTL
            try self.saveSession(session_id, data, additional_ttl);
        } else {
            return error.SessionNotFound;
        }
    }

    // === Variable Storage (Scoped) ===

    pub fn setVariable(self: *StateManager, scope: VariableScope, key: []const u8, value: []const u8) !void {
        const scoped_key = try scope.toKey(self.allocator, key);
        defer self.allocator.free(scoped_key);

        // Store in DragonflyDB for fast access
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        try self.dragonfly_conn.set(scoped_key, value, 0); // No TTL

        // Also keep in local cache
        const local_key = try self.allocator.dupe(u8, scoped_key);
        errdefer self.allocator.free(local_key);

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

        // Fall back to DragonflyDB
        if (try self.dragonfly_conn.get(scoped_key)) |value| {
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

        // Remove from DragonflyDB
        try self.dragonfly_conn.delete(scoped_key);

        // Remove from local cache
        if (self.local_variables.fetchRemove(scoped_key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value);
        }
    }

    pub fn listVariables(self: *StateManager, scope: VariableScope) !std.ArrayList([]const u8) {
        const StrList = std.ArrayList([]const u8);
        var result = StrList{};
        errdefer result.deinit(self.allocator);

        const prefix = try scope.toKey(self.allocator, "");
        defer self.allocator.free(prefix);

        // List from local cache (in production, would scan DragonflyDB)
        var iter = self.local_variables.iterator();
        while (iter.next()) |entry| {
            if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
                // Extract variable name (after last ':')
                if (std.mem.lastIndexOf(u8, entry.key_ptr.*, ":")) |last_colon| {
                    const var_name = entry.key_ptr.*[last_colon + 1 ..];
                    const name_copy = try self.allocator.dupe(u8, var_name);
                    try result.append(self.allocator, name_copy);
                }
            }
        }

        return result;
    }

    // === State Recovery ===

    pub fn createCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8, state: *const WorkflowState) !void {
        const serialized = try state.serialize(self.allocator);
        defer self.allocator.free(serialized);

        var recovery_point = try RecoveryPoint.init(self.allocator, workflow_id, execution_id, checkpoint_id, serialized);
        defer recovery_point.deinit(self.allocator);

        const query = "INSERT INTO recovery_points (workflow_id, execution_id, checkpoint_id, state_data, timestamp) VALUES ($1, $2, $3, $4, $5)";
        const timestamp_str = try std.fmt.allocPrint(self.allocator, "{d}", .{recovery_point.timestamp});
        defer self.allocator.free(timestamp_str);

        const params = [_][]const u8{ workflow_id, execution_id, checkpoint_id, serialized, timestamp_str };
        try self.postgres_conn.execute(query, &params);
    }

    pub fn loadCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8) !WorkflowState {
        const query = "SELECT state_data FROM recovery_points WHERE workflow_id = $1 AND execution_id = $2 AND checkpoint_id = $3";
        const params = [_][]const u8{ workflow_id, execution_id, checkpoint_id };

        const result = try self.postgres_conn.query(query, &params);
        defer self.allocator.free(result);

        if (result.len == 2) { // Empty array "[]"
            return error.CheckpointNotFound;
        }

        return WorkflowState.deserialize(self.allocator, result);
    }

    pub fn listCheckpoints(self: *StateManager, workflow_id: []const u8, execution_id: []const u8) !std.ArrayList([]const u8) {
        const query = "SELECT checkpoint_id FROM recovery_points WHERE workflow_id = $1 AND execution_id = $2 ORDER BY timestamp DESC";
        const params = [_][]const u8{ workflow_id, execution_id };

        const result = try self.postgres_conn.query(query, &params);
        defer self.allocator.free(result);

        const StrList = std.ArrayList([]const u8);
        const checkpoints = StrList{};
        // Mock: would parse JSON array
        return checkpoints;
    }

    pub fn deleteCheckpoint(self: *StateManager, workflow_id: []const u8, execution_id: []const u8, checkpoint_id: []const u8) !void {
        const query = "DELETE FROM recovery_points WHERE workflow_id = $1 AND execution_id = $2 AND checkpoint_id = $3";
        const params = [_][]const u8{ workflow_id, execution_id, checkpoint_id };
        try self.postgres_conn.execute(query, &params);
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
            .postgres_connected = self.postgres_conn.connected,
            .dragonfly_connected = self.dragonfly_conn.connected,
            .local_variable_count = self.local_variables.count(),
            .cache_entry_count = self.dragonfly_conn.cache.count(),
        };
    }
};

pub const StateStats = struct {
    postgres_connected: bool,
    dragonfly_connected: bool,
    local_variable_count: usize,
    cache_entry_count: usize,
};

// === Tests ===

test "StateManager initialization" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    const stats = manager.getStats();
    try std.testing.expect(stats.postgres_connected);
    try std.testing.expect(stats.dragonfly_connected);
    try std.testing.expectEqual(@as(usize, 0), stats.local_variable_count);
}

test "Session management" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    // Save session
    try manager.saveSession("sess123", "{\"user\":\"alice\"}", 3600);

    // Load session
    const data = try manager.loadSession("sess123");
    try std.testing.expect(data != null);
    try std.testing.expectEqualStrings("{\"user\":\"alice\"}", data.?);

    // Delete session
    try manager.deleteSession("sess123");
    const deleted = try manager.loadSession("sess123");
    try std.testing.expect(deleted == null);
}

test "Variable scopes" {
    const allocator = std.testing.allocator;

    // Global scope
    const global = VariableScope{ .global = {} };
    const global_key = try global.toKey(allocator, "config");
    defer allocator.free(global_key);
    try std.testing.expectEqualStrings("global:config", global_key);

    // Workflow scope
    const workflow = VariableScope{ .workflow = "wf123" };
    const wf_key = try workflow.toKey(allocator, "status");
    defer allocator.free(wf_key);
    try std.testing.expectEqualStrings("workflow:wf123:status", wf_key);

    // Session scope
    const session = VariableScope{ .session = "sess456" };
    const sess_key = try session.toKey(allocator, "temp");
    defer allocator.free(sess_key);
    try std.testing.expectEqualStrings("session:sess456:temp", sess_key);

    // User scope
    const user = VariableScope{ .user = "user789" };
    const user_key = try user.toKey(allocator, "preference");
    defer allocator.free(user_key);
    try std.testing.expectEqualStrings("user:user789:preference", user_key);
}

test "Variable storage and retrieval" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    // Set global variable
    const global_scope = VariableScope{ .global = {} };
    try manager.setVariable(global_scope, "api_key", "secret123");

    // Get global variable
    const value = try manager.getVariable(global_scope, "api_key");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("secret123", value.?);

    // Delete global variable
    try manager.deleteVariable(global_scope, "api_key");
    const deleted = try manager.getVariable(global_scope, "api_key");
    try std.testing.expect(deleted == null);
}

test "Workflow-scoped variables" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    const scope = VariableScope{ .workflow = "wf123" };

    // Set multiple variables
    try manager.setVariable(scope, "counter", "42");
    try manager.setVariable(scope, "status", "running");

    // Get variables
    const counter = try manager.getVariable(scope, "counter");
    try std.testing.expectEqualStrings("42", counter.?);

    const status = try manager.getVariable(scope, "status");
    try std.testing.expectEqualStrings("running", status.?);

    // List variables
    var vars = try manager.listVariables(scope);
    defer {
        for (vars.items) |item| {
            allocator.free(item);
        }
        vars.deinit(allocator);
    }
    try std.testing.expectEqual(@as(usize, 2), vars.items.len);
}

test "User-scoped variables isolation" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    // User 1
    const user1_scope = VariableScope{ .user = "alice" };
    try manager.setVariable(user1_scope, "theme", "dark");

    // User 2
    const user2_scope = VariableScope{ .user = "bob" };
    try manager.setVariable(user2_scope, "theme", "light");

    // Check isolation
    const alice_theme = try manager.getVariable(user1_scope, "theme");
    try std.testing.expectEqualStrings("dark", alice_theme.?);

    const bob_theme = try manager.getVariable(user2_scope, "theme");
    try std.testing.expectEqualStrings("light", bob_theme.?);
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
    try std.testing.expect(std.mem.indexOf(u8, serialized, "exec456") != null);
}

test "DragonflyDB TTL expiration" {
    const allocator = std.testing.allocator;

    const df_config = DragonflyConfig.default();
    var conn = try DragonflyConnection.init(allocator, df_config);
    defer conn.deinit();

    // Set with very short TTL (already expired)
    const past = std.time.timestamp() - 100;
    try conn.cache.put(try allocator.dupe(u8, "expired_key"), .{
        .value = try allocator.dupe(u8, "value"),
        .expires_at = past,
    });

    // Try to get expired value
    const value = try conn.get("expired_key");
    try std.testing.expect(value == null);
}

test "Cache clearing" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    // Set some variables
    const scope = VariableScope{ .global = {} };
    try manager.setVariable(scope, "key1", "value1");
    try manager.setVariable(scope, "key2", "value2");

    var stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.local_variable_count);

    // Clear cache
    manager.clearCache();

    stats = manager.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.local_variable_count);
}

test "StateStats reporting" {
    const allocator = std.testing.allocator;

    const pg_config = PostgresConfig{
        .host = "localhost",
        .port = 5432,
        .database = "test",
        .username = "test",
        .password = "test",
    };
    const df_config = DragonflyConfig.default();

    var manager = try StateManager.init(allocator, pg_config, df_config);
    defer manager.deinit();

    const stats = manager.getStats();
    try std.testing.expect(stats.postgres_connected);
    try std.testing.expect(stats.dragonfly_connected);
    try std.testing.expectEqual(@as(usize, 0), stats.local_variable_count);
    try std.testing.expectEqual(@as(usize, 0), stats.cache_entry_count);
}
