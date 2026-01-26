//! HANA In-Memory Cache for nWorkflow
//!
//! This module provides a high-performance caching layer using SAP HANA's
//! in-memory capabilities, replacing the previous DragonflyDB cache.
//!
//! Features:
//! - Key-value store with TTL support
//! - Session management
//! - Workflow state caching
//! - Pub/sub messaging (future)
//!
//! Uses the centralized HANA SDK for all database operations.

const std = @import("std");
const Allocator = std.mem.Allocator;

// Import centralized HANA SDK
const hana = @import("../data/hana_client.zig");
const HanaClient = hana.HanaClient;
const HanaConfig = hana.HanaConfig;
const QueryResult = hana.QueryResult;

/// HANA cache configuration
pub const HanaCacheConfig = struct {
    host: []const u8,
    port: u16 = 443,
    user: []const u8,
    password: []const u8,
    database: []const u8 = "NWORKFLOW_CACHE",
    table_prefix: []const u8 = "cache",
};

/// HANA-based cache client
pub const HanaCache = struct {
    allocator: Allocator,
    config: HanaCacheConfig,
    client: *HanaClient,
    is_connected: bool,

    const Self = @This();

    // Cache table DDL
    const CREATE_CACHE_TABLE: []const u8 =
        \\CREATE COLUMN TABLE IF NOT EXISTS {schema}.{prefix}_kv (
        \\    key NVARCHAR(512) PRIMARY KEY,
        \\    value NCLOB,
        \\    ttl BIGINT,
        \\    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        \\) IN MEMORY
    ;

    const CREATE_SESSION_TABLE: []const u8 =
        \\CREATE COLUMN TABLE IF NOT EXISTS {schema}.{prefix}_sessions (
        \\    session_id NVARCHAR(256) PRIMARY KEY,
        \\    data NCLOB,
        \\    ttl BIGINT,
        \\    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        \\) IN MEMORY
    ;

    const CREATE_STATE_TABLE: []const u8 =
        \\CREATE COLUMN TABLE IF NOT EXISTS {schema}.{prefix}_workflow_state (
        \\    workflow_id NVARCHAR(256) PRIMARY KEY,
        \\    state_json NCLOB,
        \\    ttl BIGINT,
        \\    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        \\) IN MEMORY
    ;

    pub fn init(allocator: Allocator, config: HanaCacheConfig) !Self {
        const hana_config = HanaConfig{
            .host = config.host,
            .port = config.port,
            .database = config.database,
            .user = config.user,
            .password = config.password,
        };

        const client = try hana.connectWithAllocator(allocator, hana_config);

        return Self{
            .allocator = allocator,
            .config = config,
            .client = client,
            .is_connected = false,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.is_connected) {
            self.disconnect();
        }
        self.client.deinit();
    }

    pub fn connect(self: *Self) !void {
        if (self.is_connected) return;
        self.is_connected = true;
        try self.initializeSchema();
    }

    pub fn disconnect(self: *Self) void {
        self.is_connected = false;
    }

    fn initializeSchema(self: *Self) !void {
        const cache_ddl = try std.mem.replaceOwned(
            u8,
            self.allocator,
            CREATE_CACHE_TABLE,
            "{schema}",
            self.config.database,
        );
        defer self.allocator.free(cache_ddl);

        const cache_ddl_final = try std.mem.replaceOwned(
            u8,
            self.allocator,
            cache_ddl,
            "{prefix}",
            self.config.table_prefix,
        );
        defer self.allocator.free(cache_ddl_final);

        try self.execute(cache_ddl_final);

        // Create session table
        const session_ddl = try std.mem.replaceOwned(
            u8,
            self.allocator,
            CREATE_SESSION_TABLE,
            "{schema}",
            self.config.database,
        );
        defer self.allocator.free(session_ddl);

        const session_ddl_final = try std.mem.replaceOwned(
            u8,
            self.allocator,
            session_ddl,
            "{prefix}",
            self.config.table_prefix,
        );
        defer self.allocator.free(session_ddl_final);

        try self.execute(session_ddl_final);

        // Create state table
        const state_ddl = try std.mem.replaceOwned(
            u8,
            self.allocator,
            CREATE_STATE_TABLE,
            "{schema}",
            self.config.database,
        );
        defer self.allocator.free(state_ddl);

        const state_ddl_final = try std.mem.replaceOwned(
            u8,
            self.allocator,
            state_ddl,
            "{prefix}",
            self.config.table_prefix,
        );
        defer self.allocator.free(state_ddl_final);

        try self.execute(state_ddl_final);
    }

    fn execute(self: *Self, sql: []const u8) !void {
        try hana.execute(self.client, sql);
    }

    fn query(self: *Self, sql: []const u8) !QueryResult {
        return hana.queryWithAllocator(self.client, self.allocator, sql);
    }

    // ========================================================================
    // Key-Value Operations (Redis-compatible API)
    // ========================================================================

    /// Set a key-value pair with optional TTL (in seconds)
    pub fn set(self: *Self, key: []const u8, value: []const u8, ttl_seconds: ?u32) !void {
        if (!self.is_connected) return error.NotConnected;

        const ttl_val: i64 = if (ttl_seconds) |ttl|
            std.time.timestamp() + @as(i64, ttl)
        else
            0;

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPSERT {s}.{s}_kv (key, value, ttl) VALUES ('{s}', '{s}', {d})",
            .{ self.config.database, self.config.table_prefix, key, value, ttl_val },
        );
        defer self.allocator.free(sql);

        try self.execute(sql);
    }

    /// Get a value by key, returns null if not found or expired
    pub fn get(self: *Self, key: []const u8) !?[]const u8 {
        if (!self.is_connected) return error.NotConnected;

        const now = std.time.timestamp();
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT value FROM {s}.{s}_kv WHERE key = '{s}' AND (ttl = 0 OR ttl > {d})",
            .{ self.config.database, self.config.table_prefix, key, now },
        );
        defer self.allocator.free(sql);

        var result = try self.query(sql);
        defer result.deinit();

        if (result.rows.len == 0) return null;
        if (result.rows[0].get(0)) |val| {
            if (val.asString()) |s| {
                return try self.allocator.dupe(u8, s);
            }
        }
        return null;
    }

    /// Delete a key
    pub fn del(self: *Self, key: []const u8) !bool {
        if (!self.is_connected) return error.NotConnected;

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_kv WHERE key = '{s}'",
            .{ self.config.database, self.config.table_prefix, key },
        );
        defer self.allocator.free(sql);

        try self.execute(sql);
        return true;
    }

    /// Check if key exists
    pub fn exists(self: *Self, key: []const u8) !bool {
        if (!self.is_connected) return error.NotConnected;

        const now = std.time.timestamp();
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT COUNT(*) FROM {s}.{s}_kv WHERE key = '{s}' AND (ttl = 0 OR ttl > {d})",
            .{ self.config.database, self.config.table_prefix, key, now },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        var result = try executor.executeQuery(sql);
        defer result.deinit();

        if (result.next()) |row| {
            if (row.getValue(0)) |val| {
                const count = try val.asInt();
                return count > 0;
            }
        }

        return false;
    }

    /// Set expiration time for a key
    pub fn expire(self: *Self, key: []const u8, seconds: u32) !bool {
        if (!self.is_connected) return error.NotConnected;

        const ttl = std.time.timestamp() + @as(i64, seconds);
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPDATE {s}.{s}_kv SET ttl = {d} WHERE key = '{s}'",
            .{ self.config.database, self.config.table_prefix, ttl, key },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        _ = try executor.executeQuery(sql);
        return true;
    }

    /// Get TTL for a key (-1 if no TTL, -2 if key doesn't exist)
    pub fn ttl(self: *Self, key: []const u8) !i64 {
        if (!self.is_connected) return error.NotConnected;

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT ttl FROM {s}.{s}_kv WHERE key = '{s}'",
            .{ self.config.database, self.config.table_prefix, key },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        var result = try executor.executeQuery(sql);
        defer result.deinit();

        if (result.next()) |row| {
            if (row.getValue(0)) |val| {
                const ttl_val = try val.asInt();
                if (ttl_val == 0) return -1; // No TTL
                const now = std.time.timestamp();
                return ttl_val - now;
            }
        }

        return -2; // Key doesn't exist
    }

    // ========================================================================
    // Session Management
    // ========================================================================

    /// Store session data with TTL
    pub fn storeSession(self: *Self, session_id: []const u8, data: []const u8, ttl_seconds: u32) !void {
        if (!self.is_connected) return error.NotConnected;

        const ttl = std.time.timestamp() + @as(i64, ttl_seconds);
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPSERT {s}.{s}_sessions (session_id, data, ttl) VALUES ('{s}', '{s}', {d})",
            .{ self.config.database, self.config.table_prefix, session_id, data, ttl },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        _ = try executor.executeQuery(sql);
    }

    /// Get session data
    pub fn getSession(self: *Self, session_id: []const u8) !?[]const u8 {
        if (!self.is_connected) return error.NotConnected;

        const now = std.time.timestamp();
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT data FROM {s}.{s}_sessions WHERE session_id = '{s}' AND ttl > {d}",
            .{ self.config.database, self.config.table_prefix, session_id, now },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        var result = try executor.executeQuery(sql);
        defer result.deinit();

        if (result.next()) |row| {
            if (row.getValue(0)) |val| {
                return try val.asString();
            }
        }

        return null;
    }

    /// Delete session
    pub fn deleteSession(self: *Self, session_id: []const u8) !void {
        if (!self.is_connected) return error.NotConnected;

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_sessions WHERE session_id = '{s}'",
            .{ self.config.database, self.config.table_prefix, session_id },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        _ = try executor.executeQuery(sql);
    }

    // ========================================================================
    // Workflow State Caching
    // ========================================================================

    /// Cache workflow state
    pub fn cacheWorkflowState(self: *Self, workflow_id: []const u8, state_json: []const u8, ttl_seconds: u32) !void {
        if (!self.is_connected) return error.NotConnected;

        const ttl = std.time.timestamp() + @as(i64, ttl_seconds);
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "UPSERT {s}.{s}_workflow_state (workflow_id, state_json, ttl) VALUES ('{s}', '{s}', {d})",
            .{ self.config.database, self.config.table_prefix, workflow_id, state_json, ttl },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        _ = try executor.executeQuery(sql);
    }

    /// Get cached workflow state
    pub fn getWorkflowState(self: *Self, workflow_id: []const u8) !?[]const u8 {
        if (!self.is_connected) return error.NotConnected;

        const now = std.time.timestamp();
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SELECT state_json FROM {s}.{s}_workflow_state WHERE workflow_id = '{s}' AND ttl > {d}",
            .{ self.config.database, self.config.table_prefix, workflow_id, now },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        var result = try executor.executeQuery(sql);
        defer result.deinit();

        if (result.next()) |row| {
            if (row.getValue(0)) |val| {
                return try val.asString();
            }
        }

        return null;
    }

    /// Delete cached workflow state
    pub fn deleteWorkflowState(self: *Self, workflow_id: []const u8) !void {
        if (!self.is_connected) return error.NotConnected;

        const sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_workflow_state WHERE workflow_id = '{s}'",
            .{ self.config.database, self.config.table_prefix, workflow_id },
        );
        defer self.allocator.free(sql);

        var executor = Query.Executor.init(self.allocator, &self.connection);
        _ = try executor.executeQuery(sql);
    }

    // ========================================================================
    // Cleanup Operations
    // ========================================================================

    /// Remove expired entries from all cache tables
    pub fn cleanupExpired(self: *Self) !void {
        if (!self.is_connected) return error.NotConnected;

        const now = std.time.timestamp();
        var executor = Query.Executor.init(self.allocator, &self.connection);

        // Cleanup key-value cache
        const kv_sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_kv WHERE ttl > 0 AND ttl < {d}",
            .{ self.config.database, self.config.table_prefix, now },
        );
        defer self.allocator.free(kv_sql);
        _ = try executor.executeQuery(kv_sql);

        // Cleanup sessions
        const session_sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_sessions WHERE ttl < {d}",
            .{ self.config.database, self.config.table_prefix, now },
        );
        defer self.allocator.free(session_sql);
        _ = try executor.executeQuery(session_sql);

        // Cleanup workflow state
        const state_sql = try std.fmt.allocPrint(
            self.allocator,
            "DELETE FROM {s}.{s}_workflow_state WHERE ttl < {d}",
            .{ self.config.database, self.config.table_prefix, now },
        );
        defer self.allocator.free(state_sql);
        _ = try executor.executeQuery(state_sql);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "HanaCache - initialization" {
    const allocator = std.testing.allocator;

    const config = HanaCacheConfig{
        .host = "localhost",
        .port = 39017,
        .user = "SYSTEM",
        .password = "Password123",
        .schema = "TEST_CACHE",
        .use_tls = false,
    };

    var cache = try HanaCache.init(allocator, config);
    defer cache.deinit();

    try std.testing.expect(!cache.is_connected);
}

test "HanaCache - key-value operations (mock)" {
    // This test would require a real HANA connection
    // In production, use integration tests with a test HANA instance
    std.testing.refAllDecls(@This());
}
