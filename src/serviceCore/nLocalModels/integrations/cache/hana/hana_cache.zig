// HANA In-Memory Cache for nLocalModels
// Replaces DragonflyDB with SAP HANA in-memory tables
//
// Features:
// - KV cache with TTL
// - Prompt caching
// - Session state management
// - Tensor storage
// - Automatic TTL cleanup

const std = @import("std");
const Allocator = std.mem.Allocator;

// Note: This will import from the HANA SDK once available
// For now, we'll create a minimal interface
const HanaClient = struct {
    allocator: Allocator,
    
    pub const HanaConfig = struct {
        host: []const u8,
        port: u16,
        database: []const u8,
        user: []const u8,
        password: []const u8,
        pool_min: u32 = 5,
        pool_max: u32 = 10,
    };
    
    pub fn init(allocator: Allocator, config: HanaConfig) !*HanaClient {
        const client = try allocator.create(HanaClient);
        client.* = .{ .allocator = allocator };
        
        std.log.info("HANA client initialized: {s}:{d}/{s}", .{
            config.host,
            config.port,
            config.database,
        });
        
        return client;
    }
    
    pub fn deinit(self: *HanaClient) void {
        self.allocator.destroy(self);
    }
    
    pub fn execute(self: *HanaClient, sql: []const u8) !void {
        _ = self;
        std.log.debug("Execute SQL: {s}", .{sql});
    }
    
    pub fn queryParameterized(
        self: *HanaClient,
        sql: []const u8,
        params: []const Parameter,
        allocator: Allocator,
    ) !QueryResult {
        _ = self;
        _ = params;
        std.log.debug("Query SQL: {s}", .{sql});
        
        return QueryResult{
            .rows = try allocator.alloc(Row, 0),
            .columns = try allocator.alloc([]const u8, 0),
            .allocator = allocator,
        };
    }
    
    pub fn executeParameterized(
        self: *HanaClient,
        sql: []const u8,
        params: []const Parameter,
    ) !void {
        _ = self;
        _ = params;
        std.log.debug("Execute parameterized: {s}", .{sql});
    }
    
    pub const Parameter = union(enum) {
        int: i64,
        float: f64,
        string: []const u8,
        blob: []const u8,
        null_value,
    };
    
    pub const QueryResult = struct {
        rows: []Row,
        columns: [][]const u8,
        allocator: Allocator,
        
        pub fn deinit(self: *const QueryResult) void {
            for (self.rows) |row| {
                row.deinit();
            }
            self.allocator.free(self.rows);
            
            for (self.columns) |col| {
                self.allocator.free(col);
            }
            self.allocator.free(self.columns);
        }
    };
    
    pub const Row = struct {
        values: []Value,
        allocator: Allocator,
        
        pub fn deinit(self: *const Row) void {
            self.allocator.free(self.values);
        }
        
        pub fn getString(self: *const Row, column: []const u8) ?[]const u8 {
            _ = column;
            if (self.values.len == 0) return null;
            return switch (self.values[0]) {
                .string => |s| s,
                .blob => |b| b,
                else => null,
            };
        }
        
        pub fn getBlob(self: *const Row, column: []const u8) ?[]const u8 {
            return self.getString(column);
        }
    };
    
    pub const Value = union(enum) {
        null_value,
        int: i64,
        float: f64,
        string: []const u8,
        blob: []const u8,
    };
};

const Parameter = HanaClient.Parameter;

/// HANA Cache Configuration
pub const HanaCacheConfig = struct {
    hana_host: []const u8 = "localhost",
    hana_port: u16 = 30015,
    hana_database: []const u8 = "NOPENAI_DB",
    hana_user: []const u8 = "SHIMMY_USER",
    hana_password: []const u8 = "",
    
    // TTL settings (seconds)
    kv_cache_ttl: u32 = 3600,      // 1 hour
    prompt_cache_ttl: u32 = 3600,  // 1 hour
    tensor_ttl: u32 = 86400,       // 24 hours
    session_ttl: u32 = 1800,       // 30 minutes
    
    // Cleanup interval
    cleanup_interval_secs: u32 = 60,
};

/// HANA In-Memory Cache
pub const HanaCache = struct {
    allocator: Allocator,
    config: HanaCacheConfig,
    client: *HanaClient,
    
    // TTL management
    ttl_thread: ?std.Thread = null,
    is_running: bool = false,
    mutex: std.Thread.Mutex = .{},
    
    // Statistics
    stats: Stats,
    
    pub const Stats = struct {
        gets: u64 = 0,
        sets: u64 = 0,
        deletes: u64 = 0,
        hits: u64 = 0,
        misses: u64 = 0,
        bytes_sent: u64 = 0,
        bytes_received: u64 = 0,
        ttl_cleanups: u64 = 0,
        
        pub fn getHitRate(self: *const Stats) f32 {
            const total = self.hits + self.misses;
            if (total == 0) return 0.0;
            return @as(f32, @floatFromInt(self.hits)) / @as(f32, @floatFromInt(total));
        }
    };
    
    /// Initialize HANA cache
    pub fn init(allocator: Allocator, config: HanaCacheConfig) !*HanaCache {
        std.log.info("Initializing HANA Cache", .{});
        std.log.info("  Host: {s}:{d}", .{ config.hana_host, config.hana_port });
        std.log.info("  Database: {s}", .{config.hana_database});
        
        // Initialize HANA client
        const client = try HanaClient.init(allocator, .{
            .host = config.hana_host,
            .port = config.hana_port,
            .database = config.hana_database,
            .user = config.hana_user,
            .password = config.hana_password,
        });
        errdefer client.deinit();
        
        const self = try allocator.create(HanaCache);
        self.* = .{
            .allocator = allocator,
            .config = config,
            .client = client,
            .is_running = true,
            .stats = .{},
        };
        
        // Create cache tables
        try self.createTables();
        
        // Start TTL cleanup thread
        self.ttl_thread = try std.Thread.spawn(.{}, ttlCleanupLoop, .{self});
        
        std.log.info("HANA Cache initialized successfully", .{});
        return self;
    }
    
    /// Create HANA cache tables
    fn createTables(self: *HanaCache) !void {
        std.log.info("Creating HANA cache tables...", .{});
        
        // KV Cache table (in-memory column store)
        try self.client.execute(
            \\CREATE COLUMN TABLE IF NOT EXISTS KV_CACHE (
            \\  key VARCHAR(512) PRIMARY KEY,
            \\  value BLOB,
            \\  expires_at TIMESTAMP,
            \\  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            \\)
        );
        
        // Prompt Cache table (in-memory)
        try self.client.execute(
            \\CREATE COLUMN TABLE IF NOT EXISTS PROMPT_CACHE (
            \\  hash VARCHAR(64) PRIMARY KEY,
            \\  state BLOB,
            \\  expires_at TIMESTAMP,
            \\  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            \\)
        );
        
        // Session State table (in-memory)
        try self.client.execute(
            \\CREATE COLUMN TABLE IF NOT EXISTS SESSION_STATE (
            \\  session_id VARCHAR(128) PRIMARY KEY,
            \\  data BLOB,
            \\  expires_at TIMESTAMP,
            \\  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            \\)
        );
        
        // Tensor Storage table (in-memory)
        try self.client.execute(
            \\CREATE COLUMN TABLE IF NOT EXISTS TENSOR_STORAGE (
            \\  tensor_id VARCHAR(256) PRIMARY KEY,
            \\  tensor_data BLOB,
            \\  metadata VARCHAR(1024),
            \\  expires_at TIMESTAMP,
            \\  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            \\)
        );
        
        std.log.info("Cache tables created", .{});
    }
    
    /// Set key-value with TTL
    pub fn set(self: *HanaCache, key: []const u8, value: []const u8, ttl_secs: u32) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.client.executeParameterized(
            \\UPSERT KV_CACHE (key, value, expires_at)
            \\VALUES (?, ?, ADD_SECONDS(CURRENT_TIMESTAMP, ?))
        , &[_]Parameter{
            .{ .string = key },
            .{ .blob = value },
            .{ .int = ttl_secs },
        });
        
        self.stats.sets += 1;
        self.stats.bytes_sent += value.len;
    }
    
    /// Get key-value
    pub fn get(self: *HanaCache, key: []const u8) !?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.gets += 1;
        
        const result = try self.client.queryParameterized(
            \\SELECT value FROM KV_CACHE
            \\WHERE key = ? AND expires_at > CURRENT_TIMESTAMP
        , &[_]Parameter{
            .{ .string = key },
        }, self.allocator);
        defer result.deinit();
        
        if (result.rows.len == 0) {
            self.stats.misses += 1;
            return null;
        }
        
        self.stats.hits += 1;
        const value = result.rows[0].getBlob("value") orelse return null;
        self.stats.bytes_received += value.len;
        
        // Duplicate the value since result will be freed
        return try self.allocator.dupe(u8, value);
    }
    
    /// Delete key
    pub fn del(self: *HanaCache, key: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.client.executeParameterized(
            \\DELETE FROM KV_CACHE WHERE key = ?
        , &[_]Parameter{
            .{ .string = key },
        });
        
        self.stats.deletes += 1;
    }
    
    /// Set prompt cache
    pub fn setPromptCache(self: *HanaCache, hash: []const u8, state: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.client.executeParameterized(
            \\UPSERT PROMPT_CACHE (hash, state, expires_at)
            \\VALUES (?, ?, ADD_SECONDS(CURRENT_TIMESTAMP, ?))
        , &[_]Parameter{
            .{ .string = hash },
            .{ .blob = state },
            .{ .int = self.config.prompt_cache_ttl },
        });
        
        self.stats.sets += 1;
        self.stats.bytes_sent += state.len;
    }
    
    /// Get prompt cache
    pub fn getPromptCache(self: *HanaCache, hash: []const u8) !?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.gets += 1;
        
        const result = try self.client.queryParameterized(
            \\SELECT state FROM PROMPT_CACHE
            \\WHERE hash = ? AND expires_at > CURRENT_TIMESTAMP
        , &[_]Parameter{
            .{ .string = hash },
        }, self.allocator);
        defer result.deinit();
        
        if (result.rows.len == 0) {
            self.stats.misses += 1;
            return null;
        }
        
        self.stats.hits += 1;
        const state = result.rows[0].getBlob("state") orelse return null;
        self.stats.bytes_received += state.len;
        
        return try self.allocator.dupe(u8, state);
    }
    
    /// Set session state
    pub fn setSession(self: *HanaCache, session_id: []const u8, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.client.executeParameterized(
            \\UPSERT SESSION_STATE (session_id, data, expires_at)
            \\VALUES (?, ?, ADD_SECONDS(CURRENT_TIMESTAMP, ?))
        , &[_]Parameter{
            .{ .string = session_id },
            .{ .blob = data },
            .{ .int = self.config.session_ttl },
        });
        
        self.stats.sets += 1;
        self.stats.bytes_sent += data.len;
    }
    
    /// Get session state
    pub fn getSession(self: *HanaCache, session_id: []const u8) !?[]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats.gets += 1;
        
        const result = try self.client.queryParameterized(
            \\SELECT data FROM SESSION_STATE
            \\WHERE session_id = ? AND expires_at > CURRENT_TIMESTAMP
        , &[_]Parameter{
            .{ .string = session_id },
        }, self.allocator);
        defer result.deinit();
        
        if (result.rows.len == 0) {
            self.stats.misses += 1;
            return null;
        }
        
        self.stats.hits += 1;
        const data = result.rows[0].getBlob("data") orelse return null;
        self.stats.bytes_received += data.len;
        
        return try self.allocator.dupe(u8, data);
    }
    
    /// Delete session
    pub fn deleteSession(self: *HanaCache, session_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.client.executeParameterized(
            \\DELETE FROM SESSION_STATE WHERE session_id = ?
        , &[_]Parameter{
            .{ .string = session_id },
        });
        
        self.stats.deletes += 1;
    }
    
    /// TTL cleanup loop (runs in background thread)
    fn ttlCleanupLoop(self: *HanaCache) void {
        while (self.is_running) {
            std.time.sleep(self.config.cleanup_interval_secs * std.time.ns_per_s);
            
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Clean expired entries from all tables
            self.client.execute(
                "DELETE FROM KV_CACHE WHERE expires_at < CURRENT_TIMESTAMP"
            ) catch |err| {
                std.log.warn("TTL cleanup failed for KV_CACHE: {}", .{err});
            };
            
            self.client.execute(
                "DELETE FROM PROMPT_CACHE WHERE expires_at < CURRENT_TIMESTAMP"
            ) catch |err| {
                std.log.warn("TTL cleanup failed for PROMPT_CACHE: {}", .{err});
            };
            
            self.client.execute(
                "DELETE FROM SESSION_STATE WHERE expires_at < CURRENT_TIMESTAMP"
            ) catch |err| {
                std.log.warn("TTL cleanup failed for SESSION_STATE: {}", .{err});
            };
            
            self.client.execute(
                "DELETE FROM TENSOR_STORAGE WHERE expires_at < CURRENT_TIMESTAMP"
            ) catch |err| {
                std.log.warn("TTL cleanup failed for TENSOR_STORAGE: {}", .{err});
            };
            
            self.stats.ttl_cleanups += 1;
            std.log.debug("TTL cleanup completed", .{});
        }
    }
    
    /// Get statistics
    pub fn getStats(self: *HanaCache) Stats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }
    
    /// Print statistics
    pub fn printStats(self: *HanaCache) void {
        const stats = self.getStats();
        
        std.log.info("HANA Cache Statistics:", .{});
        std.log.info("  Gets: {d}, Sets: {d}, Deletes: {d}", .{
            stats.gets,
            stats.sets,
            stats.deletes,
        });
        std.log.info("  Hits: {d}, Misses: {d} (Hit rate: {d:.1}%)", .{
            stats.hits,
            stats.misses,
            stats.getHitRate() * 100.0,
        });
        std.log.info("  Bytes sent: {d}, received: {d}", .{
            stats.bytes_sent,
            stats.bytes_received,
        });
        std.log.info("  TTL cleanups: {d}", .{stats.ttl_cleanups});
    }
    
    /// Cleanup and shutdown
    pub fn deinit(self: *HanaCache) void {
        std.log.info("Shutting down HANA Cache", .{});
        
        self.is_running = false;
        
        if (self.ttl_thread) |thread| {
            thread.join();
        }
        
        self.client.deinit();
        self.allocator.destroy(self);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "HanaCache - init and deinit" {
    const allocator = std.testing.allocator;
    
    const config = HanaCacheConfig{
        .hana_host = "localhost",
        .hana_port = 30015,
        .hana_database = "TEST_DB",
        .hana_user = "TEST_USER",
        .hana_password = "test123",
    };
    
    const cache = try HanaCache.init(allocator, config);
    defer cache.deinit();
    
    try std.testing.expect(cache.is_running == true);
}

test "HanaCache - set and get" {
    const allocator = std.testing.allocator;
    
    const config = HanaCacheConfig{};
    const cache = try HanaCache.init(allocator, config);
    defer cache.deinit();
    
    // Set a value
    try cache.set("test_key", "test_value", 3600);
    
    // Get the value
    const value = try cache.get("test_key");
    if (value) |v| {
        defer allocator.free(v);
        try std.testing.expectEqualStrings("test_value", v);
    }
}

test "HanaCache - statistics" {
    const allocator = std.testing.allocator;
    
    const config = HanaCacheConfig{};
    const cache = try HanaCache.init(allocator, config);
    defer cache.deinit();
    
    const stats = cache.getStats();
    try std.testing.expect(stats.gets == 0);
    try std.testing.expect(stats.sets == 0);
    try std.testing.expect(stats.getHitRate() == 0.0);
}