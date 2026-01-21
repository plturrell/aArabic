const std = @import("std");
const connection_mod = @import("connection.zig");
const protocol = @import("protocol.zig");

const SqliteConnection = connection_mod.SqliteConnection;
const SqliteConfig = protocol.SqliteConfig;
const DbError = @import("../../errors.zig").DbError;

/// SQLite connection pool configuration
pub const SqlitePoolConfig = struct {
    min_size: usize = 1,
    max_size: usize = 5,
    acquire_timeout_ms: u32 = 5000,
    idle_timeout_ms: u32 = 300000, // 5 minutes
    
    pub fn validate(self: SqlitePoolConfig) !void {
        if (self.min_size == 0) return DbError.InvalidConfiguration;
        if (self.max_size < self.min_size) return DbError.InvalidConfiguration;
    }
};

/// Pooled connection wrapper
const PooledConnection = struct {
    connection: SqliteConnection,
    in_use: bool,
    last_used: i64,
    
    fn init(allocator: std.mem.Allocator, config: SqliteConfig) !PooledConnection {
        var conn = try SqliteConnection.init(allocator, config);
        try conn.connect();
        
        return PooledConnection{
            .connection = conn,
            .in_use = false,
            .last_used = std.time.milliTimestamp(),
        };
    }
    
    fn deinit(self: *PooledConnection) void {
        self.connection.deinit();
    }
    
    fn isHealthy(self: *PooledConnection) bool {
        return self.connection.ping() catch false;
    }
};

/// SQLite connection pool
pub const SqliteConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: SqlitePoolConfig,
    db_config: SqliteConfig,
    connections: std.ArrayList(PooledConnection),
    mutex: std.Thread.Mutex,
    
    pub fn init(
        allocator: std.mem.Allocator,
        config: SqlitePoolConfig,
        db_config: SqliteConfig,
    ) !SqliteConnectionPool {
        try config.validate();
        
        var pool = SqliteConnectionPool{
            .allocator = allocator,
            .config = config,
            .db_config = db_config,
            .connections = std.ArrayList(PooledConnection).init(allocator),
            .mutex = std.Thread.Mutex{},
        };
        
        // Create minimum connections
        for (0..config.min_size) |_| {
            const conn = try PooledConnection.init(allocator, db_config);
            try pool.connections.append(conn);
        }
        
        return pool;
    }
    
    pub fn deinit(self: *SqliteConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |*conn| {
            conn.deinit();
        }
        self.connections.deinit();
    }
    
    /// Acquire connection from pool
    pub fn acquire(self: *SqliteConnectionPool) !*SqliteConnection {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Find available connection
        for (self.connections.items) |*pooled| {
            if (!pooled.in_use) {
                // Check health
                if (!pooled.isHealthy()) {
                    // Reconnect if unhealthy
                    pooled.connection.connect() catch continue;
                }
                
                pooled.in_use = true;
                pooled.last_used = std.time.milliTimestamp();
                return &pooled.connection;
            }
        }
        
        // Create new connection if under max
        if (self.connections.items.len < self.config.max_size) {
            var pooled = try PooledConnection.init(self.allocator, self.db_config);
            pooled.in_use = true;
            try self.connections.append(pooled);
            return &self.connections.items[self.connections.items.len - 1].connection;
        }
        
        return DbError.ConnectionPoolExhausted;
    }
    
    /// Release connection back to pool
    pub fn release(self: *SqliteConnectionPool, connection: *SqliteConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |*pooled| {
            if (&pooled.connection == connection) {
                pooled.in_use = false;
                pooled.last_used = std.time.milliTimestamp();
                return;
            }
        }
    }
    
    /// Maintain pool (remove idle connections, keep minimum)
    pub fn maintain(self: *SqliteConnectionPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const now = std.time.milliTimestamp();
        const idle_threshold = @as(i64, @intCast(self.config.idle_timeout_ms));
        
        var i: usize = 0;
        while (i < self.connections.items.len) {
            const pooled = &self.connections.items[i];
            
            // Keep minimum connections
            if (self.connections.items.len <= self.config.min_size) {
                i += 1;
                continue;
            }
            
            // Remove idle connections
            if (!pooled.in_use and (now - pooled.last_used) > idle_threshold) {
                var removed = self.connections.orderedRemove(i);
                removed.deinit();
                continue; // Don't increment i
            }
            
            i += 1;
        }
    }
    
    /// Get pool statistics
    pub fn getStats(self: *SqliteConnectionPool) PoolStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var in_use: usize = 0;
        for (self.connections.items) |pooled| {
            if (pooled.in_use) in_use += 1;
        }
        
        return PoolStats{
            .total = self.connections.items.len,
            .in_use = in_use,
            .available = self.connections.items.len - in_use,
            .min_size = self.config.min_size,
            .max_size = self.config.max_size,
        };
    }
};

/// Pool statistics
pub const PoolStats = struct {
    total: usize,
    in_use: usize,
    available: usize,
    min_size: usize,
    max_size: usize,
};

// ============================================================================
// Unit Tests
// ============================================================================

test "SqlitePoolConfig - validation" {
    const valid = SqlitePoolConfig{
        .min_size = 2,
        .max_size = 10,
    };
    try valid.validate();
    
    const invalid1 = SqlitePoolConfig{
        .min_size = 0,
        .max_size = 10,
    };
    try std.testing.expectError(DbError.InvalidConfiguration, invalid1.validate());
    
    const invalid2 = SqlitePoolConfig{
        .min_size = 10,
        .max_size = 5,
    };
    try std.testing.expectError(DbError.InvalidConfiguration, invalid2.validate());
}

test "SqliteConnectionPool - init and deinit" {
    const allocator = std.testing.allocator;
    const pool_config = SqlitePoolConfig{
        .min_size = 2,
        .max_size = 5,
    };
    const db_config = SqliteConfig.inMemory();
    
    var pool = try SqliteConnectionPool.init(allocator, pool_config, db_config);
    defer pool.deinit();
    
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.total);
    try std.testing.expectEqual(@as(usize, 0), stats.in_use);
}

test "SqliteConnectionPool - acquire and release" {
    const allocator = std.testing.allocator;
    const pool_config = SqlitePoolConfig{
        .min_size = 1,
        .max_size = 3,
    };
    const db_config = SqliteConfig.inMemory();
    
    var pool = try SqliteConnectionPool.init(allocator, pool_config, db_config);
    defer pool.deinit();
    
    var conn1 = try pool.acquire();
    var stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.in_use);
    
    var conn2 = try pool.acquire();
    stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.in_use);
    
    pool.release(conn1);
    stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.in_use);
    
    pool.release(conn2);
    stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.in_use);
}

test "SqliteConnectionPool - max size limit" {
    const allocator = std.testing.allocator;
    const pool_config = SqlitePoolConfig{
        .min_size = 1,
        .max_size = 2,
    };
    const db_config = SqliteConfig.inMemory();
    
    var pool = try SqliteConnectionPool.init(allocator, pool_config, db_config);
    defer pool.deinit();
    
    var conn1 = try pool.acquire();
    var conn2 = try pool.acquire();
    
    // Should fail - pool exhausted
    try std.testing.expectError(DbError.ConnectionPoolExhausted, pool.acquire());
    
    pool.release(conn1);
    pool.release(conn2);
}

test "SqliteConnectionPool - getStats" {
    const allocator = std.testing.allocator;
    const pool_config = SqlitePoolConfig{
        .min_size = 2,
        .max_size = 5,
    };
    const db_config = SqliteConfig.inMemory();
    
    var pool = try SqliteConnectionPool.init(allocator, pool_config, db_config);
    defer pool.deinit();
    
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 2), stats.total);
    try std.testing.expectEqual(@as(usize, 0), stats.in_use);
    try std.testing.expectEqual(@as(usize, 2), stats.available);
    try std.testing.expectEqual(@as(usize, 2), stats.min_size);
    try std.testing.expectEqual(@as(usize, 5), stats.max_size);
}
