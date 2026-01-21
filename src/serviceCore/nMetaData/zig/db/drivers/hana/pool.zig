const std = @import("std");
const connection_mod = @import("connection.zig");

/// Pool configuration
pub const PoolConfig = struct {
    min_size: u32 = 2,
    max_size: u32 = 10,
    acquire_timeout_ms: u32 = 5000,
    idle_timeout_ms: u32 = 300000,
    max_lifetime_ms: u32 = 1800000,
    health_check_interval_ms: u32 = 60000,
    
    pub fn validate(self: PoolConfig) !void {
        if (self.min_size > self.max_size) {
            return error.InvalidPoolSize;
        }
        if (self.max_size == 0) {
            return error.InvalidPoolSize;
        }
    }
};

/// Pooled connection wrapper
pub const PooledConnection = struct {
    connection: *connection_mod.HanaConnection,
    created_at: i64,
    last_used: i64,
    use_count: u64,
    in_use: bool,
    
    pub fn init(connection: *connection_mod.HanaConnection) PooledConnection {
        const now = std.time.milliTimestamp();
        return PooledConnection{
            .connection = connection,
            .created_at = now,
            .last_used = now,
            .use_count = 0,
            .in_use = false,
        };
    }
    
    pub fn isHealthy(self: PooledConnection, config: PoolConfig) bool {
        const now = std.time.milliTimestamp();
        
        // Check connection
        if (!self.connection.isConnected()) {
            return false;
        }
        
        // Check max lifetime
        if (config.max_lifetime_ms > 0) {
            if (now - self.created_at > config.max_lifetime_ms) {
                return false;
            }
        }
        
        return true;
    }
    
    pub fn isIdle(self: PooledConnection, config: PoolConfig) bool {
        if (self.in_use) {
            return false;
        }
        
        const now = std.time.milliTimestamp();
        return (now - self.last_used) > config.idle_timeout_ms;
    }
};

/// Connection pool
pub const HanaConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: PoolConfig,
    connection_config: connection_mod.HanaConnectionConfig,
    connections: std.ArrayList(PooledConnection),
    mutex: std.Thread.Mutex,
    total_connections: u32,
    
    pub fn init(
        allocator: std.mem.Allocator,
        pool_config: PoolConfig,
        connection_config: connection_mod.HanaConnectionConfig,
    ) !HanaConnectionPool {
        try pool_config.validate();
        try connection_config.validate();
        
        return HanaConnectionPool{
            .allocator = allocator,
            .config = pool_config,
            .connection_config = connection_config,
            .connections = std.ArrayList(PooledConnection).init(allocator),
            .mutex = std.Thread.Mutex{},
            .total_connections = 0,
        };
    }
    
    pub fn deinit(self: *HanaConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |*pc| {
            pc.connection.deinit();
            self.allocator.destroy(pc.connection);
        }
        self.connections.deinit();
    }
    
    /// Acquire a connection from the pool
    pub fn acquire(self: *HanaConnectionPool) !*connection_mod.HanaConnection {
        const deadline = std.time.milliTimestamp() + @as(i64, self.config.acquire_timeout_ms);
        
        while (std.time.milliTimestamp() < deadline) {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            // Try to find an available connection
            for (self.connections.items) |*pc| {
                if (!pc.in_use and pc.isHealthy(self.config)) {
                    pc.in_use = true;
                    pc.last_used = std.time.milliTimestamp();
                    pc.use_count += 1;
                    return pc.connection;
                }
            }
            
            // Create new connection if under max
            if (self.total_connections < self.config.max_size) {
                const conn = try self.allocator.create(connection_mod.HanaConnection);
                errdefer self.allocator.destroy(conn);
                
                conn.* = try connection_mod.HanaConnection.init(self.allocator, self.connection_config);
                errdefer conn.deinit();
                
                var pc = PooledConnection.init(conn);
                pc.in_use = true;
                
                try self.connections.append(pc);
                self.total_connections += 1;
                
                return conn;
            }
            
            // Wait a bit and retry
            std.time.sleep(10 * std.time.ns_per_ms);
        }
        
        return error.AcquireTimeout;
    }
    
    /// Release a connection back to the pool
    pub fn release(self: *HanaConnectionPool, connection: *connection_mod.HanaConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |*pc| {
            if (pc.connection == connection) {
                pc.in_use = false;
                pc.last_used = std.time.milliTimestamp();
                return;
            }
        }
    }
    
    /// Perform health check and cleanup
    pub fn maintain(self: *HanaConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var i: usize = 0;
        while (i < self.connections.items.len) {
            const pc = &self.connections.items[i];
            
            // Remove unhealthy or idle connections (if over min_size)
            const should_remove = !pc.isHealthy(self.config) or
                (self.total_connections > self.config.min_size and pc.isIdle(self.config));
            
            if (should_remove and !pc.in_use) {
                pc.connection.deinit();
                self.allocator.destroy(pc.connection);
                _ = self.connections.swapRemove(i);
                self.total_connections -= 1;
            } else {
                i += 1;
            }
        }
        
        // Ensure minimum connections
        while (self.total_connections < self.config.min_size) {
            const conn = self.allocator.create(connection_mod.HanaConnection) catch break;
            conn.* = connection_mod.HanaConnection.init(self.allocator, self.connection_config) catch {
                self.allocator.destroy(conn);
                break;
            };
            
            const pc = PooledConnection.init(conn);
            self.connections.append(pc) catch {
                conn.deinit();
                self.allocator.destroy(conn);
                break;
            };
            
            self.total_connections += 1;
        }
    }
    
    /// Get pool statistics
    pub fn getStats(self: *HanaConnectionPool) PoolStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var active: u32 = 0;
        var idle: u32 = 0;
        
        for (self.connections.items) |pc| {
            if (pc.in_use) {
                active += 1;
            } else {
                idle += 1;
            }
        }
        
        return PoolStats{
            .total = self.total_connections,
            .active = active,
            .idle = idle,
            .max_size = self.config.max_size,
        };
    }
};

/// Pool statistics
pub const PoolStats = struct {
    total: u32,
    active: u32,
    idle: u32,
    max_size: u32,
};

// ============================================================================
// Unit Tests
// ============================================================================

test "PoolConfig - validation" {
    const valid = PoolConfig{
        .min_size = 2,
        .max_size = 10,
    };
    try valid.validate();
    
    const invalid1 = PoolConfig{
        .min_size = 10,
        .max_size = 2,
    };
    try std.testing.expectError(error.InvalidPoolSize, invalid1.validate());
    
    const invalid2 = PoolConfig{
        .min_size = 0,
        .max_size = 0,
    };
    try std.testing.expectError(error.InvalidPoolSize, invalid2.validate());
}

test "PooledConnection - init" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    const pc = PooledConnection.init(&conn);
    
    try std.testing.expect(!pc.in_use);
    try std.testing.expectEqual(@as(u64, 0), pc.use_count);
}

test "PooledConnection - health check" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    const pc = PooledConnection.init(&conn);
    
    const pool_config = PoolConfig{};
    try std.testing.expect(pc.isHealthy(pool_config));
}

test "HanaConnectionPool - init and deinit" {
    const allocator = std.testing.allocator;
    
    const pool_config = PoolConfig{
        .min_size = 2,
        .max_size = 5,
    };
    
    const conn_config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var pool = try HanaConnectionPool.init(allocator, pool_config, conn_config);
    defer pool.deinit();
    
    try std.testing.expectEqual(@as(u32, 0), pool.total_connections);
}

test "HanaConnectionPool - statistics" {
    const allocator = std.testing.allocator;
    
    const pool_config = PoolConfig{
        .min_size = 2,
        .max_size = 5,
    };
    
    const conn_config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var pool = try HanaConnectionPool.init(allocator, pool_config, conn_config);
    defer pool.deinit();
    
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u32, 0), stats.total);
    try std.testing.expectEqual(@as(u32, 0), stats.active);
    try std.testing.expectEqual(@as(u32, 0), stats.idle);
    try std.testing.expectEqual(@as(u32, 5), stats.max_size);
}
