const std = @import("std");
const connection_mod = @import("connection.zig");
const PgConnection = connection_mod.PgConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;

/// PostgreSQL-specific connection state
pub const PgConnectionState = enum {
    idle,       // Available for use
    in_use,     // Currently being used
    invalid,    // Connection failed, needs cleanup
    validating, // Health check in progress
};

/// Pooled PostgreSQL connection
pub const PooledPgConnection = struct {
    connection: PgConnection,
    state: PgConnectionState,
    last_used: i64,
    use_count: usize,
    id: usize,
    created_at: i64,
    
    pub fn init(allocator: std.mem.Allocator, config: ConnectionConfig, id: usize) !PooledPgConnection {
        var conn = try PgConnection.init(allocator, config);
        
        return PooledPgConnection{
            .connection = conn,
            .state = .idle,
            .last_used = std.time.milliTimestamp(),
            .use_count = 0,
            .id = id,
            .created_at = std.time.milliTimestamp(),
        };
    }
    
    pub fn deinit(self: *PooledPgConnection) void {
        self.connection.deinit();
    }
    
    pub fn markUsed(self: *PooledPgConnection) void {
        self.state = .in_use;
        self.last_used = std.time.milliTimestamp();
        self.use_count += 1;
    }
    
    pub fn markIdle(self: *PooledPgConnection) void {
        self.state = .idle;
        self.last_used = std.time.milliTimestamp();
    }
    
    pub fn markInvalid(self: *PooledPgConnection) void {
        self.state = .invalid;
    }
    
    pub fn isHealthy(self: PooledPgConnection) bool {
        return self.state != .invalid and self.connection.isConnected();
    }
    
    pub fn getAge(self: PooledPgConnection) i64 {
        return std.time.milliTimestamp() - self.created_at;
    }
    
    pub fn getIdleTime(self: PooledPgConnection) i64 {
        return std.time.milliTimestamp() - self.last_used;
    }
};

/// PostgreSQL connection pool configuration
pub const PgPoolConfig = struct {
    connection_config: ConnectionConfig,
    min_size: usize = 2,
    max_size: usize = 10,
    acquire_timeout_ms: i64 = 5000,
    idle_timeout_ms: i64 = 300000,      // 5 minutes
    max_lifetime_ms: i64 = 1800000,     // 30 minutes
    health_check_interval_ms: i64 = 60000, // 1 minute
    validation_query: []const u8 = "SELECT 1",
    
    pub fn validate(self: PgPoolConfig) !void {
        if (self.min_size > self.max_size) {
            return error.InvalidPoolConfig;
        }
        if (self.max_size == 0) {
            return error.InvalidPoolConfig;
        }
        if (self.acquire_timeout_ms <= 0) {
            return error.InvalidPoolConfig;
        }
        try self.connection_config.validate();
    }
};

/// Pool metrics
pub const PgPoolMetrics = struct {
    total_connections: usize,
    idle_connections: usize,
    active_connections: usize,
    invalid_connections: usize,
    total_acquires: u64,
    total_releases: u64,
    total_timeouts: u64,
    total_errors: u64,
    total_created: u64,
    total_destroyed: u64,
    avg_wait_time_ms: f64,
    avg_connection_age_ms: f64,
    
    pub fn format(self: PgPoolMetrics, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "PgPoolMetrics{{ total={d}, idle={d}, active={d}, acquires={d}, timeouts={d}, avg_wait={d:.2}ms }}",
            .{
                self.total_connections,
                self.idle_connections,
                self.active_connections,
                self.total_acquires,
                self.total_timeouts,
                self.avg_wait_time_ms,
            },
        );
    }
};

/// PostgreSQL connection pool
pub const PgConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: PgPoolConfig,
    connections: std.ArrayList(PooledPgConnection),
    mutex: std.Thread.Mutex,
    next_id: usize,
    last_health_check: i64,
    
    // Metrics
    total_acquires: u64,
    total_releases: u64,
    total_timeouts: u64,
    total_errors: u64,
    total_created: u64,
    total_destroyed: u64,
    total_wait_time_ms: u64,
    
    pub fn init(allocator: std.mem.Allocator, config: PgPoolConfig) !PgConnectionPool {
        try config.validate();
        
        return PgConnectionPool{
            .allocator = allocator,
            .config = config,
            .connections = std.ArrayList(PooledPgConnection).init(allocator),
            .mutex = std.Thread.Mutex{},
            .next_id = 0,
            .last_health_check = std.time.milliTimestamp(),
            .total_acquires = 0,
            .total_releases = 0,
            .total_timeouts = 0,
            .total_errors = 0,
            .total_created = 0,
            .total_destroyed = 0,
            .total_wait_time_ms = 0,
        };
    }
    
    pub fn deinit(self: *PgConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |*conn| {
            conn.deinit();
        }
        self.connections.deinit();
    }
    
    /// Acquire a connection from the pool
    pub fn acquire(self: *PgConnectionPool) !*PooledPgConnection {
        const start_time = std.time.milliTimestamp();
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.total_acquires += 1;
        
        // Try to find idle, healthy connection
        if (try self.findHealthyConnection()) |conn| {
            conn.markUsed();
            const wait_time = std.time.milliTimestamp() - start_time;
            self.total_wait_time_ms += @intCast(wait_time);
            return conn;
        }
        
        // No healthy connection available, try to create new one
        if (self.connections.items.len < self.config.max_size) {
            const conn = try self.createConnection();
            try conn.connection.connect();
            conn.markUsed();
            const wait_time = std.time.milliTimestamp() - start_time;
            self.total_wait_time_ms += @intCast(wait_time);
            return conn;
        }
        
        // Pool is full and no connections available
        const elapsed = std.time.milliTimestamp() - start_time;
        if (elapsed >= self.config.acquire_timeout_ms) {
            self.total_timeouts += 1;
            return error.AcquireTimeout;
        }
        
        self.total_timeouts += 1;
        return error.PoolExhausted;
    }
    
    /// Release a connection back to the pool
    pub fn release(self: *PgConnectionPool, conn: *PooledPgConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.total_releases += 1;
        
        if (conn.isHealthy()) {
            conn.markIdle();
        } else {
            conn.markInvalid();
        }
    }
    
    /// Validate a connection's health
    pub fn validateConnection(self: *PgConnectionPool, conn: *PooledPgConnection) !bool {
        _ = self;
        
        // Check if still connected
        if (!conn.connection.isConnected()) {
            return false;
        }
        
        // Try to execute validation query
        // In real implementation, would execute self.config.validation_query
        // For now, just check connection state
        return true;
    }
    
    /// Perform health check on all connections
    pub fn healthCheck(self: *PgConnectionPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const now = std.time.milliTimestamp();
        self.last_health_check = now;
        
        var i: usize = 0;
        while (i < self.connections.items.len) {
            var conn = &self.connections.items[i];
            
            // Skip connections in use
            if (conn.state == .in_use) {
                i += 1;
                continue;
            }
            
            // Check age - destroy if too old
            if (conn.getAge() > self.config.max_lifetime_ms) {
                conn.markInvalid();
            }
            
            // Check idle timeout
            if (conn.state == .idle and conn.getIdleTime() > self.config.idle_timeout_ms) {
                conn.markInvalid();
            }
            
            // Remove invalid connections
            if (conn.state == .invalid) {
                var removed = self.connections.orderedRemove(i);
                removed.deinit();
                self.total_destroyed += 1;
                continue;
            }
            
            i += 1;
        }
        
        // Ensure minimum pool size
        while (self.connections.items.len < self.config.min_size) {
            var new_conn = try self.createConnection();
            try new_conn.connection.connect();
            new_conn.markIdle();
        }
    }
    
    /// Get pool metrics
    pub fn getMetrics(self: *PgConnectionPool) PgPoolMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var idle: usize = 0;
        var active: usize = 0;
        var invalid: usize = 0;
        var total_age: i64 = 0;
        
        for (self.connections.items) |*conn| {
            switch (conn.state) {
                .idle => idle += 1,
                .in_use => active += 1,
                .invalid => invalid += 1,
                .validating => {},
            }
            total_age += conn.getAge();
        }
        
        const avg_wait = if (self.total_acquires > 0)
            @as(f64, @floatFromInt(self.total_wait_time_ms)) / @as(f64, @floatFromInt(self.total_acquires))
        else
            0.0;
        
        const avg_age = if (self.connections.items.len > 0)
            @as(f64, @floatFromInt(total_age)) / @as(f64, @floatFromInt(self.connections.items.len))
        else
            0.0;
        
        return PgPoolMetrics{
            .total_connections = self.connections.items.len,
            .idle_connections = idle,
            .active_connections = active,
            .invalid_connections = invalid,
            .total_acquires = self.total_acquires,
            .total_releases = self.total_releases,
            .total_timeouts = self.total_timeouts,
            .total_errors = self.total_errors,
            .total_created = self.total_created,
            .total_destroyed = self.total_destroyed,
            .avg_wait_time_ms = avg_wait,
            .avg_connection_age_ms = avg_age,
        };
    }
    
    /// Ensure minimum pool size
    pub fn ensureMinSize(self: *PgConnectionPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        while (self.connections.items.len < self.config.min_size) {
            var conn = try self.createConnection();
            try conn.connection.connect();
            conn.markIdle();
        }
    }
    
    /// Shutdown pool gracefully
    pub fn shutdown(self: *PgConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Mark all connections as invalid
        for (self.connections.items) |*conn| {
            if (conn.state != .in_use) {
                conn.markInvalid();
            }
        }
    }
    
    /// Find a healthy idle connection (caller must hold mutex)
    fn findHealthyConnection(self: *PgConnectionPool) !?*PooledPgConnection {
        for (self.connections.items) |*conn| {
            if (conn.state == .idle and conn.isHealthy()) {
                // Quick validation check
                if (try self.validateConnection(conn)) {
                    return conn;
                } else {
                    conn.markInvalid();
                }
            }
        }
        return null;
    }
    
    /// Create a new connection (caller must hold mutex)
    fn createConnection(self: *PgConnectionPool) !*PooledPgConnection {
        var conn = try PooledPgConnection.init(
            self.allocator,
            self.config.connection_config,
            self.next_id,
        );
        errdefer conn.deinit();
        
        self.next_id += 1;
        self.total_created += 1;
        
        try self.connections.append(conn);
        return &self.connections.items[self.connections.items.len - 1];
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "PgPoolConfig - validation" {
    const conn_config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    
    // Valid config
    const valid = PgPoolConfig{
        .connection_config = conn_config,
        .min_size = 2,
        .max_size = 10,
    };
    try valid.validate();
    
    // Invalid: min > max
    const invalid1 = PgPoolConfig{
        .connection_config = conn_config,
        .min_size = 10,
        .max_size = 5,
    };
    try std.testing.expectError(error.InvalidPoolConfig, invalid1.validate());
    
    // Invalid: max = 0
    const invalid2 = PgPoolConfig{
        .connection_config = conn_config,
        .min_size = 0,
        .max_size = 0,
    };
    try std.testing.expectError(error.InvalidPoolConfig, invalid2.validate());
}

test "PgConnectionState - enum values" {
    const idle: PgConnectionState = .idle;
    const in_use: PgConnectionState = .in_use;
    const invalid: PgConnectionState = .invalid;
    const validating: PgConnectionState = .validating;
    
    try std.testing.expect(idle != in_use);
    try std.testing.expect(idle != invalid);
    try std.testing.expect(in_use != invalid);
    try std.testing.expect(validating != invalid);
}

test "PgConnectionPool - init and deinit" {
    const allocator = std.testing.allocator;
    
    const conn_config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    
    const config = PgPoolConfig{
        .connection_config = conn_config,
        .min_size = 1,
        .max_size = 5,
    };
    
    var pool = try PgConnectionPool.init(allocator, config);
    defer pool.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), pool.connections.items.len);
    try std.testing.expectEqual(@as(usize, 0), pool.next_id);
}

test "PgConnectionPool - metrics" {
    const allocator = std.testing.allocator;
    
    const conn_config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    
    const config = PgPoolConfig{
        .connection_config = conn_config,
        .min_size = 2,
        .max_size = 5,
    };
    
    var pool = try PgConnectionPool.init(allocator, config);
    defer pool.deinit();
    
    const metrics = pool.getMetrics();
    try std.testing.expectEqual(@as(usize, 0), metrics.total_connections);
    try std.testing.expectEqual(@as(u64, 0), metrics.total_acquires);
    try std.testing.expectEqual(@as(f64, 0.0), metrics.avg_wait_time_ms);
}

test "PgPoolMetrics - format" {
    const metrics = PgPoolMetrics{
        .total_connections = 10,
        .idle_connections = 3,
        .active_connections = 5,
        .invalid_connections = 2,
        .total_acquires = 100,
        .total_releases = 95,
        .total_timeouts = 2,
        .total_errors = 1,
        .total_created = 10,
        .total_destroyed = 0,
        .avg_wait_time_ms = 15.5,
        .avg_connection_age_ms = 60000.0,
    };
    
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try fbs.writer().print("{}", .{metrics});
    
    const result = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, result, "total=10") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "idle=3") != null);
}

test "PooledPgConnection - age and idle time" {
    const allocator = std.testing.allocator;
    
    const conn_config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    
    var conn = try PooledPgConnection.init(allocator, conn_config, 1);
    defer conn.deinit();
    
    const age = conn.getAge();
    const idle_time = conn.getIdleTime();
    
    try std.testing.expect(age >= 0);
    try std.testing.expect(idle_time >= 0);
}
