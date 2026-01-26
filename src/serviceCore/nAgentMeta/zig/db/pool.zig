const std = @import("std");
const client = @import("client.zig");
const DbClient = client.DbClient;

/// Connection state in the pool
pub const ConnectionState = enum {
    idle, // Available for use
    in_use, // Currently being used
    invalid, // Connection failed, needs cleanup
};

/// Pooled connection wrapper
pub const PooledConnection = struct {
    client: *DbClient,
    state: ConnectionState,
    last_used: i64, // Unix timestamp in milliseconds
    use_count: usize,
    id: usize,

    pub fn markUsed(self: *PooledConnection) void {
        self.state = .in_use;
        self.last_used = std.time.milliTimestamp();
        self.use_count += 1;
    }

    pub fn markIdle(self: *PooledConnection) void {
        self.state = .idle;
        self.last_used = std.time.milliTimestamp();
    }

    pub fn markInvalid(self: *PooledConnection) void {
        self.state = .invalid;
    }

    pub fn isHealthy(self: PooledConnection) bool {
        return self.state != .invalid;
    }
};

/// Connection pool configuration
pub const PoolConfig = struct {
    min_size: usize = 2,
    max_size: usize = 10,
    acquire_timeout_ms: i64 = 5000,
    idle_timeout_ms: i64 = 300000, // 5 minutes
    max_lifetime_ms: i64 = 1800000, // 30 minutes
    health_check_interval_ms: i64 = 60000, // 1 minute

    pub fn validate(self: PoolConfig) !void {
        if (self.min_size > self.max_size) {
            return error.InvalidPoolConfig;
        }
        if (self.max_size == 0) {
            return error.InvalidPoolConfig;
        }
        if (self.acquire_timeout_ms <= 0) {
            return error.InvalidPoolConfig;
        }
    }
};

/// Wait queue entry for blocked acquire requests
const WaitEntry = struct {
    timestamp: i64,
    completed: bool,
    connection: ?*PooledConnection,
};

/// Connection pool metrics
pub const PoolMetrics = struct {
    total_connections: usize,
    idle_connections: usize,
    active_connections: usize,
    invalid_connections: usize,
    wait_queue_size: usize,
    total_acquires: u64,
    total_releases: u64,
    total_timeouts: u64,
    total_errors: u64,
    avg_wait_time_ms: f64,

    pub fn format(self: PoolMetrics, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "PoolMetrics{{ total={d}, idle={d}, active={d}, invalid={d}, waiting={d}, acquires={d}, timeouts={d} }}",
            .{
                self.total_connections,
                self.idle_connections,
                self.active_connections,
                self.invalid_connections,
                self.wait_queue_size,
                self.total_acquires,
                self.total_timeouts,
            },
        );
    }
};

/// Thread-safe connection pool
pub const ConnectionPool = struct {
    allocator: std.mem.Allocator,
    config: PoolConfig,
    connections: std.ArrayList(PooledConnection),
    mutex: std.Thread.Mutex,
    wait_queue: std.ArrayList(WaitEntry),
    next_id: usize,
    
    // Metrics
    total_acquires: u64,
    total_releases: u64,
    total_timeouts: u64,
    total_errors: u64,
    total_wait_time_ms: u64,

    pub fn init(allocator: std.mem.Allocator, config: PoolConfig) !ConnectionPool {
        try config.validate();

        return ConnectionPool{
            .allocator = allocator,
            .config = config,
            .connections = std.ArrayList(PooledConnection){},
            .mutex = std.Thread.Mutex{},
            .wait_queue = std.ArrayList(WaitEntry){},
            .next_id = 0,
            .total_acquires = 0,
            .total_releases = 0,
            .total_timeouts = 0,
            .total_errors = 0,
            .total_wait_time_ms = 0,
        };
    }

    pub fn deinit(self: *ConnectionPool) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up all connections
        for (self.connections.items) |*conn| {
            conn.client.disconnect();
        }
        self.connections.deinit();
        self.wait_queue.deinit();
    }

    /// Acquire a connection from the pool (thread-safe)
    pub fn acquire(self: *ConnectionPool) !*PooledConnection {
        const start_time = std.time.milliTimestamp();
        
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_acquires += 1;

        // Try to find idle connection
        if (self.findIdleConnection()) |conn| {
            conn.markUsed();
            const wait_time = std.time.milliTimestamp() - start_time;
            self.total_wait_time_ms += @intCast(wait_time);
            return conn;
        }

        // No idle connection, try to create new one if under max
        if (self.connections.items.len < self.config.max_size) {
            const conn = try self.createConnection();
            conn.markUsed();
            const wait_time = std.time.milliTimestamp() - start_time;
            self.total_wait_time_ms += @intCast(wait_time);
            return conn;
        }

        // Pool is full, check timeout
        const elapsed = std.time.milliTimestamp() - start_time;
        if (elapsed >= self.config.acquire_timeout_ms) {
            self.total_timeouts += 1;
            return error.AcquireTimeout;
        }

        // Would need to wait - for now return timeout
        // In a full implementation, this would add to wait_queue
        self.total_timeouts += 1;
        return error.AcquireTimeout;
    }

    /// Release a connection back to the pool (thread-safe)
    pub fn release(self: *ConnectionPool, conn: *PooledConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_releases += 1;

        // Check if connection is still healthy
        if (conn.isHealthy()) {
            conn.markIdle();
            // Wake up any waiting requests
            self.tryWakeWaiter();
        } else {
            // Connection is invalid, mark for removal
            conn.markInvalid();
        }
    }

    /// Get pool metrics (thread-safe)
    pub fn getMetrics(self: *ConnectionPool) PoolMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        var idle: usize = 0;
        var active: usize = 0;
        var invalid: usize = 0;

        for (self.connections.items) |conn| {
            switch (conn.state) {
                .idle => idle += 1,
                .in_use => active += 1,
                .invalid => invalid += 1,
            }
        }

        const avg_wait = if (self.total_acquires > 0)
            @as(f64, @floatFromInt(self.total_wait_time_ms)) / @as(f64, @floatFromInt(self.total_acquires))
        else
            0.0;

        return PoolMetrics{
            .total_connections = self.connections.items.len,
            .idle_connections = idle,
            .active_connections = active,
            .invalid_connections = invalid,
            .wait_queue_size = self.wait_queue.items.len,
            .total_acquires = self.total_acquires,
            .total_releases = self.total_releases,
            .total_timeouts = self.total_timeouts,
            .total_errors = self.total_errors,
            .avg_wait_time_ms = avg_wait,
        };
    }

    /// Perform health check on all connections (thread-safe)
    pub fn healthCheck(self: *ConnectionPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const now = std.time.milliTimestamp();

        for (self.connections.items) |*conn| {
            // Skip connections in use
            if (conn.state == .in_use) continue;

            // Check idle timeout
            const idle_time = now - conn.last_used;
            if (idle_time > self.config.idle_timeout_ms) {
                conn.markInvalid();
                continue;
            }

            // Try to ping connection
            const is_alive = conn.client.ping() catch false;
            if (!is_alive) {
                conn.markInvalid();
            }
        }

        // Remove invalid connections
        try self.cleanupInvalidConnections();
    }

    /// Ensure minimum pool size (thread-safe)
    pub fn ensureMinSize(self: *ConnectionPool) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        while (self.connections.items.len < self.config.min_size) {
            _ = try self.createConnection();
        }
    }

    /// Find an idle connection (caller must hold mutex)
    fn findIdleConnection(self: *ConnectionPool) ?*PooledConnection {
        for (self.connections.items) |*conn| {
            if (conn.state == .idle) {
                return conn;
            }
        }
        return null;
    }

    /// Create a new connection (caller must hold mutex)
    fn createConnection(self: *ConnectionPool) !*PooledConnection {
        // In real implementation, this would create actual DbClient
        // For now, we'll create a placeholder
        const conn_client = try self.allocator.create(DbClient);
        errdefer self.allocator.destroy(conn_client);

        const pooled = PooledConnection{
            .client = conn_client,
            .state = .idle,
            .last_used = std.time.milliTimestamp(),
            .use_count = 0,
            .id = self.next_id,
        };

        self.next_id += 1;
        try self.connections.append(pooled);

        return &self.connections.items[self.connections.items.len - 1];
    }

    /// Remove invalid connections (caller must hold mutex)
    fn cleanupInvalidConnections(self: *ConnectionPool) !void {
        var i: usize = 0;
        while (i < self.connections.items.len) {
            if (self.connections.items[i].state == .invalid) {
                const conn = self.connections.orderedRemove(i);
                conn.client.disconnect();
                self.allocator.destroy(conn.client);
            } else {
                i += 1;
            }
        }
    }

    /// Try to wake up a waiting request (caller must hold mutex)
    fn tryWakeWaiter(self: *ConnectionPool) void {
        if (self.wait_queue.items.len > 0) {
            // Find first incomplete wait entry
            for (self.wait_queue.items) |*entry| {
                if (!entry.completed) {
                    if (self.findIdleConnection()) |conn| {
                        entry.connection = conn;
                        entry.completed = true;
                        conn.markUsed();
                        break;
                    }
                }
            }
        }
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "PoolConfig - validation" {
    // Valid config
    const valid = PoolConfig{
        .min_size = 2,
        .max_size = 10,
    };
    try valid.validate();

    // Invalid: min > max
    const invalid1 = PoolConfig{
        .min_size = 10,
        .max_size = 5,
    };
    try std.testing.expectError(error.InvalidPoolConfig, invalid1.validate());

    // Invalid: max = 0
    const invalid2 = PoolConfig{
        .min_size = 0,
        .max_size = 0,
    };
    try std.testing.expectError(error.InvalidPoolConfig, invalid2.validate());
}

test "ConnectionPool - init and deinit" {
    const allocator = std.testing.allocator;

    const config = PoolConfig{
        .min_size = 1,
        .max_size = 5,
    };

    var pool = try ConnectionPool.init(allocator, config);
    defer pool.deinit();

    try std.testing.expectEqual(@as(usize, 0), pool.connections.items.len);
}

test "ConnectionPool - metrics" {
    const allocator = std.testing.allocator;

    const config = PoolConfig{
        .min_size = 2,
        .max_size = 5,
    };

    var pool = try ConnectionPool.init(allocator, config);
    defer pool.deinit();

    const metrics = pool.getMetrics();
    try std.testing.expectEqual(@as(usize, 0), metrics.total_connections);
    try std.testing.expectEqual(@as(u64, 0), metrics.total_acquires);
}

test "PooledConnection - state transitions" {
    const allocator = std.testing.allocator;

    var client_instance = DbClient{
        .vtable = undefined,
        .context = undefined,
        .allocator = allocator,
    };

    var conn = PooledConnection{
        .client = &client_instance,
        .state = .idle,
        .last_used = 0,
        .use_count = 0,
        .id = 1,
    };

    try std.testing.expectEqual(ConnectionState.idle, conn.state);
    try std.testing.expect(conn.isHealthy());

    conn.markUsed();
    try std.testing.expectEqual(ConnectionState.in_use, conn.state);
    try std.testing.expectEqual(@as(usize, 1), conn.use_count);

    conn.markIdle();
    try std.testing.expectEqual(ConnectionState.idle, conn.state);

    conn.markInvalid();
    try std.testing.expectEqual(ConnectionState.invalid, conn.state);
    try std.testing.expect(!conn.isHealthy());
}

test "ConnectionState - enum values" {
    const idle: ConnectionState = .idle;
    const in_use: ConnectionState = .in_use;
    const invalid: ConnectionState = .invalid;

    try std.testing.expect(idle != in_use);
    try std.testing.expect(idle != invalid);
    try std.testing.expect(in_use != invalid);
}

test "PoolMetrics - format" {
    const metrics = PoolMetrics{
        .total_connections = 10,
        .idle_connections = 3,
        .active_connections = 5,
        .invalid_connections = 2,
        .wait_queue_size = 1,
        .total_acquires = 100,
        .total_releases = 95,
        .total_timeouts = 2,
        .total_errors = 1,
        .avg_wait_time_ms = 15.5,
    };

    var buf: [200]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try fbs.writer().print("{}", .{metrics});

    const result = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, result, "total=10") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "idle=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "active=5") != null);
}
