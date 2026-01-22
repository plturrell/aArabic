const std = @import("std");
const Allocator = std.mem.Allocator;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;

/// HANA Database Client with Connection Pooling
/// Provides thread-safe connection management for SAP HANA database
pub const HanaClient = struct {
    allocator: Allocator,
    config: HanaConfig,
    pool: ConnectionPool,
    metrics: ConnectionMetrics,
    is_running: bool,
    health_check_thread: ?std.Thread = null,

    /// HANA connection configuration
    pub const HanaConfig = struct {
        host: []const u8,
        port: u16,
        database: []const u8,
        user: []const u8,
        password: []const u8,
        pool_min: u32 = 5,
        pool_max: u32 = 10,
        idle_timeout_ms: u64 = 30000,
        connection_timeout_ms: u64 = 5000,
        max_retry_attempts: u32 = 3,
        initial_retry_delay_ms: u64 = 100,
        max_retry_delay_ms: u64 = 5000,
    };

    /// Connection pool state
    const ConnectionPool = struct {
        connections: std.ArrayList(*Connection),
        available: std.ArrayList(*Connection),
        mutex: Mutex,
        condition: Condition,
        allocator: Allocator,

        fn init(allocator: Allocator) ConnectionPool {
            return .{
                .connections = std.ArrayList(*Connection).init(allocator),
                .available = std.ArrayList(*Connection).init(allocator),
                .mutex = .{},
                .condition = .{},
                .allocator = allocator,
            };
        }

        fn deinit(self: *ConnectionPool) void {
            for (self.connections.items) |conn| {
                conn.close();
                self.allocator.destroy(conn);
            }
            self.connections.deinit();
            self.available.deinit();
        }
    };

    /// Connection metrics
    pub const ConnectionMetrics = struct {
        total_connections: usize = 0,
        active_connections: usize = 0,
        idle_connections: usize = 0,
        failed_connections: usize = 0,
        total_queries: usize = 0,
        failed_queries: usize = 0,
        mutex: Mutex = .{},

        pub fn recordQuery(self: *ConnectionMetrics, success: bool) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            self.total_queries += 1;
            if (!success) {
                self.failed_queries += 1;
            }
        }

        pub fn recordConnection(self: *ConnectionMetrics, success: bool) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            if (success) {
                self.total_connections += 1;
            } else {
                self.failed_connections += 1;
            }
        }

        pub fn updatePoolState(self: *ConnectionMetrics, active: usize, idle: usize) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            
            self.active_connections = active;
            self.idle_connections = idle;
        }
    };

    /// Individual database connection
    pub const Connection = struct {
        id: u32,
        handle: ?*anyopaque = null, // ODBC connection handle (platform-specific)
        is_healthy: bool = true,
        last_used: i64,
        created_at: i64,
        query_count: usize = 0,
        allocator: Allocator,

        pub fn init(allocator: Allocator, id: u32, config: *const HanaConfig) !*Connection {
            const conn = try allocator.create(Connection);
            const now = std.time.milliTimestamp();
            
            conn.* = .{
                .id = id,
                .allocator = allocator,
                .last_used = now,
                .created_at = now,
            };

            // TODO: Initialize actual ODBC connection
            // For now, simulate connection creation
            _ = config;
            conn.handle = @ptrFromInt(@as(usize, id + 1000)); // Placeholder
            
            return conn;
        }

        pub fn close(self: *Connection) void {
            // TODO: Close ODBC connection
            self.handle = null;
            self.is_healthy = false;
        }

        pub fn healthCheck(self: *Connection) !bool {
            if (self.handle == null) return false;
            
            // TODO: Execute actual health check query
            // SELECT 1 FROM DUMMY
            
            return self.is_healthy;
        }

        pub fn execute(self: *Connection, query: []const u8) !void {
            if (!self.is_healthy or self.handle == null) {
                return error.ConnectionUnhealthy;
            }

            // TODO: Execute actual SQL query via ODBC
            _ = query;
            
            self.query_count += 1;
            self.last_used = std.time.milliTimestamp();
        }

        pub fn query(self: *Connection, sql: []const u8, allocator: Allocator) ![][]const u8 {
            if (!self.is_healthy or self.handle == null) {
                return error.ConnectionUnhealthy;
            }

            // TODO: Execute actual SQL query and return results
            _ = sql;
            
            self.query_count += 1;
            self.last_used = std.time.milliTimestamp();
            
            // Return empty result set for now
            return try allocator.alloc([]const u8, 0);
        }
    };

    /// Initialize HANA client with connection pool
    pub fn init(allocator: Allocator, config: HanaConfig) !*HanaClient {
        const client = try allocator.create(HanaClient);
        
        client.* = .{
            .allocator = allocator,
            .config = config,
            .pool = ConnectionPool.init(allocator),
            .metrics = .{},
            .is_running = false,
        };

        // Initialize minimum connections
        try client.initializePool();
        
        // Start health check thread
        client.is_running = true;
        client.health_check_thread = try std.Thread.spawn(.{}, healthCheckLoop, .{client});

        return client;
    }

    /// Deinitialize and cleanup
    pub fn deinit(self: *HanaClient) void {
        // Stop health check thread
        self.is_running = false;
        if (self.health_check_thread) |thread| {
            thread.join();
        }

        // Cleanup pool
        self.pool.deinit();
        self.allocator.destroy(self);
    }

    /// Initialize minimum number of connections in pool
    fn initializePool(self: *HanaClient) !void {
        var i: u32 = 0;
        while (i < self.config.pool_min) : (i += 1) {
            const conn = try self.createConnection(i);
            try self.pool.connections.append(conn);
            try self.pool.available.append(conn);
            self.metrics.recordConnection(true);
        }
    }

    /// Create a new connection
    fn createConnection(self: *HanaClient, id: u32) !*Connection {
        var attempts: u32 = 0;
        var delay_ms = self.config.initial_retry_delay_ms;

        while (attempts < self.config.max_retry_attempts) : (attempts += 1) {
            const conn = Connection.init(self.allocator, id, &self.config) catch |err| {
                if (attempts + 1 < self.config.max_retry_attempts) {
                    std.time.sleep(delay_ms * std.time.ns_per_ms);
                    delay_ms = @min(delay_ms * 2, self.config.max_retry_delay_ms);
                    continue;
                }
                return err;
            };

            return conn;
        }

        return error.ConnectionFailed;
    }

    /// Acquire a connection from the pool
    pub fn getConnection(self: *HanaClient) !*Connection {
        self.pool.mutex.lock();
        defer self.pool.mutex.unlock();

        // Try to get an available connection
        while (self.pool.available.items.len == 0) {
            // If we can create more connections, do so
            if (self.pool.connections.items.len < self.config.pool_max) {
                const new_id = @as(u32, @intCast(self.pool.connections.items.len));
                const conn = try self.createConnection(new_id);
                try self.pool.connections.append(conn);
                self.metrics.recordConnection(true);
                return conn;
            }

            // Wait for a connection to become available
            self.pool.condition.wait(&self.pool.mutex);
        }

        // Get the first available connection
        const conn = self.pool.available.pop();
        
        // Verify connection health
        const healthy = try conn.healthCheck();
        if (!healthy) {
            // Recreate unhealthy connection
            conn.close();
            const new_conn = try self.createConnection(conn.id);
            
            // Replace in connections list
            for (self.pool.connections.items, 0..) |c, i| {
                if (c.id == conn.id) {
                    self.pool.connections.items[i] = new_conn;
                    break;
                }
            }
            
            self.allocator.destroy(conn);
            return new_conn;
        }

        return conn;
    }

    /// Release a connection back to the pool
    pub fn releaseConnection(self: *HanaClient, conn: *Connection) void {
        self.pool.mutex.lock();
        defer self.pool.mutex.unlock();

        // Update last used timestamp
        conn.last_used = std.time.milliTimestamp();

        // Add back to available pool
        self.pool.available.append(conn) catch {
            // If append fails, close the connection
            conn.close();
            return;
        };

        // Update metrics
        const active = self.pool.connections.items.len - self.pool.available.items.len;
        self.metrics.updatePoolState(active, self.pool.available.items.len);

        // Notify waiting threads
        self.pool.condition.signal();
    }

    /// Execute a query with automatic connection management
    pub fn execute(self: *HanaClient, query: []const u8) !void {
        const conn = try self.getConnection();
        defer self.releaseConnection(conn);

        conn.execute(query) catch |err| {
            self.metrics.recordQuery(false);
            return err;
        };

        self.metrics.recordQuery(true);
    }

    /// Execute a query and return results
    pub fn query(self: *HanaClient, sql: []const u8, allocator: Allocator) ![][]const u8 {
        const conn = try self.getConnection();
        defer self.releaseConnection(conn);

        const result = conn.query(sql, allocator) catch |err| {
            self.metrics.recordQuery(false);
            return err;
        };

        self.metrics.recordQuery(true);
        return result;
    }

    /// Get current metrics
    pub fn getMetrics(self: *HanaClient) ConnectionMetrics {
        self.metrics.mutex.lock();
        defer self.metrics.mutex.unlock();
        return self.metrics;
    }

    /// Health check loop (runs in separate thread)
    fn healthCheckLoop(self: *HanaClient) void {
        while (self.is_running) {
            std.time.sleep(60 * std.time.ns_per_s); // Check every 60 seconds

            self.pool.mutex.lock();
            defer self.pool.mutex.unlock();

            // Check all connections
            for (self.pool.connections.items) |conn| {
                const healthy = conn.healthCheck() catch false;
                if (!healthy) {
                    conn.is_healthy = false;
                    std.log.warn("Connection {d} is unhealthy", .{conn.id});
                }

                // Check for idle connections
                const now = std.time.milliTimestamp();
                const idle_time = now - conn.last_used;
                if (idle_time > self.config.idle_timeout_ms) {
                    std.log.info("Connection {d} idle for {d}ms", .{ conn.id, idle_time });
                }
            }

            // Update metrics
            const active = self.pool.connections.items.len - self.pool.available.items.len;
            self.metrics.updatePoolState(active, self.pool.available.items.len);
        }
    }

    /// Perform overall health check
    pub fn healthCheck(self: *HanaClient) !bool {
        const conn = try self.getConnection();
        defer self.releaseConnection(conn);

        return try conn.healthCheck();
    }
};

// Configuration helper for loading from environment
pub fn loadConfigFromEnv(allocator: Allocator) !HanaClient.HanaConfig {
    const env_map = try std.process.getEnvMap(allocator);
    defer env_map.deinit();

    return .{
        .host = try allocator.dupe(u8, env_map.get("HANA_HOST") orelse "localhost"),
        .port = try std.fmt.parseInt(u16, env_map.get("HANA_PORT") orelse "30015", 10),
        .database = try allocator.dupe(u8, env_map.get("HANA_DATABASE") orelse "NOPENAI_DB"),
        .user = try allocator.dupe(u8, env_map.get("HANA_USER") orelse "NUCLEUS_APP"),
        .password = try allocator.dupe(u8, env_map.get("HANA_PASSWORD") orelse ""),
        .pool_min = try std.fmt.parseInt(u32, env_map.get("HANA_POOL_MIN") orelse "5", 10),
        .pool_max = try std.fmt.parseInt(u32, env_map.get("HANA_POOL_MAX") orelse "10", 10),
    };
}

test "HanaClient initialization" {
    const allocator = std.testing.allocator;
    
    const config = HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 2,
        .pool_max = 5,
    };

    const client = try HanaClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expect(client.pool.connections.items.len == 2);
    try std.testing.expect(client.pool.available.items.len == 2);
}

test "HanaClient connection acquisition and release" {
    const allocator = std.testing.allocator;
    
    const config = HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 2,
        .pool_max = 5,
    };

    const client = try HanaClient.init(allocator, config);
    defer client.deinit();

    // Acquire a connection
    const conn = try client.getConnection();
    try std.testing.expect(client.pool.available.items.len == 1);

    // Release it back
    client.releaseConnection(conn);
    try std.testing.expect(client.pool.available.items.len == 2);
}

test "HanaClient metrics tracking" {
    const allocator = std.testing.allocator;
    
    const config = HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 2,
        .pool_max = 5,
    };

    const client = try HanaClient.init(allocator, config);
    defer client.deinit();

    const metrics = client.getMetrics();
    try std.testing.expect(metrics.total_connections == 2);
    try std.testing.expect(metrics.active_connections == 0);
    try std.testing.expect(metrics.idle_connections == 2);
}
