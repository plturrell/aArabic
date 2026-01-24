const std = @import("std");
const Allocator = std.mem.Allocator;
const Mutex = std.Thread.Mutex;
const Condition = std.Thread.Condition;
const StringHashMap = std.StringHashMap;

// ============================================================================
// Health Status Types
// ============================================================================

/// Health status enumeration for services
pub const ServiceStatus = enum {
    healthy,
    degraded,
    unhealthy,

    pub fn toString(self: ServiceStatus) []const u8 {
        return switch (self) {
            .healthy => "healthy",
            .degraded => "degraded",
            .unhealthy => "unhealthy",
        };
    }

    pub fn fromLatency(latency_ms: u64, threshold_ms: u64) ServiceStatus {
        if (latency_ms == 0) return .unhealthy;
        if (latency_ms < threshold_ms) return .healthy;
        if (latency_ms < threshold_ms * 3) return .degraded;
        return .unhealthy;
    }
};

/// Health status for a single service
pub const HealthStatus = struct {
    service: []const u8,
    status: ServiceStatus,
    latency_ms: u64,
    last_check: i64,
    error_message: ?[]const u8,
    allocator: Allocator,

    pub fn init(allocator: Allocator, service: []const u8, status: ServiceStatus, latency_ms: u64) !HealthStatus {
        return .{
            .service = try allocator.dupe(u8, service),
            .status = status,
            .latency_ms = latency_ms,
            .last_check = std.time.timestamp(),
            .error_message = null,
            .allocator = allocator,
        };
    }

    pub fn initWithError(allocator: Allocator, service: []const u8, error_msg: []const u8) !HealthStatus {
        return .{
            .service = try allocator.dupe(u8, service),
            .status = .unhealthy,
            .latency_ms = 0,
            .last_check = std.time.timestamp(),
            .error_message = try allocator.dupe(u8, error_msg),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HealthStatus) void {
        self.allocator.free(self.service);
        if (self.error_message) |msg| {
            self.allocator.free(msg);
        }
    }

    pub fn isHealthy(self: *const HealthStatus) bool {
        return self.status == .healthy;
    }
};

/// Aggregated health report for all services
pub const HealthReport = struct {
    services: []HealthStatus,
    overall_status: ServiceStatus,
    timestamp: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator, services: []HealthStatus) HealthReport {
        var overall: ServiceStatus = .healthy;
        for (services) |s| {
            if (s.status == .unhealthy) {
                overall = .unhealthy;
                break;
            } else if (s.status == .degraded and overall == .healthy) {
                overall = .degraded;
            }
        }

        return .{
            .services = services,
            .overall_status = overall,
            .timestamp = std.time.timestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *HealthReport) void {
        for (self.services) |*s| {
            s.deinit();
        }
        self.allocator.free(self.services);
    }

    pub fn getServiceStatus(self: *const HealthReport, service_name: []const u8) ?ServiceStatus {
        for (self.services) |s| {
            if (std.mem.eql(u8, s.service, service_name)) {
                return s.status;
            }
        }
        return null;
    }
};

// ============================================================================
// Circuit Breaker Pattern
// ============================================================================

/// Circuit breaker state
pub const CircuitState = enum {
    closed, // Normal operation, requests pass through
    open, // Failure threshold exceeded, requests blocked
    half_open, // Testing if service recovered

    pub fn toString(self: CircuitState) []const u8 {
        return switch (self) {
            .closed => "closed",
            .open => "open",
            .half_open => "half_open",
        };
    }
};

/// Circuit breaker configuration
pub const CircuitBreakerConfig = struct {
    failure_threshold: u32 = 5,
    success_threshold: u32 = 3,
    reset_timeout_ms: u32 = 30000,
    half_open_max_requests: u32 = 3,
};

/// Circuit breaker for preventing cascading failures
pub const CircuitBreaker = struct {
    name: []const u8,
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: i64,
    config: CircuitBreakerConfig,
    mutex: Mutex,
    allocator: Allocator,

    pub fn init(allocator: Allocator, name: []const u8, config: CircuitBreakerConfig) !*CircuitBreaker {
        const cb = try allocator.create(CircuitBreaker);
        cb.* = .{
            .name = try allocator.dupe(u8, name),
            .state = .closed,
            .failure_count = 0,
            .success_count = 0,
            .last_failure_time = 0,
            .config = config,
            .mutex = .{},
            .allocator = allocator,
        };
        return cb;
    }

    pub fn deinit(self: *CircuitBreaker) void {
        self.allocator.free(self.name);
        self.allocator.destroy(self);
    }

    /// Check if request should be allowed through
    pub fn allowRequest(self: *CircuitBreaker) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        switch (self.state) {
            .closed => return true,
            .open => {
                const now = std.time.milliTimestamp();
                const elapsed = now - self.last_failure_time;
                if (elapsed >= self.config.reset_timeout_ms) {
                    self.state = .half_open;
                    self.success_count = 0;
                    return true;
                }
                return false;
            },
            .half_open => {
                return self.success_count < self.config.half_open_max_requests;
            },
        }
    }

    /// Record a successful request
    pub fn recordSuccess(self: *CircuitBreaker) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.failure_count = 0;
        if (self.state == .half_open) {
            self.success_count += 1;
            if (self.success_count >= self.config.success_threshold) {
                self.state = .closed;
            }
        }
    }

    /// Record a failed request
    pub fn recordFailure(self: *CircuitBreaker) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.failure_count += 1;
        self.last_failure_time = std.time.milliTimestamp();

        if (self.state == .half_open) {
            self.state = .open;
        } else if (self.failure_count >= self.config.failure_threshold) {
            self.state = .open;
        }
    }

    /// Reset the circuit breaker
    pub fn reset(self: *CircuitBreaker) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.state = .closed;
        self.failure_count = 0;
        self.success_count = 0;
        self.last_failure_time = 0;
    }

    /// Get current state string
    pub fn getState(self: *CircuitBreaker) CircuitState {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.state;
    }
};

// ============================================================================
// Generic Connection Pool
// ============================================================================

/// Connection pool configuration
pub const ConnectionPoolConfig = struct {
    min_connections: u32 = 2,
    max_connections: u32 = 10,
    connection_timeout_ms: u32 = 5000,
    idle_timeout_ms: u32 = 60000,
    max_lifetime_ms: u32 = 3600000,
    health_check_interval_ms: u32 = 30000,
};

/// Generic connection wrapper
pub fn PooledConnection(comptime T: type) type {
    return struct {
        connection: *T,
        created_at: i64,
        last_used: i64,
        in_use: bool,

        const Self = @This();

        pub fn init(conn: *T) Self {
            const now = std.time.milliTimestamp();
            return .{
                .connection = conn,
                .created_at = now,
                .last_used = now,
                .in_use = false,
            };
        }

        pub fn markUsed(self: *Self) void {
            self.last_used = std.time.milliTimestamp();
            self.in_use = true;
        }

        pub fn markIdle(self: *Self) void {
            self.in_use = false;
        }

        pub fn isExpired(self: *const Self, max_lifetime_ms: u32) bool {
            const now = std.time.milliTimestamp();
            return (now - self.created_at) >= max_lifetime_ms;
        }

        pub fn isIdle(self: *const Self, idle_timeout_ms: u32) bool {
            const now = std.time.milliTimestamp();
            return !self.in_use and (now - self.last_used) >= idle_timeout_ms;
        }
    };
}


/// Generic connection pool for any database connection type
pub fn ConnectionPool(comptime T: type, comptime ConnectorFn: type) type {
    return struct {
        allocator: Allocator,
        config: ConnectionPoolConfig,
        connections: std.ArrayList(PooledConnection(T)),
        available_count: u32,
        total_count: u32,
        mutex: Mutex,
        condition: Condition,
        connector: ConnectorFn,
        circuit_breaker: *CircuitBreaker,
        metrics: PoolMetrics,
        shutdown: bool,

        const Self = @This();
        const PooledConn = PooledConnection(T);

        /// Pool metrics for monitoring
        pub const PoolMetrics = struct {
            total_acquisitions: u64 = 0,
            total_releases: u64 = 0,
            total_timeouts: u64 = 0,
            total_errors: u64 = 0,
            current_active: u32 = 0,
            current_idle: u32 = 0,
            max_wait_time_ms: u64 = 0,
        };

        pub fn init(allocator: Allocator, config: ConnectionPoolConfig, connector: ConnectorFn) !*Self {
            const pool = try allocator.create(Self);
            errdefer allocator.destroy(pool);

            const cb = try CircuitBreaker.init(allocator, "connection_pool", .{});
            errdefer cb.deinit();

            pool.* = .{
                .allocator = allocator,
                .config = config,
                .connections = std.ArrayList(PooledConn).init(allocator),
                .available_count = 0,
                .total_count = 0,
                .mutex = .{},
                .condition = .{},
                .connector = connector,
                .circuit_breaker = cb,
                .metrics = .{},
                .shutdown = false,
            };

            // Create minimum connections
            try pool.ensureMinConnections();

            return pool;
        }

        pub fn deinit(self: *Self) void {
            self.mutex.lock();
            self.shutdown = true;
            self.condition.broadcast();
            self.mutex.unlock();

            // Close all connections
            for (self.connections.items) |*pooled| {
                self.connector.close(pooled.connection);
                self.allocator.destroy(pooled.connection);
            }
            self.connections.deinit();
            self.circuit_breaker.deinit();
            self.allocator.destroy(self);
        }

        fn ensureMinConnections(self: *Self) !void {
            while (self.total_count < self.config.min_connections) {
                try self.createConnection();
            }
        }

        fn createConnection(self: *Self) !void {
            const conn = try self.allocator.create(T);
            errdefer self.allocator.destroy(conn);

            try self.connector.connect(conn);

            var pooled = PooledConn.init(conn);
            try self.connections.append(pooled);
            self.total_count += 1;
            self.available_count += 1;
        }

        /// Acquire a connection from the pool
        pub fn acquire(self: *Self) !*T {
            const start_time = std.time.milliTimestamp();

            self.mutex.lock();
            defer self.mutex.unlock();

            if (self.shutdown) return error.PoolShutdown;

            if (!self.circuit_breaker.allowRequest()) {
                self.metrics.total_errors += 1;
                return error.CircuitOpen;
            }

            // Try to find an available connection
            while (true) {
                for (self.connections.items) |*pooled| {
                    if (!pooled.in_use) {
                        // Check if connection is expired
                        if (pooled.isExpired(self.config.max_lifetime_ms)) {
                            self.removeConnection(pooled);
                            continue;
                        }

                        pooled.markUsed();
                        self.available_count -= 1;
                        self.metrics.total_acquisitions += 1;
                        self.metrics.current_active += 1;
                        self.metrics.current_idle = self.available_count;

                        const wait_time = @as(u64, @intCast(std.time.milliTimestamp() - start_time));
                        if (wait_time > self.metrics.max_wait_time_ms) {
                            self.metrics.max_wait_time_ms = wait_time;
                        }

                        self.circuit_breaker.recordSuccess();
                        return pooled.connection;
                    }
                }

                // No available connections, try to create one
                if (self.total_count < self.config.max_connections) {
                    self.createConnection() catch |err| {
                        self.circuit_breaker.recordFailure();
                        self.metrics.total_errors += 1;
                        return err;
                    };
                    continue;
                }

                // Wait for a connection to become available
                const elapsed = std.time.milliTimestamp() - start_time;
                if (elapsed >= self.config.connection_timeout_ms) {
                    self.metrics.total_timeouts += 1;
                    return error.ConnectionTimeout;
                }

                self.condition.timedWait(&self.mutex, self.config.connection_timeout_ms * std.time.ns_per_ms) catch {
                    self.metrics.total_timeouts += 1;
                    return error.ConnectionTimeout;
                };

                if (self.shutdown) return error.PoolShutdown;
            }
        }

        /// Release a connection back to the pool
        pub fn release(self: *Self, conn: *T) void {
            self.mutex.lock();
            defer self.mutex.unlock();

            for (self.connections.items) |*pooled| {
                if (pooled.connection == conn) {
                    pooled.markIdle();
                    self.available_count += 1;
                    self.metrics.total_releases += 1;
                    self.metrics.current_active -= 1;
                    self.metrics.current_idle = self.available_count;
                    self.condition.signal();
                    return;
                }
            }
        }

        fn removeConnection(self: *Self, pooled: *PooledConn) void {
            self.connector.close(pooled.connection);
            self.allocator.destroy(pooled.connection);

            // Find and remove from list
            for (self.connections.items, 0..) |*p, i| {
                if (p == pooled) {
                    _ = self.connections.orderedRemove(i);
                    break;
                }
            }

            self.total_count -= 1;
            if (!pooled.in_use) {
                self.available_count -= 1;
            }
        }

        /// Perform health check on all connections
        pub fn healthCheck(self: *Self) bool {
            self.mutex.lock();
            defer self.mutex.unlock();

            var healthy_count: u32 = 0;
            for (self.connections.items) |*pooled| {
                if (!pooled.in_use) {
                    if (self.connector.ping(pooled.connection)) {
                        healthy_count += 1;
                    } else {
                        self.removeConnection(pooled);
                    }
                }
            }

            // Ensure minimum connections
            self.ensureMinConnections() catch {
                return false;
            };

            return healthy_count > 0 or self.total_count >= self.config.min_connections;
        }

        /// Get current pool metrics
        pub fn getMetrics(self: *Self) PoolMetrics {
            self.mutex.lock();
            defer self.mutex.unlock();
            return self.metrics;
        }

        /// Get pool statistics
        pub fn getStats(self: *Self) struct { total: u32, active: u32, idle: u32 } {
            self.mutex.lock();
            defer self.mutex.unlock();
            return .{
                .total = self.total_count,
                .active = self.total_count - self.available_count,
                .idle = self.available_count,
            };
        }
    };
}



// ============================================================================
// Data Layer Configuration
// ============================================================================


/// Qdrant configuration
pub const QdrantConfig = struct {
    url: []const u8 = "http://localhost:6333",
    api_key: ?[]const u8 = null,
    collection: []const u8 = "workflow_vectors",
    timeout_ms: u32 = 5000,
};

/// Memgraph configuration
pub const MemgraphConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 7687,
    user: []const u8 = "memgraph",
    password: []const u8 = "",
    pool: ConnectionPoolConfig = .{},
};

/// Marquez configuration
pub const MarquezConfig = struct {
    url: []const u8 = "http://localhost:5000",
    namespace: []const u8 = "nworkflow",
    api_key: ?[]const u8 = null,
    timeout_ms: u32 = 5000,
};

/// SAP HANA configuration
pub const HanaPoolConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 443, // Default port for HANA Cloud
    user: []const u8 = "DBADMIN",
    password: []const u8 = "",
    schema: []const u8 = "DBADMIN",
    use_tls: bool = true, // Default true for cloud
    min_connections: u32 = 2,
    max_connections: u32 = 10,
    connection_timeout_ms: u32 = 5000,

    /// Build a connection string for HANA
    pub fn buildConnectionString(self: *const HanaPoolConfig, allocator: Allocator) ![]u8 {
        const protocol = if (self.use_tls) "hdbsqls" else "hdbsql";
        return std.fmt.allocPrint(allocator, "{s}://{s}:{d}?user={s}&schema={s}", .{
            protocol,
            self.host,
            self.port,
            self.user,
            self.schema,
        });
    }
};

/// Complete data layer configuration
pub const DataLayerConfig = struct {
    hana: HanaPoolConfig,
    qdrant: QdrantConfig = .{},
    memgraph: MemgraphConfig = .{},
    marquez: MarquezConfig = .{},
    health_check_interval_ms: u32 = 30000,
    enable_metrics: bool = true,
    enable_circuit_breakers: bool = true,
};

// ============================================================================
// HANA Pool Functions
// ============================================================================

/// HANA connection placeholder (in production, would use actual HANA client)
pub const HanaConnection = struct {
    connected: bool = false,
    host: []const u8 = "",
    port: u16 = 443,
    schema: []const u8 = "",
};

/// HANA connector for use with ConnectionPool
pub const HanaConnector = struct {
    config: HanaPoolConfig,

    pub fn connect(self: *const HanaConnector, conn: *HanaConnection) !void {
        // In production, this would establish actual HANA connection
        // using SAP HANA client libraries
        conn.* = .{
            .connected = true,
            .host = self.config.host,
            .port = self.config.port,
            .schema = self.config.schema,
        };
    }

    pub fn close(_: *const HanaConnector, conn: *HanaConnection) void {
        conn.connected = false;
    }

    pub fn ping(_: *const HanaConnector, conn: *HanaConnection) bool {
        // In production, would execute "SELECT 1 FROM DUMMY"
        return conn.connected;
    }
};

/// HANA connection pool type
pub const HanaPool = ConnectionPool(HanaConnection, HanaConnector);

/// Initialize a HANA connection pool from configuration
pub fn initHanaPool(allocator: Allocator, config: HanaPoolConfig) !*HanaPool {
    const pool_config = ConnectionPoolConfig{
        .min_connections = config.min_connections,
        .max_connections = config.max_connections,
        .connection_timeout_ms = config.connection_timeout_ms,
        .idle_timeout_ms = 60000,
        .max_lifetime_ms = 3600000,
        .health_check_interval_ms = 30000,
    };

    const connector = HanaConnector{ .config = config };
    return try HanaPool.init(allocator, pool_config, connector);
}

/// JSON parsing error type
pub const HanaConfigError = error{
    InvalidJson,
    MissingField,
    InvalidFieldType,
    OutOfMemory,
};

/// Load HANA configuration from JSON string
pub fn loadHanaConfigFromJson(allocator: Allocator, json: []const u8) HanaConfigError!HanaPoolConfig {
    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json, .{}) catch {
        return HanaConfigError.InvalidJson;
    };
    defer parsed.deinit();

    const root = parsed.value;
    if (root != .object) return HanaConfigError.InvalidJson;

    const database = root.object.get("database") orelse return HanaConfigError.MissingField;
    if (database != .object) return HanaConfigError.InvalidFieldType;

    const connection = database.object.get("connection") orelse return HanaConfigError.MissingField;
    if (connection != .object) return HanaConfigError.InvalidFieldType;

    const pool = database.object.get("pool") orelse return HanaConfigError.MissingField;
    if (pool != .object) return HanaConfigError.InvalidFieldType;

    // Extract connection fields
    const host_val = connection.object.get("host") orelse return HanaConfigError.MissingField;
    const host = if (host_val == .string) host_val.string else return HanaConfigError.InvalidFieldType;

    const port_val = connection.object.get("port") orelse return HanaConfigError.MissingField;
    const port: u16 = if (port_val == .integer) @intCast(port_val.integer) else return HanaConfigError.InvalidFieldType;

    const user_val = connection.object.get("user") orelse return HanaConfigError.MissingField;
    const user = if (user_val == .string) user_val.string else return HanaConfigError.InvalidFieldType;

    const schema_val = connection.object.get("schema") orelse return HanaConfigError.MissingField;
    const schema = if (schema_val == .string) schema_val.string else return HanaConfigError.InvalidFieldType;

    const use_tls_val = connection.object.get("use_tls");
    const use_tls = if (use_tls_val) |v| (if (v == .bool) v.bool else true) else true;

    // Extract pool fields
    const min_size_val = pool.object.get("min_size") orelse return HanaConfigError.MissingField;
    const min_size: u32 = if (min_size_val == .integer) @intCast(min_size_val.integer) else return HanaConfigError.InvalidFieldType;

    const max_size_val = pool.object.get("max_size") orelse return HanaConfigError.MissingField;
    const max_size: u32 = if (max_size_val == .integer) @intCast(max_size_val.integer) else return HanaConfigError.InvalidFieldType;

    // Duplicate strings to ensure they outlive the parsed JSON
    const host_copy = allocator.dupe(u8, host) catch return HanaConfigError.OutOfMemory;
    errdefer allocator.free(host_copy);
    const user_copy = allocator.dupe(u8, user) catch return HanaConfigError.OutOfMemory;
    errdefer allocator.free(user_copy);
    const schema_copy = allocator.dupe(u8, schema) catch return HanaConfigError.OutOfMemory;

    return HanaPoolConfig{
        .host = host_copy,
        .port = port,
        .user = user_copy,
        .password = "", // Password should be loaded separately for security
        .schema = schema_copy,
        .use_tls = use_tls,
        .min_connections = min_size,
        .max_connections = max_size,
        .connection_timeout_ms = 5000,
    };
}

// ============================================================================
// Health Checker
// ============================================================================

/// Health checker for all data services
pub const HealthChecker = struct {
    allocator: Allocator,
    config: DataLayerConfig,
    last_check_results: ?[]HealthStatus,
    check_in_progress: bool,
    mutex: Mutex,

    pub fn init(allocator: Allocator, config: DataLayerConfig) !*HealthChecker {
        const checker = try allocator.create(HealthChecker);
        checker.* = .{
            .allocator = allocator,
            .config = config,
            .last_check_results = null,
            .check_in_progress = false,
            .mutex = .{},
        };
        return checker;
    }

    pub fn deinit(self: *HealthChecker) void {
        if (self.last_check_results) |results| {
            for (results) |*r| {
                r.deinit();
            }
            self.allocator.free(results);
        }
        self.allocator.destroy(self);
    }


    /// Check Qdrant health
    pub fn checkQdrant(self: *HealthChecker, url: []const u8) !HealthStatus {
        _ = url;
        const start = std.time.milliTimestamp();

        // Simulate health check - in production would call GET /health
        const latency = @as(u64, @intCast(std.time.milliTimestamp() - start));
        const status = ServiceStatus.fromLatency(latency + 1, self.config.health_check_interval_ms / 10);

        return HealthStatus.init(self.allocator, "qdrant", status, latency + 1);
    }

    /// Check Memgraph health
    pub fn checkMemgraph(self: *HealthChecker, conn_string: []const u8) !HealthStatus {
        _ = conn_string;
        const start = std.time.milliTimestamp();

        // Simulate health check - in production would execute RETURN 1
        const latency = @as(u64, @intCast(std.time.milliTimestamp() - start));
        const status = ServiceStatus.fromLatency(latency + 1, self.config.health_check_interval_ms / 10);

        return HealthStatus.init(self.allocator, "memgraph", status, latency + 1);
    }

    /// Check Marquez health
    pub fn checkMarquez(self: *HealthChecker, url: []const u8) !HealthStatus {
        _ = url;
        const start = std.time.milliTimestamp();

        // Simulate health check - in production would call GET /api/v1/namespaces
        const latency = @as(u64, @intCast(std.time.milliTimestamp() - start));
        const status = ServiceStatus.fromLatency(latency + 1, self.config.health_check_interval_ms / 10);

        return HealthStatus.init(self.allocator, "marquez", status, latency + 1);
    }

    /// Check SAP HANA health
    pub fn checkHana(self: *HealthChecker, config: HanaPoolConfig) !HealthStatus {
        const start = std.time.milliTimestamp();

        // Simulate health check - in production would execute "SELECT 1 FROM DUMMY"
        // and verify connection to HANA Cloud endpoint
        _ = config;
        const latency = @as(u64, @intCast(std.time.milliTimestamp() - start));
        const status = ServiceStatus.fromLatency(latency + 1, self.config.health_check_interval_ms / 10);

        return HealthStatus.init(self.allocator, "hana", status, latency + 1);
    }

    /// Check all services
    pub fn checkAll(self: *HealthChecker) ![]HealthStatus {
        self.mutex.lock();
        if (self.check_in_progress) {
            self.mutex.unlock();
            return error.CheckInProgress;
        }
        self.check_in_progress = true;
        self.mutex.unlock();

        defer {
            self.mutex.lock();
            self.check_in_progress = false;
            self.mutex.unlock();
        }

        const service_count: usize = 4; // HANA + Qdrant + Memgraph + Marquez

        var results = try self.allocator.alloc(HealthStatus, service_count);
        errdefer self.allocator.free(results);

        results[0] = try self.checkHana(self.config.hana);
        results[1] = try self.checkQdrant(self.config.qdrant.url);
        results[2] = try self.checkMemgraph("placeholder");
        results[3] = try self.checkMarquez(self.config.marquez.url);

        // Store results
        self.mutex.lock();
        if (self.last_check_results) |old| {
            for (old) |*r| {
                r.deinit();
            }
            self.allocator.free(old);
        }
        // Clone results for storage
        self.last_check_results = try self.allocator.alloc(HealthStatus, service_count);
        for (results, 0..) |r, i| {
            self.last_check_results.?[i] = try HealthStatus.init(self.allocator, r.service, r.status, r.latency_ms);
        }
        self.mutex.unlock();

        return results;
    }

    /// Get last check results without running new checks
    pub fn getLastResults(self: *HealthChecker) ?[]HealthStatus {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.last_check_results;
    }
};