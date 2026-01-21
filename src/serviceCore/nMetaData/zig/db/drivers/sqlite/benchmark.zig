const std = @import("std");
const connection_mod = @import("connection.zig");
const query_mod = @import("query.zig");
const pool_mod = @import("pool.zig");
const protocol = @import("protocol.zig");
const client_types = @import("../../client.zig");

const SqliteConnection = connection_mod.SqliteConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;
const QueryExecutor = query_mod.QueryExecutor;
const SqliteConnectionPool = pool_mod.SqliteConnectionPool;
const SqlitePoolConfig = pool_mod.SqlitePoolConfig;
const Value = client_types.Value;

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    num_queries: usize = 1000,
    num_threads: usize = 1, // SQLite typically single-writer
    pool_size: usize = 3,   // Small pool for SQLite
    database_path: []const u8 = ":memory:",
    
    pub fn validate(self: BenchmarkConfig) !void {
        if (self.num_queries == 0) return error.InvalidBenchmarkConfig;
        if (self.num_threads == 0) return error.InvalidBenchmarkConfig;
        if (self.pool_size == 0) return error.InvalidBenchmarkConfig;
    }
};

/// Benchmark result
pub const BenchmarkResult = struct {
    total_queries: usize,
    duration_ms: i64,
    queries_per_second: f64,
    avg_latency_ms: f64,
    min_latency_ms: i64,
    max_latency_ms: i64,
    p50_latency_ms: i64,
    p95_latency_ms: i64,
    p99_latency_ms: i64,
    
    pub fn format(self: BenchmarkResult, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            \\
            \\Benchmark Results:
            \\  Total Queries: {d}
            \\  Duration: {d}ms
            \\  QPS: {d:.2}
            \\  Avg Latency: {d:.2}ms
            \\  Min Latency: {d}ms
            \\  Max Latency: {d}ms
            \\  P50 Latency: {d}ms
            \\  P95 Latency: {d}ms
            \\  P99 Latency: {d}ms
            \\
        ,
            .{
                self.total_queries,
                self.duration_ms,
                self.queries_per_second,
                self.avg_latency_ms,
                self.min_latency_ms,
                self.max_latency_ms,
                self.p50_latency_ms,
                self.p95_latency_ms,
                self.p99_latency_ms,
            },
        );
    }
};

/// Benchmark runner
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    latencies: std.ArrayList(i64),
    
    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !BenchmarkRunner {
        try config.validate();
        
        return BenchmarkRunner{
            .allocator = allocator,
            .config = config,
            .latencies = std.ArrayList(i64).init(allocator),
        };
    }
    
    pub fn deinit(self: *BenchmarkRunner) void {
        self.latencies.deinit();
    }
    
    /// Run simple query benchmark
    pub fn benchmarkSimpleQueries(self: *BenchmarkRunner) !BenchmarkResult {
        const conn_config = ConnectionConfig{
            .path = self.config.database_path,
            .mode = .memory,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        const start_time = std.time.milliTimestamp();
        
        // Run SELECT 1 queries
        for (0..self.config.num_queries) |_| {
            const query_start = std.time.nanoTimestamp();
            
            const result = try executor.executeQuery("SELECT 1", &[_]Value{});
            result.deinit();
            
            const query_end = std.time.nanoTimestamp();
            const latency_ns = query_end - query_start;
            try self.latencies.append(@divFloor(latency_ns, 1_000_000)); // Convert to ms
        }
        
        const end_time = std.time.milliTimestamp();
        const duration = end_time - start_time;
        
        return try self.calculateResult(duration);
    }
    
    /// Run prepared statement benchmark
    pub fn benchmarkPreparedStatements(self: *BenchmarkRunner) !BenchmarkResult {
        const conn_config = ConnectionConfig{
            .path = self.config.database_path,
            .mode = .memory,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        const start_time = std.time.milliTimestamp();
        
        // Run parameterized queries
        for (0..self.config.num_queries) |i| {
            const query_start = std.time.nanoTimestamp();
            
            const params = [_]Value{
                Value{ .integer = @intCast(i) },
                Value{ .integer = @intCast(i + 1) },
            };
            const result = try executor.executeQuery("SELECT ?1 + ?2", &params);
            result.deinit();
            
            const query_end = std.time.nanoTimestamp();
            const latency_ns = query_end - query_start;
            try self.latencies.append(@divFloor(latency_ns, 1_000_000));
        }
        
        const end_time = std.time.milliTimestamp();
        const duration = end_time - start_time;
        
        return try self.calculateResult(duration);
    }
    
    /// Run insert benchmark
    pub fn benchmarkInserts(self: *BenchmarkRunner) !BenchmarkResult {
        const conn_config = ConnectionConfig{
            .path = self.config.database_path,
            .mode = .memory,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Create test table
        const create_result = try executor.executeQuery(
            "CREATE TABLE bench_test (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        create_result.deinit();
        
        const start_time = std.time.milliTimestamp();
        
        // Run INSERT queries
        for (0..self.config.num_queries) |i| {
            const query_start = std.time.nanoTimestamp();
            
            const params = [_]Value{
                Value{ .text = "test value" },
            };
            const result = try executor.executeQuery(
                "INSERT INTO bench_test (value) VALUES (?1)",
                &params,
            );
            result.deinit();
            
            _ = i;
            const query_end = std.time.nanoTimestamp();
            const latency_ns = query_end - query_start;
            try self.latencies.append(@divFloor(latency_ns, 1_000_000));
        }
        
        const end_time = std.time.milliTimestamp();
        const duration = end_time - start_time;
        
        return try self.calculateResult(duration);
    }
    
    /// Run transaction benchmark
    pub fn benchmarkTransactions(self: *BenchmarkRunner) !BenchmarkResult {
        const conn_config = ConnectionConfig{
            .path = self.config.database_path,
            .mode = .memory,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Create test table
        const create_result = try executor.executeQuery(
            "CREATE TABLE bench_tx (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        create_result.deinit();
        
        const start_time = std.time.milliTimestamp();
        
        // Run transactions
        for (0..self.config.num_queries) |_| {
            const query_start = std.time.nanoTimestamp();
            
            // BEGIN -> INSERT -> COMMIT cycle
            const begin_result = try executor.executeQuery("BEGIN", &[_]Value{});
            begin_result.deinit();
            
            const insert_result = try executor.executeQuery(
                "INSERT INTO bench_tx (value) VALUES ('test')",
                &[_]Value{},
            );
            insert_result.deinit();
            
            const commit_result = try executor.executeQuery("COMMIT", &[_]Value{});
            commit_result.deinit();
            
            const query_end = std.time.nanoTimestamp();
            const latency_ns = query_end - query_start;
            try self.latencies.append(@divFloor(latency_ns, 1_000_000));
        }
        
        const end_time = std.time.milliTimestamp();
        const duration = end_time - start_time;
        
        return try self.calculateResult(duration);
    }
    
    /// Run connection pool benchmark
    pub fn benchmarkConnectionPool(self: *BenchmarkRunner) !BenchmarkResult {
        const conn_config = ConnectionConfig{
            .path = self.config.database_path,
            .mode = .memory,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
        
        const pool_config = SqlitePoolConfig{
            .connection_config = conn_config,
            .min_size = 1,
            .max_size = self.config.pool_size,
            .acquire_timeout_ms = 5000,
            .idle_timeout_ms = 300000,
        };
        
        var pool = try SqliteConnectionPool.init(self.allocator, pool_config);
        defer pool.deinit();
        
        const start_time = std.time.milliTimestamp();
        
        // Run pool acquire/release cycles with queries
        for (0..self.config.num_queries) |_| {
            const query_start = std.time.nanoTimestamp();
            
            var conn = try pool.acquire();
            
            var executor = QueryExecutor.init(self.allocator, conn);
            defer executor.deinit();
            
            const result = try executor.executeQuery("SELECT 1", &[_]Value{});
            result.deinit();
            
            try pool.release(conn);
            
            const query_end = std.time.nanoTimestamp();
            const latency_ns = query_end - query_start;
            try self.latencies.append(@divFloor(latency_ns, 1_000_000));
        }
        
        const end_time = std.time.milliTimestamp();
        const duration = end_time - start_time;
        
        return try self.calculateResult(duration);
    }
    
    /// Calculate benchmark result from latencies
    fn calculateResult(self: *BenchmarkRunner, duration_ms: i64) !BenchmarkResult {
        if (self.latencies.items.len == 0) {
            return error.NoLatencyData;
        }
        
        // Sort latencies for percentile calculation
        std.mem.sort(i64, self.latencies.items, {}, comptime std.sort.asc(i64));
        
        // Calculate statistics
        var total_latency: i64 = 0;
        var min_latency: i64 = std.math.maxInt(i64);
        var max_latency: i64 = 0;
        
        for (self.latencies.items) |lat| {
            total_latency += lat;
            if (lat < min_latency) min_latency = lat;
            if (lat > max_latency) max_latency = lat;
        }
        
        const avg_latency = @as(f64, @floatFromInt(total_latency)) / 
                           @as(f64, @floatFromInt(self.latencies.items.len));
        
        const qps = if (duration_ms > 0)
            (@as(f64, @floatFromInt(self.latencies.items.len)) / 
             @as(f64, @floatFromInt(duration_ms))) * 1000.0
        else
            0.0;
        
        // Calculate percentiles
        const p50_idx = self.latencies.items.len / 2;
        const p95_idx = (self.latencies.items.len * 95) / 100;
        const p99_idx = (self.latencies.items.len * 99) / 100;
        
        return BenchmarkResult{
            .total_queries = self.latencies.items.len,
            .duration_ms = duration_ms,
            .queries_per_second = qps,
            .avg_latency_ms = avg_latency,
            .min_latency_ms = min_latency,
            .max_latency_ms = max_latency,
            .p50_latency_ms = self.latencies.items[p50_idx],
            .p95_latency_ms = self.latencies.items[p95_idx],
            .p99_latency_ms = self.latencies.items[p99_idx],
        };
    }
};

/// Run all benchmarks
pub fn runAllBenchmarks(allocator: std.mem.Allocator, config: BenchmarkConfig) !void {
    std.debug.print("\n=== SQLite Benchmark Suite ===\n\n", .{});
    
    // Simple queries
    {
        std.debug.print("Running: Simple Queries...\n", .{});
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();
        
        const result = try runner.benchmarkSimpleQueries();
        std.debug.print("{}\n", .{result});
    }
    
    // Prepared statements
    {
        std.debug.print("Running: Prepared Statements...\n", .{});
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();
        
        const result = try runner.benchmarkPreparedStatements();
        std.debug.print("{}\n", .{result});
    }
    
    // Inserts
    {
        std.debug.print("Running: Inserts...\n", .{});
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();
        
        const result = try runner.benchmarkInserts();
        std.debug.print("{}\n", .{result});
    }
    
    // Transactions
    {
        std.debug.print("Running: Transactions...\n", .{});
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();
        
        const result = try runner.benchmarkTransactions();
        std.debug.print("{}\n", .{result});
    }
    
    // Connection pool
    {
        std.debug.print("Running: Connection Pool...\n", .{});
        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();
        
        const result = try runner.benchmarkConnectionPool();
        std.debug.print("{}\n", .{result});
    }
    
    std.debug.print("=== Benchmarks Complete ===\n\n", .{});
}

// ============================================================================
// Unit Tests
// ============================================================================

test "BenchmarkConfig - validation" {
    const valid = BenchmarkConfig{
        .num_queries = 100,
        .num_threads = 1,
        .pool_size = 3,
    };
    try valid.validate();
    
    const invalid = BenchmarkConfig{
        .num_queries = 0,
        .num_threads = 1,
        .pool_size = 3,
    };
    try std.testing.expectError(error.InvalidBenchmarkConfig, invalid.validate());
}

test "BenchmarkRunner - init and deinit" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .num_queries = 10,
        .num_threads = 1,
        .pool_size = 2,
    };
    
    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), runner.latencies.items.len);
}

test "BenchmarkRunner - calculateResult" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{};
    
    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();
    
    // Add some sample latencies
    try runner.latencies.append(10);
    try runner.latencies.append(20);
    try runner.latencies.append(30);
    try runner.latencies.append(40);
    try runner.latencies.append(50);
    
    const result = try runner.calculateResult(100);
    
    try std.testing.expectEqual(@as(usize, 5), result.total_queries);
    try std.testing.expectEqual(@as(i64, 100), result.duration_ms);
    try std.testing.expectEqual(@as(i64, 10), result.min_latency_ms);
    try std.testing.expectEqual(@as(i64, 50), result.max_latency_ms);
    try std.testing.expectEqual(@as(f64, 30.0), result.avg_latency_ms);
}
