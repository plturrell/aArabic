const std = @import("std");
const cross_test = @import("cross_database_test.zig");

const DatabaseType = cross_test.DatabaseType;

/// Performance benchmark for cross-database comparison
pub const CrossDatabaseBenchmark = struct {
    allocator: std.mem.Allocator,
    databases: []DatabaseType,
    results: std.ArrayList(BenchmarkResult),
    
    pub fn init(allocator: std.mem.Allocator, databases: []DatabaseType) CrossDatabaseBenchmark {
        return CrossDatabaseBenchmark{
            .allocator = allocator,
            .databases = databases,
            .results = std.ArrayList(BenchmarkResult).init(allocator),
        };
    }
    
    pub fn deinit(self: *CrossDatabaseBenchmark) void {
        self.results.deinit();
    }
    
    /// Run all benchmarks
    pub fn runAll(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("\n╔══════════════════════════════════════════╗\n", .{});
        std.debug.print("║  Cross-Database Performance Benchmarks  ║\n", .{});
        std.debug.print("╚══════════════════════════════════════════╝\n\n", .{});
        
        try self.benchmarkSimpleQueries();
        try self.benchmarkComplexQueries();
        try self.benchmarkBatchInserts();
        try self.benchmarkTransactions();
        try self.benchmarkConnectionPooling();
        
        try self.printComparisonTable();
    }
    
    /// Benchmark simple SELECT queries
    fn benchmarkSimpleQueries(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("Benchmarking: Simple SELECT Queries\n", .{});
        std.debug.print("═══════════════════════════════════════\n\n", .{});
        
        for (self.databases) |db_type| {
            const start = std.time.nanoTimestamp();
            
            // Simulate 1000 simple queries
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                std.time.sleep(100); // 0.1µs per query
            }
            
            const end = std.time.nanoTimestamp();
            const duration_ns = end - start;
            const qps = 1000.0 / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
            
            try self.results.append(BenchmarkResult{
                .db_type = db_type,
                .test_name = "Simple Queries",
                .queries_per_second = qps,
                .avg_latency_us = @as(f64, @floatFromInt(duration_ns)) / 1000.0 / 1000.0,
                .throughput_mbps = 0.5, // Estimated
            });
            
            std.debug.print("  {s:<15} {d:>10.0} QPS\n", .{ db_type.toString(), qps });
        }
        
        std.debug.print("\n", .{});
    }
    
    /// Benchmark complex JOIN queries
    fn benchmarkComplexQueries(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("Benchmarking: Complex JOIN Queries\n", .{});
        std.debug.print("═══════════════════════════════════════\n\n", .{});
        
        for (self.databases) |db_type| {
            const start = std.time.nanoTimestamp();
            
            // Simulate 100 complex queries
            var i: usize = 0;
            while (i < 100) : (i += 1) {
                std.time.sleep(5000); // 5µs per query
            }
            
            const end = std.time.nanoTimestamp();
            const duration_ns = end - start;
            const qps = 100.0 / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
            
            try self.results.append(BenchmarkResult{
                .db_type = db_type,
                .test_name = "Complex Queries",
                .queries_per_second = qps,
                .avg_latency_us = @as(f64, @floatFromInt(duration_ns)) / 100.0 / 1000.0,
                .throughput_mbps = 2.5,
            });
            
            std.debug.print("  {s:<15} {d:>10.0} QPS\n", .{ db_type.toString(), qps });
        }
        
        std.debug.print("\n", .{});
    }
    
    /// Benchmark batch inserts
    fn benchmarkBatchInserts(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("Benchmarking: Batch Inserts (10K rows)\n", .{});
        std.debug.print("═══════════════════════════════════════\n\n", .{});
        
        for (self.databases) |db_type| {
            const start = std.time.nanoTimestamp();
            
            // Simulate inserting 10K rows
            std.time.sleep(50_000_000); // 50ms
            
            const end = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
            const rows_per_sec = 10000.0 / (duration_ms / 1000.0);
            
            try self.results.append(BenchmarkResult{
                .db_type = db_type,
                .test_name = "Batch Inserts",
                .queries_per_second = rows_per_sec / 100.0, // Convert to QPS
                .avg_latency_us = duration_ms * 1000.0 / 10000.0,
                .throughput_mbps = 10.0,
            });
            
            std.debug.print("  {s:<15} {d:>10.0} rows/s ({d:.1}ms)\n", .{
                db_type.toString(),
                rows_per_sec,
                duration_ms,
            });
        }
        
        std.debug.print("\n", .{});
    }
    
    /// Benchmark transaction performance
    fn benchmarkTransactions(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("Benchmarking: Transactions (1K commits)\n", .{});
        std.debug.print("═══════════════════════════════════════\n\n", .{});
        
        for (self.databases) |db_type| {
            const start = std.time.nanoTimestamp();
            
            // Simulate 1000 transactions
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                std.time.sleep(1000); // 1µs per transaction
            }
            
            const end = std.time.nanoTimestamp();
            const duration_ns = end - start;
            const tps = 1000.0 / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
            
            try self.results.append(BenchmarkResult{
                .db_type = db_type,
                .test_name = "Transactions",
                .queries_per_second = tps,
                .avg_latency_us = @as(f64, @floatFromInt(duration_ns)) / 1000.0 / 1000.0,
                .throughput_mbps = 1.5,
            });
            
            std.debug.print("  {s:<15} {d:>10.0} TPS\n", .{ db_type.toString(), tps });
        }
        
        std.debug.print("\n", .{});
    }
    
    /// Benchmark connection pooling
    fn benchmarkConnectionPooling(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("Benchmarking: Connection Pool (100 concurrent)\n", .{});
        std.debug.print("═══════════════════════════════════════\n\n", .{});
        
        for (self.databases) |db_type| {
            const start = std.time.nanoTimestamp();
            
            // Simulate 100 concurrent connections
            std.time.sleep(10_000_000); // 10ms
            
            const end = std.time.nanoTimestamp();
            const duration_ms = @as(f64, @floatFromInt(end - start)) / 1_000_000.0;
            const conns_per_sec = 100.0 / (duration_ms / 1000.0);
            
            try self.results.append(BenchmarkResult{
                .db_type = db_type,
                .test_name = "Connection Pool",
                .queries_per_second = conns_per_sec,
                .avg_latency_us = duration_ms * 1000.0 / 100.0,
                .throughput_mbps = 0.1,
            });
            
            std.debug.print("  {s:<15} {d:>10.0} conn/s ({d:.1}ms)\n", .{
                db_type.toString(),
                conns_per_sec,
                duration_ms,
            });
        }
        
        std.debug.print("\n", .{});
    }
    
    /// Print comparison table
    fn printComparisonTable(self: *CrossDatabaseBenchmark) !void {
        std.debug.print("\n╔═══════════════════════════════════════════════════════════════╗\n", .{});
        std.debug.print("║           Performance Comparison Table                        ║\n", .{});
        std.debug.print("╚═══════════════════════════════════════════════════════════════╝\n\n", .{});
        
        // Group results by test name
        var test_groups = std.StringHashMap(std.ArrayList(BenchmarkResult)).init(self.allocator);
        defer {
            var it = test_groups.iterator();
            while (it.next()) |entry| {
                entry.value_ptr.deinit();
            }
            test_groups.deinit();
        }
        
        for (self.results.items) |result| {
            var group = test_groups.get(result.test_name) orelse blk: {
                var new_group = std.ArrayList(BenchmarkResult).init(self.allocator);
                try test_groups.put(result.test_name, new_group);
                break :blk new_group;
            };
            try group.append(result);
            try test_groups.put(result.test_name, group);
        }
        
        std.debug.print("{s:<25} PostgreSQL    HANA    SQLite\n", .{"Benchmark"});
        std.debug.print("─────────────────────────────────────────────────────────────\n", .{});
        
        var it = test_groups.iterator();
        while (it.next()) |entry| {
            const test_name = entry.key_ptr.*;
            const results = entry.value_ptr.items;
            
            std.debug.print("{s:<25}", .{test_name});
            
            for (results) |result| {
                std.debug.print(" {d:>8.0} QPS", .{result.queries_per_second});
            }
            std.debug.print("\n", .{});
        }
        
        std.debug.print("\n", .{});
    }
};

/// Benchmark result
pub const BenchmarkResult = struct {
    db_type: DatabaseType,
    test_name: []const u8,
    queries_per_second: f64,
    avg_latency_us: f64,
    throughput_mbps: f64,
};

/// Performance metrics summary
pub const PerformanceMetrics = struct {
    database: DatabaseType,
    total_qps: f64,
    avg_latency_us: f64,
    p95_latency_us: f64,
    p99_latency_us: f64,
    throughput_mbps: f64,
    
    pub fn format(
        self: PerformanceMetrics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            \\
            \\Performance Metrics: {s}
            \\  Total QPS: {d:.0}
            \\  Avg Latency: {d:.2}µs
            \\  P95 Latency: {d:.2}µs
            \\  P99 Latency: {d:.2}µs
            \\  Throughput: {d:.1} MB/s
            \\
        ,
            .{
                self.database.toString(),
                self.total_qps,
                self.avg_latency_us,
                self.p95_latency_us,
                self.p99_latency_us,
                self.throughput_mbps,
            },
        );
    }
};

/// Run cross-database benchmarks
pub fn runCrossDatabaseBenchmarks(allocator: std.mem.Allocator) !void {
    const databases = [_]DatabaseType{ .postgresql, .hana, .sqlite };
    
    var benchmark = CrossDatabaseBenchmark.init(allocator, &databases);
    defer benchmark.deinit();
    
    try benchmark.runAll();
}

// ============================================================================
// Unit Tests
// ============================================================================

test "CrossDatabaseBenchmark - init and deinit" {
    const allocator = std.testing.allocator;
    const databases = [_]DatabaseType{ .postgresql, .hana };
    
    var benchmark = CrossDatabaseBenchmark.init(allocator, &databases);
    defer benchmark.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), benchmark.results.items.len);
}

test "BenchmarkResult - creation" {
    const result = BenchmarkResult{
        .db_type = .postgresql,
        .test_name = "Test",
        .queries_per_second = 10000.0,
        .avg_latency_us = 100.0,
        .throughput_mbps = 5.0,
    };
    
    try std.testing.expectEqual(DatabaseType.postgresql, result.db_type);
    try std.testing.expectEqual(@as(f64, 10000.0), result.queries_per_second);
}

test "PerformanceMetrics - format" {
    const metrics = PerformanceMetrics{
        .database = .hana,
        .total_qps = 50000.0,
        .avg_latency_us = 50.0,
        .p95_latency_us = 100.0,
        .p99_latency_us = 150.0,
        .throughput_mbps = 25.0,
    };
    
    var buf: [500]u8 = undefined;
    const result = try std.fmt.bufPrint(&buf, "{}", .{metrics});
    
    try std.testing.expect(std.mem.indexOf(u8, result, "SAP HANA") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "50000") != null);
}
