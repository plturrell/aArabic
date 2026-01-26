const std = @import("std");
const http = @import("../../lib/std/http.zig");
const testing = std.testing;

/// Load testing framework for HTTP and database operations
/// Measures throughput, latency, and resource usage under load

pub const LoadTestConfig = struct {
    num_requests: usize = 10000,
    concurrency: usize = 100,
    duration_seconds: ?u64 = null,
    warmup_requests: usize = 1000,
    report_interval_ms: u64 = 1000,
};

pub const LoadTestResult = struct {
    total_requests: usize,
    successful: usize,
    failed: usize,
    duration_ms: u64,
    requests_per_second: f64,
    avg_latency_ms: f64,
    min_latency_ms: f64,
    max_latency_ms: f64,
    p50_latency_ms: f64,
    p95_latency_ms: f64,
    p99_latency_ms: f64,

    pub fn print(self: *const LoadTestResult) void {
        std.debug.print("\n=== Load Test Results ===\n", .{});
        std.debug.print("Total Requests:    {d}\n", .{self.total_requests});
        std.debug.print("Successful:        {d}\n", .{self.successful});
        std.debug.print("Failed:            {d}\n", .{self.failed});
        std.debug.print("Duration:          {d}ms\n", .{self.duration_ms});
        std.debug.print("Throughput:        {d:.2} req/s\n", .{self.requests_per_second});
        std.debug.print("Avg Latency:       {d:.2}ms\n", .{self.avg_latency_ms});
        std.debug.print("Min Latency:       {d:.2}ms\n", .{self.min_latency_ms});
        std.debug.print("Max Latency:       {d:.2}ms\n", .{self.max_latency_ms});
        std.debug.print("P50 Latency:       {d:.2}ms\n", .{self.p50_latency_ms});
        std.debug.print("P95 Latency:       {d:.2}ms\n", .{self.p95_latency_ms});
        std.debug.print("P99 Latency:       {d:.2}ms\n", .{self.p99_latency_ms});
        std.debug.print("========================\n\n", .{});
    }
};

pub const LoadTester = struct {
    allocator: Allocator,
    config: LoadTestConfig,
    latencies: std.ArrayList(u64),
    mutex: std.Thread.Mutex,

    const Allocator = std.mem.Allocator;

    pub fn init(allocator: Allocator, config: LoadTestConfig) LoadTester {
        return .{
            .allocator = allocator,
            .config = config,
            .latencies = std.ArrayList(u64).init(allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *LoadTester) void {
        self.latencies.deinit();
    }

    /// Execute load test with given operation
    pub fn run(self: *LoadTester, context: anytype, operation: *const fn (@TypeOf(context)) anyerror!void) !LoadTestResult {
        // Warmup
        std.debug.print("Warming up with {d} requests...\n", .{self.config.warmup_requests});
        for (0..self.config.warmup_requests) |_| {
            operation(context) catch {};
        }

        // Clear latencies from warmup
        self.latencies.clearRetainingCapacity();

        std.debug.print("Starting load test: {d} requests, {d} concurrent\n", .{
            self.config.num_requests,
            self.config.concurrency,
        });

        const start_time = std.time.milliTimestamp();
        var successful: usize = 0;
        var failed: usize = 0;

        // Run with concurrency
        var threads = try self.allocator.alloc(std.Thread, self.config.concurrency);
        defer self.allocator.free(threads);

        const requests_per_thread = self.config.num_requests / self.config.concurrency;

        for (threads, 0..) |*thread, i| {
            thread.* = try std.Thread.spawn(.{}, workerThread, .{
                self,
                context,
                operation,
                requests_per_thread,
                i,
            });
        }

        // Wait for all threads
        for (threads) |thread| {
            thread.join();
        }

        const end_time = std.time.milliTimestamp();
        const duration_ms: u64 = @intCast(end_time - start_time);

        // Calculate statistics
        self.mutex.lock();
        defer self.mutex.unlock();

        std.mem.sort(u64, self.latencies.items, {}, comptime std.sort.asc(u64));

        const total = self.latencies.items.len;
        successful = total;
        failed = self.config.num_requests - successful;

        var sum: u64 = 0;
        var min: u64 = std.math.maxInt(u64);
        var max: u64 = 0;

        for (self.latencies.items) |lat| {
            sum += lat;
            if (lat < min) min = lat;
            if (lat > max) max = lat;
        }

        const avg = @as(f64, @floatFromInt(sum)) / @as(f64, @floatFromInt(total));
        const p50_idx = total * 50 / 100;
        const p95_idx = total * 95 / 100;
        const p99_idx = total * 99 / 100;

        return LoadTestResult{
            .total_requests = self.config.num_requests,
            .successful = successful,
            .failed = failed,
            .duration_ms = duration_ms,
            .requests_per_second = @as(f64, @floatFromInt(successful)) / (@as(f64, @floatFromInt(duration_ms)) / 1000.0),
            .avg_latency_ms = avg / 1_000_000.0,
            .min_latency_ms = @as(f64, @floatFromInt(min)) / 1_000_000.0,
            .max_latency_ms = @as(f64, @floatFromInt(max)) / 1_000_000.0,
            .p50_latency_ms = @as(f64, @floatFromInt(self.latencies.items[p50_idx])) / 1_000_000.0,
            .p95_latency_ms = @as(f64, @floatFromInt(self.latencies.items[p95_idx])) / 1_000_000.0,
            .p99_latency_ms = @as(f64, @floatFromInt(self.latencies.items[p99_idx])) / 1_000_000.0,
        };
    }

    fn workerThread(
        self: *LoadTester,
        context: anytype,
        operation: *const fn (@TypeOf(context)) anyerror!void,
        num_requests: usize,
        worker_id: usize,
    ) void {
        _ = worker_id;
        for (0..num_requests) |_| {
            const start = std.time.nanoTimestamp();
            operation(context) catch {
                continue;
            };
            const end = std.time.nanoTimestamp();
            const latency: u64 = @intCast(end - start);

            self.mutex.lock();
            defer self.mutex.unlock();
            self.latencies.append(latency) catch {};
        }
    }
};

test "Load Test - HTTP endpoint throughput" {
    const allocator = testing.allocator;

    const Context = struct {
        counter: std.atomic.Value(usize),
    };

    var ctx = Context{
        .counter = std.atomic.Value(usize).init(0),
    };

    const operation = struct {
        fn op(c: *Context) !void {
            _ = c.counter.fetchAdd(1, .monotonic);
            // Simulate HTTP handler work
            std.time.sleep(1 * std.time.ns_per_ms);
        }
    }.op;

    var tester = LoadTester.init(allocator, .{
        .num_requests = 1000,
        .concurrency = 10,
        .warmup_requests = 100,
    });
    defer tester.deinit();

    const result = try tester.run(&ctx, operation);
    result.print();

    try testing.expect(result.successful > 0);
    try testing.expect(result.requests_per_second > 0);
}

test "Load Test - Database query performance" {
    const allocator = testing.allocator;

    const Context = struct {
        query_count: std.atomic.Value(usize),
    };

    var ctx = Context{
        .query_count = std.atomic.Value(usize).init(0),
    };

    const operation = struct {
        fn op(c: *Context) !void {
            _ = c.query_count.fetchAdd(1, .monotonic);
            // Simulate database query
            std.time.sleep(2 * std.time.ns_per_ms);
        }
    }.op;

    var tester = LoadTester.init(allocator, .{
        .num_requests = 500,
        .concurrency = 5,
        .warmup_requests = 50,
    });
    defer tester.deinit();

    const result = try tester.run(&ctx, operation);
    result.print();

    try testing.expect(result.successful > 0);
}

test "Load Test - Batch operation throughput" {
    const allocator = testing.allocator;

    var batch = hana.BatchOperations.init(allocator, .{
        .batch_size = 100,
    });
    defer batch.deinit();

    // Add many operations
    for (0..1000) |i| {
        const params = [_]hana.Parameter{.{ .int = @intCast(i) }};
        try batch.add("INSERT INTO test VALUES (?)", &params);
    }

    try testing.expect(batch.operations.items.len == 1000);
}