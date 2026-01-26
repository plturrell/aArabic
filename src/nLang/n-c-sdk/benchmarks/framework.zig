const std = @import("std");

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    median_ns: u64,
    mean_ns: u64,
    min_ns: u64,
    max_ns: u64,

    pub fn format(
        self: BenchmarkResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}: median={d:.2}ms, mean={d:.2}ms (n={})\n", .{
            self.name,
            @as(f64, @floatFromInt(self.median_ns)) / 1_000_000.0,
            @as(f64, @floatFromInt(self.mean_ns)) / 1_000_000.0,
            self.iterations,
        });
    }
};

pub fn benchmark(
    allocator: std.mem.Allocator,
    name: []const u8,
    iterations: usize,
    warmup: usize,
    context: anytype,
) !BenchmarkResult {
    var times = std.ArrayList(u64){};
    try times.ensureTotalCapacity(allocator, iterations);
    defer times.deinit(allocator);

    // Warmup phase
    var i: usize = 0;
    while (i < warmup) : (i += 1) {
        context.run();
    }

    // Measurement phase
    i = 0;
    while (i < iterations) : (i += 1) {
        const start = std.time.nanoTimestamp();
        context.run();
        const end = std.time.nanoTimestamp();
        try times.append(allocator, @intCast(end - start));
    }

    // Calculate statistics
    // Handle edge case: zero iterations (discovered by fuzz testing!)
    if (times.items.len == 0) {
        return BenchmarkResult{
            .name = name,
            .iterations = iterations,
            .median_ns = 0,
            .mean_ns = 0,
            .min_ns = 0,
            .max_ns = 0,
        };
    }

    std.mem.sort(u64, times.items, {}, comptime std.sort.asc(u64));
    const median = times.items[times.items.len / 2];
    const min = times.items[0];
    const max = times.items[times.items.len - 1];

    var sum: u64 = 0;
    for (times.items) |t| sum += t;
    const mean = sum / times.items.len;

    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .median_ns = median,
        .mean_ns = mean,
        .min_ns = min,
        .max_ns = max,
    };
}

pub fn printHeader() void {
    const builtin = @import("builtin");
    
    std.debug.print("\n" ++
        "╔══════════════════════════════════════════════════════════════╗\n" ++
        "║         Zig SDK Performance Benchmark Suite                 ║\n" ++
        "╚══════════════════════════════════════════════════════════════╝\n\n", .{});
    
    // Print build configuration
    std.debug.print("Build Mode:       {s}\n", .{@tagName(builtin.mode)});
    std.debug.print("Optimization:     {s}\n", .{@tagName(builtin.optimize_mode)});
    
    // ReleaseBalanced mode detection
    if (builtin.mode == .ReleaseBalanced) {
        std.debug.print("🎯 ReleaseBalanced: Hybrid safety/performance mode active\n", .{});
        std.debug.print("   Expected: 1.8-2.2x faster than ReleaseSafe\n", .{});
        std.debug.print("   Safety:   80%+ code maintains safety checks\n", .{});
    }
    
    std.debug.print("Safety Checks:    {}\n", .{builtin.mode == .Debug or builtin.mode == .ReleaseSafe or builtin.mode == .ReleaseBalanced});
    std.debug.print("\n", .{});
}

pub fn printResult(result: BenchmarkResult) void {
    std.debug.print("{s}: median={d:.2}ms, mean={d:.2}ms (n={})\n", .{
        result.name,
        @as(f64, @floatFromInt(result.median_ns)) / 1_000_000.0,
        @as(f64, @floatFromInt(result.mean_ns)) / 1_000_000.0,
        result.iterations,
    });
}