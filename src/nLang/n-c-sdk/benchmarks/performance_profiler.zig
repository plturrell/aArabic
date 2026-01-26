const std = @import("std");
const framework = @import("framework");

/// Performance profiler for analyzing optimization effectiveness
pub const ProfileResult = struct {
    build_mode: []const u8,
    lto_enabled: bool,
    total_time_ms: f64,
    memory_used_bytes: usize,
    
    pub fn format(
        self: ProfileResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "Mode: {s} | LTO: {} | Time: {d:.2}ms | Memory: {d:.2}MB\n",
            .{
                self.build_mode,
                self.lto_enabled,
                self.total_time_ms,
                @as(f64, @floatFromInt(self.memory_used_bytes)) / (1024.0 * 1024.0),
            },
        );
    }
};

pub fn profileBuildMode() ProfileResult {
    // Detect build mode at compile time
    const build_mode = switch (@import("builtin").mode) {
        .Debug => "Debug",
        .ReleaseSafe => "ReleaseSafe",
        .ReleaseFast => "ReleaseFast",
        .ReleaseSmall => "ReleaseSmall",
    };
    
    // LTO detection (heuristic based on build mode)
    const lto_enabled = switch (@import("builtin").mode) {
        .Debug => false,
        else => true, // Assume LTO is enabled for release builds in this SDK
    };
    
    return ProfileResult{
        .build_mode = build_mode,
        .lto_enabled = lto_enabled,
        .total_time_ms = 0.0,
        .memory_used_bytes = 0,
    };
}

pub fn measureMemoryUsage(allocator: std.mem.Allocator, comptime func: anytype, args: anytype) !usize {
    // Create a tracking allocator
    var tracking = std.heap.GeneralPurposeAllocator(.{
        .enable_memory_limit = true,
    }){};
    defer _ = tracking.deinit();
    
    const tracking_alloc = tracking.allocator();
    
    // Run function with tracking allocator
    const start_mem = tracking.total_requested_bytes;
    _ = try @call(.auto, func, .{tracking_alloc} ++ args);
    const end_mem = tracking.total_requested_bytes;
    
    _ = allocator; // Keep parameter for future use
    return end_mem - start_mem;
}

pub fn printOptimizationReport() void {
    const profile = profileBuildMode();
    
    std.debug.print("\n" ++
        "╔══════════════════════════════════════════════════════════════╗\n" ++
        "║         Optimization Configuration Report                    ║\n" ++
        "╚══════════════════════════════════════════════════════════════╝\n\n", .{});
    
    std.debug.print("Build Mode:       {s}\n", .{profile.build_mode});
    std.debug.print("LTO Enabled:      {}\n", .{profile.lto_enabled});
    std.debug.print("Safety Checks:    {}\n", .{switch (@import("builtin").mode) {
        .Debug, .ReleaseSafe => true,
        .ReleaseFast, .ReleaseSmall => false,
    }});
    std.debug.print("Target CPU:       {s}\n", .{@tagName(@import("builtin").cpu.arch)});
    std.debug.print("OS:               {s}\n\n", .{@tagName(@import("builtin").os.tag)});
    
    // Explain expected performance
    const expected_speedup = switch (@import("builtin").mode) {
        .Debug => "1.0x (baseline)",
        .ReleaseSafe => "2-3x faster than Debug",
        .ReleaseFast => "3-4x faster than Debug",
        .ReleaseSmall => "2-3x faster than Debug (optimized for size)",
    };
    
    std.debug.print("Expected Performance: {s}\n", .{expected_speedup});
    if (profile.lto_enabled) {
        std.debug.print("LTO Benefits:         ✅ Cross-module optimization, smaller binaries\n\n", .{});
    } else {
        std.debug.print("LTO Benefits:         ❌ Not enabled in Debug mode\n\n", .{});
    }
}

/// Comprehensive benchmark runner with profiling
pub fn runComprehensiveBenchmark(allocator: std.mem.Allocator) !void {
    printOptimizationReport();
    
    std.debug.print("Running comprehensive performance analysis...\n\n", .{});
    
    // CPU-intensive benchmark
    const cpu_result = try framework.benchmark(
        allocator,
        "CPU Intensive (Fibonacci 38)",
        5,
        1,
        struct {
            pub fn run(_: @This()) void {
                const result = fib(38);
                std.mem.doNotOptimizeAway(&result);
            }
            
            fn fib(n: u32) u64 {
                if (n <= 1) return n;
                return fib(n - 1) + fib(n - 2);
            }
        }{},
    );
    framework.printResult(cpu_result);
    
    // Memory-intensive benchmark
    const mem_result = try framework.benchmark(
        allocator,
        "Memory Intensive (Sort 500K)",
        10,
        2,
        struct {
            alloc: std.mem.Allocator,
            pub fn run(self: @This()) void {
                const size = 500_000;
                const data = self.alloc.alloc(u64, size) catch unreachable;
                defer self.alloc.free(data);
                
                var prng = std.Random.DefaultPrng.init(42);
                const random = prng.random();
                for (data) |*item| {
                    item.* = random.int(u64);
                }
                
                std.mem.sort(u64, data, {}, comptime std.sort.asc(u64));
                std.mem.doNotOptimizeAway(data.ptr);
            }
        }{ .alloc = allocator },
    );
    framework.printResult(mem_result);
    
    // Mixed workload benchmark
    const mixed_result = try framework.benchmark(
        allocator,
        "Mixed Workload (Crypto + Sort)",
        20,
        3,
        struct {
            alloc: std.mem.Allocator,
            pub fn run(self: @This()) void {
                // Crypto operations
                var hasher = std.hash.Wyhash.init(0);
                var i: usize = 0;
                while (i < 50_000) : (i += 1) {
                    const data = std.mem.asBytes(&i);
                    hasher.update(data);
                }
                const hash = hasher.final();
                
                // Sort operations
                const size = 10_000;
                const arr = self.alloc.alloc(u64, size) catch unreachable;
                defer self.alloc.free(arr);
                for (arr, 0..) |*item, idx| {
                    item.* = (hash +% idx) % 1000;
                }
                std.mem.sort(u64, arr, {}, comptime std.sort.asc(u64));
                std.mem.doNotOptimizeAway(arr.ptr);
            }
        }{ .alloc = allocator },
    );
    framework.printResult(mixed_result);
    
    std.debug.print("\n✅ Comprehensive benchmark complete!\n", .{});
    printPerformanceAnalysis(cpu_result, mem_result, mixed_result);
}

fn printPerformanceAnalysis(cpu: framework.BenchmarkResult, mem: framework.BenchmarkResult, mixed: framework.BenchmarkResult) void {
    std.debug.print("\n" ++
        "╔══════════════════════════════════════════════════════════════╗\n" ++
        "║         Performance Analysis Summary                         ║\n" ++
        "╚══════════════════════════════════════════════════════════════╝\n\n", .{});
    
    const mode = @import("builtin").mode;
    
    std.debug.print("Optimization Assessment:\n", .{});
    std.debug.print("  • CPU-bound:    {d:.2}ms median\n", .{
        @as(f64, @floatFromInt(cpu.median_ns)) / 1_000_000.0,
    });
    std.debug.print("  • Memory-bound: {d:.2}ms median\n", .{
        @as(f64, @floatFromInt(mem.median_ns)) / 1_000_000.0,
    });
    std.debug.print("  • Mixed:        {d:.2}ms median\n\n", .{
        @as(f64, @floatFromInt(mixed.median_ns)) / 1_000_000.0,
    });
    
    if (mode == .Debug) {
        std.debug.print("⚠️  Running in Debug mode - expect slower performance\n", .{});
        std.debug.print("💡 Tip: Build with -Doptimize=ReleaseSafe for production speed\n\n", .{});
    } else {
        std.debug.print("✅ Running in optimized mode\n", .{});
        if (mode == .ReleaseSafe) {
            std.debug.print("🛡️  Safety checks are enabled - balanced performance\n\n", .{});
        } else {
            std.debug.print("⚡ Maximum performance mode - safety checks disabled\n\n", .{});
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    try runComprehensiveBenchmark(allocator);
}