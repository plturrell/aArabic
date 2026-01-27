// Comprehensive benchmark suite for zig-libc
// Phase 1.1 Month 5: Performance Benchmarking vs musl

const std = @import("std");
const zig_libc = @import("zig-libc");
const builtin = @import("builtin");

const BenchmarkResult = struct {
    name: []const u8,
    operations: usize,
    duration_ns: u64,
    ops_per_sec: f64,
};

pub fn main() !void {
    const print = std.debug.print;
    
    print("═══════════════════════════════════════════════════════════════════════\n", .{});
    print("  zig-libc Benchmark Suite\n", .{});
    print("═══════════════════════════════════════════════════════════════════════\n", .{});
    print("Phase 1.1: Foundation (Month 5)\n", .{});
    print("Version: {d}.{d}.{d} ({s})\n", .{
        zig_libc.version.major,
        zig_libc.version.minor,
        zig_libc.version.patch,
        zig_libc.version.phase,
    });
    print("Platform: {s} {s}\n", .{ @tagName(builtin.os.tag), @tagName(builtin.cpu.arch) });
    print("Build Mode: {s}\n\n", .{@tagName(builtin.mode)});

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var results = std.ArrayList(BenchmarkResult).initCapacity(allocator, 15) catch unreachable;
    defer results.deinit(allocator);

    // String benchmarks
    print("String Operations:\n", .{});
    try results.append(allocator, try benchmarkStrlen());
    try results.append(allocator, try benchmarkStrcmp());
    try results.append(allocator, try benchmarkStrchr());
    try results.append(allocator, try benchmarkStrstr());
    
    // Character benchmarks
    print("\nCharacter Classification:\n", .{});
    try results.append(allocator, try benchmarkIsalpha());
    try results.append(allocator, try benchmarkIsdigit());
    try results.append(allocator, try benchmarkToupper());
    try results.append(allocator, try benchmarkTolower());
    
    // Memory benchmarks
    print("\nMemory Operations:\n", .{});
    try results.append(allocator, try benchmarkMemcmp());
    try results.append(allocator, try benchmarkMemchr());

    // Summary
    print("\n═══════════════════════════════════════════════════════════════════════\n", .{});
    print("  Benchmark Summary\n", .{});
    print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
    print("{s:<30} {s:>15} {s:>15}\n", .{ "Function", "Operations", "Ops/Second" });
    print("{s:-<30} {s:->15} {s:->15}\n", .{ "", "", "" });
    
    for (results.items) |result| {
        print("{s:<30} {d:>15} {d:>15.2}\n", .{
            result.name,
            result.operations,
            result.ops_per_sec,
        });
    }
    
    print("\n40 functions implemented, {d} benchmarks run\n", .{results.items.len});
    print("All benchmarks completed successfully! ✅\n", .{});
}

// Helper function to run a benchmark
fn runBenchmark(
    comptime name: []const u8,
    comptime iterations: usize,
    comptime func: fn () void,
) !BenchmarkResult {
    const print = std.debug.print;
    
    var timer = try std.time.Timer.start();
    const start = timer.lap();
    
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        func();
    }
    
    const duration = timer.read() - start;
    const ops_per_sec = @as(f64, @floatFromInt(iterations)) / (@as(f64, @floatFromInt(duration)) / 1_000_000_000.0);
    
    print("  {s:<28} {d:>10} ops in {d:>8.2}ms ({d:>12.0} ops/sec)\n", .{
        name,
        iterations,
        @as(f64, @floatFromInt(duration)) / 1_000_000.0,
        ops_per_sec,
    });
    
    return BenchmarkResult{
        .name = name,
        .operations = iterations,
        .duration_ns = duration,
        .ops_per_sec = ops_per_sec,
    };
}

// String benchmarks
fn benchmarkStrlen() !BenchmarkResult {
    const test_str = "Hello, World! This is a test string for benchmarking.";
    return runBenchmark("strlen", 1_000_000, struct {
        fn bench() void {
            _ = zig_libc.string.strlen(test_str);
        }
    }.bench);
}

fn benchmarkStrcmp() !BenchmarkResult {
    const str1 = "Hello, World!";
    const str2 = "Hello, World!";
    return runBenchmark("strcmp", 1_000_000, struct {
        fn bench() void {
            _ = zig_libc.string.strcmp(str1, str2);
        }
    }.bench);
}

fn benchmarkStrchr() !BenchmarkResult {
    const test_str = "Hello, World! This is a test string for benchmarking.";
    return runBenchmark("strchr", 1_000_000, struct {
        fn bench() void {
            _ = zig_libc.string.strchr(test_str, 'W');
        }
    }.bench);
}

fn benchmarkStrstr() !BenchmarkResult {
    const haystack = "Hello, World! This is a test string for benchmarking.";
    const needle = "test";
    return runBenchmark("strstr", 500_000, struct {
        fn bench() void {
            _ = zig_libc.string.strstr(haystack, needle);
        }
    }.bench);
}

// Character benchmarks
fn benchmarkIsalpha() !BenchmarkResult {
    return runBenchmark("isalpha", 10_000_000, struct {
        fn bench() void {
            _ = zig_libc.ctype.isalpha('A');
            _ = zig_libc.ctype.isalpha('z');
            _ = zig_libc.ctype.isalpha('5');
        }
    }.bench);
}

fn benchmarkIsdigit() !BenchmarkResult {
    return runBenchmark("isdigit", 10_000_000, struct {
        fn bench() void {
            _ = zig_libc.ctype.isdigit('0');
            _ = zig_libc.ctype.isdigit('9');
            _ = zig_libc.ctype.isdigit('A');
        }
    }.bench);
}

fn benchmarkToupper() !BenchmarkResult {
    return runBenchmark("toupper", 10_000_000, struct {
        fn bench() void {
            _ = zig_libc.ctype.toupper('a');
            _ = zig_libc.ctype.toupper('z');
            _ = zig_libc.ctype.toupper('A');
        }
    }.bench);
}

fn benchmarkTolower() !BenchmarkResult {
    return runBenchmark("tolower", 10_000_000, struct {
        fn bench() void {
            _ = zig_libc.ctype.tolower('A');
            _ = zig_libc.ctype.tolower('Z');
            _ = zig_libc.ctype.tolower('a');
        }
    }.bench);
}

// Memory benchmarks
fn benchmarkMemcmp() !BenchmarkResult {
    var buf1: [1024]u8 = undefined;
    var buf2: [1024]u8 = undefined;
    @memset(&buf1, 0xAA);
    @memset(&buf2, 0xAA);
    
    return runBenchmark("memcmp (1KB)", 1_000_000, struct {
        fn bench() void {
            var b1: [1024]u8 = undefined;
            var b2: [1024]u8 = undefined;
            @memset(&b1, 0xAA);
            @memset(&b2, 0xAA);
            _ = zig_libc.memory.memcmp(&b1, &b2, 1024);
        }
    }.bench);
}

fn benchmarkMemchr() !BenchmarkResult {
    var buf: [1024]u8 = undefined;
    @memset(&buf, 0xAA);
    buf[512] = 0xFF;
    
    return runBenchmark("memchr (1KB)", 1_000_000, struct {
        fn bench() void {
            var b: [1024]u8 = undefined;
            @memset(&b, 0xAA);
            b[512] = 0xFF;
            _ = zig_libc.memory.memchr(&b, 0xFF, 1024);
        }
    }.bench);
}
