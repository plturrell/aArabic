const std = @import("std");
const framework = @import("framework");

/// Fuzz test for benchmark framework to discover edge cases
/// Run with: zig build-exe fuzz_test.zig && ./fuzz_test

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n" ++
        "╔══════════════════════════════════════════════════════════════╗\n" ++
        "║         Fuzz Testing Suite - Edge Case Detection            ║\n" ++
        "╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    var prng = std.Random.DefaultPrng.init(@intCast(std.time.timestamp()));
    const random = prng.random();

    var passed: usize = 0;
    var failed: usize = 0;

    // Test 1: Zero iterations
    std.debug.print("Test 1: Zero iterations... ", .{});
    if (testZeroIterations(allocator)) |_| {
        std.debug.print("✅ Passed\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        failed += 1;
    }

    // Test 2: Large iteration count
    std.debug.print("Test 2: Large iteration count (10000)... ", .{});
    if (testLargeIterations(allocator)) |_| {
        std.debug.print("✅ Passed\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        failed += 1;
    }

    // Test 3: Random iteration patterns
    std.debug.print("Test 3: Random iteration patterns... ", .{});
    if (testRandomIterations(allocator, random)) |_| {
        std.debug.print("✅ Passed\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        failed += 1;
    }

    // Test 4: Memory stress test
    std.debug.print("Test 4: Memory stress test... ", .{});
    if (testMemoryStress(allocator)) |_| {
        std.debug.print("✅ Passed\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        failed += 1;
    }

    // Test 5: Edge case values
    std.debug.print("Test 5: Edge case values... ", .{});
    if (testEdgeCases(allocator)) |_| {
        std.debug.print("✅ Passed\n", .{});
        passed += 1;
    } else |err| {
        std.debug.print("❌ Failed: {}\n", .{err});
        failed += 1;
    }

    // Summary
    std.debug.print("\n" ++
        "╔══════════════════════════════════════════════════════════════╗\n" ++
        "║         Fuzz Test Results                                    ║\n" ++
        "╚══════════════════════════════════════════════════════════════╝\n\n", .{});

    std.debug.print("Total Tests:  {}\n", .{passed + failed});
    std.debug.print("Passed:       {} ✅\n", .{passed});
    std.debug.print("Failed:       {} {s}\n\n", .{ failed, if (failed == 0) "✅" else "❌" });

    if (failed > 0) {
        std.debug.print("⚠️  Some edge cases failed. Review and fix.\n", .{});
        return error.FuzzTestsFailed;
    } else {
        std.debug.print("✅ All fuzz tests passed! Code is robust.\n", .{});
    }
}

fn testZeroIterations(allocator: std.mem.Allocator) !void {
    // Edge case: What happens with 0 iterations?
    // Should handle gracefully without crashes
    const result = try framework.benchmark(
        allocator,
        "Zero iterations test",
        0,
        0,
        struct {
            pub fn run(_: @This()) void {
                // No-op
            }
        }{},
    );
    
    // Verify sensible results
    if (result.iterations != 0) return error.UnexpectedIterationCount;
}

fn testLargeIterations(allocator: std.mem.Allocator) !void {
    // Test with large iteration count to ensure no overflow
    const result = try framework.benchmark(
        allocator,
        "Large iterations test",
        10_000,
        100,
        struct {
            pub fn run(_: @This()) void {
                var sum: u64 = 0;
                var i: usize = 0;
                while (i < 100) : (i += 1) {
                    sum +%= i;
                }
                std.mem.doNotOptimizeAway(&sum);
            }
        }{},
    );
    
    // Verify results are reasonable
    if (result.iterations != 10_000) return error.IterationMismatch;
    if (result.mean_ns == 0) return error.InvalidTiming;
}

fn testRandomIterations(allocator: std.mem.Allocator, random: std.Random) !void {
    // Test with random iteration counts
    const iterations = random.intRangeAtMost(usize, 10, 100);
    
    const result = try framework.benchmark(
        allocator,
        "Random iterations test",
        iterations,
        2,
        struct {
            pub fn run(_: @This()) void {
                const x = 42;
                std.mem.doNotOptimizeAway(&x);
            }
        }{},
    );
    
    if (result.iterations != iterations) return error.IterationMismatch;
}

fn testMemoryStress(allocator: std.mem.Allocator) !void {
    // Stress test memory allocation in benchmark
    const result = try framework.benchmark(
        allocator,
        "Memory stress test",
        50,
        5,
        struct {
            alloc: std.mem.Allocator,
            pub fn run(self: @This()) void {
                // Allocate and free in each iteration
                const data = self.alloc.alloc(u64, 10_000) catch unreachable;
                defer self.alloc.free(data);
                
                for (data) |*item| {
                    item.* = 42;
                }
                std.mem.doNotOptimizeAway(data.ptr);
            }
        }{ .alloc = allocator },
    );
    
    // Verify no memory issues
    if (result.iterations != 50) return error.MemoryStressFailed;
}

fn testEdgeCases(allocator: std.mem.Allocator) !void {
    // Test with edge case values
    const result = try framework.benchmark(
        allocator,
        "Edge cases test",
        1,
        0,
        struct {
            pub fn run(_: @This()) void {
                // Test max values
                const max_u64: u64 = std.math.maxInt(u64);
                const max_minus_one = max_u64 -% 1;
                
                // Test zero values
                const zero: u64 = 0;
                
                // Test boundary conditions
                var arr = [_]u64{ zero, max_minus_one, max_u64 };
                std.mem.doNotOptimizeAway(&arr);
            }
        }{},
    );
    
    if (result.iterations != 1) return error.EdgeCaseFailed;
}