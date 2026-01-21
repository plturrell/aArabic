const std = @import("std");
const sampler = @import("sampler");

/// Day 11 Tests: Advanced Sampling Strategies
/// 
/// Tests:
/// 1. Greedy sampling
/// 2. Temperature sampling
/// 3. Top-k sampling
/// 4. Top-p (nucleus) sampling
/// 5. Softmax correctness
/// 6. Sampling diversity

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 11 TESTS: ADVANCED SAMPLING STRATEGIES\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run sampler tests
    try sampler.test_sampler(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 11 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Greedy sampling working\n", .{});
    std.debug.print("   ✅ Temperature sampling working\n", .{});
    std.debug.print("   ✅ Top-k sampling working\n", .{});
    std.debug.print("   ✅ Top-p sampling working\n", .{});
    std.debug.print("   ✅ Softmax verified\n", .{});
    std.debug.print("   ✅ Sampling diversity confirmed\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 Advanced sampling strategies ready! Week 3 Day 11 complete!\n", .{});
    std.debug.print("\n", .{});
}
