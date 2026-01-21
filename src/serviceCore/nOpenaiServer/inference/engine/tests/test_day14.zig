const std = @import("std");
const thread_pool = @import("thread_pool");

/// Day 14 Tests: Multi-threading Basics
/// 
/// Tests:
/// 1. Basic task submission
/// 2. Parallel map
/// 3. Parallel reduce
/// 4. Performance comparison

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 14 TESTS: MULTI-THREADING BASICS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run thread pool tests
    try thread_pool.test_thread_pool(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 14 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Thread pool creation and shutdown\n", .{});
    std.debug.print("   ✅ Task submission and execution\n", .{});
    std.debug.print("   ✅ Parallel map operations\n", .{});
    std.debug.print("   ✅ Parallel reduce operations\n", .{});
    std.debug.print("   ✅ Performance speedup verified\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 Multi-threading ready! Week 3 Day 14 complete!\n", .{});
    std.debug.print("\n", .{});
}
