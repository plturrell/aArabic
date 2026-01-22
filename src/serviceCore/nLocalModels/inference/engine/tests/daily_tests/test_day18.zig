const std = @import("std");
const flash_attention = @import("flash_attention");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 18 TESTS: FLASH ATTENTION OPTIMIZATION\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    try flash_attention.test_flash_attention(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 18 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 Summary:\n", .{});
    std.debug.print("   ✅ Flash attention correctness\n", .{});
    std.debug.print("   ✅ Memory efficiency (>90%% savings)\n", .{});
    std.debug.print("   ✅ Block tiling\n", .{});
    std.debug.print("\n🎊 Flash Attention ready! Week 4 Day 18 complete!\n", .{});
}
