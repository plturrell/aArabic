const std = @import("std");
const advanced_attention = @import("advanced_attention");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 19 TESTS: ADVANCED ATTENTION PATTERNS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    try advanced_attention.test_advanced_attention(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 19 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 Summary:\n", .{});
    std.debug.print("   ✅ Causal attention (autoregressive)\n", .{});
    std.debug.print("   ✅ Multi-query attention (4x KV cache reduction)\n", .{});
    std.debug.print("   ✅ Grouped-query attention (4x KV cache reduction)\n", .{});
    std.debug.print("\n🎊 Advanced attention patterns ready! Week 4 Day 19 complete!\n", .{});
}
