const std = @import("std");
const advanced_sampler = @import("advanced_sampler");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  ADVANCED SAMPLER TESTS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    try advanced_sampler.test_advanced_sampler(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL ADVANCED SAMPLER TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 Sampling Strategies Available:\n", .{});
    std.debug.print("   • Temperature control (0.0 = greedy, 1.0+ = creative)\n", .{});
    std.debug.print("   • Top-K: Keep only K most probable tokens\n", .{});
    std.debug.print("   • Top-P (nucleus): Keep tokens with cumulative prob P\n", .{});
    std.debug.print("   • Repetition penalty: Discourage repeated tokens\n", .{});
    std.debug.print("   • Frequency/presence penalties: Advanced control\n", .{});
}
