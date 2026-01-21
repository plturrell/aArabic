const std = @import("std");
const llama_model = @import("llama_model");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧪 Week 1 Day 5: Full Model Integration Test Suite\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run Llama model tests
    try llama_model.test_llama_model(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 5 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Llama Model (config, weights, forward pass)\n", .{});
    std.debug.print("   ✅ Multi-layer processing (2 layers tested)\n", .{});
    std.debug.print("   ✅ Token embedding & output projection\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 WEEK 1 COMPLETE! Full Zig inference engine ready!\n", .{});
    std.debug.print("\n", .{});
}
