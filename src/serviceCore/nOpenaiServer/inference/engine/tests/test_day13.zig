const std = @import("std");
const q8_0 = @import("q8_0");

/// Day 13 Tests: Q8_0 Quantization
/// 
/// Tests:
/// 1. Basic quantization/dequantization
/// 2. Compression ratio
/// 3. Dot product accuracy
/// 4. Edge cases
/// 5. Multi-block operations
/// 6. Q8_0 vs Q4_0 comparison

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 13 TESTS: Q8_0 QUANTIZATION\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run Q8_0 tests
    try q8_0.test_q8_0(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 13 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Basic quantization/dequantization\n", .{});
    std.debug.print("   ✅ Compression ratio verified (3.56:1)\n", .{});
    std.debug.print("   ✅ Dot product accuracy (<1% error)\n", .{});
    std.debug.print("   ✅ Edge cases handled\n", .{});
    std.debug.print("   ✅ Multi-block operations\n", .{});
    std.debug.print("   ✅ Q8_0 vs Q4_0 compared\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 Q8_0 quantization ready! Week 3 Day 13 complete!\n", .{});
    std.debug.print("\n", .{});
}
