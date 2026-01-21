const std = @import("std");
const matrix_ops = @import("matrix_ops");
const common = @import("quantization_common");
const q4_0 = @import("q4_0");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧪 Week 1 Day 2: Matrix Operations & Quantization Test Suite\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    // Test matrix operations
    try matrix_ops.test_operations(allocator);

    // Test quantization commons
    try common.test_conversions();

    // Test Q4_0 quantization
    try q4_0.test_q4_0(allocator);

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 2 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 Summary:\n", .{});
    std.debug.print("   ✅ Matrix operations (SIMD-optimized)\n", .{});
    std.debug.print("   ✅ Quantization commons (f16, packing)\n", .{});
    std.debug.print("   ✅ Q4_0 quantization (encode/decode)\n", .{});
    std.debug.print("\n🎯 Ready for Day 3: Tensor loading & model integration\n\n", .{});
}
