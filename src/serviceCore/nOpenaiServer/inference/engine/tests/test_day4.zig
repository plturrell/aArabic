const std = @import("std");
const attention = @import("attention");
const feed_forward = @import("feed_forward");
const transformer = @import("transformer");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧪 Week 1 Day 4: Transformer Layer Test Suite\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run attention tests
    try attention.test_attention(allocator);
    
    // Run feed-forward tests
    try feed_forward.test_feed_forward(allocator);
    
    // Run transformer tests
    try transformer.test_transformer(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 4 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Attention (RoPE, multi-head, KV cache)\n", .{});
    std.debug.print("   ✅ Feed-Forward (SwiGLU, 3-layer MLP)\n", .{});
    std.debug.print("   ✅ Transformer (Complete layer with residuals)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎯 Ready for Day 5: Full model inference!\n", .{});
    std.debug.print("\n", .{});
}
