const std = @import("std");
const tokenizer = @import("tokenizer");
const kv_cache = @import("kv_cache");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧪 Week 1 Day 3: Tokenizer & KV Cache Test Suite\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Run tokenizer tests
    try tokenizer.test_tokenizer(allocator);
    
    // Run KV cache tests
    try kv_cache.test_kv_cache(allocator);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 3 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Tokenizer (encode/decode, sampling, filtering)\n", .{});
    std.debug.print("   ✅ KV Cache (store/retrieve, multi-position, statistics)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎯 Ready for Day 4: Transformer layers!\n", .{});
    std.debug.print("\n", .{});
}
