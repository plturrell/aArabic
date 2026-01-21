const std = @import("std");
const bridge = @import("hf_to_llama_bridge");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  HF → LLAMA BRIDGE END-TO-END TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const model_path = "/Users/user/Documents/arabic_folder/vendor/layerModels/huggingFace/Qwen/Qwen2.5-0.5B-Instruct";
    
    std.debug.print("\n🎯 Testing with real model:\n", .{});
    std.debug.print("   Path: {s}\n", .{model_path});
    std.debug.print("   Size: 942MB\n", .{});
    std.debug.print("   Tensors: 290\n", .{});
    std.debug.print("   Layers: 24\n", .{});
    
    // Run the test
    try bridge.test_hf_to_llama_bridge(allocator, model_path);
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ BRIDGE TEST COMPLETE - READY FOR TEXT GENERATION!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
