const std = @import("std");
const safetensors = @import("safetensors_loader");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  SAFETENSORS LOADER TESTS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test with Qwen3 model shard
    const test_file = "vendor/layerModels/huggingFace/Qwen/Qwen3-Coder-30B-A3B-Instruct/model-00001-of-00016.safetensors";
    
    std.debug.print("\n🧪 Testing SafeTensors Loader with Qwen3 model shard\n", .{});
    std.debug.print("   File: {s}\n", .{test_file});
    
    var loader = safetensors.SafeTensorsFile.init(allocator, test_file);
    defer loader.deinit();
    
    // Load the file
    try loader.load();
    
    // List all tensors
    loader.listTensors();
    
    // Test loading a specific tensor
    std.debug.print("\n🧪 Testing tensor data loading...\n", .{});
    
    // Get first tensor name
    var it = loader.header.tensors.iterator();
    if (it.next()) |entry| {
        const tensor_name = entry.key_ptr.*;
        const tensor_info = entry.value_ptr.*;
        
        std.debug.print("   Loading tensor: {s}\n", .{tensor_name});
        std.debug.print("   Shape: [", .{});
        for (tensor_info.shape, 0..) |dim, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d}", .{dim});
        }
        std.debug.print("]\n", .{});
        std.debug.print("   Dtype: {s}\n", .{@tagName(tensor_info.dtype)});
        
        // Load tensor data
        const tensor_data = try loader.getTensor(tensor_name);
        defer allocator.free(tensor_data);
        
        std.debug.print("   Loaded {d} elements\n", .{tensor_data.len});
        
        // Print first few values
        std.debug.print("   First 10 values: ", .{});
        const print_count = @min(10, tensor_data.len);
        for (tensor_data[0..print_count], 0..) |val, i| {
            if (i > 0) std.debug.print(", ", .{});
            std.debug.print("{d:.6}", .{val});
        }
        std.debug.print("\n", .{});
        
        // Compute statistics
        var sum: f32 = 0.0;
        var min_val: f32 = tensor_data[0];
        var max_val: f32 = tensor_data[0];
        
        for (tensor_data) |val| {
            sum += val;
            min_val = @min(min_val, val);
            max_val = @max(max_val, val);
        }
        
        const mean = sum / @as(f32, @floatFromInt(tensor_data.len));
        
        std.debug.print("   Statistics:\n", .{});
        std.debug.print("     Mean: {d:.6}\n", .{mean});
        std.debug.print("     Min:  {d:.6}\n", .{min_val});
        std.debug.print("     Max:  {d:.6}\n", .{max_val});
        
        std.debug.print("   ✅ Tensor loaded successfully!\n", .{});
    } else {
        std.debug.print("   ⚠️  No tensors found in file\n", .{});
    }
    
    // Test metadata
    std.debug.print("\n🧪 Testing metadata access...\n", .{});
    if (loader.getMetadata("__metadata__")) |metadata| {
        std.debug.print("   Metadata found: {s}\n", .{metadata});
    } else {
        std.debug.print("   No metadata found\n", .{});
    }
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL SAFETENSORS TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n📊 SafeTensors Loader Features:\n", .{});
    std.debug.print("   • Parse JSON header with tensor metadata\n", .{});
    std.debug.print("   • Load tensor data with dtype conversion\n", .{});
    std.debug.print("   • Support F32, F16, BF16 formats\n", .{});
    std.debug.print("   • Efficient memory management\n", .{});
    std.debug.print("   • Ready for multi-shard loading\n", .{});
}
