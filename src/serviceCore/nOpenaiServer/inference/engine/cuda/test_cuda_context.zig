// Unit tests for CUDA Context Management
// Tests device initialization, property querying, and memory management

const std = @import("std");
const testing = std.testing;
const CudaContext = @import("cuda_context.zig").CudaContext;
const cuda_context = @import("cuda_context.zig");

test "cuda_context: initialization with device 0" {
    std.debug.print("\n=== Testing CUDA Context Initialization ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            std.debug.print("   This test requires a CUDA-capable GPU\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    // Validate context state
    try testing.expect(ctx.initialized);
    try testing.expect(ctx.device_id == 0);
    try testing.expect(ctx.properties.name.len > 0);
    
    std.debug.print("\n✅ Context initialized successfully\n", .{});
    std.debug.print("   Device: {s}\n", .{ctx.properties.name});
    std.debug.print("   Compute: {d}.{d}\n", .{
        ctx.properties.compute_capability.major,
        ctx.properties.compute_capability.minor,
    });
}

test "cuda_context: device properties validation" {
    std.debug.print("\n=== Testing Device Properties ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const props = ctx.properties;
    
    // Validate basic properties
    try testing.expect(props.name.len > 0);
    try testing.expect(props.total_memory_gb > 0);
    try testing.expect(props.multiprocessor_count > 0);
    try testing.expect(props.max_threads_per_block > 0);
    try testing.expect(props.warp_size == 32); // NVIDIA GPUs have warp size 32
    
    // Validate compute capability
    try testing.expect(props.compute_capability.major >= 3); // At least Kepler
    try testing.expect(props.compute_capability.minor >= 0);
    try testing.expect(props.compute_capability.minor < 10);
    
    std.debug.print("\n✅ All property validations passed\n", .{});
    std.debug.print("   Architecture: {s}\n", .{props.getArchitectureName()});
    std.debug.print("   SMs: {d}\n", .{props.multiprocessor_count});
    std.debug.print("   Warp Size: {d}\n", .{props.warp_size});
}

test "cuda_context: T4 detection" {
    std.debug.print("\n=== Testing T4 GPU Detection ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const is_t4 = ctx.properties.isT4();
    
    if (is_t4) {
        std.debug.print("\n✅ Tesla T4 detected!\n", .{});
        
        // Validate T4 properties
        try testing.expect(ctx.properties.compute_capability.major == 7);
        try testing.expect(ctx.properties.compute_capability.minor == 5);
        try testing.expect(ctx.properties.hasTensorCores());
        
        // T4 has ~16GB memory
        const mem_gb = @as(u32, @intFromFloat(ctx.properties.total_memory_gb));
        try testing.expect(mem_gb >= 15);
        try testing.expect(mem_gb <= 17);
        
        // Check T4-specific recommendations
        try testing.expect(ctx.properties.getRecommendedBatchSize() == 8);
        try testing.expect(ctx.properties.getRecommendedKVCacheTokens() == 2048);
        
        std.debug.print("   Batch size: {d}\n", .{ctx.properties.getRecommendedBatchSize()});
        std.debug.print("   KV tokens: {d}\n", .{ctx.properties.getRecommendedKVCacheTokens()});
    } else {
        std.debug.print("\nℹ️  Not a T4 GPU: {s}\n", .{ctx.properties.name});
        std.debug.print("   Compute: {d}.{d}\n", .{
            ctx.properties.compute_capability.major,
            ctx.properties.compute_capability.minor,
        });
    }
}

test "cuda_context: Tensor Core detection" {
    std.debug.print("\n=== Testing Tensor Core Detection ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const has_tensor_cores = ctx.properties.hasTensorCores();
    
    std.debug.print("   GPU: {s}\n", .{ctx.properties.name});
    std.debug.print("   Compute: {d}.{d}\n", .{
        ctx.properties.compute_capability.major,
        ctx.properties.compute_capability.minor,
    });
    std.debug.print("   Tensor Cores: {s}\n", .{if (has_tensor_cores) "✅ Yes" else "❌ No"});
    
    // Tensor Cores should be present on Volta (7.0) and newer
    if (ctx.properties.compute_capability.major >= 7) {
        try testing.expect(has_tensor_cores);
    }
}

test "cuda_context: memory info retrieval" {
    std.debug.print("\n=== Testing Memory Info Retrieval ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const mem = try ctx.getMemoryInfo();
    
    // Validate memory info
    try testing.expect(mem.total_mb > 0);
    try testing.expect(mem.free_mb <= mem.total_mb);
    try testing.expect(mem.used_mb == mem.total_mb - mem.free_mb);
    try testing.expect(mem.utilization_percent >= 0.0);
    try testing.expect(mem.utilization_percent <= 100.0);
    
    std.debug.print("\n📊 Memory Status:\n", .{});
    std.debug.print("   Total: {d} MB ({d:.2} GB)\n", .{
        mem.total_mb,
        @as(f32, @floatFromInt(mem.total_mb)) / 1024.0,
    });
    std.debug.print("   Used: {d} MB ({d:.1}%)\n", .{
        mem.used_mb,
        mem.utilization_percent,
    });
    std.debug.print("   Free: {d} MB ({d:.2} GB)\n", .{
        mem.free_mb,
        @as(f32, @floatFromInt(mem.free_mb)) / 1024.0,
    });
    
    std.debug.print("\n✅ Memory info retrieved successfully\n", .{});
}

test "cuda_context: memory status printing" {
    std.debug.print("\n=== Testing Memory Status Display ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    try ctx.printMemoryStatus();
}

test "cuda_context: device synchronization" {
    std.debug.print("\n=== Testing Device Synchronization ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    // Synchronize should succeed (no operations pending, but should not error)
    try ctx.synchronize();
    
    std.debug.print("✅ Device synchronized successfully\n", .{});
}

test "cuda_context: convenience function - initCUDA" {
    std.debug.print("\n=== Testing Convenience Function: initCUDA ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = cuda_context.initCUDA(allocator) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    try testing.expect(ctx.device_id == 0);
    try testing.expect(ctx.initialized);
    
    std.debug.print("✅ initCUDA() initialized device 0\n", .{});
}

test "cuda_context: convenience function - selectBestDevice" {
    std.debug.print("\n=== Testing Convenience Function: selectBestDevice ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = cuda_context.selectBestDevice(allocator) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    try testing.expect(ctx.initialized);
    try testing.expect(ctx.device_id >= 0);
    
    std.debug.print("✅ Selected best device: {d}\n", .{ctx.device_id});
}

test "cuda_context: convenience function - listDevices" {
    std.debug.print("\n=== Testing Convenience Function: listDevices ===\n", .{});
    
    const allocator = testing.allocator;
    
    cuda_context.listDevices(allocator) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
}

test "cuda_context: recommended configurations" {
    std.debug.print("\n=== Testing Recommended Configurations ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    const batch_size = ctx.properties.getRecommendedBatchSize();
    const kv_tokens = ctx.properties.getRecommendedKVCacheTokens();
    
    // Validate recommendations are reasonable
    try testing.expect(batch_size >= 1);
    try testing.expect(batch_size <= 32);
    try testing.expect(kv_tokens >= 1024);
    try testing.expect(kv_tokens <= 8192);
    
    std.debug.print("\n📋 Recommended Configuration:\n", .{});
    std.debug.print("   GPU: {s}\n", .{ctx.properties.name});
    std.debug.print("   Memory: {d:.1} GB\n", .{ctx.properties.total_memory_gb});
    std.debug.print("   Batch Size: {d}\n", .{batch_size});
    std.debug.print("   KV Cache Tokens: {d}\n", .{kv_tokens});
    std.debug.print("   Use Tensor Cores: {s}\n", .{
        if (ctx.properties.hasTensorCores()) "Yes" else "No"
    });
    
    std.debug.print("\n✅ Configuration recommendations valid\n", .{});
}

test "cuda_context: error handling - invalid device ID" {
    std.debug.print("\n=== Testing Error Handling: Invalid Device ID ===\n", .{});
    
    const allocator = testing.allocator;
    
    // Try to initialize with a very high device ID
    const result = CudaContext.init(allocator, 999);
    
    if (result) |ctx| {
        ctx.deinit();
        std.debug.print("❌ Should have failed with invalid device ID\n", .{});
        return error.TestFailed;
    } else |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        
        try testing.expect(err == error.InvalidDeviceId or err == error.CudaError);
        std.debug.print("✅ Correctly rejected invalid device ID\n", .{});
    }
}

test "cuda_context: multiple initialization and cleanup" {
    std.debug.print("\n=== Testing Multiple Init/Deinit Cycles ===\n", .{});
    
    const allocator = testing.allocator;
    
    var i: u32 = 0;
    while (i < 3) : (i += 1) {
        const ctx = CudaContext.init(allocator, 0) catch |err| {
            if (err == error.NoGPUFound) {
                std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
                return;
            }
            return err;
        };
        defer ctx.deinit();
        
        try testing.expect(ctx.initialized);
        
        std.debug.print("   Cycle {d}: ✅\n", .{i + 1});
    }
    
    std.debug.print("\n✅ Multiple init/deinit cycles successful\n", .{});
}

test "cuda_context: memory consistency" {
    std.debug.print("\n=== Testing Memory Consistency ===\n", .{});
    
    const allocator = testing.allocator;
    
    const ctx = CudaContext.init(allocator, 0) catch |err| {
        if (err == error.NoGPUFound) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer ctx.deinit();
    
    // Get memory info multiple times
    const mem1 = try ctx.getMemoryInfo();
    const mem2 = try ctx.getMemoryInfo();
    
    // Total memory should be consistent
    try testing.expect(mem1.total_mb == mem2.total_mb);
    
    // Used memory should be approximately the same (allow small variations)
    const used_diff = if (mem1.used_mb > mem2.used_mb)
        mem1.used_mb - mem2.used_mb
    else
        mem2.used_mb - mem1.used_mb;
    
    // Allow up to 10MB variation
    try testing.expect(used_diff <= 10);
    
    std.debug.print("✅ Memory info is consistent across calls\n", .{});
    std.debug.print("   Used diff: {d} MB\n", .{used_diff});
}
