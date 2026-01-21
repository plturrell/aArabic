// Unit tests for nvidia-smi wrapper
// Tests GPU detection, parsing, and configuration recommendations

const std = @import("std");
const testing = std.testing;
const nvidia_smi = @import("nvidia_smi.zig");

test "nvidia_smi: check availability" {
    std.debug.print("\n=== Testing NVIDIA GPU Availability ===\n", .{});
    
    const has_gpu = nvidia_smi.hasNvidiaGPU();
    std.debug.print("NVIDIA GPU available: {}\n", .{has_gpu});
    
    if (!has_gpu) {
        std.debug.print("⚠️  No NVIDIA GPU found - nvidia-smi may not be installed\n", .{});
        std.debug.print("   This is expected on non-GPU systems\n", .{});
    }
}

test "nvidia_smi: list GPUs (quick)" {
    std.debug.print("\n=== Testing Quick GPU List ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpu_list = nvidia_smi.listGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed or err == error.FileNotFound) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpu_list) |gpu| allocator.free(gpu);
        allocator.free(gpu_list);
    }
    
    std.debug.print("Found {d} GPU(s):\n", .{gpu_list.len});
    for (gpu_list) |gpu| {
        std.debug.print("  • {s}\n", .{gpu});
    }
    
    if (gpu_list.len > 0) {
        try testing.expect(gpu_list[0].len > 0);
    }
}

test "nvidia_smi: detect GPUs with full info" {
    std.debug.print("\n=== Testing Full GPU Detection ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            std.debug.print("   To test with GPU, ensure:\n", .{});
            std.debug.print("   1. NVIDIA driver is installed\n", .{});
            std.debug.print("   2. nvidia-smi is in PATH\n", .{});
            std.debug.print("   3. Running on a machine with NVIDIA GPU\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    std.debug.print("Detected {d} GPU(s)\n", .{gpus.len});
    
    if (gpus.len > 0) {
        // Validate structure
        for (gpus) |gpu| {
            nvidia_smi.printGPUInfo(gpu);
            
            // Basic validation
            try testing.expect(gpu.name.len > 0);
            try testing.expect(gpu.uuid.len > 0);
            try testing.expect(gpu.memory_total_mb > 0);
            
            // Test T4 detection
            if (gpu.isT4()) {
                std.debug.print("\n✅ T4 GPU detected correctly!\n", .{});
                try testing.expect(gpu.compute_capability.major == 7);
                try testing.expect(gpu.compute_capability.minor == 5);
                try testing.expect(gpu.hasTensorCores());
            }
            
            // Test recommended config
            const config = nvidia_smi.getRecommendedConfig(gpu);
            std.debug.print("\n   Recommended Configuration:\n", .{});
            std.debug.print("     Max batch size: {d}\n", .{config.max_batch_size});
            std.debug.print("     KV cache tokens: {d}\n", .{config.kv_cache_tokens});
            std.debug.print("     Use FP16: {}\n", .{config.use_fp16});
            std.debug.print("     Use Tensor Cores: {}\n", .{config.use_tensor_cores});
            
            // Validate config makes sense
            try testing.expect(config.max_batch_size > 0);
            try testing.expect(config.kv_cache_tokens >= 1024);
            
            // T4 should always recommend FP16 and Tensor Cores
            if (gpu.isT4()) {
                try testing.expect(config.use_fp16);
                try testing.expect(config.use_tensor_cores);
                try testing.expect(config.kv_cache_tokens == 2048);
                try testing.expect(config.max_batch_size == 8);
            }
        }
    } else {
        std.debug.print("ℹ️  No GPUs detected (expected on non-GPU systems)\n", .{});
    }
}

test "nvidia_smi: memory calculations" {
    std.debug.print("\n=== Testing Memory Calculations ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    for (gpus) |gpu| {
        std.debug.print("\nGPU [{d}] Memory Analysis:\n", .{gpu.index});
        std.debug.print("   Total: {d} MB ({d:.1} GB)\n", .{
            gpu.memory_total_mb,
            @as(f32, @floatFromInt(gpu.memory_total_mb)) / 1024.0,
        });
        std.debug.print("   Used: {d} MB ({d:.1}%)\n", .{
            gpu.memory_used_mb,
            @as(f32, @floatFromInt(gpu.memory_used_mb)) / @as(f32, @floatFromInt(gpu.memory_total_mb)) * 100.0,
        });
        std.debug.print("   Free: {d} MB ({d:.1} GB)\n", .{
            gpu.memory_free_mb,
            @as(f32, @floatFromInt(gpu.memory_free_mb)) / 1024.0,
        });
        
        // Validate memory consistency
        const calculated_free = gpu.memory_total_mb -| gpu.memory_used_mb;
        const memory_diff = if (calculated_free > gpu.memory_free_mb)
            calculated_free - gpu.memory_free_mb
        else
            gpu.memory_free_mb - calculated_free;
        
        // Allow small difference due to rounding
        if (memory_diff > 100) {
            std.debug.print("⚠️  Memory accounting inconsistent (diff: {d} MB)\n", .{memory_diff});
        } else {
            std.debug.print("✅ Memory accounting consistent\n", .{});
        }
    }
}

test "nvidia_smi: utilization and temperature" {
    std.debug.print("\n=== Testing GPU Utilization & Temperature ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    for (gpus) |gpu| {
        std.debug.print("\nGPU [{d}] Runtime Metrics:\n", .{gpu.index});
        std.debug.print("   GPU Utilization: {d}%\n", .{gpu.utilization_gpu});
        std.debug.print("   Memory Utilization: {d}%\n", .{gpu.utilization_memory});
        std.debug.print("   Temperature: {d}°C", .{gpu.temperature_c});
        
        // Temperature warnings
        if (gpu.temperature_c >= 80) {
            std.debug.print(" ⚠️  HIGH!", .{});
        } else if (gpu.temperature_c >= 70) {
            std.debug.print(" ⚠️  Warm", .{});
        } else {
            std.debug.print(" ✅ Normal", .{});
        }
        std.debug.print("\n", .{});
        
        std.debug.print("   Power: {d}W / {d}W ({d:.1}%)\n", .{
            gpu.power_draw_w,
            gpu.power_limit_w,
            if (gpu.power_limit_w > 0)
                @as(f32, @floatFromInt(gpu.power_draw_w)) / @as(f32, @floatFromInt(gpu.power_limit_w)) * 100.0
            else
                0.0,
        });
        
        // Validate ranges
        try testing.expect(gpu.utilization_gpu <= 100);
        try testing.expect(gpu.utilization_memory <= 100);
        try testing.expect(gpu.temperature_c < 200); // Reasonable upper bound
    }
}

test "nvidia_smi: T4 specific detection" {
    std.debug.print("\n=== Testing T4 GPU Specific Detection ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    var found_t4 = false;
    for (gpus) |gpu| {
        if (gpu.isT4()) {
            found_t4 = true;
            std.debug.print("\n✅ T4 GPU Found!\n", .{});
            std.debug.print("   Name: {s}\n", .{gpu.name});
            std.debug.print("   Compute: {d}.{d} (Turing)\n", .{
                gpu.compute_capability.major,
                gpu.compute_capability.minor,
            });
            std.debug.print("   Memory: {d} GB\n", .{gpu.memory_total_mb / 1024});
            std.debug.print("   Tensor Cores: {}\n", .{gpu.hasTensorCores()});
            
            // T4 should have these properties
            try testing.expect(gpu.compute_capability.major == 7);
            try testing.expect(gpu.compute_capability.minor == 5);
            try testing.expect(gpu.hasTensorCores());
            try testing.expect(gpu.memory_total_mb >= 15000); // ~16GB
            try testing.expect(gpu.memory_total_mb <= 17000);
            
            const config = nvidia_smi.getRecommendedConfig(gpu);
            std.debug.print("\n   T4 Optimized Config:\n", .{});
            std.debug.print("     Batch size: {d}\n", .{config.max_batch_size});
            std.debug.print("     KV tokens: {d}\n", .{config.kv_cache_tokens});
            std.debug.print("     FP16: {}\n", .{config.use_fp16});
            std.debug.print("     Tensor Cores: {}\n", .{config.use_tensor_cores});
        }
    }
    
    if (!found_t4 and gpus.len > 0) {
        std.debug.print("\nℹ️  No T4 GPU found, but detected:\n", .{});
        for (gpus) |gpu| {
            std.debug.print("   • {s} (Compute {d}.{d})\n", .{
                gpu.name,
                gpu.compute_capability.major,
                gpu.compute_capability.minor,
            });
        }
    } else if (gpus.len == 0) {
        std.debug.print("\nℹ️  No GPUs detected on this system\n", .{});
    }
}

test "nvidia_smi: driver and CUDA version" {
    std.debug.print("\n=== Testing Driver & CUDA Version Detection ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    if (gpus.len > 0) {
        const gpu = gpus[0];
        
        std.debug.print("Driver Version: {s}\n", .{gpu.driver_version});
        std.debug.print("CUDA Version: {s}\n", .{gpu.cuda_version});
        
        if (gpu.driver_version.len > 0) {
            std.debug.print("✅ Driver version detected\n", .{});
        }
        
        if (gpu.cuda_version.len > 0) {
            std.debug.print("✅ CUDA version detected\n", .{});
        }
    }
}

test "nvidia_smi: configuration recommendations" {
    std.debug.print("\n=== Testing Configuration Recommendations ===\n", .{});
    
    const allocator = testing.allocator;
    
    const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
        if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
            std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
            return;
        }
        return err;
    };
    defer {
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    for (gpus) |gpu| {
        const config = nvidia_smi.getRecommendedConfig(gpu);
        
        std.debug.print("\nGPU [{d}] {s} - Recommended Config:\n", .{ gpu.index, gpu.name });
        std.debug.print("   Batch Size: {d}\n", .{config.max_batch_size});
        std.debug.print("   KV Tokens: {d}\n", .{config.kv_cache_tokens});
        std.debug.print("   FP16: {} ", .{config.use_fp16});
        if (config.use_fp16) {
            std.debug.print("✅\n", .{});
        } else {
            std.debug.print("(FP32 only)\n", .{});
        }
        std.debug.print("   Tensor Cores: {} ", .{config.use_tensor_cores});
        if (config.use_tensor_cores) {
            std.debug.print("✅\n", .{});
        } else {
            std.debug.print("(not available)\n", .{});
        }
        
        // Validate recommendations
        try testing.expect(config.max_batch_size >= 1);
        try testing.expect(config.max_batch_size <= 32);
        try testing.expect(config.kv_cache_tokens >= 1024);
        try testing.expect(config.kv_cache_tokens <= 8192);
        
        // Tensor Cores should match compute capability
        if (gpu.compute_capability.major >= 7) {
            try testing.expect(config.use_tensor_cores);
        }
    }
}

test "nvidia_smi: stress test - multiple calls" {
    std.debug.print("\n=== Stress Testing nvidia-smi (10 calls) ===\n", .{});
    
    const allocator = testing.allocator;
    
    const start_time = std.time.milliTimestamp();
    
    var i: u32 = 0;
    while (i < 10) : (i += 1) {
        const gpus = nvidia_smi.detectGPUs(allocator) catch |err| {
            if (err == error.NvidiaSmiNotFound or err == error.NvidiaSmiExecutionFailed) {
                std.debug.print("⚠️  Test skipped: nvidia-smi not available\n", .{});
                return;
            }
            return err;
        };
        
        for (gpus) |*gpu| gpu.deinit(allocator);
        allocator.free(gpus);
    }
    
    const elapsed = std.time.milliTimestamp() - start_time;
    const avg_ms = @divFloor(elapsed, 10);
    
    std.debug.print("Completed 10 calls in {d}ms\n", .{elapsed});
    std.debug.print("Average call time: {d}ms\n", .{avg_ms});
    
    if (avg_ms > 1000) {
        std.debug.print("⚠️  nvidia-smi calls are slow (>{d}ms)\n", .{avg_ms});
    } else {
        std.debug.print("✅ Performance acceptable\n", .{});
    }
}
