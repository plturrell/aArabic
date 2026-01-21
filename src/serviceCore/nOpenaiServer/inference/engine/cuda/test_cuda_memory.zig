// Unit tests for CUDA Memory Management
// Tests allocation, transfers, pools, and tracking

const std = @import("std");
const testing = std.testing;
const cuda_memory = @import("cuda_memory.zig");
const DeviceMemory = cuda_memory.DeviceMemory;
const PinnedMemory = cuda_memory.PinnedMemory;
const MemoryPool = cuda_memory.MemoryPool;
const AllocationTracker = cuda_memory.AllocationTracker;

test "cuda_memory: device memory allocation" {
    std.debug.print("\n=== Testing Device Memory Allocation ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 1024 * 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    try testing.expect(mem.size == 1024 * 1024);
    std.debug.print("✅ Allocated 1 MB device memory\n", .{});
}

test "cuda_memory: zero device memory" {
    std.debug.print("\n=== Testing Zero Device Memory ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    try mem.zero();
    std.debug.print("✅ Zeroed device memory\n", .{});
}

test "cuda_memory: host to device transfer" {
    std.debug.print("\n=== Testing Host to Device Transfer ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Create host data
    const host_data = [_]u8{1, 2, 3, 4, 5, 6, 7, 8};
    
    // Copy to device
    try mem.copyFromHost(&host_data);
    
    std.debug.print("✅ Transferred {d} bytes to device\n", .{host_data.len});
}

test "cuda_memory: device to host transfer" {
    std.debug.print("\n=== Testing Device to Host Transfer ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Zero device memory
    try mem.zero();
    
    // Read back
    var host_buffer: [1024]u8 = undefined;
    try mem.copyToHost(&host_buffer);
    
    // Verify all zeros
    for (host_buffer) |byte| {
        try testing.expect(byte == 0);
    }
    
    std.debug.print("✅ Transferred {d} bytes from device\n", .{host_buffer.len});
    std.debug.print("✅ Verified zeros\n", .{});
}

test "cuda_memory: round-trip transfer" {
    std.debug.print("\n=== Testing Round-Trip Transfer ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 256) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Create pattern
    var original: [256]u8 = undefined;
    for (&original, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    
    // Upload to device
    try mem.copyFromHost(&original);
    
    // Download from device
    var retrieved: [256]u8 = undefined;
    try mem.copyToHost(&retrieved);
    
    // Verify match
    try testing.expectEqualSlices(u8, &original, &retrieved);
    
    std.debug.print("✅ Round-trip transfer verified\n", .{});
}

test "cuda_memory: device to device copy" {
    std.debug.print("\n=== Testing Device to Device Copy ===\n", .{});
    
    const allocator = testing.allocator;
    
    var src = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer src.deinit();
    
    var dst = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            return;
        }
        return err;
    };
    defer dst.deinit();
    
    // Upload data to source
    var data: [1024]u8 = undefined;
    for (&data, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    try src.copyFromHost(&data);
    
    // Copy device to device
    try dst.copyFromDevice(&src);
    
    // Verify
    var result: [1024]u8 = undefined;
    try dst.copyToHost(&result);
    try testing.expectEqualSlices(u8, &data, &result);
    
    std.debug.print("✅ Device-to-device copy verified\n", .{});
}

test "cuda_memory: pinned memory allocation" {
    std.debug.print("\n=== Testing Pinned Memory Allocation ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = PinnedMemory.alloc(allocator, 1024 * 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    try testing.expect(mem.size == 1024 * 1024);
    std.debug.print("✅ Allocated 1 MB pinned memory\n", .{});
}

test "cuda_memory: pinned memory access" {
    std.debug.print("\n=== Testing Pinned Memory Access ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = PinnedMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Write to pinned memory
    const slice = mem.asSlice();
    for (slice, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    
    // Verify
    try testing.expect(slice[0] == 0);
    try testing.expect(slice[255] == 255);
    
    std.debug.print("✅ Pinned memory read/write works\n", .{});
}

test "cuda_memory: pinned to device transfer" {
    std.debug.print("\n=== Testing Pinned to Device Transfer ===\n", .{});
    
    const allocator = testing.allocator;
    
    var pinned = PinnedMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer pinned.deinit();
    
    var device = DeviceMemory.alloc(allocator, 1024) catch |err| {
        if (err == error.CudaError) {
            return;
        }
        return err;
    };
    defer device.deinit();
    
    // Fill pinned memory
    const slice = pinned.asSlice();
    for (slice, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    
    // Transfer to device
    try pinned.copyToDevice(&device);
    
    // Verify
    var result: [1024]u8 = undefined;
    try device.copyToHost(&result);
    try testing.expectEqualSlices(u8, slice, &result);
    
    std.debug.print("✅ Pinned-to-device transfer verified\n", .{});
}

test "cuda_memory: memory pool basic operations" {
    std.debug.print("\n=== Testing Memory Pool ===\n", .{});
    
    const allocator = testing.allocator;
    
    var pool = MemoryPool.init(allocator, 1024 * 1024) catch |err| {
        std.debug.print("⚠️  Test skipped: {}\n", .{err});
        return;
    };
    defer pool.deinit();
    
    // Allocate from pool
    const mem1 = pool.alloc() catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    
    _ = pool.alloc() catch |err| {
        if (err == error.CudaError) {
            return;
        }
        return err;
    };
    
    std.debug.print("✅ Allocated 2 blocks from pool\n", .{});
    
    // Check stats
    var stats = pool.getStats();
    try testing.expect(stats.currently_allocated == 2);
    try testing.expect(stats.free_blocks == 0);
    std.debug.print("   Currently allocated: {d}\n", .{stats.currently_allocated});
    
    // Return one to pool
    try pool.free(mem1);
    
    stats = pool.getStats();
    try testing.expect(stats.currently_allocated == 1);
    try testing.expect(stats.free_blocks == 1);
    std.debug.print("✅ Returned 1 block to pool\n", .{});
    std.debug.print("   Free blocks: {d}\n", .{stats.free_blocks});
    
    // Allocate again (should reuse)
    _ = pool.alloc() catch |err| {
        if (err == error.CudaError) {
            return;
        }
        return err;
    };
    
    stats = pool.getStats();
    try testing.expect(stats.free_blocks == 0);
    std.debug.print("✅ Reused block from pool\n", .{});
}

test "cuda_memory: allocation tracker" {
    std.debug.print("\n=== Testing Allocation Tracker ===\n", .{});
    
    const allocator = testing.allocator;
    
    var tracker = AllocationTracker.init(allocator);
    defer tracker.deinit();
    
    var mem1 = DeviceMemory.alloc(allocator, 1024 * 1024) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    try tracker.trackAllocation(mem1.ptr, mem1.size);
    
    var mem2 = DeviceMemory.alloc(allocator, 2 * 1024 * 1024) catch |err| {
        if (err == error.CudaError) {
            mem1.deinit();
            return;
        }
        return err;
    };
    try tracker.trackAllocation(mem2.ptr, mem2.size);
    
    // Check stats
    var stats = tracker.getStats();
    try testing.expect(stats.total_allocated == 3 * 1024 * 1024);
    try testing.expect(stats.active_allocations == 2);
    
    std.debug.print("   Total allocated: {d:.2} MB\n", .{
        @as(f32, @floatFromInt(stats.total_allocated)) / (1024.0 * 1024.0)
    });
    std.debug.print("   Active allocations: {d}\n", .{stats.active_allocations});
    
    // Free one
    try tracker.trackFree(mem1.ptr);
    mem1.deinit();
    
    stats = tracker.getStats();
    try testing.expect(stats.active_allocations == 1);
    std.debug.print("✅ Tracker updated after free\n", .{});
    
    // Clean up
    try tracker.trackFree(mem2.ptr);
    mem2.deinit();
    
    // Check for leaks
    try testing.expect(!tracker.checkLeaks());
    std.debug.print("✅ No memory leaks detected\n", .{});
}

test "cuda_memory: transfer utilities" {
    std.debug.print("\n=== Testing Transfer Utilities ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 256) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Test copyHostToDevice
    const host_data = [_]u8{1, 2, 3, 4, 5};
    try cuda_memory.copyHostToDevice(mem.ptr, &host_data);
    std.debug.print("✅ copyHostToDevice works\n", .{});
    
    // Test copyDeviceToHost
    var result: [5]u8 = undefined;
    try cuda_memory.copyDeviceToHost(&result, mem.ptr, 5);
    try testing.expectEqualSlices(u8, &host_data, &result);
    std.debug.print("✅ copyDeviceToHost works\n", .{});
}

test "cuda_memory: zero and set utilities" {
    std.debug.print("\n=== Testing Zero and Set Utilities ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mem = DeviceMemory.alloc(allocator, 256) catch |err| {
        if (err == error.CudaError) {
            std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer mem.deinit();
    
    // Zero memory
    try cuda_memory.zeroDeviceMemory(mem.ptr, mem.size);
    
    var result: [256]u8 = undefined;
    try mem.copyToHost(&result);
    for (result) |byte| {
        try testing.expect(byte == 0);
    }
    std.debug.print("✅ zeroDeviceMemory works\n", .{});
    
    // Set memory
    try cuda_memory.setDeviceMemory(mem.ptr, 42, mem.size);
    try mem.copyToHost(&result);
    for (result) |byte| {
        try testing.expect(byte == 42);
    }
    std.debug.print("✅ setDeviceMemory works\n", .{});
}

test "cuda_memory: multiple allocations" {
    std.debug.print("\n=== Testing Multiple Allocations ===\n", .{});
    
    const allocator = testing.allocator;
    
    var mems: [10]DeviceMemory = undefined;
    
    // Allocate multiple
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        mems[i] = DeviceMemory.alloc(allocator, 1024 * 100) catch |err| {
            if (err == error.CudaError) {
                if (i == 0) {
                    std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
                    return;
                }
                // Clean up what we allocated
                var j: usize = 0;
                while (j < i) : (j += 1) {
                    mems[j].deinit();
                }
                return err;
            }
            return err;
        };
    }
    
    std.debug.print("✅ Allocated 10 blocks of 100 KB each\n", .{});
    
    // Clean up
    for (&mems) |*mem| {
        mem.deinit();
    }
    
    std.debug.print("✅ Freed all blocks\n", .{});
}

test "cuda_memory: stress test" {
    std.debug.print("\n=== Stress Testing Memory Operations ===\n", .{});
    
    const allocator = testing.allocator;
    
    const iterations = 100;
    var i: usize = 0;
    
    while (i < iterations) : (i += 1) {
        var mem = DeviceMemory.alloc(allocator, 1024) catch |err| {
            if (err == error.CudaError) {
                if (i == 0) {
                    std.debug.print("⚠️  Test skipped: No GPU available\n", .{});
                    return;
                }
                return err;
            }
            return err;
        };
        defer mem.deinit();
        
        // Perform operations
        try mem.zero();
        
        if (i % 10 == 0) {
            std.debug.print("   Iteration {d}/{d}\n", .{ i, iterations });
        }
    }
    
    std.debug.print("✅ Completed {d} alloc/free cycles\n", .{iterations});
}
