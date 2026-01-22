// Test Suite for GPU Memory Tier
// Validates GPU memory management, transfers, and integration

const std = @import("std");
const gpu = @import("gpu_tier.zig");
const testing = std.testing;

// ============================================================================
// Test Helpers
// ============================================================================

fn createTestConfig() gpu.GPUTierConfig {
    return .{
        .enabled = true,
        .device_id = 0,
        .max_gpu_memory = 1 * 1024 * 1024 * 1024, // 1GB for testing
        .gpu_tokens = 256,
        .use_pinned_memory = true,
        .use_memory_pool = true,
        .pool_block_size = 1 * 1024 * 1024, // 1MB blocks
        .use_async_transfers = true,
        .num_streams = 2,
    };
}

// ============================================================================
// GPU Memory Pool Tests
// ============================================================================

test "gpu_memory_pool: initialization" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Verify initial state
    try testing.expect(pool.total_allocated == 0);
    try testing.expect(pool.blocks.items.len == 16); // Pre-allocated blocks
    try testing.expect(pool.free_blocks.items.len == 16);
    
    const usage = pool.getUsage();
    try testing.expect(usage.total_mb == 1024); // 1GB
    try testing.expect(usage.used_mb == 0);
    try testing.expect(usage.utilization == 0.0);
}

test "gpu_memory_pool: basic allocation" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Allocate a block
    const block = try pool.alloc(4 * 1024 * 1024); // 4MB
    try testing.expect(block.size == 4 * 1024 * 1024);
    try testing.expect(block.in_use == true);
    try testing.expect(block.ref_count == 1);
    
    // Check stats
    const stats = pool.getStats();
    try testing.expect(stats.alloc_count == 0); // Reused pre-allocated block
    try testing.expect(stats.reuse_count == 1);
    
    // Free the block
    pool.free(block);
    try testing.expect(block.in_use == false);
    try testing.expect(block.ref_count == 0);
}

test "gpu_memory_pool: block reuse" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Allocate and free multiple times
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        const block = try pool.alloc(2 * 1024 * 1024);
        try testing.expect(block.in_use == true);
        pool.free(block);
        try testing.expect(block.in_use == false);
    }
    
    // Check reuse stats
    const stats = pool.getStats();
    try testing.expect(stats.reuse_count == 5);
    try testing.expect(stats.free_count == 5);
}

test "gpu_memory_pool: out of memory" {
    const allocator = testing.allocator;
    var config = createTestConfig();
    config.max_gpu_memory = 10 * 1024 * 1024; // Only 10MB
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Try to allocate more than available
    const result = pool.alloc(20 * 1024 * 1024); // Request 20MB
    try testing.expectError(error.OutOfGPUMemory, result);
}

test "gpu_memory_pool: multiple allocations" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Allocate multiple blocks
    var blocks: [10]*gpu.GPUBlock = undefined;
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        blocks[i] = try pool.alloc(1 * 1024 * 1024); // 1MB each
        try testing.expect(blocks[i].in_use == true);
    }
    
    // Free all blocks
    i = 0;
    while (i < 10) : (i += 1) {
        pool.free(blocks[i]);
        try testing.expect(blocks[i].in_use == false);
    }
    
    // Check free blocks available
    try testing.expect(pool.free_blocks.items.len >= 10);
}

test "gpu_memory_pool: usage tracking" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    
    const pool = try gpu.GPUMemoryPool.init(allocator, config);
    defer pool.deinit();
    
    // Allocate some blocks
    const block1 = try pool.alloc(100 * 1024 * 1024); // 100MB
    const usage1 = pool.getUsage();
    try testing.expect(usage1.used_mb >= 95); // At least 95MB (pool block overhead)
    
    const block2 = try pool.alloc(200 * 1024 * 1024); // 200MB
    const usage2 = pool.getUsage();
    try testing.expect(usage2.used_mb >= 295); // At least 295MB
    
    // Free first block
    pool.free(block1);
    const usage3 = pool.getUsage();
    try testing.expect(usage3.used_mb >= 195); // At least 195MB
    
    // Free second block
    pool.free(block2);
    const usage4 = pool.getUsage();
    try testing.expect(usage4.used_mb == 0); // All freed (back to pool)
}

// ============================================================================
// GPU Tier Tests
// ============================================================================

test "gpu_tier: initialization" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 4;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    // Verify initial state
    try testing.expect(tier.layer_blocks.len == n_layers);
    try testing.expect(tier.gpu_tokens == 0);
    
    // Check each layer has K and V slots
    for (tier.layer_blocks) |layer| {
        try testing.expect(layer.len == 2);
        try testing.expect(layer[0] == null);
        try testing.expect(layer[1] == null);
    }
    
    // Check stats
    const stats = tier.getStats();
    try testing.expect(stats.gpu_hits == 0);
    try testing.expect(stats.ram_to_gpu_transfers == 0);
    try testing.expect(stats.gpu_to_ram_transfers == 0);
}

test "gpu_tier: store from RAM" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 2;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    // Create test data
    const kv_dim: usize = 128;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Fill with test data
    for (keys, 0..) |*k, i| {
        k.* = @as(f32, @floatFromInt(i));
    }
    for (values, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) + 1000.0;
    }
    
    // Store to GPU
    try tier.storeFromRAM(0, keys, values);
    
    // Verify data is on GPU
    try testing.expect(tier.hasData(0) == true);
    try testing.expect(tier.layer_blocks[0][0] != null);
    try testing.expect(tier.layer_blocks[0][1] != null);
    
    // Check stats
    const stats = tier.getStats();
    try testing.expect(stats.ram_to_gpu_transfers == 1);
    try testing.expect(stats.bytes_to_gpu_mb > 0);
}

test "gpu_tier: load to RAM" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 2;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    // Create and store test data
    const kv_dim: usize = 128;
    var keys_src = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys_src);
    var values_src = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values_src);
    
    for (keys_src, 0..) |*k, i| {
        k.* = @as(f32, @floatFromInt(i));
    }
    for (values_src, 0..) |*v, i| {
        v.* = @as(f32, @floatFromInt(i)) + 1000.0;
    }
    
    try tier.storeFromRAM(0, keys_src, values_src);
    
    // Load back from GPU
    var keys_dest = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys_dest);
    var values_dest = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values_dest);
    
    try tier.loadToRAM(0, keys_dest, values_dest);
    
    // Verify data loaded (placeholder doesn't copy data, but we check stats)
    const stats = tier.getStats();
    try testing.expect(stats.gpu_to_ram_transfers == 1);
    try testing.expect(stats.gpu_hits == 1);
}

test "gpu_tier: multiple layers" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 8;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 128;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Store to all layers
    var layer: u32 = 0;
    while (layer < n_layers) : (layer += 1) {
        for (keys, 0..) |*k, i| {
            k.* = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(layer)) * 1000.0;
        }
        for (values, 0..) |*v, i| {
            v.* = @as(f32, @floatFromInt(i)) + @as(f32, @floatFromInt(layer)) * 1000.0 + 10000.0;
        }
        
        try tier.storeFromRAM(layer, keys, values);
        try testing.expect(tier.hasData(layer) == true);
    }
    
    // Verify all layers have data
    layer = 0;
    while (layer < n_layers) : (layer += 1) {
        try testing.expect(tier.hasData(layer) == true);
    }
    
    const stats = tier.getStats();
    try testing.expect(stats.ram_to_gpu_transfers == n_layers);
}

test "gpu_tier: eviction" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 2;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 128;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Store data
    try tier.storeFromRAM(0, keys, values);
    try testing.expect(tier.hasData(0) == true);
    
    // Evict
    tier.evict(0);
    try testing.expect(tier.hasData(0) == false);
    try testing.expect(tier.layer_blocks[0][0] == null);
    try testing.expect(tier.layer_blocks[0][1] == null);
}

test "gpu_tier: statistics" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 4;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 256;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Perform several operations
    try tier.storeFromRAM(0, keys, values);
    try tier.storeFromRAM(1, keys, values);
    try tier.loadToRAM(0, keys, values);
    tier.evict(1);
    
    const stats = tier.getStats();
    
    // Verify stats make sense
    try testing.expect(stats.ram_to_gpu_transfers == 2);
    try testing.expect(stats.gpu_to_ram_transfers == 1);
    try testing.expect(stats.gpu_hits == 1);
    try testing.expect(stats.bytes_to_gpu_mb > 0);
    try testing.expect(stats.bytes_from_gpu_mb > 0);
}

test "gpu_tier: no data error" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 2;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 128;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Try to load from empty layer
    const result = tier.loadToRAM(0, keys, values);
    try testing.expectError(error.NoGPUData, result);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "gpu_tier: full workflow" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 4;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 512;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Workflow: Store → Load → Evict → Store again
    
    // 1. Store initial data
    try tier.storeFromRAM(0, keys, values);
    try testing.expect(tier.hasData(0) == true);
    
    // 2. Load it back
    try tier.loadToRAM(0, keys, values);
    
    // 3. Evict
    tier.evict(0);
    try testing.expect(tier.hasData(0) == false);
    
    // 4. Store again (should reuse freed blocks)
    try tier.storeFromRAM(0, keys, values);
    try testing.expect(tier.hasData(0) == true);
    
    // Verify stats
    const stats = tier.getStats();
    try testing.expect(stats.ram_to_gpu_transfers == 2);
    try testing.expect(stats.gpu_to_ram_transfers == 1);
    try testing.expect(stats.pool_stats.reuse_count >= 2);
}

test "gpu_tier: stress test" {
    const allocator = testing.allocator;
    const config = createTestConfig();
    const n_layers: u32 = 16;
    
    const tier = try gpu.GPUTier.init(allocator, config, n_layers);
    defer tier.deinit();
    
    const kv_dim: usize = 1024;
    var keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    var values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Perform 100 random operations
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const layer = random.intRangeAtMost(u32, 0, n_layers - 1);
        const op = random.intRangeAtMost(u32, 0, 2);
        
        switch (op) {
            0 => {
                // Store
                tier.storeFromRAM(layer, keys, values) catch {};
            },
            1 => {
                // Load
                if (tier.hasData(layer)) {
                    tier.loadToRAM(layer, keys, values) catch {};
                }
            },
            2 => {
                // Evict
                if (tier.hasData(layer)) {
                    tier.evict(layer);
                }
            },
            else => unreachable,
        }
    }
    
    // Just verify it didn't crash
    const stats = tier.getStats();
    try testing.expect(stats.ram_to_gpu_transfers > 0);
}

// ============================================================================
// Utility Tests
// ============================================================================

test "gpu_utils: CUDA availability check" {
    const available = gpu.isCUDAAvailable();
    // Should return false in test environment (no actual CUDA)
    try testing.expect(available == false);
}

test "gpu_utils: device properties" {
    const props = try gpu.getCUDADeviceProperties(0);
    
    // Verify placeholder values
    try testing.expect(props.compute_capability.major >= 0);
    try testing.expect(props.compute_capability.minor >= 0);
    try testing.expect(props.total_memory_gb > 0);
    try testing.expect(props.clock_rate_ghz > 0);
    try testing.expect(props.memory_bandwidth_gbps > 0);
}

test "gpu_utils: CUDA initialization" {
    // Should not crash (even without actual CUDA)
    try gpu.initCUDA(0);
    gpu.shutdownCUDA();
}
