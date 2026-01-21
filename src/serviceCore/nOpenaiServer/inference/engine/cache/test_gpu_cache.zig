// Comprehensive GPU KV Cache Tests
// Tests for GPU-accelerated KV cache implementation

const std = @import("std");
const testing = std.testing;
const GpuKvCache = @import("gpu_kv_cache.zig").GpuKvCache;
const GpuCacheEntry = @import("gpu_kv_cache.zig").GpuCacheEntry;
const GpuCacheConfig = @import("gpu_cache_config.zig").GpuCacheConfig;
const CudaManager = @import("../cuda/cuda_manager.zig").CudaManager;

// ============================================================================
// Test Helpers
// ============================================================================

fn skipIfNoGpu(err: anyerror) bool {
    return err == error.CudaError or err == error.NoGPUFound;
}

fn createTestConfig() GpuCacheConfig {
    return GpuCacheConfig{
        .n_layers = 4,
        .n_heads = 8,
        .head_dim = 64,
        .max_seq_len = 128,
        .batch_size = 2,
        .enable_metrics = true,
        .track_hit_rate = true,
    };
}

fn createTestData(allocator: std.mem.Allocator, size: usize) ![]f32 {
    const data = try allocator.alloc(f32, size);
    for (0..size) |i| {
        data[i] = @floatFromInt(i);
    }
    return data;
}

// ============================================================================
// Basic Functionality Tests
// ============================================================================

test "gpu_cache: initialization and cleanup" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    try testing.expect(cache.entries.items.len == 0);
    try testing.expect(cache.total_allocated == 0);
    try testing.expect(cache.current_pos == 0);
    try testing.expect(cache.hits == 0);
    try testing.expect(cache.misses == 0);
    try testing.expect(cache.evictions == 0);
    
    std.debug.print("✓ GPU cache initialization and cleanup working\n", .{});
}

test "gpu_cache: config validation" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    // Valid config should work
    const valid_config = createTestConfig();
    var cache1 = try GpuKvCache.init(allocator, valid_config, cuda_manager);
    defer cache1.deinit();
    
    // Invalid configs should fail
    var invalid_config = createTestConfig();
    invalid_config.n_layers = 0;
    try testing.expectError(error.InvalidLayers, GpuKvCache.init(allocator, invalid_config, cuda_manager));
    
    std.debug.print("✓ Config validation working\n", .{});
}

// ============================================================================
// Cache Entry Management Tests
// ============================================================================

test "gpu_cache: store single entry" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    try cache.store(0, 0, key_data, value_data);
    
    try testing.expect(cache.entries.items.len == 1);
    try testing.expect(cache.total_allocated > 0);
    try testing.expect(cache.contains(0, 0));
    
    std.debug.print("✓ Store single entry working\n", .{});
}

test "gpu_cache: store multiple entries" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Store entries for different layers and batches
    try cache.store(0, 0, key_data, value_data);
    try cache.store(0, 1, key_data, value_data);
    try cache.store(1, 0, key_data, value_data);
    
    try testing.expect(cache.entries.items.len == 3);
    try testing.expect(cache.contains(0, 0));
    try testing.expect(cache.contains(0, 1));
    try testing.expect(cache.contains(1, 0));
    try testing.expect(!cache.contains(1, 1));
    
    std.debug.print("✓ Store multiple entries working\n", .{});
}

test "gpu_cache: retrieve entries" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    try cache.store(0, 0, key_data, value_data);
    
    // Retrieve should succeed
    const keys = try cache.getKeys(0, 0);
    try testing.expect(keys != null);
    
    const values = try cache.getValues(0, 0);
    try testing.expect(values != null);
    
    // Non-existent entry should return null
    const no_keys = try cache.getKeys(1, 1);
    try testing.expect(no_keys == null);
    
    std.debug.print("✓ Retrieve entries working\n", .{});
}

test "gpu_cache: entry bounds checking" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Out of range layer should fail
    try testing.expectError(error.LayerOutOfRange, cache.store(999, 0, key_data, value_data));
    
    // Out of range batch should fail
    try testing.expectError(error.BatchOutOfRange, cache.store(0, 999, key_data, value_data));
    
    // Wrong size should fail
    const wrong_size = try allocator.alloc(f32, 10);
    defer allocator.free(wrong_size);
    try testing.expectError(error.InvalidSize, cache.store(0, 0, wrong_size, value_data));
    
    std.debug.print("✓ Bounds checking working\n", .{});
}

// ============================================================================
// Eviction Policy Tests
// ============================================================================

test "gpu_cache: LRU eviction" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    var config = createTestConfig();
    config.eviction_policy = .lru;
    config.max_gpu_memory_mb = 1; // Force eviction
    
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Store first entry
    try cache.store(0, 0, key_data, value_data);
    std.time.sleep(std.time.ns_per_ms * 10); // Small delay
    
    // Store second entry (will trigger eviction of first)
    try cache.store(1, 0, key_data, value_data);
    
    try testing.expect(cache.evictions > 0);
    
    std.debug.print("✓ LRU eviction working\n", .{});
}

test "gpu_cache: FIFO eviction" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    var config = createTestConfig();
    config.eviction_policy = .fifo;
    config.max_gpu_memory_mb = 1; // Force eviction
    
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    try cache.store(0, 0, key_data, value_data);
    try cache.store(1, 0, key_data, value_data);
    
    try testing.expect(cache.evictions > 0);
    
    std.debug.print("✓ FIFO eviction working\n", .{});
}

test "gpu_cache: clear all entries" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Add multiple entries
    try cache.store(0, 0, key_data, value_data);
    try cache.store(1, 0, key_data, value_data);
    try cache.store(2, 0, key_data, value_data);
    
    try testing.expect(cache.entries.items.len == 3);
    
    // Clear all
    try cache.clear();
    
    try testing.expect(cache.entries.items.len == 0);
    try testing.expect(cache.total_allocated == 0);
    
    std.debug.print("✓ Clear all entries working\n", .{});
}

// ============================================================================
// Statistics and Monitoring Tests
// ============================================================================

test "gpu_cache: hit rate tracking" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Store entry
    try cache.store(0, 0, key_data, value_data);
    
    // Hit
    _ = try cache.getKeys(0, 0);
    try testing.expect(cache.hits == 1);
    
    // Miss
    _ = try cache.getKeys(1, 1);
    try testing.expect(cache.misses == 2); // Store counts as miss + this miss
    
    // Check stats
    const stats = cache.getStats();
    try testing.expect(stats.hits == 1);
    try testing.expect(stats.misses == 2);
    try testing.expect(stats.hit_rate > 0.0 and stats.hit_rate < 1.0);
    
    std.debug.print("✓ Hit rate tracking working\n", .{});
}

test "gpu_cache: memory usage tracking" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    const initial_allocated = cache.total_allocated;
    try testing.expect(initial_allocated == 0);
    
    try cache.store(0, 0, key_data, value_data);
    
    try testing.expect(cache.total_allocated > initial_allocated);
    
    const stats = cache.getStats();
    try testing.expect(stats.total_allocated_bytes == cache.total_allocated);
    
    std.debug.print("✓ Memory tracking working\n", .{});
}

test "gpu_cache: statistics printing" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    // Should not crash
    cache.printStats();
    
    std.debug.print("✓ Statistics printing working\n", .{});
}

// ============================================================================
// Entry Metadata Tests
// ============================================================================

test "gpu_cache: entry metadata tracking" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    try cache.store(0, 0, key_data, value_data);
    
    const entry = &cache.entries.items[0];
    try testing.expect(entry.layer_id == 0);
    try testing.expect(entry.batch_id == 0);
    try testing.expect(entry.sequence_length == 1);
    try testing.expect(entry.access_count == 1);
    try testing.expect(entry.last_access > 0);
    
    // Access again
    _ = try cache.getKeys(0, 0);
    try testing.expect(entry.access_count == 2);
    
    std.debug.print("✓ Entry metadata tracking working\n", .{});
}

test "gpu_cache: entry memory usage calculation" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    try cache.store(0, 0, key_data, value_data);
    
    const entry = &cache.entries.items[0];
    const mem_usage = entry.getMemoryUsage();
    
    const expected = kv_size * @sizeOf(f32) * 2; // keys + values
    try testing.expect(mem_usage == expected);
    
    std.debug.print("✓ Entry memory calculation working\n", .{});
}

// ============================================================================
// Integration Tests
// ============================================================================

test "gpu_cache: full lifecycle" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Store
    try cache.store(0, 0, key_data, value_data);
    try testing.expect(cache.contains(0, 0));
    
    // Retrieve
    const keys = try cache.getKeys(0, 0);
    try testing.expect(keys != null);
    
    // Update stats
    try testing.expect(cache.hits > 0);
    
    // Clear
    try cache.clear();
    try testing.expect(!cache.contains(0, 0));
    
    std.debug.print("✓ Full lifecycle working\n", .{});
}

test "gpu_cache: multiple layers and batches" {
    const allocator = testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (skipIfNoGpu(err)) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = createTestConfig();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try createTestData(allocator, kv_size);
    defer allocator.free(key_data);
    const value_data = try createTestData(allocator, kv_size);
    defer allocator.free(value_data);
    
    // Store entries for all layers and batches
    for (0..config.n_layers) |layer| {
        for (0..config.batch_size) |batch| {
            try cache.store(@intCast(layer), @intCast(batch), key_data, value_data);
        }
    }
    
    const expected_entries = config.n_layers * config.batch_size;
    try testing.expect(cache.entries.items.len == expected_entries);
    
    // Verify all entries exist
    for (0..config.n_layers) |layer| {
        for (0..config.batch_size) |batch| {
            try testing.expect(cache.contains(@intCast(layer), @intCast(batch)));
        }
    }
    
    std.debug.print("✓ Multiple layers and batches working\n", .{});
}

// ============================================================================
// Run All Tests
// ============================================================================

test {
    std.testing.refAllDecls(@This());
}
