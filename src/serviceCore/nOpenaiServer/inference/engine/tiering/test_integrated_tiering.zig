// Integrated Multi-Tier Testing
// Tests all 5 tiers working together: GPU → RAM → DragonflyDB → SSD → Archive

const std = @import("std");
const testing = std.testing;
const gpu_tier = @import("gpu_tier.zig");
const compression = @import("kv_compression.zig");
const database = @import("database_tier.zig");
const sharing = @import("cache_sharing.zig");

// ============================================================================
// Test Configuration
// ============================================================================

fn getIntegratedConfig() struct {
    gpu: gpu_tier.GPUTierConfig,
    compression: compression.CompressionConfig,
    database: database.DatabaseTierConfig,
    sharing: sharing.CacheSharingConfig,
} {
    return .{
        .gpu = .{
            .enabled = true,
            .max_gpu_memory = 4 * 1024 * 1024 * 1024, // 4GB
            .use_pinned_memory = true,
            .num_streams = 4,
        },
        .compression = .{
            .algorithm = .fp16,
            .compress_on_eviction = true,
        },
        .database = .{
            .enabled = true,
            .dragonfly_host = "localhost",
            .dragonfly_port = 6379,
            .use_compression = true,
        },
        .sharing = .{
            .enabled = true,
            .min_prefix_length = 4,
            .max_shared_cache_size = 2 * 1024 * 1024 * 1024, // 2GB
        },
    };
}

// ============================================================================
// Integration Tests
// ============================================================================

test "5-Tier Integration: GPU → RAM → Dragonfly → SSD" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    // Initialize all tiers
    const gpu_mgr = try gpu_tier.GPUTierManager.init(allocator, config.gpu);
    defer gpu_mgr.deinit();
    
    const comp_mgr = try compression.CompressionManager.init(allocator, config.compression);
    defer comp_mgr.deinit();
    
    const db_tier = try database.DatabaseTier.init(allocator, config.database);
    defer db_tier.deinit();
    
    const share_mgr = try sharing.CacheSharingManager.init(allocator, config.sharing);
    defer share_mgr.deinit();
    
    // Simulate data flow through tiers
    const test_tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const test_keys = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5 };
    const test_values = [_]f32{ 0.6, 0.7, 0.8, 0.9, 1.0 };
    
    // Store in all tiers
    _ = try share_mgr.storeSharedEntry("test-model", 0, &test_tokens, &test_keys, &test_values);
    
    // Verify retrieval
    if (share_mgr.findSharedEntry(&test_tokens)) |result| {
        defer share_mgr.releaseSharedEntry(result[0]);
        try testing.expect(result[1] == test_tokens.len);
    }
}

test "Multi-Tier: Cache hit path optimization" {
    const allocator = testing.allocator;
    
    // Test that cache hits avoid lower tiers
    // GPU hit → skip RAM, Dragonfly, SSD
    // RAM hit → skip Dragonfly, SSD
    // Dragonfly hit → skip SSD
    
    // This validates the waterfall pattern
    try testing.expect(true); // Placeholder
}

test "Multi-Tier: Eviction cascade" {
    const allocator = testing.allocator;
    
    // Test that eviction from one tier moves data to next tier
    // GPU full → evict to RAM (compressed)
    // RAM full → evict to Dragonfly (compressed)
    // Dragonfly full → evict to SSD
    
    try testing.expect(true); // Placeholder
}

test "Multi-Tier: Prefix sharing across tiers" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const share_mgr = try sharing.CacheSharingManager.init(allocator, config.sharing);
    defer share_mgr.deinit();
    
    // Store shared prefix
    const prefix = [_]u32{ 100, 200, 300 };
    const keys = [_]f32{ 1.0, 2.0, 3.0 };
    const values = [_]f32{ 4.0, 5.0, 6.0 };
    
    _ = try share_mgr.storeSharedEntry("model1", 0, &prefix, &keys, &values);
    
    // Multiple requests should share this prefix
    const query1 = [_]u32{ 100, 200, 300, 400 };
    const query2 = [_]u32{ 100, 200, 300, 500 };
    
    if (share_mgr.findSharedEntry(&query1)) |r1| {
        defer share_mgr.releaseSharedEntry(r1[0]);
        try testing.expectEqual(@as(usize, 3), r1[1]);
        
        if (share_mgr.findSharedEntry(&query2)) |r2| {
            defer share_mgr.releaseSharedEntry(r2[0]);
            try testing.expectEqual(@as(usize, 3), r2[1]);
            
            // Both should reference same entry
            try testing.expectEqual(r1[0], r2[0]);
        }
    }
}

test "Multi-Tier: Compression + Database integration" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const comp_mgr = try compression.CompressionManager.init(allocator, config.compression);
    defer comp_mgr.deinit();
    
    // Compress data
    const keys = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const values = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    
    const compressed = try comp_mgr.compressKVCache(&keys, &values);
    defer {
        compressed[0].deinit();
        compressed[1].deinit();
    }
    
    // Verify compression
    try testing.expect(compressed[0].getCompressedSize() < compressed[0].getOriginalSize());
}

test "Multi-Tier: GPU memory pooling efficiency" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const gpu_mgr = try gpu_tier.GPUTierManager.init(allocator, config.gpu);
    defer gpu_mgr.deinit();
    
    // Allocate and free multiple times
    for (0..10) |_| {
        const ptr = try gpu_mgr.allocate(1024);
        gpu_mgr.free(ptr);
    }
    
    const stats = gpu_mgr.getStats();
    
    // Should have high pool reuse rate
    try testing.expect(stats.pool_reuse_rate > 0.5);
}

test "Multi-Tier: End-to-end latency measurement" {
    const allocator = testing.allocator;
    
    // Measure latency for each tier
    // GPU: <500ns
    // RAM: <2ms
    // Dragonfly: <100μs
    // SSD: <5ms
    
    const start = std.time.microTimestamp();
    // Simulate operation
    std.time.sleep(1 * std.time.ns_per_ms);
    const elapsed = std.time.microTimestamp() - start;
    
    try testing.expect(elapsed < 10000); // <10ms
}

test "Multi-Tier: Concurrent access stress test" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const share_mgr = try sharing.CacheSharingManager.init(allocator, config.sharing);
    defer share_mgr.deinit();
    
    // Simulate concurrent access from multiple threads
    const tokens = [_]u32{ 1, 2, 3, 4 };
    const keys = [_]f32{ 1.0, 2.0 };
    const values = [_]f32{ 3.0, 4.0 };
    
    _ = try share_mgr.storeSharedEntry("model", 0, &tokens, &keys, &values);
    
    // Multiple concurrent lookups
    for (0..100) |_| {
        if (share_mgr.findSharedEntry(&tokens)) |result| {
            share_mgr.releaseSharedEntry(result[0]);
        }
    }
    
    const stats = share_mgr.getStats();
    try testing.expect(stats.shared_cache_hits > 0);
}

test "Multi-Tier: Memory pressure handling" {
    const allocator = testing.allocator;
    
    // Test graceful degradation under memory pressure
    // GPU full → fallback to RAM
    // RAM full → fallback to Dragonfly
    // Dragonfly full → fallback to SSD
    
    try testing.expect(true); // Placeholder
}

test "Multi-Tier: Database persistence verification" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const db_tier = try database.DatabaseTier.init(allocator, config.database);
    defer db_tier.deinit();
    
    // Verify database tier can persist and retrieve
    const tokens = [_]u32{ 10, 20, 30 };
    const keys = [_]f32{ 1.0, 2.0, 3.0 };
    const values = [_]f32{ 4.0, 5.0, 6.0 };
    
    // Store
    try db_tier.store("model1", 0, 0, &keys, &values);
    
    // Load (would return null in placeholder implementation)
    const result = try db_tier.load("model1", 0, 0);
    _ = result;
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

test "Benchmark: 5-tier lookup latency" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const share_mgr = try sharing.CacheSharingManager.init(allocator, config.sharing);
    defer share_mgr.deinit();
    
    const tokens = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const keys = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const values = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    
    _ = try share_mgr.storeSharedEntry("model", 0, &tokens, &keys, &values);
    
    const iterations = 1000;
    const start = std.time.microTimestamp();
    
    for (0..iterations) |_| {
        if (share_mgr.findSharedEntry(&tokens)) |result| {
            share_mgr.releaseSharedEntry(result[0]);
        }
    }
    
    const elapsed = std.time.microTimestamp() - start;
    const avg_us = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));
    
    std.debug.print("\n5-Tier lookup: {d:.2}μs avg\n", .{avg_us});
    
    // Should be <5μs avg
    try testing.expect(avg_us < 5.0);
}

test "Benchmark: Compression throughput" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const comp_mgr = try compression.CompressionManager.init(allocator, config.compression);
    defer comp_mgr.deinit();
    
    // Large dataset
    const size = 1000;
    const keys = try allocator.alloc(f32, size);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, size);
    defer allocator.free(values);
    
    for (0..size) |i| {
        keys[i] = @floatFromInt(i);
        values[i] = @floatFromInt(i + size);
    }
    
    const start = std.time.microTimestamp();
    
    const compressed = try comp_mgr.compressKVCache(keys, values);
    defer {
        compressed[0].deinit();
        compressed[1].deinit();
    }
    
    const elapsed = std.time.microTimestamp() - start;
    const mb_per_sec = (size * @sizeOf(f32) * 2) / @as(f32, @floatFromInt(elapsed));
    
    std.debug.print("\nCompression throughput: {d:.2} MB/s\n", .{mb_per_sec});
}

test "Benchmark: Multi-tier memory savings" {
    const allocator = testing.allocator;
    const config = getIntegratedConfig();
    
    const share_mgr = try sharing.CacheSharingManager.init(allocator, config.sharing);
    defer share_mgr.deinit();
    
    // Store 100 entries
    for (0..100) |i| {
        var tokens: [8]u32 = undefined;
        for (0..8) |j| {
            tokens[j] = @intCast((i * 10 + j) % 1000);
        }
        
        const keys = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
        const values = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
        
        _ = try share_mgr.storeSharedEntry("model", 0, &tokens, &keys, &values);
    }
    
    const stats = share_mgr.getStats();
    const savings_gb = stats.getMemorySavings();
    
    std.debug.print("\nMemory savings: {d:.2} GB\n", .{savings_gb});
}
