// Tests for Cache Sharing System
// Validates prefix detection, reference counting, and shared cache management

const std = @import("std");
const testing = std.testing;
const sharing = @import("cache_sharing.zig");

// ============================================================================
// Test Configuration
// ============================================================================

fn getTestConfig() sharing.CacheSharingConfig {
    return .{
        .enabled = true,
        .min_prefix_length = 2,
        .max_trie_depth = 32,
        .auto_detect_prefixes = true,
        .protect_shared_entries = true,
        .shared_entry_ttl = 300,
        .max_shared_cache_size = 100 * 1024 * 1024, // 100MB
        .compress_shared_prefixes = false,
        .enable_replication = false,
    };
}

// ============================================================================
// Basic Tests
// ============================================================================

test "SharedCacheEntry: creation and reference counting" {
    const allocator = testing.allocator;
    
    const tokens = [_]u32{ 1, 2, 3, 4, 5 };
    const keys = [_]f32{ 0.1, 0.2, 0.3 };
    const values = [_]f32{ 0.4, 0.5, 0.6 };
    
    const entry = try sharing.SharedCacheEntry.init(
        allocator,
        12345,
        "test-model",
        0,
        &tokens,
        &keys,
        &values,
    );
    defer entry.deinit(allocator);
    
    // Check initial state
    try testing.expectEqual(@as(u64, 12345), entry.id);
    try testing.expectEqual(@as(u32, 0), entry.layer);
    try testing.expectEqual(@as(usize, 5), entry.tokens.len);
    try testing.expectEqual(@as(u32, 0), entry.getRefCount());
    try testing.expect(entry.canEvict());
    
    // Test acquire/release
    entry.acquire();
    try testing.expectEqual(@as(u32, 1), entry.getRefCount());
    try testing.expect(!entry.canEvict());
    
    entry.acquire();
    try testing.expectEqual(@as(u32, 2), entry.getRefCount());
    
    entry.release();
    try testing.expectEqual(@as(u32, 1), entry.getRefCount());
    
    entry.release();
    try testing.expectEqual(@as(u32, 0), entry.getRefCount());
    try testing.expect(entry.canEvict());
}

test "PrefixTree: insert and find" {
    const allocator = testing.allocator;
    
    const tree = try sharing.PrefixTree.init(allocator);
    defer tree.deinit();
    
    // Create test entries
    const tokens1 = [_]u32{ 1, 2, 3 };
    const tokens2 = [_]u32{ 1, 2, 3, 4, 5 };
    const keys = [_]f32{ 0.1, 0.2 };
    const values = [_]f32{ 0.3, 0.4 };
    
    const entry1 = try sharing.SharedCacheEntry.init(
        allocator,
        111,
        "model1",
        0,
        &tokens1,
        &keys,
        &values,
    );
    defer entry1.deinit(allocator);
    
    const entry2 = try sharing.SharedCacheEntry.init(
        allocator,
        222,
        "model2",
        0,
        &tokens2,
        &keys,
        &values,
    );
    defer entry2.deinit(allocator);
    
    // Insert into tree
    try tree.insert(&tokens1, entry1);
    try tree.insert(&tokens2, entry2);
    
    // Find exact match
    const query1 = [_]u32{ 1, 2, 3 };
    if (tree.findLongestPrefix(&query1)) |result| {
        try testing.expectEqual(entry1, result[0]);
        try testing.expectEqual(@as(usize, 3), result[1]);
    } else {
        try testing.expect(false);
    }
    
    // Find longer match
    const query2 = [_]u32{ 1, 2, 3, 4, 5, 6 };
    if (tree.findLongestPrefix(&query2)) |result| {
        try testing.expectEqual(entry2, result[0]);
        try testing.expectEqual(@as(usize, 5), result[1]);
    } else {
        try testing.expect(false);
    }
    
    // Find partial match
    const query3 = [_]u32{ 1, 2, 3, 4 };
    if (tree.findLongestPrefix(&query3)) |result| {
        // Should match entry1 (prefix [1,2,3])
        try testing.expectEqual(entry1, result[0]);
        try testing.expectEqual(@as(usize, 3), result[1]);
    } else {
        try testing.expect(false);
    }
    
    // No match
    const query4 = [_]u32{ 9, 8, 7 };
    try testing.expect(tree.findLongestPrefix(&query4) == null);
}

test "CacheSharingManager: basic operations" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store a shared entry
    const tokens = [_]u32{ 100, 200, 300 };
    const keys = [_]f32{ 1.0, 2.0, 3.0 };
    const values = [_]f32{ 4.0, 5.0, 6.0 };
    
    const id = try manager.storeSharedEntry("test-model", 0, &tokens, &keys, &values);
    try testing.expect(id > 0);
    
    // Verify stats
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 1), stats.shared_entries);
    try testing.expectEqual(@as(u64, 1), stats.total_entries);
}

test "CacheSharingManager: find and acquire" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store entry
    const tokens = [_]u32{ 10, 20, 30, 40 };
    const keys = [_]f32{ 1.0, 2.0 };
    const values = [_]f32{ 3.0, 4.0 };
    
    _ = try manager.storeSharedEntry("model1", 0, &tokens, &keys, &values);
    
    // Find exact match
    const query1 = [_]u32{ 10, 20, 30, 40 };
    if (manager.findSharedEntry(&query1)) |result| {
        const entry = result[0];
        const match_len = result[1];
        defer manager.releaseSharedEntry(entry);
        
        try testing.expectEqual(@as(usize, 4), match_len);
        try testing.expectEqual(@as(u32, 1), entry.getRefCount());
    } else {
        try testing.expect(false);
    }
    
    // Find prefix match
    const query2 = [_]u32{ 10, 20, 30, 40, 50, 60 };
    if (manager.findSharedEntry(&query2)) |result| {
        const entry = result[0];
        const match_len = result[1];
        defer manager.releaseSharedEntry(entry);
        
        try testing.expectEqual(@as(usize, 4), match_len);
    } else {
        try testing.expect(false);
    }
    
    // Verify stats
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 2), stats.shared_cache_hits);
    try testing.expectEqual(@as(u64, 2), stats.prefix_matches);
}

test "CacheSharingManager: concurrent access" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store entry
    const tokens = [_]u32{ 1, 2, 3 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    _ = try manager.storeSharedEntry("model", 0, &tokens, &keys, &values);
    
    // Acquire multiple times (simulating concurrent requests)
    const query = [_]u32{ 1, 2, 3 };
    
    const result1 = manager.findSharedEntry(&query).?;
    const entry1 = result1[0];
    
    const result2 = manager.findSharedEntry(&query).?;
    const entry2 = result2[0];
    
    const result3 = manager.findSharedEntry(&query).?;
    const entry3 = result3[0];
    
    // All should point to same entry
    try testing.expectEqual(entry1, entry2);
    try testing.expectEqual(entry2, entry3);
    try testing.expectEqual(@as(u32, 3), entry1.getRefCount());
    
    // Release all
    manager.releaseSharedEntry(entry1);
    manager.releaseSharedEntry(entry2);
    manager.releaseSharedEntry(entry3);
    
    try testing.expectEqual(@as(u32, 0), entry1.getRefCount());
}

test "CacheSharingManager: prefix too short" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Try to store entry shorter than min_prefix_length
    const tokens = [_]u32{1}; // Only 1 token, min is 2
    const keys = [_]f32{1.0};
    const values = [_]f32{2.0};
    
    const result = manager.storeSharedEntry("model", 0, &tokens, &keys, &values);
    try testing.expectError(error.PrefixTooShort, result);
}

test "CacheSharingManager: eviction on size limit" {
    const allocator = testing.allocator;
    var config = getTestConfig();
    config.max_shared_cache_size = 1000; // Very small limit
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store first entry
    const tokens1 = [_]u32{ 1, 2, 3 };
    const keys1 = [_]f32{ 1.0, 2.0 };
    const values1 = [_]f32{ 3.0, 4.0 };
    
    _ = try manager.storeSharedEntry("model1", 0, &tokens1, &keys1, &values1);
    
    // Sleep briefly to ensure different timestamp
    std.time.sleep(10 * std.time.ns_per_ms);
    
    // Store second entry (should trigger eviction of first)
    const tokens2 = [_]u32{ 4, 5, 6 };
    const keys2 = [_]f32{ 5.0, 6.0 };
    const values2 = [_]f32{ 7.0, 8.0 };
    
    _ = try manager.storeSharedEntry("model2", 0, &tokens2, &keys2, &values2);
    
    // First entry should be evicted, second should remain
    const query1 = [_]u32{ 1, 2, 3 };
    try testing.expect(manager.findSharedEntry(&query1) == null);
    
    const query2 = [_]u32{ 4, 5, 6 };
    try testing.expect(manager.findSharedEntry(&query2) != null);
}

test "CacheSharingManager: protected entries not evicted" {
    const allocator = testing.allocator;
    var config = getTestConfig();
    config.max_shared_cache_size = 1000;
    config.protect_shared_entries = true;
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store and acquire first entry
    const tokens1 = [_]u32{ 1, 2, 3 };
    const keys1 = [_]f32{ 1.0, 2.0 };
    const values1 = [_]f32{ 3.0, 4.0 };
    
    _ = try manager.storeSharedEntry("model1", 0, &tokens1, &keys1, &values1);
    
    const query1 = [_]u32{ 1, 2, 3 };
    const result1 = manager.findSharedEntry(&query1).?;
    const entry1 = result1[0];
    // Don't release - keep ref count > 0
    
    // Try to store second entry
    const tokens2 = [_]u32{ 4, 5, 6 };
    const keys2 = [_]f32{ 5.0, 6.0 };
    const values2 = [_]f32{ 7.0, 8.0 };
    
    _ = try manager.storeSharedEntry("model2", 0, &tokens2, &keys2, &values2);
    
    // First entry should still be there (protected by refcount)
    if (manager.findSharedEntry(&query1)) |result| {
        const entry = result[0];
        defer manager.releaseSharedEntry(entry);
        try testing.expectEqual(entry1, entry);
    } else {
        try testing.expect(false);
    }
    
    // Clean up
    manager.releaseSharedEntry(entry1);
}

test "CacheSharingManager: statistics" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store entries
    const tokens1 = [_]u32{ 1, 2, 3 };
    const tokens2 = [_]u32{ 4, 5, 6 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    _ = try manager.storeSharedEntry("model1", 0, &tokens1, &keys, &values);
    _ = try manager.storeSharedEntry("model2", 0, &tokens2, &keys, &values);
    
    // Find first entry multiple times
    const query1 = [_]u32{ 1, 2, 3 };
    for (0..5) |_| {
        if (manager.findSharedEntry(&query1)) |result| {
            manager.releaseSharedEntry(result[0]);
        }
    }
    
    // Find second entry once
    const query2 = [_]u32{ 4, 5, 6 };
    if (manager.findSharedEntry(&query2)) |result| {
        manager.releaseSharedEntry(result[0]);
    }
    
    // Miss
    const query3 = [_]u32{ 9, 9, 9 };
    _ = manager.findSharedEntry(&query3);
    
    // Check stats
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 2), stats.shared_entries);
    try testing.expectEqual(@as(u64, 6), stats.shared_cache_hits);
    try testing.expectEqual(@as(u64, 1), stats.shared_cache_misses);
    try testing.expectEqual(@as(u64, 6), stats.full_prefix_reuse);
    
    // Check hit rate
    const hit_rate = stats.getSharedHitRate();
    try testing.expect(hit_rate > 0.85); // 6/7 ≈ 85.7%
}

test "CacheSharingManager: full vs partial prefix reuse" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store entry
    const tokens = [_]u32{ 10, 20, 30 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    _ = try manager.storeSharedEntry("model", 0, &tokens, &keys, &values);
    
    // Full reuse (exact match)
    const query_full = [_]u32{ 10, 20, 30 };
    if (manager.findSharedEntry(&query_full)) |result| {
        manager.releaseSharedEntry(result[0]);
    }
    
    // Partial reuse (prefix match)
    const query_partial = [_]u32{ 10, 20, 30, 40, 50 };
    if (manager.findSharedEntry(&query_partial)) |result| {
        manager.releaseSharedEntry(result[0]);
    }
    
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 1), stats.full_prefix_reuse);
    try testing.expectEqual(@as(u64, 1), stats.partial_prefix_reuse);
}

test "CacheSharingManager: multiple layers" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    const tokens = [_]u32{ 1, 2, 3 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    // Store same tokens for different layers
    _ = try manager.storeSharedEntry("model", 0, &tokens, &keys, &values);
    _ = try manager.storeSharedEntry("model", 1, &tokens, &keys, &values);
    _ = try manager.storeSharedEntry("model", 2, &tokens, &keys, &values);
    
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 3), stats.shared_entries);
}

test "CacheSharingManager: disabled sharing" {
    const allocator = testing.allocator;
    var config = getTestConfig();
    config.enabled = false;
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    const tokens = [_]u32{ 1, 2, 3 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    // Should fail when disabled
    const result = manager.storeSharedEntry("model", 0, &tokens, &keys, &values);
    try testing.expectError(error.SharingDisabled, result);
    
    // Find should return null
    const query = [_]u32{ 1, 2, 3 };
    try testing.expect(manager.findSharedEntry(&query) == null);
}

test "PrefixTree: deep nesting" {
    const allocator = testing.allocator;
    
    const tree = try sharing.PrefixTree.init(allocator);
    defer tree.deinit();
    
    // Create progressively longer sequences
    var tokens_buf: [20]u32 = undefined;
    for (0..20) |i| {
        tokens_buf[i] = @intCast(i);
    }
    
    const keys = [_]f32{1.0};
    const values = [_]f32{2.0};
    
    // Insert multiple entries with increasing length
    for (1..11) |len| {
        const entry = try sharing.SharedCacheEntry.init(
            allocator,
            @intCast(len),
            "model",
            0,
            tokens_buf[0..len],
            &keys,
            &values,
        );
        defer entry.deinit(allocator);
        
        try tree.insert(tokens_buf[0..len], entry);
    }
    
    // Find longest match for full sequence
    if (tree.findLongestPrefix(tokens_buf[0..15])) |result| {
        try testing.expectEqual(@as(usize, 10), result[1]);
    } else {
        try testing.expect(false);
    }
}

test "CacheSharingManager: hash collision handling" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const manager = try sharing.CacheSharingManager.init(allocator, config);
    defer manager.deinit();
    
    // Store first entry
    const tokens1 = [_]u32{ 1, 2, 3 };
    const keys1 = [_]f32{ 1.0 };
    const values1 = [_]f32{ 2.0 };
    
    const id1 = try manager.storeSharedEntry("model1", 0, &tokens1, &keys1, &values1);
    
    // Try to store same tokens again (should detect existing entry)
    const id2 = try manager.storeSharedEntry("model1", 0, &tokens1, &keys1, &values1);
    
    try testing.expectEqual(id1, id2);
    
    const stats = manager.getStats();
    try testing.expectEqual(@as(u64, 1), stats.shared_entries); // Only one entry
}

test "CacheReplicationManager: initialization" {
    const allocator = testing.allocator;
    const config = getTestConfig();
    
    const repl = try sharing.CacheReplicationManager.init(allocator, config, "node-1");
    defer repl.deinit();
    
    try testing.expectEqualStrings("node-1", repl.node_id);
}

// ============================================================================
// Performance Tests
// ============================================================================

test "Performance: prefix tree lookup speed" {
    const allocator = testing.allocator;
    
    const tree = try sharing.PrefixTree.init(allocator);
    defer tree.deinit();
    
    // Insert 100 entries
    const keys = [_]f32{1.0};
    const values = [_]f32{2.0};
    
    var tokens_buf: [10]u32 = undefined;
    for (0..100) |i| {
        for (0..10) |j| {
            tokens_buf[j] = @intCast((i * 10 + j) % 1000);
        }
        
        const entry = try sharing.SharedCacheEntry.init(
            allocator,
            @intCast(i),
            "model",
            0,
            &tokens_buf,
            &keys,
            &values,
        );
        defer entry.deinit(allocator);
        
        try tree.insert(&tokens_buf, entry);
    }
    
    // Benchmark lookups
    const start = std.time.microTimestamp();
    const iterations = 10000;
    
    for (0..iterations) |i| {
        for (0..10) |j| {
            tokens_buf[j] = @intCast(((i % 100) * 10 + j) % 1000);
        }
        _ = tree.findLongestPrefix(&tokens_buf);
    }
    
    const elapsed = std.time.microTimestamp() - start;
    const avg_us = @as(f64, @floatFromInt(elapsed)) / @as(f64, @floatFromInt(iterations));
    
    std.debug.print("\nPrefix tree lookup: {d:.2}μs avg ({d} iterations)\n", .{ avg_us, iterations });
    
    // Should be very fast (< 5μs avg)
    try testing.expect(avg_us < 5.0);
}

test "Performance: reference counting overhead" {
    const allocator = testing.allocator;
    
    const tokens = [_]u32{ 1, 2, 3 };
    const keys = [_]f32{ 1.0 };
    const values = [_]f32{ 2.0 };
    
    const entry = try sharing.SharedCacheEntry.init(
        allocator,
        1,
        "model",
        0,
        &tokens,
        &keys,
        &values,
    );
    defer entry.deinit(allocator);
    
    const iterations = 1000000;
    const start = std.time.microTimestamp();
    
    for (0..iterations) |_| {
        entry.acquire();
        entry.release();
    }
    
    const elapsed = std.time.microTimestamp() - start;
    const avg_ns = (@as(f64, @floatFromInt(elapsed)) * 1000.0) / @as(f64, @floatFromInt(iterations));
    
    std.debug.print("\nReference counting: {d:.2}ns avg ({d} iterations)\n", .{ avg_ns, iterations });
    
    // Should be very fast (< 100ns)
    try testing.expect(avg_ns < 100.0);
}
