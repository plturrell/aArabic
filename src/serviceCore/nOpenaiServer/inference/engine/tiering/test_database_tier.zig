// Test Suite for Database-Backed KV Cache Tier
// Validates multi-database integration and functionality

const std = @import("std");
const db_tier = @import("database_tier.zig");
const testing = std.testing;

// ============================================================================
// Test Helpers
// ============================================================================

fn createTestData(allocator: std.mem.Allocator, size: usize) ![]f32 {
    var data = try allocator.alloc(f32, size);
    var prng = std.rand.DefaultPrng.init(42);
    const random = prng.random();
    
    for (data) |*val| {
        val.* = random.float(f32) * 2.0 - 1.0;
    }
    
    return data;
}

// ============================================================================
// Configuration Tests
// ============================================================================

test "database_tier_config: default values" {
    const config = db_tier.DatabaseTierConfig{};
    
    try testing.expect(config.enabled == true);
    try testing.expectEqualStrings(config.dragonfly_host, "localhost");
    try testing.expect(config.dragonfly_port == 6379);
    try testing.expect(config.dragonfly_ttl_seconds == 3600);
    
    try testing.expectEqualStrings(config.postgres_host, "localhost");
    try testing.expect(config.postgres_port == 5432);
    try testing.expectEqualStrings(config.postgres_database, "kv_cache");
    
    try testing.expectEqualStrings(config.qdrant_host, "localhost");
    try testing.expect(config.qdrant_port == 6333);
    
    try testing.expect(config.use_compression == true);
    try testing.expect(config.connection_pool_size == 10);
}

test "database_tier_config: custom values" {
    const config = db_tier.DatabaseTierConfig{
        .dragonfly_host = "redis.example.com",
        .dragonfly_port = 6380,
        .postgres_database = "custom_cache",
        .use_compression = false,
    };
    
    try testing.expectEqualStrings(config.dragonfly_host, "redis.example.com");
    try testing.expect(config.dragonfly_port == 6380);
    try testing.expectEqualStrings(config.postgres_database, "custom_cache");
    try testing.expect(config.use_compression == false);
}

// ============================================================================
// Statistics Tests
// ============================================================================

test "database_tier_stats: initialization" {
    const stats = db_tier.DatabaseTierStats{};
    
    try testing.expect(stats.dragonfly_hits == 0);
    try testing.expect(stats.dragonfly_misses == 0);
    try testing.expect(stats.postgres_reads == 0);
    try testing.expect(stats.qdrant_upserts == 0);
}

test "database_tier_stats: hit rate calculation" {
    var stats = db_tier.DatabaseTierStats{
        .dragonfly_hits = 80,
        .dragonfly_misses = 20,
    };
    
    const hit_rate = stats.getDragonflyHitRate();
    try testing.expect(@abs(hit_rate - 0.8) < 0.001);
}

test "database_tier_stats: hit rate edge cases" {
    var stats = db_tier.DatabaseTierStats{};
    
    // No accesses
    try testing.expect(stats.getDragonflyHitRate() == 0.0);
    
    // All hits
    stats.dragonfly_hits = 100;
    try testing.expect(stats.getDragonflyHitRate() == 1.0);
    
    // All misses
    stats.dragonfly_hits = 0;
    stats.dragonfly_misses = 100;
    try testing.expect(stats.getDragonflyHitRate() == 0.0);
}

test "database_tier_stats: qdrant precision" {
    var stats = db_tier.DatabaseTierStats{
        .qdrant_searches = 100,
        .qdrant_hits = 75,
    };
    
    const precision = stats.getQdrantPrecision();
    try testing.expect(@abs(precision - 0.75) < 0.001);
}

// ============================================================================
// Metadata Tests
// ============================================================================

test "kv_cache_metadata: initialization" {
    const metadata = db_tier.KVCacheMetadata.init("test-model", 0, 0, 128);
    
    try testing.expectEqualStrings(metadata.model_id, "test-model");
    try testing.expect(metadata.layer == 0);
    try testing.expect(metadata.token_start == 0);
    try testing.expect(metadata.token_end == 128);
    try testing.expect(metadata.version == 1);
    try testing.expect(metadata.access_count == 0);
}

test "kv_cache_metadata: timestamps" {
    const metadata = db_tier.KVCacheMetadata.init("test-model", 0, 0, 128);
    
    try testing.expect(metadata.created_at > 0);
    try testing.expect(metadata.accessed_at > 0);
    try testing.expect(metadata.created_at == metadata.accessed_at);
}

// ============================================================================
// DragonflyDB Client Tests
// ============================================================================

test "dragonfly_client: initialization" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    try testing.expect(client.connected == false);
}

test "dragonfly_client: connect and disconnect" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    try client.connect();
    try testing.expect(client.connected == true);
    
    client.disconnect();
    try testing.expect(client.connected == false);
}

test "dragonfly_client: get without connection" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    const result = client.get("test-key");
    try testing.expectError(error.NotConnected, result);
}

test "dragonfly_client: set without connection" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    const result = client.set("test-key", "test-value", null);
    try testing.expectError(error.NotConnected, result);
}

test "dragonfly_client: get returns null for cache miss" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.DragonflyClient.init(allocator, config);
    defer client.deinit();
    
    try client.connect();
    
    const result = try client.get("nonexistent-key");
    try testing.expect(result == null);
}

// ============================================================================
// PostgreSQL Client Tests
// ============================================================================

test "postgres_client: initialization" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.PostgresClient.init(allocator, config);
    defer client.deinit();
    
    try testing.expect(client.connected == false);
}

test "postgres_client: connect and disconnect" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.PostgresClient.init(allocator, config);
    defer client.deinit();
    
    try client.connect();
    try testing.expect(client.connected == true);
    
    client.disconnect();
    try testing.expect(client.connected == false);
}

test "postgres_client: operations without connection" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.PostgresClient.init(allocator, config);
    defer client.deinit();
    
    // Create schema
    try testing.expectError(error.NotConnected, client.createSchema());
    
    // Insert metadata
    const metadata = db_tier.KVCacheMetadata.init("test", 0, 0, 128);
    try testing.expectError(error.NotConnected, client.insertMetadata(&metadata));
}

// ============================================================================
// Qdrant Client Tests
// ============================================================================

test "qdrant_client: initialization" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.QdrantClient.init(allocator, config);
    defer client.deinit();
    
    try testing.expect(client.connected == false);
}

test "qdrant_client: connect and disconnect" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.QdrantClient.init(allocator, config);
    defer client.deinit();
    
    try client.connect();
    try testing.expect(client.connected == true);
    
    client.disconnect();
    try testing.expect(client.connected == false);
}

test "qdrant_client: operations without connection" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const client = try db_tier.QdrantClient.init(allocator, config);
    defer client.deinit();
    
    // Create collection
    try testing.expectError(error.NotConnected, client.createCollection(512));
    
    // Upsert vector
    const vector = [_]f32{0.1, 0.2, 0.3};
    try testing.expectError(error.NotConnected, 
        client.upsertVector("test-id", &vector, null));
}

// ============================================================================
// Database Tier Manager Tests
// ============================================================================

test "database_tier: initialization" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try testing.expect(tier.stats.dragonfly_hits == 0);
    try testing.expect(tier.stats.postgres_reads == 0);
}

test "database_tier: initialization with compression" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{
        .use_compression = true,
    };
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try testing.expect(tier.compression_mgr != null);
}

test "database_tier: initialization without compression" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{
        .use_compression = false,
    };
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try testing.expect(tier.compression_mgr == null);
}

test "database_tier: connect all databases" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    try testing.expect(tier.dragonfly.connected == true);
    try testing.expect(tier.postgres.connected == true);
    try testing.expect(tier.qdrant.connected == true);
}

test "database_tier: store updates statistics" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{
        .use_compression = false, // Simplify test
    };
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    // Store data
    try tier.store("test-model", 0, 0, keys, values);
    
    // Check statistics
    try testing.expect(tier.stats.dragonfly_sets == 1);
    try testing.expect(tier.stats.postgres_writes == 1);
    try testing.expect(tier.stats.qdrant_upserts == 1);
}

test "database_tier: load with cache miss" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    // Try to load non-existent data
    const result = try tier.load("test-model", 0, 0);
    try testing.expect(result == null);
    
    // Check miss recorded
    try testing.expect(tier.stats.dragonfly_misses == 1);
}

test "database_tier: get statistics" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    const stats = tier.getStats();
    try testing.expect(stats.dragonfly_hits == 0);
    try testing.expect(stats.postgres_reads == 0);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "integration: store and load cycle" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{
        .use_compression = true,
    };
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 256);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 256);
    defer allocator.free(values);
    
    // Store
    try tier.store("test-model", 0, 0, keys, values);
    
    // Load (will miss in placeholder implementation)
    const result = try tier.load("test-model", 0, 0);
    
    // Verify statistics
    try testing.expect(tier.stats.dragonfly_sets >= 1);
    try testing.expect(tier.stats.postgres_writes >= 1);
    
    // Note: In real implementation, result should contain data
    _ = result;
}

test "integration: multiple models" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    // Store for model 1
    try tier.store("model-1", 0, 0, keys, values);
    
    // Store for model 2
    try tier.store("model-2", 0, 0, keys, values);
    
    // Check statistics
    try testing.expect(tier.stats.dragonfly_sets == 2);
    try testing.expect(tier.stats.postgres_writes == 2);
}

test "integration: multiple layers" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    // Store multiple layers
    var layer: u32 = 0;
    while (layer < 5) : (layer += 1) {
        try tier.store("test-model", layer, 0, keys, values);
    }
    
    // Check statistics
    try testing.expect(tier.stats.dragonfly_sets == 5);
    try testing.expect(tier.stats.qdrant_upserts == 5);
}

// ============================================================================
// Performance Tests
// ============================================================================

test "performance: store throughput" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{
        .use_compression = true,
    };
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 256);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 256);
    defer allocator.free(values);
    
    const start = std.time.nanoTimestamp();
    
    // Perform 10 stores
    var i: usize = 0;
    while (i < 10) : (i += 1) {
        try tier.store("test-model", 0, @intCast(i * 256), keys, values);
    }
    
    const end = std.time.nanoTimestamp();
    const elapsed_us = @divTrunc(end - start, 1000);
    
    // Should complete in reasonable time (<1s for 10 operations)
    try testing.expect(elapsed_us < 1_000_000);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "edge_cases: empty data" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const empty_keys = [_]f32{};
    const empty_values = [_]f32{};
    
    // Should handle empty data gracefully
    try tier.store("test-model", 0, 0, &empty_keys, &empty_values);
}

test "edge_cases: large token range" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 10000);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 10000);
    defer allocator.free(values);
    
    // Should handle large ranges
    try tier.store("test-model", 0, 0, keys, values);
    
    try testing.expect(tier.stats.dragonfly_sets == 1);
}

test "edge_cases: model id with special characters" {
    const allocator = testing.allocator;
    const config = db_tier.DatabaseTierConfig{};
    
    const tier = try db_tier.DatabaseTier.init(allocator, config);
    defer tier.deinit();
    
    try tier.connect();
    
    const keys = try createTestData(allocator, 128);
    defer allocator.free(keys);
    const values = try createTestData(allocator, 128);
    defer allocator.free(values);
    
    // Model ID with special characters
    try tier.store("model:with:colons", 0, 0, keys, values);
    try tier.store("model/with/slashes", 0, 0, keys, values);
    
    try testing.expect(tier.stats.dragonfly_sets == 2);
}
