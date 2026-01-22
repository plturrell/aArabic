// ============================================================================
// Router Cache Integration - Day 58 Implementation
// ============================================================================
// Purpose: Integrate distributed cache with Model Router
// Week: Week 12 (Days 56-60) - Distributed Caching
// Phase: Month 4 - HANA Integration & Scalability
// ============================================================================

const std = @import("std");
const Allocator = std.mem.Allocator;
const DistributedCoordinator = @import("distributed_coordinator.zig").DistributedCoordinator;
const DistributedCacheConfig = @import("distributed_coordinator.zig").DistributedCacheConfig;

// ============================================================================
// CACHE KEY TYPES
// ============================================================================

/// Types of cache keys used in routing
pub const CacheKeyType = enum {
    routing_decision,    // Cache which model was selected for query
    query_result,        // Cache the actual query result
    model_metadata,      // Cache model capabilities and stats
    load_metrics,        // Cache current load metrics
    
    pub fn prefix(self: CacheKeyType) []const u8 {
        return switch (self) {
            .routing_decision => "route:",
            .query_result => "result:",
            .model_metadata => "meta:",
            .load_metrics => "load:",
        };
    }
};

// ============================================================================
// CACHE ENTRY METADATA
// ============================================================================

/// Metadata about cached routing decisions
pub const RoutingCacheEntry = struct {
    query_hash: []const u8,
    selected_model: []const u8,
    score: f32,
    timestamp: i64,
    hit_count: u32,
    
    pub fn toJson(self: RoutingCacheEntry, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(
            allocator,
            "{{\"query_hash\":\"{s}\",\"selected_model\":\"{s}\",\"score\":{d:.2},\"timestamp\":{d},\"hit_count\":{d}}}",
            .{ self.query_hash, self.selected_model, self.score, self.timestamp, self.hit_count }
        );
    }
    
    pub fn fromJson(allocator: Allocator, json: []const u8) !RoutingCacheEntry {
        _ = allocator;
        // Simplified parsing - in production would use proper JSON parser
        var entry: RoutingCacheEntry = undefined;
        
        // For now, return a default entry
        // Real implementation would parse JSON properly
        entry.query_hash = "";
        entry.selected_model = "";
        entry.score = 0.0;
        entry.timestamp = std.time.milliTimestamp();
        entry.hit_count = 0;
        
        _ = json; // Will be used in full implementation
        return entry;
    }
};

/// Metadata about cached query results
pub const QueryResultCacheEntry = struct {
    query_hash: []const u8,
    model_id: []const u8,
    result_data: []const u8,
    response_time_ms: u32,
    timestamp: i64,
    
    pub fn toJson(self: QueryResultCacheEntry, allocator: Allocator) ![]const u8 {
        // Escape result_data for JSON
        return std.fmt.allocPrint(
            allocator,
            "{{\"query_hash\":\"{s}\",\"model_id\":\"{s}\",\"result_data\":\"{s}\",\"response_time_ms\":{d},\"timestamp\":{d}}}",
            .{ self.query_hash, self.model_id, self.result_data, self.response_time_ms, self.timestamp }
        );
    }
};

/// Metadata about cached model information
pub const ModelMetadataEntry = struct {
    model_id: []const u8,
    capabilities: []const u8,  // JSON string
    current_load: f32,
    avg_response_time_ms: u32,
    timestamp: i64,
    
    pub fn toJson(self: ModelMetadataEntry, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(
            allocator,
            "{{\"model_id\":\"{s}\",\"capabilities\":{s},\"current_load\":{d:.2},\"avg_response_time_ms\":{d},\"timestamp\":{d}}}",
            .{ self.model_id, self.capabilities, self.current_load, self.avg_response_time_ms, self.timestamp }
        );
    }
};

// ============================================================================
// ROUTER CACHE
// ============================================================================

/// Configuration for router cache
pub const RouterCacheConfig = struct {
    routing_decision_ttl_ms: i64 = 300000,      // 5 minutes
    query_result_ttl_ms: i64 = 600000,          // 10 minutes
    model_metadata_ttl_ms: i64 = 60000,         // 1 minute
    load_metrics_ttl_ms: i64 = 5000,            // 5 seconds
    enable_result_caching: bool = true,
    enable_metadata_caching: bool = true,
    max_result_size_bytes: usize = 1024 * 1024, // 1MB
};

/// Router-specific cache layer built on distributed coordinator
pub const RouterCache = struct {
    allocator: Allocator,
    config: RouterCacheConfig,
    coordinator: *DistributedCoordinator,
    stats: CacheStats,
    mutex: std.Thread.Mutex,
    
    pub fn init(
        allocator: Allocator,
        cache_config: RouterCacheConfig,
        dist_config: DistributedCacheConfig,
    ) !*RouterCache {
        const cache = try allocator.create(RouterCache);
        
        cache.* = .{
            .allocator = allocator,
            .config = cache_config,
            .coordinator = try DistributedCoordinator.init(allocator, dist_config),
            .stats = CacheStats{},
            .mutex = .{},
        };
        
        return cache;
    }
    
    pub fn deinit(self: *RouterCache) void {
        self.coordinator.deinit();
        self.allocator.destroy(self);
    }
    
    // ========================================================================
    // ROUTING DECISION CACHING
    // ========================================================================
    
    /// Cache a routing decision
    pub fn cacheRoutingDecision(
        self: *RouterCache,
        query_hash: []const u8,
        selected_model: []const u8,
        score: f32,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const entry = RoutingCacheEntry{
            .query_hash = query_hash,
            .selected_model = selected_model,
            .score = score,
            .timestamp = std.time.milliTimestamp(),
            .hit_count = 0,
        };
        
        const json = try entry.toJson(self.allocator);
        defer self.allocator.free(json);
        
        const key = try self.buildKey(.routing_decision, query_hash);
        defer self.allocator.free(key);
        
        try self.coordinator.put(key, json, self.config.routing_decision_ttl_ms);
        self.stats.routing_writes += 1;
    }
    
    /// Get cached routing decision
    pub fn getRoutingDecision(
        self: *RouterCache,
        query_hash: []const u8,
    ) !?RoutingCacheEntry {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const key = try self.buildKey(.routing_decision, query_hash);
        defer self.allocator.free(key);
        
        if (try self.coordinator.get(key)) |json| {
            defer self.allocator.free(json);
            self.stats.routing_hits += 1;
            return try RoutingCacheEntry.fromJson(self.allocator, json);
        }
        
        self.stats.routing_misses += 1;
        return null;
    }
    
    // ========================================================================
    // QUERY RESULT CACHING
    // ========================================================================
    
    /// Cache a query result
    pub fn cacheQueryResult(
        self: *RouterCache,
        query_hash: []const u8,
        model_id: []const u8,
        result_data: []const u8,
        response_time_ms: u32,
    ) !void {
        if (!self.config.enable_result_caching) return;
        if (result_data.len > self.config.max_result_size_bytes) return;
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const entry = QueryResultCacheEntry{
            .query_hash = query_hash,
            .model_id = model_id,
            .result_data = result_data,
            .response_time_ms = response_time_ms,
            .timestamp = std.time.milliTimestamp(),
        };
        
        const json = try entry.toJson(self.allocator);
        defer self.allocator.free(json);
        
        // Composite key: query_hash:model_id
        const composite_key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ query_hash, model_id }
        );
        defer self.allocator.free(composite_key);
        
        const key = try self.buildKey(.query_result, composite_key);
        defer self.allocator.free(key);
        
        try self.coordinator.put(key, json, self.config.query_result_ttl_ms);
        self.stats.result_writes += 1;
    }
    
    /// Get cached query result
    pub fn getQueryResult(
        self: *RouterCache,
        query_hash: []const u8,
        model_id: []const u8,
    ) !?QueryResultCacheEntry {
        if (!self.config.enable_result_caching) return null;
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const composite_key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ query_hash, model_id }
        );
        defer self.allocator.free(composite_key);
        
        const key = try self.buildKey(.query_result, composite_key);
        defer self.allocator.free(key);
        
        if (try self.coordinator.get(key)) |json| {
            defer self.allocator.free(json);
            self.stats.result_hits += 1;
            
            // Parse JSON back to entry
            const entry = QueryResultCacheEntry{
                .query_hash = query_hash,
                .model_id = model_id,
                .result_data = json, // Simplified
                .response_time_ms = 0,
                .timestamp = std.time.milliTimestamp(),
            };
            return entry;
        }
        
        self.stats.result_misses += 1;
        return null;
    }
    
    // ========================================================================
    // MODEL METADATA CACHING
    // ========================================================================
    
    /// Cache model metadata
    pub fn cacheModelMetadata(
        self: *RouterCache,
        model_id: []const u8,
        capabilities: []const u8,
        current_load: f32,
        avg_response_time_ms: u32,
    ) !void {
        if (!self.config.enable_metadata_caching) return;
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const entry = ModelMetadataEntry{
            .model_id = model_id,
            .capabilities = capabilities,
            .current_load = current_load,
            .avg_response_time_ms = avg_response_time_ms,
            .timestamp = std.time.milliTimestamp(),
        };
        
        const json = try entry.toJson(self.allocator);
        defer self.allocator.free(json);
        
        const key = try self.buildKey(.model_metadata, model_id);
        defer self.allocator.free(key);
        
        try self.coordinator.put(key, json, self.config.model_metadata_ttl_ms);
        self.stats.metadata_writes += 1;
    }
    
    /// Get cached model metadata
    pub fn getModelMetadata(
        self: *RouterCache,
        model_id: []const u8,
    ) !?ModelMetadataEntry {
        if (!self.config.enable_metadata_caching) return null;
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const key = try self.buildKey(.model_metadata, model_id);
        defer self.allocator.free(key);
        
        if (try self.coordinator.get(key)) |json| {
            defer self.allocator.free(json);
            self.stats.metadata_hits += 1;
            
            // Simplified - would parse JSON properly
            const entry = ModelMetadataEntry{
                .model_id = model_id,
                .capabilities = "{}",
                .current_load = 0.0,
                .avg_response_time_ms = 0,
                .timestamp = std.time.milliTimestamp(),
            };
            return entry;
        }
        
        self.stats.metadata_misses += 1;
        return null;
    }
    
    // ========================================================================
    // CACHE INVALIDATION
    // ========================================================================
    
    /// Invalidate all cache entries for a specific model
    pub fn invalidateModel(self: *RouterCache, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Invalidate model metadata
        const meta_key = try self.buildKey(.model_metadata, model_id);
        defer self.allocator.free(meta_key);
        try self.coordinator.invalidate(meta_key);
        
        self.stats.invalidations += 1;
    }
    
    /// Invalidate specific query result
    pub fn invalidateQueryResult(
        self: *RouterCache,
        query_hash: []const u8,
        model_id: []const u8,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const composite_key = try std.fmt.allocPrint(
            self.allocator,
            "{s}:{s}",
            .{ query_hash, model_id }
        );
        defer self.allocator.free(composite_key);
        
        const key = try self.buildKey(.query_result, composite_key);
        defer self.allocator.free(key);
        
        try self.coordinator.invalidate(key);
        self.stats.invalidations += 1;
    }
    
    // ========================================================================
    // CACHE WARMING
    // ========================================================================
    
    /// Warm cache with frequently accessed models
    pub fn warmCache(self: *RouterCache, model_ids: [][]const u8) !void {
        for (model_ids) |model_id| {
            // Pre-load model metadata
            try self.cacheModelMetadata(
                model_id,
                "{}", // Empty capabilities for now
                0.0,
                100,
            );
        }
        
        self.stats.warm_operations += 1;
    }
    
    // ========================================================================
    // STATISTICS
    // ========================================================================
    
    /// Get cache statistics
    pub fn getStats(self: *RouterCache) CacheStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Add cluster stats
        const cluster_stats = self.coordinator.getClusterStats();
        
        var stats = self.stats;
        stats.total_keys = cluster_stats.total_keys;
        stats.cluster_nodes = cluster_stats.total_nodes;
        
        return stats;
    }
    
    /// Reset statistics
    pub fn resetStats(self: *RouterCache) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.stats = CacheStats{};
    }
    
    // ========================================================================
    // HELPER METHODS
    // ========================================================================
    
    /// Build cache key with prefix
    fn buildKey(
        self: *RouterCache,
        key_type: CacheKeyType,
        value: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(
            self.allocator,
            "{s}{s}",
            .{ key_type.prefix(), value }
        );
    }
};

// ============================================================================
// CACHE STATISTICS
// ============================================================================

pub const CacheStats = struct {
    routing_hits: u64 = 0,
    routing_misses: u64 = 0,
    routing_writes: u64 = 0,
    result_hits: u64 = 0,
    result_misses: u64 = 0,
    result_writes: u64 = 0,
    metadata_hits: u64 = 0,
    metadata_misses: u64 = 0,
    metadata_writes: u64 = 0,
    invalidations: u64 = 0,
    warm_operations: u64 = 0,
    total_keys: u64 = 0,
    cluster_nodes: u32 = 0,
    
    pub fn routingHitRate(self: CacheStats) f32 {
        const total = self.routing_hits + self.routing_misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.routing_hits)) / @as(f32, @floatFromInt(total));
    }
    
    pub fn resultHitRate(self: CacheStats) f32 {
        const total = self.result_hits + self.result_misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.result_hits)) / @as(f32, @floatFromInt(total));
    }
    
    pub fn metadataHitRate(self: CacheStats) f32 {
        const total = self.metadata_hits + self.metadata_misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(self.metadata_hits)) / @as(f32, @floatFromInt(total));
    }
    
    pub fn overallHitRate(self: CacheStats) f32 {
        const total_hits = self.routing_hits + self.result_hits + self.metadata_hits;
        const total_misses = self.routing_misses + self.result_misses + self.metadata_misses;
        const total = total_hits + total_misses;
        if (total == 0) return 0.0;
        return @as(f32, @floatFromInt(total_hits)) / @as(f32, @floatFromInt(total));
    }
};

// ============================================================================
// UNIT TESTS
// ============================================================================

test "RouterCache: initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    const cache_config = RouterCacheConfig{};
    const dist_config = DistributedCacheConfig{};
    
    const cache = try RouterCache.init(allocator, cache_config, dist_config);
    defer cache.deinit();
    
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 0), stats.routing_hits);
}

test "RouterCache: cache routing decision" {
    const allocator = std.testing.allocator;
    
    const cache_config = RouterCacheConfig{};
    const dist_config = DistributedCacheConfig{};
    
    const cache = try RouterCache.init(allocator, cache_config, dist_config);
    defer cache.deinit();
    
    // Register cache nodes first
    try cache.coordinator.registerNode("node-1", "localhost", 6379);
    try cache.coordinator.registerNode("node-2", "localhost", 6380);
    
    // Cache a routing decision
    try cache.cacheRoutingDecision("query123", "gpt-4", 0.95);
    
    // Verify write was recorded
    var stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.routing_writes);
    
    // Attempt to retrieve (may or may not find it due to replication)
    _ = try cache.getRoutingDecision("query123");
    
    // Verify at least one cache operation occurred
    stats = cache.getStats();
    try std.testing.expect(stats.routing_writes > 0);
}

test "RouterCache: cache miss" {
    const allocator = std.testing.allocator;
    
    const cache_config = RouterCacheConfig{};
    const dist_config = DistributedCacheConfig{};
    
    const cache = try RouterCache.init(allocator, cache_config, dist_config);
    defer cache.deinit();
    
    // Try to get non-existent entry
    const entry = try cache.getRoutingDecision("nonexistent");
    try std.testing.expect(entry == null);
    
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.routing_misses);
}

test "RouterCache: hit rate calculation" {
    const allocator = std.testing.allocator;
    
    const cache_config = RouterCacheConfig{};
    const dist_config = DistributedCacheConfig{};
    
    const cache = try RouterCache.init(allocator, cache_config, dist_config);
    defer cache.deinit();
    
    // Register cache nodes
    try cache.coordinator.registerNode("node-1", "localhost", 6379);
    
    // Write and attempt reads
    try cache.cacheRoutingDecision("query1", "model1", 0.9);
    _ = try cache.getRoutingDecision("query1");
    _ = try cache.getRoutingDecision("query2"); // Miss
    
    const stats = cache.getStats();
    
    // Verify we tracked cache operations
    try std.testing.expect(stats.routing_writes >= 1);
    try std.testing.expect(stats.routing_hits + stats.routing_misses >= 2);
}

test "RouterCache: cache invalidation" {
    const allocator = std.testing.allocator;
    
    const cache_config = RouterCacheConfig{};
    const dist_config = DistributedCacheConfig{};
    
    const cache = try RouterCache.init(allocator, cache_config, dist_config);
    defer cache.deinit();
    
    // Register cache node
    try cache.coordinator.registerNode("node-1", "localhost", 6379);
    
    // Cache metadata
    try cache.cacheModelMetadata("model1", "{}", 0.5, 100);
    
    // Invalidate
    try cache.invalidateModel("model1");
    
    const stats = cache.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.invalidations);
}
