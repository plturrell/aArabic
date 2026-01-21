// Eviction Policy Implementation
// Advanced eviction strategies for GPU KV cache management

const std = @import("std");
const GpuCacheEntry = @import("gpu_kv_cache.zig").GpuCacheEntry;
const GpuCacheConfig = @import("gpu_cache_config.zig").GpuCacheConfig;
const EvictionPolicy = @import("gpu_cache_config.zig").EvictionPolicy;

// ============================================================================
// Eviction Strategy
// ============================================================================

/// Eviction strategy selector and executor
pub const EvictionStrategy = struct {
    policy: EvictionPolicy,
    
    pub fn init(policy: EvictionPolicy) EvictionStrategy {
        return .{ .policy = policy };
    }
    
    /// Select entry to evict from cache
    pub fn selectVictim(
        self: EvictionStrategy,
        entries: []GpuCacheEntry,
    ) ?usize {
        if (entries.len == 0) return null;
        
        return switch (self.policy) {
            .lru => self.selectLRU(entries),
            .lfu => self.selectLFU(entries),
            .fifo => self.selectFIFO(entries),
            .size_based => self.selectLargest(entries),
        };
    }
    
    // ========================================================================
    // LRU (Least Recently Used)
    // ========================================================================
    
    fn selectLRU(self: EvictionStrategy, entries: []GpuCacheEntry) usize {
        _ = self;
        var oldest_idx: usize = 0;
        var oldest_time = entries[0].last_access;
        
        for (entries, 0..) |entry, i| {
            if (entry.last_access < oldest_time) {
                oldest_time = entry.last_access;
                oldest_idx = i;
            }
        }
        
        return oldest_idx;
    }
    
    // ========================================================================
    // LFU (Least Frequently Used)
    // ========================================================================
    
    fn selectLFU(self: EvictionStrategy, entries: []GpuCacheEntry) usize {
        _ = self;
        var lowest_idx: usize = 0;
        var lowest_count = entries[0].access_count;
        
        for (entries, 0..) |entry, i| {
            if (entry.access_count < lowest_count) {
                lowest_count = entry.access_count;
                lowest_idx = i;
            }
        }
        
        return lowest_idx;
    }
    
    // ========================================================================
    // FIFO (First In First Out)
    // ========================================================================
    
    fn selectFIFO(self: EvictionStrategy, entries: []GpuCacheEntry) usize {
        _ = self;
        _ = entries;
        // Always evict the first entry (oldest insertion)
        return 0;
    }
    
    // ========================================================================
    // Size-Based
    // ========================================================================
    
    fn selectLargest(self: EvictionStrategy, entries: []GpuCacheEntry) usize {
        _ = self;
        var largest_idx: usize = 0;
        var largest_size = entries[0].getMemoryUsage();
        
        for (entries, 0..) |entry, i| {
            const size = entry.getMemoryUsage();
            if (size > largest_size) {
                largest_size = size;
                largest_idx = i;
            }
        }
        
        return largest_idx;
    }
    
    // ========================================================================
    // Advanced Policies
    // ========================================================================
    
    /// LRU with frequency threshold (combines LRU and LFU)
    pub fn selectLRUWithFrequency(
        self: EvictionStrategy,
        entries: []GpuCacheEntry,
        min_frequency: u32,
    ) ?usize {
        _ = self;
        
        // First, find entries below frequency threshold
        var candidates = std.ArrayList(usize).init(std.heap.page_allocator);
        defer candidates.deinit();
        
        for (entries, 0..) |entry, i| {
            if (entry.access_count < min_frequency) {
                candidates.append(i) catch continue;
            }
        }
        
        // If no low-frequency entries, fall back to standard LRU
        if (candidates.items.len == 0) {
            return self.selectLRU(entries);
        }
        
        // Among low-frequency entries, find the least recently used
        var oldest_idx = candidates.items[0];
        var oldest_time = entries[oldest_idx].last_access;
        
        for (candidates.items) |idx| {
            if (entries[idx].last_access < oldest_time) {
                oldest_time = entries[idx].last_access;
                oldest_idx = idx;
            }
        }
        
        return oldest_idx;
    }
    
    /// Adaptive eviction based on cache state
    pub fn selectAdaptive(
        self: EvictionStrategy,
        entries: []GpuCacheEntry,
        cache_utilization: f32,
    ) ?usize {
        // High utilization: prioritize space (evict largest)
        if (cache_utilization > 0.9) {
            return self.selectLargest(entries);
        }
        
        // Medium utilization: balance recency and frequency
        if (cache_utilization > 0.7) {
            return self.selectLRUWithFrequency(entries, 5);
        }
        
        // Low utilization: standard LRU
        return self.selectLRU(entries);
    }
};

// ============================================================================
// Eviction Scorer
// ============================================================================

/// Score entries for eviction decisions
pub const EvictionScorer = struct {
    /// Calculate eviction score for an entry (higher = more likely to evict)
    pub fn scoreEntry(entry: *const GpuCacheEntry, current_time: i64) f64 {
        const recency_weight = 0.4;
        const frequency_weight = 0.3;
        const size_weight = 0.3;
        
        // Recency score (older = higher score)
        const age = @as(f64, @floatFromInt(current_time - entry.last_access));
        const recency_score = age / 3600.0; // Normalize by hour
        
        // Frequency score (less accessed = higher score)
        const frequency_score = 1.0 / (@as(f64, @floatFromInt(entry.access_count)) + 1.0);
        
        // Size score (larger = higher score)
        const size_mb = @as(f64, @floatFromInt(entry.getMemoryUsage())) / (1024.0 * 1024.0);
        const size_score = size_mb / 100.0; // Normalize by 100MB
        
        return recency_weight * recency_score +
               frequency_weight * frequency_score +
               size_weight * size_score;
    }
    
    /// Find entry with highest eviction score
    pub fn findHighestScore(entries: []GpuCacheEntry) ?usize {
        if (entries.len == 0) return null;
        
        const current_time = std.time.timestamp();
        var highest_idx: usize = 0;
        var highest_score = scoreEntry(&entries[0], current_time);
        
        for (entries, 0..) |*entry, i| {
            const score = scoreEntry(entry, current_time);
            if (score > highest_score) {
                highest_score = score;
                highest_idx = i;
            }
        }
        
        return highest_idx;
    }
};

// ============================================================================
// Eviction Statistics
// ============================================================================

/// Track eviction statistics for analysis
pub const EvictionStats = struct {
    total_evictions: u64,
    lru_evictions: u64,
    lfu_evictions: u64,
    fifo_evictions: u64,
    size_evictions: u64,
    
    pub fn init() EvictionStats {
        return .{
            .total_evictions = 0,
            .lru_evictions = 0,
            .lfu_evictions = 0,
            .fifo_evictions = 0,
            .size_evictions = 0,
        };
    }
    
    pub fn recordEviction(self: *EvictionStats, policy: EvictionPolicy) void {
        self.total_evictions += 1;
        switch (policy) {
            .lru => self.lru_evictions += 1,
            .lfu => self.lfu_evictions += 1,
            .fifo => self.fifo_evictions += 1,
            .size_based => self.size_evictions += 1,
        }
    }
    
    pub fn print(self: *const EvictionStats) void {
        std.debug.print("\n🔄 Eviction Statistics\n", .{});
        std.debug.print("   Total: {}\n", .{self.total_evictions});
        std.debug.print("   LRU: {}\n", .{self.lru_evictions});
        std.debug.print("   LFU: {}\n", .{self.lfu_evictions});
        std.debug.print("   FIFO: {}\n", .{self.fifo_evictions});
        std.debug.print("   Size-based: {}\n", .{self.size_evictions});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "eviction_policy: LRU selection" {
    const strategy = EvictionStrategy.init(.lru);
    
    var entries: [3]GpuCacheEntry = undefined;
    for (&entries, 0..) |*entry, i| {
        entry.* = GpuCacheEntry{
            .keys = undefined,
            .values = undefined,
            .layer_id = @intCast(i),
            .batch_id = 0,
            .sequence_length = 1,
            .last_access = @intCast(1000 + i * 100),
            .access_count = 1,
            .stream = null,
        };
    }
    
    const victim = strategy.selectVictim(&entries);
    try std.testing.expect(victim.? == 0); // Oldest access
    
    std.debug.print("✓ LRU selection working\n", .{});
}

test "eviction_policy: LFU selection" {
    const strategy = EvictionStrategy.init(.lfu);
    
    var entries: [3]GpuCacheEntry = undefined;
    for (&entries, 0..) |*entry, i| {
        entry.* = GpuCacheEntry{
            .keys = undefined,
            .values = undefined,
            .layer_id = @intCast(i),
            .batch_id = 0,
            .sequence_length = 1,
            .last_access = std.time.timestamp(),
            .access_count = @intCast(10 - i * 2),
            .stream = null,
        };
    }
    
    const victim = strategy.selectVictim(&entries);
    try std.testing.expect(victim.? == 2); // Lowest frequency
    
    std.debug.print("✓ LFU selection working\n", .{});
}

test "eviction_policy: FIFO selection" {
    const strategy = EvictionStrategy.init(.fifo);
    
    var entries: [3]GpuCacheEntry = undefined;
    for (&entries, 0..) |*entry, i| {
        entry.* = GpuCacheEntry{
            .keys = undefined,
            .values = undefined,
            .layer_id = @intCast(i),
            .batch_id = 0,
            .sequence_length = 1,
            .last_access = std.time.timestamp(),
            .access_count = 1,
            .stream = null,
        };
    }
    
    const victim = strategy.selectVictim(&entries);
    try std.testing.expect(victim.? == 0); // First entry
    
    std.debug.print("✓ FIFO selection working\n", .{});
}

test "eviction_policy: stats tracking" {
    var stats = EvictionStats.init();
    
    stats.recordEviction(.lru);
    stats.recordEviction(.lru);
    stats.recordEviction(.lfu);
    
    try std.testing.expect(stats.total_evictions == 3);
    try std.testing.expect(stats.lru_evictions == 2);
    try std.testing.expect(stats.lfu_evictions == 1);
    
    std.debug.print("✓ Eviction stats working\n", .{});
}
