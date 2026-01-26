// Cache Metrics Implementation
// Performance monitoring and analytics for GPU KV cache

const std = @import("std");
const GpuKvCache = @import("gpu_kv_cache.zig").GpuKvCache;
const CacheStats = @import("gpu_kv_cache.zig").CacheStats;
const OperationStats = @import("cache_operations.zig").OperationStats;
const EvictionStats = @import("eviction_policy.zig").EvictionStats;

// ============================================================================
// Performance Metrics
// ============================================================================

/// Detailed performance metrics for cache operations
pub const PerformanceMetrics = struct {
    // Latency tracking (microseconds)
    avg_insert_latency_us: f64,
    avg_lookup_latency_us: f64,
    avg_update_latency_us: f64,
    avg_eviction_latency_us: f64,
    
    // Throughput (operations per second)
    insert_throughput: f64,
    lookup_throughput: f64,
    
    // Resource utilization
    gpu_memory_utilization: f64, // 0.0 to 1.0
    cache_hit_rate: f64,          // 0.0 to 1.0
    
    // Sample counts
    sample_count: u64,
    
    pub fn init() PerformanceMetrics {
        return .{
            .avg_insert_latency_us = 0.0,
            .avg_lookup_latency_us = 0.0,
            .avg_update_latency_us = 0.0,
            .avg_eviction_latency_us = 0.0,
            .insert_throughput = 0.0,
            .lookup_throughput = 0.0,
            .gpu_memory_utilization = 0.0,
            .cache_hit_rate = 0.0,
            .sample_count = 0,
        };
    }
    
    pub fn print(self: *const PerformanceMetrics) void {
        std.debug.print("\n⚡ Performance Metrics\n", .{});
        std.debug.print("   Latencies (μs):\n", .{});
        std.debug.print("     Insert: {d:.2}\n", .{self.avg_insert_latency_us});
        std.debug.print("     Lookup: {d:.2}\n", .{self.avg_lookup_latency_us});
        std.debug.print("     Update: {d:.2}\n", .{self.avg_update_latency_us});
        std.debug.print("     Eviction: {d:.2}\n", .{self.avg_eviction_latency_us});
        std.debug.print("   Throughput (ops/sec):\n", .{});
        std.debug.print("     Insert: {d:.2}\n", .{self.insert_throughput});
        std.debug.print("     Lookup: {d:.2}\n", .{self.lookup_throughput});
        std.debug.print("   Utilization:\n", .{});
        std.debug.print("     GPU Memory: {d:.2}%\n", .{self.gpu_memory_utilization * 100.0});
        std.debug.print("     Cache Hit Rate: {d:.2}%\n", .{self.cache_hit_rate * 100.0});
        std.debug.print("   Samples: {}\n", .{self.sample_count});
    }
};

// ============================================================================
// Latency Tracker
// ============================================================================

/// Track operation latencies over time
pub const LatencyTracker = struct {
    allocator: std.mem.Allocator,
    samples: std.ArrayList(f64),
    window_size: usize,
    
    pub fn init(allocator: std.mem.Allocator, window_size: usize) !*LatencyTracker {
        const self = try allocator.create(LatencyTracker);
        self.* = LatencyTracker{
            .allocator = allocator,
            .samples = std.ArrayList(f64){},
            .window_size = window_size,
        };
        return self;
    }
    
    pub fn deinit(self: *LatencyTracker) void {
        self.samples.deinit();
        self.allocator.destroy(self);
    }
    
    /// Record a latency sample
    pub fn record(self: *LatencyTracker, latency_us: f64) !void {
        try self.samples.append(latency_us);
        
        // Keep only most recent samples
        if (self.samples.items.len > self.window_size) {
            _ = self.samples.orderedRemove(0);
        }
    }
    
    /// Get average latency
    pub fn getAverage(self: *const LatencyTracker) f64 {
        if (self.samples.items.len == 0) return 0.0;
        
        var sum: f64 = 0.0;
        for (self.samples.items) |sample| {
            sum += sample;
        }
        
        return sum / @as(f64, @floatFromInt(self.samples.items.len));
    }
    
    /// Get percentile latency
    pub fn getPercentile(self: *LatencyTracker, percentile: f64) !f64 {
        if (self.samples.items.len == 0) return 0.0;
        
        // Create sorted copy
        var sorted = try self.allocator.alloc(f64, self.samples.items.len);
        defer self.allocator.free(sorted);
        @memcpy(sorted, self.samples.items);
        std.mem.sort(f64, sorted, {}, comptime std.sort.asc(f64));
        
        const idx = @as(usize, @intFromFloat(@as(f64, @floatFromInt(sorted.len)) * percentile / 100.0));
        return sorted[@min(idx, sorted.len - 1)];
    }
    
    /// Get minimum latency
    pub fn getMin(self: *const LatencyTracker) f64 {
        if (self.samples.items.len == 0) return 0.0;
        
        var min_val = self.samples.items[0];
        for (self.samples.items[1..]) |sample| {
            if (sample < min_val) min_val = sample;
        }
        
        return min_val;
    }
    
    /// Get maximum latency
    pub fn getMax(self: *const LatencyTracker) f64 {
        if (self.samples.items.len == 0) return 0.0;
        
        var max_val = self.samples.items[0];
        for (self.samples.items[1..]) |sample| {
            if (sample > max_val) max_val = sample;
        }
        
        return max_val;
    }
};

// ============================================================================
// Cache Metrics Collector
// ============================================================================

/// Collect and aggregate cache metrics
pub const CacheMetricsCollector = struct {
    allocator: std.mem.Allocator,
    
    // Latency trackers
    insert_latency: *LatencyTracker,
    lookup_latency: *LatencyTracker,
    update_latency: *LatencyTracker,
    eviction_latency: *LatencyTracker,
    
    // Throughput tracking
    start_time: i64,
    total_inserts: u64,
    total_lookups: u64,
    
    // Eviction tracking
    eviction_stats: EvictionStats,
    
    pub fn init(allocator: std.mem.Allocator) !*CacheMetricsCollector {
        const self = try allocator.create(CacheMetricsCollector);
        self.* = CacheMetricsCollector{
            .allocator = allocator,
            .insert_latency = try LatencyTracker.init(allocator, 1000),
            .lookup_latency = try LatencyTracker.init(allocator, 1000),
            .update_latency = try LatencyTracker.init(allocator, 1000),
            .eviction_latency = try LatencyTracker.init(allocator, 1000),
            .start_time = std.time.timestamp(),
            .total_inserts = 0,
            .total_lookups = 0,
            .eviction_stats = EvictionStats.init(),
        };
        return self;
    }
    
    pub fn deinit(self: *CacheMetricsCollector) void {
        self.insert_latency.deinit();
        self.lookup_latency.deinit();
        self.update_latency.deinit();
        self.eviction_latency.deinit();
        self.allocator.destroy(self);
    }
    
    /// Record insert operation
    pub fn recordInsert(self: *CacheMetricsCollector, latency_us: f64) !void {
        try self.insert_latency.record(latency_us);
        self.total_inserts += 1;
    }
    
    /// Record lookup operation
    pub fn recordLookup(self: *CacheMetricsCollector, latency_us: f64) !void {
        try self.lookup_latency.record(latency_us);
        self.total_lookups += 1;
    }
    
    /// Record update operation
    pub fn recordUpdate(self: *CacheMetricsCollector, latency_us: f64) !void {
        try self.update_latency.record(latency_us);
    }
    
    /// Record eviction operation
    pub fn recordEviction(self: *CacheMetricsCollector, latency_us: f64, policy: @import("gpu_cache_config.zig").EvictionPolicy) !void {
        try self.eviction_latency.record(latency_us);
        self.eviction_stats.recordEviction(policy);
    }
    
    /// Get comprehensive metrics
    pub fn getMetrics(self: *CacheMetricsCollector, cache: *const GpuKvCache) PerformanceMetrics {
        const elapsed_sec = @as(f64, @floatFromInt(std.time.timestamp() - self.start_time));
        const cache_stats = cache.getStats();
        
        return PerformanceMetrics{
            .avg_insert_latency_us = self.insert_latency.getAverage(),
            .avg_lookup_latency_us = self.lookup_latency.getAverage(),
            .avg_update_latency_us = self.update_latency.getAverage(),
            .avg_eviction_latency_us = self.eviction_latency.getAverage(),
            .insert_throughput = if (elapsed_sec > 0) @as(f64, @floatFromInt(self.total_inserts)) / elapsed_sec else 0.0,
            .lookup_throughput = if (elapsed_sec > 0) @as(f64, @floatFromInt(self.total_lookups)) / elapsed_sec else 0.0,
            .gpu_memory_utilization = if (cache.config.gpuMemorySize() > 0)
                @as(f64, @floatFromInt(cache.total_allocated)) / @as(f64, @floatFromInt(cache.config.gpuMemorySize()))
            else
                0.0,
            .cache_hit_rate = cache_stats.hit_rate,
            .sample_count = self.insert_latency.samples.items.len,
        };
    }
    
    /// Print comprehensive report
    pub fn printReport(self: *CacheMetricsCollector, cache: *const GpuKvCache) !void {
        const metrics = self.getMetrics(cache);
        metrics.print();
        
        std.debug.print("\n📈 Latency Distribution\n", .{});
        std.debug.print("   Insert (μs): min={d:.2}, p50={d:.2}, p95={d:.2}, p99={d:.2}, max={d:.2}\n", .{
            self.insert_latency.getMin(),
            try self.insert_latency.getPercentile(50),
            try self.insert_latency.getPercentile(95),
            try self.insert_latency.getPercentile(99),
            self.insert_latency.getMax(),
        });
        
        std.debug.print("   Lookup (μs): min={d:.2}, p50={d:.2}, p95={d:.2}, p99={d:.2}, max={d:.2}\n", .{
            self.lookup_latency.getMin(),
            try self.lookup_latency.getPercentile(50),
            try self.lookup_latency.getPercentile(95),
            try self.lookup_latency.getPercentile(99),
            self.lookup_latency.getMax(),
        });
        
        self.eviction_stats.print();
    }
};

// ============================================================================
// Benchmark Results
// ============================================================================

/// Store benchmark results for comparison
pub const BenchmarkResults = struct {
    operation_name: []const u8,
    avg_latency_us: f64,
    p50_latency_us: f64,
    p95_latency_us: f64,
    p99_latency_us: f64,
    throughput_ops_sec: f64,
    
    pub fn print(self: *const BenchmarkResults) void {
        std.debug.print("\n🎯 Benchmark: {s}\n", .{self.operation_name});
        std.debug.print("   Avg Latency: {d:.2} μs\n", .{self.avg_latency_us});
        std.debug.print("   P50 Latency: {d:.2} μs\n", .{self.p50_latency_us});
        std.debug.print("   P95 Latency: {d:.2} μs\n", .{self.p95_latency_us});
        std.debug.print("   P99 Latency: {d:.2} μs\n", .{self.p99_latency_us});
        std.debug.print("   Throughput: {d:.2} ops/sec\n", .{self.throughput_ops_sec});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "cache_metrics: latency tracker" {
    const allocator = std.testing.allocator;
    
    var tracker = try LatencyTracker.init(allocator, 100);
    defer tracker.deinit();
    
    try tracker.record(10.0);
    try tracker.record(20.0);
    try tracker.record(30.0);
    
    const avg = tracker.getAverage();
    try std.testing.expect(avg == 20.0);
    
    std.debug.print("✓ Latency tracker working\n", .{});
}

test "cache_metrics: performance metrics" {
    var metrics = PerformanceMetrics.init();
    
    metrics.avg_insert_latency_us = 10.5;
    metrics.cache_hit_rate = 0.85;
    
    try std.testing.expect(metrics.avg_insert_latency_us == 10.5);
    try std.testing.expect(metrics.cache_hit_rate == 0.85);
    
    std.debug.print("✓ Performance metrics working\n", .{});
}

test "cache_metrics: percentiles" {
    const allocator = std.testing.allocator;
    
    var tracker = try LatencyTracker.init(allocator, 100);
    defer tracker.deinit();
    
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try tracker.record(@floatFromInt(i));
    }
    
    const p50 = try tracker.getPercentile(50);
    const p95 = try tracker.getPercentile(95);
    
    try std.testing.expect(p50 >= 40.0 and p50 <= 60.0);
    try std.testing.expect(p95 >= 90.0);
    
    std.debug.print("✓ Percentile calculation working\n", .{});
}
