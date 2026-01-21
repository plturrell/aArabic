// Benchmark Suite for Tiered KV Cache - Day 4 SIMD Optimizations
// Tests performance improvements from SIMD vectorization and batch processing

const std = @import("std");
const tiered = @import("tiered_kv_cache.zig");

/// Benchmark configuration
const BenchConfig = struct {
    n_layers: u32 = 32,
    n_heads: u32 = 8,
    head_dim: u32 = 128,
    max_seq_len: u32 = 4096,
    hot_tokens: u32 = 2048,
    iterations: u32 = 10000,
};

/// Benchmark results
const BenchResult = struct {
    name: []const u8,
    total_time_ns: u64,
    tokens_processed: u64,
    tokens_per_sec: f64,
    throughput_gbps: f64,
    
    pub fn print(self: BenchResult) void {
        const time_ms = @as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000.0;
        std.debug.print("\n📊 {s}\n", .{self.name});
        std.debug.print("   Time: {d:.2} ms\n", .{time_ms});
        std.debug.print("   Tokens: {d}\n", .{self.tokens_processed});
        std.debug.print("   Rate: {d:.0} tokens/sec\n", .{self.tokens_per_sec});
        std.debug.print("   Throughput: {d:.2} GB/s\n", .{self.throughput_gbps});
    }
};

/// Benchmark single token store operations
fn benchmarkSingleStore(allocator: std.mem.Allocator, config: BenchConfig) !BenchResult {
    const cache_config = tiered.TieredKVConfig{
        .n_layers = config.n_layers,
        .n_heads = config.n_heads,
        .head_dim = config.head_dim,
        .max_seq_len = config.max_seq_len,
        .hot_tokens = config.hot_tokens,
        .eviction_policy = .adaptive_lru,
        .test_mode = true,  // Day 5: Use minimal SSD allocation for benchmarks
    };
    
    const cache = try tiered.TieredKVCache.init(allocator, cache_config);
    defer cache.deinit();
    
    const kv_dim = cache_config.kvDim();
    const keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    // Fill with test data
    for (keys, 0..) |*k, i| k.* = @floatFromInt(i);
    for (values, 0..) |*v, i| v.* = @floatFromInt(i + 1000);
    
    // Warmup
    for (0..100) |_| {
        try cache.store(0, keys, values);
        cache.advance();
    }
    cache.reset();
    
    // Benchmark
    const start = std.time.nanoTimestamp();
    for (0..config.iterations) |_| {
        try cache.store(0, keys, values);
        cache.advance();
    }
    const elapsed = std.time.nanoTimestamp() - start;
    
    const bytes_per_token = kv_dim * 2 * @sizeOf(f32); // keys + values
    const total_bytes = config.iterations * bytes_per_token;
    const throughput = (@as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(elapsed))) * 1e9;
    
    return BenchResult{
        .name = "Single Token Store (SIMD)",
        .total_time_ns = @intCast(elapsed),
        .tokens_processed = config.iterations,
        .tokens_per_sec = (@as(f64, @floatFromInt(config.iterations)) / @as(f64, @floatFromInt(elapsed))) * 1e9,
        .throughput_gbps = throughput / 1e9,
    };
}

/// Benchmark batch store operations
fn benchmarkBatchStore(allocator: std.mem.Allocator, config: BenchConfig, batch_size: u32) !BenchResult {
    const cache_config = tiered.TieredKVConfig{
        .n_layers = config.n_layers,
        .n_heads = config.n_heads,
        .head_dim = config.head_dim,
        .max_seq_len = config.max_seq_len,
        .hot_tokens = config.hot_tokens,
        .eviction_policy = .adaptive_lru,
        .test_mode = true,  // Day 5: Use minimal SSD allocation for benchmarks
    };
    
    const cache = try tiered.TieredKVCache.init(allocator, cache_config);
    defer cache.deinit();
    
    const kv_dim = cache_config.kvDim();
    const keys_batch = try allocator.alloc(f32, batch_size * kv_dim);
    defer allocator.free(keys_batch);
    const values_batch = try allocator.alloc(f32, batch_size * kv_dim);
    defer allocator.free(values_batch);
    
    // Fill with test data
    for (keys_batch, 0..) |*k, i| k.* = @floatFromInt(i);
    for (values_batch, 0..) |*v, i| v.* = @floatFromInt(i + 1000);
    
    const num_batches = config.iterations / batch_size;
    
    // Warmup
    for (0..10) |_| {
        try cache.storeBatch(0, keys_batch, values_batch, batch_size);
    }
    cache.reset();
    
    // Benchmark
    const start = std.time.nanoTimestamp();
    for (0..num_batches) |_| {
        try cache.storeBatch(0, keys_batch, values_batch, batch_size);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    
    const total_tokens = num_batches * batch_size;
    const bytes_per_token = kv_dim * 2 * @sizeOf(f32);
    const total_bytes = total_tokens * bytes_per_token;
    const throughput = (@as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(elapsed))) * 1e9;
    
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "Batch Store (batch_size={d})", .{batch_size});
    
    return BenchResult{
        .name = name,
        .total_time_ns = @intCast(elapsed),
        .tokens_processed = total_tokens,
        .tokens_per_sec = (@as(f64, @floatFromInt(total_tokens)) / @as(f64, @floatFromInt(elapsed))) * 1e9,
        .throughput_gbps = throughput / 1e9,
    };
}

/// Benchmark memory copy operations directly
fn benchmarkMemcpy(allocator: std.mem.Allocator, size: usize) !BenchResult {
    const src = try allocator.alloc(f32, size);
    defer allocator.free(src);
    const dst = try allocator.alloc(f32, size);
    defer allocator.free(dst);
    
    // Fill with test data
    for (src, 0..) |*s, i| s.* = @floatFromInt(i);
    
    const iterations: usize = 1000;
    
    // Warmup
    for (0..10) |_| {
        @memcpy(dst, src);
    }
    
    // Benchmark standard memcpy
    const start = std.time.nanoTimestamp();
    for (0..iterations) |_| {
        @memcpy(dst, src);
    }
    const elapsed = std.time.nanoTimestamp() - start;
    
    const bytes_per_iter = size * @sizeOf(f32);
    const total_bytes = iterations * bytes_per_iter;
    const throughput = (@as(f64, @floatFromInt(total_bytes)) / @as(f64, @floatFromInt(elapsed))) * 1e9;
    
    var name_buf: [128]u8 = undefined;
    const name = try std.fmt.bufPrint(&name_buf, "Standard memcpy ({d} floats)", .{size});
    
    return BenchResult{
        .name = name,
        .total_time_ns = @intCast(elapsed),
        .tokens_processed = iterations,
        .tokens_per_sec = (@as(f64, @floatFromInt(iterations)) / @as(f64, @floatFromInt(elapsed))) * 1e9,
        .throughput_gbps = throughput / 1e9,
    };
}

/// Compare single vs batch performance
fn compareStorePerformance(allocator: std.mem.Allocator) !void {
    const config = BenchConfig{};
    
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("Day 4 SIMD Optimization Benchmark Suite\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    
    std.debug.print("\nConfiguration:\n", .{});
    std.debug.print("   Layers: {d}\n", .{config.n_layers});
    std.debug.print("   Heads: {d}\n", .{config.n_heads});
    std.debug.print("   Head dim: {d}\n", .{config.head_dim});
    std.debug.print("   KV dim: {d}\n", .{config.n_heads * config.head_dim});
    std.debug.print("   Iterations: {d}\n", .{config.iterations});
    
    // Test 1: Single token stores with SIMD
    const single_result = try benchmarkSingleStore(allocator, config);
    single_result.print();
    
    // Test 2-5: Batch stores with different batch sizes
    const batch_sizes = [_]u32{ 4, 8, 16, 32 };
    var batch_results: [4]BenchResult = undefined;
    
    for (batch_sizes, 0..) |batch_size, i| {
        batch_results[i] = try benchmarkBatchStore(allocator, config, batch_size);
        batch_results[i].print();
    }
    
    // Test 6-8: Raw memory copy benchmarks
    std.debug.print("\n--- Raw Memory Copy Performance ---\n", .{});
    
    const memcpy_sizes = [_]usize{ 128, 1024, 8192 };
    for (memcpy_sizes) |size| {
        const memcpy_result = try benchmarkMemcpy(allocator, size);
        memcpy_result.print();
    }
    
    // Summary
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("Performance Summary\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    
    std.debug.print("\nSingle Token Store:\n", .{});
    std.debug.print("   {d:.0} tokens/sec | {d:.2} GB/s\n", .{
        single_result.tokens_per_sec,
        single_result.throughput_gbps,
    });
    
    std.debug.print("\nBatch Store Performance:\n", .{});
    for (batch_results, batch_sizes) |result, batch_size| {
        const speedup = result.tokens_per_sec / single_result.tokens_per_sec;
        std.debug.print("   Batch {d:2}: {d:.0} tokens/sec ({d:.2}x speedup)\n", .{
            batch_size,
            result.tokens_per_sec,
            speedup,
        });
    }
    
    // Find best batch size
    var best_idx: usize = 0;
    var best_rate: f64 = 0;
    for (batch_results, 0..) |result, i| {
        if (result.tokens_per_sec > best_rate) {
            best_rate = result.tokens_per_sec;
            best_idx = i;
        }
    }
    
    std.debug.print("\n✅ Optimal batch size: {d} tokens\n", .{batch_sizes[best_idx]});
    std.debug.print("✅ Best performance: {d:.0} tokens/sec\n", .{best_rate});
    std.debug.print("✅ Total speedup: {d:.2}x over single-token baseline\n", .{
        best_rate / single_result.tokens_per_sec,
    });
    
    // Week 1 target check
    const week1_target: f64 = 50000.0; // 50K tokens/sec
    std.debug.print("\n📈 Week 1 Target Progress:\n", .{});
    std.debug.print("   Target: {d:.0} tokens/sec\n", .{week1_target});
    std.debug.print("   Current: {d:.0} tokens/sec\n", .{best_rate});
    const progress = (best_rate / week1_target) * 100.0;
    std.debug.print("   Progress: {d:.1}%\n", .{progress});
    
    if (best_rate >= week1_target) {
        std.debug.print("   ✅ TARGET EXCEEDED!\n", .{});
    } else {
        const remaining = week1_target - best_rate;
        std.debug.print("   ⚠️  Remaining: {d:.0} tokens/sec\n", .{remaining});
    }
    
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    try compareStorePerformance(allocator);
}

test "SIMD memcpy correctness" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const size: usize = 1024;
    const src = try allocator.alloc(f32, size);
    defer allocator.free(src);
    const dst = try allocator.alloc(f32, size);
    defer allocator.free(dst);
    
    // Fill with test data
    for (src, 0..) |*val, i| {
        val.* = @floatFromInt(i);
    }
    
    // Test SIMD copy via store function
    const cache_config = tiered.TieredKVConfig{
        .n_layers = 1,
        .n_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
    };
    
    const cache = try tiered.TieredKVCache.init(allocator, cache_config);
    defer cache.deinit();
    
    const kv_dim = cache_config.kvDim();
    const keys = try allocator.alloc(f32, kv_dim);
    defer allocator.free(keys);
    const values = try allocator.alloc(f32, kv_dim);
    defer allocator.free(values);
    
    for (keys, 0..) |*k, i| k.* = @floatFromInt(i);
    for (values, 0..) |*v, i| v.* = @floatFromInt(i + 1000);
    
    // Store and verify
    try cache.store(0, keys, values);
    
    const stored_keys = cache.getHotKeys(0);
    const stored_values = cache.getHotValues(0);
    
    try testing.expectEqual(keys.len, stored_keys.len);
    try testing.expectEqual(values.len, stored_values.len);
    
    for (keys, stored_keys) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
    
    for (values, stored_values) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
}

test "Batch store correctness" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const cache_config = tiered.TieredKVConfig{
        .n_layers = 1,
        .n_heads = 8,
        .head_dim = 128,
        .max_seq_len = 4096,
    };
    
    const cache = try tiered.TieredKVCache.init(allocator, cache_config);
    defer cache.deinit();
    
    const batch_size: u32 = 16;
    const kv_dim = cache_config.kvDim();
    const keys_batch = try allocator.alloc(f32, batch_size * kv_dim);
    defer allocator.free(keys_batch);
    const values_batch = try allocator.alloc(f32, batch_size * kv_dim);
    defer allocator.free(values_batch);
    
    // Fill with test data
    for (keys_batch, 0..) |*k, i| k.* = @floatFromInt(i);
    for (values_batch, 0..) |*v, i| v.* = @floatFromInt(i + 10000);
    
    // Store batch
    try cache.storeBatch(0, keys_batch, values_batch, batch_size);
    
    // Verify
    try testing.expectEqual(cache.seq_pos, batch_size);
    
    const stored_keys = cache.getHotKeys(0);
    const stored_values = cache.getHotValues(0);
    
    try testing.expectEqual(batch_size * kv_dim, stored_keys.len);
    try testing.expectEqual(batch_size * kv_dim, stored_values.len);
    
    for (keys_batch, stored_keys) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
    
    for (values_batch, stored_values) |expected, actual| {
        try testing.expectEqual(expected, actual);
    }
}
