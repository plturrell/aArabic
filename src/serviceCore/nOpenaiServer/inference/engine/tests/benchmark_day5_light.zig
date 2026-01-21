// Day 5 Lightweight Benchmark - Focus on KV Cache Performance
// Reduced memory footprint to avoid OutOfSpace errors

const std = @import("std");
const tiered = @import("tiered_kv_cache.zig");

/// Lightweight benchmark configuration
const BenchConfig = struct {
    n_layers: u32 = 8,  // Reduced from 32
    n_heads: u32 = 8,
    head_dim: u32 = 128,
    max_seq_len: u32 = 1024,  // Reduced from 4096
    hot_tokens: u32 = 512,     // Reduced from 2048
    iterations: u32 = 5000,    // Reduced from 10000
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
    for (0..50) |_| {
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
    
    const bytes_per_token = kv_dim * 2 * @sizeOf(f32);
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
    for (0..5) |_| {
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

/// Compare performance
fn compareStorePerformance(allocator: std.mem.Allocator) !void {
    const config = BenchConfig{};
    
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("DAY 5 FINAL BENCHMARK - Week 1 Validation\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    
    std.debug.print("\nConfiguration:\n", .{});
    std.debug.print("   Layers: {d}\n", .{config.n_layers});
    std.debug.print("   Heads: {d}\n", .{config.n_heads});
    std.debug.print("   Head dim: {d}\n", .{config.head_dim});
    std.debug.print("   KV dim: {d}\n", .{config.n_heads * config.head_dim});
    std.debug.print("   Iterations: {d}\n", .{config.iterations});
    
    // Historical baselines
    const day1_baseline: f64 = 5046.0;   // Day 1
    const day3_baseline: f64 = 10038.0;  // Day 3 with adaptive eviction
    
    // Test 1: Single token stores with SIMD
    std.debug.print("\n🔬 Running Single Token Benchmark...\n", .{});
    const single_result = try benchmarkSingleStore(allocator, config);
    single_result.print();
    
    // Test 2-5: Batch stores
    std.debug.print("\n🔬 Running Batch Benchmarks...\n", .{});
    const batch_sizes = [_]u32{ 4, 8, 16, 32 };
    var batch_results: [4]BenchResult = undefined;
    
    for (batch_sizes, 0..) |batch_size, i| {
        batch_results[i] = try benchmarkBatchStore(allocator, config, batch_size);
        batch_results[i].print();
    }
    
    // === COMPREHENSIVE ANALYSIS ===
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("📊 COMPREHENSIVE PERFORMANCE ANALYSIS\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    
    // Find best batch performance
    var best_idx: usize = 0;
    var best_rate: f64 = 0;
    for (batch_results, 0..) |result, i| {
        if (result.tokens_per_sec > best_rate) {
            best_rate = result.tokens_per_sec;
            best_idx = i;
        }
    }
    
    std.debug.print("\n1️⃣  SINGLE TOKEN PERFORMANCE (SIMD)\n", .{});
    std.debug.print("   Current: {d:.0} tokens/sec\n", .{single_result.tokens_per_sec});
    std.debug.print("   vs Day 1: {d:.2}x speedup\n", .{single_result.tokens_per_sec / day1_baseline});
    std.debug.print("   vs Day 3: {d:.2}x speedup\n", .{single_result.tokens_per_sec / day3_baseline});
    
    std.debug.print("\n2️⃣  BATCH PROCESSING PERFORMANCE\n", .{});
    for (batch_results, batch_sizes) |result, batch_size| {
        const speedup_single = result.tokens_per_sec / single_result.tokens_per_sec;
        const speedup_day1 = result.tokens_per_sec / day1_baseline;
        std.debug.print("   Batch {d:2}: {d:.0} tok/s ({d:.2}x vs single, {d:.2}x vs Day 1)\n", .{
            batch_size,
            result.tokens_per_sec,
            speedup_single,
            speedup_day1,
        });
    }
    
    std.debug.print("\n3️⃣  OPTIMAL CONFIGURATION\n", .{});
    std.debug.print("   ✅ Best batch size: {d} tokens\n", .{batch_sizes[best_idx]});
    std.debug.print("   ✅ Best performance: {d:.0} tokens/sec\n", .{best_rate});
    std.debug.print("   ✅ Throughput: {d:.2} GB/s\n", .{batch_results[best_idx].throughput_gbps});
    
    std.debug.print("\n4️⃣  TOTAL IMPROVEMENTS\n", .{});
    const total_speedup_day1 = best_rate / day1_baseline;
    const total_speedup_day3 = best_rate / day3_baseline;
    std.debug.print("   📈 Day 1 → Day 5: {d:.2}x total speedup\n", .{total_speedup_day1});
    std.debug.print("   📈 Day 3 → Day 5: {d:.2}x SIMD+Batch speedup\n", .{total_speedup_day3});
    std.debug.print("   📈 Day 1 baseline: {d:.0} tok/s\n", .{day1_baseline});
    std.debug.print("   📈 Day 3 baseline: {d:.0} tok/s\n", .{day3_baseline});
    std.debug.print("   📈 Day 5 final: {d:.0} tok/s\n", .{best_rate});
    
    // Week 1 target check
    const week1_target: f64 = 50000.0;
    std.debug.print("\n5️⃣  WEEK 1 TARGET VALIDATION\n", .{});
    std.debug.print("   🎯 Target: {d:.0} tokens/sec\n", .{week1_target});
    std.debug.print("   📊 Achieved: {d:.0} tokens/sec\n", .{best_rate});
    const progress = (best_rate / week1_target) * 100.0;
    std.debug.print("   📈 Progress: {d:.1}%\n", .{progress});
    
    if (best_rate >= week1_target) {
        const exceed_pct = ((best_rate / week1_target) - 1.0) * 100.0;
        std.debug.print("   ✅ ✅ ✅ TARGET EXCEEDED BY {d:.1}%!\n", .{exceed_pct});
        std.debug.print("   🎉 WEEK 1 GOAL ACHIEVED!\n", .{});
    } else {
        const remaining = week1_target - best_rate;
        const remaining_pct = (remaining / week1_target) * 100.0;
        std.debug.print("   ⚠️  Remaining: {d:.0} tokens/sec ({d:.1}%)\n", .{remaining, remaining_pct});
        std.debug.print("   📝 Next steps: Profile hot paths, add explicit NEON intrinsics\n", .{});
    }
    
    std.debug.print("\n" ++ "=" ** 80 ++ "\n", .{});
    std.debug.print("✅ DAY 5 BENCHMARK COMPLETE\n", .{});
    std.debug.print("=" ** 80 ++ "\n\n", .{});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    try compareStorePerformance(allocator);
}
