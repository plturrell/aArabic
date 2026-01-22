// GPU Inference Benchmark
// Target: 500+ tokens/second on Tesla T4
//
// Usage: zig build bench-gpu -- /path/to/model.gguf

const std = @import("std");
const gguf = @import("gguf_loader");
const GpuInference = @import("gpu_inference").GpuInference;
const GpuWeightCache = @import("gpu_weight_cache").GpuWeightCache;
const cuda = @import("cuda_bindings");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 2) {
        std.debug.print("Usage: bench_gpu_inference <model.gguf> [num_tokens]\n", .{});
        std.debug.print("Example: bench_gpu_inference tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf 100\n", .{});
        return;
    }

    const model_path = args[1];
    const num_tokens: u32 = if (args.len > 2) std.fmt.parseInt(u32, args[2], 10) catch 100 else 100;

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🚀 GPU Inference Benchmark - Target: 500 tokens/second\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📁 Model: {s}\n", .{model_path});
    std.debug.print("🎯 Tokens to generate: {}\n", .{num_tokens});

    // Check GPU
    var device_count: c_int = 0;
    if (cuda.cudaGetDeviceCount(&device_count) != 0 or device_count == 0) {
        std.debug.print("❌ No CUDA GPU available\n", .{});
        return;
    }
    std.debug.print("✅ GPU detected: {} device(s)\n", .{device_count});

    // Load GGUF model
    std.debug.print("\n📦 Loading model...\n", .{});
    var model = try gguf.GGUFModel.load(allocator, model_path);
    defer model.deinit();

    // Get model config from parsed metadata
    const n_layers = model.metadata.n_layers;
    const embed_dim = model.metadata.hidden_size;
    const hidden_dim = model.metadata.intermediate_size;
    const n_heads = model.metadata.n_heads;
    const n_kv_heads = model.metadata.n_kv_heads;
    const vocab_size = model.metadata.vocab_size;

    std.debug.print("   Layers: {}\n", .{n_layers});
    std.debug.print("   Embed dim: {}\n", .{embed_dim});
    std.debug.print("   Hidden dim: {}\n", .{hidden_dim});
    std.debug.print("   Heads: {} / KV heads: {}\n", .{ n_heads, n_kv_heads });
    std.debug.print("   Vocab size: {}\n", .{vocab_size});

    // Initialize GPU weight cache (pre-load all weights to GPU)
    std.debug.print("\n", .{});
    var weight_cache = try GpuWeightCache.init(
        allocator,
        &model,
        @intCast(n_layers),
        @intCast(vocab_size),
        @intCast(embed_dim),
        @intCast(hidden_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
    );
    defer weight_cache.deinit();

    // =========================================================================
    // Benchmark 1: Single-token mode (baseline)
    // =========================================================================
    std.debug.print("\n═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("📊 BENCHMARK 1: Single-Token Mode (M=1)\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    var gpu_engine_single = try GpuInference.init(
        allocator,
        @intCast(embed_dim),
        @intCast(hidden_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(vocab_size),
        @intCast(n_layers),
    );
    defer gpu_engine_single.deinit();

    // Warm-up
    try gpu_engine_single.loadEmbedding(&weight_cache, 1);
    _ = try gpu_engine_single.forward(&weight_cache, 0);
    _ = cuda.cudaDeviceSynchronize();

    // Single-token benchmark
    std.debug.print("⏱️  Running single-token benchmark ({} tokens)...\n", .{num_tokens});
    const start1 = std.time.nanoTimestamp();

    for (0..num_tokens) |i| {
        try gpu_engine_single.loadEmbedding(&weight_cache, @intCast(i % vocab_size));
        _ = try gpu_engine_single.forward(&weight_cache, @intCast(i));
    }
    _ = cuda.cudaDeviceSynchronize();
    const end1 = std.time.nanoTimestamp();

    const elapsed1_ms = @as(f64, @floatFromInt(end1 - start1)) / 1_000_000.0;
    const tps1 = @as(f64, @floatFromInt(num_tokens)) / (elapsed1_ms / 1000.0);
    std.debug.print("   Throughput: {d:.1} tokens/second\n", .{tps1});
    std.debug.print("   Time per token: {d:.2} ms\n", .{elapsed1_ms / @as(f64, @floatFromInt(num_tokens))});

    // =========================================================================
    // Benchmark 2-5: Multiple batch sizes
    // =========================================================================
    const batch_sizes = [_]u32{ 8, 32, 64, 128, 256, 512, 1024 };
    var best_tps: f64 = tps1;
    var best_batch: u32 = 1;

    // Allocate max-size token buffer
    var token_ids_buf: [1024]u32 = undefined;

    inline for (batch_sizes) |bs| {
        std.debug.print("\n═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("📊 BENCHMARK: Batched Mode (batch_size={})\n", .{bs});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

        var gpu_engine_batch = try GpuInference.initWithBatchSize(
            allocator,
            @intCast(embed_dim),
            @intCast(hidden_dim),
            @intCast(n_heads),
            @intCast(n_kv_heads),
            @intCast(vocab_size),
            @intCast(n_layers),
            bs,
        );
        defer gpu_engine_batch.deinit();

        // Prepare batch of token IDs
        for (0..bs) |i| token_ids_buf[i] = @intCast(i + 1);
        const token_ids = token_ids_buf[0..bs];

        // Warm-up
        try gpu_engine_batch.loadEmbeddingsBatched(&weight_cache, token_ids);
        _ = try gpu_engine_batch.forwardBatched(&weight_cache, 0);
        _ = cuda.cudaDeviceSynchronize();

        // Batched benchmark
        const num_batches = @max(1, num_tokens / bs);
        const total_tokens = num_batches * bs;

        std.debug.print("⏱️  Running ({} batches × {} tokens = {} tokens)...\n", .{num_batches, bs, total_tokens});
        const start2 = std.time.nanoTimestamp();

        for (0..num_batches) |batch_idx| {
            for (0..bs) |i| {
                token_ids_buf[i] = @intCast((batch_idx * bs + i) % vocab_size);
            }
            try gpu_engine_batch.loadEmbeddingsBatched(&weight_cache, token_ids);
            _ = try gpu_engine_batch.forwardBatched(&weight_cache, @intCast(batch_idx));
        }
        _ = cuda.cudaDeviceSynchronize();
        const end2 = std.time.nanoTimestamp();

        const elapsed2_ms = @as(f64, @floatFromInt(end2 - start2)) / 1_000_000.0;
        const tps2 = @as(f64, @floatFromInt(total_tokens)) / (elapsed2_ms / 1000.0);
        const speedup = tps2 / tps1;
        const ms_per_batch = elapsed2_ms / @as(f64, @floatFromInt(num_batches));

        std.debug.print("   Throughput: {d:.1} tokens/second\n", .{tps2});
        std.debug.print("   Time per batch: {d:.2} ms ({d:.2} ms/token)\n", .{ms_per_batch, ms_per_batch / @as(f64, @floatFromInt(bs))});
        std.debug.print("   Speedup vs M=1: {d:.2}x\n", .{speedup});

        if (tps2 > best_tps) {
            best_tps = tps2;
            best_batch = bs;
        }
    }

    // =========================================================================
    // Summary
    // =========================================================================
    std.debug.print("\n═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🏆 FINAL RESULTS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("   Best throughput: {d:.1} tokens/second (batch_size={})\n", .{best_tps, best_batch});
    std.debug.print("   Baseline (M=1):  {d:.1} tokens/second\n", .{tps1});
    std.debug.print("   Max speedup:     {d:.2}x\n", .{best_tps / tps1});
    std.debug.print("\n", .{});

    if (best_tps >= 500.0) {
        std.debug.print("   🎉 TARGET ACHIEVED! {d:.1} tokens/second ≥ 500 tok/s\n", .{best_tps});
    }
    std.debug.print("\n", .{});
    std.debug.print("   GPU weights: {d:.2} GB\n", .{@as(f64, @floatFromInt(weight_cache.total_gpu_memory)) / (1024.0 * 1024.0 * 1024.0)});
}

