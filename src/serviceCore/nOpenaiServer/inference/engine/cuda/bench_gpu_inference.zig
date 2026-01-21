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

    // Initialize GPU inference engine
    var gpu_engine = try GpuInference.init(
        allocator,
        @intCast(embed_dim),
        @intCast(hidden_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(vocab_size),
        @intCast(n_layers),
    );
    defer gpu_engine.deinit();

    // Allocate host buffer for logits
    const logits_buffer = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits_buffer);

    // Warm-up run
    std.debug.print("\n🔥 Warming up...\n", .{});
    try gpu_engine.loadEmbedding(&weight_cache, 1); // Token ID 1
    _ = try gpu_engine.forward(&weight_cache, 0);
    _ = cuda.cudaDeviceSynchronize();

    // Benchmark
    std.debug.print("\n⏱️  Running benchmark ({} tokens)...\n", .{num_tokens});
    const start_time = std.time.nanoTimestamp();

    for (0..num_tokens) |i| {
        // Load token embedding
        try gpu_engine.loadEmbedding(&weight_cache, @intCast(i % vocab_size));

        // Forward pass (all on GPU)
        _ = try gpu_engine.forward(&weight_cache, @intCast(i));
    }

    // Sync before measuring time
    _ = cuda.cudaDeviceSynchronize();
    const end_time = std.time.nanoTimestamp();

    // Get final logits (only copy once at end)
    try gpu_engine.getLogits(logits_buffer);

    // Calculate metrics
    const elapsed_ns = end_time - start_time;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000.0;
    const elapsed_s = elapsed_ms / 1000.0;
    const tokens_per_sec = @as(f64, @floatFromInt(num_tokens)) / elapsed_s;
    const ms_per_token = elapsed_ms / @as(f64, @floatFromInt(num_tokens));

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("📊 BENCHMARK RESULTS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("   Tokens generated: {}\n", .{num_tokens});
    std.debug.print("   Total time: {d:.2} ms\n", .{elapsed_ms});
    std.debug.print("   Time per token: {d:.2} ms\n", .{ms_per_token});
    std.debug.print("   Throughput: {d:.1} tokens/second\n", .{tokens_per_sec});
    std.debug.print("\n", .{});

    if (tokens_per_sec >= 500.0) {
        std.debug.print("   🎉 TARGET ACHIEVED! ≥500 tokens/second\n", .{});
    } else if (tokens_per_sec >= 100.0) {
        std.debug.print("   ✅ Good progress: {d:.0}% of target\n", .{tokens_per_sec / 5.0});
    } else {
        std.debug.print("   ⚠️  Needs optimization: {d:.1}% of target\n", .{tokens_per_sec / 5.0});
    }
    std.debug.print("\n", .{});

    // Memory stats
    std.debug.print("📈 Memory Usage:\n", .{});
    std.debug.print("   GPU weights: {d:.2} GB\n", .{@as(f64, @floatFromInt(weight_cache.total_gpu_memory)) / (1024.0 * 1024.0 * 1024.0)});
}

