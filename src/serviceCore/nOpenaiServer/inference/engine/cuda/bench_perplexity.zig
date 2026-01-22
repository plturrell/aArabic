const std = @import("std");
const cuda = @import("cuda_bindings");
const GpuInference = @import("gpu_inference").GpuInference;
const GpuWeightCache = @import("gpu_weight_cache").GpuWeightCache;
const gguf = @import("gguf_loader");
const tokenizer_mod = @import("tokenizer");

/// Calculate perplexity over a text dataset
/// PPL = exp(-1/N * Σ log P(token_i | context))
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 3) {
        std.debug.print("Usage: bench_perplexity <model.gguf> <dataset.txt> [context_size]\n", .{});
        std.debug.print("Example: bench_perplexity tinyllama.Q4_K_M.gguf wiki.test.raw 512\n", .{});
        return;
    }

    const model_path = args[1];
    const dataset_path = args[2];
    const context_size: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 512;

    std.debug.print("\n🧮 Perplexity Benchmark\n", .{});
    std.debug.print("=" ** 60 ++ "\n", .{});
    std.debug.print("Model: {s}\n", .{model_path});
    std.debug.print("Dataset: {s}\n", .{dataset_path});
    std.debug.print("Context size: {d}\n", .{context_size});

    // Load dataset
    std.debug.print("\n📂 Loading dataset...\n", .{});
    const dataset_file = try std.fs.cwd().openFile(dataset_path, .{});
    defer dataset_file.close();
    const dataset_text = try dataset_file.readToEndAlloc(allocator, 100 * 1024 * 1024);
    defer allocator.free(dataset_text);
    std.debug.print("   Dataset size: {d} bytes\n", .{dataset_text.len});

    // Initialize CUDA
    var device_count: c_int = 0;
    _ = cuda.cudaGetDeviceCount(&device_count);
    if (device_count == 0) return error.NoGPU;
    std.debug.print("✅ GPU detected\n", .{});

    // Load model
    std.debug.print("\n📦 Loading model...\n", .{});
    var model = try gguf.GGUFModel.load(allocator, model_path);
    defer model.deinit();

    const n_layers = model.metadata.n_layers;
    const embed_dim = model.metadata.hidden_size;
    const hidden_dim = model.metadata.intermediate_size;
    const n_heads = model.metadata.n_heads;
    const n_kv_heads = model.metadata.n_kv_heads;
    const vocab_size = model.metadata.vocab_size;
    const rope_theta: f32 = 10000.0; // Default for LLaMA models

    std.debug.print("   Layers: {d}, Embed: {d}, Hidden: {d}, Heads: {d}/{d}, Vocab: {d}\n", .{
        n_layers, embed_dim, hidden_dim, n_heads, n_kv_heads, vocab_size,
    });

    // Load tokenizer
    std.debug.print("\n🔤 Loading tokenizer...\n", .{});
    var tok = try tokenizer_mod.Tokenizer.loadFromModel(allocator, &model);
    defer tok.deinit();

    // Tokenize dataset
    std.debug.print("   Tokenizing dataset...\n", .{});
    const tokens = try tok.encode(dataset_text, allocator);
    defer allocator.free(tokens);
    std.debug.print("   Total tokens: {d}\n", .{tokens.len});

    // Calculate number of chunks
    const num_chunks = tokens.len / context_size;
    std.debug.print("   Chunks ({d} tokens each): {d}\n", .{ context_size, num_chunks });

    // Load weights to GPU
    std.debug.print("\n🎮 Loading weights to GPU...\n", .{});
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
    std.debug.print("   GPU memory: {d:.2} GB\n", .{@as(f64, @floatFromInt(weight_cache.total_gpu_memory)) / (1024 * 1024 * 1024)});

    // Initialize GPU inference engine
    var gpu_engine = try GpuInference.init(
        allocator,
        @intCast(embed_dim),
        @intCast(hidden_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(vocab_size),
        @intCast(n_layers),
        rope_theta,
    );
    defer gpu_engine.deinit();

    // Calculate perplexity
    std.debug.print("\n⏱️  Calculating perplexity over {d} chunks...\n", .{num_chunks});
    const start_time = std.time.nanoTimestamp();

    var total_nll: f64 = 0.0; // Negative log likelihood sum
    var total_tokens_evaluated: usize = 0;

    // Allocate buffer for logits
    const logits_host = try allocator.alloc(f32, vocab_size);
    defer allocator.free(logits_host);

    for (0..num_chunks) |chunk_idx| {
        const chunk_start = chunk_idx * context_size;
        const chunk_tokens = tokens[chunk_start .. chunk_start + context_size];

        // For each token in chunk (except first), calculate NLL
        for (1..context_size) |pos| {
            const input_token = chunk_tokens[pos - 1];
            const target_token = chunk_tokens[pos];

            // Run forward pass
            try gpu_engine.loadEmbedding(&weight_cache, input_token);
            const logits_gpu = try gpu_engine.forward(&weight_cache, @intCast(pos - 1));
            try logits_gpu.copyToHostF32(logits_host);

            // Compute log softmax and get NLL for target token
            const nll = computeNLL(logits_host, target_token);
            total_nll += nll;
            total_tokens_evaluated += 1;
        }

        // Progress update every 10 chunks
        if ((chunk_idx + 1) % 10 == 0 or chunk_idx == num_chunks - 1) {
            const current_ppl = @exp(total_nll / @as(f64, @floatFromInt(total_tokens_evaluated)));
            std.debug.print("   [{d}/{d}] Current PPL: {d:.4}\n", .{ chunk_idx + 1, num_chunks, current_ppl });
        }
    }

    const elapsed_ns = std.time.nanoTimestamp() - start_time;
    const elapsed_s = @as(f64, @floatFromInt(elapsed_ns)) / 1_000_000_000.0;

    // Final perplexity
    const perplexity = @exp(total_nll / @as(f64, @floatFromInt(total_tokens_evaluated)));

    std.debug.print("\n" ++ "=" ** 60 ++ "\n", .{});
    std.debug.print("📊 RESULTS\n", .{});
    std.debug.print("   Tokens evaluated: {d}\n", .{total_tokens_evaluated});
    std.debug.print("   Time: {d:.2}s\n", .{elapsed_s});
    std.debug.print("   Speed: {d:.1} tokens/s\n", .{@as(f64, @floatFromInt(total_tokens_evaluated)) / elapsed_s});
    std.debug.print("\n   🎯 PERPLEXITY: {d:.4}\n", .{perplexity});
    std.debug.print("=" ** 60 ++ "\n", .{});
}

/// Compute negative log likelihood for target token given logits
fn computeNLL(logits: []const f32, target: u32) f64 {
    // Find max for numerical stability
    var max_logit: f32 = logits[0];
    for (logits[1..]) |l| {
        if (l > max_logit) max_logit = l;
    }

    // Compute log-sum-exp
    var sum_exp: f64 = 0.0;
    for (logits) |l| {
        sum_exp += @exp(@as(f64, l - max_logit));
    }
    const log_sum_exp = @as(f64, max_logit) + @log(sum_exp);

    // NLL = -log P(target) = log_sum_exp - logit[target]
    return log_sum_exp - @as(f64, logits[target]);
}

