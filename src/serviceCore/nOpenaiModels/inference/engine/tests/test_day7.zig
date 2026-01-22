const std = @import("std");
const matrix_ops = @import("matrix_ops");
const batch_processor = @import("batch_processor");
const llama = @import("llama_model");
const gguf = @import("gguf_loader");
const transformer = @import("transformer");
const tokenizer = @import("tokenizer");

/// Day 7 Tests: Batch Processing
///
/// Tests:
/// 1. Batch state initialization
/// 2. Batch embedding retrieval
/// 3. Batch forward pass
/// 4. Memory efficiency
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  DAY 7 TESTS: BATCH PROCESSING\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    // Test 1: Batch processor unit tests
    try batch_processor.test_batch_processor(allocator);

    // Test 2: Batch with model
    try test_batch_with_model(allocator);

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL DAY 7 TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("📊 Summary:\n", .{});
    std.debug.print("   ✅ Batch state initialization\n", .{});
    std.debug.print("   ✅ Batch embedding retrieval\n", .{});
    std.debug.print("   ✅ Batch model integration\n", .{});
    std.debug.print("   ✅ Memory-efficient batching\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🎊 Batch processing ready! Week 2 Day 7 complete!\n", .{});
    std.debug.print("\n", .{});
}

fn test_batch_with_model(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Batch with Model\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    std.debug.print("\n1️⃣  Creating test model...\n", .{});

    // Create minimal test configuration
    const config = llama.LlamaConfig{
        .vocab_size = 100,
        .n_layers = 2,
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .max_seq_len = 32,
    };

    // Create dummy weights
    // Note: All these allocations are passed to LlamaWeights which is owned by LlamaModel.
    // LlamaModel.deinit() calls LlamaWeights.deinit() which frees all of these.
    // Do NOT use defer to free them here as that would cause double-free.
    const token_embedding = try allocator.alloc(f32, config.vocab_size * config.embed_dim);
    @memset(token_embedding, 0.1);

    const output_norm = try allocator.alloc(f32, config.embed_dim);
    for (output_norm) |*w| w.* = 1.0;

    const output_weight = try allocator.alloc(f32, config.embed_dim * config.vocab_size);
    @memset(output_weight, 0.1);

    // Create layer weights
    const layer_weights = try allocator.alloc(transformer.TransformerWeights, config.n_layers);

    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;

    for (0..config.n_layers) |i| {
        const attn_norm = try allocator.alloc(f32, config.embed_dim);
        for (attn_norm) |*w| w.* = 1.0;

        const wq = try allocator.alloc(f32, config.embed_dim * q_dim);
        @memset(wq, 0.1);
        const wk = try allocator.alloc(f32, config.embed_dim * kv_dim);
        @memset(wk, 0.1);
        const wv = try allocator.alloc(f32, config.embed_dim * kv_dim);
        @memset(wv, 0.1);
        const wo = try allocator.alloc(f32, q_dim * config.embed_dim);
        @memset(wo, 0.1);

        const ffn_norm = try allocator.alloc(f32, config.embed_dim);
        for (ffn_norm) |*w| w.* = 1.0;

        const w_gate = try allocator.alloc(f32, config.embed_dim * config.ffn_dim);
        @memset(w_gate, 0.1);
        const w_up = try allocator.alloc(f32, config.embed_dim * config.ffn_dim);
        @memset(w_up, 0.1);
        const w_down = try allocator.alloc(f32, config.ffn_dim * config.embed_dim);
        @memset(w_down, 0.1);

        layer_weights[i] = transformer.TransformerWeights{
            .allocator = allocator,
            .attn_norm = attn_norm,
            .wq = .{ .f32 = wq },
            .wk = .{ .f32 = wk },
            .wv = .{ .f32 = wv },
            .wo = .{ .f32 = wo },
            .ffn_norm = ffn_norm,
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
    }

    // Note: LlamaWeights.deinit() is called by LlamaModel.deinit() which frees
    // all layer weights. Do NOT add a defer block to free them here as that
    // would cause a double-free.

    const weights = llama.LlamaWeights{
        .token_embedding = matrix_ops.Weight{ .f32 = token_embedding },
        .output_norm = output_norm,
        .output_weight = matrix_ops.Weight{ .f32 = output_weight },
        .layer_weights = layer_weights,
        .allocator = allocator,
    };

    // Create simple tokenizer
    // Note: We create a dummy model ONLY to pass metadata to the tokenizer.
    // We use empty slices and undefined file since we won't actually load from it.
    // Since deinit() would try to close the undefined file, we manually clean up
    // the small allocations we made instead of calling deinit().
    const vocab_tokens_slice = try allocator.alloc([]u8, 0);
    defer allocator.free(vocab_tokens_slice);
    const vocab_scores_slice = try allocator.alloc(f32, 0);
    defer allocator.free(vocab_scores_slice);
    const tensors_slice = try allocator.alloc(gguf.TensorInfo, 0);
    defer allocator.free(tensors_slice);

    var dummy_model = gguf.GGUFModel{
        .allocator = allocator,
        .file = undefined,
        .header = undefined,
        .metadata = .{
            .architecture = .Llama,
            .vocab_size = config.vocab_size,
            .n_layers = config.n_layers,
            .n_heads = config.n_heads,
            .n_kv_heads = config.n_kv_heads,
            .hidden_size = config.embed_dim,
            .intermediate_size = config.ffn_dim,
            .max_seq_len = config.max_seq_len,
            .rope_theta = 10000.0,
            .rms_norm_eps = config.rms_norm_eps,
            .conv_kernel = 3,
        },
        .vocab_tokens = vocab_tokens_slice,
        .vocab_scores = vocab_scores_slice,
        .tensors = tensors_slice,
        .tensor_data_offset = 0,
    };
    // Do NOT call dummy_model.deinit() - it would try to close an undefined file handle

    // Note: LlamaModel.init() takes ownership of the tokenizer and will call deinit on it.
    // Do NOT defer tok.deinit() here as that would cause a double-free.
    const tok = try tokenizer.Tokenizer.loadFromModel(allocator, &dummy_model);

    std.debug.print("   ✅ Test model created\n", .{});

    std.debug.print("\n2️⃣  Initializing batch model...\n", .{});

    var model = try llama.LlamaModel.init(allocator, config, weights, tok);
    defer model.deinit();

    const batch_config = batch_processor.BatchConfig{
        .max_batch_size = 8,
        .enable_parallel = false,
    };

    var batch_model = try batch_processor.BatchLlamaModel.init(
        allocator,
        &model,
        batch_config,
    );
    defer batch_model.deinit();

    std.debug.print("   ✅ Batch model initialized\n", .{});

    std.debug.print("\n3️⃣  Testing batch forward pass...\n", .{});

    const token_ids = [_]u32{ 1, 2, 3, 4 };
    const positions = [_]u32{ 0, 1, 2, 3 };

    const logits = try batch_model.forwardBatch(&token_ids, &positions);
    defer allocator.free(logits);

    const expected_size = token_ids.len * config.vocab_size;
    if (logits.len != expected_size) {
        std.debug.print("   ❌ Logits size mismatch: {d} vs {d}\n", .{ logits.len, expected_size });
        return error.TestFailed;
    }

    std.debug.print("   Logits size: {d} (batch={d} × vocab={d})\n", .{
        logits.len,
        token_ids.len,
        config.vocab_size,
    });
    std.debug.print("   ✅ Batch forward pass working\n", .{});

    std.debug.print("\n✅ Batch model integration tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
