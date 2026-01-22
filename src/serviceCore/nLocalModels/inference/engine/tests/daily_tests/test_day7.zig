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
    const token_embedding = try allocator.alloc(f32, config.vocab_size * config.embed_dim);
    defer allocator.free(token_embedding);
    @memset(token_embedding, 0.1);

    const output_norm = try allocator.alloc(f32, config.embed_dim);
    defer allocator.free(output_norm);
    for (output_norm) |*w| w.* = 1.0;

    const output_weight = try allocator.alloc(f32, config.embed_dim * config.vocab_size);
    defer allocator.free(output_weight);
    @memset(output_weight, 0.1);

    // Create layer weights
    const layer_weights = try allocator.alloc(transformer.TransformerWeights, config.n_layers);
    defer allocator.free(layer_weights);

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

    defer {
        for (layer_weights) |lw| {
            allocator.free(lw.attn_norm);
            switch (lw.wq) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            switch (lw.wk) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            switch (lw.wv) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            switch (lw.wo) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            allocator.free(lw.ffn_norm);
            switch (lw.w_gate) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            switch (lw.w_up) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
            switch (lw.w_down) {
                .f32 => |s| allocator.free(s),
                else => {},
            }
        }
    }

    const weights = llama.LlamaWeights{
        .token_embedding = matrix_ops.Weight{ .f32 = token_embedding },
        .output_norm = output_norm,
        .output_weight = matrix_ops.Weight{ .f32 = output_weight },
        .layer_weights = layer_weights,
        .allocator = allocator,
    };

    // Create simple tokenizer
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
        .vocab_tokens = try allocator.alloc([]u8, 0),
        .vocab_scores = try allocator.alloc(f32, 0),
        .tensors = try allocator.alloc(gguf.TensorInfo, 0),
        .tensor_data_offset = 0,
    };
    defer dummy_model.deinit();

    var tok = try tokenizer.Tokenizer.loadFromModel(allocator, &dummy_model);
    defer tok.deinit();

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
