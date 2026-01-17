const std = @import("std");
const lfm2 = @import("lfm2_model");
const tokenizer = @import("tokenizer");
const gguf = @import("gguf_loader");
const matrix_ops = @import("matrix_ops");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    try test_lfm2_smoke(gpa.allocator());
}

fn test_lfm2_smoke(allocator: std.mem.Allocator) !void {
    const vocab: u32 = 8;
    const n_layers: u32 = 2;
    const hidden: u32 = 16;
    const head_dim: u32 = 8;
    const heads: u32 = 2;
    const ffn: u32 = 32;
    const kernel: u32 = 3;

    // Build config slices that Lfm2Model will free.
    const layer_types = try allocator.alloc(lfm2.LayerType, n_layers);
    for (layer_types, 0..) |*lt, idx| {
        lt.* = if (idx % 2 == 0) .conv else .ffn;
    }
    const full_attn_idxs = try allocator.alloc(u32, 1);
    full_attn_idxs[0] = 0;

    const cfg = lfm2.Lfm2Config{
        .vocab_size = vocab,
        .n_layers = n_layers,
        .hidden_size = hidden,
        .intermediate_size = ffn,
        .n_heads = heads,
        .n_kv_heads = heads,
        .head_dim = head_dim,
        .max_seq_len = 16,
        .rope_theta = 1_000_000.0,
        .norm_eps = 1e-5,
        .conv_kernel = kernel,
        .conv_bias = false,
        .layer_types = layer_types,
        .full_attn_idxs = full_attn_idxs,
    };

    // Tokenizer with dummy vocab.
    var dummy_model = gguf.GGUFModel{
        .allocator = allocator,
        .file = undefined,
        .header = undefined,
        .metadata = .{
            .architecture = .Lfm2,
            .vocab_size = vocab,
            .n_layers = n_layers,
            .n_heads = heads,
            .n_kv_heads = heads,
            .hidden_size = hidden,
            .intermediate_size = ffn,
            .max_seq_len = cfg.max_seq_len,
            .rope_theta = cfg.rope_theta,
            .rms_norm_eps = cfg.norm_eps,
            .conv_kernel = cfg.conv_kernel,
        },
        .tensors = &[_]gguf.TensorInfo{},
        .vocab_tokens = &[_][]u8{},
        .vocab_scores = &[_]f32{},
    };
    const tok = try tokenizer.Tokenizer.loadFromModel(allocator, &dummy_model);

    // Shared embedding/head
    const emb_len = vocab * hidden;
    const token_embedding_data = try allocator.alloc(f32, emb_len);
    @memset(token_embedding_data, 0.01);
    const head_weight = matrix_ops.Weight{ .f32 = token_embedding_data };

    // Layer weights
    const layers = try allocator.alloc(lfm2.Lfm2LayerWeights, n_layers);

    for (layers, 0..) |*lw, idx| {
        const conv_weight = try allocator.alloc(f32, kernel * hidden);
        @memset(conv_weight, 0.02);
        const in_proj = try allocator.alloc(f32, hidden * hidden * 2);
        @memset(in_proj, 0.01);
        const out_proj = try allocator.alloc(f32, hidden * hidden);
        @memset(out_proj, 0.01);

        const attn_norm = try allocator.alloc(f32, hidden);
        @memset(attn_norm, 1.0);
        const has_attn = (idx == 0);

        var q_norm: []f32 = &[_]f32{};
        var k_norm: []f32 = &[_]f32{};
        var wq_weight: matrix_ops.Weight = .{ .f32 = &[_]f32{} };
        var wk_weight: matrix_ops.Weight = .{ .f32 = &[_]f32{} };
        var wv_weight: matrix_ops.Weight = .{ .f32 = &[_]f32{} };
        var wo_weight: matrix_ops.Weight = .{ .f32 = &[_]f32{} };

        if (has_attn) {
            q_norm = try allocator.alloc(f32, head_dim);
            k_norm = try allocator.alloc(f32, head_dim);
            @memset(q_norm, 1.0);
            @memset(k_norm, 1.0);

            const q_dim = heads * head_dim;
            const kv_dim = heads * head_dim;
            const wq = try allocator.alloc(f32, hidden * q_dim);
            const wk = try allocator.alloc(f32, hidden * kv_dim);
            const wv = try allocator.alloc(f32, hidden * kv_dim);
            const wo = try allocator.alloc(f32, q_dim * hidden);
            @memset(wq, 0.01);
            @memset(wk, 0.01);
            @memset(wv, 0.01);
            @memset(wo, 0.01);
            wq_weight = .{ .f32 = wq };
            wk_weight = .{ .f32 = wk };
            wv_weight = .{ .f32 = wv };
            wo_weight = .{ .f32 = wo };
        }

        const ffn_norm = try allocator.alloc(f32, hidden);
        @memset(ffn_norm, 1.0);
        const ffn_gate = try allocator.alloc(f32, hidden * ffn);
        const ffn_up = try allocator.alloc(f32, hidden * ffn);
        const ffn_down = try allocator.alloc(f32, ffn * hidden);
        @memset(ffn_gate, 0.01);
        @memset(ffn_up, 0.01);
        @memset(ffn_down, 0.01);

        lw.* = lfm2.Lfm2LayerWeights{
            .conv_weight = conv_weight,
            .in_proj = .{ .f32 = in_proj },
            .out_proj = .{ .f32 = out_proj },
            .attn_norm = attn_norm,
            .has_attn = has_attn,
            .wq = wq_weight,
            .wk = wk_weight,
            .wv = wv_weight,
            .wo = wo_weight,
            .q_norm = q_norm,
            .k_norm = k_norm,
            .ffn_gate = .{ .f32 = ffn_gate },
            .ffn_up = .{ .f32 = ffn_up },
            .ffn_down = .{ .f32 = ffn_down },
            .ffn_norm = ffn_norm,
        };
    }

    const weights = lfm2.Lfm2Weights{
        .allocator = allocator,
        .token_embedding = head_weight,
        .output_weight = head_weight,
        .output_tied = true,
        .layers = layers,
    };

    var model = try lfm2.Lfm2Model.init(allocator, cfg, weights, tok);
    defer model.deinit();

    const logits = try model.forwardToken(1, 0);
    defer allocator.free(logits);

    try std.testing.expectEqual(@as(usize, vocab), logits.len);
    for (logits) |v| {
        try std.testing.expect(!std.math.isNan(v));
    }
}
