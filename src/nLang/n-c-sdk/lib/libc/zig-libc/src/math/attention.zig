// Attention mechanisms for neural networks - Pure Zig implementations
// Includes scaled dot-product, causal, multi-head, and flash attention variants

const std = @import("std");
const math = std.math;

// ============================================================================
// Constants
// ============================================================================

const NEG_INF: f32 = -math.inf(f32);

// ============================================================================
// Scaled Dot-Product Attention
// ============================================================================

/// Compute attention scores: scores = Q * K^T * scale
/// q: Query matrix [seq_len_q x head_dim]
/// k: Key matrix [seq_len_kv x head_dim]
/// seq_len_q: Query sequence length
/// seq_len_kv: Key/Value sequence length
/// head_dim: Dimension of each head
/// scale: Scaling factor (typically 1/sqrt(head_dim))
/// scores: Output matrix [seq_len_q x seq_len_kv]
pub export fn n_attention_scores(
    q: [*]const f32,
    k: [*]const f32,
    seq_len_q: c_int,
    seq_len_kv: c_int,
    head_dim: c_int,
    scale: f32,
    scores: [*]f32,
) void {
    if (seq_len_q <= 0 or seq_len_kv <= 0 or head_dim <= 0) return;
    const sq: usize = @intCast(seq_len_q);
    const skv: usize = @intCast(seq_len_kv);
    const hd: usize = @intCast(head_dim);

    // scores[i][j] = sum_d(q[i][d] * k[j][d]) * scale
    for (0..sq) |i| {
        for (0..skv) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            scores[i * skv + j] = dot * scale;
        }
    }
}

/// Scaled dot-product attention: output = softmax(Q * K^T / sqrt(d_k)) * V
/// q: Query matrix [seq_len x head_dim]
/// k: Key matrix [seq_len x head_dim]
/// v: Value matrix [seq_len x head_dim]
/// output: Output matrix [seq_len x head_dim]
/// seq_len: Sequence length
/// head_dim: Dimension of each head
pub export fn n_scaled_dot_product_attention(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    output: [*]f32,
    seq_len: c_int,
    head_dim: c_int,
) void {
    if (seq_len <= 0 or head_dim <= 0) return;
    const sl: usize = @intCast(seq_len);
    const hd: usize = @intCast(head_dim);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // For each query position
    for (0..sl) |i| {
        // Compute attention scores for this query
        var max_score: f32 = NEG_INF;
        for (0..sl) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            const score = dot * scale;
            if (score > max_score) max_score = score;
        }

        // Compute softmax and weighted sum simultaneously
        var sum_exp: f32 = 0.0;
        // First pass: compute exp and sum
        for (0..sl) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            sum_exp += @exp(dot * scale - max_score);
        }

        // Initialize output to zero
        for (0..hd) |d| {
            output[i * hd + d] = 0.0;
        }

        // Second pass: weighted sum
        for (0..sl) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            const weight = @exp(dot * scale - max_score) / sum_exp;
            for (0..hd) |d| {
                output[i * hd + d] += weight * v[j * hd + d];
            }
        }
    }
}

// ============================================================================
// Causal (Autoregressive) Attention
// ============================================================================

/// Apply causal mask to attention scores
/// Sets future positions (j > i) to -inf
/// scores: Attention scores matrix [seq_len x seq_len] (modified in place)
/// seq_len: Sequence length
pub export fn n_apply_causal_mask(scores: [*]f32, seq_len: c_int) void {
    if (seq_len <= 0) return;
    const sl: usize = @intCast(seq_len);

    for (0..sl) |i| {
        for (i + 1..sl) |j| {
            scores[i * sl + j] = NEG_INF;
        }
    }
}

/// Causal attention: masked attention where position i can only attend to positions <= i
/// q: Query matrix [seq_len x head_dim]
/// k: Key matrix [seq_len x head_dim]
/// v: Value matrix [seq_len x head_dim]
/// output: Output matrix [seq_len x head_dim]
/// seq_len: Sequence length
/// head_dim: Dimension of each head
pub export fn n_causal_attention(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    output: [*]f32,
    seq_len: c_int,
    head_dim: c_int,
) void {
    if (seq_len <= 0 or head_dim <= 0) return;
    const sl: usize = @intCast(seq_len);
    const hd: usize = @intCast(head_dim);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // For each query position
    for (0..sl) |i| {
        // Only attend to positions <= i (causal mask)
        var max_score: f32 = NEG_INF;
        for (0..i + 1) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            const score = dot * scale;
            if (score > max_score) max_score = score;
        }

        // Handle edge case where max_score is still -inf
        if (max_score == NEG_INF) max_score = 0.0;

        var sum_exp: f32 = 0.0;
        for (0..i + 1) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            sum_exp += @exp(dot * scale - max_score);
        }

        // Initialize output to zero
        for (0..hd) |d| {
            output[i * hd + d] = 0.0;
        }

        // Weighted sum over attended positions
        for (0..i + 1) |j| {
            var dot: f32 = 0.0;
            for (0..hd) |d| {
                dot += q[i * hd + d] * k[j * hd + d];
            }
            const weight = @exp(dot * scale - max_score) / sum_exp;
            for (0..hd) |d| {
                output[i * hd + d] += weight * v[j * hd + d];
            }
        }
    }
}

// ============================================================================
// Multi-Head Attention Helpers
// ============================================================================

/// Split input tensor for multi-head attention
/// Reshapes from [batch x seq_len x (n_heads * head_dim)] to [batch x n_heads x seq_len x head_dim]
/// input: Input tensor [batch * seq_len * n_heads * head_dim]
/// batch: Batch size
/// seq_len: Sequence length
/// n_heads: Number of attention heads
/// head_dim: Dimension of each head
/// output: Output tensor [batch * n_heads * seq_len * head_dim]
pub export fn n_split_heads(
    input: [*]const f32,
    batch: c_int,
    seq_len: c_int,
    n_heads: c_int,
    head_dim: c_int,
    output: [*]f32,
) void {
    if (batch <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0) return;
    const b: usize = @intCast(batch);
    const sl: usize = @intCast(seq_len);
    const nh: usize = @intCast(n_heads);
    const hd: usize = @intCast(head_dim);

    // input: [b, sl, nh, hd] -> output: [b, nh, sl, hd]
    for (0..b) |bi| {
        for (0..sl) |si| {
            for (0..nh) |hi| {
                for (0..hd) |di| {
                    const in_idx = bi * sl * nh * hd + si * nh * hd + hi * hd + di;
                    const out_idx = bi * nh * sl * hd + hi * sl * hd + si * hd + di;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

/// Merge heads back after multi-head attention
/// Reshapes from [batch x n_heads x seq_len x head_dim] to [batch x seq_len x (n_heads * head_dim)]
/// input: Input tensor [batch * n_heads * seq_len * head_dim]
/// batch: Batch size
/// seq_len: Sequence length
/// n_heads: Number of attention heads
/// head_dim: Dimension of each head
/// output: Output tensor [batch * seq_len * n_heads * head_dim]
pub export fn n_merge_heads(
    input: [*]const f32,
    batch: c_int,
    seq_len: c_int,
    n_heads: c_int,
    head_dim: c_int,
    output: [*]f32,
) void {
    if (batch <= 0 or seq_len <= 0 or n_heads <= 0 or head_dim <= 0) return;
    const b: usize = @intCast(batch);
    const sl: usize = @intCast(seq_len);
    const nh: usize = @intCast(n_heads);
    const hd: usize = @intCast(head_dim);

    // input: [b, nh, sl, hd] -> output: [b, sl, nh, hd]
    for (0..b) |bi| {
        for (0..nh) |hi| {
            for (0..sl) |si| {
                for (0..hd) |di| {
                    const in_idx = bi * nh * sl * hd + hi * sl * hd + si * hd + di;
                    const out_idx = bi * sl * nh * hd + si * nh * hd + hi * hd + di;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}


// ============================================================================
// Flash Attention (Memory-Efficient)
// ============================================================================

/// Block size for flash attention tiling
pub const FLASH_BLOCK_SIZE: usize = 64;

/// Flash attention context with workspace buffers
/// Uses tiled computation to reduce memory from O(n²) to O(n)
pub const FlashAttentionContext = extern struct {
    /// Workspace for storing partial row max values
    row_max: [*]f32,
    /// Workspace for storing partial row sum values
    row_sum: [*]f32,
    /// Workspace for temporary score storage (block_size x block_size)
    score_block: [*]f32,
    /// Block size used for tiling
    block_size: c_int,
};

/// Initialize flash attention context
pub export fn n_flash_attention_init(
    ctx: *FlashAttentionContext,
    row_max: [*]f32,
    row_sum: [*]f32,
    score_block: [*]f32,
    block_size: c_int,
) void {
    ctx.row_max = row_max;
    ctx.row_sum = row_sum;
    ctx.score_block = score_block;
    ctx.block_size = block_size;
}

/// Flash attention forward pass with tiled computation
/// Uses online softmax to compute attention without materializing full attention matrix
/// ctx: Flash attention context with workspace buffers
/// q: Query matrix [seq_len_q x head_dim]
/// k: Key matrix [seq_len_kv x head_dim]
/// v: Value matrix [seq_len_kv x head_dim]
/// output: Output matrix [seq_len_q x head_dim]
/// seq_len_q: Query sequence length
/// seq_len_kv: Key/Value sequence length
/// head_dim: Dimension of each head
pub export fn n_flash_attention_forward(
    ctx: *FlashAttentionContext,
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    output: [*]f32,
    seq_len_q: c_int,
    seq_len_kv: c_int,
    head_dim: c_int,
) void {
    if (seq_len_q <= 0 or seq_len_kv <= 0 or head_dim <= 0) return;
    const sq: usize = @intCast(seq_len_q);
    const skv: usize = @intCast(seq_len_kv);
    const hd: usize = @intCast(head_dim);
    const bs: usize = @intCast(ctx.block_size);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // Initialize output and statistics
    for (0..sq) |i| {
        ctx.row_max[i] = NEG_INF;
        ctx.row_sum[i] = 0.0;
        for (0..hd) |d| {
            output[i * hd + d] = 0.0;
        }
    }

    // Process in blocks
    const num_kv_blocks = (skv + bs - 1) / bs;

    for (0..num_kv_blocks) |kv_block| {
        const kv_start = kv_block * bs;
        const kv_end = @min(kv_start + bs, skv);

        // For each query position
        for (0..sq) |i| {
            const old_max = ctx.row_max[i];
            var new_max = old_max;

            // Compute scores for this block and find new max
            for (kv_start..kv_end) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[i * hd + d] * k[j * hd + d];
                }
                const score = dot * scale;
                ctx.score_block[(i % bs) * bs + (j - kv_start)] = score;
                if (score > new_max) new_max = score;
            }

            // Update running statistics using online softmax
            const old_sum = ctx.row_sum[i];
            var block_sum: f32 = 0.0;

            // Compute exp for this block
            for (kv_start..kv_end) |j| {
                const score = ctx.score_block[(i % bs) * bs + (j - kv_start)];
                block_sum += @exp(score - new_max);
            }

            // Rescale old sum and combine
            const rescale = @exp(old_max - new_max);
            const new_sum = old_sum * rescale + block_sum;

            // Rescale existing output
            for (0..hd) |d| {
                output[i * hd + d] *= old_sum * rescale / new_sum;
            }

            // Add contribution from this block
            for (kv_start..kv_end) |j| {
                const score = ctx.score_block[(i % bs) * bs + (j - kv_start)];
                const weight = @exp(score - new_max) / new_sum;
                for (0..hd) |d| {
                    output[i * hd + d] += weight * v[j * hd + d];
                }
            }

            ctx.row_max[i] = new_max;
            ctx.row_sum[i] = new_sum;
        }
    }
}

// ============================================================================
// Online Softmax Utilities
// ============================================================================

/// Update online softmax running statistics
/// Given old max and sum, update with new scores
/// m_old: Previous maximum value
/// l_old: Previous sum of exponentials
/// scores: New scores to incorporate [n]
/// n: Number of new scores
/// m_new: Output - new maximum value
/// l_new: Output - new sum of exponentials
pub export fn n_online_softmax_update(
    m_old: f32,
    l_old: f32,
    scores: [*]const f32,
    n: c_int,
    m_new: *f32,
    l_new: *f32,
) void {
    if (n <= 0) {
        m_new.* = m_old;
        l_new.* = l_old;
        return;
    }
    const nu: usize = @intCast(n);

    // Find maximum in new scores
    var block_max: f32 = NEG_INF;
    for (0..nu) |i| {
        if (scores[i] > block_max) block_max = scores[i];
    }

    // Compute new global max
    const global_max = @max(m_old, block_max);

    // Compute rescaled sum
    const old_sum_rescaled = l_old * @exp(m_old - global_max);
    var block_sum: f32 = 0.0;
    for (0..nu) |i| {
        block_sum += @exp(scores[i] - global_max);
    }

    m_new.* = global_max;
    l_new.* = old_sum_rescaled + block_sum;
}

/// Finalize online softmax by dividing by the sum
/// output: Values to normalize (modified in place) [n]
/// l: Sum of exponentials
/// n: Number of elements
pub export fn n_online_softmax_finalize(output: [*]f32, l: f32, n: c_int) void {
    if (n <= 0 or l == 0.0) return;
    const nu: usize = @intCast(n);
    const inv_l = 1.0 / l;

    for (0..nu) |i| {
        output[i] *= inv_l;
    }
}


// ============================================================================
// Attention Variants
// ============================================================================

/// Multi-Query Attention (MQA): 1 KV head shared across multiple Q heads
/// q: Query matrix [n_heads x seq_len x head_dim]
/// k: Key matrix [seq_len x head_dim] (single head)
/// v: Value matrix [seq_len x head_dim] (single head)
/// output: Output matrix [n_heads x seq_len x head_dim]
/// n_heads: Number of query heads
/// seq_len: Sequence length
/// head_dim: Dimension of each head
pub export fn n_multi_query_attention(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    output: [*]f32,
    n_heads: c_int,
    seq_len: c_int,
    head_dim: c_int,
) void {
    if (n_heads <= 0 or seq_len <= 0 or head_dim <= 0) return;
    const nh: usize = @intCast(n_heads);
    const sl: usize = @intCast(seq_len);
    const hd: usize = @intCast(head_dim);
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // For each head
    for (0..nh) |h| {
        // For each query position
        for (0..sl) |i| {
            const q_offset = h * sl * hd + i * hd;

            // Find max score for numerical stability
            var max_score: f32 = NEG_INF;
            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[j * hd + d];
                }
                const score = dot * scale;
                if (score > max_score) max_score = score;
            }

            // Compute softmax denominator
            var sum_exp: f32 = 0.0;
            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[j * hd + d];
                }
                sum_exp += @exp(dot * scale - max_score);
            }

            // Compute weighted sum
            const out_offset = h * sl * hd + i * hd;
            for (0..hd) |d| {
                output[out_offset + d] = 0.0;
            }

            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[j * hd + d];
                }
                const weight = @exp(dot * scale - max_score) / sum_exp;
                for (0..hd) |d| {
                    output[out_offset + d] += weight * v[j * hd + d];
                }
            }
        }
    }
}

/// Grouped Query Attention (GQA): KV heads shared across groups of Q heads
/// q: Query matrix [n_heads x seq_len x head_dim]
/// k: Key matrix [n_kv_heads x seq_len x head_dim]
/// v: Value matrix [n_kv_heads x seq_len x head_dim]
/// output: Output matrix [n_heads x seq_len x head_dim]
/// n_heads: Number of query heads
/// n_kv_heads: Number of key/value heads (n_heads must be divisible by n_kv_heads)
/// seq_len: Sequence length
/// head_dim: Dimension of each head
pub export fn n_grouped_query_attention(
    q: [*]const f32,
    k: [*]const f32,
    v: [*]const f32,
    output: [*]f32,
    n_heads: c_int,
    n_kv_heads: c_int,
    seq_len: c_int,
    head_dim: c_int,
) void {
    if (n_heads <= 0 or n_kv_heads <= 0 or seq_len <= 0 or head_dim <= 0) return;
    if (@mod(@as(usize, @intCast(n_heads)), @as(usize, @intCast(n_kv_heads))) != 0) return;

    const nh: usize = @intCast(n_heads);
    const nkv: usize = @intCast(n_kv_heads);
    const sl: usize = @intCast(seq_len);
    const hd: usize = @intCast(head_dim);
    const heads_per_kv = nh / nkv;
    const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hd)));

    // For each head
    for (0..nh) |h| {
        const kv_head = h / heads_per_kv; // Which KV head this Q head uses
        const kv_offset = kv_head * sl * hd;

        // For each query position
        for (0..sl) |i| {
            const q_offset = h * sl * hd + i * hd;

            // Find max score
            var max_score: f32 = NEG_INF;
            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[kv_offset + j * hd + d];
                }
                const score = dot * scale;
                if (score > max_score) max_score = score;
            }

            // Compute softmax denominator
            var sum_exp: f32 = 0.0;
            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[kv_offset + j * hd + d];
                }
                sum_exp += @exp(dot * scale - max_score);
            }

            // Compute weighted sum
            const out_offset = h * sl * hd + i * hd;
            for (0..hd) |d| {
                output[out_offset + d] = 0.0;
            }

            for (0..sl) |j| {
                var dot: f32 = 0.0;
                for (0..hd) |d| {
                    dot += q[q_offset + d] * k[kv_offset + j * hd + d];
                }
                const weight = @exp(dot * scale - max_score) / sum_exp;
                for (0..hd) |d| {
                    output[out_offset + d] += weight * v[kv_offset + j * hd + d];
                }
            }
        }
    }
}


// ============================================================================
// Tests
// ============================================================================

test "attention_scores basic" {
    const seq_len = 2;
    const head_dim = 2;
    var q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var scores: [seq_len * seq_len]f32 = undefined;

    n_attention_scores(&q, &k, seq_len, seq_len, head_dim, 1.0, &scores);

    // q[0] dot k[0] = 1, q[0] dot k[1] = 0
    // q[1] dot k[0] = 0, q[1] dot k[1] = 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), scores[2], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[3], 1e-5);
}

test "scaled_dot_product_attention identity" {
    const seq_len = 2;
    const head_dim = 2;
    // Q = K = I (identity-like), V = some values
    var q = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [seq_len * head_dim]f32 = undefined;

    n_scaled_dot_product_attention(&q, &k, &v, &output, seq_len, head_dim);

    // Output should be weighted average of values
    // First query attends more to first key, second to second key
    // Check that output is reasonable (between min and max of values)
    for (0..seq_len * head_dim) |i| {
        try std.testing.expect(output[i] >= 0.5 and output[i] <= 5.0);
    }
}

test "causal_mask" {
    const seq_len = 3;
    var scores = [_]f32{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };

    n_apply_causal_mask(&scores, seq_len);

    // Row 0: only [0] should remain, [1] and [2] should be -inf
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[0], 1e-5);
    try std.testing.expect(scores[1] == NEG_INF);
    try std.testing.expect(scores[2] == NEG_INF);

    // Row 1: [0] and [1] remain, [2] is -inf
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[3], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[4], 1e-5);
    try std.testing.expect(scores[5] == NEG_INF);

    // Row 2: all remain
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[6], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[7], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), scores[8], 1e-5);
}

test "split_and_merge_heads_roundtrip" {
    const batch = 1;
    const seq_len = 2;
    const n_heads = 2;
    const head_dim = 2;
    const total = batch * seq_len * n_heads * head_dim;

    var input: [total]f32 = undefined;
    for (0..total) |i| {
        input[i] = @floatFromInt(i);
    }
    var split_out: [total]f32 = undefined;
    var merged: [total]f32 = undefined;

    n_split_heads(&input, batch, seq_len, n_heads, head_dim, &split_out);
    n_merge_heads(&split_out, batch, seq_len, n_heads, head_dim, &merged);

    // Should be identity after roundtrip
    for (0..total) |i| {
        try std.testing.expectApproxEqAbs(input[i], merged[i], 1e-5);
    }
}

test "online_softmax_update" {
    var scores = [_]f32{ 1.0, 2.0, 3.0 };
    var m_new: f32 = undefined;
    var l_new: f32 = undefined;

    n_online_softmax_update(NEG_INF, 0.0, &scores, 3, &m_new, &l_new);

    try std.testing.expectApproxEqAbs(@as(f32, 3.0), m_new, 1e-5);
    // l_new = exp(1-3) + exp(2-3) + exp(3-3) = exp(-2) + exp(-1) + 1
    const expected_l = @exp(@as(f32, -2.0)) + @exp(@as(f32, -1.0)) + 1.0;
    try std.testing.expectApproxEqAbs(expected_l, l_new, 1e-5);
}

test "multi_query_attention basic" {
    const n_heads = 2;
    const seq_len = 2;
    const head_dim = 2;

    // Q: [2 heads x 2 seq x 2 dim]
    var q = [_]f32{
        1.0, 0.0, 0.0, 1.0, // head 0
        1.0, 0.0, 0.0, 1.0, // head 1
    };
    // K, V: single head
    var k = [_]f32{ 1.0, 0.0, 0.0, 1.0 };
    var v = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var output: [n_heads * seq_len * head_dim]f32 = undefined;

    n_multi_query_attention(&q, &k, &v, &output, n_heads, seq_len, head_dim);

    // Both heads should produce similar output since they use same K,V
    for (0..seq_len * head_dim) |i| {
        try std.testing.expectApproxEqAbs(output[i], output[seq_len * head_dim + i], 1e-5);
    }
}

test "grouped_query_attention basic" {
    const n_heads = 4;
    const n_kv_heads = 2;
    const seq_len = 2;
    const head_dim = 2;

    var q: [n_heads * seq_len * head_dim]f32 = undefined;
    for (0..n_heads * seq_len * head_dim) |i| {
        q[i] = 0.5;
    }
    var k: [n_kv_heads * seq_len * head_dim]f32 = undefined;
    var v: [n_kv_heads * seq_len * head_dim]f32 = undefined;
    for (0..n_kv_heads * seq_len * head_dim) |i| {
        k[i] = 0.5;
        v[i] = @floatFromInt(i);
    }
    var output: [n_heads * seq_len * head_dim]f32 = undefined;

    n_grouped_query_attention(&q, &k, &v, &output, n_heads, n_kv_heads, seq_len, head_dim);

    // Heads 0,1 share KV head 0; heads 2,3 share KV head 1
    // Outputs for heads 0,1 should be identical
    for (0..seq_len * head_dim) |i| {
        try std.testing.expectApproxEqAbs(output[i], output[seq_len * head_dim + i], 1e-5);
    }
    // Outputs for heads 2,3 should be identical
    for (0..seq_len * head_dim) |i| {
        try std.testing.expectApproxEqAbs(
            output[2 * seq_len * head_dim + i],
            output[3 * seq_len * head_dim + i],
            1e-5,
        );
    }
}
