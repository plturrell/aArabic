const std = @import("std");
const matrix_ops = @import("matrix_ops");
const kv_cache = @import("kv_cache");
const thread_pool = @import("thread_pool");
const compute = @import("compute");
const gguf = @import("gguf_loader");
const config_parser = @import("config_parser");

/// Multi-head self-attention with KV caching for Llama models
/// Implements scaled dot-product attention with RoPE position encoding
/// Supports RoPE scaling for context window extension (linear, dynamic NTK, YaRN)

// ============================================================================ 
// Structures
// ============================================================================ 

pub const AttentionConfig = struct {
    n_heads: u32,
    n_kv_heads: u32,  // For grouped-query attention
    head_dim: u32,
    rope_theta: f32 = 10000.0,
    rope_scaling: ?config_parser.RopeScalingConfig = null,
};

pub const AttentionWeights = struct {
    wq: matrix_ops.Weight,  // Query projection [n_heads * head_dim, embed_dim]
    wk: matrix_ops.Weight,  // Key projection [n_kv_heads * head_dim, embed_dim]
    wv: matrix_ops.Weight,  // Value projection [n_kv_heads * head_dim, embed_dim]
    wo: matrix_ops.Weight,  // Output projection [embed_dim, n_heads * head_dim]
};

/// Backend parameters for GPU matmul operations
pub const BackendParams = struct {
    data: []const u8,
    quant_type: gguf.QuantizationType,
};

/// Convert a matrix_ops.Weight to backend parameters (data pointer and quantization type)
/// for use with ComputeBackend.matmul()
pub fn weightToBackendParams(weight: matrix_ops.Weight) BackendParams {
    return switch (weight) {
        .f32 => |data| BackendParams{
            .data = std.mem.sliceAsBytes(data),
            .quant_type = .F32,
        },
        .q4_0 => |data| BackendParams{
            .data = data,
            .quant_type = .Q4_0,
        },
        .q4_k => |data| BackendParams{
            .data = data,
            .quant_type = .Q4_K,
        },
        .q6_k => |data| BackendParams{
            .data = data,
            .quant_type = .Q6_K,
        },
    };
}

// ============================================================================
// RoPE (Rotary Position Embedding) with Scaling Support
// ============================================================================

/// Compute scaled RoPE frequency for a given position
/// Supports linear, dynamic NTK-aware, and YaRN scaling methods
fn computeScaledFreq(
    base_freq: f32,
    position: u32,
    dim_idx: usize,
    head_dim: u32,
    scaling_config: ?config_parser.RopeScalingConfig,
) f32 {
    const config = scaling_config orelse return base_freq;
    
    const pos_f = @as(f32, @floatFromInt(position));
    const original_max = @as(f32, @floatFromInt(config.original_max_position_embeddings));
    
    return switch (config.type) {
        .none => base_freq,
        
        .linear => base_freq / config.factor,
        
        .dynamic => blk: {
            // Dynamic NTK-aware scaling (Code Llama style)
            // Only applies scaling beyond original context window
            if (position < config.original_max_position_embeddings) {
                break :blk base_freq;
            }

            // Scale factor based on sequence length ratio
            // scale = (current_length / original_length) ^ (dim / (dim - 2))
            const length_ratio = pos_f / original_max;
            const dim_f = @as(f32, @floatFromInt(head_dim));

            // NTK-aware interpolation exponent
            const exponent = dim_f / (dim_f - 2.0);
            const ntk_scale = std.math.pow(f32, length_ratio, exponent);

            break :blk base_freq / ntk_scale;
        },

        .yarn => blk: {
            // YaRN (Yet another RoPE extensioN method)
            // Most sophisticated, interpolates between low and high frequency dimensions
            if (position < config.original_max_position_embeddings) {
                break :blk base_freq;
            }

            const dim_f = @as(f32, @floatFromInt(head_dim));
            const dim_idx_f = @as(f32, @floatFromInt(dim_idx * 2));

            // YaRN parameters with defaults
            const attn_factor = config.attention_factor orelse 1.0;
            const beta_fast = config.beta_fast orelse 32.0;
            const beta_slow = config.beta_slow orelse 1.0;

            // Compute interpolation ramp based on dimension position
            // Low frequencies (high wavelengths) get more scaling
            // High frequencies (low wavelengths) get less scaling
            const ramp = (dim_idx_f / dim_f - beta_fast) / (beta_slow - beta_fast);
            const ramp_clamped = @max(0.0, @min(1.0, ramp));

            // Interpolate scale factor between 1.0 (no scaling) and config.factor
            const scale = config.factor * ramp_clamped + 1.0 * (1.0 - ramp_clamped);

            break :blk base_freq / (scale * attn_factor);
        },
    };
}

/// Precompute RoPE frequencies with optional scaling for position encoding
/// Supports context window extension via linear, dynamic NTK, or YaRN scaling
pub fn precomputeRopeFreqs(
    allocator: std.mem.Allocator,
    head_dim: u32,
    max_seq_len: u32,
    theta: f32,
    scaling_config: ?config_parser.RopeScalingConfig,
) ![]f32 {
    const freq_size = (head_dim / 2) * max_seq_len;
    const freqs = try allocator.alloc(f32, freq_size * 2); // cos and sin
    
    // Log scaling information if present
    if (scaling_config) |sc| {
        std.debug.print("   🔄 RoPE Scaling enabled: {s}\n", .{@tagName(sc.type)});
        std.debug.print("      Factor: {d:.2}x, Original: {d} → Extended: {d}\n", .{
            sc.factor,
            sc.original_max_position_embeddings,
            sc.getExtendedSeqLen(),
        });
    }
    
    // Compute frequencies with scaling
    for (0..head_dim / 2) |i| {
        // Base frequency (unscaled)
        const base_freq = 1.0 / std.math.pow(
            f32,
            theta,
            @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(head_dim))
        );
        
        // For each position
        for (0..max_seq_len) |pos| {
            // Apply scaling if configured
            const scaled_freq = computeScaledFreq(
                base_freq,
                @intCast(pos),
                i,
                head_dim,
                scaling_config,
            );
            
            const angle = @as(f32, @floatFromInt(pos)) * scaled_freq;
            const idx = pos * (head_dim / 2) + i;
            freqs[idx] = @cos(angle);  // cos
            freqs[freq_size + idx] = @sin(angle);  // sin
        }
    }
    
    return freqs;
}

/// Apply RoPE to query or key vectors
/// Uses "split" format (LLaMA/Qwen style): first half and second half of dims
pub fn applyRope(
    output: []f32,
    input: []const f32,
    position: u32,
    rope_freqs: []const f32,
    head_dim: u32,
) void {
    const half_dim = head_dim / 2;
    const freq_size = rope_freqs.len / 2;
    const pos_offset = position * half_dim;

    // Split format: x0 = input[0..half], x1 = input[half..dim]
    // Rotation: out[0..half] = x0 * cos - x1 * sin
    //           out[half..dim] = x1 * cos + x0 * sin
    for (0..half_dim) |i| {
        const cos_val = rope_freqs[pos_offset + i];
        const sin_val = rope_freqs[freq_size + pos_offset + i];

        const x0 = input[i];              // First half
        const x1 = input[half_dim + i];   // Second half

        output[i] = x0 * cos_val - x1 * sin_val;
        output[half_dim + i] = x1 * cos_val + x0 * sin_val;
    }
}

// ============================================================================ 
// Attention Operations
// ============================================================================ 

/// Compute self-attention for a single token
pub fn computeAttention(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: AttentionWeights,
    cache: *kv_cache.KVCache,
    layer: u32,
    position: u32,
    config: AttentionConfig,
    rope_freqs: []const f32,
    pool: ?*thread_pool.ThreadPool,
) !void {
    const embed_dim = input.len;
    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;
    const head_dim = config.head_dim;
    
    // Allocate workspace for projections
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);
    
    // Project to Q, K, V
    try matrix_ops.matmul(q, weights.wq, input, q_dim, 1, embed_dim, allocator, pool);
    try matrix_ops.matmul(k, weights.wk, input, kv_dim, 1, embed_dim, allocator, pool);
    try matrix_ops.matmul(v, weights.wv, input, kv_dim, 1, embed_dim, allocator, pool);
    
    // Apply RoPE to Q and K
    const q_rope = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rope);
    const k_rope = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rope);
    
    // Apply RoPE (Parallelize if needed, but fast enough serial for single token)
    for (0..config.n_heads) |h| {
        const q_head = q[h * head_dim .. (h + 1) * head_dim];
        const q_rope_head = q_rope[h * head_dim .. (h + 1) * head_dim];
        applyRope(q_rope_head, q_head, position, rope_freqs, head_dim);
    }
    
    for (0..config.n_kv_heads) |h| {
        const k_head = k[h * head_dim .. (h + 1) * head_dim];
        const k_rope_head = k_rope[h * head_dim .. (h + 1) * head_dim];
        applyRope(k_rope_head, k_head, position, rope_freqs, head_dim);
    }
    
    // Store K and V in cache
    cache.store(layer, k_rope, v);
    
    // Prepare for attention computation
    const attn_output = try allocator.alloc(f32, q_dim);
    defer allocator.free(attn_output);
    
    const seq_len = cache.getSequenceLength();
    const BLOCK_SIZE = 128;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    
    // Define Head Context for parallel execution
    const HeadContext = struct {
        head_idx: usize,
        q_rope: []const f32,
        cache: *kv_cache.KVCache,
        layer: u32,
        attn_output: []f32,
        allocator: std.mem.Allocator, // Thread-safe allocator needed? Using main allocator.
        config: AttentionConfig,
        seq_len: u32,
        scale: f32,
    };
    
    const compute_head = struct {
        fn run(ctx: *anyopaque) void {
            const context = @as(*HeadContext, @ptrCast(@alignCast(ctx)));
            const h = context.head_idx;
            const head_dim_local = context.config.head_dim; // Rename to avoid capture
            
            // Allocate thread-local buffers (small)
            // K_block: [BLOCK_SIZE, head_dim]
            const k_block = context.allocator.alloc(f32, BLOCK_SIZE * head_dim_local) catch return;
            defer context.allocator.free(k_block);
            
            // V_block: [BLOCK_SIZE, head_dim]
            const v_block = context.allocator.alloc(f32, BLOCK_SIZE * head_dim_local) catch return;
            defer context.allocator.free(v_block);
            
            // Scores: [BLOCK_SIZE]
            const scores = context.allocator.alloc(f32, BLOCK_SIZE) catch return;
            defer context.allocator.free(scores);
            
            // Get query for this head
            const q_head = context.q_rope[h * head_dim_local .. (h + 1) * head_dim_local];
            
            // Map to KV head (GQA)
            const kv_head_idx = h * context.config.n_kv_heads / context.config.n_heads;
            
            // Online Softmax State
            var max_score: f32 = -std.math.inf(f32);
            var sum_exp: f32 = 0.0;
            
            // Output accumulator for this head
            const out_head = context.attn_output[h * head_dim_local .. (h + 1) * head_dim_local];
            @memset(out_head, 0.0);
            
            // Loop over blocks
            var start_pos: u32 = 0;
            while (start_pos < context.seq_len) {
                const end_pos = @min(start_pos + BLOCK_SIZE, context.seq_len);
                const current_block_size = end_pos - start_pos;
                
                // 1. Gather K and V for this block
                context.cache.gatherHeadKeys(context.layer, @intCast(kv_head_idx), start_pos, end_pos, k_block);
                context.cache.gatherHeadValues(context.layer, @intCast(kv_head_idx), start_pos, end_pos, v_block);
                
                // 2. Compute Scores: S = Q . K^T
                // Q: [1, D]. K: [B, D]. S: [1, B].
                // We implement simple dot product loop here for vectorization
                const Vec = @Vector(8, f32);
                
                for (0..current_block_size) |b_idx| {
                    var dot: f32 = 0.0;
                    var vec_dot: Vec = @splat(0.0);
                    
                    const k_ptr = k_block[b_idx * head_dim_local .. (b_idx + 1) * head_dim_local];
                    
                    var d: usize = 0;
                    while (d + 8 <= head_dim_local) : (d += 8) {
                        const vq = @as(Vec, q_head[d..][0..8].*);
                        const vk = @as(Vec, k_ptr[d..][0..8].*);
                        vec_dot += vq * vk;
                    }
                    dot = @reduce(.Add, vec_dot);
                    while (d < head_dim_local) : (d += 1) {
                        dot += q_head[d] * k_ptr[d];
                    }
                    
                    scores[b_idx] = dot * context.scale;
                }
                
                // 3. Online Softmax Update
                var local_max: f32 = -std.math.inf(f32);
                for (scores[0..current_block_size]) |s| {
                    if (s > local_max) local_max = s;
                }
                
                const new_max = @max(max_score, local_max);
                const factor_old = @exp(max_score - new_max);
                _ = @exp(local_max - new_max); // factor_new - scale for new block (unused)

                // Rescale accumulator
                for (out_head) |*val| {
                    val.* *= factor_old;
                }
                sum_exp *= factor_old;
                
                // Accumulate new block
                for (0..current_block_size) |b_idx| {
                    const score = scores[b_idx];
                    const weight = @exp(score - new_max);
                    sum_exp += weight;
                    
                    const v_ptr = v_block[b_idx * head_dim_local .. (b_idx + 1) * head_dim_local];
                    
                    // Vectorized accumulation: out += weight * v
                    const vec_w: Vec = @splat(weight);
                    var d: usize = 0;
                    while (d + 8 <= head_dim_local) : (d += 8) {
                        var vec_o = @as(Vec, out_head[d..][0..8].*);
                        const vec_v = @as(Vec, v_ptr[d..][0..8].*);
                        vec_o += vec_w * vec_v;
                        out_head[d..][0..8].* = vec_o;
                    }
                    while (d < head_dim_local) : (d += 1) {
                        out_head[d] += weight * v_ptr[d];
                    }
                }
                
                max_score = new_max;
                start_pos += BLOCK_SIZE;
            }
            
            // Final normalization
            if (sum_exp > 0.0) {
                const inv_sum = 1.0 / sum_exp;
                matrix_ops.vec_scale(out_head, out_head, inv_sum);
            }
        }
    }.run;
    
    // Execute heads in parallel
    if (pool) |tp| {
        var contexts = try allocator.alloc(HeadContext, config.n_heads);
        defer allocator.free(contexts);
        
        for (0..config.n_heads) |h| {
            contexts[h] = HeadContext{
                .head_idx = h,
                .q_rope = q_rope,
                .cache = cache,
                .layer = layer,
                .attn_output = attn_output,
                .allocator = allocator,
                .config = config,
                .seq_len = seq_len,
                .scale = scale,
            };
            
            tp.submit(.{
                .work_fn = compute_head,
                .context = &contexts[h],
            });
        }
        tp.waitAll();
    } else {
        // Serial fallback
        for (0..config.n_heads) |h| {
            var ctx = HeadContext{
                .head_idx = h,
                .q_rope = q_rope,
                .cache = cache,
                .layer = layer,
                .attn_output = attn_output,
                .allocator = allocator,
                .config = config,
                .seq_len = seq_len,
                .scale = scale,
            };
            compute_head(&ctx);
        }
    }
    
    // Project back to embed_dim
    // wo: [embed, q_dim]. attn_output: [q_dim, 1].
    try matrix_ops.matmul(output, weights.wo, attn_output, embed_dim, 1, q_dim, allocator, pool);
}

/// Compute self-attention for a single token using GPU backend
/// Uses ComputeBackend.matmul() for GPU-accelerated matrix operations
pub fn computeAttentionGpu(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: AttentionWeights,
    cache: *kv_cache.KVCache,
    layer: u32,
    position: u32,
    config: AttentionConfig,
    rope_freqs: []const f32,
    pool: ?*thread_pool.ThreadPool,
    backend: compute.ComputeBackend,
) !void {
    const embed_dim = input.len;
    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;
    const head_dim = config.head_dim;

    // Allocate workspace for projections
    const q = try allocator.alloc(f32, q_dim);
    defer allocator.free(q);
    const k = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k);
    const v = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v);

    // Project to Q, K, V using GPU backend
    const wq_params = weightToBackendParams(weights.wq);
    const wk_params = weightToBackendParams(weights.wk);
    const wv_params = weightToBackendParams(weights.wv);

    try backend.matmul(q, wq_params.data, wq_params.quant_type, input, q_dim, 1, embed_dim);
    try backend.matmul(k, wk_params.data, wk_params.quant_type, input, kv_dim, 1, embed_dim);
    try backend.matmul(v, wv_params.data, wv_params.quant_type, input, kv_dim, 1, embed_dim);

    // Apply RoPE to Q and K
    const q_rope = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_rope);
    const k_rope = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_rope);

    // Apply RoPE (Parallelize if needed, but fast enough serial for single token)
    for (0..config.n_heads) |h| {
        const q_head = q[h * head_dim .. (h + 1) * head_dim];
        const q_rope_head = q_rope[h * head_dim .. (h + 1) * head_dim];
        applyRope(q_rope_head, q_head, position, rope_freqs, head_dim);
    }

    for (0..config.n_kv_heads) |h| {
        const k_head = k[h * head_dim .. (h + 1) * head_dim];
        const k_rope_head = k_rope[h * head_dim .. (h + 1) * head_dim];
        applyRope(k_rope_head, k_head, position, rope_freqs, head_dim);
    }

    // Store K and V in cache
    cache.store(layer, k_rope, v);

    // Prepare for attention computation
    const attn_output = try allocator.alloc(f32, q_dim);
    defer allocator.free(attn_output);

    const seq_len = cache.getSequenceLength();
    const BLOCK_SIZE = 128;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    // Define Head Context for parallel execution
    const HeadContext = struct {
        head_idx: usize,
        q_rope: []const f32,
        cache: *kv_cache.KVCache,
        layer: u32,
        attn_output: []f32,
        allocator: std.mem.Allocator,
        config: AttentionConfig,
        seq_len: u32,
        scale: f32,
    };

    const compute_head = struct {
        fn run(ctx: *anyopaque) void {
            const context = @as(*HeadContext, @ptrCast(@alignCast(ctx)));
            const h = context.head_idx;
            const head_dim_local = context.config.head_dim;

            // Allocate thread-local buffers (small)
            const k_block = context.allocator.alloc(f32, BLOCK_SIZE * head_dim_local) catch return;
            defer context.allocator.free(k_block);

            const v_block = context.allocator.alloc(f32, BLOCK_SIZE * head_dim_local) catch return;
            defer context.allocator.free(v_block);

            const scores = context.allocator.alloc(f32, BLOCK_SIZE) catch return;
            defer context.allocator.free(scores);

            // Get query for this head
            const q_head = context.q_rope[h * head_dim_local .. (h + 1) * head_dim_local];

            // Map to KV head (GQA)
            const kv_head_idx = h * context.config.n_kv_heads / context.config.n_heads;

            // Online Softmax State
            var max_score: f32 = -std.math.inf(f32);
            var sum_exp: f32 = 0.0;

            // Output accumulator for this head
            const out_head = context.attn_output[h * head_dim_local .. (h + 1) * head_dim_local];
            @memset(out_head, 0.0);

            // Loop over blocks
            var start_pos: u32 = 0;
            while (start_pos < context.seq_len) {
                const end_pos = @min(start_pos + BLOCK_SIZE, context.seq_len);
                const current_block_size = end_pos - start_pos;

                // 1. Gather K and V for this block
                context.cache.gatherHeadKeys(context.layer, @intCast(kv_head_idx), start_pos, end_pos, k_block);
                context.cache.gatherHeadValues(context.layer, @intCast(kv_head_idx), start_pos, end_pos, v_block);

                // 2. Compute Scores: S = Q . K^T
                const Vec = @Vector(8, f32);

                for (0..current_block_size) |b_idx| {
                    var dot: f32 = 0.0;
                    var vec_dot: Vec = @splat(0.0);

                    const k_ptr = k_block[b_idx * head_dim_local .. (b_idx + 1) * head_dim_local];

                    var d: usize = 0;
                    while (d + 8 <= head_dim_local) : (d += 8) {
                        const vq = @as(Vec, q_head[d..][0..8].*);
                        const vk = @as(Vec, k_ptr[d..][0..8].*);
                        vec_dot += vq * vk;
                    }
                    dot = @reduce(.Add, vec_dot);
                    while (d < head_dim_local) : (d += 1) {
                        dot += q_head[d] * k_ptr[d];
                    }

                    scores[b_idx] = dot * context.scale;
                }

                // 3. Online Softmax Update
                var local_max: f32 = -std.math.inf(f32);
                for (scores[0..current_block_size]) |s| {
                    if (s > local_max) local_max = s;
                }

                const new_max = @max(max_score, local_max);
                const factor_old = @exp(max_score - new_max);
                _ = @exp(local_max - new_max);

                // Rescale accumulator
                for (out_head) |*val| {
                    val.* *= factor_old;
                }
                sum_exp *= factor_old;

                // Accumulate new block
                for (0..current_block_size) |b_idx| {
                    const score = scores[b_idx];
                    const weight = @exp(score - new_max);
                    sum_exp += weight;

                    const v_ptr = v_block[b_idx * head_dim_local .. (b_idx + 1) * head_dim_local];

                    // Vectorized accumulation: out += weight * v
                    const vec_w: Vec = @splat(weight);
                    var d: usize = 0;
                    while (d + 8 <= head_dim_local) : (d += 8) {
                        var vec_o = @as(Vec, out_head[d..][0..8].*);
                        const vec_v = @as(Vec, v_ptr[d..][0..8].*);
                        vec_o += vec_w * vec_v;
                        out_head[d..][0..8].* = vec_o;
                    }
                    while (d < head_dim_local) : (d += 1) {
                        out_head[d] += weight * v_ptr[d];
                    }
                }

                max_score = new_max;
                start_pos += BLOCK_SIZE;
            }

            // Final normalization
            if (sum_exp > 0.0) {
                const inv_sum = 1.0 / sum_exp;
                matrix_ops.vec_scale(out_head, out_head, inv_sum);
            }
        }
    }.run;

    // Execute heads in parallel
    if (pool) |tp| {
        var contexts = try allocator.alloc(HeadContext, config.n_heads);
        defer allocator.free(contexts);

        for (0..config.n_heads) |h| {
            contexts[h] = HeadContext{
                .head_idx = h,
                .q_rope = q_rope,
                .cache = cache,
                .layer = layer,
                .attn_output = attn_output,
                .allocator = allocator,
                .config = config,
                .seq_len = seq_len,
                .scale = scale,
            };

            tp.submit(.{
                .work_fn = compute_head,
                .context = &contexts[h],
            });
        }
        tp.waitAll();
    } else {
        // Serial fallback
        for (0..config.n_heads) |h| {
            var ctx = HeadContext{
                .head_idx = h,
                .q_rope = q_rope,
                .cache = cache,
                .layer = layer,
                .attn_output = attn_output,
                .allocator = allocator,
                .config = config,
                .seq_len = seq_len,
                .scale = scale,
            };
            compute_head(&ctx);
        }
    }

    // Project back to embed_dim using GPU backend
    const wo_params = weightToBackendParams(weights.wo);
    try backend.matmul(output, wo_params.data, wo_params.quant_type, attn_output, embed_dim, 1, q_dim);
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_attention(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Attention\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const config = AttentionConfig{
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .rope_theta = 10000.0,
    };
    
    const embed_dim: u32 = 64;
    const max_seq_len: u32 = 32;
    
    // Test 3: Attention with dummy weights
    {
        const q_dim = config.n_heads * config.head_dim;
        const kv_dim = config.n_kv_heads * config.head_dim;
        
        const wq = try allocator.alloc(f32, embed_dim * q_dim);
        defer allocator.free(wq);
        const wk = try allocator.alloc(f32, embed_dim * kv_dim);
        defer allocator.free(wk);
        const wv = try allocator.alloc(f32, embed_dim * kv_dim);
        defer allocator.free(wv);
        const wo = try allocator.alloc(f32, q_dim * embed_dim);
        defer allocator.free(wo);
        
        @memset(wq, 1.0);
        @memset(wk, 1.0);
        @memset(wv, 1.0);
        @memset(wo, 1.0);
        
        const weights = AttentionWeights{
            .wq = .{ .f32 = wq },
            .wk = .{ .f32 = wk },
            .wv = .{ .f32 = wv },
            .wo = .{ .f32 = wo },
        };
        
        var cache = try kv_cache.KVCache.init(allocator, 1, config.n_kv_heads, config.head_dim, max_seq_len);
        defer cache.deinit();
        
        const freqs = try precomputeRopeFreqs(
            allocator,
            config.head_dim,
            max_seq_len,
            config.rope_theta,
            null, // No scaling for tests
        );
        defer allocator.free(freqs);
        
        const input = try allocator.alloc(f32, embed_dim);
        defer allocator.free(input);
        @memset(input, 1.0);
        
        const output = try allocator.alloc(f32, embed_dim);
        defer allocator.free(output);
        
        try computeAttention(
            allocator,
            output,
            input,
            weights,
            &cache,
            0,
            0,
            config,
            freqs,
            null, // No pool for tests
        );
    }
}
