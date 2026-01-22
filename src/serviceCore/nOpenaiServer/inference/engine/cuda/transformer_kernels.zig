// Transformer Kernel Bindings for Zig
// FFI layer for CUDA transformer operations (RMSNorm, SiLU, Softmax)
//
// Links against: libtransformer_kernels.so
// Build: cd cuda/kernels && make
//
// These kernels implement the missing GPU operations for complete transformer forward pass.

const std = @import("std");
const cuda = @import("cuda_bindings");
const GpuTensor = @import("gpu_tensor").GpuTensor;

const log = std.log.scoped(.transformer_kernels);

// ============================================================================
// External CUDA Transformer Kernels (from libtransformer_kernels.so)
// All kernels use FP16 (f16) to match GpuTensor format
// ============================================================================

/// RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight
pub extern "transformer_kernels" fn cuda_rms_norm(
    input: [*]const f16,
    weight: [*]const f16,
    output: [*]f16,
    size: c_int,
    eps: f32,
    stream: ?*anyopaque,
) c_int;

/// SiLU activation: y = x * sigmoid(x)
pub extern "transformer_kernels" fn cuda_silu(
    input: [*]const f16,
    output: [*]f16,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Fused SiLU + elementwise multiply: out = silu(gate) * up
pub extern "transformer_kernels" fn cuda_silu_mul(
    gate: [*]const f16,
    up: [*]const f16,
    output: [*]f16,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Elementwise multiply: out = a * b
pub extern "transformer_kernels" fn cuda_elementwise_mul(
    a: [*]const f16,
    b: [*]const f16,
    output: [*]f16,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Vector add: out = a + b
pub extern "transformer_kernels" fn cuda_vector_add(
    a: [*]const f16,
    b: [*]const f16,
    output: [*]f16,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Row-wise softmax: each row gets exp(x - max) / sum
pub extern "transformer_kernels" fn cuda_softmax(
    data: [*]f16,
    rows: c_int,
    cols: c_int,
    stream: ?*anyopaque,
) c_int;

/// Apply Rotary Position Embedding (RoPE) to Q and K tensors
pub extern "transformer_kernels" fn cuda_apply_rope(
    q: [*]f16,
    k: [*]f16,
    position: c_int,
    head_dim: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    rope_theta: f32,
    stream: ?*anyopaque,
) c_int;

/// Compute attention scores: Q @ K^T / sqrt(head_dim)
pub extern "transformer_kernels" fn cuda_attention_scores(
    q: [*]const f16,
    k_cache: [*]const f16,
    scores: [*]f16,
    position: c_int,
    head_dim: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    max_seq_len: c_int,
    stream: ?*anyopaque,
) c_int;

/// Apply causal mask to attention scores
pub extern "transformer_kernels" fn cuda_apply_causal_mask(
    scores: [*]f16,
    n_heads: c_int,
    seq_len: c_int,
    current_pos: c_int,
    stream: ?*anyopaque,
) c_int;

/// Compute attention output: scores @ V (attention weighted sum)
pub extern "transformer_kernels" fn cuda_attention_output(
    scores: [*]const f16,
    v_cache: [*]const f16,
    output: [*]f16,
    position: c_int,
    head_dim: c_int,
    n_heads: c_int,
    n_kv_heads: c_int,
    max_seq_len: c_int,
    stream: ?*anyopaque,
) c_int;

/// Copy K and V to KV cache at the given position
pub extern "transformer_kernels" fn cuda_copy_to_kv_cache(
    k: [*]const f16,
    v: [*]const f16,
    k_cache: [*]f16,
    v_cache: [*]f16,
    position: c_int,
    head_dim: c_int,
    n_kv_heads: c_int,
    max_seq_len: c_int,
    stream: ?*anyopaque,
) c_int;

// ============================================================================
// High-Level Wrapper Functions with Error Handling
// ============================================================================

pub const TransformerKernelError = error{
    RMSNormFailed,
    SiLUFailed,
    SiLUMulFailed,
    ElementwiseMulFailed,
    VectorAddFailed,
    SoftmaxFailed,
    RoPEFailed,
    AttentionScoresFailed,
    CausalMaskFailed,
    AttentionOutputFailed,
    KVCacheCopyFailed,
};

/// Apply RMSNorm to input tensor
/// output = input * rsqrt(mean(input²) + eps) * weight
pub fn rmsNorm(
    output: *GpuTensor,
    input: *const GpuTensor,
    weight: *const GpuTensor,
    eps: f32,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const size = input.len;
    
    const result = cuda_rms_norm(
        @ptrCast(@alignCast(input.ptr)),
        @ptrCast(@alignCast(weight.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(size),
        eps,
        stream,
    );
    
    if (result != 0) {
        log.err("cuda_rms_norm failed with code: {}", .{result});
        return TransformerKernelError.RMSNormFailed;
    }
}

/// Apply SiLU activation: output = input * sigmoid(input)
pub fn silu(
    output: *GpuTensor,
    input: *const GpuTensor,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const size = input.len;

    const result = cuda_silu(
        @ptrCast(@alignCast(input.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(size),
        stream,
    );
    
    if (result != 0) {
        log.err("cuda_silu failed with code: {}", .{result});
        return TransformerKernelError.SiLUFailed;
    }
}

/// Fused SiLU + multiply: output = silu(gate) * up
pub fn siluMul(
    output: *GpuTensor,
    gate: *const GpuTensor,
    up: *const GpuTensor,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const size = gate.len;

    const result = cuda_silu_mul(
        @ptrCast(@alignCast(gate.ptr)),
        @ptrCast(@alignCast(up.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(size),
        stream,
    );

    if (result != 0) {
        log.err("cuda_silu_mul failed with code: {}", .{result});
        return TransformerKernelError.SiLUMulFailed;
    }
}

/// Elementwise multiply: output = a * b
pub fn elementwiseMul(
    output: *GpuTensor,
    a: *const GpuTensor,
    b: *const GpuTensor,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const size = a.len;

    const result = cuda_elementwise_mul(
        @ptrCast(@alignCast(a.ptr)),
        @ptrCast(@alignCast(b.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(size),
        stream,
    );

    if (result != 0) {
        log.err("cuda_elementwise_mul failed with code: {}", .{result});
        return TransformerKernelError.ElementwiseMulFailed;
    }
}

/// Vector add: output = a + b
pub fn vectorAdd(
    output: *GpuTensor,
    a: *const GpuTensor,
    b: *const GpuTensor,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const size = a.len;

    const result = cuda_vector_add(
        @ptrCast(@alignCast(a.ptr)),
        @ptrCast(@alignCast(b.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(size),
        stream,
    );

    if (result != 0) {
        log.err("cuda_vector_add failed with code: {}", .{result});
        return TransformerKernelError.VectorAddFailed;
    }
}

/// Row-wise softmax (in-place)
/// Each row: out[i] = exp(x[i] - max) / sum(exp(x - max))
pub fn softmax(
    data: *GpuTensor,
    rows: usize,
    cols: usize,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_softmax(
        @ptrCast(@alignCast(data.ptr)),
        @intCast(rows),
        @intCast(cols),
        stream,
    );

    if (result != 0) {
        log.err("cuda_softmax failed with code: {}", .{result});
        return TransformerKernelError.SoftmaxFailed;
    }
}

/// Apply Rotary Position Embedding (RoPE) to Q and K tensors in-place
/// Rotates Q and K vectors based on position for relative position encoding
pub fn applyRope(
    q: *GpuTensor,
    k: *GpuTensor,
    position: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    rope_theta: f32,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_apply_rope(
        @ptrCast(@alignCast(q.ptr)),
        @ptrCast(@alignCast(k.ptr)),
        @intCast(position),
        @intCast(head_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        rope_theta,
        stream,
    );

    if (result != 0) {
        log.err("cuda_apply_rope failed with code: {}", .{result});
        return TransformerKernelError.RoPEFailed;
    }
}

/// Compute attention scores: scores = Q @ K^T / sqrt(head_dim)
/// Uses KV cache for efficient incremental inference
pub fn attentionScores(
    scores: *GpuTensor,
    q: *const GpuTensor,
    k_cache: *const GpuTensor,
    position: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_attention_scores(
        @ptrCast(@alignCast(q.ptr)),
        @ptrCast(@alignCast(k_cache.ptr)),
        @ptrCast(@alignCast(scores.ptr)),
        @intCast(position),
        @intCast(head_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(max_seq_len),
        stream,
    );

    if (result != 0) {
        log.err("cuda_attention_scores failed with code: {}", .{result});
        return TransformerKernelError.AttentionScoresFailed;
    }
}

/// Apply causal mask to attention scores (in-place)
/// Sets future positions to -inf to prevent attending to future tokens
pub fn applyCausalMask(
    scores: *GpuTensor,
    n_heads: usize,
    seq_len: usize,
    current_pos: usize,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_apply_causal_mask(
        @ptrCast(@alignCast(scores.ptr)),
        @intCast(n_heads),
        @intCast(seq_len),
        @intCast(current_pos),
        stream,
    );

    if (result != 0) {
        log.err("cuda_apply_causal_mask failed with code: {}", .{result});
        return TransformerKernelError.CausalMaskFailed;
    }
}

/// Compute attention output: output = scores @ V
/// Computes weighted sum of values based on attention scores
pub fn attentionOutput(
    output: *GpuTensor,
    scores: *const GpuTensor,
    v_cache: *const GpuTensor,
    position: usize,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_attention_output(
        @ptrCast(@alignCast(scores.ptr)),
        @ptrCast(@alignCast(v_cache.ptr)),
        @ptrCast(@alignCast(output.ptr)),
        @intCast(position),
        @intCast(head_dim),
        @intCast(n_heads),
        @intCast(n_kv_heads),
        @intCast(max_seq_len),
        stream,
    );

    if (result != 0) {
        log.err("cuda_attention_output failed with code: {}", .{result});
        return TransformerKernelError.AttentionOutputFailed;
    }
}

/// Copy K and V to their positions in the KV cache
/// K layout: [n_kv_heads * head_dim] -> cache[n_kv_heads, max_seq_len, head_dim]
pub fn copyToKVCache(
    k: *const GpuTensor,
    v: *const GpuTensor,
    k_cache: *GpuTensor,
    v_cache: *GpuTensor,
    position: usize,
    head_dim: usize,
    n_kv_heads: usize,
    max_seq_len: usize,
    stream: ?*anyopaque,
) TransformerKernelError!void {
    const result = cuda_copy_to_kv_cache(
        @ptrCast(@alignCast(k.ptr)),
        @ptrCast(@alignCast(v.ptr)),
        @ptrCast(@alignCast(k_cache.ptr)),
        @ptrCast(@alignCast(v_cache.ptr)),
        @intCast(position),
        @intCast(head_dim),
        @intCast(n_kv_heads),
        @intCast(max_seq_len),
        stream,
    );

    if (result != 0) {
        log.err("cuda_copy_to_kv_cache failed with code: {}", .{result});
        return TransformerKernelError.KVCacheCopyFailed;
    }
}

// ============================================================================
// Test
// ============================================================================

test "transformer kernels linkage" {
    // This test verifies that the extern declarations link correctly
    // Actual kernel execution requires a CUDA-capable GPU
    _ = cuda_rms_norm;
    _ = cuda_silu;
    _ = cuda_silu_mul;
    _ = cuda_elementwise_mul;
    _ = cuda_vector_add;
    _ = cuda_softmax;
    _ = cuda_apply_rope;
    _ = cuda_attention_scores;
    _ = cuda_apply_causal_mask;
    _ = cuda_attention_output;
    _ = cuda_copy_to_kv_cache;
}

