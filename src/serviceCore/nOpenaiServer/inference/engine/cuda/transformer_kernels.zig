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
// ============================================================================

/// RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight
pub extern "transformer_kernels" fn cuda_rms_norm(
    input: [*]const f32,
    weight: [*]const f32,
    output: [*]f32,
    size: c_int,
    eps: f32,
    stream: ?*anyopaque,
) c_int;

/// SiLU activation: y = x * sigmoid(x)
pub extern "transformer_kernels" fn cuda_silu(
    input: [*]const f32,
    output: [*]f32,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Fused SiLU + elementwise multiply: out = silu(gate) * up
pub extern "transformer_kernels" fn cuda_silu_mul(
    gate: [*]const f32,
    up: [*]const f32,
    output: [*]f32,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Elementwise multiply: out = a * b
pub extern "transformer_kernels" fn cuda_elementwise_mul(
    a: [*]const f32,
    b: [*]const f32,
    output: [*]f32,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Vector add: out = a + b
pub extern "transformer_kernels" fn cuda_vector_add(
    a: [*]const f32,
    b: [*]const f32,
    output: [*]f32,
    size: c_int,
    stream: ?*anyopaque,
) c_int;

/// Row-wise softmax: each row gets exp(x - max) / sum
pub extern "transformer_kernels" fn cuda_softmax(
    data: [*]f32,
    rows: c_int,
    cols: c_int,
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
}

