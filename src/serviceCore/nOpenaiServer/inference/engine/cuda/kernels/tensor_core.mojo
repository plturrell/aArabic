"""
Tensor Core Attention Implementation for CUDA
FP16 Q @ K^T matmul and FP16 softmax with FP32 accumulation.
Optimized for T4/A100/H100 GPU architectures.

FFI Bindings: cublasGemmEx, softmax_fp16_fp32_accum_kernel
"""

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer

from .matmul_cuda import (CublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, CUDA_R_16F,
                          CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)


struct TensorCoreAttention:
    """Tensor Core accelerated attention computation."""
    var _cublas: CublasHandle
    var _lib: DLHandle
    var _head_dim: Int32
    var _scale: Float32

    fn __init__(inout self, cublas_lib: String = "libcublas.so",
                kernel_lib: String = "libcuda_kernels.so", head_dim: Int32 = 64) raises:
        self._cublas = CublasHandle(cublas_lib)
        self._lib = DLHandle(kernel_lib)
        self._head_dim = head_dim
        self._scale = 1.0 / Float32(head_dim).sqrt()

    fn compute_qk_scores(self, Q: UnsafePointer[Float16], K: UnsafePointer[Float16],
                        scores: UnsafePointer[Float16], batch_size: Int32, num_heads: Int32,
                        seq_len: Int32, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Compute Q @ K^T attention scores using Tensor Cores with FP32 accumulation."""
        var M = seq_len
        var N = seq_len
        var K_dim = self._head_dim
        var stride = seq_len * self._head_dim
        var out_stride = seq_len * seq_len

        for b in range(batch_size * num_heads):
            var q_ptr = Q.offset(b * stride)
            var k_ptr = K.offset(b * stride)
            var s_ptr = scores.offset(b * out_stride)
            var status = self._cublas.gemm_ex(
                CUBLAS_OP_T, CUBLAS_OP_N, N, M, K_dim, self._scale,
                k_ptr.bitcast[NoneType](), CUDA_R_16F, K_dim,
                q_ptr.bitcast[NoneType](), CUDA_R_16F, K_dim,
                0.0, s_ptr.bitcast[NoneType](), CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            if status != 0:
                return status
        return 0

    fn softmax_fp16_fp32_accum(self, input: UnsafePointer[Float16], output: UnsafePointer[Float16],
                               batch_size: Int32, seq_len: Int32,
                               stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Softmax with FP16 I/O but FP32 accumulation for numerical stability."""
        return external_call["softmax_fp16_fp32_accum_kernel", Int32](input, output, batch_size, seq_len, stream)

    fn apply_attention_weights(self, attn_weights: UnsafePointer[Float16], V: UnsafePointer[Float16],
                               output: UnsafePointer[Float16], batch_size: Int32, num_heads: Int32,
                               seq_len: Int32, stream: UnsafePointer[NoneType] = UnsafePointer[NoneType]()) raises -> Int32:
        """Apply attention weights to values: output = attn_weights @ V."""
        var M = seq_len
        var N = self._head_dim
        var K_dim = seq_len
        var attn_stride = seq_len * seq_len
        var v_stride = seq_len * self._head_dim

        for b in range(batch_size * num_heads):
            var a_ptr = attn_weights.offset(b * attn_stride)
            var v_ptr = V.offset(b * v_stride)
            var o_ptr = output.offset(b * v_stride)
            var status = self._cublas.gemm_ex(
                CUBLAS_OP_N, CUBLAS_OP_N, N, M, K_dim, 1.0,
                v_ptr.bitcast[NoneType](), CUDA_R_16F, N,
                a_ptr.bitcast[NoneType](), CUDA_R_16F, K_dim,
                0.0, o_ptr.bitcast[NoneType](), CUDA_R_16F, N,
                CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            if status != 0:
                return status
        return 0


fn attention_forward(Q: UnsafePointer[Float16], K: UnsafePointer[Float16], V: UnsafePointer[Float16],
                    output: UnsafePointer[Float16], attn_scores: UnsafePointer[Float16],
                    batch_size: Int32, num_heads: Int32, seq_len: Int32, head_dim: Int32,
                    attention: TensorCoreAttention) raises -> Int32:
    """Full attention forward pass with Tensor Core acceleration."""
    var status = attention.compute_qk_scores(Q, K, attn_scores, batch_size, num_heads, seq_len)
    if status != 0:
        return status
    status = attention.softmax_fp16_fp32_accum(attn_scores, attn_scores, batch_size * num_heads, seq_len)
    if status != 0:
        return status
    return attention.apply_attention_weights(attn_scores, V, output, batch_size, num_heads, seq_len)


# ============================================================================
# C-Compatible Export Functions for Zig FFI
# ============================================================================

# Global context storage for exported functions
var _global_tensor_core_ctx: UnsafePointer[TensorCoreAttention] = UnsafePointer[TensorCoreAttention]()


@export
fn mojo_tensor_core_attention_create(cublas_lib: UnsafePointer[UInt8], kernel_lib: UnsafePointer[UInt8],
                                      head_dim: Int32) -> UnsafePointer[NoneType]:
    """
    C-compatible export: Create TensorCoreAttention context.

    Called from Zig as: mojo_tensor_core_attention_create(cublas_lib, kernel_lib, head_dim)

    Args:
        cublas_lib: Path to cuBLAS library (C string)
        kernel_lib: Path to CUDA kernels library (C string)
        head_dim: Attention head dimension (e.g., 64, 128)

    Returns:
        Opaque pointer to TensorCoreAttention context, or null on failure
    """
    try:
        var cublas_path = String(cublas_lib)
        var kernel_path = String(kernel_lib)
        var ctx_ptr = UnsafePointer[TensorCoreAttention].alloc(1)
        ctx_ptr.init_pointee_move(TensorCoreAttention(cublas_path, kernel_path, head_dim))
        return ctx_ptr.bitcast[NoneType]()
    except:
        return UnsafePointer[NoneType]()


@export
fn mojo_tensor_core_attention_destroy(ctx: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Destroy TensorCoreAttention context.

    Called from Zig as: mojo_tensor_core_attention_destroy(ctx)

    Args:
        ctx: Opaque pointer to TensorCoreAttention context

    Returns:
        0 on success, -1 on failure
    """
    if not ctx:
        return -1
    try:
        var typed_ctx = ctx.bitcast[TensorCoreAttention]()
        typed_ctx.destroy_pointee()
        typed_ctx.free()
        return 0
    except:
        return -1


@export
fn mojo_compute_qk_scores(ctx: UnsafePointer[NoneType], Q: UnsafePointer[Float16],
                          K: UnsafePointer[Float16], scores: UnsafePointer[Float16],
                          batch_size: Int32, num_heads: Int32, seq_len: Int32,
                          stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Compute Q @ K^T attention scores.

    Called from Zig as: mojo_compute_qk_scores(ctx, Q, K, scores, batch_size, num_heads, seq_len, stream)

    Computes scaled dot-product attention scores using Tensor Cores with FP32 accumulation.
    scores = (Q @ K^T) * scale, where scale = 1/sqrt(head_dim)

    Args:
        ctx: Opaque pointer to TensorCoreAttention context
        Q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        K: Key tensor [batch_size, num_heads, seq_len, head_dim]
        scores: Output scores tensor [batch_size, num_heads, seq_len, seq_len]
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        stream: CUDA stream for async execution

    Returns:
        0 on success, non-zero CUDA/cuBLAS error code on failure
    """
    if not ctx:
        return -1
    try:
        var typed_ctx = ctx.bitcast[TensorCoreAttention]()
        return typed_ctx[].compute_qk_scores(Q, K, scores, batch_size, num_heads, seq_len, stream)
    except:
        return -1


@export
fn mojo_apply_attention_weights(ctx: UnsafePointer[NoneType], attn_weights: UnsafePointer[Float16],
                                 V: UnsafePointer[Float16], output: UnsafePointer[Float16],
                                 batch_size: Int32, num_heads: Int32, seq_len: Int32,
                                 stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Apply attention weights to values (attn @ V).

    Called from Zig as: mojo_apply_attention_weights(ctx, attn_weights, V, output, batch_size, num_heads, seq_len, stream)

    Computes output = attn_weights @ V using Tensor Cores with FP32 accumulation.

    Args:
        ctx: Opaque pointer to TensorCoreAttention context
        attn_weights: Attention weights [batch_size, num_heads, seq_len, seq_len]
        V: Value tensor [batch_size, num_heads, seq_len, head_dim]
        output: Output tensor [batch_size, num_heads, seq_len, head_dim]
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        stream: CUDA stream for async execution

    Returns:
        0 on success, non-zero CUDA/cuBLAS error code on failure
    """
    if not ctx:
        return -1
    try:
        var typed_ctx = ctx.bitcast[TensorCoreAttention]()
        return typed_ctx[].apply_attention_weights(attn_weights, V, output, batch_size, num_heads, seq_len, stream)
    except:
        return -1


@export
fn mojo_attention_forward(ctx: UnsafePointer[NoneType], Q: UnsafePointer[Float16],
                          K: UnsafePointer[Float16], V: UnsafePointer[Float16],
                          output: UnsafePointer[Float16], attn_scores: UnsafePointer[Float16],
                          batch_size: Int32, num_heads: Int32, seq_len: Int32, head_dim: Int32,
                          stream: UnsafePointer[NoneType]) -> Int32:
    """
    C-compatible export: Full attention forward pass.

    Called from Zig as: mojo_attention_forward(ctx, Q, K, V, output, attn_scores, batch_size, num_heads, seq_len, head_dim, stream)

    Computes full attention: output = softmax(Q @ K^T / sqrt(head_dim)) @ V
    Uses Tensor Cores for matrix multiplications with FP32 accumulation.

    Args:
        ctx: Opaque pointer to TensorCoreAttention context
        Q: Query tensor [batch_size, num_heads, seq_len, head_dim]
        K: Key tensor [batch_size, num_heads, seq_len, head_dim]
        V: Value tensor [batch_size, num_heads, seq_len, head_dim]
        output: Output tensor [batch_size, num_heads, seq_len, head_dim]
        attn_scores: Scratch buffer for attention scores [batch_size, num_heads, seq_len, seq_len]
        batch_size: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Head dimension
        stream: CUDA stream for async execution

    Returns:
        0 on success, non-zero CUDA/cuBLAS error code on failure
    """
    if not ctx:
        return -1
    try:
        var typed_ctx = ctx.bitcast[TensorCoreAttention]()
        # Compute Q @ K^T scores
        var status = typed_ctx[].compute_qk_scores(Q, K, attn_scores, batch_size, num_heads, seq_len, stream)
        if status != 0:
            return status
        # Apply softmax with FP32 accumulation
        status = typed_ctx[].softmax_fp16_fp32_accum(attn_scores, attn_scores, batch_size * num_heads, seq_len, stream)
        if status != 0:
            return status
        # Apply attention weights to values
        return typed_ctx[].apply_attention_weights(attn_scores, V, output, batch_size, num_heads, seq_len, stream)
    except:
        return -1
