// cuBLAS API bindings for Zig
// FFI layer for interfacing with NVIDIA cuBLAS library
//
// Links against: libcublas.so
// Requires: CUDA Toolkit 11.8+ or 12.x
//
// Provides: GEMM operations with Tensor Core support via cublasGemmEx

const std = @import("std");
const cuda = @import("cuda_bindings.zig");

// ============================================================================
// cuBLAS Types and Constants
// ============================================================================

/// Opaque cuBLAS handle type
pub const cublasHandle_t = *anyopaque;

/// cuBLAS operation types
pub const cublasOperation_t = c_int;
pub const CUBLAS_OP_N: cublasOperation_t = 0; // No transpose
pub const CUBLAS_OP_T: cublasOperation_t = 1; // Transpose
pub const CUBLAS_OP_C: cublasOperation_t = 2; // Conjugate transpose

/// CUDA data types for cublasGemmEx
pub const cudaDataType_t = c_int;
pub const CUDA_R_16F: cudaDataType_t = 2; // FP16 (half)
pub const CUDA_R_32F: cudaDataType_t = 0; // FP32 (float)
pub const CUDA_R_16BF: cudaDataType_t = 14; // BF16
pub const CUDA_R_8I: cudaDataType_t = 3; // INT8

/// cuBLAS compute types
pub const cublasComputeType_t = c_int;
pub const CUBLAS_COMPUTE_16F: cublasComputeType_t = 64; // FP16 compute
pub const CUBLAS_COMPUTE_32F: cublasComputeType_t = 68; // FP32 compute
pub const CUBLAS_COMPUTE_32F_FAST_16F: cublasComputeType_t = 74; // FP32 accumulate with FP16 input (Tensor Cores)
pub const CUBLAS_COMPUTE_32F_FAST_TF32: cublasComputeType_t = 77; // TF32 Tensor Cores (Ampere+)

/// cuBLAS GEMM algorithms
pub const cublasGemmAlgo_t = c_int;
pub const CUBLAS_GEMM_DEFAULT: cublasGemmAlgo_t = -1;
pub const CUBLAS_GEMM_DEFAULT_TENSOR_OP: cublasGemmAlgo_t = 99; // Use Tensor Cores

/// cuBLAS status codes
pub const cublasStatus_t = c_int;
pub const CUBLAS_STATUS_SUCCESS: cublasStatus_t = 0;
pub const CUBLAS_STATUS_NOT_INITIALIZED: cublasStatus_t = 1;
pub const CUBLAS_STATUS_ALLOC_FAILED: cublasStatus_t = 3;
pub const CUBLAS_STATUS_INVALID_VALUE: cublasStatus_t = 7;
pub const CUBLAS_STATUS_ARCH_MISMATCH: cublasStatus_t = 8;
pub const CUBLAS_STATUS_EXECUTION_FAILED: cublasStatus_t = 13;
pub const CUBLAS_STATUS_INTERNAL_ERROR: cublasStatus_t = 14;
pub const CUBLAS_STATUS_NOT_SUPPORTED: cublasStatus_t = 15;

// ============================================================================
// cuBLAS Core Functions (link against libcublas.so)
// ============================================================================

/// Create cuBLAS handle
pub extern "cublas" fn cublasCreate_v2(handle: *cublasHandle_t) cublasStatus_t;

/// Destroy cuBLAS handle
pub extern "cublas" fn cublasDestroy_v2(handle: cublasHandle_t) cublasStatus_t;

/// Set cuBLAS stream
pub extern "cublas" fn cublasSetStream_v2(handle: cublasHandle_t, stream: ?*anyopaque) cublasStatus_t;

/// Get cuBLAS stream
pub extern "cublas" fn cublasGetStream_v2(handle: cublasHandle_t, stream: *?*anyopaque) cublasStatus_t;

// ============================================================================
// GEMM Operations
// ============================================================================

/// FP32 matrix multiplication: C = alpha * A @ B + beta * C
/// A[m,k], B[k,n], C[m,n] - Column-major layout
pub extern "cublas" fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    A: *const f32,
    lda: c_int,
    B: *const f32,
    ldb: c_int,
    beta: *const f32,
    C: *f32,
    ldc: c_int,
) cublasStatus_t;

/// FP16 matrix multiplication: C = alpha * A @ B + beta * C
/// Uses Tensor Cores on Turing+ GPUs
pub extern "cublas" fn cublasHgemm(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f16,
    A: *const f16,
    lda: c_int,
    B: *const f16,
    ldb: c_int,
    beta: *const f16,
    C: *f16,
    ldc: c_int,
) cublasStatus_t;

/// Mixed precision GEMM with Tensor Core support
/// Allows different input/output/compute types
pub extern "cublas" fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const anyopaque,
    A: *const anyopaque,
    Atype: cudaDataType_t,
    lda: c_int,
    B: *const anyopaque,
    Btype: cudaDataType_t,
    ldb: c_int,
    beta: *const anyopaque,
    C: *anyopaque,
    Ctype: cudaDataType_t,
    ldc: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) cublasStatus_t;

// ============================================================================
// Helper Functions
// ============================================================================

pub fn checkCublasError(status: cublasStatus_t, comptime context: []const u8) !void {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std.debug.print("cuBLAS Error in {s}: status code {d}\n", .{ context, status });
        return error.CublasError;
    }
}

/// Get error string for cuBLAS status
pub fn getCublasErrorString(status: cublasStatus_t) []const u8 {
    return switch (status) {
        CUBLAS_STATUS_SUCCESS => "CUBLAS_STATUS_SUCCESS",
        CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED",
        CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED",
        CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE",
        CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH",
        CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED",
        CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR",
        CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED",
        else => "CUBLAS_STATUS_UNKNOWN",
    };
}

// ============================================================================
// High-Level cuBLAS Context
// ============================================================================

/// cuBLAS context wrapper with RAII-style resource management
pub const CublasContext = struct {
    handle: cublasHandle_t,
    stream: ?*anyopaque,
    use_tensor_cores: bool,

    const Self = @This();

    /// Initialize cuBLAS context
    pub fn init(use_tensor_cores: bool) !Self {
        var handle: cublasHandle_t = undefined;
        try checkCublasError(cublasCreate_v2(&handle), "cublasCreate_v2");

        return Self{
            .handle = handle,
            .stream = null,
            .use_tensor_cores = use_tensor_cores,
        };
    }

    /// Deinitialize cuBLAS context
    pub fn deinit(self: *Self) void {
        _ = cublasDestroy_v2(self.handle);
    }

    /// Set CUDA stream for async operations
    pub fn setStream(self: *Self, stream: ?*anyopaque) !void {
        try checkCublasError(cublasSetStream_v2(self.handle, stream), "cublasSetStream_v2");
        self.stream = stream;
    }

    /// FP32 GEMM: C[m,n] = A[m,k] @ B[k,n]
    /// Row-major input, cuBLAS expects column-major, so we compute C^T = B^T @ A^T
    pub fn sgemm(
        self: *const Self,
        c: [*]f32,
        a: [*]const f32,
        b: [*]const f32,
        m: usize,
        n: usize,
        k: usize,
    ) !void {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;

        // For row-major: C = A @ B is equivalent to C^T = B^T @ A^T in col-major
        // cuBLAS: C[n,m] = B[n,k] @ A[k,m] with lda=k, ldb=n, ldc=n
        const status = cublasSgemm_v2(
            self.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            @intCast(n), // m in cuBLAS terms
            @intCast(m), // n in cuBLAS terms
            @intCast(k),
            &alpha,
            b, // B matrix
            @intCast(n), // lda = n (B is k x n)
            a, // A matrix
            @intCast(k), // ldb = k (A is m x k)
            &beta,
            c, // C matrix
            @intCast(n), // ldc = n (C is m x n)
        );
        try checkCublasError(status, "cublasSgemm_v2");
    }

    /// Mixed precision GEMM with FP16 inputs and FP32 accumulation
    /// Uses Tensor Cores on Turing+ GPUs for 8x speedup
    pub fn gemmEx_fp16(
        self: *const Self,
        c: *anyopaque,
        a: *const anyopaque,
        b: *const anyopaque,
        m: usize,
        n: usize,
        k: usize,
        c_type: cudaDataType_t,
    ) !void {
        const alpha: f32 = 1.0;
        const beta: f32 = 0.0;

        const algo = if (self.use_tensor_cores)
            CUBLAS_GEMM_DEFAULT_TENSOR_OP
        else
            CUBLAS_GEMM_DEFAULT;

        const compute_type = if (self.use_tensor_cores)
            CUBLAS_COMPUTE_32F_FAST_16F
        else
            CUBLAS_COMPUTE_32F;

        const status = cublasGemmEx(
            self.handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            @intCast(n),
            @intCast(m),
            @intCast(k),
            &alpha,
            b,
            CUDA_R_16F,
            @intCast(n),
            a,
            CUDA_R_16F,
            @intCast(k),
            &beta,
            c,
            c_type,
            @intCast(n),
            compute_type,
            algo,
        );
        try checkCublasError(status, "cublasGemmEx");
    }
};
