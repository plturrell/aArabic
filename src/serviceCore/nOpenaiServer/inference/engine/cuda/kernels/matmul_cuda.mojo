"""
cuBLAS-based Matrix Multiplication Kernels for CUDA
FP16/FP32 GEMM with Tensor Core support via cublasGemmEx.

FFI Bindings: cublasCreate, cublasSgemm, cublasHgemm, cublasGemmEx
"""

from sys.ffi import DLHandle, external_call
from memory import UnsafePointer

# cuBLAS constants
alias CUBLAS_OP_N: Int32 = 0  # No transpose
alias CUBLAS_OP_T: Int32 = 1  # Transpose
alias CUDA_R_16F: Int32 = 2   # FP16
alias CUDA_R_32F: Int32 = 0   # FP32
alias CUBLAS_COMPUTE_32F_FAST_16F: Int32 = 74  # Tensor Core path
alias CUBLAS_GEMM_DEFAULT_TENSOR_OP: Int32 = 99


struct CublasHandle:
    """Wrapper for cuBLAS handle with FFI bindings."""
    var _handle: UnsafePointer[NoneType]
    var _lib: DLHandle
    var _initialized: Bool

    fn __init__(inout self, lib_path: String = "libcublas.so") raises:
        """Initialize cuBLAS handle."""
        self._lib = DLHandle(lib_path)
        self._handle = UnsafePointer[NoneType]()
        self._initialized = False
        var status = external_call["cublasCreate_v2", Int32](UnsafePointer.address_of(self._handle))
        if status != 0:
            raise Error("Failed to create cuBLAS handle: " + String(status))
        self._initialized = True

    fn __del__(owned self):
        if self._initialized:
            _ = external_call["cublasDestroy_v2", Int32](self._handle)

    fn sgemm(self, trans_a: Int32, trans_b: Int32, m: Int32, n: Int32, k: Int32,
             alpha: Float32, A: UnsafePointer[Float32], lda: Int32,
             B: UnsafePointer[Float32], ldb: Int32, beta: Float32,
             C: UnsafePointer[Float32], ldc: Int32) raises -> Int32:
        """FP32 matrix multiplication: C = alpha * A @ B + beta * C."""
        var alpha_ptr = UnsafePointer[Float32].alloc(1)
        var beta_ptr = UnsafePointer[Float32].alloc(1)
        alpha_ptr[0] = alpha
        beta_ptr[0] = beta
        var status = external_call["cublasSgemm_v2", Int32](
            self._handle, trans_a, trans_b, m, n, k, alpha_ptr, A, lda, B, ldb, beta_ptr, C, ldc)
        alpha_ptr.free()
        beta_ptr.free()
        return status

    fn hgemm(self, trans_a: Int32, trans_b: Int32, m: Int32, n: Int32, k: Int32,
             alpha: Float16, A: UnsafePointer[Float16], lda: Int32,
             B: UnsafePointer[Float16], ldb: Int32, beta: Float16,
             C: UnsafePointer[Float16], ldc: Int32) raises -> Int32:
        """FP16 matrix multiplication: C = alpha * A @ B + beta * C."""
        var alpha_ptr = UnsafePointer[Float16].alloc(1)
        var beta_ptr = UnsafePointer[Float16].alloc(1)
        alpha_ptr[0] = alpha
        beta_ptr[0] = beta
        var status = external_call["cublasHgemm", Int32](
            self._handle, trans_a, trans_b, m, n, k, alpha_ptr, A, lda, B, ldb, beta_ptr, C, ldc)
        alpha_ptr.free()
        beta_ptr.free()
        return status

    fn gemm_ex(self, trans_a: Int32, trans_b: Int32, m: Int32, n: Int32, k: Int32,
               alpha: Float32, A: UnsafePointer[NoneType], a_type: Int32, lda: Int32,
               B: UnsafePointer[NoneType], b_type: Int32, ldb: Int32, beta: Float32,
               C: UnsafePointer[NoneType], c_type: Int32, ldc: Int32,
               compute_type: Int32 = CUBLAS_COMPUTE_32F_FAST_16F,
               algo: Int32 = CUBLAS_GEMM_DEFAULT_TENSOR_OP) raises -> Int32:
        """Mixed precision GEMM with Tensor Core utilization via cublasGemmEx."""
        var alpha_ptr = UnsafePointer[Float32].alloc(1)
        var beta_ptr = UnsafePointer[Float32].alloc(1)
        alpha_ptr[0] = alpha
        beta_ptr[0] = beta
        var status = external_call["cublasGemmEx", Int32](
            self._handle, trans_a, trans_b, m, n, k, alpha_ptr,
            A, a_type, lda, B, b_type, ldb, beta_ptr, C, c_type, ldc, compute_type, algo)
        alpha_ptr.free()
        beta_ptr.free()
        return status


fn matmul_fp32(handle: CublasHandle, A: UnsafePointer[Float32], B: UnsafePointer[Float32],
               C: UnsafePointer[Float32], M: Int32, N: Int32, K: Int32) raises -> Int32:
    """FP32 matmul: C[M,N] = A[M,K] @ B[K,N]."""
    return handle.sgemm(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 1.0, B, N, A, K, 0.0, C, N)


fn matmul_fp16_tensor_core(handle: CublasHandle, A: UnsafePointer[Float16],
                           B: UnsafePointer[Float16], C: UnsafePointer[Float16],
                           M: Int32, N: Int32, K: Int32) raises -> Int32:
    """FP16 matmul using Tensor Cores with FP32 accumulation."""
    return handle.gemm_ex(CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, 1.0,
        A.bitcast[NoneType](), CUDA_R_16F, N, B.bitcast[NoneType](), CUDA_R_16F, K,
        0.0, C.bitcast[NoneType](), CUDA_R_16F, N)
