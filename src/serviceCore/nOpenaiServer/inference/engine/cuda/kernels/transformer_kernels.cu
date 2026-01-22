/**
 * CUDA Transformer Kernels for GPU Inference
 *
 * Implements RMSNorm, SiLU activation, elementwise operations, and softmax
 * for complete transformer forward pass on GPU.
 *
 * All kernels use FP16 (half) for input/output to match GpuTensor format,
 * but use FP32 internally for numerical stability.
 *
 * Compile: nvcc -shared -o libtransformer_kernels.so transformer_kernels.cu -arch=sm_75 --compiler-options '-fPIC'
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// RMSNorm Kernel (FP16 in/out, FP32 compute) with NaN protection
// y = x * rsqrt(mean(x²) + eps) * weight
// ============================================================================
__global__ void rms_norm_kernel(const half* __restrict__ input,
                                 const half* __restrict__ weight,
                                 half* __restrict__ output,
                                 int size,
                                 float eps) {
    // One block per vector, use shared memory for reduction
    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Calculate sum of squares with grid-stride loop (read FP16, compute FP32)
    // Also check for NaN/Inf and clamp
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += stride) {
        float val = __half2float(input[i]);
        // Clamp extreme values to prevent overflow
        val = fminf(fmaxf(val, -65504.0f), 65504.0f);  // FP16 range
        if (!isfinite(val)) val = 0.0f;  // Replace NaN/Inf with 0
        sum_sq += val * val;
    }

    // Store in shared memory
    shared[tid] = sum_sq;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Compute normalization factor with additional safety
    float variance = shared[0] / (float)size + eps;
    float rms = rsqrtf(fmaxf(variance, eps));  // Ensure we never divide by zero
    if (!isfinite(rms)) rms = 1.0f;  // Fallback to no normalization if rms is NaN/Inf

    // Apply normalization and weight (compute in FP32, write FP16)
    for (int i = tid; i < size; i += stride) {
        float val = __half2float(input[i]);
        float w = __half2float(weight[i]);
        // Clamp input and weights
        val = fminf(fmaxf(val, -65504.0f), 65504.0f);
        w = fminf(fmaxf(w, -65504.0f), 65504.0f);
        if (!isfinite(val)) val = 0.0f;
        if (!isfinite(w)) w = 1.0f;
        float result = val * rms * w;
        // Clamp output to FP16 range
        result = fminf(fmaxf(result, -65504.0f), 65504.0f);
        output[i] = __float2half(result);
    }
}

// ============================================================================
// SiLU (Swish) Activation Kernel (FP16 in/out, FP32 compute) with NaN protection
// y = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================
__global__ void silu_kernel(const half* __restrict__ input,
                            half* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = __half2float(input[idx]);
        // Handle NaN/Inf inputs
        if (!isfinite(x)) {
            output[idx] = __float2half(0.0f);
            return;
        }
        // Clamp x to prevent exp overflow (exp(-x) overflows for x < -88)
        x = fminf(fmaxf(x, -88.0f), 88.0f);
        float result = x / (1.0f + expf(-x));
        // Clamp output
        result = fminf(fmaxf(result, -65504.0f), 65504.0f);
        output[idx] = __float2half(result);
    }
}

// ============================================================================
// SiLU + Elementwise Multiply (Fused) (FP16 in/out, FP32 compute) with NaN protection
// out = silu(gate) * up
// ============================================================================
__global__ void silu_mul_kernel(const half* __restrict__ gate,
                                 const half* __restrict__ up,
                                 half* __restrict__ output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = __half2float(gate[idx]);
        float u = __half2float(up[idx]);
        // Handle NaN/Inf inputs
        if (!isfinite(g) || !isfinite(u)) {
            output[idx] = __float2half(0.0f);
            return;
        }
        // Clamp g to prevent exp overflow
        g = fminf(fmaxf(g, -88.0f), 88.0f);
        float silu_g = g / (1.0f + expf(-g));
        float result = silu_g * u;
        // Clamp output to FP16 range
        result = fminf(fmaxf(result, -65504.0f), 65504.0f);
        output[idx] = __float2half(result);
    }
}

// ============================================================================
// Elementwise Multiply Kernel (FP16 in/out, FP32 compute)
// out = a * b
// ============================================================================
__global__ void elementwise_mul_kernel(const half* __restrict__ a,
                                        const half* __restrict__ b,
                                        half* __restrict__ output,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va * vb);
    }
}

// ============================================================================
// Vector Add Kernel (FP16 in/out, FP32 compute)
// out = a + b
// ============================================================================
__global__ void vector_add_kernel(const half* __restrict__ a,
                                   const half* __restrict__ b,
                                   half* __restrict__ output,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float va = __half2float(a[idx]);
        float vb = __half2float(b[idx]);
        output[idx] = __float2half(va + vb);
    }
}

// ============================================================================
// Row-wise Softmax Kernel (FP16 in/out, FP32 compute) with NaN/zero protection
// For each row: out[i] = exp(x[i] - max) / sum(exp(x - max))
// ============================================================================
__global__ void softmax_kernel(half* __restrict__ data,
                                int rows,
                                int cols) {
    extern __shared__ float shared[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;

    if (row >= rows) return;

    half* row_data = data + row * cols;

    // Find max (for numerical stability) - read FP16, compute FP32
    // Also handle NaN values by treating them as -inf
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += stride) {
        float val = __half2float(row_data[i]);
        if (isfinite(val)) {
            max_val = fmaxf(max_val, val);
        }
    }
    shared[tid] = max_val;
    __syncthreads();

    // Reduce to find global max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] = fmaxf(shared[tid], shared[tid + s]);
        }
        __syncthreads();
    }
    max_val = shared[0];
    __syncthreads();

    // If max is still -inf (all values were NaN/inf), set uniform distribution
    bool all_invalid = !isfinite(max_val);
    if (all_invalid) {
        float uniform_val = 1.0f / (float)cols;
        for (int i = tid; i < cols; i += stride) {
            row_data[i] = __float2half(uniform_val);
        }
        return;
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float val = __half2float(row_data[i]);
        float exp_val = 0.0f;
        if (isfinite(val)) {
            // Clamp (val - max_val) to prevent exp underflow/overflow
            float diff = fminf(fmaxf(val - max_val, -88.0f), 0.0f);  // diff <= 0 always
            exp_val = expf(diff);
        }
        row_data[i] = __float2half(exp_val);  // Store intermediate in FP16
        sum += exp_val;
    }
    shared[tid] = sum;
    __syncthreads();

    // Reduce to find sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    float total_sum = shared[0];

    // Protect against divide by zero - use uniform distribution if sum is 0
    float inv_sum;
    if (total_sum > 1e-12f) {
        inv_sum = 1.0f / total_sum;
    } else {
        // Sum is 0 or very small - set uniform distribution
        for (int i = tid; i < cols; i += stride) {
            row_data[i] = __float2half(1.0f / (float)cols);
        }
        return;
    }

    // Normalize - read back FP16, compute FP32, write FP16
    for (int i = tid; i < cols; i += stride) {
        float val = __half2float(row_data[i]);
        float result = val * inv_sum;
        // Clamp to valid probability range
        result = fminf(fmaxf(result, 0.0f), 1.0f);
        row_data[i] = __float2half(result);
    }
}

// ============================================================================
// C-Compatible Export Functions for Zig FFI
// All functions use half* (FP16) to match GpuTensor format
// ============================================================================

extern "C" {

#define THREADS_PER_BLOCK 256

int cuda_rms_norm(const half* input, const half* weight, half* output,
                  int size, float eps, cudaStream_t stream) {
    if (size <= 0) return 0;
    // Use one block with shared memory for reduction
    int threads = min(THREADS_PER_BLOCK, size);
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, output, size, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_silu(const half* input, half* output, int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    silu_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_silu_mul(const half* gate, const half* up, half* output,
                  int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    silu_mul_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(gate, up, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_elementwise_mul(const half* a, const half* b, half* output,
                         int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_mul_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(a, b, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_vector_add(const half* a, const half* b, half* output,
                    int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(a, b, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_softmax(half* data, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return 0;
    int threads = min(THREADS_PER_BLOCK, cols);
    softmax_kernel<<<rows, threads, threads * sizeof(float), stream>>>(data, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"

