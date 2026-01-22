/**
 * CUDA Transformer Kernels for GPU Inference
 * 
 * Implements RMSNorm, SiLU activation, elementwise operations, and softmax
 * for complete transformer forward pass on GPU.
 * 
 * Compile: nvcc -shared -o libtransformer_kernels.so transformer_kernels.cu -arch=sm_75 --compiler-options '-fPIC'
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

// ============================================================================
// RMSNorm Kernel
// y = x * rsqrt(mean(x²) + eps) * weight
// ============================================================================
__global__ void rms_norm_kernel(const float* __restrict__ input,
                                 const float* __restrict__ weight,
                                 float* __restrict__ output,
                                 int size,
                                 float eps) {
    // One block per vector, use shared memory for reduction
    extern __shared__ float shared[];
    
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // Calculate sum of squares with grid-stride loop
    float sum_sq = 0.0f;
    for (int i = tid; i < size; i += stride) {
        float val = input[i];
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
    
    // Compute normalization factor
    float rms = rsqrtf(shared[0] / (float)size + eps);
    
    // Apply normalization and weight
    for (int i = tid; i < size; i += stride) {
        output[i] = input[i] * rms * weight[i];
    }
}

// ============================================================================
// SiLU (Swish) Activation Kernel
// y = x * sigmoid(x) = x / (1 + exp(-x))
// ============================================================================
__global__ void silu_kernel(const float* __restrict__ input,
                            float* __restrict__ output,
                            int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = input[idx];
        output[idx] = x / (1.0f + expf(-x));
    }
}

// ============================================================================
// SiLU + Elementwise Multiply (Fused)
// out = silu(gate) * up
// ============================================================================
__global__ void silu_mul_kernel(const float* __restrict__ gate,
                                 const float* __restrict__ up,
                                 float* __restrict__ output,
                                 int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        output[idx] = silu_g * up[idx];
    }
}

// ============================================================================
// Elementwise Multiply Kernel
// out = a * b
// ============================================================================
__global__ void elementwise_mul_kernel(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float* __restrict__ output,
                                        int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] * b[idx];
    }
}

// ============================================================================
// Vector Add Kernel
// out = a + b
// ============================================================================
__global__ void vector_add_kernel(const float* __restrict__ a,
                                   const float* __restrict__ b,
                                   float* __restrict__ output,
                                   int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = a[idx] + b[idx];
    }
}

// ============================================================================
// Row-wise Softmax Kernel (for attention scores)
// For each row: out[i] = exp(x[i] - max) / sum(exp(x - max))
// ============================================================================
__global__ void softmax_kernel(float* __restrict__ data,
                                int rows,
                                int cols) {
    extern __shared__ float shared[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    if (row >= rows) return;
    
    float* row_data = data + row * cols;
    
    // Find max (for numerical stability)
    float max_val = -INFINITY;
    for (int i = tid; i < cols; i += stride) {
        max_val = fmaxf(max_val, row_data[i]);
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

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (int i = tid; i < cols; i += stride) {
        float exp_val = expf(row_data[i] - max_val);
        row_data[i] = exp_val;
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

    // Normalize
    float inv_sum = 1.0f / total_sum;
    for (int i = tid; i < cols; i += stride) {
        row_data[i] *= inv_sum;
    }
}

// ============================================================================
// C-Compatible Export Functions for Zig FFI
// ============================================================================

extern "C" {

#define THREADS_PER_BLOCK 256

int cuda_rms_norm(const float* input, const float* weight, float* output,
                  int size, float eps, cudaStream_t stream) {
    if (size <= 0) return 0;
    // Use one block with shared memory for reduction
    int threads = min(THREADS_PER_BLOCK, size);
    rms_norm_kernel<<<1, threads, threads * sizeof(float), stream>>>(
        input, weight, output, size, eps);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_silu(const float* input, float* output, int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    silu_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_silu_mul(const float* gate, const float* up, float* output,
                  int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    silu_mul_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(gate, up, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_elementwise_mul(const float* a, const float* b, float* output,
                         int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    elementwise_mul_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(a, b, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_vector_add(const float* a, const float* b, float* output,
                    int size, cudaStream_t stream) {
    if (size <= 0) return 0;
    int grid_size = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    vector_add_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(a, b, output, size);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_softmax(float* data, int rows, int cols, cudaStream_t stream) {
    if (rows <= 0 || cols <= 0) return 0;
    int threads = min(THREADS_PER_BLOCK, cols);
    softmax_kernel<<<rows, threads, threads * sizeof(float), stream>>>(data, rows, cols);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"

