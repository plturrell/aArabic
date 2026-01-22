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
// RoPE (Rotary Position Embeddings) Kernel - LLaMA style "split" format
// For each dimension pair (i, i+half_dim):
//   x'[i] = x[i] * cos - x[i+half_dim] * sin
//   x'[i+half_dim] = x[i+half_dim] * cos + x[i] * sin
// ============================================================================
__global__ void rope_kernel(half* __restrict__ q,
                            half* __restrict__ k,
                            int position,
                            int head_dim,
                            int n_heads,
                            int n_kv_heads,
                            float rope_theta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_dim = head_dim / 2;
    int total_q_pairs = n_heads * half_dim;
    int total_k_pairs = n_kv_heads * half_dim;
    int total_pairs = total_q_pairs + total_k_pairs;

    if (idx >= total_pairs) return;

    // Determine if this is Q or K, and which head/dimension
    bool is_q = idx < total_q_pairs;
    int local_idx = is_q ? idx : (idx - total_q_pairs);
    int n_heads_for_tensor = is_q ? n_heads : n_kv_heads;
    half* tensor = is_q ? q : k;

    int head_idx = local_idx / half_dim;
    int dim_idx = local_idx % half_dim;

    // Compute frequency: freq = 1.0 / pow(rope_theta, 2*dim_idx / head_dim)
    float freq = 1.0f / powf(rope_theta, (float)(2 * dim_idx) / (float)head_dim);
    float angle = (float)position * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    // Get indices for the pair
    int base_idx = head_idx * head_dim;
    int idx_lo = base_idx + dim_idx;
    int idx_hi = base_idx + dim_idx + half_dim;

    // Read values (FP16 -> FP32)
    float x_lo = __half2float(tensor[idx_lo]);
    float x_hi = __half2float(tensor[idx_hi]);

    // Handle NaN/Inf
    if (!isfinite(x_lo)) x_lo = 0.0f;
    if (!isfinite(x_hi)) x_hi = 0.0f;

    // Apply rotation
    float new_lo = x_lo * cos_val - x_hi * sin_val;
    float new_hi = x_hi * cos_val + x_lo * sin_val;

    // Clamp to FP16 range and write back
    new_lo = fminf(fmaxf(new_lo, -65504.0f), 65504.0f);
    new_hi = fminf(fmaxf(new_hi, -65504.0f), 65504.0f);

    tensor[idx_lo] = __float2half(new_lo);
    tensor[idx_hi] = __float2half(new_hi);
}

// ============================================================================
// Attention Scores Kernel - Compute Q @ K^T / sqrt(head_dim)
// Supports grouped query attention (n_heads Q heads share n_kv_heads KV heads)
// Output: scores[n_heads][position+1]
// ============================================================================
__global__ void attention_scores_kernel(const half* __restrict__ q,
                                         const half* __restrict__ k_cache,
                                         half* __restrict__ scores,
                                         int position,
                                         int head_dim,
                                         int n_heads,
                                         int n_kv_heads,
                                         int max_seq_len) {
    // Each block handles one (query_head, key_position) pair
    // blockIdx.x = query head index
    // blockIdx.y = key position index (0 to position inclusive)
    int q_head = blockIdx.x;
    int k_pos = blockIdx.y;
    int seq_len = position + 1;

    if (q_head >= n_heads || k_pos >= seq_len) return;

    // Grouped query attention: map query head to KV head
    int kv_head = q_head / (n_heads / n_kv_heads);

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute dot product Q[q_head] @ K[kv_head, k_pos]
    // Q is laid out as [n_heads, head_dim]
    // K_cache is laid out as [n_kv_heads, max_seq_len, head_dim]
    const half* q_ptr = q + q_head * head_dim;
    const half* k_ptr = k_cache + kv_head * max_seq_len * head_dim + k_pos * head_dim;

    float sum = 0.0f;
    for (int i = tid; i < head_dim; i += stride) {
        float q_val = __half2float(q_ptr[i]);
        float k_val = __half2float(k_ptr[i]);
        if (!isfinite(q_val)) q_val = 0.0f;
        if (!isfinite(k_val)) k_val = 0.0f;
        sum += q_val * k_val;
    }

    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final score with scaling
    if (tid == 0) {
        float score = shared[0] / sqrtf((float)head_dim);
        score = fminf(fmaxf(score, -65504.0f), 65504.0f);
        // Output layout: [n_heads, seq_len]
        scores[q_head * seq_len + k_pos] = __float2half(score);
    }
}

// ============================================================================
// Causal Mask Kernel - Set future positions to -inf (large negative)
// scores layout: [n_heads, seq_len], current_pos is the query position
// ============================================================================
__global__ void causal_mask_kernel(half* __restrict__ scores,
                                    int n_heads,
                                    int seq_len,
                                    int current_pos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n_heads * seq_len;

    if (idx >= total) return;

    int head = idx / seq_len;
    int pos = idx % seq_len;

    // Mask out positions after current_pos (future tokens)
    // For autoregressive decoding, current_pos == seq_len - 1, so no masking needed
    // But for prefill or if current_pos < seq_len - 1, mask future positions
    if (pos > current_pos) {
        // Use -65504 (min FP16) as -inf substitute
        scores[head * seq_len + pos] = __float2half(-65504.0f);
    }
}

// ============================================================================
// Attention Output Kernel - Compute softmax(scores) @ V for each head
// Assumes scores have already been softmaxed
// Output: [n_heads * head_dim]
// ============================================================================
__global__ void attention_output_kernel(const half* __restrict__ scores,
                                         const half* __restrict__ v_cache,
                                         half* __restrict__ output,
                                         int position,
                                         int head_dim,
                                         int n_heads,
                                         int n_kv_heads,
                                         int max_seq_len) {
    // Each block handles one (query_head, output_dim) pair
    // blockIdx.x = query head index
    // blockIdx.y = output dimension index within head
    int q_head = blockIdx.x;
    int dim_idx = blockIdx.y;
    int seq_len = position + 1;

    if (q_head >= n_heads || dim_idx >= head_dim) return;

    // Grouped query attention: map query head to KV head
    int kv_head = q_head / (n_heads / n_kv_heads);

    extern __shared__ float shared[];

    int tid = threadIdx.x;
    int stride = blockDim.x;

    // Compute weighted sum: sum over positions of scores[q_head, pos] * V[kv_head, pos, dim_idx]
    // scores layout: [n_heads, seq_len]
    // V_cache layout: [n_kv_heads, max_seq_len, head_dim]
    const half* score_ptr = scores + q_head * seq_len;
    const half* v_ptr = v_cache + kv_head * max_seq_len * head_dim + dim_idx;

    float sum = 0.0f;
    for (int pos = tid; pos < seq_len; pos += stride) {
        float s = __half2float(score_ptr[pos]);
        float v = __half2float(v_ptr[pos * head_dim]);
        if (!isfinite(s)) s = 0.0f;
        if (!isfinite(v)) v = 0.0f;
        sum += s * v;
    }

    shared[tid] = sum;
    __syncthreads();

    // Parallel reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Thread 0 writes the final output
    if (tid == 0) {
        float result = shared[0];
        result = fminf(fmaxf(result, -65504.0f), 65504.0f);
        // Output layout: [n_heads, head_dim] flattened
        output[q_head * head_dim + dim_idx] = __float2half(result);
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

int cuda_apply_rope(half* q, half* k, int position, int head_dim,
                    int n_heads, int n_kv_heads, float rope_theta,
                    cudaStream_t stream) {
    if (head_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0) return 0;
    int half_dim = head_dim / 2;
    int total_pairs = n_heads * half_dim + n_kv_heads * half_dim;
    int grid_size = (total_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    rope_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
        q, k, position, head_dim, n_heads, n_kv_heads, rope_theta);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_attention_scores(const half* q, const half* k_cache, half* scores,
                          int position, int head_dim, int n_heads,
                          int n_kv_heads, int max_seq_len, cudaStream_t stream) {
    if (head_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0) return 0;
    int seq_len = position + 1;
    // Grid: [n_heads, seq_len], each block computes one score
    dim3 grid(n_heads, seq_len);
    int threads = min(THREADS_PER_BLOCK, head_dim);
    attention_scores_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        q, k_cache, scores, position, head_dim, n_heads, n_kv_heads, max_seq_len);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_apply_causal_mask(half* scores, int n_heads, int seq_len,
                           int current_pos, cudaStream_t stream) {
    if (n_heads <= 0 || seq_len <= 0) return 0;
    int total = n_heads * seq_len;
    int grid_size = (total + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    causal_mask_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(
        scores, n_heads, seq_len, current_pos);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_attention_output(const half* scores, const half* v_cache, half* output,
                          int position, int head_dim, int n_heads,
                          int n_kv_heads, int max_seq_len, cudaStream_t stream) {
    if (head_dim <= 0 || n_heads <= 0 || n_kv_heads <= 0) return 0;
    int seq_len = position + 1;
    // Grid: [n_heads, head_dim], each block computes one output element
    dim3 grid(n_heads, head_dim);
    int threads = min(THREADS_PER_BLOCK, seq_len);
    attention_output_kernel<<<grid, threads, threads * sizeof(float), stream>>>(
        scores, v_cache, output, position, head_dim, n_heads, n_kv_heads, max_seq_len);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int cuda_copy_to_kv_cache(const half* k, const half* v,
                          half* k_cache, half* v_cache,
                          int position, int head_dim, int n_kv_heads,
                          int max_seq_len, cudaStream_t stream) {
    if (head_dim <= 0 || n_kv_heads <= 0) return 0;
    // Copy K and V to their respective positions in the cache
    // K layout: [n_kv_heads * head_dim] -> cache[n_kv_heads, max_seq_len, head_dim] at position
    // V layout: same
    for (int h = 0; h < n_kv_heads; h++) {
        size_t src_offset = h * head_dim;
        size_t dst_offset = h * max_seq_len * head_dim + position * head_dim;
        cudaMemcpyAsync(k_cache + dst_offset, k + src_offset,
                        head_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
        cudaMemcpyAsync(v_cache + dst_offset, v + src_offset,
                        head_dim * sizeof(half), cudaMemcpyDeviceToDevice, stream);
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

} // extern "C"

