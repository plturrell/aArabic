/**
 * CUDA Dequantization Kernels for GGUF Quantized Weights
 * 
 * Supports Q4_0, Q8_0, Q4_K, and Q6_K formats with FP16 output for Tensor Core consumption.
 * 
 * Compile: nvcc -shared -o libdequant_kernels.so dequant_kernels.cu -arch=sm_75 --compiler-options '-fPIC'
 */

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Block sizes matching GGUF spec
#define Q4_0_BLOCK_SIZE 32
#define Q4_0_BLOCK_BYTES 18
#define Q8_0_BLOCK_SIZE 32
#define Q8_0_BLOCK_BYTES 36
#define Q4_K_BLOCK_SIZE 256
#define Q4_K_BLOCK_BYTES 144
#define Q6_K_BLOCK_SIZE 256
#define Q6_K_BLOCK_BYTES 210

// ============================================================================
// Q4_0 Dequantization Kernel
// Format: 2 bytes f16 scale + 16 bytes packed 4-bit values = 32 FP16 outputs
// ============================================================================
__global__ void dequant_q4_0_kernel(const uint8_t* __restrict__ input,
                                     __half* __restrict__ output,
                                     int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    const uint8_t* block = input + block_idx * Q4_0_BLOCK_BYTES;
    __half* out = output + block_idx * Q4_0_BLOCK_SIZE;
    
    // Read scale (stored as f16)
    __half scale = *reinterpret_cast<const __half*>(block);
    const uint8_t* qs = block + 2;
    
    // Dequantize 32 values (16 bytes, 2 values per byte)
    for (int i = 0; i < 16; i++) {
        uint8_t packed = qs[i];
        int8_t v0 = (packed & 0x0F) - 8;  // Low nibble
        int8_t v1 = (packed >> 4) - 8;     // High nibble
        out[i * 2] = __hmul(scale, __float2half((float)v0));
        out[i * 2 + 1] = __hmul(scale, __float2half((float)v1));
    }
}

// ============================================================================
// Q8_0 Dequantization Kernel
// Format: 4 bytes f32 scale + 32 bytes int8 values = 32 FP16 outputs
// ============================================================================
__global__ void dequant_q8_0_kernel(const uint8_t* __restrict__ input,
                                     __half* __restrict__ output,
                                     int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    const uint8_t* block = input + block_idx * Q8_0_BLOCK_BYTES;
    __half* out = output + block_idx * Q8_0_BLOCK_SIZE;
    
    // Read scale (stored as f32)
    float scale = *reinterpret_cast<const float*>(block);
    const int8_t* qs = reinterpret_cast<const int8_t*>(block + 4);
    
    // Dequantize 32 values
    for (int i = 0; i < 32; i++) {
        out[i] = __float2half((float)qs[i] * scale);
    }
}

// ============================================================================
// Q4_K Dequantization Kernel  
// Format: 144 bytes -> 256 FP16 values (K-quant format)
// ============================================================================
__global__ void dequant_q4_k_kernel(const uint8_t* __restrict__ input,
                                     __half* __restrict__ output,
                                     int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;
    
    const uint8_t* block = input + block_idx * Q4_K_BLOCK_BYTES;
    __half* out = output + block_idx * Q4_K_BLOCK_SIZE;
    
    // Q4_K block structure:
    // - d: f16 super-block scale (2 bytes)
    // - dmin: f16 super-block min (2 bytes)  
    // - scales: 12 bytes packed scales/mins for 8 sub-blocks
    // - qs: 128 bytes packed 4-bit values
    
    __half d = *reinterpret_cast<const __half*>(block);
    __half dmin = *reinterpret_cast<const __half*>(block + 2);
    const uint8_t* scales = block + 4;
    const uint8_t* qs = block + 16;
    
    // Process 8 sub-blocks of 32 values each
    for (int sb = 0; sb < 8; sb++) {
        // Extract scale and min for this sub-block from packed format
        uint8_t sc_packed = scales[sb * 3 / 2];
        uint8_t sc = (sb % 2 == 0) ? (sc_packed & 0x3F) : (sc_packed >> 6) | ((scales[sb * 3 / 2 + 1] & 0x0F) << 2);
        uint8_t m = (sb % 2 == 0) ? ((scales[sb * 3 / 2] >> 6) | ((scales[sb * 3 / 2 + 1] & 0x03) << 2)) : (scales[sb * 3 / 2 + 1] >> 4);
        
        float scale_val = __half2float(d) * (float)sc;
        float min_val = __half2float(dmin) * (float)m;
        
        // Dequantize 32 values in this sub-block
        const uint8_t* sub_qs = qs + sb * 16;
        __half* sub_out = out + sb * 32;
        
        for (int i = 0; i < 16; i++) {
            uint8_t packed = sub_qs[i];
            float v0 = (float)(packed & 0x0F) * scale_val - min_val;
            float v1 = (float)(packed >> 4) * scale_val - min_val;
            sub_out[i * 2] = __float2half(v0);
            sub_out[i * 2 + 1] = __float2half(v1);
        }
    }
}

// ============================================================================
// Q6_K Dequantization Kernel
// Format: 210 bytes -> 256 FP16 values
// ============================================================================
__global__ void dequant_q6_k_kernel(const uint8_t* __restrict__ input,
                                     __half* __restrict__ output,
                                     int num_blocks) {
    int block_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_idx >= num_blocks) return;

    const uint8_t* block = input + block_idx * Q6_K_BLOCK_BYTES;
    __half* out = output + block_idx * Q6_K_BLOCK_SIZE;

    // Q6_K structure: ql[128] + qh[64] + scales[16] + d (f16)
    const uint8_t* ql = block;
    const uint8_t* qh = block + 128;
    const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
    __half d = *reinterpret_cast<const __half*>(block + 208);

    float d_val = __half2float(d);

    // Process 16 sub-blocks of 16 values each
    for (int sb = 0; sb < 16; sb++) {
        float scale = d_val * (float)scales[sb];
        int ql_offset = sb * 8;
        int qh_offset = sb * 4;
        __half* sub_out = out + sb * 16;

        for (int i = 0; i < 8; i++) {
            uint8_t ql_val = ql[ql_offset + i];
            uint8_t qh_val = qh[qh_offset + i / 2];
            int qh_shift = (i % 2) * 4;

            int8_t q0 = (ql_val & 0x0F) | (((qh_val >> qh_shift) & 0x03) << 4);
            int8_t q1 = (ql_val >> 4) | (((qh_val >> (qh_shift + 2)) & 0x03) << 4);

            sub_out[i * 2] = __float2half((float)(q0 - 32) * scale);
            sub_out[i * 2 + 1] = __float2half((float)(q1 - 32) * scale);
        }
    }
}

// ============================================================================
// C-Compatible Export Functions for Zig FFI
// ============================================================================

extern "C" {

// Optimal thread block size for T4
#define THREADS_PER_BLOCK 256

int dequant_q4_0_fp16(const uint8_t* input, __half* output, int num_blocks, cudaStream_t stream) {
    if (num_blocks <= 0) return 0;
    int grid_size = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dequant_q4_0_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, num_blocks);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int dequant_q8_0_fp16(const uint8_t* input, __half* output, int num_blocks, cudaStream_t stream) {
    if (num_blocks <= 0) return 0;
    int grid_size = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dequant_q8_0_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, num_blocks);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int dequant_q4_k_fp16(const uint8_t* input, __half* output, int num_blocks, cudaStream_t stream) {
    if (num_blocks <= 0) return 0;
    int grid_size = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dequant_q4_k_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, num_blocks);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

int dequant_q6_k_fp16(const uint8_t* input, __half* output, int num_blocks, cudaStream_t stream) {
    if (num_blocks <= 0) return 0;
    int grid_size = (num_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dequant_q6_k_kernel<<<grid_size, THREADS_PER_BLOCK, 0, stream>>>(input, output, num_blocks);
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

// Utility functions
int dequant_get_output_size(int quant_type, int num_blocks) {
    switch (quant_type) {
        case 2:  return num_blocks * Q4_0_BLOCK_SIZE;   // Q4_0
        case 8:  return num_blocks * Q8_0_BLOCK_SIZE;   // Q8_0
        case 12: return num_blocks * Q4_K_BLOCK_SIZE;   // Q4_K
        case 14: return num_blocks * Q6_K_BLOCK_SIZE;   // Q6_K
        default: return 0;
    }
}

int dequant_get_input_size(int quant_type, int num_blocks) {
    switch (quant_type) {
        case 2:  return num_blocks * Q4_0_BLOCK_BYTES;  // Q4_0
        case 8:  return num_blocks * Q8_0_BLOCK_BYTES;  // Q8_0
        case 12: return num_blocks * Q4_K_BLOCK_BYTES;  // Q4_K
        case 14: return num_blocks * Q6_K_BLOCK_BYTES;  // Q6_K
        default: return 0;
    }
}

} // extern "C"

