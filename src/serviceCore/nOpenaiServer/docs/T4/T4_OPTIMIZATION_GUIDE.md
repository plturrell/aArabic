# T4 GPU Optimization Guide
## Maximizing Performance on NVIDIA Tesla T4

**Target Hardware**: NVIDIA Tesla T4 (16GB VRAM, Compute 7.5)
**Last Updated**: 2026-01-21
**Audience**: ML Engineers, DevOps

---

## Table of Contents

1. [T4 Hardware Specifications](#t4-hardware-specifications)
2. [Memory Optimization](#memory-optimization)
3. [Tensor Core Utilization](#tensor-core-utilization)
4. [Quantization Strategies](#quantization-strategies)
5. [Performance Tuning](#performance-tuning)
6. [Troubleshooting](#troubleshooting)

---

## T4 Hardware Specifications

### Core Specs
- **Architecture**: Turing (TU104)
- **Compute Capability**: 7.5
- **CUDA Cores**: 2,560
- **Tensor Cores**: 320 (2nd generation)
- **Memory**: 16 GB GDDR6
- **Memory Bandwidth**: 320 GB/s
- **TDP**: 70W
- **FP32 Performance**: 8.1 TFLOPS
- **FP16 Performance**: 65 TFLOPS (with Tensor Cores)
- **INT8 Performance**: 130 TOPS (with Tensor Cores)

### Key Features
✅ **Tensor Cores**: Hardware acceleration for matrix operations
✅ **Mixed Precision**: Native FP16 support
✅ **INT8 Quantization**: Hardware-accelerated inference
✅ **Low Power**: Ideal for inference workloads
✅ **PCIe Gen3 x16**: High bandwidth to host

### Limitations
⚠️ **Limited VRAM**: 16GB may be tight for 70B+ models
⚠️ **Memory Bandwidth**: Lower than A100/H100
⚠️ **Compute**: Optimized for inference, not training

---

## Memory Optimization

### 16GB VRAM Budget Strategy

**Recommended Allocation**:
```
Total: 16,384 MB
├── Model Weights: 4,096 MB (25%)
├── KV Cache (Hot): 10,240 MB (62.5%)
├── Activation Memory: 1,536 MB (9.4%)
└── System/Overhead: 512 MB (3.1%)
```

### Model Size Guidelines

| Model Size | Quantization | VRAM Used | Fits in T4? | Notes |
|------------|-------------|-----------|-------------|-------|
| 7B | Q4_K_M | ~4 GB | ✅ Yes | Ideal, plenty of room for KV cache |
| 13B | Q4_K_M | ~7 GB | ✅ Yes | Good fit with tiering |
| 33B | Q4_K_M | ~18 GB | ⚠️ Tight | Requires aggressive tiering |
| 70B | Q4_K_M | ~40 GB | ❌ No | Needs RAM/SSD tiering |

### Tiered Memory Configuration

For models that don't fit entirely in VRAM:

```zig
// Optimal T4 configuration
pub const T4Config = struct {
    // GPU Tier (Hottest)
    gpu_tokens: u32 = 2048,           // ~8GB for KV cache
    gpu_model_layers: u32 = 8,        // Keep first/last layers on GPU
    
    // RAM Tier (Hot)
    ram_tokens: u32 = 8192,           // ~32GB system RAM
    ram_model_layers: u32 = 24,       // Majority in RAM
    
    // SSD Tier (Cold)
    ssd_tokens: u32 = 65536,          // Long context on SSD
    ssd_model_layers: u32 = 0,        // Model stays in RAM/GPU
    
    // Compression
    enable_kv_compression: bool = true,
    compression_ratio: f32 = 0.5,     // 2x reduction
};
```

### Memory-Mapped Model Loading

For large models, use memory-mapped GGUF files:

```zig
const loader_config = TieredModelConfig{
    .model_path = "llama-70b-q4km.gguf",
    .max_ram_mb = 4096,              // Only 4GB in RAM
    .hot_layers = &[_][]const u8{
        "model.layers.0",             // Input layer
        "model.layers.79",            // Output layer
        "lm_head",                    // Final projection
    },
};
```

**Benefits**:
- Zero-copy access to model weights
- OS handles paging automatically
- Only hot layers in memory

---

## Tensor Core Utilization

### When to Use Tensor Cores

Tensor Cores provide massive speedups for specific operations:

**Ideal Operations**:
- ✅ Matrix multiplication (M x N x K where M, N, K ≥ 16)
- ✅ Attention mechanism (Q @ K^T, scores @ V)
- ✅ Feed-forward layers (matmul in MLP)

**Not Beneficial**:
- ❌ Element-wise operations (ReLU, LayerNorm)
- ❌ Small matrices (< 16x16)
- ❌ Memory-bound operations

### Enabling Tensor Cores

**FP16 Mixed Precision** (Recommended):
```mojo
# In matmul_cuda.mojo
fn attention_with_tensor_cores(
    q: Tensor[DType.float16],
    k: Tensor[DType.float16],
    v: Tensor[DType.float16]
) -> Tensor[DType.float16]:
    # Q @ K^T using Tensor Cores
    var scores = Tensor[DType.float16](seq_len, seq_len)
    external_call["cublasGemmEx"](
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_T,
        seq_len, seq_len, head_dim,
        alpha, q.data, CUDA_R_16F,
        k.data, CUDA_R_16F,
        beta, scores.data, CUDA_R_16F,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP  # Enable Tensor Cores
    )
    
    # Softmax (regular CUDA kernel)
    softmax_inplace(scores)
    
    # scores @ V using Tensor Cores
    var output = Tensor[DType.float16](seq_len, head_dim)
    external_call["cublasGemmEx"](
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        seq_len, head_dim, seq_len,
        alpha, scores.data, CUDA_R_16F,
        v.data, CUDA_R_16F,
        beta, output.data, CUDA_R_16F,
        CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    )
    
    return output
```

**INT8 Quantization** (Maximum Performance):
```mojo
fn matmul_int8_tensor_core(
    c: Tensor[DType.float32],
    a: Tensor[DType.int8],
    b: Tensor[DType.int8],
    scale_a: Float32,
    scale_b: Float32
):
    external_call["cublasGemmEx"](
        cublas_handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        m, n, k,
        alpha, a.data, CUDA_R_8I,
        b.data, CUDA_R_8I,
        beta, c.data, CUDA_R_32F,
        CUDA_R_32I, CUBLAS_GEMM_DEFAULT_TENSOR_OP
    )
    
    # Scale output
    scale_kernel(c, scale_a * scale_b)
```

### Performance Expectations

| Operation | FP32 (TFLOPS) | FP16 (TFLOPS) | INT8 (TOPS) | Speedup |
|-----------|---------------|---------------|-------------|---------|
| MatMul (no TC) | 8.1 | 8.1 | - | 1x |
| MatMul (FP16 TC) | - | 65 | - | **8x** |
| MatMul (INT8 TC) | - | - | 130 | **16x** |

---

## Quantization Strategies

### Recommended Quantization Levels

| Quantization | Bits/Weight | Model Size | Quality | Speed | Use Case |
|--------------|-------------|------------|---------|-------|----------|
| **Q4_K_M** | 4.5 | ~0.5x | ⭐⭐⭐⭐ | Fast | **Recommended for T4** |
| Q6_K | 6.5 | ~0.7x | ⭐⭐⭐⭐⭐ | Medium | High quality needed |
| Q8_0 | 8.5 | ~0.9x | ⭐⭐⭐⭐⭐ | Slow | Maximum accuracy |
| Q4_0 | 4.0 | ~0.4x | ⭐⭐⭐ | Fastest | Experimental |

### Q4_K_M Configuration

**Why Q4_K_M is optimal for T4**:
- ✅ 4.5 bits per weight (good compression)
- ✅ K-means quantization (better quality than Q4_0)
- ✅ Mixed precision blocks (critical weights at higher precision)
- ✅ Fast dequantization kernels

**Loading Q4_K_M on T4**:
```zig
const model_config = ModelConfig{
    .id = "llama-70b-q4km",
    .path = "Llama-3.3-70B-Q4_K_M.gguf",
    .quantization = .Q4_K_M,
    .tier_config = .{
        .max_ram_mb = 12288,          // 12GB for model + KV
        .kv_cache_ram_mb = 8192,      // 8GB for KV cache
        .enable_ssd_tier = true,
    },
};
```

### Dynamic Quantization

For maximum flexibility, implement runtime quantization:

```zig
pub fn optimizeForT4(model: *Model) !void {
    const mem_info = try cuda_ctx.getMemoryInfo();
    
    if (mem_info.free_mb < 8192) {
        // Low memory: aggressive quantization
        try model.quantize(.Q4_0);
        try model.enableKVCompression(0.4);
    } else if (mem_info.free_mb < 12288) {
        // Medium memory: balanced
        try model.quantize(.Q4_K_M);
        try model.enableKVCompression(0.5);
    } else {
        // Plenty of memory: higher quality
        try model.quantize(.Q6_K);
        try model.disableKVCompression();
    }
}
```

---

## Performance Tuning

### Batch Size Optimization

T4 performs best with specific batch sizes:

```zig
pub fn calculateOptimalBatchSize(
    model_size_mb: u32,
    seq_len: u32,
    precision: Precision
) u32 {
    const available_mb = 16384 - model_size_mb;
    const kv_per_token_mb = switch (precision) {
        .FP16 => 0.5,
        .FP32 => 1.0,
        .INT8 => 0.25,
    };
    
    const max_batch = available_mb / (seq_len * kv_per_token_mb);
    
    // Prefer powers of 2 for coalesced memory access
    return @max(1, @min(32, std.math.ceilPowerOfTwo(u32, max_batch) catch 1));
}
```

**Recommended Batch Sizes**:
- 7B model: batch_size = 8-16
- 13B model: batch_size = 4-8
- 33B model: batch_size = 1-2
- 70B model: batch_size = 1 (with tiering)

### Context Length Tuning

**Short Context (≤2K tokens)**:
- Keep entire KV cache on GPU
- No tiering needed
- Maximum throughput

```zig
const config = GPUConfig{
    .gpu_tokens = 2048,
    .enable_tiering = false,
};
```

**Medium Context (2K-8K tokens)**:
- Hot tokens on GPU
- Recent tokens in RAM
- Older tokens on SSD

```zig
const config = TieredConfig{
    .gpu_tokens = 2048,
    .ram_tokens = 6144,
    .enable_tiering = true,
};
```

**Long Context (>8K tokens)**:
- Aggressive tiering
- KV cache compression
- Sliding window attention

```zig
const config = TieredConfig{
    .gpu_tokens = 1024,
    .ram_tokens = 4096,
    .ssd_tokens = 32768,
    .enable_compression = true,
    .compression_ratio = 0.5,
};
```

### Kernel Launch Configuration

Optimize CUDA kernel launches for T4:

```mojo
fn get_optimal_grid_config(n_elements: Int) -> (Int, Int):
    # T4 has 40 SMs, each with 64 warps
    let threads_per_block = 256  # 8 warps per block
    let blocks_per_sm = 2        # Maximize occupancy
    let num_sms = 40
    
    let total_blocks = (n_elements + threads_per_block - 1) // threads_per_block
    let grid_size = min(total_blocks, num_sms * blocks_per_sm)
    
    return (grid_size, threads_per_block)
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory Errors

**Symptoms**:
```
CUDA Error: cudaErrorMemoryAllocation (2)
OutOfGPUMemory
```

**Solutions**:
1. Reduce batch size
2. Enable KV cache tiering
3. Use more aggressive quantization (Q4_0 instead of Q6_K)
4. Reduce context length
5. Enable KV cache compression

```bash
# Check current memory usage
nvidia-smi

# Enable tiering
export GPU_TOKENS=1024
export ENABLE_TIERING=true
```

#### 2. Low GPU Utilization

**Symptoms**:
```
GPU Utilization: 30%
Memory Bandwidth: Low
```

**Causes**:
- Batch size too small
- Not using Tensor Cores
- Memory-bound operations

**Solutions**:
1. Increase batch size
2. Enable FP16 mixed precision
3. Use async memory transfers
4. Optimize kernel launch config

#### 3. Slow Inference

**Benchmark**:
```bash
# Expected performance
7B Q4_K_M: ~40-50 tokens/sec
13B Q4_K_M: ~20-30 tokens/sec
33B Q4_K_M: ~8-12 tokens/sec
```

**If slower**:
1. Check Tensor Core usage: `nvidia-smi dmon -s u`
2. Profile with: `nsys profile ./openai_server`
3. Verify FP16 enabled
4. Check PCIe bandwidth

#### 4. Temperature Throttling

**Symptoms**:
```
Temperature: >80°C
Performance degradation
```

**Solutions**:
1. Improve datacenter cooling
2. Reduce power limit: `nvidia-smi -pl 65`
3. Add delays between requests
4. Monitor with: `nvidia-smi dmon -s pct`

---

## Performance Checklist

### Before Deployment

- [ ] Model quantized to Q4_K_M or Q6_K
- [ ] FP16 mixed precision enabled
- [ ] Tensor Cores utilized for matmul/attention
- [ ] KV cache tiering configured
- [ ] Batch size optimized for workload
- [ ] Memory bandwidth tested (>250 GB/s)
- [ ] GPU utilization >70% during inference
- [ ] Temperature <75°C under load
- [ ] Benchmarks meet targets (see above)

### Monitoring

```bash
# Real-time monitoring
watch -n 1 nvidia-smi

# Detailed metrics
nvidia-smi dmon -s puct -c 100

# Profile a run
nsys profile -o report ./openai_server
nsys stats report.qdrep
```

---

## Best Practices

1. **Always use Q4_K_M** for production (best quality/performance trade-off)
2. **Enable FP16** for Tensor Core acceleration
3. **Configure tiering** for models >7B
4. **Monitor GPU metrics** continuously
5. **Test with target workload** before production
6. **Keep CUDA drivers updated** (470+)
7. **Use pinned memory** for faster transfers
8. **Batch requests** when possible

---

## Additional Resources

- [NVIDIA T4 Datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf)
- [Tensor Core Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#tensor-cores)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/index.html)
- [GGUF Quantization](https://github.com/ggerganov/llama.cpp/blob/master/gguf-py/README.md)

---

**Next**: [AI Core Deployment Guide](./AICORE_DEPLOYMENT_GUIDE.md)
