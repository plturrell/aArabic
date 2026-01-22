# Zig LLM Inference Engine Architecture

## System Overview

This document describes the architecture of the Zig-based LLM inference engine optimized for SAP AI Core deployment on NVIDIA Tesla T4 GPUs.

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          OpenAI-Compatible API Layer                         │
│                    (/v1/chat/completions, /v1/completions)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                              Batch Processor                                 │
│              (Continuous Batching, Dynamic Request Scheduling)               │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Inference Engine Core                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │  Tokenizer  │  │ Transformer │  │  KV Cache   │  │   Sampler           │ │
│  │    (BPE)    │  │   Layers    │  │  (Per-Seq)  │  │ (Temp/Top-p/Top-k)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────────────┘ │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Compute Backend Abstraction                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    ComputeBackend (compute.zig)                         │ │
│  │         matmul() | rms_norm() | softmax() | rope() | alloc()            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│       ▲                      ▲                       ▲                       │
│  ┌────┴─────┐          ┌─────┴──────┐          ┌─────┴──────┐               │
│  │ CPU      │          │  CUDA/T4   │          │   Metal    │               │
│  │ Backend  │          │  Backend   │          │  Backend   │               │
│  └──────────┘          └────────────┘          └────────────┘               │
├─────────────────────────────────────────────────────────────────────────────┤
│                        GPU Acceleration Layer                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ GPU Weight      │  │ cuBLAS          │  │ CUDA Dequant Kernels        │  │
│  │ Cache (FP16)    │  │ GemmEx/Tensor   │  │ (Q4_0/Q8_0/Q4_K/Q6_K→FP16)  │  │
│  │ ~2.05GB TinyLlama│ │ Cores           │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│                           Model Loading Layer                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  GGUF Loader │ Quantization │ Tiered Loading │ HuggingFace Bridge       │ │
│  │  (v3 format) │ (Q4_K ~7:1)  │ (mmap/SSD)     │ (SafeTensors)            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Design Goals

- **Target Platform**: SAP AI Core on NVIDIA Tesla T4 GPUs (16GB VRAM, Compute 7.5)
- **Performance Target**: 500+ tokens/second for TinyLlama 1.1B
- **Memory Efficiency**: ~7:1 compression with Q4_K quantization
- **Deployment**: OpenAI-compatible API for seamless integration

### Technology Stack

| Component | Technology | Version |
|-----------|------------|---------|
| Core Language | Zig | 0.15.2 |
| GPU Compute | CUDA | 12.x |
| Matrix Ops | cuBLAS | GemmEx with Tensor Cores |
| Model Format | GGUF | v3 |
| Quantization | Q4_0, Q8_0, Q4_K, Q6_K | GGML spec |

---

## Core Components

### 1. Model Loader (`core/gguf_loader.zig`)

The GGUF loader parses GGUF v3 format files and extracts model weights and metadata.

**Key Structures:**

```zig
pub const QuantizationType = enum(u32) {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14, Q8_K = 15,
};

pub const GGUFModel = struct {
    metadata: ModelMetadata,    // vocab_size, n_layers, hidden_size, etc.
    tensors: []TensorInfo,      // Weight tensor descriptors
    vocab_tokens: [][]const u8, // Vocabulary strings
    vocab_scores: []f32,        // BPE merge scores
};
```

**Loading Flow:**
1. Parse GGUF header (magic: `0x46554747`)
2. Read metadata key-value pairs
3. Load tensor info (name, shape, quantization type, offset)
4. Memory-map tensor data for efficient access

### 2. Tokenizer (`tokenization/tokenizer.zig`)

BPE (Byte Pair Encoding) tokenizer for text encoding/decoding.

```zig
pub const Tokenizer = struct {
    vocab: []Token,                    // [vocab_size] tokens
    vocab_size: u32,
    token_map: std.StringHashMap(u32), // text → token_id lookup
    bos_token: u32,                    // Beginning of sequence (1)
    eos_token: u32,                    // End of sequence (2)
    pad_token: u32,                    // Padding (0)
    unk_token: u32,                    // Unknown token
    
    pub fn loadFromModel(allocator, model: *GGUFModel) !Tokenizer;
    pub fn encode(self, text: []const u8) ![]u32;
    pub fn decode(self, tokens: []const u32) ![]u8;
};
```

### 3. Quantization (`quantization/`)

Supports multiple GGML quantization formats with ~7:1 compression ratio.

| Format | Block Size | Bytes/Block | Values | Compression |
|--------|------------|-------------|--------|-------------|
| Q4_0 | 32 | 18 | 32 | ~7.1:1 |
| Q8_0 | 32 | 36 | 32 | ~3.6:1 |
| Q4_K | 256 | 144 | 256 | ~7.1:1 |
| Q6_K | 256 | 210 | 256 | ~4.9:1 |

**Block Structures (`quantization/common.zig`):**

```zig
/// Q4_0: 18 bytes → 32 float values
pub const BlockQ4_0 = extern struct {
    scale: u16,      // f16 scale factor
    qs: [16]u8,      // 32x 4-bit values (packed, 2 per byte)
};

/// Q4_K: 144 bytes → 256 float values (K-quant format)
pub const BlockQ4_K = extern struct {
    d: u16,          // f16 super-block scale
    dmin: u16,       // f16 super-block min
    scales: [12]u8,  // Packed scales/mins for 8 sub-blocks
    qs: [128]u8,     // 256x 4-bit values
};
```

### 4. Transformer Layers (`core/transformer.zig`)

Implements the LLaMA transformer architecture with RMSNorm, attention, and SwiGLU FFN.

```zig
pub const TransformerConfig = struct {
    embed_dim: u32,
    ffn_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,     // Grouped Query Attention (GQA)
    head_dim: u32,
    rope_theta: f32,     // RoPE base frequency (default: 10000.0)
    rms_norm_eps: f32,   // RMSNorm epsilon (default: 1e-5)
};

pub fn computeTransformerLayer(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: TransformerWeights,
    cache: *kv_cache.KVCache,
    layer: u32,
    position: u32,
    config: TransformerConfig,
    rope_freqs: []const f32,
) !void;
```

**Layer Computation Flow:**
1. **Pre-attention RMSNorm**: `normed = RMSNorm(input, attn_norm)`
2. **Self-Attention**: Q/K/V projections → RoPE → Scaled dot-product → Output projection
3. **Residual Connection**: `residual1 = input + attn_out`
4. **Pre-FFN RMSNorm**: `normed = RMSNorm(residual1, ffn_norm)`
5. **Feed-Forward**: SwiGLU activation: `FFN(x) = (SiLU(x·W_gate) ⊙ x·W_up)·W_down`
6. **Residual Connection**: `output = residual1 + ffn_out`

### 5. KV Cache (`core/kv_cache.zig`)

Memory-efficient key-value caching for autoregressive generation.

```zig
pub const KVCache = struct {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    seq_pos: u32,              // Current position in sequence
    cache: [][]f32,            // [n_layers][2 * max_seq_len * kv_dim]

    pub fn store(self, layer: u32, keys: []const f32, values: []const f32) void;
    pub fn getKeys(self, layer: u32) []const f32;
    pub fn getValues(self, layer: u32) []const f32;
    pub fn advance(self) void;  // Increment position for next token
    pub fn reset(self) void;    // Clear for new sequence
};
```

**Memory Layout:**
```
cache[layer] = [keys_0, keys_1, ..., keys_max | values_0, values_1, ..., values_max]
                ←───── max_seq_len × kv_dim ─────→←───── max_seq_len × kv_dim ──────→
```

### 6. Batch Processor (`core/batch_processor.zig`, `batch/batch_inference.zig`)

Continuous batching for high-throughput server workloads.

```zig
pub const BatchConfig = struct {
    max_batch_size: u32,
    max_seq_len: u32,
    head_dim: u32,
    n_heads: u32,
    timeout_ms: u64 = 100,  // Batching wait time
};

pub const BatchRequest = struct {
    id: u64,
    tokens: []const u32,
    max_new_tokens: u32,
    temperature: f32 = 1.0,
};

pub const BatchState = struct {
    batch_embeddings: []f32,  // [max_batch_size, embed_dim]
    batch_hidden: []f32,      // [max_batch_size, embed_dim]
    batch_output: []f32,      // [max_batch_size, embed_dim]
};
```

---

## GPU Backend Architecture

### 1. CUDA Backend (`core/backend_cuda.zig`)

Implements the `ComputeBackend` interface for NVIDIA GPUs with Tensor Core support.

```zig
pub const CudaBackend = struct {
    context: *cuda_context.CudaContext,
    stream: cuda_streams.CudaStream,
    cublas_ctx: cublas.CublasContext,
    device_id: i32,
    has_tensor_cores: bool,    // Compute 7.0+ (T4 = 7.5)
    fp16_supported: bool,      // Compute 6.0+
    total_memory: usize,
    free_memory: usize,
    dequant_ctx: dequant.DequantContext,  // GPU dequantization
};
```

**Tesla T4 Optimizations:**
- Compute 7.5 with 320 Tensor Cores
- 16GB GDDR6 VRAM
- FP16 Tensor Core operations (8x faster than FP32 CUDA cores)
- cuBLAS GemmEx for mixed-precision matmul

### 2. GPU Weight Cache (`cuda/gpu_weight_cache.zig`)

Pre-dequantizes and caches all model weights on GPU in FP16 format.

```zig
pub const GpuLayerWeights = struct {
    // Attention weights (FP16 on GPU)
    wq: GpuTensor,     // [n_heads * head_dim, embed_dim]
    wk: GpuTensor,     // [n_kv_heads * head_dim, embed_dim]
    wv: GpuTensor,     // [n_kv_heads * head_dim, embed_dim]
    wo: GpuTensor,     // [embed_dim, n_heads * head_dim]

    // FFN weights (FP16 on GPU)
    w_gate: GpuTensor, // [hidden_dim, embed_dim]
    w_up: GpuTensor,   // [hidden_dim, embed_dim]
    w_down: GpuTensor, // [embed_dim, hidden_dim]
};

pub const GpuWeightCache = struct {
    token_embedding: GpuTensor,  // [vocab_size, embed_dim]
    output_weight: GpuTensor,    // [vocab_size, embed_dim]
    layers: []GpuLayerWeights,   // Per-layer FP16 weights
    total_gpu_memory: usize,     // ~2.05 GB for TinyLlama 1.1B
};
```

**Benefits:**
- Eliminates per-matmul CPU→GPU weight transfer
- Zero-copy GPU weight access during inference
- FP16 storage saves 50% memory vs FP32

### 3. GPU Inference (`cuda/gpu_inference.zig`)

Fully GPU-resident forward pass with minimal data transfer.

```zig
pub const GpuInference = struct {
    cublas_handle: cublas.cublasHandle_t,
    stream: ?*anyopaque,
    graph: ?cuda.cudaGraph_t,         // CUDA Graph for kernel replay
    graph_exec: ?cuda.cudaGraphExec_t,

    // Pre-allocated GPU activation buffers
    hidden_state: GpuTensor,   // [batch_size * embed_dim]
    residual: GpuTensor,       // [batch_size * embed_dim]
    attn_out: GpuTensor,       // [batch_size * embed_dim]
    q_proj: GpuTensor,         // [batch_size * n_heads * head_dim]
    k_proj: GpuTensor,         // [batch_size * n_kv_heads * head_dim]
    v_proj: GpuTensor,         // [batch_size * n_kv_heads * head_dim]
    ffn_gate: GpuTensor,       // [batch_size * hidden_dim]
    ffn_up: GpuTensor,         // [batch_size * hidden_dim]
    logits: GpuTensor,         // [batch_size * vocab_size]

    pub fn init(allocator, embed_dim, hidden_dim, n_heads, n_kv_heads, vocab_size, n_layers) !Self;
    pub fn initWithBatchSize(allocator, ..., batch_size: u32) !Self;
};
```

### 4. Dequantization Kernels (`cuda/kernels/dequant_kernels.cu`)

Custom CUDA kernels for GPU-side weight dequantization.

```cuda
// Q4_0: 18 bytes → 32 FP16 values
__global__ void dequant_q4_0_kernel(
    const uint8_t* input, __half* output, int num_blocks
);

// Q8_0: 36 bytes → 32 FP16 values
__global__ void dequant_q8_0_kernel(
    const uint8_t* input, __half* output, int num_blocks
);

// Q4_K: 144 bytes → 256 FP16 values (K-quant)
__global__ void dequant_q4_k_kernel(
    const uint8_t* input, __half* output, int num_blocks
);

// Q6_K: 210 bytes → 256 FP16 values
__global__ void dequant_q6_k_kernel(
    const uint8_t* input, __half* output, int num_blocks
);
```

**Compilation:**
```bash
nvcc -shared -o libdequant_kernels.so dequant_kernels.cu \
     -arch=sm_75 --compiler-options '-fPIC'
```

---

## Memory Management

### GPU Weight Caching Strategy

For TinyLlama 1.1B (Q4_K quantized → FP16 on GPU):

| Component | Size (FP16) | Notes |
|-----------|-------------|-------|
| Token Embedding | 128 MB | 32000 × 2048 × 2B |
| 22 Transformer Layers | ~1.85 GB | 7 weights/layer × 22 |
| Output Projection | 128 MB | 32000 × 2048 × 2B |
| **Total** | **~2.05 GB** | Fits in T4's 16GB VRAM |

### Activation Buffer Allocation

Activation buffers scale with batch size:

```
Buffer Size = batch_size × dimension × sizeof(f32)

For batch_size=8, embed_dim=2048:
- hidden_state: 8 × 2048 × 4 = 64 KB
- ffn_gate/up:  8 × 5632 × 4 = 176 KB each
- logits:       8 × 32000 × 4 = 1 MB
```

### KV Cache Memory

```
KV Cache Size = 2 × n_layers × max_seq_len × n_kv_heads × head_dim × sizeof(f32)

For TinyLlama (22 layers, 4 KV heads, 64 head_dim, 2048 max_seq_len):
= 2 × 22 × 2048 × 4 × 64 × 4 bytes = 46 MB per sequence
```

---

## Data Flow

### Model Loading Pipeline

```
┌─────────────┐    ┌───────────────┐    ┌──────────────────┐    ┌──────────────┐
│ GGUF File   │───▶│ GGUF Loader   │───▶│ GPU Dequant      │───▶│ GPU Weight   │
│ (Q4_K)      │    │ (mmap tensors)│    │ (Q4_K → FP16)    │    │ Cache (FP16) │
└─────────────┘    └───────────────┘    └──────────────────┘    └──────────────┘
```

### Inference Pipeline

```
┌─────────┐   ┌──────────┐   ┌───────────┐   ┌───────────────┐   ┌──────────┐
│ Tokens  │──▶│ Embedding│──▶│ Layer 0   │──▶│ Layer 1...N   │──▶│ Output   │
│ (CPU)   │   │ (GPU)    │   │ Attention │   │ Transformer   │   │ Logits   │
└─────────┘   └──────────┘   │ + FFN     │   │ Blocks        │   │ (GPU)    │
                             └───────────┘   └───────────────┘   └──────────┘
                                  │                                   │
                                  └────────── KV Cache ───────────────┘
```

### Batched Inference Pipeline

```
Requests    Batch         GPU Parallel        Sampling      Responses
────────    ─────         ────────────        ────────      ─────────
[Req 1] ─┐              ┌─ Token Embed ─┐   ┌─ Logits 1 ─┐  ┌─ [Resp 1]
[Req 2] ─┼─▶ [Batch] ──▶│   N × matmul  │──▶│ Top-k/p    │─▶├─ [Resp 2]
[Req 3] ─┤              │  (M=batch)    │   │ Temperature│  ├─ [Resp 3]
...      │              └───────────────┘   └────────────┘  │  ...
[Req N] ─┘                                                  └─ [Resp N]
```

