# Context Window Architecture for nOpenaiServer

## Overview

This document describes how context windows and sliding window mechanisms are implemented in the nOpenaiServer for single-model inference. Each model in the system has a **fixed maximum context window** that is determined by its architecture and configuration.

## Architecture Components

### 1. Model Configuration (`config_parser.zig`)

Each model's context window is defined in its HuggingFace `config.json` and parsed through the `ConfigParser`:

```zig
pub const ModelConfig = struct {
    // Context and sequence settings
    max_position_embeddings: usize,  // Fixed max context window
    sliding_window: ?usize,           // Optional sliding window size
    
    // ... other fields
};
```

**Key Parameters:**
- **`max_position_embeddings`**: The absolute maximum context length supported by the model architecture (e.g., 2048, 4096, 8192, 32768 tokens)
- **`sliding_window`**: Optional parameter for models like Mistral that use sliding window attention (e.g., 4096 tokens)

### 2. Model Registry (`config.json`)

Models are registered with their tier configurations that include cache memory allocations:

```json
{
    "id": "hymt-1.5-7b-q4km",
    "tier_config": {
        "max_ram_mb": 6000,
        "kv_cache_ram_mb": 1024
    }
}
```

**Tier Configuration:**
- `max_ram_mb`: Total RAM budget for the model
- `kv_cache_ram_mb`: Dedicated memory for KV cache (affects how much context can be cached)

### 3. KV Cache Management

#### A. Tiered KV Cache (`kv_cache_tiered.zig`)

The tiered cache system provides automatic memory management:

```zig
pub const TieredKVCacheConfig = struct {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,           // Fixed max context window
    
    // Tiering options
    hot_tokens: u32 = 2048,     // Tokens kept in RAM
    ssd_path: []const u8,
    max_ssd_mb: u32 = 16384,    // SSD tier for cold tokens
    enable_tiering: bool = true,
};
```

**Two-Tier Architecture:**
1. **Hot Tier (RAM)**: Recent tokens (default: 2048) kept in fast memory
2. **Cold Tier (SSD)**: Older tokens offloaded to SSD storage when capacity exceeded

#### B. Cache Management Strategies (`cache_manager.zig`)

Multiple strategies for managing the fixed context window:

```zig
pub const CacheStrategy = enum {
    fifo,           // First-in-first-out (drop oldest)
    sliding_window, // Keep last N tokens (rolling window)
    keep_first,     // Keep first tokens, drop middle (prefix caching)
    adaptive,       // Dynamic based on importance
};
```

**Sliding Window Implementation:**
```zig
pub const ManagedCacheConfig = struct {
    max_seq_len: u32,           // Max context window (fixed per model)
    window_size: u32 = 1024,    // Sliding window size
    strategy: CacheStrategy = .sliding_window,
};
```

When using sliding window strategy:
- **`max_seq_len`**: The model's fixed maximum context window
- **`window_size`**: The actual sliding window size (must be ≤ max_seq_len)
- Oldest tokens are automatically evicted when window_size is reached
- Maintains constant memory usage regardless of input length

### 4. Request Validation (`openai_http_server.zig`)

API requests are validated against context limits:

```zig
pub const CompletionRequest = struct {
    model: []const u8,
    prompt: []const u8,
    max_tokens: ?u32 = null,    // Generation limit (1-100,000)
    temperature: ?f32 = null,
    // ...
};

// Validation
if (request.max_tokens) |max_tokens| {
    if (max_tokens == 0 or max_tokens > 100000) {
        return ValidationError.InvalidMaxTokens;
    }
}
```

**Request Parameters:**
- `max_tokens`: Maximum tokens to **generate** (not total context)
- Combined with prompt length, must fit within model's `max_position_embeddings`

### 5. Resource Quotas (`resource_quotas.zig`)

Rate limiting and quota management per tier:

```zig
pub const ResourceQuota = struct {
    max_tokens_per_second: f32 = 1000.0,
    max_tokens_per_hour: u64 = 1_000_000,
    max_tokens_per_day: u64 = 10_000_000,
    max_requests_per_hour: u64 = 10_000,
    // ...
};
```

## Context Window Per Model Type

### Small Models (1-2B parameters)
```json
{
    "id": "lfm2.5-1.2b-f16",
    "max_position_embeddings": 2048,
    "tier_config": {
        "max_ram_mb": 3000,
        "kv_cache_ram_mb": 512
    }
}
```
- **Context Window**: 2048 tokens (fixed)
- **KV Cache**: 512 MB RAM
- **Use Case**: Short conversations, code snippets

### Medium Models (7B parameters)
```json
{
    "id": "hymt-1.5-7b-q4km",
    "max_position_embeddings": 4096,
    "sliding_window": 4096,
    "tier_config": {
        "max_ram_mb": 6000,
        "kv_cache_ram_mb": 1024
    }
}
```
- **Context Window**: 4096 tokens (fixed)
- **Sliding Window**: 4096 tokens (full attention)
- **KV Cache**: 1024 MB RAM
- **Use Case**: Standard conversations, document Q&A

### Large Models (33B+ parameters)
```json
{
    "id": "deepseek-coder-33b",
    "max_position_embeddings": 16384,
    "tier_config": {
        "max_ram_mb": 24000,
        "kv_cache_ram_mb": 4096,
        "max_ssd_mb": 8192
    }
}
```
- **Context Window**: 16,384 tokens (fixed)
- **KV Cache**: 4096 MB RAM + 8192 MB SSD tier
- **Hot Tokens**: ~2048 in RAM
- **Cold Tokens**: Remaining in SSD tier
- **Use Case**: Long documents, codebases

### Very Large Models (70B+ parameters)
```json
{
    "id": "llama-3.3-70b",
    "max_position_embeddings": 32768,
    "tier_config": {
        "max_ram_mb": 48000,
        "kv_cache_ram_mb": 8192,
        "max_ssd_mb": 16384,
        "enable_distributed": true
    }
}
```
- **Context Window**: 32,768 tokens (fixed)
- **KV Cache**: 8192 MB RAM + 16384 MB SSD tier
- **Distributed**: Supports multi-GPU/multi-node
- **Use Case**: Very long contexts, book-length documents

## Sliding Window Attention

For models with sliding window attention (e.g., Mistral architecture):

```
Model Context: 32768 tokens (max_position_embeddings)
Sliding Window: 4096 tokens (sliding_window)

┌─────────────────────────────────────┐
│   Total Context Buffer (32768)      │
│                                      │
│  [Prefix Cache]                      │
│       ↓                              │
│  ┌──────────────┐                   │
│  │  Window 4096 │ ← Active attention│
│  └──────────────┘                   │
│       ↑                              │
│  [Old tokens]                        │
└─────────────────────────────────────┘
```

**Behavior:**
- Model can **store** up to 32,768 tokens in context
- Each new token only **attends to** last 4,096 tokens
- Reduces computation from O(n²) to O(n×window_size)
- Maintains fixed memory and compute per token

## Memory Management Flow

```
User Request
    ↓
[Tokenize Prompt] → token_count
    ↓
[Check: prompt + max_tokens ≤ max_position_embeddings]
    ↓
[Allocate KV Cache]
    ├─→ [RAM Cache] (hot_tokens)
    └─→ [SSD Tier] (cold tokens, if enabled)
    ↓
[Generation Loop]
    ├─→ [Store KV] → Check cache strategy
    │       ├─→ sliding_window: evict if > window_size
    │       ├─→ fifo: evict oldest
    │       └─→ keep_first: evict middle
    ├─→ [Forward Pass]
    └─→ [Generate Token]
    ↓
[Return Response]
```

## Configuration Best Practices

### 1. Match Window to Use Case

```zig
// Short conversations
.window_size = 1024,
.hot_tokens = 1024,

// Standard documents
.window_size = 4096,
.hot_tokens = 2048,

// Long contexts
.window_size = 8192,
.hot_tokens = 2048,
.enable_tiering = true,
```

### 2. Memory Budget

```
KV Cache Size = n_layers × 2 × seq_len × n_heads × head_dim × sizeof(f32)

Example (7B model):
- Layers: 32
- Heads: 32  
- Head dim: 128
- Sequence: 4096

RAM = 32 × 2 × 4096 × 32 × 128 × 4 bytes
    = 32 × 2 × 4096 × 32 × 128 × 4
    = 1,073,741,824 bytes
    = 1024 MB
```

### 3. Tiering Strategy

```zig
// Enable tiering for long contexts
if (max_seq_len > 8192) {
    config.enable_tiering = true;
    config.hot_tokens = 2048;
    config.max_ssd_mb = max_seq_len * kv_size_per_token / 1024;
}
```

## API Usage

### Request with Context Window

```bash
curl -X POST http://localhost:9632/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hymt-1.5-7b-q4km",
    "prompt": "Your prompt here",
    "max_tokens": 512,
    "temperature": 0.7
  }'
```

**Constraints:**
- `prompt.length + max_tokens ≤ model.max_position_embeddings`
- Model automatically manages context window with configured strategy
- Sliding window attention (if supported) automatically applied

## Performance Characteristics

| Model Size | Context | RAM Cache | SSD Tier | Tokens/sec |
|------------|---------|-----------|----------|------------|
| 1.2B       | 2K      | 512 MB    | -        | 50-100     |
| 7B         | 4K      | 1 GB      | -        | 20-40      |
| 33B        | 16K     | 4 GB      | 8 GB     | 5-10       |
| 70B        | 32K     | 8 GB      | 16 GB    | 2-5        |

## Summary

The nOpenaiServer implements **fixed context windows per model** with intelligent caching strategies:

1. **Fixed Window**: Each model has a `max_position_embeddings` defined by its architecture
2. **Sliding Window**: Optional attention mechanism that limits attention span while maintaining full context storage
3. **Tiered Cache**: Automatic RAM/SSD tiering for memory efficiency
4. **Multiple Strategies**: FIFO, sliding window, keep-first, and adaptive eviction
5. **Per-Request Control**: `max_tokens` controls generation length, not total context

The system ensures that regardless of input length, memory usage remains bounded by the configured limits while maximizing context utilization through intelligent cache management.
