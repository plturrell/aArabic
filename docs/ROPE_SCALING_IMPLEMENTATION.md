# RoPE Scaling Implementation for Context Window Extension

## Overview

This document describes the implementation of RoPE (Rotary Position Embedding) scaling for dynamic context window extension in the nOpenaiServer. RoPE scaling allows models to handle sequences longer than their original training context without retraining.

## Implementation Date

**January 22, 2026** - Added support for linear, dynamic NTK-aware, and YaRN RoPE scaling methods.

## Architecture

### 1. Configuration Layer (`config_parser.zig`)

#### RopeScalingConfig Structure

```zig
pub const RopeScalingConfig = struct {
    type: RopeScalingType,
    factor: f32,  // Scaling factor (e.g., 2.0 for 2x extension)
    original_max_position_embeddings: u32,  // Original context window
    
    // YaRN-specific parameters
    attention_factor: ?f32 = null,
    beta_fast: ?f32 = null,
    beta_slow: ?f32 = null,
};
```

**Key Methods:**
- `getExtendedSeqLen()`: Calculates extended context length (original × factor)
- `isWithinOriginalRange()`: Checks if position needs scaling

#### Parsing from HuggingFace Config

The parser automatically detects and loads RoPE scaling configuration from `config.json`:

```json
{
  "max_position_embeddings": 4096,
  "rope_scaling": {
    "type": "linear",
    "factor": 8.0,
    "original_max_position_embeddings": 4096
  }
}
```

### 2. Attention Layer (`attention.zig`)

#### Three Scaling Methods Implemented

##### a) Linear Scaling

**Use Case**: Simple, uniform scaling for moderate extensions (2-4x)

**Formula**: `freq_scaled = freq / scale_factor`

**Characteristics:**
- Simplest method
- Uniform scaling across all dimensions
- Works well for 2-4x extensions
- May cause perplexity degradation at higher scales

**Implementation:**
```zig
.linear => {
    base_freq / config.factor
}
```

##### b) Dynamic NTK-Aware Scaling

**Use Case**: Code Llama style, better for longer extensions

**Formula**: `scale = (length / original_length) ^ (dim / (dim - 2))`

**Characteristics:**
- Only applies beyond original context window
- Dynamically adjusts based on current position
- Better preservation of model quality
- Dimension-aware interpolation

**Implementation:**
```zig
.dynamic => {
    if (position < config.original_max_position_embeddings) {
        return base_freq;  // No scaling within original range
    }
    
    const length_ratio = pos_f / original_max;
    const exponent = dim_f / (dim_f - 2.0);
    const ntk_scale = std.math.pow(f32, length_ratio, exponent);
    
    base_freq / ntk_scale
}
```

##### c) YaRN (Yet another RoPE extensioN)

**Use Case**: Most sophisticated, best quality for long contexts

**Formula**: Interpolates between low and high frequency dimensions

**Characteristics:**
- Dimension-specific scaling
- Low frequencies (high wavelengths) get more scaling
- High frequencies (low wavelengths) get less scaling
- Best quality retention at extreme extensions (8x+)

**Parameters:**
- `attention_factor`: Overall attention scaling (default: 1.0)
- `beta_fast`: Fast dimension threshold (default: 32.0)
- `beta_slow`: Slow dimension threshold (default: 1.0)

**Implementation:**
```zig
.yarn => {
    if (position < config.original_max_position_embeddings) {
        return base_freq;
    }
    
    const attn_factor = config.attention_factor orelse 1.0;
    const beta_fast = config.beta_fast orelse 32.0;
    const beta_slow = config.beta_slow orelse 1.0;
    
    // Compute interpolation ramp
    const ramp = (dim_idx_f / dim_f - beta_fast) / (beta_slow - beta_fast);
    const ramp_clamped = @max(0.0, @min(1.0, ramp));
    
    // Interpolate scale factor
    const scale = config.factor * ramp_clamped + 1.0 * (1.0 - ramp_clamped);
    
    base_freq / (scale * attn_factor)
}
```

### 3. Frequency Precomputation

The `precomputeRopeFreqs()` function now supports scaling:

```zig
pub fn precomputeRopeFreqs(
    allocator: std.mem.Allocator,
    head_dim: u32,
    max_seq_len: u32,  // Extended length if scaling enabled
    theta: f32,
    scaling_config: ?RopeScalingConfig,
) ![]f32
```

**Process:**
1. Compute base frequencies for each dimension
2. For each position, apply scaling based on configuration
3. Compute cos/sin values for rotary embedding
4. Store in precomputed array for fast lookup

## Usage Examples

### Example 1: Linear Scaling (8x Extension)

**Model Config:**
```json
{
  "max_position_embeddings": 4096,
  "rope_theta": 10000.0,
  "rope_scaling": {
    "type": "linear",
    "factor": 8.0
  }
}
```

**Result:**
- Original context: 4,096 tokens
- Extended context: **32,768 tokens**
- Method: Uniform frequency scaling

### Example 2: Dynamic NTK Scaling (4x Extension)

**Model Config:**
```json
{
  "max_position_embeddings": 8192,
  "rope_theta": 10000.0,
  "rope_scaling": {
    "type": "dynamic",
    "factor": 4.0,
    "original_max_position_embeddings": 8192
  }
}
```

**Result:**
- Original context: 8,192 tokens
- Extended context: **32,768 tokens**
- Method: NTK-aware dynamic scaling

### Example 3: YaRN Scaling (16x Extension)

**Model Config:**
```json
{
  "max_position_embeddings": 2048,
  "rope_theta": 10000.0,
  "rope_scaling": {
    "type": "yarn",
    "factor": 16.0,
    "attention_factor": 1.0,
    "beta_fast": 32.0,
    "beta_slow": 1.0
  }
}
```

**Result:**
- Original context: 2,048 tokens
- Extended context: **32,768 tokens**
- Method: YaRN dimension-aware interpolation

## Memory Requirements

### KV Cache Allocation

The KV cache must be allocated for the **extended** sequence length:

```zig
const extended_max_seq_len = if (rope_scaling_config) |rsc|
    rsc.getExtendedSeqLen()
else
    config.max_seq_len;

// Allocate cache for extended context
const cache = try KVCache.init(
    allocator,
    n_layers,
    n_kv_heads,
    head_dim,
    extended_max_seq_len,
);
```

### Memory Calculation

```
KV Cache Size = n_layers × 2 × extended_seq_len × n_kv_heads × head_dim × sizeof(f32)

Example (7B model, 8x extension):
- Original: 4096 tokens → 1 GB
- Extended: 32768 tokens → 8 GB
```

## Performance Characteristics

| Method | Extension | Quality | Speed | Memory |
|--------|-----------|---------|-------|--------|
| Linear | 2-4x | Good | Fast | +100-300% |
| Dynamic | 4-8x | Better | Fast | +300-700% |
| YaRN | 8-16x+ | Best | Fast | +700-1500% |

**Notes:**
- Speed impact is negligible (precomputed frequencies)
- Memory scales linearly with context extension
- Quality retention varies by method and model

## Integration Points

### 1. Model Loading

When loading a model, the system:
1. Parses `rope_scaling` from config.json
2. Creates `RopeScalingConfig` if present
3. Calculates extended context length
4. Allocates KV cache for extended length

### 2. Attention Computation

During inference:
1. Uses precomputed scaled frequencies
2. Applies same RoPE rotation as before
3. No runtime performance penalty
4. Automatic scaling beyond original context

### 3. GGUF Models

For GGUF models, the system:
1. Reads `rope_scaling` from GGUF metadata
2. Parses scaling parameters
3. Applies same scaling logic
4. Works identically to HuggingFace models

## Backward Compatibility

**Models without RoPE scaling:**
- `rope_scaling_config` is `null`
- System uses original context window
- No changes to existing behavior
- Zero performance impact

**Models with RoPE scaling:**
- Automatically detected and enabled
- Context extended based on configuration
- Transparent to user

## Testing

### Test Scenarios

1. **No Scaling**: Model without rope_scaling config
2. **Linear Scaling**: 2x, 4x, 8x extensions
3. **Dynamic Scaling**: Various extension factors
4. **YaRN Scaling**: Extreme extensions (16x+)

### Validation

```zig
// Test scaling detection
if (config.rope_scaling_config) |rsc| {
    assert(rsc.factor > 1.0);
    assert(rsc.getExtendedSeqLen() > rsc.original_max_position_embeddings);
}

// Test frequency computation
const scaled_freq = computeScaledFreq(base_freq, position, dim_idx, head_dim, scaling_config);
assert(scaled_freq > 0.0);
```

## Debugging

### Enable Debug Logging

The system logs scaling information during initialization:

```
🔄 RoPE Scaling enabled: linear
   Factor: 8.00x, Original: 4096 → Extended: 32768
```

### Common Issues

1. **Out of Memory**: Extended context requires more RAM
   - Solution: Reduce scaling factor or use tiered cache

2. **Quality Degradation**: Extreme scaling without proper method
   - Solution: Use YaRN for extensions >8x

3. **Missing Configuration**: rope_scaling not detected
   - Solution: Verify config.json format matches spec

## Future Enhancements

### Potential Improvements

1. **Adaptive Scaling**: Auto-select method based on extension factor
2. **Mixed Precision**: F16 frequencies to reduce memory
3. **On-the-fly Scaling**: Compute frequencies on demand
4. **Model-specific Tuning**: Per-architecture optimal parameters

### Research Directions

1. **Learned Scaling**: Train scaling parameters
2. **Hybrid Methods**: Combine multiple scaling approaches
3. **Compression**: Reduce frequency table size
4. **Hardware Optimization**: GPU-accelerated frequency lookup

## References

### Papers

1. **Linear RoPE Scaling**: "Extending Context Window of Large Language Models via Positional Interpolation"
2. **Dynamic NTK Scaling**: Code Llama paper
3. **YaRN**: "YaRN: Efficient Context Window Extension of Large Language Models"

### Implementation Sources

- HuggingFace Transformers: rope_scaling configuration format
- llama.cpp: Reference implementations
- vLLM: Production deployment patterns

## Summary

The RoPE scaling implementation enables:

✅ **Automatic Detection**: Reads scaling from model config  
✅ **Three Methods**: Linear, Dynamic NTK, YaRN  
✅ **Zero Config**: Works automatically if metadata present  
✅ **Backward Compatible**: No impact on models without scaling  
✅ **Memory Efficient**: Only allocates extended cache when needed  
✅ **Production Ready**: Tested with multiple architectures  

Models can now handle **8-16x longer contexts** than their original training window with minimal quality loss using appropriate scaling methods.
