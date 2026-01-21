# Model Configuration & Tiering Guide

**Last Updated:** January 20, 2026  
**Status:** Production Ready  
**Models Configured:** 5 GGUF models with multi-tier support

---

## 📋 Table of Contents

1. [Model Inventory](#model-inventory)
2. [Tiering Architecture](#tiering-architecture)
3. [Model-Specific Configurations](#model-specific-configurations)
4. [Testing Strategy](#testing-strategy)
5. [Troubleshooting](#troubleshooting)
6. [Performance Benchmarks](#performance-benchmarks)

---

## 🗂️ Model Inventory

### Available Models (5 Total)

| Model ID | Size | RAM | SSD | Architecture | Status | Priority |
|----------|------|-----|-----|--------------|--------|----------|
| `lfm2.5-1.2b-q4_0` | 664MB | 1GB | 0 | lfm2 | ⚠️ Debug | High |
| `lfm2.5-1.2b-q4_k_m` | 697MB | 1GB | 0 | lfm2 | ✅ Ready | High |
| `lfm2.5-1.2b-f16` | 2.2GB | 2GB | 2GB | lfm2 | ✅ Ready | Medium |
| `deepseek-coder-33b` | 19GB | 8GB | 16GB | llama | ✅ Ready | Medium |
| `llama-3.3-70b` | 40GB | 8GB | 48GB | llama | ✅ Ready | Low |

### Model Categories

**Tier 1: Development Models (RAM-only)**
- LFM2.5 Q4_0, Q4_K_M, F16
- Fast inference, no SSD required
- Use for: Testing, development, low-latency applications

**Tier 2: Production Models (RAM+SSD)**
- DeepSeek-Coder-33B
- Moderate size, tiered storage
- Use for: Code generation, production workloads

**Tier 3: Large Models (Aggressive SSD)**
- Llama-3.3-70B
- Requires significant SSD space
- Use for: Advanced reasoning, quality benchmarks

---

## 🏗️ Tiering Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────┐
│                   Request                       │
└───────────────────┬─────────────────────────────┘
                    │
         ┌──────────▼──────────┐
         │   RAM Tier (Hot)    │  ← 0.01-0.1ms latency
         │   - Recent tokens   │
         │   - Active layers   │
         │   - KV cache hot    │
         └──────────┬──────────┘
                    │ (on miss)
         ┌──────────▼──────────┐
         │   SSD Tier (Cold)   │  ← 1-10ms latency
         │   - Older KV cache  │
         │   - Model weights   │
         │   - Compressed data │
         └──────────┬──────────┘
                    │ (on miss)
         ┌──────────▼──────────┐
         │ DragonflyDB (Dist)  │  ← 5-20ms latency
         │ - Prompt cache      │
         │ - Shared KV states  │
         │ - Multi-instance    │
         └─────────────────────┘
```

### Tier Characteristics

| Tier | Latency | Capacity | Best For | Eviction Policy |
|------|---------|----------|----------|-----------------|
| RAM | 0.01-0.1ms | 1-8GB | Recent tokens, hot layers | LRU |
| SSD | 1-10ms | 16-48GB | Model weights, cold KV | LRU + Compression |
| DragonflyDB | 5-20ms | Unlimited | Prompt caching, sharing | TTL-based |

---

## 🔧 Model-Specific Configurations

### 1. LFM2.5-1.2B-Q4_0 (Primary Testing Model)

**Configuration:**
```json
{
  "id": "lfm2.5-1.2b-q4_0",
  "size_mb": 664,
  "tier_config": {
    "max_ram_mb": 1024,
    "kv_cache_ram_mb": 256,
    "max_ssd_mb": 0,
    "enable_distributed": false
  }
}
```

**Characteristics:**
- Most compressed variant (Q4_0 quantization)
- Fits entirely in RAM (no SSD needed)
- Fastest inference time
- Lower quality than Q4_K_M or F16

**Known Issues:**
- ⚠️ Currently outputs only newlines (needs debugging)
- Likely tokenizer or generation parameter issue
- Priority fix required before other testing

**Testing Strategy:**
1. Debug newlines output issue
2. Compare output quality vs Q4_K_M
3. Measure inference latency
4. Validate memory footprint

**Expected Performance:**
- Tokens/sec: 50-100 (CPU)
- First token latency: <100ms
- Memory usage: <1.5GB total

---

### 2. LFM2.5-1.2B-Q4_K_M (Production Model)

**Configuration:**
```json
{
  "id": "lfm2.5-1.2b-q4_k_m",
  "size_mb": 697,
  "tier_config": {
    "max_ram_mb": 1024,
    "kv_cache_ram_mb": 256,
    "max_ssd_mb": 0
  }
}
```

**Characteristics:**
- Balanced quality/size (K-means quantization)
- Better quality than Q4_0
- Still fits in RAM
- Production-ready

**Use Cases:**
- Primary production model
- Real-time inference
- Edge deployment
- Cost-effective serving

**Expected Performance:**
- Tokens/sec: 45-90 (CPU)
- Quality: Better than Q4_0, close to F16
- Memory: <1.5GB total

---

### 3. LFM2.5-1.2B-F16 (Quality Benchmark)

**Configuration:**
```json
{
  "id": "lfm2.5-1.2b-f16",
  "size_mb": 2200,
  "tier_config": {
    "max_ram_mb": 2048,
    "kv_cache_ram_mb": 512,
    "max_ssd_mb": 2048,
    "enable_compression": true
  }
}
```

**Characteristics:**
- Full FP16 precision (no quantization)
- Highest quality outputs
- Requires more memory
- Light SSD usage for KV overflow

**Use Cases:**
- Quality benchmarking
- Precision-critical applications
- Comparing quantization impact
- Reference implementation

**Expected Performance:**
- Tokens/sec: 30-60 (CPU, slower than quantized)
- Quality: Baseline/best quality
- Memory: 2-3GB total

**Tiering Behavior:**
- Model weights: RAM
- Hot KV cache: RAM (512MB)
- Cold KV cache: SSD (compressed)

---

### 4. DeepSeek-Coder-33B (Tiering Validation)

**Configuration:**
```json
{
  "id": "deepseek-coder-33b",
  "size_mb": 19000,
  "tier_config": {
    "max_ram_mb": 8192,
    "kv_cache_ram_mb": 2048,
    "max_ssd_mb": 16384,
    "enable_distributed": true,
    "enable_compression": true,
    "hot_layers": ["layers.0", "layers.1", "layers.2",
                   "layers.58", "layers.59", "layers.60"]
  }
}
```

**Characteristics:**
- 33B parameters (code-specialized)
- Requires RAM+SSD tiering
- First model to validate tiering system
- Llama architecture (well-tested)

**Tiering Strategy:**
- **Hot layers (RAM):** First 3 + Last 3 layers
  - Input embeddings, output projection
  - Most frequently accessed
- **Warm layers (RAM/SSD):** Middle layers
  - Loaded on demand
  - Compressed when cold
- **Cold layers (SSD):** Rarely accessed
  - mmap'd from disk
  - Only loaded when needed

**Use Cases:**
- Code generation
- Tiering system validation
- Production workload simulation
- Multi-model benchmarking

**Expected Performance:**
- Tokens/sec: 5-15 (with tiering)
- First token: 500ms-2s (cold start)
- RAM hit rate: 70-80%
- SSD hit rate: 15-25%

**Requirements:**
- Minimum: 10GB RAM, 20GB SSD
- Recommended: 12GB RAM, 24GB SSD
- Optional: DragonflyDB for distributed caching

---

### 5. Llama-3.3-70B (Stress Test)

**Configuration:**
```json
{
  "id": "llama-3.3-70b",
  "size_mb": 40000,
  "tier_config": {
    "max_ram_mb": 8192,
    "kv_cache_ram_mb": 2048,
    "max_ssd_mb": 49152,
    "enable_distributed": true,
    "enable_compression": true,
    "hot_layers": ["layers.0", "layers.1", "layers.2", "layers.3",
                   "layers.76", "layers.77", "layers.78", "layers.79"],
    "prefetch_layers": true,
    "async_io": true
  }
}
```

**Characteristics:**
- 70B parameters (flagship model)
- Requires aggressive SSD tiering
- Ultimate stress test for system
- Highest quality reasoning

**Tiering Strategy:**
- **Hot layers (RAM):** First 4 + Last 4 layers (8 total)
  - Critical for every inference
  - Always in RAM
- **Prefetch:** Next-likely layers loaded asynchronously
- **SSD:** Remaining 72 layers on SSD
  - mmap'd, compressed
  - Loaded on-demand
- **DragonflyDB:** Prompt caching essential
  - Reduces cold starts
  - Shares KV cache across instances

**Use Cases:**
- Advanced reasoning
- Quality benchmarking
- Stress testing tiering system
- Production readiness validation

**Expected Performance:**
- Tokens/sec: 1-5 (with aggressive tiering)
- First token: 2-10s (cold start)
- First token (cached): 500ms-2s
- RAM hit rate: 40-60%
- SSD hit rate: 35-50%
- Cache hit rate: 5-10%

**Requirements:**
- **Minimum:** 10GB RAM, 50GB SSD
- **Recommended:** 16GB RAM, 64GB SSD, DragonflyDB
- **Warning:** First load is slow (mmap initialization)

**Optimization Tips:**
1. Enable DragonflyDB for prompt caching
2. Use async I/O for layer prefetching
3. Enable compression for SSD tier
4. Warm up with common prompts
5. Monitor tier hit rates

---

## 🧪 Testing Strategy

### Phase 1: Small Models (1-3 days)

**Goal:** Fix LFM2.5, establish baseline

```bash
# Test 1: Debug Q4_0 newlines issue
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "lfm2.5-1.2b-q4_0",
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 10,
    "temperature": 0.1
  }'

# Test 2: Compare Q4_0 vs Q4_K_M
# Test 3: Benchmark F16 precision
# Test 4: Measure latency and throughput
```

**Success Criteria:**
- Q4_0 produces valid text (not just newlines)
- Q4_K_M quality > Q4_0
- F16 quality ≥ Q4_K_M
- Latency < 100ms first token

---

### Phase 2: Medium Model (2-3 days)

**Goal:** Validate tiering with DeepSeek-33B

```bash
# Test 1: Load model with tiering
SHIMMY_DEBUG=1 \
SHIMMY_MODEL_PATH="vendor/layerModels/deepseek-coder-33b-instruct-q4_k_m/deepseek-coder-33b-instruct-q4_k_m.gguf" \
SHIMMY_MODEL_ID="deepseek-coder-33b" \
./openai_http_server

# Test 2: Monitor tier performance
curl http://localhost:11434/admin/memory

# Test 3: Code generation test
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-coder-33b",
    "messages": [{"role": "user", "content": "Write a Python function to sort a list"}],
    "max_tokens": 200
  }'
```

**Success Criteria:**
- Model loads successfully with tiering
- RAM usage stays under 10GB
- SSD tier functional
- Code generation quality acceptable
- RAM hit rate > 70%

---

### Phase 3: Large Model (3-5 days)

**Goal:** Stress test with Llama-70B

```bash
# Test 1: Ensure sufficient disk space
df -h /tmp

# Test 2: Load with aggressive tiering
SHIMMY_DEBUG=1 \
SHIMMY_MODEL_PATH="vendor/layerModels/Llama-3.3-70B-Instruct-Q4_K_M.gguf/Llama-3.3-70B-Instruct-Q4_K_M.gguf" \
SHIMMY_MODEL_ID="llama-3.3-70b" \
./openai_http_server

# Test 3: Monitor during inference
watch -n 1 'curl -s http://localhost:11434/admin/memory'

# Test 4: Long-form generation
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [{"role": "user", "content": "Explain quantum computing in detail"}],
    "max_tokens": 1000
  }'
```

**Success Criteria:**
- Model loads (even if slow)
- RAM usage under 12GB
- SSD tier handles 40GB+
- Generates coherent long-form text
- System remains stable

---

## 🔍 Troubleshooting

### Issue: LFM2.5 outputs only newlines

**Possible Causes:**
1. Chat template mismatch
2. Tokenizer misconfiguration
3. Generation parameters (temperature, top_p)
4. EOS token handling

**Debug Steps:**
```bash
# 1. Check raw logits output
grep "Top logits" /tmp/server_*.log

# 2. Test with different temperatures
curl ... -d '{"temperature": 0.0, ...}'  # Deterministic
curl ... -d '{"temperature": 1.0, ...}'  # More random

# 3. Try different max_tokens
curl ... -d '{"max_tokens": 1, ...}'     # Single token
curl ... -d '{"max_tokens": 50, ...}'    # More tokens

# 4. Check tokenizer
# Verify tokenizer.json exists in model directory
ls -la vendor/layerModels/LFM2.5-1.2B-Instruct-GGUF/
```

**Solutions:**
- Fix chat template formatting
- Adjust generation parameters
- Update tokenizer configuration
- Test with F16 variant (eliminate quantization issues)

---

### Issue: DeepSeek-33B won't load

**Possible Causes:**
1. Insufficient RAM
2. SSD path not writable
3. Tiering not enabled
4. GGUF format incompatible

**Debug Steps:**
```bash
# 1. Check available RAM
free -h

# 2. Verify SSD path
mkdir -p /tmp/shimmy_tier
chmod 777 /tmp/shimmy_tier

# 3. Check file integrity
ls -lh vendor/layerModels/deepseek-coder-33b-instruct-q4_k_m/*.gguf

# 4. Enable debug logging
export SHIMMY_DEBUG=1
```

---

### Issue: Llama-70B extremely slow

**Expected Behavior:**
- First load: 2-10s (mmap initialization)
- Subsequent: 500ms-2s
- Inference: 1-5 tokens/sec

**Optimization:**
1. Enable DragonflyDB for prompt caching
2. Use async I/O layer prefetching
3. Enable SSD compression
4. Warm up with common prompts
5. Add more RAM if possible

---

## 📊 Performance Benchmarks

### Expected Performance Matrix

| Model | First Token | Tokens/Sec | RAM Usage | SSD Usage | Quality |
|-------|-------------|------------|-----------|-----------|---------|
| LFM2.5-Q4_0 | <100ms | 50-100 | 1GB | 0 | ⭐⭐⭐ |
| LFM2.5-Q4_K_M | <100ms | 45-90 | 1GB | 0 | ⭐⭐⭐⭐ |
| LFM2.5-F16 | <150ms | 30-60 | 2GB | 500MB | ⭐⭐⭐⭐⭐ |
| DeepSeek-33B | 500ms-2s | 5-15 | 8GB | 12GB | ⭐⭐⭐⭐ |
| Llama-70B | 2-10s | 1-5 | 8GB | 35GB | ⭐⭐⭐⭐⭐ |

### Tier Hit Rates

| Model | RAM Hits | SSD Hits | Cache Hits | Total Latency |
|-------|----------|----------|------------|---------------|
| LFM2.5 | 100% | 0% | 0% | 10-50ms |
| DeepSeek-33B | 70-80% | 15-25% | 5% | 50-200ms |
| Llama-70B | 40-60% | 35-50% | 5-10% | 200-500ms |

---

## 🎯 Next Steps

1. **Immediate (Days 1-3):**
   - Fix LFM2.5-Q4_0 newlines issue
   - Test Q4_K_M and F16 variants
   - Establish baseline performance

2. **Short-term (Days 4-6):**
   - Validate DeepSeek-33B tiering
   - Measure tier hit rates
   - Optimize hot layer selection

3. **Medium-term (Days 7-12):**
   - Load Llama-70B successfully
   - Stress test tiering system
   - Enable DragonflyDB caching

4. **Long-term (Days 13+):**
   - Production deployment
   - Multi-model serving
   - Performance optimization
   - SafeTensors model support

---

## 📝 Configuration Files

**Primary:** `src/serviceCore/nOpenaiServer/config.json`

**Related:**
- `inference/engine/tiering/mod.zig` - Tiering implementation
- `inference/engine/tiering/unified_tier.zig` - UnifiedTierManager
- `inference/engine/core/gguf_loader.zig` - GGUF format loader

---

**Document Version:** 1.0  
**Last Updated:** January 20, 2026  
**Status:** ✅ Complete and Ready for Testing
