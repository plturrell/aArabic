# mHC Migration Guide

**Document Version**: 1.0  
**Date**: January 19, 2026  
**Reference**: MHC_INTEGRATION_TECHNICAL_SPEC.md  
**Status**: Active

---

## Table of Contents

1. [Upgrading from Non-mHC to mHC](#1-upgrading-from-non-mhc-to-mhc)
2. [Configuration Migration](#2-configuration-migration)
3. [API Changes](#3-api-changes)
4. [Code Migration Examples](#4-code-migration-examples)
5. [Migration Checklist](#5-migration-checklist)
6. [Rollback Procedures](#6-rollback-procedures)

---

## 1. Upgrading from Non-mHC to mHC

### 1.1 Overview

Migrating to mHC is designed to be **backward compatible**. Your existing code will continue to work, and you can enable mHC features incrementally.

### 1.2 Migration Phases

**Phase 1: Preparation** (No changes required)
- Update to latest nOpenaiServer version
- Review current configuration
- Test with mHC disabled

**Phase 2: Auto-Detection** (Minimal changes)
- Enable `auto_detect: true`
- mHC activates only for mHC-aware models

**Phase 3: Explicit Enable** (Full adoption)
- Set `enabled: true`
- Configure layer-specific settings
- Enable stability tracking

### 1.3 Zero-Downtime Migration

```bash
# Step 1: Update server without enabling mHC
git pull origin main
./scripts/build.sh

# Step 2: Restart with auto-detect only
export SHIMMY_MHC_ENABLED=false
export SHIMMY_MHC_AUTO_DETECT=true
./scripts/start_server.sh

# Step 3: Enable mHC at runtime (no restart needed)
curl -X POST http://localhost:8080/admin/config/mhc/enable

# Step 4: Verify operation
curl http://localhost:8080/admin/metrics/mhc
```

---

## 2. Configuration Migration

### 2.1 Before: Non-mHC Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "inference": {
    "model_path": "models/llama-3.3-70b.gguf",
    "max_tokens": 2048,
    "temperature": 0.7
  }
}
```

### 2.2 After: mHC-Ready Configuration

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080
  },
  "inference": {
    "model_path": "models/llama-3.3-70b.gguf",
    "max_tokens": 2048,
    "temperature": 0.7,
    "mhc": {
      "enabled": false,
      "auto_detect": true,
      "sinkhorn_iterations": 10,
      "manifold_epsilon": 1e-6,
      "stability_threshold": 1e-4,
      "log_metrics": false,
      "layers": {
        "apply_to_attention": false,
        "apply_to_ffn": false,
        "layer_range": null
      }
    }
  }
}
```

### 2.3 Environment Variable Migration

**Before:**
```bash
export SHIMMY_MODEL_PATH=models/llama-3.3-70b.gguf
export SHIMMY_MAX_TOKENS=2048
```

**After:**
```bash
# Existing variables unchanged
export SHIMMY_MODEL_PATH=models/llama-3.3-70b.gguf
export SHIMMY_MAX_TOKENS=2048

# New mHC variables (optional)
export SHIMMY_MHC_ENABLED=false
export SHIMMY_MHC_AUTO_DETECT=true
export SHIMMY_MHC_SINKHORN_ITERS=10
export SHIMMY_MHC_LOG_METRICS=false
```

### 2.4 Service-Level Configuration Migration

**Translation Service - Before:**
```json
{
  "services": {
    "translation": {
      "cache_enabled": true,
      "timeout_ms": 30000
    }
  }
}
```

**Translation Service - After:**
```json
{
  "services": {
    "translation": {
      "cache_enabled": true,
      "timeout_ms": 30000,
      "mhc_stability_tracking": true,
      "stability_threshold": 0.8,
      "log_unstable": true,
      "cache_stable_only": false
    }
  }
}
```

---

## 3. API Changes

### 3.1 New Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/admin/config/mhc` | GET | Get mHC configuration |
| `/admin/config/mhc` | POST | Update mHC configuration |
| `/admin/config/mhc/enable` | POST | Enable mHC globally |
| `/admin/config/mhc/disable` | POST | Disable mHC globally |
| `/admin/metrics/mhc` | GET | Get mHC metrics |
| `/admin/diagnostics/mhc` | GET | Full mHC diagnostics |

### 3.2 Request Parameter Extensions

**Completions API - Extended:**

```bash
# Before (still works)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "prompt": "Hello",
    "max_tokens": 100
  }'

# After (new mhc_config parameter)
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "prompt": "Hello",
    "max_tokens": 100,
    "mhc_config": {
      "enabled": true,
      "sinkhorn_iterations": 12
    }
  }'
```

### 3.3 Response Extensions

**Before:**
```json
{
  "id": "cmpl-123",
  "choices": [
    {"text": "Hello world", "index": 0}
  ]
}
```

**After (with mHC metrics):**
```json
{
  "id": "cmpl-123",
  "choices": [
    {"text": "Hello world", "index": 0}
  ],
  "mhc_metrics": {
    "enabled": true,
    "avg_stability": 0.94,
    "layers_processed": 80,
    "unstable_layers": []
  }
}
```

---

## 4. Code Migration Examples

### 4.1 Zig: Matrix Operations Migration

**Before (standard matmul):**
```zig
const matrix_ops = @import("inference/engine/core/matrix_ops.zig");

pub fn forward(output: []f32, weights: Weight, input: []const f32) !void {
    try matrix_ops.matmul(output, weights, input, m, n, k, allocator, pool);
}
```

**After (mHC-enabled matmul):**
```zig
const matrix_ops = @import("inference/engine/core/matrix_ops.zig");

pub fn forward(output: []f32, weights: Weight, input: []const f32, layer_id: u32) !?mhc.StabilityMetrics {
    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .layer_id = layer_id,
        .mhc_config = .{
            .enabled = true,
            .sinkhorn_iterations = 10,
        },
    };

    return try matrix_ops.matmul_with_mhc(
        output, weights, input,
        m, n, k,
        config,
        allocator,
        pool,
    );
}
```

### 4.2 Zig: Transformer Layer Migration

**Before:**
```zig
const transformer = @import("inference/engine/core/transformer.zig");

pub fn processLayer(output: []f32, input: []const f32, layer: u32) !void {
    const config = transformer.TransformerConfig{
        .hidden_dim = 8192,
        .n_heads = 64,
    };
    try transformer.forward_layer(output, input, layer, weights, config, allocator, pool);
}
```

**After:**
```zig
const transformer = @import("inference/engine/core/transformer.zig");

pub fn processLayer(output: []f32, input: []const f32, layer: u32) !void {
    const config = transformer.TransformerConfig{
        .hidden_dim = 8192,
        .n_heads = 64,
        .mhc_config = .{
            .enabled = true,
            .attention_enabled = true,
            .ffn_enabled = true,
            .layer_range = null,  // Apply to all layers
            .core = .{
                .sinkhorn_iterations = 10,
                .manifold_epsilon = 1e-6,
            },
        },
    };
    try transformer.forward_layer(output, input, layer, weights, config, allocator, pool);
}
```

### 4.3 Mojo: Translation Service Migration

**Before:**
```mojo
from services.translation import MojoTranslationService

fn translate_text(text: String, target: String) -> String:
    var service = MojoTranslationService()
    return service.translate(text, "en", target)
```

**After:**
```mojo
from services.translation import MojoTranslationService

fn translate_text(text: String, target: String) -> Tuple[String, Float32]:
    var service = MojoTranslationService()
    service.mhc_enabled = True
    service.stability_threshold = 0.8

    var result = service.translate_with_stability(text, "en", target)
    return (result.text, result.stability_score)
```

### 4.4 Mojo: Embedding Service Migration

**Before:**
```mojo
from services.embedding import MojoEmbeddingService

fn get_embedding(text: String) -> DynamicVector[Float32]:
    var service = MojoEmbeddingService()
    return service.embed(text)
```

**After:**
```mojo
from services.embedding import MojoEmbeddingService

fn get_embedding(text: String) -> Tuple[DynamicVector[Float32], Float32]:
    var service = MojoEmbeddingService()
    service.mhc_enabled = True
    service.consistency_check = True

    var result = service.embed_with_stability(text)
    return (result.embedding, result.stability_score)
```

### 4.5 Python Client Migration

**Before:**
```python
import requests

def get_completion(prompt):
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={"model": "llama-3.3-70b", "prompt": prompt, "max_tokens": 100}
    )
    return response.json()["choices"][0]["text"]
```

**After:**
```python
import requests

def get_completion(prompt, mhc_enabled=True):
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={
            "model": "llama-3.3-70b",
            "prompt": prompt,
            "max_tokens": 100,
            "mhc_config": {
                "enabled": mhc_enabled,
                "sinkhorn_iterations": 10
            }
        }
    )
    result = response.json()
    text = result["choices"][0]["text"]
    stability = result.get("mhc_metrics", {}).get("avg_stability", None)
    return text, stability
```

### 4.6 GGUF Loader Migration

**Before:**
```zig
const gguf = @import("inference/engine/core/gguf_loader.zig");

pub fn loadModel(path: []const u8) !Model {
    return try gguf.loadModel(path, allocator);
}
```

**After:**
```zig
const gguf = @import("inference/engine/core/gguf_loader.zig");

pub fn loadModel(path: []const u8) !Model {
    const model = try gguf.loadModel(path, allocator);

    // Auto-configure mHC from model metadata
    if (model.metadata.hasMHC()) {
        const mhc_config = model.metadata.getMHCConfig().?;
        std.debug.print("Model has mHC: iterations={}\n", .{mhc_config.sinkhorn_iterations});
    }

    return model;
}
```

---

## 5. Migration Checklist

### Pre-Migration

- [ ] Backup current configuration files
- [ ] Document current performance baselines
- [ ] Update to latest nOpenaiServer version
- [ ] Review existing model compatibility
- [ ] Set up monitoring for mHC metrics

### Configuration Migration

- [ ] Add `mhc` section to config.json
- [ ] Set initial values: `enabled: false`, `auto_detect: true`
- [ ] Configure service-level mHC settings
- [ ] Set up environment variables (if using)
- [ ] Validate configuration with validation script

### Code Migration

- [ ] Update matrix operations to use `matmul_with_mhc`
- [ ] Update transformer layers to include `mhc_config`
- [ ] Update service handlers to track stability
- [ ] Add mHC metrics to API responses
- [ ] Update client code to handle new response format

### Testing

- [ ] Run unit tests with mHC disabled
- [ ] Run unit tests with mHC enabled
- [ ] Compare output quality before/after
- [ ] Benchmark performance impact
- [ ] Test stability metrics accuracy

### Deployment

- [ ] Deploy with mHC disabled initially
- [ ] Enable auto_detect in staging
- [ ] Monitor metrics for 24 hours
- [ ] Gradually enable mHC in production
- [ ] Set up alerts for instability

---

## 6. Rollback Procedures

### Immediate Rollback (Runtime)

```bash
# Disable mHC instantly (no restart)
curl -X POST http://localhost:8080/admin/config/mhc/disable

# Verify disabled
curl http://localhost:8080/admin/config/mhc | jq '.enabled'
# Should return: false
```

### Environment Variable Rollback

```bash
export SHIMMY_MHC_ENABLED=false
# Server will use this on next request
```

### Configuration File Rollback

```json
{
  "inference": {
    "mhc": {
      "enabled": false,
      "auto_detect": false
    }
  }
}
```

### Full Code Rollback

```bash
# Revert to previous version
git checkout previous-version
./scripts/build.sh
./scripts/restart_server.sh
```

---

## Migration Support

### Common Migration Issues

| Issue | Solution |
|-------|----------|
| Performance regression | Reduce `sinkhorn_iterations` to 8 |
| API response changed | Check for `mhc_metrics` field |
| Stability warnings | Increase iterations or adjust threshold |
| Memory increase | Disable `log_metrics` |

### Getting Help

- Review [MHC_TROUBLESHOOTING_GUIDE.md](MHC_TROUBLESHOOTING_GUIDE.md)
- Check [MHC_CONFIGURATION_GUIDE.md](MHC_CONFIGURATION_GUIDE.md)
- Consult [MHC_INTEGRATION_TECHNICAL_SPEC.md](MHC_INTEGRATION_TECHNICAL_SPEC.md)

---

**End of Migration Guide**

