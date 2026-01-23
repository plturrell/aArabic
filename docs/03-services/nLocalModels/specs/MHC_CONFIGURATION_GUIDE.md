# mHC Configuration Guide

**Document Version**: 1.0  
**Date**: January 19, 2026  
**Reference**: MHC_INTEGRATION_TECHNICAL_SPEC.md  
**Status**: Active Configuration Reference

---

## Executive Summary

This guide provides comprehensive documentation for configuring mHC (Manifold-Constrained Hyper-Connections) in nOpenaiServer. It covers all configuration methods, from JSON files to environment variables to runtime API calls, with examples for common use cases.

### Configuration Philosophy

**Layered Configuration**: Multiple configuration sources with clear precedence  
**Safe Defaults**: Conservative settings that work out-of-the-box  
**Runtime Flexibility**: Change settings without restarting  
**Validation**: Automatic validation with helpful error messages

---

## Table of Contents

1. [Configuration Hierarchy](#1-configuration-hierarchy)
2. [JSON Configuration](#2-json-configuration)
3. [Environment Variables](#3-environment-variables)
4. [Runtime API](#4-runtime-api)
5. [Common Scenarios](#5-common-scenarios)
6. [Troubleshooting](#6-troubleshooting)
7. [Best Practices](#7-best-practices)
8. [Migration Guide](#8-migration-guide)

---

## 1. Configuration Hierarchy

### 1.1 Precedence Order

Configuration sources are applied in this order (highest to lowest priority):

```
1. Runtime API Calls (highest priority)
   ↓
2. Environment Variables
   ↓
3. JSON Configuration File
   ↓
4. Default Values (lowest priority)
```

**Example**: If `mhc.enabled=true` in JSON but `SHIMMY_MHC_ENABLED=false` in environment, then **mHC is disabled** (env var wins).

### 1.2 Configuration Loading Process

```
Server Start:
    ↓
Load Default Config (hardcoded)
    ↓
Load JSON Config (if exists)
    ↓
Override with Environment Variables
    ↓
Apply Runtime Updates (via API)
    ↓
Validate Final Configuration
    ↓
Server Ready
```

---

## 2. JSON Configuration

### 2.1 Complete Configuration File

**Location**: `src/serviceCore/nLocalModels/config.json`

```json
{
  "server": {
    "host": "0.0.0.0",
    "port": 8080,
    "num_workers": 4
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
  },
  
  "services": {
    "translation": {
      "mhc_stability_tracking": true,
      "stability_threshold": 0.8,
      "log_unstable": true
    },
    
    "embedding": {
      "mhc_enabled": false,
      "consistency_check": true
    },
    
    "rag": {
      "mhc_multi_doc": false,
      "stability_threshold": 0.75
    }
  },
  
  "orchestration": {
    "kto_policy": {
      "mhc_stability_weight": 0.1,
      "track_stability": true,
      "min_stability": 0.7
    },
    
    "recursive_llm": {
      "mhc_depth_threshold": 5,
      "max_stable_depth": 15,
      "track_per_depth": true
    },
    
    "tau2_bench": {
      "include_mhc_metrics": true,
      "stability_weight": 0.2
    }
  }
}
```

### 2.2 Core mHC Parameters

#### enabled
- **Type**: Boolean
- **Default**: `false`
- **Description**: Global mHC enable/disable switch
- **Example**: `"enabled": true`
- **Impact**: Controls all mHC features

#### auto_detect
- **Type**: Boolean
- **Default**: `true`
- **Description**: Automatically enable mHC if model metadata indicates support
- **Example**: `"auto_detect": true`
- **Impact**: Overrides `enabled` when mHC model detected

#### sinkhorn_iterations
- **Type**: Integer
- **Range**: 5-50
- **Default**: `10`
- **Description**: Number of Sinkhorn-Knopp normalization iterations
- **Tuning**: 
  - Lower (5-8): Faster but less accurate
  - Medium (10-15): Balanced (recommended)
  - Higher (20-50): More accurate but slower

#### manifold_epsilon
- **Type**: Float
- **Range**: 1e-8 to 1e-3
- **Default**: `1e-6`
- **Description**: Convergence threshold for Sinkhorn-Knopp
- **Tuning**:
  - Smaller (1e-8): Stricter convergence
  - Larger (1e-4): Faster convergence

#### stability_threshold
- **Type**: Float
- **Range**: 1e-6 to 1.0
- **Default**: `1e-4`
- **Description**: Threshold for flagging unstable activations
- **Usage**: Activations with norm > threshold trigger warnings

#### log_metrics
- **Type**: Boolean
- **Default**: `false`
- **Description**: Enable detailed stability metrics logging
- **Impact**: Performance cost ~2% when enabled

### 2.3 Layer-Specific Configuration

#### apply_to_attention
- **Type**: Boolean
- **Default**: `false`
- **Description**: Apply mHC constraints to attention layer outputs
- **Cost**: ~3% overhead per layer
- **Benefit**: 15-20% stability improvement

#### apply_to_ffn
- **Type**: Boolean
- **Default**: `false`
- **Description**: Apply mHC constraints to FFN layer outputs
- **Cost**: ~2% overhead per layer
- **Benefit**: 10-15% stability improvement

#### layer_range
- **Type**: Object or null
- **Default**: `null` (apply to all layers)
- **Format**: `{"start": 10, "end": 70}`
- **Description**: Apply mHC only to specific layer range
- **Use Case**: Gradual rollout, testing specific layers

**Example**:
```json
{
  "layers": {
    "apply_to_attention": true,
    "apply_to_ffn": true,
    "layer_range": {
      "start": 20,
      "end": 60
    }
  }
}
```

### 2.4 Service-Level Configuration

#### Translation Service

```json
{
  "services": {
    "translation": {
      "mhc_stability_tracking": true,
      "stability_threshold": 0.8,
      "log_unstable": true,
      "cache_stable_only": true
    }
  }
}
```

Parameters:
- `mhc_stability_tracking`: Enable stability monitoring
- `stability_threshold`: Minimum stability score (0-1)
- `log_unstable`: Log warnings for unstable translations
- `cache_stable_only`: Only cache translations with stability > threshold

#### KTO Policy

```json
{
  "orchestration": {
    "kto_policy": {
      "mhc_stability_weight": 0.1,
      "track_stability": true,
      "min_stability": 0.7,
      "stabilize_exploration": true
    }
  }
}
```

Parameters:
- `mhc_stability_weight`: Weight for stability in action selection (0-1)
- `track_stability`: Collect per-action stability metrics
- `min_stability`: Minimum acceptable stability score
- `stabilize_exploration`: Apply constraints during exploration

#### Recursive LLM

```json
{
  "orchestration": {
    "recursive_llm": {
      "mhc_depth_threshold": 5,
      "max_stable_depth": 15,
      "track_per_depth": true,
      "strict_mode_depth": 10
    }
  }
}
```

Parameters:
- `mhc_depth_threshold`: Depth at which to enable mHC constraints
- `max_stable_depth`: Maximum recursion depth with stability tracking
- `track_per_depth`: Collect metrics per recursion level
- `strict_mode_depth`: Depth at which to apply stricter constraints

---

## 3. Environment Variables

### 3.1 Core Variables

```bash
# Global mHC control
export SHIMMY_MHC_ENABLED=true
export SHIMMY_MHC_AUTO_DETECT=true

# Algorithm parameters
export SHIMMY_MHC_SINKHORN_ITERS=10
export SHIMMY_MHC_EPSILON=1e-6
export SHIMMY_MHC_STABILITY_THRESHOLD=1e-4

# Logging
export SHIMMY_MHC_LOG_METRICS=false
export SHIMMY_MHC_LOG_LEVEL=info  # debug, info, warn, error
```

### 3.2 Layer Configuration

```bash
# Layer-specific settings
export SHIMMY_MHC_APPLY_ATTENTION=false
export SHIMMY_MHC_APPLY_FFN=false

# Layer range
export SHIMMY_MHC_LAYER_START=0
export SHIMMY_MHC_LAYER_END=80  # Apply to all 80 layers
```

### 3.3 Service Variables

```bash
# Translation service
export SHIMMY_TRANSLATION_MHC_TRACKING=true
export SHIMMY_TRANSLATION_STABILITY_THRESHOLD=0.8

# Embedding service
export SHIMMY_EMBEDDING_MHC_ENABLED=false

# RAG service
export SHIMMY_RAG_MHC_MULTI_DOC=false

# KTO policy
export SHIMMY_KTO_MHC_WEIGHT=0.1

# Recursive LLM
export SHIMMY_RECURSIVE_MHC_THRESHOLD=5
```

### 3.4 Environment Variable Precedence

```bash
# Example: JSON says enabled=false, env says true
# Result: mHC is ENABLED (env var wins)

# config.json:
{
  "inference": {
    "mhc": {
      "enabled": false
    }
  }
}

# Environment:
export SHIMMY_MHC_ENABLED=true

# Effective configuration: mHC is ENABLED
```

---

## 4. Runtime API

### 4.1 Configuration Endpoints

#### Get Current Configuration

```bash
GET /admin/config/mhc

Response:
{
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
```

#### Update Configuration

```bash
POST /admin/config/mhc
Content-Type: application/json

{
  "enabled": true,
  "sinkhorn_iterations": 15,
  "log_metrics": true
}

Response:
{
  "success": true,
  "message": "Configuration updated",
  "requires_reload": false
}
```

#### Enable mHC Globally

```bash
POST /admin/config/mhc/enable

Response:
{
  "success": true,
  "message": "mHC enabled globally"
}
```

#### Disable mHC Globally

```bash
POST /admin/config/mhc/disable

Response:
{
  "success": true,
  "message": "mHC disabled globally"
}
```

### 4.2 Per-Request Configuration

Some endpoints support per-request mHC configuration:

```bash
POST /v1/completions
Content-Type: application/json

{
  "model": "llama-3.3-70b",
  "prompt": "Translate to Arabic: Hello world",
  "max_tokens": 100,
  "mhc_config": {
    "enabled": true,
    "sinkhorn_iterations": 12
  }
}
```

### 4.3 Metrics Endpoint

```bash
GET /admin/metrics/mhc

Response:
{
  "global_stats": {
    "total_inferences": 1542,
    "mhc_enabled_count": 847,
    "avg_stability": 0.92,
    "unstable_count": 23
  },
  "layer_stats": {
    "layers_with_mhc": [20, 21, 22, ..., 59, 60],
    "avg_amplification": 1.02,
    "unstable_layers": []
  },
  "service_stats": {
    "translation": {
      "total": 423,
      "avg_stability": 0.89,
      "unstable": 12
    },
    "kto_policy": {
      "total_actions": 1247,
      "avg_stability": 0.94
    }
  }
}
```

---

## 5. Common Scenarios

### 5.1 Development Environment

**Goal**: Enable mHC with detailed logging for debugging

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "log_metrics": true,
      "layers": {
        "apply_to_attention": true,
        "apply_to_ffn": true
      }
    }
  }
}
```

```bash
export SHIMMY_MHC_LOG_LEVEL=debug
export SHIMMY_MHC_LOG_METRICS=true
```

**Result**: Full mHC with detailed logs, suitable for development/debugging

---

### 5.2 Production Environment

**Goal**: Enable mHC with minimal logging, optimized for performance

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "auto_detect": true,
      "log_metrics": false,
      "layers": {
        "apply_to_attention": false,
        "apply_to_ffn": true
      }
    }
  }
}
```

```bash
export SHIMMY_MHC_ENABLED=true
export SHIMMY_MHC_LOG_METRICS=false
export SHIMMY_MHC_LOG_LEVEL=warn
```

**Result**: mHC enabled for FFN layers only, minimal logging

---

### 5.3 Testing mHC Models

**Goal**: Test new mHC model with auto-detection

```json
{
  "inference": {
    "model_path": "models/deepseek-mhc-v1.gguf",
    "mhc": {
      "enabled": false,
      "auto_detect": true
    }
  }
}
```

**Result**: Model metadata triggers mHC auto-enable, no manual config needed

---

### 5.4 Gradual Rollout

**Goal**: Enable mHC for specific layer range only

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "layers": {
        "layer_range": {
          "start": 30,
          "end": 50
        }
      }
    }
  }
}
```

**Result**: mHC applied to layers 30-50 only, others use standard ResNet

---

### 5.5 Arabic Translation Optimization

**Goal**: Optimize for Arabic translation stability

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 12
    }
  },
  "services": {
    "translation": {
      "mhc_stability_tracking": true,
      "stability_threshold": 0.85,
      "cache_stable_only": true
    }
  }
}
```

**Result**: Stable Arabic translations with aggressive caching

---

### 5.6 Deep Recursion

**Goal**: Enable mHC for deep recursive reasoning

```json
{
  "inference": {
    "mhc": {
      "enabled": true
    }
  },
  "orchestration": {
    "recursive_llm": {
      "mhc_depth_threshold": 3,
      "max_stable_depth": 20,
      "track_per_depth": true
    }
  }
}
```

**Result**: mHC constraints kick in at depth 3, stable to depth 20

---

### 5.7 Disabled (Standard Mode)

**Goal**: Run without mHC (baseline performance)

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

```bash
export SHIMMY_MHC_ENABLED=false
export SHIMMY_MHC_AUTO_DETECT=false
```

**Result**: Pure ResNet architecture, no mHC overhead

---

## 6. Troubleshooting

### 6.1 Common Issues

#### Issue: mHC Not Activating

**Symptoms**: Model loads but mHC remains disabled

**Diagnosis**:
```bash
# Check configuration
curl http://localhost:8080/admin/config/mhc

# Check model metadata
curl http://localhost:8080/admin/model/info
```

**Solutions**:
1. Verify `enabled: true` in config
2. Check `auto_detect: true` if using mHC model
3. Restart server after config changes
4. Check environment variables aren't overriding

---

#### Issue: Performance Regression

**Symptoms**: Inference slower with mHC enabled

**Diagnosis**:
```bash
# Get metrics
curl http://localhost:8080/admin/metrics/mhc

# Check Sinkhorn iterations
grep "sinkhorn_iterations" config.json
```

**Solutions**:
1. Reduce `sinkhorn_iterations` (try 8 or 10)
2. Disable `log_metrics` in production
3. Apply mHC selectively (FFN only, not attention)
4. Use layer range to limit mHC application

---

#### Issue: Instability Warnings

**Symptoms**: Frequent "unstable activation" warnings

**Diagnosis**:
```bash
# Check stability metrics
curl http://localhost:8080/admin/metrics/mhc | jq '.layer_stats.unstable_layers'
```

**Solutions**:
1. Increase `sinkhorn_iterations` (try 15-20)
2. Lower `manifold_epsilon` (try 1e-7)
3. Increase `stability_threshold` (less strict validation)
4. Check model quantization quality

---

#### Issue: Memory Usage Increase

**Symptoms**: Higher memory consumption with mHC

**Diagnosis**:
```bash
# Check memory usage
curl http://localhost:8080/admin/memory
```

**Solutions**:
1. Expected overhead: ~5MB per layer
2. For 80-layer model: ~400MB additional
3. Disable `log_metrics` to save memory
4. Use layer range to limit mHC scope

---

### 6.2 Validation Errors

#### Error: Invalid sinkhorn_iterations

```
Error: sinkhorn_iterations must be between 5 and 50
```

**Fix**: Update config with valid value:
```json
{
  "sinkhorn_iterations": 10
}
```

---

#### Error: Invalid layer_range

```
Error: layer_range.start must be < layer_range.end
```

**Fix**: Ensure start < end:
```json
{
  "layer_range": {
    "start": 10,
    "end": 50
  }
}
```

---

### 6.3 Logging

#### Enable Debug Logging

```bash
export SHIMMY_MHC_LOG_LEVEL=debug
```

**Output**:
```
[DEBUG] mHC: Layer 25 attention output
  - Norm before: 12.3
  - Norm after: 12.1
  - Amplification: 0.98
  - Stable: true
```

#### View Stability Logs

```bash
# Real-time logs
tail -f logs/mhc_stability.log

# Search for unstable events
grep "unstable" logs/mhc_stability.log
```

---

## 7. Best Practices

### 7.1 Development Best Practices

1. **Start Conservative**: Begin with mHC disabled, enable gradually
2. **Test Incrementally**: Enable for one layer range first, then expand
3. **Monitor Metrics**: Keep `log_metrics=true` during development
4. **Validate Numerics**: Compare outputs with/without mHC
5. **Profile Performance**: Benchmark each configuration change

### 7.2 Production Best Practices

1. **Disable Verbose Logging**: Set `log_metrics=false`
2. **Use Auto-Detect**: Let model metadata control mHC
3. **Monitor Stability**: Track `unstable_count` metric
4. **Set Alerts**: Alert if stability < 0.7 for >1% of requests
5. **Have Rollback Plan**: Keep mHC=false config ready

### 7.3 Performance Optimization

#### Minimize Overhead

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 8,
      "log_metrics": false,
      "layers": {
        "apply_to_ffn": true,
        "apply_to_attention": false
      }
    }
  }
}
```

**Result**: ~2% overhead, 10-15% stability improvement

#### Maximum Stability

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 20,
      "manifold_epsilon": 1e-7,
      "layers": {
        "apply_to_attention": true,
        "apply_to_ffn": true
      }
    }
  }
}
```

**Result**: ~5-8% overhead, 25-35% stability improvement

### 7.4 Configuration Management

#### Use Version Control

```bash
# Store configs in git
git add config.json config.production.json config.development.json
git commit -m "Add mHC configurations"
```

#### Environment-Specific Configs

```
configs/
├── config.development.json  # mHC enabled, logging on
├── config.staging.json      # mHC enabled, logging off
├── config.production.json   # mHC auto-detect only
└── config.testing.json      # mHC forced on for tests
```

#### Configuration Validation Script

```bash
#!/bin/bash
# validate_mhc_config.sh

# Validate JSON syntax
jq . config.json > /dev/null || exit 1

# Check required fields
jq -e '.inference.mhc.enabled' config.json > /dev/null || exit 1

# Validate ranges
iters=$(jq '.inference.mhc.sinkhorn_iterations' config.json)
if [ $iters -lt 5 ] || [ $iters -gt 50 ]; then
    echo "Error: sinkhorn_iterations must be 5-50"
    exit 1
fi

echo "✅ Configuration valid"
```

---

## 8. Migration Guide

### 8.1 Migrating from Standard Configuration

#### Before (No mHC)

```json
{
  "inference": {
    "model_path": "models/llama-3.3-70b.gguf",
    "max_tokens": 2048
  }
}
```

#### After (With mHC)

```json
{
  "inference": {
    "model_path": "models/llama-3.3-70b.gguf",
    "max_tokens": 2048,
    "mhc": {
      "enabled": false,
      "auto_detect": true
    }
  }
}
```

**Impact**: No change in behavior (mHC disabled), but ready for mHC models

---

### 8.2 Enabling mHC for Existing Services

#### Translation Service Migration

**Before**:
```mojo
var service = MojoTranslationService()
var (translation, quality) = service.translate(text, "ar", "en")
```

**After**:
```mojo
var service = MojoTranslationService()
service.mhc_enabled = True  # Enable stability tracking
var (translation, quality, stability) = service.translate(text, "ar", "en")

# Check stability
if stability < 0.8:
    print("⚠️ Translation may be unstable")
```

**API Change**: Returns 3-tuple instead of 2-tuple (backward compatible via default)

---

### 8.3 Configuration Schema Evolution

#### Version 1.0 (Current)

```json
{
  "inference": {
    "mhc": {
      "enabled": false
    }
  }
}
```

#### Future Version 2.0 (Example)

```json
{
  "inference": {
    "mhc": {
      "version": "2.0",
      "enabled": false,
      "advanced": {
        "adaptive_iterations": true,
        "dynamic_epsilon": true
      }
    }
  }
}
```

**Migration Path**: v1.0 configs will work with v2.0 (backward compatible)

---

## 9. Configuration Examples

### 9.1 Conservative Production

```json
{
  "inference": {
    "mhc": {
      "enabled": false,
      "auto_detect": true,
      "sinkhorn_iterations": 10,
      "log_metrics": false
    }
  },
  "services": {
    "translation": {
      "mhc_stability_tracking": true
    }
  }
}
```

**Use Case**: Production with auto-detection, stability tracking only

---

### 9.2 Aggressive Optimization

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 8,
      "log_metrics": false,
      "layers": {
        "apply_to_ffn": true,
        "apply_to_attention": false
      }
    }
  }
}
```

**Use Case**: Maximize performance, accept slightly lower stability

---

### 9.3 Maximum Stability

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 20,
      "manifold_epsilon": 1e-7,
      "layers": {
        "apply_to_attention": true,
        "apply_to_ffn": true
      }
    }
  },
  "services": {
    "translation": {
      "stability_threshold": 0.9
    }
  },
  "orchestration": {
    "recursive_llm": {
      "mhc_depth_threshold": 3
    }
  }
}
```

**Use Case**: Critical applications requiring maximum stability

---

### 9.4 Testing Configuration

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "log_metrics": true,
      "stability_threshold": 1e-2,
      "layers": {
        "layer_range": {
          "start": 0,
          "end": 5
        }
      }
    }
  }
}
```

**Use Case**: Test mHC on first 5 layers only, detailed logging

---

### 9.5 Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  nopenaiserver:
    image: nopenaiserver:mhc-v1.0
    environment:
      - SHIMMY_MHC_ENABLED=true
      - SHIMMY_MHC_AUTO_DETECT=true
      - SHIMMY_MHC_SINKHORN_ITERS=10
      - SHIMMY_MHC_LOG_METRICS=false
      - SHIMMY_MHC_APPLY_FFN=true
    volumes:
      - ./config.json:/app/config.json
      - ./models:/models
    ports:
      - "8080:8080"
```

**Use Case**: Container deployment with env var configuration

---

### 9.6 Kubernetes Deployment

```yaml
# kubernetes/deployment.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: nopenaiserver-mhc-config
data:
  config.json: |
    {
      "inference": {
        "mhc": {
          "enabled": true,
          "auto_detect": true,
          "sinkhorn_iterations": 10
        }
      }
    }
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nopenaiserver
spec:
  template:
    spec:
      containers:
      - name: server
        image: nopenaiserver:mhc-v1.0
        env:
        - name: SHIMMY_MHC_ENABLED
          value: "true"
        - name: SHIMMY_MHC_LOG_LEVEL
          value: "info"
        volumeMounts:
        - name: config
          mountPath: /app/config.json
          subPath: config.json
      volumes:
      - name: config
        configMap:
          name: nopenaiserver-mhc-config
```

**Use Case**: K8s deployment with ConfigMap

---

## 10. Advanced Configuration

### 10.1 Dynamic Configuration Updates

**Use Case**: Change mHC settings without restart

```bash
# Enable mHC runtime
curl -X POST http://localhost:8080/admin/config/mhc/enable

# Update iterations
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"sinkhorn_iterations": 15}'

# Verify update
curl http://localhost:8080/admin/config/mhc
```

**Impact**: Takes effect immediately for new requests

---

### 10.2 Feature Flags

```json
{
  "feature_flags": {
    "mhc_enabled": true,
    "mhc_attention": false,
    "mhc_ffn": true,
    "mhc_recursive": true
  }
}
```

**Use Case**: Granular feature control via flags

---

### 10.3 A/B Testing Configuration

```json
{
  "experiments": {
    "mhc_ab_test": {
      "enabled": true,
      "variant_a": {
        "mhc_enabled": false
      },
      "variant_b": {
        "mhc_enabled": true,
        "sinkhorn_iterations": 10
      },
      "split_ratio": 0.5
    }
  }
}
```

**Use Case**: Compare mHC vs standard in production

---

### 10.4 Per-Model Configuration

```json
{
  "models": {
    "llama-3.3-70b": {
      "mhc": {
        "enabled": false,
        "auto_detect": true
      }
    },
    "deepseek-mhc-v1": {
      "mhc": {
        "enabled": true,
        "sinkhorn_iterations": 12
      }
    }
  }
}
```

**Use Case**: Different mHC settings per model

---

## 11. Configuration Reference

### 11.1 Complete Parameter List

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `enabled` | bool | false | - | Global enable/disable |
| `auto_detect` | bool | true | - | Auto-enable for mHC models |
| `sinkhorn_iterations` | int | 10 | 5-50 | Normalization iterations |
| `manifold_epsilon` | float | 1e-6 | 1e-8 to 1e-3 | Convergence threshold |
| `stability_threshold` | float | 1e-4 | 1e-6 to 1.0 | Instability threshold |
| `log_metrics` | bool | false | - | Enable metrics logging |
| `apply_to_attention` | bool | false | - | mHC in attention layers |
| `apply_to_ffn` | bool | false | - | mHC in FFN layers |
| `layer_range.start` | int | 0 | 0+ | First layer for mHC |
| `layer_range.end` | int | null | 0+ | Last layer for mHC |

### 11.2 Environment Variable Mapping

| JSON Path | Environment Variable | Example |
|-----------|---------------------|---------|
| `inference.mhc.enabled` | `SHIMMY_MHC_ENABLED` | `true` |
| `inference.mhc.auto_detect` | `SHIMMY_MHC_AUTO_DETECT` | `true` |
| `inference.mhc.sinkhorn_iterations` | `SHIMMY_MHC_SINKHORN_ITERS` | `10` |
| `inference.mhc.manifold_epsilon` | `SHIMMY_MHC_EPSILON` | `1e-6` |
| `inference.mhc.log_metrics` | `SHIMMY_MHC_LOG_METRICS` | `false` |
| `services.translation.mhc_stability_tracking` | `SHIMMY_TRANSLATION_MHC_TRACKING` | `true` |
| `orchestration.kto_policy.mhc_stability_weight` | `SHIMMY_KTO_MHC_WEIGHT` | `0.1` |
| `orchestration.recursive_llm.mhc_depth_threshold` | `SHIMMY_RECURSIVE_MHC_THRESHOLD` | `5` |

---

## 12. Configuration Validation

### 12.1 Validation Script

```bash
#!/bin/bash
# scripts/validate_mhc_config.sh

CONFIG_FILE=${1:-config.json}

echo "Validating mHC configuration: $CONFIG_FILE"

# Check JSON syntax
if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
    echo "❌ Invalid JSON syntax"
    exit 1
fi

# Validate sinkhorn_iterations
ITERS=$(jq -r '.inference.mhc.sinkhorn_iterations // 10' "$CONFIG_FILE")
if [ "$ITERS" -lt 5 ] || [ "$ITERS" -gt 50 ]; then
    echo "❌ sinkhorn_iterations must be 5-50 (got: $ITERS)"
    exit 1
fi

# Validate manifold_epsilon
EPS=$(jq -r '.inference.mhc.manifold_epsilon // 1e-6' "$CONFIG_FILE")
# Note: Bash doesn't handle scientific notation well, use Python for this
python3 -c "eps = $EPS; assert 1e-8 <= eps <= 1e-3, 'Invalid epsilon'" || {
    echo "❌ manifold_epsilon must be 1e-8 to 1e-3"
    exit 1
}

# Validate layer_range if present
if jq -e '.inference.mhc.layers.layer_range' "$CONFIG_FILE" > /dev/null; then
    START=$(jq -r '.inference.mhc.layers.layer_range.start' "$CONFIG_FILE")
    END=$(jq -r '.inference.mhc.layers.layer_range.end' "$CONFIG_FILE")
    
    if [ "$START" -ge "$END" ]; then
        echo "❌ layer_range.start must be < layer_range.end"
        exit 1
    fi
fi

echo "✅ Configuration valid"
```

### 12.2 Server-Side Validation

The server validates configuration on startup:

```
Starting nOpenaiServer...
Loading configuration from config.json...
  ✅ JSON syntax valid
  ✅ mHC parameters valid
  ✅ Service configurations valid
  ✅ Orchestration settings valid
Configuration loaded successfully.
```

If validation fails:
```
Starting nOpenaiServer...
Loading configuration from config.json...
  ❌ Error: sinkhorn_iterations=3 (must be 5-50)
Configuration invalid. Server not started.
```

---

## 13. Monitoring & Metrics

### 13.1 Prometheus Metrics

```
# mHC-specific metrics
mhc_enabled{service="inference"} 1
mhc_stability_score{layer="25"} 0.94
mhc_unstable_count{service="translation"} 5
mhc_sinkhorn_iterations{layer="30"} 10
mhc_convergence_time_ms{layer="30"} 0.045
```

### 13.2 Metrics Dashboard

```grafana
# Grafana dashboard queries

# Average stability score
avg(mhc_stability_score)

# Unstable activation rate
rate(mhc_unstable_count[5m])

# Performance overhead
(inference_time_with_mhc - inference_time_baseline) / inference_time_baseline * 100
```

---

## 14. FAQ

### Q1: Should I enable mHC in production?

**A**: Only if:
- You have mHC-trained models
- You've tested thoroughly in staging
- Performance overhead is acceptable (<5%)
- You have monitoring in place

**Recommendation**: Start with `auto_detect=true`, `enabled=false`

---

### Q2: What's the performance cost?

**A**: Depends on configuration:
- FFN only: ~2-3% overhead
- Attention only: ~3-4% overhead
- Both: ~5% overhead
- With logging: +1-2% additional

---

### Q3: Can I enable mHC for specific requests?

**A**: Yes, via per-request config:
```json
{
  "prompt": "...",
  "mhc_config": {
    "enabled": true
  }
}
```

---

### Q4: How do I know if mHC is working?

**A**: Check metrics:
```bash
curl http://localhost:8080/admin/metrics/mhc
```

Look for:
- `avg_stability` near 0.9-1.0
- `unstable_count` low (<1% of requests)
- `mhc_enabled_count` > 0

---

### Q5: What if mHC causes issues?

**A**: Immediate rollback:
```bash
# Option 1: Environment variable
export SHIMMY_MHC_ENABLED=false

# Option 2: Runtime API
curl -X POST http://localhost:8080/admin/config/mhc/disable

# Option 3: Update config and restart
# Edit config.json: "enabled": false
./scripts/restart_server.sh
```

---

## 15. Summary

### Quick Start

**Minimal Configuration** (auto-detect only):
```json
{
  "inference": {
    "mhc": {
      "auto_detect": true
    }
  }
}
```

**Recommended Production**:
```json
{
  "inference": {
    "mhc": {
      "enabled": false,
      "auto_detect": true,
      "sinkhorn_iterations": 10,
      "log_metrics": false
    }
  },
  "services": {
    "translation": {
      "mhc_stability_tracking": true
    }
  }
}
```

**Maximum Stability**:
```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 20,
      "layers": {
        "apply_to_attention": true,
        "apply_to_ffn": true
      }
    }
  }
}
```

### Configuration Checklist

- [ ] JSON syntax valid
- [ ] Parameters in valid ranges
- [ ] Environment variables set (if needed)
- [ ] Validation script passed
- [ ] Monitoring configured
- [ ] Rollback plan ready
- [ ] Team trained on configuration

---

**End of Configuration Guide**

For implementation details, see MHC_INTEGRATION_TECHNICAL_SPEC.md  
For day-by-day tasks, see MHC_IMPLEMENTATION_ROADMAP.md
