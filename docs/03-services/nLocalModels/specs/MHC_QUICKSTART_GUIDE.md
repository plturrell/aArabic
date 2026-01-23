# mHC Quickstart Guide

**Document Version**: 1.0  
**Date**: January 19, 2026  
**Reference**: MHC_INTEGRATION_TECHNICAL_SPEC.md  
**Status**: Active

---

## Getting Started in 5 Minutes

This guide will have you running mHC-enabled inference in just 5 minutes. mHC (Manifold-Constrained Hyper-Connections) provides mathematically-guaranteed stability for deep neural network inference.

### Prerequisites

- nOpenaiServer installed and running
- A GGUF model file (with or without mHC metadata)
- Basic familiarity with HTTP APIs

---

## Step 1: Verify Installation (30 seconds)

Check that the server is running with mHC support:

```bash
# Check server health
curl http://localhost:8080/health

# Expected response:
# {"status": "healthy", "mhc_available": true, "version": "1.0.0"}
```

---

## Step 2: Check mHC Configuration (30 seconds)

View the current mHC configuration:

```bash
curl http://localhost:8080/admin/config/mhc

# Example response:
# {
#   "enabled": false,
#   "auto_detect": true,
#   "sinkhorn_iterations": 10,
#   "manifold_epsilon": 1e-6,
#   "stability_threshold": 1e-4
# }
```

---

## Step 3: Enable mHC (1 minute)

### Option A: Enable via API (runtime)

```bash
# Enable mHC globally
curl -X POST http://localhost:8080/admin/config/mhc/enable

# Response:
# {"success": true, "message": "mHC enabled globally"}
```

### Option B: Enable via Configuration File

Edit `config.json`:

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "auto_detect": true,
      "sinkhorn_iterations": 10
    }
  }
}
```

### Option C: Enable via Environment Variable

```bash
export SHIMMY_MHC_ENABLED=true
./scripts/start_server.sh
```

---

## Step 4: Run Your First mHC Inference (2 minutes)

### Example 1: Basic Completion

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "prompt": "Translate to Arabic: Hello, how are you today?",
    "max_tokens": 100
  }'
```

### Example 2: Chat Completion with mHC

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Explain quantum computing in simple terms."}
    ],
    "max_tokens": 500
  }'
```

### Example 3: Per-Request mHC Configuration

```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-3.3-70b",
    "prompt": "Write a haiku about programming",
    "max_tokens": 50,
    "mhc_config": {
      "enabled": true,
      "sinkhorn_iterations": 15
    }
  }'
```

---

## Step 5: Verify mHC is Working (1 minute)

Check mHC metrics:

```bash
curl http://localhost:8080/admin/metrics/mhc

# Expected response:
# {
#   "global_stats": {
#     "total_inferences": 3,
#     "mhc_enabled_count": 3,
#     "avg_stability": 0.94,
#     "unstable_count": 0
#   }
# }
```

**Key indicators of successful mHC operation:**
- `mhc_enabled_count` > 0
- `avg_stability` between 0.9 and 1.0
- `unstable_count` should be low (< 1% of total)

---

## 🎉 Congratulations!

You've successfully run mHC-enabled inference! Here's what happened:

1. **Sinkhorn-Knopp normalization** was applied to layer activations
2. **Signal stability** was maintained through deep layers (80+ layers for Llama 3.3 70B)
3. **Manifold constraints** prevented signal explosion/vanishing

---

## Basic Configuration Reference

### Minimal Configuration

The simplest mHC configuration:

```json
{
  "inference": {
    "mhc": {
      "enabled": true
    }
  }
}
```

### Recommended Development Configuration

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "auto_detect": true,
      "sinkhorn_iterations": 10,
      "log_metrics": true
    }
  }
}
```

### Recommended Production Configuration

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "auto_detect": true,
      "sinkhorn_iterations": 10,
      "log_metrics": false,
      "layers": {
        "apply_to_ffn": true,
        "apply_to_attention": false
      }
    }
  }
}
```

---

## Code Examples

### Example 4: Zig - Direct mHC Constraint Application

```zig
const mhc = @import("inference/engine/core/mhc_constraints.zig");

pub fn applyMHCToActivations(activations: []f32, allocator: std.mem.Allocator) !void {
    const config = mhc.MHCConfig{
        .enabled = true,
        .sinkhorn_iterations = 10,
        .manifold_epsilon = 1e-6,
        .stability_threshold = 1e-4,
    };

    // Apply Sinkhorn-Knopp normalization
    const iterations = try mhc.sinkhorn_normalize(
        activations,
        64,   // rows
        64,   // cols
        config,
        allocator,
    );

    std.debug.print("Converged in {} iterations\n", .{iterations});
}
```

### Example 5: Zig - Stability Check

```zig
const mhc = @import("inference/engine/core/mhc_constraints.zig");

pub fn checkStability(activations: []const f32) bool {
    const config = mhc.MHCConfig{
        .stability_threshold = 1e-4,
    };

    // Compute stability metrics
    const metrics = mhc.compute_stability_metrics(activations, config);

    // Check if stable
    return metrics.is_stable;
}
```

### Example 6: Zig - Matrix Multiplication with mHC

```zig
const matrix_ops = @import("inference/engine/core/matrix_ops.zig");

pub fn mhcMatMul(
    output: []f32,
    weights: matrix_ops.Weight,
    input: []const f32,
    m: usize, n: usize, k: usize,
    allocator: std.mem.Allocator,
) !?mhc.StabilityMetrics {
    const config = matrix_ops.MatMulConfig{
        .use_mhc = true,
        .layer_id = 25,
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
        null,
    );
}
```

### Example 7: Mojo - Translation Service with mHC

```mojo
from services.translation import MojoTranslationService

fn main() raises:
    var service = MojoTranslationService()
    service.mhc_enabled = True
    service.stability_threshold = 0.85

    # Translate with stability tracking
    var result = service.translate(
        "Hello, how are you?",
        source_lang="en",
        target_lang="ar"
    )

    print("Translation:", result.text)
    print("Stability:", result.stability_score)
    print("Quality:", result.quality_score)
```

### Example 8: Python - Client with mHC Config

```python
import requests

def inference_with_mhc(prompt: str, mhc_iterations: int = 10):
    response = requests.post(
        "http://localhost:8080/v1/completions",
        json={
            "model": "llama-3.3-70b",
            "prompt": prompt,
            "max_tokens": 100,
            "mhc_config": {
                "enabled": True,
                "sinkhorn_iterations": mhc_iterations
            }
        }
    )
    return response.json()

# Example usage
result = inference_with_mhc("Explain mHC in one sentence")
print(result["choices"][0]["text"])
```

---

## Next Steps

1. **Configuration Deep Dive**: See [MHC_CONFIGURATION_GUIDE.md](MHC_CONFIGURATION_GUIDE.md)
2. **Troubleshooting**: See [MHC_TROUBLESHOOTING_GUIDE.md](MHC_TROUBLESHOOTING_GUIDE.md)
3. **Migration from non-mHC**: See [MHC_MIGRATION_GUIDE.md](MHC_MIGRATION_GUIDE.md)
4. **Technical Details**: See [MHC_INTEGRATION_TECHNICAL_SPEC.md](MHC_INTEGRATION_TECHNICAL_SPEC.md)

---

## Common Quick Fixes

| Issue | Quick Fix |
|-------|-----------|
| mHC not enabled | `curl -X POST http://localhost:8080/admin/config/mhc/enable` |
| Performance slow | Reduce `sinkhorn_iterations` to 8 |
| Instability warnings | Increase `sinkhorn_iterations` to 15-20 |
| Memory high | Set `log_metrics: false` |

---

**End of Quickstart Guide**

