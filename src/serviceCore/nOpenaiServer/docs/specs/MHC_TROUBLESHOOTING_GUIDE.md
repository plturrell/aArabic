# mHC Troubleshooting Guide

**Document Version**: 1.0  
**Date**: January 19, 2026  
**Reference**: MHC_INTEGRATION_TECHNICAL_SPEC.md  
**Status**: Active

---

## Table of Contents

1. [Common Issues and Solutions](#1-common-issues-and-solutions)
2. [Error Messages Explained](#2-error-messages-explained)
3. [Performance Tuning Tips](#3-performance-tuning-tips)
4. [Diagnostic Commands](#4-diagnostic-commands)
5. [Debug Mode](#5-debug-mode)

---

## 1. Common Issues and Solutions

### Issue 1: mHC Not Activating

**Symptoms:**
- Metrics show `mhc_enabled_count: 0`
- No stability metrics in responses
- Performance identical to non-mHC mode

**Diagnosis:**

```bash
# Check current configuration
curl http://localhost:8080/admin/config/mhc | jq

# Check model metadata for mHC support
curl http://localhost:8080/admin/model/info | jq '.mhc_enabled'
```

**Solutions:**

```bash
# Solution 1: Enable mHC explicitly
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"enabled": true}'

# Solution 2: Check environment variable isn't overriding
echo $SHIMMY_MHC_ENABLED
# If false, set to true:
export SHIMMY_MHC_ENABLED=true

# Solution 3: Verify auto_detect is working
curl http://localhost:8080/admin/config/mhc | jq '.auto_detect'
# Should be true for mHC models
```

**Example Fix in config.json:**

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "auto_detect": true
    }
  }
}
```

---

### Issue 2: Performance Degradation

**Symptoms:**
- Inference 10-20% slower than expected
- Higher latency for first request
- Memory usage increased significantly

**Diagnosis:**

```bash
# Check current iterations setting
curl http://localhost:8080/admin/config/mhc | jq '.sinkhorn_iterations'

# Check if logging is enabled
curl http://localhost:8080/admin/config/mhc | jq '.log_metrics'

# Get performance metrics
curl http://localhost:8080/admin/metrics | jq '.latency_ms'
```

**Solutions:**

```bash
# Solution 1: Reduce Sinkhorn iterations (default: 10, try: 8)
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"sinkhorn_iterations": 8}'

# Solution 2: Disable metrics logging
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"log_metrics": false}'

# Solution 3: Apply mHC to FFN only (not attention)
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{
    "layers": {
      "apply_to_ffn": true,
      "apply_to_attention": false
    }
  }'
```

**Performance-Optimized Configuration:**

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 8,
      "log_metrics": false,
      "early_stopping": true,
      "layers": {
        "apply_to_ffn": true,
        "apply_to_attention": false
      }
    }
  }
}
```

---

### Issue 3: Frequent Instability Warnings

**Symptoms:**
- Log messages: "unstable activation detected"
- `unstable_count` metric increasing
- Occasional output quality degradation

**Diagnosis:**

```bash
# Check stability metrics
curl http://localhost:8080/admin/metrics/mhc | jq '.global_stats'

# Check which layers are unstable
curl http://localhost:8080/admin/metrics/mhc | jq '.layer_stats.unstable_layers'

# Check current thresholds
curl http://localhost:8080/admin/config/mhc | jq '{stability_threshold, manifold_epsilon}'
```

**Solutions:**

```bash
# Solution 1: Increase Sinkhorn iterations for better convergence
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"sinkhorn_iterations": 15}'

# Solution 2: Decrease epsilon for stricter normalization
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"manifold_epsilon": 1e-7}'

# Solution 3: Relax stability threshold if warnings are too frequent
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"stability_threshold": 1e-3}'
```

**Stability-Optimized Configuration:**

```json
{
  "inference": {
    "mhc": {
      "enabled": true,
      "sinkhorn_iterations": 20,
      "manifold_epsilon": 1e-7,
      "stability_threshold": 1e-4,
      "manifold_beta": 10.0
    }
  }
}
```

---

### Issue 4: Memory Usage Too High

**Symptoms:**
- OOM (Out of Memory) errors
- Swap usage increasing
- Slow garbage collection

**Diagnosis:**

```bash
# Check memory usage
curl http://localhost:8080/admin/memory | jq

# Check if logging is causing memory growth
curl http://localhost:8080/admin/config/mhc | jq '.log_metrics'
```

**Solutions:**

```bash
# Solution 1: Disable metrics logging (saves ~5-10% memory)
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"log_metrics": false}'

# Solution 2: Limit mHC to specific layer range
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{
    "layers": {
      "layer_range": {"start": 20, "end": 60}
    }
  }'
```

---

### Issue 5: Model Not Loading with mHC

**Symptoms:**
- "Failed to load model" error
- mHC metadata parsing errors
- Model loads but mHC disabled

**Diagnosis:**

```bash
# Check model file exists and is readable
ls -la models/your-model.gguf

# Check GGUF metadata
zig build run-gguf-info -- models/your-model.gguf
```

**Solution - Verify Model Compatibility:**

```zig
// Example: Check if model has mHC metadata
const model = try gguf.loadModel("models/llama-3.3-70b.gguf", allocator);
defer model.deinit();

if (model.metadata.hasMHC()) {
    std.debug.print("mHC enabled in model\n", .{});
    const config = model.metadata.getMHCConfig().?;
    std.debug.print("  Sinkhorn iterations: {}\n", .{config.sinkhorn_iterations});
} else {
    std.debug.print("Model does not have mHC metadata\n", .{});
}
```

---

## 2. Error Messages Explained

### Error: "sinkhorn_iterations must be between 5 and 50"

**Cause:** Invalid configuration value
**Fix:**
```json
{"sinkhorn_iterations": 10}  // Valid: 5-50
```

### Error: "layer_range.start must be < layer_range.end"

**Cause:** Invalid layer range
**Fix:**
```json
{"layer_range": {"start": 10, "end": 50}}  // start < end
```

### Error: "manifold_epsilon out of range"

**Cause:** Epsilon too large or too small
**Fix:**
```json
{"manifold_epsilon": 1e-6}  // Valid: 1e-8 to 1e-3
```

### Error: "stability validation failed"

**Cause:** Activation exceeded stability bounds
**Fix:** Increase `sinkhorn_iterations` or `manifold_beta`

### Error: "Sinkhorn normalization did not converge"

**Cause:** Matrix too ill-conditioned
**Fix:**
```bash
curl -X POST http://localhost:8080/admin/config/mhc \
  -H "Content-Type: application/json" \
  -d '{"sinkhorn_iterations": 25, "manifold_epsilon": 1e-5}'
```

### Warning: "unstable activation at layer N"

**Cause:** Layer N produced unstable output
**Fix:** Enable mHC for that layer range or increase iterations

```json
{
  "layers": {
    "layer_range": {"start": 0, "end": 80},
    "apply_to_attention": true,
    "apply_to_ffn": true
  }
}
```

---

## 3. Performance Tuning Tips

### Tip 1: Balance Iterations vs. Accuracy

| Iterations | Latency Impact | Stability | Use Case |
|------------|---------------|-----------|----------|
| 5-8 | Minimal (~1-2%) | Moderate | High-throughput production |
| 10-15 | Low (~3-5%) | Good | Standard production |
| 20-30 | Moderate (~5-8%) | Excellent | Critical applications |
| 40-50 | High (~10%+) | Maximum | Research/debugging |

**Recommended Starting Point:**
```json
{"sinkhorn_iterations": 10}
```

### Tip 2: Selective Layer Application

Apply mHC only where it matters most:

```json
{
  "layers": {
    "apply_to_attention": false,
    "apply_to_ffn": true,
    "layer_range": {"start": 20, "end": 60}
  }
}
```

**Rationale:**
- FFN layers benefit most from stability constraints
- Middle layers (20-60 for 80-layer models) are most prone to instability
- Attention layers have lower overhead without mHC

### Tip 3: Early Stopping

Enable early stopping for faster convergence:

```json
{
  "early_stopping": true,
  "manifold_epsilon": 1e-6
}
```

**Effect:** Stops Sinkhorn iterations when convergence detected (typically 3-5 iterations earlier)

### Tip 4: Disable Logging in Production

```bash
export SHIMMY_MHC_LOG_METRICS=false
export SHIMMY_MHC_LOG_LEVEL=warn
```

**Memory savings:** ~100MB for 80-layer models
**Latency savings:** ~1-2%

### Tip 5: Use Thread Pool for Large Batches

```zig
// Enable parallel mHC for batched inference
const pool = try thread_pool.ThreadPool.init(.{
    .num_threads = 8,
});
defer pool.deinit();

const metrics = try matrix_ops.batched_matmul_with_mhc(
    outputs, weights, inputs,
    m, n, k,
    config,
    allocator,
    pool,
);
```

---

## 4. Diagnostic Commands

### Quick Health Check

```bash
# Full diagnostic
curl http://localhost:8080/admin/diagnostics/mhc

# Response includes:
# - Configuration status
# - Memory usage
# - Stability statistics
# - Recent errors
```

### Layer-by-Layer Analysis

```bash
curl http://localhost:8080/admin/metrics/mhc/layers | jq
```

### Export Debug Report

```bash
curl http://localhost:8080/admin/diagnostics/mhc/report > mhc_debug.json
```

### Real-Time Log Monitoring

```bash
tail -f logs/mhc_stability.log | grep -E "(unstable|error|warn)"
```

---

## 5. Debug Mode

### Enable Verbose Debug Logging

```bash
export SHIMMY_MHC_LOG_LEVEL=debug
export SHIMMY_MHC_LOG_METRICS=true
./scripts/start_server.sh
```

### Debug Output Example

```
[DEBUG] mHC: Layer 25 attention output
  - Input norm: 12.34
  - Output norm: 12.31
  - Amplification: 0.998
  - Sinkhorn iterations: 8 (converged early)
  - Stable: true
[DEBUG] mHC: Layer 25 FFN output
  - Input norm: 15.67
  - Output norm: 15.62
  - Amplification: 0.997
  - Sinkhorn iterations: 10
  - Stable: true
```

### Programmatic Debug in Zig

```zig
const mhc = @import("inference/engine/core/mhc_constraints.zig");

pub fn debugMHC(activations: []f32, layer_id: u32) void {
    const config = mhc.MHCConfig{
        .log_stability_metrics = true,
    };

    const metrics = mhc.compute_stability_metrics(activations, config);

    std.debug.print("Layer {}: norm={d:.4}, amp={d:.4}, stable={}\n", .{
        layer_id,
        metrics.norm,
        metrics.amplification,
        metrics.is_stable,
    });
}
```

---

## Quick Reference: Most Common Fixes

| Problem | Command |
|---------|---------|
| Enable mHC | `curl -X POST http://localhost:8080/admin/config/mhc/enable` |
| Disable mHC | `curl -X POST http://localhost:8080/admin/config/mhc/disable` |
| Reduce latency | `{"sinkhorn_iterations": 8, "log_metrics": false}` |
| Fix instability | `{"sinkhorn_iterations": 20, "manifold_epsilon": 1e-7}` |
| Save memory | `{"log_metrics": false}`, limit layer range |
| Debug issues | `export SHIMMY_MHC_LOG_LEVEL=debug` |

---

## Related Documents

- [MHC_QUICKSTART_GUIDE.md](MHC_QUICKSTART_GUIDE.md) - Getting started
- [MHC_CONFIGURATION_GUIDE.md](MHC_CONFIGURATION_GUIDE.md) - Full configuration reference
- [MHC_MIGRATION_GUIDE.md](MHC_MIGRATION_GUIDE.md) - Upgrading to mHC
- [MHC_INTEGRATION_TECHNICAL_SPEC.md](MHC_INTEGRATION_TECHNICAL_SPEC.md) - Technical details

---

**End of Troubleshooting Guide**

