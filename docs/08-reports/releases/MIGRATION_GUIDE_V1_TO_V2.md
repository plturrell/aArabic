# Migration Guide: mHC v1.5 to v2.0

**Version:** 2.0.0  
**Date:** January 19, 2026  
**Previous Version:** 1.5.0

---

## Overview

This guide covers the migration from mHC v1.5 (Geometric Stability) to mHC v2.0 (Manifold-Constrained Hyper-Connections). v2.0 is a major release with significant new capabilities while maintaining backward compatibility with v1.5 APIs.

---

## What's New in v2.0

| Feature | v1.5 | v2.0 |
|---------|------|------|
| Euclidean constraints | ✅ | ✅ |
| Hyperbolic geometry | ❌ | ✅ |
| Spherical manifold | ❌ | ✅ |
| Product manifolds | ❌ | ✅ |
| Geometry auto-detection | ❌ | ✅ |
| Uncertainty quantification | ❌ | ✅ |
| Bayesian estimation | ❌ | ✅ |
| Speculative mHC | ❌ | ✅ |
| Production monitoring | Basic | Advanced |
| Test count | 91 | 800+ |

---

## Migration Steps

### Step 1: Backup Current Configuration

```bash
# Backup existing configuration
cp config.json config.json.v1.5.backup
cp -r docs/ docs.v1.5.backup/
```

### Step 2: Update Configuration File

#### v1.5 Configuration (existing)
```json
{
  "mhc": {
    "core": {
      "enabled": true,
      "sinkhorn_iterations": 10,
      "manifold_epsilon": 1e-6,
      "stability_threshold": 0.01,
      "manifold_beta": 10.0
    }
  }
}
```

#### v2.0 Configuration (new sections)
```json
{
  "mhc": {
    "core": {
      "enabled": true,
      "sinkhorn_iterations": 10,
      "manifold_epsilon": 1e-6,
      "stability_threshold": 0.01,
      "manifold_beta": 10.0
    },
    "geometry": {
      "auto_detect": true,
      "default_manifold": "euclidean",
      "hyperbolic": {
        "enabled": true,
        "curvature": -1.0,
        "max_norm": 0.99
      },
      "spherical": {
        "enabled": true,
        "radius": 1.0
      },
      "product": {
        "enabled": true,
        "components": [
          {"type": "hyperbolic", "dims": 128},
          {"type": "spherical", "dims": 64},
          {"type": "euclidean", "dims": 64}
        ]
      }
    },
    "uncertainty": {
      "enabled": true,
      "bootstrap_samples": 100,
      "confidence_level": 0.95,
      "detection_threshold": 0.6
    },
    "monitoring": {
      "enabled": true,
      "metrics_buffer_size": 1000,
      "alert_thresholds": {
        "curvature_warning": 0.5,
        "curvature_critical": 0.8,
        "acceptance_rate_warning": 0.6,
        "acceptance_rate_critical": 0.4
      },
      "integrations": {
        "prometheus": true,
        "pagerduty": false,
        "slack": false
      }
    },
    "speculative": {
      "enabled": false,
      "curvature_weight": 0.3,
      "distance_weight": 0.4,
      "stability_weight": 0.3,
      "acceptance_threshold": 0.5
    }
  }
}
```

### Step 3: Update Environment Variables

New environment variables in v2.0:

```bash
# Geometry settings
export MHC_AUTO_DETECT_GEOMETRY=true
export MHC_DEFAULT_MANIFOLD=euclidean
export MHC_HYPERBOLIC_ENABLED=true
export MHC_HYPERBOLIC_CURVATURE=-1.0
export MHC_SPHERICAL_ENABLED=true

# Uncertainty settings
export MHC_UNCERTAINTY_ENABLED=true
export MHC_BOOTSTRAP_SAMPLES=100
export MHC_CONFIDENCE_LEVEL=0.95

# Monitoring settings
export MHC_MONITORING_ENABLED=true
export MHC_PROMETHEUS_ENABLED=true

# Speculative settings
export MHC_SPECULATIVE_ENABLED=false
export MHC_ACCEPTANCE_THRESHOLD=0.5
```

### Step 4: Code Migration

#### API Changes

**v1.5 API (still supported)**
```zig
const mhc = @import("mhc_constraints.zig");

// Basic mHC operations
const iterations = try mhc.sinkhorn_normalize(matrix, rows, cols, config);
try mhc.apply_manifold_constraints(hidden_states, config);
const stable = mhc.check_stability(activations, threshold);
```

**v2.0 API (new capabilities)**
```zig
const mhc_hyperbolic = @import("mhc_hyperbolic.zig");
const mhc_spherical = @import("mhc_spherical.zig");
const mhc_product = @import("mhc_product_manifold.zig");
const mhc_detector = @import("mhc_geometry_detector.zig");
const mhc_uncertainty = @import("mhc_uncertainty.zig");

// Hyperbolic operations
const distance = try mhc_hyperbolic.hyperbolic_distance(x, y, config, allocator);
try mhc_hyperbolic.exp_map(result, base, tangent, config, allocator);

// Spherical operations
const dist = mhc_spherical.spherical_distance(x, y, config);
try mhc_spherical.frechet_mean(result, points, config, allocator);

// Product manifolds
const prod_dist = mhc_product.product_distance(x, y, product_config);

// Geometry detection with uncertainty
var detector = mhc_uncertainty.UncertaintyAwareGeometryDetector.init(allocator, config);
const result = try detector.detectWithUncertainty(points, num_points, dim);
```

### Step 5: New Endpoints

v2.0 adds new API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/mhc/geometry/detect` | POST | Auto-detect geometry type |
| `/v1/mhc/geometry/hyperbolic` | POST | Hyperbolic operations |
| `/v1/mhc/geometry/spherical` | POST | Spherical operations |
| `/v1/mhc/geometry/product` | POST | Product manifold operations |
| `/v1/mhc/uncertainty` | POST | Uncertainty quantification |
| `/v1/mhc/metrics/prometheus` | GET | Prometheus metrics |
| `/v1/mhc/alerts` | GET | Current alert status |

### Step 6: Run Test Suite

```bash
# Run all mHC tests (800+)
cd src/serviceCore/nLocalModels/inference/engine/core
zig build test

# Run specific test suites
zig test mhc_test_suite.zig                    # Core tests (91)
zig test mhc_geometric_test_suite.zig          # Geometric tests (148)
zig test mhc_uncertainty.zig                   # Uncertainty tests (27)
zig test mhc_bayesian.zig                      # Bayesian tests (24)
zig test mhc_failure_detection.zig             # Failure tests (28)
zig test mhc_monitor.zig                       # Monitor tests (32)
zig test mhc_speculative.zig                   # Speculative tests (37)

# Run production tests
zig test mhc_production_tests.zig              # Production suite (82)
```

### Step 7: Verify Installation

```bash
# Check v2.0 status
curl http://localhost:8080/v1/mhc/status

# Expected response:
{
  "enabled": true,
  "version": "2.0.0",
  "stable": true,
  "geometry": {
    "auto_detect": true,
    "available": ["euclidean", "hyperbolic", "spherical", "product"]
  },
  "monitoring": {
    "enabled": true,
    "prometheus": true
  }
}
```

---

## Breaking Changes

### None - Full Backward Compatibility

v2.0 maintains full backward compatibility with v1.5:

- All v1.5 APIs continue to work unchanged
- Existing configurations are valid
- Default behavior matches v1.5 (Euclidean mode)
- New features are opt-in

---

## Deprecation Notices

The following will be deprecated in v3.0:

| Deprecated | Replacement | v3.0 Status |
|------------|-------------|-------------|
| `manifold_beta` (scalar) | `geometry.euclidean.beta` | Removed |
| `MHC_ENABLED` env var | `MHC_CORE_ENABLED` | Removed |
| `/v1/mhc/config` PUT | `/v1/mhc/config/core` PUT | Removed |

---

## Rollback Procedure

If issues occur, rollback to v1.5:

```bash
# 1. Restore configuration
cp config.json.v1.5.backup config.json

# 2. Restart with v1.5 binary
./scripts/start_server_v1.5.sh

# 3. Verify rollback
curl http://localhost:8080/v1/mhc/status
# Should show "version": "1.5.0"
```

---

## Performance Considerations

### Memory Impact

| Component | Additional Memory |
|-----------|-------------------|
| Hyperbolic module | +50MB |
| Spherical module | +30MB |
| Product manifold | +20MB |
| Uncertainty | +10MB |
| Monitoring buffers | +5MB |
| **Total** | **~115MB** |

### Latency Impact

| Mode | Latency Overhead |
|------|------------------|
| Euclidean (default) | 0% (unchanged) |
| Auto-detect enabled | +5-10ms per request |
| Hyperbolic mode | +180% vs Euclidean |
| Product mode | +220% vs Euclidean |

---

## Support

- **Documentation**: `src/serviceCore/nLocalModels/docs/`
- **Release Notes**: `docs/RELEASE_NOTES_V2.md`
- **Operator Runbook**: `docs/operations/OPERATOR_RUNBOOK.md`

---

**Migration Complete!** 🚀

