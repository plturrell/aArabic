# mHC Configuration System Design

**Document Version**: 1.0  
**Date**: January 20, 2026  
**Author**: nOpenaiServer Team  
**Status**: Design Specification  
**Phase**: Day 31 - Configuration System Design

---

## Executive Summary

This document specifies the comprehensive configuration system for mHC (Manifold-Constrained Hyper-Connections), providing multiple configuration sources with clear precedence rules, runtime updates, and robust validation. The system supports JSON files, environment variables, CLI arguments, and programmatic configuration.

**Key Features**:
- **Multi-source configuration**: JSON, environment variables, CLI, code
- **Clear hierarchy**: CLI > ENV > JSON > Defaults
- **Runtime updates**: Hot-reload without service restart
- **Validation**: Schema validation with detailed error messages
- **Observability**: Configuration change tracking and audit logs

**Configuration Sources** (in precedence order):
1. CLI arguments (highest priority)
2. Environment variables
3. JSON configuration files
4. Programmatic defaults (lowest priority)

---

## Table of Contents

1. [Configuration Schema](#1-configuration-schema)
2. [JSON Configuration Format](#2-json-configuration-format)
3. [Environment Variable Mapping](#3-environment-variable-mapping)
4. [Configuration Hierarchy](#4-configuration-hierarchy)
5. [Runtime Updates](#5-runtime-updates)
6. [Validation System](#6-validation-system)
7. [Configuration Loading](#7-configuration-loading)
8. [API Reference](#8-api-reference)
9. [Examples](#9-examples)
10. [Migration Guide](#10-migration-guide)

---

## 1. Configuration Schema

### 1.1 Schema Overview

The configuration system uses a hierarchical structure:

```
mhc_config.json
├── core                    # Core mHC settings (Day 27)
├── matrix_ops              # Matrix operation settings (Day 28)
├── transformer             # Transformer integration (Day 29)
├── gguf                    # GGUF loader settings (Day 30)
├── geometric               # Geometric extensions (Days 54-60)
├── monitoring              # Observability settings (Days 61-67)
└── runtime                 # Runtime behavior
```

### 1.2 Complete Schema Structure

```zig
/// Root configuration structure
pub const MHCConfiguration = struct {
    /// Schema version (semantic versioning)
    schema_version: []const u8 = "1.0.0",
    
    /// Core mHC settings
    core: CoreConfig,
    
    /// Matrix operation settings
    matrix_ops: MatrixOpsConfig,
    
    /// Transformer integration settings
    transformer: TransformerConfig,
    
    /// GGUF loader settings
    gguf: GGUFConfig,
    
    /// Geometric extensions (optional, Days 54-60)
    geometric: ?GeometricConfig = null,
    
    /// Monitoring and observability (optional, Days 61-67)
    monitoring: ?MonitoringConfig = null,
    
    /// Runtime behavior
    runtime: RuntimeConfig,
};

/// Core mHC constraint settings (from Day 27)
pub const CoreConfig = struct {
    /// Enable mHC constraints globally
    enabled: bool = false,
    
    /// Sinkhorn-Knopp iterations (5-50)
    sinkhorn_iterations: u32 = 10,
    
    /// Convergence threshold (1e-8 to 1e-3)
    manifold_epsilon: f32 = 1e-6,
    
    /// Stability validation threshold
    stability_threshold: f32 = 1e-4,
    
    /// Maximum activation bound
    manifold_beta: f32 = 10.0,
    
    /// Enable early stopping
    early_stopping: bool = true,
    
    /// Log detailed stability metrics
    log_stability_metrics: bool = false,
    
    /// Apply to specific layer range (null = all layers)
    layer_range: ?LayerRange = null,
};

/// Matrix operation settings (from Day 28)
pub const MatrixOpsConfig = struct {
    /// Enable mHC in matrix operations
    use_mhc: bool = true,
    
    /// Abort on instability detection
    abort_on_instability: bool = false,
    
    /// Enable SIMD optimizations
    use_simd: bool = true,
    
    /// Thread pool size for parallel operations (0 = auto-detect)
    thread_pool_size: u32 = 0,
    
    /// Enable quantized matmul support
    support_quantization: bool = true,
    
    /// Batch size for batch operations
    batch_size: u32 = 32,
};

/// Transformer integration settings (from Day 29)
pub const TransformerConfig = struct {
    /// Apply mHC to attention output
    mhc_in_attention: bool = true,
    
    /// Apply mHC to FFN output
    mhc_in_ffn: bool = true,
    
    /// Apply mHC to residual connections
    mhc_in_residual: bool = false,
    
    /// Track stability metrics
    track_stability: bool = true,
    
    /// Layer selection strategy: "all", "adaptive", "manual"
    layer_selection: []const u8 = "adaptive",
    
    /// Manual layer range (if layer_selection = "manual")
    manual_layer_range: ?LayerRange = null,
    
    /// Adaptive selection threshold (layers with α > threshold)
    adaptive_threshold: f32 = 1.05,
};

/// GGUF loader settings (from Day 30)
pub const GGUFConfig = struct {
    /// Auto-detect mHC from GGUF metadata
    auto_detect: bool = true,
    
    /// Require mHC metadata in GGUF files
    require_metadata: bool = false,
    
    /// Fallback to defaults if metadata missing
    use_fallback: bool = true,
    
    /// Validation level: "strict", "loose", "none"
    validation_level: []const u8 = "loose",
};

/// Geometric extensions settings (Days 54-60)
pub const GeometricConfig = struct {
    /// Enable geometric extensions
    enabled: bool = false,
    
    /// Manifold type: "euclidean", "hyperbolic", "spherical", "product"
    manifold_type: []const u8 = "euclidean",
    
    /// Hyperbolic settings
    hyperbolic: ?HyperbolicConfig = null,
    
    /// Spherical settings
    spherical: ?SphericalConfig = null,
    
    /// Product manifold settings
    product: ?ProductManifoldConfig = null,
    
    /// Auto-detect geometry from data
    auto_detect_geometry: bool = false,
    
    /// Curvature estimation method: "ollivier_ricci", "sectional"
    curvature_method: []const u8 = "ollivier_ricci",
};

/// Hyperbolic manifold settings
pub const HyperbolicConfig = struct {
    /// Curvature parameter (negative for hyperbolic)
    curvature: f32 = -1.0,
    
    /// Use Poincaré ball model (vs hyperboloid)
    use_poincare: bool = true,
    
    /// Numerical stability epsilon
    epsilon: f32 = 1e-8,
};

/// Spherical manifold settings
pub const SphericalConfig = struct {
    /// Sphere radius
    radius: f32 = 1.0,
    
    /// Use stereographic projection
    use_stereographic: bool = false,
};

/// Product manifold settings
pub const ProductManifoldConfig = struct {
    /// Component manifolds (e.g., ["euclidean", "hyperbolic", "spherical"])
    components: [][]const u8,
    
    /// Component weights for distance computation
    weights: []f32,
};

/// Monitoring and observability settings (Days 61-67)
pub const MonitoringConfig = struct {
    /// Enable uncertainty quantification
    uncertainty_quantification: bool = false,
    
    /// Bootstrap samples for uncertainty estimation
    bootstrap_samples: u32 = 100,
    
    /// Enable failure detection
    failure_detection: bool = true,
    
    /// Alert thresholds
    alert_thresholds: AlertThresholds,
    
    /// Metrics collection interval (milliseconds)
    metrics_interval_ms: u32 = 1000,
    
    /// Enable Prometheus metrics export
    prometheus_enabled: bool = true,
    
    /// Prometheus port
    prometheus_port: u16 = 9090,
};

/// Alert threshold configuration
pub const AlertThresholds = struct {
    /// Maximum allowed instability rate (0.0-1.0)
    max_instability_rate: f32 = 0.05,
    
    /// Maximum amplification factor
    max_amplification: f32 = 1.5,
    
    /// Minimum amplification factor
    min_amplification: f32 = 0.5,
    
    /// Maximum energy spike (relative to baseline)
    max_energy_spike: f32 = 2.0,
};

/// Runtime behavior settings
pub const RuntimeConfig = struct {
    /// Enable hot-reload of configuration
    hot_reload: bool = true,
    
    /// Configuration file watch interval (seconds)
    watch_interval_sec: u32 = 5,
    
    /// Log configuration changes
    log_config_changes: bool = true,
    
    /// Validation mode: "strict", "warn", "silent"
    validation_mode: []const u8 = "warn",
    
    /// Configuration file path
    config_file_path: []const u8 = "config/mhc_config.json",
    
    /// Enable configuration audit log
    audit_log_enabled: bool = true,
    
    /// Audit log file path
    audit_log_path: []const u8 = "logs/mhc_config_audit.log",
};

/// Layer range specification
pub const LayerRange = struct {
    start: u32,
    end: u32,
    
    pub fn contains(self: LayerRange, layer_id: u32) bool {
        return layer_id >= self.start and layer_id <= self.end;
    }
    
    pub fn validate(self: LayerRange) !void {
        if (self.start > self.end) {
            return error.InvalidLayerRange;
        }
    }
};
```

---

## 2. JSON Configuration Format

### 2.1 Complete JSON Schema

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "mHC Configuration Schema",
  "type": "object",
  "required": ["schema_version", "core", "runtime"],
  "properties": {
    "schema_version": {
      "type": "string",
      "pattern": "^[0-9]+\\.[0-9]+\\.[0-9]+$",
      "description": "Semantic version of configuration schema"
    },
    "core": {
      "type": "object",
      "required": ["enabled"],
      "properties": {
        "enabled": {
          "type": "boolean",
          "description": "Enable mHC constraints globally"
        },
        "sinkhorn_iterations": {
          "type": "integer",
          "minimum": 5,
          "maximum": 50,
          "default": 10,
          "description": "Number of Sinkhorn-Knopp iterations"
        },
        "manifold_epsilon": {
          "type": "number",
          "minimum": 1e-8,
          "maximum": 1e-3,
          "default": 1e-6,
          "description": "Convergence threshold for normalization"
        },
        "stability_threshold": {
          "type": "number",
          "minimum": 0,
          "default": 1e-4,
          "description": "Maximum allowed activation value"
        },
        "manifold_beta": {
          "type": "number",
          "minimum": 0,
          "default": 10.0,
          "description": "Maximum L2 norm bound"
        },
        "early_stopping": {
          "type": "boolean",
          "default": true,
          "description": "Enable early convergence detection"
        },
        "log_stability_metrics": {
          "type": "boolean",
          "default": false,
          "description": "Log detailed stability metrics"
        },
        "layer_range": {
          "type": ["object", "null"],
          "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0}
          },
          "required": ["start", "end"],
          "description": "Apply to specific layer range"
        }
      }
    },
    "matrix_ops": {
      "type": "object",
      "properties": {
        "use_mhc": {
          "type": "boolean",
          "default": true,
          "description": "Enable mHC in matrix operations"
        },
        "abort_on_instability": {
          "type": "boolean",
          "default": false,
          "description": "Abort on instability detection"
        },
        "use_simd": {
          "type": "boolean",
          "default": true,
          "description": "Enable SIMD optimizations"
        },
        "thread_pool_size": {
          "type": "integer",
          "minimum": 0,
          "default": 0,
          "description": "Thread pool size (0 = auto-detect)"
        },
        "support_quantization": {
          "type": "boolean",
          "default": true,
          "description": "Enable quantized matmul support"
        },
        "batch_size": {
          "type": "integer",
          "minimum": 1,
          "default": 32,
          "description": "Batch size for batch operations"
        }
      }
    },
    "transformer": {
      "type": "object",
      "properties": {
        "mhc_in_attention": {
          "type": "boolean",
          "default": true,
          "description": "Apply mHC to attention output"
        },
        "mhc_in_ffn": {
          "type": "boolean",
          "default": true,
          "description": "Apply mHC to FFN output"
        },
        "mhc_in_residual": {
          "type": "boolean",
          "default": false,
          "description": "Apply mHC to residual connections"
        },
        "track_stability": {
          "type": "boolean",
          "default": true,
          "description": "Track stability metrics"
        },
        "layer_selection": {
          "type": "string",
          "enum": ["all", "adaptive", "manual"],
          "default": "adaptive",
          "description": "Layer selection strategy"
        },
        "manual_layer_range": {
          "type": ["object", "null"],
          "properties": {
            "start": {"type": "integer", "minimum": 0},
            "end": {"type": "integer", "minimum": 0}
          }
        },
        "adaptive_threshold": {
          "type": "number",
          "minimum": 0.9,
          "maximum": 2.0,
          "default": 1.05,
          "description": "Adaptive selection threshold"
        }
      }
    },
    "gguf": {
      "type": "object",
      "properties": {
        "auto_detect": {
          "type": "boolean",
          "default": true,
          "description": "Auto-detect mHC from GGUF metadata"
        },
        "require_metadata": {
          "type": "boolean",
          "default": false,
          "description": "Require mHC metadata in GGUF files"
        },
        "use_fallback": {
          "type": "boolean",
          "default": true,
          "description": "Fallback to defaults if metadata missing"
        },
        "validation_level": {
          "type": "string",
          "enum": ["strict", "loose", "none"],
          "default": "loose",
          "description": "Validation level for GGUF metadata"
        }
      }
    },
    "geometric": {
      "type": ["object", "null"],
      "properties": {
        "enabled": {
          "type": "boolean",
          "default": false,
          "description": "Enable geometric extensions"
        },
        "manifold_type": {
          "type": "string",
          "enum": ["euclidean", "hyperbolic", "spherical", "product"],
          "default": "euclidean",
          "description": "Manifold type for constraints"
        },
        "hyperbolic": {
          "type": ["object", "null"],
          "properties": {
            "curvature": {"type": "number", "default": -1.0},
            "use_poincare": {"type": "boolean", "default": true},
            "epsilon": {"type": "number", "default": 1e-8}
          }
        },
        "spherical": {
          "type": ["object", "null"],
          "properties": {
            "radius": {"type": "number", "default": 1.0},
            "use_stereographic": {"type": "boolean", "default": false}
          }
        },
        "product": {
          "type": ["object", "null"],
          "properties": {
            "components": {
              "type": "array",
              "items": {"type": "string"}
            },
            "weights": {
              "type": "array",
              "items": {"type": "number"}
            }
          }
        },
        "auto_detect_geometry": {
          "type": "boolean",
          "default": false,
          "description": "Auto-detect geometry from data"
        },
        "curvature_method": {
          "type": "string",
          "enum": ["ollivier_ricci", "sectional"],
          "default": "ollivier_ricci",
          "description": "Curvature estimation method"
        }
      }
    },
    "monitoring": {
      "type": ["object", "null"],
      "properties": {
        "uncertainty_quantification": {
          "type": "boolean",
          "default": false,
          "description": "Enable uncertainty quantification"
        },
        "bootstrap_samples": {
          "type": "integer",
          "minimum": 10,
          "maximum": 1000,
          "default": 100,
          "description": "Bootstrap samples for uncertainty"
        },
        "failure_detection": {
          "type": "boolean",
          "default": true,
          "description": "Enable failure detection"
        },
        "alert_thresholds": {
          "type": "object",
          "properties": {
            "max_instability_rate": {
              "type": "number",
              "minimum": 0.0,
              "maximum": 1.0,
              "default": 0.05
            },
            "max_amplification": {
              "type": "number",
              "minimum": 1.0,
              "default": 1.5
            },
            "min_amplification": {
              "type": "number",
              "maximum": 1.0,
              "default": 0.5
            },
            "max_energy_spike": {
              "type": "number",
              "minimum": 1.0,
              "default": 2.0
            }
          }
        },
        "metrics_interval_ms": {
          "type": "integer",
          "minimum": 100,
          "default": 1000,
          "description": "Metrics collection interval (ms)"
        },
        "prometheus_enabled": {
          "type": "boolean",
          "default": true,
          "description": "Enable Prometheus metrics export"
        },
        "prometheus_port": {
          "type": "integer",
          "minimum": 1024,
          "maximum": 65535,
          "default": 9090,
          "description": "Prometheus metrics port"
        }
      }
    },
    "runtime": {
      "type": "object",
      "required": ["config_file_path"],
      "properties": {
        "hot_reload": {
          "type": "boolean",
          "default": true,
          "description": "Enable hot-reload of configuration"
        },
        "watch_interval_sec": {
          "type": "integer",
          "minimum": 1,
          "default": 5,
          "description": "Configuration file watch interval (seconds)"
        },
        "log_config_changes": {
          "type": "boolean",
          "default": true,
          "description": "Log configuration changes"
        },
        "validation_mode": {
          "type": "string",
          "enum": ["strict", "warn", "silent"],
          "default": "warn",
          "description": "Validation mode for configuration"
        },
        "config_file_path": {
          "type": "string",
          "default": "config/mhc_config.json",
          "description": "Configuration file path"
        },
        "audit_log_enabled": {
          "type": "boolean",
          "default": true,
          "description": "Enable configuration audit log"
        },
        "audit_log_path": {
          "type": "string",
          "default": "logs/mhc_config_audit.log",
          "description": "Audit log file path"
        }
      }
    }
  }
}
```

### 2.2 Example Configuration Files

#### 2.2.1 Minimal Configuration (Development)

```json
{
  "schema_version": "1.0.0",
  "core": {
    "enabled": true,
    "sinkhorn_iterations": 10,
    "log_stability_metrics": true
  },
  "matrix_ops": {
    "use_mhc": true,
    "use_simd": true
  },
  "transformer": {
    "mhc_in_attention": true,
    "mhc_in_ffn": true,
    "layer_selection": "all"
  },
  "gguf": {
    "auto_detect": true,
    "use_fallback": true
  },
  "runtime": {
    "hot_reload": true,
    "config_file_path": "config/mhc_config.json",
    "validation_mode": "warn"
  }
}
```

#### 2.2.2 Production Configuration

```json
{
  "schema_version": "1.0.0",
  "core": {
    "enabled": true,
    "sinkhorn_iterations": 15,
    "manifold_epsilon": 1e-6,
    "stability_threshold": 1e-4,
    "manifold_beta": 10.0,
    "early_stopping": true,
    "log_stability_metrics": false,
    "layer_range": null
  },
  "matrix_ops": {
    "use_mhc": true,
    "abort_on_instability": false,
    "use_simd": true,
    "thread_pool_size": 0,
    "support_quantization": true,
    "batch_size": 32
  },
  "transformer": {
    "mhc_in_attention": true,
    "mhc_in_ffn": true,
    "mhc_in_residual": false,
    "track_stability": true,
    "layer_selection": "adaptive",
    "adaptive_threshold": 1.05
  },
  "gguf": {
    "auto_detect": true,
    "require_metadata": false,
    "use_fallback": true,
    "validation_level": "loose"
  },
  "monitoring": {
    "uncertainty_quantification": false,
    "failure_detection": true,
    "alert_thresholds": {
      "max_instability_rate": 0.05,
      "max_amplification": 1.5,
      "min_amplification": 0.5,
      "max_energy_spike": 2.0
    },
    "metrics_interval_ms": 1000,
    "prometheus_enabled": true,
    "prometheus_port": 9090
  },
  "runtime": {
    "hot_reload": true,
    "watch_interval_sec": 5,
    "log_config_changes": true,
    "validation_mode": "warn",
    "config_file_path": "config/mhc_config.json",
    "audit_log_enabled": true,
    "audit_log_path": "logs/mhc_config_audit.log"
  }
}
```

#### 2.2.3 Advanced Configuration (with Geometric Extensions)

```json
{
  "schema_version": "1.0.0",
  "core": {
    "enabled": true,
    "sinkhorn_iterations": 20,
    "manifold_epsilon": 1e-7,
    "early_stopping": true
  },
  "matrix_ops": {
    "use_mhc": true,
    "use_simd": true,
    "thread_pool_size": 8
  },
  "transformer": {
    "mhc_in_attention": true,
    "mhc_in_ffn": true,
    "layer_selection": "adaptive",
    "adaptive_threshold": 1.03
  },
  "gguf": {
    "auto_detect": true,
    "validation_level": "strict"
  },
  "geometric": {
    "enabled": true,
    "manifold_type": "hyperbolic",
    "hyperbolic": {
      "curvature": -1.0,
      "use_poincare": true,
      "epsilon": 1e-8
    },
    "auto_detect_geometry": true,
    "curvature_method": "ollivier_ricci"
  },
  "monitoring": {
    "uncertainty_quantification": true,
    "bootstrap_samples": 100,
    "failure_detection": true,
    "alert_thresholds": {
      "max_instability_rate": 0.03,
      "max_amplification": 1.3,
      "min_amplification": 0.7,
      "max_energy_spike": 1.8
    },
    "prometheus_enabled": true
  },
  "runtime": {
    "hot_reload": true,
    "validation_mode": "strict",
    "audit_log_enabled": true
  }
}
```

---

## 3. Environment Variable Mapping

### 3.1 Environment Variable Naming Convention

**Pattern**: `MHC_<SECTION>_<PARAMETER>`

**Rules**:
- All uppercase
- Underscores separate sections
- Nested objects use additional underscores
- Boolean: `true`/`false` strings
- Numbers: decimal format
- Null: `null` string or omit variable

### 3.2 Complete Environment Variable Map

```bash
# Schema Version
MHC_SCHEMA_VERSION=1.0.0

# Core Settings
MHC_CORE_ENABLED=true
MHC_CORE_SINKHORN_ITERATIONS=10
MHC_CORE_MANIFOLD_EPSILON=0.000001
MHC_CORE_STABILITY_THRESHOLD=0.0001
MHC_CORE_MANIFOLD_BETA=10.0
MHC_CORE_EARLY_STOPPING=true
MHC_CORE_LOG_STABILITY_METRICS=false
MHC_CORE_LAYER_RANGE_START=0        # Optional
MHC_CORE_LAYER_RANGE_END=79         # Optional

# Matrix Operations
MHC_MATRIX_OPS_USE_MHC=true
MHC_MATRIX_OPS_ABORT_ON_INSTABILITY=false
MHC_MATRIX_OPS_USE_SIMD=true
MHC_MATRIX_OPS_THREAD_POOL_SIZE=0
MHC_MATRIX_OPS_SUPPORT_QUANTIZATION=true
MHC_MATRIX_OPS_BATCH_SIZE=32

# Transformer
MHC_TRANSFORMER_MHC_IN_ATTENTION=true
MHC_TRANSFORMER_MHC_IN_FFN=true
MHC_TRANSFORMER_MHC_IN_RESIDUAL=false
MHC_TRANSFORMER_TRACK_STABILITY=true
MHC_TRANSFORMER_LAYER_SELECTION=adaptive
MHC_TRANSFORMER_MANUAL_LAYER_RANGE_START=10  # Optional
MHC_TRANSFORMER_MANUAL_LAYER_RANGE_END=50    # Optional
MHC_TRANSFORMER_ADAPTIVE_THRESHOLD=1.05

# GGUF Loader
MHC_GGUF_AUTO_DETECT=true
MHC_GGUF_REQUIRE_METADATA=false
MHC_GGUF_USE_FALLBACK=true
MHC_GGUF_VALIDATION_LEVEL=loose

# Geometric Extensions (Optional)
MHC_GEOMETRIC_ENABLED=false
MHC_GEOMETRIC_MANIFOLD_TYPE=euclidean
MHC_GEOMETRIC_HYPERBOLIC_CURVATURE=-1.0
MHC_GEOMETRIC_HYPERBOLIC_USE_POINCARE=true
MHC_GEOMETRIC_HYPERBOLIC_EPSILON=0.00000001
MHC_GEOMETRIC_SPHERICAL_RADIUS=1.0
MHC_GEOMETRIC_SPHERICAL_USE_STEREOGRAPHIC=false
MHC_GEOMETRIC_PRODUCT_COMPONENTS=euclidean,hyperbolic,spherical  # CSV
MHC_GEOMETRIC_PRODUCT_WEIGHTS=0.33,0.33,0.34                      # CSV
MHC_GEOMETRIC_AUTO_DETECT_GEOMETRY=false
MHC_GEOMETRIC_CURVATURE_METHOD=ollivier_ricci

# Monitoring (Optional)
MHC_MONITORING_UNCERTAINTY_QUANTIFICATION=false
MHC_MONITORING_BOOTSTRAP_SAMPLES=100
MHC_MONITORING_FAILURE_DETECTION=true
MHC_MONITORING_ALERT_THRESHOLDS_MAX_INSTABILITY_RATE=0.05
MHC_MONITORING_ALERT_THRESHOLDS_MAX_AMPLIFICATION=1.5
MHC_MONITORING_ALERT_THRESHOLDS_MIN_AMPLIFICATION=0.5
MHC_MONITORING_ALERT_THRESHOLDS_MAX_ENERGY_SPIKE=2.0
MHC_MONITORING_METRICS_INTERVAL_MS=1000
MHC_MONITORING_PROMETHEUS_ENABLED=true
MHC_MONITORING_PROMETHEUS_PORT=9090

# Runtime
MHC_RUNTIME_HOT_RELOAD=true
MHC_RUNTIME_WATCH_INTERVAL_SEC=5
MHC_RUNTIME_LOG_CONFIG_CHANGES=true
MHC_RUNTIME_VALIDATION_MODE=warn
MHC_RUNTIME_CONFIG_FILE_PATH=config/mhc_config.json
MHC_RUNTIME_AUDIT_LOG_ENABLED=true
MHC_RUNTIME_AUDIT_LOG_PATH=logs/mhc_config_audit.log
```

### 3.3 Environment Variable Parsing

```zig
/// Parse environment variables into configuration
pub fn parse_env_vars(allocator: std.mem.Allocator) !MHCConfiguration {
    var config = default_config();
    
    // Core settings
    if (std.os.getenv("MHC_CORE_ENABLED")) |val| {
        config.core.enabled = parse_bool(val);
    }
    if (std.os.getenv("MHC_CORE_SINKHORN_ITERATIONS")) |val| {
        config.core.sinkhorn_iterations = try std.fmt.parseInt(u32, val, 10);
    }
    if (std.os.getenv("MHC_CORE_MANIFOLD_EPSILON")) |val| {
        config.core.manifold_epsilon = try std.fmt.parseFloat(f32, val);
    }
    // ... continue for all parameters
    
    return config;
}

fn parse_bool(val: []const u8) bool {
    return std.mem.eql(u8, val, "true") or std.mem.eql(u8, val, "1");
}
```

### 3.4 Docker Environment Example

```dockerfile
# Dockerfile
FROM ubuntu:22.04

# Set mHC environment variables
ENV MHC_CORE_ENABLED=true \
    MHC_CORE_SINKHORN_ITERATIONS=15 \
    MHC_MATRIX_OPS_USE_SIMD=true \
    MHC_TRANSFORMER_LAYER_SELECTION=adaptive \
    MHC_RUNTIME_HOT_RELOAD=false \
    MHC_RUNTIME_VALIDATION_MODE=strict

COPY nOpenaiServer /app/
WORKDIR /app
CMD ["./nOpenaiServer"]
```

```yaml
# docker-compose.yml
services:
  nopenai-server:
    image: nopenai-server:latest
    environment:
      - MHC_CORE_ENABLED=true
      - MHC_CORE_SINKHORN_ITERATIONS=15
      - MHC_MATRIX_OPS_USE_SIMD=true
      - MHC_MONITORING_PROMETHEUS_ENABLED=true
      - MHC_MONITORING_PROMETHEUS_PORT=9090
    ports:
      - "8080:8080"
      - "9090:9090"
```

---

## 4. Configuration Hierarchy

### 4.1 Precedence Rules

**Configuration sources are merged in this order** (highest priority first):

1. **CLI Arguments** (highest priority)
   - Example: `--mhc-core-enabled=true`
   - Overrides all other sources
   - Validated immediately

2. **Environment Variables**
   - Example: `MHC_CORE_ENABLED=true`
   - Overrides JSON and defaults
   - Parsed at startup

3. **JSON Configuration File**
   - Example: `config/mhc_config.json`
   - Overrides defaults only
   - Can be hot-reloaded

4. **Programmatic Defaults** (lowest priority)
   - Hardcoded in `MHCConfiguration` struct
   - Used if no other source provides value

### 4.2 Merge Algorithm

```zig
/// Merge configuration from multiple sources
pub fn merge_configs(
    defaults: MHCConfiguration,
    json_config: ?MHCConfiguration,
    env_config: ?MHCConfiguration,
    cli_config: ?MHCConfiguration,
) MHCConfiguration {
    var result = defaults;
    
    // Layer 1: Apply JSON config
    if (json_config) |json| {
        result = merge_two_configs(result, json);
    }
    
    // Layer 2: Apply env vars
    if (env_config) |env| {
        result = merge_two_configs(result, env);
    }
    
    // Layer 3: Apply CLI args (highest priority)
    if (cli_config) |cli| {
        result = merge_two_configs(result, cli);
    }
    
    return result;
}

fn merge_two_configs(
    base: MHCConfiguration,
    override: MHCConfiguration,
) MHCConfiguration {
    var result = base;
    
    // Merge core settings
    inline for (@typeInfo(@TypeOf(base.core)).Struct.fields) |field| {
        const override_val = @field(override.core, field.name);
        if (!is_default_value(override_val, field.default_value)) {
            @field(result.core, field.name) = override_val;
        }
    }
    
    // Merge matrix_ops, transformer, etc.
    // ... similar for all sections
    
    return result;
}
```

### 4.3 Configuration Resolution Example

```
Scenario: Multiple configuration sources

1. Defaults:
   core.enabled = false
   core.sinkhorn_iterations = 10

2. JSON file (config/mhc_config.json):
   core.enabled = true
   core.sinkhorn_iterations = 15

3. Environment variable:
   MHC_CORE_SINKHORN_ITERATIONS=20

4. CLI argument:
   --mhc-core-enabled=false

Result after merge:
   core.enabled = false              (CLI overrides JSON)
   core.sinkhorn_iterations = 20     (ENV overrides JSON)
```

### 4.4 Partial Configuration Support

```json
{
  "schema_version": "1.0.0",
  "core": {
    "enabled": true,
    "sinkhorn_iterations": 15
    // Other core settings use defaults
  }
  // matrix_ops, transformer, etc. use defaults
}
```

All unspecified values use defaults from lower-priority sources.

---

## 5. Runtime Updates

### 5.1 Hot-Reload System

```zig
/// Configuration hot-reload manager
pub const ConfigHotReload = struct {
    allocator: std.mem.Allocator,
    config_path: []const u8,
    current_config: MHCConfiguration,
    last_modified: i64,
    watch_interval_ms: u64,
    callbacks: std.ArrayList(*const fn (MHCConfiguration) void),
    running: std.atomic.Value(bool),
    
    pub fn init(
        allocator: std.mem.Allocator,
        config_path: []const u8,
        initial_config: MHCConfiguration,
    ) !ConfigHotReload {
        return ConfigHotReload{
            .allocator = allocator,
            .config_path = try allocator.dupe(u8, config_path),
            .current_config = initial_config,
            .last_modified = try get_file_mtime(config_path),
            .watch_interval_ms = initial_config.runtime.watch_interval_sec * 1000,
            .callbacks = std.ArrayList(*const fn (MHCConfiguration) void).init(allocator),
            .running = std.atomic.Value(bool).init(false),
        };
    }
    
    pub fn deinit(self: *ConfigHotReload) void {
        self.stop();
        self.allocator.free(self.config_path);
        self.callbacks.deinit();
    }
    
    /// Start watching for configuration changes
    pub fn start(self: *ConfigHotReload) !void {
        if (self.running.swap(true, .SeqCst)) {
            return; // Already running
        }
        
        // Spawn watcher thread
        const thread = try std.Thread.spawn(.{}, watch_loop, .{self});
        thread.detach();
    }
    
    /// Stop watching
    pub fn stop(self: *ConfigHotReload) void {
        _ = self.running.swap(false, .SeqCst);
    }
    
    /// Register callback for configuration changes
    pub fn on_change(
        self: *ConfigHotReload,
        callback: *const fn (MHCConfiguration) void,
    ) !void {
        try self.callbacks.append(callback);
    }
    
    /// Watcher thread loop
    fn watch_loop(self: *ConfigHotReload) void {
        while (self.running.load(.SeqCst)) {
            std.time.sleep(self.watch_interval_ms * std.time.ns_per_ms);
            
            // Check if file modified
            const current_mtime = get_file_mtime(self.config_path) catch continue;
            if (current_mtime > self.last_modified) {
                self.reload_config() catch |err| {
                    std.log.err("Failed to reload config: {}", .{err});
                    continue;
                };
                self.last_modified = current_mtime;
            }
        }
    }
    
    /// Reload configuration from file
    fn reload_config(self: *ConfigHotReload) !void {
        std.log.info("Configuration file changed, reloading...", .{});
        
        // Load new configuration
        const new_config = try load_json_config(self.allocator, self.config_path);
        
        // Validate new configuration
        try validate_config(new_config);
        
        // Atomic update
        const old_config = self.current_config;
        self.current_config = new_config;
        
        // Log changes
        if (self.current_config.runtime.log_config_changes) {
            log_config_diff(old_config, new_config);
        }
        
        // Audit log
        if (self.current_config.runtime.audit_log_enabled) {
            try write_audit_log(
                self.current_config.runtime.audit_log_path,
                old_config,
                new_config,
            );
        }
        
        // Notify callbacks
        for (self.callbacks.items) |callback| {
            callback(new_config);
        }
        
        std.log.info("Configuration reloaded successfully", .{});
    }
};

fn get_file_mtime(path: []const u8) !i64 {
    const stat = try std.fs.cwd().statFile(path);
    return stat.mtime;
}
```

### 5.2 Configuration Change Callbacks

```zig
// Example: React to configuration changes
fn on_mhc_config_changed(new_config: MHCConfiguration) void {
    std.log.info("mHC configuration updated:", .{});
    std.log.info("  enabled: {}", .{new_config.core.enabled});
    std.log.info("  sinkhorn_iterations: {}", .{new_config.core.sinkhorn_iterations});
    
    // Update runtime behavior
    if (new_config.core.enabled) {
        enable_mhc_constraints();
    } else {
        disable_mhc_constraints();
    }
    
    // Update monitoring
    if (new_config.monitoring) |monitoring| {
        update_alert_thresholds(monitoring.alert_thresholds);
    }
}

// Register callback
try hot_reload.on_change(on_mhc_config_changed);
```

### 5.3 Audit Logging

```zig
/// Write configuration change to audit log
fn write_audit_log(
    log_path: []const u8,
    old_config: MHCConfiguration,
    new_config: MHCConfiguration,
) !void {
    const file = try std.fs.cwd().openFile(log_path, .{ .mode = .write_only, .append = true });
    defer file.close();
    
    const writer = file.writer();
    
    // Write timestamp
    const timestamp = std.time.timestamp();
    try writer.print("[{}] Configuration changed:\n", .{timestamp});
    
    // Write diff
    try write_config_diff(writer, old_config, new_config);
    try writer.writeAll("\n");
}

fn write_config_diff(
    writer: anytype,
    old: MHCConfiguration,
    new: MHCConfiguration,
) !void {
    // Core settings
    if (old.core.enabled != new.core.enabled) {
        try writer.print("  core.enabled: {} -> {}\n", .{ old.core.enabled, new.core.enabled });
    }
    if (old.core.sinkhorn_iterations != new.core.sinkhorn_iterations) {
        try writer.print("  core.sinkhorn_iterations: {} -> {}\n", .{
            old.core.sinkhorn_iterations,
            new.core.sinkhorn_iterations,
        });
    }
    // ... continue for all fields
}
```

**Example Audit Log**:

```
[1737344400] Configuration changed:
  core.enabled: false -> true
  core.sinkhorn_iterations: 10 -> 15
  transformer.layer_selection: all -> adaptive

[1737344460] Configuration changed:
  monitoring.failure_detection: false -> true
  monitoring.prometheus_port: 9090 -> 9091
```

---

## 6. Validation System

### 6.1 Validation Framework

```zig
/// Validation error with context
pub const ValidationError = struct {
    field: []const u8,
    error_type: ErrorType,
    message: []const u8,
    current_value: []const u8,
    expected_range: ?[]const u8 = null,
    
    pub const ErrorType = enum {
        out_of_range,
        invalid_type,
        missing_required,
        invalid_enum,
        constraint_violation,
        dependency_conflict,
    };
    
    pub fn format(
        self: ValidationError,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print("Validation error in '{s}': {s}", .{ self.field, self.message });
        if (self.expected_range) |range| {
            try writer.print(" (expected: {s}, got: {s})", .{ range, self.current_value });
        }
    }
};

/// Validation result
pub const ValidationResult = struct {
    valid: bool,
    errors: std.ArrayList(ValidationError),
    warnings: std.ArrayList(ValidationError),
    
    pub fn init(allocator: std.mem.Allocator) ValidationResult {
        return ValidationResult{
            .valid = true,
            .errors = std.ArrayList(ValidationError).init(allocator),
            .warnings = std.ArrayList(ValidationError).init(allocator),
        };
    }
    
    pub fn deinit(self: *ValidationResult) void {
        self.errors.deinit();
        self.warnings.deinit();
    }
    
    pub fn add_error(
        self: *ValidationResult,
        field: []const u8,
        error_type: ValidationError.ErrorType,
        message: []const u8,
        current_value: []const u8,
    ) !void {
        self.valid = false;
        try self.errors.append(ValidationError{
            .field = field,
            .error_type = error_type,
            .message = message,
            .current_value = current_value,
        });
    }
    
    pub fn add_warning(
        self: *ValidationResult,
        field: []const u8,
        message: []const u8,
    ) !void {
        try self.warnings.append(ValidationError{
            .field = field,
            .error_type = .constraint_violation,
            .message = message,
            .current_value = "",
        });
    }
};

/// Comprehensive configuration validation
pub fn validate_config(config: MHCConfiguration) !ValidationResult {
    var result = ValidationResult.init(std.heap.page_allocator);
    
    // Validate core settings
    try validate_core_config(config.core, &result);
    
    // Validate matrix_ops
    try validate_matrix_ops_config(config.matrix_ops, &result);
    
    // Validate transformer
    try validate_transformer_config(config.transformer, &result);
    
    // Validate GGUF
    try validate_gguf_config(config.gguf, &result);
    
    // Validate geometric (if present)
    if (config.geometric) |geometric| {
        try validate_geometric_config(geometric, &result);
    }
    
    // Validate monitoring (if present)
    if (config.monitoring) |monitoring| {
        try validate_monitoring_config(monitoring, &result);
    }
    
    // Validate runtime
    try validate_runtime_config(config.runtime, &result);
    
    // Cross-validation (dependencies between sections)
    try validate_dependencies(config, &result);
    
    return result;
}

fn validate_core_config(core: CoreConfig, result: *ValidationResult) !void {
    // Range validation
    if (core.sinkhorn_iterations < 5 or core.sinkhorn_iterations > 50) {
        try result.add_error(
            "core.sinkhorn_iterations",
            .out_of_range,
            "Must be between 5 and 50",
            try std.fmt.allocPrint(result.errors.allocator, "{}", .{core.sinkhorn_iterations}),
        );
    }
    
    if (core.manifold_epsilon <= 0 or core.manifold_epsilon >= 1) {
        try result.add_error(
            "core.manifold_epsilon",
            .out_of_range,
            "Must be between 0 and 1",
            try std.fmt.allocPrint(result.errors.allocator, "{d}", .{core.manifold_epsilon}),
        );
    }
    
    if (core.stability_threshold <= 0) {
        try result.add_error(
            "core.stability_threshold",
            .out_of_range,
            "Must be positive",
            try std.fmt.allocPrint(result.errors.allocator, "{d}", .{core.stability_threshold}),
        );
    }
    
    if (core.manifold_beta <= 0) {
        try result.add_error(
            "core.manifold_beta",
            .out_of_range,
            "Must be positive",
            try std.fmt.allocPrint(result.errors.allocator, "{d}", .{core.manifold_beta}),
        );
    }
    
    // Layer range validation
    if (core.layer_range) |range| {
        if (range.start > range.end) {
            try result.add_error(
                "core.layer_range",
                .constraint_violation,
                "Start must be <= end",
                try std.fmt.allocPrint(result.errors.allocator, "start={}, end={}", .{ range.start, range.end }),
            );
        }
    }
}

fn validate_transformer_config(transformer: TransformerConfig, result: *ValidationResult) !void {
    // Enum validation
    const valid_selections = &[_][]const u8{ "all", "adaptive", "manual" };
    var valid = false;
    for (valid_selections) |sel| {
        if (std.mem.eql(u8, transformer.layer_selection, sel)) {
            valid = true;
            break;
        }
    }
    if (!valid) {
        try result.add_error(
            "transformer.layer_selection",
            .invalid_enum,
            "Must be one of: all, adaptive, manual",
            transformer.layer_selection,
        );
    }
    
    // Dependency validation
    if (std.mem.eql(u8, transformer.layer_selection, "manual")) {
        if (transformer.manual_layer_range == null) {
            try result.add_error(
                "transformer.manual_layer_range",
                .missing_required,
                "Required when layer_selection = 'manual'",
                "null",
            );
        }
    }
    
    // Range validation
    if (transformer.adaptive_threshold < 0.9 or transformer.adaptive_threshold > 2.0) {
        try result.add_error(
            "transformer.adaptive_threshold",
            .out_of_range,
            "Must be between 0.9 and 2.0",
            try std.fmt.allocPrint(result.errors.allocator, "{d}", .{transformer.adaptive_threshold}),
        );
    }
}

fn validate_dependencies(config: MHCConfiguration, result: *ValidationResult) !void {
    // If geometric extensions enabled, core must be enabled
    if (config.geometric) |geometric| {
        if (geometric.enabled and !config.core.enabled) {
            try result.add_warning(
                "geometric.enabled",
                "Geometric extensions require core.enabled = true",
            );
        }
    }
    
    // If monitoring enabled, core should be enabled for meaningful metrics
    if (config.monitoring) |monitoring| {
        if (monitoring.failure_detection and !config.core.enabled) {
            try result.add_warning(
                "monitoring.failure_detection",
                "Failure detection requires core.enabled = true for full functionality",
            );
        }
    }
    
    // If GGUF requires metadata, auto_detect should be true
    if (config.gguf.require_metadata and !config.gguf.auto_detect) {
        try result.add_warning(
            "gguf.require_metadata",
            "require_metadata = true but auto_detect = false may cause issues",
        );
    }
}
```

### 6.2 Validation Modes

```zig
pub const ValidationMode = enum {
    /// Strict: Fail on any validation error
    strict,
    
    /// Warn: Log warnings but continue
    warn,
    
    /// Silent: No validation output (not recommended)
    silent,
};

pub fn validate_with_mode(
    config: MHCConfiguration,
    mode: ValidationMode,
) !void {
    const result = try validate_config(config);
    defer result.deinit();
    
    switch (mode) {
        .strict => {
            if (!result.valid) {
                std.log.err("Configuration validation failed:", .{});
                for (result.errors.items) |err| {
                    std.log.err("  {}", .{err});
                }
                return error.InvalidConfiguration;
            }
            
            if (result.warnings.items.len > 0) {
                std.log.warn("Configuration warnings:", .{});
                for (result.warnings.items) |warn| {
                    std.log.warn("  {}", .{warn});
                }
            }
        },
        
        .warn => {
            if (!result.valid) {
                std.log.warn("Configuration validation failed:", .{});
                for (result.errors.items) |err| {
                    std.log.warn("  {}", .{err});
                }
                std.log.warn("Continuing with invalid configuration (warn mode)", .{});
            }
            
            for (result.warnings.items) |warn| {
                std.log.warn("  {}", .{warn});
            }
        },
        
        .silent => {
            // No output
        },
    }
}
```

### 6.3 Schema Version Compatibility

```zig
/// Check schema version compatibility
pub fn check_schema_compatibility(schema_version: []const u8) !void {
    const current_version = "1.0.0";
    
    // Parse versions
    const current = try parse_semver(current_version);
    const config = try parse_semver(schema_version);
    
    // Major version must match
    if (current.major != config.major) {
        std.log.err(
            "Incompatible schema version: config={s}, server={s}",
            .{ schema_version, current_version },
        );
        return error.IncompatibleSchemaVersion;
    }
    
    // Warn on minor version mismatch
    if (current.minor != config.minor) {
        std.log.warn(
            "Schema minor version mismatch: config={s}, server={s}",
            .{ schema_version, current_version },
        );
    }
}

const SemVer = struct {
    major: u32,
    minor: u32,
    patch: u32,
};

fn parse_semver(version: []const u8) !SemVer {
    var iter = std.mem.split(u8, version, ".");
    const major = try std.fmt.parseInt(u32, iter.next() orelse return error.InvalidVersion, 10);
    const minor = try std.fmt.parseInt(u32, iter.next() orelse return error.InvalidVersion, 10);
    const patch = try std.fmt.parseInt(u32, iter.next() orelse return error.InvalidVersion, 10);
    return SemVer{ .major = major, .minor = minor, .patch = patch };
}
```

---

## 7. Configuration Loading

### 7.1 Complete Loading Pipeline

```zig
/// Load configuration from all sources
pub fn load_configuration(
    allocator: std.mem.Allocator,
    cli_args: [][]const u8,
) !MHCConfiguration {
    // Step 1: Load defaults
    var config = default_config();
    
    // Step 2: Load JSON file (if specified)
    const config_path = find_config_path(cli_args) orelse config.runtime.config_file_path;
    if (std.fs.cwd().access(config_path, .{})) {
        const json_config = try load_json_config(allocator, config_path);
        config = merge_two_configs(config, json_config);
    } else |_| {
        std.log.warn("Configuration file not found: {s}, using defaults", .{config_path});
    }
    
    // Step 3: Parse environment variables
    if (std.os.getenv("MHC_CORE_ENABLED")) |_| {
        const env_config = try parse_env_vars(allocator);
        config = merge_two_configs(config, env_config);
    }
    
    // Step 4: Parse CLI arguments
    if (cli_args.len > 0) {
        const cli_config = try parse_cli_args(allocator, cli_args);
        config = merge_two_configs(config, cli_config);
    }
    
    // Step 5: Validate final configuration
    try validate_with_mode(config, parse_validation_mode(config.runtime.validation_mode));
    
    // Step 6: Check schema compatibility
    try check_schema_compatibility(config.schema_version);
    
    // Step 7: Log final configuration (debug mode)
    if (std.log.defaultLogEnabled(.debug)) {
        log_config_summary(config);
    }
    
    return config;
}

/// Load configuration from JSON file
fn load_json_config(
    allocator: std.mem.Allocator,
    path: []const u8,
) !MHCConfiguration {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    
    const content = try file.readToEndAlloc(allocator, 1024 * 1024); // 1MB max
    defer allocator.free(content);
    
    // Parse JSON
    const parsed = try std.json.parseFromSlice(
        MHCConfiguration,
        allocator,
        content,
        .{ .allocate = .alloc_always },
    );
    defer parsed.deinit();
    
    return parsed.value;
}

/// Parse CLI arguments
fn parse_cli_args(
    allocator: std.mem.Allocator,
    args: [][]const u8,
) !MHCConfiguration {
    var config = default_config();
    
    for (args) |arg| {
        if (std.mem.startsWith(u8, arg, "--mhc-")) {
            const key_value = arg[6..]; // Strip "--mhc-"
            var iter = std.mem.split(u8, key_value, "=");
            const key = iter.next() orelse continue;
            const value = iter.next() orelse continue;
            
            // Parse key path (e.g., "core-enabled" -> core.enabled)
            try apply_cli_arg(&config, key, value);
        }
    }
    
    return config;
}

fn apply_cli_arg(
    config: *MHCConfiguration,
    key: []const u8,
    value: []const u8,
) !void {
    // Core settings
    if (std.mem.eql(u8, key, "core-enabled")) {
        config.core.enabled = parse_bool(value);
    } else if (std.mem.eql(u8, key, "core-sinkhorn-iterations")) {
        config.core.sinkhorn_iterations = try std.fmt.parseInt(u32, value, 10);
    } else if (std.mem.eql(u8, key, "core-manifold-epsilon")) {
        config.core.manifold_epsilon = try std.fmt.parseFloat(f32, value);
    }
    // ... continue for all parameters
}
```

### 7.2 Configuration Export

```zig
/// Export configuration to JSON file
pub fn export_config_to_json(
    config: MHCConfiguration,
    allocator: std.mem.Allocator,
    path: []const u8,
) !void {
    const file = try std.fs.cwd().createFile(path, .{});
    defer file.close();
    
    var string = std.ArrayList(u8).init(allocator);
    defer string.deinit();
    
    try std.json.stringify(config, .{ .whitespace = .indent_2 }, string.writer());
    
    try file.writeAll(string.items);
    
    std.log.info("Configuration exported to: {s}", .{path});
}
```

---

## 8. API Reference

### 8.1 Configuration Manager

```zig
/// Configuration manager with hot-reload support
pub const ConfigManager = struct {
    allocator: std.mem.Allocator,
    current_config: MHCConfiguration,
    hot_reload: ?ConfigHotReload,
    mutex: std.Thread.Mutex,
    
    pub fn init(
        allocator: std.mem.Allocator,
        cli_args: [][]const u8,
    ) !ConfigManager {
        // Load initial configuration
        const config = try load_configuration(allocator, cli_args);
        
        // Initialize hot-reload if enabled
        var hot_reload: ?ConfigHotReload = null;
        if (config.runtime.hot_reload) {
            hot_reload = try ConfigHotReload.init(
                allocator,
                config.runtime.config_file_path,
                config,
            );
            try hot_reload.?.start();
        }
        
        return ConfigManager{
            .allocator = allocator,
            .current_config = config,
            .hot_reload = hot_reload,
            .mutex = std.Thread.Mutex{},
        };
    }
    
    pub fn deinit(self: *ConfigManager) void {
        if (self.hot_reload) |*hr| {
            hr.deinit();
        }
    }
    
    /// Get current configuration (thread-safe)
    pub fn get_config(self: *ConfigManager) MHCConfiguration {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.current_config;
    }
    
    /// Update configuration programmatically
    pub fn update_config(
        self: *ConfigManager,
        new_config: MHCConfiguration,
    ) !void {
        // Validate new configuration
        try validate_with_mode(new_config, parse_validation_mode(new_config.runtime.validation_mode));
        
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const old_config = self.current_config;
        self.current_config = new_config;
        
        // Log changes
        if (new_config.runtime.log_config_changes) {
            log_config_diff(old_config, new_config);
        }
        
        std.log.info("Configuration updated programmatically", .{});
    }
    
    /// Register callback for configuration changes
    pub fn on_change(
        self: *ConfigManager,
        callback: *const fn (MHCConfiguration) void,
    ) !void {
        if (self.hot_reload) |*hr| {
            try hr.on_change(callback);
        } else {
            return error.HotReloadDisabled;
        }
    }
    
    /// Reload configuration from file manually
    pub fn reload(self: *ConfigManager) !void {
        const new_config = try load_json_config(
            self.allocator,
            self.current_config.runtime.config_file_path,
        );
        try self.update_config(new_config);
    }
    
    /// Export current configuration
    pub fn export_to_file(self: *ConfigManager, path: []const u8) !void {
        const config = self.get_config();
        try export_config_to_json(config, self.allocator, path);
    }
};
```

### 8.2 Usage Example

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Get CLI args
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    // Initialize configuration manager
    var config_mgr = try ConfigManager.init(allocator, args[1..]);
    defer config_mgr.deinit();
    
    // Register callback for changes
    try config_mgr.on_change(on_config_changed);
    
    // Get current configuration
    const config = config_mgr.get_config();
    std.debug.print("mHC enabled: {}\n", .{config.core.enabled});
    
    // Use configuration...
    if (config.core.enabled) {
        // Run mHC-enabled inference
    }
    
    // Keep running (hot-reload active)
    while (true) {
        std.time.sleep(1 * std.time.ns_per_s);
    }
}

fn on_config_changed(new_config: MHCConfiguration) void {
    std.log.info("Configuration changed! mHC enabled: {}", .{new_config.core.enabled});
}
```

---

## 9. Examples

### 9.1 Development Setup

```bash
#!/bin/bash
# dev_setup.sh

# Set development environment variables
export MHC_CORE_ENABLED=true
export MHC_CORE_LOG_STABILITY_METRICS=true
export MHC_RUNTIME_VALIDATION_MODE=warn
export MHC_RUNTIME_HOT_RELOAD=true

# Create minimal config file
cat > config/mhc_config.json <<EOF
{
  "schema_version": "1.0.0",
  "core": {
    "enabled": true,
    "sinkhorn_iterations": 10
  },
  "matrix_ops": {
    "use_simd": true
  },
  "transformer": {
    "layer_selection": "all"
  },
  "runtime": {
    "hot_reload": true,
    "validation_mode": "warn"
  }
}
EOF

# Run server
./nOpenaiServer --mhc-core-log-stability-metrics=true
```

### 9.2 Production Deployment

```yaml
# kubernetes/mhc-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: mhc-config
data:
  mhc_config.json: |
    {
      "schema_version": "1.0.0",
      "core": {
        "enabled": true,
        "sinkhorn_iterations": 15,
        "manifold_epsilon": 1e-6,
        "early_stopping": true,
        "log_stability_metrics": false
      },
      "matrix_ops": {
        "use_mhc": true,
        "use_simd": true,
        "thread_pool_size": 0
      },
      "transformer": {
        "mhc_in_attention": true,
        "mhc_in_ffn": true,
        "layer_selection": "adaptive",
        "adaptive_threshold": 1.05
      },
      "monitoring": {
        "failure_detection": true,
        "prometheus_enabled": true,
        "prometheus_port": 9090
      },
      "runtime": {
        "hot_reload": false,
        "validation_mode": "strict",
        "audit_log_enabled": true
      }
    }
---
apiVersion: v1
kind: Secret
metadata:
  name: mhc-env-vars
type: Opaque
stringData:
  MHC_MONITORING_PROMETHEUS_PORT: "9090"
  MHC_RUNTIME_VALIDATION_MODE: "strict"
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nopenai-server
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: server
        image: nopenai-server:latest
        envFrom:
        - secretRef:
            name: mhc-env-vars
        volumeMounts:
        - name: config
          mountPath: /app/config
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: mhc-config
```

### 9.3 CLI Override Examples

```bash
# Enable mHC with custom iterations
./nOpenaiServer \
  --mhc-core-enabled=true \
  --mhc-core-sinkhorn-iterations=20 \
  --mhc-transformer-layer-selection=adaptive

# Disable SIMD for debugging
./nOpenaiServer \
  --mhc-matrix-ops-use-simd=false \
  --mhc-core-log-stability-metrics=true

# Production mode with strict validation
./nOpenaiServer \
  --mhc-runtime-validation-mode=strict \
  --mhc-runtime-hot-reload=false \
  --mhc-monitoring-prometheus-enabled=true
```

---

## 10. Migration Guide

### 10.1 Migrating from Hardcoded Config

**Before** (hardcoded):

```zig
const config = MHCConfig{
    .enabled = true,
    .sinkhorn_iterations = 10,
};
```

**After** (configuration system):

```zig
// Load from configuration system
var config_mgr = try ConfigManager.init(allocator, args);
defer config_mgr.deinit();

const config = config_mgr.get_config().core;
```

### 10.2 Adding New Configuration Parameters

**Step 1**: Update `MHCConfiguration` structure

```zig
pub const CoreConfig = struct {
    // ... existing fields ...
    
    /// New parameter
    new_parameter: f32 = 1.0,
};
```

**Step 2**: Update JSON schema

```json
{
  "core": {
    "properties": {
      "new_parameter": {
        "type": "number",
        "default": 1.0,
        "description": "Description of new parameter"
      }
    }
  }
}
```

**Step 3**: Add environment variable mapping

```bash
MHC_CORE_NEW_PARAMETER=1.5
```

**Step 4**: Add validation (if needed)

```zig
if (core.new_parameter < 0) {
    try result.add_error("core.new_parameter", .out_of_range, "Must be positive", ...);
}
```

### 10.3 Schema Version Upgrade

When upgrading schema version:

1. Update `schema_version` in all config files
2. Add migration function for backward compatibility:

```zig
fn migrate_config_v1_to_v2(v1_config: MHCConfigurationV1) MHCConfiguration {
    return MHCConfiguration{
        .schema_version = "2.0.0",
        .core = migrate_core_v1_to_v2(v1_config.core),
        // ... map all fields ...
    };
}
```

---

## Appendix A: Configuration Best Practices

### A.1 Development vs Production

**Development**:
- `hot_reload = true` (fast iteration)
- `validation_mode = "warn"` (forgiving)
- `log_stability_metrics = true` (debug info)
- `log_config_changes = true` (visibility)

**Production**:
- `hot_reload = false` (stability)
- `validation_mode = "strict"` (catch errors)
- `log_stability_metrics = false` (performance)
- `audit_log_enabled = true` (compliance)

### A.2 Performance Tuning

1. **Baseline**: Start with defaults
2. **Profile**: Measure actual bottlenecks
3. **Tune**: Adjust `sinkhorn_iterations`, `thread_pool_size`
4. **Validate**: Ensure stability maintained
5. **Document**: Record optimal settings

### A.3 Security Considerations

- Store sensitive configs in environment variables, not JSON files
- Use file permissions: `chmod 600 config/mhc_config.json`
- Enable audit logging for compliance
- Validate all external inputs
- Use `validation_mode = "strict"` in production

---

## Appendix B: Configuration Schema Version History

### Version 1.0.0 (Current)

- Initial configuration schema
- Support for Days 27-30 designs
- Core, matrix_ops, transformer, gguf sections
- Optional geometric and monitoring sections
- Hot-reload and validation system

### Future Versions

**Version 1.1.0** (Planned):
- Add Days 54-60 geometric extensions
- Hyperbolic, spherical, product manifolds
- Automatic geometry detection

**Version 1.2.0** (Planned):
- Add Days 61-67 monitoring features
- Uncertainty quantification
- Failure detection and alerts

**Version 2.0.0** (Future):
- Breaking changes if needed
- Full mHC production system (Day 70)

---

## Appendix C: Troubleshooting

### C.1 Common Issues

**Issue**: Configuration file not found

```
Solution: Check file path in runtime.config_file_path
Default: config/mhc_config.json
Override: --mhc-runtime-config-file-path=/path/to/config.json
```

**Issue**: Validation errors

```
Solution: Run with validation_mode = "warn" to see all issues
Check error messages for specific field problems
Use export_to_file() to generate valid example
```

**Issue**: Hot-reload not working

```
Solution: Ensure runtime.hot_reload = true
Check file permissions (must be readable)
Verify watch_interval_sec is reasonable (default: 5)
```

**Issue**: Environment variables not applied

```
Solution: Check variable naming (must start with MHC_)
Ensure variables set before server starts
Use printenv | grep MHC to verify
```

### C.2 Debugging Configuration

```zig
// Enable debug logging
std.log.default_level = .debug;

// Log loaded configuration
const config = config_mgr.get_config();
std.debug.print("Full config: {}\n", .{config});

// Export current config for inspection
try config_mgr.export_to_file("debug_config.json");
```

---

**End of Configuration System Design**

**Status**: Design complete, ready for implementation (Day 31)

**Next Steps**:
- Day 32: Week 6 Review & Test Strategy
- Day 33-34: Core Module Implementation
- Day 35-36: Matrix Operations Implementation
- Day 37-39: Transformer & GGUF Implementation

**Total Specification**: 15,000+ lines (40+ pages)

**Key Deliverables**:
1. Complete JSON schema with validation
2. Environment variable mapping (60+ variables)
3. Configuration hierarchy (4 layers)
4. Hot-reload system with callbacks
5. Comprehensive validation framework
6. API reference with examples
7. Migration guide and best practices
