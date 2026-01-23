# GGUF Loader mHC Metadata Enhancement Specification

**Document Version:** 1.0  
**Created:** 2026-01-19  
**Author:** Development Team  
**Status:** Design Complete  

---

## Table of Contents

1. [Overview](#overview)
2. [GGUF Format Background](#gguf-format-background)
3. [Metadata Schema Extensions](#metadata-schema-extensions)
4. [Auto-Detection Logic](#auto-detection-logic)
5. [Configuration Loading](#configuration-loading)
6. [Model Detection Pipeline](#model-detection-pipeline)
7. [Backward Compatibility](#backward-compatibility)
8. [Implementation Details](#implementation-details)
9. [Testing Strategy](#testing-strategy)
10. [Examples](#examples)

---

## 1. Overview

### 1.1 Purpose

This document specifies the enhancement of the GGUF (GPT-Generated Unified Format) loader to support mHC (Manifold-Constrained Hyper-Connections) metadata. The design enables:

1. **Metadata Storage**: Store mHC configuration in GGUF files
2. **Auto-Detection**: Automatically detect if a model was trained with mHC
3. **Configuration Loading**: Load mHC settings from model metadata
4. **Backward Compatibility**: Work seamlessly with existing GGUF files

### 1.2 Key Goals

- **Seamless Integration**: Load mHC config automatically from GGUF metadata
- **Fallback Support**: Use default config if metadata is missing
- **Version Control**: Track mHC version for compatibility
- **Validation**: Ensure loaded config is valid and consistent
- **Documentation**: Clear metadata format specification

### 1.3 Design Principles

- **Optional Metadata**: mHC metadata is optional, not required
- **Backward Compatible**: Existing GGUF files work without changes
- **Forward Compatible**: Design supports future mHC extensions
- **Self-Documenting**: Metadata includes version and description
- **Validation**: Validate all loaded values before use

---

## 2. GGUF Format Background

### 2.1 GGUF Structure

GGUF files consist of:

```
┌─────────────────────────────┐
│ Header (Magic + Version)    │
├─────────────────────────────┤
│ Tensor Count                │
├─────────────────────────────┤
│ Metadata KV Pairs           │ ← We extend this section
│   - general.name            │
│   - general.architecture    │
│   - llama.attention.head... │
│   - [mHC metadata]          │ ← New mHC keys added here
├─────────────────────────────┤
│ Tensor Info                 │
├─────────────────────────────┤
│ Padding                     │
├─────────────────────────────┤
│ Tensor Data                 │
└─────────────────────────────┘
```

### 2.2 Metadata Key-Value Format

GGUF metadata uses typed key-value pairs:

```
Key: String (null-terminated)
Type: uint32 (GGUF_TYPE_*)
Value: Type-specific encoding
```

**Supported Types:**
- `GGUF_TYPE_UINT8` - Unsigned 8-bit integer
- `GGUF_TYPE_INT8` - Signed 8-bit integer
- `GGUF_TYPE_UINT32` - Unsigned 32-bit integer
- `GGUF_TYPE_FLOAT32` - 32-bit floating point
- `GGUF_TYPE_BOOL` - Boolean (uint8)
- `GGUF_TYPE_STRING` - UTF-8 string
- `GGUF_TYPE_ARRAY` - Array of values

### 2.3 Existing Metadata Examples

```
general.name = "Llama 3.3 70B Instruct"
general.architecture = "llama"
llama.context_length = 131072
llama.attention.head_count = 64
llama.block_count = 80
llama.embedding_length = 8192
```

---

## 3. Metadata Schema Extensions

### 3.1 Metadata Namespace

All mHC metadata uses the `mhc.*` namespace prefix:

```
mhc.enabled = bool
mhc.version = string
mhc.config.* = various types
```

### 3.2 Core Metadata Keys

#### 3.2.1 Version and Status

```
Key: "mhc.enabled"
Type: GGUF_TYPE_BOOL
Description: Whether the model was trained/fine-tuned with mHC
Default: false
Example: true
```

```
Key: "mhc.version"
Type: GGUF_TYPE_STRING
Description: mHC implementation version (semver)
Default: "1.0.0"
Example: "1.0.0", "1.1.0", "2.0.0-beta"
```

```
Key: "mhc.description"
Type: GGUF_TYPE_STRING
Description: Human-readable description of mHC configuration
Optional: true
Example: "Deep layer stabilization for Arabic NLP"
```

#### 3.2.2 Core mHC Parameters

```
Key: "mhc.config.sinkhorn_iterations"
Type: GGUF_TYPE_UINT32
Description: Number of Sinkhorn-Knopp iterations
Default: 10
Range: [1, 100]
Example: 10
```

```
Key: "mhc.config.manifold_epsilon"
Type: GGUF_TYPE_FLOAT32
Description: Numerical stability epsilon
Default: 1e-6
Range: [1e-10, 1e-3]
Example: 1e-6
```

```
Key: "mhc.config.stability_threshold"
Type: GGUF_TYPE_FLOAT32
Description: Threshold for stability detection
Default: 1e-4
Range: [1e-6, 1e-2]
Example: 1e-4
```

```
Key: "mhc.config.manifold_beta"
Type: GGUF_TYPE_FLOAT32
Description: Manifold projection strength
Default: 10.0
Range: [0.1, 100.0]
Example: 10.0
```

```
Key: "mhc.config.manifold_type"
Type: GGUF_TYPE_STRING
Description: Manifold geometry type
Default: "Euclidean"
Values: ["Euclidean", "Hyperbolic", "Spherical", "Product"]
Example: "Euclidean"
```

```
Key: "mhc.config.early_stopping"
Type: GGUF_TYPE_BOOL
Description: Enable early stopping in Sinkhorn iterations
Default: true
Example: true
```

#### 3.2.3 Transformer-Specific Metadata

```
Key: "mhc.transformer.attention_enabled"
Type: GGUF_TYPE_BOOL
Description: Apply mHC to attention outputs
Default: true
Example: true
```

```
Key: "mhc.transformer.ffn_enabled"
Type: GGUF_TYPE_BOOL
Description: Apply mHC to FFN outputs
Default: true
Example: true
```

```
Key: "mhc.transformer.residual_enabled"
Type: GGUF_TYPE_BOOL
Description: Apply mHC to residual connections
Default: false
Example: false
```

```
Key: "mhc.transformer.layer_range_start"
Type: GGUF_TYPE_UINT32
Description: First layer to apply mHC (inclusive)
Optional: true (null = all layers)
Example: 60
```

```
Key: "mhc.transformer.layer_range_end"
Type: GGUF_TYPE_UINT32
Description: Last layer to apply mHC (exclusive)
Optional: true (null = all layers)
Example: 80
```

#### 3.2.4 Training Metadata (Optional)

```
Key: "mhc.training.trained_with_mhc"
Type: GGUF_TYPE_BOOL
Description: Whether model was trained from scratch with mHC
Default: false
Example: true
```

```
Key: "mhc.training.finetuned_with_mhc"
Type: GGUF_TYPE_BOOL
Description: Whether model was fine-tuned with mHC
Default: false
Example: true
```

```
Key: "mhc.training.training_steps"
Type: GGUF_TYPE_UINT32
Description: Number of training steps with mHC
Optional: true
Example: 100000
```

```
Key: "mhc.training.stability_history"
Type: GGUF_TYPE_ARRAY (FLOAT32)
Description: Historical stability metrics during training
Optional: true
Example: [0.95, 0.96, 0.97, 0.98]
```

### 3.3 Complete Metadata Example

```
# Core mHC
mhc.enabled = true
mhc.version = "1.0.0"
mhc.description = "Deep layer stabilization (layers 60-79)"

# Core Config
mhc.config.sinkhorn_iterations = 10
mhc.config.manifold_epsilon = 1e-6
mhc.config.stability_threshold = 1e-4
mhc.config.manifold_beta = 10.0
mhc.config.manifold_type = "Euclidean"
mhc.config.early_stopping = true

# Transformer Config
mhc.transformer.attention_enabled = true
mhc.transformer.ffn_enabled = true
mhc.transformer.residual_enabled = false
mhc.transformer.layer_range_start = 60
mhc.transformer.layer_range_end = 80

# Training Info (optional)
mhc.training.trained_with_mhc = false
mhc.training.finetuned_with_mhc = true
mhc.training.training_steps = 50000
```

---

## 4. Auto-Detection Logic

### 4.1 Detection Strategy

The GGUF loader automatically detects mHC configuration using a multi-level approach:

```
┌─────────────────────────────────────────────┐
│ 1. Check for "mhc.enabled" key             │
│    - If present and true → Load mHC config │
│    - If present and false → Disable mHC    │
│    - If absent → Proceed to heuristics     │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 2. Check for any "mhc.*" keys              │
│    - If any present → Assume mHC enabled   │
│    - Load available config                 │
│    - Use defaults for missing keys         │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ 3. No mHC metadata found                   │
│    - Use default config (mHC disabled)     │
│    - Log: "No mHC metadata, using defaults"│
└─────────────────────────────────────────────┘
```

### 4.2 Detection Function

```zig
/// Auto-detect mHC configuration from GGUF metadata
fn detectMHCConfig(metadata: *const GGUFMetadata) MHCDetectionResult {
    var result: MHCDetectionResult = .{
        .detected = false,
        .confidence = 0.0,
        .config = null,
        .source = .None,
    };
    
    // Level 1: Explicit mhc.enabled flag
    if (metadata.get("mhc.enabled")) |value| {
        const enabled = value.asBool() catch false;
        result.detected = enabled;
        result.confidence = 1.0;
        result.source = .Explicit;
        
        if (enabled) {
            result.config = try loadMHCConfigFromMetadata(metadata);
        }
        
        return result;
    }
    
    // Level 2: Presence of any mhc.* keys
    const mhc_keys = metadata.keysWithPrefix("mhc.");
    if (mhc_keys.len > 0) {
        result.detected = true;
        result.confidence = 0.9;  // High confidence
        result.source = .Heuristic;
        result.config = try loadMHCConfigFromMetadata(metadata);
        
        std.log.info(
            "Detected mHC config from {} metadata keys",
            .{mhc_keys.len},
        );
        
        return result;
    }
    
    // Level 3: No mHC metadata
    result.detected = false;
    result.confidence = 1.0;  // Confident it's not mHC
    result.source = .None;
    
    std.log.debug("No mHC metadata found, using defaults", .{});
    
    return result;
}

const MHCDetectionResult = struct {
    detected: bool,
    confidence: f32,  // [0.0, 1.0]
    config: ?mhc.MHCConfig,
    source: DetectionSource,
};

const DetectionSource = enum {
    None,       // No mHC metadata
    Explicit,   // mhc.enabled flag
    Heuristic,  // Inferred from mhc.* keys
};
```

### 4.3 Confidence Scoring

- **1.0 (Certain)**: Explicit `mhc.enabled` flag present
- **0.9 (High)**: Multiple `mhc.*` keys present
- **0.5 (Medium)**: Single `mhc.*` key present (might be accidental)
- **0.0 (None)**: No mHC metadata found

---

## 5. Configuration Loading

### 5.1 Loading Function

```zig
/// Load mHC configuration from GGUF metadata
fn loadMHCConfigFromMetadata(
    metadata: *const GGUFMetadata,
) !mhc.MHCConfig {
    var config: mhc.MHCConfig = .{
        .enabled = true,  // Already detected
        .sinkhorn_iterations = 10,
        .manifold_epsilon = 1e-6,
        .stability_threshold = 1e-4,
        .manifold_beta = 10.0,
        .manifold_type = .Euclidean,
        .log_metrics = false,
        .early_stopping = true,
    };
    
    // Load core parameters
    if (metadata.get("mhc.config.sinkhorn_iterations")) |value| {
        const iters = try value.asU32();
        if (iters >= 1 and iters <= 100) {
            config.sinkhorn_iterations = iters;
        } else {
            std.log.warn(
                "Invalid sinkhorn_iterations {}, using default 10",
                .{iters},
            );
        }
    }
    
    if (metadata.get("mhc.config.manifold_epsilon")) |value| {
        const eps = try value.asF32();
        if (eps >= 1e-10 and eps <= 1e-3) {
            config.manifold_epsilon = eps;
        } else {
            std.log.warn(
                "Invalid manifold_epsilon {d}, using default 1e-6",
                .{eps},
            );
        }
    }
    
    if (metadata.get("mhc.config.stability_threshold")) |value| {
        const threshold = try value.asF32();
        if (threshold >= 1e-6 and threshold <= 1e-2) {
            config.stability_threshold = threshold;
        } else {
            std.log.warn(
                "Invalid stability_threshold {d}, using default 1e-4",
                .{threshold},
            );
        }
    }
    
    if (metadata.get("mhc.config.manifold_beta")) |value| {
        const beta = try value.asF32();
        if (beta >= 0.1 and beta <= 100.0) {
            config.manifold_beta = beta;
        } else {
            std.log.warn(
                "Invalid manifold_beta {d}, using default 10.0",
                .{beta},
            );
        }
    }
    
    if (metadata.get("mhc.config.manifold_type")) |value| {
        const type_str = try value.asString();
        config.manifold_type = parseManifoldType(type_str) orelse blk: {
            std.log.warn(
                "Invalid manifold_type '{s}', using Euclidean",
                .{type_str},
            );
            break :blk .Euclidean;
        };
    }
    
    if (metadata.get("mhc.config.early_stopping")) |value| {
        config.early_stopping = try value.asBool();
    }
    
    return config;
}

fn parseManifoldType(s: []const u8) ?mhc.ManifoldType {
    if (std.mem.eql(u8, s, "Euclidean")) return .Euclidean;
    if (std.mem.eql(u8, s, "Hyperbolic")) return .Hyperbolic;
    if (std.mem.eql(u8, s, "Spherical")) return .Spherical;
    if (std.mem.eql(u8, s, "Product")) return .Product;
    return null;
}
```

### 5.2 Transformer Configuration Loading

```zig
/// Load transformer-specific mHC config from metadata
fn loadTransformerMHCConfig(
    metadata: *const GGUFMetadata,
) !MHCTransformerConfig {
    var config: MHCTransformerConfig = .{
        .enabled = true,
        .attention_enabled = true,
        .ffn_enabled = true,
        .residual_enabled = false,
        .layer_range = null,
        .core = try loadMHCConfigFromMetadata(metadata),
    };
    
    // Load transformer-specific settings
    if (metadata.get("mhc.transformer.attention_enabled")) |value| {
        config.attention_enabled = try value.asBool();
    }
    
    if (metadata.get("mhc.transformer.ffn_enabled")) |value| {
        config.ffn_enabled = try value.asBool();
    }
    
    if (metadata.get("mhc.transformer.residual_enabled")) |value| {
        config.residual_enabled = try value.asBool();
    }
    
    // Load layer range (if specified)
    const has_start = metadata.get("mhc.transformer.layer_range_start");
    const has_end = metadata.get("mhc.transformer.layer_range_end");
    
    if (has_start != null and has_end != null) {
        const start = try has_start.?.asU32();
        const end = try has_end.?.asU32();
        
        if (start < end) {
            config.layer_range = .{ .start = start, .end = end };
            std.log.info(
                "Layer range: {}-{} (exclusive)",
                .{start, end},
            );
        } else {
            std.log.warn(
                "Invalid layer range {}-{}, ignoring",
                .{start, end},
            );
        }
    } else if (has_start != null or has_end != null) {
        std.log.warn(
            "Incomplete layer range (need both start and end), ignoring",
            .{},
        );
    }
    
    return config;
}
```

### 5.3 Validation

```zig
/// Validate loaded mHC configuration
fn validateMHCConfig(config: mhc.MHCConfig) !void {
    // Validate sinkhorn_iterations
    if (config.sinkhorn_iterations < 1 or config.sinkhorn_iterations > 100) {
        return error.InvalidSinkhornIterations;
    }
    
    // Validate manifold_epsilon
    if (config.manifold_epsilon < 1e-10 or config.manifold_epsilon > 1e-3) {
        return error.InvalidManifoldEpsilon;
    }
    
    // Validate stability_threshold
    if (config.stability_threshold < 1e-6 or config.stability_threshold > 1e-2) {
        return error.InvalidStabilityThreshold;
    }
    
    // Validate manifold_beta
    if (config.manifold_beta < 0.1 or config.manifold_beta > 100.0) {
        return error.InvalidManifoldBeta;
    }
    
    std.log.debug("mHC config validation passed", .{});
}
```

---

## 6. Model Detection Pipeline

### 6.1 Complete Loading Pipeline

```zig
/// Load model with automatic mHC detection
pub fn loadModel(
    filepath: []const u8,
    allocator: std.mem.Allocator,
) !LoadedModel {
    // 1. Parse GGUF file
    const gguf = try parseGGUF(filepath, allocator);
    defer gguf.deinit();
    
    // 2. Load standard metadata
    const model_metadata = try loadStandardMetadata(gguf);
    
    // 3. Detect and load mHC configuration
    const mhc_detection = detectMHCConfig(gguf.metadata);
    
    var model = LoadedModel{
        .metadata = model_metadata,
        .mhc_enabled = mhc_detection.detected,
        .mhc_config = null,
        .tensors = undefined,
    };
    
    if (mhc_detection.detected) {
        std.log.info(
            "mHC detected (confidence: {d:.2f}, source: {s})",
            .{
                mhc_detection.confidence,
                @tagName(mhc_detection.source),
            },
        );
        
        // Load full mHC config
        const mhc_config = mhc_detection.config orelse
            try loadMHCConfigFromMetadata(gguf.metadata);
        
        // Validate config
        try validateMHCConfig(mhc_config);
        
        model.mhc_config = mhc_config;
        
        // Log loaded configuration
        logMHCConfig(mhc_config);
    } else {
        std.log.info("No mHC configuration detected", .{});
    }
    
    // 4. Load tensors
    model.tensors = try loadTensors(gguf, allocator);
    
    return model;
}

const LoadedModel = struct {
    metadata: ModelMetadata,
    mhc_enabled: bool,
    mhc_config: ?mhc.MHCConfig,
    tensors: TensorCollection,
    
    pub fn deinit(self: *LoadedModel) void {
        self.tensors.deinit();
    }
};
```

### 6.2 Logging Helper

```zig
fn logMHCConfig(config: mhc.MHCConfig) void {
    std.log.info("mHC Configuration:", .{});
    std.log.info("  Sinkhorn iterations: {}", .{config.sinkhorn_iterations});
    std.log.info("  Manifold epsilon: {d:.2e}", .{config.manifold_epsilon});
    std.log.info("  Stability threshold: {d:.2e}", .{config.stability_threshold});
    std.log.info("  Manifold beta: {d:.1f}", .{config.manifold_beta});
    std.log.info("  Manifold type: {s}", .{@tagName(config.manifold_type)});
    std.log.info("  Early stopping: {}", .{config.early_stopping});
}
```

---

## 7. Backward Compatibility

### 7.1 Compatibility Strategy

**100% Backward Compatible:**
- Existing GGUF files work without any modifications
- Missing mHC metadata → Default config (mHC disabled)
- No changes required to existing loaders

**Forward Compatible:**
- Unknown `mhc.*` keys → Ignored with warning
- Newer mHC versions → Check version compatibility
- Invalid values → Use defaults with warning

### 7.2 Version Compatibility

```zig
/// Check mHC version compatibility
fn checkMHCVersionCompatibility(
    version_str: []const u8,
) !CompatibilityResult {
    const version = try parseSemanticVersion(version_str);
    const current = SemanticVersion{ .major = 1, .minor = 0, .patch = 0 };
    
    // Major version mismatch → Incompatible
    if (version.major != current.major) {
        std.log.err(
            "mHC version mismatch: file={s}, current={}.{}.{}",
            .{version_str, current.major, current.minor, current.patch},
        );
        return .{ .compatible = false, .reason = "Major version mismatch" };
    }
    
    // Minor version ahead → Warning (may work)
    if (version.minor > current.minor) {
        std.log.warn(
            "mHC file version {s} is newer than current {}.{}.{}",
            .{version_str, current.major, current.minor, current.patch},
        );
        return .{ .compatible = true, .reason = "Newer minor version" };
    }
    
    // Compatible
    return .{ .compatible = true, .reason = "Version compatible" };
}

const SemanticVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,
};

const CompatibilityResult = struct {
    compatible: bool,
    reason: []const u8,
};
```

### 7.3 Migration Path

**For Model Authors:**

1. **Add mHC metadata to GGUF files** (optional):
   ```python
   # Using gguf-py library
   import gguf
   
   writer = gguf.GGUFWriter("model.gguf", "llama")
   
   # Add mHC metadata
   writer.add_bool("mhc.enabled", True)
   writer.add_string("mhc.version", "1.0.0")
   writer.add_uint32("mhc.config.sinkhorn_iterations", 10)
   writer.add_float32("mhc.config.manifold_epsilon", 1e-6)
   # ... other mHC keys ...
   
   writer.write()
   ```

2. **Or use CLI tool**:
   ```bash
   # Add mHC metadata to existing GGUF
   gguf-add-metadata \
     --input model.gguf \
     --output model-mhc.gguf \
     --mhc-enabled true \
     --mhc-sinkhorn-iterations 10 \
     --mhc-layer-range 60-80
   ```

**For Inference Users:**
- No changes required
- mHC automatically detected and applied
- Override with CLI flags if needed

---

## 8. Implementation Details

### 8.1 Enhanced ModelMetadata Structure

```zig
const ModelMetadata = struct {
    // Existing fields
    name: []const u8,
    architecture: []const u8,
    context_length: u32,
    embedding_length: u32,
    block_count: u32,
    head_count: u32,
    
    // NEW: mHC fields
    mhc_enabled: bool = false,
    mhc_version: ?[]const u8 = null,
    mhc_description: ?[]const u8 = null,
    mhc_config: ?mhc.MHCConfig = null,
    mhc_transformer_config: ?MHCTransformerConfig = null,
    
    pub fn hasMHC(self: ModelMetadata) bool {
        return self.mhc_enabled and self.mhc_config != null;
    }
    
    pub fn getMHCConfig(self: ModelMetadata) ?mhc.MHCConfig {
        if (!self.mhc_enabled) return null;
        return self.mhc_config;
    }
};
```

### 8.2 GGUF Loader Integration

```zig
// In gguf_loader.zig

pub fn load(
    path: []const u8,
    allocator: std.mem.Allocator,
) !LoadedModel {
    // Parse GGUF file
    var file = try std.fs.cwd().openFile(path, .{});
    defer file.close();
    
    const gguf = try parseGGUFHeader(file, allocator);
    defer gguf.deinit();
    
    // Load standard metadata
    var metadata = try loadStandardMetadata(gguf, allocator);
    
    // Auto-detect and load mHC configuration
    const mhc_result = detectMHCConfig(gguf.metadata);
    
    if (mhc_result.detected) {
        metadata.mhc_enabled = true;
        metadata.mhc_config = mhc_result.config;
        
        // Load version info
        if (gguf.metadata.get("mhc.version")) |v| {
            metadata.mhc_version = try v.asString();
            
            // Check compatibility
            const compat = try checkMHCVersionCompatibility(
                metadata.mhc_version.?,
            );
            if (!compat.compatible) {
                return error.IncompatibleMHCVersion;
            }
        }
        
        // Load transformer config
        metadata.mhc_transformer_config =
            try loadTransformerMHCConfig(gguf.metadata);
    }
    
    // Load tensors
    const tensors = try loadTensors(gguf, file, allocator);
    
    return LoadedModel{
        .metadata = metadata,
        .tensors = tensors,
    };
}
```

### 8.3 CLI Override Support

```zig
/// CLI arguments for mHC override
const CLIArgs = struct {
    // Model path
    model_path: []const u8,
    
    // mHC overrides (optional)
    mhc_override_enabled: ?bool = null,
    mhc_override_iterations: ?u32 = null,
    mhc_override_epsilon: ?f32 = null,
    mhc_override_layer_range: ?LayerRange = null,
    
    /// Apply CLI overrides to loaded metadata
    pub fn applyOverrides(
        self: CLIArgs,
        metadata: *ModelMetadata,
    ) void {
        if (self.mhc_override_enabled) |enabled| {
            metadata.mhc_enabled = enabled;
            std.log.info("CLI override: mhc_enabled = {}", .{enabled});
        }
        
        if (metadata.mhc_config) |*config| {
            if (self.mhc_override_iterations) |iters| {
                config.sinkhorn_iterations = iters;
                std.log.info(
                    "CLI override: sinkhorn_iterations = {}",
                    .{iters},
                );
            }
            
            if (self.mhc_override_epsilon) |eps| {
                config.manifold_epsilon = eps;
                std.log.info(
                    "CLI override: manifold_epsilon = {d:.2e}",
                    .{eps},
                );
            }
        }
        
        if (metadata.mhc_transformer_config) |*tconfig| {
            if (self.mhc_override_layer_range) |range| {
                tconfig.layer_range = range;
                std.log.info(
                    "CLI override: layer_range = {}-{}",
                    .{range.start, range.end},
                );
            }
        }
    }
};
```

---

## 9. Testing Strategy

### 9.1 Unit Tests

```zig
// Test 1: Load GGUF without mHC metadata
test "gguf_load_no_mhc" {
    const path = "test_models/llama-base.gguf";
    const model = try loadModel(path, testing.allocator);
    defer model.deinit();
    
    try testing.expect(!model.mhc_enabled);
    try testing.expect(model.mhc_config == null);
}

// Test 2: Load GGUF with complete mHC metadata
test "gguf_load_with_mhc" {
    const path = "test_models/llama-mhc-full.gguf";
    const model = try loadModel(path, testing.allocator);
    defer model.deinit();
    
    try testing.expect(model.mhc_enabled);
    try testing.expect(model.mhc_config != null);
    
    const config = model.mhc_config.?;
    try testing.expectEqual(@as(u32, 10), config.sinkhorn_iterations);
    try testing.expectApproxEqAbs(@as(f32, 1e-6), config.manifold_epsilon, 1e-10);
    try testing.expectEqual(mhc.ManifoldType.Euclidean, config.manifold_type);
}

// Test 3: Load GGUF with partial mHC metadata (use defaults)
test "gguf_load_partial_mhc" {
    const path = "test_models/llama-mhc-partial.gguf";
    const model = try loadModel(path, testing.allocator);
    defer model.deinit();
    
    try testing.expect(model.mhc_enabled);
    try testing.expect(model.mhc_config != null);
    
    const config = model.mhc_config.?;
    // Custom value from metadata
    try testing.expectEqual(@as(u32, 15), config.sinkhorn_iterations);
    // Default value (not in metadata)
    try testing.expectApproxEqAbs(@as(f32, 1e-6), config.manifold_epsilon, 1e-10);
}

// Test 4: Invalid mHC metadata (validation fails)
test "gguf_load_invalid_mhc" {
    const path = "test_models/llama-mhc-invalid.gguf";
    
    // Should fail validation
    try testing.expectError(
        error.InvalidSinkhornIterations,
        loadModel(path, testing.allocator),
    );
}

// Test 5: Version compatibility check
test "mhc_version_compatibility" {
    // Compatible version
    const compat1 = try checkMHCVersionCompatibility("1.0.0");
    try testing.expect(compat1.compatible);
    
    // Newer minor version (warning but compatible)
    const compat2 = try checkMHCVersionCompatibility("1.1.0");
    try testing.expect(compat2.compatible);
    
    // Major version mismatch (incompatible)
    const compat3 = try checkMHCVersionCompatibility("2.0.0");
    try testing.expect(!compat3.compatible);
}

// Test 6: CLI overrides
test "mhc_cli_overrides" {
    var metadata = ModelMetadata{
        .mhc_enabled = true,
        .mhc_config = .{
            .enabled = true,
            .sinkhorn_iterations = 10,
            .manifold_epsilon = 1e-6,
        },
    };
    
    const args = CLIArgs{
        .model_path = "model.gguf",
        .mhc_override_iterations = 20,
        .mhc_override_epsilon = 1e-5,
    };
    
    args.applyOverrides(&metadata);
    
    try testing.expectEqual(@as(u32, 20), metadata.mhc_config.?.sinkhorn_iterations);
    try testing.expectApproxEqAbs(
        @as(f32, 1e-5),
        metadata.mhc_config.?.manifold_epsilon,
        1e-10,
    );
}

// Test 7: Transformer config loading
test "gguf_load_transformer_mhc" {
    const path = "test_models/llama-mhc-transformer.gguf";
    const model = try loadModel(path, testing.allocator);
    defer model.deinit();
    
    try testing.expect(model.metadata.mhc_transformer_config != null);
    
    const tconfig = model.metadata.mhc_transformer_config.?;
    try testing.expect(tconfig.attention_enabled);
    try testing.expect(tconfig.ffn_enabled);
    try testing.expect(!tconfig.residual_enabled);
    
    try testing.expect(tconfig.layer_range != null);
    try testing.expectEqual(@as(u32, 60), tconfig.layer_range.?.start);
    try testing.expectEqual(@as(u32, 80), tconfig.layer_range.?.end);
}
```

### 9.2 Integration Tests

```zig
// Test 8: End-to-end model loading and inference
test "gguf_mhc_integration" {
    // This test requires actual GGUF file with mHC metadata
    if (true) return error.SkipZigTest;
    
    // Load model
    const model = try loadModel("llama-3.3-70b-mhc.gguf", testing.allocator);
    defer model.deinit();
    
    // Verify mHC config loaded
    try testing.expect(model.mhc_enabled);
    try testing.expect(model.mhc_config != null);
    
    // Create transformer with loaded config
    const transformer_config = TransformerConfig{
        .n_layers = model.metadata.block_count,
        .n_heads = model.metadata.head_count,
        .d_model = model.metadata.embedding_length,
        // ... other fields ...
        .mhc_config = model.metadata.mhc_transformer_config.?,
    };
    
    // Run inference with mHC
    const input = try allocator.alloc(f32, 128 * 8192);
    defer allocator.free(input);
    
    const output = try transformerForward(input, transformer_config, allocator);
    defer allocator.free(output);
    
    try testing.expect(output.len == input.len);
}
```

---

## 10. Examples

### 10.1 Example 1: Basic Model Loading

```zig
const std = @import("std");
const gguf = @import("gguf_loader.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Load model (auto-detects mHC)
    const model = try gguf.loadModel(
        "models/llama-3.3-70b-instruct.gguf",
        allocator,
    );
    defer model.deinit();
    
    // Check if mHC is enabled
    if (model.metadata.hasMHC()) {
        const config = model.metadata.getMHCConfig().?;
        std.debug.print("mHC enabled:\n", .{});
        std.debug.print("  Version: {s}\n", .{model.metadata.mhc_version.?});
        std.debug.print("  Sinkhorn iterations: {}\n", .{config.sinkhorn_iterations});
        std.debug.print("  Manifold type: {s}\n", .{@tagName(config.manifold_type)});
    } else {
        std.debug.print("mHC not enabled\n", .{});
    }
    
    // Use model for inference...
}
```

### 10.2 Example 2: CLI Override

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse CLI args
    const args = try parseArgs(allocator);
    
    // Load model
    var model = try gguf.loadModel(args.model_path, allocator);
    defer model.deinit();
    
    // Apply CLI overrides
    if (args.mhc_override_enabled) |enabled| {
        model.metadata.mhc_enabled = enabled;
        std.log.info("mHC override: {}", .{enabled});
    }
    
    if (args.mhc_override_layer_range) |range| {
        if (model.metadata.mhc_transformer_config) |*tc| {
            tc.layer_range = range;
            std.log.info("Layer range override: {}-{}", .{range.start, range.end});
        }
    }
    
    // Run inference with overridden config...
}
```

### 10.3 Example 3: Creating mHC-Enabled GGUF (Python)

```python
#!/usr/bin/env python3
"""Add mHC metadata to GGUF file"""

import sys
import gguf

def add_mhc_metadata(input_path, output_path):
    # Read existing GGUF
    reader = gguf.GGUFReader(input_path)
    
    # Create writer with same architecture
    writer = gguf.GGUFWriter(output_path, reader.arch)
    
    # Copy existing metadata
    for key, value in reader.metadata.items():
        writer.add_key_value(key, value)
    
    # Add mHC metadata
    writer.add_bool("mhc.enabled", True)
    writer.add_string("mhc.version", "1.0.0")
    writer.add_string("mhc.description", "Deep layer stabilization")
    
    # Core config
    writer.add_uint32("mhc.config.sinkhorn_iterations", 10)
    writer.add_float32("mhc.config.manifold_epsilon", 1e-6)
    writer.add_float32("mhc.config.stability_threshold", 1e-4)
    writer.add_float32("mhc.config.manifold_beta", 10.0)
    writer.add_string("mhc.config.manifold_type", "Euclidean")
    writer.add_bool("mhc.config.early_stopping", True)
    
    # Transformer config
    writer.add_bool("mhc.transformer.attention_enabled", True)
    writer.add_bool("mhc.transformer.ffn_enabled", True)
    writer.add_bool("mhc.transformer.residual_enabled", False)
    writer.add_uint32("mhc.transformer.layer_range_start", 60)
    writer.add_uint32("mhc.transformer.layer_range_end", 80)
    
    # Training info (optional)
    writer.add_bool("mhc.training.trained_with_mhc", False)
    writer.add_bool("mhc.training.finetuned_with_mhc", True)
    writer.add_uint32("mhc.training.training_steps", 50000)
    
    # Copy tensors
    for tensor in reader.tensors:
        writer.add_tensor(tensor)
    
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    
    print(f"Created mHC-enabled GGUF: {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: add_mhc_metadata.py <input.gguf> <output.gguf>")
        sys.exit(1)
    
    add_mhc_metadata(sys.argv[1], sys.argv[2])
```

### 10.4 Example 4: Metadata Inspection Tool

```zig
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    
    if (args.len != 2) {
        std.debug.print("Usage: inspect_mhc <model.gguf>\n", .{});
        return error.InvalidArgs;
    }
    
    // Load model
    const model = try gguf.loadModel(args[1], allocator);
    defer model.deinit();
    
    // Print model info
    std.debug.print("Model: {s}\n", .{model.metadata.name});
    std.debug.print("Architecture: {s}\n", .{model.metadata.architecture});
    std.debug.print("Layers: {}\n", .{model.metadata.block_count});
    std.debug.print("\n", .{});
    
    // Print mHC info
    if (model.metadata.hasMHC()) {
        std.debug.print("mHC: ENABLED\n", .{});
        std.debug.print("Version: {s}\n", .{model.metadata.mhc_version.?});
        
        if (model.metadata.mhc_description) |desc| {
            std.debug.print("Description: {s}\n", .{desc});
        }
        
        const config = model.metadata.getMHCConfig().?;
        std.debug.print("\nCore Configuration:\n", .{});
        std.debug.print("  Sinkhorn iterations: {}\n", .{config.sinkhorn_iterations});
        std.debug.print("  Manifold epsilon: {d:.2e}\n", .{config.manifold_epsilon});
        std.debug.print("  Stability threshold: {d:.2e}\n", .{config.stability_threshold});
        std.debug.print("  Manifold beta: {d:.1f}\n", .{config.manifold_beta});
        std.debug.print("  Manifold type: {s}\n", .{@tagName(config.manifold_type)});
        std.debug.print("  Early stopping: {}\n", .{config.early_stopping});
        
        if (model.metadata.mhc_transformer_config) |tc| {
            std.debug.print("\nTransformer Configuration:\n", .{});
            std.debug.print("  Attention: {}\n", .{tc.attention_enabled});
            std.debug.print("  FFN: {}\n", .{tc.ffn_enabled});
            std.debug.print("  Residual: {}\n", .{tc.residual_enabled});
            
            if (tc.layer_range) |range| {
                std.debug.print("  Layer range: {}-{}\n", .{range.start, range.end});
            } else {
                std.debug.print("  Layer range: all layers\n", .{});
            }
        }
    } else {
        std.debug.print("mHC: DISABLED\n", .{});
    }
}
```

---

## Summary

This specification provides a complete design for integrating mHC metadata into GGUF files with:

1. **Complete Metadata Schema**: 15+ metadata keys covering all mHC parameters
2. **Auto-Detection**: 3-level detection strategy (explicit → heuristic → default)
3. **Backward Compatible**: 100% compatible with existing GGUF files
4. **Forward Compatible**: Supports future mHC versions with version checking
5. **Validation**: Comprehensive validation of all loaded values
6. **CLI Override**: Command-line override support for runtime configuration
7. **Well-Tested**: 8 unit tests + integration tests
8. **Documented**: 4 complete examples (loading, CLI, Python, inspection)

**Implementation Effort:** ~350 lines of core code + 150 lines of tests + 100 lines of examples = **600+ lines total**

**Key Benefits:**
- Automatic mHC configuration from model files
- No manual configuration required
- Seamless integration with existing workflows
- Clear migration path for model authors

---

**Document Status:** ✅ COMPLETE  
**Ready for Implementation:** YES  
**Next Steps:** Begin Day 38 implementation using this specification  
**Dependencies:** Day 27 (mhc_constraints.zig), Day 29 (transformer.zig)
