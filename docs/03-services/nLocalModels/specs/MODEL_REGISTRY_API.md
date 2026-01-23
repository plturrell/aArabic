# Model Registry API Documentation - Day 11

**Version:** 1.0.0  
**Date:** 2026-01-19  
**Status:** ✅ Production Ready

## Overview

The Enhanced Model Registry provides comprehensive multi-model management with versioning, metadata tracking, and filesystem integration. It automatically discovers models from `vendor/layerModels` and manages metadata in `vendor/layerData`.

## Features

- ✅ **Multi-Model Support** - Manage multiple models simultaneously
- ✅ **Version Management** - Semantic versioning (major.minor.patch)
- ✅ **Auto-Discovery** - Automatic model scanning from filesystem
- ✅ **Rich Metadata** - Architecture, quantization, size, tags, etc.
- ✅ **Health Tracking** - Model health status and usage statistics
- ✅ **JSON API** - OpenAI-compatible API endpoint format
- ✅ **Zero Dependencies** - Uses vendor/layerData (not SQLite/JSON files)

## Architecture

```
┌─────────────────────────────────────────────────┐
│          Model Registry (Zig)                   │
│  ┌───────────────────────────────────────────┐  │
│  │ Multi-Model HashMap                       │  │
│  │ - StringHashMap<ModelConfig>             │  │
│  │ - Version tracking per model             │  │
│  │ - Default model selection                │  │
│  └───────────────────────────────────────────┘  │
│                      │                          │
│  ┌──────────────────┴────────────────────────┐  │
│  │                                           │  │
│  ▼                                           ▼  │
│ [Discovery]                              [API]  │
│ vendor/layerModels                    toJson()  │
│ - Llama-3.2-1B/                      listModels()│
│ - Qwen2.5-0.5B/                      get()      │
│ - phi-2/                             register() │
│                                                  │
└─────────────────────────────────────────────────┘
```

## Core Data Structures

### ModelVersion

Semantic version representation:

```zig
pub const ModelVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,
    
    pub fn parse(str: []const u8) !ModelVersion
    pub fn compare(self: ModelVersion, other: ModelVersion) std.math.Order
    pub fn format(...) // For printing
};
```

**Example:**
```zig
const v1 = try ModelVersion.parse("1.2.3");
const v2 = ModelVersion{ .major = 2, .minor = 0, .patch = 0 };
const cmp = v1.compare(v2); // .lt (less than)
```

### ModelMetadata

Rich model information:

```zig
pub const ModelMetadata = struct {
    architecture: []const u8,      // "llama", "phi", "qwen", "gemma", "nemotron"
    quantization: []const u8,      // "Q4_K_M", "Q8_0", "F16", etc.
    parameter_count: []const u8,   // "1B", "3B", "7B", "70B", etc.
    format: []const u8,            // "gguf", "safetensors", "pytorch"
    context_length: u32,           // 4096, 8192, 32768, etc.
    tags: []const []const u8,      // ["local", "quantized", "instruct"]
    source: []const u8,            // "huggingface", "local", "ollama"
    license: []const u8,           // "MIT", "Apache-2.0", "Llama-2"
    created_at: i64,               // Unix timestamp
    size_bytes: u64,               // Total size in bytes
};
```

### ModelConfig

Complete model configuration:

```zig
pub const ModelConfig = struct {
    id: []const u8,                    // Unique identifier
    path: []const u8,                  // Filesystem path
    display_name: []const u8,          // Human-readable name
    version: ModelVersion,             // Semantic version
    metadata: ModelMetadata,           // Rich metadata
    preload: bool,                     // Preload on startup?
    max_workers: ?u32,                 // Worker thread limit
    max_tokens: ?u32,                  // Token generation limit
    temperature: ?f32,                 // Default temperature
    enabled: bool,                     // Model enabled?
    health_status: HealthStatus,       // Current health
    last_used: ?i64,                   // Last usage timestamp
    use_count: u64,                    // Usage counter
    
    pub const HealthStatus = enum {
        unknown, healthy, degraded, unhealthy, loading
    };
    
    pub fn markUsed(self: *ModelConfig) void
    pub fn updateHealthStatus(self: *ModelConfig, status: HealthStatus) void
};
```

## Model Registry API

### Initialization

```zig
const allocator = std.heap.page_allocator;

// Initialize registry with paths
var registry = try ModelRegistry.init(
    allocator,
    "vendor/layerModels",  // Model files directory
    "vendor/layerData"     // Metadata storage directory
);
defer registry.deinit();
```

### Model Discovery

Automatically scan filesystem for models:

```zig
const stats = try registry.discoverModels();

std.debug.print("Total scanned: {}\n", .{stats.total_scanned});
std.debug.print("Models found: {}\n", .{stats.models_found});
std.debug.print("Models added: {}\n", .{stats.models_added});
std.debug.print("Models updated: {}\n", .{stats.models_updated});
std.debug.print("Errors: {}\n", .{stats.errors});
```

**Discovery Stats:**
```zig
pub const DiscoveryStats = struct {
    total_scanned: u32 = 0,    // Directories scanned
    models_found: u32 = 0,     // Valid models found
    models_added: u32 = 0,     // New models added
    models_updated: u32 = 0,   // Existing models updated
    errors: u32 = 0,           // Errors encountered
};
```

### Model Registration

Manually register a model:

```zig
// Create metadata
const tags = try allocator.alloc([]const u8, 2);
tags[0] = try allocator.dupe(u8, "local");
tags[1] = try allocator.dupe(u8, "instruct");

const metadata = ModelMetadata{
    .architecture = try allocator.dupe(u8, "llama"),
    .quantization = try allocator.dupe(u8, "Q4_K_M"),
    .parameter_count = try allocator.dupe(u8, "3B"),
    .format = try allocator.dupe(u8, "gguf"),
    .context_length = 8192,
    .tags = tags,
    .source = try allocator.dupe(u8, "huggingface"),
    .license = try allocator.dupe(u8, "Llama-3"),
    .created_at = std.time.timestamp(),
    .size_bytes = 2_147_483_648, // 2GB
};

// Create config
const config = try ModelConfig.init(allocator, .{
    .id = "llama-3.2-3b-instruct",
    .path = "vendor/layerModels/Llama-3.2-3B",
    .display_name = "Llama 3.2 3B Instruct",
    .version = .{ .major = 3, .minor = 2, .patch = 0 },
    .metadata = metadata,
    .preload = true,
    .max_workers = 4,
    .max_tokens = 4096,
    .temperature = 0.7,
});

// Register
try registry.register(config);
```

### Model Retrieval

```zig
// Get by ID
if (registry.get("llama-3.2-3b-instruct")) |model| {
    std.debug.print("Found: {s}\n", .{model.display_name});
    std.debug.print("Architecture: {s}\n", .{model.metadata.architecture});
    std.debug.print("Size: {} GB\n", .{model.metadata.size_bytes / 1_073_741_824});
}

// Get mutable reference
if (registry.getMut("llama-3.2-3b-instruct")) |model| {
    model.markUsed();
    model.updateHealthStatus(.healthy);
}

// Get by version
const version = ModelVersion{ .major = 3, .minor = 2, .patch = 0 };
if (registry.getByVersion("llama-3.2-3b-instruct", version)) |model| {
    std.debug.print("Found version: {}\n", .{model.version});
}
```

### Default Model Management

```zig
// Get default model
if (registry.default()) |model| {
    std.debug.print("Default: {s}\n", .{model.id});
}

// Set default model
try registry.setDefault("llama-3.2-3b-instruct");
```

### Model Listing

```zig
// List all model IDs
const models = try registry.listModels(allocator);
defer {
    for (models) |model| allocator.free(model);
    allocator.free(models);
}

for (models) |model_id| {
    std.debug.print("  - {s}\n", .{model_id});
}

// List only healthy models
const healthy = try registry.getHealthyModels(allocator);
defer {
    for (healthy) |model| allocator.free(model);
    allocator.free(healthy);
}
```

### Version Management

```zig
// Get all versions of a model
if (registry.getVersions("llama-3.2-3b-instruct")) |versions| {
    for (versions) |version| {
        std.debug.print("Version: {}\n", .{version});
    }
}
```

### JSON Serialization

OpenAI-compatible API format:

```zig
const json = try registry.toJson(allocator);
defer allocator.free(json);

// Returns JSON like:
// {
//   "object": "list",
//   "data": [
//     {
//       "id": "llama-3.2-3b-instruct",
//       "display_name": "Llama 3.2 3B Instruct",
//       "path": "vendor/layerModels/Llama-3.2-3B",
//       "version": "3.2.0",
//       "architecture": "llama",
//       "parameter_count": "3B",
//       "enabled": true,
//       "health_status": "healthy",
//       "use_count": 42,
//       "size_bytes": 2147483648,
//       "preload": true,
//       "max_workers": 4,
//       "max_tokens": 4096,
//       "temperature": 0.7
//     }
//   ]
// }
```

### Health Status Management

```zig
// Update health status
if (registry.getMut("model-id")) |model| {
    model.updateHealthStatus(.healthy);    // or .degraded, .unhealthy, .loading
}

// Track usage
if (registry.getMut("model-id")) |model| {
    model.markUsed(); // Increments use_count, updates last_used
}

// Get healthy models only
const healthy = try registry.getHealthyModels(allocator);
defer {
    for (healthy) |model| allocator.free(model);
    allocator.free(healthy);
}
```

## Integration with Existing Systems

### With Discovery (model_scanner.mojo)

The Mojo model scanner discovers models, then the Zig registry manages them:

```mojo
# Mojo side: Discover models
var scanner = ModelScanner()
var count = scanner.scan()

# Pass to Zig registry via FFI
# (Registry auto-discovers on init)
```

### With Orchestration (llm_integration)

The registry provides models to the LLM orchestration system:

```zig
// Get model for inference
if (registry.get("llama-3.2-3b-instruct")) |model| {
    // Pass model.path to inference engine
    // Use model.metadata for optimal config
    // Track usage with model.markUsed()
}
```

### With Observability Stack

The registry integrates with Days 6-9 features:

```zig
// Structured logging (Day 6)
logger.info("Model registered", .{
    .model_id = config.id,
    .version = config.version,
    .architecture = config.metadata.architecture,
});

// Health checks (Day 9)
pub fn checkModelHealth(registry: *ModelRegistry) !HealthStatus {
    const healthy_count = (try registry.getHealthyModels(allocator)).len;
    const total_count = registry.len();
    
    if (healthy_count == total_count) return .healthy;
    if (healthy_count > 0) return .degraded;
    return .unhealthy;
}
```

## Usage Examples

### Example 1: Simple Discovery

```zig
const std = @import("std");
const ModelRegistry = @import("model_registry.zig").ModelRegistry;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    const stats = try registry.discoverModels();
    std.debug.print("Discovered {} models\n", .{stats.models_found});

    const json = try registry.toJson(allocator);
    defer allocator.free(json);
    std.debug.print("{s}\n", .{json});
}
```

### Example 2: Model Selection

```zig
pub fn selectBestModel(registry: *const ModelRegistry, task: []const u8) ?*const ModelConfig {
    // Get all healthy models
    const models = registry.getHealthyModels(allocator) catch return null;
    defer allocator.free(models);

    // Select based on task requirements
    for (models) |model_id| {
        if (registry.get(model_id)) |model| {
            // Prefer larger models for complex tasks
            if (std.mem.eql(u8, task, "complex")) {
                if (std.mem.indexOf(u8, model.metadata.parameter_count, "7B") != null or
                    std.mem.indexOf(u8, model.metadata.parameter_count, "3B") != null) {
                    return model;
                }
            }
            // Prefer smaller models for simple tasks
            else if (std.mem.eql(u8, task, "simple")) {
                if (std.mem.indexOf(u8, model.metadata.parameter_count, "1B") != null or
                    std.mem.indexOf(u8, model.metadata.parameter_count, "0.5B") != null) {
                    return model;
                }
            }
        }
    }

    return registry.default();
}
```

### Example 3: Health Monitoring

```zig
pub fn monitorModels(registry: *ModelRegistry) !void {
    var iter = registry.models.iterator();
    
    while (iter.next()) |entry| {
        const model = entry.value_ptr;
        
        // Check if model is stale (not used in 24 hours)
        if (model.last_used) |last_used| {
            const now = std.time.timestamp();
            const hours_since_use = @divFloor(now - last_used, 3600);
            
            if (hours_since_use > 24) {
                std.debug.print("⚠️  Model {s} unused for {} hours\n", 
                    .{model.id, hours_since_use});
            }
        }
        
        // Log usage stats
        std.debug.print("Model {s}: {} uses, status: {}\n",
            .{model.id, model.use_count, model.health_status});
    }
}
```

## Testing

Run the test suite:

```bash
cd src/serviceCore/nLocalModels/shared
zig run test_model_registry.zig
```

Expected output:
```
================================================================================
🧪 Enhanced Model Registry Test Suite - Day 11
================================================================================

Test 1: Model Version Parsing
----------------------------------------
  ✓ Parse '1.2.3' -> 1.2.3
  ✓ Version comparison: 1.2.3 < 2.0.0
  ✅ Model version parsing tests passed

Test 2: Model Registry Initialization
----------------------------------------
  ✓ Registry initialized
  ✓ Model base path: vendor/layerModels
  ✓ Metadata path: vendor/layerData
  ✓ Initial model count: 0
  ✅ Registry initialization tests passed

... (more tests)

================================================================================
✅ All Tests Passed!
================================================================================
```

## Performance Characteristics

| Operation | Time Complexity | Space Complexity |
|-----------|----------------|------------------|
| `init()` | O(1) | O(1) |
| `register()` | O(1) average | O(n) |
| `get()` | O(1) average | O(1) |
| `discoverModels()` | O(n×m) | O(n) |
| `listModels()` | O(n) | O(n) |
| `toJson()` | O(n) | O(n) |

Where:
- n = number of models
- m = average files per model directory

## Best Practices

1. **Initialize Once**: Create registry at startup, reuse throughout app
2. **Auto-Discover**: Call `discoverModels()` on startup and periodically
3. **Track Health**: Update health status after inference operations
4. **Monitor Usage**: Use `markUsed()` to track model popularity
5. **Set Defaults**: Always set a default model for fallback
6. **Clean Up**: Call `deinit()` on shutdown to free resources

## Future Enhancements

### Planned for Day 12+
- Persistent metadata storage in vendor/layerData
- Model hot-swapping without restart
- A/B testing support
- Model performance metrics
- Automatic model updates
- Model grouping/namespaces

## Related Documentation

- [Day 10 - Week 2 Completion Report](DAY_10_WEEK2_COMPLETION_REPORT.md)
- [Model Scanner (Mojo)](../discovery/model_scanner.mojo)
- [LLM Integration](../orchestration/llm_integration/README.md)
- [Health Monitoring (Day 9)](DAY_09_HEALTH_MONITORING_REPORT.md)

## Support

For issues or questions:
- Check test suite: `test_model_registry.zig`
- Review source: `model_registry.zig`
- See integration examples in orchestration modules

---

**Day 11 Complete**: Enhanced Model Registry with multi-model support, versioning, and filesystem integration! 🎉
