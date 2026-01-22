# Multi-Model Cache API Documentation - Day 12

**Version:** 1.0.0  
**Date:** 2026-01-19  
**Status:** ✅ Production Ready

## Overview

The Multi-Model Cache Manager coordinates tiered KV caches across multiple models, providing fair resource allocation, intelligent eviction policies, and per-model metrics tracking.

## Features

- ✅ **Multi-Model Coordination** - Manage caches for unlimited models
- ✅ **Fair Resource Allocation** - 4 allocation strategies
- ✅ **Cross-Model Eviction** - 4 global eviction policies
- ✅ **Per-Model Namespacing** - Isolated caches per model
- ✅ **Usage Tracking** - Comprehensive per-model and global metrics
- ✅ **Thread-Safe** - Mutex-protected operations
- ✅ **Integration Ready** - Works with Day 11 Model Registry

## Architecture

```
┌─────────────────────────────────────────────────────┐
│      Multi-Model Cache Manager (Day 12)            │
│  ┌──────────────────────────────────────────────┐  │
│  │ StringHashMap<ModelCacheState>              │  │
│  │                                             │  │
│  │  model-1 → TieredKVCache (RAM + SSD)       │  │
│  │  model-2 → TieredKVCache (RAM + SSD)       │  │
│  │  model-3 → TieredKVCache (RAM + SSD)       │  │
│  │  ...                                        │  │
│  └──────────────────────────────────────────────┘  │
│                                                     │
│  Global Resource Management:                       │
│  - Total RAM: 4GB (fair/priority allocation)       │
│  - Total SSD: 32GB (fair/priority allocation)      │
│  - Cross-model eviction (LRU/LFU/size/round-robin) │
│  - Global statistics & monitoring                  │
│                                                     │
└─────────────────────────────────────────────────────┘
         │                    │
         ▼                    ▼
    [Model Registry]    [Existing TieredKVCache]
     (Day 11)           (Days 1-5 optimized)
```

## Core Components

### 1. MultiModelCacheConfig

Global configuration for the multi-model cache system:

```zig
pub const MultiModelCacheConfig = struct {
    // Resource limits
    total_ram_mb: u64 = 4096,              // Total RAM for all models
    total_ssd_mb: u64 = 32768,             // Total SSD for all models (32GB)
    
    // Allocation strategy
    allocation_strategy: AllocationStrategy = .fair_share,
    
    // Minimum per model
    min_ram_per_model_mb: u64 = 256,       // At least 256MB RAM per model
    min_ssd_per_model_mb: u64 = 1024,      // At least 1GB SSD per model
    
    // Eviction policy
    global_eviction_policy: GlobalEvictionPolicy = .least_recently_used_model,
    
    // Paths
    ssd_base_path: []const u8 = "/tmp/shimmy_multi_model_cache",
    
    // Performance
    hot_tokens_per_model: u32 = 2048,      // Hot tokens in RAM per model
    enable_cross_model_sharing: bool = false,  // Future feature
};
```

### 2. Allocation Strategies

Four strategies for distributing resources:

```zig
pub const AllocationStrategy = enum {
    fair_share,           // Equal resources per model
    proportional,         // Based on model size/usage
    priority_based,       // Based on model priority (1-10)
    dynamic,             // Adapt based on usage patterns
};
```

**Fair Share Example:**
- 4 models, 4096MB total RAM → 1024MB per model
- 4 models, 32768MB total SSD → 8192MB per model

**Priority-Based Example:**
- High priority (10/10): 2048MB RAM
- Medium priority (5/10): 1024MB RAM
- Low priority (1/10): 256MB RAM (minimum)

### 3. Global Eviction Policies

Four policies for choosing which model to evict from:

```zig
pub const GlobalEvictionPolicy = enum {
    least_recently_used_model,   // Evict from oldest accessed model
    least_frequently_used_model, // Evict from least used model
    smallest_model_first,        // Evict from model with least RAM
    round_robin,                // Rotate eviction fairly
};
```

### 4. ModelCacheState

Per-model cache state tracking:

```zig
pub const ModelCacheState = struct {
    model_id: []const u8,
    cache: *TieredKVCache,               // Individual tiered cache
    
    // Resource allocation
    allocated_ram_mb: u64,
    allocated_ssd_mb: u64,
    
    // Usage tracking
    last_access_time: i64,               // Unix timestamp (ms)
    access_count: u64,                   // Total accesses
    total_tokens_processed: u64,         // Total tokens
    
    // Configuration
    priority: u8 = 5,                    // 1-10 priority
    is_active: bool = true,
    is_preloaded: bool = false,
    
    pub fn markAccess(self: *ModelCacheState) void
    pub fn getUsageScore(self: *const ModelCacheState) f32
};
```

## API Reference

### Initialization

```zig
const allocator = std.heap.page_allocator;

const config = MultiModelCacheConfig{
    .total_ram_mb = 4096,
    .total_ssd_mb = 32768,
    .allocation_strategy = .fair_share,
    .global_eviction_policy = .least_recently_used_model,
    .hot_tokens_per_model = 2048,
};

var manager = try MultiModelCacheManager.init(allocator, config);
defer manager.deinit();
```

### Model Registration

Register a model and allocate its cache:

```zig
try manager.registerModel("llama-3.2-1b", .{
    .n_layers = 16,
    .n_heads = 16,
    .head_dim = 64,
    .max_seq_len = 4096,
    .priority = 8,  // High priority (1-10 scale)
});
```

### Getting Model Cache

Retrieve cache for a specific model:

```zig
const cache = try manager.getModelCache("llama-3.2-1b");

// Use the cache (same API as TieredKVCache)
try cache.store(layer, keys, values);
try cache.getKeys(layer, start_pos, end_pos, dest);
```

### Model Unregistration

Remove a model and free its resources:

```zig
try manager.unregisterModel("llama-3.2-1b");
// Resources automatically freed
```

### Statistics

**Per-Model Statistics:**
```zig
const stats = try manager.getModelStats("llama-3.2-1b");

std.debug.print("Model: {s}\n", .{stats.model_id});
std.debug.print("RAM: {d} MB\n", .{stats.allocated_ram_mb});
std.debug.print("SSD: {d} MB\n", .{stats.allocated_ssd_mb});
std.debug.print("Access count: {d}\n", .{stats.access_count});
std.debug.print("Tokens processed: {d}\n", .{stats.tokens_processed});
std.debug.print("Cache hits: {d}\n", .{stats.cache_hits});
std.debug.print("Cache misses: {d}\n", .{stats.cache_misses});
std.debug.print("Usage score: {d:.2}\n", .{stats.usage_score});
```

**Global Statistics:**
```zig
const global = manager.getGlobalStats();

std.debug.print("Total models: {d}\n", .{global.total_models});
std.debug.print("Active models: {d}\n", .{global.active_models});
std.debug.print("Total RAM: {d}/{d} MB\n", .{
    global.total_ram_used_mb,
    manager.config.total_ram_mb,
});
std.debug.print("Total SSD: {d}/{d} MB\n", .{
    global.total_ssd_used_mb,
    manager.config.total_ssd_mb,
});
std.debug.print("Cross-model evictions: {d}\n", .{global.cross_model_evictions});
```

### Listing Models

```zig
const models = try manager.listModels(allocator);
defer {
    for (models) |model| allocator.free(model);
    allocator.free(models);
}

for (models) |model_id| {
    std.debug.print("Model: {s}\n", .{model_id});
}
```

### Status Display

```zig
manager.printStatus();
// Outputs comprehensive status for all models
```

### Cross-Model Eviction

Manually trigger global eviction:

```zig
try manager.performGlobalEviction();
// Evicts from model selected by global_eviction_policy
```

## Integration Examples

### Example 1: Integration with Model Registry (Day 11)

```zig
const ModelRegistry = @import("../../shared/model_registry.zig").ModelRegistry;

pub fn initializeMultiModelCache(
    allocator: std.mem.Allocator,
    registry: *ModelRegistry,
) !*MultiModelCacheManager {
    // Create cache manager
    const cache_config = MultiModelCacheConfig{
        .total_ram_mb = 8192,  // 8GB total
        .total_ssd_mb = 65536, // 64GB total
        .allocation_strategy = .priority_based,
    };
    
    var manager = try MultiModelCacheManager.init(allocator, cache_config);
    errdefer manager.deinit();
    
    // Register caches for all healthy models in registry
    const model_list = try registry.listModels(allocator);
    defer {
        for (model_list) |model| allocator.free(model);
        allocator.free(model_list);
    }
    
    for (model_list) |model_id| {
        if (registry.get(model_id)) |model_config| {
            if (model_config.health_status == .healthy) {
                // Extract model architecture details
                // (In real code, parse from model metadata)
                try manager.registerModel(model_id, .{
                    .n_layers = 16,  // Extract from model
                    .n_heads = 16,   // Extract from model
                    .head_dim = 64,  // Extract from model
                    .max_seq_len = 4096,
                    .priority = 5,
                });
            }
        }
    }
    
    return manager;
}
```

### Example 2: Model Selection with Cache

```zig
pub fn selectModelAndGetCache(
    registry: *ModelRegistry,
    cache_manager: *MultiModelCacheManager,
    task_type: []const u8,
) !struct { model_id: []const u8, cache: *TieredKVCache } {
    // Select model from registry (Day 11)
    const healthy_models = try registry.getHealthyModels(allocator);
    defer allocator.free(healthy_models);
    
    // Pick first healthy model (or implement smarter selection)
    const model_id = healthy_models[0];
    
    // Get cache for selected model
    const cache = try cache_manager.getModelCache(model_id);
    
    return .{
        .model_id = model_id,
        .cache = cache,
    };
}
```

### Example 3: Request Routing with Cache Awareness

```zig
pub fn routeRequest(
    request: Request,
    cache_manager: *MultiModelCacheManager,
) !*TieredKVCache {
    // Get cache statistics for all models
    const models = try cache_manager.listModels(allocator);
    defer {
        for (models) |m| allocator.free(m);
        allocator.free(models);
    }
    
    // Find model with best cache hit rate
    var best_model: ?[]const u8 = null;
    var best_score: f32 = 0.0;
    
    for (models) |model_id| {
        const stats = try cache_manager.getModelStats(model_id);
        const hit_rate = if (stats.cache_hits + stats.cache_misses > 0)
            @as(f32, @floatFromInt(stats.cache_hits)) /
            @as(f32, @floatFromInt(stats.cache_hits + stats.cache_misses))
        else
            0.0;
        
        if (hit_rate > best_score) {
            best_score = hit_rate;
            best_model = model_id;
        }
    }
    
    if (best_model) |id| {
        return try cache_manager.getModelCache(id);
    }
    
    return error.NoSuitableModel;
}
```

## Performance Characteristics

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| `init()` | O(1) | Constant time |
| `registerModel()` | O(1) | HashMap insert |
| `getModelCache()` | O(1) | HashMap lookup |
| `performGlobalEviction()` | O(n) | Scan all models |
| `listModels()` | O(n) | Iterate all models |
| `getModelStats()` | O(1) | HashMap lookup |

### Space Complexity

| Structure | Complexity | Notes |
|-----------|------------|-------|
| ModelCacheState | O(1) | Fixed size per model |
| Manager | O(n) | n models registered |
| Individual Caches | O(m) | m = max_seq_len per model |

### Resource Usage

**Per Model:**
- RAM: 256MB - 2048MB (configurable)
- SSD: 1GB - 16GB (configurable)
- Overhead: ~1KB (ModelCacheState tracking)

**Global (6 models):**
- RAM: 4GB total (fair share: 682MB each)
- SSD: 32GB total (fair share: 5.3GB each)
- Overhead: ~6KB (tracking + HashMap)

## Allocation Strategies Explained

### 1. Fair Share (Default)

**Algorithm:**
```zig
ram_per_model = total_ram / num_models
ssd_per_model = total_ssd / num_models
```

**Example (4 models, 4GB RAM, 32GB SSD):**
- Model 1: 1GB RAM, 8GB SSD
- Model 2: 1GB RAM, 8GB SSD
- Model 3: 1GB RAM, 8GB SSD
- Model 4: 1GB RAM, 8GB SSD

**Best For:** Equal-sized models, balanced workloads

### 2. Proportional

**Algorithm:**
```zig
factor = model_size / total_size
ram_for_model = total_ram * factor
```

**Example (4GB RAM):**
- 7B model: 2GB RAM (50%)
- 3B model: 1GB RAM (25%)
- 1B model: 0.5GB RAM (12.5%)
- 0.5B model: 0.5GB RAM (12.5%)

**Best For:** Mixed model sizes

### 3. Priority-Based

**Algorithm:**
```zig
factor = priority / 10.0  // 1-10 scale
ram_for_model = (total_ram * factor) / num_models
```

**Example (priority 10, 5, 1 with 4GB RAM, 3 models):**
- P10: 1.3GB RAM (higher allocation)
- P5: 0.65GB RAM (medium allocation)
- P1: 0.13GB RAM + 143MB minimum = 256MB

**Best For:** Production/staging mix, VIP users

### 4. Dynamic

**Algorithm:** Starts with fair share, adapts based on usage

**Adaptation Triggers:**
- High usage model → increase allocation
- Low usage model → decrease allocation
- Rebalance every N requests

**Best For:** Unpredictable workloads

## Global Eviction Policies Explained

### 1. Least Recently Used Model (LRU)

Evicts from the model with oldest `last_access_time`:

```zig
// Finds model with oldest access time
oldest_model = model with min(last_access_time)
```

**Best For:** Time-sensitive applications

### 2. Least Frequently Used Model (LFU)

Evicts from model with lowest `access_count`:

```zig
// Finds model with lowest access count
least_used_model = model with min(access_count)
```

**Best For:** Usage-based optimization

### 3. Smallest Model First

Evicts from model with least RAM allocated:

```zig
// Finds model with smallest RAM footprint
smallest_model = model with min(allocated_ram_mb)
```

**Best For:** Protecting large critical models

### 4. Round Robin

Evicts fairly across all models in rotation:

```zig
// Cycles through models in order
next_model = models[round_robin_index++ % num_models]
```

**Best For:** Fair treatment, preventing starvation

## Usage Patterns

### Pattern 1: Static Multi-Model Serving

```zig
// Initialize once at startup
var manager = try MultiModelCacheManager.init(allocator, config);
defer manager.deinit();

// Register all models
for (model_ids) |id| {
    try manager.registerModel(id, model_config);
}

// Serve requests
while (true) {
    const request = try receiveRequest();
    const cache = try manager.getModelCache(request.model_id);
    const response = try runInference(cache, request);
    try sendResponse(response);
}
```

### Pattern 2: Dynamic Model Loading

```zig
// Start with empty manager
var manager = try MultiModelCacheManager.init(allocator, config);
defer manager.deinit();

// Register models on-demand
fn handleRequest(manager: *MultiModelCacheManager, request: Request) !Response {
    // Check if model cache exists
    const cache = manager.getModelCache(request.model_id) catch |err| {
        if (err == error.ModelCacheNotFound) {
            // Register model on first use
            try manager.registerModel(request.model_id, request.model_config);
            return try manager.getModelCache(request.model_id);
        }
        return err;
    };
    
    return try runInference(cache, request);
}
```

### Pattern 3: Priority-Based Serving

```zig
const config = MultiModelCacheConfig{
    .allocation_strategy = .priority_based,
    .global_eviction_policy = .smallest_model_first,
};

var manager = try MultiModelCacheManager.init(allocator, config);

// Register production models with high priority
try manager.registerModel("prod-model", .{
    .n_layers = 32,
    .n_heads = 32,
    .head_dim = 128,
    .max_seq_len = 8192,
    .priority = 10,  // Highest priority
});

// Register staging models with lower priority
try manager.registerModel("staging-model", .{
    .n_layers = 16,
    .n_heads = 16,
    .head_dim = 64,
    .max_seq_len = 4096,
    .priority = 3,   // Lower priority
});

// Production model gets more resources automatically
```

## Monitoring & Observability

### Integration with Days 6-9 Observability Stack

```zig
// Structured logging (Day 6)
log.info("Cache operation", .{
    .operation = "getModelCache",
    .model_id = model_id,
    .access_count = stats.access_count,
    .cache_hit_rate = hit_rate,
});

// Health checks (Day 9)
pub fn checkCacheManagerHealth(manager: *MultiModelCacheManager) HealthStatus {
    const stats = manager.getGlobalStats();
    const ram_usage = @as(f32, @floatFromInt(stats.total_ram_used_mb)) /
                      @as(f32, @floatFromInt(manager.config.total_ram_mb));
    
    if (ram_usage > 0.95) return .unhealthy;
    if (ram_usage > 0.85) return .degraded;
    return .healthy;
}

// Metrics export (Day 9)
pub fn exportPrometheusMetrics(manager: *MultiModelCacheManager) ![]u8 {
    var buffer = std.ArrayList(u8).init(allocator);
    
    // Global metrics
    const stats = manager.getGlobalStats();
    try buffer.writer().print("cache_total_models {d}\n", .{stats.total_models});
    try buffer.writer().print("cache_active_models {d}\n", .{stats.active_models});
    try buffer.writer().print("cache_ram_used_mb {d}\n", .{stats.total_ram_used_mb});
    try buffer.writer().print("cache_ssd_used_mb {d}\n", .{stats.total_ssd_used_mb});
    
    // Per-model metrics
    const models = try manager.listModels(allocator);
    defer allocator.free(models);
    
    for (models) |model_id| {
        const model_stats = try manager.getModelStats(model_id);
        try buffer.writer().print(
            "cache_model_hits{{model=\"{s}\"}} {d}\n",
            .{ model_id, model_stats.cache_hits },
        );
    }
    
    return buffer.toOwnedSlice();
}
```

## Testing

Run the test suite:

```bash
cd src/serviceCore/nOpenaiServer/inference/engine/tiering
zig run test_multi_model_cache.zig
```

Expected output:
```
================================================================================
🧪 Multi-Model Cache Manager Test Suite - Day 12
================================================================================

Test 1: Manager Initialization
----------------------------------------
  ✓ Manager initialized successfully
  ✓ Initial state verified
  ✅ Manager initialization tests passed

Test 2: Fair Share Allocation
----------------------------------------
  ✓ Registered 4 models
  ✓ Total RAM allocated: 4096 MB
  ✓ Total SSD allocated: 32768 MB
  ✓ llama-1b: 1024 MB RAM, 8192 MB SSD
  ✓ phi-2: 1024 MB RAM, 8192 MB SSD
  ✓ qwen-0.5b: 1024 MB RAM, 8192 MB SSD
  ✓ gemma-270m: 1024 MB RAM, 8192 MB SSD
  ✅ Fair share allocation tests passed

... (8 more tests)

================================================================================
✅ All Tests Passed! (10/10)
================================================================================
```

## Best Practices

1. **Initialize Early**: Create manager at system startup
2. **Register Upfront**: Register all known models during initialization
3. **Monitor Resources**: Track global RAM/SSD usage
4. **Choose Policy Wisely**: Match eviction policy to workload
5. **Set Priorities**: Use priority-based allocation for mixed environments
6. **Track Metrics**: Monitor per-model cache efficiency
7. **Clean Up**: Unregister unused models to free resources

## Comparison: Single vs Multi-Model

### Before (Days 1-11): Single Model

```zig
// One cache per process
var cache = try TieredKVCache.init(allocator, config);
defer cache.deinit();

// No model isolation
// No fair allocation
// Manual resource management
```

### After (Day 12): Multi-Model

```zig
// One manager for all models
var manager = try MultiModelCacheManager.init(allocator, config);
defer manager.deinit();

// Automatic model isolation
// Fair resource allocation
// Intelligent eviction
// Per-model metrics
```

## Future Enhancements

### Planned for Day 13+

1. **Dynamic Reallocation**
   - Automatic resource adjustment based on usage
   - Shrink underused models, grow heavily-used models

2. **Cross-Model Cache Sharing**
   - Share common prompt prefixes
   - Reference-counted cache entries
   - 30%+ speedup for similar prompts

3. **Advanced Eviction**
   - Machine learning-based eviction prediction
   - Workload-aware policies
   - Cost-based eviction (evict high-cost-to-reload last)

4. **Cache Warming**
   - Pre-populate caches from SSD on startup
   - Predict needed caches based on schedule
   - Reduce cold starts

## Related Documentation

- [Day 11 - Model Registry](DAY_11_MODEL_REGISTRY_REPORT.md)
- [Day 10 - Week 2 Completion](DAY_10_WEEK2_COMPLETION_REPORT.md)
- [Days 1-5 - Tiered KV Cache](DAY_05_COMPLETION_REPORT.md)
- [Day 3 - Adaptive Eviction](DAY_03_EVICTION_REPORT.md)

---

**Day 12 Complete**: Multi-Model Cache with fair allocation and intelligent eviction! 🎉
