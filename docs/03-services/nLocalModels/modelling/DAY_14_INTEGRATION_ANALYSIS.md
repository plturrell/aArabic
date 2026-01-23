# Day 14 Request Routing - Integration Analysis

## Overview

This document explains how **Day 14 Request Routing** integrates with the **orchestration** and **discovery** subsystems to provide intelligent, multi-model request routing for the nOpenAI server.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     Client Request                              │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  REQUEST ROUTER (Day 14)                        │
│                  [Zig Implementation]                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Routing Strategies:                                     │  │
│  │  • Round-robin      • Least-loaded                       │  │
│  │  • Cache-aware      • Quota-aware                        │  │
│  │  • Latency-based    • Affinity-based                     │  │
│  │  • Weighted-random  • A/B testing                        │  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────┬──────────────┬──────────────┬──────────────────────┘
            │              │              │
            │              │              │
    ┌───────▼─────┐  ┌────▼──────┐  ┌───▼──────────┐
    │   Model     │  │   Cache   │  │   Quota      │
    │  Registry   │  │  Manager  │  │   Manager    │
    │  (Zig)      │  │  (Zig)    │  │   (Zig)      │
    └───────┬─────┘  └───────────┘  └──────────────┘
            │
            │ Populated by
            │
    ┌───────▼──────────────────────────────────────────────────┐
    │           DISCOVERY SYSTEM                               │
    │           [Mojo Implementation]                          │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  Model Scanner (model_scanner.mojo)                │  │
    │  │  • Scans HuggingFace cache                         │  │
    │  │  • Scans Ollama models                             │  │
    │  │  • Scans local directories                         │  │
    │  │  • Extracts metadata (quantization, architecture)  │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
            │
            │ Feeds metadata to
            │
    ┌───────▼──────────────────────────────────────────────────┐
    │          ORCHESTRATION SYSTEM                            │
    │          [Mojo Implementation]                           │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  Tool Registry (tools/registry.mojo)               │  │
    │  │  • Registers model endpoints                       │  │
    │  │  • Maps capabilities to models                     │  │
    │  │  • Stores execution metadata                       │  │
    │  │  • Provides SIMD-optimized lookup                  │  │
    │  └────────────────────────────────────────────────────┘  │
    │                                                            │
    │  ┌────────────────────────────────────────────────────┐  │
    │  │  Job Control (job_control/job_control.mojo)        │  │
    │  │  • Manages model execution lifecycles              │  │
    │  │  • Coordinates multi-model workflows               │  │
    │  └────────────────────────────────────────────────────┘  │
    └──────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Discovery → Model Registry Integration

**Purpose**: Auto-discover and register available models

#### Discovery System (Mojo)
```mojo
// src/serviceCore/nLocalModels/discovery/model_scanner.mojo

struct ModelScanner:
    """Scans filesystem for GGUF models"""
    var models: List[ModelInfo]
    var scan_paths: List[String]
    
    fn scan(inout self) -> Int:
        """
        Scans:
        - ~/.cache/huggingface/hub
        - ~/.ollama/models
        - ./models
        - ~/models
        """
        # Discovers models and extracts:
        # - Architecture (llama, phi, qwen, gemma)
        # - Quantization (Q4_0, Q4_K_M, Q8_0, F16)
        # - Size (bytes)
        # - Path
```

#### Model Registry (Zig)
```zig
// src/serviceCore/nLocalModels/shared/model_registry.zig

pub const ModelRegistry = struct {
    models: std.StringHashMap(ModelConfig),
    model_base_path: []const u8,  // vendor/layerModels
    
    pub fn discoverModels(self: *ModelRegistry) !DiscoveryStats {
        // Scans model_base_path directory
        // Creates ModelConfig for each discovered model
        // Registers in internal HashMap for O(1) lookup
    }
    
    pub fn register(self: *ModelRegistry, config: ModelConfig) !void {
        // Stores model configuration
        // Tracks versions
        // Updates default if first model
    }
};
```

**Data Flow**:
1. **Discovery Scanner** → Scans filesystem for GGUF files
2. **Model Info Extraction** → Parses filenames/metadata
3. **Model Registry** → Stores configurations with health status
4. **Request Router** → Queries registry for available models

### 2. Model Registry → Request Router Integration

**Purpose**: Provide router with model availability and health information

#### Request Router Dependencies
```zig
// src/serviceCore/nLocalModels/inference/engine/routing/request_router.zig

pub const RequestRouter = struct {
    registry: ?*ModelRegistry = null,  // Model availability
    cache_manager: ?*MultiModelCacheManager = null,  // Cache stats
    quota_manager: ?*ResourceQuotaManager = null,  // Quota limits
    
    pub fn route(self: *RequestRouter, request: ...) !RoutingDecision {
        // 1. Check preferred model in registry
        // 2. Check affinity (sticky routing)
        // 3. Apply routing strategy:
        
        switch (self.config.strategy) {
            .least_loaded => {
                // Queries registry for all models
                // Checks health_status
                // Scores based on load
            },
            .cache_aware => {
                // Queries cache_manager for hit rates
                // Prefers models with high cache hits
            },
            .quota_aware => {
                // Queries quota_manager for availability
                // Avoids models near limits
            },
        }
    }
    
    fn getAvailableModels(self: *RequestRouter) ![][]const u8 {
        const all_models = try registry.listModels(allocator);
        
        // Filter by:
        // 1. Health status (registry.get().health_status == .healthy)
        // 2. Enabled flag (registry.get().enabled == true)
        // 3. Quota availability (quota_manager.checkQuota())
    }
};
```

#### Model Health Tracking
```zig
pub const ModelConfig = struct {
    health_status: HealthStatus = .unknown,
    last_used: ?i64 = null,
    use_count: u64 = 0,
    
    pub const HealthStatus = enum {
        unknown,
        healthy,
        degraded,
        unhealthy,
        loading,
    };
    
    pub fn markUsed(self: *ModelConfig) void {
        self.use_count += 1;
        self.last_used = std.time.timestamp();
    }
};
```

### 3. Orchestration → Request Router Integration

**Purpose**: Higher-level workflow orchestration and tool execution

#### Tool Registry (Mojo)
```mojo
// src/serviceCore/nLocalModels/orchestration/tools/registry.mojo

struct ToolRegistry:
    """Manages tool definitions and model endpoints"""
    var tools: Dict[String, ToolDefinition]
    var models: Dict[String, ModelDefinition]
    
    fn register_model(inout self, model: ModelDefinition):
        """
        Registers model as executable tool
        Example:
          ModelDefinition(
            name="shimmy_local_inference",
            endpoint="http://localhost:11435/v1/chat/completions",
            model_type="chat",
            model_name="nvidia/Orchestrator-8B"
          )
        """
```

#### Integration Flow

```
┌─────────────────────────────────────────────────────────┐
│  Orchestration Layer (Mojo)                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │  LLM Integration (llm_integration/)               │  │
│  │  • Receives user query                            │  │
│  │  │                                                 │  │
│  │  ▼                                                 │  │
│  │  Query Translator (query_translation/)            │  │
│  │  • Determines required capabilities               │  │
│  │  │                                                 │  │
│  │  ▼                                                 │  │
│  │  Tool Registry (tools/registry.mojo)              │  │
│  │  • find_tools_by_capability("code_analysis")      │  │
│  │  • Returns list of capable models                 │  │
│  └───────────────────────┬───────────────────────────┘  │
└────────────────────────────┼─────────────────────────────┘
                             │
                             ▼ Model selection request
┌─────────────────────────────────────────────────────────┐
│  Request Router (Zig)                                   │
│  • Receives capability requirements                     │
│  • Queries Model Registry for matching models           │
│  • Applies routing strategy                             │
│  • Returns RoutingDecision                              │
└─────────────────────────────────────────────────────────┘
```

### 4. Routing Strategy Examples

#### Strategy 1: Cache-Aware Routing
```zig
fn routeCacheAware(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
    const candidates = try self.scoreModels(estimated_tokens);
    
    // For each model:
    // 1. Query cache_manager.getModelStats()
    // 2. Calculate hit rate = hits / (hits + misses)
    // 3. Select model with highest hit rate
    
    var best = candidates[0];
    for (candidates[1..]) |candidate| {
        if (candidate.cache_hit_rate > best.cache_hit_rate) {
            best = candidate;
        }
    }
    
    return RoutingDecision{
        .model_id = best.model_id,
        .reason = "Cache-aware routing",
        .cache_score = best.cache_hit_rate,
    };
}
```

#### Strategy 2: Quota-Aware Routing
```zig
fn routeQuotaAware(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
    const candidates = try self.scoreModels(estimated_tokens);
    
    // For each model:
    // 1. Query quota_manager.getModelReport()
    // 2. Calculate availability = 100 - hourly_quota_used
    // 3. Select model with most quota remaining
    
    var best = candidates[0];
    for (candidates[1..]) |candidate| {
        if (candidate.quota_available > best.quota_available) {
            best = candidate;
        }
    }
    
    return RoutingDecision{
        .model_id = best.model_id,
        .reason = "Quota-aware routing",
        .quota_score = best.quota_available,
    };
}
```

#### Strategy 3: Least-Loaded Routing
```zig
fn routeLeastLoaded(self: *RequestRouter, estimated_tokens: u64) !RoutingDecision {
    const candidates = try self.scoreModels(estimated_tokens);
    
    // For each model:
    // 1. Get current load from quota_manager (RAM utilization)
    // 2. Select model with lowest load
    
    var best = candidates[0];
    for (candidates[1..]) |candidate| {
        if (candidate.load < best.load) {
            best = candidate;
        }
    }
    
    return RoutingDecision{
        .model_id = best.model_id,
        .reason = "Least loaded model",
        .load_score = 1.0 - best.load,
    };
}
```

## Complete Request Flow

### Scenario: User requests Arabic NLP processing

```
1. REQUEST ARRIVAL
   ┌────────────────────────────────────────┐
   │ POST /v1/chat/completions              │
   │ { model: "auto",                       │
   │   messages: [...],                     │
   │   capabilities: ["arabic_nlp"] }       │
   └────────────────────────────────────────┘
                    │
                    ▼
2. ORCHESTRATION LAYER
   ┌────────────────────────────────────────┐
   │ Tool Registry (Mojo)                   │
   │ • find_tools_by_capability("arabic_nlp")│
   │ • Returns: ["Llama-3.3-70B", "Qwen-2.5"]│
   └────────────────────────────────────────┘
                    │
                    ▼
3. REQUEST ROUTER
   ┌────────────────────────────────────────┐
   │ RequestRouter.route()                  │
   │ • Strategy: cache_aware                │
   │ • Queries Model Registry               │
   │   - Check health_status                │
   │   - Check enabled flag                 │
   │ • Queries Cache Manager                │
   │   - Llama-3.3-70B: 85% hit rate       │
   │   - Qwen-2.5: 62% hit rate            │
   │ • Decision: Llama-3.3-70B             │
   └────────────────────────────────────────┘
                    │
                    ▼
4. MODEL REGISTRY UPDATE
   ┌────────────────────────────────────────┐
   │ ModelConfig.markUsed()                 │
   │ • Increment use_count                  │
   │ • Update last_used timestamp           │
   └────────────────────────────────────────┘
                    │
                    ▼
5. EXECUTION
   ┌────────────────────────────────────────┐
   │ Model execution via selected endpoint  │
   │ • Load from cache if available         │
   │ • Update cache statistics              │
   │ • Update quota usage                   │
   └────────────────────────────────────────┘
```

## Key Integration Benefits

### 1. **Automatic Model Discovery**
- **Discovery Scanner** finds models without manual configuration
- **Model Registry** maintains up-to-date inventory
- **Request Router** always has current model list

### 2. **Intelligent Routing**
- **Cache Manager** provides hit rate statistics
- **Quota Manager** provides utilization data
- **Router** makes data-driven decisions

### 3. **Health-Aware Routing**
- **Model Registry** tracks health status
- **Router** skips unhealthy models
- **Graceful degradation** when models fail

### 4. **Orchestration Flexibility**
- **Tool Registry** provides high-level abstractions
- **Router** handles low-level model selection
- **Clean separation** of concerns

### 5. **Performance Optimization**
- **Mojo Discovery**: Fast filesystem scanning
- **Zig Router**: Zero-allocation routing decisions
- **O(1) lookups**: HashMap-based model registry

## Configuration Example

### Discovery Configuration
```mojo
// Auto-scan standard paths
var scanner = ModelScanner()
scanner.add_scan_path("~/models")
scanner.add_scan_path("./vendor/layerModels")
var count = scanner.scan()
```

### Registry Configuration
```zig
// Initialize registry with discovered models
var registry = try ModelRegistry.init(
    allocator,
    "vendor/layerModels",
    "vendor/layerData"
);

// Auto-discover models
const stats = try registry.discoverModels();
// stats.models_found, stats.models_added
```

### Router Configuration
```zig
// Initialize router with strategy
var router = try RequestRouter.init(allocator, .{
    .strategy = .cache_aware,
    .enable_health_checks = true,
    .enable_quota_checks = true,
    .enable_cache_optimization = true,
});

// Connect integrations
router.setRegistry(registry);
router.setCacheManager(cache_manager);
router.setQuotaManager(quota_manager);

// Route request
const decision = try router.route(.{
    .estimated_tokens = 100,
    .required_capabilities = &[_][]const u8{"arabic_nlp"},
});
```

## API Boundaries

### Discovery → Registry
```mojo
// Discovery provides
struct ModelInfo:
    var name: String
    var path: String
    var size_bytes: Int
    var quantization: String
    var architecture: String
```

### Registry → Router
```zig
// Registry provides
pub const ModelConfig = struct {
    id: []const u8,
    path: []const u8,
    health_status: HealthStatus,
    enabled: bool,
    use_count: u64,
};

pub fn listModels() ![][]const u8;
pub fn get(id: []const u8) ?*const ModelConfig;
pub fn getHealthyModels() ![][]const u8;
```

### Router → Orchestration
```zig
// Router provides
pub const RoutingDecision = struct {
    model_id: []const u8,
    reason: []const u8,
    total_score: f32,
};

pub fn route(request: RequestParams) !RoutingDecision;
```

## Future Enhancements

### 1. **Bi-directional Integration**
- Router feedback to Discovery (usage patterns)
- Dynamic re-scanning based on load
- Predictive model loading

### 2. **Advanced Orchestration**
- Multi-model ensemble routing
- Capability-based auto-scaling
- Cross-model result aggregation

### 3. **Performance Optimization**
- Shared memory between Mojo/Zig
- Direct FFI calls avoiding serialization
- Zero-copy model metadata passing

### 4. **Monitoring Integration**
- Route decision telemetry
- Model health metrics export
- Discovery audit logs

## Conclusion

Day 14 Request Routing creates a cohesive system by:

1. **Leveraging Discovery** for automatic model detection and metadata extraction
2. **Using Model Registry** as the source of truth for available models and health
3. **Integrating with Orchestration** for high-level workflow coordination
4. **Providing intelligent routing** based on cache, quota, and load metrics

The multi-language architecture (Mojo for discovery/orchestration, Zig for routing/execution) enables both high-level flexibility and low-level performance optimization.
