# nLocalModels Orchestration System

**Centralized Model Selection & Routing Engine**

## Overview

The nLocalModels orchestration system is the **single source of truth** for intelligent model selection and routing across the entire arabic_folder platform. Both nFlow (workflow layer) and nLocalModels (inference layer) use this centralized system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 Orchestration Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  Centralized System (nLocalModels/orchestration)            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  - model_selector.zig (core engine)                 │   │
│  │  - catalog/task_categories.json (9 categories)      │   │
│  │  - catalog/schema_*.mojo (catalog management)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                             │                                 │
│                             │ (shared)                        │
│              ┌──────────────┴──────────────┐                │
│              │                              │                │
│   ┌──────────▼────────────┐   ┌───────────▼──────────┐    │
│   │ nFlow Layer           │   │ nLocalModels Layer   │    │
│   │ (Workflow Engine)     │   │ (Inference Engine)   │    │
│   │                       │   │                      │    │
│   │ - Integration module  │   │ - Direct access     │    │
│   │ - LLM nodes          │   │ - Runtime routing   │    │
│   │ - Task orchestration │   │ - Load balancing    │    │
│   └───────────────────────┘   └──────────────────────┘    │
│                                                               │
│  Shared Resources                                            │
│  - vendor/layerModels/MODEL_REGISTRY.json (7 models)        │
│  - catalog/task_categories.json (9 categories)              │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### Core Module: model_selector.zig

**Location:** `src/serviceCore/nLocalModels/orchestration/model_selector.zig`

Intelligent model selection engine that:
- Loads MODEL_REGISTRY.json (7 models with full metadata)
- Loads task_categories.json (9 categories with benchmarks)
- Applies GPU memory constraints (T4, A100-40GB, A100-80GB)
- Filters by agent type (inference, tool, orchestrator)
- Scores models (base 50 + bonuses up to 50)
- Returns optimal selection with reasoning

### Catalog System

**Location:** `src/serviceCore/nLocalModels/orchestration/catalog/`

- **task_categories.json** - 9 task categories with 19 benchmarks
- **schema_registry.mojo** - Mojo integration for catalog management
- **schema_loader.mojo** - Dynamic schema loading
- **schema_introspector.mojo** - Schema analysis

### Integration Points

#### 1. nFlow Integration
**Location:** `src/serviceCore/nFlow/orchestration/nLocalModels_integration.zig`

Provides nFlow workflows with access to centralized orchestration:
```zig
// Import nLocalModels orchestration
const nLocalModelsOrch = @import("../../orchestration/nLocalModels_integration.zig");
const ModelSelector = nLocalModelsOrch.ModelSelector;

// Initialize with default paths
const selector = try nLocalModelsOrch.initDefault(allocator);

// Select for workflow
const result = try nLocalModelsOrch.selectForWorkflow(
    selector,
    "code",
    gpu_limit_mb,
);
```

#### 2. nLocalModels Direct Access

Native access from nLocalModels services:
```zig
const ModelSelector = @import("orchestration/model_selector.zig").ModelSelector;

const selector = try ModelSelector.init(
    allocator,
    "vendor/layerModels/MODEL_REGISTRY.json",
    "src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json",
);
```

## Usage Examples

### Example 1: Code Generation with T4 GPU

```zig
const selector = try ModelSelector.init(...);
try selector.loadRegistry();
try selector.loadCategories();

const constraints = SelectionConstraints{
    .max_gpu_memory_mb = 14 * 1024, // T4 (14GB usable)
    .required_agent_type = "inference",
};

var result = try selector.selectModel("code", constraints);
defer result.deinit(allocator);

// Result: google-gemma-3-270m-it (score: 70.0)
// Reason: Small model bonus (20) + base score (50)
```

### Example 2: Translation with Tool Capability

```zig
const constraints = SelectionConstraints{
    .max_gpu_memory_mb = 16 * 1024,
    .required_agent_type = "tool",
};

var result = try selector.selectModel("relational", constraints);
// Result: HY-MT1.5-7B (inference + tool)
```

### Example 3: Preferred Model

```zig
const preferred = [_][]const u8{"microsoft-phi-2"};
const constraints = SelectionConstraints{
    .max_gpu_memory_mb = 14 * 1024,
    .preferred_models = &preferred,
};

var result = try selector.selectModel("code", constraints);
// Result: microsoft-phi-2 (score: 90.0 with preference bonus)
```

## Task Categories

| Category | Models | Agent Types | Use Cases |
|----------|--------|-------------|-----------|
| **code** | 3 | inference | Programming, debugging |
| **relational** | 2 | inference, tool | SQL, translation |
| math | 0 | inference | Calculations, proofs |
| reasoning | 0 | inference, orchestrator | Logic, analysis |
| summarization | 0 | inference, orchestrator | Document summaries |
| time_series | 0 | inference | Forecasting |
| graph | 0 | inference, tool | Knowledge graphs |
| vector_search | 0 | inference, tool | RAG, embeddings |
| ocr_extraction | 0 | inference, tool | OCR, forms |

## Model Registry

### Current Models (7 total)

| Model | Size | GPU | Categories | Agent Types |
|-------|------|-----|------------|-------------|
| google-gemma-3-270m-it | 540MB | 2GB | code | inference |
| LFM2.5-1.2B-Instruct-GGUF | 1.2GB | 4GB | - | inference |
| HY-MT1.5-7B | 4.2GB | 8GB | relational | inference, tool |
| microsoft-phi-2 | 5.2GB | 6GB | code | inference |
| deepseek-coder-33b-instruct | 19GB | 22GB | code | inference |
| Llama-3.3-70B-Instruct | 43GB | 48GB | - | inference |
| translategemma-27b-it-GGUF | 16.6GB | 20GB | relational, translation | inference |

## Performance

### Selection Time (validated)
- **Mean:** 0.002ms (2 microseconds)
- **P95:** 0.004ms
- **P99:** 0.046ms
- **Throughput:** 100,000+ selections/second

### Consistency
- **Code category:** 100% (deterministic)
- **Relational category:** 100% (deterministic)
- **Algorithm:** No randomness, fully reproducible

## Integration with nFlow

nFlow workflows use the nLocalModels orchestration via integration module:

```zig
// In nFlow LLM nodes
const nLocalModelsOrch = @import("../../orchestration/nLocalModels_integration.zig");

// LLM node with auto-selection
const node = try LLMChatNode.init(
    allocator,
    "chat1",
    "Code Helper",
    null, // No explicit model
    "code", // Task category - auto-selects from nLocalModels
    0.7,
    1000,
    system_prompt,
    service_config,
);

node.setModelSelector(selector); // Uses nLocalModels orchestration
```

## Integration with nLocalModels Services

Direct integration in nLocalModels inference services:

```zig
// In nLocalModels inference modules
const ModelSelector = @import("orchestration/model_selector.zig").ModelSelector;

pub fn handleInferenceRequest(
    allocator: Allocator,
    task_category: []const u8,
) ![]const u8 {
    const selector = try ModelSelector.init(...);
    defer selector.deinit();
    
    try selector.loadRegistry();
    try selector.loadCategories();
    
    const result = try selector.selectModel(task_category, constraints);
    defer result.deinit(allocator);
    
    return result.model.name;
}
```

## Scoring System

| Factor | Points | Condition |
|--------|--------|-----------|
| Base Score | 50 | All models |
| Small Model | +20 | < 4GB GPU |
| Medium Model | +10 | 4-8GB GPU |
| Preferred | +30 | In preference list |
| **Future:** Benchmark | +0-50 | Based on scores |

## Testing

### Run Tests

```bash
# Python validation
python3 scripts/orchestration/benchmark_routing_performance.py --iterations 1000

# Zig integration tests
zig test tests/orchestration/test_model_selection_integration.zig

# Benchmark validator
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json --report
```

### Test Coverage
✅ Registry loading  
✅ Category loading  
✅ GPU constraint filtering  
✅ Agent type filtering  
✅ Model preferences  
✅ Model exclusion  
✅ GPU memory parsing  
✅ Fallback handling  
✅ Scoring system  
✅ Selection consistency  

## Maintenance

### Update Model Metadata

```bash
# Re-enrich from HuggingFace
python3 scripts/models/hf_model_card_extractor.py vendor/layerModels/MODEL_REGISTRY.json --verbose
```

### Add New Model

1. Add entry to `vendor/layerModels/MODEL_REGISTRY.json`
2. Run enrichment script
3. Update `task_categories.json` model assignments
4. Validate

### Add New Category

1. Add to `catalog/task_categories.json`
2. Define benchmarks
3. Assign models
4. Update documentation

## Migration from nOpenaiServer

The orchestration system was migrated from nOpenaiServer to nLocalModels to:
- **Centralize** model selection logic
- **Eliminate duplication** between layers
- **Improve maintainability** with single source of truth
- **Enable** direct nLocalModels integration

### Migration Summary

| Component | Old Location | New Location | Status |
|-----------|-------------|--------------|--------|
| model_selector.zig | nFlow/orchestration | **nLocalModels/orchestration** | ✅ Migrated |
| task_categories.json | nOpenaiServer | **nLocalModels/orchestration/catalog** | ✅ Migrated |
| Integration | Direct in nFlow | **nFlow→nLocalModels** | ✅ Created |

## Documentation

- **Architecture:** [docs/01-architecture/MODEL_ORCHESTRATION_MAPPING.md](../../../docs/01-architecture/MODEL_ORCHESTRATION_MAPPING.md)
- **Validation Report:** [docs/08-reports/validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md](../../../docs/08-reports/validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md)
- **nFlow README:** [src/serviceCore/nFlow/orchestration/README.md](../../nFlow/orchestration/README.md)

## Support

- **Issues:** https://github.com/plturrell/aArabic/issues
- **Discussions:** Use GitHub Discussions for questions

## License

Part of the arabic_folder project.
