# Model Orchestration System

Intelligent model selection and routing system for nFlow workflow engine.

## Overview

The Model Orchestration System provides intelligent, constraint-aware model selection based on task categories, enabling efficient multi-model workflows with automatic routing.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         Model Orchestration Flow                     │
├─────────────────────────────────────────────────────┤
│                                                       │
│  1. Task Request → task_category: "code"            │
│                                                       │
│  2. Load Configurations                              │
│     ├─ MODEL_REGISTRY.json (7 models)               │
│     └─ task_categories.json (9 categories)          │
│                                                       │
│  3. Apply Constraints                                │
│     ├─ GPU Memory: 14GB (T4)                        │
│     ├─ Agent Type: inference                        │
│     └─ Preferred/Excluded models                    │
│                                                       │
│  4. Score & Select                                   │
│     ├─ Base score: 50.0                             │
│     ├─ Small model bonus: +20.0                     │
│     ├─ Preference bonus: +30.0                      │
│     └─ Best score: 70.0                             │
│                                                       │
│  5. Execute with Selected Model                      │
│     └─ google-gemma-3-270m-it                       │
│                                                       │
└─────────────────────────────────────────────────────┘
```

## Components

### 1. ModelSelector (`model_selector.zig`)

Core orchestration engine that:
- Loads MODEL_REGISTRY.json and task_categories.json
- Applies constraint-based filtering
- Scores models based on multiple factors
- Returns optimal model with reasoning

**Key Features:**
- GPU memory constraint handling
- Agent type filtering (inference, tool, orchestrator)
- Preferred/excluded model lists
- Fallback strategies
- Scoring system with configurable weights

### 2. Enhanced LLM Nodes (`../nodes/llm/llm_nodes.zig`)

LLM integration nodes with intelligent routing:
- **LLMChatNode** - Chat completion with auto-selection
- **LLMEmbedNode** - Text embedding generation
- **PromptTemplateNode** - Template-based prompts
- **ResponseParserNode** - Response parsing & validation

**New Capabilities:**
- Task category-based auto-selection
- Explicit model override option
- Selection metadata tracking
- Backward compatible API

### 3. Configuration Files

#### task_categories.json
Defines 9 orchestration categories with:
- Category descriptions
- Associated benchmarks
- Model assignments
- GPU routing rules

#### MODEL_REGISTRY.json
7 models with comprehensive metadata:
- HuggingFace metadata (downloads, likes, tags)
- Orchestration categories
- Agent types
- GPU memory requirements
- Specifications & benchmarks

## Usage

### Basic Model Selection

```zig
const std = @import("std");
const ModelSelector = @import("orchestration/model_selector.zig").ModelSelector;
const SelectionConstraints = @import("orchestration/model_selector.zig").SelectionConstraints;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize selector
    const selector = try ModelSelector.init(
        allocator,
        "vendor/layerModels/MODEL_REGISTRY.json",
        "src/serviceCore/nOpenaiServer/orchestration/catalog/task_categories.json",
    );
    defer selector.deinit();
    
    // Load data
    try selector.loadRegistry();
    try selector.loadCategories();
    
    // Select model for code generation with T4 GPU
    const constraints = SelectionConstraints{
        .max_gpu_memory_mb = 14 * 1024, // 14GB
        .required_agent_type = "inference",
    };
    
    var result = try selector.selectModel("code", constraints);
    defer result.deinit(allocator);
    
    std.log.info("Selected: {s} (score: {d:.2})", .{
        result.model.name,
        result.score,
    });
}
```

### LLM Node with Auto-Selection

```zig
const LLMChatNode = @import("nodes/llm/llm_nodes.zig").LLMChatNode;

// Create node with task category (auto-select model)
const node = try LLMChatNode.init(
    allocator,
    "chat1",
    "Code Helper",
    null, // No explicit model
    "code", // Task category
    0.7, // Temperature
    1000, // Max tokens
    "You are a coding assistant.",
    service_config,
);

// Set model selector for intelligent routing
node.setModelSelector(selector);

// Execute - will auto-select best model for "code" category
const result = try node.execute(ctx);
```

### With Explicit Model

```zig
// Create node with explicit model (skip auto-selection)
const node = try LLMChatNode.init(
    allocator,
    "chat1",
    "Code Helper",
    "microsoft-phi-2", // Explicit model
    null, // No category needed
    0.7,
    1000,
    "You are a coding assistant.",
    service_config,
);

// Execute with specified model
const result = try node.execute(ctx);
```

## Task Categories

| Category | Description | Models | Use Cases |
|----------|-------------|--------|-----------|
| **code** | Code generation & understanding | 3 | Programming, debugging |
| **relational** | SQL & cross-lingual | 2 | Translation, databases |
| **math** | Mathematical reasoning | 0 | Calculations, proofs |
| **reasoning** | Complex reasoning | 0 | Logic, analysis |
| **summarization** | Text summarization | 0 | Document summaries |
| **time_series** | Forecasting | 0 | Financial, IoT |
| **graph** | Graph databases | 0 | Neo4j, knowledge graphs |
| **vector_search** | Semantic search | 0 | RAG, embeddings |
| **ocr_extraction** | Document extraction | 0 | OCR, forms |

## Selection Constraints

### GPU Memory Constraints

```zig
const SelectionConstraints = .{
    .max_gpu_memory_mb = 14 * 1024, // T4 GPU (14GB usable)
};
```

**GPU Profiles:**
- **T4 (16GB):** 14GB constraint → Small models (< 6GB)
- **A100-40GB:** 38GB constraint → Medium models (< 22GB)
- **A100-80GB:** 76GB constraint → Large models (< 48GB)

### Agent Type Constraints

```zig
const SelectionConstraints = .{
    .required_agent_type = "tool", // Require tool capability
};
```

**Agent Types:**
- **inference** - All models (text generation)
- **tool** - API/integration capable (vector_search, ocr_extraction, relational, graph)
- **orchestrator** - Complex orchestration (reasoning, summarization)

### Model Preferences

```zig
const preferred = [_][]const u8{"microsoft-phi-2"};
const excluded = [_][]const u8{"deepseek-coder-33b-instruct-q4_k_m"};

const SelectionConstraints = .{
    .preferred_models = &preferred,
    .excluded_models = &excluded,
};
```

## Scoring System

Models are scored based on multiple factors:

| Factor | Points | Description |
|--------|--------|-------------|
| Base Score | 50.0 | All models start here |
| Small Model (< 4GB) | +20.0 | Fast inference bonus |
| Medium Model (4-8GB) | +10.0 | Balanced bonus |
| Preferred Model | +30.0 | User preference bonus |
| Benchmark Score | +0-50.0 | Performance-based (future) |

**Example Scores:**
- google-gemma-3-270m-it (2GB): 70.0 (50 + 20)
- microsoft-phi-2 (6GB): 60.0 (50 + 10)
- microsoft-phi-2 (preferred): 90.0 (50 + 10 + 30)

## Performance Metrics

### Selection Time Benchmarks

| Category | Mean | Median | P95 | P99 |
|----------|------|--------|-----|-----|
| code | 0.0034ms | 0.0028ms | 0.0044ms | 0.0456ms |
| relational | 0.0019ms | 0.0018ms | 0.0021ms | 0.0033ms |
| other | 0.0002ms | 0.0002ms | 0.0002ms | 0.0003ms |

**Throughput:** 100,000+ selections/second

### Consistency

- **Code category:** 100% consistent (100/100 iterations)
- **Relational category:** 100% consistent (100/100 iterations)
- **Selection algorithm:** Deterministic (no randomness)

## Testing

### Run Integration Tests

```bash
# Python performance benchmark
cd /Users/user/Documents/arabic_folder
python3 scripts/orchestration/benchmark_routing_performance.py --iterations 1000

# Benchmark report
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json --report

# Zig tests (requires build.zig configuration)
zig test tests/orchestration/test_model_selection_integration.zig
```

### Test Scenarios Covered

1. ✅ Load MODEL_REGISTRY.json successfully
2. ✅ Load task_categories.json successfully
3. ✅ Select model with GPU constraints
4. ✅ Select model with agent type requirements
5. ✅ Preferred model selection
6. ✅ Model exclusion
7. ✅ GPU memory parsing
8. ✅ Fallback when no models match
9. ✅ Scoring system validation
10. ✅ Selection consistency

## Maintenance

### Add New Model

1. Add to MODEL_REGISTRY.json:
```json
{
  "name": "new-model",
  "hf_repo": "vendor/model-name",
  "gpu_memory": "8GB",
  ...
}
```

2. Enrich with metadata:
```bash
python3 scripts/models/hf_model_card_extractor.py vendor/layerModels/MODEL_REGISTRY.json
```

3. Validate:
```bash
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json
```

### Add New Category

1. Add to task_categories.json:
```json
{
  "categories": {
    "new_category": {
      "id": "new_category",
      "name": "New Category",
      "description": "...",
      "models": ["model1", "model2"],
      "benchmarks": [...]
    }
  }
}
```

2. Update mappings in hf_model_card_extractor.py
3. Re-enrich registry

### Update Model Metadata

```bash
# Re-run extractor to get latest HF data
python3 scripts/models/hf_model_card_extractor.py vendor/layerModels/MODEL_REGISTRY.json --verbose

# Validate changes
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json
```

## Known Limitations

1. **Category Coverage:** Only 2/9 categories have models (code, relational)
2. **Benchmark Extraction:** Limited by HF API access (some models gated)
3. **Multi-Category Support:** Models assigned to single category only
4. **Dynamic Routing:** GPU load not considered yet

## Future Enhancements (Phase 4)

- [ ] Multi-category model support with weighted scoring
- [ ] Real-time GPU load monitoring for dynamic routing
- [ ] A/B testing framework for model comparison
- [ ] Benchmark-based scoring integration
- [ ] Automated vendor sync (periodic HF refresh)
- [ ] Extended taxonomy (domain-specific categories)
- [ ] Cost-based routing
- [ ] Latency-based routing

## Support

- **Documentation:** [docs/01-architecture/MODEL_ORCHESTRATION_MAPPING.md](../../../docs/01-architecture/MODEL_ORCHESTRATION_MAPPING.md)
- **Validation Report:** [docs/08-reports/validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md](../../../docs/08-reports/validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md)
- **Issues:** https://github.com/plturrell/aArabic/issues

## License

Part of the arabic_folder project. See LICENSE for details.
