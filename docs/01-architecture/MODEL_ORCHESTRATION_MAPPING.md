# Model Orchestration Category Mapping

## Overview

This document explains how HuggingFace model cards are mapped to our unified orchestration task categories, enabling intelligent model selection and routing based on task requirements.

## Orchestration Categories

Our system uses **9 task categories** aligned with the Unified Orchestration Architecture:

| Category | Description | Example Use Cases | Benchmarks |
|----------|-------------|-------------------|------------|
| `math` | Mathematical reasoning & calculations | GSM8K, MATH problems | GSM8K, MATH |
| `code` | Code generation & understanding | Programming, debugging | HumanEval, MBPP |
| `reasoning` | Complex multi-step reasoning | Logic puzzles, analysis | ARC-Challenge, BIG-Bench |
| `summarization` | Text summarization & condensation | Document summaries, logs | SummScreen, GovReport |
| `time_series` | Time series forecasting & analysis | Financial, IoT predictions | M4, Monash |
| `relational` | Relational data & SQL | Database queries, tables | Spider |
| `graph` | Graph database queries | Knowledge graphs, Neo4j | GraphQA, AQuA |
| `vector_search` | Semantic search & embeddings | RAG, document retrieval | BEIR, MSMARCO |
| `ocr_extraction` | Document & image text extraction | Invoice processing, OCR | DocVQA, ChartQA |

## Agent Types

Models are classified into **3 agent types** based on their capabilities:

1. **inference** - All models (core text generation)
2. **tool** - Models suitable for tool/API integration (vector_search, ocr_extraction, relational, graph)
3. **orchestrator** - Models for complex orchestration (reasoning, summarization)

## HuggingFace → Orchestration Mapping

### Direct Mappings (pipeline_tag & tags)

```python
{
    # Math & Reasoning
    "math": ["math"],
    "reasoning": ["reasoning", "math"],
    "question-answering": ["reasoning"],
    
    # Code
    "text-generation": ["code"],
    "code": ["code"],
    "code-generation": ["code"],
    
    # Summarization
    "summarization": ["summarization"],
    "document-question-answering": ["ocr_extraction", "reasoning"],
    
    # Translation & Multilingual
    "translation": ["relational"],  # Cross-lingual understanding
    "multilingual": ["relational"],
    
    # Time Series
    "time-series": ["time_series"],
    "forecasting": ["time_series"],
    
    # Vector Search
    "feature-extraction": ["vector_search"],
    "sentence-similarity": ["vector_search"],
    "embeddings": ["vector_search"],
    
    # Relational & Graph
    "table-question-answering": ["relational", "graph"],
    "tabular": ["relational"],
    
    # OCR
    "image-to-text": ["ocr_extraction"],
    "ocr": ["ocr_extraction"],
}
```

### Model Name Pattern Detection

The system also infers categories from model names:

```python
{
    "math": ["gsm", "math", "calc"],
    "code": ["code", "coder", "starcoder", "codegen"],
    "reasoning": ["reason", "think", "chain-of-thought", "cot"],
    "summarization": ["summar", "abstract"],
    "time_series": ["forecast", "timeseries"],
    "relational": ["sql", "table", "tabular"],
    "graph": ["graph", "cypher", "neo4j"],
    "vector_search": ["embed", "retriev", "rag"],
    "ocr_extraction": ["ocr", "document", "vision"],
}
```

### Language-Specific Enhancements

Arabic and multilingual models get additional categorization:

```python
{
    "ar": ["relational"],  # Cross-lingual relationships
    "multilingual": ["relational"],
}
```

## Current Model Registry Mappings

### google-gemma-3-270m-it
- **Orchestration Categories:** `code`
- **Agent Types:** `inference`
- **HF Pipeline:** `text-generation`
- **Downloads:** 113,884
- **Use Case:** Testing, general-purpose text generation

### HY-MT1.5-7B (Arabic Translation)
- **Orchestration Categories:** `relational`
- **Agent Types:** `inference`, `tool`
- **HF Pipeline:** `translation`
- **Downloads:** 119,010
- **Use Case:** Arabic ↔ English translation, cross-lingual understanding

### microsoft-phi-2
- **Orchestration Categories:** `code`
- **Agent Types:** `inference`
- **HF Pipeline:** `text-generation`
- **Downloads:** 1,294,247
- **Use Case:** Code generation, reasoning

### deepseek-coder-33b-instruct
- **Orchestration Categories:** `code`
- **Agent Types:** `inference`
- **Downloads:** 9,578
- **Use Case:** Advanced code generation

## Automatic Enrichment

### Enrichment Script

The `hf_model_card_extractor.py` script enriches `MODEL_REGISTRY.json` with comprehensive vendor data:

1. **Orchestration categories** - Automatically mapped from HF metadata
2. **Agent types** - Determined by orchestration categories
3. **HF metadata**:
   - Download counts and likes
   - License information
   - Languages supported
   - Datasets used
   - Pipeline tag and model tags
4. **Model specifications**:
   - Parameter count
   - Context length
   - Architecture type
   - Quantization format
5. **Hardware requirements**:
   - Minimum GPU memory
   - CPU compatibility
6. **Training information**:
   - Training datasets
   - Training tokens
7. **Benchmark scores**:
   - Extracted from model cards
   - Validated against known ranges

### Usage

```bash
# Enrich entire registry
python3 scripts/models/hf_model_card_extractor.py vendor/layerModels/MODEL_REGISTRY.json

# Test single model with verbose output
python3 scripts/models/hf_model_card_extractor.py --test "google/gemma-3-270m-it" --verbose

# Validate benchmarks
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json

# Generate benchmark report
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json --report

# Compare models on specific benchmark
python3 scripts/models/benchmark_validator.py vendor/layerModels/MODEL_REGISTRY.json --compare humaneval
```

## Integration with Orchestration Layer

### Task Categories Catalog

The orchestration system uses `src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json` to map tasks to models:

```json
{
  "categories": {
    "code": {
      "models": [
        "google-gemma-3-270m-it",
        "microsoft-phi-2",
        "deepseek-coder-33b-instruct-q4_k_m"
      ],
      "benchmarks": [
        {"name": "HumanEval", "metric": "pass@1"},
        {"name": "MBPP", "metric": "pass@1"}
      ]
    },
    "relational": {
      "models": [
        "HY-MT1.5-7B",
        "translategemma-27b-it-GGUF"
      ],
      "benchmarks": [
        {"name": "Spider", "metric": "exact match"}
      ]
    }
  },
  "routing_rules": {
    "gpu_constraints": {
      "t4_16gb": {
        "max_model_memory": "14GB",
        "recommended_models": [
          "google-gemma-3-270m-it",
          "microsoft-phi-2"
        ]
      }
    }
  }
}
```

### Model Selection in Zig (Planned Implementation)

```zig
// Future implementation in nFlow/nodes/llm/
pub fn selectModelForTask(
    allocator: Allocator,
    task_category: []const u8,
    gpu_memory_available: usize
) ![]const u8 {
    // Load task_categories.json
    const categories = try loadTaskCategories(allocator);
    
    // Get models for this category
    const category = categories.get(task_category) orelse 
        return error.UnknownCategory;
    
    // Filter by GPU constraints
    for (category.models) |model_name| {
        const model = try loadModelFromRegistry(allocator, model_name);
        
        if (model.gpu_memory_mb <= gpu_memory_available) {
            return model.name;
        }
    }
    
    return error.NoSuitableModel;
}
```

## Benefits

### 1. Intelligent Model Selection
- **Task-based routing:** Automatically select best model for each task category
- **Multi-model orchestration:** Route different parts of complex queries to specialized models
- **Capability scoring:** Track performance per category for informed decisions

### 2. Enhanced Discoverability
- **Popularity metrics:** Downloads and likes from HuggingFace
- **License awareness:** Know licensing constraints upfront
- **Language support:** Identify multilingual capabilities

### 3. Orchestration Integration
- **Agent type mapping:** Know which models can be tools vs orchestrators
- **Benchmark tracking:** Link models to standard evaluation datasets
- **Performance monitoring:** Track category-specific performance in production

### 4. Maintenance Benefits
- **Automatic updates:** Re-run extractor to get latest HF metadata
- **Consistent categorization:** Unified taxonomy across all models
- **Extensible:** Easy to add new categories or mappings

## Future Enhancements

### ✅ Phase 5 Completed Additions

1. **✅ Benchmark Integration** (Enhancement #4)
   - Module: `src/serviceCore/nLocalModels/orchestration/benchmark_scoring.zig`
   - Extract scores from HF model cards ✅
   - Link to evaluation datasets (16 benchmarks configured) ✅
   - Category-specific benchmark weighting ✅
   - Normalized scoring (0-50 points) ✅

2. **✅ Multi-Category Support** (Enhancement #1)
   - Module: `src/serviceCore/nLocalModels/orchestration/multi_category.zig`
   - Models excel at multiple tasks ✅
   - Weighted scoring per category with confidence levels (0.7-0.95) ✅
   - Per-category performance tracking ✅
   - MultiCategoryRegistry for centralized management ✅

3. **✅ Dynamic GPU-Aware Routing** (Enhancement #2)
   - Module: `src/serviceCore/nLocalModels/orchestration/gpu_monitor.zig`
   - Real-time GPU load monitoring via nvidia-smi ✅
   - Load-balanced model selection ✅
   - Health checking (temperature, utilization, power) ✅
   - Multi-GPU support ✅

### 📋 Future Additions (Medium/Low Priority)

1. **A/B Testing Framework** (Enhancement #3 - Medium Priority)
   - Compare model performance across categories
   - Track selection accuracy
   - Automated performance regression detection

2. **Category Expansion** (Enhancement #5 - Medium Priority)
   - Add domain-specific categories (medical, legal, finance)
   - Fine-grained subcategories
   - Custom user-defined categories

3. **Python → Zig/Mojo Migration** (Enhancement #6 - Low Priority)
   - Convert model selection logic to Zig
   - Mojo integration for performance-critical paths
   - Zero-copy model metadata handling

## Usage Examples

### Example 1: Find Code Generation Models

```bash
# Query registry for code models
jq '.models[] | select(.orchestration_categories[] == "code") | .name' \
  vendor/layerModels/MODEL_REGISTRY.json
```

Output:
```
"google-gemma-3-270m-it"
"microsoft-phi-2"
"deepseek-coder-33b-instruct"
```

### Example 2: Get Most Popular Models

```bash
# Sort by downloads
jq '.models | sort_by(.hf_metadata.downloads) | reverse | .[0:3] | 
    .[] | {name, downloads: .hf_metadata.downloads}' \
  vendor/layerModels/MODEL_REGISTRY.json
```

### Example 3: Find Tool-Capable Models

```bash
# Models that can be used as tools
jq '.models[] | select(.agent_types[] == "tool") | .name' \
  vendor/layerModels/MODEL_REGISTRY.json
```

Output:
```
"HY-MT1.5-7B"
```

## Current Implementation Status

### ✅ Phase 5 Completed (All High-Priority Items)

#### Infrastructure & Tooling
- **HF Model Card Extractor** - `scripts/models/hf_model_card_extractor.py` ✅
- **Benchmark Validator** - `scripts/models/benchmark_validator.py` ✅
- **Task Categories Catalog** - `src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json` ✅
- **MODEL_REGISTRY.json Enrichment** - All 7 models enriched with metadata ✅
- **Orchestration Category Mapping** - Models mapped to task categories ✅

#### Core Orchestration Modules (Zig)
- **Model Selector** - `src/serviceCore/nLocalModels/orchestration/model_selector.zig` ✅
  - Basic category-based selection ✅
  - Multi-category selection with weighted scoring ✅
  - GPU-aware dynamic routing ✅
  - Fallback strategies ✅
  
- **Benchmark Scoring** - `src/serviceCore/nLocalModels/orchestration/benchmark_scoring.zig` ✅
  - 16 benchmark weights configured ✅
  - Category-specific scoring ✅
  - Normalized scoring (0-50 points) ✅
  - 6 passing tests ✅

- **Multi-Category Support** - `src/serviceCore/nLocalModels/orchestration/multi_category.zig` ✅
  - Per-category confidence scoring ✅
  - MultiCategoryRegistry ✅
  - MultiCategoryBuilder helpers ✅
  - 10 passing tests ✅

- **GPU Monitoring** - `src/serviceCore/nLocalModels/orchestration/gpu_monitor.zig` ✅
  - nvidia-smi integration ✅
  - Real-time load monitoring ✅
  - Health checking ✅
  - Load-balanced selection ✅
  - 5 passing tests ✅

**Total Test Coverage:** 23 passing tests across all modules ✅

### 📋 Future Work (Medium/Low Priority)

- **A/B Testing Framework** - Compare model performance across categories
- **Automated Vendor Sync** - Periodic HF metadata refresh
- **Extended Taxonomy** - Domain-specific categories (medical, legal, finance)
- **Python → Zig Migration** - Convert remaining Python orchestration logic

## References

- [Task Categories Catalog](../../src/serviceCore/nLocalModels/orchestration/catalog/task_categories.json)
- [Model Registry](../../vendor/layerModels/MODEL_REGISTRY.json)
- [HuggingFace Model API](https://huggingface.co/docs/api-inference/index)
- [LLM Integration Nodes](../../src/serviceCore/nFlow/nodes/llm/llm_nodes.zig)

## Support

For questions or issues:
- Check existing model mappings in `MODEL_REGISTRY.json`
- Test mappings with `--test` flag
- Contribute new mappings via PR
- Report issues: https://github.com/plturrell/aArabic/issues
