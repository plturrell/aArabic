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

### When Models Are Downloaded

The `download_models_on_brev.sh` script automatically enriches `MODEL_REGISTRY.json` with:

1. **Orchestration categories** mapped from HF metadata
2. **Agent types** determined by category
3. **HF metadata**:
   - Download counts
   - Like counts
   - License information
   - Languages supported
   - Datasets used
   - Pipeline tag

### Manual Enrichment

You can manually enrich the registry at any time:

```bash
# Enrich entire registry
python3 scripts/models/hf_model_card_extractor.py vendor/layerModels/MODEL_REGISTRY.json

# Test single model
python3 scripts/models/hf_model_card_extractor.py --test "google/gemma-3-270m-it"
```

## Integration with Orchestration Layer

### Model Selection by Category

```zig
// In orchestration/catalog/task_categories.json
{
  "math": {
    "models": ["google-gemma-3-270m-it"],  // Has "math" category
    "benchmarks": ["GSM8K", "MATH"]
  },
  "code": {
    "models": [
      "google-gemma-3-270m-it",
      "microsoft-phi-2",
      "deepseek-coder-33b-instruct"
    ],
    "benchmarks": ["HumanEval", "MBPP"]
  },
  "relational": {
    "models": ["HY-MT1.5-7B"],  // Translation model
    "benchmarks": ["Spider"]
  }
}
```

### Routing Logic

```zig
// Example routing based on task category
fn selectModelForTask(task_category: []const u8) ![]const u8 {
    const registry = try loadModelRegistry();
    
    for (registry.models) |model| {
        if (model.orchestration_categories.contains(task_category)) {
            // Check if model fits GPU constraints
            if (model.gpu_memory <= available_vram) {
                return model.name;
            }
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

### Planned Additions

1. **Benchmark Integration**
   - Extract actual scores from HF model cards
   - Link to evaluation datasets
   - Track performance over time

2. **Multi-Category Support**
   - Models can excel at multiple tasks
   - Weighted scoring per category
   - Automatic A/B testing

3. **Dynamic Routing**
   - Real-time model selection based on:
     - Current GPU load
     - Task complexity
     - Latency requirements
     - Cost constraints

4. **Category Expansion**
   - Add domain-specific categories (medical, legal, finance)
   - Fine-grained subcategories
   - Custom user-defined categories

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

## References

- [Unified Orchestration Architecture](../src/serviceCore/nOpenaiServer/orchestration/UNIFIED_ORCHESTRATION.md)
- [Task Categories Catalog](../src/serviceCore/nOpenaiServer/orchestration/catalog/task_categories.json)
- [HuggingFace Model API](https://huggingface.co/docs/api-inference/index)
- [Model Registry](../vendor/layerModels/MODEL_REGISTRY.json)

## Support

For questions or issues:
- Check existing model mappings in `MODEL_REGISTRY.json`
- Test mappings with `--test` flag
- Contribute new mappings via PR
- Report issues: https://github.com/plturrell/aArabic/issues
