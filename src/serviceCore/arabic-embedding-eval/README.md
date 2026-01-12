# Arabic Translation Embedding Evaluation Framework

Comprehensive framework for evaluating and comparing embedding models for Arabic-English translation quality, inspired by [aimo-model-exp](https://github.com/aimo/aimo-model-exp).

## Features

✅ **Model Comparison** - Compare cross-lingual similarity and translation quality  
✅ **Performance Benchmarking** - Measure latency, throughput, and percentiles  
✅ **Arabic-Specific Metrics** - Financial term accuracy, domain adaptation  
✅ **CLI Tools** - Easy-to-use command-line interfaces  
✅ **Automated Reports** - Markdown reports with recommendations  

## Supported Models

| Model | Dimension | Endpoint | Description |
|-------|-----------|----------|-------------|
| **Multilingual-MiniLM** | 384 | `/embed/general` | Fast, multilingual (100+ languages) |
| **CamelBERT-Financial** | 768 | `/embed/financial` | Arabic financial domain specialized |
| **all-MiniLM-L6** | 384 | `/embed/general` | Baseline for comparison |

## Installation

```bash
cd src/serviceCore/arabic-embedding-eval
cargo build --release
```

## Quick Start

### 1. Start Embedding Service

```bash
# Start the embedding service first
cd ../serviceEmbedding-rust
docker-compose up -d

# Verify it's running
curl http://localhost:8007/health
```

### 2. Run Model Comparison

```bash
cd ../arabic-embedding-eval

# Compare both models using sample data
cargo run --bin arabic-model-comparison -- \
  --test-cases data/sample_test_cases.json \
  --models "multilingual,camelbert" \
  --output comparison_report.md
```

**Example Output:**
```
🚀 Arabic Model Comparison Tool

📚 Loading test cases...
✅ Loaded 10 test cases

🔬 Comparing 2 models:
   • paraphrase-multilingual-MiniLM-L12-v2 (384D)
   • CamelBERT-Financial (768D)

─────────────────────────────────────────────────────
📊 Evaluating model: paraphrase-multilingual-MiniLM-L12-v2
   Cross-lingual similarity: 0.8542
   Financial accuracy: 0.7234
   Avg latency: 12.45ms

📊 Evaluating model: CamelBERT-Financial
   Cross-lingual similarity: 0.8234
   Financial accuracy: 0.9123
   Avg latency: 18.76ms

═════════════════════════════════════════════════════
📊 COMPARISON RESULTS

┌────────────────────────────────┬──────────┬──────┬──────────────┬────────────┬─────────────┬────────────┐
│ Model                          │ Type     │ Dims │ Cross-Ling   │ Financial  │ Latency (ms)│ Throughput │
├────────────────────────────────┼──────────┼──────┼──────────────┼────────────┼─────────────┼────────────┤
│ paraphrase-multilingual-...    │ general  │ 384  │ 0.8542       │ 0.7234     │ 12.45       │ 80.3       │
│ CamelBERT-Financial            │ financial│ 768  │ 0.8234       │ 0.9123     │ 18.76       │ 53.3       │
└────────────────────────────────┴──────────┴──────┴──────────────┴────────────┴─────────────┴────────────┘

🏆 WINNERS

   Overall: paraphrase-multilingual-MiniLM-L12-v2
   General: paraphrase-multilingual-MiniLM-L12-v2
   Financial: CamelBERT-Financial

✅ Report saved to: comparison_report.md
```

### 3. Run Performance Benchmark

```bash
# Benchmark performance
cargo run --bin arabic-benchmark -- \
  --models "multilingual,camelbert" \
  --warmup 10 \
  --iterations 100 \
  --output benchmark_results.json
```

**Example Output:**
```
🔬 Arabic Model Benchmark Tool

📊 Benchmarking 2 models:
   • paraphrase-multilingual-MiniLM-L12-v2 (384D)
   • CamelBERT-Financial (768D)

🔬 Benchmarking: paraphrase-multilingual-MiniLM-L12-v2
  Warming up (10 iterations)...
  Benchmarking single encoding...
  Benchmarking batch encoding (32 texts)...
✅ Benchmark complete!
  Single encode: 12.34ms
  Batch encode (32): 125.67ms
  Throughput: 81.0 texts/sec

═══════════════════════════════════════════════════
📊 BENCHMARK RESULTS

┌────────────────────────────────┬──────────────┬────────────────┬───────────────┬──────────┬──────────┬──────────┐
│ Model                          │ Single (ms)  │ Batch/32 (ms)  │ Throughput    │ P50 (ms) │ P95 (ms) │ P99 (ms) │
├────────────────────────────────┼──────────────┼────────────────┼───────────────┼──────────┼──────────┼──────────┤
│ paraphrase-multilingual-...    │ 12.34        │ 125.67         │ 81            │ 12.10    │ 14.50    │ 16.20    │
│ CamelBERT-Financial            │ 18.92        │ 189.45         │ 53            │ 18.50    │ 21.30    │ 23.80    │
└────────────────────────────────┴──────────────┴────────────────┴───────────────┴──────────┴──────────┴──────────┘

🏆 PERFORMANCE WINNERS

   ⚡ Fastest (single): paraphrase-multilingual-MiniLM-L12-v2
   🚀 Best throughput: paraphrase-multilingual-MiniLM-L12-v2

✅ Results saved to: benchmark_results.json
```

## Metrics Explained

### Cross-Lingual Similarity
Measures how well Arabic and English translations align in embedding space (0-1).
- **Higher is better**
- Indicates semantic preservation across languages
- Critical for translation quality

### Financial Term Accuracy  
Accuracy of financial term preservation in embeddings (0-1).
- **Higher is better**
- Measures domain-specific performance
- Important for financial document processing

### Latency
Average embedding generation time per text.
- **Lower is better**
- Single text latency for real-time applications
- Batch latency for bulk processing

### Throughput
Number of texts that can be embedded per second.
- **Higher is better**
- Important for high-volume scenarios
- Calculated as 1000/latency_ms

## Test Data Format

Create `test_cases.json`:

```json
[
  {
    "id": "test_001",
    "arabic_text": "فاتورة رقم ١٢٣٤٥ بمبلغ ٥٠٠٠ ريال",
    "english_translation": "Invoice number 12345 for 5000 Riyals",
    "domain": "financial",
    "ground_truth_similar": [],
    "ground_truth_dissimilar": []
  }
]
```

**Fields:**
- `id`: Unique test case identifier
- `arabic_text`: Arabic source text
- `english_translation`: English translation
- `domain`: "general" or "financial"
- `ground_truth_similar`: Similar documents (future use)
- `ground_truth_dissimilar`: Dissimilar documents (future use)

## CLI Reference

### model_comparison

Compare multiple models on translation quality.

```bash
cargo run --bin arabic-model-comparison -- [OPTIONS]
```

**Options:**
- `-t, --test-cases <PATH>` - Test cases JSON file (required)
- `-m, --models <LIST>` - Comma-separated model names
- `-o, --output <PATH>` - Output report path (default: comparison_report.md)

**Model Names:**
- `multilingual` - Multilingual-MiniLM
- `camelbert` - CamelBERT-Financial
- `baseline` - all-MiniLM-L6-v2

### arabic-benchmark

Benchmark model performance.

```bash
cargo run --bin arabic-benchmark -- [OPTIONS]
```

**Options:**
- `-m, --models <LIST>` - Comma-separated model names
- `-w, --warmup <N>` - Warmup iterations (default: 10)
- `-i, --iterations <N>` - Benchmark iterations (default: 100)
- `-o, --output <PATH>` - Optional JSON output

## Model Selection Guide

### For General Translation
**→ Multilingual-MiniLM**
- Fastest inference (~12ms single)
- Good cross-lingual similarity (>0.85)
- Best for real-time applications
- Lower memory footprint (384d)

### For Financial Documents
**→ CamelBERT-Financial**
- Best financial term accuracy (>0.90)
- Specialized for Arabic financial text
- Higher accuracy for domain terms
- Worth the extra latency for critical docs

### Balanced Approach
Use both models with smart routing:
- Multilingual for general content
- CamelBERT for financial/invoices
- Route based on content classification

## Library Usage

```rust
use arabic_embedding_eval::{
    ArabicEvaluator, ArabicTestCase, ModelConfig,
};

#[tokio::main]
async fn main() -> Result<()> {
    // Load test cases
    let test_cases = vec![
        ArabicTestCase {
            id: "test_001".to_string(),
            arabic_text: "فاتورة".to_string(),
            english_translation: "Invoice".to_string(),
            domain: "financial".to_string(),
            ground_truth_similar: vec![],
            ground_truth_dissimilar: vec![],
        }
    ];
    
    // Create evaluator
    let evaluator = ArabicEvaluator::new();
    
    // Evaluate model
    let config = ModelConfig::multilingual_minilm();
    let result = evaluator.evaluate_model(&config, &test_cases).await?;
    
    println!("Cross-lingual similarity: {:.4}", 
        result.semantic_metrics.cross_lingual_similarity);
    println!("Latency: {:.2}ms", result.avg_latency_ms);
    
    Ok(())
}
```

## Integration with RAG Pipeline

Use evaluation results to configure your RAG system:

```rust
// Based on evaluation, select best model
let embedding_config = if document.is_financial() {
    ModelConfig::camelbert_financial()  // Best for financial
} else {
    ModelConfig::multilingual_minilm()  // Faster for general
};

// Use in RAG pipeline
let embeddings = generate_embeddings(&embedding_config, &texts).await?;
store_in_qdrant(embeddings).await?;
```

## Troubleshooting

### Service Not Running
```bash
# Check if embedding service is up
curl http://localhost:8007/health

# Start the service
cd ../serviceEmbedding-rust
docker-compose up -d
```

### Out of Memory
```bash
# Reduce test cases or use smaller batches
# Edit benchmark iterations:
cargo run --bin arabic-benchmark -- --iterations 50
```

### Slow Evaluation
```bash
# Reduce warmup and iterations for quick tests
cargo run --bin arabic-model-comparison -- \
  --test-cases data/sample_test_cases.json \
  --models "multilingual"
```

### Model Endpoint Errors
```bash
# Verify endpoints are correct
curl -X POST http://localhost:8007/embed/general \
  -H "Content-Type: application/json" \
  -d '{"texts":["test"],"normalize":true}'
```

## Contributing

Add new models in `src/lib.rs`:

```rust
impl ModelConfig {
    pub fn my_custom_model() -> Self {
        Self {
            name: "my-model".to_string(),
            endpoint: "http://localhost:8007/embed/custom".to_string(),
            dimension: 512,
            model_type: "specialized".to_string(),
            description: "Custom model for X".to_string(),
            supported_languages: vec!["ar".to_string(), "en".to_string()],
        }
    }
}
```

Then use in CLI:
```bash
# Update model_comparison.rs to recognize "custom"
cargo run --bin arabic-model-comparison -- \
  --models "multilingual,custom"
```

## Performance Tips

1. **Use appropriate batch sizes** for your GPU/CPU
2. **Cache embeddings** for repeated texts
3. **Warm up** before benchmarking
4. **Profile with smaller datasets** first
5. **Consider trade-offs** between speed and accuracy

## References

- [Phase 2 Implementation](../../../docs/implementation/PHASE2_EMBEDDING_ENHANCEMENT.md)
- [Translation Service](../serviceTranslation-rust/README.md)
- [Embedding Service v2](../serviceEmbedding-rust/README.md)
- [Inspiration: aimo-model-exp](https://github.com/aimo)

---

**Evaluate with confidence. Choose models backed by data!** 📊
