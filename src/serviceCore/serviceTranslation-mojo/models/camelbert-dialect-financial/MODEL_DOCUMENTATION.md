# CamelBERT (Unified: All Variants)

## Overview

CamelBERT is a family of Arabic BERT models fine-tuned for Arabic dialect classification and text embeddings. This unified service supports all CamelBERT variants, automatically selecting the best model based on domain and availability.

**Base Model**: [CAMeL-Lab/bert-base-arabic-camelbert-msa](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-msa)  
**License**: Apache 2.0

## Model Details

### Technical Specifications
- **Architecture**: BERT (Bidirectional Encoder Representations from Transformers)
- **Base Model**: CAMeL-Lab/bert-base-arabic-camelbert-msa
- **Hidden Size**: 768
- **Layers**: 12
- **Attention Heads**: 12
- **Max Sequence Length**: 512 tokens (128 for classification)
- **Vocabulary Size**: 30,000
- **Task**: Sequence Classification (8 Arabic dialects)

### Model Variants

#### camelbert-dialect-financial
- **Purpose**: Financial domain Arabic + dialect classification
- **Training Data**:
  - AraFinNews (212K articles) - Domain adaptation
  - QADI dataset (3,303 samples) - Dialect classification
  - ArBanking77 (40,402 samples) - Dialect-labeled banking intents
- **Dialects**: 8 Arabic dialects (egyptian, sudanese, iraqi, yemeni, saudi, gulf, palestinian, maghrebi)
- **Best For**: Financial Arabic text, banking intents
- **Performance**: 60-70% on QADI, 82-88% with Banking77, 90% with ensembling

#### camelbert-fusion-ultimate
- **Purpose**: Best performing fusion model
- **Training Data**: All available Arabic datasets (QADI, Banking77, AraFinNews, ARBML)
- **Best For**: General Arabic dialect classification (highest accuracy)
- **Location**: May have checkpoint subdirectory (e.g., `checkpoint-25528`)

#### camelbert-fusion-base
- **Purpose**: Base fusion model
- **Training Data**: General Arabic text
- **Best For**: General Arabic tasks, fallback option

#### camelbert-fusion-recursive
- **Purpose**: Recursive refinement variant
- **Architecture**: Base model + recursive refinement head (`refinement_head.pt`)
- **Best For**: Ambiguous cases, low-confidence predictions, code-switching
- **Features**: Iterative refinement cycles for improved accuracy

#### kuwain-1.5B (AraBERT)
- **Purpose**: Arabic BERT for embeddings and masked language modeling
- **Base Model**: AraBERT v0.2-base (`aubmindlab/bert-base-arabertv02`)
- **Architecture**: BERT (BertForMaskedLM) - **NOT suitable for classification**
- **Trained By**: AUB MIND Lab (American University of Beirut)
- **Training Data**: 200M sentences / 77GB / 8.6B words
  - OSCAR unshuffled (filtered)
  - Arabic Wikipedia dump (2020/09/01)
  - 1.5B words Arabic Corpus
  - OSIAN Corpus
  - Assafir news articles
- **Training Hardware**: TPUv3-8, 3M training steps
- **Best For**: Text embeddings, masked language modeling, general Arabic understanding
- **Features**: 64K vocabulary, 768 hidden size, 12 layers, 12 attention heads
- **Note**: 
  - Integrated into CamelBERT service for embedding tasks
  - **Cannot be used for classification** (it's MLM, not sequence classification)
  - Automatically selected for embedding tasks when available
  - Used in TOON generation for financial domain language injection

### Dialect Mappings
- **0**: egyptian (Egyptian Arabic)
- **1**: sudanese (Sudanese Arabic)
- **2**: iraqi (Iraqi Arabic)
- **3**: yemeni (Yemeni Arabic)
- **4**: saudi (Saudi Arabic)
- **5**: gulf (Gulf Arabic)
- **6**: palestinian (Palestinian Arabic)
- **7**: maghrebi (Maghrebi Arabic)

## Service & Integration

### Unified FastAPI Service
The model is exposed via a unified FastAPI microservice (`services/camelbert-service/app/main.py`) that:

- **Automatically detects** available model variants
- **Selects the best model** based on domain (financial domain → financial variant)
- **Supports classification** (dialect classification)
- **Supports embeddings** (text embeddings for GNN features)
- **Hides implementation details** from clients

Endpoints:
- `GET /health` → Liveness probe (shows available models)
- `GET /ready` → Readiness probe (ensures at least one model is ready)
- `POST /classify` → Single text dialect classification
- `POST /classify/batch` → Batch dialect classification
- `POST /embed` → Single text embedding
- `POST /embed/batch` → Batch text embeddings
- `GET /metrics` → Prometheus metrics endpoint

### Environment Variables
- `CAMELBERT_BASE_PATH` (default `/models/arabic_models`) - Base path for all models
- `CAMELBERT_FINANCIAL_PATH` - Path to financial variant
- `CAMELBERT_ULTIMATE_PATH` - Path to ultimate variant
- `CAMELBERT_BASE_MODEL_PATH` - Path to base variant
- `CAMELBERT_RECURSIVE_PATH` - Path to recursive variant
- `CAMELBERT_REQUEST_TIMEOUT` (default `60` seconds) - Request timeout

### Model Selection Logic
The service automatically selects models in this priority:

**For Classification Tasks:**
1. **Financial variant** (if domain contains "financial" and available)
2. **Ultimate variant** (if available, best general performance)
3. **Base variant** (fallback)
4. **Recursive variant** (if available, for refinement)

**For Embedding Tasks:**
1. **Kuwain (AraBERT)** (if available, optimized for embeddings)
2. **Financial variant** (if domain contains "financial" and available)
3. **Ultimate variant** (if available)
4. **Base variant** (fallback)

Clients can optionally specify:
- `domain`: Hint for domain-specific selection (e.g., "financial")
- `variant`: Force specific variant (financial, ultimate, base, recursive, kuwain)

### Gateway Integration
- **Gateway router**: `services/gateway/app/routers/camelbert.py`
- **Endpoints**:
  - `GET /api/camelbert/healthz` - Health check
  - `GET /api/camelbert/ready` - Readiness check
  - `POST /api/camelbert/classify` - Single classification
  - `POST /api/camelbert/classify/batch` - Batch classification
  - `POST /api/camelbert/embed` - Single embedding
  - `POST /api/camelbert/embed/batch` - Batch embeddings
- **Rate limiting**: 60 req/min for single, 20 req/min for batch
- **JWT headers**: Automatically injected by gateway
- **Correlation IDs**: Tracked across service boundaries

### APISIX Routes
Routes configured in `infrastructure/apisix/routes.yaml`:
- `/camelbert/classify` - Single classification (60 req/min)
- `/camelbert/classify/batch` - Batch classification (20 req/min)
- `/camelbert/embed` - Single embedding (60 req/min)
- `/camelbert/embed/batch` - Batch embeddings (20 req/min)
- `/camelbert/healthz` - Health check
- `/camelbert/ready` - Readiness check

All routes include:
- Rate limiting
- Prometheus metrics
- Request ID tracking
- CORS support

### n8n Workflow
Workflow template available at `services/camelbert-workflow.json`:
- Webhook trigger at `/camelbert`
- Task type detection (classify vs embed)
- Batch mode detection
- Single and batch paths for both tasks
- Error handling with retry logic (up to 2 retries)
- Supports domain and variant selection
- Uses APISIX base URL from environment

### UI Integration
- **Widget Component**: `services/gateway/frontend/src/components/CamelBERTWidget.tsx`
- **Command Palette**: "CamelBERT Arabic Classification" (Ctrl+Shift+B)
- **Keyboard Shortcut**: `Ctrl+Shift+B` (or `Cmd+Shift+B` on Mac)
- **Features**:
  - Task selection (Classification / Embedding)
  - Text input (single or batch)
  - Domain selection (for automatic model selection)
  - Variant selection (force specific variant)
  - Classification results with dialect probabilities
  - Embedding visualization
  - Processing history

## Usage Examples

### Dialect Classification
```python
import requests

response = requests.post(
    "http://localhost:8000/api/camelbert/classify",
    json={
        "text": "أريد تحويل مبلغ من حسابي البنكي",
        "domain": "financial",
        "return_all_scores": True
    }
)
print(f"Dialect: {response.json()['dialect']}")
print(f"Confidence: {response.json()['confidence']}")
print(f"Variant: {response.json()['variant']}")
```

### Batch Classification
```python
response = requests.post(
    "http://localhost:8000/api/camelbert/classify/batch",
    json={
        "texts": [
            "مرحبا كيف حالك",
            "أريد فتح حساب",
            "شكرا جزيلا"
        ],
        "domain": "financial",
        "return_all_scores": True
    }
)

for idx, result in enumerate(response.json()["results"]):
    print(f"Text {idx + 1}: {result['dialect']} ({result['confidence']:.3f})")
```

### Text Embedding
```python
response = requests.post(
    "http://localhost:8000/api/camelbert/embed",
    json={
        "text": "نص عربي للتحليل",
        "domain": "financial",
        "pooling": "mean"
    }
)
embedding = response.json()["embedding"]
print(f"Embedding dimension: {response.json()['dimension']}")
```

### Batch Embeddings
```python
response = requests.post(
    "http://localhost:8000/api/camelbert/embed/batch",
    json={
        "texts": ["نص 1", "نص 2", "نص 3"],
        "pooling": "mean",
        "batch_size": 32
    }
)
embeddings = response.json()["embeddings"]
```

### Force Specific Variant
```python
response = requests.post(
    "http://localhost:8000/api/camelbert/classify",
    json={
        "text": "نص عربي",
        "variant": "ultimate"  # Force ultimate variant
    }
)
```

## Recommendations

### Production Deployment
1. **Model Selection**: Use financial variant for financial domain, ultimate for general use
2. **GPU Requirements**: Use GPU for better performance (CPU fallback available)
3. **Batch Processing**: Use batch endpoints for multiple texts
4. **Model Caching**: Models are cached in memory after first load

### Performance Optimization
1. **Variant Selection**: Choose appropriate variant for your domain
   - Financial domain → financial variant
   - General use → ultimate variant
   - Ambiguous cases → recursive variant
2. **Batch Processing**: Group requests into batches for better throughput
3. **Embedding Pooling**: Use "mean" for general embeddings, "cls" for classification-focused
4. **Sequence Length**: Classification uses 128 tokens, embeddings use 512 tokens

### Security Best Practices
1. **Input Validation**: Validate and sanitize all Arabic text inputs
2. **Rate Limiting**: Enforce rate limits at APISIX and service levels
3. **Content Filtering**: Implement content filtering for sensitive applications
4. **Model Isolation**: Ensure models are properly isolated in production

### Monitoring & Observability
1. **Variant Metrics**: Track which variant is used via Prometheus labels
2. **Performance Comparison**: Compare variant performance and accuracy
3. **Availability**: Monitor variant availability and automatic selection
4. **Structured Logging**: All logs include variant and task information

### Resource Management
1. **GPU Allocation**: Use GPU for better performance, CPU fallback available
2. **Memory**: Models are loaded on-demand and cached
3. **Concurrent Requests**: Limit concurrent batch requests to avoid OOM
4. **Model Loading**: Models are loaded once and reused

### Development & Testing
1. **Variant Testing**: Test with all available variants
2. **Domain Testing**: Test domain-based selection logic
3. **Batch Testing**: Test batch processing with various sizes
4. **Embedding Testing**: Test embedding generation and pooling methods

### Cost Optimization
1. **Variant Selection**: Use appropriate variant for your use case
2. **Batch Processing**: Group requests into batches to reduce overhead
3. **Caching**: Embeddings can be cached for repeated texts
4. **GPU Usage**: Use GPU when available for better throughput

### Documentation & Maintenance
1. **Variant Documentation**: Document when to use each variant
2. **Domain Documentation**: Document domain-based selection logic
3. **API Compatibility**: Ensure all variants return compatible responses
4. **Version Tracking**: Track model versions and updates

### Integration Best Practices
1. **Unified Interface**: Always use the unified service, not variant-specific endpoints
2. **Transparency**: Don't expose variant details to end users unless needed
3. **Error Handling**: Handle variant-specific errors gracefully
4. **Monitoring**: Track variant usage patterns and performance

