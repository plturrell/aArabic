# Local Model Inventory

## Available Models (No External LLM Access)

### 1. **Qwen3-Coder-30B-A3B-Instruct**
- **Location:** `vendor/layerModels/huggingFace/Qwen/Qwen3-Coder-30B-A3B-Instruct/`
- **Type:** Code generation, instruction following
- **Size:** 30B parameters (16 shards)
- **Use Cases:**
  - Backend code generation
  - Shimmy AI orchestration
  - n8n workflow code nodes
- **Format:** Safetensors (GGUF compatible)

### 2. **HY-MT1.5-1.8B (Tencent)** ⭐ PRIMARY TRANSLATION
- **Location:** `vendor/layerModels/huggingFace/tencent/HY-MT1.5-1.8B/`
- **Type:** Machine translation (General purpose)
- **Size:** 1.8B parameters
- **Use Cases:**
  - **General translation** (English ↔ Arabic)
  - Business document translation
  - Email and correspondence translation
  - Technical documentation translation
  - Contract and agreement translation
- **Format:** Safetensors
- **Performance:** Fast, accurate general translation

### 3. **Arabic Financial Models** ⭐ SPECIALIZED FINANCIAL
- **Location:** `vendor/layerModels/folderRepos/arabic_models/`
- **Type:** Domain-specific Arabic NLP (Financial trained)
- **Models Included:**
  - CamelBERT-Dialect-Financial
  - M2M100-418M (Arabic financial fine-tuned)
- **Use Cases:**
  - **Financial Arabic NLP** (accounting, banking, finance terminology)
  - Arabic invoice processing with financial context
  - Financial reports and statements (Arabic)
  - Budget and forecast documents
  - Tax and VAT documentation (Arabic)
  - Financial contracts and agreements
  - Banking correspondence
- **Training:** Specialized on Arabic financial corpora
- **Advantage:** Understands financial terminology, accounting concepts

### 4. **RLM (Reinforcement Learning Models)** ⭐ INTELLIGENT TRANSLATION
- **Location:** `vendor/layerModels/folderRepos/rlm/`
- **Type:** RL-trained translation agents
- **Use Cases:**
  - **Context-aware translation** with business logic
  - Multi-document translation workflows
  - Quality assurance and translation validation
  - Adaptive translation based on domain
  - Learning from correction feedback
- **Advantage:** Can learn and improve translation quality over time

### 5. **MiniMax-M2.1**
- **Location:** `vendor/layerModels/huggingFace/MiniMaxAI/MiniMax-M2.1/`
- **Type:** General purpose LLM
- **Size:** Large (130 shards)
- **Use Cases:**
  - General text processing
  - Document analysis
  - Fallback for complex queries
- **Format:** Safetensors

### 5. **DeepSeek-Math**
- **Location:** `vendor/layerModels/folderRepos/deepseek-math/`
- **Type:** Mathematical reasoning
- **Use Cases:**
  - Financial calculations
  - VAT computation
  - Invoice validation

### 6. **RLM (Reinforcement Learning Models)**
- **Location:** `vendor/layerModels/folderRepos/rlm/`
- **Type:** RL-trained agents
- **Use Cases:**
  - Workflow optimization
  - Decision making

### 7. **ToolOrchestra**
- **Location:** `vendor/layerModels/folderRepos/toolorchestra/`
- **Type:** Tool-use specialized models
- **Use Cases:**
  - API orchestration
  - Multi-tool workflows

## Service Configuration

### Backend API
```python
MODEL_PATH = "/models"
DEFAULT_MODEL = "Qwen3-Coder-30B-A3B-Instruct"
TRANSLATION_MODEL = "HY-MT1.5-1.8B"
ARABIC_MODEL = "arabic_models/camelbert"
```

### Langflow
- Configure to use local Ollama/vLLM endpoint
- Point to model server on backend

### Shimmy
- Uses models from `/app/models` mount
- Primary: Qwen3-Coder for orchestration

### n8n
- Configure AI nodes to use backend API
- No external OpenAI/Anthropic calls

## Model Server Setup

Deploy a local model server (Ollama/vLLM) that serves these models:

```bash
# Option 1: Ollama (simpler)
docker run -d -v $(pwd)/vendor/layerModels:/models \
  -p 11434:11434 ollama/ollama

# Option 2: vLLM (better performance)
docker run -d -v $(pwd)/vendor/layerModels:/models \
  -p 8000:8000 vllm/vllm-openai:latest \
  --model /models/huggingFace/Qwen/Qwen3-Coder-30B-A3B-Instruct
```

## Validation

Ensure NO services have:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- External API endpoints
- Internet access for model downloads

All model access goes through:
1. Local model server (Ollama/vLLM)
2. Backend API model endpoints
3. Mounted model directories