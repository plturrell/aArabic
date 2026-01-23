# nOpenaiServer

Local LLM inference server with OpenAI-compatible API.

## Overview

**Language**: Zig/Mojo  
**Port**: 11434  
**Status**: Production  
**Repository**: `src/serviceCore/nLocalModels/`

nOpenaiServer is a high-performance local LLM inference engine that provides an OpenAI-compatible API for running models locally without any external dependencies.

## Key Features

- **Zero External Dependencies** - All inference runs locally
- **OpenAI-Compatible API** - Drop-in replacement for OpenAI API
- **GGUF Model Support** - Runs quantized models efficiently
- **Multi-Model Support** - Load and switch between models
- **GPU Acceleration** - Optional CUDA/Metal support
- **High Performance** - Zig/Mojo implementation for maximum speed

## Quick Start

### Running Locally

```bash
# Pull models first
dvc pull

# Run with Docker
docker run -p 11434:11434 \
  -v $(pwd)/models:/models \
  plturrell/nopenaiserver:latest

# Or use docker-compose
docker-compose -f docker/compose/docker-compose.servicecore.yml up nopenaiserver
```

### Test Inference

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [
      {"role": "user", "content": "Write a hello world in Python"}
    ]
  }'
```

## Available Models

### Code Generation
- **Qwen3-Coder-30B-A3B-Instruct** - Primary code generation model
- Size: ~15GB (quantized)
- Context: 32K tokens

### Translation
- **HY-MT1.5-1.8B** - Machine translation
- Size: ~2GB
- Languages: Multiple including Arabic

### Arabic NLP
- **CamelBERT** - Arabic language understanding
- Size: ~500MB
- Specialized for Arabic dialects

### Mathematics
- **DeepSeek-Math** - Mathematical reasoning
- Size: ~7GB
- Specialized for math problems

## API Endpoints

### Chat Completions (OpenAI Compatible)

```bash
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen3-Coder-30B-A3B-Instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Hello!"}
  ],
  "temperature": 0.7,
  "max_tokens": 2000
}
```

### List Models

```bash
GET /v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3-Coder-30B-A3B-Instruct",
      "object": "model",
      "owned_by": "local",
      "permission": []
    }
  ]
}
```

### Health Check

```bash
GET /health
```

### Model Info

```bash
GET /v1/models/{model_name}
```

## Configuration

### Environment Variables

```bash
# Server configuration
NOPENAISERVER_PORT=11434
NOPENAISERVER_HOST=0.0.0.0

# Model configuration
MODEL_BASE_PATH=/models
DEFAULT_MODEL=Qwen3-Coder-30B-A3B-Instruct

# Performance tuning
N_GPU_LAYERS=35              # GPU layers (0 for CPU only)
N_CTX=32768                  # Context window
N_BATCH=512                  # Batch size
N_THREADS=8                  # CPU threads

# SAP HANA Cloud logging
HANA_ODATA_URL=https://...
HANA_USERNAME=DBADMIN
HANA_PASSWORD=...
LOG_TO_HANA=true
```

## Integration Examples

### Python

```python
from openai import OpenAI

# Point to local server
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="not-needed"  # Local server doesn't require auth
)

# Use exactly like OpenAI API
response = client.chat.completions.create(
    model="Qwen3-Coder-30B-A3B-Instruct",
    messages=[
        {"role": "user", "content": "Write a bubble sort in Python"}
    ]
)

print(response.choices[0].message.content)
```

### JavaScript/TypeScript

```typescript
import OpenAI from 'openai';

const client = new OpenAI({
  baseURL: 'http://localhost:11434/v1',
  apiKey: 'not-needed'
});

const response = await client.chat.completions.create({
  model: 'Qwen3-Coder-30B-A3B-Instruct',
  messages: [
    { role: 'user', content: 'Hello!' }
  ]
});

console.log(response.choices[0].message.content);
```

### cURL

```bash
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": false
  }'
```

## Performance Tuning

### CPU-Only Mode

```bash
docker run -p 11434:11434 \
  -e N_GPU_LAYERS=0 \
  -e N_THREADS=16 \
  -v $(pwd)/models:/models \
  plturrell/nopenaiserver:latest
```

### GPU Mode (NVIDIA)

```bash
docker run --gpus all -p 11434:11434 \
  -e N_GPU_LAYERS=35 \
  -v $(pwd)/models:/models \
  plturrell/nopenaiserver:latest
```

### GPU Mode (Apple Silicon)

```bash
docker run -p 11434:11434 \
  -e N_GPU_LAYERS=35 \
  -e USE_METAL=1 \
  -v $(pwd)/models:/models \
  plturrell/nopenaiserver:latest
```

## Model Management

### Adding New Models

1. Add model to DVC:
```bash
dvc add models/new-model.gguf
dvc push
```

2. Update configuration to include new model

3. Restart nOpenaiServer

### Model Selection

Models are selected via the `model` parameter in API calls:

```json
{
  "model": "Qwen3-Coder-30B-A3B-Instruct"
}
```

## Monitoring

### Metrics

Available at `/metrics`:
- `inference_requests_total` - Total inference requests
- `inference_duration_seconds` - Request duration histogram
- `model_load_duration_seconds` - Model load time
- `active_requests` - Current active requests
- `tokens_generated_total` - Total tokens generated

### Logs

All requests logged to SAP HANA Cloud:
```json
{
  "timestamp": "2026-01-22T23:55:00Z",
  "level": "info",
  "service": "nopenaiserver",
  "model": "Qwen3-Coder-30B-A3B-Instruct",
  "prompt_tokens": 20,
  "completion_tokens": 150,
  "duration_ms": 3500
}
```

## Troubleshooting

### Model Not Found

**Problem**: "Model not found" error  
**Solution**:
- Run `dvc pull` to download models
- Verify model path in configuration
- Check MODEL_BASE_PATH environment variable

### Out of Memory

**Problem**: Server crashes with OOM  
**Solution**:
- Reduce N_CTX (context window)
- Reduce N_BATCH (batch size)
- Use smaller quantized model
- Increase Docker memory limit

### Slow Inference

**Problem**: Inference takes too long  
**Solution**:
- Enable GPU acceleration (N_GPU_LAYERS > 0)
- Increase N_THREADS for CPU inference
- Use smaller model
- Reduce N_CTX

### GPU Not Detected

**Problem**: GPU not being used  
**Solution**:
- Verify CUDA/Metal drivers installed
- Check `--gpus all` flag (NVIDIA)
- Set N_GPU_LAYERS > 0
- Check GPU compatibility

## Architecture

See detailed documentation:
- [Authentication Setup](./AUTHENTICATION_SETUP.md)
- Additional nOpenaiServer docs in this directory

## Related Documentation

- [Architecture Overview](../../01-architecture/)
- [DVC Setup](../../02-setup/DVC_SAP_S3_SETUP.md)
- [Model Orchestration](../../01-architecture/MODEL_ORCHESTRATION_MAPPING.md)

---

**Language**: Zig/Mojo  
**Status**: Production  
**Port**: 11434  
**Last Updated**: January 22, 2026
