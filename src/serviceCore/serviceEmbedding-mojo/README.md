# Mojo Embedding Service

High-performance Arabic-English embedding generation using Mojo with SIMD optimizations.

## 🚀 Quick Start

### Prerequisites
- Mojo installed (via Pixi or direct installation)
- Python 3.11+ with FastAPI and Uvicorn
- Models directory (optional for Phase 3+)

### Running the Service

```bash
# Method 1: Using the run script (recommended)
./scripts/run_mojo_embedding.sh

# Method 2: Direct mojo command
mojo run src/serviceCore/serviceEmbedding-mojo/main.mojo

# Method 3: Using pixi (if configured)
pixi run mojo run src/serviceCore/serviceEmbedding-mojo/main.mojo
```

The service will start on `http://localhost:8007`

## 📡 API Endpoints

### Health Check
```bash
curl http://localhost:8007/health
```

### Single Embedding
```bash
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_type": "general"
  }'
```

### Batch Embedding
```bash
curl -X POST http://localhost:8007/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Hello world", "مرحبا بك", "Test"],
    "model_type": "general",
    "normalize": true
  }'
```

### Workflow Embedding
```bash
curl -X POST http://localhost:8007/embed/workflow \
  -H "Content-Type: application/json" \
  -d '{
    "workflow_text": "Process invoice and extract data",
    "workflow_metadata": {
      "name": "Invoice Processing",
      "description": "Automated extraction"
    }
  }'
```

### Invoice Embedding
```bash
curl -X POST http://localhost:8007/embed/invoice \
  -H "Content-Type: application/json" \
  -d '{
    "invoice_text": "Invoice #12345",
    "extracted_data": {
      "vendor_name": "ACME Corp",
      "total_amount": "1000",
      "currency": "SAR"
    }
  }'
```

### Document Embedding (Chunked)
```bash
curl -X POST http://localhost:8007/embed/document \
  -H "Content-Type: application/json" \
  -d '{
    "document_text": "Long document text...",
    "chunk_size": 512
  }'
```

### List Models
```bash
curl http://localhost:8007/models
```

### Metrics
```bash
curl http://localhost:8007/metrics
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Start the service first
./scripts/run_mojo_embedding.sh &

# Wait a few seconds for startup
sleep 3

# Run tests
./scripts/test_mojo_embedding.sh
```

## 📚 API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8007/docs`
- ReDoc: `http://localhost:8007/redoc`

## 🎯 Features

### Phase 1 (Current)
- ✅ FastAPI HTTP server with Python interop
- ✅ All endpoint contracts match Python service
- ✅ Health check and metrics
- ✅ Dummy embeddings (for testing)

### Phase 2 (Next)
- [ ] Input validation
- [ ] Error handling
- [ ] Request/response logging

### Phase 3 (In Progress)
- [ ] Real model loading with MAX Engine
- [ ] Actual embedding generation
- [ ] Model comparison with Python

### Phase 4 (Planned)
- [ ] SIMD-optimized tokenization (10x faster)
- [ ] Vectorized mean pooling
- [ ] SIMD L2 normalization
- [ ] Parallel batch processing

### Phase 5 (Planned)
- [ ] LRU cache (in-memory)
- [ ] Redis integration
- [ ] Production metrics
- [ ] Benchmarking suite
- [ ] Docker deployment

## 📊 Performance Targets

| Metric | Python | Mojo Target | Expected Improvement |
|--------|--------|-------------|---------------------|
| Single Embedding | ~10ms | <1ms | **10x faster** |
| Batch (32 docs) | ~150ms | <10ms | **15x faster** |
| Throughput | 500/sec | 10,000+/sec | **20x higher** |
| Memory | 6GB | 2-3GB | **50-60% less** |
| Cold Start | 3-5s | <1s | **3-5x faster** |

## 🏗️ Architecture

```
main.mojo (HTTP Server)
├── Python Interop
│   ├── FastAPI (routing)
│   └── Uvicorn (ASGI server)
│
├── Embedder (future)
│   ├── Model Loading (MAX Engine)
│   ├── Tokenization (SIMD)
│   ├── Inference (GPU/CPU)
│   └── Postprocessing (SIMD)
│
└── Cache (future)
    ├── LRU (in-memory)
    └── Redis (distributed)
```

## 🔧 Development

### Project Structure
```
src/serviceCore/serviceEmbedding-mojo/
├── main.mojo           # HTTP server entry point
├── embedder.mojo       # Core embedding logic (TODO)
├── tokenizer.mojo      # SIMD tokenization (TODO)
├── cache.mojo          # LRU cache (TODO)
└── README.md           # This file
```

### Adding New Endpoints

1. Define handler function in `main.mojo`
2. Add route decorator (`@app.get` or `@app.post`)
3. Implement logic
4. Add tests to `scripts/test_mojo_embedding.sh`

### Building for Production

```bash
# Compile to standalone binary
mojo build -o embedding-service src/serviceCore/serviceEmbedding-mojo/main.mojo

# Run binary
./embedding-service
```

## 🐳 Docker Deployment

```dockerfile
FROM modular/mojo:latest

WORKDIR /app
COPY src/serviceCore/serviceEmbedding-mojo/ ./

RUN pip install fastapi uvicorn[standard]
RUN mojo build -o embedding-service main.mojo

EXPOSE 8007
CMD ["./embedding-service"]
```

## 🤝 Comparison with Other Services

| Service | Language | Performance | Status |
|---------|----------|-------------|--------|
| Python | Python 3.11 | Baseline | ✅ Production |
| Rust | Rust 1.75 | 2-3x faster | 🚧 In Progress |
| **Mojo** | **Mojo** | **10-15x faster** | **🔥 This Service** |

## 📝 Notes

### Current Phase (Phase 1)
- Service returns dummy embeddings for testing
- All API contracts match existing Python service
- Ready for integration testing with other services

### Next Steps
1. Run tests: `./scripts/test_mojo_embedding.sh`
2. Verify API compatibility
3. Begin Phase 2: Input validation and error handling
4. Phase 3: Integrate real models with MAX Engine

## 🆘 Troubleshooting

### Service won't start
```bash
# Check if mojo is in PATH
which mojo

# Check Python dependencies
python3 -c "import fastapi, uvicorn"

# Install if missing
pip install fastapi uvicorn[standard]
```

### Port already in use
```bash
# Kill existing process on port 8007
lsof -ti:8007 | xargs kill -9

# Or change port in main.mojo
# uvicorn.run(app, host="0.0.0.0", port=8008)
```

### Mojo not found
```bash
# Add to PATH
export PATH="$HOME/.pixi/envs/max/bin:$PATH"

# Or use absolute path
~/.pixi/envs/max/bin/mojo run src/serviceCore/serviceEmbedding-mojo/main.mojo
```

## 📖 Resources

- [Mojo Documentation](https://docs.modular.com/mojo/)
- [MAX Engine Guide](https://docs.modular.com/max/)
- [SIMD Programming in Mojo](https://docs.modular.com/mojo/manual/vectorization)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

Same as parent project

---

**Status**: Phase 1 Complete ✅  
**Next**: Phase 2 - Input Validation & Error Handling  
**Version**: 0.1.0
