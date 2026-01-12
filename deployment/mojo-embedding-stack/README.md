# 🚀 Mojo Embedding Stack - Deployment Package

Complete production-ready deployment package for the Mojo Embedding Service with Redis cache and Qdrant vector database.

## 📦 What's Included

This deployment package provides a **complete, turnkey solution** for deploying a high-performance embedding service with:

- **Mojo Embedding Service** (FastAPI) - 384d + 768d models
- **Redis Distributed Cache** - 1.67ms cached responses
- **Qdrant Vector Database** - Semantic search & RAG
- **Docker + Portainer** - Container orchestration
- **Automated deployment** - One-command setup
- **Health checks & monitoring** - Production-grade observability

## 🎯 Quick Start (5 Minutes)

```bash
# 1. Clone or copy this directory to your server
cd mojo-embedding-stack

# 2. Deploy everything
./deploy.sh

# 3. Test it works
curl http://localhost:8007/health
```

**That's it!** The service is now running with all components.

---

## 📊 System Overview

### **Architecture**

```
┌─────────────────────────────────────┐
│         Applications/Clients         │
└──────────────┬──────────────────────┘
               │ HTTP/JSON
┌──────────────▼──────────────────────┐
│   Mojo Embedding Service (8007)     │
│   • FastAPI HTTP API                │
│   • 2 Models: 384d + 768d           │
│   • Request tracking                │
└──────────┬────────────────────────┬─┘
           │                        │
    ┌──────▼─────┐          ┌───────▼────────┐
    │   Redis    │          │    Qdrant      │
    │   Cache    │          │   Vector DB    │
    │  (6379)    │          │    (6333)      │
    └────────────┘          └────────────────┘
```

### **Performance**

| Metric | Value |
|--------|-------|
| **Startup Time** | ~90 seconds (first-time model download) |
| **Uncached Latency** | 28-108ms (depending on model) |
| **Cached Latency** | 1.67ms (Redis distributed) |
| **Cache Speedup** | 16-64x faster |
| **Throughput** | 40+ texts/second (CPU) |
| **Memory** | ~2GB (with both models loaded) |

### **Models**

| Model | Dimensions | Languages | Use Case |
|-------|-----------|-----------|----------|
| **General** | 384d | 50+ | Multilingual, fast |
| **Financial** | 768d | Arabic | Financial domain |

---

## 🛠️ Installation

### **Prerequisites**

- **Docker** (20.10+)
- **Docker Compose** (2.0+)
- **Portainer** (optional, for UI management)
- **4GB RAM minimum** (8GB recommended)
- **10GB disk space** (for models)

### **Step 1: Copy Package**

```bash
# Copy this entire directory to your server
scp -r mojo-embedding-stack user@server:/path/to/deployment/
```

### **Step 2: Configure (Optional)**

```bash
# Review and adjust settings
cp .env.example .env
nano .env  # Edit ports, memory limits, etc.
```

### **Step 3: Deploy**

```bash
# One-command deployment
./deploy.sh
```

The script will:
1. Check prerequisites
2. Set up environment
3. Build Docker images
4. Start all services
5. Wait for health checks
6. Run verification tests
7. Display connection info

---

## 📝 Configuration

### **Environment Variables (.env)**

```bash
# Service Ports
EMBEDDING_PORT=8007    # Embedding service API
REDIS_PORT=6379        # Redis cache
QDRANT_PORT=6333       # Qdrant vector DB

# Redis Configuration
REDIS_MEMORY=1gb       # Max memory for cache
REDIS_TTL=3600         # Cache TTL in seconds

# Qdrant Configuration
QDRANT_MEMORY=2gb      # Max memory for Qdrant

# Logging
LOG_LEVEL=INFO         # DEBUG, INFO, WARNING, ERROR

# Models (auto-downloaded on first run)
GENERAL_MODEL=paraphrase-multilingual-MiniLM-L12-v2
FINANCIAL_MODEL=CAMeL-Lab/bert-base-arabic-camelbert-mix
```

### **Port Customization**

If default ports conflict, edit `.env` before deploying:

```bash
EMBEDDING_PORT=9007  # Instead of 8007
REDIS_PORT=7379      # Instead of 6379
QDRANT_PORT=7333     # Instead of 6333
```

---

## 🔧 Usage

### **API Endpoints**

**Base URL:** `http://localhost:8007`

#### **1. Single Embedding**

```bash
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "model_type": "general"
  }'
```

Response:
```json
{
  "embedding": [0.1, 0.2, ...],  # 384 floats
  "dimensions": 384,
  "processing_time_ms": 28.5,
  "cached": false,
  "request_id": "abc-123-def"
}
```

#### **2. Batch Embeddings**

```bash
curl -X POST http://localhost:8007/embed/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": ["Text 1", "Text 2", "Text 3"],
    "model_type": "general"
  }'
```

#### **3. Arabic Financial (CamelBERT)**

```bash
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{
    "text": "فاتورة مالية بمبلغ 5000 ريال",
    "model_type": "financial"
  }'
```

Response includes 768-dimensional embedding.

#### **4. Health Check**

```bash
curl http://localhost:8007/health
```

#### **5. Metrics**

```bash
curl http://localhost:8007/metrics
```

Response:
```json
{
  "requests_total": 150,
  "cache_hit_rate": 0.45,
  "average_latency_ms": 15.2,
  "cache_type": "Redis (distributed)",
  "device": "cpu"
}
```

### **Interactive API Docs**

Visit `http://localhost:8007/docs` for Swagger UI with:
- Interactive API testing
- Request/response schemas
- Example payloads
- Authentication (if enabled)

---

## 🎛️ Management

### **Docker Compose Commands**

```bash
# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f mojo-embedding

# Stop all services
docker-compose down

# Stop and remove volumes (clears cache)
docker-compose down -v

# Restart services
docker-compose restart

# Check status
docker-compose ps

# Update and rebuild
docker-compose up -d --build
```

### **Portainer Management**

If Portainer is available (`http://localhost:9000`):

1. **Navigate to Stacks** → `mojo-embedding-stack`
2. **View containers:** See all running services
3. **View logs:** Real-time log streaming
4. **Console access:** Shell into containers
5. **Resource monitoring:** CPU, memory, network
6. **Quick actions:** Start, stop, restart

---

## 🧪 Testing

### **Run Full Test Suite**

```bash
./scripts/test-deployment.sh
```

Tests include:
- Health checks (all services)
- Embedding generation (single + batch)
- Model support (384d + 768d)
- Cache performance (speedup verification)
- Metrics endpoints

### **Manual Tests**

```bash
# Test general model (384d)
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{"text":"test","model_type":"general"}' | python3 -m json.tool

# Test financial model (768d Arabic)
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{"text":"فاتورة","model_type":"financial"}' | python3 -m json.tool

# Test cache speedup
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{"text":"cache test"}' | python3 -c "import json,sys; print('Time:', json.load(sys.stdin)['processing_time_ms'], 'ms')"

# Run again (should be cached)
curl -X POST http://localhost:8007/embed/single \
  -H "Content-Type: application/json" \
  -d '{"text":"cache test"}' | python3 -c "import json,sys; print('Time:', json.load(sys.stdin)['processing_time_ms'], 'ms', '(cached)')"
```

---

## 📊 Monitoring

### **Built-in Metrics**

```bash
# Get current metrics
curl http://localhost:8007/metrics | python3 -m json.tool
```

Metrics include:
- Total requests
- Requests per second
- Average latency
- Cache hit rate
- Cache hits/misses
- Cache size
- Embeddings generated
- Uptime

### **Health Checks**

```bash
# Service health
curl http://localhost:8007/health

# Redis health
docker exec redis-cache redis-cli ping

# Qdrant health
curl http://localhost:6333/readyz
```

### **Logs**

```bash
# All services
docker-compose logs -f

# Embedding service only
docker-compose logs -f mojo-embedding

# Last 100 lines
docker-compose logs --tail=100

# Since specific time
docker-compose logs --since 2h
```

---

## 🔒 Production Considerations

### **Security**

1. **Change default passwords** in `.env`
2. **Enable HTTPS** with reverse proxy (nginx/Caddy)
3. **Add authentication** to API endpoints
4. **Restrict network access** with firewall rules
5. **Use secrets management** for sensitive data

### **Performance Tuning**

```bash
# For high-load scenarios, adjust:

# 1. Redis memory
REDIS_MEMORY=4gb  # Increase cache size

# 2. Qdrant memory
QDRANT_MEMORY=4gb  # More vectors in memory

# 3. Add more embedding service replicas
docker-compose up -d --scale mojo-embedding=3
```

### **Backup & Recovery**

```bash
# Backup Redis data
docker cp redis-cache:/data ./redis-backup

# Backup Qdrant data
docker cp qdrant:/qdrant/storage ./qdrant-backup

# Restore Redis
docker cp ./redis-backup redis-cache:/data
docker-compose restart redis-cache

# Restore Qdrant
docker cp ./qdrant-backup qdrant:/qdrant/storage
docker-compose restart qdrant
```

---

## 🐛 Troubleshooting

### **Port Already in Use**

```bash
# Check what's using the port
lsof -i :8007

# Option 1: Stop conflicting service
# Option 2: Change port in .env
EMBEDDING_PORT=9007
```

### **Service Won't Start**

```bash
# Check logs
docker-compose logs mojo-embedding

# Common issues:
# 1. Out of memory → Increase Docker memory limit
# 2. Port conflict → Change ports in .env
# 3. Model download failed → Check internet connection
```

### **Slow Performance**

```bash
# 1. Check device
curl http://localhost:8007/health | grep device

# 2. Enable GPU (if available)
# Edit docker-compose.yml, add:
#   deploy:
#     resources:
#       reservations:
#         devices:
#           - driver: nvidia
#             count: 1
#             capabilities: [gpu]

# 3. Check cache hit rate
curl http://localhost:8007/metrics | grep cache_hit_rate
# Low hit rate? Consider increasing Redis memory
```

### **Cache Not Working**

```bash
# 1. Check Redis connection
docker exec redis-cache redis-cli ping

# 2. Check cache type in metrics
curl http://localhost:8007/metrics | grep cache_type
# Should show "Redis (distributed)"

# 3. Test cache manually
redis-cli -h localhost -p 6379 KEYS "emb:*"
```

---

## 📚 Documentation

- **API Docs:** http://localhost:8007/docs
- **Health:** http://localhost:8007/health
- **Metrics:** http://localhost:8007/metrics
- **Qdrant UI:** http://localhost:6333/dashboard
- **Portainer:** http://localhost:9000 (if available)

---

## 🎉 Success Criteria

After deployment, you should see:

✅ All services running (`docker-compose ps`)  
✅ Health check passes (`curl http://localhost:8007/health`)  
✅ Embeddings generate (`curl -X POST ... /embed/single`)  
✅ Cache working (2nd call much faster)  
✅ Metrics available (`curl http://localhost:8007/metrics`)  
✅ Qdrant accessible (`curl http://localhost:6333/readyz`)  

---

## 📞 Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Run tests: `./scripts/test-deployment.sh`
3. Review troubleshooting section above
4. Check original documentation in parent repository

---

## 📄 License

This deployment package is part of the Arabic Invoice Processing project.

---

**🚀 Ready to deploy? Run `./deploy.sh` and you're live in 5 minutes!**
