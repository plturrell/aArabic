# Quick Start Guide

Get up and running with serviceCore in under 30 minutes.

## Prerequisites

- Docker Desktop or Docker Engine installed
- Git installed
- Python 3.11+ installed
- 20GB free disk space
- Internet connection for initial setup

## Step 1: Clone Repository

```bash
git clone https://github.com/plturrell/aArabic.git
cd aArabic
```

## Step 2: Configure Environment

```bash
# Copy environment template
cp .env.servicecore.example .env

# The .env file is already configured with:
# - SAP HANA Cloud credentials
# - SAP Object Store credentials  
# - All service configurations
```

## Step 3: Pull Models (DVC)

```bash
# Install DVC
pip install dvc[s3]

# Pull all models from SAP Object Store
dvc pull
```

## Step 4: Start Services

```bash
# Start all serviceCore services
docker-compose -f docker/compose/docker-compose.servicecore.yml up -d

# Check status
docker-compose -f docker/compose/docker-compose.servicecore.yml ps
```

## Step 5: Verify Services

```bash
# Check service registry
curl http://localhost:8100/health

# Check nWebServe (API Gateway)
curl http://localhost:8080/health

# Check nOpenaiServer (LLM)
curl http://localhost:11434/health

# Check nExtract
curl http://localhost:8200/health
```

## Step 6: Test Inference

```bash
# Test local LLM inference
curl -X POST http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-Coder-30B-A3B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Available Services

Once running, you'll have access to:

| Service | Port | Description |
|---------|------|-------------|
| service-registry | 8100 | Service discovery |
| nWebServe | 8080 | API Gateway |
| nOpenaiServer | 11434 | LLM Inference |
| nExtract | 8200 | Document extraction |
| nAudioLab | 8300 | Audio processing |
| nCode | 8400 | Code generation |

## Common Commands

### View Logs
```bash
# All services
docker-compose -f docker/compose/docker-compose.servicecore.yml logs -f

# Specific service
docker logs servicecore_nopenaiserver -f
```

### Stop Services
```bash
docker-compose -f docker/compose/docker-compose.servicecore.yml down
```

### Restart Service
```bash
docker-compose -f docker/compose/docker-compose.servicecore.yml restart nwebserve
```

### Update Services
```bash
# Pull latest changes
git pull

# Rebuild images
docker-compose -f docker/compose/docker-compose.servicecore.yml build

# Restart
docker-compose -f docker/compose/docker-compose.servicecore.yml up -d
```

## Models Available

The platform includes these local models (via DVC):

- **Qwen3-Coder-30B** - Code generation and analysis
- **HY-MT1.5-1.8B** - Translation
- **CamelBERT** - Arabic NLP
- **DeepSeek-Math** - Mathematical reasoning

## Architecture

```
SAP BTP Cloud (HANA + S3)
    ↓ OData/REST
serviceCore Services
    ↓
Local Development
```

## Troubleshooting

### Services Won't Start

**Check Docker:**
```bash
docker ps
docker logs <container-name>
```

**Common issues:**
- Port already in use → Change ports in .env
- Insufficient memory → Increase Docker memory limit
- Network conflicts → Check docker network ls

### Can't Connect to HANA

**Verify connection:**
```bash
curl -u DBADMIN:Initial@1 \
  https://d93a8739-44a8-4845-bef3-8ec724dea2ce.hana.prod-us10.hanacloud.ondemand.com/odata/v4
```

### DVC Pull Fails

**Check credentials:**
```bash
dvc remote list
dvc status --remote
aws s3 ls s3://hcp-055af4b0-2344-40d2-88fe-ddc1c4aad6c5/ --region us-east-1
```

### Model Not Found

**Re-pull models:**
```bash
dvc pull --force
ls -lh models/
```

## Next Steps

Now that you're up and running:

1. **Explore Architecture**: [Architecture Docs](../01-architecture/)
2. **Learn Services**: [Service Documentation](../03-services/)
3. **Start Developing**: [Contributing Guide](../05-development/CONTRIBUTING.md)
4. **Deploy to Production**: [Deployment Guide](../06-deployment/)

## Getting Help

- **Documentation**: [Full docs index](../README.md)
- **Issues**: GitHub Issues
- **Support**: Platform team

---

**Estimated Setup Time**: 30 minutes  
**Difficulty**: Beginner-friendly  
**Last Updated**: January 22, 2026
