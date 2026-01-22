# Multi-Model User Guide

**nOpenAI Server - Week 3 Deliverable**  
**Version**: 1.0  
**Date**: January 19, 2026

---

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Model Management](#model-management)
4. [Request Routing](#request-routing)
5. [Resource Management](#resource-management)
6. [Monitoring & Observability](#monitoring--observability)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Introduction

The nOpenAI Server multi-model system enables running multiple Large Language Models (LLMs) simultaneously with intelligent resource management, request routing, and comprehensive observability.

### Key Features

✅ **Unlimited Models**: Run as many models as your hardware supports  
✅ **Smart Routing**: 8 routing strategies (round-robin, least-loaded, cache-aware, etc.)  
✅ **Resource Quotas**: Per-model RAM/SSD/token/request limits  
✅ **Auto-Discovery**: Automatic model detection from filesystem  
✅ **Health Tracking**: Real-time health monitoring and failover  
✅ **Session Affinity**: Sticky routing for stateful applications  
✅ **A/B Testing**: Split traffic for model evaluation  
✅ **Complete Observability**: Structured logging, tracing, metrics

### Architecture Overview

```
Client Request
     ↓
Request Router (8 strategies)
     ↓
Model Registry (health + metadata)
     ↓
Multi-Model Cache (fair allocation)
     ↓
Resource Quotas (limits + enforcement)
     ↓
Selected Model Execution
```

---

## Quick Start

### Prerequisites

- Zig 0.11+ installed
- At least 32GB RAM (for 70B models)
- SSD with 500GB+ free space
- Docker (optional, for monitoring stack)

### Installation

```bash
# Clone repository
git clone https://github.com/plturrell/aArabic.git
cd aArabic/src/serviceCore/nOpenaiServer

# Build server
zig build -Doptimize=ReleaseFast

# Run server
./zig-out/bin/nopenai-server
```

### Add Your First Model

```bash
# Place GGUF model in vendor/layerModels
mkdir -p vendor/layerModels/Llama-3.3-70B-Instruct
cp /path/to/model.gguf vendor/layerModels/Llama-3.3-70B-Instruct/

# Server will auto-discover on startup
# Check logs for: "Model discovered: Llama-3.3-70B-Instruct"
```

### Make Your First Request

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

---

## Model Management

### Model Discovery

The server automatically discovers models from `vendor/layerModels`:

```
vendor/layerModels/
├── Llama-3.3-70B-Instruct/
│   └── model.gguf
├── Qwen-2.5-72B-Instruct/
│   └── model.gguf
└── phi-4/
    └── model.gguf
```

**Discovery Process**:
1. Scans `vendor/layerModels` on startup
2. Extracts metadata (architecture, quantization, size)
3. Registers in Model Registry
4. Sets health status to `unknown`
5. Performs initial health check

### Model Registry API

#### List All Models

```bash
GET /v1/models
```

Response:
```json
{
  "object": "list",
  "data": [
    {
      "id": "Llama-3.3-70B-Instruct",
      "display_name": "Llama-3.3-70B-Instruct",
      "path": "vendor/layerModels/Llama-3.3-70B-Instruct",
      "version": "1.0.0",
      "architecture": "llama",
      "parameter_count": "70B",
      "enabled": true,
      "health_status": "healthy",
      "use_count": 1247,
      "size_bytes": 42949672960,
      "preload": false
    }
  ]
}
```

#### Get Model Details

```bash
GET /v1/models/Llama-3.3-70B-Instruct
```

#### Set Default Model

```bash
POST /v1/models/default
{
  "model_id": "Llama-3.3-70B-Instruct"
}
```

### Model Versioning

Models support semantic versioning (major.minor.patch):

```bash
# Add new version
POST /v1/models/register
{
  "id": "Llama-3.3-70B-Instruct",
  "version": "1.1.0",
  "path": "vendor/layerModels/Llama-3.3-70B-Instruct-v1.1"
}

# List versions
GET /v1/models/Llama-3.3-70B-Instruct/versions
```

### Model Health Status

Health states:
- `unknown`: Initial state, not yet checked
- `healthy`: Fully operational
- `degraded`: Operational but suboptimal
- `unhealthy`: Not operational
- `loading`: Currently loading into memory

Health checks run every 30 seconds and check:
- Model file integrity
- Memory allocation success
- Response time threshold (<5s)
- Error rate (<1%)

---

## Request Routing

### Routing Strategies

The router supports 8 strategies:

#### 1. Round-Robin
Distributes requests evenly across all healthy models.

```json
{
  "routing_config": {
    "strategy": "round_robin"
  }
}
```

**Best for**: Uniform workloads, testing

#### 2. Least-Loaded
Routes to model with lowest current load (RAM utilization).

```json
{
  "routing_config": {
    "strategy": "least_loaded",
    "max_load_threshold": 0.85
  }
}
```

**Best for**: Variable request sizes, peak traffic

#### 3. Cache-Aware (Recommended)
Prefers models with highest cache hit rates.

```json
{
  "routing_config": {
    "strategy": "cache_aware",
    "min_cache_hit_rate": 0.3
  }
}
```

**Best for**: Repeated similar requests, production

#### 4. Quota-Aware
Avoids models approaching quota limits.

```json
{
  "routing_config": {
    "strategy": "quota_aware"
  }
}
```

**Best for**: Rate-limited environments, multi-tenant

#### 5. Random
Random selection across healthy models.

```json
{
  "routing_config": {
    "strategy": "random"
  }
}
```

**Best for**: Load testing, benchmarking

#### 6. Weighted-Random
Random selection weighted by model scores.

```json
{
  "routing_config": {
    "strategy": "weighted_random"
  }
}
```

**Best for**: Gradual model rollouts

#### 7. Latency-Based
Prefers models with lowest latency.

```json
{
  "routing_config": {
    "strategy": "latency_based"
  }
}
```

**Best for**: Latency-sensitive applications

#### 8. Affinity-Based (Sticky Routing)
Routes same session to same model.

```json
{
  "routing_config": {
    "strategy": "affinity_based",
    "affinity_timeout_sec": 300
  }
}
```

**Best for**: Stateful applications, conversations

### Request Routing Examples

#### Basic Request (Auto-Routing)
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Preferred Model
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Llama-3.3-70B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Session Affinity
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: user-abc-123" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

#### Capability-Based Routing
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "capabilities": ["arabic_nlp"],
    "messages": [{"role": "user", "content": "مرحبا"}]
  }'
```

### A/B Testing

Split traffic between two models:

```json
{
  "routing_config": {
    "strategy": "least_loaded",
    "ab_test_enabled": true,
    "ab_test_split": 0.7,
    "ab_test_model_a": "Llama-3.3-70B-Instruct",
    "ab_test_model_b": "Qwen-2.5-72B-Instruct"
  }
}
```

70% of requests go to Model A, 30% to Model B.

### Routing Metrics

Monitor routing decisions:

```bash
GET /metrics/routing
```

Response:
```json
{
  "total_routes": 152847,
  "successful_routes": 152821,
  "failed_routes": 26,
  "fallback_routes": 12,
  "avg_routing_time_us": 0.85,
  "per_strategy": {
    "round_robin": 12483,
    "least_loaded": 89234,
    "cache_aware": 48192,
    "quota_aware": 2891,
    "affinity": 47
  }
}
```

---

## Resource Management

### Per-Model Quotas

Set limits for each model:

```json
{
  "model_id": "Llama-3.3-70B-Instruct",
  "quotas": {
    "max_ram_mb": 81920,
    "max_ssd_gb": 200,
    "hourly_token_limit": 1000000,
    "hourly_request_limit": 10000,
    "daily_token_limit": 20000000,
    "daily_request_limit": 200000,
    "burst_token_limit": 50000,
    "max_concurrent_requests": 50
  }
}
```

### Quota Types

1. **Hourly Quotas**: Reset every hour
2. **Daily Quotas**: Reset at midnight
3. **Burst Quotas**: Maximum tokens in single request
4. **Concurrent Limits**: Maximum simultaneous requests

### Soft Limits

Quotas have soft limits (80-95%) that trigger warnings:

```
80% → INFO log
85% → WARN log
90% → WARN log + throttling begins
95% → ERROR log + aggressive throttling
100% → Hard limit, requests rejected
```

### Graceful Degradation Modes

When quotas exceeded:

1. **Normal**: All systems operational
2. **Soft Limit**: Warnings issued
3. **Throttled**: Request rate reduced
4. **Hard Limit**: Requests queued or rejected
5. **Emergency**: Only priority requests accepted

### Resource Allocation Strategies

Configure how cache is allocated across models:

#### Fair Share (Default)
Equal allocation regardless of usage.

```json
{
  "allocation_strategy": "fair_share"
}
```

#### Proportional
Allocation based on usage patterns.

```json
{
  "allocation_strategy": "proportional"
}
```

#### Priority-Based
Models have priority levels.

```json
{
  "allocation_strategy": "priority_based",
  "model_priorities": {
    "Llama-3.3-70B-Instruct": 10,
    "Qwen-2.5-72B-Instruct": 8,
    "phi-4": 5
  }
}
```

#### Dynamic
Adapts based on real-time demand.

```json
{
  "allocation_strategy": "dynamic"
}
```

### Global Eviction Policies

When cache is full across all models:

1. **LRU**: Evict least recently used globally
2. **LFU**: Evict least frequently used globally
3. **Smallest Model First**: Evict from smallest models
4. **Round-Robin**: Evict from models in rotation

```json
{
  "global_eviction_policy": "lru"
}
```

---

## Monitoring & Observability

### Structured Logging

Logs are in JSON format with multiple levels:

```json
{
  "timestamp": "2026-01-19T08:00:00Z",
  "level": "INFO",
  "module": "request_router",
  "message": "Routed request to model",
  "model_id": "Llama-3.3-70B-Instruct",
  "strategy": "cache_aware",
  "score": 0.92,
  "routing_time_us": 0.8
}
```

**Log Levels**:
- `DEBUG`: Detailed debugging information
- `INFO`: General operational messages
- `WARN`: Warning conditions (soft limits, degraded health)
- `ERROR`: Error conditions (hard limits, failures)
- `FATAL`: Critical failures requiring restart

### Distributed Tracing

Integrated with OpenTelemetry and Jaeger:

```bash
# View traces
open http://localhost:16686
```

**Trace Structure**:
```
request (root span)
├── routing_decision (child span)
├── cache_lookup (child span)
├── quota_check (child span)
└── model_execution (child span)
```

### Prometheus Metrics

Key metrics exposed at `/metrics`:

```
# Routing metrics
nopenai_routing_total{strategy="cache_aware"} 48192
nopenai_routing_duration_seconds{quantile="0.99"} 0.0000032

# Model metrics
nopenai_model_requests_total{model="Llama-3.3-70B-Instruct"} 152847
nopenai_model_health_status{model="Llama-3.3-70B-Instruct"} 1

# Cache metrics
nopenai_cache_hit_rate{model="Llama-3.3-70B-Instruct"} 0.85
nopenai_cache_size_bytes{model="Llama-3.3-70B-Instruct"} 42949672960

# Quota metrics
nopenai_quota_usage_percent{model="Llama-3.3-70B-Instruct",type="hourly"} 67.5
nopenai_quota_remaining_tokens{model="Llama-3.3-70B-Instruct"} 325000
```

### Grafana Dashboards

Pre-configured dashboards available:

1. **Multi-Model Overview**: All models at a glance
2. **Routing Performance**: Routing metrics and decisions
3. **Resource Utilization**: RAM, SSD, quotas per model
4. **Cache Performance**: Hit rates, evictions, allocations
5. **Health Status**: Model health across time

Access at: http://localhost:3000 (default Grafana)

---

## Best Practices

### Model Selection

✅ **DO**:
- Use quantized models (Q4_K_M, Q8_0) for production
- Match model size to available RAM
- Test models before production deployment
- Monitor cache hit rates and adjust strategies

❌ **DON'T**:
- Run FP16/FP32 models on limited memory
- Mix drastically different model sizes without quotas
- Ignore health warnings
- Deploy untested routing strategies

### Routing Strategy Selection

| Use Case | Recommended Strategy | Alternative |
|----------|---------------------|-------------|
| Production API | `cache_aware` | `least_loaded` |
| Multi-tenant | `quota_aware` | `fair_share` allocation |
| Conversations | `affinity_based` | `least_loaded` |
| Load testing | `round_robin` | `random` |
| Model comparison | `ab_test` | Manual switching |
| Latency-critical | `latency_based` | `cache_aware` |

### Resource Configuration

**Small Deployment (1-2 models)**:
```json
{
  "global_cache_size_gb": 100,
  "allocation_strategy": "fair_share",
  "per_model_quota": {
    "max_ram_mb": 40960,
    "hourly_token_limit": 500000
  }
}
```

**Medium Deployment (3-5 models)**:
```json
{
  "global_cache_size_gb": 300,
  "allocation_strategy": "proportional",
  "per_model_quota": {
    "max_ram_mb": 81920,
    "hourly_token_limit": 1000000
  }
}
```

**Large Deployment (5+ models)**:
```json
{
  "global_cache_size_gb": 500,
  "allocation_strategy": "dynamic",
  "global_eviction_policy": "lru",
  "per_model_quota": {
    "max_ram_mb": 81920,
    "hourly_token_limit": 2000000
  }
}
```

### Performance Tuning

1. **Enable Cache-Aware Routing**: +15% hit rate improvement
2. **Set Appropriate Quotas**: Prevent resource exhaustion
3. **Use Session Affinity**: Better cache utilization for conversations
4. **Monitor Eviction Rates**: Adjust cache sizes if too high
5. **Tune Soft Limits**: Balance between warnings and hard limits

### Security

✅ **Enable**:
- Request rate limiting per model
- Quota enforcement
- Health checks with automatic failover
- Structured logging for audit trails
- Authentication/authorization (when available)

❌ **Disable** (in production):
- Debug logging (performance impact)
- Unlimited quotas
- Health check bypass
- Unmonitored deployments

---

## Troubleshooting

### Common Issues

#### Issue: Model Not Discovered

**Symptoms**: Model not in `/v1/models` list

**Solutions**:
1. Check model location: `vendor/layerModels/ModelName/model.gguf`
2. Verify file permissions: `chmod 644 model.gguf`
3. Check logs for discovery errors
4. Restart server to trigger re-discovery
5. Manually register model via API

#### Issue: High Routing Time

**Symptoms**: `avg_routing_time_us > 5.0`

**Solutions**:
1. Reduce number of models (filter unhealthy)
2. Simplify routing strategy (use `round_robin`)
3. Disable quota checks if not needed
4. Check for contention (high concurrent requests)
5. Profile with `perf` or `Instruments`

#### Issue: Quota Exceeded Frequently

**Symptoms**: Many `ERROR` logs with "Quota exceeded"

**Solutions**:
1. Increase quota limits
2. Use `quota_aware` routing strategy
3. Enable request throttling
4. Add more models to distribute load
5. Implement request prioritization

#### Issue: Low Cache Hit Rate

**Symptoms**: `cache_hit_rate < 0.3`

**Solutions**:
1. Use `cache_aware` routing strategy
2. Enable session affinity
3. Increase cache size allocation
4. Check request patterns (too random?)
5. Consider prompt caching strategies

#### Issue: Model Stuck in `loading` State

**Symptoms**: Health status remains `loading` for >5 minutes

**Solutions**:
1. Check available RAM (may be insufficient)
2. Verify model file integrity (`md5sum`)
3. Check system logs for OOM errors
4. Restart model loading process
5. Try smaller quantized version

### Debug Mode

Enable debug logging:

```bash
export NOPENAI_LOG_LEVEL=DEBUG
./zig-out/bin/nopenai-server
```

Debug logs include:
- Every routing decision with scores
- Cache operations (hits/misses/evictions)
- Quota checks and calculations
- Health check details
- Model loading progress

### Health Check Diagnostics

```bash
# Check overall health
GET /health

# Check specific model health
GET /health/model/Llama-3.3-70B-Instruct

# Force health check
POST /health/check
{
  "model_id": "Llama-3.3-70B-Instruct"
}
```

### Performance Profiling

```bash
# Enable profiling
export NOPENAI_PROFILE=1

# Profile routing decisions
curl http://localhost:8080/metrics/routing?detailed=true

# Profile cache operations
curl http://localhost:8080/metrics/cache?detailed=true

# Profile quota checks
curl http://localhost:8080/metrics/quotas?detailed=true
```

---

## API Reference

### Model Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List all models |
| `/v1/models/{id}` | GET | Get model details |
| `/v1/models/register` | POST | Register new model |
| `/v1/models/default` | POST | Set default model |
| `/v1/models/{id}/versions` | GET | List model versions |
| `/v1/models/{id}/health` | GET | Get model health |

### Request Routing

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/chat/completions` | POST | Chat completion with routing |
| `/v1/completions` | POST | Text completion with routing |
| `/v1/embeddings` | POST | Generate embeddings |
| `/routing/config` | GET | Get routing configuration |
| `/routing/config` | POST | Update routing configuration |
| `/routing/stats` | GET | Get routing statistics |

### Resource Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quotas/{model_id}` | GET | Get model quotas |
| `/quotas/{model_id}` | POST | Set model quotas |
| `/quotas/{model_id}/report` | GET | Get quota usage report |
| `/cache/allocation` | GET | Get cache allocation |
| `/cache/allocation` | POST | Update cache allocation |
| `/cache/stats` | GET | Get cache statistics |

### Monitoring

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Overall health status |
| `/health/model/{id}` | GET | Model health status |
| `/metrics` | GET | Prometheus metrics |
| `/metrics/routing` | GET | Routing metrics |
| `/metrics/cache` | GET | Cache metrics |
| `/metrics/quotas` | GET | Quota metrics |

---

## Example Workflows

### Workflow 1: Adding a New Model

```bash
# 1. Place model files
mkdir -p vendor/layerModels/new-model
cp /path/to/model.gguf vendor/layerModels/new-model/

# 2. Wait for auto-discovery (or restart server)
tail -f logs/server.log | grep "Model discovered"

# 3. Verify model registered
curl http://localhost:8080/v1/models | jq '.data[] | select(.id=="new-model")'

# 4. Check model health
curl http://localhost:8080/health/model/new-model

# 5. Test inference
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "new-model", "messages": [{"role": "user", "content": "Test"}]}'
```

### Workflow 2: Configuring Multi-Model Routing

```bash
# 1. List available models
curl http://localhost:8080/v1/models

# 2. Configure cache-aware routing
curl -X POST http://localhost:8080/routing/config \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "cache_aware",
    "enable_health_checks": true,
    "enable_quota_checks": true,
    "enable_cache_optimization": true
  }'

# 3. Make requests and monitor routing
for i in {1..100}; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "auto", "messages": [{"role": "user", "content": "Hello"}]}'
done

# 4. Check routing statistics
curl http://localhost:8080/routing/stats | jq '.per_strategy'
```

### Workflow 3: Setting Up A/B Testing

```bash
# 1. Configure A/B test (70/30 split)
curl -X POST http://localhost:8080/routing/config \
  -H "Content-Type: application/json" \
  -d '{
    "strategy": "least_loaded",
    "ab_test_enabled": true,
    "ab_test_split": 0.7,
    "ab_test_model_a": "Llama-3.3-70B-Instruct",
    "ab_test_model_b": "Qwen-2.5-72B-Instruct"
  }'

# 2. Run test workload
for i in {1..1000}; do
  curl -X POST http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "auto", "messages": [{"role": "user", "content": "Test"}]}'
done

# 3. Compare model performance
curl http://localhost:8080/metrics/cache | \
  jq '.per_model | to_entries | map({model: .key, hit_rate: .value.hit_rate})'

# 4. Select winner and disable A/B test
curl -X POST http://localhost:8080/routing/config \
  -H "Content-Type: application/json" \
  -d '{"ab_test_enabled": false}'
```

---

## Next Steps

1. **Deploy Monitoring Stack**: Set up Prometheus + Grafana + Jaeger
2. **Configure Quotas**: Set appropriate limits for your use case
3. **Test Routing Strategies**: Find optimal strategy for your workload
4. **Enable Auto-Discovery**: Add models to `vendor/layerModels`
5. **Monitor Performance**: Track cache hit rates and routing metrics
6. **Scale Horizontally**: Add more models as needed

## Support

- **Documentation**: https://docs.nopenai.server/
- **Issues**: https://github.com/plturrell/aArabic/issues
- **Community**: https://discord.gg/nopenai

---

**Last Updated**: January 19, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
