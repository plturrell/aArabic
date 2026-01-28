# nLocalModels - Kyma Deployment Guide

## 📋 Overview

This guide provides step-by-step instructions to deploy the nLocalModels LLM inference service to SAP Kyma.

**Service**: nLocalModels  
**Namespace**: nservicecore  
**Image**: `plturrell/nservices:nLocalModels`  
**Port**: 11434  

---

## 🔧 Prerequisites

### 1. **SAP BTP Infrastructure** (Critical)

#### A. SAP HANA Cloud Instance
```bash
# Required for:
- KV cache metadata storage
- Metrics and monitoring data
- Session state persistence
- Routing decisions

# Database Requirements:
Database Name: NOPENAI_DB
User: NUCLEUS_APP (or custom)
Tables Required:
  - KV_CACHE_METADATA
  - PROMPT_CACHE
  - SESSION_STATE
  - ROUTING_DECISIONS
  - INFERENCE_METRICS
```

**Setup Steps**:
1. Provision HANA Cloud instance in SAP BTP Cockpit
2. Create database `NOPENAI_DB`
3. Run table creation scripts (see `src/serviceCore/nLocalModels/database/`)
4. Create user with appropriate permissions
5. Note connection details (host, port, credentials)

#### B. SAP Object Store
```bash
# Required for:
- KV cache tensor storage (large binary data)
- Model weights storage (optional)

# Bucket Requirements:
Bucket Name: nlocalmodels-kv-cache
Access: Private (credentials required)
Region: Same as Kyma cluster for low latency
```

**Setup Steps**:
1. Enable Object Store service in SAP BTP
2. Create bucket `nlocalmodels-kv-cache`
3. Generate access credentials (access key + secret key)
4. Note endpoint URL

#### C. SAP AI Core (Optional)
```bash
# Required only if using:
- AI Core deployment management
- Model lifecycle automation
- Centralized inference tracking
```

### 2. **Kyma Cluster Access**

```bash
# Verify cluster access
kubectl get nodes
kubectl get ns

# Minimum cluster requirements:
- Kubernetes 1.24+
- Kyma 2.x
- Istio service mesh enabled
- 16GB+ RAM available per node
- 4+ CPU cores per node
```

### 3. **Container Registry Access**

```bash
# Verify image availability
docker pull plturrell/nservices:nLocalModels

# Or check GitHub Actions workflow:
# .github/workflows/docker-build-backend.yml
```

### 4. **Required Tools**

```bash
# Install required CLI tools
kubectl version --client
kustomize version

# Optional but recommended
kubectx  # For switching contexts
kubens   # For switching namespaces
k9s      # Kubernetes UI
```

### 5. **LLM Model Files** (Optional)

**Option A**: Models baked into Docker image (current)
- Models included during build
- Larger image size (~5-10GB)
- Fast startup

**Option B**: Download at startup (future enhancement)
- Smaller image
- Slower first startup
- Requires init container

**Option C**: Persistent Volume (for large models)
- Shared across pods
- Requires PVC setup
- Good for >10GB models

---

## 📝 Configuration Steps

### Step 1: Update Secrets

Edit `nlocalmodels-secret.yaml` with your actual credentials:

```yaml
stringData:
  # HANA Cloud Connection (REQUIRED)
  HANA_HOST: "abc123-xyz.hanacloud.ondemand.com"
  HANA_PORT: "443"
  HANA_DATABASE: "NOPENAI_DB"
  HANA_USER: "NUCLEUS_APP"
  HANA_PASSWORD: "your-actual-password"
  
  # SAP Object Store (REQUIRED)
  OBJECT_STORE_URL: "https://objectstore.cfapps.eu10.hana.ondemand.com"
  OBJECT_STORE_BUCKET: "nlocalmodels-kv-cache"
  OBJECT_STORE_ACCESS_KEY: "your-actual-access-key"
  OBJECT_STORE_SECRET_KEY: "your-actual-secret-key"
  
  # SAP AI Core (OPTIONAL - only if using AI Core)
  AI_CORE_URL: "https://api.ai.eu-central-1.aws.ml.hana.ondemand.com"
  AI_CORE_CLIENT_ID: "your-client-id"
  AI_CORE_CLIENT_SECRET: "your-client-secret"
```

**⚠️ Security Note**: Never commit real credentials to Git!

### Step 2: Adjust Resource Limits

Edit `nlocalmodels-deployment.yaml` resources based on your cluster:

```yaml
# For small clusters (16GB RAM nodes)
resources:
  requests:
    memory: "4Gi"
    cpu: "1000m"
  limits:
    memory: "8Gi"
    cpu: "2000m"

# For medium clusters (32GB RAM nodes) - DEFAULT
resources:
  requests:
    memory: "8Gi"
    cpu: "2000m"
  limits:
    memory: "16Gi"
    cpu: "4000m"

# For large clusters (64GB+ RAM nodes)
resources:
  requests:
    memory: "16Gi"
    cpu: "4000m"
  limits:
    memory: "32Gi"
    cpu: "8000m"
```

### Step 3: Configure Memory Tiers

Edit `nlocalmodels-configmap.yaml`:

```yaml
data:
  # Adjust based on pod memory limits
  MAX_RAM_MB: "8192"        # 50% of pod memory limit
  KV_CACHE_RAM_MB: "2048"   # 25% of MAX_RAM_MB
  TENSOR_HOT_MB: "2048"     # 25% of MAX_RAM_MB
  TENSOR_WARM_MB: "2048"    # 25% of MAX_RAM_MB
```

**Memory Planning Guidelines**:
- `MAX_RAM_MB` = 50% of pod memory limit
- `KV_CACHE_RAM_MB` = 25% of `MAX_RAM_MB`
- `TENSOR_HOT_MB` = 25% of `MAX_RAM_MB`
- `TENSOR_WARM_MB` = remaining
- Leave 50% for OS, buffers, overhead

### Step 4: GPU Configuration (Optional)

If deploying to GPU nodes, uncomment in `nlocalmodels-deployment.yaml`:

```yaml
# In ConfigMap:
ENABLE_GPU: "true"

# In Deployment:
resources:
  limits:
    nvidia.com/gpu: "1"  # Request 1 GPU

nodeSelector:
  gpu: "true"

tolerations:
- key: nvidia.com/gpu
  operator: Exists
  effect: NoSchedule
```

### Step 5: Update APIRule Host

Edit `nlocalmodels-apirule.yaml` for your domain:

```yaml
spec:
  host: nlocalmodels.your-kyma-cluster.kyma.ondemand.com
```

---

## 🚀 Deployment Steps

### Option A: Deploy with Kustomize (Recommended)

```bash
# 1. Navigate to deployment directory
cd /Users/karthikeyan/git/aArabic/infrastructure/kyma/nLocalModels

# 2. Preview deployment
kubectl kustomize .

# 3. Apply all resources
kubectl apply -k .

# 4. Verify deployment
kubectl get all -n nservicecore
```

### Option B: Deploy Manually

```bash
# 1. Create namespace
kubectl apply -f namespace.yaml

# 2. Create secrets (IMPORTANT: Update credentials first!)
kubectl apply -f nlocalmodels-secret.yaml

# 3. Create configmap
kubectl apply -f nlocalmodels-configmap.yaml

# 4. Create deployment
kubectl apply -f nlocalmodels-deployment.yaml

# 5. Create service
kubectl apply -f nlocalmodels-service.yaml

# 6. Create APIRule (for external access)
kubectl apply -f nlocalmodels-apirule.yaml
```

---

## ✅ Verification

### 1. Check Pod Status

```bash
# Watch pods starting
kubectl get pods -n nservicecore -w

# Expected output:
# NAME                            READY   STATUS    RESTARTS   AGE
# nlocalmodels-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
# nlocalmodels-xxxxxxxxxx-xxxxx   1/1     Running   0          2m
```

### 2. Check Pod Logs

```bash
# View logs
kubectl logs -n nservicecore -l app=nlocalmodels --tail=100 -f

# Expected log entries:
# ✓ HANA connection pool initialized (5 connections)
# ✓ Object Store client configured
# ✓ Model loaded: /app/models/llama-7b.gguf
# ✓ GPU detected: NVIDIA Tesla T4
# ✓ Server listening on :11434
```

### 3. Check Service

```bash
# Get service details
kubectl get svc -n nservicecore nlocalmodels

# Port forward for testing
kubectl port-forward -n nservicecore svc/nlocalmodels 11434:11434
```

### 4. Test Health Endpoint

```bash
# From port-forward
curl http://localhost:11434/health

# Expected response:
{
  "status": "healthy",
  "uptime_seconds": 120,
  "hana_connected": true,
  "object_store_connected": true,
  "models_loaded": 1
}
```

### 5. Test Inference

```bash
# Chat completion
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama-7b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 6. Check APIRule

```bash
# Get APIRule status
kubectl get apirule -n nservicecore nlocalmodels

# Get external URL
kubectl get apirule -n nservicecore nlocalmodels -o jsonpath='{.status.virtualService.gateway}'

# Test external access
curl https://nlocalmodels.your-cluster.kyma.ondemand.com/health
```

---

## 🔍 Monitoring

### 1. View Real-time Metrics

```bash
# Pod resource usage
kubectl top pods -n nservicecore

# Detailed pod metrics
kubectl describe pod -n nservicecore -l app=nlocalmodels
```

### 2. Access UI5 Dashboard

The service includes a built-in UI5 dashboard at:
```
https://nlocalmodels.your-cluster.kyma.ondemand.com/webapp/
```

Features:
- Real-time inference metrics
- KV cache hit rates
- Model performance charts
- Request latency histograms
- Error rates and logs

### 3. Query HANA Metrics

```sql
-- Connect to HANA Cloud and run:

-- Request volume (last hour)
SELECT COUNT(*) as total_requests
FROM ROUTING_DECISIONS
WHERE CREATED_AT > ADD_HOURS(CURRENT_TIMESTAMP, -1);

-- Average latency
SELECT AVG(LATENCY_MS) as avg_latency_ms
FROM ROUTING_DECISIONS
WHERE SUCCESS = TRUE
  AND CREATED_AT > ADD_DAYS(CURRENT_TIMESTAMP, -1);

-- Cache hit rate
SELECT 
  COUNT(CASE WHEN CACHE_HIT = TRUE THEN 1 END) * 100.0 / COUNT(*) as hit_rate
FROM KV_CACHE_METADATA
WHERE CREATED_AT > ADD_HOURS(CURRENT_TIMESTAMP, -1);
```

---

## 🐛 Troubleshooting

### Issue: Pods CrashLoopBackOff

```bash
# Check logs
kubectl logs -n nservicecore -l app=nlocalmodels --previous

# Common causes:
# 1. HANA connection failed
#    - Verify HANA_HOST, HANA_PASSWORD in secret
#    - Check network connectivity
#    - Verify HANA instance is running

# 2. Object Store connection failed
#    - Verify OBJECT_STORE credentials
#    - Check bucket exists

# 3. Out of memory
#    - Increase memory limits
#    - Reduce MAX_RAM_MB in ConfigMap

# 4. Model files missing
#    - Verify model baked into image
#    - Or add init container to download
```

### Issue: Health Check Failing

```bash
# Check pod events
kubectl describe pod -n nservicecore -l app=nlocalmodels

# Common causes:
# 1. Port 11434 not listening
#    - Check application logs
#    - Verify PORT env var

# 2. Health endpoint timeout
#    - Increase initialDelaySeconds
#    - Check if model loading is slow
```

### Issue: External Access Not Working

```bash
# Check APIRule status
kubectl get apirule -n nservicecore nlocalmodels -o yaml

# Verify Istio gateway
kubectl get gateway -n kyma-system kyma-gateway

# Check Istio virtual service
kubectl get virtualservice -n nservicecore

# Common causes:
# 1. APIRule not ready
#    - Wait for status to be OK
#    - Check Kyma API Gateway controller logs

# 2. DNS not resolving
#    - Verify host matches cluster domain
#    - Check DNS propagation
```

### Issue: High Memory Usage

```bash
# Check memory usage
kubectl top pod -n nservicecore -l app=nlocalmodels

# Solutions:
# 1. Reduce memory limits in ConfigMap
# 2. Enable more aggressive cache eviction
# 3. Reduce MAX_SEQ_LEN
# 4. Use smaller models
```

### Issue: Slow Inference

```bash
# Check logs for performance metrics
kubectl logs -n nservicecore -l app=nlocalmodels | grep "latency"

# Solutions:
# 1. Enable GPU (set ENABLE_GPU=true)
# 2. Increase TENSOR_HOT_MB
# 3. Use quantized models (Q4_0, Q8_0)
# 4. Enable prompt caching
# 5. Check HANA query performance
```

---

## 📊 Scaling

### Horizontal Scaling

```bash
# Scale up
kubectl scale deployment -n nservicecore nlocalmodels --replicas=5

# Scale down
kubectl scale deployment -n nservicecore nlocalmodels --replicas=1

# Auto-scaling (create HPA)
kubectl autoscale deployment nlocalmodels \
  -n nservicecore \
  --cpu-percent=70 \
  --min=2 \
  --max=10
```

### Vertical Scaling

```bash
# Edit deployment to increase resources
kubectl edit deployment -n nservicecore nlocalmodels

# Update memory/CPU requests and limits
# Pods will be recreated with new limits
```

---

## 🔄 Updates

### Update Docker Image

```bash
# After new image is built and pushed:

# Option 1: Using kubectl
kubectl set image deployment/nlocalmodels \
  -n nservicecore \
  nlocalmodels=plturrell/nservices:nLocalModels-new-tag

# Option 2: Update kustomization.yaml
# Edit images.newTag, then:
kubectl apply -k .

# Monitor rollout
kubectl rollout status deployment/nlocalmodels -n nservicecore
```

### Update Configuration

```bash
# Edit ConfigMap
kubectl edit configmap -n nservicecore nlocalmodels-config

# Restart pods to pick up changes
kubectl rollout restart deployment/nlocalmodels -n nservicecore
```

### Update Secrets

```bash
# Edit Secret
kubectl edit secret -n nservicecore nlocalmodels-secret

# Restart pods
kubectl rollout restart deployment/nlocalmodels -n nservicecore
```

---

## 🗑️ Cleanup

### Remove All Resources

```bash
# Using kustomize
kubectl delete -k .

# Or manually
kubectl delete namespace nservicecore

# Verify deletion
kubectl get all -n nservicecore
```

---

## 📚 Additional Resources

### Documentation
- [nLocalModels README](../../../src/serviceCore/nLocalModels/README.md)
- [Production Readiness Audit](../../../src/serviceCore/nLocalModels/docs/PRODUCTION_READINESS_AUDIT.md)
- [Dockerfile](../../../docker/Dockerfile.nlocalmodels)

### SAP BTP Documentation
- [SAP HANA Cloud](https://help.sap.com/docs/HANA_CLOUD)
- [SAP Object Store](https://help.sap.com/docs/ObjectStore)
- [SAP Kyma Runtime](https://help.sap.com/docs/BTP/65de2977205c403bbc107264b8eccf4b/468c2f3c3ca24c2c8497ef9f83154c44.html)

### Kubernetes Resources
- [Deployments](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/)
- [ConfigMaps](https://kubernetes.io/docs/concepts/configuration/configmap/)
- [Secrets](https://kubernetes.io/docs/concepts/configuration/secret/)

---

## 📞 Support

For issues or questions:
1. Check logs: `kubectl logs -n nservicecore -l app=nlocalmodels`
2. Review troubleshooting section above
3. Check HANA Cloud connectivity
4. Verify Object Store access
5. Review production readiness audit document

---

**Last Updated**: January 27, 2026  
**Version**: 1.0  
**Status**: Production Ready ✅
