# nLocalModels - Kyma Deployment Package

## 📦 Package Contents

Complete Kyma deployment manifests for the nLocalModels LLM inference service.

**Namespace**: `nservicecore`  
**Service**: nLocalModels  
**Image**: `plturrell/nservices:nLocalModels`  
**Port**: 11434  

---

## 📁 Files Overview

### **Core Manifests** (Required)
- `namespace.yaml` - Namespace definition
- `nlocalmodels-secret.yaml` - HANA credentials (⚠️ gitignored)
- `nlocalmodels-configmap.yaml` - Service configuration
- `nlocalmodels-service.yaml` - Kubernetes Service
- `nlocalmodels-apirule.yaml` - External access (Kyma)
- `kustomization.yaml` - Kustomize orchestration

### **Deployment Options** (Choose one)
- `nlocalmodels-deployment.yaml` - **Object Store mode** (default)
- `nlocalmodels-deployment-pvc.yaml` - **PVC mode** (for AI Core)
- `nlocalmodels-pvc.yaml` - PVC definition (for PVC mode)

### **Documentation**
- `README.md` - This file
- `DEPLOYMENT_GUIDE.md` - Complete deployment guide
- `PREREQUISITES.md` - Dependencies and setup
- `STORAGE_OPTIONS.md` - Object Store vs PVC comparison

### **Scripts**
- `setup-hana.sh` - Automated HANA table creation
- `.gitignore` - Protects secrets from Git

---

## 🚀 Quick Start

### **For SAP AI Core** (Recommended: PVC Mode)

```bash
# 1. Run HANA setup (you provide credentials interactively)
./setup-hana.sh

# 2. Update secret with HANA credentials only
vim nlocalmodels-secret.yaml
# Fill in HANA_* fields, leave Object Store fields as "not-used"

# 3. Deploy with PVC mode
kubectl apply -f namespace.yaml
kubectl apply -f nlocalmodels-secret.yaml
kubectl apply -f nlocalmodels-configmap.yaml
kubectl apply -f nlocalmodels-pvc.yaml
kubectl apply -f nlocalmodels-deployment-pvc.yaml
kubectl apply -f nlocalmodels-service.yaml
kubectl apply -f nlocalmodels-apirule.yaml

# 4. Verify
kubectl get pods -n nservicecore -w
```

### **For SAP BTP Kyma** (Recommended: Object Store Mode)

```bash
# 1. Run HANA setup
./setup-hana.sh

# 2. Update secret with HANA + Object Store credentials
vim nlocalmodels-secret.yaml
# Fill in all HANA_* and OBJECT_STORE_* fields

# 3. Deploy with Object Store mode
kubectl apply -f namespace.yaml
kubectl apply -f nlocalmodels-secret.yaml
kubectl apply -f nlocalmodels-configmap.yaml
kubectl apply -f nlocalmodels-deployment.yaml
kubectl apply -f nlocalmodels-service.yaml
kubectl apply -f nlocalmodels-apirule.yaml

# 4. Verify
kubectl get pods -n nservicecore -w
```

---

## 📋 Prerequisites Checklist

### **Critical (Always Required)**
- [ ] SAP HANA Cloud instance running
- [ ] HANA database `NOPENAI_DB` created
- [ ] HANA tables created (run `setup-hana.sh`)
- [ ] HANA credentials obtained
- [ ] Kyma cluster accessible
- [ ] kubectl configured
- [ ] Docker image available

### **For Object Store Mode**
- [ ] SAP Object Store enabled
- [ ] Bucket `nlocalmodels-kv-cache` created
- [ ] Object Store credentials obtained

### **For PVC Mode**
- [ ] Kyma storage class supports ReadWriteMany
- [ ] Sufficient storage quota available (100Gi+)

---

## 📚 Documentation

### **Start Here**
1. **[PREREQUISITES.md](PREREQUISITES.md)** - What you need before deploying
2. **[STORAGE_OPTIONS.md](STORAGE_OPTIONS.md)** - Choose Object Store or PVC
3. **[DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)** - Step-by-step deployment

### **Reference**
- Service source: `../../src/serviceCore/nLocalModels/`
- Dockerfile: `../../docker/Dockerfile.nlocalmodels`
- GitHub workflow: `../../.github/workflows/docker-build-backend.yml`
- Production audit: `../../src/serviceCore/nLocalModels/docs/PRODUCTION_READINESS_AUDIT.md`

---

## 🎯 Deployment Modes Summary

### **Object Store Mode** (Production-grade)
```bash
Files needed:
  ✓ namespace.yaml
  ✓ nlocalmodels-secret.yaml (HANA + Object Store)
  ✓ nlocalmodels-configmap.yaml
  ✓ nlocalmodels-deployment.yaml
  ✓ nlocalmodels-service.yaml
  ✓ nlocalmodels-apirule.yaml

Prerequisites:
  ✓ HANA Cloud
  ✓ Object Store + bucket
  ✓ Both credentials
```

### **PVC Mode** (AI Core / Simple)
```bash
Files needed:
  ✓ namespace.yaml
  ✓ nlocalmodels-secret.yaml (HANA only)
  ✓ nlocalmodels-configmap.yaml
  ✓ nlocalmodels-pvc.yaml
  ✓ nlocalmodels-deployment-pvc.yaml
  ✓ nlocalmodels-service.yaml
  ✓ nlocalmodels-apirule.yaml

Prerequisites:
  ✓ HANA Cloud
  ✓ Kyma storage (RWX)
  ✓ HANA credentials only
```

---

## 🔧 Configuration

### **Credentials** (nlocalmodels-secret.yaml)

**⚠️ IMPORTANT**: This file is gitignored. Update locally only!

```yaml
# For Object Store mode:
HANA_HOST: "your-actual-hana-host"
HANA_PASSWORD: "your-actual-password"
OBJECT_STORE_URL: "your-actual-url"
OBJECT_STORE_ACCESS_KEY: "your-actual-key"

# For PVC mode:
HANA_HOST: "your-actual-hana-host"
HANA_PASSWORD: "your-actual-password"
OBJECT_STORE_URL: "not-used"
OBJECT_STORE_ACCESS_KEY: "not-used"
```

### **Resources** (nlocalmodels-configmap.yaml)

Adjust based on your cluster:
```yaml
MAX_RAM_MB: "8192"        # 50% of pod memory limit
KV_CACHE_RAM_MB: "2048"   # 25% of MAX_RAM_MB
TENSOR_HOT_MB: "2048"     # For hot model layers
```

---

## ✅ Verification

```bash
# Check deployment
kubectl get all -n nservicecore

# Check logs
kubectl logs -n nservicecore -l app=nlocalmodels -f

# Port forward
kubectl port-forward -n nservicecore svc/nlocalmodels 11434:11434

# Test health
curl http://localhost:11434/health

# Test inference
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"llama-7b","messages":[{"role":"user","content":"Hi"}]}'
```

---

## 🆘 Troubleshooting

### Quick Checks
```bash
# Pod status
kubectl get pods -n nservicecore

# Pod logs
kubectl logs -n nservicecore -l app=nlocalmodels --tail=100

# Pod events
kubectl describe pod -n nservicecore -l app=nlocalmodels

# Secret verification
kubectl get secret nlocalmodels-secret -n nservicecore -o yaml

# PVC status (if using PVC mode)
kubectl get pvc -n nservicecore
```

### Common Issues

**CrashLoopBackOff**:
1. Check HANA credentials
2. Verify HANA tables exist
3. Check resource limits
4. Review pod logs

**ImagePullBackOff**:
1. Verify image exists: `docker pull plturrell/nservices:nLocalModels`
2. Check registry credentials (if private)

**Health check failing**:
1. Increase `initialDelaySeconds`
2. Check model loading time
3. Verify port 11434 is correct

---

## 📞 Support

**Need help?**
1. Review [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)
2. Check [PREREQUISITES.md](PREREQUISITES.md)
3. See [STORAGE_OPTIONS.md](STORAGE_OPTIONS.md)
4. Check service logs: `kubectl logs -n nservicecore -l app=nlocalmodels`

---

## 📊 Resource Requirements

| Component | CPU | Memory | Storage |
|-----------|-----|--------|---------|
| Per Pod | 2-4 cores | 8-16Gi | - |
| PVC (optional) | - | - | 100Gi |
| Total (2 replicas) | 4-8 cores | 16-32Gi | 100Gi |

---

## 🎓 Next Steps After Deployment

1. **Access UI5 Dashboard**
   ```
   https://nlocalmodels.your-cluster.kyma.ondemand.com/webapp/
   ```

2. **Monitor Metrics in HANA**
   - Query `INFERENCE_METRICS` table
   - Query `ROUTING_DECISIONS` table
   - Monitor cache hit rates

3. **Test API Endpoints**
   - `/health` - Health check
   - `/v1/chat/completions` - Chat API
   - `/v1/completions` - Completion API
   - `/v1/models` - List models

4. **Scale as Needed**
   ```bash
   kubectl scale deployment nlocalmodels -n nservicecore --replicas=5
   ```

---

**Status**: ✅ Ready for Deployment  
**Last Updated**: January 27, 2026  
**Package Version**: 1.0
