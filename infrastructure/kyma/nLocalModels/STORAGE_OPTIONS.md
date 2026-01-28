# nLocalModels - Storage Options Guide

## 📋 Overview

nLocalModels supports **two storage modes** for KV cache tensors:

1. **Object Store Mode** (SAP Object Store) - Cloud-native, scalable
2. **PVC Mode** (Persistent Volume) - Simple, Kyma-native

Both modes use **HANA Cloud** for metadata. The choice depends on your deployment environment.

---

## 🎯 Storage Architecture

```
┌─────────────────────────────────────────────────────┐
│         nLocalModels Service                         │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐         ┌──────────────┐         │
│  │  KV Cache    │         │  Metadata    │         │
│  │  Manager     │────────▶│  (HANA)      │         │
│  └──────┬───────┘         └──────────────┘         │
│         │                                            │
│         │  Tensor Storage (choose one):             │
│         │                                            │
│         ├──────▶ Option 1: Object Store             │
│         │        (SAP BTP)                           │
│         │                                            │
│         └──────▶ Option 2: PVC                       │
│                  (Kyma Volume)                       │
└─────────────────────────────────────────────────────┘
```

---

## 🔄 Option 1: Object Store Mode (Default)

### **When to Use**
- ✅ Deploying to SAP BTP with Object Store access
- ✅ Need unlimited scalable storage
- ✅ Multi-cluster deployments (shared cache)
- ✅ Production environments with high availability

### **Deployment Files**
```bash
namespace.yaml                    # Namespace
nlocalmodels-secret.yaml         # HANA + Object Store credentials
nlocalmodels-configmap.yaml      # Service configuration
nlocalmodels-deployment.yaml     # Main deployment (Object Store)
nlocalmodels-service.yaml        # Service
nlocalmodels-apirule.yaml        # External access
```

### **Secret Configuration**
Edit `nlocalmodels-secret.yaml`:
```yaml
stringData:
  # HANA Cloud (REQUIRED)
  HANA_HOST: "your-hana.hanacloud.ondemand.com"
  HANA_PORT: "443"
  HANA_DATABASE: "NOPENAI_DB"
  HANA_USER: "NUCLEUS_APP"
  HANA_PASSWORD: "your-password"
  
  # Object Store (REQUIRED for Object Store mode)
  OBJECT_STORE_URL: "https://objectstore.cfapps.eu10.hana.ondemand.com"
  OBJECT_STORE_BUCKET: "nlocalmodels-kv-cache"
  OBJECT_STORE_ACCESS_KEY: "your-access-key"
  OBJECT_STORE_SECRET_KEY: "your-secret-key"
```

### **Deployment**
```bash
# Deploy with Object Store mode
kubectl apply -f namespace.yaml
kubectl apply -f nlocalmodels-secret.yaml  # With Object Store credentials
kubectl apply -f nlocalmodels-configmap.yaml
kubectl apply -f nlocalmodels-deployment.yaml  # Uses Object Store
kubectl apply -f nlocalmodels-service.yaml
kubectl apply -f nlocalmodels-apirule.yaml
```

### **Advantages**
- ✅ Unlimited scalable storage
- ✅ No PVC size limits
- ✅ Shared across clusters
- ✅ Managed by SAP BTP
- ✅ Automatic replication

### **Prerequisites**
- SAP Object Store enabled in BTP
- Bucket created: `nlocalmodels-kv-cache`
- Access credentials obtained

---

## 💾 Option 2: PVC Mode (Simple Alternative)

### **When to Use**
- ✅ Object Store not available/accessible
- ✅ Simple Kyma/Kubernetes deployment
- ✅ Single cluster deployment
- ✅ Development/testing environments
- ✅ **Deploying to SAP AI Core** (uses Kyma PVC)

### **Deployment Files**
```bash
namespace.yaml                      # Namespace
nlocalmodels-secret.yaml           # HANA credentials ONLY
nlocalmodels-configmap.yaml        # Service configuration
nlocalmodels-pvc.yaml              # Persistent Volume Claim (NEW)
nlocalmodels-deployment-pvc.yaml   # Deployment using PVC (NEW)
nlocalmodels-service.yaml          # Service
nlocalmodels-apirule.yaml          # External access
```

### **Secret Configuration**
Edit `nlocalmodels-secret.yaml` (simplified):
```yaml
stringData:
  # HANA Cloud (REQUIRED)
  HANA_HOST: "your-hana.hanacloud.ondemand.com"
  HANA_PORT: "443"
  HANA_DATABASE: "NOPENAI_DB"
  HANA_USER: "NUCLEUS_APP"
  HANA_PASSWORD: "your-password"
  
  # Object Store fields - leave as placeholders (not used in PVC mode)
  OBJECT_STORE_URL: "not-used"
  OBJECT_STORE_BUCKET: "not-used"
  OBJECT_STORE_ACCESS_KEY: "not-used"
  OBJECT_STORE_SECRET_KEY: "not-used"
```

### **Deployment**
```bash
# Deploy with PVC mode
kubectl apply -f namespace.yaml
kubectl apply -f nlocalmodels-secret.yaml  # HANA only
kubectl apply -f nlocalmodels-configmap.yaml
kubectl apply -f nlocalmodels-pvc.yaml  # Create persistent volume
kubectl apply -f nlocalmodels-deployment-pvc.yaml  # Uses PVC
kubectl apply -f nlocalmodels-service.yaml
kubectl apply -f nlocalmodels-apirule.yaml
```

### **Advantages**
- ✅ No Object Store dependency
- ✅ Simpler setup (fewer credentials)
- ✅ Native Kubernetes storage
- ✅ **Works in SAP AI Core**
- ✅ Lower cost (no Object Store charges)

### **Limitations**
- ⚠️ Storage limited by PVC size (default: 100Gi)
- ⚠️ Not shared across clusters
- ⚠️ Requires ReadWriteMany storage class

### **Prerequisites**
- Kyma cluster with persistent storage
- Storage class supporting ReadWriteMany (RWX)

---

## 🔍 Comparison Matrix

| Feature | Object Store Mode | PVC Mode |
|---------|------------------|----------|
| **Setup Complexity** | Medium (needs Object Store) | Simple (Kyma-native) |
| **Storage Capacity** | Unlimited | Limited by PVC (100Gi default) |
| **Scalability** | Excellent | Good (single cluster) |
| **Cross-cluster Sharing** | ✅ Yes | ❌ No |
| **Cost** | Object Store charges | PVC storage charges |
| **SAP AI Core Compatible** | ✅ Yes | ✅ **Yes (Recommended)** |
| **Credentials Required** | HANA + Object Store | HANA only |
| **Storage Location** | SAP Object Store | Kyma Persistent Volume |
| **Best For** | Production, Multi-cluster | Development, AI Core |

---

## 🚀 Deployment Decision Tree

```
Start
  │
  ├─ Deploying to SAP AI Core?
  │   └─ YES ──▶ Use PVC Mode ✅
  │   └─ NO  ──▶ Continue...
  │
  ├─ Have SAP Object Store access?
  │   └─ NO  ──▶ Use PVC Mode ✅
  │   └─ YES ──▶ Continue...
  │
  ├─ Need multi-cluster sharing?
  │   └─ YES ──▶ Use Object Store Mode ✅
  │   └─ NO  ──▶ Continue...
  │
  ├─ Need >100Gi cache storage?
  │   └─ YES ──▶ Use Object Store Mode ✅
  │   └─ NO  ──▶ Either mode works
  │
  └─ Default: Use PVC Mode (simpler) ✅
```

---

## 📝 Migration Between Modes

### **From Object Store to PVC**

```bash
# 1. Scale down deployment
kubectl scale deployment nlocalmodels -n nservicecore --replicas=0

# 2. Create PVC
kubectl apply -f nlocalmodels-pvc.yaml

# 3. Switch deployment
kubectl delete -f nlocalmodels-deployment.yaml
kubectl apply -f nlocalmodels-deployment-pvc.yaml

# 4. Verify
kubectl get pods -n nservicecore -w
```

### **From PVC to Object Store**

```bash
# 1. Scale down deployment
kubectl scale deployment nlocalmodels -n nservicecore --replicas=0

# 2. Update secret with Object Store credentials
kubectl apply -f nlocalmodels-secret.yaml

# 3. Switch deployment
kubectl delete -f nlocalmodels-deployment-pvc.yaml
kubectl apply -f nlocalmodels-deployment.yaml

# 4. Delete PVC (optional)
kubectl delete -f nlocalmodels-pvc.yaml

# 5. Verify
kubectl get pods -n nservicecore -w
```

---

## 🎯 Recommendation by Environment

### **SAP AI Core**
```bash
✅ Use: PVC Mode
Why: AI Core uses Kyma, PVC is native and simpler
```

### **SAP BTP Kyma (Production)**
```bash
✅ Use: Object Store Mode
Why: Scalable, managed, production-grade
```

### **SAP BTP Kyma (Dev/Test)**
```bash
✅ Use: PVC Mode
Why: Simpler, no Object Store setup needed
```

### **On-premises Kubernetes**
```bash
✅ Use: PVC Mode
Why: No SAP Object Store available
```

---

## ⚙️ Configuration Differences

### **Environment Variables**

**Object Store Mode** (`nlocalmodels-deployment.yaml`):
```yaml
env:
- name: STORAGE_MODE
  value: "objectstore"  # Default if not set
- name: OBJECT_STORE_URL
  valueFrom:
    secretKeyRef:
      name: nlocalmodels-secret
      key: OBJECT_STORE_URL
# ... other Object Store credentials
```

**PVC Mode** (`nlocalmodels-deployment-pvc.yaml`):
```yaml
env:
- name: STORAGE_MODE
  value: "pvc"
- name: CACHE_PATH
  value: "/cache"

volumeMounts:
- name: cache
  mountPath: /cache
  subPath: kv-cache

volumes:
- name: cache
  persistentVolumeClaim:
    claimName: nlocalmodels-cache-pvc
```

---

## 🔧 Troubleshooting

### **Object Store Mode Issues**

**Problem**: "Object Store connection failed"
```bash
# Check credentials
kubectl get secret nlocalmodels-secret -n nservicecore -o yaml

# Test from pod
kubectl exec -it nlocalmodels-xxx -n nservicecore -- sh
curl -v ${OBJECT_STORE_URL}
```

### **PVC Mode Issues**

**Problem**: "Volume mount failed"
```bash
# Check PVC status
kubectl get pvc -n nservicecore
kubectl describe pvc nlocalmodels-cache-pvc -n nservicecore

# Check storage class
kubectl get storageclass
```

**Problem**: "ReadWriteMany not supported"
```bash
# Your cluster's storage class may not support RWX
# Option 1: Use ReadWriteOnce and set replicas=1
# Option 2: Use NFS-based storage class
# Option 3: Switch to Object Store mode
```

---

## 📞 Support

**Object Store Mode**:
- Check SAP BTP Object Store service status
- Verify bucket permissions
- Confirm network connectivity from Kyma

**PVC Mode**:
- Check Kyma storage provisioner
- Verify storage class supports ReadWriteMany
- Monitor PVC usage: `kubectl top pvc -n nservicecore`

---

## ✅ Summary

**Both modes are fully supported:**

- **Object Store Mode**: Best for production, scalable, multi-cluster
- **PVC Mode**: Best for AI Core, development, simpler setup

**Same secret file (`nlocalmodels-secret.yaml`) works for both:**
- Object Store mode: Uses all credentials
- PVC mode: Only uses HANA credentials, ignores Object Store fields

**Choose based on your environment and requirements!**

---

**Last Updated**: January 27, 2026  
**Version**: 1.0
