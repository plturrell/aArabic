# nLocalModels - Model Management Guide

## 🎯 Current Situation

**Dockerfile Analysis**:
```dockerfile
# Creates empty models directory
RUN mkdir -p /app/models && chown -R appuser:appuser /app

# Service expects models at:
/app/models/
```

**Problem**: ❌ No models are included in the Docker image!  
**Solution**: ✅ Three options available (choose one)

---

## 📦 Model Storage Options

### **Option 1: Init Container Download** ✅ (Recommended for AI Core)

**How it works**:
- Init container downloads models before main container starts
- Models stored in shared emptyDir volume
- Downloaded once per pod startup

**Advantages**:
- ✅ Small Docker image
- ✅ Flexible (change models without rebuilding)
- ✅ Works in any Kubernetes/Kyma cluster
- ✅ No external storage needed

**Disadvantages**:
- ⚠️ Slower first startup (5-10 minutes)
- ⚠️ Downloads on every pod restart
- ⚠️ Requires internet access from pods

**Implementation**: See deployment file below

---

### **Option 2: Persistent Volume (PVC)** ✅ (Best for Production)

**How it works**:
- Models stored in Kyma PVC
- Mounted to all pods (ReadWriteMany)
- Downloaded once, used by all pods

**Advantages**:
- ✅ Fast startup (models already present)
- ✅ Shared across all pods
- ✅ Survives pod restarts
- ✅ One-time download

**Disadvantages**:
- ⚠️ Requires ReadWriteMany storage class
- ⚠️ Additional storage costs
- ⚠️ Manual model upload needed

**Implementation**: See PVC deployment below

---

### **Option 3: Bake into Docker Image** (Not Recommended)

**How it works**:
- Models copied into image during build
- Part of the image layers

**Advantages**:
- ✅ Fastest startup
- ✅ No runtime dependencies

**Disadvantages**:
- ❌ Huge image size (5-20GB per model)
- ❌ Slow builds and deployments
- ❌ Must rebuild for model updates
- ❌ Large registry storage costs

**Not recommended** - Use Option 1 or 2 instead

---

## 🚀 Implementation

### **Option 1: Init Container Deployment**

Create: `nlocalmodels-deployment-with-init.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nlocalmodels
  namespace: nservicecore
spec:
  replicas: 2
  selector:
    matchLabels:
      app: nlocalmodels
  template:
    metadata:
      labels:
        app: nlocalmodels
    spec:
      # Init container downloads models before main container starts
      initContainers:
      - name: model-downloader
        image: curlimages/curl:latest
        command:
        - sh
        - -c
        - |
          echo "Downloading LLM models..."
          
          # Download llama-2-7b-chat (Q4_K_M quantized - ~4GB)
          if [ ! -f /models/llama-2-7b-chat.Q4_K_M.gguf ]; then
            echo "Downloading llama-2-7b-chat..."
            curl -L -o /models/llama-2-7b-chat.Q4_K_M.gguf \
              https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
          else
            echo "Model already exists, skipping download"
          fi
          
          # Download gemma-2-2b-it (Q4_K_M - ~1.5GB)
          if [ ! -f /models/gemma-2-2b-it.Q4_K_M.gguf ]; then
            echo "Downloading gemma-2-2b-it..."
            curl -L -o /models/gemma-2-2b-it.Q4_K_M.gguf \
              https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf
          else
            echo "Model already exists, skipping download"
          fi
          
          echo "Model download complete!"
          ls -lh /models/
        volumeMounts:
        - name: models
          mountPath: /models
      
      containers:
      - name: nlocalmodels
        image: plturrell/nservices:nLocalModels
        imagePullPolicy: Always
        
        env:
        - name: MODEL_PATH
          value: "/app/models"
        # ... other env vars from configmap/secret
        
        volumeMounts:
        - name: models
          mountPath: /app/models
        
        # ... ports, health checks, etc.
      
      volumes:
      - name: models
        emptyDir:
          sizeLimit: 20Gi  # Adjust based on model sizes
```

**Deployment**:
```bash
kubectl apply -f nlocalmodels-deployment-with-init.yaml
```

---

### **Option 2: Persistent Volume for Models**

#### Step 1: Create Models PVC

Create: `nlocalmodels-models-pvc.yaml`

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nlocalmodels-models-pvc
  namespace: nservicecore
spec:
  accessModes:
  - ReadWriteMany  # Shared across pods
  resources:
    requests:
      storage: 50Gi  # Adjust based on model count
  storageClassName: default
```

#### Step 2: Upload Models to PVC

```bash
# Create temporary pod to upload models
kubectl apply -f - <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: model-uploader
  namespace: nservicecore
spec:
  containers:
  - name: uploader
    image: curlimages/curl:latest
    command: ["sleep", "3600"]
    volumeMounts:
    - name: models
      mountPath: /models
  volumes:
  - name: models
    persistentVolumeClaim:
      claimName: nlocalmodels-models-pvc
EOF

# Wait for pod to be ready
kubectl wait --for=condition=ready pod/model-uploader -n nservicecore --timeout=60s

# Download models into PVC
kubectl exec -n nservicecore model-uploader -- sh -c '
  cd /models
  
  # Download llama-2-7b-chat
  curl -L -o llama-2-7b-chat.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf
  
  # Download gemma-2-2b-it
  curl -L -o gemma-2-2b-it.Q4_K_M.gguf \
    https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf
  
  ls -lh
'

# Verify models uploaded
kubectl exec -n nservicecore model-uploader -- ls -lh /models

# Delete uploader pod
kubectl delete pod model-uploader -n nservicecore
```

#### Step 3: Use PVC in Deployment

Update deployment to mount models PVC:

```yaml
# In nlocalmodels-deployment.yaml or nlocalmodels-deployment-pvc.yaml
spec:
  template:
    spec:
      containers:
      - name: nlocalmodels
        volumeMounts:
        - name: models
          mountPath: /app/models
          readOnly: true
      
      volumes:
      - name: models
        persistentVolumeClaim:
          claimName: nlocalmodels-models-pvc
```

---

## 🔍 Model Loading Mechanism

Based on the service architecture:

```zig
// Service startup (simplified)
pub fn main() !void {
    // 1. Read MODEL_PATH from environment
    const model_path = std.os.getenv("MODEL_PATH") orelse "/app/models";
    
    // 2. List available models
    var dir = try std.fs.openDirAbsolute(model_path, .{ .iterate = true });
    defer dir.close();
    
    // 3. Load GGUF models
    var iter = dir.iterate();
    while (try iter.next()) |entry| {
        if (std.mem.endsWith(u8, entry.name, ".gguf")) {
            // Load model into memory
            const model = try loadGGUFModel(model_path, entry.name);
            try model_registry.register(model);
        }
    }
    
    // 4. Start HTTP server
    try startServer(11434);
}
```

**Model Discovery**:
- Scans `/app/models/` directory
- Loads all `.gguf` files
- Registers in model registry
- Makes available via API

---

## 📊 Model Size Reference

| Model | Quantization | Size | RAM | Use Case |
|-------|-------------|------|-----|----------|
| Gemma 2B | Q4_K_M | 1.5GB | 2GB | Fast, small tasks |
| LLaMA 2 7B | Q4_K_M | 4GB | 6GB | Balanced |
| LLaMA 2 7B | Q8_0 | 7GB | 10GB | Higher quality |
| LLaMA 2 13B | Q4_K_M | 7.5GB | 10GB | Better reasoning |
| LLaMA 2 70B | Q4_K_M | 40GB | 50GB+ | Best quality |

**Quantization Guide**:
- `Q4_K_M` - Good balance (recommended)
- `Q8_0` - Higher quality, 2x size
- `F16` - Full precision, 4x size

---

## 🎯 Recommended Approach

### **For SAP AI Core** ✅

**Use: Init Container + emptyDir**

Why:
- ✅ Simple setup
- ✅ No PVC complexity
- ✅ Works in restricted environments
- ✅ Models always fresh

Trade-off:
- ⚠️ 5-10 minute startup per pod
- ⚠️ Downloads on every restart

### **For Production Kyma** ✅

**Use: PVC with pre-uploaded models**

Why:
- ✅ Fast startup (<1 minute)
- ✅ Shared across all pods
- ✅ No repeated downloads
- ✅ Persistent storage

Trade-off:
- ⚠️ One-time manual upload needed
- ⚠️ Storage costs

---

## 🛠️ Model Management Commands

### **List Models in Running Pod**

```bash
# Check what models are loaded
kubectl exec -n nservicecore -it $(kubectl get pod -n nservicecore -l app=nlocalmodels -o jsonpath='{.items[0].metadata.name}') -- ls -lh /app/models/
```

### **Add New Model to PVC**

```bash
# 1. Create uploader pod (if not exists)
kubectl run model-uploader -n nservicecore \
  --image=curlimages/curl:latest \
  --restart=Never \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "uploader",
      "image": "curlimages/curl:latest",
      "command": ["sleep", "3600"],
      "volumeMounts": [{
        "name": "models",
        "mountPath": "/models"
      }]
    }],
    "volumes": [{
      "name": "models",
      "persistentVolumeClaim": {
        "claimName": "nlocalmodels-models-pvc"
      }
    }]
  }
}'

# 2. Download new model
kubectl exec -n nservicecore model-uploader -- sh -c '
  cd /models
  curl -L -o mistral-7b.Q4_K_M.gguf \
    https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf
'

# 3. Restart deployment to load new model
kubectl rollout restart deployment/nlocalmodels -n nservicecore
```

### **Check Available Models via API**

```bash
# Port forward
kubectl port-forward -n nservicecore svc/nlocalmodels 11434:11434

# List models
curl http://localhost:11434/v1/models

# Expected response:
{
  "object": "list",
  "data": [
    {"id": "llama-2-7b-chat", "object": "model"},
    {"id": "gemma-2-2b-it", "object": "model"}
  ]
}
```

---

## 🎓 Recommended Model Sets

### **Starter Set** (6GB total)
```bash
- gemma-2-2b-it.Q4_K_M.gguf          # 1.5GB - Fast, small tasks
- llama-2-7b-chat.Q4_K_M.gguf        # 4GB - General purpose
```

### **Production Set** (15GB total)
```bash
- gemma-2-2b-it.Q4_K_M.gguf          # 1.5GB - Quick responses
- llama-2-7b-chat.Q4_K_M.gguf        # 4GB - General chat
- mistral-7b-instruct.Q4_K_M.gguf    # 4GB - Instruction following
- llama-2-13b-chat.Q4_K_M.gguf       # 7.5GB - Better reasoning
```

### **Enterprise Set** (30GB+ total)
```bash
- Multiple 7B models for different tasks
- One 13B model for complex reasoning
- Specialized models (code, math, etc.)
```

---

## 📝 Model Configuration

### **Environment Variable**

In ConfigMap (`nlocalmodels-configmap.yaml`):
```yaml
data:
  MODEL_PATH: "/app/models"  # Models directory
  DEFAULT_MODEL: "llama-2-7b-chat"  # Default model name
```

### **Model Discovery**

The service automatically:
1. Scans `MODEL_PATH` directory
2. Loads all `.gguf` files
3. Registers in model registry
4. Exposes via `/v1/models` API

---

## 🔧 Implementation Files

I'll create two deployment variants with model management:

1. **`nlocalmodels-deployment-with-init.yaml`** - Init container download
2. **`nlocalmodels-models-pvc.yaml`** - Separate PVC for models
3. **`model-upload-job.yaml`** - One-time upload job

---

## 🎯 Decision Matrix

| Factor | Init Container | Models PVC |
|--------|---------------|------------|
| **Startup Time** | 5-10 min (first) | <1 min |
| **Storage Cost** | None (emptyDir) | PVC charges |
| **Setup Complexity** | Low | Medium |
| **Model Updates** | Edit deployment | Upload to PVC |
| **Pod Restarts** | Re-downloads | Persists |
| **Best For** | AI Core, Dev | Production |

---

## 📞 Model Sources

### **Hugging Face Hub** (Primary)
```bash
# LLaMA 2 models
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF

# Gemma 2 models
https://huggingface.co/lmstudio-community/gemma-2-2b-it-GGUF

# Mistral models
https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF
```

### **Model Format**
- ✅ **GGUF** (llama.cpp format)
- ✅ Quantized (Q4_K_M, Q8_0)
- ❌ Not PyTorch (.bin, .safetensors)
- ❌ Not TensorFlow (.pb, .h5)

---

## ⚡ Quick Start Commands

### **Deploy with Init Container** (Easy)
```bash
# Will download models on startup
kubectl apply -f nlocalmodels-deployment-with-init.yaml

# Watch download progress
kubectl logs -n nservicecore -l app=nlocalmodels -c model-downloader -f
```

### **Deploy with Models PVC** (Fast)
```bash
# 1. Create PVC
kubectl apply -f nlocalmodels-models-pvc.yaml

# 2. Upload models (one-time)
kubectl apply -f model-upload-job.yaml

# 3. Wait for upload to complete
kubectl wait --for=condition=complete job/model-uploader -n nservicecore --timeout=600s

# 4. Deploy service
kubectl apply -f nlocalmodels-deployment-pvc.yaml
```

---

**Last Updated**: January 28, 2026  
**Status**: Ready to implement
