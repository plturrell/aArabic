# nLocalModels - Prerequisites & Dependencies

## 📋 Quick Summary

Before deploying nLocalModels to SAP Kyma, ensure you have:

✅ **SAP HANA Cloud instance** with NOPENAI_DB database  
✅ **SAP Object Store** bucket for KV cache storage  
✅ **Kyma cluster** with 16GB+ RAM nodes  
✅ **Docker image** built and pushed (`plturrell/nservices:nLocalModels`)  
✅ **kubectl & kustomize** CLI tools installed  

---

## 🎯 Critical Dependencies

### 1. SAP HANA Cloud (REQUIRED)

**Why**: Stores metadata for KV cache, metrics, routing decisions, session state

**What you need**:
- Instance URL: `your-instance.hanacloud.ondemand.com`
- Database name: `NOPENAI_DB`
- User: `NUCLEUS_APP` (or custom)
- Password: (secure credential)
- Port: `443` (HTTPS/ODBC)

**Setup checklist**:
- [ ] HANA Cloud instance provisioned in BTP Cockpit
- [ ] Database `NOPENAI_DB` created
- [ ] User created with appropriate permissions
- [ ] Required tables created (see table schema below)
- [ ] Connection tested from local machine
- [ ] Credentials securely stored

**Required HANA Tables**:
```sql
-- Run these in HANA Studio or HANA Cloud Central

-- 1. KV Cache Metadata
CREATE COLUMN TABLE KV_CACHE_METADATA (
  SESSION_ID VARCHAR(128),
  LAYER INT,
  SIZE INT,
  COMPRESSION VARCHAR(16),
  OBJECT_KEY VARCHAR(256),
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  EXPIRES_AT TIMESTAMP,
  PRIMARY KEY (SESSION_ID, LAYER)
);

-- 2. Prompt Cache
CREATE COLUMN TABLE PROMPT_CACHE (
  HASH VARCHAR(64) PRIMARY KEY,
  STATE BLOB,
  EXPIRES_AT TIMESTAMP,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Session State
CREATE COLUMN TABLE SESSION_STATE (
  SESSION_ID VARCHAR(128) PRIMARY KEY,
  DATA BLOB,
  EXPIRES_AT TIMESTAMP,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 4. Routing Decisions
CREATE COLUMN TABLE ROUTING_DECISIONS (
  DECISION_ID VARCHAR(128) PRIMARY KEY,
  REQUEST_ID VARCHAR(128),
  TASK_TYPE VARCHAR(64),
  AGENT_ID VARCHAR(128),
  MODEL_ID VARCHAR(128),
  CAPABILITY_SCORE DECIMAL(5,2),
  PERFORMANCE_SCORE DECIMAL(5,2),
  COMPOSITE_SCORE DECIMAL(5,2),
  STRATEGY_USED VARCHAR(64),
  LATENCY_MS INT,
  SUCCESS BOOLEAN,
  FALLBACK_USED BOOLEAN,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 5. Inference Metrics
CREATE COLUMN TABLE INFERENCE_METRICS (
  METRIC_ID VARCHAR(128) PRIMARY KEY,
  MODEL_ID VARCHAR(128),
  LATENCY_MS INT,
  TTFT_MS INT,
  TOKENS_GENERATED INT,
  CACHE_HIT BOOLEAN,
  CREATED_AT TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_routing_agent ON ROUTING_DECISIONS(AGENT_ID);
CREATE INDEX idx_routing_model ON ROUTING_DECISIONS(MODEL_ID);
CREATE INDEX idx_routing_created ON ROUTING_DECISIONS(CREATED_AT);
CREATE INDEX idx_metrics_model ON INFERENCE_METRICS(MODEL_ID);
CREATE INDEX idx_metrics_created ON INFERENCE_METRICS(CREATED_AT);
```

---

### 2. SAP Object Store (REQUIRED)

**Why**: Stores large binary KV cache tensors (too big for HANA inline storage)

**What you need**:
- Endpoint URL: `https://objectstore.cfapps.{region}.hana.ondemand.com`
- Bucket name: `nlocalmodels-kv-cache`
- Access Key ID: (credential)
- Secret Access Key: (credential)

**Setup checklist**:
- [ ] Object Store service enabled in BTP
- [ ] Bucket `nlocalmodels-kv-cache` created
- [ ] Bucket set to private access
- [ ] Access credentials generated
- [ ] Lifecycle policy configured (optional - for auto-cleanup)
- [ ] Connection tested

**Bucket Configuration**:
```json
{
  "name": "nlocalmodels-kv-cache",
  "region": "eu-central-1",
  "access": "private",
  "lifecycle_rules": [
    {
      "id": "expire-old-cache",
      "status": "enabled",
      "expiration": {
        "days": 7
      },
      "filter": {
        "prefix": "kv-cache/"
      }
    }
  ]
}
```

---

### 3. SAP Kyma Cluster (REQUIRED)

**Why**: Runtime platform for containerized services

**What you need**:
- Kubernetes version: 1.24+
- Kyma version: 2.x
- Node specifications:
  - Memory: 16GB+ per node
  - CPU: 4+ cores per node
  - Storage: 100GB+ per node
- Istio service mesh: Enabled
- kubectl context: Configured

**Setup checklist**:
- [ ] Kyma runtime provisioned in BTP
- [ ] kubectl configured with cluster context
- [ ] Can run: `kubectl get nodes`
- [ ] Can run: `kubectl get ns`
- [ ] Sufficient resources available
- [ ] Istio gateway accessible

**Verify cluster**:
```bash
# Check cluster connection
kubectl cluster-info

# Check nodes
kubectl get nodes -o wide

# Check available resources
kubectl top nodes

# Check Kyma components
kubectl get pods -n kyma-system

# Check Istio gateway
kubectl get gateway -n kyma-system kyma-gateway
```

---

### 4. Docker Image (REQUIRED)

**Why**: Contains the nLocalModels service binary

**What you need**:
- Image: `plturrell/nservices:nLocalModels`
- Registry: Docker Hub (or custom registry)
- Architecture: linux/amd64, linux/arm64 (multi-arch)

**Setup checklist**:
- [ ] GitHub Actions workflow executed successfully
- [ ] Image pushed to registry
- [ ] Image can be pulled: `docker pull plturrell/nservices:nLocalModels`
- [ ] Registry credentials configured in Kyma (if private)

**Build image manually** (if needed):
```bash
# Navigate to repo root
cd /Users/karthikeyan/git/aArabic

# Build image
docker build -f docker/Dockerfile.nlocalmodels -t plturrell/nservices:nLocalModels .

# Test locally
docker run -p 11434:11434 plturrell/nservices:nLocalModels

# Push to registry
docker push plturrell/nservices:nLocalModels
```

---

## 🔧 Optional Dependencies

### 5. SAP AI Core (OPTIONAL)

**Why**: Centralized model deployment and lifecycle management

**When to use**:
- Managing multiple model deployments
- Need centralized inference tracking
- Using SAP AI Core for model training

**What you need**:
- AI Core URL: `https://api.ai.{region}.aws.ml.hana.ondemand.com`
- Client ID: (OAuth credential)
- Client Secret: (OAuth credential)
- Auth URL: `https://{subdomain}.authentication.{region}.hana.ondemand.com/oauth/token`
- Resource Group: `default` (or custom)

**Setup checklist**:
- [ ] AI Core service enabled
- [ ] Service key created
- [ ] OAuth credentials extracted
- [ ] Resource group created

---

### 6. GPU Nodes (OPTIONAL)

**Why**: 10-100x faster inference for large models

**When to use**:
- Deploying models >7B parameters
- Need low latency (<500ms)
- High throughput requirements

**What you need**:
- GPU type: NVIDIA T4, A10G, or A100
- CUDA version: 11.8+ or 12.x
- GPU operator: Installed in Kyma
- Node labels: `gpu: "true"`

**Setup checklist**:
- [ ] GPU nodes available in cluster
- [ ] NVIDIA GPU Operator installed
- [ ] GPU device plugin running
- [ ] Can verify: `kubectl get nodes -l gpu=true`

---

## 🛠️ Required CLI Tools

### kubectl (REQUIRED)

```bash
# Install kubectl
# macOS
brew install kubectl

# Linux
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
chmod +x kubectl
sudo mv kubectl /usr/local/bin/

# Verify
kubectl version --client
```

### kustomize (REQUIRED)

```bash
# Install kustomize
# macOS
brew install kustomize

# Linux
curl -s "https://raw.githubusercontent.com/kubernetes-sigs/kustomize/master/hack/install_kustomize.sh" | bash
sudo mv kustomize /usr/local/bin/

# Verify
kustomize version
```

### Helpful Tools (OPTIONAL)

```bash
# kubectx/kubens - Switch contexts/namespaces easily
brew install kubectx

# k9s - Terminal UI for Kubernetes
brew install derailed/k9s/k9s

# stern - Multi-pod log tailing
brew install stern

# jq - JSON processor
brew install jq
```

---

## 📊 Resource Requirements

### Per Pod

| Resource | Minimum | Recommended | Maximum |
|----------|---------|-------------|---------|
| Memory | 4Gi | 8Gi | 32Gi |
| CPU | 1 core | 2 cores | 8 cores |
| GPU (optional) | - | 1x T4 | 1x A100 |
| Ephemeral Storage | 10Gi | 20Gi | 100Gi |

### Per Namespace (2 replicas)

| Resource | Total |
|----------|-------|
| Memory | 16-64Gi |
| CPU | 4-16 cores |
| Storage | 20-200Gi |

### HANA Database

| Metric | Estimate |
|--------|----------|
| Storage | 10-100GB (depends on cache TTL) |
| IOPS | 1000-5000 |
| Connections | 5-10 per pod |

### Object Store

| Metric | Estimate |
|--------|----------|
| Storage | 50-500GB (depends on model size and cache) |
| Requests | 100-1000 per minute |
| Bandwidth | 10-100 MB/s |

---

## 🔐 Security Requirements

### Network Access

**Kyma to HANA**:
- [ ] Outbound HTTPS (port 443) allowed
- [ ] HANA endpoint reachable from Kyma pods
- [ ] Network firewall rules configured

**Kyma to Object Store**:
- [ ] Outbound HTTPS (port 443) allowed
- [ ] Object Store endpoint reachable
- [ ] Network firewall rules configured

### Credentials Management

**NEVER**:
- ❌ Commit credentials to Git
- ❌ Use plaintext passwords
- ❌ Share credentials in chat/email

**ALWAYS**:
- ✅ Use Kubernetes Secrets
- ✅ Rotate credentials regularly
- ✅ Use least-privilege access
- ✅ Enable audit logging

---

## ✅ Pre-Deployment Checklist

### Infrastructure
- [ ] HANA Cloud instance running
- [ ] HANA database `NOPENAI_DB` created
- [ ] HANA tables created and verified
- [ ] Object Store bucket created
- [ ] Kyma cluster accessible
- [ ] Sufficient cluster resources available

### Credentials
- [ ] HANA credentials obtained
- [ ] Object Store credentials obtained
- [ ] Credentials tested locally
- [ ] Secrets YAML updated (but not committed!)

### Image
- [ ] Docker image built successfully
- [ ] Image pushed to registry
- [ ] Image can be pulled from Kyma

### Tools
- [ ] kubectl installed and configured
- [ ] kustomize installed
- [ ] Can access Kyma cluster

### Configuration
- [ ] Resource limits reviewed
- [ ] Memory tiers configured
- [ ] APIRule host updated
- [ ] Namespace name confirmed (nservicecore)

---

## 🚀 Ready to Deploy?

If all checkboxes above are complete, you're ready to deploy!

**Next steps**:
1. Review the [Deployment Guide](DEPLOYMENT_GUIDE.md)
2. Update credentials in `nlocalmodels-secret.yaml`
3. Deploy with: `kubectl apply -k .`

---

## 📞 Support

**Issues with prerequisites?**

1. **HANA Cloud**: Contact SAP BTP Support
2. **Object Store**: Check SAP BTP Cockpit service status
3. **Kyma Cluster**: Review Kyma documentation
4. **Docker Image**: Check GitHub Actions build logs

**Common setup issues**:
- HANA connection timeout → Check firewall rules
- Object Store access denied → Verify credentials and bucket permissions
- kubectl not working → Configure kubeconfig correctly
- Image pull failed → Check registry credentials

---

**Last Updated**: January 27, 2026  
**Version**: 1.0
