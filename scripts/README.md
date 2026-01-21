# Scripts Directory

Organized collection of automation and utility scripts for the Arabic NLP project.

## 📁 Directory Structure

### 🖥️ `gpu/` - GPU & T4 Testing
Scripts for GPU testing, monitoring, and T4 optimization.

- **test_t4_gpu.sh** - Comprehensive T4 GPU test suite
- **monitor_t4_gpu.sh** - Real-time GPU monitoring dashboard
- **T4_GPU_QUICKSTART.md** - Quick start guide for T4 GPU testing

**Usage:**
```bash
# Run T4 GPU test suite
./gpu/test_t4_gpu.sh

# Monitor GPU in real-time
./gpu/monitor_t4_gpu.sh
```

### 📦 `models/` - Model Management
Scripts for downloading, syncing, and managing ML models.

- **sync_models_to_brev.sh** - Sync local models to Brev instance
- **download_models_on_brev.sh** - Download models from HuggingFace on Brev
- **download-local-models.sh** - Download models locally
- **download_kaggle_datasets.sh** - Download datasets from Kaggle

**Usage:**
```bash
# Sync models to Brev
./models/sync_models_to_brev.sh awesome-gpu-nucleus

# Download models on Brev instance
./models/download_models_on_brev.sh all-testing
```

### 🚀 `deployment/` - Deployment & Infrastructure
Scripts for deploying services and managing infrastructure.

- **bootstrap_ouroboros.py** - Bootstrap Ouroboros system
- **deploy_mojo_to_portainer.sh** - Deploy Mojo services to Portainer
- **docker-down.sh** - Stop all Docker services
- **docker-up.sh** - Start all Docker services
- **init-platform.sh** - Initialize platform
- **portainer_setup.sh** - Set up Portainer

**Usage:**
```bash
# Start all services
./deployment/docker-up.sh

# Deploy to Portainer
./deployment/deploy_mojo_to_portainer.sh
```

### 🔧 `services/` - Service Management
Scripts for starting, stopping, and managing services.

- **start_all_services.sh** - Start all services
- **start_dashboard_stack.sh** - Start dashboard services
- **start_services.sh** - Start core services
- **start_vendor_services.sh** - Start vendor services
- **start-liquidai-server.sh** - Start LiquidAI server
- **start-local-inference.sh** - Start local inference server
- **run_mojo_embedding.sh** - Run Mojo embedding service
- **run_mojo_translation.sh** - Run Mojo translation service
- **access_services.sh** - Access service endpoints
- **open_all_uis.sh** - Open all UIs in browser
- **check_status.sh** - Check service status
- **consolidate_services.sh** - Consolidate services
- **fix_healthchecks.sh** - Fix service healthchecks

**Usage:**
```bash
# Start all services
./services/start_all_services.sh

# Check service status
./services/check_status.sh
```

### 📊 `benchmarks/` - Performance Benchmarking
Scripts for benchmarking performance across different tiers.

- **benchmark_cache_sharing.sh** - Benchmark cache sharing
- **benchmark_database_tier.sh** - Benchmark database tier
- **benchmark_gpu_tier.sh** - Benchmark GPU tier
- **benchmark_integrated_tiering.sh** - Benchmark integrated tiering

**Usage:**
```bash
# Run GPU benchmarks
./benchmarks/benchmark_gpu_tier.sh
```

### 🔨 `build/` - Build & Compilation
Scripts for building projects and installing dependencies.

- **build_all_rust_clients.sh** - Build all Rust clients
- **build_json_lib.sh** - Build JSON library
- **build_rust_clients.sh** - Build Rust clients
- **install_rust_clients.sh** - Install Rust clients

**Usage:**
```bash
# Build all Rust clients
./build/build_all_rust_clients.sh
```

### 🧪 `test/` - Testing
Test scripts for various components.

- **test_mojo_embedding.sh** - Test Mojo embedding service
- **test_mojo_translation.sh** - Test Mojo translation service
- **test_rag_integration.sh** - Test RAG integration

**Usage:**
```bash
# Test RAG integration
./test/test_rag_integration.sh
```

### 🔧 `maintenance/` - Maintenance Scripts
System maintenance and operational scripts.

### 🎯 `chaos/` - Chaos Engineering
Chaos engineering and resilience testing scripts.

### ⚙️ `setup/` - Setup & Configuration
Initial setup and configuration scripts.

## 🚀 Quick Start

### For T4 GPU Testing on Brev
```bash
# 1. Download models
./models/download_models_on_brev.sh all-testing

# 2. Run GPU tests
./gpu/test_t4_gpu.sh

# 3. Monitor GPU (separate terminal)
./gpu/monitor_t4_gpu.sh
```

### For Local Development
```bash
# 1. Initialize platform
./deployment/init-platform.sh

# 2. Start services
./services/start_all_services.sh

# 3. Check status
./services/check_status.sh
```

## 📝 Script Naming Conventions

- **test_*.sh** - Testing scripts
- **start_*.sh** - Service starter scripts
- **run_*.sh** - Runtime execution scripts
- **build_*.sh** - Build and compilation scripts
- **benchmark_*.sh** - Performance benchmarking scripts
- **download_*.sh** - Download and fetch scripts
- **deploy_*.sh** - Deployment scripts

## 🔗 Related Documentation

- [T4 GPU Quick Start Guide](gpu/T4_GPU_QUICKSTART.md)
- [Model Registry](../vendor/layerModels/MODEL_REGISTRY.json)
- [T4 Optimization Guide](../src/serviceCore/nOpenaiServer/docs/T4/T4_OPTIMIZATION_GUIDE.md)
- [Implementation Plan](../src/serviceCore/nOpenaiServer/docs/T4/IMPLEMENTATION_PLAN.md)

## 💡 Tips

- All scripts should be run from the project root or their respective directories
- Use `--help` flag where available for detailed usage information
- Check script permissions: `chmod +x script_name.sh`
- Review logs in `/tmp/` for troubleshooting

## 🐛 Troubleshooting

If a script fails:
1. Check file permissions (`ls -l script_name.sh`)
2. Verify dependencies are installed
3. Check logs in `/tmp/` directory
4. Ensure you're in the correct directory
5. Review script output for specific error messages

## 📧 Support

For issues or questions:
- Check documentation in respective directories
- Review test logs
- Consult project README
- Open GitHub issue: https://github.com/plturrell/aArabic/issues
