# Scripts Reference Guide

Complete guide to all automation scripts in the serviceCore platform.

## Overview

All scripts are located in the `scripts/` directory and organized by category.

```
scripts/
├── README.md                 # Scripts overview
├── benchmarks/               # Performance benchmarking
├── build/                    # Build automation
├── chaos/                    # Chaos engineering tests
├── deployment/               # Deployment automation
├── docker/                   # Docker utilities
├── gpu/                      # GPU management
├── maintenance/              # Maintenance tasks
├── models/                   # Model management
├── services/                 # Service management
├── setup/                    # Initial setup
└── test/                     # Testing scripts
```

## Quick Reference

### Essential Scripts

| Script | Purpose | Usage |
|--------|---------|-------|
| `deployment/docker-up.sh` | Start all services | `./scripts/deployment/docker-up.sh` |
| `deployment/docker-down.sh` | Stop all services | `./scripts/deployment/docker-down.sh` |
| `models/download-local-models.sh` | Download models | `./scripts/models/download-local-models.sh` |
| `services/check_status.sh` | Check service health | `./scripts/services/check_status.sh` |
| `gpu/test_t4_gpu.sh` | Test GPU setup | `./scripts/gpu/test_t4_gpu.sh` |

## Categories

### 1. Benchmarks (`scripts/benchmarks/`)

Performance testing and benchmarking scripts.

#### benchmark_cache_sharing.sh
Tests cache sharing performance across services.

```bash
./scripts/benchmarks/benchmark_cache_sharing.sh
```

**What it does:**
- Tests shared cache performance
- Measures hit rates
- Reports cache efficiency
- Generates performance report

#### benchmark_database_tier.sh
Benchmarks database tier performance.

```bash
./scripts/benchmarks/benchmark_database_tier.sh
```

**What it does:**
- Tests HANA Cloud query performance
- Measures connection latency
- Tests concurrent connections
- Generates database performance report

#### benchmark_gpu_tier.sh
Benchmarks GPU inference performance.

```bash
./scripts/benchmarks/benchmark_gpu_tier.sh
```

**What it does:**
- Tests GPU inference speed
- Measures throughput
- Tests different batch sizes
- Generates GPU performance report

#### benchmark_integrated_tiering.sh
End-to-end tiered system benchmark.

```bash
./scripts/benchmarks/benchmark_integrated_tiering.sh
```

**What it does:**
- Tests full system integration
- Measures end-to-end latency
- Tests load balancing
- Generates comprehensive report

---

### 2. Deployment (`scripts/deployment/`)

Service deployment and orchestration.

#### docker-up.sh
Start all serviceCore services.

```bash
./scripts/deployment/docker-up.sh
```

**What it does:**
- Pulls latest images
- Starts docker-compose services
- Waits for health checks
- Shows service status

**Options:**
```bash
# Start specific service
./scripts/deployment/docker-up.sh nopenaiserver

# Start in detached mode
./scripts/deployment/docker-up.sh -d
```

#### docker-down.sh
Stop all services.

```bash
./scripts/deployment/docker-down.sh
```

**What it does:**
- Gracefully stops services
- Removes containers
- Preserves volumes
- Cleans up networks

**Options:**
```bash
# Remove volumes too
./scripts/deployment/docker-down.sh -v

# Force stop
./scripts/deployment/docker-down.sh -f
```

#### init-platform.sh
Initialize platform for first-time setup.

```bash
./scripts/deployment/init-platform.sh
```

**What it does:**
- Creates necessary directories
- Sets up configuration files
- Initializes databases
- Pulls models
- Starts services

---

### 3. Models (`scripts/models/`)

Model management and downloading.

#### download-local-models.sh
Download all required models.

```bash
./scripts/models/download-local-models.sh
```

**What it does:**
- Downloads models via DVC
- Verifies checksums
- Places in correct directories
- Updates model registry

#### sync_models_to_brev.sh
Sync models to Brev.dev environment.

```bash
./scripts/models/sync_models_to_brev.sh
```

**What it does:**
- Syncs models to Brev
- Handles large file transfers
- Verifies integrity
- Updates remote paths

#### hf_model_card_extractor.py
Extract model information from Hugging Face.

```bash
python scripts/models/hf_model_card_extractor.py <model-name>
```

**What it does:**
- Fetches model metadata
- Extracts configuration
- Downloads model card
- Saves to local registry

---

### 4. Services (`scripts/services/`)

Service management and utilities.

#### start_all_services.sh
Start all serviceCore services.

```bash
./scripts/services/start_all_services.sh
```

**What it does:**
- Starts in dependency order
- Waits for each service
- Registers with service-registry
- Shows status dashboard

#### check_status.sh
Check health of all services.

```bash
./scripts/services/check_status.sh
```

**What it does:**
- Queries health endpoints
- Shows service status
- Reports unhealthy services
- Shows resource usage

**Output:**
```
Service Status:
✓ service-registry (healthy) - 8100
✓ nwebserve (healthy) - 8080
✓ nopenaiserver (healthy) - 11434
✗ nextract (unhealthy) - Connection refused
✓ naudiolab (healthy) - 8300
✓ ncode (healthy) - 8400
```

#### open_all_uis.sh
Open all service UIs in browser.

```bash
./scripts/services/open_all_uis.sh
```

**What it does:**
- Opens service dashboards
- Opens monitoring UIs
- Opens API documentation
- Opens health check pages

---

### 5. GPU (`scripts/gpu/`)

GPU setup, testing, and monitoring.

#### test_t4_gpu.sh
Test NVIDIA T4 GPU setup.

```bash
./scripts/gpu/test_t4_gpu.sh
```

**What it does:**
- Checks CUDA installation
- Tests GPU visibility
- Runs inference test
- Generates performance report

#### monitor_t4_gpu.sh
Monitor GPU usage in real-time.

```bash
./scripts/gpu/monitor_t4_gpu.sh
```

**What it does:**
- Shows GPU utilization
- Shows memory usage
- Shows running processes
- Refreshes every 2 seconds

**Output:**
```
GPU 0: NVIDIA T4
  Utilization: 85%
  Memory: 12.5 GB / 16 GB
  Temperature: 72°C
  Power: 45W / 70W
```

#### generate_validation_report.sh
Generate GPU validation report.

```bash
./scripts/gpu/generate_validation_report.sh
```

**What it does:**
- Runs comprehensive GPU tests
- Benchmarks inference performance
- Tests different batch sizes
- Generates detailed report

---

### 6. Maintenance (`scripts/maintenance/`)

System maintenance and fixes.

#### fix_adapters.py
Fix model adapter configurations.

```bash
python scripts/maintenance/fix_adapters.py
```

**What it does:**
- Scans for broken adapters
- Fixes configuration issues
- Updates adapter registry
- Verifies integrity

#### fix_all_adapters.sh
Fix all adapter issues in batch.

```bash
./scripts/maintenance/fix_all_adapters.sh
```

**What it does:**
- Runs fix_adapters.py
- Verifies all adapters
- Regenerates indices
- Tests loading

---

### 7. Testing (`scripts/test/`)

Test automation and validation.

#### test_rag_integration.sh
Test RAG (Retrieval Augmented Generation) integration.

```bash
./scripts/test/test_rag_integration.sh
```

**What it does:**
- Tests document ingestion
- Tests retrieval
- Tests generation
- Validates results

#### test_mojo_embedding.sh
Test Mojo embedding service.

```bash
./scripts/test/test_mojo_embedding.sh
```

**What it does:**
- Tests embedding generation
- Validates dimensions
- Tests batch processing
- Measures performance

#### test_all_uis.sh
Test all service UIs.

```bash
./scripts/test/test_all_uis.sh
```

**What it does:**
- Tests each service UI
- Checks accessibility
- Validates responses
- Reports failures

---

### 8. Chaos Testing (`scripts/chaos/`)

Chaos engineering and resilience testing.

#### chaos_test_suite.sh
Run comprehensive chaos tests.

```bash
./scripts/chaos/chaos_test_suite.sh
```

**What it does:**
- Randomly kills services
- Simulates network issues
- Tests failover
- Validates recovery
- Generates chaos report

**Scenarios tested:**
- Service crashes
- Network partitions
- Resource exhaustion
- Database failures
- API timeouts

---

## Common Workflows

### Initial Setup

```bash
# 1. Initialize platform
./scripts/deployment/init-platform.sh

# 2. Download models
./scripts/models/download-local-models.sh

# 3. Start services
./scripts/deployment/docker-up.sh

# 4. Check status
./scripts/services/check_status.sh
```

### Daily Operations

```bash
# Check service health
./scripts/services/check_status.sh

# View service UIs
./scripts/services/open_all_uis.sh

# Monitor GPU
./scripts/gpu/monitor_t4_gpu.sh
```

### Troubleshooting

```bash
# Check service status
./scripts/services/check_status.sh

# Fix adapter issues
./scripts/maintenance/fix_all_adapters.sh

# Restart services
./scripts/deployment/docker-down.sh
./scripts/deployment/docker-up.sh
```

### Performance Testing

```bash
# Run benchmarks
./scripts/benchmarks/benchmark_integrated_tiering.sh

# Test GPU
./scripts/gpu/test_t4_gpu.sh

# Generate reports
./scripts/gpu/generate_validation_report.sh
```

## Script Development

### Creating New Scripts

1. **Choose appropriate directory** based on purpose
2. **Use consistent naming** (lowercase, underscores)
3. **Add shebang** (`#!/bin/bash`)
4. **Include help text** (`--help` flag)
5. **Add error handling** (`set -e`)
6. **Document in this guide**

### Script Template

```bash
#!/bin/bash
set -e

# Script: my_script.sh
# Purpose: Brief description
# Usage: ./scripts/category/my_script.sh [options]

# Help text
show_help() {
    cat << EOF
Usage: $0 [options]

Description of what this script does.

Options:
    -h, --help     Show this help message
    -v, --verbose  Enable verbose output
    
Examples:
    $0
    $0 --verbose
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Main script logic
main() {
    echo "Starting..."
    # Your code here
    echo "Done!"
}

main
```

## Environment Variables

Scripts respect these environment variables:

```bash
# Service URLs
SERVICE_REGISTRY_URL=http://localhost:8100
NOPENAISERVER_URL=http://localhost:11434

# Paths
MODELS_PATH=/models
SCRIPTS_PATH=/app/scripts

# Logging
LOG_LEVEL=info
LOG_TO_HANA=true

# SAP HANA Cloud
HANA_ODATA_URL=https://...
HANA_USERNAME=DBADMIN
HANA_PASSWORD=...
```

## Troubleshooting Scripts

### Script Fails to Execute

**Problem**: Permission denied  
**Solution**:
```bash
chmod +x scripts/category/script.sh
```

### Script Can't Find Dependencies

**Problem**: Command not found  
**Solution**:
- Install required tools
- Check PATH
- Use absolute paths

### Script Timeouts

**Problem**: Script hangs  
**Solution**:
- Increase timeout values
- Check service availability
- Review logs

## Related Documentation

- [Deployment Guide](../06-deployment/)
- [Operations Runbook](../04-operations/OPERATOR_RUNBOOK.md)
- [Development Guide](../05-development/CONTRIBUTING.md)

---

**Total Scripts**: 30+  
**Categories**: 8  
**Last Updated**: January 22, 2026
