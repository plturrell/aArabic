# Scripts Directory

Organized collection of utility scripts for the AI Nucleus Platform.

## Directory Structure

### 📁 Root Scripts
Core platform scripts:
- `access_services.sh` - Open all service UIs in browser
- `check_status.sh` - Check status of all running services
- `docker-up.sh` / `docker-down.sh` - Start/stop all containers
- `start_all_services.sh` - Start all platform services
- `start_backend.sh` - Start backend service only
- `start_langflow.sh` - Start Langflow only
- `start_services.sh` - Start core services
- `start_vendor_services.sh` - Start vendor services
- `start-local-inference.sh` - Start local LLM inference
- `open_all_uis.sh` - Open all web UIs
- `init-platform.sh` - Initialize platform (first-time setup)
- `download-local-models.sh` - Download required models
- `rebuild-langflow.sh` - Rebuild Langflow container
- `n8n_export_workflows.sh` - Export n8n workflows
- `n8n_import_workflows.sh` - Import n8n workflows

### 📁 setup/
Setup and configuration scripts:
- `setup_graphchat_in_cline.md` - GraphChat setup guide for Cline
- `setup_memgraph.cypher` - Memgraph sample data initialization

### 📁 test/
Testing scripts:
- `test_graphchat.sh` - Test GraphChat Bridge functionality
- `test_all_uis.sh` - Test all UI accessibility

### 📁 maintenance/
Maintenance and fix scripts:
- `fix_adapter_aliases.sh` - Fix adapter alias issues
- `fix_adapters.py` - Python adapter fixer
- `fix_all_adapters.sh` - Fix all adapter issues

## Usage

### Quick Start
```bash
# Initialize platform (first time)
./scripts/init-platform.sh

# Start all services
./scripts/start_all_services.sh

# Check service status
./scripts/check_status.sh

# Open all UIs
./scripts/access_services.sh
```

### Testing
```bash
# Test GraphChat
./scripts/test/test_graphchat.sh

# Test all UIs
./scripts/test/test_all_uis.sh
```

### Setup
```bash
# Setup Memgraph sample data
docker cp scripts/setup/setup_memgraph.cypher ai_nucleus_memgraph:/tmp/
docker exec ai_nucleus_memgraph mgconsole < /tmp/setup_memgraph.cypher
```

## Notes

- All scripts are executable (`chmod +x` already applied)
- Scripts assume they're run from project root unless otherwise noted
- Check individual script headers for detailed usage information
