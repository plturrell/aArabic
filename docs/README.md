# Documentation Index

Welcome to the arabic_folder project documentation! This index helps you find the right documentation quickly.

---

## 📚 Quick Navigation

### 🚀 Getting Started
- [Quick Start Guide](00-getting-started/QUICK_START.md) - Get up and running in 5 minutes

### 🏗️ Architecture & Design
- [Model Orchestration Mapping](01-architecture/MODEL_ORCHESTRATION_MAPPING.md) - **Core architecture document**
- [Context Window Architecture](01-architecture/CONTEXT_WINDOW_ARCHITECTURE.md)
- [RoPE Scaling Implementation](01-architecture/ROPE_SCALING_IMPLEMENTATION.md)
- [Phase 5 Orchestration Plan](01-architecture/ORCHESTRATION_PHASE5_PLAN.md)

### ⚙️ Setup & Configuration
- [Docker Build & Cloud Setup](02-setup/DOCKER_BUILD_CLOUD_SETUP.md)
- [DVC & SAP S3 Setup](02-setup/DVC_SAP_S3_SETUP.md)
- [GitHub Secrets Setup](02-setup/GITHUB_SECRETS_SETUP.md)
- [SAP BTP Setup](02-setup/SAP_BTP_SETUP.md)

### 🔧 Services Documentation
- **nOpenaiServer**: [README](03-services/nOpenaiServer/README.md) | [Auth Setup](03-services/nOpenaiServer/AUTHENTICATION_SETUP.md) | [SSL/TLS](03-services/nOpenaiServer/SSL_TLS_VERIFICATION.md)
- **nAgentMeta**: [Implementation Plan](03-services/nAgentMeta/IMPLEMENTATION_PLAN.md) | [API Reference](03-services/nAgentMeta/API_REFERENCE.md)
- **nGrounding**: [Spec](03-services/nGrounding/spec.md) | [Implementation Plan](03-services/nGrounding/implementation-plan.md)
- **Service Registry**: [README](03-services/service-registry/README.md)

### 🎯 Operations
- [Operator Runbook](04-operations/OPERATOR_RUNBOOK.md) - Production operations guide

### 👩‍💻 Development
- [Contributing Guide](05-development/CONTRIBUTING.md)
- [Prompt Modes Implementation](05-development/PROMPT_MODES_IMPLEMENTATION.md)

### 📖 API Reference
- [Mojo API Reference](07-api-reference/MOJO_API_REFERENCE.md)
- [nFlow OpenAPI Spec](07-api-reference/nFlow-openapi.yaml)
- [nAgentMeta OpenAPI Spec](07-api-reference/nAgentMeta-openapi.yaml)

### 📊 Reports & Releases

#### Latest Migration Reports
- ⭐ **[Python to Zig Migration - User Guide](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_USER_GUIDE.md)** - How to migrate from Python tools
- ⭐ **[Python to Zig Migration Summary](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_SUMMARY.md)** - Technical details & performance
- [Phase 5 Deployment Guide](08-reports/releases/PHASE5_DEPLOYMENT_GUIDE.md)
- [Orchestration Migration Summary](08-reports/releases/ORCHESTRATION_MIGRATION_SUMMARY.md)

#### Release Notes
- [Release Notes V2](08-reports/releases/RELEASE_NOTES_V2.md)
- [Release Notes V1.5](08-reports/releases/RELEASE_NOTES_V1.5.md)
- [Migration Guide V1 to V2](08-reports/releases/MIGRATION_GUIDE_V1_TO_V2.md)
- [Docker Build Migration](08-reports/releases/DOCKER_BUILD_MIGRATION_SUMMARY.md)

#### Validation Reports
- [Model Orchestration Validation](08-reports/validation/MODEL_ORCHESTRATION_VALIDATION_REPORT.md)
- [Arabic NLP Validation](08-reports/validation/ARABIC_NLP_VALIDATION.md)
- [GPU Validation Report](08-reports/validation/GPU_VALIDATION_REPORT_SYSTEM.md)

### 📚 Reference Materials
- [Scripts Guide](09-reference/SCRIPTS_GUIDE.md)
- [T4 GPU Quickstart](09-reference/T4_GPU_QUICKSTART.md)
- [Contributing Guidelines](09-reference/CONTRIBUTING.md)
- [License](09-reference/LICENSE)
- [Third Party Notices](09-reference/THIRD_PARTY_NOTICES.md)

---

## 🔍 Find Documentation By Topic

### Model Orchestration
**Primary:** [MODEL_ORCHESTRATION_MAPPING.md](01-architecture/MODEL_ORCHESTRATION_MAPPING.md)  
**Related:** [ORCHESTRATION_PHASE5_PLAN.md](01-architecture/ORCHESTRATION_PHASE5_PLAN.md), [Phase 5 Deployment Guide](08-reports/releases/PHASE5_DEPLOYMENT_GUIDE.md)

### CLI Tools (Analytics, GPU Monitor, Benchmark, etc.)
**Primary:** [CLI Tools README](../src/serviceCore/nLocalModels/orchestration/CLI_TOOLS_README.md)  
**Migration:** [Python to Zig User Guide](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_USER_GUIDE.md)  
**Technical:** [Python to Zig Summary](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_SUMMARY.md)

### nOpenaiServer
**Main:** [nOpenaiServer README](03-services/nOpenaiServer/README.md)  
**Specs:** [03-services/nOpenaiServer/specs/](03-services/nOpenaiServer/specs/)  
**UI:** [03-services/nOpenaiServer/ui/](03-services/nOpenaiServer/ui/)  
**Daily Reports:** [03-services/nOpenaiServer/modelling/](03-services/nOpenaiServer/modelling/)

### Database & Lineage
**nAgentMeta:** [API Reference](03-services/nAgentMeta/API_REFERENCE.md)  
**Guides:** [Database Operations](03-services/nAgentMeta/DATABASE_OPERATIONS_GUIDE.md), [Performance](03-services/nAgentMeta/DATABASE_PERFORMANCE_GUIDE.md)  
**HANA:** [Graph Engine Guide](03-services/nAgentMeta/HANA_GRAPH_ENGINE_GUIDE.md)

### Mojo & Performance
**API:** [Mojo API Reference](07-api-reference/MOJO_API_REFERENCE.md)  
**MHC:** [nOpenaiServer MHC Specs](03-services/nOpenaiServer/specs/)  
**Optimization:** [Zig/Mojo Optimization Guide](03-services/nOpenaiServer/specs/ZIG_MOJO_OPTIMIZATION_GUIDE.md)

---

## 📂 Documentation Structure

```
docs/
├── 00-getting-started/      # New user onboarding
├── 01-architecture/         # System architecture & design
├── 02-setup/               # Installation & configuration
├── 03-services/            # Individual service documentation
├── 04-operations/          # Production operations
├── 05-development/         # Development guidelines
├── 06-deployment/          # Deployment procedures
├── 07-api-reference/       # API specifications
├── 08-reports/             # Reports, releases, validations
│   ├── releases/           # Release notes & migration guides
│   ├── validation/         # Validation & test reports
│   └── daily-reports/      # Daily progress reports
├── 09-reference/           # Reference materials
└── archive/                # Deprecated/historical docs
```

---

## 🆕 Recent Updates

### January 2026
- ✅ **Python to Zig Migration Complete** - All monitoring and utility scripts migrated
  - [User Guide](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_USER_GUIDE.md)
  - [Technical Summary](08-reports/releases/PYTHON_TO_ZIG_MIGRATION_SUMMARY.md)
  - [CLI Tools README](../src/serviceCore/nLocalModels/orchestration/CLI_TOOLS_README.md)

- ✅ **Model Orchestration Phase 5** - Advanced orchestration features
  - [Deployment Guide](08-reports/releases/PHASE5_DEPLOYMENT_GUIDE.md)
  - [Migration Summary](08-reports/releases/ORCHESTRATION_MIGRATION_SUMMARY.md)

---

## 🎯 Documentation Guidelines

### Finding What You Need
1. **Start here**: This README for navigation
2. **Architecture**: Check [01-architecture/](01-architecture/) for system design
3. **How-to**: Check [02-setup/](02-setup/) for setup guides
4. **Services**: Check [03-services/](03-services/) for specific services
5. **Operations**: Check [04-operations/](04-operations/) for production ops
6. **API**: Check [07-api-reference/](07-api-reference/) for APIs
7. **Recent Changes**: Check [08-reports/releases/](08-reports/releases/)

### Contributing to Documentation
See [CONTRIBUTING.md](05-development/CONTRIBUTING.md) for guidelines on:
- Documentation standards
- Writing style guide
- Review process
- Where to place new docs

---

## 📞 Support

- **Issues**: Report bugs or request features via GitHub issues
- **Questions**: Check existing documentation first, then ask in discussions
- **Updates**: Watch the repository for documentation updates

---

**Last Updated:** 2026-01-23  
**Documentation Version:** 2.0  
**Status:** ✅ Active Maintenance
