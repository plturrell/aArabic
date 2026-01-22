# serviceCore Platform Documentation

Welcome to the comprehensive documentation for the serviceCore platform. This documentation covers architecture, setup, operations, development, and all services.

## 📚 Documentation Structure

### [00. Getting Started](./00-getting-started/)
Quick start guides and prerequisites for new users.
- Installation and setup
- Prerequisites
- First steps

### [01. Architecture](./01-architecture/)
System architecture and design documentation.
- [Context Window Architecture](./01-architecture/CONTEXT_WINDOW_ARCHITECTURE.md)
- [Model Orchestration Mapping](./01-architecture/MODEL_ORCHESTRATION_MAPPING.md)
- [RoPE Scaling Implementation](./01-architecture/ROPE_SCALING_IMPLEMENTATION.md)

### [02. Setup & Configuration](./02-setup/)
Comprehensive setup guides for all components.
- [Docker Build Cloud Setup](./02-setup/DOCKER_BUILD_CLOUD_SETUP.md)
- [DVC with SAP S3 Setup](./02-setup/DVC_SAP_S3_SETUP.md)
- [GitHub Secrets Setup](./02-setup/GITHUB_SECRETS_SETUP.md)
- [SAP BTP Setup](./02-setup/SAP_BTP_SETUP.md)

### [03. Services](./03-services/)
Documentation for all serviceCore services.

**Core Services:**
- service-registry - Service discovery and orchestration
- nOpenaiServer - Local LLM inference
- nWebServe - API Gateway and web server
- nExtract - Document extraction engine
- nAudioLab - Audio processing
- nCode - Code generation and analysis
- nHyperBook - Hypertext book management
- nLeanProof - Lean theorem proving
- nMetaData - Metadata management
- nWorkflow - Workflow orchestration

### [04. Operations](./04-operations/)
Day-to-day operations and maintenance.
- [Operator Runbook](./04-operations/OPERATOR_RUNBOOK.md)
- Monitoring
- Troubleshooting
- Maintenance procedures

### [05. Development](./05-development/)
Developer guides and contribution guidelines.
- [Contributing Guide](./05-development/CONTRIBUTING.md)
- [Prompt Modes Implementation](./05-development/PROMPT_MODES_IMPLEMENTATION.md)
- Coding standards
- Testing guidelines

### [06. Deployment](./06-deployment/)
Deployment guides for various environments.
- Docker deployment
- Kubernetes deployment
- CI/CD pipelines

### [07. API Reference](./07-api-reference/)
API documentation and references.
- REST APIs
- GraphQL APIs
- Service-specific APIs

### [08. Reports](./08-reports/)
Historical reports, release notes, and validations.

**Releases:**
- [Release Notes V1.5](./08-reports/releases/RELEASE_NOTES_V1.5.md)
- [Release Notes V2.0](./08-reports/releases/RELEASE_NOTES_V2.md)
- [Migration Guide V1 to V2](./08-reports/releases/MIGRATION_GUIDE_V1_TO_V2.md)
- [Docker Build Migration](./08-reports/releases/DOCKER_BUILD_MIGRATION_SUMMARY.md)

**Daily Reports:** [Week 8](./08-reports/daily-reports/week-8/) | [Week 9](./08-reports/daily-reports/week-9/) | [Week 10](./08-reports/daily-reports/week-10/)

**Validation Reports:**
- [GPU Validation System](./08-reports/validation/GPU_VALIDATION_REPORT_SYSTEM.md)
- [Arabic NLP Validation](./08-reports/validation/ARABIC_NLP_VALIDATION.md)

### [09. Reference](./09-reference/)
Reference materials and additional resources.
- Dashboard design documents
- Glossary
- FAQ
- External resources

### [10. Archive](./archive/)
Deprecated documentation for historical reference.

## 🚀 Quick Links

### For New Users
1. [Prerequisites](./00-getting-started/PREREQUISITES.md)
2. [Quick Start](./00-getting-started/QUICK_START.md)
3. [SAP BTP Setup](./02-setup/SAP_BTP_SETUP.md)

### For Developers
1. [Contributing Guide](./05-development/CONTRIBUTING.md)
2. [Architecture Overview](./01-architecture/)
3. [Service Documentation](./03-services/)

### For Operators
1. [Operator Runbook](./04-operations/OPERATOR_RUNBOOK.md)
2. [Monitoring Guide](./04-operations/MONITORING.md)
3. [Troubleshooting](./04-operations/TROUBLESHOOTING.md)

## 🏗️ Technology Stack

- **Languages**: Zig, Mojo, Rust, Python
- **Infrastructure**: SAP HANA Cloud, SAP Object Store (AWS S3)
- **Container Orchestration**: Docker, Docker Build Cloud
- **Version Control**: Git, DVC (Data Version Control)
- **CI/CD**: GitHub Actions

## 📊 Architecture Overview

```
SAP BTP Cloud
├── HANA Cloud (Data/Logs/Metrics/Traces)
└── Object Store (Models via DVC, Documents, Audio)
    ↓ OData/REST APIs
serviceCore Services (First-Party Only)
├── service-registry  ├── nWebServe
├── nOpenaiServer    ├── nExtract
├── nAudioLab        ├── nCode
├── nHyperBook       ├── nLeanProof
├── nMetaData        └── nWorkflow
    ↓ Docker Build Cloud
Docker Hub Registry
```

## 🔗 Important Links

- **Repository**: https://github.com/plturrell/aArabic
- **Docker Hub**: https://hub.docker.com/u/plturrell
- **SAP BTP**: [Internal Tenant Configuration](./02-setup/SAP_BTP_SETUP.md)

## 📝 Documentation Standards

All documentation follows these standards:
- **Markdown format** for easy reading and editing
- **Clear headings** and hierarchical structure
- **Code examples** with syntax highlighting
- **Diagrams** where applicable
- **Cross-references** to related docs
- **Version information** and last updated dates

## 🆘 Getting Help

- **Issues**: Report bugs via GitHub Issues
- **Questions**: Check the [FAQ](./09-reference/FAQ.md)
- **Support**: Contact the platform team

## 📅 Documentation Updates

This documentation is continuously updated. Check the last modified date at the bottom of each page.

**Last Updated**: January 22, 2026  
**Version**: 2.0.0  
**Platform**: serviceCore (First-Party Only)

---

## Quick Navigation

| Category | Path | Description |
|----------|------|-------------|
| Getting Started | `00-getting-started/` | New user guides |
| Architecture | `01-architecture/` | System design |
| Setup | `02-setup/` | Configuration guides |
| Services | `03-services/` | Service docs |
| Operations | `04-operations/` | Day-to-day ops |
| Development | `05-development/` | Dev guides |
| Deployment | `06-deployment/` | Deploy guides |
| API Reference | `07-api-reference/` | API docs |
| Reports | `08-reports/` | Historical reports |
| Reference | `09-reference/` | Additional materials |
