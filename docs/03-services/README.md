# serviceCore Services Documentation

Complete documentation for all first-party serviceCore services.

## Service Overview

All services are prefixed with "n*" (nucleus) and are built with Zig, Mojo, or Rust for maximum performance and zero external dependencies.

## Core Services

### [service-registry](./service-registry/) (Rust/Actix)
**Port**: 8100  
**Purpose**: Service discovery and orchestration  
**Status**: Production

Central service registry that manages:
- Service discovery and health tracking
- Dynamic service routing
- Load balancing
- Circuit breaker patterns

**Documentation**:
- [Architecture](./service-registry/ARCHITECTURE.md)
- [API Reference](./service-registry/API.md)
- [Configuration](./service-registry/CONFIGURATION.md)

---

### [nWebServe](./nWebServe/) (Zig)
**Port**: 8080  
**Purpose**: API Gateway and web server  
**Status**: Production

Custom-built API gateway providing:
- Request routing
- Authentication/authorization
- Rate limiting
- Request/response transformation

**Documentation**:
- [Architecture](./nWebServe/ARCHITECTURE.md)
- [API Reference](./nWebServe/API.md)
- [Configuration](./nWebServe/CONFIGURATION.md)

---

### [nOpenaiServer](./nOpenaiServer/) (Zig/Mojo)
**Port**: 11434  
**Purpose**: Local LLM inference server  
**Status**: Production

OpenAI-compatible inference server with:
- GGUF model support (Qwen, Llama, Mistral, etc.)
- Local inference (no data leaves infrastructure)
- GPU acceleration support
- Multiple concurrent requests

**Models**:
- Qwen3-Coder-30B - Code generation
- HY-MT1.5-1.8B - Translation
- CamelBERT - Arabic NLP
- DeepSeek-Math - Mathematical reasoning

**Documentation**:
- [Architecture](./nOpenaiServer/ARCHITECTURE.md)
- [API Reference](./nOpenaiServer/API.md)
- [Model Guide](./nOpenaiServer/MODELS.md)
- [Configuration](./nOpenaiServer/CONFIGURATION.md)

---

### [nExtract](./nExtract/) (Zig/Mojo)
**Port**: 8200  
**Purpose**: Document extraction engine  
**Status**: Active Development

Unified document processing replacing three Python libraries:
- PDF, Office (DOCX/XLSX/PPTX) parsing
- OCR engine (pure Zig/Mojo)
- ML-based layout analysis
- Structured extraction via nOpenaiServer
- Export to Markdown, HTML, JSON

**Documentation**:
- [Architecture](./nExtract/ARCHITECTURE.md)
- [API Reference](./nExtract/API.md)
- [Supported Formats](./nExtract/FORMATS.md)
- [Configuration](./nExtract/CONFIGURATION.md)

---

### [nAudioLab](./nAudioLab/) (Python → Zig/Mojo)
**Port**: 8300  
**Purpose**: Audio processing and transcription  
**Status**: Active (Python, migrating to Zig/Mojo)

Audio processing capabilities:
- Audio transcription
- Speaker diarization
- Audio format conversion
- Speech enhancement

**Documentation**:
- [Architecture](./nAudioLab/ARCHITECTURE.md)
- [API Reference](./nAudioLab/API.md)
- [Configuration](./nAudioLab/CONFIGURATION.md)

---

### [nCode](./nCode/) (Python → Zig/Mojo)
**Port**: 8400  
**Purpose**: Code generation and analysis  
**Status**: Active (Python, migrating to Zig/Mojo)

Code-related capabilities:
- Code generation
- Code analysis and refactoring
- Syntax transformation
- Code quality metrics

**Documentation**:
- [Architecture](./nCode/ARCHITECTURE.md)
- [API Reference](./nCode/API.md)
- [Configuration](./nCode/CONFIGURATION.md)

---

### [nHyperBook](./nHyperBook/) (Python/Mojo)
**Purpose**: Hypertext book generation and management  
**Status**: Active

**Documentation**:
- [Architecture](./nHyperBook/ARCHITECTURE.md)
- [API Reference](./nHyperBook/API.md)

---

### [nLeanProof](./nLeanProof/) (Lean 4/Python/Mojo)
**Purpose**: Lean theorem proving and formal verification  
**Status**: Active

**Documentation**:
- [Architecture](./nLeanProof/ARCHITECTURE.md)
- [API Reference](./nLeanProof/API.md)

---

### [nMetaData](./nMetaData/)
**Purpose**: Metadata management and extraction  
**Status**: Active

**Documentation**:
- [Architecture](./nMetaData/ARCHITECTURE.md)
- [API Reference](./nMetaData/API.md)

---

### [nWorkflow](./nWorkflow/)
**Purpose**: Workflow orchestration and automation  
**Status**: Active

**Documentation**:
- [Architecture](./nWorkflow/ARCHITECTURE.md)
- [API Reference](./nWorkflow/API.md)

---

### [nLaunchpad](./nLaunchpad/)
**Purpose**: Service launcher and manager  
**Status**: Active

**Documentation**:
- [Architecture](./nLaunchpad/ARCHITECTURE.md)
- [API Reference](./nLaunchpad/API.md)

## Service Communication

All services communicate via:
- **HTTP/REST**: Standard RESTful APIs
- **Service Registry**: Dynamic service discovery
- **SAP HANA Cloud**: Shared data layer via OData

## Service Dependencies

```
service-registry (core)
    ↓
nWebServe (gateway)
    ↓
├── nOpenaiServer (LLM inference)
│   └── Used by: nExtract, nCode, nHyperBook
├── nExtract (document processing)
├── nAudioLab (audio processing)
├── nCode (code generation)
├── nHyperBook (book generation)
├── nLeanProof (theorem proving)
├── nMetaData (metadata management)
└── nWorkflow (orchestration)
```

## Common Patterns

### Health Checks
All services implement `/health` endpoint:
```bash
curl http://localhost:<port>/health
```

### Logging
All services log to SAP HANA Cloud:
- Structured JSON logs
- Trace IDs for correlation
- Automatic metric extraction

### Configuration
All services use environment variables:
- Defined in `.env`
- Documented in service-specific CONFIGURATION.md
- Validated on startup

## Development Status

| Service | Language | Status | Migration Target |
|---------|----------|--------|------------------|
| service-registry | Rust | ✅ Production | - |
| nWebServe | Zig | ✅ Production | - |
| nOpenaiServer | Zig/Mojo | ✅ Production | - |
| nExtract | Zig/Mojo | 🚧 Development | - |
| nAudioLab | Python | ✅ Active | Zig/Mojo |
| nCode | Python | ✅ Active | Zig/Mojo |
| nHyperBook | Python/Mojo | ✅ Active | Full Mojo |
| nLeanProof | Lean 4/Python/Mojo | ✅ Active | Full Mojo |
| nMetaData | - | 🚧 Planning | Zig/Mojo |
| nWorkflow | - | 🚧 Planning | Zig/Mojo |

## Service Deployment

### Docker Images

All services have Docker images available:
```
plturrell/service-registry:latest
plturrell/nopenaiserver:latest
plturrell/nwebserve:latest
plturrell/nextract:latest
plturrell/naudiolab:latest
plturrell/ncode:latest
```

### Docker Compose

Start all services:
```bash
docker-compose -f docker/compose/docker-compose.servicecore.yml up -d
```

### Individual Services

Start specific service:
```bash
docker-compose -f docker/compose/docker-compose.servicecore.yml up service-registry
```

## Service Integration Guide

### Using nOpenaiServer from Your Service

```python
import requests

response = requests.post(
    "http://nopenaiserver:11434/v1/chat/completions",
    json={
        "model": "Qwen3-Coder-30B-A3B-Instruct",
        "messages": [{"role": "user", "content": "Your prompt"}]
    }
)
```

### Using nExtract for Document Processing

```python
import requests

# Upload document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://nextract:8200/extract",
        files={"file": f},
        data={"format": "markdown"}
    )
```

### Registering with service-registry

```python
import requests

requests.post(
    "http://service-registry:8100/register",
    json={
        "name": "my-service",
        "host": "my-service",
        "port": 8500,
        "health_endpoint": "/health"
    }
)
```

## Next Steps

- **Architecture**: [Understanding the System](../01-architecture/)
- **Setup**: [Detailed Setup Guides](../02-setup/)
- **Development**: [Contributing to Services](../05-development/)
- **Operations**: [Running in Production](../04-operations/)

## Support

- **Service Issues**: Check service-specific troubleshooting docs
- **API Questions**: See service-specific API docs
- **General Support**: Contact platform team

---

**Last Updated**: January 22, 2026  
**Total Services**: 11  
**Production-Ready**: 6
