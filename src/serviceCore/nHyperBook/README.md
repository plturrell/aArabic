# HyperShimmy: Pure Mojo/Zig Research Assistant

A complete replacement of HyperbookLM built with 100% Mojo/Zig technology stack.

## 🎯 Overview

HyperShimmy is an enterprise-grade research assistant that provides:

- **Multi-Source Ingestion** - Web scraping and PDF processing
- **AI-Powered Chat** - Context-aware conversations using local LLMs
- **Research Summaries** - Comprehensive multi-document analysis
- **Interactive Mindmaps** - Visual knowledge graph exploration
- **Audio Overviews** - Podcast-style audio generation
- **Presentation Slides** - Auto-generated slide decks

## 🏗️ Architecture

```
┌─────────────────────────────────────────┐
│     SAPUI5 Freestyle Web Interface      │
│   (Enterprise UI with Fiori Design)     │
└──────────────┬──────────────────────────┘
               │ HTTP + OData V4
┌──────────────▼──────────────────────────┐
│      Zig HTTP + OData V4 Server         │
│  • Request routing & middleware         │
│  • OData service implementation         │
│  • WebSocket for streaming              │
└──────────────┬──────────────────────────┘
               │ FFI
┌──────────────▼──────────────────────────┐
│       Mojo Business Logic Core          │
│  • LLM inference (Shimmy integration)   │
│  • Document processing & embeddings     │
│  • Knowledge graph & mindmap generation │
│  • TOON encoding for optimization       │
└──────────────┬──────────────────────────┘
               │
┌──────────────▼──────────────────────────┐
│          Zig I/O Layer                   │
│  • Web scraping (HTTP + HTML parsing)   │
│  • PDF parsing & text extraction        │
│  • File system operations               │
└──────────────────────────────────────────┘
```

## 📁 Directory Structure

```
nHyperBook/
├── server/           # Zig HTTP + OData server
├── core/             # Mojo business logic
├── io/               # Zig I/O operations
├── webapp/           # SAPUI5 Freestyle UI
│   ├── controller/   # MVC controllers
│   ├── view/         # XML views
│   ├── model/        # Data models
│   ├── css/          # Styling
│   └── i18n/         # Internationalization
├── metadata/         # OData V4 metadata
├── lib/              # Compiled libraries
├── scripts/          # Build & deployment
├── tests/            # Test suite
├── docs/             # Documentation
└── templates/        # Generation templates
```

## 🚀 Quick Start

### Prerequisites

- **Mojo** 24.5+
- **Zig** 0.13.0+
- **LLVM** 17+
- Existing Shimmy-Mojo installation

### Build

```bash
# Build all components
./scripts/build_all.sh

# Build individual components
./scripts/build_server.sh  # Zig server
./scripts/build_core.sh    # Mojo core
./scripts/build_ui.sh      # SAPUI5 UI (optional)
```

### Run

```bash
# Start server
./scripts/start.sh

# Server will run on http://localhost:11434
# UI accessible at http://localhost:11434/
```

### Test

```bash
# Run all tests
./scripts/test.sh

# Run specific test suites
./scripts/test_unit.sh
./scripts/test_integration.sh
```

## 🔧 Technology Stack

### Backend
- **Zig 0.13.0** - HTTP server, I/O operations, OData implementation
- **Mojo 24.5+** - Business logic, LLM inference, ML operations
- **Shimmy-Mojo** - LLM inference engine with TOON encoding

### Frontend
- **SAPUI5** - Enterprise UI framework (SAP Fiori)
- **OData V4** - Data protocol
- **WebSocket** - Real-time streaming

### Integration
- **Qdrant** - Vector database for embeddings
- **LLaMA 3.2** - Local language models (1B/3B)
- **Graph Toolkit** - Knowledge graph operations

## 📊 Features

### ✅ Current Status (Week 1 - Day 1)

- [x] Project structure created
- [ ] Build system configured
- [ ] Basic HTTP server
- [ ] OData V4 metadata
- [ ] SAPUI5 bootstrap
- [ ] FFI bridge (Zig ↔ Mojo)

### 🚧 In Development

See [docs/implementation-plan.md](docs/implementation-plan.md) for the complete 12-week implementation schedule.

### 🎯 Planned Features

- Multi-source document ingestion (URLs, PDFs)
- Semantic search with vector embeddings
- Context-aware AI chat
- Research summary generation
- Interactive mindmap visualization
- Audio overview generation
- Presentation slide creation

## 🔌 API Endpoints

### OData V4 Service

**Base URL:** `http://localhost:11434/odata/v4/research/`

**Entities:**
- `Sources` - Document sources (CRUD)
- `Messages` - Chat messages
- `Summaries` - Generated summaries
- `MindmapNodes` - Mindmap data

**Actions:**
- `Chat` - Interactive chat with sources
- `GenerateSummary` - Create research summary
- `GenerateMindmap` - Generate knowledge graph
- `GenerateAudio` - Create audio overview
- `GenerateSlides` - Generate presentation

See [docs/api-reference.md](docs/api-reference.md) for detailed API documentation.

## 🧪 Testing

### Unit Tests

```bash
# Mojo unit tests
mojo test tests/unit/

# Zig unit tests
zig test tests/unit/
```

### Integration Tests

```bash
# End-to-end tests
./scripts/test_integration.sh
```

### Coverage

Target: 80%+ code coverage for production readiness

## 📚 Documentation

- [User Guide](docs/user-guide.md) - How to use HyperShimmy
- [API Reference](docs/api-reference.md) - Complete API documentation
- [Developer Guide](docs/developer-guide.md) - Development setup and contribution
- [Architecture](docs/architecture.md) - System design and components
- [Implementation Plan](docs/implementation-plan.md) - 12-week development schedule

## 🔒 Security

- Request rate limiting
- Input validation
- File upload restrictions
- CORS configuration
- Authentication (basic auth)
- CSP headers

See [docs/security.md](docs/security.md) for security guidelines.

## 🚀 Deployment

### Docker

```bash
# Build image
docker build -t hypershimmy:latest .

# Run container
docker-compose up -d
```

### Production

See [docs/deployment.md](docs/deployment.md) for production deployment guide.

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

1. Install prerequisites (Mojo, Zig, LLVM)
2. Clone repository
3. Run `./scripts/setup-dev.sh`
4. Build with `./scripts/build_all.sh`
5. Run tests with `./scripts/test.sh`

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- Built on top of [Shimmy-Mojo](../serviceShimmy-mojo/) LLM inference stack
- Inspired by [HyperbookLM](../../vendor/layerIntelligence/hyperbooklm/)
- Uses SAPUI5 for enterprise UI
- Integrates with SAP OData toolkit

## 📈 Project Status

**Version:** 0.1.0-dev  
**Status:** 🚧 In Development  
**Week:** 1 of 12  
**Day:** 1 of 60  
**Completion:** 0%

### Current Sprint (Week 1)

- [x] Day 1: Project setup
- [ ] Day 2: OData server foundation
- [ ] Day 3: OData V4 metadata
- [ ] Day 4: SAPUI5 bootstrap
- [ ] Day 5: FlexibleColumnLayout

---

**Contact:** [Your contact information]  
**Repository:** [Your repository URL]  
**Documentation:** [Your docs URL]

---

Made with 🔥 Mojo + ⚡ Zig
