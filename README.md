# AI Nucleus - Arabic Invoice Processing System

**Intelligent invoice processing with ToolOrchestra orchestration and A2UI agent-generated interfaces**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Rust](https://img.shields.io/badge/Rust-1.70+-orange.svg)](https://www.rust-lang.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)

## Overview

AI Nucleus is a comprehensive system for processing Arabic invoices using AI models (CamelBERT for analysis, M2M100 for translation) with intelligent orchestration powered by ToolOrchestra-style workflows and A2UI for agent-generated user interfaces.

### Key Features

- 🤖 **AI-Powered Processing**: Arabic invoice translation and analysis
- 🔄 **Intelligent Orchestration**: ToolOrchestra-style workflow engine
- 🎨 **A2UI Integration**: Agent-generated declarative UIs
- 🏗️ **Hybrid Architecture**: Rust (performance) + Python (ML models)
- 🎯 **SAP Design System**: Professional UI matching SAP Fiori
- 📊 **Visual Workflow Designer**: OpenCanvas-style flow editor
- 🔒 **Enterprise Security**: Authentication, rate limiting, input validation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (HTML/JS)                   │
│              SAP-styled UI + A2UI Renderer               │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/REST
┌────────────────────▼────────────────────────────────────┐
│              Python FastAPI Backend                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Model Service│  │ Orchestration │  │ A2UI Adapter │ │
│  │ (CamelBERT,  │  │   Adapter     │  │              │ │
│  │   M2M100)    │  │               │  │              │ │
│  └──────────────┘  └──────┬───────┘  └──────────────┘ │
└───────────────────────────┼────────────────────────────┘
                            │ HTTP
┌───────────────────────────▼────────────────────────────┐
│            Rust Workflow Engine (Shimmy)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Orchestration│  │  Workflow    │  │   A2UI       │ │
│  │   Service    │  │   Engine     │  │   Types      │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.8+
- Rust 1.70+ (for workflow engine)
- Node.js (optional, for frontend development)

### Installation

1. **Clone and setup environment:**

```bash
cd arabic_folder

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install development dependencies (optional)
pip install -r requirements-dev.txt
```

2. **Configure environment:**

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# At minimum, set:
# - RUST_BACKEND_URL (if Rust service is on different port)
# - CORS_ORIGINS (for frontend access)
```

3. **Start the backend:**

```bash
# Start Python FastAPI server
python -m backend.api.server

# Or using uvicorn directly
uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

4. **Start Rust workflow engine (optional):**

```bash
cd shimmy-ai
cargo build --release
./target/release/shimmy serve --bind 127.0.0.1:11435
```

5. **Open frontend:**

```bash
# Open frontend/index.html in your browser
# Or serve with a simple HTTP server:
python -m http.server 3000 -d frontend
```

## Project Structure

```
arabic_folder/
├── backend/                 # Python backend
│   ├── api/                 # API routes and server
│   │   ├── routes/          # Route modules
│   │   ├── errors.py        # Error handling
│   │   └── server.py        # FastAPI app
│   ├── adapters/           # External service adapters
│   │   ├── orchestration.py # Rust workflow adapter
│   │   └── a2ui.py          # A2UI adapter
│   ├── schemas/            # Pydantic schemas
│   │   └── workflow.py      # Workflow definitions
│   ├── services/           # Business logic
│   │   └── model_service.py # ML model service
│   ├── config/             # Configuration
│   │   └── settings.py      # Settings management
│   ├── utils/              # Utilities
│   │   ├── logging.py       # Logging setup
│   │   └── validation.py    # Input validation
│   ├── auth/               # Authentication
│   └── constants.py         # Application constants
│
├── frontend/               # Frontend application
│   ├── index.html          # Main application
│   └── static/             # Static assets
│       ├── css/
│       ├── js/
│       └── assets/
│
├── vendor/                # Vendored dependencies
│   ├── shimmy-ai/         # Rust workflow engine
│   │   ├── src/
│   │   │   ├── orchestration.rs # Orchestration service
│   │   │   ├── workflow.rs      # Workflow engine
│   │   │   ├── a2ui.rs          # A2UI types
│   │   │   ├── memgraph.rs      # Memgraph integration
│   │   │   └── ...
│   │   └── Cargo.toml
│   ├── a2ui/              # Google A2UI
│   └── open-canvas/       # LangChain Open Canvas
│
├── models/                # ML models
│   └── arabic_models/
│       ├── camelbert-dialect-financial/
│       └── m2m100-418M/
│
├── data/                  # Data files
│   ├── invoices/
│   ├── templates/
│   └── emails/
│
├── tests/                 # Tests
│   ├── python/            # Python tests
│   │   ├── unit/
│   │   └── integration/
│   └── rust/              # Rust tests
│
├── config/                # Configuration files
│   ├── development.yaml
│   └── production.yaml
│
├── vendor/                # Vendored dependencies
│   └── a2ui/              # A2UI integration
│
├── docs/                  # Documentation
│   ├── architecture.md
│   ├── api.md
│   └── deployment.md
│
├── requirements.txt        # Production dependencies
├── requirements-dev.txt    # Development dependencies
├── requirements-test.txt   # Test dependencies
├── pytest.ini             # Pytest configuration
├── .env.example           # Environment template
└── README.md              # This file
```

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Application
APP_NAME=AI Nucleus Backend
ENVIRONMENT=development
DEBUG=true

# Server
HOST=0.0.0.0
PORT=8000

# CORS (comma-separated)
CORS_ORIGINS=http://localhost:8000,http://127.0.0.1:8000

# Rust Backend
RUST_BACKEND_URL=http://127.0.0.1:11435

# Security
ENABLE_AUTH=false
API_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=60

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

### Model Paths

Models are loaded from `./models/arabic_models/` by default. You can override paths:

```bash
MODELS_DIR=./models/arabic_models
CAMELBERT_PATH=./models/arabic_models/camelbert-dialect-financial
M2M100_PATH=./models/arabic_models/m2m100-418M
```

## API Documentation

### Model Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/translate` - Translate Arabic text
- `POST /api/v1/analyze` - Analyze invoice text

### Orchestration Endpoints

- `POST /orchestrate/execute` - Execute workflow
- `GET /orchestrate/workflows` - List workflows
- `GET /orchestrate/workflows/{id}` - Get workflow
- `POST /orchestrate/workflows` - Save workflow
- `DELETE /orchestrate/workflows/{id}` - Delete workflow

### A2UI Endpoints

- `POST /a2ui/generate` - Generate A2UI from data
- `POST /a2ui/workflow-result` - Convert workflow result to A2UI

### Interactive API Docs

When `DEBUG=true`, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test file
pytest tests/python/unit/test_schemas.py

# Run integration tests
pytest tests/python/integration/
```

### Code Quality

```bash
# Format code
black backend/ tests/

# Lint code
ruff check backend/ tests/

# Type checking
mypy backend/
```

### Project Setup Script

```bash
# Run setup script (if available)
./scripts/setup.sh
```

## Deployment

### Production Checklist

- [ ] Set `ENVIRONMENT=production` in `.env`
- [ ] Set `DEBUG=false`
- [ ] Configure `CORS_ORIGINS` with production domains
- [ ] Set `ENABLE_AUTH=true` and configure `API_KEY`
- [ ] Use proper secrets management
- [ ] Configure logging to file/external service
- [ ] Set up monitoring and alerting
- [ ] Configure reverse proxy (nginx/traefik)
- [ ] Set up SSL/TLS certificates

### Docker Deployment

```bash
# Build Docker image
docker build -t ai-nucleus:latest .

# Run container
docker run -p 8000:8000 --env-file .env ai-nucleus:latest
```

## Architecture Details

### Workflow Orchestration

The system uses a hybrid approach:
- **Rust workflow engine** for performance-critical orchestration
- **Python adapters** for ML model integration
- **ToolOrchestra-style** intelligent routing and model selection

### A2UI Integration

A2UI (Agent-to-User Interface) allows AI agents to generate declarative UI components:
- Secure: Declarative format prevents code injection
- Framework-agnostic: Same JSON works across platforms
- LLM-friendly: Easy for models to generate

See [A2UI_INTEGRATION_SUMMARY.md](A2UI_INTEGRATION_SUMMARY.md) for details.

### Security Features

- ✅ Environment-based configuration
- ✅ API key authentication (optional)
- ✅ Rate limiting
- ✅ Input validation and sanitization
- ✅ Path validation
- ✅ Security headers
- ✅ CORS configuration

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Run tests and linting
6. Submit a pull request

## License

[Specify your license]

## Support

For issues and questions:
- Create an issue in the repository
- Check documentation in `docs/`
- Review [CODE_REVIEW.md](CODE_REVIEW.md) for architecture details

## Acknowledgments

- [ToolOrchestra](https://github.com/NVlabs/ToolOrchestra) - Workflow orchestration inspiration
- [A2UI](https://github.com/google/A2UI) - Agent-to-User Interface
- [Shimmy](https://github.com/Michael-A-Kuykendall/shimmy) - Rust workflow engine
- SAP Fiori Design System - UI design guidelines

