# serviceCore

**Purpose:** Core intellectual property - Custom-built "n*" projects

---

## Overview

This directory contains our proprietary technology stack, all prefixed with "n*" (nucleus). These are custom-built services primarily implemented in Zig and Mojo for maximum performance and zero external dependencies.

## Projects

### 🎵 nAudioLab
Audio processing, transcription, and analysis service.

**Technologies:** Python, ML models  
**Status:** Active

---

### 💻 nCode
Code generation, analysis, and transformation service.

**Technologies:** Python, Zig  
**Status:** Active

---

### 📄 nExtract
**NEW - Unified document extraction engine**

Replaces three Python libraries with a single Zig/Mojo solution:
- ✅ Docling (document parsing)
- ✅ MarkItDown (markdown conversion)
- ✅ LangExtract (structured extraction)

**Technologies:** Zig, Mojo  
**Status:** Planning (155-day implementation)  
**Documentation:** 
- `nExtract/NEXTRAKT_155_DAY_MASTER_PLAN.md`
- `nExtract/INTEGRATION_SUMMARY.md`

**Key Features:**
- PDF, Office (DOCX/XLSX/PPTX), HTML, CSV, Markdown parsing
- OCR engine (pure Zig/Mojo)
- ML-based layout analysis
- Structured extraction via nOpenaiServer
- Export to Markdown, HTML, JSON, DocTags
- Chunking for RAG applications
- Zero external dependencies

---

### 📚 nHyperBook
Hypertext book generation and management system.

**Technologies:** Python, Mojo  
**Status:** Active

---

### 🔬 nLeanProof
Lean theorem proving and formal verification service.

**Technologies:** Lean 4, Python, Mojo  
**Status:** Active

---

### 🤖 nOpenaiServer
**Formerly:** serviceShimmy-mojo

Local LLM inference server with OpenAI-compatible API.

**Technologies:** Zig, Mojo  
**Status:** Active  
**Port:** 11434

**Key Features:**
- GGUF model support (Qwen, Llama, Mistral, etc.)
- OpenAI-compatible REST API
- Local inference (no data leaves infrastructure)
- Used by nExtract for structured extraction
- GPU acceleration support

---

## Architecture Principles

### 1. Zero External Dependencies
All n* projects aim to minimize external dependencies, building functionality from scratch where it makes sense for performance, security, and maintainability.

### 2. High Performance
- Zig for low-level, performance-critical code
- Mojo for high-level orchestration with native speed
- SIMD optimizations throughout

### 3. Type Safety
- Leveraging Zig's compile-time safety
- Mojo's ownership and borrowing system
- Strong typing across all projects

### 4. Local-First
- No cloud API dependencies
- nOpenaiServer provides local LLM inference
- Full privacy and security control

---

## Development Guidelines

### Language Choices

**Zig** (Primary for new projects):
- Performance-critical parsers and algorithms
- System-level code
- FFI boundaries

**Mojo** (Primary for new projects):
- High-level APIs and orchestration
- Service layer implementation
- ML inference pipelines

**Python** (Legacy, being phased out):
- Existing services (nAudioLab, nCode, nHyperBook, nLeanProof)
- To be gradually migrated to Zig/Mojo

### Project Structure

Each n* project should follow this structure:
```
nProjectName/
├── README.md           # Project documentation
├── zig/                # Zig implementation
│   ├── build.zig      # Build configuration
│   └── src/           # Source code
├── mojo/               # Mojo implementation
│   └── src/           # Source code
├── tests/              # Test suite
└── docs/               # Additional documentation
```

### Testing Requirements
- Unit tests: 80%+ coverage
- Integration tests for all major features
- Performance benchmarks
- Fuzzing for parsers and security-critical code

---

## Migration from Python

As we migrate services from Python to Zig/Mojo:

1. **Maintain API compatibility** during transition
2. **Run in parallel** for validation period
3. **Benchmark performance** improvements
4. **Document differences** in behavior
5. **Archive Python code** before removal

---

## Relationship with serviceVendor

- **serviceCore**: Our proprietary technology (this directory)
- **serviceVendor**: External/vendor services and integrations

Services in serviceCore should minimize dependencies on serviceVendor, maintaining independence and portability.

---

## Getting Started

### Prerequisites
- Zig 0.13+ (for Zig projects)
- Mojo SDK v1.0+ (for Mojo projects)
- Python 3.10+ (for legacy projects)

### Building Projects

Each project has its own build system. Refer to individual project READMEs for specific instructions.

---

## Future Roadmap

### Short-term (Q1-Q2 2026)
- ✅ Complete nExtract implementation (155 days)
- ✅ Deprecate Python-based document processing
- ✅ Integrate nOpenaiServer with all n* projects

### Medium-term (Q3-Q4 2026)
- Migrate nAudioLab to Zig/Mojo
- Migrate nCode to Zig/Mojo
- Migrate nHyperBook to Zig/Mojo
- Migrate nLeanProof to Zig/Mojo

### Long-term (2027+)
- Complete Python phase-out
- Unified service framework across all n* projects
- Advanced ML capabilities built into core
- Multi-modal processing (text, image, audio, video)

---

**Last Updated:** January 17, 2026  
**Total Projects:** 6  
**Primary Languages:** Zig, Mojo, Python (legacy)
