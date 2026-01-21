# nExtract - Document Extraction Engine

**Version:** 1.0.0-dev  
**Status:** In Development (Day 1/155)  
**License:** MIT  

## Overview

nExtract is a production-ready document extraction engine built entirely in **Zig** and **Mojo** with **zero external dependencies**. It replaces three Python libraries with a unified, high-performance solution:

- ✅ **Replaces Docling** - Complete document parsing and layout analysis
- ✅ **Replaces MarkItDown** - Document-to-Markdown conversion with full feature parity  
- ✅ **Replaces LangExtract** - LLM-powered structured extraction with source grounding
- ✅ **Integrates with nOpenaiServer** - Local LLM inference (no cloud dependencies)

## Key Features

### Core Capabilities
- **Zero External Dependencies** - All parsers, codecs, and ML inference built from scratch
- **Pure Zig/Mojo Implementation** - No MuPDF, Tesseract, or ONNX Runtime
- **Production Quality** - Memory-safe, fast, thoroughly tested
- **Local LLM Integration** - Uses nOpenaiServer for structured extraction

### Supported Formats

**Input Formats:**
- PDF (1.4-2.0)
- Office: DOCX, XLSX, PPTX (OOXML)
- Text: CSV, Markdown, HTML, XML, TXT
- Images: PNG, JPEG (with OCR)
- Archives: ZIP, GZIP, ZLIB

**Output Formats:**
- Markdown
- HTML5
- JSON
- DocTags (custom format)

### Advanced Features
- **OCR Engine** - Character recognition from scratch
- **ML-Based Layout Analysis** - Custom neural network inference
- **Structured Extraction** - LLM-powered with source grounding
- **Chunking Strategies** - Hierarchical, semantic, token-based, hybrid
- **HTTP Service** - REST API with Shimmy pattern
- **CLI Tool** - Complete command-line interface

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                        nExtract                         │
│  (Document Processing + Structured Extraction)          │
└───────────────┬─────────────────────────────────────────┘
                │
                ├──► Document Parsing (Zig)
                │    └─► PDF, Office, HTML, etc.
                │
                ├──► Layout Analysis (Zig + ML)
                │    └─► OCR, Table Detection, Reading Order
                │
                ├──► Export (Mojo)
                │    └─► Markdown, HTML, JSON
                │
                └──► Structured Extraction (Mojo + nOpenaiServer)
                     ├─► Few-shot prompting
                     ├─► Chunking & parallel processing
                     ├─► Source grounding & span extraction
                     └─► HTTP call to nOpenaiServer:11434
                         └─► Local GGUF models (Qwen, Llama, etc.)
```

## Project Structure

```
src/serviceCore/nExtract/
├── zig/                    # Zig implementation (low-level)
│   ├── build.zig          # Build system configuration
│   ├── core/              # Core data structures
│   ├── parsers/           # Document parsers
│   ├── ocr/               # OCR engine
│   ├── ml/                # ML inference engine
│   ├── pdf/               # PDF processing
│   └── tests/             # Unit tests
├── mojo/                  # Mojo implementation (high-level)
│   ├── ffi.mojo          # FFI bindings
│   ├── core.mojo         # Core API
│   ├── pipeline.mojo     # Processing pipelines
│   ├── service.mojo      # HTTP service
│   └── cli.mojo          # CLI tool
├── docs/                  # Documentation
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- **Zig 0.13+** - [Installation guide](https://ziglang.org/download/)
- **Mojo SDK v1.0+** - [Installation guide](https://docs.modular.com/mojo/)
- **Git** - For version control

### Building from Source

```bash
# Clone repository
cd src/serviceCore/nExtract

# Build Zig libraries
cd zig
zig build

# Install libraries
zig build install

# Run tests
zig build test

# Build documentation
zig build docs
```

### Running Tests

```bash
# All tests
zig build test

# Specific test suite
zig build test --test-filter "core"

# Benchmarks
zig build bench

# Fuzzing
zig build fuzz
```

## Development Status

🚧 **Currently implementing Day 1/155 of the master plan**

### Phase 1: Foundation & Core Infrastructure (Days 1-25)
- [x] Day 1: Project Architecture & Build System (In Progress)
- [ ] Day 2: Core Data Structures
- [ ] Day 3: Mojo FFI Layer
- [ ] Day 4: String & Text Utilities
- [ ] Day 5: Memory Management Infrastructure

See [NEXTRAKT_155_DAY_MASTER_PLAN.md](NEXTRAKT_155_DAY_MASTER_PLAN.md) for complete implementation plan.

## Project Statistics (Target)

| Component | Language | Lines | Status |
|-----------|----------|-------|--------|
| Zig Core & Parsers | Zig | ~25,000 | 📋 Planned |
| Mojo API Layer | Mojo | ~10,000 | 📋 Planned |
| Tests | Both | ~15,000 | 📋 Planned |
| Documentation | Markdown | ~5,000 | 📋 Planned |
| **Total** | | **~55,000** | |

### Test Coverage Goal
- Unit tests: 1,000+
- Integration tests: 200+
- Fuzzing: Continuous
- Target coverage: 85%+

## Performance Targets

- **Speed:** 2-5x faster than Python Docling
- **Memory:** 50% less than Python Docling
- **Accuracy:** >95% OCR character accuracy
- **Throughput:** 10+ pages/second (PDF)

## Contributing

This project is currently in active development. Contributions will be welcome after v1.0.0 release.

### Development Workflow

1. Follow the 155-day master plan
2. Write tests first (TDD approach)
3. Run `zig build fmt` before committing
4. Run full test suite: `zig build test`
5. Update documentation as needed

## License

MIT License - See LICENSE file for details

## Contact

For questions or issues, please refer to the project's issue tracker.

## Acknowledgments

This project replaces and improves upon:
- **Docling** - Document parsing and layout analysis
- **MarkItDown** - Document-to-Markdown conversion
- **LangExtract** - LLM-powered structured extraction

Built with:
- **Zig** - Low-level systems programming language
- **Mojo** - High-performance Python-like language
- **nOpenaiServer** - Local LLM inference engine

---

**Last Updated:** January 17, 2026  
**Current Phase:** Phase 1 - Foundation & Core Infrastructure  
**Days Completed:** 0/155
