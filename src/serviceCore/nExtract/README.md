# nExtract - Document Extraction Engine 🚀

**Version:** 1.0.0 (In Development)  
**Status:** Planning Phase  
**Target Completion:** June 2026  
**Location:** `/src/serviceCore/nExtract/`

---

## Overview

**nExtract** is a production-ready document extraction engine built entirely in **Zig** and **Mojo** with **zero external dependencies**. It replaces the Python-based Docling library with full feature parity, delivering 2-5x performance improvements and 50% memory reduction.

### Key Features

✅ **Zero External Dependencies** - All parsers, codecs, and ML models built from scratch  
✅ **Multi-Format Support** - PDF, DOCX, XLSX, PPTX, HTML, Markdown, CSV, images, audio  
✅ **Advanced PDF Processing** - Complete PDF 1.4-2.0 parser with text, images, and layout  
✅ **OCR Engine** - Character recognition from scratch (pure Zig/Mojo)  
✅ **ML Inference** - Custom neural network engine for layout analysis  
✅ **Image Codecs** - PNG and JPEG decoders  
✅ **Smart Chunking** - Hierarchical, semantic, token-based, and hybrid strategies  
✅ **Export Formats** - Markdown, HTML, JSON, DocTags  
✅ **HTTP Service** - REST API using Mojo SDK's Shimmy pattern  
✅ **CLI Tool** - Comprehensive command-line interface  

---

## Architecture

### Technology Stack

- **Zig 0.13+** - Low-level implementation (parsers, codecs, algorithms)
- **Mojo SDK v1.0** - High-level orchestration, API layer, service framework
- **mojo-bindgen** - Auto-generate FFI bindings (Zig → Mojo)
- **Pure Implementation** - No external libraries (no MuPDF, no Tesseract, no ONNX Runtime)

### Design Principles

1. **Performance First** - Zig for performance-critical components
2. **Safety First** - Memory-safe by design (Zig + Mojo SDK)
3. **Type Safety** - FFI bridge with auto-generated bindings
4. **Zero Dependencies** - Everything built from scratch
5. **Production Quality** - 85%+ test coverage, continuous fuzzing

---

## Project Structure

```
src/serviceCore/nExtract/
├── README.md                           # This file
├── NEXTRAKT_155_DAY_MASTER_PLAN.md    # Complete 155-day implementation plan
├── ARCHITECTURE.md                     # System architecture (to be created)
├── .gitignore                          # Git ignore patterns
├── zig/                                # Zig implementation
│   ├── build.zig                      # Build configuration
│   ├── core/                          # Core data structures
│   │   ├── types.zig                  # Document types (BoundingBox, Element, etc.)
│   │   ├── string.zig                 # UTF-8 string utilities
│   │   ├── allocator.zig              # Arena allocator, object pool
│   │   └── profiler.zig               # Memory profiler
│   ├── parsers/                       # Format parsers
│   │   ├── csv.zig                    # CSV parser (RFC 4180)
│   │   ├── markdown.zig               # Markdown parser (CommonMark + GFM)
│   │   ├── xml.zig                    # XML parser (XML 1.0)
│   │   ├── html.zig                   # HTML parser (HTML5)
│   │   ├── deflate.zig                # DEFLATE decompressor (RFC 1951)
│   │   ├── zip.zig                    # ZIP archive handler
│   │   ├── gzip.zig                   # GZIP format
│   │   ├── zlib.zig                   # ZLIB format
│   │   ├── ooxml.zig                  # Office Open XML structure
│   │   ├── png.zig                    # PNG decoder
│   │   └── jpeg.zig                   # JPEG decoder
│   ├── pdf/                           # PDF processing
│   │   ├── objects.zig                # PDF object model
│   │   ├── streams.zig                # Stream decompression
│   │   ├── content.zig                # Content stream parsing
│   │   ├── fonts.zig                  # Font handling
│   │   ├── text_positioning.zig       # Text positioning
│   │   ├── text_extraction.zig        # Text extraction
│   │   ├── unicode_mapping.zig        # Unicode mapping
│   │   ├── images.zig                 # Image extraction
│   │   ├── image_decode.zig           # Image decoding
│   │   ├── graphics.zig               # Vector graphics
│   │   ├── form_xobjects.zig          # Form XObjects
│   │   ├── annotations.zig            # Annotations
│   │   ├── outline.zig                # Bookmarks
│   │   ├── metadata.zig               # Metadata
│   │   └── forms.zig                  # AcroForm
│   ├── ocr/                           # OCR engine
│   │   ├── colorspace.zig             # Color space conversions
│   │   ├── filters.zig                # Image filters
│   │   ├── transform.zig              # Image transformations
│   │   ├── threshold.zig              # Thresholding
│   │   ├── line_detection.zig         # Text line detection
│   │   ├── char_segmentation.zig      # Character segmentation
│   │   ├── recognition.zig            # Character recognition
│   │   ├── ocr.zig                    # OCR pipeline
│   │   ├── multilang.zig              # Multi-language support
│   │   ├── layout.zig                 # Layout analysis for OCR
│   │   └── enhance.zig                # Image enhancement
│   ├── ml/                            # ML inference engine
│   │   ├── tensor.zig                 # Tensor operations
│   │   ├── nn.zig                     # Neural network layers
│   │   ├── layout_model.zig           # Layout detection model
│   │   ├── table_model.zig            # Table structure model
│   │   └── reading_order.zig          # Reading order model
│   └── tests/                         # Zig unit tests
│       ├── unit_tests.zig
│       ├── compression_test.zig
│       ├── ooxml_test.zig
│       ├── image_test.zig
│       ├── image_processing_test.zig
│       ├── ocr_test.zig
│       ├── ml_test.zig
│       ├── pdf_parser_test.zig
│       ├── pdf_text_test.zig
│       ├── pdf_graphics_test.zig
│       ├── pdf_advanced_test.zig
│       ├── layout_test.zig
│       ├── docx_test.zig
│       ├── xlsx_test.zig
│       └── pptx_test.zig
├── mojo/                              # Mojo high-level API
│   ├── ffi.mojo                       # Auto-generated FFI bindings
│   ├── core.mojo                      # Core types (wrappers)
│   ├── pipeline.mojo                  # Pipeline framework
│   ├── simple_pipeline.mojo           # Simple pipeline
│   ├── paginated_pipeline.mojo        # Paginated pipeline
│   ├── standard_pipeline.mojo         # Standard pipeline (caching)
│   ├── assembly.mojo                  # Element assembly
│   ├── hierarchy.mojo                 # Hierarchy construction
│   ├── metadata.mojo                  # Metadata extraction
│   ├── provenance.mojo                # Provenance tracking
│   ├── reading_order.mojo             # Reading order algorithm
│   ├── structure_inference.mojo       # Structure inference
│   ├── semantic_analysis.mojo         # Semantic analysis
│   ├── export_markdown.mojo           # Markdown export
│   ├── export_html.mojo               # HTML export
│   ├── export_json.mojo               # JSON export
│   ├── export_doctags.mojo            # DocTags export
│   ├── chunking/                      # Chunking strategies
│   │   ├── hierarchical.mojo
│   │   ├── semantic.mojo
│   │   ├── token_based.mojo
│   │   └── hybrid.mojo
│   ├── image_classification.mojo      # Image classification
│   ├── image_captioning.mojo          # Image captioning
│   ├── audio_processing.mojo          # Audio metadata
│   ├── webvtt_parser.mojo             # WebVTT parser
│   ├── converter.mojo                 # DocumentConverter API
│   ├── streaming.mojo                 # Streaming support
│   ├── concurrency.mojo               # Concurrency management
│   ├── service.mojo                   # HTTP service (Shimmy pattern)
│   ├── http_handlers.mojo             # Request/response handlers
│   ├── middleware.mojo                # Middleware stack
│   ├── cli.mojo                       # CLI tool
│   ├── cli_advanced.mojo              # Advanced CLI features
│   ├── config.mojo                    # Configuration system
│   ├── logging.mojo                   # Progress & logging
│   └── tests/                         # Mojo integration tests
│       ├── unit_tests.mojo
│       ├── assembly_test.mojo
│       ├── structure_test.mojo
│       ├── export_test.mojo
│       ├── chunking_test.mojo
│       ├── media_test.mojo
│       ├── converter_test.mojo
│       └── service_test.mojo
├── docs/                              # Documentation
│   ├── getting-started.md
│   ├── api/                           # API documentation
│   ├── ARCHITECTURE.md                # Architecture docs
│   ├── USER_GUIDE.md                  # User guide
│   ├── CONTRIBUTING.md                # Developer guide
│   ├── MIGRATION.md                   # Migration from Docling
│   ├── SECURITY.md                    # Security documentation
│   ├── security_audit.md
│   ├── fuzzing_results.md
│   ├── static_analysis.md
│   ├── code_review.md
│   ├── PARITY_VALIDATION.md
│   ├── DOCLING_DIFFERENCES.md
│   ├── MIGRATION_PLAYBOOK.md
│   └── ROLLBACK_PLAN.md
├── tests/                             # End-to-end tests
│   ├── fixtures/                      # Test documents
│   │   ├── pdf/
│   │   ├── docx/
│   │   ├── xlsx/
│   │   ├── pptx/
│   │   ├── html/
│   │   ├── markdown/
│   │   └── csv/
│   ├── integration/                   # Integration tests
│   │   └── integration_tests.mojo
│   ├── benchmarks/                    # Performance benchmarks
│   │   └── benchmarks.mojo
│   ├── fuzz/                          # Fuzzing infrastructure
│   │   ├── fuzz_pdf.zig
│   │   ├── fuzz_xml.zig
│   │   ├── fuzz_html.zig
│   │   ├── fuzz_zip.zig
│   │   ├── fuzz_deflate.zig
│   │   ├── fuzz_png.zig
│   │   ├── fuzz_jpeg.zig
│   │   ├── fuzz_ocr.zig
│   │   └── fuzz_ml.zig
│   └── reporting/                     # Test reporting
│       └── coverage_report.html
└── tools/                             # Development tools
    └── model_converter.zig            # ONNX → custom format converter
```

---

## Implementation Plan

See **[NEXTRAKT_155_DAY_MASTER_PLAN.md](NEXTRAKT_155_DAY_MASTER_PLAN.md)** for the complete 155-day implementation plan with day-by-day breakdown.

### Timeline Overview

| Phase | Days | Focus | Status |
|-------|------|-------|--------|
| Phase 1 | 1-25 | Foundation & Core Infrastructure | 📋 Planned |
| Phase 2 | 26-45 | Advanced Image Processing & OCR | 📋 Planned |
| Phase 3 | 46-70 | PDF Processing | 📋 Planned |
| Phase 4 | 71-85 | Office Format Implementation | 📋 Planned |
| Phase 5 | 86-105 | Pipeline & API | 📋 Planned |
| Phase 6 | 106-115 | Advanced Features | 📋 Planned |
| Phase 7 | 116-125 | Service & CLI | 📋 Planned |
| Phase 8 | 126-135 | CLI & Tooling | 📋 Planned |
| Phase 9 | 136-150 | Finalization & Release | 📋 Planned |
| Post-Impl | 151-155 | Cleanup & Deployment | 📋 Planned |

---

## Quick Start (Future)

Once implemented, usage will be:

### CLI

```bash
# Convert single document
nextract convert document.pdf output.md

# Batch convert
nextract batch ./documents/ ./output/ --recursive

# Extract components
nextract extract document.pdf --images ./images/

# Show document info
nextract info document.pdf
```

### Mojo API

```mojo
from nExtract.converter import DocumentConverter

fn main():
    let converter = DocumentConverter()
    
    # Convert single document
    let result = converter.convert("document.pdf")
    if result.is_ok():
        let doc = result.unwrap()
        let markdown = doc.export_to_markdown()
        print(markdown)
```

### HTTP API

```bash
# Convert document via REST API
curl -X POST http://localhost:8080/convert \
  -F "file=@document.pdf" \
  -H "Accept: application/json"
```

---

## Supported Formats

### Input Formats

| Format | Extension | Status | Notes |
|--------|-----------|--------|-------|
| PDF | .pdf | 📋 Planned | Full PDF 1.4-2.0 support |
| Word | .docx | 📋 Planned | OOXML format |
| Excel | .xlsx | 📋 Planned | OOXML format |
| PowerPoint | .pptx | 📋 Planned | OOXML format |
| HTML | .html, .htm | 📋 Planned | HTML5 parser |
| Markdown | .md | 📋 Planned | CommonMark + GFM |
| CSV | .csv | 📋 Planned | RFC 4180 |
| Images | .png, .jpg | 📋 Planned | PNG, JPEG decoders |
| Audio | .mp3, .wav | 📋 Planned | Metadata only |
| WebVTT | .vtt | 📋 Planned | Video subtitles |

### Export Formats

| Format | Extension | Features |
|--------|-----------|----------|
| Markdown | .md | GFM tables, code blocks, math |
| HTML | .html | Semantic HTML5, CSS styling |
| JSON | .json | Lossless serialization |
| DocTags | .xml | Custom tag-based format |

---

## Features

### Document Processing

- **Text Extraction** - Precise text extraction with positioning
- **Layout Analysis** - ML-based page segmentation and reading order
- **Table Recognition** - Detect and extract table structures
- **Image Extraction** - Extract all images with metadata
- **OCR Support** - Process scanned documents
- **Formula Detection** - Detect and extract mathematical formulas
- **Code Block Detection** - Identify code snippets
- **Metadata Extraction** - Extract title, author, dates, etc.

### Advanced Features

- **Smart Chunking** - 4 chunking strategies (hierarchical, semantic, token-based, hybrid)
- **Multi-Language** - Support for Latin, Cyrillic, Greek scripts
- **Right-to-Left** - RTL language support (Arabic, Hebrew)
- **Streaming** - Process large documents incrementally
- **Parallel Processing** - Multi-threaded page processing
- **Caching** - Incremental processing with result caching

### API & Tools

- **REST API** - Complete HTTP service with middleware stack
- **CLI Tool** - Comprehensive command-line interface
- **Configuration** - TOML config files, environment variables
- **Progress Tracking** - Real-time progress bars and callbacks
- **Structured Logging** - JSON logging with request tracing

---

## Performance Goals

| Metric | Target | vs Docling |
|--------|--------|------------|
| Conversion Speed | 12+ pages/sec | 2-5x faster |
| Memory Usage | <50 MB/doc | 50% less |
| OCR Accuracy | >95% CAR | Comparable |
| Test Coverage | >85% | Better |

---

## Development

### Prerequisites

- **Zig 0.13+** - For building Zig components
- **Mojo SDK v1.0** - Located at `/src/serviceCore/serviceShimmy-mojo/mojo-sdk`
- **mojo-bindgen** - Available in Mojo SDK tools

### Building from Source

```bash
# Navigate to project
cd src/serviceCore/nExtract

# Build Zig components
cd zig
zig build

# Generate Mojo FFI bindings
mojo-bindgen zig/core/types.zig --output mojo/ffi.mojo

# Build Mojo components
cd ../mojo
mojo build

# Run tests
zig build test  # Zig tests
mojo test mojo/tests/  # Mojo tests
```

### Testing

```bash
# Run all tests
./test_runner.sh

# Run specific test suite
zig test zig/tests/pdf_parser_test.zig

# Run fuzzing
cd tests/fuzz
./fuzz_pdf corpus/ -max_total_time=3600
```

---

## Contributing

This project is in active development. See [NEXTRAKT_155_DAY_MASTER_PLAN.md](NEXTRAKT_155_DAY_MASTER_PLAN.md) for the implementation roadmap.

### Development Phases

Current phase: **Phase 0 - Planning** ✅

Next phase: **Phase 1 - Foundation & Core Infrastructure** (Days 1-25)

---

## Documentation

- **[Master Plan](NEXTRAKT_155_DAY_MASTER_PLAN.md)** - Complete 155-day implementation plan
- **[Architecture](ARCHITECTURE.md)** - System architecture (to be created)
- **[User Guide](docs/USER_GUIDE.md)** - User documentation (to be created)
- **[API Reference](docs/api/)** - API documentation (to be created)
- **[Contributing](docs/CONTRIBUTING.md)** - Developer guide (to be created)

---

## Related Projects

- **[Mojo SDK](../serviceShimmy-mojo/mojo-sdk/)** - Custom Mojo implementation with compiler, stdlib, and tooling
- **[Docling (Original)](../../vendor/layerCore/docling/)** - Python-based document extraction (to be replaced)

---

## License

MIT License - See LICENSE file for details

---

## Status

**Current Status:** Planning Complete ✅  
**Next Step:** Begin Phase 1 (Foundation) - Day 1  
**Estimated Completion:** June 2026  

**Project Goals:**
- ✅ Zero external dependencies
- ✅ Full feature parity with Docling
- ✅ 2-5x performance improvement
- ✅ 50% memory reduction
- ✅ Production-grade quality
- ✅ 85%+ test coverage

---

**Last Updated:** January 17, 2026  
**Maintained by:** Development Team  
**Questions?** See documentation or open an issue.
