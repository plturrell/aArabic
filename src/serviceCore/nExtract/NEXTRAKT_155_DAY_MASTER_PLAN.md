# nExtract - Complete 155-Day Implementation Plan

**Version:** 1.0  
**Project:** nExtract - Document Extraction Engine  
**Start Date:** January 2026  
**Target Completion:** June 2026  
**Architecture:** Pure Zig/Mojo Implementation (Zero External Dependencies)  
**Target Location:** `/src/serviceCore/nExtract/`

---

## Executive Summary

This plan details the complete implementation of **nExtract**, a production-ready document extraction engine built entirely in Zig and Mojo with **zero external dependencies**. This project will:

1. **Replace** the legacy Docling vendor package - Complete document parsing and layout analysis
2. **Replace** the legacy MarkItDown vendor service - Document-to-Markdown conversion with full feature parity
3. **Replace** the legacy LangExtract vendor service - LLM-powered structured extraction with source grounding
4. **Leverage** `/src/serviceCore/nOpenaiServer` - Local LLM inference engine (already implemented!)

### Key Objectives

1. ✅ **Zero External Dependencies** - All parsers, codecs, and ML inference built from scratch
2. ✅ **Full Feature Parity** - Replace three Python libraries with unified Zig/Mojo solution
3. ✅ **Production Quality** - Memory-safe, fast, thoroughly tested
4. ✅ **Mojo SDK Integration** - Leverage existing infrastructure (FFI, Service Framework, Async)
5. ✅ **Pure Implementation** - No MuPDF, no Tesseract, no ONNX Runtime
6. ✅ **Local LLM Integration** - Use nOpenaiServer for structured extraction (no cloud API dependencies)

### What We're Building

#### Core Document Processing (replaces Docling & MarkItDown)
- **PDF Parser** - Complete PDF 1.4-2.0 implementation
- **Office Formats** - DOCX, XLSX, PPTX parsers (OOXML)
- **Text Formats** - CSV, Markdown, HTML, XML parsers
- **OCR Engine** - Character recognition from scratch
- **ML Inference** - Custom neural network inference engine
- **Image Codecs** - PNG and JPEG decoders
- **Layout Analysis** - Geometric and ML-based document understanding
- **Export Formats** - Markdown, HTML, JSON, DocTags
- **Audio Processing** - Audio metadata and transcription support
- **Video/YouTube** - Video transcription support

#### Structured Extraction (replaces LangExtract)
- **LLM-Powered Extraction** - Use nOpenaiServer for entity extraction
- **Source Grounding** - Map extractions to exact source locations
- **Schema Enforcement** - Structured output with validation
- **Chunking Strategies** - Optimized for long documents
- **Visualization** - Interactive HTML highlighting
- **Few-Shot Learning** - Define extraction tasks with examples

#### Service Layer
- **HTTP Service** - REST API using Mojo SDK's Shimmy pattern
- **CLI Tool** - Command-line interface for conversions and extraction
- **Streaming API** - Process large documents incrementally

---

## Technology Stack

### Languages & Tools
- **Zig 0.13+** - Low-level implementation (parsers, codecs, algorithms)
- **Mojo SDK v1.0** - High-level orchestration, API layer, service framework
- **mojo-bindgen** - Auto-generate FFI bindings (Zig → Mojo)
- **nOpenaiServer** - Local LLM inference (GGUF models, OpenAI-compatible API)
- **LLVM** - Backend compilation (via Mojo SDK)

### Architecture Principles
1. **Zig for Performance** - All performance-critical code in Zig
2. **Mojo for Ergonomics** - API, pipelines, service layer in Mojo
3. **FFI Bridge** - Type-safe communication via generated bindings
4. **Memory Safety** - Leverage both languages' safety features
5. **SIMD Optimization** - Use Mojo SDK's SIMD support where applicable
6. **Local LLM Integration** - nOpenaiServer provides structured extraction without cloud dependencies

### Integration Architecture

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

---

## Project Statistics (Estimated)

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

---

# PHASE 1: Foundation & Core Infrastructure (Days 1-25)

## WEEK 1: Days 1-5 - Project Foundation

### **DAY 1: Project Architecture & Build System**

**Goals:**
1. Initialize project structure in `/src/serviceCore/nExtract/`
2. Configure Zig build system (`build.zig`)
3. Set up Mojo package integration
4. Design FFI architecture
5. CI/CD pipeline (GitHub Actions or similar)

**Deliverables:**
- `build.zig` with library targets
- `.gitignore` for Zig/Mojo artifacts
- `README.md` with project overview
- CI configuration (test, build, lint)
- Git hooks for pre-commit checks

**Files Created:**
- `src/serviceCore/nExtract/zig/build.zig`
- `src/serviceCore/nExtract/.gitignore`
- `src/serviceCore/nExtract/README.md`
- `.github/workflows/nExtract.yml`

---

### **DAY 2: Core Data Structures (Zig)**

**Goals:**
1. Implement geometric types
2. Document structure representation
3. Element type definitions
4. Memory management patterns

**Deliverables:**

**File:** `zig/core/types.zig` (~500 lines)
```zig
// Core geometric types
pub const Point = struct {
    x: f32,
    y: f32,
};

pub const Size = struct {
    width: f32,
    height: f32,
};

pub const BoundingBox = struct {
    x: f32,
    y: f32,
    width: f32,
    height: f32,
};

// Document structure
pub const DoclingDocument = struct {
    pages: []Page,
    metadata: Metadata,
    elements: []Element,
    allocator: *Allocator,
};

pub const ElementType = enum {
    Text,
    Heading,
    Paragraph,
    Table,
    Image,
    Code,
    Formula,
    List,
    ListItem,
};

pub const Element = struct {
    type: ElementType,
    bbox: BoundingBox,
    content: []const u8,
    properties: Properties,
};
```

**Exported Functions:**
```zig
export fn nExtract_Document_create() *DoclingDocument;
export fn nExtract_Document_destroy(doc: *DoclingDocument) void;
export fn nExtract_Element_create(type: ElementType) *Element;
```

---

### **DAY 3: Mojo FFI Layer**

**Goals:**
1. Generate FFI bindings with mojo-bindgen
2. Create high-level Mojo wrappers
3. Test data marshalling
4. Establish memory ownership patterns

**Deliverables:**

**File:** `mojo/ffi.mojo` (auto-generated, ~300 lines)
```mojo
from sys.ffi import external_call, DLHandle

# Auto-generated by mojo-bindgen
struct CDoclingDocument:
    var handle: UnsafePointer[NoneType]
    
    fn __init__(inout self):
        self.handle = external_call["nExtract_Document_create"]()
    
    fn __del__(owned self):
        external_call["nExtract_Document_destroy"](self.handle)
```

**File:** `mojo/core.mojo` (~400 lines)
```mojo
from .ffi import CDoclingDocument

struct DoclingDocument:
    """High-level Mojo wrapper for document processing."""
    var _handle: CDoclingDocument
    
    fn __init__(inout self):
        self._handle = CDoclingDocument()
    
    fn add_element(inout self, element: Element) -> Result[None, Error]:
        # Type-safe wrapper around C FFI
        pass
```

**Tests:**
- Round-trip data marshalling
- Memory leak detection
- Performance benchmarks

---

### **DAY 4: String & Text Utilities (Zig)**

**Goals:**
1. UTF-8 validation and manipulation
2. String builder implementation
3. Unicode normalization
4. Text processing utilities

**Deliverables:**

**File:** `zig/core/string.zig` (~800 lines)

**Features:**
- UTF-8 validation and iteration
- Dynamic string builder with SSO (Small String Optimization)
- Unicode normalization (NFD, NFC, NFKC, NFKD)
- Grapheme cluster iteration
- Case folding (upper, lower, title)
- String searching (Boyer-Moore, KMP algorithms)
- Whitespace trimming and normalization

**Tests:**
- UTF-8 edge cases (invalid sequences, overlong encodings)
- Unicode normalization correctness
- Performance benchmarks (vs stdlib)

---

### **DAY 5: Memory Management Infrastructure**

**Goals:**
1. Arena allocator for document processing
2. Object pooling for frequent allocations
3. Memory profiling integration
4. Leak detection tooling

**Deliverables:**

**File:** `zig/core/allocator.zig` (~600 lines)

**Features:**
- Arena allocator with configurable block size
- Object pool with type-safe reuse
- Memory usage tracking
- Leak detection (debug builds)
- Integration with Zig's standard allocator interface

**File:** `zig/core/profiler.zig` (~300 lines)
- Memory allocation profiler
- Performance metrics collection
- Integration with benchmarking tools

**Tests:**
- Allocation/deallocation stress tests
- Memory leak detection
- Performance benchmarks

---

## WEEK 2: Days 6-10 - Core Parsers

### **DAY 6: CSV Parser (Pure Zig)**

**Goals:**
1. RFC 4180 compliant CSV parser
2. Streaming parser for large files
3. Encoding detection
4. Delimiter auto-detection

**Deliverables:**

**File:** `zig/parsers/csv.zig` (~700 lines)

**Features:**
- RFC 4180 full compliance
- Quoted fields with escape sequences
- Multi-line field support
- Delimiter auto-detection (comma, tab, semicolon, pipe)
- Encoding detection (UTF-8, UTF-16, Latin1)
- Streaming parser (O(1) memory for large files)
- Configurable quote character
- Header row detection

**Export Functions:**
```zig
export fn nExtract_CSV_parse(data: [*]const u8, len: usize) *CsvDocument;
export fn nExtract_CSV_parse_stream(reader: *Reader) *CsvDocument;
```

**Tests:**
- Edge cases (empty fields, quotes, newlines)
- Large file handling (GB+)
- Encoding detection accuracy
- Malformed CSV recovery

---

### **DAY 7: Markdown Parser (Pure Zig)**

**Goals:**
1. Full CommonMark 0.30 spec
2. GitHub Flavored Markdown (GFM) extensions
3. AST generation
4. HTML block handling

**Deliverables:**

**File:** `zig/parsers/markdown.zig` (~1,200 lines)

**Features:**
- CommonMark 0.30 compliance
- GFM extensions:
  - Tables
  - Task lists
  - Strikethrough
  - Autolinks
- Footnotes
- Math blocks (LaTeX syntax)
- HTML blocks and inline HTML
- AST (Abstract Syntax Tree) generation
- Link reference definitions
- Code fences with syntax highlighting hints

**AST Nodes:**
- Document, Heading, Paragraph, BlockQuote
- List (ordered/unordered), ListItem
- CodeBlock, InlineCode
- Link, Image
- Emphasis, Strong, Strikethrough
- Table, TableRow, TableCell

**Tests:**
- CommonMark spec test suite
- GFM test suite
- Complex nested structures
- Edge cases (unclosed tags, etc.)

---

### **DAY 8-9: XML Parser (Pure Zig)**

**Goals:**
1. XML 1.0 spec compliance
2. SAX and DOM parsing modes
3. Namespace resolution
4. Entity expansion (with bomb protection)

**Deliverables:**

**File:** `zig/parsers/xml.zig` (~1,500 lines)

**Features:**
- Full XML 1.0 spec
- SAX (event-based) parsing
- DOM tree construction
- Namespace support (xmlns)
- DTD validation (optional)
- Entity expansion with size limits (prevent billion laughs)
- XPath subset for querying
- Streaming parser for large files
- Attribute parsing and validation
- CDATA section handling
- Processing instruction support

**SAX Events:**
- startElement, endElement
- characters, whitespace
- comment, processingInstruction

**DOM API:**
- Node, Element, Attribute, Text
- Tree traversal (children, siblings)
- XPath query

**Tests:**
- XML 1.0 conformance tests
- Namespace resolution
- Entity expansion limits
- Malformed XML recovery
- Large file handling

---

### **DAY 10: HTML Parser (Pure Zig)**

**Goals:**
1. HTML5 parsing algorithm (WHATWG spec)
2. DOM tree construction
3. Tag soup recovery
4. CSS selector engine

**Deliverables:**

**File:** `zig/parsers/html.zig` (~2,000 lines)

**Features:**
- HTML5 parsing algorithm
- DOM tree construction
- Tag soup recovery (auto-close tags, etc.)
- Character encoding detection
- CSS selector engine for traversal:
  - ID selector (#id)
  - Class selector (.class)
  - Tag selector (div)
  - Attribute selector ([attr=value])
  - Descendant combinator (div p)
  - Child combinator (div > p)
- Script/style tag handling
- DOCTYPE parsing
- Foreign content (SVG, MathML)

**DOM Structure:**
- Document, Element, Text, Comment
- Attributes
- Tree navigation (parent, children, siblings)

**Tests:**
- HTML5 parsing tests
- Tag soup recovery scenarios
- CSS selector correctness
- Encoding detection
- Large HTML file handling

---

## WEEK 3: Days 11-15 - Compression & Archives

### **DAY 11-12: DEFLATE Implementation (Pure Zig)**

**Goals:**
1. RFC 1951 DEFLATE algorithm
2. Huffman coding (static and dynamic)
3. LZ77 sliding window
4. Streaming decompressor

**Deliverables:**

**File:** `zig/parsers/deflate.zig` (~1,800 lines)

**Features:**
- RFC 1951 full implementation
- Huffman decoding (static and dynamic tables)
- LZ77 decompression with sliding window
- Streaming decompressor (O(window_size) memory)
- Compression support (for future use):
  - Dynamic Huffman table generation
  - LZ77 string matching
  - Configurable compression levels
- SIMD optimization where applicable
- Bit-level operations

**Algorithm:**
1. Read block header (BFINAL, BTYPE)
2. Decode Huffman tables (for dynamic blocks)
3. Decode literal/length symbols
4. Decode distance codes
5. Copy from sliding window

**Tests:**
- RFC 1951 test vectors
- Edge cases (no compression, fixed Huffman, dynamic Huffman)
- Large file decompression
- Corrupt stream handling
- Performance benchmarks

---

### **DAY 13: ZIP Archive Handler (Pure Zig)**

**Goals:**
1. ZIP format parsing (PKZIP 2.0+)
2. ZIP64 support
3. CRC32 verification
4. Streaming extraction

**Deliverables:**

**File:** `zig/parsers/zip.zig` (~1,200 lines)

**Features:**
- ZIP format (PKZIP 2.0+)
- ZIP64 support (large files > 4GB)
- Central directory parsing
- Local file header parsing
- File extraction with streaming
- CRC32 verification
- Compression methods:
  - Store (no compression)
  - Deflate (via deflate.zig)
- Directory traversal
- Metadata extraction (timestamps, permissions)

**ZIP Structure:**
- Local file headers
- Central directory
- End of central directory record
- ZIP64 end of central directory

**Tests:**
- Standard ZIP archives
- ZIP64 large files
- Password-protected ZIPs (basic RC4)
- Corrupt archive recovery
- Multi-file extraction

---

### **DAY 14: GZIP/ZLIB Support**

**Goals:**
1. RFC 1952 (GZIP) format
2. RFC 1950 (ZLIB) wrapper
3. Header parsing
4. Integration with DEFLATE

**Deliverables:**

**File:** `zig/parsers/gzip.zig` (~400 lines)

**GZIP Features:**
- Header parsing (ID1, ID2, CM, FLG, MTIME)
- Flags (FTEXT, FHCRC, FEXTRA, FNAME, FCOMMENT)
- CRC16 verification
- Footer parsing (CRC32, ISIZE)
- Integration with DEFLATE decompressor

**File:** `zig/parsers/zlib.zig` (~300 lines)

**ZLIB Features:**
- Header parsing (CMF, FLG)
- Compression method and window size
- FCHECK, FDICT, FLEVEL
- Adler32 checksum
- Integration with DEFLATE decompressor

**Tests:**
- GZIP test files
- ZLIB test files
- Checksum verification
- Malformed header handling

---

### **DAY 15: Compression Testing**

**Goals:**
1. Comprehensive test suite
2. Fuzzing infrastructure
3. Performance benchmarks
4. Memory profiling

**Deliverables:**

**File:** `zig/tests/compression_test.zig` (~600 lines)

**Test Coverage:**
- All compression formats (DEFLATE, GZIP, ZLIB, ZIP)
- Edge cases and corner cases
- Malformed input handling
- Large file processing
- Memory usage validation
- Performance benchmarks vs reference implementations

**Fuzzing:**
- Fuzz DEFLATE decoder
- Fuzz ZIP parser
- Fuzz GZIP/ZLIB parsers
- Crash detection and reporting

**Benchmarks:**
- Decompression speed
- Memory usage
- Comparison with zlib (C library)

---

## WEEK 4: Days 16-20 - Office Formats Foundation

### **DAY 16-17: OOXML Structure Parser**

**Goals:**
1. Office Open XML (ISO 29500) spec
2. Package relationships
3. Content types
4. Part naming conventions

**Deliverables:**

**File:** `zig/parsers/ooxml.zig` (~1,500 lines)

**Features:**
- OOXML package structure understanding
- `_rels/.rels` relationship parsing
- `[Content_Types].xml` parsing
- Part naming conventions (e.g., `/word/document.xml`)
- Relationship types (officeDocument, image, style, etc.)
- Digital signatures (optional, metadata only)
- Package validation

**OOXML Structure:**
```
docx/
├── [Content_Types].xml
├── _rels/
│   └── .rels
├── word/
│   ├── document.xml
│   ├── styles.xml
│   ├── numbering.xml
│   ├── _rels/
│   │   └── document.xml.rels
│   └── media/
│       └── image1.png
└── docProps/
    ├── core.xml
    └── app.xml
```

**Relationship Types:**
- officeDocument
- styles
- numbering
- image
- hyperlink
- footer
- header

**Tests:**
- Valid OOXML packages
- Missing relationships
- Invalid content types
- Relationship resolution

---

### **DAY 18: Shared String Table (XLSX)**

**Goals:**
1. SST XML parsing
2. String deduplication
3. Rich text handling
4. Phonetic properties

**Deliverables:**

**File:** `zig/parsers/xlsx_sst.zig` (~600 lines)

**Features:**
- SharedStringTable (`xl/sharedStrings.xml`)
- String deduplication (reference by index)
- Rich text (`<r>` elements with formatting)
- Phonetic properties (for Japanese/Chinese)
- Unicode support
- Whitespace preservation
- String count and unique count

**XML Structure:**
```xml
<sst count="3" uniqueCount="3">
  <si><t>Simple text</t></si>
  <si>
    <r><t>Rich </t></r>
    <r><rPr><b/></rPr><t>bold</t></r>
    <r><t> text</t></r>
  </si>
  <si><t>Another string</t></si>
</sst>
```

**Tests:**
- Simple strings
- Rich text with formatting
- Large SST (1M+ strings)
- Unicode characters

---

### **DAY 19: Style System (Office)**

**Goals:**
1. Font, color, border styles
2. Number formatting
3. Conditional formatting
4. Theme parsing

**Deliverables:**

**File:** `zig/parsers/office_styles.zig` (~800 lines)

**Features:**
- Font styles (family, size, bold, italic, underline, color)
- Cell borders (top, bottom, left, right, diagonal)
- Background colors and patterns
- Number formats (general, currency, date, percentage, custom)
- Conditional formatting rules
- Theme colors (accent1-6, hyperlink, followedHyperlink)
- Cell alignment (horizontal, vertical, wrap text)

**Style Types:**
- CellXf (cell format)
- CellStyleXf (cell style format)
- Dxf (differential format for conditional formatting)

**Tests:**
- Style inheritance
- Theme color resolution
- Number format application
- Complex conditional formatting

---

### **DAY 20: Office Format Testing**

**Goals:**
1. Complex OOXML documents
2. Nested structures
3. Large file handling
4. Edge case testing

**Deliverables:**

**File:** `zig/tests/ooxml_test.zig` (~500 lines)

**Test Coverage:**
- Complex DOCX documents
- Large XLSX spreadsheets
- Multi-slide PPTX presentations
- Relationship resolution
- Style application
- Image extraction
- Malformed package handling

**Test Documents:**
- Real-world Office documents
- Edge cases (empty documents, maximum nesting)
- Corrupt packages

---

## WEEK 5: Days 21-25 - Image Codec Foundations

### **DAY 21-22: PNG Decoder (Pure Zig)**

**Goals:**
1. Full PNG spec (ISO/IEC 15948)
2. All chunk types
3. Interlacing support
4. Color type support

**Deliverables:**

**File:** `zig/parsers/png.zig` (~2,000 lines)

**Features:**
- Full PNG specification
- Critical chunks:
  - IHDR (image header)
  - PLTE (palette)
  - IDAT (image data)
  - IEND (image trailer)
- Ancillary chunks:
  - tEXt, zTXt, iTXt (text)
  - tIME (timestamp)
  - pHYs (physical pixel dimensions)
  - bKGD (background color)
  - tRNS (transparency)
- Interlacing (Adam7)
- Color types:
  - Grayscale (0)
  - RGB (2)
  - Palette (3)
  - Grayscale + Alpha (4)
  - RGB + Alpha (6)
- Bit depths: 1, 2, 4, 8, 16
- Filtering (None, Sub, Up, Average, Paeth)
- CRC validation
- DEFLATE decompression (via deflate.zig)

**Decoding Pipeline:**
1. Parse signature (89 50 4E 47 0D 0A 1A 0A)
2. Read chunks (length, type, data, CRC)
3. Validate IHDR
4. Decompress IDAT chunks
5. Apply filters
6. Deinterlace (if Adam7)
7. Convert to RGBA

**Tests:**
- PNG test suite (PngSuite)
- All color types and bit depths
- Interlaced images
- Ancillary chunks
- Corrupt PNG handling

---

### **DAY 23-24: JPEG Decoder (Pure Zig)**

**Goals:**
1. JPEG/JFIF spec (ISO/IEC 10918-1)
2. Huffman decoding
3. DCT (Discrete Cosine Transform)
4. Color space conversion

**Deliverables:**

**File:** `zig/parsers/jpeg.zig` (~2,500 lines)

**Features:**
- JPEG/JFIF format
- Marker parsing (SOI, SOF, DHT, DQT, SOS, EOI)
- Huffman decoding (DC and AC coefficients)
- DCT (Discrete Cosine Transform) - IDCT
- YCbCr to RGB conversion
- Progressive JPEG (multiple scans)
- EXIF metadata extraction
- Thumbnail extraction (JFIF, EXIF)
- Chroma subsampling (4:4:4, 4:2:2, 4:2:0)

**Marker Types:**
- SOI (Start of Image)
- SOF0-SOF15 (Start of Frame)
- DHT (Define Huffman Table)
- DQT (Define Quantization Table)
- DRI (Define Restart Interval)
- SOS (Start of Scan)
- EOI (End of Image)
- APP0-APP15 (Application segments)
- COM (Comment)

**Decoding Pipeline:**
1. Parse markers
2. Build Huffman tables
3. Build quantization tables
4. Decode MCU (Minimum Coded Unit)
5. Inverse DCT
6. Dequantization
7. YCbCr to RGB conversion

**Tests:**
- JPEG test suite
- Progressive JPEG
- Various subsampling modes
- EXIF metadata
- Corrupt JPEG handling

---

### **DAY 25: Image Testing**

**Goals:**
1. Comprehensive image tests
2. Color space conversions
3. Performance benchmarks
4. Error handling

**Deliverables:**

**File:** `zig/tests/image_test.zig` (~600 lines)

**Test Coverage:**
- PNG decoder (all color types, bit depths)
- JPEG decoder (baseline, progressive)
- Color space conversions
- Large image handling
- Corrupt image recovery
- Memory usage validation

**Benchmarks:**
- Decoding speed (PNG, JPEG)
- Memory usage
- Comparison with reference implementations (libpng, libjpeg)

---

# PHASE 2: Advanced Image Processing & OCR (Days 26-45)

## WEEK 6: Days 26-30 - Image Processing Primitives

### **DAY 26: Color Space Conversions**

**Goals:**
1. RGB ↔ Grayscale
2. RGB ↔ HSV/HSL
3. RGB ↔ YCbCr
4. Gamma correction

**Deliverables:**

**File:** `zig/ocr/colorspace.zig` (~800 lines)

**Features:**
- RGB to Grayscale (weighted average)
- RGB ↔ HSV (Hue, Saturation, Value)
- RGB ↔ HSL (Hue, Saturation, Lightness)
- RGB ↔ YCbCr (JPEG color space)
- CMYK support (basic)
- Gamma correction (encode/decode)
- Color temperature adjustment
- SIMD optimization for batch conversions

**Conversion Formulas:**
- Grayscale: `Y = 0.299*R + 0.587*G + 0.114*B`
- HSV/HSL: Geometric transformations
- YCbCr: ITU-R BT.601 standard

**Tests:**
- Conversion accuracy
- Round-trip conversions
- Edge cases (black, white, saturated colors)
- Performance benchmarks

---

### **DAY 27: Image Filtering**

**Goals:**
1. Gaussian blur
2. Edge detection
3. Morphological operations
4. Noise reduction

**Deliverables:**

**File:** `zig/ocr/filters.zig` (~1,200 lines)

**Features:**
- Gaussian blur (configurable sigma)
- Median filter (noise reduction)
- Sharpen filter (unsharp mask)
- Edge detection:
  - Sobel operator (horizontal, vertical)
  - Canny edge detector (multi-stage)
  - Laplacian of Gaussian
- Morphological operations:
  - Erosion
  - Dilation
  - Opening (erosion + dilation)
  - Closing (dilation + erosion)
- Bilateral filter (edge-preserving smoothing)
- SIMD optimization for convolution

**Convolution Kernels:**
- 3x3, 5x5, 7x7 kernels
- Separable filters (Gaussian)
- Custom kernel support

**Tests:**
- Filter correctness
- Edge detection accuracy
- Performance benchmarks
- Memory usage

---

### **DAY 28: Image Transformations**

**Goals:**
1. Rotation (arbitrary angles)
2. Scaling (interpolation)
3. Affine transformations
4. Perspective correction

**Deliverables:**

**File:** `zig/ocr/transform.zig` (~1,000 lines)

**Features:**
- Rotation (any angle, with interpolation)
- Scaling:
  - Nearest neighbor
  - Bilinear interpolation
  - Bicubic interpolation
- Affine transformations (rotation, scale, shear, translation)
- Perspective transformation (4-point homography)
- Flip (horizontal, vertical)
- Crop and resize

**Interpolation Methods:**
- Nearest neighbor (fast, low quality)
- Bilinear (medium speed, good quality)
- Bicubic (slow, high quality)

**Tests:**
- Rotation accuracy
- Scaling quality
- Perspective correction
- Performance benchmarks

---

### **DAY 29: Thresholding & Binarization**

**Goals:**
1. Global thresholding
2. Otsu's method
3. Adaptive thresholding
4. Sauvola binarization

**Deliverables:**

**File:** `zig/ocr/threshold.zig` (~800 lines)

**Features:**
- Global thresholding (single threshold value)
- Otsu's method (automatic threshold selection)
- Adaptive thresholding:
  - Mean adaptive
  - Gaussian adaptive
- Sauvola binarization (for document images)
- Bradley adaptive thresholding
- Niblack method
- Hysteresis thresholding (Canny edge detector)

**Otsu's Method:**
- Histogram-based automatic threshold
- Maximizes inter-class variance
- Optimal for bimodal histograms

**Adaptive Thresholding:**
- Local threshold per pixel
- Window-based calculation
- Handles varying illumination

**Tests:**
- Document image binarization
- Various lighting conditions
- Comparison with reference implementations
- Performance benchmarks

---

### **DAY 30: Image Processing Tests**

**Goals:**
1. Quality assessment
2. Performance benchmarks
3. SIMD optimization validation
4. Integration tests

**Deliverables:**

**File:** `zig/tests/image_processing_test.zig` (~600 lines)

**Test Coverage:**
- All filters and transformations
- Color space conversions
- Thresholding methods
- Large image processing
- Memory usage validation

**Quality Metrics:**
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Visual inspection (saved test outputs)

**Benchmarks:**
- Processing speed per operation
- Memory usage
- SIMD vs scalar performance

---

## WEEK 7: Days 31-35 - OCR Engine (Pure Zig/Mojo)

### **DAY 31: Text Line Detection**

**Goals:**
1. Connected component analysis
2. Line segmentation
3. Skew detection and correction
4. Baseline detection

**Deliverables:**

**File:** `zig/ocr/line_detection.zig` (~1,000 lines)

**Features:**
- Connected component analysis (CCA)
  - 4-connectivity and 8-connectivity
  - Component labeling
  - Bounding box extraction
- Line segmentation:
  - Horizontal projection profile
  - Vertical projection profile
  - Line detection via valley detection
- Skew detection:
  - Hough transform
  - Projection profile method
- Skew correction (rotation)
- Baseline detection (bottom of characters)

**Algorithm:**
1. Binarize image
2. Connected component analysis
3. Filter noise (small components)
4. Detect skew angle
5. Rotate to correct skew
6. Segment into lines (projection profile)
7. Detect baselines

**Tests:**
- Various document layouts
- Skewed documents (0-15 degrees)
- Multi-column documents
- Noisy images

---

### **DAY 32: Character Segmentation**

**Goals:**
1. Character boundary detection
2. Touching character separation
3. Projection profiles
4. Noise removal

**Deliverables:**

**File:** `zig/ocr/char_segmentation.zig` (~900 lines)

**Features:**
- Character segmentation:
  - Vertical projection profile
  - Connected component analysis per line
  - Bounding box extraction per character
- Touching character separation:
  - Vertical cuts at narrowest points
  - Template matching for split validation
- Noise removal:
  - Filter small components (dots, specks)
  - Remove large components (page borders)
- Character normalization:
  - Resize to standard height (e.g., 32 pixels)
  - Center in bounding box
  - Aspect ratio preservation

**Algorithm:**
1. For each text line:
   a. Vertical projection profile
   b. Detect valleys (character boundaries)
   c. Extract character images
   d. Normalize character size
2. Handle touching characters (split if needed)

**Tests:**
- Various fonts
- Touching characters
- Broken characters
- Noise and artifacts

---

### **DAY 33-34: Feature Extraction & Recognition**

**Goals:**
1. Feature extraction (HOG, templates)
2. Character classification
3. Neural network inference (simple)
4. Confidence scoring

**Deliverables:**

**File:** `zig/ocr/recognition.zig` (~2,000 lines)

**Features:**
- Feature extraction:
  - HOG (Histogram of Oriented Gradients)
  - Pixel intensity features
  - Zoning features (divide into zones)
- Template matching (for simple recognition)
- Simple neural network classifier:
  - Input: Feature vector
  - Hidden layers: 2-3 layers
  - Output: Character class (A-Z, a-z, 0-9, symbols)
- Character set:
  - Latin uppercase (A-Z)
  - Latin lowercase (a-z)
  - Digits (0-9)
  - Common symbols (. , ! ? - etc.)
- Confidence scoring (softmax output)

**Neural Network:**
- Architecture: Input → Hidden1 → Hidden2 → Output
- Activation: ReLU (hidden), Softmax (output)
- Forward pass only (inference)
- Pre-trained weights (from synthetic data or public dataset)

**Tests:**
- Character recognition accuracy
- Confidence calibration
- Various fonts and sizes
- Degraded images

---

### **DAY 35: OCR Integration & Testing**

**Goals:**
1. Word formation from characters
2. Dictionary-based correction (optional)
3. Confidence thresholding
4. End-to-end OCR pipeline

**Deliverables:**

**File:** `zig/ocr/ocr.zig` (~800 lines)

**Features:**
- Full OCR pipeline:
  1. Preprocess image (binarize, denoise)
  2. Detect and correct skew
  3. Segment into lines
  4. Segment into characters
  5. Recognize characters
  6. Form words (space detection)
  7. Output text with bounding boxes
- Word formation:
  - Detect spaces (gap > threshold)
  - Concatenate characters
- Dictionary-based correction (optional):
  - Check against English word list
  - Suggest corrections for low-confidence words
- Confidence thresholding:
  - Reject low-confidence characters
  - Mark uncertain regions

**Export Function:**
```zig
export fn nExtract_OCR_process(
    image_data: [*]const u8,
    width: u32,
    height: u32
) *OcrResult;
```

**Tests:**
- Scanned documents
- Various fonts and sizes
- Noisy images
- Accuracy metrics (character/word accuracy)
- Performance benchmarks

---

## WEEK 8: Days 36-40 - Advanced OCR Features

### **DAY 36: Language Support**

**Goals:**
1. UTF-8 character recognition
2. Multi-language detection
3. Character set expansion
4. Unicode support

**Deliverables:**

**File:** `zig/ocr/multilang.zig` (~1,000 lines)

**Features:**
- Extended character sets:
  - Latin (A-Z, a-z, À-ÿ)
  - Cyrillic (А-Я, а-я)
  - Greek (Α-Ω, α-ω)
  - Common diacritics
- Language detection (heuristic):
  - Character frequency analysis
  - Detect script (Latin, Cyrillic, Greek, etc.)
- Unicode normalization
- Right-to-left text detection (Arabic, Hebrew)

**Character Classifier:**
- Expanded neural network output (1000+ classes)
- Language-specific models (optional)

**Tests:**
- Multi-language documents
- Mixed scripts
- Diacritics and accents
- Accuracy per language

---

### **DAY 37: Layout Analysis for OCR**

**Goals:**
1. Text block detection
2. Column separation
3. Reading order determination
4. Table detection (basic)

**Deliverables:**

**File:** `zig/ocr/layout.zig` (~1,200 lines)

**Features:**
- Text block detection:
  - Cluster lines into blocks (proximity-based)
  - Detect block boundaries
- Column separation:
  - Detect vertical gaps (columns)
  - Left-to-right or right-to-left ordering
- Reading order determination:
  - Top-to-bottom, left-to-right (English)
  - Top-to-bottom, right-to-left (Arabic)
  - Column-aware ordering
- Table detection (rule-based):
  - Detect grid lines
  - Identify table regions
  - Extract table structure (rows, columns)

**Algorithm:**
1. Segment page into lines
2. Cluster lines into blocks (vertical proximity)
3. Detect columns (vertical gaps)
4. Order blocks (top-to-bottom, then left-to-right per column)
5. Detect tables (grid lines, cell alignment)

**Tests:**
- Single-column documents
- Multi-column documents
- Documents with tables
- Complex layouts

---

### **DAY 38: OCR Quality Enhancement**

**Goals:**
1. Denoising
2. Contrast enhancement
3. Super-resolution (simple)
4. Pre-processing pipeline

**Deliverables:**

**File:** `zig/ocr/enhance.zig` (~800 lines)

**Features:**
- Denoising:
  - Median filter
  - Bilateral filter
  - Morphological opening/closing
- Contrast enhancement:
  - Histogram equalization
  - CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Super-resolution (simple):
  - Bilinear/bicubic upscaling
  - Edge-aware interpolation
- Binarization (multiple methods):
  - Otsu
  - Adaptive (Gaussian, Mean)
  - Sauvola

**Pre-processing Pipeline:**
1. Convert to grayscale
2. Denoise
3. Enhance contrast
4. Super-resolve (if low resolution)
5. Binarize

**Tests:**
- Low-quality scans
- Noisy images
- Low-resolution images
- Accuracy improvement metrics

---

### **DAY 39-40: OCR Testing & Optimization**

**Goals:**
1. Comprehensive test suite
2. Accuracy benchmarks
3. Speed optimization
4. Memory optimization

**Deliverables:**

**File:** `zig/tests/ocr_test.zig` (~800 lines)

**Test Coverage:**
- Character recognition accuracy
- Word recognition accuracy
- End-to-end pipeline tests
- Various document types:
  - Printed documents
  - Scanned documents
  - Photocopies
  - Screenshots
- Various fonts and sizes
- Noisy and degraded images

**Accuracy Metrics:**
- Character accuracy rate (CAR)
- Word accuracy rate (WAR)
- Confusion matrix (common mistakes)

**Optimization:**
- SIMD for image processing
- Parallel processing (multiple pages)
- Memory usage reduction
- Pre-compute feature extractors

**Benchmarks:**
- Processing speed (pages per second)
- Memory usage per page
- Comparison with Tesseract (if available for reference)

---

## WEEK 9: Days 41-45 - ML Model Inference Engine

### **DAY 41-42: Tensor Operations (Pure Zig/Mojo)**

**Goals:**
1. Multi-dimensional arrays
2. Matrix multiplication (GEMM)
3. Convolution operations
4. Activation functions

**Deliverables:**

**File:** `zig/ml/tensor.zig` (~1,500 lines)

**Features:**
- Tensor type (multi-dimensional array)
  - 1D (vector), 2D (matrix), 3D, 4D, ND
  - Shape, strides, data pointer
- Basic operations:
  - Element-wise (add, sub, mul, div)
  - Reduction (sum, mean, max, min)
  - Reshape, transpose, slice
- Matrix multiplication:
  - Naive implementation
  - Blocked/tiled GEMM
  - SIMD optimization
- Convolution operations:
  - 2D convolution (im2col + GEMM)
  - Padding (same, valid)
  - Stride support
- Activation functions:
  - ReLU (max(0, x))
  - Sigmoid (1 / (1 + e^(-x)))
  - Tanh
  - Softmax (e^x / sum(e^x))
- Pooling:
  - Max pooling
  - Average pooling

**SIMD Optimization:**
- Vectorized operations where possible
- Use Mojo SDK's SIMD support for Mojo layer

**Tests:**
- Tensor operations correctness
- Matrix multiplication (against reference)
- Convolution correctness
- Performance benchmarks

---

### **DAY 43: Neural Network Inference**

**Goals:**
1. Layer types (Dense, Conv2D, BatchNorm)
2. Forward pass implementation
3. Model weight loading
4. Quantization support

**Deliverables:**

**File:** `zig/ml/nn.zig` (~1,800 lines)

**Features:**
- Layer types:
  - Dense (fully connected)
  - Conv2D (2D convolution)
  - BatchNorm (batch normalization)
  - MaxPool2D, AvgPool2D
  - Flatten
  - Dropout (inference mode, disabled)
- Forward pass:
  - Sequential model execution
  - Layer-by-layer inference
  - Activation application
- Model structure:
  - Graph representation (layers, connections)
  - Input/output tensor shapes
- Quantization support:
  - INT8 quantization (weights and activations)
  - Dequantization for accumulation
  - Performance improvement

**Model Definition:**
```zig
const Model = struct {
    layers: []Layer,
    
    fn forward(input: Tensor) Tensor {
        var x = input;
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x;
    }
};
```

**Tests:**
- Layer forward pass correctness
- Full model inference
- Quantized model inference
- Performance benchmarks

---

### **DAY 44: Model Conversion Tool**

**Goals:**
1. Convert ONNX → custom format
2. Weight extraction
3. Graph optimization
4. Quantization

**Deliverables:**

**File:** `tools/model_converter.zig` (~1,200 lines)

**Features:**
- ONNX file parsing (protobuf)
- Graph extraction (nodes, edges, weights)
- Operator mapping (ONNX → custom)
- Weight extraction and conversion
- Graph optimization:
  - Constant folding
  - Operator fusion (Conv + BatchNorm + ReLU)
  - Dead code elimination
- Quantization:
  - FP32 → INT8 conversion
  - Calibration (min/max per layer)
- Custom format export (binary format for fast loading)

**ONNX Operators Supported:**
- Conv, Gemm, MatMul
- Relu, Sigmoid, Tanh, Softmax
- BatchNormalization
- MaxPool, AveragePool
- Flatten, Reshape, Transpose
- Add, Mul, Concat

**Usage:**
```bash
./model_converter --input model.onnx --output model.custom --quantize int8
```

**Tests:**
- Convert reference models
- Validate output correctness
- Quantization accuracy

---

### **DAY 45: ML Testing**

**Goals:**
1. Validation against reference implementations
2. Performance benchmarks
3. Memory usage optimization
4. Integration with OCR

**Deliverables:**

**File:** `zig/tests/ml_test.zig` (~600 lines)

**Test Coverage:**
- Tensor operations
- Layer forward pass
- Full model inference
- Quantized model inference
- Model conversion

**Validation:**
- Compare outputs with reference (ONNX Runtime, PyTorch)
- Acceptable error tolerance (< 1e-5 for FP32, < 1% for INT8)

**Benchmarks:**
- Inference speed (images per second)
- Memory usage per inference
- Quantized vs FP32 performance

**Integration:**
- Use ML model for OCR character recognition
- Use ML model for layout analysis (next phase)

---

# PHASE 3: PDF Processing (Days 46-70)

## WEEK 10: Days 46-50 - PDF Parser Core

### **DAY 46-47: PDF Object Model**

**Goals:**
1. PDF cross-reference table (xref)
2. Object stream parsing
3. Indirect object resolution
4. Document catalog

**Deliverables:**

**File:** `zig/pdf/objects.zig` (~2,000 lines)

**Features:**
- PDF version detection (1.0-1.7, 2.0)
- Cross-reference table (xref):
  - Traditional xref table
  - Cross-reference streams (PDF 1.5+)
  - Incremental updates (multiple xref sections)
- Object types:
  - Null, Boolean, Integer, Real
  - String (literal, hexadecimal)
  - Name
  - Array
  - Dictionary
  - Stream
  - Indirect object reference
- Object stream parsing:
  - Read object by ID
  - Resolve indirect references
  - Circular reference detection
- Document catalog:
  - Root object
  - Pages tree
  - Metadata
  - Outlines (bookmarks)

**PDF Structure:**
```
%PDF-1.7
1 0 obj
<< /Type /Catalog /Pages 2 0 R >>
endobj
...
xref
0 5
0000000000 65535 f
0000000015 00000 n
...
trailer
<< /Size 5 /Root 1 0 R >>
startxref
123
%%EOF
```

**Tests:**
- Various PDF versions
- Linearized PDFs
- Incremental updates
- Corrupt xref recovery

---

### **DAY 48: PDF Streams**

**Goals:**
1. Stream dictionary parsing
2. Filter support (multiple filters)
3. Stream decompression
4. Predictor functions

**Deliverables:**

**File:** `zig/pdf/streams.zig` (~1,500 lines)

**Features:**
- Stream dictionary parsing
- Filters:
  - FlateDecode (DEFLATE)
  - DCTDecode (JPEG)
  - LZWDecode
  - ASCIIHexDecode
  - ASCII85Decode
  - RunLengthDecode
  - CCITTFaxDecode (basic)
- Filter parameters (decode params)
- Predictor functions:
  - PNG predictor (for FlateDecode)
  - TIFF predictor
- Chained filters (e.g., ASCII85 → FlateDecode)
- Stream decompression with buffering

**LZW Decompression:**
- Implement LZW algorithm (TIFF/PDF variant)
- Dynamic dictionary
- Clear code, EOD code

**ASCII85/Hex Decoding:**
- Base85 decoding
- Hexadecimal decoding

**Tests:**
- All filter types
- Chained filters
- Predictor functions
- Large stream decompression

---

### **DAY 49: PDF Content Streams**

**Goals:**
1. Content stream tokenizer
2. Operator parsing
3. Graphics state stack
4. Path construction

**Deliverables:**

**File:** `zig/pdf/content.zig` (~2,000 lines)

**Features:**
- Content stream tokenizer:
  - Parse operators and operands
  - Handle whitespace and comments
- Text operators:
  - BT, ET (Begin/End text)
  - Tf (Set font)
  - Tj, TJ, ' , " (Show text)
  - Td, TD, Tm, T* (Text positioning)
  - Tc, Tw, Tz, TL, Ts (Text state)
- Graphics state operators:
  - q, Q (Save/Restore graphics state)
  - cm (Concatenate matrix)
  - w, J, j, M, d (Line attributes)
  - CS, cs, SC, sc, G, g, RG, rg, K, k (Color)
- Path construction operators:
  - m, l, c, v, y, h (Path construction)
  - re (Rectangle)
  - S, s, f, F, f*, B, B*, b, b*, n (Path painting)
- XObject operators:
  - Do (Invoke XObject)

**Graphics State Stack:**
- Current transformation matrix (CTM)
- Current font and size
- Current color
- Line width, cap, join
- Clipping path

**Tests:**
- Parse various content streams
- Graphics state tracking
- Path construction
- Text extraction (basic)

---

### **DAY 50: PDF Testing**

**Goals:**
1. Parse various PDF versions
2. Linearized PDFs
3. Encrypted PDFs (basic)
4. Malformed PDF recovery

**Deliverables:**

**File:** `zig/tests/pdf_parser_test.zig` (~600 lines)

**Test Coverage:**
- PDF versions 1.4-2.0
- Linearized PDFs (fast web view)
- Encrypted PDFs (RC4, AES)
- Object streams
- Cross-reference streams
- Incremental updates
- Corrupt PDF recovery

**Test PDFs:**
- Real-world PDF samples
- PDF specification examples
- Malformed PDFs (missing xref, etc.)

---

## WEEK 11: Days 51-55 - PDF Text Extraction

### **DAY 51: Font Handling**

**Goals:**
1. Font dictionary parsing
2. Font types (Type1, TrueType, CID)
3. Encoding resolution
4. CMap parsing

**Deliverables:**

**File:** `zig/pdf/fonts.zig` (~2,000 lines)

**Features:**
- Font types:
  - Type1 (PostScript fonts)
  - TrueType
  - Type0 (Composite fonts for CJK)
  - Type3 (User-defined fonts)
- Font dictionary:
  - BaseFont, Encoding, Widths
  - FontDescriptor (for metrics)
  - ToUnicode CMap
- Encoding:
  - WinAnsiEncoding
  - MacRomanEncoding
  - MacExpertEncoding
  - StandardEncoding
  - Custom encoding (Differences array)
- CMap parsing:
  - Identity-H, Identity-V (for CID fonts)
  - Custom CMaps
  - Character code → CID → Unicode
- Font metrics:
  - Character widths
  - Ascent, descent, cap height
- Font subset detection

**CMap Structure:**
```
/CIDInit /ProcSet findresource begin
12 dict begin
begincmap
/CIDSystemInfo << /Registry (Adobe) /Ordering (Japan1) /Supplement 2 >> def
/CMapName /90ms-RKSJ-H def
/CMapType 1 def
1 begincodespacerange
<0000> <FFFF>
endcodespacerange
100 beginbfchar
<0020> <0020>
...
endbfchar
endcmap
```

**Tests:**
- Various font types
- Encoding resolution
- CMap parsing (CJK fonts)
- Font subset handling

---

### **DAY 52: Text Positioning**

**Goals:**
1. Text matrix calculations
2. Character spacing
3. Word spacing
4. Glyph positioning

**Deliverables:**

**File:** `zig/pdf/text_positioning.zig` (~1,200 lines)

**Features:**
- Text matrix (Tm):
  - Current text position
  - Text line matrix (Tlm)
  - Updated by Td, TD, Tm, T*
- Text state parameters:
  - Tc (character spacing)
  - Tw (word spacing)
  - Th (horizontal scaling)
  - Tl (leading)
  - Ts (text rise)
- Glyph positioning:
  - Calculate glyph position from Tm
  - Apply character and word spacing
  - Apply horizontal scaling
  - Handle text rise (superscript/subscript)
- Text rendering modes:
  - Fill, Stroke, Fill+Stroke
  - Invisible (for searching)
  - Clipping

**Text Positioning Algorithm:**
1. Start with text matrix Tm
2. For each character:
   a. Calculate position: (x, y) = Tm * (0, 0)
   b. Advance: dx = (width + Tc + Tw_if_space) * Th
   c. Update Tm: Tm = Tm * translate(dx, 0)

**Tests:**
- Text positioning accuracy
- Character and word spacing
- Horizontal scaling
- Text rise

---

### **DAY 53: Text Extraction Algorithm**

**Goals:**
1. Extract text with position
2. Detect word boundaries
3. Line breaking
4. Paragraph detection

**Deliverables:**

**File:** `zig/pdf/text_extraction.zig` (~1,500 lines)

**Features:**
- Text extraction:
  - Extract characters with bounding boxes
  - Track text position and size
  - Preserve formatting (bold, italic)
- Word boundary detection:
  - Detect spaces (gap > threshold)
  - Word boundary heuristics
- Line breaking:
  - Detect line ends (vertical gap)
  - Handle multi-column layouts
- Paragraph detection:
  - Detect paragraph boundaries (larger vertical gap)
  - Indentation detection
- Reading order:
  - Left-to-right, top-to-bottom
  - Multi-column handling
  - Table detection (basic)

**Text Extraction Output:**
```zig
struct ExtractedText {
    text: []const u8,
    bbox: BoundingBox,
    font_size: f32,
    font_name: []const u8,
    is_bold: bool,
    is_italic: bool,
};
```

**Tests:**
- Single-column documents
- Multi-column documents
- Various fonts and sizes
- Tables and lists

---

### **DAY 54: Unicode Mapping**

**Goals:**
1. ToUnicode CMap
2. Character code to Unicode conversion
3. Glyph name to Unicode
4. UTF-8 output

**Deliverables:**

**File:** `zig/pdf/unicode_mapping.zig` (~1,000 lines)

**Features:**
- ToUnicode CMap parsing
  - bfchar, bfrange mappings
  - Character code → Unicode
- Glyph name to Unicode:
  - Adobe Glyph List (AGL)
  - Common glyph names (e.g., "A", "space", "eacute")
- Fallback strategies:
  - Use encoding (if available)
  - Use font's built-in encoding
  - Heuristic (if StandardEncoding)
- UTF-8 output:
  - Convert Unicode codepoints to UTF-8
  - Preserve non-breaking spaces

**Adobe Glyph List (AGL):**
- Map glyph names to Unicode
- Example: "A" → U+0041, "eacute" → U+00E9

**Tests:**
- ToUnicode CMap parsing
- Glyph name resolution
- Non-Latin scripts (CJK, Arabic, etc.)
- UTF-8 encoding correctness

---

### **DAY 55: Text Extraction Testing**

**Goals:**
1. Complex PDFs
2. Multi-column layouts
3. Non-Latin scripts
4. Accuracy validation

**Deliverables:**

**File:** `zig/tests/pdf_text_test.zig` (~600 lines)

**Test Coverage:**
- Single-column documents
- Multi-column documents
- Rotated text
- Vertical text (CJK)
- Mixed fonts and sizes
- Tables and lists
- Right-to-left text (Arabic, Hebrew)
- Complex layouts

**Accuracy Metrics:**
- Character accuracy
- Word accuracy
- Preserve reading order
- Preserve formatting

**Test PDFs:**
- Academic papers (multi-column)
- Books (single-column, paragraphs)
- Forms (tables, mixed content)
- Multilingual documents

---

## WEEK 12: Days 56-60 - PDF Images & Graphics

### **DAY 56: Image XObject Extraction**

**Goals:**
1. Image dictionary parsing
2. ColorSpace handling
3. Image data extraction
4. Inline images

**Deliverables:**

**File:** `zig/pdf/images.zig` (~1,500 lines)

**Features:**
- Image XObject:
  - Dictionary parsing (/Type /XObject /Subtype /Image)
  - Width, Height, BitsPerComponent
  - ColorSpace (DeviceRGB, DeviceGray, DeviceCMYK, ICC)
  - Filter (DCTDecode, FlateDecode, etc.)
  - Decode array (color adjustment)
  - ImageMask (for masks)
- ColorSpace handling:
  - DeviceGray, DeviceRGB, DeviceCMYK
  - Indexed (palette-based)
  - Separation (spot colors)
  - ICCBased (ICC profile)
- Image data extraction:
  - Decompress stream (via filters)
  - Apply decode array
  - Convert to RGB/RGBA
- Inline images (BI...ID...EI):
  - Parse inline image dictionary
  - Extract image data
  - Convert to standard image format

**Image Extraction Pipeline:**
1. Parse image dictionary
2. Decompress stream
3. Apply color space conversion
4. Apply decode array
5. Output as PNG/JPEG/raw pixels

**Tests:**
- Various color spaces
- Compressed images (JPEG, Flate)
- Inline images
- Image masks

---

### **DAY 57: Image Decoding**

**Goals:**
1. DCTDecode (JPEG) integration
2. FlateDecode (PNG-like)
3. JBIG2Decode (black/white)
4. Color space conversion

**Deliverables:**

**File:** `zig/pdf/image_decode.zig` (~1,200 lines)

**Features:**
- DCTDecode:
  - Use JPEG decoder (from Day 23-24)
  - Handle JPEG images embedded in PDF
- FlateDecode:
  - Decompress with DEFLATE
  - Apply PNG predictor (if specified)
  - Convert to RGB/RGBA
- JBIG2Decode:
  - Basic JBIG2 support (monochrome images)
  - Huffman decoding
  - Context-based arithmetic coding (if needed)
- JPXDecode (JPEG2000):
  - Skip for now or basic metadata only
- Color space conversion:
  - CMYK → RGB
  - Indexed → RGB (via palette)
  - ICCBased → RGB (via ICC profile, simplified)

**CMYK to RGB Conversion:**
```
R = (1 - C) * (1 - K)
G = (1 - M) * (1 - K)
B = (1 - Y) * (1 - K)
```

**Tests:**
- JPEG images
- Flate-compressed images
- JBIG2 images
- Color space conversions

---

### **DAY 58: Vector Graphics**

**Goals:**
1. Path extraction
2. Rectangle detection
3. Line extraction
4. Shape recognition

**Deliverables:**

**File:** `zig/pdf/graphics.zig` (~1,000 lines)

**Features:**
- Path extraction:
  - Parse path construction operators (m, l, c, h)
  - Convert to points/curves
  - Detect closed paths
- Rectangle detection:
  - Detect rectangles from paths (re operator or 4-line closed path)
  - Used for borders, table cells, etc.
- Line extraction:
  - Detect straight lines
  - Line width, color, dash pattern
- Shape recognition:
  - Circles, ellipses (from Bezier curves)
  - Polygons
- Clipping paths:
  - Extract clipping regions
  - Apply to subsequent drawing operations

**Graphics Primitives:**
```zig
struct Path {
    subpaths: []SubPath,
};

struct SubPath {
    closed: bool,
    segments: []Segment,
};

struct Segment {
    type: SegmentType,  // Line, BezierCurve
    points: []Point,
};
```

**Tests:**
- Path extraction
- Rectangle detection
- Line detection
- Complex shapes

---

### **DAY 59: Form XObjects**

**Goals:**
1. Nested content streams
2. Transformation matrices
3. Transparency groups
4. Recursive rendering

**Deliverables:**

**File:** `zig/pdf/form_xobjects.zig` (~800 lines)

**Features:**
- Form XObject:
  - Dictionary parsing (/Type /XObject /Subtype /Form)
  - BBox (bounding box)
  - Matrix (transformation)
  - Resources (fonts, images, etc.)
  - Content stream
- Nested content streams:
  - Parse and render form content
  - Apply transformation matrix
  - Recursive rendering (forms can contain forms)
- Transparency groups:
  - Isolated and knockout groups
  - Blend modes (multiply, screen, overlay, etc.)
- Do operator:
  - Invoke XObject (image or form)
  - Apply CTM (current transformation matrix)

**Rendering Pipeline:**
1. Save graphics state
2. Apply form matrix
3. Process form content stream
4. Restore graphics state

**Tests:**
- Forms with images
- Nested forms
- Transformation matrices
- Transparency

---

### **DAY 60: PDF Graphics Testing**

**Goals:**
1. Image extraction accuracy
2. Vector graphics extraction
3. Complex layouts
4. Integration tests

**Deliverables:**

**File:** `zig/tests/pdf_graphics_test.zig` (~500 lines)

**Test Coverage:**
- Image extraction (all formats)
- Vector graphics (paths, shapes)
- Form XObjects
- Clipping paths
- Transparency
- Complex documents (mixed text, images, graphics)

**Test PDFs:**
- Image-heavy documents
- Technical drawings
- Forms with vector graphics

---

## WEEK 13: Days 61-65 - PDF Advanced Features

### **DAY 61: Annotations**

**Goals:**
1. Annotation types
2. Annotation extraction
3. Hyperlink resolution
4. Comments and markup

**Deliverables:**

**File:** `zig/pdf/annotations.zig` (~1,000 lines)

**Features:**
- Annotation types:
  - Text (notes, comments)
  - Link (hyperlinks)
  - FreeText (callouts)
  - Line, Square, Circle (markup)
  - Highlight, Underline, StrikeOut, Squiggly
  - Stamp, Ink (handwritten)
  - FileAttachment
- Annotation dictionary:
  - Rect (bounding box)
  - Contents (text)
  - A (action, e.g., GoTo, URI)
  - Dest (destination for links)
- Hyperlink resolution:
  - Parse link action (URI, GoTo)
  - Resolve destination (page, named destination)
- Comments and markup:
  - Extract comment text
  - Extract author, date
  - Reply threads

**Annotation Output:**
```zig
struct Annotation {
    type: AnnotationType,
    bbox: BoundingBox,
    contents: []const u8,
    action: ?Action,  // Link action
    author: []const u8,
    date: []const u8,
};
```

**Tests:**
- Link annotations
- Comment annotations
- Markup annotations
- Link resolution

---

### **DAY 62: Bookmarks & Outline**

**Goals:**
1. Outline dictionary parsing
2. Bookmark hierarchy
3. Destination resolution
4. Table of contents generation

**Deliverables:**

**File:** `zig/pdf/outline.zig` (~800 lines)

**Features:**
- Outline dictionary:
  - Outlines (root)
  - OutlineItem (each bookmark)
  - Title, Dest, A (action)
  - First, Last, Next, Prev, Parent (hierarchy)
  - Count (number of descendants)
- Bookmark hierarchy:
  - Tree structure
  - Nested bookmarks
  - Expanded/collapsed state (Count +/-)
- Destination resolution:
  - Named destinations (Dests dictionary)
  - Explicit destinations ([page /XYZ x y zoom])
  - GoTo actions
- Table of contents generation:
  - Extract bookmark titles
  - Generate hierarchical TOC
  - Resolve page numbers

**Bookmark Output:**
```zig
struct Bookmark {
    title: []const u8,
    page: u32,  // Page number (0-indexed)
    x: f32, y: f32,  // Position on page
    children: []Bookmark,
};
```

**Tests:**
- Flat bookmark list
- Nested bookmarks
- Destination resolution
- Named destinations

---

### **DAY 63: Metadata**

**Goals:**
1. Document info dictionary
2. XMP metadata parsing
3. Creation/modification dates
4. Author, title, subject extraction

**Deliverables:**

**File:** `zig/pdf/metadata.zig` (~700 lines)

**Features:**
- Document info dictionary:
  - Title, Author, Subject, Keywords
  - Creator, Producer
  - CreationDate, ModDate
- XMP metadata (XML):
  - Parse XMP stream (in Metadata object)
  - Dublin Core elements (dc:title, dc:creator, dc:description)
  - PDF properties (pdf:Producer, pdf:Keywords)
  - XMP basic (xmp:CreateDate, xmp:ModifyDate)
- Date parsing:
  - PDF date format: (D:YYYYMMDDHHmmSSOHH'mm')
  - Convert to ISO 8601 or Unix timestamp

**Metadata Output:**
```zig
struct Metadata {
    title: ?[]const u8,
    author: ?[]const u8,
    subject: ?[]const u8,
    keywords: ?[]const u8,
    creator: ?[]const u8,
    producer: ?[]const u8,
    creation_date: ?i64,  // Unix timestamp
    mod_date: ?i64,
};
```

**Tests:**
- Info dictionary parsing
- XMP metadata parsing
- Date format parsing
- Missing metadata handling

---

### **DAY 64: Forms (AcroForm)**

**Goals:**
1. Form field extraction
2. Field values
3. Field types (text, checkbox, radio)
4. Form data export

**Deliverables:**

**File:** `zig/pdf/forms.zig` (~1,000 lines)

**Features:**
- AcroForm dictionary:
  - Fields (array of field dictionaries)
  - NeedAppearances, SigFlags
- Field types:
  - Text (Tx): Single-line, multi-line
  - Button (Btn): Pushbutton, checkbox, radio
  - Choice (Ch): List box, combo box
  - Signature (Sig)
- Field properties:
  - FT (field type)
  - T (field name)
  - V (field value)
  - Rect (field position)
  - Flags (ReadOnly, Required, NoExport, etc.)
- Field hierarchy:
  - Parent-child relationships
  - Inherited properties
- Form data export:
  - Extract field names and values
  - Export as JSON or XML

**Form Output:**
```zig
struct FormField {
    name: []const u8,
    type: FieldType,
    value: ?[]const u8,
    bbox: BoundingBox,
    flags: u32,
};
```

**Tests:**
- Text fields
- Checkboxes and radio buttons
- List boxes and combo boxes
- Form data extraction

---

### **DAY 65: PDF Advanced Testing**

**Goals:**
1. Interactive PDFs
2. Tagged PDFs
3. Portfolios
4. Complex documents

**Deliverables:**

**File:** `zig/tests/pdf_advanced_test.zig` (~500 lines)

**Test Coverage:**
- Annotations (all types)
- Bookmarks and outline
- Metadata (info dict, XMP)
- Forms (AcroForms)
- Interactive elements
- Tagged PDFs (structure tree, basic)
- PDF portfolios (embedded files)

**Test PDFs:**
- Forms (government forms, surveys)
- Interactive PDFs (links, bookmarks)
- Tagged PDFs (accessibility)

---

## WEEK 14: Days 66-70 - Layout Analysis (ML-Based)

### **DAY 66-67: Layout Detection Model**

**Goals:**
1. Implement layout segmentation model
2. Page region classification
3. Bounding box detection
4. Model integration

**Deliverables:**

**File:** `zig/ml/layout_model.zig` (~2,000 lines)

**Features:**
- Layout segmentation model (CNN-based):
  - Input: Page image (e.g., 800x1200 pixels)
  - Output: Region masks and labels
- Region classification:
  - Text (body text)
  - Heading
  - Table
  - Figure/Image
  - Caption
  - List
  - Code block
  - Formula
- Bounding box detection:
  - Detect bounding boxes around regions
  - Confidence scores per region
- Model architecture (example):
  - Backbone: ResNet-like (feature extraction)
  - Neck: FPN (Feature Pyramid Network)
  - Head: Detection head (bounding box + class)
- Integration with custom inference engine:
  - Load pre-trained weights
  - Forward pass
  - Post-processing (NMS, score threshold)

**Model Pipeline:**
1. Render PDF page as image
2. Preprocess (resize, normalize)
3. Forward pass through model
4. Post-process (NMS to remove duplicates)
5. Extract regions with labels and bboxes

**Pre-trained Weights:**
- Train on PubLayNet or DocBank dataset
- Convert to custom format (via model converter)

**Tests:**
- Layout detection accuracy
- Bounding box accuracy
- Various document types

---

### **DAY 68: Table Structure Recognition**

**Goals:**
1. Table detection model
2. Row/column boundary detection
3. Cell merging detection
4. Table structure output

**Deliverables:**

**File:** `zig/ml/table_model.zig` (~1,500 lines)

**Features:**
- Table detection:
  - Detect table regions (from layout model or dedicated model)
- Table structure recognition:
  - Detect row boundaries (horizontal lines or gaps)
  - Detect column boundaries (vertical lines or gaps)
  - Identify cell merging (rowspan, colspan)
- Model (optional):
  - Table structure model (e.g., TableFormer-like)
  - Input: Table region image
  - Output: Row/column grid
- Rule-based alternative:
  - Use extracted graphics (lines, rectangles)
  - Text alignment detection
  - Grid inference from text positions

**Table Structure Output:**
```zig
struct Table {
    bbox: BoundingBox,
    rows: u32,
    cols: u32,
    cells: [][]Cell,  // 2D array of cells
};

struct Cell {
    text: []const u8,
    rowspan: u32,
    colspan: u32,
};
```

**Tests:**
- Simple tables (grid lines)
- Complex tables (merged cells)
- Tables without grid lines
- Accuracy metrics

---

### **DAY 69: Reading Order Model**

**Goals:**
1. Reading order prediction
2. Multi-column handling
3. Complex layout support
4. Integration with layout model

**Deliverables:**

**File:** `zig/ml/reading_order.zig` (~1,000 lines)

**Features:**
- Reading order prediction:
  - Given regions (from layout model), determine reading order
  - Left-to-right, top-to-bottom (primary)
  - Column-aware (detect columns, order within columns)
  - Handle complex layouts (sidebars, footnotes)
- Model (optional):
  - Reading order model (GNN-based or sequence model)
  - Input: Region positions and types
  - Output: Reading order (sequence of region IDs)
- Rule-based alternative:
  - Geometric heuristics (top-to-bottom, left-to-right)
  - Column detection (vertical gaps)
  - XY-cut algorithm (recursive splitting)

**Reading Order Algorithm (Rule-Based):**
1. Sort regions by Y position (top to bottom)
2. For each horizontal slice:
   a. Detect columns (vertical gaps)
   b. Sort regions within slice by X position (left to right per column)
3. Output ordered sequence of regions

**Tests:**
- Single-column documents
- Multi-column documents
- Complex layouts (sidebars, text boxes)
- Accuracy validation

---

### **DAY 70: Layout Testing**

**Goals:**
1. Test on diverse document layouts
2. Accuracy metrics
3. Performance optimization
4. Integration tests

**Deliverables:**

**File:** `zig/tests/layout_test.zig` (~500 lines)

**Test Coverage:**
- Layout segmentation accuracy
- Table structure recognition
- Reading order accuracy
- Various document types:
  - Academic papers (2-column)
  - Newspapers (multi-column, complex)
  - Books (single-column)
  - Forms (tables, mixed)

**Accuracy Metrics:**
- Intersection over Union (IoU) for bounding boxes
- F1 score for region classification
- Edit distance for reading order
- Table cell accuracy

**Performance:**
- Inference speed (pages per second)
- Memory usage
- Optimization (model quantization, batch processing)

---

# PHASE 4: Office Format Implementation (Days 71-85)

## WEEK 15: Days 71-75 - DOCX Full Implementation

### **DAY 71: Document Structure**

**Goals:**
1. document.xml full parsing
2. Section properties
3. Page setup
4. Headers/footers

**Deliverables:**

**File:** `zig/parsers/docx.zig` (~1,500 lines)

**Features:**
- Document structure (`word/document.xml`):
  - Body element
  - Paragraphs (<w:p>)
  - Runs (<w:r>)
  - Tables (<w:tbl>)
  - Sections (<w:sectPr>)
- Section properties:
  - Page size (width, height)
  - Page margins (top, bottom, left, right)
  - Page orientation (portrait, landscape)
  - Columns (number, spacing)
  - Headers/footers (first page, even/odd pages)
- Page setup:
  - Paper size
  - Orientation
  - Margins
- Headers/footers:
  - Parse header1.xml, header2.xml, footer1.xml, etc.
  - Extract content (text, images)
  - Apply to pages

**Document Structure:**
```xml
<w:document>
  <w:body>
    <w:p>...</w:p>  <!-- Paragraph -->
    <w:tbl>...</w:tbl>  <!-- Table -->
    <w:sectPr>...</w:sectPr>  <!-- Section properties -->
  </w:body>
</w:document>
```

**Tests:**
- Multi-section documents
- Headers and footers
- Various page layouts

---

### **DAY 72: Paragraphs & Runs**

**Goals:**
1. Paragraph properties
2. Run properties
3. Text and formatting extraction
4. Nested formatting

**Deliverables:**

**File:** `zig/parsers/docx_text.zig` (~1,200 lines)

**Features:**
- Paragraph properties (<w:pPr>):
  - Alignment (left, center, right, justify)
  - Indentation (left, right, first line, hanging)
  - Spacing (before, after, line spacing)
  - Borders
  - Shading (background color)
- Run properties (<w:rPr>):
  - Font family (<w:rFonts>)
  - Font size (<w:sz>)
  - Bold (<w:b>)
  - Italic (<w:i>)
  - Underline (<w:u>)
  - Strikethrough (<w:strike>)
  - Color (<w:color>)
  - Highlight (<w:highlight>)
- Text extraction:
  - <w:t> elements (text content)
  - Preserve spaces (<xml:space="preserve">)
  - Tab characters (<w:tab>)
  - Line breaks (<w:br>)
- Nested formatting:
  - Runs within paragraphs
  - Style inheritance (document → paragraph → run)

**Text Extraction Output:**
```zig
struct Paragraph {
    text: []const u8,
    alignment: Alignment,
    runs: []Run,
};

struct Run {
    text: []const u8,
    font_family: []const u8,
    font_size: f32,
    is_bold: bool,
    is_italic: bool,
    is_underline: bool,
    color: u32,  // RGB
};
```

**Tests:**
- Various paragraph alignments
- Text formatting (bold, italic, etc.)
- Nested formatting
- Tab and line breaks

---

### **DAY 73: Tables**

**Goals:**
1. Table structure
2. Row/column properties
3. Cell merging
4. Table styles

**Deliverables:**

**File:** `zig/parsers/docx_tables.zig` (~1,000 lines)

**Features:**
- Table structure (<w:tbl>):
  - Table rows (<w:tr>)
  - Table cells (<w:tc>)
  - Table properties (<w:tblPr>)
- Row properties (<w:trPr>):
  - Row height
  - Header row (repeat on each page)
- Cell properties (<w:tcPr>):
  - Cell width
  - Cell borders
  - Cell shading (background)
  - Vertical merge (vMerge)
  - Horizontal merge (gridSpan)
- Cell merging:
  - vMerge (vertical merge): "restart" and "continue"
  - gridSpan (horizontal merge): span multiple columns
- Table styles:
  - Table style ID
  - Apply conditional formatting (first row, last row, etc.)

**Table Structure:**
```xml
<w:tbl>
  <w:tr>  <!-- Row -->
    <w:tc>  <!-- Cell -->
      <w:tcPr>
        <w:gridSpan w:val="2"/>  <!-- Merge 2 columns -->
      </w:tcPr>
      <w:p><w:r><w:t>Cell text</w:t></w:r></w:p>
    </w:tc>
  </w:tr>
</w:tbl>
```

**Tests:**
- Simple tables
- Merged cells (rows and columns)
- Table styles
- Nested tables

---

### **DAY 74: Advanced Features**

**Goals:**
1. Numbered and bulleted lists
2. Hyperlinks
3. Bookmarks
4. Comments and track changes

**Deliverables:**

**File:** `zig/parsers/docx_advanced.zig` (~1,200 lines)

**Features:**
- Lists:
  - Numbering definition (`word/numbering.xml`)
  - List level (indent, number format)
  - Paragraph numbering properties (<w:numPr>)
  - Bulleted lists (symbol bullets)
  - Numbered lists (1, 2, 3 or a, b, c or i, ii, iii)
- Hyperlinks:
  - Internal links (<w:hyperlink>)
  - External links (r:id references)
  - Link text and destination
- Bookmarks:
  - Bookmark start (<w:bookmarkStart>)
  - Bookmark end (<w:bookmarkEnd>)
  - Bookmark names and IDs
- Comments:
  - Comment references (<w:commentReference>)
  - Comment text (`word/comments.xml`)
  - Author and date
- Track changes:
  - Insertions (<w:ins>)
  - Deletions (<w:del>)
  - Author and timestamp

**List Numbering:**
```xml
<w:p>
  <w:pPr>
    <w:numPr>
      <w:ilvl w:val="0"/>  <!-- Level 0 -->
      <w:numId w:val="1"/>  <!-- Numbering ID -->
    </w:numPr>
  </w:pPr>
  <w:r><w:t>List item</w:t></w:r>
</w:p>
```

**Tests:**
- Bulleted lists
- Numbered lists (various formats)
- Hyperlinks (internal, external)
- Bookmarks
- Comments

---

### **DAY 75: DOCX Testing**

**Goals:**
1. Complex documents
2. Nested structures
3. Style inheritance
4. Real-world DOCX files

**Deliverables:**

**File:** `zig/tests/docx_test.zig` (~500 lines)

**Test Coverage:**
- Paragraphs with various formatting
- Tables (simple and complex)
- Lists (bulleted, numbered, nested)
- Hyperlinks and bookmarks
- Headers and footers
- Multi-section documents
- Comments and track changes

**Test Documents:**
- Academic papers
- Reports (with tables and figures)
- Resumes (formatting-heavy)
- Contracts (complex structure)

---

## WEEK 16: Days 76-80 - XLSX Full Implementation

### **DAY 76: Worksheet Parsing**

**Goals:**
1. worksheet.xml structure
2. Cell data extraction
3. Cell references
4. Formulas

**Deliverables:**

**File:** `zig/parsers/xlsx.zig` (~1,500 lines)

**Features:**
- Worksheet structure (`xl/worksheets/sheet1.xml`):
  - <sheetData> (cell data)
  - <row> (rows)
  - <c> (cells)
  - Dimension (used range)
- Cell data:
  - Cell reference (A1, B2, etc.)
  - Cell type (t="s" for shared string, t="n" for number, t="b" for boolean)
  - Cell value (<v>)
  - Cell formula (<f>)
- Cell references:
  - A1 notation (e.g., "A1", "B10")
  - R1C1 notation (optional)
  - Range references (e.g., "A1:B10")
- Formulas:
  - Formula text (<f>)
  - Shared formulas (si attribute)
  - Array formulas (ref attribute)
  - Calculated values (cached in <v>)

**Cell Structure:**
```xml
<row r="1">
  <c r="A1" t="s">  <!-- Shared string -->
    <v>0</v>  <!-- Index into shared string table -->
  </c>
  <c r="B1">  <!-- Number -->
    <v>123.45</v>
  </c>
  <c r="C1">  <!-- Formula -->
    <f>A1+B1</f>
    <v>123.45</v>  <!-- Cached result -->
  </c>
</row>
```

**Tests:**
- Cell data extraction
- Cell references (A1 notation)
- Formulas (simple, shared, array)
- Large worksheets

---

### **DAY 77: Shared Strings & Styles**

**Goals:**
1. Shared string table
2. Style application
3. Number formats
4. Conditional formatting

**Deliverables:**

**File:** `zig/parsers/xlsx_styles.zig` (~1,200 lines)

**Features:**
- Shared string table:
  - Already implemented (Day 18)
  - Resolve cell values via SST index
- Styles (`xl/styles.xml`):
  - Cell formats (cellXfs)
  - Cell styles (cellStyles)
  - Fonts, fills, borders
  - Apply style to cells (s attribute)
- Number formats:
  - Built-in formats (0 = General, 1 = 0, 2 = 0.00, etc.)
  - Custom formats (e.g., "$#,##0.00")
  - Apply to cell values
- Conditional formatting:
  - Rules (<conditionalFormatting>)
  - Color scales, data bars, icon sets
  - Metadata extraction (rules, ranges)

**Style Application:**
```zig
struct Cell {
    value: CellValue,
    style: ?CellStyle,
};

struct CellStyle {
    font: Font,
    fill: Fill,
    border: Border,
    number_format: NumberFormat,
};
```

**Tests:**
- Shared strings
- Style application
- Number formats (currency, date, percentage)
- Conditional formatting

---

### **DAY 78: Charts & Drawings**

**Goals:**
1. Chart extraction
2. Drawing objects
3. Image embedding
4. Chart metadata

**Deliverables:**

**File:** `zig/parsers/xlsx_charts.zig` (~1,000 lines)

**Features:**
- Charts (`xl/charts/chart1.xml`):
  - Chart type (bar, line, pie, scatter, etc.)
  - Chart title
  - Data series (categories, values)
  - Axis labels
- Drawing objects (`xl/drawings/drawing1.xml`):
  - Shapes (rectangles, circles, etc.)
  - Text boxes
  - Images
  - Anchor (absolute or one-cell/two-cell)
- Image embedding:
  - Image relationships
  - Extract image data (from `xl/media/`)
  - Image position (anchor)

**Chart Metadata:**
```zig
struct Chart {
    type: ChartType,
    title: []const u8,
    series: []DataSeries,
    x_axis_label: []const u8,
    y_axis_label: []const u8,
};
```

**Tests:**
- Various chart types
- Images in worksheets
- Drawing objects
- Chart data extraction

---

### **DAY 79: Advanced Features**

**Goals:**
1. Named ranges
2. Data validation
3. Filters and sorting metadata
4. Pivot tables (metadata)

**Deliverables:**

**File:** `zig/parsers/xlsx_advanced.zig` (~800 lines)

**Features:**
- Named ranges (`xl/workbook.xml`):
  - Defined names (<definedNames>)
  - Name, reference (e.g., "MyRange" = "Sheet1!$A$1:$B$10")
  - Scope (workbook or worksheet)
- Data validation:
  - Validation rules (<dataValidations>)
  - Type (list, whole, decimal, date, etc.)
  - Formula1, formula2 (constraints)
  - Error message
- Filters and sorting:
  - AutoFilter (<autoFilter>)
  - Filter range
  - Sort state (metadata)
- Pivot tables:
  - Pivot table definition (`xl/pivotTables/`)
  - Source data range
  - Row/column fields
  - Data fields

**Tests:**
- Named ranges
- Data validation rules
- AutoFilter
- Pivot table metadata

---

### **DAY 80: XLSX Testing**

**Goals:**
1. Large spreadsheets
2. Formula evaluation (basic)
3. Complex formatting
4. Real-world XLSX files

**Deliverables:**

**File:** `zig/tests/xlsx_test.zig` (~500 lines)

**Test Coverage:**
- Cell data (various types)
- Formulas (simple calculations)
- Styles and number formats
- Charts and drawings
- Named ranges and data validation
- Large worksheets (100K+ rows)

**Test Spreadsheets:**
- Financial reports (formulas, charts)
- Data analysis (pivot tables, filters)
- Dashboards (conditional formatting, charts)

---

## WEEK 17: Days 81-85 - PPTX Full Implementation

### **DAY 81: Presentation Structure**

**Goals:**
1. presentation.xml parsing
2. Slide master/layout relationships
3. Slide sequence
4. Presentation properties

**Deliverables:**

**File:** `zig/parsers/pptx.zig` (~1,200 lines)

**Features:**
- Presentation structure (`ppt/presentation.xml`):
  - Slide list (<sldIdLst>)
  - Slide master list (<sldMasterIdLst>)
  - Slide size (width, height)
- Slide relationships:
  - Slide → Slide layout → Slide master
  - Inheritance of properties
- Slide sequence:
  - Slide order (by ID)
  - Hidden slides (show attribute)
- Presentation properties:
  - Default text styles
  - Color schemes (themes)

**Presentation Structure:**
```xml
<p:presentation>
  <p:sldSz cx="9144000" cy="6858000"/>  <!-- Slide size -->
  <p:sldIdLst>
    <p:sldId id="256" r:id="rId2"/>  <!-- Slide 1 -->
    <p:sldId id="257" r:id="rId3"/>  <!-- Slide 2 -->
  </p:sldIdLst>
</p:presentation>
```

**Tests:**
- Multi-slide presentations
- Slide masters and layouts
- Hidden slides
- Various slide sizes

---

### **DAY 82: Slide Content**

**Goals:**
1. Shape extraction
2. Text boxes
3. Text runs and formatting
4. Shape properties

**Deliverables:**

**File:** `zig/parsers/pptx_slides.zig` (~1,500 lines)

**Features:**
- Slide content (`ppt/slides/slide1.xml`):
  - Shape tree (<p:spTree>)
  - Shapes (<p:sp>)
  - Text boxes (<p:txBody>)
  - Pictures (<p:pic>)
  - Groups (<p:grpSp>)
- Shapes:
  - Shape type (rectangle, ellipse, etc.)
  - Shape position (x, y)
  - Shape size (width, height)
  - Shape properties (fill, border)
- Text boxes:
  - Text content (<a:t>)
  - Paragraphs (<a:p>)
  - Runs (<a:r>)
  - Text properties (<a:rPr>)
- Text formatting:
  - Font family, size
  - Bold, italic, underline
  - Color
  - Alignment (left, center, right)

**Shape Structure:**
```xml
<p:sp>
  <p:nvSpPr>...</p:nvSpPr>  <!-- Non-visual properties -->
  <p:spPr>...</p:spPr>  <!-- Shape properties -->
  <p:txBody>  <!-- Text body -->
    <a:p>  <!-- Paragraph -->
      <a:r>  <!-- Run -->
        <a:rPr>...</a:rPr>  <!-- Run properties -->
        <a:t>Text content</a:t>
      </a:r>
    </a:p>
  </p:txBody>
</p:sp>
```

**Tests:**
- Text boxes
- Various shapes
- Text formatting
- Nested shapes (groups)

---

### **DAY 83: Images & Media**

**Goals:**
1. Image extraction from slides
2. Video/audio metadata
3. SmartArt parsing
4. Embedded objects

**Deliverables:**

**File:** `zig/parsers/pptx_media.zig` (~1,000 lines)

**Features:**
- Images:
  - Picture shape (<p:pic>)
  - Image relationships (r:embed)
  - Extract image data (from `ppt/media/`)
  - Image position and size
- Video/audio:
  - Video shape
  - Audio shape
  - Media metadata (duration, codec)
  - File reference (no actual decoding)
- SmartArt:
  - SmartArt shape
  - SmartArt data (`diagrams/data1.xml`)
  - Text extraction from SmartArt nodes
- Embedded objects:
  - OLE objects (e.g., Excel embedded in PPT)
  - Extract as binary data

**Image Extraction:**
```zig
struct Image {
    data: []const u8,
    format: ImageFormat,  // PNG, JPEG, etc.
    width: u32,
    height: u32,
    position: Point,
};
```

**Tests:**
- Slides with images
- Video/audio metadata
- SmartArt extraction
- Embedded objects

---

### **DAY 84: Transitions & Animations**

**Goals:**
1. Transition metadata
2. Animation sequences
3. Timing information
4. Effects extraction

**Deliverables:**

**File:** `zig/parsers/pptx_animations.zig` (~800 lines)

**Features:**
- Transitions:
  - Transition type (fade, wipe, push, etc.)
  - Transition duration
  - Transition direction
- Animations:
  - Animation sequences (<p:timing>)
  - Animation effects (entrance, emphasis, exit)
  - Target shapes
  - Timing (delay, duration)
- Effects:
  - Effect type (fly in, zoom, rotate, etc.)
  - Effect properties (speed, direction)

**Animation Metadata:**
```zig
struct Animation {
    type: AnimationType,
    target_shape: u32,  // Shape ID
    effect: AnimationEffect,
    delay: f32,  // seconds
    duration: f32,
};
```

**Tests:**
- Slide transitions
- Animation sequences
- Timing extraction
- Various effects

---

### **DAY 85: PPTX Testing**

**Goals:**
1. Complex presentations
2. Master slide inheritance
3. Grouped shapes
4. Real-world PPTX files

**Deliverables:**

**File:** `zig/tests/pptx_test.zig` (~500 lines)

**Test Coverage:**
- Multi-slide presentations
- Text and images
- SmartArt
- Animations and transitions
- Master slides and layouts
- Grouped shapes
- Embedded objects

**Test Presentations:**
- Corporate presentations (templates, branding)
- Educational slides (diagrams, SmartArt)
- Photo albums (image-heavy)

---

# PHASE 5: Pipeline & API (Days 86-105)

## WEEK 18: Days 86-90 - Pipeline Framework

### **DAY 86-87: Base Pipeline (Mojo)**

**Goals:**
1. Pipeline trait/interface
2. Stage-based processing
3. Error propagation
4. Progress tracking

**Deliverables:**

**File:** `mojo/pipeline.mojo` (~1,500 lines)

**Features:**
- Pipeline trait:
  - execute() method
  - Status reporting
  - Error handling
- Stage-based processing:
  - Input → Stage1 → Stage2 → ... → Output
  - Each stage processes document
  - Stages can be composed
- Error propagation:
  - Result[T, Error] return type
  - Error context (where, why)
  - Partial success (some pages OK, some failed)
- Progress tracking:
  - Callback mechanism
  - Progress percentage
  - Current stage information

**Pipeline Trait:**
```mojo
trait Pipeline:
    fn execute(inout self, doc: InputDocument) -> Result[ConversionResult, Error]:
        pass
    
    fn name(self) -> String:
        pass
```

**Stages:**
- Parse stage (format-specific)
- OCR stage (if needed)
- Layout analysis stage (ML-based)
- Assembly stage (combine elements)
- Export stage (generate output)

**Tests:**
- Pipeline execution
- Error handling
- Progress tracking
- Stage composition

---

### **DAY 88: Simple Pipeline**

**Goals:**
1. For text formats (CSV, MD, HTML)
2. Single-pass processing
3. Synchronous execution
4. Minimal overhead

**Deliverables:**

**File:** `mojo/simple_pipeline.mojo` (~600 lines)

**Features:**
- Simple pipeline for declarative formats:
  - CSV, Markdown, HTML, TXT
  - No page-by-page processing
  - Direct parse → DoclingDocument
- Synchronous execution:
  - No concurrency (not needed for simple formats)
  - Fast and straightforward
- Minimal overhead:
  - No ML models
  - No OCR
  - Direct conversion

**SimplePipeline:**
```mojo
struct SimplePipeline(Pipeline):
    fn execute(inout self, doc: InputDocument) -> Result[ConversionResult, Error]:
        # 1. Detect format
        # 2. Call appropriate parser (CSV, MD, HTML)
        # 3. Create DoclingDocument
        # 4. Return result
        pass
```

**Tests:**
- CSV files
- Markdown files
- HTML files
- Error handling

---

### **DAY 89: Paginated Pipeline**

**Goals:**
1. For page-based formats (PDF, PPTX)
2. Page-level parallel processing
3. Result aggregation
4. Memory efficiency

**Deliverables:**

**File:** `mojo/paginated_pipeline.mojo` (~1,000 lines)

**Features:**
- Paginated pipeline for page-based formats:
  - PDF, PPTX, Images (multi-page TIFF)
- Page-level parallelism:
  - Process pages concurrently (use Mojo SDK's concurrency)
  - Thread pool (configurable size)
  - Parallel OCR, parallel layout analysis
- Result aggregation:
  - Combine page results
  - Maintain page order
  - Handle partial failures (some pages OK, some failed)
- Memory efficiency:
  - Stream pages (don't load all at once)
  - Release page resources after processing

**PaginatedPipeline:**
```mojo
struct PaginatedPipeline(Pipeline):
    fn execute(inout self, doc: InputDocument) -> Result[ConversionResult, Error]:
        # 1. Load page count
        # 2. For each page (parallel):
        #    a. Render page
        #    b. OCR (if needed)
        #    c. Layout analysis
        #    d. Extract text/images
        # 3. Aggregate results
        # 4. Assemble document
        pass
```

**Tests:**
- PDF documents (multi-page)
- PPTX presentations
- Parallel processing correctness
- Memory usage

---

### **DAY 90: Standard Pipeline**

**Goals:**
1. Advanced pipeline with caching
2. Incremental processing
3. Dependency tracking
4. Optimization

**Deliverables:**

**File:** `mojo/standard_pipeline.mojo` (~1,200 lines)

**Features:**
- Standard pipeline with advanced features:
  - Caching (parsed pages, OCR results, ML results)
  - Incremental processing (re-use cached results)
  - Dependency tracking (which stages depend on others)
- Caching:
  - Cache key (document hash + page number + stage name)
  - Cache storage (memory or disk)
  - Cache expiration (LRU eviction)
- Incremental processing:
  - If document unchanged, use cached results
  - If page unchanged, use cached page results
- Optimization:
  - Skip unnecessary stages (e.g., no OCR if text embedded)
  - Batch ML inference (multiple pages at once)

**StandardPipeline:**
```mojo
struct StandardPipeline(Pipeline):
    var cache: Cache
    
    fn execute(inout self, doc: InputDocument) -> Result[ConversionResult, Error]:
        # 1. Check cache
        # 2. Execute stages (with caching)
        # 3. Store results in cache
        pass
```

**Tests:**
- Caching behavior
- Incremental processing
- Cache eviction
- Performance improvement

---

## WEEK 19: Days 91-95 - Document Assembly

### **DAY 91: Element Assembly**

**Goals:**
1. Combine parsed elements
2. Generate unified DoclingDocument
3. Cross-reference resolution
4. Element deduplication

**Deliverables:**

**File:** `mojo/assembly.mojo` (~1,000 lines)

**Features:**
- Element assembly:
  - Collect elements from all pages
  - Maintain order (reading order)
  - Remove duplicates (e.g., page headers/footers)
- Unified DoclingDocument:
  - Single document with all elements
  - Page boundaries preserved
  - Element provenance (source page, position)
- Cross-reference resolution:
  - Link footnotes to references
  - Link TOC entries to sections
  - Link hyperlinks to targets
- Element deduplication:
  - Detect repeated headers/footers
  - Merge duplicate elements (e.g., page numbers)

**Assembly Process:**
1. For each page:
   a. Extract elements (from parser output)
   b. Add to document element list
2. Resolve cross-references
3. Deduplicate elements
4. Generate final DoclingDocument

**Tests:**
- Multi-page documents
- Cross-references
- Deduplication
- Element order

---

### **DAY 92: Hierarchy Construction**

**Goals:**
1. Section tree building
2. Heading levels
3. Nested list structures
4. Table of contents

**Deliverables:**

**File:** `mojo/hierarchy.mojo` (~800 lines)

**Features:**
- Section tree:
  - Build hierarchical structure from headings
  - Heading levels (H1, H2, H3, ...)
  - Nest paragraphs under headings
- Heading detection:
  - From explicit markup (PDF bookmarks, HTML <h1>-<h6>, DOCX headings)
  - From font size and style (bold, larger than body text)
- Nested lists:
  - Detect list hierarchy (bullet, numbered)
  - Nest sub-lists
  - List item numbering
- Table of contents:
  - Generate from heading hierarchy
  - Include page numbers (if available)

**Hierarchy Output:**
```mojo
struct Section:
    var title: String
    var level: Int  # 1, 2, 3, ...
    var children: List[Section]
    var elements: List[Element]  # Paragraphs, tables, etc.
```

**Tests:**
- Flat documents (no headings)
- Hierarchical documents (nested sections)
- List structures
- TOC generation

---

### **DAY 93: Metadata Extraction**

**Goals:**
1. Document properties
2. Language detection
3. Page count, word count
4. Statistics

**Deliverables:**

**File:** `mojo/metadata.mojo` (~600 lines)

**Features:**
- Document properties:
  - Title, author, subject, keywords (from PDF/Office)
  - Creation date, modification date
  - Creator, producer
- Language detection (heuristic):
  - Character frequency analysis
  - Common words detection
  - Script detection (Latin, Cyrillic, Arabic, etc.)
- Statistics:
  - Page count
  - Word count
  - Character count
  - Average words per page
- Additional metadata:
  - Number of images
  - Number of tables
  - Document complexity score

**Language Detection:**
- Analyze text content
- Count character frequencies (e.g., 'e', 't' high in English)
- Detect common words ("the", "and", "is" → English)
- Unicode script blocks (U+0600-U+06FF → Arabic)

**Tests:**
- Various languages
- Mixed-language documents
- Statistics accuracy

---

### **DAY 94: Provenance Tracking**

**Goals:**
1. Source page/position for each element
2. Original format information
3. Transformation log
4. Quality metrics

**Deliverables:**

**File:** `mojo/provenance.mojo` (~500 lines)

**Features:**
- Provenance tracking:
  - Track source page number for each element
  - Track original position (bounding box)
  - Track extraction method (parser, OCR, ML)
- Original format information:
  - Source file format (PDF, DOCX, etc.)
  - Original element type (from source)
  - Formatting preservation
- Transformation log:
  - Record each processing stage
  - Timestamps
  - Stage results (success, partial, failure)
- Quality metrics:
  - OCR confidence (per element)
  - Layout analysis confidence
  - Overall document quality score

**Provenance Data:**
```mojo
struct Provenance:
    var source_page: Int
    var source_bbox: BoundingBox
    var extraction_method: String  # "parser", "ocr", "ml"
    var confidence: Float32
    var timestamp: Int64
```

**Tests:**
- Provenance tracking
- Transformation log
- Quality metrics

---

### **DAY 95: Assembly Testing**

**Goals:**
1. End-to-end assembly tests
2. Complex documents
3. Cross-reference resolution
4. Integration tests

**Deliverables:**

**File:** `mojo/tests/assembly_test.mojo` (~500 lines)

**Test Coverage:**
- Element assembly
- Hierarchy construction
- Metadata extraction
- Provenance tracking
- Multi-page documents
- Various formats (PDF, DOCX, etc.)

**Test Documents:**
- Academic papers (sections, references)
- Books (chapters, TOC)
- Reports (tables, figures)

---

## WEEK 20: Days 96-100 - Reading Order & Structure

### **DAY 96-97: Reading Order Algorithm**

**Goals:**
1. Z-order traversal
2. Multi-column detection
3. Right-to-left language support
4. Vertical text support

**Deliverables:**

**File:** `mojo/reading_order.mojo` (~1,200 lines)

**Features:**
- Reading order algorithm:
  - Z-order traversal (left-to-right, top-to-bottom)
  - Multi-column detection (vertical gaps)
  - Column ordering (left-to-right per column)
- Right-to-left (RTL) support:
  - Detect RTL languages (Arabic, Hebrew)
  - Right-to-left ordering within columns
  - Mixed LTR/RTL handling
- Vertical text support:
  - Detect vertical text (CJK)
  - Top-to-bottom ordering
  - Column ordering (right-to-left for vertical)
- Complex layouts:
  - Sidebars
  - Text boxes
  - Footnotes (bottom of page)
  - Page numbers (top/bottom)

**XY-Cut Algorithm:**
1. Sort elements by Y (top to bottom)
2. Find horizontal gaps (split into rows)
3. For each row:
   a. Sort by X (left to right)
   b. Find vertical gaps (split into columns)
   c. Order elements within columns
4. Output ordered sequence

**Tests:**
- Single-column documents
- Multi-column documents
- RTL documents
- Vertical text (CJK)
- Complex layouts

---

### **DAY 98: Document Structure Inference**

**Goals:**
1. Heading hierarchy detection
2. Section boundaries
3. List structure
4. Table of contents generation

**Deliverables:**

**File:** `mojo/structure_inference.mojo` (~1,000 lines)

**Features:**
- Heading hierarchy:
  - Detect headings (font size, bold, position)
  - Determine heading levels (H1 > H2 > H3)
  - Nest sections under headings
- Section boundaries:
  - Detect section starts (headings, page breaks)
  - Group elements into sections
- List structure:
  - Detect list markers (bullets, numbers)
  - Nest sub-lists
  - Determine list type (bullet, numbered, etc.)
- Table of contents:
  - Generate from heading hierarchy
  - Include page numbers
  - Hierarchical structure

**Heading Detection Heuristics:**
- Font size > body text
- Bold or strong style
- Preceded by whitespace
- Followed by paragraph

**Tests:**
- Various document structures
- Heading detection accuracy
- List structure correctness
- TOC generation

---

### **DAY 99: Semantic Analysis**

**Goals:**
1. Paragraph classification
2. Code block detection
3. Formula detection
4. Citation detection

**Deliverables:**

**File:** `mojo/semantic_analysis.mojo` (~800 lines)

**Features:**
- Paragraph classification:
  - Body text
  - Caption (figure, table)
  - Footnote
  - Header/footer
  - Page number
- Code block detection:
  - Monospace font
  - Indentation
  - Syntax highlighting (from source)
- Formula detection:
  - LaTeX syntax (from PDF/Word)
  - Math symbols (∑, ∫, ∂, etc.)
  - Fraction notation
- Citation detection:
  - In-text citations ([1], (Smith 2020))
  - Bibliography entries
  - Reference numbers

**Semantic Labels:**
```mojo
enum SemanticLabel:
    BodyText
    Caption
    Footnote
    HeaderFooter
    PageNumber
    CodeBlock
    Formula
    Citation
```

**Tests:**
- Various paragraph types
- Code blocks
- Formulas
- Citations

---

### **DAY 100: Structure Testing**

**Goals:**
1. End-to-end structure tests
2. Reading order accuracy
3. Semantic analysis accuracy
4. Integration tests

**Deliverables:**

**File:** `mojo/tests/structure_test.mojo` (~500 lines)

**Test Coverage:**
- Reading order algorithm
- Multi-column documents
- RTL documents
- Heading hierarchy
- List structures
- Semantic classification

**Test Documents:**
- Academic papers (complex structure)
- Books (chapters, sections)
- Newspapers (multi-column)
- Technical documents (code, formulas)

---

## WEEK 21: Days 101-105 - Export Formats

### **DAY 101: Markdown Export**

**Goals:**
1. DoclingDocument → Markdown
2. Preserve formatting
3. Table rendering
4. Image embedding

**Deliverables:**

**File:** `mojo/export_markdown.mojo` (~1,000 lines)

**Features:**
- Markdown export:
  - Headings (# H1, ## H2, ### H3)
  - Paragraphs (double newline)
  - Bold (**bold**), italic (*italic*), code (`code`)
  - Lists (bullets - or *, numbered 1. 2. 3.)
  - Tables (GitHub Flavored Markdown)
  - Images (![alt](url) or ![alt](data:image/png;base64,...))
  - Links ([text](url))
  - Code blocks (```language\ncode\n```)
  - Blockquotes (> quote)
- Table rendering:
  - GFM table syntax (| col1 | col2 |)
  - Header separator (|------|------|)
  - Cell alignment (left, center, right)
- Image embedding:
  - External images (URL)
  - Embedded images (data URI with base64)
- Math blocks:
  - LaTeX syntax ($$...$$, or $...$)

**Markdown Output Example:**
```markdown
# Document Title

This is a **bold** paragraph with *italic* text.

## Section 1

- Bullet point 1
- Bullet point 2

### Subsection

1. Numbered item
2. Another item

| Header 1 | Header 2 |
|----------|----------|
| Cell 1   | Cell 2   |

![Image](image.png)

```python
print("Hello, world!")
```
```

**Tests:**
- Various elements
- Tables
- Images
- Code blocks
- Round-trip (Markdown → parse → export → compare)

---

### **DAY 102: HTML Export**

**Goals:**
1. Semantic HTML5
2. CSS styling
3. Table rendering
4. Syntax highlighting for code

**Deliverables:**

**File:** `mojo/export_html.mojo` (~1,200 lines)

**Features:**
- Semantic HTML5:
  - <article>, <section>, <header>, <footer>
  - <h1>-<h6> for headings
  - <p> for paragraphs
  - <ul>, <ol>, <li> for lists
  - <table>, <tr>, <td> for tables
  - <figure>, <figcaption> for images
  - <pre>, <code> for code blocks
- CSS styling:
  - Embedded CSS (in <style> tag)
  - External CSS (link to stylesheet)
  - Responsive design (media queries)
  - Print styles
- Table rendering:
  - <thead>, <tbody>, <tfoot>
  - Rowspan, colspan for merged cells
  - CSS for table styling
- Syntax highlighting:
  - Use CSS classes for code tokens
  - Pre-generate HTML with classes (e.g., <span class="keyword">)
  - Include CSS for highlighting (e.g., Monokai theme)

**HTML Output Example:**
```html
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>Document Title</title>
  <style>
    body { font-family: Arial, sans-serif; }
    h1 { color: #333; }
  </style>
</head>
<body>
  <article>
    <h1>Document Title</h1>
    <p>This is a <strong>bold</strong> paragraph.</p>
    <table>
      <tr><th>Header 1</th><th>Header 2</th></tr>
      <tr><td>Cell 1</td><td>Cell 2</td></tr>
    </table>
  </article>
</body>
</html>
```

**Tests:**
- HTML validation (W3C validator)
- CSS styling
- Table rendering
- Code syntax highlighting

---

### **DAY 103: JSON Export**

**Goals:**
1. Lossless serialization
2. Schema definition
3. Nested structure preservation
4. Binary data encoding

**Deliverables:**

**File:** `mojo/export_json.mojo` (~800 lines)

**Features:**
- JSON export:
  - Lossless serialization of DoclingDocument
  - All elements, metadata, provenance
- Schema:
  - Well-defined JSON schema
  - Versioning (schema version field)
- Nested structures:
  - Preserve hierarchy (sections, lists)
  - Preserve relationships (footnotes, references)
- Binary data:
  - Encode images as base64
  - Encode fonts/attachments as base64

**JSON Schema Example:**
```json
{
  "schema_version": "1.0",
  "metadata": {
    "title": "Document Title",
    "author": "John Doe",
    "page_count": 10
  },
  "pages": [
    {
      "page_number": 1,
      "width": 595,
      "height": 842,
      "elements": [
        {
          "type": "heading",
          "level": 1,
          "text": "Introduction",
          "bbox": {"x": 72, "y": 100, "width": 450, "height": 30}
        },
        {
          "type": "paragraph",
          "text": "This is the first paragraph...",
          "bbox": {"x": 72, "y": 140, "width": 450, "height": 100}
        }
      ]
    }
  ]
}
```

**Tests:**
- Serialization correctness
- Deserialization (round-trip)
- Schema validation
- Large document handling

---

### **DAY 104: DocTags Export**

**Goals:**
1. Custom format for downstream processing
2. Tag-based representation
3. Maintain provenance
4. Extensibility

**Deliverables:**

**File:** `mojo/export_doctags.mojo` (~600 lines)

**Features:**
- DocTags format:
  - Tag-based markup (similar to XML)
  - Custom tags for each element type
  - Attributes for metadata
- Element tags:
  - <heading level="1">...</heading>
  - <paragraph>...</paragraph>
  - <table>...</table>
  - <image src="..." />
  - <code language="python">...</code>
- Provenance:
  - Include source page, position
  - Include confidence scores
- Extensibility:
  - Custom attributes
  - Custom tags (for specialized elements)

**DocTags Example:**
```xml
<document>
  <metadata title="Document Title" author="John Doe" />
  <page number="1">
    <heading level="1" bbox="72,100,450,30">Introduction</heading>
    <paragraph bbox="72,140,450,100">This is the first paragraph...</paragraph>
    <table rows="2" cols="2" bbox="72,250,450,100">
      <row>
        <cell>Header 1</cell>
        <cell>Header 2</cell>
      </row>
      <row>
        <cell>Cell 1</cell>
        <cell>Cell 2</cell>
      </row>
    </table>
  </page>
</document>
```

**Tests:**
- DocTags generation
- Parsing (round-trip)
- Provenance preservation
- Custom attributes

---

### **DAY 105: Export Testing**

**Goals:**
1. Round-trip tests
2. Visual validation
3. Format-specific tests
4. Integration tests

**Deliverables:**

**File:** `mojo/tests/export_test.mojo` (~600 lines)

**Test Coverage:**
- Markdown export
- HTML export (W3C validation)
- JSON export (schema validation)
- DocTags export
- Round-trip tests (parse → export → parse → compare)
- Visual validation (HTML output in browser)

**Test Documents:**
- Various document types
- Complex structures
- Images and tables
- Code blocks

---

# PHASE 6: Advanced Features (Days 106-115)

## WEEK 22: Days 106-110 - Chunking System

### **DAY 106: Hierarchical Chunker**

**Goals:**
1. Respect document structure
2. Section-aware chunking
3. Heading-based splitting
4. Configurable chunk size

**Deliverables:**

**File:** `mojo/chunking/hierarchical.mojo` (~800 lines)

**Features:**
- Hierarchical chunking:
  - Respect section boundaries (never split sections)
  - Split at heading boundaries (H1, H2, H3)
  - Configurable max chunk size (characters or tokens)
- Section-aware:
  - Keep sections together (if possible)
  - Split large sections at sub-heading boundaries
- Heading context:
  - Include parent headings in chunk metadata
  - Example: "Chapter 1 / Section 1.2 / Subsection 1.2.3"
- Chunk metadata:
  - Chunk ID
  - Source page numbers
  - Heading context
  - Chunk size

**Hierarchical Chunking Algorithm:**
1. Traverse document hierarchy (depth-first)
2. For each section:
   a. If section size < max_chunk_size:
      - Add entire section to chunk
   b. Else:
      - Split at sub-heading boundaries
      - Recursively chunk sub-sections
3. Output chunks with metadata

**Tests:**
- Small documents (single chunk)
- Large documents (multiple chunks)
- Various chunk sizes
- Hierarchy preservation

---

### **DAY 107: Semantic Chunker**

**Goals:**
1. Sentence boundary detection
2. Paragraph-aware splitting
3. Maintain semantic coherence
4. Configurable overlap

**Deliverables:**

**File:** `mojo/chunking/semantic.mojo` (~700 lines)

**Features:**
- Semantic chunking:
  - Split at sentence boundaries (never mid-sentence)
  - Prefer paragraph boundaries
  - Maintain semantic coherence (related sentences together)
- Sentence boundary detection:
  - Period followed by space and capital letter
  - Abbreviation handling (Dr., Mr., etc.)
  - Multiple sentence-ending punctuation (. ! ?)
- Paragraph-aware:
  - Prefer to split at paragraph boundaries
  - Keep paragraphs together (if possible)
- Configurable overlap:
  - Overlap N sentences between chunks
  - Helps maintain context for downstream tasks (e.g., QA)

**Semantic Chunking Algorithm:**
1. Split document into sentences
2. Group sentences into chunks:
   a. Add sentences to current chunk
   b. If chunk size > max_size:
      - Finalize current chunk (at sentence boundary)
      - Start new chunk (with overlap)
3. Output chunks

**Tests:**
- Sentence boundary detection
- Paragraph boundaries
- Overlap handling
- Various chunk sizes

---

### **DAY 108: Token-Based Chunker**

**Goals:**
1. Tokenization
2. Token counting
3. Sliding window chunks
4. Overlap management

**Deliverables:**

**File:** `mojo/chunking/token_based.mojo` (~600 lines)

**Features:**
- Tokenization:
  - Simple tokenization (whitespace + punctuation)
  - Approximate BPE-like tokenization (for LLM compatibility)
  - Token counting
- Token counting:
  - Count tokens (not characters)
  - Configurable max tokens per chunk
- Sliding window:
  - Fixed-size chunks (e.g., 512 tokens)
  - Sliding window with overlap (e.g., 50 tokens)
- Overlap management:
  - Configurable overlap size
  - Helps with context preservation

**Token-Based Chunking Algorithm:**
1. Tokenize document
2. Split into fixed-size chunks:
   a. Take N tokens
   b. Output chunk
   c. Slide window by (N - overlap) tokens
   d. Repeat
3. Output chunks

**Tokenization:**
- Simple: Split on whitespace and punctuation
- Approximate BPE: Use common subword splits (for English)
- Character count heuristic: tokens ≈ characters / 4

**Tests:**
- Tokenization accuracy
- Chunk size consistency
- Overlap handling
- Edge cases (empty document, single token)

---

### **DAY 109: Hybrid Chunker**

**Goals:**
1. Combine semantic + structural + token-based
2. Optimized for RAG
3. Preserve citations and references
4. Chunk quality metrics

**Deliverables:**

**File:** `mojo/chunking/hybrid.mojo` (~1,000 lines)

**Features:**
- Hybrid chunking:
  - Respect document structure (headings, sections)
  - Maintain semantic coherence (sentences, paragraphs)
  - Enforce token limits (max tokens per chunk)
- Optimized for RAG (Retrieval-Augmented Generation):
  - Include heading context in metadata
  - Preserve citations (keep reference numbers with text)
  - Include chunk metadata (page, section, keywords)
- Chunk quality:
  - Avoid orphaned sentences (very short chunks)
  - Avoid mid-sentence splits
  - Balance chunk sizes (min/max constraints)
- Metadata enrichment:
  - Extract keywords from chunk
  - Generate chunk summary (optional, first sentence)
  - Heading hierarchy (breadcrumb)

**Hybrid Chunking Algorithm:**
1. Hierarchical split (at headings)
2. For each section:
   a. Semantic split (at paragraphs/sentences)
   b. For each chunk:
      - Count tokens
      - If > max_tokens:
        - Split further (at sentence boundary)
3. Enrich metadata
4. Output chunks

**Tests:**
- RAG-optimized chunks
- Citation preservation
- Metadata enrichment
- Various document types

---

### **DAY 110: Chunking Testing**

**Goals:**
1. Various document types
2. Chunk quality metrics
3. Performance benchmarks
4. Integration tests

**Deliverables:**

**File:** `mojo/tests/chunking_test.mojo` (~500 lines)

**Test Coverage:**
- All chunking strategies
- Various document types (academic, books, reports)
- Chunk quality metrics:
  - Average chunk size
  - Chunk size variance
  - Semantic coherence (manual review)
- Performance benchmarks

**Chunk Quality Metrics:**
- Average chunk size (should be close to target)
- Variance (should be low)
- Orphan rate (short chunks)
- Split quality (sentence boundaries vs mid-sentence)

---

## WEEK 23: Days 111-115 - Image & Media Processing

### **DAY 111: Image Classification**

**Goals:**
1. Rule-based classification
2. Simple CNN model
3. Confidence scoring
4. Chart/photo/diagram classification

**Deliverables:**

**File:** `mojo/image_classification.mojo` (~800 lines)

**Features:**
- Rule-based classification:
  - Aspect ratio (wide → chart, square → photo)
  - Position on page (top/bottom → header/footer image)
  - Color distribution (grayscale → technical drawing)
- CNN model (optional, simple):
  - Input: Image (e.g., 224x224 pixels)
  - Output: Class (chart, photo, diagram, table, etc.)
  - Pre-trained on document image dataset
- Confidence scoring:
  - Softmax output (probabilities per class)
  - Threshold for low-confidence (classify as "unknown")
- Class types:
  - Chart (bar, line, pie, scatter)
  - Photo (photographic image)
  - Diagram (flowchart, UML, etc.)
  - Table (table rendered as image)
  - Logo
  - Icon

**Rule-Based Heuristics:**
- Aspect ratio:
  - > 2:1 or < 1:2 → likely chart
  - ≈ 1:1 → likely photo or icon
- Color:
  - Grayscale → technical drawing or chart
  - High color variance → photo
- Position:
  - Top of page → likely logo or header
  - Bottom of page → likely footer

**Tests:**
- Various image types
- Classification accuracy
- Confidence calibration

---

### **DAY 112: Image Captioning (Basic)**

**Goals:**
1. Extract image context from surrounding text
2. Metadata extraction (alt text, captions)
3. Reference resolution
4. Caption generation (basic)

**Deliverables:**

**File:** `mojo/image_captioning.mojo` (~600 lines)

**Features:**
- Context extraction:
  - Extract text before/after image (window of N sentences)
  - Identify captions (e.g., "Figure 1: ...")
  - Identify references (e.g., "as shown in Figure 1")
- Metadata extraction:
  - Alt text (from HTML, PDF annotations)
  - Figure captions (from document structure)
  - Image labels (from OCR, if text in image)
- Reference resolution:
  - Link figure references to actual figures
  - Extract figure numbers (e.g., "Figure 1", "Fig. 2")
- Caption generation (basic):
  - Use existing caption (if available)
  - Generate from context (e.g., "Image showing [context]")
  - Use image classification (e.g., "Chart showing data")

**Caption Output:**
```mojo
struct ImageCaption:
    var caption: String
    var source: String  # "alt_text", "figure_caption", "generated"
    var confidence: Float32
```

**Tests:**
- Caption extraction
- Reference resolution
- Caption generation accuracy

---

### **DAY 113: Audio File Handling**

**Goals:**
1. Format detection
2. Metadata extraction
3. Duration calculation
4. Waveform analysis (basic)

**Deliverables:**

**File:** `mojo/audio_processing.mojo` (~500 lines)

**Features:**
- Format detection:
  - MP3, WAV, FLAC, OGG, AAC, M4A
  - Magic byte detection
- Metadata extraction:
  - ID3 tags (MP3):
    - Title, artist, album, year
    - Track number, genre
    - Cover art (embedded image)
  - Vorbis comments (FLAC, OGG):
    - Similar to ID3 tags
  - RIFF INFO (WAV)
- Duration calculation:
  - Parse audio header (sample rate, bit rate)
  - Calculate: duration = file_size / bit_rate
- Waveform analysis (basic):
  - Sample audio at regular intervals
  - Compute amplitude (for visualization)
  - No actual audio decoding (just metadata)

**Audio Metadata:**
```mojo
struct AudioMetadata:
    var format: String  # "MP3", "WAV", etc.
    var duration: Float32  # seconds
    var sample_rate: Int  # Hz
    var bit_rate: Int  # kbps
    var title: String
    var artist: String
    var album: String
```

**Tests:**
- Various audio formats
- Metadata extraction
- Duration calculation
- Large audio files

---

### **DAY 114: WebVTT Parser**

**Goals:**
1. Full WebVTT spec
2. Timestamp parsing
3. Cue text formatting
4. Style and region support

**Deliverables:**

**File:** `mojo/webvtt_parser.mojo` (~700 lines)

**Features:**
- WebVTT format:
  - Signature (WEBVTT)
  - Cue blocks (timestamp → text)
  - Style blocks
  - Region blocks
- Timestamp parsing:
  - Format: hh:mm:ss.mmm (hours, minutes, seconds, milliseconds)
  - Start and end timestamps for each cue
- Cue text:
  - Plain text
  - Formatting tags (<b>, <i>, <u>, <c>, <v>, etc.)
  - Voice tags (<v Speaker Name>)
- Styles:
  - CSS-like styling for cues
  - Position, size, alignment
- Regions:
  - Define layout regions for cues

**WebVTT Example:**
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, welcome to the video.

00:00:02.500 --> 00:00:05.000
<v Speaker 1>This is the first speaker.</v>

00:00:05.000 --> 00:00:08.000
<v Speaker 2>And this is the second speaker.</v>
```

**WebVTT Output:**
```mojo
struct WebVTTCue:
    var start: Float32  # seconds
    var end: Float32
    var text: String
    var speaker: String  # if specified
```

**Tests:**
- Timestamp parsing
- Cue text extraction
- Formatting tags
- Multiple speakers

---

### **DAY 115: Media Testing**

**Goals:**
1. Image classification tests
2. Audio metadata tests
3. WebVTT parsing tests
4. Integration tests

**Deliverables:**

**File:** `mojo/tests/media_test.mojo` (~400 lines)

**Test Coverage:**
- Image classification (all types)
- Image captioning
- Audio metadata extraction
- WebVTT parsing
- Error handling (corrupt files)

**Test Files:**
- Various image types (charts, photos, diagrams)
- Various audio formats (MP3, WAV, FLAC)
- WebVTT files (different formats)

---

# PHASE 7: Service & CLI (Days 116-125)

## WEEK 24: Days 116-120 - DocumentConverter API

### **DAY 116-117: Main API (Mojo)**

**Goals:**
1. DocumentConverter class
2. convert() method
3. convert_all() batch method
4. Format auto-detection

**Deliverables:**

**File:** `mojo/converter.mojo` (~1,500 lines)

**Features:**
- DocumentConverter class:
  - Main API for document conversion
  - Configure pipelines, options
- Methods:
  - `convert(source)` - Convert single document
  - `convert_all(sources)` - Batch conversion
  - `convert_string(content, format)` - Convert from string
- Format auto-detection:
  - Magic bytes (PDF: %PDF, ZIP: PK, etc.)
  - File extension (.pdf, .docx, .html, etc.)
  - Content sniffing (if needed)
- Configuration options:
  - Output format (Markdown, HTML, JSON, DocTags)
  - OCR enabled/disabled
  - Layout analysis enabled/disabled
  - Chunking strategy
  - Custom pipeline

**DocumentConverter API:**
```mojo
struct DocumentConverter:
    fn convert(self, source: String) -> Result[ConversionResult, Error]:
        # 1. Detect format
        # 2. Select pipeline
        # 3. Execute pipeline
        # 4. Return result
        pass
    
    fn convert_all(self, sources: List[String]) -> List[Result[ConversionResult, Error]]:
        # Batch conversion (parallel)
        pass
    
    fn convert_string(self, content: String, format: InputFormat) -> Result[ConversionResult, Error]:
        # Convert from string (Markdown, HTML)
        pass
```

**ConversionResult:**
```mojo
struct ConversionResult:
    var document: DoclingDocument
    var metadata: Metadata
    var provenance: Provenance
    var status: ConversionStatus
    var errors: List[Error]
```

**Tests:**
- Single document conversion
- Batch conversion
- Format auto-detection
- Error handling

---

### **DAY 118: Streaming Support**

**Goals:**
1. Incremental document processing
2. Memory-efficient buffering
3. Progress callbacks
4. Cancellation support

**Deliverables:**

**File:** `mojo/streaming.mojo` (~800 lines)

**Features:**
- Streaming support:
  - Process large documents incrementally
  - Don't load entire document in memory
  - Stream pages one at a time
- Memory-efficient:
  - Fixed memory budget
  - Release processed pages
  - Buffering strategy
- Progress callbacks:
  - Callback function (page_done, progress_percent)
  - Real-time updates
- Cancellation:
  - Check cancellation flag
  - Stop processing gracefully
  - Return partial results

**Streaming API:**
```mojo
fn convert_streaming(
    self,
    source: String,
    progress_callback: fn(Int, Int) -> None  # (current_page, total_pages)
) -> Result[ConversionResult, Error]:
    # Stream pages, call callback after each
    pass
```

**Tests:**
- Large document streaming
- Memory usage validation
- Progress callback
- Cancellation

---

### **DAY 119: Concurrency Management**

**Goals:**
1. Thread pool for parallel conversion
2. Async document conversion
3. Rate limiting
4. Resource management

**Deliverables:**

**File:** `mojo/concurrency.mojo` (~1,000 lines)

**Features:**
- Thread pool:
  - Configurable number of threads
  - Queue documents for processing
  - Parallel page processing
- Async support (use Mojo SDK's async when ready):
  - `async fn convert_async()`
  - Await multiple conversions
  - Concurrent batch processing
- Rate limiting:
  - Limit conversions per second
  - Prevent resource exhaustion
  - Token bucket algorithm
- Resource management:
  - Memory limits (per conversion)
  - CPU limits (time per page)
  - Automatic cleanup

**Concurrency API:**
```mojo
fn convert_batch_parallel(
    self,
    sources: List[String],
    num_threads: Int
) -> List[Result[ConversionResult, Error]]:
    # Parallel batch conversion
    pass
```

**Tests:**
- Parallel conversion correctness
- Thread safety
- Rate limiting
- Resource limits

---

### **DAY 120: API Testing**

**Goals:**
1. Unit tests for all methods
2. Integration tests
3. Error handling tests
4. Performance tests

**Deliverables:**

**File:** `mojo/tests/converter_test.mojo` (~600 lines)

**Test Coverage:**
- convert() method
- convert_all() batch method
- Streaming conversion
- Format auto-detection
- Error handling (invalid files, corrupt files)
- Concurrency (parallel conversion)

**Performance Tests:**
- Single document speed
- Batch conversion speed
- Memory usage
- Parallel speedup

---

## WEEK 25: Days 121-125 - HTTP Service

### **DAY 121-122: Service Implementation (Shimmy Pattern)**

**Goals:**
1. Implement Service trait
2. REST API endpoints
3. Request routing
4. Response formatting

**Deliverables:**

**File:** `mojo/service.mojo` (~1,500 lines)

**Features:**
- Service trait implementation:
  - Use Mojo SDK's Service trait
  - Handle HTTP requests
  - Return HTTP responses
- REST API endpoints:
  - `POST /convert` - Single document (multipart/form-data)
    - Upload file, return converted document
  - `POST /convert/batch` - Multiple documents
    - Upload multiple files, return array of results
  - `POST /convert/url` - Convert from URL
    - Provide URL, fetch and convert
  - `GET /status/:id` - Async conversion status
    - Check status of background conversion
  - `GET /formats` - List supported formats
    - Return array of supported input formats
  - `GET /health` - Health check
    - Return service status
  - `GET /version` - API version
- Request routing:
  - Use Router from Mojo SDK
  - Path parameter extraction (e.g., /status/:id)
  - Query parameter parsing
- Response formatting:
  - JSON responses (default)
  - Binary responses (for converted files)
  - Error responses (RFC 7807 Problem Details)

**Service Implementation:**
```mojo
struct NExtractService(Service):
    fn handle_request(inout self, ctx: Context) -> Response:
        let route = ctx.path()
        
        if route == "/convert":
            return self.handle_convert(ctx)
        elif route == "/formats":
            return self.handle_formats(ctx)
        elif route == "/health":
            return self.handle_health(ctx)
        else:
            return Response(404, "Not Found")
    
    fn handle_convert(inout self, ctx: Context) -> Response:
        # Parse multipart form data
        # Extract file
        # Convert document
        # Return result (JSON or binary)
        pass
```

**Tests:**
- All endpoints
- Request parsing
- Response formatting
- Error handling

---

### **DAY 123: Request/Response Handling**

**Goals:**
1. Multipart file upload parsing
2. JSON request/response
3. Error response standards
4. Content negotiation

**Deliverables:**

**File:** `mojo/http_handlers.mojo` (~1,000 lines)

**Features:**
- Multipart file upload:
  - Parse multipart/form-data
  - Extract file(s) from request
  - Handle multiple files (batch)
- JSON request/response:
  - Parse JSON request body
  - Generate JSON response
  - Use Mojo SDK's JSON utilities
- Error responses:
  - RFC 7807 Problem Details format
  - HTTP status codes (400, 404, 500, etc.)
  - Detailed error messages
  - Error stack trace (in debug mode)
- Content negotiation:
  - Accept header parsing
  - Support multiple output formats (JSON, XML, etc.)
  - Content-Type header (response)
- CORS headers:
  - Access-Control-Allow-Origin
  - Access-Control-Allow-Methods
  - Preflight request handling

**Error Response Example:**
```json
{
  "type": "https://nextr.act/errors/invalid-format",
  "title": "Invalid Document Format",
  "status": 400,
  "detail": "The uploaded file is not a valid PDF document.",
  "instance": "/convert/req-12345"
}
```

**Tests:**
- Multipart parsing
- JSON encoding/decoding
- Error response format
- CORS headers

---

### **DAY 124: Middleware Stack**

**Goals:**
1. LoggingMiddleware
2. RequestIdMiddleware
3. RateLimitMiddleware
4. RecoveryMiddleware

**Deliverables:**

**File:** `mojo/middleware.mojo` (~1,000 lines)

**Features:**
- LoggingMiddleware:
  - Log all requests (method, path, status, duration)
  - Structured logging (JSON format)
  - Log level (info, warn, error)
- RequestIdMiddleware:
  - Generate unique request ID (UUID)
  - Add to response headers (X-Request-Id)
  - Include in logs (for tracing)
- RateLimitMiddleware:
  - Token bucket algorithm
  - Limit requests per IP
  - Configurable rate (e.g., 10 req/sec)
  - Return 429 Too Many Requests (if exceeded)
- AuthMiddleware (optional):
  - Bearer token authentication
  - API key authentication
  - JWT validation
- RecoveryMiddleware:
  - Catch panics/errors
  - Return 500 Internal Server Error
  - Log error details
- CompressionMiddleware:
  - Gzip response compression
  - Accept-Encoding header check

**Middleware Chain:**
```mojo
let router = Router()
router.add_middleware(LoggingMiddleware())
router.add_middleware(RequestIdMiddleware())
router.add_middleware(RateLimitMiddleware(rate=10.0))  # 10 req/sec
router.add_middleware(RecoveryMiddleware())

router.post("/convert", handle_convert)
```

**Tests:**
- Logging output
- Request ID generation
- Rate limiting behavior
- Panic recovery

---

### **DAY 125: Service Testing**

**Goals:**
1. HTTP integration tests
2. Load testing
3. Stress testing
4. Error scenarios

**Deliverables:**

**File:** `mojo/tests/service_test.mojo` (~800 lines)

**Test Coverage:**
- All endpoints (unit tests)
- File upload (multipart)
- Batch conversion
- Error handling (invalid files, timeouts)
- Rate limiting
- CORS
- Authentication (if enabled)

**Load Testing:**
- Concurrent requests (100+ simultaneous)
- Large file handling (100+ MB)
- Throughput (requests per second)
- Latency (p50, p95, p99)

**Stress Testing:**
- Resource limits (memory, CPU)
- File size limits (reject files > max size)
- Rate limit enforcement

---

# PHASE 8: CLI & Tooling (Days 126-135)

## WEEK 26: Days 126-130 - CLI Tool

### **DAY 126-127: CLI Implementation**

**Goals:**
1. Command-line interface
2. Convert command
3. Batch command
4. Info and validate commands

**Deliverables:**

**File:** `mojo/cli.mojo` (~1,500 lines)

**Features:**
- CLI tool: `nextract` (or `nhyperbook`)
- Commands:
  - `nextract convert <input> [output]`
    - Convert single document
    - Input: file path, URL, or stdin (-)
    - Output: file path or stdout (-)
    - Options: --from <format>, --to <format>
  - `nextract batch <directory> [output-dir]`
    - Batch convert all files in directory
    - Recursive: --recursive
    - Parallel: --jobs <N>
    - Skip errors: --skip-errors
  - `nextract info <file>`
    - Show document metadata
    - Format, page count, word count
    - Author, title, date
  - `nextract validate <file>`
    - Check if file is parseable
    - Report format issues
    - Return exit code (0 = valid, 1 = invalid)
- Options:
  - `--from <format>` - Input format (auto-detect if not specified)
  - `--to <format>` - Output format (markdown, html, json, doctags)
  - `--ocr` - Enable OCR for scanned documents
  - `--no-ocr` - Disable OCR
  - `--layout` - Enable layout analysis
  - `--no-layout` - Disable layout analysis
  - `--config <file>` - Configuration file
  - `--verbose` / `-v` - Verbose output
  - `--quiet` / `-q` - Quiet mode
  - `--help` / `-h` - Show help

**CLI Examples:**
```bash
# Convert PDF to Markdown
nextract convert document.pdf document.md

# Convert from URL
nextract convert https://example.com/doc.pdf output.md

# Convert from stdin
cat document.html | nextract convert --from html --to markdown - output.md

# Batch convert
nextract batch ./documents/ ./output/ --recursive --jobs 4

# Show document info
nextract info document.pdf

# Validate document
nextract validate document.docx
```

**Tests:**
- All commands
- Various options
- Error handling (invalid files, missing arguments)
- Exit codes

---

### **DAY 128: Advanced CLI Features**

**Goals:**
1. Extract command
2. Compare command
3. Merge command
4. Additional utilities

**Deliverables:**

**File:** `mojo/cli_advanced.mojo` (~1,000 lines)

**Features:**
- Advanced commands:
  - `nextract extract <file> --images|--text|--tables`
    - Extract specific components
    - `--images <dir>` - Extract all images to directory
    - `--text <file>` - Extract text to file
    - `--tables <file>` - Extract tables to CSV/JSON
  - `nextract compare <file1> <file2>`
    - Compare two documents (text diff)
    - Show added, deleted, modified lines
    - Output: unified diff or side-by-side
  - `nextract merge <file1> <file2> [output]`
    - Merge multiple documents
    - Concatenate pages
    - Preserve formatting
  - `nextract chunk <file> [output-dir]`
    - Split document into chunks
    - Options: --strategy <hierarchical|semantic|token>
    - Options: --chunk-size <N>, --overlap <N>
- Utility commands:
  - `nextract formats`
    - List all supported formats (input and output)
  - `nextract version`
    - Show version information
  - `nextract doctor`
    - Check system dependencies
    - Verify installation

**CLI Examples:**
```bash
# Extract images
nextract extract document.pdf --images ./images/

# Compare documents
nextract compare doc1.pdf doc2.pdf > diff.txt

# Merge documents
nextract merge doc1.pdf doc2.pdf merged.pdf

# Chunk document
nextract chunk large-doc.pdf ./chunks/ --strategy semantic --chunk-size 512
```

**Tests:**
- Extract commands
- Compare functionality
- Merge functionality
- Chunking

---

### **DAY 129: Configuration System**

**Goals:**
1. TOML config file
2. CLI flags override config
3. Environment variables
4. Per-project config

**Deliverables:**

**File:** `mojo/config.mojo` (~800 lines)

**Features:**
- Configuration file (`~/.nextract/config.toml` or `.nextract.toml`):
  - Global config (in home directory)
  - Per-project config (in project directory)
- Configuration options:
  - Default output format
  - OCR enabled/disabled
  - Layout analysis enabled/disabled
  - Chunking strategy
  - API endpoint (for service)
  - Logging level
- CLI flags override:
  - CLI flags > project config > global config > defaults
- Environment variables:
  - `NEXTRACT_CONFIG` - Path to config file
  - `NEXTRACT_OCR` - Enable OCR (true/false)
  - `NEXTRACT_LAYOUT` - Enable layout analysis (true/false)
  - `NEXTRACT_LOG_LEVEL` - Logging level (debug, info, warn, error)

**Config File Example:**
```toml
[default]
output_format = "markdown"
ocr_enabled = true
layout_enabled = true
log_level = "info"

[chunking]
strategy = "semantic"
chunk_size = 512
overlap = 50

[api]
endpoint = "http://localhost:8080"
timeout = 30
```

**Tests:**
- Config file parsing
- CLI override
- Environment variables
- Priority order

---

### **DAY 130: Progress & Logging**

**Goals:**
1. Progress bars
2. Verbose mode
3. Quiet mode
4. Structured logging

**Deliverables:**

**File:** `mojo/logging.mojo` (~600 lines)

**Features:**
- Progress bars:
  - Show progress for long operations
  - Page-by-page progress (for PDF)
  - Batch progress (N/M files converted)
  - Estimated time remaining
- Verbose mode (`-v`, `-vv`, `-vvv`):
  - `-v` - Basic info (file names, progress)
  - `-vv` - Detailed info (stages, timings)
  - `-vvv` - Debug info (internal details)
- Quiet mode (`-q`):
  - No output (except errors)
  - Useful for scripting
- Logging:
  - Structured logging (JSON format, optional)
  - Log to file (`--log-file <file>`)
  - Log levels (debug, info, warn, error)
  - Colorized output (terminal)

**Progress Bar Example:**
```
Converting document.pdf...
[████████████████████████████████] 100% (25/25 pages) - 2.5s
```

**Logging Example:**
```
[INFO] Converting document.pdf
[INFO] Format detected: PDF
[INFO] Processing page 1/25
[INFO] OCR completed (confidence: 0.95)
[INFO] Layout analysis completed
[INFO] Conversion complete (2.5s)
```

**Tests:**
- Progress bar rendering
- Verbose output
- Quiet mode
- Log file output

---

## WEEK 27: Days 131-135 - Testing Infrastructure

### **DAY 131: Unit Test Suite**

**Goals:**
1. Test all components
2. Edge case handling
3. Error scenarios
4. Code coverage (80%+)

**Deliverables:**

**File:** `zig/tests/unit_tests.zig` and `mojo/tests/unit_tests.mojo` (~2,000 lines)

**Test Coverage:**
- All parsers (CSV, Markdown, XML, HTML, PDF, Office)
- Image processing (filters, transformations, thresholding)
- OCR engine (line detection, character segmentation, recognition)
- ML inference (tensor operations, layers, models)
- Pipeline framework (all pipelines)
- Document assembly
- Reading order
- Chunking
- Export formats
- API (DocumentConverter)
- Service (HTTP endpoints)
- CLI (all commands)

**Test Organization:**
- Zig tests: `zig/tests/` (for Zig components)
- Mojo tests: `mojo/tests/` (for Mojo components)
- Integration tests: `tests/integration/`

**Run Tests:**
```bash
# Zig tests
zig build test

# Mojo tests
mojo test mojo/tests/

# All tests
./test_runner.sh
```

**Tests:**
- 1,000+ unit tests
- Coverage report
- Test execution time

---

### **DAY 132: Integration Test Suite**

**Goals:**
1. End-to-end conversion tests
2. All format combinations
3. Real-world documents
4. Performance regression tests

**Deliverables:**

**File:** `tests/integration/integration_tests.mojo` (~1,500 lines)

**Test Coverage:**
- End-to-end conversion:
  - PDF → Markdown, HTML, JSON
  - DOCX → Markdown, HTML, JSON
  - XLSX → Markdown, HTML, JSON
  - PPTX → Markdown, HTML, JSON
  - HTML → Markdown, JSON
  - Markdown → HTML, JSON
- Real-world documents:
  - Academic papers (arXiv PDFs)
  - Government forms (PDFs)
  - Books (EPUBs converted to HTML)
  - Spreadsheets (financial reports)
  - Presentations (corporate decks)
- Performance regression:
  - Benchmark conversion speed
  - Detect regressions (> 10% slower)
  - Memory usage benchmarks

**Integration Test Framework:**
```mojo
fn test_pdf_to_markdown():
    let converter = DocumentConverter()
    let result = converter.convert("tests/fixtures/paper.pdf")
    
    assert(result.is_ok())
    let doc = result.unwrap()
    
    # Validate output
    assert(doc.page_count() == 10)
    assert(doc.word_count() > 5000)
    
    # Export to Markdown
    let markdown = doc.export_to_markdown()
    assert(markdown.contains("# Introduction"))
```

**Tests:**
- 200+ integration tests
- Real-world documents
- Performance benchmarks

---

### **DAY 133: Fuzzing Infrastructure**

**Goals:**
1. Fuzz all parsers
2. Malformed input handling
3. Crash detection
4. Memory leak detection

**Deliverables:**

**File:** `tests/fuzz/` (fuzzing infrastructure)

**Fuzzing Targets:**
- PDF parser (malformed PDFs)
- XML parser (malformed XML)
- HTML parser (malformed HTML)
- ZIP parser (malformed archives)
- DEFLATE decompressor (malformed streams)
- PNG decoder (malformed PNGs)
- JPEG decoder (malformed JPEGs)
- OCR engine (adversarial images)
- ML inference (invalid tensors)

**Fuzzing Strategy:**
- LibFuzzer integration (via Zig)
- Continuous fuzzing (run in CI/CD)
- Crash corpus collection
- Automatic bug reporting

**Fuzzing Setup:**
```bash
# Build fuzz targets
zig build-exe tests/fuzz/fuzz_pdf.zig -fsanitize=address

# Run fuzzer
./fuzz_pdf corpus/ -max_total_time=3600  # 1 hour
```

**Tests:**
- Fuzz targets (10+)
- Crash detection
- Memory leak detection
- Continuous fuzzing in CI

---

### **DAY 134: Performance Benchmarks**

**Goals:**
1. Conversion speed benchmarks
2. Memory usage profiling
3. Comparison with original Docling
4. Optimization opportunities

**Deliverables:**

**File:** `tests/benchmarks/benchmarks.mojo` (~800 lines)

**Benchmarks:**
- Conversion speed:
  - PDF → Markdown (pages per second)
  - DOCX → Markdown
  - XLSX → JSON
  - Large files (100+ pages)
- Memory usage:
  - Peak memory per conversion
  - Memory per page
  - Memory leaks (long-running tests)
- Component benchmarks:
  - PDF parsing (pages per second)
  - OCR (pages per second)
  - Layout analysis (pages per second)
  - Image decoding (images per second)
  - ML inference (inferences per second)
- Comparison with Docling:
  - Speed (faster/slower)
  - Memory (less/more)
  - Accuracy (character/word accuracy)

**Benchmark Framework:**
```mojo
fn benchmark_pdf_conversion():
    let converter = DocumentConverter()
    
    let start = time.now()
    for i in range(100):
        let result = converter.convert("tests/fixtures/paper.pdf")
    let end = time.now()
    
    let duration = end - start
    let pages_per_sec = 100 * 10 / duration  # 10 pages per doc
    
    print("PDF conversion: {pages_per_sec} pages/sec")
```

**Tests:**
- 50+ benchmarks
- Automated benchmarking in CI
- Performance dashboard

---

### **DAY 135: Test Reporting**

**Goals:**
1. Test coverage reports
2. Performance dashboard
3. Continuous benchmarking
4. Quality metrics

**Deliverables:**

**File:** `tests/reporting/` (test reporting tools)

**Features:**
- Test coverage:
  - Line coverage report (HTML)
  - Branch coverage
  - Function coverage
  - Target: 80%+ coverage
- Performance dashboard:
  - Track conversion speed over time
  - Track memory usage over time
  - Detect regressions
  - Visualizations (charts)
- Continuous benchmarking:
  - Run benchmarks on every commit
  - Store results in database
  - Alert on regressions
- Quality metrics:
  - Test pass rate (target: 100%)
  - Fuzzing coverage (branches discovered)
  - Bug count (open/closed)
  - Code quality score

**Test Report Example:**
```
=== Test Summary ===
Total tests: 1,234
Passed: 1,234
Failed: 0
Coverage: 87.3%

=== Performance ===
PDF conversion: 12.5 pages/sec (+5% vs baseline)
Memory usage: 45 MB per document (-10% vs baseline)

=== Fuzzing ===
Total iterations: 1,000,000
Crashes found: 0
Coverage: 85.2% of branches
```

**Tests:**
- Coverage reporting
- Performance tracking
- Dashboard generation

---

# PHASE 9: Finalization & Release (Days 136-150)

## WEEK 28: Days 136-140 - Security & Quality

### **DAY 136: Security Audit**

**Goals:**
1. Input validation
2. Buffer overflow protection
3. Integer overflow checks
4. Memory safety verification

**Deliverables:**

**File:** `docs/security_audit.md`

**Security Review:**
- Input validation:
  - File size limits (reject files > max size)
  - File type validation (magic bytes)
  - Path traversal protection (sanitize file paths)
  - URL validation (for remote fetching)
- Buffer overflow protection:
  - Bounds checking (all array accesses)
  - Safe string operations
  - Stack overflow protection
- Integer overflow checks:
  - Check arithmetic operations (especially in parsers)
  - Use saturating arithmetic where appropriate
- Denial-of-service protection:
  - Resource limits (memory, CPU time)
  - Zip bomb detection (compressed size vs uncompressed)
  - XML bomb detection (billion laughs, exponential entity expansion)
  - Regular expression DoS (ReDoS) prevention
- Memory safety:
  - Zig safety features (bounds checking, no use-after-free)
  - Mojo SDK's borrow checker
  - ASAN/MSAN testing (address sanitizer, memory sanitizer)

**Security Tests:**
- Malicious file handling (zip bombs, XML bombs)
- Path traversal attempts
- Integer overflow tests
- Memory safety tests

---

### **DAY 137: Fuzz Testing Results**

**Goals:**
1. Review fuzzer findings
2. Fix discovered issues
3. Harden parsers
4. Re-run fuzz campaigns

**Deliverables:**

**File:** `docs/fuzzing_results.md`

**Fuzzing Review:**
- Analyze crash reports
- Reproduce crashes
- Fix root causes:
  - Out-of-bounds accesses
  - Null pointer dereferences
  - Assertion failures
  - Memory leaks
- Harden parsers:
  - Add input validation
  - Add bounds checks
  - Add error handling
- Re-run fuzzing:
  - Verify fixes
  - Increase coverage
  - Run longer campaigns (24+ hours)

**Fuzzing Stats:**
- Iterations run: 10,000,000+
- Crashes found: 0 (after fixes)
- Coverage: 90%+ of branches
- New test cases generated: 1,000+

---

### **DAY 138: Static Analysis**

**Goals:**
1. Run static analyzers
2. Address warnings
3. Code quality improvements
4. Linting

**Deliverables:**

**File:** `docs/static_analysis.md`

**Static Analysis Tools:**
- Zig built-in analyzer:
  - `zig build --release=safe` (runtime safety checks)
  - Detect undefined behavior
- Custom analyzers:
  - Unused variable detection
  - Dead code detection
  - Unreachable code
- Linters:
  - `zig fmt` (code formatting)
  - Custom linting rules
- Code complexity:
  - Cyclomatic complexity (keep < 15 per function)
  - Function length (keep < 100 lines)

**Issues to Address:**
- Unused imports/variables
- Unreachable code
- Code duplication
- Complex functions (refactor if needed)
- Magic numbers (replace with constants)

---

### **DAY 139: Code Review**

**Goals:**
1. Peer review of critical components
2. Architecture review
3. API design review
4. Documentation review

**Deliverables:**

**File:** `docs/code_review.md`

**Code Review:**
- Critical components:
  - PDF parser (most complex)
  - OCR engine (accuracy-critical)
  - ML inference engine (performance-critical)
  - FFI layer (correctness-critical)
- Architecture review:
  - Module boundaries
  - Separation of concerns
  - Extensibility (can add new formats?)
- API design:
  - Ergonomics (easy to use?)
  - Safety (hard to misuse?)
  - Completeness (covers all use cases?)
- Documentation:
  - Inline comments (explain complex logic)
  - API documentation (complete?)
  - Examples (sufficient?)

**Review Checklist:**
- Code clarity
- Error handling
- Performance considerations
- Memory management
- Test coverage
- Documentation quality

---

### **DAY 140: Security Documentation**

**Goals:**
1. Security considerations
2. Threat model
3. Best practices for deployment
4. Security advisories

**Deliverables:**

**File:** `docs/SECURITY.md`

**Security Documentation:**
- Threat model:
  - Malicious input files (PDFs, Office docs)
  - Denial-of-service attacks
  - Memory exhaustion
  - Path traversal
- Mitigations:
  - Input validation (file sizes, types)
  - Resource limits (memory, CPU time)
  - Sandboxing (optional, for untrusted files)
- Best practices:
  - Run service with limited privileges
  - Set resource limits (ulimit, cgroups)
  - Use HTTPS for remote file fetching
  - Validate all user inputs
- Reporting vulnerabilities:
  - Security contact email
  - Responsible disclosure policy
  - Bug bounty (optional)

---

## WEEK 29: Days 141-145 - Documentation

### **DAY 141: API Documentation**

**Goals:**
1. Mojo API reference
2. Zig FFI interface documentation
3. Code examples for all APIs
4. Tutorials

**Deliverables:**

**File:** `docs/api/` (API documentation)

**API Documentation:**
- Mojo API:
  - DocumentConverter class
  - Pipeline classes
  - Export functions
  - Chunking functions
  - Configuration options
- Zig FFI:
  - Exported functions
  - Data structures
  - Memory management
  - Callback mechanisms
- Code examples:
  - Basic conversion
  - Batch conversion
  - Custom pipeline
  - Streaming conversion
  - HTTP service usage
- Tutorials:
  - Getting started
  - Converting PDFs
  - Extracting tables
  - OCR usage
  - Chunking for RAG

---

### **DAY 142: Architecture Documentation**

**Goals:**
1. System architecture diagrams
2. Component interactions
3. Data flow diagrams
4. Performance characteristics

**Deliverables:**

**File:** `docs/ARCHITECTURE.md`

**Architecture Documentation:**
- System overview:
  - High-level architecture diagram
  - Component layers (Zig, Mojo, Service)
- Components:
  - Parsers (PDF, Office, text)
  - OCR engine
  - ML inference
  - Pipeline framework
  - API layer
  - Service layer
- Data flow:
  - Input file → Parser → Pipeline → DoclingDocument → Export
  - Diagrams (flow charts)
- Performance:
  - Bottlenecks (OCR, ML inference)
  - Optimization strategies (caching, parallelism)
  - Scalability (horizontal scaling for service)

---

### **DAY 143: User Guide**

**Goals:**
1. Getting started
2. Installation instructions
3. CLI usage guide
4. HTTP API guide

**Deliverables:**

**File:** `docs/USER_GUIDE.md`

**User Guide:**
- Getting started:
  - Installation (binaries, build from source)
  - Quick start (convert first document)
- Installation:
  - Prerequisites (Zig, Mojo SDK)
  - Download binaries (macOS, Linux, Windows)
  - Build from source
  - Docker image
- CLI usage:
  - All commands
  - Options and flags
  - Examples
  - Configuration
- HTTP API:
  - API endpoints
  - Request/response format
  - Authentication (if enabled)
  - Rate limits
  - Examples (curl, Python requests)

---

### **DAY 144: Developer Guide**

**Goals:**
1. Contributing guidelines
2. Building from source
3. Testing procedures
4. Adding new format support

**Deliverables:**

**File:** `docs/CONTRIBUTING.md`

**Developer Guide:**
- Contributing:
  - Code of conduct
  - How to report bugs
  - How to submit pull requests
  - Code style guide
- Building from source:
  - Clone repository
  - Install dependencies (Zig, Mojo SDK)
  - Build commands
  - Run tests
- Testing:
  - Unit tests
  - Integration tests
  - Fuzzing
  - Benchmarks
- Adding new formats:
  - Create parser (in `zig/parsers/`)
  - Export functions for FFI
  - Create Mojo wrapper
  - Add to DocumentConverter
  - Add tests

---

### **DAY 145: Migration Guide**

**Goals:**
1. From Python Docling to nExtract
2. API compatibility notes
3. Performance improvements
4. Feature differences

**Deliverables:**

**File:** `docs/MIGRATION.md`

**Migration Guide:**
- API changes:
  - DocumentConverter (similar API)
  - convert() vs convert_all()
  - Configuration options
- Code examples:
  - Python Docling code
  - Equivalent nExtract code (Mojo)
- Performance:
  - Speed improvements (2-5x faster)
  - Memory usage (50% less)
- Features:
  - Feature parity table (what's supported)
  - New features (streaming, async)
  - Missing features (if any)
- Workflow changes:
  - CLI differences
  - Service deployment
  - Configuration

---

## WEEK 30: Days 146-150 - Release Preparation

### **DAY 146: Version 1.0.0 Preparation**

**Goals:**
1. Finalize version number
2. CHANGELOG.md
3. LICENSE file
4. Release notes

**Deliverables:**

**Files:**
- `CHANGELOG.md`
- `LICENSE` (MIT)
- `RELEASE_NOTES_v1.0.0.md`

**Version 1.0.0:**
- Semantic versioning (1.0.0)
- Major release (first stable release)

**CHANGELOG.md:**
```markdown
# Changelog

## [1.0.0] - 2026-06-XX

### Added
- Full PDF parsing support
- DOCX, XLSX, PPTX parsing
- CSV, Markdown, HTML parsing
- OCR engine (pure Zig/Mojo)
- ML-based layout analysis
- Image codecs (PNG, JPEG)
- Export formats (Markdown, HTML, JSON, DocTags)
- Chunking system (hierarchical, semantic, token-based, hybrid)
- HTTP service (REST API)
- CLI tool (convert, batch, info, validate, extract)
- Complete test suite (1,200+ tests)

### Performance
- 2-5x faster than Python Docling
- 50% less memory usage
- Zero external dependencies

### Documentation
- Complete API documentation
- User guide
- Developer guide
- Architecture documentation
```

**Release Notes:**
- Summary of features
- Installation instructions
- Breaking changes (N/A for 1.0.0)
- Known issues
- Roadmap for future releases

---

### **DAY 147: Build Artifacts**

**Goals:**
1. macOS binaries (x86_64, arm64)
2. Linux binaries (x86_64, arm64)
3. Windows binaries (x86_64)
4. Docker image

**Deliverables:**

**Build Artifacts:**
- `nextract-v1.0.0-macos-x86_64.tar.gz`
- `nextract-v1.0.0-macos-arm64.tar.gz`
- `nextract-v1.0.0-linux-x86_64.tar.gz`
- `nextract-v1.0.0-linux-arm64.tar.gz`
- `nextract-v1.0.0-windows-x86_64.zip`
- Docker image: `nextract:1.0.0`

**Build Process:**
```bash
# macOS (arm64)
zig build -Dtarget=aarch64-macos -Doptimize=ReleaseFast

# Linux (x86_64)
zig build -Dtarget=x86_64-linux -Doptimize=ReleaseFast

# Windows (x86_64)
zig build -Dtarget=x86_64-windows -Doptimize=ReleaseFast
```

**Docker Image:**
```dockerfile
FROM alpine:latest
COPY nextract /usr/local/bin/
COPY mojo-sdk /opt/mojo-sdk/
ENV PATH="/usr/local/bin:$PATH"
ENTRYPOINT ["nextract"]
```

---

### **DAY 148: Deployment Scripts**

**Goals:**
1. Systemd service file
2. Docker Compose setup
3. Kubernetes manifests
4. Ansible playbook

**Deliverables:**

**Systemd Service:**
```ini
[Unit]
Description=nExtract Document Extraction Service
After=network.target

[Service]
Type=simple
User=nextract
ExecStart=/usr/local/bin/nextract serve --port 8080
Restart=always

[Install]
WantedBy=multi-user.target
```

**Docker Compose:**
```yaml
version: '3.8'
services:
  nextract:
    image: nextract:1.0.0
    ports:
      - "8080:8080"
    volumes:
      - ./documents:/data
    environment:
      - NEXTRACT_LOG_LEVEL=info
```

**Kubernetes Deployment:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nextract
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nextract
  template:
    metadata:
      labels:
        app: nextract
    spec:
      containers:
      - name: nextract
        image: nextract:1.0.0
        ports:
        - containerPort: 8080
```

---

### **DAY 149: Release Testing**

**Goals:**
1. Fresh install testing
2. Cross-platform testing
3. Performance validation
4. Documentation review

**Deliverables:**

**Testing Checklist:**
- Install on fresh systems (macOS, Linux, Windows)
- Run all CLI commands
- Start HTTP service
- Convert test documents (all formats)
- Run test suite
- Verify documentation links
- Performance benchmarks (vs baseline)

**Cross-Platform Testing:**
- macOS (x86_64, arm64)
- Linux (x86_64, arm64, Ubuntu, Debian, Fedora)
- Windows (x86_64, Windows 10, Windows 11)

**Performance Validation:**
- Conversion speed (within 10% of baseline)
- Memory usage (within 10% of baseline)
- No regressions

---

### **DAY 150: Release Day**

**Goals:**
1. Tag v1.0.0
2. Publish binaries
3. Docker registry push
4. Announce release

**Deliverables:**

**Release Process:**
```bash
# Tag release
git tag -a v1.0.0 -m "Release version 1.0.0"
git push origin v1.0.0

# Publish binaries (GitHub Releases)
gh release create v1.0.0 \
  --title "nExtract v1.0.0" \
  --notes-file RELEASE_NOTES_v1.0.0.md \
  nextract-*.tar.gz

# Push Docker image
docker tag nextract:1.0.0 dockerhub/nextract:1.0.0
docker push dockerhub/nextract:1.0.0
```

**Announcement:**
- Blog post
- Social media (Twitter, LinkedIn)
- Reddit (r/programming, r/machinelearning)
- Hacker News
- Email to users

**Release Checklist:**
- ✅ Version tagged
- ✅ Binaries uploaded
- ✅ Docker image pushed
- ✅ Documentation live
- ✅ Announcement published
- ✅ Celebration! 🎉

---

# Post-Implementation: Days 151-155

## DAY 151-152: Feature Parity Validation

**Goals:**
1. Comprehensive test against original Docling
2. Feature checklist
3. Performance comparison
4. Quality assessment

**Deliverables:**

**File:** `docs/PARITY_VALIDATION.md`

**Validation Process:**
- Test nExtract against Docling:
  - Same input documents
  - Compare outputs (text accuracy, structure preservation)
  - Measure performance (speed, memory)
- Feature checklist:
  - All formats supported? (PDF, DOCX, XLSX, PPTX, HTML, MD, CSV)
  - OCR support? (for scanned PDFs)
  - Layout analysis? (tables, figures, reading order)
  - Export formats? (Markdown, HTML, JSON)
  - All documented features implemented?
- Performance comparison:
  - Conversion speed (nExtract vs Docling)
  - Memory usage
  - Accuracy (character/word accuracy for OCR)
- Quality assessment:
  - Text extraction quality
  - Structure preservation
  - Formatting preservation

**Validation Results:**
- Feature parity: 100% (all features implemented)
- Performance: 2-5x faster than Docling
- Memory: 50% less than Docling
- Accuracy: >95% (OCR character accuracy)

**Gaps (if any):**
- Document for future releases
- Prioritize for next version

---

## DAY 153: Backup & Archive

**Goals:**
1. Create archive of the retired Docling vendor package
2. Store in safe location
3. Document differences
4. Create migration playbook

**Deliverables:**

**Backup Process:**
```bash
# Create archive
tar -czf docling_backup_$(date +%Y%m%d).tar.gz /path/to/docling

# Store in multiple locations
# 1. External drive
# 2. Cloud storage (S3, Google Drive)
# 3. Company backup system
```

**Documentation:**
- `docs/DOCLING_DIFFERENCES.md` - Differences between nExtract and Docling
- `docs/MIGRATION_PLAYBOOK.md` - Step-by-step migration guide
- `docs/ROLLBACK_PLAN.md` - How to rollback if needed

**Archive Contents:**
- Original Docling source code
- Documentation
- Test results
- Performance benchmarks
- Migration notes

---

## DAY 154: Delete Original

**Goals:**
1. Remove the Docling vendor package
2. Update all references
3. Update documentation
4. Update configuration files

**Deliverables:**

**Deletion Process:**
```bash
# Remove Docling
rm -rf /path/to/docling

# Commit removal
git add -A
git commit -m "Remove legacy Docling vendor package (replaced by nExtract)"
git push
```

**Update References:**
- Import statements (Python → Mojo)
- Configuration files (update paths)
- Documentation (update links)
- CI/CD pipelines (use nExtract)
- Deployment scripts (use nExtract binaries)

**Verification:**
- No broken links
- No missing imports
- All tests pass
- CI/CD pipeline passes

---

## DAY 155: Final Validation & Deployment

**Goals:**
1. Full system test in production environment
2. Monitor initial usage
3. Performance metrics
4. User feedback collection

**Deliverables:**

**Production Deployment:**
- Deploy nExtract to production servers
- Configure monitoring (logs, metrics)
- Set up alerting (errors, performance degradation)
- Gradual rollout (canary deployment, 10% → 50% → 100%)

**Validation:**
- Run production workload
- Monitor performance (latency, throughput)
- Check error rates
- Verify output quality

**Monitoring:**
- Request rate
- Response time (p50, p95, p99)
- Error rate
- Memory usage
- CPU usage
- Conversion success rate

**User Feedback:**
- Collect feedback from initial users
- Identify issues
- Prioritize fixes/improvements
- Plan next release

**Celebrate! 🎉**
- Project complete!
- 155 days of implementation
- Zero external dependencies
- Full feature parity
- Production-ready system

---

# Summary

## Project Statistics (Final)

| Metric | Value |
|--------|-------|
| Total days | 155 |
| Total code | ~55,000 lines |
| Zig code | ~25,000 lines |
| Mojo code | ~10,000 lines |
| Test code | ~15,000 lines |
| Documentation | ~5,000 lines |
| Total tests | 1,200+ |
| Code coverage | 85%+ |
| Fuzzing iterations | 10M+ |
| Supported formats | 10+ |
| Export formats | 4 |

## Key Achievements

✅ **Zero External Dependencies** - All components built from scratch  
✅ **Full Feature Parity** - Matches Docling functionality  
✅ **2-5x Performance** - Faster than Python Docling  
✅ **50% Less Memory** - More efficient resource usage  
✅ **Production Quality** - Thoroughly tested, documented, secure  
✅ **Pure Zig/Mojo** - Leverages Mojo SDK infrastructure  
✅ **HTTP Service** - REST API with Shimmy pattern  
✅ **CLI Tool** - Complete command-line interface  
✅ **Extensible** - Easy to add new formats  

## Next Steps (Future Releases)

**v1.1.0** (Optional, future enhancements):
- Additional language support (OCR)
- EPUB format support
- Improved chart understanding
- Formula parsing improvements
- PDF generation (export to PDF)

**v2.0.0** (Major enhancements):
- Real-time collaborative editing
- Cloud-based processing
- API authentication & user management
- Advanced analytics dashboard
- Custom ML model training

---

**Document Status:** Complete  
**Last Updated:** January 17, 2026  
**Prepared by:** AI Assistant  
**Project:** nExtract (Docling Replacement)
