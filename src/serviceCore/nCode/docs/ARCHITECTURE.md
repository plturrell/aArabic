# nCode Architecture

This document provides a deep technical dive into nCode's architecture, data model, and implementation details.

## Table of Contents

- [System Overview](#system-overview)
- [SCIP Protocol](#scip-protocol)
- [Component Architecture](#component-architecture)
- [Data Flow](#data-flow)
- [Database Integration Strategy](#database-integration-strategy)
- [Performance Considerations](#performance-considerations)
- [Security Model](#security-model)

---

## System Overview

nCode is a code intelligence platform that bridges language-specific indexers with universal query interfaces and database backends. The architecture follows a three-tier model:

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIENT LAYER (Tier 1)                        │
│  HTTP API │ Python Client │ CLI Tools │ Web UI │ IDE Extensions │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PROCESSING LAYER (Tier 2)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │ HTTP Server  │  │ SCIP Parser  │  │  Query Engine      │  │
│  │  (Zig)       │  │  (Zig)       │  │  (Zig + Mojo)      │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STORAGE LAYER (Tier 3)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐  │
│  │   Qdrant     │  │  Memgraph    │  │     Marquez        │  │
│  │  (Vector DB) │  │  (Graph DB)  │  │  (Lineage DB)      │  │
│  └──────────────┘  └──────────────┘  └─────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Design Principles

1. **Language Agnostic**: Accept SCIP from any indexer, regardless of source language
2. **Fast by Default**: Zero-copy parsing, efficient data structures, compiled languages
3. **Scalable Storage**: Offload complex queries to specialized databases
4. **Standard Protocols**: SCIP (code), HTTP (API), OpenLineage (lineage), Cypher (graph)
5. **Composable**: Each component can be used independently

---

## SCIP Protocol

SCIP (Source Code Intelligence Protocol) is a protobuf-based format for representing code intelligence data. nCode implements the full SCIP specification.

### Core Data Model

```protobuf
message Index {
  Metadata metadata = 1;
  repeated Document documents = 2;
  repeated SymbolInformation external_symbols = 3;
}

message Document {
  string relative_path = 1;
  string language = 4;
  repeated Occurrence occurrences = 2;
  repeated SymbolInformation symbols = 3;
  string text = 5;  // Optional: full source text
}

message SymbolInformation {
  string symbol = 1;                      // Unique symbol ID
  repeated string documentation = 3;       // Docstrings/comments
  repeated Relationship relationships = 4; // Type, implementations
  SymbolKind kind = 5;                    // Function, Class, etc.
  string display_name = 6;                // Human-readable name
  string enclosing_symbol = 8;            // Parent symbol
}

message Occurrence {
  repeated int32 range = 1;     // [start_line, start_char, end_line, end_char]
  string symbol = 2;            // Symbol ID
  int32 symbol_roles = 3;       // Definition, Reference, etc.
  SyntaxKind syntax_kind = 5;   // Identifier, Keyword, etc.
}

message Relationship {
  string symbol = 1;            // Related symbol ID
  bool is_reference = 2;
  bool is_implementation = 3;
  bool is_type_definition = 4;
  bool is_definition = 5;
}
```

### Symbol Naming Convention

SCIP uses a hierarchical naming scheme:

```
<scheme> <package> <descriptor>* <symbol>
```

Examples:
- `scip-typescript npm @types/node v16.0.0 fs.readFile().`
- `scip-python my_package/__init__.py MyClass#my_method().`
- `scip-go github.com/user/repo pkg/server Server#Start().`

### Symbol Kinds

nCode recognizes 26 symbol kinds:

| Code | Kind | Description |
|------|------|-------------|
| 1 | UnspecifiedKind | Unknown type |
| 3 | Macro | Preprocessor macro |
| 5 | Type | Type/class definition |
| 6 | Parameter | Function parameter |
| 7 | Variable | Variable declaration |
| 8 | Property | Class property/field |
| 9 | Enum | Enumeration |
| 10 | EnumMember | Enum value |
| 11 | Function | Function/method |
| 12 | Method | Class method |
| 13 | Constructor | Constructor |
| 14 | Interface | Interface definition |
| 19 | Namespace | Namespace/module |
| 21 | TypeParameter | Generic type param |
| 22 | Trait | Trait definition |

---

## Component Architecture

### 1. HTTP Server (`server/main.zig`)

**Technology**: Zig 0.15.2, std.http.Server

**Responsibilities**:
- Accept HTTP requests on port 18003
- Route requests to appropriate handlers
- Parse JSON request bodies
- Serialize JSON responses
- Maintain in-memory SCIP index

**Key Data Structures**:
```zig
const ScipIndex = struct {
    loaded: bool,
    path: ?[]const u8,
    documents: std.ArrayList(Document),
    symbols: std.StringHashMap(SymbolInfo),
    allocator: mem.Allocator,
};
```

**Endpoints** (see [API.md](API.md) for details):
- `GET /health` - Health check
- `POST /v1/index/load` - Load SCIP file
- `POST /v1/definition` - Find definition
- `POST /v1/references` - Find references
- `POST /v1/hover` - Get hover info
- `POST /v1/symbols` - List symbols
- `POST /v1/document-symbols` - Document outline

### 2. SCIP Parser (`scip_reader.zig`)

**Technology**: Zig 0.15.2, manual protobuf parsing

**Responsibilities**:
- Parse SCIP protobuf format
- Extract metadata, documents, symbols, occurrences
- Build symbol lookup tables
- Handle malformed/corrupt indexes gracefully

**Key Functions**:
```zig
pub fn loadScipIndex(path: []const u8, allocator: Allocator) !ScipIndex
pub fn findDefinition(index: *ScipIndex, file: []const u8, line: i32, char: i32) ?Symbol
pub fn findReferences(index: *ScipIndex, symbol: []const u8) []Occurrence
pub fn getHover(index: *ScipIndex, file: []const u8, line: i32, char: i32) ?HoverInfo
```

**Protobuf Parsing Strategy**:
- Wire format: varint, fixed32, fixed64, length-delimited
- Zero-copy wherever possible (slice into mmap'd buffer)
- Lazy parsing: only parse requested documents
- Memory pooling for repeated allocations

### 3. SCIP Writer (`zig_scip_writer.zig`)

**Technology**: Zig 0.15.2, protobuf serialization

**Responsibilities**:
- Generate SCIP indexes from source code analysis
- Write protobuf messages efficiently
- Support streaming writes for large codebases

**C ABI Exports** (for Mojo/Python interop):
```zig
pub export fn scip_init(path: [*:0]const u8) c_int
pub export fn scip_write_metadata(version: c_int, tool_name: [*:0]const u8, ...) c_int
pub export fn scip_begin_document(language: [*:0]const u8, path: [*:0]const u8) c_int
pub export fn scip_add_occurrence(start_line: i32, start_char: i32, ...) c_int
pub export fn scip_add_symbol_info(symbol: [*:0]const u8, doc: [*:0]const u8, kind: c_int) c_int
pub export fn scip_close() c_int
```

### 4. Tree-Sitter Indexer (`treesitter_indexer.zig`)

**Technology**: Zig 0.15.2, pattern matching

**Responsibilities**:
- Index data languages without full parsers
- Extract symbols via regex and heuristics
- Generate SCIP indexes for JSON, XML, YAML, SQL, etc.

**Symbol Extraction Strategies**:

| Language | Strategy | Symbols Extracted |
|----------|----------|-------------------|
| JSON | Key detection | Property keys |
| YAML | Key detection | YAML keys, section headers |
| SQL | Keyword parsing | Tables, columns, views, indexes, procedures |
| XML/HTML | Tag parsing | Elements, attributes |
| CSS | Selector parsing | Classes, IDs, properties, at-rules |
| GraphQL | Type parsing | Types, queries, mutations, enums |
| TOML | Section parsing | Tables, array tables, keys |

**CLI Interface**:
```bash
ncode-treesitter index --language <lang> [--output <path>] <input>
```

### 5. Database Loaders (`loaders/`)

**Technology**: Python 3.9+, async/await

**Architecture**:
```
loaders/
├── __init__.py           # Package init
├── scip_parser.py        # Pure Python SCIP parser
├── qdrant_loader.py      # Vector DB integration
├── memgraph_loader.py    # Graph DB integration
└── marquez_loader.py     # Lineage tracking
```

Each loader implements:
- `connect()` - Establish DB connection
- `load_scip_index(scip_path)` - Load and transform SCIP data
- `query(...)` - Database-specific query methods

---

## Data Flow

### Indexing Flow

```
┌─────────────┐
│ Source Code │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│ Language Indexer    │ (scip-typescript, scip-python, etc.)
│ - Parse source      │
│ - Extract symbols   │
│ - Generate SCIP     │
└──────┬──────────────┘
       │
       ▼ index.scip (protobuf)
┌─────────────────────┐
│ nCode Server        │
│ POST /v1/index/load │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ SCIP Parser         │
│ - Parse protobuf    │
│ - Build symbol map  │
│ - Store in memory   │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ In-Memory Index     │ (ScipIndex struct)
└─────────────────────┘
```

### Query Flow

```
┌─────────────┐
│ HTTP Client │
└──────┬──────┘
       │ POST /v1/definition
       ▼
┌─────────────────────┐
│ HTTP Server         │
│ - Parse JSON        │
│ - Validate request  │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ Query Handler       │
│ - findDefinition()  │
│ - Lookup symbol     │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐
│ In-Memory Index     │
│ - Symbol hash map   │
│ - Occurrence list   │
└──────┬──────────────┘
       │
       ▼ JSON response
┌─────────────────────┐
│ HTTP Client         │
└─────────────────────┘
```

### Database Export Flow

```
┌──────────────┐
│ index.scip   │
└──────┬───────┘
       │
       ▼
┌─────────────────────────┐
│ load_to_databases.py    │
│ - Parse SCIP (Python)   │
│ - Extract symbols       │
└──────┬──────────────────┘
       │
       ├──────────────────────┬──────────────────────┐
       │                      │                      │
       ▼                      ▼                      ▼
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Qdrant       │      │ Memgraph     │      │ Marquez      │
│ - Embed text │      │ - Create     │      │ - Track      │
│ - Upsert     │      │   nodes      │      │   lineage    │
│   vectors    │      │ - Create     │      │ - Emit       │
│              │      │   edges      │      │   events     │
└──────────────┘      └──────────────┘      └──────────────┘
```

---

## Database Integration Strategy

### Qdrant (Vector Database)

**Use Case**: Semantic code search ("find functions that parse JSON")

**Data Model**:
- Collection: `code_symbols`
- Vector size: 384 (sentence-transformers/all-MiniLM-L6-v2)
- Distance metric: Cosine similarity

**Payload**:
```json
{
  "symbol": "scip-typescript npm my-pkg index.ts MyClass#",
  "display_name": "MyClass",
  "kind": 5,
  "kind_name": "Type",
  "documentation": "A utility class for...",
  "file_path": "src/index.ts",
  "language": "typescript",
  "enclosing_symbol": null,
  "project_root": "/path/to/project",
  "indexer": "scip-typescript"
}
```

**Query Flow**:
1. Embed query text using sentence-transformers
2. Search Qdrant with embedding
3. Apply filters (language, kind)
4. Return top-k results with scores

### Memgraph (Graph Database)

**Use Case**: Code relationship analysis ("show call graph", "find all implementations")

**Schema**:
```cypher
// Nodes
(:Symbol {symbol, display_name, kind, kind_name, documentation})
(:Document {path, language})

// Relationships
(:Symbol)-[:DEFINED_IN]->(:Document)
(:Symbol)-[:REFERENCES]->(:Symbol)
(:Symbol)-[:IMPLEMENTS]->(:Symbol)
(:Symbol)-[:TYPE_DEFINITION]->(:Symbol)
(:Symbol)-[:ENCLOSES]->(:Symbol)
```

**Example Queries**:
```cypher
// Find all implementations of an interface
MATCH (impl:Symbol)-[:IMPLEMENTS]->(iface:Symbol {symbol: $symbol})
RETURN impl

// Get call graph (transitive)
MATCH path = (caller:Symbol {symbol: $symbol})-[:REFERENCES*1..5]->(callee:Symbol)
WHERE callee.kind IN [5, 8, 11]  // Method, Constructor, Function
RETURN path

// Find all symbols in a file
MATCH (s:Symbol)-[:DEFINED_IN]->(d:Document {path: $file_path})
RETURN s
```

### Marquez (Data Lineage)

**Use Case**: Track indexing runs, source file provenance

**Data Model** (OpenLineage):
- Namespace: Project name or repository URL
- Job: `scip-index-{project}`
- Run: UUID per indexing run
- Input Datasets: Source files
- Output Datasets: SCIP index file

**Event Structure**:
```json
{
  "eventType": "COMPLETE",
  "eventTime": "2026-01-17T19:00:00Z",
  "run": {
    "runId": "uuid-here",
    "facets": {
      "nominalTime": {"nominalStartTime": "..."}
    }
  },
  "job": {
    "namespace": "my-project",
    "name": "scip-index-my-project",
    "facets": {
      "sourceCodeLocation": {"url": "/path/to/project"},
      "documentation": {"description": "SCIP indexing job"}
    }
  },
  "inputs": [
    {"namespace": "my-project", "name": "src/main.ts"},
    {"namespace": "my-project", "name": "src/utils.ts"}
  ],
  "outputs": [
    {"namespace": "my-project", "name": "index.scip"}
  ]
}
```

---

## Performance Considerations

### Memory Management

**In-Memory Index**:
- Average: ~500 bytes per symbol
- 100K symbols ≈ 50MB
- 1M symbols ≈ 500MB
- Use arena allocator for batch operations

**Optimization Strategies**:
1. Lazy document parsing (parse on demand)
2. Symbol interning (reuse common strings)
3. Compact occurrence representation (packed int32 array)
4. Memory pooling for temporary allocations

### Parsing Performance

**Protobuf Parsing**:
- Zero-copy slicing where possible
- varint decoding: ~10ns per int
- Length-delimited: O(1) to skip
- Full index parse: ~100ms for 10K symbols

**Bottlenecks**:
- String allocations (use string interning)
- Hash map insertions (use FNV hash)
- JSON serialization (use streaming)

### Query Performance

| Operation | Complexity | Typical Time |
|-----------|-----------|--------------|
| Find definition | O(1) hash lookup | <1ms |
| Find references | O(n) symbol scan | <10ms |
| Get hover | O(1) hash lookup | <1ms |
| List symbols | O(m) document scan | <5ms |
| Document symbols | O(m) occurrence scan | <3ms |

**Caching Strategy**:
- Cache parsed documents (LRU, max 1000)
- Cache symbol lookups (indefinite, cleared on reload)
- No query result caching (fast enough without)

### Database Export Performance

**Qdrant**:
- Embedding generation: ~50ms per symbol
- Batch upsert: 100 symbols per batch
- 10K symbols: ~5 minutes

**Memgraph**:
- Cypher execution: ~1ms per query
- Batch import: 1000 symbols per transaction
- 10K symbols: ~30 seconds

**Marquez**:
- Event emission: ~10ms per event
- 2 events per run (START, COMPLETE)
- Negligible overhead

---

## Security Model

### Input Validation

**SCIP Files**:
- Maximum file size: 1GB
- Protobuf field limits enforced
- String length limits (symbol: 1KB, doc: 10KB)
- UTF-8 validation on all strings

**HTTP Requests**:
- JSON schema validation
- Path traversal prevention
- File path canonicalization
- Integer overflow checks

### Access Control

**Current State**: No authentication/authorization
- Suitable for internal/development use
- Server binds to localhost only by default

**Production Recommendations**:
1. Add API key authentication
2. Implement rate limiting
3. Use TLS for encryption
4. Add audit logging
5. Consider OAuth2 integration

### Database Security

**Qdrant**: Use API keys, configure network policies
**Memgraph**: Use Bolt encryption, configure user auth
**Marquez**: Use API authentication, restrict network access

---

## Deployment Architecture

### Development
```
┌────────────────────┐
│ Developer Machine  │
│  nCode Server      │
│  :18003            │
└────────────────────┘
```

### Production (Docker Compose)
```
┌────────────────────────────────────────────────┐
│              Docker Network                    │
│                                                │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  nCode   │  │  Qdrant  │  │ Memgraph │   │
│  │  :18003  │  │  :6333   │  │  :7687   │   │
│  └──────────┘  └──────────┘  └──────────┘   │
│                                                │
│  ┌──────────┐                                 │
│  │ Marquez  │                                 │
│  │  :5000   │                                 │
│  └──────────┘                                 │
└────────────────────────────────────────────────┘
```

### Kubernetes
```yaml
apiVersion: v1
kind: Service
metadata:
  name: ncode
spec:
  selector:
    app: ncode
  ports:
  - port: 18003
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ncode
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: ncode
        image: ncode:latest
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Future Enhancements

### Performance
- [ ] Incremental index updates (delta parsing)
- [ ] Distributed index (shard by project)
- [ ] GPU-accelerated embedding generation
- [ ] gRPC API (faster than HTTP/JSON)

### Features
- [ ] Real-time indexing (watch file system)
- [ ] Cross-project symbol resolution
- [ ] Semantic diff (compare index versions)
- [ ] IDE protocol support (LSP/DAP)

### Scalability
- [ ] Horizontal scaling (stateless server)
- [ ] Index sharding (by project/language)
- [ ] Query result streaming
- [ ] Distributed tracing (OpenTelemetry)

---

**Last Updated**: 2026-01-17  
**Version**: 1.0
