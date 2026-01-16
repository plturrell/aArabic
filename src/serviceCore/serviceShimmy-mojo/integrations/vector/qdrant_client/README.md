# Qdrant Client for Shimmy

High-performance Qdrant vector database client with Zig backend and Mojo domain logic layer.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Shimmy Core & Recursive LLM            │
│            (Business Logic & Orchestration)             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│               Domain Logic Layer (Mojo)                 │
│        src/serviceCore/serviceShimmy-mojo/clients/      │
│                                                          │
│  • qdrant/qdrant_domain.mojo - Vector operations & RAG  │
│  • dragonfly/dragonfly_cache.mojo - Cache operations    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              FFI Wrapper Layer (Mojo)                   │
│        src/serviceCore/serviceShimmy-mojo/clients/      │
│                                                          │
│  • qdrant_client.mojo - Low-level FFI bindings          │
│  • dragonfly_cache.mojo - Cache FFI bindings            │
└────────────────────┬────────────────────────────────────┘
                     │ C ABI
                     ▼
┌─────────────────────────────────────────────────────────┐
│           Native Clients (Zig - Compiled .dylib)        │
│        src/serviceCore/serviceShimmy-mojo/clients/      │
│                                                          │
│  • qdrant_client.zig - HTTP REST client                 │
│  • dragonfly_client.zig - RESP protocol client          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              External Services                          │
│                                                          │
│  • Qdrant (localhost:6333)                              │
│  • DragonflyDB (localhost:6379)                         │
└─────────────────────────────────────────────────────────┘
```

## Adapters

## Files in This Directory

### qdrant_client.zig (500 lines)
**Low-level HTTP REST client for Qdrant**
- Pure Zig implementation
- JSON serialization/deserialization
- Vector search, upsert, delete operations
- C ABI exports for FFI
- Target: 5-10x faster than Python

### qdrant_client.mojo (150 lines)
**FFI wrapper layer**
- Low-level bindings to Zig client via C ABI
- Memory-safe wrappers
- Basic operations: search, results handling
- Used by domain logic layer

### qdrant_domain.mojo (600 lines)

**Purpose:** High-level domain operations for Qdrant vector database

**Features:**
- ✅ Workflow embedding storage and similarity search
- ✅ Invoice similarity and duplicate detection  
- ✅ Tool discovery and capability matching
- ✅ Integration with recursive LLM
- ✅ Memgraph synchronization (stub)

**Performance Target:** 5-10x faster than Python `qdrant.py`

**Usage Example:**

```mojo
from clients.qdrant.qdrant_domain import (
    QdrantDomain, 
    WorkflowEmbedding,
    InvoiceEmbedding,
    ToolEmbedding
)

# Initialize domain layer
let domain = QdrantDomain(host="127.0.0.1", port=6333)

# Store a workflow embedding
var embedding = List[Float32]()
for i in range(768):
    embedding.append(0.1)

let workflow = WorkflowEmbedding(
    workflow_id="wf_001",
    name="AP Invoice Processing",
    description="Automated invoice validation and routing",
    embedding=embedding,
    status="active"
)
domain.store_workflow_embedding(workflow)

# Find similar workflows
let similar = domain.find_similar_workflows(
    query_embedding=embedding,
    limit=5,
    min_score=0.7
)

# Match invoice to workflow
let matched_workflow = domain.match_invoice_to_workflow(
    invoice_embedding=embedding
)

# Find relevant tools for a task
let tools = domain.find_relevant_tools(
    task_description="Extract text from PDF",
    task_embedding=embedding,
    limit=5
)

# Detect duplicate invoices
let invoice = InvoiceEmbedding(
    invoice_id="inv_001",
    vendor_name="ACME Corp",
    invoice_number="INV-2026-001",
    embedding=embedding
)
let duplicates = domain.find_duplicate_invoices(
    invoice=invoice,
    similarity_threshold=0.95
)
```

## Domain Types

### WorkflowEmbedding
Represents a workflow with vector embedding and metadata for similarity search and recommendations.

**Fields:**
- `workflow_id: String` - Unique identifier
- `name: String` - Workflow name
- `description: String` - Human-readable description
- `embedding: List[Float32]` - Vector representation (typically 768-dim)
- `status: String` - active/completed/failed
- `created_at: String` - ISO timestamp
- `tags: String` - Comma-separated tags

### InvoiceEmbedding
Represents an invoice with vector embedding for duplicate detection and workflow matching.

**Fields:**
- `invoice_id: String` - Unique identifier
- `vendor_name: String` - Vendor name
- `invoice_number: String` - Invoice number
- `embedding: List[Float32]` - Vector representation
- `amount: String` - Invoice amount
- `currency: String` - Currency code (USD, EUR, etc.)
- `invoice_date: String` - ISO date
- `status: String` - pending/processed/rejected

### ToolEmbedding
Represents a tool with vector embedding for semantic discovery and capability matching.

**Fields:**
- `tool_id: String` - Unique identifier
- `tool_name: String` - Tool name
- `description: String` - Tool description
- `embedding: List[Float32]` - Vector representation
- `capabilities: String` - Comma-separated capabilities
- `category: String` - Tool category
- `version: String` - Version string

## Collections

The domain layer manages four Qdrant collections:

1. **workflows** - Workflow embeddings for similarity search
2. **invoices** - Invoice embeddings for duplicate detection
3. **tools** - Tool embeddings for semantic discovery
4. **documents** - Document embeddings for RAG operations

## Integration Points

### Recursive LLM
- Workflow recommendations during reasoning
- Tool discovery for task execution
- Document retrieval for context

### Tool Orchestration
- Automatic tool selection via `find_relevant_tools()`
- Capability-based routing

### Workflow Orchestration
- Workflow matching via `match_invoice_to_workflow()`
- Similarity-based workflow suggestions

### Memgraph Integration
- Bi-directional sync between vector and graph databases
- Maintained via `sync_with_memgraph()` (to be implemented)

## Performance Characteristics

Based on the Zig client layer:

**Search Operations:**
- Python baseline: 50-100ms per search
- Zig+Mojo target: 10-20ms per search
- **Expected improvement: 5-10x faster**

**Batch Operations:**
- Efficient bulk upsert for migrations
- Parallel processing support (future)

**Memory:**
- Zero-copy where possible
- Efficient string handling
- Minimal allocations in hot paths

## TODOs

### Short-term (Week 2)
- [ ] Extend `qdrant_client.mojo` with upsert/delete operations
- [ ] Implement proper JSON parsing in helper methods
- [ ] Add integration tests
- [ ] Benchmark against Python baseline

### Medium-term (Week 3-4)
- [ ] Add filtered search support (by status, category, etc.)
- [ ] Implement batch operations
- [ ] Add connection pooling
- [ ] Implement actual Memgraph sync

### Long-term (Week 5+)
- [ ] Add async/await support for concurrent operations
- [ ] Implement caching layer (with DragonflyDB)
- [ ] Add retry logic and error recovery
- [ ] Metrics and monitoring integration

## Directory Structure

```
src/serviceCore/serviceShimmy-mojo/clients/qdrant/
├── build.zig                  # Zig build configuration
├── qdrant_client.zig          # Low-level HTTP client (500 lines)
├── qdrant_client.mojo         # FFI wrapper (150 lines)
├── qdrant_domain.mojo         # Domain logic layer (600 lines)
└── README.md                  # This file (includes usage examples)
```

## Related Clients

**DragonflyDB Cache:**
- `../dragonfly/dragonfly_client.zig` - RESP protocol client
- `../dragonfly/dragonfly_cache.mojo` - Cache domain logic
- `../dragonfly/README.md` - Cache client documentation

## Migration Status

### ✅ Completed (Week 1-2)
- [x] DragonflyDB client (Zig + Mojo)
- [x] Qdrant client (Zig + Mojo)
- [x] Qdrant domain logic (Mojo)

### ⏳ In Progress (Week 2)
- [ ] Client extensions (upsert/delete)
- [ ] Integration testing
- [ ] Performance benchmarking

### 📋 Planned (Week 3+)
- [ ] Tool orchestration domain logic
- [ ] Workflow orchestration domain logic
- [ ] Memgraph client (Zig + Mojo)
- [ ] Graph operations domain logic

## References

- [Mojo Migration Roadmap](../../../MOJO_MIGRATION_ROADMAP.md)
- [Adapters Migration Plan](../../../ADAPTERS_MIGRATION_PLAN.md)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
