# Qdrant Vector Database Integration

## Overview

This document describes the integration of Qdrant vector database with the Arabic invoice processing system. Qdrant works alongside Memgraph to provide a comprehensive dual-database architecture for AI-powered workflow orchestration.

## Architecture

### Dual Database Design

```
┌─────────────────┐    ┌─────────────────┐
│   Memgraph      │    │    Qdrant       │
│  Graph Database │    │ Vector Database │
├─────────────────┤    ├─────────────────┤
│ • Workflows     │    │ • Embeddings    │
│ • Dependencies  │    │ • Similarity    │
│ • Execution     │    │ • Semantic      │
│ • Analytics     │    │ • Recommendations│
└─────────────────┘    └─────────────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌─────────────────┐
         │ Hybrid          │
         │ Orchestration   │
         │ Adapter         │
         └─────────────────┘
```

### Key Components

1. **QdrantAdapter** - Core vector database operations
2. **HybridOrchestrationAdapter** - Combines graph + vector processing
3. **Vector Collections** - Organized by data type
4. **API Routes** - RESTful endpoints for vector operations

## Collections

### Default Collections

| Collection | Vector Size | Distance | Purpose |
|------------|-------------|----------|---------|
| `workflows` | 768 | Cosine | Workflow similarity and recommendations |
| `documents` | 1536 | Cosine | Document semantic search |
| `invoices` | 768 | Cosine | Arabic invoice similarity matching |
| `tools` | 384 | Cosine | Intelligent tool orchestration |
| `a2ui_components` | 512 | Cosine | UI component generation |

### Vector Types

```python
class VectorType(Enum):
    WORKFLOW_EMBEDDING = "workflow_embedding"
    DOCUMENT_EMBEDDING = "document_embedding" 
    INVOICE_EMBEDDING = "invoice_embedding"
    SEMANTIC_SEARCH = "semantic_search"
    TOOL_EMBEDDING = "tool_embedding"
    A2UI_COMPONENT = "a2ui_component"
```

## Setup and Installation

### 1. Start Qdrant Service

```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant

# Or using the vendored version
cd vendor/qdrant
docker-compose up -d
```

### 2. Initialize Collections

```bash
# Run the initialization script
python backend/scripts/initialize_qdrant.py
```

### 3. Verify Installation

```bash
# Check Qdrant health
curl http://localhost:6333/

# List collections
curl http://localhost:6333/collections
```

## API Endpoints

### Core Vector Operations

```http
# Health check
GET /api/qdrant/health

# List collections
GET /api/qdrant/collections

# Search vectors
POST /api/qdrant/search
{
  "query_vector": [0.1, 0.2, ...],
  "collection_name": "workflows",
  "limit": 10,
  "score_threshold": 0.7
}

# Upsert vectors
POST /api/qdrant/upsert
{
  "collection_name": "workflows",
  "points": [
    {
      "id": "workflow_001",
      "vector": [0.1, 0.2, ...],
      "payload": {"name": "Sample Workflow"}
    }
  ]
}
```

### AI-Specific Operations

```http
# Store workflow embedding
POST /api/qdrant/workflows/store
{
  "workflow_id": "wf_001",
  "workflow_name": "Arabic Invoice Processing",
  "embedding": [0.1, 0.2, ...],
  "metadata": {"type": "invoice_processing"}
}

# Find similar workflows
POST /api/qdrant/workflows/search
{
  "query_embedding": [0.1, 0.2, ...],
  "limit": 5,
  "score_threshold": 0.7
}

# Store invoice embedding
POST /api/qdrant/invoices/store
{
  "invoice_id": "inv_001",
  "invoice_text": "فاتورة رقم ١٢٣٤٥",
  "embedding": [0.1, 0.2, ...],
  "extracted_data": {"total": 1500.00}
}

# Get tool recommendations
POST /api/qdrant/tools/search
{
  "task_embedding": [0.1, 0.2, ...],
  "limit": 5,
  "tool_category": "ocr"
}
```

### Hybrid Orchestration

```http
# Execute hybrid workflow
POST /api/hybrid/execute
{
  "workflow_data": {
    "id": "wf_001",
    "name": "Arabic Invoice Processing",
    "tools": [...]
  },
  "context": {"user_id": "user123"}
}

# Get intelligent tool recommendations
POST /api/hybrid/tools/recommend
{
  "task_description": "Process Arabic invoice images",
  "context": {"domain": "finance"}
}

# Optimize workflow
POST /api/hybrid/optimize
{
  "workflow_data": {...}
}

# Get workflow health score
GET /api/hybrid/workflows/{workflow_id}/health
```

## Usage Examples

### Storing Workflow Embeddings

```python
from backend.adapters.qdrant import get_qdrant_adapter

# Get adapter
qdrant = await get_qdrant_adapter()

# Store workflow embedding
await qdrant.store_workflow_embedding(
    workflow_id="arabic_invoice_wf_001",
    workflow_name="Arabic Invoice Processing Workflow",
    embedding=workflow_embedding,  # 768-dimensional vector
    metadata={
        "type": "invoice_processing",
        "language": "arabic",
        "complexity": "medium",
        "success_rate": 0.95
    }
)
```

### Finding Similar Workflows

```python
# Find similar workflows
similar_workflows = await qdrant.find_similar_workflows(
    query_embedding=query_embedding,
    limit=5,
    score_threshold=0.7,
    workflow_type="invoice_processing"
)

for workflow in similar_workflows:
    print(f"Workflow: {workflow.payload['workflow_name']}")
    print(f"Similarity: {workflow.score:.3f}")
```

### Hybrid Workflow Execution

```python
from backend.adapters.hybrid_orchestration import get_hybrid_orchestration_adapter

# Get hybrid adapter
hybrid = await get_hybrid_orchestration_adapter()

# Execute workflow with both graph and vector processing
result = await hybrid.execute_hybrid_workflow(
    workflow_data=workflow_data,
    context={"user_id": "user123"}
)

print(f"Execution time: {result.execution_time:.2f}s")
print(f"Graph analytics: {result.graph_analytics}")
print(f"Vector recommendations: {result.vector_recommendations}")
print(f"Similar workflows: {len(result.similar_workflows)}")
```

## Integration with Memgraph

### Data Synchronization

The system automatically synchronizes data between Qdrant and Memgraph:

```python
# Sync workflow data between databases
await qdrant.sync_with_memgraph(
    memgraph_adapter=memgraph_adapter,
    workflow_id="wf_001",
    sync_embeddings=True
)
```

### Dual Database Queries

```python
# Get comprehensive workflow analytics
analytics = await hybrid.get_workflow_analytics(workflow_id)

# Includes data from both databases:
# - Graph structure from Memgraph
# - Vector similarities from Qdrant
# - Combined recommendations
```

## Performance Considerations

### Vector Dimensions

- **Workflows**: 768d (balance between accuracy and performance)
- **Documents**: 1536d (high accuracy for semantic search)
- **Invoices**: 768d (optimized for Arabic text)
- **Tools**: 384d (lightweight for fast recommendations)
- **A2UI Components**: 512d (UI-specific embeddings)

### Batch Operations

```python
# Batch upsert for better performance
points = [VectorPoint(...) for _ in range(100)]
await qdrant.upsert_points("workflows", points, batch_size=50)
```

### Indexing

Qdrant uses HNSW (Hierarchical Navigable Small World) indexing for fast similarity search:

- **Build time**: Optimized for write-heavy workloads
- **Search time**: Sub-millisecond for most queries
- **Memory usage**: Configurable based on accuracy requirements

## Monitoring and Maintenance

### Health Monitoring

```python
# Check Qdrant health
health = await qdrant.health_check()

# Get collection statistics
info = await qdrant.get_collection_info("workflows")
print(f"Vectors: {info.vectors_count}")
print(f"Indexed: {info.indexed_vectors_count}")
```

### Analytics

```python
# Get comprehensive analytics
summary = await qdrant.get_analytics_summary()
print(f"Total collections: {summary['total_collections']}")
print(f"Total vectors: {summary['total_vectors']}")
```

## Troubleshooting

### Common Issues

1. **Connection Failed**
   ```bash
   # Check if Qdrant is running
   curl http://localhost:6333/
   
   # Check Docker container
   docker ps | grep qdrant
   ```

2. **Collection Not Found**
   ```bash
   # Reinitialize collections
   python backend/scripts/initialize_qdrant.py
   ```

3. **Vector Dimension Mismatch**
   ```python
   # Ensure embedding dimensions match collection config
   assert len(embedding) == 768  # For workflows
   ```

4. **Performance Issues**
   ```python
   # Use batch operations for large datasets
   await qdrant.upsert_points(collection, points, batch_size=100)
   ```

### Logs

```bash
# Check application logs
tail -f logs/qdrant_adapter.log

# Check Qdrant service logs
docker logs qdrant_container
```

## Future Enhancements

1. **Advanced Embeddings**: Integration with transformer models
2. **Multi-modal Vectors**: Support for image + text embeddings
3. **Federated Search**: Cross-collection similarity search
4. **Real-time Updates**: WebSocket-based vector updates
5. **Clustering**: Automatic workflow pattern discovery
