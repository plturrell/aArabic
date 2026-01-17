# Graph Data Product Catalog Architecture

## 🎯 Overview

The Graph Data Product Catalog provides intelligent schema management for graph databases, enabling natural language to Cypher query translation with full context awareness.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Graph Data Product Catalog                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌─────────────────────┐     ┌──────────────────────┐          │
│  │  Manual Schema      │     │  Auto-Introspection  │          │
│  │  (JSON Config)      │     │  (Live Databases)    │          │
│  │                     │     │                       │          │
│  │  • Node types       │     │  • Neo4j CALL db.*   │          │
│  │  • Relationships    │     │  • Memgraph SHOW *   │          │
│  │  • Properties       │     │  • HANA metadata     │          │
│  │  • Descriptions     │     │  • Live properties   │          │
│  │  • Sample queries   │     │  • Constraints       │          │
│  └──────────┬──────────┘     └──────────┬───────────┘          │
│             │                           │                        │
│             └───────────┬───────────────┘                        │
│                         ↓                                        │
│                ┌────────────────────┐                           │
│                │  Schema Merger     │                           │
│                │  • Priority rules  │                           │
│                │  • Conflict resolve│                           │
│                │  • Versioning      │                           │
│                └─────────┬──────────┘                           │
│                          ↓                                       │
│                ┌────────────────────┐                           │
│                │  Unified Registry  │                           │
│                │  • O(1) lookup     │                           │
│                │  • Property index  │                           │
│                │  • Capability map  │                           │
│                └────────────────────┘                           │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

## 📦 Components

### 1. Schema Registry (`schema_registry.mojo`)

Core data structures for graph schema management:

```mojo
struct GraphSchema:
    """Complete graph schema with nodes, relationships, and metadata"""
    var node_types: Dict[String, NodeSchema]
    var relationship_types: Dict[String, RelationshipSchema]
    var property_index: Dict[String, List[String]]
    var metadata: SchemaMetadata

struct NodeSchema:
    """Schema for a graph node type"""
    var label: String
    var description: String
    var properties: Dict[String, PropertyMetadata]
    var sample_queries: List[String]
    var constraints: List[String]

struct RelationshipSchema:
    """Schema for a graph relationship type"""
    var type: String
    var description: String
    var from_node: String
    var to_node: String
    var properties: Dict[String, PropertyMetadata]
    var cardinality: String  # "one-to-one", "one-to-many", "many-to-many"

struct PropertyMetadata:
    """Metadata about a property"""
    var name: String
    var type: String  # "string", "integer", "float", "date", "boolean"
    var description: String
    var indexed: Bool
    var required: Bool
    var constraints: List[String]
    var examples: List[String]
```

### 2. Schema Loader (`schema_loader.mojo`)

Loads manual schema definitions from JSON config:

```json
{
  "version": "1.0",
  "graphs": {
    "supply_chain": {
      "description": "Supply chain management graph",
      "nodes": {
        "Product": {
          "description": "Physical or digital products",
          "properties": {
            "id": {
              "type": "string",
              "description": "Unique product identifier",
              "indexed": true,
              "required": true
            },
            "name": {
              "type": "string",
              "description": "Product name",
              "indexed": true
            },
            "sku": {
              "type": "string",
              "description": "Stock keeping unit"
            }
          },
          "sample_queries": [
            "MATCH (p:Product {name: $name}) RETURN p",
            "MATCH (p:Product) WHERE p.name CONTAINS $keyword RETURN p LIMIT 10"
          ]
        },
        "Supplier": {
          "description": "Supply chain vendors",
          "properties": {
            "id": {"type": "string", "indexed": true, "required": true},
            "name": {"type": "string", "indexed": true},
            "status": {
              "type": "string",
              "description": "Supplier status",
              "constraints": ["IN ['active', 'delayed', 'blocked']"]
            }
          }
        }
      },
      "relationships": {
        "SUPPLIED_BY": {
          "description": "Products supplied by suppliers",
          "from": "Product",
          "to": "Supplier",
          "cardinality": "many-to-one",
          "properties": {
            "lead_time": {
              "type": "integer",
              "description": "Lead time in days"
            },
            "delay": {
              "type": "integer",
              "description": "Current delay in days"
            }
          }
        }
      }
    }
  }
}
```

### 3. Schema Introspector (`schema_introspector.mojo`)

Auto-discovers schema from live databases:

**Neo4j Introspection:**
```cypher
CALL db.schema.visualization()
CALL db.schema.nodeTypeProperties()
CALL db.schema.relTypeProperties()
CALL db.constraints()
CALL db.indexes()
```

**Memgraph Introspection:**
```cypher
SHOW NODE_LABELS
SHOW REL_TYPES  
SHOW CONSTRAINTS
SHOW INDEX INFO
```

**HANA Graph Introspection:**
```sql
SELECT * FROM SYS.GRAPH_WORKSPACES
SELECT * FROM GRAPH_WORKSPACE_SCHEMA
```

### 4. Schema Merger (`schema_merger.mojo`)

Merges manual and auto-discovered schemas with conflict resolution:

**Priority Rules:**
1. Manual descriptions override auto-discovered
2. Auto-discovered properties supplement manual
3. Constraints from database take precedence
4. Indexes from database are authoritative

**Merge Strategy:**
```mojo
fn merge_schemas(
    manual: GraphSchema,
    auto: GraphSchema
) -> GraphSchema:
    """
    Merge manual and auto schemas.
    
    Priority:
    1. Descriptions: manual > auto
    2. Properties: union (manual + auto)
    3. Constraints: database (auto)
    4. Indexes: database (auto)
    """
```

## 🎓 Usage Examples

### Example 1: Load Schema

```mojo
from orchestration.catalog.schema_registry import GraphSchema
from orchestration.catalog.schema_loader import load_schema_from_json

# Load manual schema
var manual_schema = load_schema_from_json("config/graph_schemas.json")

print("Loaded", len(manual_schema.node_types), "node types")
print("Loaded", len(manual_schema.relationship_types), "relationship types")
```

### Example 2: Introspect Database

```mojo
from orchestration.catalog.schema_introspector import Neo4jIntrospector
from graph_toolkit.lib.clients.neo4j_client import Neo4jClient

# Connect to Neo4j
var client = Neo4jClient(uri="bolt://localhost:7687")
client.connect()

# Introspect schema
var introspector = Neo4jIntrospector(client)
var auto_schema = introspector.discover_schema()

print("Discovered", len(auto_schema.node_types), "node types")
```

### Example 3: Merge Schemas

```mojo
from orchestration.catalog.schema_merger import merge_schemas

# Merge manual and auto
var unified = merge_schemas(manual_schema, auto_schema)

print("Unified schema:")
print("  Nodes:", len(unified.node_types))
print("  Relationships:", len(unified.relationship_types))
print("  Indexed properties:", len(unified.property_index))
```

### Example 4: Query Schema

```mojo
# Find all node types with a specific property
var nodes_with_delay = unified.find_nodes_with_property("delay")
print("Nodes with 'delay' property:", nodes_with_delay)

# Get relationship between two nodes
var rel = unified.get_relationship("Product", "Supplier")
print("Relationship:", rel.type, "-", rel.description)

# Find sample queries for a node
var product = unified.node_types["Product"]
print("Sample queries for Product:")
for query in product.sample_queries:
    print("  -", query)
```

## 🔄 Integration with Query Translation

The catalog feeds into the NL → Cypher translation pipeline:

```mojo
from services.graph_query.translator import QueryTranslator

# Initialize translator with catalog
var translator = QueryTranslator(schema=unified)

# Translate natural language to Cypher
var result = translator.translate(
    "Find all suppliers with delays over 5 days"
)

print("Generated Cypher:", result.cypher)
print("Confidence:", result.confidence)
print("Used nodes:", result.node_types_used)
print("Used relationships:", result.relationship_types_used)
```

## 📊 Schema Caching

All schemas are cached in DragonflyDB for fast access:

```
Cache Keys:
- schema:supply_chain:manual          # Manual schema
- schema:supply_chain:auto:neo4j      # Auto schema from Neo4j
- schema:supply_chain:unified         # Merged schema
- schema:index:property:delay         # Property index

TTL: 1 hour (auto-refresh from database)
```

## 🔒 Security Considerations

1. **Schema Validation**: All manual schemas validated before use
2. **Database Credentials**: Stored securely (never in schema files)
3. **Query Limits**: Introspection queries have timeouts
4. **Access Control**: Schema access logged for audit

## 🎯 Next Steps

1. Implement schema_registry.mojo (core structures)
2. Implement schema_loader.mojo (JSON parsing)
3. Implement schema_introspector.mojo (database discovery)
4. Implement schema_merger.mojo (merge logic)
5. Create config/graph_schemas.json (sample schemas)
6. Integration tests with Neo4j/Memgraph/HANA

## 📚 References

- Neo4j Schema API: https://neo4j.com/docs/cypher-manual/current/schema/
- Memgraph Schema: https://memgraph.com/docs/memgraph/reference-guide/schema
- HANA Graph: https://help.sap.com/docs/HANA_SERVICE_CF/11afa2e60a5f4192a381df30f94863f9/

---

**Status**: Architecture complete, ready for implementation
**Estimated**: 4-6 hours for complete catalog system
**Dependencies**: graph-toolkit-mojo (Neo4j/Memgraph/HANA clients)
