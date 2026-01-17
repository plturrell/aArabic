"""
Graph Data Product Catalog

Intelligent schema management for graph databases enabling natural language
to Cypher query translation with full context awareness.

Components:
- schema_registry: Core data structures for graph schemas
- schema_loader: Load manual schemas from JSON config
- schema_introspector: Auto-discover schemas from live databases
- schema_merger: Merge manual and auto-discovered schemas

Usage:
    from orchestration.catalog.schema_registry import GraphSchema
    from orchestration.catalog.schema_loader import load_schema_from_json
    
    var schema = load_schema_from_json("config/graph_schemas.json")
    print("Loaded", len(schema.node_types), "node types")
"""

# Module exports
from .schema_registry import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
    PropertyMetadata,
    SchemaMetadata
)
