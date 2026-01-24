"""
Schema Introspector

Generic schema discovery framework for graph databases.
Uses protocol-based abstraction for extensibility.
"""

from collections import Dict, List
from sys.ffi import DLHandle
from memory import UnsafePointer
from .schema_registry import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
    PropertyMetadata
)


# ============================================================================
# Schema Introspection Protocol
# ============================================================================

trait SchemaDiscoveryProtocol:
    """
    Protocol for schema discovery implementations.
    
    Each database type implements this protocol to provide
    schema introspection capabilities.
    """
    
    fn discover_nodes(inout self, client: object) raises -> Dict[String, NodeSchema]:
        """Discover node types and properties from database"""
        ...
    
    fn discover_relationships(inout self, client: object) raises -> Dict[String, RelationshipSchema]:
        """Discover relationship types and properties from database"""
        ...
    
    fn discover_constraints(inout self, client: object) raises -> Dict[String, List[String]]:
        """Discover database constraints (UNIQUE, NOT NULL, etc.)"""
        ...
    
    fn discover_indexes(inout self, client: object) raises -> Dict[String, List[String]]:
        """Discover database indexes"""
        ...


# ============================================================================
# Generic Schema Introspector
# ============================================================================

struct GenericSchemaIntrospector:
    """
    Generic schema introspector with plugin architecture.
    
    Supports multiple database types through protocol implementations:
    - Neo4j (Bolt protocol)
    - Memgraph (Bolt protocol)
    - HANA Graph (HTTP/REST)
    """
    var graph_name: String
    var db_type: String
    var enabled: Bool
    var verbose: Bool
    
    fn __init__(
        inout self,
        graph_name: String = "auto_schema",
        db_type: String = "hana",
        enabled: Bool = True,
        verbose: Bool = False
    ):
        """
        Initialize schema introspector.
        
        Args:
            graph_name: Name for discovered schema
            db_type: Database type ("hana", "hana", "hana")
            enabled: Enable introspection
            verbose: Debug output
        """
        self.graph_name = graph_name
        self.db_type = db_type
        self.enabled = enabled
        self.verbose = verbose
        
        if verbose:
            print(f"[SchemaIntrospector] Initialized for {db_type}")
    
    fn discover(inout self, client: object) raises -> GraphSchema:
        """
        Discover schema from connected database client.
        
        Args:
            client: Database client (Neo4jClient, MemgraphClient, HanaGraphClient)
            
        Returns:
            GraphSchema with auto-discovered structure
        """
        if not self.enabled:
            if self.verbose:
                print("[SchemaIntrospector] Disabled - returning empty schema")
            return GraphSchema(self.graph_name)
        
        var schema = GraphSchema(self.graph_name)
        schema.metadata.source = f"{self.db_type}_auto"
        schema.metadata.description = f"Auto-discovered from {self.db_type}"
        
        if self.verbose:
            print(f"[SchemaIntrospector] Discovering schema from {self.db_type}...")
        
        # Delegate to database-specific implementation
        var nodes = self._discover_nodes_impl(client)
        var rels = self._discover_relationships_impl(client)

        # Add nodes to schema
        for node_key in nodes.keys():
            schema.add_node(nodes[node_key[]])

        # Add relationships to schema
        for rel_key in rels.keys():
            schema.add_relationship(rels[rel_key[]])

        if self.verbose:
            print(f"[SchemaIntrospector] Discovered {schema.total_nodes} nodes, {schema.total_relationships} relationships")
        
        return schema
    
    fn _discover_nodes_impl(inout self, client: object) raises -> Dict[String, NodeSchema]:
        """Database-specific node discovery implementation"""
        var nodes = Dict[String, NodeSchema]()
        
        if self.db_type == "hana":
                    elif self.db_type == "hana":
                    elif self.db_type == "hana":
            return self._discover_hana_nodes(client)
        
        return nodes
    
    fn _discover_relationships_impl(inout self, client: object) raises -> Dict[String, RelationshipSchema]:
        """Database-specific relationship discovery implementation"""
        var rels = Dict[String, RelationshipSchema]()
        
        if self.db_type == "hana":
                    elif self.db_type == "hana":
                    elif self.db_type == "hana":
            return self._discover_hana_relationships(client)
        
        return rels
    
    # Database-specific implementations    fn _discover_hana_nodes(inout self, client: object) raises -> Dict[String, NodeSchema]:
        """
        HANA Graph: GET /workspaces/{workspace}/graphs/{graph}/schema
        """
        from python import Python
        var nodes = Dict[String, NodeSchema]()

        try:
            let requests = Python.import_module("requests")
            let json_mod = Python.import_module("json")

            # Assume client has workspace and graph properties
            let workspace = String(client.workspace)
            let graph = String(client.graph_name)
            let base_url = String(client.base_url)

            let url = f"{base_url}/workspaces/{workspace}/graphs/{graph}/schema"
            let response = requests.get(url, headers=client.headers)

            if int(response.status_code) == 200:
                let schema_data = json_mod.loads(response.text)

                # Parse node types from HANA schema response
                if "nodeTypes" in schema_data:
                    let node_types = schema_data["nodeTypes"]
                    for i in range(len(node_types)):
                        let node_def = node_types[i]
                        let label = String(node_def.get("name", "Unknown"))
                        var node = NodeSchema(label, String(node_def.get("description", "")))

                        # Parse properties
                        if "properties" in node_def:
                            let props = node_def["properties"]
                            for j in range(len(props)):
                                let prop_def = props[j]
                                var prop = PropertyMetadata(
                                    String(prop_def.get("name", "")),
                                    self._hana_type_to_standard(String(prop_def.get("type", "NVARCHAR"))),
                                    String(prop_def.get("description", "")),
                                    bool(prop_def.get("indexed", False)),
                                    bool(prop_def.get("required", False))
                                )
                                node.add_property(prop)

                        nodes[label] = node

        except e:
            if self.verbose:
                print(f"[SchemaIntrospector] HANA discovery error: {e}")

        return nodes

    fn _discover_hana_relationships(inout self, client: object) raises -> Dict[String, RelationshipSchema]:
        """
        HANA Graph: GET /workspaces/{workspace}/graphs/{graph}/schema
        """
        from python import Python
        var rels = Dict[String, RelationshipSchema]()

        try:
            let requests = Python.import_module("requests")
            let json_mod = Python.import_module("json")

            let workspace = String(client.workspace)
            let graph = String(client.graph_name)
            let base_url = String(client.base_url)

            let url = f"{base_url}/workspaces/{workspace}/graphs/{graph}/schema"
            let response = requests.get(url, headers=client.headers)

            if int(response.status_code) == 200:
                let schema_data = json_mod.loads(response.text)

                # Parse edge types from HANA schema response
                if "edgeTypes" in schema_data:
                    let edge_types = schema_data["edgeTypes"]
                    for i in range(len(edge_types)):
                        let edge_def = edge_types[i]
                        let rel_type = String(edge_def.get("name", "Unknown"))
                        var rel = RelationshipSchema(
                            rel_type,
                            String(edge_def.get("description", "")),
                            String(edge_def.get("sourceType", "")),
                            String(edge_def.get("targetType", "")),
                            String(edge_def.get("cardinality", "many-to-many"))
                        )

                        # Parse properties
                        if "properties" in edge_def:
                            let props = edge_def["properties"]
                            for j in range(len(props)):
                                let prop_def = props[j]
                                var prop = PropertyMetadata(
                                    String(prop_def.get("name", "")),
                                    self._hana_type_to_standard(String(prop_def.get("type", "NVARCHAR"))),
                                    String(prop_def.get("description", ""))
                                )
                                rel.add_property(prop)

                        rels[rel_type] = rel

        except e:
            if self.verbose:
                print(f"[SchemaIntrospector] HANA relationship discovery error: {e}")

        return rels

    fn _normalize_type(self, db_type: String) -> String:
        """Normalize database-specific types to standard types"""
        let lower_type = db_type.lower()

        if "int" in lower_type or "long" in lower_type:
            return "integer"
        elif "float" in lower_type or "double" in lower_type or "decimal" in lower_type:
            return "float"
        elif "bool" in lower_type:
            return "boolean"
        elif "date" in lower_type or "time" in lower_type:
            return "date"
        elif "list" in lower_type or "array" in lower_type:
            return "array"
        else:
            return "string"

    fn _hana_type_to_standard(self, hana_type: String) -> String:
        """Convert HANA SQL types to standard types"""
        let upper_type = hana_type.upper()

        if upper_type == "INTEGER" or upper_type == "BIGINT" or upper_type == "SMALLINT":
            return "integer"
        elif upper_type == "DOUBLE" or upper_type == "DECIMAL" or upper_type == "REAL":
            return "float"
        elif upper_type == "BOOLEAN":
            return "boolean"
        elif upper_type == "DATE" or upper_type == "TIMESTAMP" or upper_type == "TIME":
            return "date"
        else:
            return "string"

    fn _infer_type_from_value(self, value: object) -> String:
        """Infer type from a Python value"""
        from python import Python

        let type_name = String(type(value).__name__)

        if type_name == "int":
            return "integer"
        elif type_name == "float":
            return "float"
        elif type_name == "bool":
            return "boolean"
        elif type_name == "list":
            return "array"
        elif type_name == "datetime" or type_name == "date":
            return "date"
        else:
            return "string"


# ============================================================================
# Schema Discovery Utilities
# ============================================================================

fn infer_property_type(sample_values: List[String]) -> String:
    """
    Infer property type from sample values.
    
    Args:
        sample_values: Sample property values
        
    Returns:
        Inferred type ("string", "integer", "float", "boolean", "date")
    """
    if len(sample_values) == 0:
        return "string"
    
    # Check if all values are numeric
    var all_numeric = True
    for i in range(len(sample_values)):
        var val = sample_values[i]
        var is_digit = True
        for j in range(len(val)):
            if val[j] < "0" or val[j] > "9":
                is_digit = False
                break
        if not is_digit:
            all_numeric = False
            break
    
    if all_numeric:
        return "integer"
    
    # Check for boolean
    if len(sample_values) > 0:
        var first = sample_values[0]
        if first == "true" or first == "false":
            return "boolean"
    
    return "string"


fn parse_constraint_type(constraint_info: String) -> String:
    """
    Parse constraint type from database metadata.
    
    Args:
        constraint_info: Raw constraint string
        
    Returns:
        Constraint type ("UNIQUE", "INDEX", "NOT_NULL", etc.)
    """
    if "UNIQUE" in constraint_info:
        return "UNIQUE"
    elif "INDEX" in constraint_info:
        return "INDEX"
    elif "NOT NULL" in constraint_info:
        return "NOT_NULL"
    return "UNKNOWN"


# ============================================================================
# Factory Function
# ============================================================================

fn create_schema_introspector(
    graph_name: String = "auto_schema",
    db_type: String = "hana",
    enabled: Bool = True,
    verbose: Bool = False
) -> GenericSchemaIntrospector:
    """
    Create schema introspector for specified database type.
    
    This is the recommended way to create an introspector.
    
    Args:
        graph_name: Name for discovered schema
        db_type: Database type ("hana", "hana", "hana")
        enabled: Enable introspection
        verbose: Debug output
        
    Returns:
        Configured schema introspector
    """
    return GenericSchemaIntrospector(
        graph_name,
        db_type,
        enabled,
        verbose
    )


# ============================================================================
# Export for Zig/C Integration
# ============================================================================

@export
fn schema_discover(
    client_ptr: UnsafePointer[UInt8],
    db_type_ptr: UnsafePointer[UInt8],
    db_type_len: Int,
    graph_name_ptr: UnsafePointer[UInt8],
    graph_name_len: Int
) -> UnsafePointer[UInt8]:
    """
    C ABI function for schema discovery.
    
    Args:
        client_ptr: Database client pointer
        db_type_ptr: Database type string
        db_type_len: Database type length
        graph_name_ptr: Graph name string
        graph_name_len: Graph name length
        
    Returns:
        JSON-encoded schema string
    """
    var db_type = String(db_type_ptr, db_type_len)
    var graph_name = String(graph_name_ptr, graph_name_len)
    
    var introspector = create_schema_introspector(
        graph_name,
        db_type,
        enabled=True,
        verbose=True
    )
    
    # Note: client_ptr needs proper type handling
    # var schema = introspector.discover(client)
    
    # Return empty schema JSON for now
    var result = String('{"nodes":{},"relationships":{}}')
    return result.unsafe_ptr()
