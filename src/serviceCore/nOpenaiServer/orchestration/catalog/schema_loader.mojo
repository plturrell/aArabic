"""
Schema Loader

Loads graph schemas from JSON configuration files.
ABSOLUTE ZERO PYTHON DEPENDENCIES - Pure Mojo + Zig.
"""

from collections import Dict, List
from .schema_registry import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
    PropertyMetadata,
    SchemaMetadata
)

# Import mojo-sdk JSON parser (zero Python!)
import sys
sys.path.append("../../../mojo-sdk/stdlib")
from json import JsonParser


# ============================================================================
# Schema Loader (100% Python-Free!)
# ============================================================================

fn load_schema_from_json(json_path: String) raises -> Dict[String, GraphSchema]:
    """
    Load all graph schemas from JSON configuration file.
    
    ABSOLUTE ZERO PYTHON DEPENDENCIES!
    All operations use Zig std.json backend.
    
    Args:
        json_path: Path to graph_schemas.json
        
    Returns:
        Dict mapping graph name to GraphSchema
        
    Example:
        var schemas = load_schema_from_json("config/graph_schemas.json")
        var supply_chain = schemas["supply_chain"]
        print("Loaded", supply_chain.total_nodes, "node types")
    """
    # Use mojo-sdk JSON parser (Zig backend - zero Python!)
    var parser = JsonParser(verbose=True)
    var json_content = parser.parse_file(json_path)
    
    print("[SchemaLoader] Using Zig for ALL operations (zero Python!)")
    
    # Build schemas using Zig dict/array operations
    var schemas = Dict[String, GraphSchema]()
    
    # Get graphs object
    var graphs_json = parser.get_nested_value(json_content, "graphs")
    var graph_keys = parser.get_keys(graphs_json)
    
    # Parse comma-separated keys
    var graph_names = _split_string(graph_keys, ",")
    
    print(f"[SchemaLoader] Found {len(graph_names)} graphs")
    
    for i in range(len(graph_names)):
        var graph_name = graph_names[i]
        
        # Get graph data
        var graph_json = parser.get_value(graphs_json, graph_name)
        
        # Create schema
        var schema = GraphSchema(graph_name)
        
        # Get description
        if parser.has_key(graph_json, "description"):
            var desc = parser.get_value(graph_json, "description")
            schema.metadata.description = _strip_quotes(desc)
        
        schema.metadata.source = "manual"
        
        # Load nodes
        if parser.has_key(graph_json, "nodes"):
            var nodes_json = parser.get_value(graph_json, "nodes")
            var node_keys = parser.get_keys(nodes_json)
            var node_labels = _split_string(node_keys, ",")
            
            for j in range(len(node_labels)):
                var node_label = node_labels[j]
                var node_json = parser.get_value(nodes_json, node_label)
                
                var node = NodeSchema(
                    node_label,
                    _get_string_value(parser, node_json, "description"),
                    _get_string_value(parser, node_json, "category", "entity")
                )
                
                # Load properties
                if parser.has_key(node_json, "properties"):
                    var props_json = parser.get_value(node_json, "properties")
                    var prop_keys = parser.get_keys(props_json)
                    var prop_names = _split_string(prop_keys, ",")
                    
                    for k in range(len(prop_names)):
                        var prop_name = prop_names[k]
                        var prop_json = parser.get_value(props_json, prop_name)
                        
                        var prop = PropertyMetadata(
                            prop_name,
                            _get_string_value(parser, prop_json, "type", "string"),
                            _get_string_value(parser, prop_json, "description"),
                            _get_bool_value(parser, prop_json, "indexed", False),
                            _get_bool_value(parser, prop_json, "required", False)
                        )
                        
                        # Add constraints
                        if parser.has_key(prop_json, "constraints"):
                            var constraints_json = parser.get_value(prop_json, "constraints")
                            var constraint_count = parser.get_array_length(constraints_json)
                            for c in range(constraint_count):
                                var constraint = parser.get_array_item(constraints_json, c)
                                prop.add_constraint(_strip_quotes(constraint))
                        
                        # Add examples
                        if parser.has_key(prop_json, "examples"):
                            var examples_json = parser.get_value(prop_json, "examples")
                            var example_count = parser.get_array_length(examples_json)
                            for e in range(example_count):
                                var example = parser.get_array_item(examples_json, e)
                                prop.add_example(_strip_quotes(example))
                        
                        node.add_property(prop)
                
                # Load sample queries
                if parser.has_key(node_json, "sample_queries"):
                    var queries_json = parser.get_value(node_json, "sample_queries")
                    var query_count = parser.get_array_length(queries_json)
                    for q in range(query_count):
                        var query = parser.get_array_item(queries_json, q)
                        node.add_sample_query(_strip_quotes(query))
                
                schema.add_node(node)
        
        # Load relationships
        if parser.has_key(graph_json, "relationships"):
            var rels_json = parser.get_value(graph_json, "relationships")
            var rel_keys = parser.get_keys(rels_json)
            var rel_types = _split_string(rel_keys, ",")
            
            for j in range(len(rel_types)):
                var rel_type = rel_types[j]
                var rel_json = parser.get_value(rels_json, rel_type)
                
                var rel = RelationshipSchema(
                    rel_type,
                    _get_string_value(parser, rel_json, "description"),
                    _get_string_value(parser, rel_json, "from"),
                    _get_string_value(parser, rel_json, "to"),
                    _get_string_value(parser, rel_json, "cardinality", "many-to-many")
                )
                
                rel.bidirectional = _get_bool_value(parser, rel_json, "bidirectional", False)
                
                # Load properties
                if parser.has_key(rel_json, "properties"):
                    var props_json = parser.get_value(rel_json, "properties")
                    var prop_keys = parser.get_keys(props_json)
                    var prop_names = _split_string(prop_keys, ",")
                    
                    for k in range(len(prop_names)):
                        var prop_name = prop_names[k]
                        var prop_json = parser.get_value(props_json, prop_name)
                        
                        var prop = PropertyMetadata(
                            prop_name,
                            _get_string_value(parser, prop_json, "type", "string"),
                            _get_string_value(parser, prop_json, "description")
                        )
                        
                        # Add constraints
                        if parser.has_key(prop_json, "constraints"):
                            var constraints_json = parser.get_value(prop_json, "constraints")
                            var constraint_count = parser.get_array_length(constraints_json)
                            for c in range(constraint_count):
                                var constraint = parser.get_array_item(constraints_json, c)
                                prop.add_constraint(_strip_quotes(constraint))
                        
                        rel.add_property(prop)
                
                schema.add_relationship(rel)
        
        schemas[graph_name] = schema
        
        print("✅ Loaded schema:", graph_name, "(100% Zig!)")
        print("   Nodes:", schema.total_nodes)
        print("   Relationships:", schema.total_relationships)
    
    return schemas


# ============================================================================
# Helper Functions (Pure Mojo String Operations)
# ============================================================================

fn _split_string(text: String, delimiter: String) -> List[String]:
    """Split string by delimiter (pure Mojo)"""
    var parts = List[String]()
    var start = 0
    
    for i in range(len(text)):
        if text[i:i+1] == delimiter:
            if i > start:
                parts.append(text[start:i])
            start = i + 1
    
    # Add last part
    if start < len(text):
        parts.append(text[start:])
    
    return parts


fn _strip_quotes(text: String) -> String:
    """Remove surrounding quotes from JSON string value"""
    if len(text) >= 2 and text[0] == '"' and text[len(text)-1] == '"':
        return text[1:len(text)-1]
    return text


fn _get_string_value(parser: JsonParser, json: String, key: String, default: String = "") -> String:
    """Get string value with default (zero Python!)"""
    if not parser.has_key(json, key):
        return default
    
    try:
        var value = parser.get_value(json, key)
        return _strip_quotes(value)
    except:
        return default


fn _get_bool_value(parser: JsonParser, json: String, key: String, default: Bool = False) -> Bool:
    """Get boolean value with default (zero Python!)"""
    if not parser.has_key(json, key):
        return default
    
    try:
        var value = parser.get_value(json, key)
        if value == "true":
            return True
        elif value == "false":
            return False
        return default
    except:
        return default


fn load_single_schema(json_path: String, graph_name: String) raises -> GraphSchema:
    """
    Load a single graph schema by name.
    
    Args:
        json_path: Path to graph_schemas.json
        graph_name: Name of the graph
        
    Returns:
        GraphSchema for the specified graph
    """
    var all_schemas = load_schema_from_json(json_path)
    
    if graph_name in all_schemas:
        return all_schemas[graph_name]
    
    raise Error("Graph schema not found: " + graph_name)


# ============================================================================
# Schema Export (for debugging/inspection)
# ============================================================================

fn schema_to_string(schema: GraphSchema) -> String:
    """Convert schema to human-readable string"""
    var output = String("Graph Schema: ") + schema.metadata.graph_name + String("\n")
    output += String("Version: ") + schema.metadata.version + String("\n")
    output += String("Description: ") + schema.metadata.description + String("\n")
    output += String("\n")
    output += String("Node Types: ") + String(schema.total_nodes) + String("\n")
    output += String("Relationship Types: ") + String(schema.total_relationships) + String("\n")
    
    return output
