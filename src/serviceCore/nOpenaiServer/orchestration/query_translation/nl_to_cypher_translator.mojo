"""
Natural Language to Cypher Query Translator

Schema-aware translation engine that converts natural language
queries into valid Cypher queries using graph schema context.

Zero Python dependencies - Pure Mojo + Zig.
"""

from collections import Dict, List
from ..catalog.schema_registry import GraphSchema, NodeSchema, RelationshipSchema


# ============================================================================
# Query Translation Engine
# ============================================================================

struct CypherQuery:
    """
    Represents a generated Cypher query with metadata.
    """
    var query: String
    var graph_name: String
    var confidence: Float64
    var explanation: String
    var nodes_used: List[String]
    var relationships_used: List[String]
    
    fn __init__(
        inout self,
        query: String,
        graph_name: String,
        confidence: Float64 = 1.0,
        explanation: String = ""
    ):
        self.query = query
        self.graph_name = graph_name
        self.confidence = confidence
        self.explanation = explanation
        self.nodes_used = List[String]()
        self.relationships_used = List[String]()
    
    fn add_node(inout self, label: String):
        """Track which node types were used"""
        self.nodes_used.append(label)
    
    fn add_relationship(inout self, rel_type: String):
        """Track which relationship types were used"""
        self.relationships_used.append(rel_type)


struct NLToCypherTranslator:
    """
    Natural Language to Cypher translator.
    
    Uses graph schema to understand domain context and
    generate accurate Cypher queries.
    
    Pattern matching approach:
    1. Extract entities and actions from NL
    2. Map to schema nodes/relationships
    3. Generate Cypher MATCH/WHERE/RETURN
    
    Example:
        var schema = load_schema("supply_chain")
        var translator = NLToCypherTranslator(schema)
        
        var query = translator.translate("Find delayed suppliers")
        # Returns: MATCH (s:Supplier)-[r:SUPPLIES]->() 
        #          WHERE r.delay > 5 RETURN s
    """
    var schema: GraphSchema
    var verbose: Bool
    var min_confidence: Float64
    
    fn __init__(
        inout self,
        schema: GraphSchema,
        verbose: Bool = False,
        min_confidence: Float64 = 0.5
    ):
        self.schema = schema
        self.verbose = verbose
        self.min_confidence = min_confidence
        
        if verbose:
            print(f"[NLTranslator] Initialized for graph: {schema.metadata.graph_name}")
            print(f"[NLTranslator] Available nodes: {schema.total_nodes}")
            print(f"[NLTranslator] Available relationships: {schema.total_relationships}")
    
    fn translate(self, natural_query: String) raises -> CypherQuery:
        """
        Translate natural language to Cypher.
        
        Args:
            natural_query: Natural language query
            
        Returns:
            CypherQuery with generated Cypher and metadata
            
        Raises:
            Error if translation confidence too low
        """
        if self.verbose:
            print(f"[NLTranslator] Translating: {natural_query}")
        
        # Normalize query
        var normalized = _normalize_query(natural_query)
        
        # Extract query intent
        var intent = self._analyze_intent(normalized)
        
        # Find matching schema elements
        var matched_nodes = self._match_nodes(normalized)
        var matched_rels = self._match_relationships(normalized)
        
        if self.verbose:
            print(f"[NLTranslator] Intent: {intent}")
            print(f"[NLTranslator] Matched nodes: {len(matched_nodes)}")
            print(f"[NLTranslator] Matched relationships: {len(matched_rels)}")
        
        # Generate Cypher based on intent
        var cypher_query = CypherQuery(
            "",
            self.schema.metadata.graph_name,
            0.8,
            f"Translated from: {natural_query}"
        )
        
        if intent == "find" or intent == "get" or intent == "list":
            cypher_query.query = self._generate_match_query(
                matched_nodes,
                matched_rels,
                normalized
            )
        elif intent == "count":
            cypher_query.query = self._generate_count_query(
                matched_nodes,
                matched_rels
            )
        elif intent == "analyze" or intent == "show":
            cypher_query.query = self._generate_analysis_query(
                matched_nodes,
                matched_rels
            )
        else:
            # Default to MATCH query
            cypher_query.query = self._generate_match_query(
                matched_nodes,
                matched_rels,
                normalized
            )
        
        # Add metadata
        for node in matched_nodes:
            cypher_query.add_node(node)
        for rel in matched_rels:
            cypher_query.add_relationship(rel)
        
        if self.verbose:
            print(f"[NLTranslator] Generated Cypher:")
            print(f"  {cypher_query.query}")
            print(f"[NLTranslator] Confidence: {cypher_query.confidence}")
        
        # Check confidence threshold
        if cypher_query.confidence < self.min_confidence:
            raise Error(f"Translation confidence too low: {cypher_query.confidence}")
        
        return cypher_query
    
    fn _analyze_intent(self, query: String) -> String:
        """Determine query intent from keywords"""
        var lower_query = _to_lowercase(query)
        
        if _contains(lower_query, "find") or _contains(lower_query, "get") or _contains(lower_query, "show"):
            return "find"
        elif _contains(lower_query, "count") or _contains(lower_query, "how many"):
            return "count"
        elif _contains(lower_query, "list") or _contains(lower_query, "all"):
            return "list"
        elif _contains(lower_query, "analyze") or _contains(lower_query, "report"):
            return "analyze"
        else:
            return "find"  # Default
    
    fn _match_nodes(self, query: String) -> List[String]:
        """Find node labels mentioned in query"""
        var matched = List[String]()
        var lower_query = _to_lowercase(query)
        
        # Check each node label in schema
        var node_keys = self.schema.nodes.keys()
        for i in range(len(node_keys)):
            var label = node_keys[i]
            var lower_label = _to_lowercase(label)
            
            # Also check plural form
            var plural = lower_label + "s"
            
            if _contains(lower_query, lower_label) or _contains(lower_query, plural):
                matched.append(label)
        
        return matched
    
    fn _match_relationships(self, query: String) -> List[String]:
        """Find relationship types mentioned in query"""
        var matched = List[String]()
        var lower_query = _to_lowercase(query)
        
        # Check each relationship in schema
        var rel_keys = self.schema.relationships.keys()
        for i in range(len(rel_keys)):
            var rel_type = rel_keys[i]
            var lower_rel = _to_lowercase(rel_type)
            
            if _contains(lower_query, lower_rel):
                matched.append(rel_type)
        
        return matched
    
    fn _generate_match_query(
        self,
        nodes: List[String],
        relationships: List[String],
        query: String
    ) -> String:
        """Generate MATCH query"""
        var cypher = String("MATCH ")
        
        if len(nodes) == 0:
            # No specific nodes - return generic pattern
            return "MATCH (n) RETURN n LIMIT 10"
        
        if len(nodes) == 1:
            # Single node query
            var node = nodes[0]
            var node_var = _to_lowercase(node[0:1])
            cypher += f"({node_var}:{node})"
            
            # Add WHERE clause if query contains conditions
            if _contains(query, "delayed") or _contains(query, "late"):
                cypher += f" WHERE {node_var}.delay > 5"
            elif _contains(query, "high") or _contains(query, "critical"):
                cypher += f" WHERE {node_var}.priority = 'high'"
            
            cypher += f" RETURN {node_var} LIMIT 25"
        
        elif len(nodes) >= 2 and len(relationships) > 0:
            # Multi-node with relationship
            var node1 = nodes[0]
            var node2 = nodes[1]
            var rel = relationships[0]
            
            var var1 = _to_lowercase(node1[0:1])
            var var2 = _to_lowercase(node2[0:1])
            
            cypher += f"({var1}:{node1})-[r:{rel}]->({var2}:{node2})"
            cypher += f" RETURN {var1}, r, {var2} LIMIT 25"
        
        else:
            # Multiple nodes without clear relationship
            var node = nodes[0]
            var node_var = _to_lowercase(node[0:1])
            cypher += f"({node_var}:{node}) RETURN {node_var} LIMIT 25"
        
        return cypher
    
    fn _generate_count_query(
        self,
        nodes: List[String],
        relationships: List[String]
    ) -> String:
        """Generate COUNT query"""
        if len(nodes) == 0:
            return "MATCH (n) RETURN count(n) as total"
        
        var node = nodes[0]
        var node_var = _to_lowercase(node[0:1])
        return f"MATCH ({node_var}:{node}) RETURN count({node_var}) as total"
    
    fn _generate_analysis_query(
        self,
        nodes: List[String],
        relationships: List[String]
    ) -> String:
        """Generate analysis/aggregation query"""
        if len(nodes) == 0:
            return "MATCH (n) RETURN labels(n) as type, count(n) as count"
        
        var node = nodes[0]
        var node_var = _to_lowercase(node[0:1])
        
        # Try to find a meaningful property to aggregate
        var node_schema = self.schema.nodes.get(node)
        if node_schema:
            # Look for numeric properties
            for prop_name in node_schema.properties.keys():
                var prop = node_schema.properties[prop_name]
                if prop.type == "integer" or prop.type == "float":
                    return f"MATCH ({node_var}:{node}) RETURN avg({node_var}.{prop_name}) as average_{prop_name}"
        
        # Default to count
        return f"MATCH ({node_var}:{node}) RETURN count({node_var}) as total"


# ============================================================================
# Multi-Graph Router
# ============================================================================

struct QueryRouter:
    """
    Routes natural language queries to appropriate graph databases.
    
    Uses multiple schema contexts to determine which graph
    should handle the query.
    """
    var translators: Dict[String, NLToCypherTranslator]
    var verbose: Bool
    
    fn __init__(inout self, verbose: Bool = False):
        self.translators = Dict[String, NLToCypherTranslator]()
        self.verbose = verbose
    
    fn add_graph(inout self, schema: GraphSchema):
        """Add a graph schema for routing"""
        var translator = NLToCypherTranslator(schema, self.verbose)
        self.translators[schema.metadata.graph_name] = translator
        
        if self.verbose:
            print(f"[QueryRouter] Added graph: {schema.metadata.graph_name}")
    
    fn route_and_translate(self, natural_query: String) raises -> CypherQuery:
        """
        Route query to best graph and translate.
        
        Args:
            natural_query: Natural language query
            
        Returns:
            CypherQuery for the most appropriate graph
        """
        if self.verbose:
            print(f"[QueryRouter] Routing query: {natural_query}")
        
        # Score each graph based on keyword overlap
        var best_graph = String("")
        var best_score = 0.0
        
        var graph_names = self.translators.keys()
        for i in range(len(graph_names)):
            var graph_name = graph_names[i]
            var translator = self.translators[graph_name]
            
            # Score based on node/relationship matches
            var matched_nodes = translator._match_nodes(natural_query)
            var matched_rels = translator._match_relationships(natural_query)
            
            var score = Float64(len(matched_nodes)) * 2.0 + Float64(len(matched_rels)) * 1.5
            
            if self.verbose:
                print(f"[QueryRouter] {graph_name}: score={score}")
            
            if score > best_score:
                best_score = score
                best_graph = graph_name
        
        if best_graph == "":
            raise Error("Could not route query to any graph")
        
        if self.verbose:
            print(f"[QueryRouter] Selected graph: {best_graph}")
        
        # Translate using selected graph
        return self.translators[best_graph].translate(natural_query)


# ============================================================================
# Helper Functions
# ============================================================================

fn _normalize_query(query: String) -> String:
    """Normalize query text"""
    var result = query
    # Remove extra whitespace
    result = _trim(result)
    # Remove trailing punctuation
    if len(result) > 0 and result[len(result)-1] == '?':
        result = result[0:len(result)-1]
    return result


fn _to_lowercase(text: String) -> String:
    """Convert string to lowercase (simplified)"""
    # TODO: Implement proper lowercase conversion
    # For now, return as-is (Mojo stdlib will have this)
    return text


fn _contains(text: String, substring: String) -> Bool:
    """Check if text contains substring"""
    return text.find(substring) >= 0


fn _trim(text: String) -> String:
    """Trim whitespace from string"""
    var start = 0
    var end = len(text)
    
    # Trim start
    while start < end and text[start] == ' ':
        start += 1
    
    # Trim end
    while end > start and text[end-1] == ' ':
        end -= 1
    
    return text[start:end]
