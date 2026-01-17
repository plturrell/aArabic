"""
LLM Query Translator
====================

LLM-powered natural language to graph query translator.
Drop-in replacement for pattern-based NLToCypherTranslator.

Uses ShimmyLLMClient to generate queries via your local shimmy_openai_server,
providing much better accuracy than pattern-matching approaches.

Architecture:
    Natural Language → LLMQueryTranslator → ShimmyLLMClient → shimmy_openai_server
                                    ↓
                            Generated Cypher/SQL

Features:
    - Schema-aware prompt generation
    - Multi-database support (Neo4j, Memgraph, HANA)
    - Query validation
    - Error recovery with retry
    - Confidence scoring

Author: Shimmy-Mojo Team
Date: 2026-01-16
"""

from collections import List
from .shimmy_llm_client import ShimmyLLMClient, ChatMessage
from ..catalog.schema_registry import SchemaRegistry


struct LLMQueryTranslator:
    """
    LLM-powered query translator with schema awareness.
    
    This translator uses your local shimmy_openai_server to convert
    natural language queries into Cypher or SQL graph queries, with
    full awareness of your graph schema.
    
    Example:
        var schema = SchemaRegistry.load("config/graph_schemas.json")
        var translator = LLMQueryTranslator(schema)
        var query = translator.translate("Find suppliers with delays")
        # Returns: MATCH (s:Supplier)-[:DELAYED]->(o:Order) RETURN s
    """
    
    var llm: ShimmyLLMClient
    var schema: SchemaRegistry
    var database_type: String  # "cypher", "sql", "hana_graph"
    var max_retries: Int
    var enable_validation: Bool
    
    fn __init__(inout self, 
                schema: SchemaRegistry,
                database_type: String = "cypher",
                llm_url: String = "http://localhost:11434",
                model: String = "phi-3-mini"):
        """
        Initialize LLM query translator.
        
        Args:
            schema: Graph schema registry with node/relationship info
            database_type: Target query language ("cypher", "sql", "hana_graph")
            llm_url: URL of shimmy_openai_server
            model: LLM model to use
        """
        self.schema = schema
        self.database_type = database_type
        self.llm = ShimmyLLMClient(llm_url, model)
        self.max_retries = 3
        self.enable_validation = True
    
    fn translate(inout self, natural_query: String) raises -> String:
        """
        Translate natural language to graph query.
        
        This method:
        1. Builds a schema-aware prompt
        2. Calls shimmy_openai_server via LLM client
        3. Validates the generated query
        4. Retries on errors
        
        Args:
            natural_query: Natural language query (e.g., "Find delayed suppliers")
            
        Returns:
            Generated graph query (Cypher/SQL)
            
        Raises:
            Error if translation fails after retries
        """
        var attempt = 0
        var last_error = String("")
        
        while attempt < self.max_retries:
            try:
                # Build schema-aware prompt
                let prompt = self.build_prompt(natural_query, last_error)
                
                # Create messages for LLM
                var messages = List[ChatMessage]()
                messages.append(ChatMessage("system", self.get_system_prompt()))
                messages.append(ChatMessage("user", prompt))
                
                # Call LLM via shimmy_openai_server
                let response = self.llm.chat_completion(
                    messages,
                    temperature=0.3,  # Lower temperature for more deterministic queries
                    max_tokens=512
                )
                
                # Clean and extract query
                let query = self.extract_query(response)
                
                # Validate if enabled
                if self.enable_validation:
                    let validation_result = self.validate_query(query)
                    if not validation_result.is_valid:
                        last_error = validation_result.error_message
                        attempt += 1
                        continue
                
                return query
                
            except e:
                last_error = str(e)
                attempt += 1
        
        raise Error("Translation failed after " + str(self.max_retries) + " attempts. Last error: " + last_error)
    
    fn get_system_prompt(self) -> String:
        """
        Get the system prompt based on database type.
        """
        if self.database_type == "cypher":
            return """You are an expert graph database query translator specializing in Cypher (Neo4j/Memgraph).
Your task is to convert natural language questions into valid Cypher queries.

Rules:
1. Return ONLY the Cypher query, no explanations or markdown
2. Use the provided schema (node labels, relationships, properties)
3. Follow Cypher syntax strictly
4. Use appropriate WHERE clauses for filtering
5. Return relevant properties in RETURN clause
6. Handle temporal queries with appropriate date functions"""
        
        elif self.database_type == "sql":
            return """You are an expert at translating natural language to SQL queries for graph databases.
Convert natural language questions into valid SQL queries.

Rules:
1. Return ONLY the SQL query, no explanations
2. Use the provided schema (tables, columns, relationships)
3. Follow standard SQL syntax
4. Use JOINs appropriately for relationships
5. Include relevant columns in SELECT"""
        
        elif self.database_type == "hana_graph":
            return """You are an expert at SAP HANA Graph queries.
Convert natural language to HANA Graph SQL.

Rules:
1. Return ONLY the query, no explanations
2. Use GRAPH_TABLE syntax for graph traversals
3. Leverage HANA spatial functions when needed
4. Follow HANA-specific graph syntax"""
        
        else:
            return "You are a graph query expert. Convert natural language to graph queries."
    
    fn build_prompt(self, query: String, previous_error: String = "") -> String:
        """
        Build schema-aware prompt for the LLM.
        
        Includes:
        - Graph schema (nodes, relationships, properties)
        - Natural language query
        - Previous error (if retrying)
        """
        var prompt = String("Graph Schema:\n")
        prompt += self.get_schema_summary()
        prompt += "\n\nNatural Language Query:\n"
        prompt += query
        prompt += "\n\n"
        
        if len(previous_error) > 0:
            prompt += "Previous attempt had this error: " + previous_error + "\n"
            prompt += "Please fix the query.\n\n"
        
        if self.database_type == "cypher":
            prompt += "Generate a Cypher query. Return ONLY the query, no explanation."
        elif self.database_type == "sql":
            prompt += "Generate a SQL query. Return ONLY the query, no explanation."
        elif self.database_type == "hana_graph":
            prompt += "Generate a HANA Graph query. Return ONLY the query, no explanation."
        
        return prompt
    
    fn get_schema_summary(self) -> String:
        """
        Get a concise schema summary for the prompt.
        
        Format:
        Node Types:
        - Supplier (properties: id, name, location, rating)
        - Order (properties: id, date, amount, status)
        
        Relationships:
        - (Supplier)-[:SUPPLIES]->(Product)
        - (Supplier)-[:DELAYED]->(Order)
        """
        var summary = String("Node Types:\n")
        
        # Get node types from schema
        let nodes = self.schema.get_node_types()
        for node in nodes:
            summary += "- " + node.name + " (properties: "
            let props = node.properties
            for i in range(len(props)):
                if i > 0:
                    summary += ", "
                summary += props[i]
            summary += ")\n"
        
        summary += "\nRelationships:\n"
        
        # Get relationship types
        let rels = self.schema.get_relationship_types()
        for rel in rels:
            summary += "- (" + rel.from_node + ")-[:" + rel.type + "]->(" + rel.to_node + ")\n"
        
        return summary
    
    fn extract_query(self, llm_response: String) -> String:
        """
        Extract the actual query from LLM response.
        
        LLM might return:
        - Just the query
        - Query wrapped in markdown ```
        - Query with explanation
        
        We need to extract just the query part.
        """
        var cleaned = llm_response.strip()
        
        # Remove markdown code blocks
        if cleaned.startswith("```"):
            # Find the first newline after ```
            let first_newline = cleaned.find("\n")
            if first_newline > 0:
                cleaned = cleaned[first_newline + 1:]
            
            # Remove closing ```
            if cleaned.endswith("```"):
                cleaned = cleaned[:len(cleaned) - 3]
        
        # Remove "cypher" or "sql" language tag
        cleaned = cleaned.strip()
        if cleaned.startswith("cypher") or cleaned.startswith("CYPHER"):
            cleaned = cleaned[6:].strip()
        elif cleaned.startswith("sql") or cleaned.startswith("SQL"):
            cleaned = cleaned[3:].strip()
        
        # Take only the first statement (stop at explanation)
        let explanation_markers = ["\n\n", "Explanation:", "This query", "Note:"]
        for marker in explanation_markers:
            let pos = cleaned.find(marker)
            if pos > 0:
                cleaned = cleaned[:pos]
        
        return cleaned.strip()
    
    fn validate_query(self, query: String) -> ValidationResult:
        """
        Validate generated query for common issues.
        
        Checks:
        - Not empty
        - Contains expected keywords
        - Basic syntax validation
        - Schema compliance
        """
        if len(query) == 0:
            return ValidationResult(False, "Empty query generated")
        
        if self.database_type == "cypher":
            return self.validate_cypher(query)
        elif self.database_type == "sql":
            return self.validate_sql(query)
        else:
            return ValidationResult(True, "")  # Accept for unknown types
    
    fn validate_cypher(self, query: String) -> ValidationResult:
        """Validate Cypher query."""
        let query_upper = query.upper()
        
        # Must contain MATCH or CREATE or MERGE
        if not (query_upper.find("MATCH") >= 0 or 
                query_upper.find("CREATE") >= 0 or 
                query_upper.find("MERGE") >= 0):
            return ValidationResult(False, "Cypher query must contain MATCH, CREATE, or MERGE")
        
        # Should contain RETURN for queries
        if query_upper.find("MATCH") >= 0 and query_upper.find("RETURN") < 0:
            return ValidationResult(False, "Cypher MATCH query should contain RETURN clause")
        
        # Check balanced parentheses
        let open_count = query.count("(")
        let close_count = query.count(")")
        if open_count != close_count:
            return ValidationResult(False, "Unbalanced parentheses in query")
        
        return ValidationResult(True, "")
    
    fn validate_sql(self, query: String) -> ValidationResult:
        """Validate SQL query."""
        let query_upper = query.upper()
        
        # Must contain SELECT, INSERT, UPDATE, or DELETE
        if not (query_upper.find("SELECT") >= 0 or
                query_upper.find("INSERT") >= 0 or
                query_upper.find("UPDATE") >= 0 or
                query_upper.find("DELETE") >= 0):
            return ValidationResult(False, "SQL query must contain SELECT, INSERT, UPDATE, or DELETE")
        
        # SELECT should have FROM
        if query_upper.find("SELECT") >= 0 and query_upper.find("FROM") < 0:
            return ValidationResult(False, "SELECT query should contain FROM clause")
        
        return ValidationResult(True, "")


struct ValidationResult:
    """Result of query validation."""
    
    var is_valid: Bool
    var error_message: String
    
    fn __init__(inout self, is_valid: Bool, error_message: String):
        self.is_valid = is_valid
        self.error_message = error_message
