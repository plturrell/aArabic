"""
Schema-Aware Prompt Builder

Builds context-rich prompts for LLM translation by including
relevant schema information, examples, and constraints.

Optimizes token usage while maximizing translation accuracy.
"""

from collections import Dict, List
from ..catalog.schema_registry import (
    GraphSchema,
    NodeSchema,
    RelationshipSchema,
    PropertyMetadata
)


# ============================================================================
# Prompt Builder
# ============================================================================

struct SchemaPromptBuilder:
    """
    Builds LLM prompts with schema context for accurate translation.
    
    Strategy:
    1. Include relevant schema elements (not entire schema)
    2. Provide examples from schema
    3. Add constraints and validation rules
    4. Keep token count minimal
    
    Example:
        var schema = load_schema("supply_chain")
        var builder = SchemaPromptBuilder(schema)
        
        var prompt = builder.build_prompt(
            "Find delayed suppliers",
            include_examples=True
        )
    """
    var schema: GraphSchema
    var max_tokens: Int
    var include_property_details: Bool
    var include_sample_queries: Bool
    var verbose: Bool
    
    fn __init__(
        inout self,
        schema: GraphSchema,
        max_tokens: Int = 2000,
        include_property_details: Bool = True,
        include_sample_queries: Bool = True,
        verbose: Bool = False
    ):
        self.schema = schema
        self.max_tokens = max_tokens
        self.include_property_details = include_property_details
        self.include_sample_queries = include_sample_queries
        self.verbose = verbose
        
        if verbose:
            print(f"[PromptBuilder] Initialized for: {schema.metadata.graph_name}")
            print(f"[PromptBuilder] Max tokens: {max_tokens}")
    
    fn build_translation_prompt(
        self,
        natural_query: String,
        matched_nodes: List[String],
        matched_rels: List[String]
    ) -> String:
        """
        Build complete prompt for LLM translation.
        
        Args:
            natural_query: User's natural language query
            matched_nodes: Node labels relevant to query
            matched_rels: Relationship types relevant to query
            
        Returns:
            Complete prompt for LLM
        """
        var prompt = String("")
        
        # System context
        prompt += "You are a Cypher query expert. Convert natural language to Cypher.\n\n"
        
        # Graph context
        prompt += f"# Graph Database: {self.schema.metadata.graph_name}\n"
        prompt += f"Description: {self.schema.metadata.description}\n\n"
        
        # Schema context (relevant parts only)
        if len(matched_nodes) > 0:
            prompt += "# Relevant Node Types:\n"
            for i in range(len(matched_nodes)):
                var label = matched_nodes[i]
                prompt += self._format_node_schema(label)
                prompt += "\n"
        
        if len(matched_rels) > 0:
            prompt += "# Relevant Relationships:\n"
            for i in range(len(matched_rels)):
                var rel_type = matched_rels[i]
                prompt += self._format_relationship_schema(rel_type)
                prompt += "\n"
        
        # Examples (if available)
        if self.include_sample_queries:
            var examples = self._get_relevant_examples(matched_nodes)
            if len(examples) > 0:
                prompt += "# Example Queries:\n"
                prompt += examples
                prompt += "\n"
        
        # Cypher best practices
        prompt += "# Rules:\n"
        prompt += "- Use MATCH for pattern matching\n"
        prompt += "- Use WHERE for filtering\n"
        prompt += "- Always include RETURN clause\n"
        prompt += "- Use LIMIT for large result sets\n"
        prompt += "- Property names are case-sensitive\n\n"
        
        # The actual query to translate
        prompt += f"# Translate to Cypher:\n"
        prompt += f"\"{natural_query}\"\n\n"
        prompt += "Cypher Query:"
        
        if self.verbose:
            print(f"[PromptBuilder] Generated prompt ({len(prompt)} chars)")
        
        return prompt
    
    fn build_validation_prompt(
        self,
        cypher_query: String
    ) -> String:
        """
        Build prompt for validating generated Cypher.
        
        Args:
            cypher_query: Cypher query to validate
            
        Returns:
            Validation prompt
        """
        var prompt = String("")
        
        prompt += "Validate this Cypher query for correctness:\n\n"
        prompt += f"Query: {cypher_query}\n\n"
        prompt += "Check for:\n"
        prompt += "1. Syntax correctness\n"
        prompt += "2. Valid node labels and relationship types\n"
        prompt += "3. Valid property names\n"
        prompt += "4. Performance issues (missing indexes, etc.)\n\n"
        prompt += "Available schema:\n"
        prompt += self._format_schema_summary()
        prompt += "\nIs this query valid? If not, what are the issues?"
        
        return prompt
    
    fn _format_node_schema(self, label: String) -> String:
        """Format node schema for prompt"""
        var output = String("")
        
        if label not in self.schema.nodes:
            return output
        
        var node = self.schema.nodes[label]
        
        output += f"- {label}: {node.description}\n"
        
        if self.include_property_details and len(node.properties) > 0:
            output += "  Properties:\n"
            var prop_keys = node.properties.keys()
            for i in range(min(5, len(prop_keys))):  # Limit to 5 properties
                var prop_name = prop_keys[i]
                var prop = node.properties[prop_name]
                output += f"    - {prop_name} ({prop.type})"
                if prop.required:
                    output += " [required]"
                if prop.indexed:
                    output += " [indexed]"
                output += f": {prop.description}\n"
        
        return output
    
    fn _format_relationship_schema(self, rel_type: String) -> String:
        """Format relationship schema for prompt"""
        var output = String("")
        
        if rel_type not in self.schema.relationships:
            return output
        
        var rel = self.schema.relationships[rel_type]
        
        output += f"- {rel_type}: {rel.description}\n"
        output += f"  Pattern: ({rel.from_node})-[:{rel_type}]->({rel.to_node})\n"
        output += f"  Cardinality: {rel.cardinality}\n"
        
        if self.include_property_details and len(rel.properties) > 0:
            output += "  Properties:\n"
            var prop_keys = rel.properties.keys()
            for i in range(min(3, len(prop_keys))):  # Limit to 3 properties
                var prop_name = prop_keys[i]
                var prop = rel.properties[prop_name]
                output += f"    - {prop_name} ({prop.type}): {prop.description}\n"
        
        return output
    
    fn _get_relevant_examples(self, node_labels: List[String]) -> String:
        """Get relevant sample queries from schema"""
        var examples = String("")
        
        for i in range(len(node_labels)):
            var label = node_labels[i]
            if label not in self.schema.nodes:
                continue
            
            var node = self.schema.nodes[label]
            if len(node.sample_queries) > 0:
                # Include first 2 examples
                for j in range(min(2, len(node.sample_queries))):
                    examples += f"- {node.sample_queries[j]}\n"
        
        return examples
    
    fn _format_schema_summary(self) -> String:
        """Format complete schema summary"""
        var output = String("")
        
        output += "Node Types: "
        var node_keys = self.schema.nodes.keys()
        for i in range(len(node_keys)):
            if i > 0:
                output += ", "
            output += node_keys[i]
        output += "\n"
        
        output += "Relationship Types: "
        var rel_keys = self.schema.relationships.keys()
        for i in range(len(rel_keys)):
            if i > 0:
                output += ", "
            output += rel_keys[i]
        output += "\n"
        
        return output


# ============================================================================
# Prompt Templates
# ============================================================================

struct PromptTemplate:
    """
    Pre-defined prompt templates for common scenarios.
    """
    
    @staticmethod
    fn get_basic_translation_template() -> String:
        """Basic translation template"""
        return """You are a Cypher query expert.

Translate this natural language query to Cypher:
"{query}"

Use this schema:
{schema}

Respond with ONLY the Cypher query, no explanation."""
    
    @staticmethod
    fn get_few_shot_template() -> String:
        """Few-shot learning template with examples"""
        return """You are a Cypher query expert.

Examples:
{examples}

Translate to Cypher:
"{query}"

Schema:
{schema}

Cypher:"""
    
    @staticmethod
    fn get_chain_of_thought_template() -> String:
        """Chain-of-thought template for complex queries"""
        return """You are a Cypher query expert.

Query: "{query}"

Schema: {schema}

Think step-by-step:
1. What entities are mentioned?
2. What relationships are implied?
3. What filters or conditions are needed?
4. What should be returned?

Now generate the Cypher query:"""


# ============================================================================
# Token Optimizer
# ============================================================================

struct TokenOptimizer:
    """
    Optimizes prompts to fit within token limits.
    
    Strategies:
    - Remove less relevant properties
    - Truncate descriptions
    - Limit examples
    - Compress whitespace
    """
    var target_tokens: Int
    var verbose: Bool
    
    fn __init__(inout self, target_tokens: Int = 2000, verbose: Bool = False):
        self.target_tokens = target_tokens
        self.verbose = verbose
    
    fn optimize_prompt(self, prompt: String) -> String:
        """
        Optimize prompt to fit token limit.
        
        Rough estimate: 1 token ≈ 4 characters
        """
        var estimated_tokens = len(prompt) / 4
        
        if estimated_tokens <= self.target_tokens:
            return prompt
        
        if self.verbose:
            print(f"[TokenOptimizer] Prompt too long: {estimated_tokens} tokens")
            print(f"[TokenOptimizer] Compressing to {self.target_tokens} tokens")
        
        # Simple compression: truncate to character limit
        var target_chars = self.target_tokens * 4
        if len(prompt) > target_chars:
            return prompt[0:target_chars] + "\n..."
        
        return prompt
    
    fn estimate_tokens(self, text: String) -> Int:
        """Rough token estimate"""
        return len(text) / 4


# ============================================================================
# Prompt Cache
# ============================================================================

struct PromptCache:
    """
    Caches generated prompts for reuse.
    
    Reduces redundant prompt generation for similar queries.
    """
    var cache: Dict[String, String]
    var max_size: Int
    var hits: Int
    var misses: Int
    
    fn __init__(inout self, max_size: Int = 100):
        self.cache = Dict[String, String]()
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    fn get(inout self, key: String) -> String:
        """Get cached prompt"""
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        else:
            self.misses += 1
            return ""
    
    fn put(inout self, key: String, prompt: String):
        """Cache prompt"""
        if len(self.cache) >= self.max_size:
            # Simple eviction: clear cache when full
            self.cache = Dict[String, String]()
        
        self.cache[key] = prompt
    
    fn get_hit_rate(self) -> Float64:
        """Get cache hit rate"""
        var total = self.hits + self.misses
        if total == 0:
            return 0.0
        return Float64(self.hits) / Float64(total)
