"""
Query Translation Module

Natural Language to Cypher translation with schema awareness.

Exports:
    - NLToCypherTranslator: Core translator
    - CypherQuery: Query result structure
    - QueryRouter: Multi-graph routing
    - SchemaPromptBuilder: LLM prompt generation
    - PromptTemplate: Pre-defined templates
    - TokenOptimizer: Prompt optimization
    - PromptCache: Prompt caching

Usage:
    from orchestration.query_translation import (
        NLToCypherTranslator,
        QueryRouter,
        SchemaPromptBuilder
    )
    
    # Load schema
    var schema = load_schema_from_json("config/graph_schemas.json")["supply_chain"]
    
    # Create translator
    var translator = NLToCypherTranslator(schema, verbose=True)
    
    # Translate query
    var result = translator.translate("Find delayed suppliers")
    print(result.query)
"""

from .nl_to_cypher_translator import (
    NLToCypherTranslator,
    CypherQuery,
    QueryRouter
)

from .schema_prompt_builder import (
    SchemaPromptBuilder,
    PromptTemplate,
    TokenOptimizer,
    PromptCache
)

__all__ = [
    "NLToCypherTranslator",
    "CypherQuery",
    "QueryRouter",
    "SchemaPromptBuilder",
    "PromptTemplate",
    "TokenOptimizer",
    "PromptCache"
]
