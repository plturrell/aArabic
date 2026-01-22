"""
LLM Integration Module
======================

Provides LLM-powered query translation for graph databases.

This module integrates your local shimmy_openai_server with the graph
orchestration system, enabling natural language queries with much better
accuracy than pattern-based approaches.

Public API:
    - ShimmyLLMClient: Client for shimmy_openai_server
    - ChatMessage: Message structure for conversations
    - LLMQueryTranslator: Main translator class
    - ValidationResult: Query validation result

Example Usage:
    from orchestration.llm_integration import LLMQueryTranslator
    from orchestration.catalog import SchemaRegistry
    
    # Load your graph schema
    var schema = SchemaRegistry.load("config/graph_schemas.json")
    
    # Create LLM-powered translator
    var translator = LLMQueryTranslator(schema)
    
    # Translate natural language to Cypher
    var query = translator.translate("Find suppliers with delays")
    print(query)
    # Output: MATCH (s:Supplier)-[:DELAYED]->(o:Order) RETURN s

Architecture:
    Natural Language
         ↓
    LLMQueryTranslator (this module)
         ↓
    ShimmyLLMClient
         ↓
    HTTPClient (existing)
         ↓
    zig_shimmy_post (existing)
         ↓
    shimmy_openai_server (your local LLM)

Author: Shimmy-Mojo Team
Date: 2026-01-16
Version: 1.0.0
"""

# Export the main client
from .shimmy_llm_client import ShimmyLLMClient, ChatMessage

# Export the translator
from .llm_query_translator import LLMQueryTranslator, ValidationResult

# Module metadata
__version__ = "1.0.0"
__author__ = "Shimmy-Mojo Team"
__all__ = ["ShimmyLLMClient", "ChatMessage", "LLMQueryTranslator", "ValidationResult"]
