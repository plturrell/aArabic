#!/usr/bin/env python3
"""
GraphChat Bridge MCP Server
Provides LLM and graph querying capabilities for Memgraph Lab's GraphChat feature.
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("graphchat-bridge")

# Configuration
MEMGRAPH_URI = os.getenv("MEMGRAPH_URI", "bolt://localhost:7687")
SHIMMY_URL = os.getenv("SHIMMY_URL", "http://shimmy:11434")
LANGFLOW_URL = os.getenv("LANGFLOW_URL", "http://langflow:7860")
MODEL_SERVER_URL = os.getenv("MODEL_SERVER_URL", "http://model-server:8000")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Qwen3-Coder-30B-A3B-Instruct")

# Initialize MCP server
app = Server("graphchat-bridge")


class MemgraphClient:
    """Client for Memgraph database operations."""
    
    def __init__(self, uri: str):
        self.uri = uri
        self._client = None
    
    async def connect(self):
        """Connect to Memgraph."""
        try:
            from neo4j import AsyncGraphDatabase
            self._client = AsyncGraphDatabase.driver(
                self.uri.replace("bolt://", "neo4j://"),
                auth=None
            )
            logger.info(f"Connected to Memgraph at {self.uri}")
        except ImportError:
            logger.warning("neo4j driver not installed, using REST fallback")
            self._client = None
    
    async def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results."""
        if not self._client:
            # Fallback to mgconsole via docker exec
            import subprocess
            result = subprocess.run(
                ["docker", "exec", "-i", "ai_nucleus_memgraph", "mgconsole", "-e", query],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                raise Exception(f"Query failed: {result.stderr}")
            return [{"output": result.stdout}]
        
        async with self._client.session() as session:
            result = await session.run(query)
            records = []
            async for record in result:
                records.append(dict(record))
            return records
    
    async def get_schema(self) -> Dict[str, Any]:
        """Get the graph schema."""
        node_types = await self.execute_query(
            "CALL schema.node_type_properties() YIELD nodeType, propertyName, propertyTypes "
            "RETURN nodeType, collect({name: propertyName, types: propertyTypes}) as properties"
        )
        
        rel_types = await self.execute_query(
            "CALL schema.rel_type_properties() YIELD relType, propertyName, propertyTypes "
            "RETURN relType, collect({name: propertyName, types: propertyTypes}) as properties"
        )
        
        return {
            "node_types": node_types,
            "relationship_types": rel_types
        }


class LLMClient:
    """Multi-LLM client with intelligent fallback."""
    
    def __init__(self, shimmy_url: str, langflow_url: str, model_server_url: str):
        self.shimmy_url = shimmy_url
        self.langflow_url = langflow_url
        self.model_server_url = model_server_url
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def generate(
        self,
        prompt: str,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Generate text using available LLM services with fallback."""
        # Try Shimmy first (fastest for simple completions)
        try:
            logger.info("Trying Shimmy...")
            response = await self.client.post(
                f"{self.shimmy_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.info("✅ Shimmy successful")
            return data["choices"][0]["text"]
        except Exception as e:
            logger.warning(f"Shimmy failed: {e}, trying Model Server...")
        
        # Try Model Server next
        try:
            response = await self.client.post(
                f"{self.model_server_url}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.info("✅ Model Server successful")
            return data["choices"][0]["text"]
        except Exception as e:
            logger.warning(f"Model Server failed: {e}, trying Langflow...")
        
        # Try Langflow as last resort
        try:
            response = await self.client.post(
                f"{self.langflow_url}/api/v1/run",
                json={
                    "input_value": prompt,
                    "output_type": "text",
                    "tweaks": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                },
                timeout=60.0
            )
            response.raise_for_status()
            data = response.json()
            logger.info("✅ Langflow successful")
            # Extract text from Langflow response
            if isinstance(data, dict) and "outputs" in data:
                return data["outputs"][0].get("outputs", [{}])[0].get("results", {}).get("text", {}).get("data", {}).get("text", "")
            return str(data)
        except Exception as e:
            logger.error(f"All LLM services failed. Langflow error: {e}")
            return f"Error: All LLM services unavailable. Last error: {str(e)}"
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = DEFAULT_MODEL,
        max_tokens: int = 2000,
        temperature: float = 0.7
    ) -> str:
        """Chat using available LLM services with fallback."""
        # Try Shimmy first
        try:
            logger.info("Trying Shimmy chat...")
            response = await self.client.post(
                f"{self.shimmy_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.info("✅ Shimmy chat successful")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Shimmy chat failed: {e}, trying Model Server...")
        
        # Try Model Server
        try:
            response = await self.client.post(
                f"{self.model_server_url}/v1/chat/completions",
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature
                },
                timeout=30.0
            )
            response.raise_for_status()
            data = response.json()
            logger.info("✅ Model Server chat successful")
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.warning(f"Model Server chat failed: {e}, falling back to completion format...")
        
        # Fallback to completion format using generate()
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
        return await self.generate(prompt, model, max_tokens, temperature)


# Initialize clients
memgraph = MemgraphClient(MEMGRAPH_URI)
llm = LLMClient(SHIMMY_URL, LANGFLOW_URL, MODEL_SERVER_URL)


@app.list_resources()
async def list_resources() -> List[Resource]:
    """List available graph resources."""
    return [
        Resource(
            uri="graph://schema",
            name="Graph Schema",
            mimeType="application/json",
            description="The current graph database schema including node and relationship types"
        ),
        Resource(
            uri="graph://stats",
            name="Graph Statistics",
            mimeType="application/json",
            description="Statistics about the graph database"
        )
    ]


@app.read_resource()
async def read_resource(uri: str) -> str:
    """Read a graph resource."""
    if uri == "graph://schema":
        schema = await memgraph.get_schema()
        return json.dumps(schema, indent=2)
    
    elif uri == "graph://stats":
        stats = await memgraph.execute_query(
            "MATCH (n) RETURN labels(n)[0] as type, count(*) as count"
        )
        return json.dumps(stats, indent=2)
    
    raise ValueError(f"Unknown resource: {uri}")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """List available tools."""
    return [
        Tool(
            name="execute_cypher",
            description="Execute a Cypher query against the graph database",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to execute"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="generate_cypher",
            description="Generate a Cypher query from a natural language question using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Natural language question about the graph"
                    },
                    "include_schema": {
                        "type": "boolean",
                        "description": "Include graph schema in the prompt",
                        "default": True
                    }
                },
                "required": ["question"]
            }
        ),
        Tool(
            name="explain_query",
            description="Explain what a Cypher query does in natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The Cypher query to explain"
                    }
                },
                "required": ["query"]
            }
        ),
        Tool(
            name="chat_about_graph",
            description="Have a conversation about the graph database",
            inputSchema={
                "type": "object",
                "properties": {
                    "message": {
                        "type": "string",
                        "description": "Your message or question"
                    },
                    "history": {
                        "type": "array",
                        "description": "Previous conversation history",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string"},
                                "content": {"type": "string"}
                            }
                        },
                        "default": []
                    }
                },
                "required": ["message"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> List[TextContent]:
    """Execute a tool."""
    
    if name == "execute_cypher":
        query = arguments["query"]
        try:
            results = await memgraph.execute_query(query)
            return [TextContent(
                type="text",
                text=json.dumps(results, indent=2, default=str)
            )]
        except Exception as e:
            return [TextContent(
                type="text",
                text=f"Error executing query: {str(e)}"
            )]
    
    elif name == "generate_cypher":
        question = arguments["question"]
        include_schema = arguments.get("include_schema", True)
        
        # Build prompt
        prompt = f"""You are a Cypher query expert for Memgraph graph database.

Given the following natural language question, generate a valid Cypher query.

"""
        
        if include_schema:
            schema = await memgraph.get_schema()
            prompt += f"""Database Schema:
{json.dumps(schema, indent=2)}

"""
        
        prompt += f"""Question: {question}

Generate ONLY the Cypher query, with no explanations or additional text.
Query:"""
        
        query = await llm.generate(prompt, temperature=0.3)
        query = query.strip().replace("```cypher", "").replace("```", "").strip()
        
        return [TextContent(
            type="text",
            text=query
        )]
    
    elif name == "explain_query":
        query = arguments["query"]
        
        prompt = f"""Explain the following Cypher query in simple, clear language:

```cypher
{query}
```

Explanation:"""
        
        explanation = await llm.generate(prompt, temperature=0.5)
        
        return [TextContent(
            type="text",
            text=explanation.strip()
        )]
    
    elif name == "chat_about_graph":
        message = arguments["message"]
        history = arguments.get("history", [])
        
        # Get current schema for context
        schema = await memgraph.get_schema()
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": f"""You are a helpful assistant that helps users understand and query their graph database.

Current Graph Schema:
{json.dumps(schema, indent=2)}

You can help users:
- Understand their data
- Write Cypher queries
- Analyze relationships
- Get insights from the graph
"""
            }
        ]
        
        # Add history
        messages.extend(history)
        
        # Add current message
        messages.append({
            "role": "user",
            "content": message
        })
        
        response = await llm.chat(messages, temperature=0.7)
        
        return [TextContent(
            type="text",
            text=response
        )]
    
    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    # Connect to Memgraph
    await memgraph.connect()
    
    # Run server
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
