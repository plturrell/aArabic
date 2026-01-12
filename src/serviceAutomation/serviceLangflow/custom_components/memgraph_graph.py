"""
Memgraph Graph Component for Langflow
Uses Memgraph AI Toolkit (langchain-memgraph) for graph operations
"""

from langflow.custom import CustomComponent
from langflow.field_typing import Text
from typing import Optional, Dict, List, Any
import os

try:
    from langchain_memgraph import MemgraphGraph
    MEMGRAPH_AVAILABLE = True
except ImportError:
    MEMGRAPH_AVAILABLE = False


class MemgraphGraphComponent(CustomComponent):
    display_name = "Memgraph Graph"
    description = "Query and store code graphs in Memgraph using LangChain integration"
    icon = "database"
    
    def build_config(self):
        return {
            "query": {
                "display_name": "Cypher Query",
                "info": "Cypher query to execute on Memgraph",
                "multiline": True,
            },
            "operation": {
                "display_name": "Operation",
                "options": ["query", "create_nodes", "create_relationships", "schema"],
                "value": "query",
                "info": "Type of operation to perform",
            },
            "memgraph_host": {
                "display_name": "Memgraph Host",
                "value": os.getenv("MEMGRAPH_HOST", "ai_nucleus_memgraph"),
                "advanced": True,
            },
            "memgraph_port": {
                "display_name": "Memgraph Port",
                "value": int(os.getenv("MEMGRAPH_PORT", "7687")),
                "advanced": True,
            },
        }
    
    def build(
        self,
        query: str = "",
        operation: str = "query",
        memgraph_host: str = "ai_nucleus_memgraph",
        memgraph_port: int = 7687,
    ) -> Dict[str, Any]:
        """Execute Memgraph operations using LangChain integration"""
        
        if not MEMGRAPH_AVAILABLE:
            return {
                "error": "langchain-memgraph not installed",
                "message": "Install: pip install langchain-memgraph",
                "status": "failed"
            }
        
        try:
            # Initialize Memgraph connection
            graph = MemgraphGraph(
                url=f"bolt://{memgraph_host}:{memgraph_port}",
                username="",  # Memgraph default has no auth
                password="",
            )
            
            result = {}
            
            if operation == "schema":
                # Get graph schema
                schema = graph.get_schema
                result = {
                    "operation": "schema",
                    "schema": schema,
                    "status": "success"
                }
            
            elif operation == "query":
                # Execute Cypher query
                if not query:
                    return {
                        "error": "Query is required for query operation",
                        "status": "failed"
                    }
                
                query_result = graph.query(query)
                result = {
                    "operation": "query",
                    "query": query,
                    "results": query_result,
                    "count": len(query_result) if isinstance(query_result, list) else 1,
                    "status": "success"
                }
            
            elif operation == "create_nodes":
                # Create nodes from query
                if not query:
                    return {
                        "error": "Query is required for create_nodes operation",
                        "status": "failed"
                    }
                
                graph.query(query)
                result = {
                    "operation": "create_nodes",
                    "query": query,
                    "message": "Nodes created successfully",
                    "status": "success"
                }
            
            elif operation == "create_relationships":
                # Create relationships from query
                if not query:
                    return {
                        "error": "Query is required for create_relationships operation",
                        "status": "failed"
                    }
                
                graph.query(query)
                result = {
                    "operation": "create_relationships",
                    "query": query,
                    "message": "Relationships created successfully",
                    "status": "success"
                }
            
            # Add connection info
            result["connection"] = {
                "host": memgraph_host,
                "port": memgraph_port,
                "url": f"bolt://{memgraph_host}:{memgraph_port}"
            }
            
            return result
            
        except Exception as e:
            return {
                "error": str(e),
                "operation": operation,
                "status": "failed",
                "connection": {
                    "host": memgraph_host,
                    "port": memgraph_port,
                }
            }


class MemgraphSCIPToGraphComponent(CustomComponent):
    display_name = "SCIP to Memgraph"
    description = "Convert SCIP index to Memgraph graph structure"
    icon = "share-2"
    
    def build_config(self):
        return {
            "scip_data": {
                "display_name": "SCIP Data",
                "info": "SCIP index data (from SCIP Indexer component)",
            },
            "project_name": {
                "display_name": "Project Name",
                "info": "Name for the project in Memgraph",
            },
            "memgraph_host": {
                "display_name": "Memgraph Host",
                "value": os.getenv("MEMGRAPH_HOST", "ai_nucleus_memgraph"),
                "advanced": True,
            },
        }
    
    def build(
        self,
        scip_data: Dict[str, Any],
        project_name: str,
        memgraph_host: str = "ai_nucleus_memgraph",
    ) -> Dict[str, Any]:
        """Convert SCIP index to Memgraph graph"""
        
        if not MEMGRAPH_AVAILABLE:
            return {
                "error": "langchain-memgraph not installed",
                "status": "failed"
            }
        
        try:
            graph = MemgraphGraph(
                url=f"bolt://{memgraph_host}:7687",
                username="",
                password="",
            )
            
            # Create project node
            create_project_query = f"""
            MERGE (p:Project {{name: '{project_name}'}})
            SET p.language = '{scip_data.get('language', 'unknown')}',
                p.indexed_at = datetime(),
                p.symbols_count = {scip_data.get('symbols_count', 0)},
                p.documents_count = {scip_data.get('documents_count', 0)}
            RETURN p
            """
            
            graph.query(create_project_query)
            
            # TODO: Parse SCIP index and create symbol nodes and relationships
            # This would involve reading the SCIP protobuf format
            # For now, we create a placeholder structure
            
            return {
                "project_name": project_name,
                "status": "success",
                "message": f"Created project node for {project_name}",
                "memgraph_host": memgraph_host,
                "next_steps": "Add symbol and relationship creation from SCIP data"
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "status": "failed"
            }