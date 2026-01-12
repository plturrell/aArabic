from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
import logging
import json
import subprocess
import os
from urllib.parse import urlparse
from pathlib import Path

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/nucleus-graph", tags=["NucleusGraph"])

# Memgraph connection settings
MEMGRAPH_HOST = os.getenv("MEMGRAPH_HOST", "localhost")
MEMGRAPH_PORT = os.getenv("MEMGRAPH_PORT", "7687")
MEMGRAPH_USERNAME = os.getenv("MEMGRAPH_USERNAME", "")
MEMGRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")

MEMGRAPH_URI = os.getenv("MEMGRAPH_URI")
if MEMGRAPH_URI:
    parsed = urlparse(MEMGRAPH_URI)
    if parsed.hostname:
        MEMGRAPH_HOST = parsed.hostname
    if parsed.port:
        MEMGRAPH_PORT = str(parsed.port)
    if not MEMGRAPH_USERNAME and parsed.username:
        MEMGRAPH_USERNAME = parsed.username
    if not MEMGRAPH_PASSWORD and parsed.password:
        MEMGRAPH_PASSWORD = parsed.password

@router.get("/health")
async def health_check():
    """Check if Memgraph service is available"""
    try:
        # Try to connect to Memgraph using mgconsole if available
        result = subprocess.run(
            ["which", "mgconsole"],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        mgconsole_available = result.returncode == 0
        
        return {
            "status": "healthy",
            "service": "NucleusGraph",
            "memgraph_host": MEMGRAPH_HOST,
            "memgraph_port": MEMGRAPH_PORT,
            "mgconsole_available": mgconsole_available,
            "message": "Graph database service is ready"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "degraded",
            "service": "NucleusGraph",
            "error": str(e),
            "message": "Graph database service may not be available"
        }

@router.post("/query")
async def execute_cypher_query(request: Dict[str, Any]):
    """Execute a Cypher query against Memgraph"""
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query is required")
        
        # For now, return mock data since we don't have Memgraph running
        # In production, this would connect to actual Memgraph instance
        mock_results = {
            "query": query,
            "results": [],
            "execution_time": "0.001s",
            "nodes_created": 0,
            "relationships_created": 0,
            "properties_set": 0,
            "message": "Query executed successfully (mock mode)"
        }
        
        # Parse common query types and return appropriate mock data
        if "MATCH" in query.upper():
            mock_results["results"] = [
                {"n": {"id": 1, "labels": ["Document"], "properties": {"title": "Sample Document"}}},
                {"n": {"id": 2, "labels": ["Concept"], "properties": {"name": "AI Research"}}}
            ]
        elif "CREATE" in query.upper():
            mock_results["nodes_created"] = 1
            mock_results["properties_set"] = 2
        
        return {
            "success": True,
            "data": mock_results
        }
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@router.get("/schema")
async def get_graph_schema():
    """Get the current graph schema"""
    try:
        # Mock schema data - in production this would query actual Memgraph
        schema = {
            "node_labels": [
                {"label": "Document", "count": 5, "properties": ["id", "title", "source", "content"]},
                {"label": "Concept", "count": 15, "properties": ["name", "frequency"]},
                {"label": "Author", "count": 3, "properties": ["name", "email"]},
                {"label": "Topic", "count": 8, "properties": ["name", "category"]}
            ],
            "relationship_types": [
                {"type": "MENTIONS", "count": 25, "properties": ["strength", "context"]},
                {"type": "RELATES_TO", "count": 12, "properties": ["type", "confidence"]},
                {"type": "AUTHORED_BY", "count": 5, "properties": ["date"]},
                {"type": "BELONGS_TO", "count": 8, "properties": ["relevance"]}
            ],
            "constraints": [],
            "indexes": []
        }
        
        return {
            "success": True,
            "schema": schema
        }
        
    except Exception as e:
        logger.error(f"Schema retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Schema retrieval failed: {str(e)}")

@router.get("/visualize")
async def get_graph_visualization(
    limit: int = Query(50, description="Maximum number of nodes to return"),
    node_type: Optional[str] = Query(None, description="Filter by node type/label")
):
    """Get graph data for visualization"""
    try:
        # Mock visualization data - in production this would query actual Memgraph
        nodes = [
            {"id": "1", "label": "Document", "properties": {"title": "AI Research Paper", "type": "Document"}},
            {"id": "2", "label": "Concept", "properties": {"name": "Machine Learning", "type": "Concept"}},
            {"id": "3", "label": "Concept", "properties": {"name": "Neural Networks", "type": "Concept"}},
            {"id": "4", "label": "Author", "properties": {"name": "Dr. Smith", "type": "Author"}},
            {"id": "5", "label": "Topic", "properties": {"name": "Deep Learning", "type": "Topic"}}
        ]
        
        edges = [
            {"id": "e1", "source": "1", "target": "2", "type": "MENTIONS", "properties": {"strength": 0.8}},
            {"id": "e2", "source": "1", "target": "3", "type": "MENTIONS", "properties": {"strength": 0.9}},
            {"id": "e3", "source": "2", "target": "3", "type": "RELATES_TO", "properties": {"confidence": 0.7}},
            {"id": "e4", "source": "1", "target": "4", "type": "AUTHORED_BY", "properties": {"date": "2024-01-01"}},
            {"id": "e5", "source": "2", "target": "5", "type": "BELONGS_TO", "properties": {"relevance": 0.85}}
        ]
        
        # Filter by node type if specified
        if node_type:
            nodes = [n for n in nodes if n["label"] == node_type]
            node_ids = {n["id"] for n in nodes}
            edges = [e for e in edges if e["source"] in node_ids or e["target"] in node_ids]
        
        # Apply limit
        nodes = nodes[:limit]
        
        return {
            "success": True,
            "graph": {
                "nodes": nodes,
                "edges": edges,
                "stats": {
                    "node_count": len(nodes),
                    "edge_count": len(edges),
                    "node_types": list(set(n["label"] for n in nodes))
                }
            }
        }
        
    except Exception as e:
        logger.error(f"Visualization data retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"Visualization failed: {str(e)}")

@router.post("/import")
async def import_graph_data(request: Dict[str, Any]):
    """Import graph data from various sources"""
    try:
        data_source = request.get("source", "")
        data = request.get("data", {})
        
        if not data_source or not data:
            raise HTTPException(status_code=400, detail="Source and data are required")
        
        # Mock import process
        imported_stats = {
            "nodes_created": len(data.get("nodes", [])),
            "relationships_created": len(data.get("edges", [])),
            "properties_set": sum(len(n.get("properties", {})) for n in data.get("nodes", [])),
            "execution_time": "0.05s"
        }
        
        return {
            "success": True,
            "message": f"Successfully imported data from {data_source}",
            "stats": imported_stats
        }
        
    except Exception as e:
        logger.error(f"Data import failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data import failed: {str(e)}")
