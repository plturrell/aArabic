"""
NucleusGraph Adapter
Integration layer for the NucleusGraph knowledge graph visualization service.
Provides graph querying, visualization, and knowledge management.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)
    node_type: Optional[str] = None


@dataclass
class GraphEdge:
    """Represents an edge in the knowledge graph."""
    id: str
    source: str
    target: str
    label: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphQueryResult:
    """Result of a graph query."""
    nodes: List[GraphNode] = field(default_factory=list)
    edges: List[GraphEdge] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NucleusGraphHealthStatus:
    """Health status of the NucleusGraph service."""
    healthy: bool
    status: str
    memgraph_connected: bool = False
    error: Optional[str] = None


class NucleusGraphAdapter:
    """
    Adapter for interacting with NucleusGraph service.

    Provides methods for:
    - Graph querying (Cypher)
    - Node/edge management
    - Visualization data retrieval
    - Knowledge extraction
    - Health monitoring
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url or os.getenv("NUCLEUS_GRAPH_URL", "http://localhost:3005")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> NucleusGraphHealthStatus:
        """Check the health of NucleusGraph."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    data = await response.json()
                    return NucleusGraphHealthStatus(
                        healthy=True,
                        status="healthy",
                        memgraph_connected=data.get("memgraph_connected", False)
                    )
                return NucleusGraphHealthStatus(healthy=False, status="unhealthy")
        except Exception as e:
            return NucleusGraphHealthStatus(
                healthy=False,
                status="unavailable",
                error=str(e)
            )

    async def execute_query(self, cypher: str) -> GraphQueryResult:
        """Execute a Cypher query."""
        session = await self._get_session()
        async with session.post(
            f"{self.base_url}/api/query",
            json={"query": cypher}
        ) as response:
            data = await response.json()

            nodes = [
                GraphNode(
                    id=n.get("id", str(i)),
                    label=n.get("label", ""),
                    properties=n.get("properties", {}),
                    node_type=n.get("type")
                )
                for i, n in enumerate(data.get("nodes", []))
            ]

            edges = [
                GraphEdge(
                    id=e.get("id", str(i)),
                    source=e.get("source", ""),
                    target=e.get("target", ""),
                    label=e.get("label", ""),
                    properties=e.get("properties", {})
                )
                for i, e in enumerate(data.get("edges", []))
            ]

            return GraphQueryResult(
                nodes=nodes,
                edges=edges,
                metadata=data.get("metadata", {})
            )

    async def get_graph_data(
        self,
        node_limit: int = 100,
        include_edges: bool = True
    ) -> GraphQueryResult:
        """Get graph visualization data."""
        session = await self._get_session()
        params = {
            "limit": node_limit,
            "edges": str(include_edges).lower()
        }
        async with session.get(
            f"{self.base_url}/api/graph",
            params=params
        ) as response:
            data = await response.json()
            return GraphQueryResult(
                nodes=[GraphNode(**n) for n in data.get("nodes", [])],
                edges=[GraphEdge(**e) for e in data.get("edges", [])]
            )

    async def create_node(self, node: GraphNode) -> GraphNode:
        """Create a new node."""
        session = await self._get_session()
        payload = {
            "label": node.label,
            "properties": node.properties,
            "type": node.node_type
        }
        async with session.post(
            f"{self.base_url}/api/nodes",
            json=payload
        ) as response:
            data = await response.json()
            return GraphNode(
                id=data.get("id"),
                label=data.get("label", node.label),
                properties=data.get("properties", {}),
                node_type=data.get("type")
            )

    async def create_edge(self, edge: GraphEdge) -> GraphEdge:
        """Create a new edge."""
        session = await self._get_session()
        payload = {
            "source": edge.source,
            "target": edge.target,
            "label": edge.label,
            "properties": edge.properties
        }
        async with session.post(
            f"{self.base_url}/api/edges",
            json=payload
        ) as response:
            data = await response.json()
            return GraphEdge(
                id=data.get("id"),
                source=data.get("source", edge.source),
                target=data.get("target", edge.target),
                label=data.get("label", edge.label),
                properties=data.get("properties", {})
            )

    async def search_nodes(
        self,
        query: str,
        node_type: Optional[str] = None,
        limit: int = 50
    ) -> List[GraphNode]:
        """Search for nodes."""
        session = await self._get_session()
        params = {"q": query, "limit": limit}
        if node_type:
            params["type"] = node_type

        async with session.get(
            f"{self.base_url}/api/search",
            params=params
        ) as response:
            data = await response.json()
            return [
                GraphNode(
                    id=n.get("id"),
                    label=n.get("label", ""),
                    properties=n.get("properties", {}),
                    node_type=n.get("type")
                )
                for n in data.get("results", [])
            ]

    async def get_node_neighbors(
        self,
        node_id: str,
        depth: int = 1
    ) -> GraphQueryResult:
        """Get neighboring nodes and edges."""
        session = await self._get_session()
        params = {"depth": depth}
        async with session.get(
            f"{self.base_url}/api/nodes/{node_id}/neighbors",
            params=params
        ) as response:
            data = await response.json()
            return GraphQueryResult(
                nodes=[GraphNode(**n) for n in data.get("nodes", [])],
                edges=[GraphEdge(**e) for e in data.get("edges", [])]
            )

    async def get_shortest_path(
        self,
        source_id: str,
        target_id: str
    ) -> GraphQueryResult:
        """Find shortest path between two nodes."""
        session = await self._get_session()
        async with session.get(
            f"{self.base_url}/api/path",
            params={"source": source_id, "target": target_id}
        ) as response:
            data = await response.json()
            return GraphQueryResult(
                nodes=[GraphNode(**n) for n in data.get("nodes", [])],
                edges=[GraphEdge(**e) for e in data.get("edges", [])]
            )

    async def import_data(
        self,
        nodes: List[Dict[str, Any]],
        edges: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """Bulk import nodes and edges."""
        session = await self._get_session()
        payload = {"nodes": nodes, "edges": edges}
        async with session.post(
            f"{self.base_url}/api/import",
            json=payload
        ) as response:
            return await response.json()

    async def clear_graph(self) -> bool:
        """Clear all graph data."""
        session = await self._get_session()
        async with session.delete(f"{self.base_url}/api/graph") as response:
            return response.status == 200


async def check_nucleusgraph_health(
    base_url: Optional[str] = None
) -> NucleusGraphHealthStatus:
    """Quick health check for NucleusGraph."""
    adapter = NucleusGraphAdapter(base_url=base_url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()


# Alias for backward compatibility
# NucleusGraphAdapter - no main class exists, adapter provides utility functions


async def check_nucleusgraph_health(nucleusgraph_url: str = "http://nucleus-graph:5000") -> Dict[str, Any]:
    """
    Check NucleusGraph service health
    
    Args:
        nucleusgraph_url: Base URL for NucleusGraph service
        
    Returns:
        Health check result
    """
    service = NucleusGraphService(base_url=nucleusgraph_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
