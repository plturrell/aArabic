"""
Memgraph Graph Database Adapter
Provides workflow dependency tracking and relationship management
Using neo4j driver for ARM64 compatibility
"""

import asyncio
import logging
import os
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
from neo4j import GraphDatabase, Driver, AsyncGraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError

logger = logging.getLogger(__name__)


class NodeStatus(str, Enum):
    """Workflow node status enumeration"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowGraphNode:
    """Workflow node representation for graph database"""
    id: str
    workflow_id: str
    name: str
    node_type: str
    status: NodeStatus = NodeStatus.PENDING
    metadata: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class WorkflowGraphEdge:
    """Workflow edge representation for graph database"""
    source_id: str
    target_id: str
    workflow_id: str
    edge_type: str = "DEPENDS_ON"
    metadata: Optional[Dict[str, Any]] = None


class MemgraphService:
    """
    Enhanced Memgraph integration for workflow management
    Provides graph-based workflow dependency tracking and analytics
    Uses neo4j driver for ARM64 compatibility
    """
    
    def __init__(self, host: str = "localhost", port: int = 7687, username: str = "", password: str = ""):
        env_uri = os.getenv("MEMGRAPH_URI")
        if env_uri and host == "localhost" and port == 7687:
            parsed = urlparse(env_uri)
            if parsed.hostname:
                host = parsed.hostname
            if parsed.port:
                port = parsed.port
            if not username and parsed.username:
                username = parsed.username
            if not password and parsed.password:
                password = parsed.password

        env_host = os.getenv("MEMGRAPH_HOST")
        if env_host and host == "localhost":
            host = env_host

        env_port = os.getenv("MEMGRAPH_PORT")
        if env_port and port == 7687:
            try:
                port = int(env_port)
            except ValueError:
                logger.warning("Invalid MEMGRAPH_PORT value: %s", env_port)

        env_username = os.getenv("MEMGRAPH_USERNAME")
        env_password = os.getenv("MEMGRAPH_PASSWORD")
        if env_username and not username:
            username = env_username
        if env_password and not password:
            password = env_password

        self.host = host
        self.port = port
        self.username = username or ""
        self.password = password or ""
        self.uri = f"bolt://{self.host}:{self.port}"
        self.driver: Optional[Driver] = None
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to Memgraph database using neo4j driver"""
        try:
            # Create driver (handles connection pooling internally)
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password) if self.username else None,
                max_connection_lifetime=3600,
                max_connection_pool_size=50,
                connection_acquisition_timeout=60
            )
            
            # Test connection
            with self.driver.session() as session:
                result = session.run("RETURN 1 AS test")
                test_result = result.single()
                
                if test_result and test_result["test"] == 1:
                    self.connected = True
                    logger.info(f"Connected to Memgraph at {self.host}:{self.port}")
                    
                    # Initialize schema
                    await self._initialize_schema()
                    return True
                else:
                    logger.error("Memgraph connection test failed")
                    return False
                    
        except (ServiceUnavailable, AuthError) as e:
            logger.warning(f"Failed to connect to Memgraph: {e}")
            logger.info("Workflow execution will continue without graph database")
            self.connected = False
            return False
        except Exception as e:
            logger.warning(f"Unexpected error connecting to Memgraph: {e}")
            self.connected = False
            return False
    
    async def _initialize_schema(self):
        """Initialize Memgraph schema for workflow tracking"""
        try:
            # Create indexes for better performance
            schema_queries = [
                "CREATE INDEX ON :WorkflowNode(id);",
                "CREATE INDEX ON :WorkflowNode(workflow_id);",
                "CREATE INDEX ON :WorkflowNode(status);",
                "CREATE INDEX ON :Workflow(id);",
                "CREATE INDEX ON :Workflow(status);"
            ]
            
            with self.driver.session() as session:
                for query in schema_queries:
                    try:
                        session.run(query)
                    except Exception as e:
                        # Index might already exist
                        logger.debug(f"Schema query failed (might already exist): {e}")
                        
            logger.info("Memgraph schema initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Memgraph schema: {e}")
    
    def is_connected(self) -> bool:
        """Check if connected to Memgraph"""
        return self.connected and self.driver is not None
    
    async def close(self):
        """Close connection to Memgraph"""
        if self.driver:
            try:
                self.driver.close()
                logger.info(f"Closed Memgraph connection for {self.host}:{self.port}")
            except Exception as e:
                logger.error(f"Error closing Memgraph connection: {e}")
            finally:
                self.driver = None
                self.connected = False
    
    async def create_workflow_graph(
        self,
        workflow_id: str,
        workflow_name: str,
        nodes: List[WorkflowGraphNode],
        edges: List[WorkflowGraphEdge],
        workflow_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Create workflow graph in Memgraph"""
        if not self.is_connected():
            logger.debug("Memgraph not connected, skipping graph creation")
            return False
            
        try:
            with self.driver.session() as session:
                # Clear existing workflow graph
                session.run(
                    "MATCH (n:WorkflowNode {workflow_id: $workflow_id}) DETACH DELETE n",
                    workflow_id=workflow_id
                )
                
                # Delete existing workflow node
                session.run(
                    "MATCH (w:Workflow {id: $workflow_id}) DELETE w",
                    workflow_id=workflow_id
                )
                
                # Create workflow node
                session.run(
                    """
                    CREATE (w:Workflow {
                        id: $id,
                        name: $name,
                        status: $status,
                        created_at: datetime(),
                        metadata: $metadata
                    })
                    """,
                    id=workflow_id,
                    name=workflow_name,
                    status="created",
                    metadata=json.dumps(workflow_metadata or {})
                )
                
                # Create workflow nodes
                for node in nodes:
                    session.run(
                        """
                        CREATE (n:WorkflowNode {
                            id: $id,
                            workflow_id: $workflow_id,
                            name: $name,
                            node_type: $node_type,
                            status: $status,
                            metadata: $metadata,
                            created_at: datetime()
                        })
                        """,
                        id=node.id,
                        workflow_id=node.workflow_id,
                        name=node.name,
                        node_type=node.node_type,
                        status=node.status.value,
                        metadata=json.dumps(node.metadata or {})
                    )
                
                # Create edges between nodes
                for edge in edges:
                    session.run(
                        """
                        MATCH (source:WorkflowNode {id: $source_id, workflow_id: $workflow_id})
                        MATCH (target:WorkflowNode {id: $target_id, workflow_id: $workflow_id})
                        CREATE (source)-[:DEPENDS_ON {
                            edge_type: $edge_type,
                            metadata: $metadata,
                            created_at: datetime()
                        }]->(target)
                        """,
                        source_id=edge.source_id,
                        target_id=edge.target_id,
                        workflow_id=edge.workflow_id,
                        edge_type=edge.edge_type,
                        metadata=json.dumps(edge.metadata or {})
                    )
            
            logger.info(f"Created workflow graph for {workflow_id} with {len(nodes)} nodes and {len(edges)} edges")
            return True

        except Exception as e:
            logger.error(f"Failed to create workflow graph: {e}")
            return False

    async def update_node_status(
        self,
        workflow_id: str,
        node_id: str,
        status: NodeStatus,
        execution_time: Optional[float] = None,
        error_message: Optional[str] = None,
        result_metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Update workflow node status"""
        if not self.is_connected():
            return False

        try:
            with self.driver.session() as session:
                # Build update parameters
                params = {
                    "node_id": node_id,
                    "workflow_id": workflow_id,
                    "status": status.value
                }
                
                set_clauses = ["n.status = $status", "n.updated_at = datetime()"]
                
                if execution_time is not None:
                    params["execution_time"] = execution_time
                    set_clauses.append("n.execution_time = $execution_time")
                if error_message:
                    params["error_message"] = error_message
                    set_clauses.append("n.error_message = $error_message")
                if result_metadata:
                    params["result_metadata"] = json.dumps(result_metadata)
                    set_clauses.append("n.result_metadata = $result_metadata")
                
                query = f"""
                MATCH (n:WorkflowNode {{id: $node_id, workflow_id: $workflow_id}})
                SET {', '.join(set_clauses)}
                """
                
                session.run(query, **params)

            logger.debug(f"Updated node {node_id} status to {status.value}")
            return True

        except Exception as e:
            logger.error(f"Failed to update node status: {e}")
            return False

    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status"""
        if not self.is_connected():
            return {"error": "Not connected to Memgraph"}

        try:
            with self.driver.session() as session:
                # Get workflow info
                workflow_result = session.run(
                    """
                    MATCH (w:Workflow {id: $workflow_id})
                    RETURN w.name AS name, w.status AS status, w.created_at AS created_at
                    """,
                    workflow_id=workflow_id
                )
                workflow_record = workflow_result.single()

                if not workflow_record:
                    return {"error": "Workflow not found"}

                # Get node statuses
                nodes_result = session.run(
                    """
                    MATCH (n:WorkflowNode {workflow_id: $workflow_id})
                    RETURN n.id AS id, n.name AS name, n.status AS status,
                           n.execution_time AS execution_time, n.error_message AS error_message
                    ORDER BY n.id
                    """,
                    workflow_id=workflow_id
                )

                nodes = [dict(record) for record in nodes_result]

                # Calculate statistics
                total_nodes = len(nodes)
                status_counts = {}
                total_execution_time = 0

                for node in nodes:
                    status = node["status"]
                    status_counts[status] = status_counts.get(status, 0) + 1

                    if node["execution_time"]:
                        total_execution_time += node["execution_time"]

                return {
                    "workflow_id": workflow_id,
                    "name": workflow_record["name"],
                    "status": workflow_record["status"],
                    "created_at": str(workflow_record["created_at"]),
                    "total_nodes": total_nodes,
                    "status_counts": status_counts,
                    "total_execution_time": total_execution_time,
                    "nodes": nodes
                }

        except Exception as e:
            logger.error(f"Failed to get workflow status: {e}")
            return {"error": str(e)}

    async def get_workflow_dependencies(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get workflow node dependencies"""
        if not self.is_connected():
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (source:WorkflowNode {workflow_id: $workflow_id})-[r:DEPENDS_ON]->(target:WorkflowNode)
                    RETURN source.id AS source_id, source.name AS source_name,
                           target.id AS target_id, target.name AS target_name,
                           r.edge_type AS edge_type
                    ORDER BY source.id, target.id
                    """,
                    workflow_id=workflow_id
                )

                return [dict(record) for record in result]

        except Exception as e:
            logger.error(f"Failed to get workflow dependencies: {e}")
            return []

    async def find_ready_nodes(self, workflow_id: str) -> List[str]:
        """Find nodes that are ready to execute (all dependencies completed)"""
        if not self.is_connected():
            return []

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:WorkflowNode {workflow_id: $workflow_id, status: 'pending'})
                    WHERE NOT EXISTS {
                        MATCH (dep:WorkflowNode)-[:DEPENDS_ON]->(n)
                        WHERE dep.status <> 'completed'
                    }
                    RETURN n.id AS node_id
                    ORDER BY n.id
                    """,
                    workflow_id=workflow_id
                )

                return [record["node_id"] for record in result]

        except Exception as e:
            logger.error(f"Failed to find ready nodes: {e}")
            return []

    async def get_workflow_analytics(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow execution analytics"""
        if not self.is_connected():
            return {}

        try:
            with self.driver.session() as session:
                result = session.run(
                    """
                    MATCH (n:WorkflowNode {workflow_id: $workflow_id})
                    RETURN
                        COUNT(n) AS total_nodes,
                        SUM(CASE WHEN n.status = 'completed' THEN 1 ELSE 0 END) AS completed_nodes,
                        SUM(CASE WHEN n.status = 'failed' THEN 1 ELSE 0 END) AS failed_nodes,
                        SUM(CASE WHEN n.status = 'running' THEN 1 ELSE 0 END) AS running_nodes,
                        SUM(CASE WHEN n.status = 'pending' THEN 1 ELSE 0 END) AS pending_nodes,
                        AVG(n.execution_time) AS avg_execution_time,
                        MAX(n.execution_time) AS max_execution_time,
                        MIN(n.execution_time) AS min_execution_time
                    """,
                    workflow_id=workflow_id
                )

                record = result.single()
                if record:
                    analytics = dict(record)
                    total = analytics["total_nodes"]
                    completed = analytics["completed_nodes"] or 0
                    analytics["completion_rate"] = (completed / total * 100) if total > 0 else 0
                    return analytics
                else:
                    return {}

        except Exception as e:
            logger.error(f"Failed to get workflow analytics: {e}")
            return {}


# Alias for backward compatibility
MemgraphAdapter = MemgraphService


async def check_memgraph_health(memgraph_url: str = "bolt://memgraph:7687") -> Dict[str, Any]:
    """
    Check Memgraph service health
    
    Args:
        memgraph_url: Bolt URL for Memgraph service
        
    Returns:
        Health check result
    """
    # Parse URL to extract host and port
    parsed = urlparse(memgraph_url)
    host = parsed.hostname or "memgraph"
    port = parsed.port or 7687
    
    service = MemgraphService(host=host, port=port)
    try:
        connected = await service.connect()
        if connected:
            return {
                "status": "healthy",
                "service": "memgraph",
                "host": host,
                "port": port,
                "connected": True
            }
        else:
            return {
                "status": "unhealthy",
                "service": "memgraph",
                "host": host,
                "port": port,
                "connected": False,
                "error": "Failed to connect"
            }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "memgraph",
            "host": host,
            "port": port,
            "connected": False,
            "error": str(e)
        }
    finally:
        await service.close()
