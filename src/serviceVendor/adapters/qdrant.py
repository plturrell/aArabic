"""
Qdrant Vector Database Adapter
Provides vector similarity search and semantic processing capabilities
Works alongside Memgraph for comprehensive AI workflow management
"""

import asyncio
import logging
import os
from urllib.parse import urlparse
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import uuid
from datetime import datetime

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, CreateCollection, PointStruct,
        Filter, FieldCondition, MatchValue, SearchRequest,
        UpdateCollection, OptimizersConfigDiff, ScalarQuantization,
        QuantizationType, CompressionRatio
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

logger = logging.getLogger(__name__)

if not QDRANT_AVAILABLE:
    logger.warning("Qdrant client not available. Install with: pip install qdrant-client")


class VectorType(str, Enum):
    """Vector type enumeration"""
    WORKFLOW_EMBEDDING = "workflow_embedding"
    DOCUMENT_EMBEDDING = "document_embedding"
    INVOICE_EMBEDDING = "invoice_embedding"
    SEMANTIC_SEARCH = "semantic_search"
    TOOL_EMBEDDING = "tool_embedding"
    A2UI_COMPONENT = "a2ui_component"


@dataclass
class VectorPoint:
    """Vector point representation for Qdrant"""
    id: str
    vector: List[float]
    payload: Dict[str, Any]
    vector_type: VectorType


@dataclass
class SearchResult:
    """Vector search result"""
    id: str
    score: float
    payload: Dict[str, Any]
    vector: Optional[List[float]] = None


@dataclass
class CollectionInfo:
    """Collection information"""
    name: str
    vectors_count: int
    indexed_vectors_count: int
    points_count: int
    segments_count: int
    config: Dict[str, Any]


class QdrantAdapter:
    """Qdrant vector database adapter for AI workflow processing"""
    
    # Class-level client pool so multiple adapters (or reinitializations)
    # can share a single underlying QdrantClient per (host, port, api_key, https, prefer_grpc)
    _client_pool: Dict[Tuple[str, int, int, str, bool, bool], QdrantClient] = {}
    _client_pool_ref_counts: Dict[Tuple[str, int, int, str, bool, bool], int] = {}
    _pool_lock: asyncio.Lock = asyncio.Lock()

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        grpc_port: int = 6334,
        prefer_grpc: bool = False,
        api_key: Optional[str] = None,
        https: bool = False,
        timeout: int = 30
    ):
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available. Some features will be disabled.")
            self.client = None
            self.collections = {}
            return
        
        self.host = host
        self.port = port
        self.grpc_port = grpc_port
        self.prefer_grpc = prefer_grpc
        self.api_key = api_key
        self.https = https
        self.timeout = timeout
        
        self.client: Optional[QdrantClient] = None
        self.collections: Dict[str, CollectionInfo] = {}
        # Key used for this adapter instance in the shared pool
        self._pool_key: Optional[Tuple[str, int, int, str, bool, bool]] = None
        
        # Default collection configurations
        self.default_collections = {
            "workflows": {
                "vector_size": 768,  # Standard embedding size
                "distance": Distance.COSINE,
                "description": "Workflow embeddings and metadata"
            },
            "documents": {
                "vector_size": 1536,  # OpenAI embedding size
                "distance": Distance.COSINE,
                "description": "Document embeddings for semantic search"
            },
            "invoices": {
                "vector_size": 768,
                "distance": Distance.COSINE,
                "description": "Arabic invoice embeddings and processing data"
            },
            "tools": {
                "vector_size": 384,  # Smaller embedding for tool descriptions
                "distance": Distance.COSINE,
                "description": "Tool embeddings for intelligent orchestration"
            },
            "a2ui_components": {
                "vector_size": 512,
                "distance": Distance.COSINE,
                "description": "A2UI component embeddings for dynamic UI generation"
            }
        }

    def _check_availability(self) -> bool:
        """Check if Qdrant client is available"""
        if not QDRANT_AVAILABLE:
            logger.warning("Qdrant client not available - operation skipped")
            return False
        return True

    async def initialize(self) -> bool:
        """Initialize Qdrant client and create default collections"""
        if not self._check_availability():
            return False

        try:
            # Initialize or reuse pooled client
            pool_key: Tuple[str, int, int, str, bool, bool] = (
                self.host,
                self.port,
                self.grpc_port,
                self.api_key or "",
                self.https,
                self.prefer_grpc,
            )
            self._pool_key = pool_key

            async with self._pool_lock:
                pooled_client = self._client_pool.get(pool_key)
                if pooled_client is not None:
                    self.client = pooled_client
                    self._client_pool_ref_counts[pool_key] = (
                        self._client_pool_ref_counts.get(pool_key, 0) + 1
                    )
                    logger.info(
                        "Reusing pooled Qdrant client for %s (ref_count=%d)",
                        f"{self.host}:{self.grpc_port if self.prefer_grpc else self.port}",
                        self._client_pool_ref_counts[pool_key],
                    )
                else:
                    if self.prefer_grpc:
                        self.client = QdrantClient(
                            host=self.host,
                            grpc_port=self.grpc_port,
                            api_key=self.api_key,
                            https=self.https,
                            timeout=self.timeout,
                            prefer_grpc=True,
                        )
                    else:
                        self.client = QdrantClient(
                            host=self.host,
                            port=self.port,
                            api_key=self.api_key,
                            https=self.https,
                            timeout=self.timeout,
                        )

                    self._client_pool[pool_key] = self.client
                    self._client_pool_ref_counts[pool_key] = 1
                    logger.info(
                        "Created new Qdrant client for %s",
                        f"{self.host}:{self.grpc_port if self.prefer_grpc else self.port}",
                    )
            
            # Test connection
            health = await self.health_check()
            if not health:
                logger.error("Qdrant health check failed")
                return False
            
            # Create default collections
            await self.create_default_collections()
            
            logger.info(f"Qdrant adapter initialized successfully on {self.host}:{self.port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant adapter: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Qdrant service health"""
        if not self._check_availability():
            return False

        try:
            if not self.client:
                return False
            
            # Use sync method in async context
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            return True
            
        except Exception as e:
            logger.warning(f"Qdrant health check failed: {e}")
            return False
    
    async def create_default_collections(self):
        """Create default collections for AI workflow processing"""
        for collection_name, config in self.default_collections.items():
            try:
                await self.create_collection(
                    collection_name=collection_name,
                    vector_size=config["vector_size"],
                    distance=config["distance"],
                    description=config.get("description", "")
                )
            except Exception as e:
                logger.warning(f"Failed to create collection {collection_name}: {e}")
    
    async def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: Distance = Distance.COSINE,
        description: str = "",
        shard_number: int = 1,
        replication_factor: int = 1,
        write_consistency_factor: int = 1,
        on_disk_payload: bool = True,
        hnsw_config: Optional[Dict] = None,
        optimizers_config: Optional[Dict] = None,
        quantization_config: Optional[Dict] = None
    ) -> bool:
        """Create a new collection with specified configuration"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")
            
            # Check if collection already exists
            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )
            
            existing_names = [col.name for col in collections.collections]
            if collection_name in existing_names:
                logger.info(f"Collection {collection_name} already exists")
                return True
            
            # Prepare HNSW configuration
            hnsw_config = hnsw_config or {
                "m": 16,
                "ef_construct": 100,
                "full_scan_threshold": 10000,
                "max_indexing_threads": 0,
                "on_disk": False,
                "payload_m": None
            }
            
            # Prepare optimizers configuration
            optimizers_config = optimizers_config or {
                "deleted_threshold": 0.2,
                "vacuum_min_vector_number": 1000,
                "default_segment_number": 0,
                "max_segment_size": None,
                "memmap_threshold": None,
                "indexing_threshold": 20000,
                "flush_interval_sec": 5,
                "max_optimization_threads": None
            }
            
            # Create collection
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=distance,
                        hnsw_config=models.HnswConfigDiff(**hnsw_config),
                        on_disk=on_disk_payload
                    ),
                    shard_number=shard_number,
                    replication_factor=replication_factor,
                    write_consistency_factor=write_consistency_factor,
                    optimizers_config=OptimizersConfigDiff(**optimizers_config),
                    quantization_config=quantization_config
                )
            )
            
            logger.info(f"Created Qdrant collection: {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create collection {collection_name}: {e}")
            return False

    async def upsert_points(
        self,
        collection_name: str,
        points: List[VectorPoint],
        wait: bool = True,
        batch_size: int = 100
    ) -> bool:
        """Insert or update vector points in a collection"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            # Process points in batches
            for i in range(0, len(points), batch_size):
                batch = points[i:i + batch_size]

                # Convert to Qdrant PointStruct format
                qdrant_points = []
                for point in batch:
                    qdrant_points.append(
                        PointStruct(
                            id=point.id,
                            vector=point.vector,
                            payload={
                                **point.payload,
                                "vector_type": point.vector_type.value,
                                "created_at": datetime.now().isoformat(),
                                "updated_at": datetime.now().isoformat()
                            }
                        )
                    )

                # Upsert batch
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.upsert(
                        collection_name=collection_name,
                        points=qdrant_points,
                        wait=wait
                    )
                )

            logger.info(f"Upserted {len(points)} points to collection {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to upsert points to {collection_name}: {e}")
            return False

    async def search_vectors(
        self,
        collection_name: str,
        query_vector: List[float],
        limit: int = 10,
        score_threshold: Optional[float] = None,
        filter_conditions: Optional[Dict[str, Any]] = None,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> List[SearchResult]:
        """Search for similar vectors in a collection"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            # Build filter if provided
            query_filter = None
            if filter_conditions:
                must_conditions = []
                for field, value in filter_conditions.items():
                    if isinstance(value, list):
                        # Multiple values - use should condition
                        should_conditions = [
                            FieldCondition(key=field, match=MatchValue(value=v))
                            for v in value
                        ]
                        must_conditions.append(models.Filter(should=should_conditions))
                    else:
                        # Single value
                        must_conditions.append(
                            FieldCondition(key=field, match=MatchValue(value=value))
                        )

                if must_conditions:
                    query_filter = Filter(must=must_conditions)

            # Perform search
            search_results = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=limit,
                    score_threshold=score_threshold,
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )
            )

            # Convert to SearchResult objects
            results = []
            for result in search_results:
                results.append(
                    SearchResult(
                        id=str(result.id),
                        score=result.score,
                        payload=result.payload or {},
                        vector=result.vector if with_vectors else None
                    )
                )

            logger.debug(f"Found {len(results)} similar vectors in {collection_name}")
            return results

        except Exception as e:
            logger.error(f"Failed to search vectors in {collection_name}: {e}")
            return []

    async def get_point(
        self,
        collection_name: str,
        point_id: str,
        with_payload: bool = True,
        with_vectors: bool = False
    ) -> Optional[SearchResult]:
        """Get a specific point by ID"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.retrieve(
                    collection_name=collection_name,
                    ids=[point_id],
                    with_payload=with_payload,
                    with_vectors=with_vectors
                )
            )

            if result and len(result) > 0:
                point = result[0]
                return SearchResult(
                    id=str(point.id),
                    score=1.0,  # Perfect match for exact retrieval
                    payload=point.payload or {},
                    vector=point.vector if with_vectors else None
                )

            return None

        except Exception as e:
            logger.error(f"Failed to get point {point_id} from {collection_name}: {e}")
            return None

    async def delete_points(
        self,
        collection_name: str,
        point_ids: List[str],
        wait: bool = True
    ) -> bool:
        """Delete points from a collection"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.delete(
                    collection_name=collection_name,
                    points_selector=models.PointIdsList(
                        points=point_ids
                    ),
                    wait=wait
                )
            )

            logger.info(f"Deleted {len(point_ids)} points from {collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete points from {collection_name}: {e}")
            return False

    async def get_collection_info(self, collection_name: str) -> Optional[CollectionInfo]:
        """Get information about a collection"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.get_collection(collection_name)
            )

            return CollectionInfo(
                name=collection_name,
                vectors_count=info.vectors_count or 0,
                indexed_vectors_count=info.indexed_vectors_count or 0,
                points_count=info.points_count or 0,
                segments_count=info.segments_count or 0,
                config=info.config.dict() if info.config else {}
            )

        except Exception as e:
            logger.error(f"Failed to get collection info for {collection_name}: {e}")
            return None

    async def list_collections(self) -> List[str]:
        """List all collections"""
        try:
            if not self.client:
                raise RuntimeError("Qdrant client not initialized")

            collections = await asyncio.get_event_loop().run_in_executor(
                None, self.client.get_collections
            )

            return [col.name for col in collections.collections]

        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    # AI-specific workflow integration methods

    async def store_workflow_embedding(
        self,
        workflow_id: str,
        workflow_name: str,
        embedding: List[float],
        metadata: Dict[str, Any],
        memgraph_node_id: Optional[str] = None
    ) -> bool:
        """Store workflow embedding with Memgraph integration"""
        try:
            point = VectorPoint(
                id=workflow_id,
                vector=embedding,
                vector_type=VectorType.WORKFLOW_EMBEDDING,
                payload={
                    "workflow_name": workflow_name,
                    "memgraph_node_id": memgraph_node_id,
                    "workflow_type": metadata.get("type", "unknown"),
                    "status": metadata.get("status", "pending"),
                    "created_by": metadata.get("created_by", "system"),
                    "tags": metadata.get("tags", []),
                    **metadata
                }
            )

            return await self.upsert_points("workflows", [point])

        except Exception as e:
            logger.error(f"Failed to store workflow embedding for {workflow_id}: {e}")
            return False

    async def find_similar_workflows(
        self,
        query_embedding: List[float],
        limit: int = 5,
        score_threshold: float = 0.7,
        workflow_type: Optional[str] = None,
        status_filter: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Find similar workflows based on embedding similarity"""
        try:
            filter_conditions = {}

            if workflow_type:
                filter_conditions["workflow_type"] = workflow_type

            if status_filter:
                filter_conditions["status"] = status_filter

            return await self.search_vectors(
                collection_name="workflows",
                query_vector=query_embedding,
                limit=limit,
                score_threshold=score_threshold,
                filter_conditions=filter_conditions if filter_conditions else None,
                with_payload=True
            )

        except Exception as e:
            logger.error(f"Failed to find similar workflows: {e}")
            return []

    async def store_invoice_embedding(
        self,
        invoice_id: str,
        invoice_text: str,
        embedding: List[float],
        extracted_data: Dict[str, Any],
        processing_status: str = "pending"
    ) -> bool:
        """Store Arabic invoice embedding with extracted data"""
        try:
            point = VectorPoint(
                id=invoice_id,
                vector=embedding,
                vector_type=VectorType.INVOICE_EMBEDDING,
                payload={
                    "invoice_text": invoice_text[:1000],  # Truncate for storage
                    "extracted_data": extracted_data,
                    "processing_status": processing_status,
                    "language": "arabic",
                    "vendor_name": extracted_data.get("vendor_name", ""),
                    "total_amount": extracted_data.get("total_amount", 0),
                    "invoice_date": extracted_data.get("invoice_date", ""),
                    "currency": extracted_data.get("currency", ""),
                    "confidence_score": extracted_data.get("confidence_score", 0.0)
                }
            )

            return await self.upsert_points("invoices", [point])

        except Exception as e:
            logger.error(f"Failed to store invoice embedding for {invoice_id}: {e}")
            return False

    async def search_similar_invoices(
        self,
        query_embedding: List[float],
        limit: int = 10,
        vendor_name: Optional[str] = None,
        amount_range: Optional[Tuple[float, float]] = None,
        processing_status: Optional[str] = None
    ) -> List[SearchResult]:
        """Search for similar invoices with filtering"""
        try:
            filter_conditions = {}

            if vendor_name:
                filter_conditions["vendor_name"] = vendor_name

            if processing_status:
                filter_conditions["processing_status"] = processing_status

            # Note: Qdrant doesn't support range queries in basic filtering
            # For amount_range, we'd need to implement post-filtering

            results = await self.search_vectors(
                collection_name="invoices",
                query_vector=query_embedding,
                limit=limit,
                filter_conditions=filter_conditions if filter_conditions else None,
                with_payload=True
            )

            # Post-filter by amount range if specified
            if amount_range:
                min_amount, max_amount = amount_range
                filtered_results = []
                for result in results:
                    amount = result.payload.get("total_amount", 0)
                    if min_amount <= amount <= max_amount:
                        filtered_results.append(result)
                results = filtered_results

            return results

        except Exception as e:
            logger.error(f"Failed to search similar invoices: {e}")
            return []

    async def store_tool_embedding(
        self,
        tool_id: str,
        tool_name: str,
        tool_description: str,
        embedding: List[float],
        tool_metadata: Dict[str, Any]
    ) -> bool:
        """Store tool embedding for intelligent orchestration"""
        try:
            point = VectorPoint(
                id=tool_id,
                vector=embedding,
                vector_type=VectorType.TOOL_EMBEDDING,
                payload={
                    "tool_name": tool_name,
                    "tool_description": tool_description,
                    "tool_category": tool_metadata.get("category", "general"),
                    "input_types": tool_metadata.get("input_types", []),
                    "output_types": tool_metadata.get("output_types", []),
                    "execution_time_avg": tool_metadata.get("execution_time_avg", 0.0),
                    "success_rate": tool_metadata.get("success_rate", 1.0),
                    "dependencies": tool_metadata.get("dependencies", []),
                    "tags": tool_metadata.get("tags", [])
                }
            )

            return await self.upsert_points("tools", [point])

        except Exception as e:
            logger.error(f"Failed to store tool embedding for {tool_id}: {e}")
            return False

    async def find_relevant_tools(
        self,
        task_embedding: List[float],
        limit: int = 5,
        tool_category: Optional[str] = None,
        required_input_types: Optional[List[str]] = None
    ) -> List[SearchResult]:
        """Find relevant tools for a given task"""
        try:
            filter_conditions = {}

            if tool_category:
                filter_conditions["tool_category"] = tool_category

            # For input types, we'd need more complex filtering
            # This is a simplified version

            return await self.search_vectors(
                collection_name="tools",
                query_vector=task_embedding,
                limit=limit,
                filter_conditions=filter_conditions if filter_conditions else None,
                with_payload=True
            )

        except Exception as e:
            logger.error(f"Failed to find relevant tools: {e}")
            return []

    async def store_a2ui_component_embedding(
        self,
        component_id: str,
        component_type: str,
        component_description: str,
        embedding: List[float],
        component_metadata: Dict[str, Any]
    ) -> bool:
        """Store A2UI component embedding for dynamic UI generation"""
        try:
            point = VectorPoint(
                id=component_id,
                vector=embedding,
                vector_type=VectorType.A2UI_COMPONENT,
                payload={
                    "component_type": component_type,
                    "component_description": component_description,
                    "ui_framework": component_metadata.get("ui_framework", "a2ui"),
                    "complexity_level": component_metadata.get("complexity_level", "medium"),
                    "supported_data_types": component_metadata.get("supported_data_types", []),
                    "interaction_patterns": component_metadata.get("interaction_patterns", []),
                    "accessibility_features": component_metadata.get("accessibility_features", []),
                    "responsive_breakpoints": component_metadata.get("responsive_breakpoints", [])
                }
            )

            return await self.upsert_points("a2ui_components", [point])

        except Exception as e:
            logger.error(f"Failed to store A2UI component embedding for {component_id}: {e}")
            return False

    async def find_suitable_ui_components(
        self,
        requirement_embedding: List[float],
        limit: int = 5,
        component_type: Optional[str] = None,
        complexity_level: Optional[str] = None
    ) -> List[SearchResult]:
        """Find suitable UI components for given requirements"""
        try:
            filter_conditions = {}

            if component_type:
                filter_conditions["component_type"] = component_type

            if complexity_level:
                filter_conditions["complexity_level"] = complexity_level

            return await self.search_vectors(
                collection_name="a2ui_components",
                query_vector=requirement_embedding,
                limit=limit,
                filter_conditions=filter_conditions if filter_conditions else None,
                with_payload=True
            )

        except Exception as e:
            logger.error(f"Failed to find suitable UI components: {e}")
            return []

    # Integration utilities for working with Memgraph

    async def sync_with_memgraph(
        self,
        memgraph_adapter,
        workflow_id: str,
        sync_embeddings: bool = True
    ) -> bool:
        """Synchronize workflow data between Qdrant and Memgraph"""
        try:
            # Get workflow from Memgraph
            workflow_data = await memgraph_adapter.get_workflow_analytics(workflow_id)

            if not workflow_data:
                logger.warning(f"Workflow {workflow_id} not found in Memgraph")
                return False

            if sync_embeddings:
                # Update workflow embedding with latest Memgraph data
                workflow_point = await self.get_point("workflows", workflow_id, with_payload=True)

                if workflow_point:
                    # Update payload with Memgraph data
                    updated_payload = {
                        **workflow_point.payload,
                        "memgraph_status": workflow_data.get("status", "unknown"),
                        "execution_progress": workflow_data.get("execution_progress", 0.0),
                        "node_count": workflow_data.get("node_count", 0),
                        "completed_nodes": workflow_data.get("completed_nodes", 0),
                        "failed_nodes": workflow_data.get("failed_nodes", 0),
                        "last_sync": datetime.now().isoformat()
                    }

                    # Create updated point
                    updated_point = VectorPoint(
                        id=workflow_id,
                        vector=workflow_point.vector or [],
                        vector_type=VectorType.WORKFLOW_EMBEDDING,
                        payload=updated_payload
                    )

                    return await self.upsert_points("workflows", [updated_point])

            return True

        except Exception as e:
            logger.error(f"Failed to sync workflow {workflow_id} with Memgraph: {e}")
            return False

    async def get_workflow_recommendations(
        self,
        current_workflow_id: str,
        limit: int = 3,
        include_similar_patterns: bool = True
    ) -> Dict[str, Any]:
        """Get workflow recommendations based on vector similarity and Memgraph relationships"""
        try:
            recommendations = {
                "similar_workflows": [],
                "suggested_tools": [],
                "optimization_hints": []
            }

            # Get current workflow embedding
            current_workflow = await self.get_point(
                "workflows",
                current_workflow_id,
                with_payload=True,
                with_vectors=True
            )

            if not current_workflow or not current_workflow.vector:
                return recommendations

            # Find similar workflows
            if include_similar_patterns:
                similar_workflows = await self.find_similar_workflows(
                    query_embedding=current_workflow.vector,
                    limit=limit,
                    score_threshold=0.6
                )

                recommendations["similar_workflows"] = [
                    {
                        "workflow_id": result.id,
                        "workflow_name": result.payload.get("workflow_name", "Unknown"),
                        "similarity_score": result.score,
                        "status": result.payload.get("status", "unknown"),
                        "success_rate": result.payload.get("success_rate", 0.0)
                    }
                    for result in similar_workflows
                    if result.id != current_workflow_id
                ]

            # Find relevant tools based on workflow context
            workflow_type = current_workflow.payload.get("workflow_type", "general")
            relevant_tools = await self.find_relevant_tools(
                task_embedding=current_workflow.vector,
                limit=limit,
                tool_category=workflow_type
            )

            recommendations["suggested_tools"] = [
                {
                    "tool_id": result.id,
                    "tool_name": result.payload.get("tool_name", "Unknown"),
                    "relevance_score": result.score,
                    "success_rate": result.payload.get("success_rate", 1.0),
                    "avg_execution_time": result.payload.get("execution_time_avg", 0.0)
                }
                for result in relevant_tools
            ]

            return recommendations

        except Exception as e:
            logger.error(f"Failed to get workflow recommendations for {current_workflow_id}: {e}")
            return {"similar_workflows": [], "suggested_tools": [], "optimization_hints": []}

    async def cleanup_old_embeddings(self, days_old: int = 30) -> bool:
        """Clean up old embeddings to manage storage"""
        try:
            cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

            collections_to_clean = ["workflows", "invoices", "tools", "a2ui_components"]

            for collection_name in collections_to_clean:
                # This is a simplified cleanup - in production you'd want more sophisticated logic
                logger.info(f"Cleanup for {collection_name} would be implemented here")
                # Implementation would involve querying by date and deleting old points

            return True

        except Exception as e:
            logger.error(f"Failed to cleanup old embeddings: {e}")
            return False

    async def get_analytics_summary(self) -> Dict[str, Any]:
        """Get analytics summary across all collections"""
        try:
            summary = {
                "collections": {},
                "total_points": 0,
                "total_vectors": 0,
                "storage_usage": "N/A"  # Would need additional API calls
            }

            collections = await self.list_collections()

            for collection_name in collections:
                info = await self.get_collection_info(collection_name)
                if info:
                    summary["collections"][collection_name] = {
                        "points_count": info.points_count,
                        "vectors_count": info.vectors_count,
                        "indexed_vectors": info.indexed_vectors_count,
                        "segments": info.segments_count
                    }
                    summary["total_points"] += info.points_count
                    summary["total_vectors"] += info.vectors_count

            return summary

        except Exception as e:
            logger.error(f"Failed to get analytics summary: {e}")
            return {}

    async def close(self):
        """
        Release this adapter's reference to the pooled Qdrant client.
        
        The underlying QdrantClient is only discarded when the last adapter
        using the same connection parameters releases it.
        """
        if not self.client or not self._pool_key:
            return

        try:
            async with self._pool_lock:
                key = self._pool_key
                ref_count = self._client_pool_ref_counts.get(key, 0)

                if ref_count > 1:
                    self._client_pool_ref_counts[key] = ref_count - 1
                    logger.info(
                        "Released pooled Qdrant client for %s (remaining ref_count=%d)",
                        f"{self.host}:{self.grpc_port if self.prefer_grpc else self.port}",
                        self._client_pool_ref_counts[key],
                    )
                else:
                    # Last reference: remove from pool; underlying client will be
                    # garbage-collected as the library doesn't expose an explicit close.
                    self._client_pool.pop(key, None)
                    self._client_pool_ref_counts.pop(key, None)
                    logger.info(
                        "Dropped pooled Qdrant client for %s (pool entry removed)",
                        f"{self.host}:{self.grpc_port if self.prefer_grpc else self.port}",
                    )
        except Exception as e:
            logger.error(f"Error closing Qdrant adapter: {e}")
        finally:
            self.client = None


# Global Qdrant adapter instance
qdrant_adapter = QdrantAdapter()

async def get_qdrant_adapter() -> QdrantAdapter:
    """Get the global Qdrant adapter instance"""
    return qdrant_adapter

async def initialize_qdrant_adapter(
    host: str = "localhost",
    port: int = 6333,
    grpc_port: int = 6334,
    prefer_grpc: bool = False,
    api_key: Optional[str] = None,
    https: bool = False
) -> bool:
    """Initialize the global Qdrant adapter"""
    global qdrant_adapter

    env_url = os.getenv("QDRANT_URL")
    if env_url:
        parsed = urlparse(env_url)
        if parsed.hostname:
            host = parsed.hostname
        if parsed.port:
            port = parsed.port
        if parsed.scheme:
            https = parsed.scheme.lower() == "https"

    env_grpc_port = os.getenv("QDRANT_GRPC_PORT")
    if env_grpc_port:
        try:
            grpc_port = int(env_grpc_port)
        except ValueError:
            logger.warning("Invalid QDRANT_GRPC_PORT value: %s", env_grpc_port)

    env_api_key = os.getenv("QDRANT_API_KEY")
    if api_key is None and env_api_key:
        api_key = env_api_key

    qdrant_adapter = QdrantAdapter(
        host=host,
        port=port,
        grpc_port=grpc_port,
        prefer_grpc=prefer_grpc,
        api_key=api_key,
        https=https
    )
    return await qdrant_adapter.initialize()

async def shutdown_qdrant_adapter():
    """Shutdown the global Qdrant adapter"""
    await qdrant_adapter.close()


async def check_qdrant_health(qdrant_url: str = "http://qdrant:6333") -> Dict[str, Any]:
    """
    Check Qdrant service health
    
    Args:
        qdrant_url: Base URL for Qdrant service
        
    Returns:
        Health check result with status boolean
    """
    from urllib.parse import urlparse
    parsed = urlparse(qdrant_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 6333
    
    adapter = QdrantAdapter(host=host, port=port)
    try:
        is_healthy = await adapter.health_check()
        return {"status": "healthy" if is_healthy else "unhealthy", "healthy": is_healthy}
    except Exception as e:
        return {"status": "error", "error": str(e), "healthy": False}
    finally:
        await adapter.close()
