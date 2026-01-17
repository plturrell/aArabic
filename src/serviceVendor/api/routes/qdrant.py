"""
Qdrant Vector Database API Routes
Provides endpoints for vector similarity search and AI processing
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel
import logging

from backend.adapters.qdrant import get_qdrant_adapter, QdrantAdapter, VectorPoint, VectorType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/qdrant", tags=["Qdrant Vector Database"])

class VectorSearchRequest(BaseModel):
    """Vector search request"""
    query_vector: List[float]
    collection_name: str
    limit: int = 10
    score_threshold: Optional[float] = None
    filter_conditions: Optional[Dict[str, Any]] = None
    with_payload: bool = True
    with_vectors: bool = False

class VectorUpsertRequest(BaseModel):
    """Vector upsert request"""
    collection_name: str
    points: List[Dict[str, Any]]  # Will be converted to VectorPoint objects

class WorkflowEmbeddingRequest(BaseModel):
    """Workflow embedding storage request"""
    workflow_id: str
    workflow_name: str
    embedding: List[float]
    metadata: Dict[str, Any]
    memgraph_node_id: Optional[str] = None

class InvoiceEmbeddingRequest(BaseModel):
    """Invoice embedding storage request"""
    invoice_id: str
    invoice_text: str
    embedding: List[float]
    extracted_data: Dict[str, Any]
    processing_status: str = "pending"

class ToolEmbeddingRequest(BaseModel):
    """Tool embedding storage request"""
    tool_id: str
    tool_name: str
    tool_description: str
    embedding: List[float]
    tool_metadata: Dict[str, Any]

@router.get("/health")
async def health_check(qdrant: QdrantAdapter = Depends(get_qdrant_adapter)):
    """Check Qdrant service health"""
    try:
        is_healthy = await qdrant.health_check()
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "service": "qdrant",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        raise HTTPException(status_code=503, detail="Qdrant service unavailable")

@router.get("/collections")
async def list_collections(qdrant: QdrantAdapter = Depends(get_qdrant_adapter)):
    """List all Qdrant collections"""
    try:
        collections = await qdrant.list_collections()
        return {"collections": collections}
    except Exception as e:
        logger.error(f"Failed to list collections: {e}")
        raise HTTPException(status_code=500, detail="Failed to list collections")

@router.get("/collections/{collection_name}/info")
async def get_collection_info(
    collection_name: str,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Get information about a specific collection"""
    try:
        info = await qdrant.get_collection_info(collection_name)
        if not info:
            raise HTTPException(status_code=404, detail=f"Collection {collection_name} not found")
        
        return {
            "collection_name": info.name,
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "points_count": info.points_count,
            "segments_count": info.segments_count,
            "config": info.config
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get collection info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get collection info")

@router.post("/search")
async def search_vectors(
    request: VectorSearchRequest,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Search for similar vectors"""
    try:
        results = await qdrant.search_vectors(
            collection_name=request.collection_name,
            query_vector=request.query_vector,
            limit=request.limit,
            score_threshold=request.score_threshold,
            filter_conditions=request.filter_conditions,
            with_payload=request.with_payload,
            with_vectors=request.with_vectors
        )
        
        return {
            "results": [
                {
                    "id": result.id,
                    "score": result.score,
                    "payload": result.payload,
                    "vector": result.vector
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Vector search failed: {e}")
        raise HTTPException(status_code=500, detail="Vector search failed")

@router.post("/upsert")
async def upsert_vectors(
    request: VectorUpsertRequest,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Insert or update vector points"""
    try:
        # Convert request points to VectorPoint objects
        vector_points = []
        for point_data in request.points:
            vector_points.append(
                VectorPoint(
                    id=point_data["id"],
                    vector=point_data["vector"],
                    vector_type=VectorType(point_data.get("vector_type", "semantic_search")),
                    payload=point_data.get("payload", {})
                )
            )
        
        success = await qdrant.upsert_points(request.collection_name, vector_points)
        
        if success:
            return {"message": f"Successfully upserted {len(vector_points)} points"}
        else:
            raise HTTPException(status_code=500, detail="Failed to upsert points")
            
    except Exception as e:
        logger.error(f"Vector upsert failed: {e}")
        raise HTTPException(status_code=500, detail="Vector upsert failed")

@router.get("/points/{collection_name}/{point_id}")
async def get_point(
    collection_name: str,
    point_id: str,
    with_payload: bool = Query(True),
    with_vectors: bool = Query(False),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Get a specific point by ID"""
    try:
        result = await qdrant.get_point(
            collection_name=collection_name,
            point_id=point_id,
            with_payload=with_payload,
            with_vectors=with_vectors
        )
        
        if not result:
            raise HTTPException(status_code=404, detail=f"Point {point_id} not found")
        
        return {
            "id": result.id,
            "score": result.score,
            "payload": result.payload,
            "vector": result.vector
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get point: {e}")
        raise HTTPException(status_code=500, detail="Failed to get point")

@router.delete("/points/{collection_name}")
async def delete_points(
    collection_name: str,
    point_ids: List[str],
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Delete points from a collection"""
    try:
        success = await qdrant.delete_points(collection_name, point_ids)
        
        if success:
            return {"message": f"Successfully deleted {len(point_ids)} points"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete points")
            
    except Exception as e:
        logger.error(f"Failed to delete points: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete points")

# AI-specific endpoints

@router.post("/workflows/store")
async def store_workflow_embedding(
    request: WorkflowEmbeddingRequest,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Store workflow embedding with metadata"""
    try:
        success = await qdrant.store_workflow_embedding(
            workflow_id=request.workflow_id,
            workflow_name=request.workflow_name,
            embedding=request.embedding,
            metadata=request.metadata,
            memgraph_node_id=request.memgraph_node_id
        )

        if success:
            return {"message": f"Workflow embedding stored for {request.workflow_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store workflow embedding")

    except Exception as e:
        logger.error(f"Failed to store workflow embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to store workflow embedding")

@router.post("/workflows/search")
async def find_similar_workflows(
    query_embedding: List[float],
    limit: int = Query(5, ge=1, le=50),
    score_threshold: float = Query(0.7, ge=0.0, le=1.0),
    workflow_type: Optional[str] = Query(None),
    status_filter: Optional[List[str]] = Query(None),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Find similar workflows based on embedding similarity"""
    try:
        results = await qdrant.find_similar_workflows(
            query_embedding=query_embedding,
            limit=limit,
            score_threshold=score_threshold,
            workflow_type=workflow_type,
            status_filter=status_filter
        )

        return {
            "similar_workflows": [
                {
                    "workflow_id": result.id,
                    "workflow_name": result.payload.get("workflow_name", "Unknown"),
                    "similarity_score": result.score,
                    "workflow_type": result.payload.get("workflow_type", "unknown"),
                    "status": result.payload.get("status", "unknown"),
                    "memgraph_node_id": result.payload.get("memgraph_node_id"),
                    "metadata": result.payload
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to find similar workflows: {e}")
        raise HTTPException(status_code=500, detail="Failed to find similar workflows")

@router.post("/invoices/store")
async def store_invoice_embedding(
    request: InvoiceEmbeddingRequest,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Store Arabic invoice embedding with extracted data"""
    try:
        success = await qdrant.store_invoice_embedding(
            invoice_id=request.invoice_id,
            invoice_text=request.invoice_text,
            embedding=request.embedding,
            extracted_data=request.extracted_data,
            processing_status=request.processing_status
        )

        if success:
            return {"message": f"Invoice embedding stored for {request.invoice_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store invoice embedding")

    except Exception as e:
        logger.error(f"Failed to store invoice embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to store invoice embedding")

@router.post("/invoices/search")
async def search_similar_invoices(
    query_embedding: List[float],
    limit: int = Query(10, ge=1, le=100),
    vendor_name: Optional[str] = Query(None),
    processing_status: Optional[str] = Query(None),
    min_amount: Optional[float] = Query(None),
    max_amount: Optional[float] = Query(None),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Search for similar Arabic invoices"""
    try:
        amount_range = None
        if min_amount is not None and max_amount is not None:
            amount_range = (min_amount, max_amount)

        results = await qdrant.search_similar_invoices(
            query_embedding=query_embedding,
            limit=limit,
            vendor_name=vendor_name,
            amount_range=amount_range,
            processing_status=processing_status
        )

        return {
            "similar_invoices": [
                {
                    "invoice_id": result.id,
                    "similarity_score": result.score,
                    "vendor_name": result.payload.get("vendor_name", ""),
                    "total_amount": result.payload.get("total_amount", 0),
                    "currency": result.payload.get("currency", ""),
                    "invoice_date": result.payload.get("invoice_date", ""),
                    "processing_status": result.payload.get("processing_status", "unknown"),
                    "confidence_score": result.payload.get("confidence_score", 0.0),
                    "extracted_data": result.payload.get("extracted_data", {})
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to search similar invoices: {e}")
        raise HTTPException(status_code=500, detail="Failed to search similar invoices")

@router.post("/tools/store")
async def store_tool_embedding(
    request: ToolEmbeddingRequest,
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Store tool embedding for intelligent orchestration"""
    try:
        success = await qdrant.store_tool_embedding(
            tool_id=request.tool_id,
            tool_name=request.tool_name,
            tool_description=request.tool_description,
            embedding=request.embedding,
            tool_metadata=request.tool_metadata
        )

        if success:
            return {"message": f"Tool embedding stored for {request.tool_id}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to store tool embedding")

    except Exception as e:
        logger.error(f"Failed to store tool embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to store tool embedding")

@router.post("/tools/search")
async def find_relevant_tools(
    task_embedding: List[float],
    limit: int = Query(5, ge=1, le=20),
    tool_category: Optional[str] = Query(None),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Find relevant tools for a given task"""
    try:
        results = await qdrant.find_relevant_tools(
            task_embedding=task_embedding,
            limit=limit,
            tool_category=tool_category
        )

        return {
            "relevant_tools": [
                {
                    "tool_id": result.id,
                    "tool_name": result.payload.get("tool_name", "Unknown"),
                    "relevance_score": result.score,
                    "tool_description": result.payload.get("tool_description", ""),
                    "tool_category": result.payload.get("tool_category", "general"),
                    "success_rate": result.payload.get("success_rate", 1.0),
                    "avg_execution_time": result.payload.get("execution_time_avg", 0.0),
                    "input_types": result.payload.get("input_types", []),
                    "output_types": result.payload.get("output_types", [])
                }
                for result in results
            ],
            "count": len(results)
        }
    except Exception as e:
        logger.error(f"Failed to find relevant tools: {e}")
        raise HTTPException(status_code=500, detail="Failed to find relevant tools")

@router.get("/workflows/{workflow_id}/recommendations")
async def get_workflow_recommendations(
    workflow_id: str,
    limit: int = Query(3, ge=1, le=10),
    include_similar_patterns: bool = Query(True),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Get AI-powered workflow recommendations"""
    try:
        recommendations = await qdrant.get_workflow_recommendations(
            current_workflow_id=workflow_id,
            limit=limit,
            include_similar_patterns=include_similar_patterns
        )

        return recommendations

    except Exception as e:
        logger.error(f"Failed to get workflow recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get workflow recommendations")

@router.get("/analytics/summary")
async def get_analytics_summary(qdrant: QdrantAdapter = Depends(get_qdrant_adapter)):
    """Get analytics summary across all collections"""
    try:
        summary = await qdrant.get_analytics_summary()
        return summary
    except Exception as e:
        logger.error(f"Failed to get analytics summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics summary")

@router.post("/sync/memgraph/{workflow_id}")
async def sync_with_memgraph(
    workflow_id: str,
    sync_embeddings: bool = Query(True),
    qdrant: QdrantAdapter = Depends(get_qdrant_adapter)
):
    """Synchronize workflow data between Qdrant and Memgraph"""
    try:
        # Import memgraph adapter
        from backend.adapters.memgraph import MemgraphAdapter
        memgraph_adapter = MemgraphAdapter()

        success = await qdrant.sync_with_memgraph(
            memgraph_adapter=memgraph_adapter,
            workflow_id=workflow_id,
            sync_embeddings=sync_embeddings
        )

        if success:
            return {"message": f"Successfully synced workflow {workflow_id} with Memgraph"}
        else:
            raise HTTPException(status_code=500, detail="Failed to sync with Memgraph")

    except Exception as e:
        logger.error(f"Failed to sync with Memgraph: {e}")
        raise HTTPException(status_code=500, detail="Failed to sync with Memgraph")
