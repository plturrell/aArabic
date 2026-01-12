"""
Hybrid Orchestration API Routes
Provides endpoints for hybrid graph + vector workflow orchestration
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Any, Optional
from pydantic import BaseModel
import logging

from backend.adapters.hybrid_orchestration import (
    get_hybrid_orchestration_adapter, 
    HybridOrchestrationAdapter,
    HybridWorkflowResult
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/hybrid", tags=["Hybrid Orchestration"])

class HybridWorkflowRequest(BaseModel):
    """Hybrid workflow execution request"""
    workflow_data: Dict[str, Any]
    context: Optional[Dict[str, Any]] = None

class ToolRecommendationRequest(BaseModel):
    """Tool recommendation request"""
    task_description: str
    context: Optional[Dict[str, Any]] = None

class WorkflowOptimizationRequest(BaseModel):
    """Workflow optimization request"""
    workflow_data: Dict[str, Any]

class PatternAnalysisRequest(BaseModel):
    """Pattern analysis request"""
    workflow_type: Optional[str] = None
    limit: int = 10

@router.get("/health")
async def health_check(
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Check hybrid orchestration health"""
    try:
        # Check if adapters are initialized
        memgraph_healthy = hybrid_adapter.memgraph_adapter is not None
        qdrant_healthy = hybrid_adapter.qdrant_adapter is not None
        base_healthy = hybrid_adapter.base_orchestration is not None
        
        return {
            "status": "healthy" if (memgraph_healthy or qdrant_healthy) and base_healthy else "degraded",
            "components": {
                "memgraph": "healthy" if memgraph_healthy else "unavailable",
                "qdrant": "healthy" if qdrant_healthy else "unavailable", 
                "base_orchestration": "healthy" if base_healthy else "unavailable"
            },
            "service": "hybrid_orchestration",
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Hybrid orchestration health check failed: {e}")
        raise HTTPException(status_code=503, detail="Hybrid orchestration service unavailable")

@router.post("/execute")
async def execute_hybrid_workflow(
    request: HybridWorkflowRequest,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Execute workflow with hybrid graph + vector processing"""
    try:
        result = await hybrid_adapter.execute_hybrid_workflow(
            workflow_data=request.workflow_data,
            context=request.context
        )
        
        return {
            "workflow_id": result.workflow_id,
            "success": result.success,
            "execution_time": result.execution_time,
            "execution_result": result.execution_result,
            "graph_analytics": result.graph_analytics,
            "vector_recommendations": result.vector_recommendations,
            "similar_workflows": result.similar_workflows
        }
        
    except Exception as e:
        logger.error(f"Hybrid workflow execution failed: {e}")
        raise HTTPException(status_code=500, detail="Hybrid workflow execution failed")

@router.post("/tools/recommend")
async def get_tool_recommendations(
    request: ToolRecommendationRequest,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Get intelligent tool recommendations using vector similarity"""
    try:
        recommendations = await hybrid_adapter.get_intelligent_tool_recommendations(
            task_description=request.task_description,
            context=request.context
        )
        
        return {
            "task_description": request.task_description,
            "recommendations": recommendations,
            "count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Tool recommendation failed: {e}")
        raise HTTPException(status_code=500, detail="Tool recommendation failed")

@router.post("/analyze/patterns")
async def analyze_workflow_patterns(
    request: PatternAnalysisRequest,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Analyze workflow patterns using both graph and vector data"""
    try:
        analysis = await hybrid_adapter.analyze_workflow_patterns(
            workflow_type=request.workflow_type,
            limit=request.limit
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Pattern analysis failed")

@router.post("/optimize")
async def optimize_workflow(
    request: WorkflowOptimizationRequest,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Optimize workflow execution using hybrid graph + vector insights"""
    try:
        optimization = await hybrid_adapter.optimize_workflow_execution(
            workflow_data=request.workflow_data
        )
        
        return optimization
        
    except Exception as e:
        logger.error(f"Workflow optimization failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow optimization failed")

@router.get("/workflows/{workflow_id}/health")
async def get_workflow_health(
    workflow_id: str,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Get comprehensive workflow health score"""
    try:
        health_score = await hybrid_adapter.get_workflow_health_score(workflow_id)
        
        return {
            "workflow_id": workflow_id,
            **health_score
        }
        
    except Exception as e:
        logger.error(f"Workflow health check failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow health check failed")

@router.get("/workflows/{workflow_id}/analytics")
async def get_workflow_analytics(
    workflow_id: str,
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Get comprehensive workflow analytics from both graph and vector data"""
    try:
        analytics = {
            "workflow_id": workflow_id,
            "graph_analytics": {},
            "vector_data": {},
            "recommendations": {}
        }
        
        # Get graph analytics
        try:
            graph_analytics = await hybrid_adapter.memgraph_adapter.get_workflow_analytics(workflow_id)
            analytics["graph_analytics"] = graph_analytics or {}
        except Exception as e:
            logger.warning(f"Failed to get graph analytics: {e}")
        
        # Get vector data
        if hybrid_adapter.qdrant_adapter:
            try:
                workflow_point = await hybrid_adapter.qdrant_adapter.get_point(
                    "workflows", workflow_id, with_payload=True
                )
                if workflow_point:
                    analytics["vector_data"] = workflow_point.payload
                
                # Get recommendations
                recommendations = await hybrid_adapter.qdrant_adapter.get_workflow_recommendations(
                    current_workflow_id=workflow_id,
                    limit=3
                )
                analytics["recommendations"] = recommendations
                
            except Exception as e:
                logger.warning(f"Failed to get vector data: {e}")
        
        return analytics
        
    except Exception as e:
        logger.error(f"Workflow analytics failed: {e}")
        raise HTTPException(status_code=500, detail="Workflow analytics failed")

@router.get("/status")
async def get_hybrid_status(
    hybrid_adapter: HybridOrchestrationAdapter = Depends(get_hybrid_orchestration_adapter)
):
    """Get detailed status of hybrid orchestration components"""
    try:
        status = {
            "hybrid_orchestration": "active",
            "components": {},
            "capabilities": []
        }
        
        # Check Memgraph status
        if hybrid_adapter.memgraph_adapter:
            try:
                memgraph_health = await hybrid_adapter.memgraph_adapter.health_check()
                status["components"]["memgraph"] = {
                    "status": "healthy" if memgraph_health else "unhealthy",
                    "capabilities": ["graph_storage", "dependency_tracking", "workflow_analytics"]
                }
                if memgraph_health:
                    status["capabilities"].extend(["graph_analytics", "dependency_optimization"])
            except Exception as e:
                status["components"]["memgraph"] = {"status": "error", "error": str(e)}
        
        # Check Qdrant status
        if hybrid_adapter.qdrant_adapter:
            try:
                qdrant_health = await hybrid_adapter.qdrant_adapter.health_check()
                status["components"]["qdrant"] = {
                    "status": "healthy" if qdrant_health else "unhealthy",
                    "capabilities": ["vector_search", "semantic_similarity", "recommendations"]
                }
                if qdrant_health:
                    status["capabilities"].extend(["semantic_search", "workflow_recommendations", "tool_suggestions"])
            except Exception as e:
                status["components"]["qdrant"] = {"status": "error", "error": str(e)}
        
        # Check base orchestration
        if hybrid_adapter.base_orchestration:
            status["components"]["base_orchestration"] = {
                "status": "healthy",
                "capabilities": ["workflow_execution", "tool_orchestration"]
            }
            status["capabilities"].extend(["workflow_execution", "basic_orchestration"])
        
        return status
        
    except Exception as e:
        logger.error(f"Failed to get hybrid status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get hybrid status")
