"""
Orchestration API routes
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, Any, List
import logging

from backend.schemas.workflow import WorkflowDefinition
from backend.adapters.orchestration import OrchestrationAdapter
from backend.adapters.toolorchestra import ToolOrchestraAdapter, OrchestrationStrategy
from backend.api.errors import NotFoundError, ServiceUnavailableError, ValidationError
from backend.config.settings import settings
from backend.constants import WORKFLOW_TYPE_DEFAULT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/orchestrate", tags=["orchestration"])

# Initialize orchestration adapters
orchestration_adapter: Optional[OrchestrationAdapter] = None
toolorchestra_adapter: Optional[ToolOrchestraAdapter] = None
workflow_storage: Dict[str, WorkflowDefinition] = {}
default_workflow: Optional[WorkflowDefinition] = None


def init_orchestration():
    """Initialize orchestration services"""
    global orchestration_adapter, toolorchestra_adapter, default_workflow

    try:
        orchestration_adapter = OrchestrationAdapter()
        default_workflow = orchestration_adapter.create_default_invoice_workflow()
        workflow_storage[default_workflow.id] = default_workflow
        logger.info("Orchestration service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize orchestration: {e}", exc_info=True)
        orchestration_adapter = None

    # Initialize ToolOrchestra adapter
    try:
        toolorchestra_adapter = ToolOrchestraAdapter(enable_rl=True)
        logger.info("ToolOrchestra adapter initialized")
    except Exception as e:
        logger.error(f"Failed to initialize ToolOrchestra: {e}", exc_info=True)
        toolorchestra_adapter = None


# Initialize on module load
try:
    init_orchestration()
except Exception as e:
    logger.warning(f"Orchestration initialization deferred: {e}")


class WorkflowExecuteRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: Optional[str] = None
    workflow: Optional[Dict[str, Any]] = None
    inputs: Dict[str, Any] = {}
    use_rl_optimization: bool = False
    orchestration_strategy: Optional[str] = None


@router.post("/execute")
async def execute_orchestrated_workflow(req: WorkflowExecuteRequest):
    """
    Execute an orchestrated workflow
    
    Args:
        req: Workflow execution request
    
    Returns:
        Workflow execution result
    """
    if not orchestration_adapter:
        raise ServiceUnavailableError("orchestration")
    
    try:
        if req.workflow_id:
            if req.workflow_id not in workflow_storage:
                raise NotFoundError("Workflow", req.workflow_id)
            workflow = workflow_storage[req.workflow_id]
        elif req.workflow:
            try:
                workflow = WorkflowDefinition(**req.workflow)
            except Exception as e:
                raise ValidationError(f"Invalid workflow definition: {str(e)}")
        else:
            if not default_workflow:
                raise ValidationError("No workflow specified and no default available")
            workflow = default_workflow
        
        result = await orchestration_adapter.execute_workflow(workflow, req.inputs)
        logger.info(f"Workflow {workflow.id} executed successfully")
        return result
    except (NotFoundError, ValidationError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"Workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow execution error: {str(e)}")


@router.get("/workflows")
def list_workflows():
    """
    List all available workflows
    
    Returns:
        List of workflow summaries
    """
    return {
        "workflows": [
            {
                "id": wf.id,
                "name": wf.name,
                "description": wf.description,
                "node_count": len(wf.nodes),
                "edge_count": len(wf.edges),
                "version": wf.version
            }
            for wf in workflow_storage.values()
        ]
    }


@router.get("/workflows/{workflow_id}")
def get_workflow(workflow_id: str):
    """
    Get a specific workflow definition
    
    Args:
        workflow_id: Workflow identifier
    
    Returns:
        Workflow definition
    """
    if workflow_id not in workflow_storage:
        raise NotFoundError("Workflow", workflow_id)
    
    return workflow_storage[workflow_id].dict()


@router.post("/workflows")
def save_workflow(workflow: Dict[str, Any]):
    """
    Save or update a workflow definition
    
    Args:
        workflow: Workflow definition dictionary
    
    Returns:
        Success response with workflow ID
    """
    try:
        wf = WorkflowDefinition(**workflow)
        workflow_storage[wf.id] = wf
        logger.info(f"Workflow {wf.id} saved")
        return {
            "success": True,
            "workflow_id": wf.id,
            "message": "Workflow saved"
        }
    except Exception as e:
        raise ValidationError(f"Invalid workflow definition: {str(e)}")


@router.delete("/workflows/{workflow_id}")
def delete_workflow(workflow_id: str):
    """
    Delete a workflow
    
    Args:
        workflow_id: Workflow identifier
    
    Returns:
        Success response
    """
    if workflow_id not in workflow_storage:
        raise NotFoundError("Workflow", workflow_id)
    
    del workflow_storage[workflow_id]
    logger.info(f"Workflow {workflow_id} deleted")
    return {
        "success": True,
        "message": f"Workflow {workflow_id} deleted"
    }


@router.post("/execute/rl-optimized")
async def execute_rl_optimized_workflow(req: WorkflowExecuteRequest):
    """
    Execute workflow with RL-based optimization using ToolOrchestra

    Args:
        req: Workflow execution request with RL optimization enabled

    Returns:
        Optimized workflow execution result with performance metrics
    """
    if not toolorchestra_adapter:
        raise ServiceUnavailableError("toolorchestra", "ToolOrchestra adapter not available")

    try:
        # Get workflow definition
        if req.workflow_id:
            if req.workflow_id not in workflow_storage:
                raise NotFoundError("Workflow", req.workflow_id)
            workflow = workflow_storage[req.workflow_id]
        elif req.workflow:
            try:
                workflow = WorkflowDefinition(**req.workflow)
            except Exception as e:
                raise ValidationError(f"Invalid workflow definition: {str(e)}")
        else:
            if not default_workflow:
                raise ValidationError("No workflow specified and no default available")
            workflow = default_workflow

        # Convert workflow to ToolOrchestra format
        workflow_dict = workflow.dict()

        # Execute with RL optimization
        result = await toolorchestra_adapter.orchestrate_workflow(
            workflow_dict,
            req.inputs
        )

        logger.info(f"RL-optimized workflow {workflow.id} executed successfully")
        return result

    except (NotFoundError, ValidationError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"RL-optimized workflow execution failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RL workflow execution error: {str(e)}")


@router.post("/optimize")
async def optimize_workflow(
    workflow_id: str,
    historical_data: Optional[List[Dict[str, Any]]] = None
):
    """
    Optimize workflow using ToolOrchestra RL insights

    Args:
        workflow_id: ID of workflow to optimize
        historical_data: Optional historical execution data for analysis

    Returns:
        Optimization recommendations and estimated improvements
    """
    if not toolorchestra_adapter:
        raise ServiceUnavailableError("toolorchestra", "ToolOrchestra adapter not available")

    if workflow_id not in workflow_storage:
        raise NotFoundError("Workflow", workflow_id)

    try:
        workflow = workflow_storage[workflow_id]
        workflow_dict = workflow.dict()

        optimization_result = await toolorchestra_adapter.optimize_workflow(
            workflow_dict,
            historical_data
        )

        logger.info(f"Workflow {workflow_id} optimization analysis completed")
        return optimization_result

    except (NotFoundError, ServiceUnavailableError):
        raise
    except Exception as e:
        logger.error(f"Workflow optimization failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Workflow optimization error: {str(e)}")


@router.get("/analytics")
def get_orchestration_analytics():
    """
    Get comprehensive orchestration analytics from ToolOrchestra

    Returns:
        Analytics data including tool metrics, performance summaries, and RL status
    """
    if not toolorchestra_adapter:
        raise ServiceUnavailableError("toolorchestra", "ToolOrchestra adapter not available")

    try:
        analytics = toolorchestra_adapter.get_orchestration_analytics()
        return analytics
    except Exception as e:
        logger.error(f"Failed to get orchestration analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Analytics error: {str(e)}")


@router.get("/strategies")
def list_orchestration_strategies():
    """
    List available orchestration strategies

    Returns:
        Available orchestration strategies and their descriptions
    """
    return {
        "strategies": [
            {
                "name": OrchestrationStrategy.SEQUENTIAL.value,
                "description": "Execute tools one after another in sequence",
                "use_case": "Simple workflows with dependencies"
            },
            {
                "name": OrchestrationStrategy.PARALLEL.value,
                "description": "Execute independent tools in parallel",
                "use_case": "Workflows with independent parallel tasks"
            },
            {
                "name": OrchestrationStrategy.ADAPTIVE.value,
                "description": "Dynamically choose between sequential and parallel",
                "use_case": "Complex workflows with mixed dependencies"
            },
            {
                "name": OrchestrationStrategy.RL_OPTIMIZED.value,
                "description": "Use reinforcement learning for optimal execution",
                "use_case": "Performance-critical workflows requiring optimization"
            }
        ],
        "default_strategy": OrchestrationStrategy.ADAPTIVE.value,
        "rl_available": toolorchestra_adapter is not None and toolorchestra_adapter.enable_rl
    }

