"""
Enhanced A2UI API routes
Implements A2UI v0.9 specification with full vendor integration
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List
import logging

from backend.adapters.a2ui import A2UIResponse, generate_invoice_processing_ui, workflow_result_to_a2ui
from backend.adapters.a2ui_enhanced import EnhancedA2UIAdapter, A2UIMessageType
from backend.api.errors import ServiceUnavailableError, ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/a2ui", tags=["a2ui"])

A2UI_AVAILABLE = True  # A2UI is always available (Python implementation)

# Global enhanced adapter instance
enhanced_adapter = EnhancedA2UIAdapter()


@router.post("/generate")
def generate_a2ui_ui(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate A2UI v0.9 messages from data

    Args:
        data: Data dictionary with type and relevant data

    Returns:
        List of A2UI v0.9 messages
    """
    if not A2UI_AVAILABLE:
        raise ServiceUnavailableError("a2ui")

    try:
        ui_type = data.get("type", "invoice")
        surface_id = data.get("surface_id")

        if ui_type == "invoice":
            invoice_data = data.get("invoice_data", {})
            messages = enhanced_adapter.generate_invoice_processing_ui(invoice_data, surface_id)
            logger.info(f"A2UI invoice UI generated with {len(messages)} messages")
            return messages

        elif ui_type == "workflow_result":
            workflow_result = data.get("workflow_result", {})
            messages = enhanced_adapter.generate_workflow_ui(workflow_result, surface_id)
            logger.info(f"A2UI workflow result UI generated with {len(messages)} messages")
            return messages

        else:
            raise ValidationError(f"Unknown UI type: {ui_type}")

    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"A2UI generation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("a2ui", f"A2UI generation error: {str(e)}")


@router.post("/generate/legacy")
def generate_a2ui_ui_legacy(data: Dict[str, Any]):
    """
    Legacy A2UI generation endpoint for backward compatibility

    Args:
        data: Data dictionary with type and invoice_data

    Returns:
        Legacy A2UI response dictionary
    """
    if not A2UI_AVAILABLE:
        raise ServiceUnavailableError("a2ui")

    try:
        ui_type = data.get("type", "invoice")
        if ui_type == "invoice":
            invoice_data = data.get("invoice_data", {})
            a2ui_response = generate_invoice_processing_ui(invoice_data)
            logger.info(f"Legacy A2UI invoice UI generated")
            return a2ui_response.to_dict()
        elif ui_type == "workflow_result":
            workflow_result = data.get("workflow_result", {})
            a2ui_response = workflow_result_to_a2ui(workflow_result)
            logger.info(f"Legacy A2UI workflow result UI generated")
            return a2ui_response.to_dict()
        else:
            raise ValidationError(f"Unknown UI type: {ui_type}")
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"Legacy A2UI generation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("a2ui", f"A2UI generation error: {str(e)}")


@router.post("/workflow-result")
def workflow_result_a2ui(workflow_result: Dict[str, Any]):
    """
    Convert workflow result to A2UI format
    
    Args:
        workflow_result: Workflow execution result dictionary
    
    Returns:
        A2UI response dictionary
    """
    if not A2UI_AVAILABLE:
        raise ServiceUnavailableError("a2ui")
    
    try:
        a2ui_response = workflow_result_to_a2ui(workflow_result)
        logger.info("Workflow result converted to A2UI")
        return a2ui_response.to_dict()
    except Exception as e:
        logger.error(f"A2UI conversion failed: {e}", exc_info=True)
        raise ServiceUnavailableError("a2ui", f"A2UI conversion error: {str(e)}")


@router.post("/surface/create")
def create_surface(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new A2UI surface

    Args:
        data: Surface creation data with optional surface_id and catalog_id

    Returns:
        Surface creation response
    """
    try:
        surface_id = data.get("surface_id")
        catalog_id = data.get("catalog_id", "https://a2ui.dev/specification/0.9/standard_catalog_definition.json")

        surface = enhanced_adapter.create_surface(surface_id, catalog_id)

        logger.info(f"A2UI surface created: {surface.surface_id}")
        return {
            "surface_id": surface.surface_id,
            "catalog_id": surface.catalog_id,
            "message": surface.to_create_message()
        }
    except Exception as e:
        logger.error(f"Surface creation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Surface creation error: {str(e)}")


@router.delete("/surface/{surface_id}")
def delete_surface(surface_id: str) -> Dict[str, Any]:
    """
    Delete an A2UI surface

    Args:
        surface_id: ID of the surface to delete

    Returns:
        Surface deletion response
    """
    try:
        if surface_id in enhanced_adapter.surfaces:
            surface = enhanced_adapter.surfaces[surface_id]
            delete_message = surface.to_delete_message()
            del enhanced_adapter.surfaces[surface_id]

            logger.info(f"A2UI surface deleted: {surface_id}")
            return {
                "surface_id": surface_id,
                "message": delete_message
            }
        else:
            raise HTTPException(status_code=404, detail=f"Surface not found: {surface_id}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Surface deletion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Surface deletion error: {str(e)}")


@router.get("/surfaces")
def list_surfaces() -> Dict[str, Any]:
    """
    List all active A2UI surfaces

    Returns:
        List of active surfaces
    """
    try:
        surfaces_info = []
        for surface_id, surface in enhanced_adapter.surfaces.items():
            surfaces_info.append({
                "surface_id": surface_id,
                "catalog_id": surface.catalog_id,
                "component_count": len(surface.components or {}),
                "has_data_model": bool(surface.data_model)
            })

        return {
            "surfaces": surfaces_info,
            "total_count": len(surfaces_info)
        }
    except Exception as e:
        logger.error(f"Surface listing failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Surface listing error: {str(e)}")


@router.post("/enhanced/workflow-result")
def generate_enhanced_workflow_result_ui(workflow_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate enhanced A2UI messages for workflow execution result

    Args:
        workflow_result: Workflow execution result data

    Returns:
        List of A2UI v0.9 messages
    """
    if not A2UI_AVAILABLE:
        raise ServiceUnavailableError("a2ui")

    try:
        messages = enhanced_adapter.generate_workflow_ui(workflow_result)
        logger.info(f"Enhanced A2UI workflow result UI generated with {len(messages)} messages")
        return messages
    except Exception as e:
        logger.error(f"Enhanced A2UI workflow result generation failed: {e}", exc_info=True)
        raise ServiceUnavailableError("a2ui", f"A2UI generation error: {str(e)}")

