"""
Enhanced A2UI (Agent to User Interface) Adapter
Implements A2UI v0.9 specification with full vendor integration
Provides dynamic UI generation capabilities for agent responses
"""

import json
import logging
import uuid
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

# A2UI v0.9 Constants
A2UI_VERSION = "0.9"
STANDARD_CATALOG_ID = "https://a2ui.dev/specification/0.9/standard_catalog_definition.json"
A2UI_MIME_TYPE = "application/json+a2ui"


class A2UIMessageType(str, Enum):
    """A2UI v0.9 message types"""
    CREATE_SURFACE = "createSurface"
    UPDATE_COMPONENTS = "updateComponents"
    UPDATE_DATA_MODEL = "updateDataModel"
    DELETE_SURFACE = "deleteSurface"


class A2UIComponentType(str, Enum):
    """A2UI v0.9 standard component types"""
    # Layout Components
    COLUMN = "Column"
    ROW = "Row"
    CARD = "Card"
    TABS = "Tabs"
    
    # Text Components
    TEXT = "Text"
    MARKDOWN = "Markdown"
    
    # Input Components
    TEXT_FIELD = "TextField"
    TEXT_AREA = "TextArea"
    BUTTON = "Button"
    CHECKBOX = "CheckBox"
    RADIO_GROUP = "RadioGroup"
    SELECT = "Select"
    MULTI_SELECT = "MultiSelect"
    
    # Display Components
    IMAGE = "Image"
    PROGRESS_BAR = "ProgressBar"
    DIVIDER = "Divider"


@dataclass
class A2UIDataBinding:
    """A2UI data binding representation"""
    path: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"path": self.path}


@dataclass
class A2UIComponent:
    """Enhanced A2UI Component for v0.9 specification"""
    component_id: str
    component_type: A2UIComponentType
    properties: Optional[Dict[str, Any]] = None
    children: Optional[Union[List[str], Dict[str, Any]]] = None
    data_bindings: Optional[Dict[str, A2UIDataBinding]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI v0.9 JSON format"""
        result = {
            "id": self.component_id,
            "component": self.component_type.value
        }
        
        if self.properties:
            result.update(self.properties)
            
        if self.children:
            if isinstance(self.children, list):
                result["children"] = self.children
            else:
                result["children"] = self.children
                
        # Handle data bindings
        if self.data_bindings:
            for prop_name, binding in self.data_bindings.items():
                result[prop_name] = binding.to_dict()
        
        return result


@dataclass
class A2UISurface:
    """A2UI Surface representation"""
    surface_id: str
    catalog_id: str = STANDARD_CATALOG_ID
    components: Optional[Dict[str, A2UIComponent]] = None
    data_model: Optional[Dict[str, Any]] = None
    theme: Optional[Dict[str, Any]] = None
    
    def to_create_message(self) -> Dict[str, Any]:
        """Generate createSurface message"""
        return {
            "createSurface": {
                "surfaceId": self.surface_id,
                "catalogId": self.catalog_id
            }
        }
    
    def to_update_components_message(self) -> Dict[str, Any]:
        """Generate updateComponents message"""
        components_dict = {}
        if self.components:
            components_dict = {
                comp_id: comp.to_dict() 
                for comp_id, comp in self.components.items()
            }
        
        message = {
            "updateComponents": {
                "surfaceId": self.surface_id,
                "components": components_dict
            }
        }
        
        if self.theme:
            message["updateComponents"]["theme"] = self.theme
            
        return message
    
    def to_update_data_message(self) -> Dict[str, Any]:
        """Generate updateDataModel message"""
        return {
            "updateDataModel": {
                "surfaceId": self.surface_id,
                "dataModel": self.data_model or {}
            }
        }
    
    def to_delete_message(self) -> Dict[str, Any]:
        """Generate deleteSurface message"""
        return {
            "deleteSurface": {
                "surfaceId": self.surface_id
            }
        }


class EnhancedA2UIAdapter:
    """
    Enhanced A2UI adapter implementing v0.9 specification
    Provides comprehensive UI generation for workflow results and agent interactions
    """
    
    def __init__(self):
        self.surfaces: Dict[str, A2UISurface] = {}
        
    def create_surface(
        self,
        surface_id: Optional[str] = None,
        catalog_id: str = STANDARD_CATALOG_ID
    ) -> A2UISurface:
        """Create a new A2UI surface"""
        if surface_id is None:
            surface_id = f"surface_{uuid.uuid4().hex[:8]}"
            
        surface = A2UISurface(
            surface_id=surface_id,
            catalog_id=catalog_id,
            components={},
            data_model={}
        )
        
        self.surfaces[surface_id] = surface
        return surface
    
    def create_component(
        self,
        component_type: A2UIComponentType,
        component_id: Optional[str] = None,
        **properties
    ) -> A2UIComponent:
        """Create a new A2UI component"""
        if component_id is None:
            component_id = f"{component_type.value.lower()}_{uuid.uuid4().hex[:8]}"
            
        return A2UIComponent(
            component_id=component_id,
            component_type=component_type,
            properties=properties
        )

    def generate_workflow_ui(
        self,
        workflow_result: Dict[str, Any],
        surface_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate A2UI messages for workflow execution result"""
        surface = self.create_surface(surface_id)

        # Create main layout
        main_column = self.create_component(
            A2UIComponentType.COLUMN,
            "main_column",
            spacing="medium"
        )

        # Workflow status card
        status_card = self.create_component(
            A2UIComponentType.CARD,
            "status_card",
            title="Workflow Execution Status"
        )

        status_text = self.create_component(
            A2UIComponentType.TEXT,
            "status_text",
            text="✅ Success" if workflow_result.get("success") else "❌ Failed",
            variant="h6"
        )

        execution_time = self.create_component(
            A2UIComponentType.TEXT,
            "execution_time",
            text=f"Execution time: {workflow_result.get('execution_time_ms', 0)}ms"
        )

        # Add components to surface
        surface.components = {
            "main_column": main_column,
            "status_card": status_card,
            "status_text": status_text,
            "execution_time": execution_time
        }

        # Set up component hierarchy
        main_column.children = ["status_card"]
        status_card.children = ["status_text", "execution_time"]

        # Add step results if available
        step_results = workflow_result.get("step_results", {})
        step_components = []

        for step_id, step_result in step_results.items():
            step_card_id = f"step_card_{step_id}"
            step_card = self.create_component(
                A2UIComponentType.CARD,
                step_card_id,
                title=f"Step: {step_id}"
            )

            step_status_id = f"step_status_{step_id}"
            step_status = self.create_component(
                A2UIComponentType.TEXT,
                step_status_id,
                text="✅ Success" if step_result.get("success") else "❌ Failed"
            )

            step_card.children = [step_status_id]
            surface.components[step_card_id] = step_card
            surface.components[step_status_id] = step_status
            step_components.append(step_card_id)

            # Add error message if present
            if step_result.get("error"):
                error_text_id = f"error_text_{step_id}"
                error_text = self.create_component(
                    A2UIComponentType.TEXT,
                    error_text_id,
                    text=f"Error: {step_result['error']}",
                    color="error"
                )
                step_card.children.append(error_text_id)
                surface.components[error_text_id] = error_text

        # Add step components to main column
        if step_components:
            main_column.children.extend(step_components)

        # Set data model
        surface.data_model = {
            "workflow": workflow_result,
            "timestamp": datetime.now().isoformat()
        }

        # Generate message sequence
        messages = [
            surface.to_create_message(),
            surface.to_update_components_message(),
            surface.to_update_data_message()
        ]

        return messages

    def generate_invoice_processing_ui(
        self,
        invoice_data: Dict[str, Any],
        surface_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Generate A2UI messages for invoice processing interface"""
        surface = self.create_surface(surface_id)

        # Main layout
        main_column = self.create_component(
            A2UIComponentType.COLUMN,
            "main_column",
            spacing="large"
        )

        # Invoice details card
        details_card = self.create_component(
            A2UIComponentType.CARD,
            "invoice_details",
            title="Invoice Details"
        )

        # Invoice fields
        invoice_number = self.create_component(
            A2UIComponentType.TEXT_FIELD,
            "invoice_number",
            label="Invoice Number",
            value=invoice_data.get("invoice_number", ""),
            required=True
        )

        supplier_name = self.create_component(
            A2UIComponentType.TEXT_FIELD,
            "supplier_name",
            label="Supplier Name",
            value=invoice_data.get("supplier_name", ""),
            required=True
        )

        amount = self.create_component(
            A2UIComponentType.TEXT_FIELD,
            "amount",
            label="Amount",
            value=str(invoice_data.get("amount", "")),
            type="number",
            required=True
        )

        # Action buttons
        button_row = self.create_component(
            A2UIComponentType.ROW,
            "button_row",
            spacing="medium",
            justifyContent="flex-end"
        )

        submit_button = self.create_component(
            A2UIComponentType.BUTTON,
            "submit_button",
            text="Submit to Ariba",
            variant="primary",
            onClick="submit_invoice"
        )

        cancel_button = self.create_component(
            A2UIComponentType.BUTTON,
            "cancel_button",
            text="Cancel",
            variant="secondary",
            onClick="cancel_invoice"
        )

        # Set up component hierarchy
        main_column.children = ["invoice_details", "button_row"]
        details_card.children = ["invoice_number", "supplier_name", "amount"]
        button_row.children = ["cancel_button", "submit_button"]

        # Add all components to surface
        surface.components = {
            "main_column": main_column,
            "invoice_details": details_card,
            "invoice_number": invoice_number,
            "supplier_name": supplier_name,
            "amount": amount,
            "button_row": button_row,
            "submit_button": submit_button,
            "cancel_button": cancel_button
        }

        # Set data model with bindings
        surface.data_model = {
            "invoice": invoice_data,
            "form_data": {
                "invoice_number": invoice_data.get("invoice_number", ""),
                "supplier_name": invoice_data.get("supplier_name", ""),
                "amount": invoice_data.get("amount", "")
            }
        }

        # Generate message sequence
        messages = [
            surface.to_create_message(),
            surface.to_update_components_message(),
            surface.to_update_data_message()
        ]

        return messages


# Alias for backward compatibility
# Alias for backward compatibility
A2UIEnhancedAdapter = A2UISurface


async def check_a2uienhanced_health(a2uienhanced_url: str = "http://a2ui-enhanced:8000") -> Dict[str, Any]:
    """
    Check A2UIEnhanced service health
    
    Args:
        a2uienhanced_url: Base URL for A2UIEnhanced service
        
    Returns:
        Health check result
    """
    service = A2UIEnhancedService(base_url=a2uienhanced_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
