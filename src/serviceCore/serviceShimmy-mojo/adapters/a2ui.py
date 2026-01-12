"""
A2UI Python Adapter
Converts between Python data structures and A2UI format
Based on https://github.com/google/A2UI
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import json


class ComponentType(str, Enum):
    """A2UI component types"""
    TEXT_FIELD = "text-field"
    BUTTON = "button"
    CARD = "card"
    TEXT = "text"
    FORM = "form"
    SELECT = "select"
    CHECKBOX = "checkbox"
    RADIO = "radio"


@dataclass
class A2UIComponent:
    """A2UI Component representation"""
    id: str
    component_type: str
    properties: Optional[Dict[str, Any]] = None
    components: Optional[List['A2UIComponent']] = None
    data_path: Optional[str] = None
    events: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI JSON format"""
        result = {
            "id": self.id,
            "type": self.component_type
        }
        if self.properties:
            result["properties"] = self.properties
        if self.components:
            result["components"] = [c.to_dict() for c in self.components]
        if self.data_path:
            result["dataPath"] = self.data_path
        if self.events:
            result["events"] = self.events
        return result

    @classmethod
    def text_field(cls, id: str, label: str, data_path: Optional[str] = None) -> 'A2UIComponent':
        """Create a text field component"""
        return cls(
            id=id,
            component_type=ComponentType.TEXT_FIELD.value,
            properties={"label": label},
            data_path=data_path
        )

    @classmethod
    def button(cls, id: str, label: str, action: str) -> 'A2UIComponent':
        """Create a button component"""
        return cls(
            id=id,
            component_type=ComponentType.BUTTON.value,
            properties={"label": label},
            events={"click": action}
        )

    @classmethod
    def card(cls, id: str, title: str) -> 'A2UIComponent':
        """Create a card component"""
        return cls(
            id=id,
            component_type=ComponentType.CARD.value,
            properties={"title": title},
            components=[]
        )

    @classmethod
    def text(cls, id: str, content: str) -> 'A2UIComponent':
        """Create a text component"""
        return cls(
            id=id,
            component_type=ComponentType.TEXT.value,
            properties={"content": content}
        )

    def add_child(self, child: 'A2UIComponent') -> 'A2UIComponent':
        """Add a child component"""
        if self.components is None:
            self.components = []
        self.components.append(child)
        return self


@dataclass
class A2UISurface:
    """A2UI Surface (container)"""
    id: str
    surface_type: str
    components: List[A2UIComponent] = field(default_factory=list)
    data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI JSON format"""
        result = {
            "id": self.id,
            "type": self.surface_type,
            "components": [c.to_dict() for c in self.components]
        }
        if self.data:
            result["data"] = self.data
        return result


@dataclass
class A2UIResponse:
    """A2UI Response - root structure"""
    surface: A2UISurface
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to A2UI JSON format"""
        result = {
            "surface": self.surface.to_dict()
        }
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


def generate_invoice_processing_ui(invoice_data: Dict[str, Any]) -> A2UIResponse:
    """Generate A2UI response for invoice processing workflow"""
    surface = A2UISurface(
        id="invoice-processing",
        surface_type="card",
        data=invoice_data,
        components=[
            A2UIComponent.card("invoice-header", "Invoice Details")
                .add_child(A2UIComponent.text("invoice-number", f"Invoice: {invoice_data.get('invoice_number', 'N/A')}"))
                .add_child(A2UIComponent.text("supplier-name", f"Supplier: {invoice_data.get('supplier_name', 'N/A')}"))
                .add_child(A2UIComponent.text("amount", f"Amount: {invoice_data.get('amount', 'N/A')}")),
            A2UIComponent.form("invoice-form")
                .add_child(A2UIComponent.text_field("supplier-field", "Supplier Name", "supplier_name"))
                .add_child(A2UIComponent.text_field("invoice-number-field", "Invoice Number", "invoice_number"))
                .add_child(A2UIComponent.text_field("amount-field", "Amount", "amount")),
            A2UIComponent.button("submit-button", "Submit to Ariba", "submit_invoice")
        ]
    )
    return A2UIResponse(surface=surface)


def workflow_result_to_a2ui(workflow_result: Dict[str, Any]) -> A2UIResponse:
    """Convert workflow execution result to A2UI response"""
    components = []
    
    # Status card
    status_card = A2UIComponent.card("workflow-status", "Workflow Execution")
    status_card.add_child(
        A2UIComponent.text("status-text", "Success" if workflow_result.get("success") else "Failed")
    )
    status_card.add_child(
        A2UIComponent.text("execution-time", f"Execution time: {workflow_result.get('execution_time_ms', 0)}ms")
    )
    components.append(status_card)
    
    # Step results
    step_results = workflow_result.get("step_results", {})
    for step_id, step_result in step_results.items():
        step_card = A2UIComponent.card(f"step-{step_id}", f"Step: {step_id}")
        step_card.add_child(
            A2UIComponent.text(f"step-{step_id}-status", 
                             "Success" if step_result.get("success") else "Failed")
        )
        if step_result.get("error"):
            step_card.add_child(
                A2UIComponent.text(f"step-{step_id}-error", step_result["error"])
            )
        components.append(step_card)
    
    surface = A2UISurface(
        id="workflow-result",
        surface_type="card",
        components=components,
        data=workflow_result
    )
    return A2UIResponse(surface=surface)



# Alias for backward compatibility
A2UIAdapter = A2UISurface


async def check_a2ui_health(a2ui_url: str = "http://a2ui:8000") -> Dict[str, Any]:
    """
    Check A2UI service health
    
    Args:
        a2ui_url: Base URL for A2UI service
        
    Returns:
        Health check result
    """
    # A2UI is a data format, not a service with health check
    return {
        "status": "healthy",
        "url": a2ui_url,
        "note": "A2UI is a data format adapter, not a service"
    }
