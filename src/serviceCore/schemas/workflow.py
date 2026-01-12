"""
Workflow Schema Definitions for ToolOrchestra Integration
Defines JSON schemas for workflow definitions and ToolOrchestra format mapping
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Literal
from pydantic import BaseModel, Field
from enum import Enum


class NodeType(str, Enum):
    """Node types in the workflow"""
    START = "start"
    END = "end"
    PROCESS = "process"
    DECISION = "decision"
    ACTION = "action"


class ProcessType(str, Enum):
    """Types of processing steps"""
    OCR = "ocr"
    TRANSLATION = "translation"
    ANALYSIS = "analysis"
    VALIDATION = "validation"
    SUBMISSION = "submission"


class WorkflowNode(BaseModel):
    """Represents a node in the workflow"""
    id: str = Field(..., description="Unique node identifier")
    type: NodeType = Field(..., description="Node type")
    label: str = Field(..., description="Display label")
    process_type: Optional[ProcessType] = Field(None, description="Process type if applicable")
    position: Dict[str, float] = Field(default_factory=lambda: {"x": 0, "y": 0}, description="Canvas position")
    
    # Configuration
    config: Dict[str, Any] = Field(default_factory=dict, description="Node-specific configuration")
    
    # Decision node specific
    condition: Optional[str] = Field(None, description="Condition expression for decision nodes")
    threshold: Optional[float] = Field(None, description="Confidence threshold")
    
    # Model selection
    model: Optional[str] = Field(None, description="Model to use (camelbert, m2m100)")
    
    # Error handling
    retry_count: int = Field(default=0, description="Number of retries on failure")
    error_handler: Optional[str] = Field(None, description="Error handling strategy")


class WorkflowEdge(BaseModel):
    """Represents an edge/connection in the workflow"""
    id: str = Field(..., description="Unique edge identifier")
    source: str = Field(..., description="Source node ID")
    target: str = Field(..., description="Target node ID")
    label: Optional[str] = Field(None, description="Edge label")
    condition: Optional[str] = Field(None, description="Condition for conditional edges")


class WorkflowDefinition(BaseModel):
    """Complete workflow definition"""
    id: str = Field(..., description="Workflow identifier")
    name: str = Field(..., description="Workflow name")
    description: str = Field(default="", description="Workflow description")
    version: str = Field(default="1.0.0", description="Workflow version")
    
    nodes: List[WorkflowNode] = Field(default_factory=list, description="Workflow nodes")
    edges: List[WorkflowEdge] = Field(default_factory=list, description="Workflow edges")
    
    # Input/output definitions
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Workflow inputs")
    outputs: List[str] = Field(default_factory=list, description="Output node IDs")
    
    # Metadata
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")
    tags: List[str] = Field(default_factory=list, description="Workflow tags")


class WorkflowExecutionRequest(BaseModel):
    """Request to execute a workflow"""
    workflow_id: str = Field(..., description="Workflow ID to execute")
    inputs: Dict[str, Any] = Field(default_factory=dict, description="Input data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")


class StepResult(BaseModel):
    """Result of a workflow step execution"""
    step_id: str
    success: bool
    result: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    execution_time_ms: float = 0.0
    confidence: Optional[float] = None


class WorkflowExecutionResult(BaseModel):
    """Result of workflow execution"""
    execution_id: str
    workflow_id: str
    status: Literal["pending", "running", "completed", "failed", "cancelled"]
    step_results: Dict[str, StepResult] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    total_time_ms: float = 0.0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


def workflow_to_toolorchestra_format(workflow: WorkflowDefinition) -> Dict[str, Any]:
    """
    Convert workflow definition to ToolOrchestra format
    """
    # Map nodes to ToolOrchestra steps
    steps = []
    for node in workflow.nodes:
        step = {
            "id": node.id,
            "type": node.type.value,
            "label": node.label,
        }
        
        if node.process_type:
            step["process_type"] = node.process_type.value
        
        if node.config:
            step["config"] = node.config
        
        if node.model:
            step["model"] = node.model
        
        if node.condition:
            step["condition"] = node.condition
        
        if node.threshold is not None:
            step["threshold"] = node.threshold
        
        steps.append(step)
    
    # Map edges to dependencies
    dependencies = {}
    for edge in workflow.edges:
        if edge.target not in dependencies:
            dependencies[edge.target] = []
        dependencies[edge.target].append({
            "source": edge.source,
            "condition": edge.condition
        })
    
    return {
        "workflow_id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "steps": steps,
        "dependencies": dependencies,
        "inputs": workflow.inputs,
        "outputs": workflow.outputs
    }


def toolorchestra_to_workflow_format(data: Dict[str, Any]) -> WorkflowDefinition:
    """
    Convert ToolOrchestra format to workflow definition
    """
    nodes = []
    for step in data.get("steps", []):
        node = WorkflowNode(
            id=step["id"],
            type=NodeType(step["type"]),
            label=step.get("label", step["id"]),
            process_type=ProcessType(step["process_type"]) if "process_type" in step else None,
            config=step.get("config", {}),
            model=step.get("model"),
            condition=step.get("condition"),
            threshold=step.get("threshold")
        )
        nodes.append(node)
    
    edges = []
    for target, deps in data.get("dependencies", {}).items():
        for dep in deps:
            edge = WorkflowEdge(
                id=f"{dep['source']}-{target}",
                source=dep["source"],
                target=target,
                condition=dep.get("condition")
            )
            edges.append(edge)
    
    return WorkflowDefinition(
        id=data["workflow_id"],
        name=data.get("name", "Untitled Workflow"),
        description=data.get("description", ""),
        nodes=nodes,
        edges=edges,
        inputs=data.get("inputs", {}),
        outputs=data.get("outputs", [])
    )

