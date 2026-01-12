"""
Orchestration Adapter Layer
Bridges ToolOrchestra-style workflows with Rust workflow engine and Python model services
"""

import json
from typing import Dict, List, Optional, Any
import aiohttp
from workflow_schema import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge,
    NodeType,
    ProcessType,
    workflow_to_toolorchestra_format,
    toolorchestra_to_workflow_format,
)


class OrchestrationAdapter:
    """Adapter between frontend workflow definitions and Rust workflow engine"""
    
    def __init__(self, rust_backend_url: str = "http://127.0.0.1:11435"):
        self.rust_backend_url = rust_backend_url
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    def workflow_to_rust_format(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Convert frontend workflow definition to Rust workflow engine format
        """
        steps = []
        for node in workflow.nodes:
            step = {
                "id": node.id,
                "depends_on": [],
                "parameters": {}
            }
            
            # Map node type to Rust workflow step type
            if node.type == NodeType.START:
                continue  # Start nodes don't need steps
            
            elif node.type == NodeType.PROCESS:
                if node.process_type == ProcessType.TRANSLATION:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": "translate_text",
                        "arguments": {
                            "text": f"{{{{step_{node.id}.text}}}}"
                        }
                    }
                elif node.process_type == ProcessType.ANALYSIS:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": "analyze_invoice",
                        "arguments": {
                            "text": f"{{{{step_{node.id}.text}}}}"
                        }
                    }
                else:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": node.process_type.value if node.process_type else "unknown",
                        "arguments": node.config
                    }
            
            elif node.type == NodeType.DECISION:
                # Decision nodes become conditional steps
                step["step_type"] = {
                    "type": "conditional",
                    "condition": node.condition or "true",
                    "if_true": None,  # Will be filled from edges
                    "if_false": None
                }
            
            elif node.type == NodeType.ACTION:
                step["step_type"] = {
                    "type": "tool",
                    "tool_name": node.label.lower().replace(" ", "_"),
                    "arguments": node.config
                }
            
            # Add dependencies from edges
            for edge in workflow.edges:
                if edge.target == node.id:
                    step["depends_on"].append(edge.source)
            
            steps.append(step)
        
        return {
            "workflow": {
                "id": workflow.id,
                "name": workflow.name,
                "description": workflow.description,
                "steps": steps,
                "inputs": workflow.inputs,
                "outputs": workflow.outputs
            },
            "context": {}
        }
    
    async def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        inputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute workflow using Rust backend
        """
        rust_format = self.workflow_to_rust_format(workflow)
        rust_format["context"] = inputs
        
        session = await self._get_session()
        url = f"{self.rust_backend_url}/api/workflows/execute"
        
        try:
            async with session.post(url, json=rust_format) as response:
                if response.status == 200:
                    result = await response.json()
                    return result
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Rust backend returned {response.status}: {error_text}"
                    }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Failed to connect to Rust backend: {str(e)}"
            }
    
    def create_default_invoice_workflow(self) -> WorkflowDefinition:
        """
        Create default invoice processing workflow
        """
        nodes = [
            WorkflowNode(
                id="start",
                type=NodeType.START,
                label="Invoice Received",
                position={"x": 100, "y": 100}
            ),
            WorkflowNode(
                id="ocr_extract",
                type=NodeType.PROCESS,
                label="OCR Extraction",
                process_type=ProcessType.OCR,
                position={"x": 100, "y": 200}
            ),
            WorkflowNode(
                id="translate",
                type=NodeType.PROCESS,
                label="Translate Arabic",
                process_type=ProcessType.TRANSLATION,
                position={"x": 100, "y": 300}
            ),
            WorkflowNode(
                id="analyze",
                type=NodeType.PROCESS,
                label="Analyze Invoice",
                process_type=ProcessType.ANALYSIS,
                position={"x": 100, "y": 400}
            ),
            WorkflowNode(
                id="confidence_check",
                type=NodeType.DECISION,
                label="Confidence > 85%?",
                condition="confidence >= 0.85",
                threshold=0.85,
                position={"x": 100, "y": 500}
            ),
            WorkflowNode(
                id="auto_submit",
                type=NodeType.ACTION,
                label="Auto Submit",
                process_type=ProcessType.SUBMISSION,
                position={"x": 50, "y": 600}
            ),
            WorkflowNode(
                id="flag_review",
                type=NodeType.ACTION,
                label="Flag for Review",
                position={"x": 150, "y": 600}
            ),
            WorkflowNode(
                id="end",
                type=NodeType.END,
                label="Complete",
                position={"x": 100, "y": 700}
            )
        ]
        
        edges = [
            WorkflowEdge(id="e1", source="start", target="ocr_extract"),
            WorkflowEdge(id="e2", source="ocr_extract", target="translate"),
            WorkflowEdge(id="e3", source="translate", target="analyze"),
            WorkflowEdge(id="e4", source="analyze", target="confidence_check"),
            WorkflowEdge(id="e5", source="confidence_check", target="auto_submit", condition="yes"),
            WorkflowEdge(id="e6", source="confidence_check", target="flag_review", condition="no"),
            WorkflowEdge(id="e7", source="auto_submit", target="end"),
            WorkflowEdge(id="e8", source="flag_review", target="end")
        ]
        
        return WorkflowDefinition(
            id="default-invoice-processing",
            name="Default Invoice Processing",
            description="Standard invoice processing pipeline",
            nodes=nodes,
            edges=edges,
            inputs={"invoice_document": ""},
            outputs=["auto_submit", "flag_review"]
        )

