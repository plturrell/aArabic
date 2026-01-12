"""
Enhanced Orchestration Adapter Layer
Bridges ToolOrchestra-style workflows with Shimmy AI workflow engine and Python model services
Provides comprehensive integration with all vendor components
"""

import json
import asyncio
import websockets
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
import logging
from backend.schemas.workflow import (
    WorkflowDefinition,
    WorkflowNode,
    WorkflowEdge,
    NodeType,
    ProcessType,
    workflow_to_toolorchestra_format,
    toolorchestra_to_workflow_format,
)

logger = logging.getLogger(__name__)


class ShimmyAdapter:
    """Enhanced adapter for Shimmy AI workflow engine integration"""

    def __init__(self, shimmy_url: Optional[str] = None):
        from backend.config.settings import settings
        self.shimmy_url = shimmy_url or settings.shimmy_backend_url
        self.default_model = settings.default_local_model
        self.session: Optional[aiohttp.ClientSession] = None
        self.ws_connection: Optional[websockets.WebSocketServerProtocol] = None
        self._health_status = {"status": "unknown", "last_check": None}

    async def health_check(self) -> Dict[str, Any]:
        """Check Shimmy service health"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.shimmy_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._health_status = {
                        "status": "healthy",
                        "last_check": health_data.get("timestamp"),
                        "service": health_data.get("service"),
                        "version": health_data.get("version"),
                        "models": health_data.get("models", {}),
                        "features": health_data.get("features", {})
                    }
                    return self._health_status
                else:
                    self._health_status = {"status": "unhealthy", "error": f"HTTP {response.status}"}
                    return self._health_status
        except Exception as e:
            self._health_status = {"status": "error", "error": str(e)}
            return self._health_status

    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Shimmy"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.shimmy_url}/api/models") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to list models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model in Shimmy"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.shimmy_url}/api/models/{model_name}/load") as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool via Shimmy"""
        try:
            session = await self._get_session()
            payload = {"parameters": parameters}
            async with session.post(f"{self.shimmy_url}/api/tools/{tool_name}/execute", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        """Close HTTP session and WebSocket connections"""
        if self.session and not self.session.closed:
            await self.session.close()
        if self.ws_connection:
            await self.ws_connection.close()

    async def stream_generation(self, prompt: str, model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """Stream text generation from Shimmy via WebSocket"""
        try:
            if model_name is None:
                model_name = self.default_model

            ws_url = self.shimmy_url.replace("http://", "ws://").replace("https://", "wss://")
            async with websockets.connect(f"{ws_url}/ws/generate") as websocket:
                payload = {
                    "model": model_name,
                    "prompt": prompt,
                    "stream": True
                }
                await websocket.send(json.dumps(payload))

                async for message in websocket:
                    data = json.loads(message)
                    if "content" in data:
                        yield data["content"]
                    elif "error" in data:
                        logger.error(f"WebSocket error: {data['error']}")
                        break
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def workflow_to_shimmy_format(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """
        Convert frontend workflow definition to Shimmy workflow engine format
        Enhanced with A2UI integration and tool orchestration
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
                            "text": f"{{{{step_{node.id}.text}}}}",
                            "source_lang": "ar",
                            "target_lang": "en"
                        }
                    }
                elif node.process_type == ProcessType.ANALYSIS:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": "analyze_invoice",
                        "arguments": {
                            "text": f"{{{{step_{node.id}.text}}}}",
                            "confidence_threshold": 0.85
                        }
                    }
                elif node.process_type == ProcessType.OCR:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": "extract_text_ocr",
                        "arguments": {
                            "document": f"{{{{step_{node.id}.document}}}}",
                            "language": "ar"
                        }
                    }
                else:
                    step["step_type"] = {
                        "type": "tool",
                        "tool_name": node.process_type.value if node.process_type else "unknown",
                        "arguments": node.config or {}
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
                "outputs": workflow.outputs,
                "metadata": {
                    "created_by": "ai_nucleus_backend",
                    "version": "1.0",
                    "vendor_integrations": ["shimmy", "a2ui", "memgraph"]
                }
            },
            "context": {},
            "execution_options": {
                "parallel_execution": True,
                "timeout_seconds": 300,
                "retry_failed_steps": True,
                "generate_a2ui": True
            }
        }
    
    async def execute_workflow(
        self,
        workflow: WorkflowDefinition,
        inputs: Dict[str, Any],
        stream_updates: bool = False
    ) -> Dict[str, Any]:
        """
        Execute workflow using Shimmy backend with enhanced capabilities
        """
        shimmy_format = self.workflow_to_shimmy_format(workflow)
        shimmy_format["context"] = inputs

        session = await self._get_session()
        url = f"{self.shimmy_url}/api/workflows/execute"

        try:
            async with session.post(url, json=shimmy_format) as response:
                if response.status == 200:
                    result = await response.json()

                    # Enhance result with A2UI generation if requested
                    if shimmy_format.get("execution_options", {}).get("generate_a2ui"):
                        result["a2ui_response"] = await self._generate_workflow_a2ui(result)

                    return result
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "error": f"Shimmy backend returned {response.status}: {error_text}",
                        "workflow_id": workflow.id
                    }
        except aiohttp.ClientError as e:
            return {
                "success": False,
                "error": f"Failed to connect to Shimmy backend: {str(e)}",
                "workflow_id": workflow.id
            }

    async def _generate_workflow_a2ui(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate A2UI response for workflow result"""
        try:
            session = await self._get_session()
            url = f"{self.shimmy_url}/api/a2ui/generate"
            payload = {
                "type": "workflow_result",
                "workflow_result": workflow_result
            }

            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"A2UI generation failed: HTTP {response.status}")
                    return {}
        except Exception as e:
            logger.error(f"Error generating A2UI: {e}")
            return {}
    
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
            description="Enhanced invoice processing pipeline with vendor integrations",
            nodes=nodes,
            edges=edges,
            inputs={"invoice_document": "", "confidence_threshold": 0.85},
            outputs=["auto_submit", "flag_review"],
            metadata={
                "vendor_integrations": ["shimmy", "a2ui", "memgraph"],
                "supports_streaming": True,
                "supports_a2ui": True
            }
        )


# Backward compatibility class
class OrchestrationAdapter(ShimmyAdapter):
    """Legacy adapter class for backward compatibility"""

    def __init__(self, rust_backend_url: Optional[str] = None):
        super().__init__(rust_backend_url)
        logger.warning("OrchestrationAdapter is deprecated. Use ShimmyAdapter instead.")

    def workflow_to_rust_format(self, workflow: WorkflowDefinition) -> Dict[str, Any]:
        """Legacy method - redirects to new implementation"""
        return self.workflow_to_shimmy_format(workflow)

