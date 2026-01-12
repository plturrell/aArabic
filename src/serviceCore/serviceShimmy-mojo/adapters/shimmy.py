"""
Shimmy AI Service Adapter
Direct integration with Shimmy workflow engine and model services
"""

import json
import asyncio
import websockets
from typing import Dict, List, Optional, Any, AsyncGenerator
import aiohttp
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ShimmyServiceStatus(str, Enum):
    """Shimmy service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass
class ShimmyModel:
    """Shimmy model information"""
    name: str
    path: str
    template: Optional[str] = None
    ctx_len: Optional[int] = None
    loaded: bool = False
    size_mb: Optional[float] = None


@dataclass
class ShimmyTool:
    """Shimmy tool information"""
    name: str
    description: str
    parameters: Dict[str, Any]
    enabled: bool = True


class ShimmyService:
    """
    Enhanced Shimmy AI service integration
    Provides comprehensive access to Shimmy's capabilities
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:11435", default_model: str = "glm4:9b"):
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self.session: Optional[aiohttp.ClientSession] = None
        self._models_cache: List[ShimmyModel] = []
        self._tools_cache: List[ShimmyTool] = []
        self._last_health_check: Optional[Dict[str, Any]] = None
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def close(self):
        """Close HTTP session"""
        if self.session and not self.session.closed:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    health_data = await response.json()
                    self._last_health_check = {
                        "status": ShimmyServiceStatus.HEALTHY,
                        "timestamp": health_data.get("timestamp"),
                        "service": health_data.get("service"),
                        "version": health_data.get("version"),
                        "models": health_data.get("models", {}),
                        "system": health_data.get("system", {}),
                        "features": health_data.get("features", {}),
                        "endpoints": health_data.get("endpoints", [])
                    }
                    return self._last_health_check
                else:
                    error_data = {
                        "status": ShimmyServiceStatus.UNHEALTHY,
                        "error": f"HTTP {response.status}",
                        "timestamp": None
                    }
                    self._last_health_check = error_data
                    return error_data
        except Exception as e:
            error_data = {
                "status": ShimmyServiceStatus.ERROR,
                "error": str(e),
                "timestamp": None
            }
            self._last_health_check = error_data
            return error_data
    
    async def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/diag") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {"error": f"HTTP {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    async def list_models(self, refresh_cache: bool = False) -> List[ShimmyModel]:
        """List available models"""
        if not refresh_cache and self._models_cache:
            return self._models_cache
            
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/models") as response:
                if response.status == 200:
                    models_data = await response.json()
                    self._models_cache = [
                        ShimmyModel(
                            name=model.get("name", ""),
                            path=model.get("path", ""),
                            template=model.get("template"),
                            ctx_len=model.get("ctx_len"),
                            loaded=model.get("loaded", False),
                            size_mb=model.get("size_mb")
                        )
                        for model in models_data.get("models", [])
                    ]
                    return self._models_cache
                else:
                    logger.error(f"Failed to list models: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def discover_models(self) -> Dict[str, Any]:
        """Trigger model discovery"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/models/discover") as response:
                result = await response.json()
                # Refresh cache after discovery
                await self.list_models(refresh_cache=True)
                return result
        except Exception as e:
            logger.error(f"Error discovering models: {e}")
            return {"success": False, "error": str(e)}

    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a specific model"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/models/{model_name}/load") as response:
                result = await response.json()
                # Refresh models cache
                await self.list_models(refresh_cache=True)
                return result
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a specific model"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/models/{model_name}/unload") as response:
                result = await response.json()
                # Refresh models cache
                await self.list_models(refresh_cache=True)
                return result
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return {"success": False, "error": str(e)}

    async def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/models/{model_name}/status") as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error getting model status for {model_name}: {e}")
            return {"error": str(e)}

    async def list_tools(self, refresh_cache: bool = False) -> List[ShimmyTool]:
        """List available tools"""
        if not refresh_cache and self._tools_cache:
            return self._tools_cache

        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/tools") as response:
                if response.status == 200:
                    tools_data = await response.json()
                    self._tools_cache = [
                        ShimmyTool(
                            name=tool.get("name", ""),
                            description=tool.get("description", ""),
                            parameters=tool.get("parameters", {}),
                            enabled=tool.get("enabled", True)
                        )
                        for tool in tools_data.get("tools", [])
                    ]
                    return self._tools_cache
                else:
                    logger.error(f"Failed to list tools: HTTP {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool with parameters"""
        try:
            session = await self._get_session()
            payload = {"parameters": parameters}
            async with session.post(f"{self.base_url}/api/tools/{tool_name}/execute", json=payload) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}")
            return {"success": False, "error": str(e)}

    async def execute_workflow(self, workflow_definition: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a workflow"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/workflows/execute", json=workflow_definition) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return {"success": False, "error": str(e)}

    async def generate_text(self, prompt: str, model: str = None, **kwargs) -> Dict[str, Any]:
        """Generate text using Shimmy's native API"""
        try:
            if model is None:
                model = self.default_model

            session = await self._get_session()
            payload = {
                "model": model,
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "stream": kwargs.get("stream", False),
                **kwargs
            }
            async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response": result.get("response", ""),
                        "model": model,
                        "prompt_tokens": result.get("prompt_eval_count", 0),
                        "completion_tokens": result.get("eval_count", 0)
                    }
                else:
                    error_text = await response.text()
                    logger.error(f"Shimmy API error: {response.status} - {error_text}")
                    return {"success": False, "error": f"HTTP {response.status}: {error_text}"}
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return {"success": False, "error": str(e)}

    async def stream_generation(self, prompt: str, model: Optional[str] = None, **kwargs) -> AsyncGenerator[str, None]:
        """Stream text generation via WebSocket"""
        try:
            if model is None:
                model = self.default_model

            ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
            async with websockets.connect(f"{ws_url}/ws/generate") as websocket:
                payload = {
                    "model": model,
                    "prompt": prompt,
                    "stream": True,
                    **kwargs
                }
                await websocket.send(json.dumps(payload))

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if "content" in data:
                            yield data["content"]
                        elif "error" in data:
                            logger.error(f"WebSocket error: {data['error']}")
                            break
                        elif data.get("done", False):
                            break
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON received: {message}")
                        continue
        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
            yield f"Error: {str(e)}"

    async def generate_a2ui(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate A2UI response"""
        try:
            session = await self._get_session()
            async with session.post(f"{self.base_url}/api/a2ui/generate", json=request_data) as response:
                return await response.json()
        except Exception as e:
            logger.error(f"Error generating A2UI: {e}")
            return {"error": str(e)}


# Alias for backward compatibility
ShimmyAdapter = ShimmyService


async def check_shimmy_health(shimmy_url: str = "http://shimmy:3001") -> Dict[str, Any]:
    """
    Check Shimmy service health
    
    Args:
        shimmy_url: Base URL for Shimmy service
        
    Returns:
        Health check result
    """
    service = ShimmyService(base_url=shimmy_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
