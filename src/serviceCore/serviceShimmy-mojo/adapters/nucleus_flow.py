"""
NucleusFlow (Langflow) Service Adapter
Integration with NucleusFlow for visual agent workflow building
"""

import aiohttp
import logging
from typing import Dict, List, Any, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class NucleusFlowStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    ERROR = "error"

@dataclass
class Flow:
    id: str
    name: str
    description: str
    data: Dict[str, Any]

class NucleusFlowService:
    """
    Adapter for communicating with NucleusFlow (Langflow)
    """

    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip('/')
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check if NucleusFlow is running"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    return {"status": NucleusFlowStatus.HEALTHY, "details": await response.json()}
                return {"status": NucleusFlowStatus.UNHEALTHY, "code": response.status}
        except Exception as e:
            return {"status": NucleusFlowStatus.ERROR, "error": str(e)}

    async def list_flows(self) -> List[Flow]:
        """List all available flows"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/v1/flows/") as response:
                if response.status == 200:
                    data = await response.json()
                    return [
                        Flow(
                            id=f["id"],
                            name=f["name"],
                            description=f.get("description", ""),
                            data=f.get("data", {})
                        ) for f in data
                    ]
                logger.error(f"Failed to list flows: {response.status}")
                return []
        except Exception as e:
            logger.error(f"Error listing flows: {e}")
            return []

    async def run_flow(self, flow_id: str, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a specific flow with inputs"""
        try:
            session = await self._get_session()
            payload = {"inputs": inputs}
            # Note: Endpoint might vary based on Langflow version
            async with session.post(f"{self.base_url}/api/v1/process/{flow_id}", json=payload) as response:
                if response.status == 200:
                    return await response.json()
                error_text = await response.text()
                logger.error(f"Failed to run flow {flow_id}: {response.status} - {error_text}")
                return {"error": error_text}
        except Exception as e:
            logger.error(f"Error running flow {flow_id}: {e}")
            return {"error": str(e)}


# Alias for backward compatibility
NucleusFlowAdapter = NucleusFlowService


async def check_nucleusflow_health(nucleusflow_url: str = "http://nucleus-flow:8000") -> Dict[str, Any]:
    """
    Check NucleusFlow service health
    
    Args:
        nucleusflow_url: Base URL for NucleusFlow service
        
    Returns:
        Health check result
    """
    service = NucleusFlowService(base_url=nucleusflow_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()

# Alias for backward compatibility
check_nucleus_flow_health = check_nucleusflow_health
