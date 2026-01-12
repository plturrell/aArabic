"""
HyperbookLM Service Adapter
Integration with HyperbookLM Research Assistant
"""

import aiohttp
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class HyperbookLMService:
    """
    Adapter for HyperbookLM
    Since HyperbookLM is primarily a frontend app, this adapter checks its availability
    and potentially interacts with its API routes if exposed.
    """

    def __init__(self, base_url: str = "http://localhost:3002"):
        # Assuming HyperbookLM runs on port 3002 to avoid conflicts
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
        """Check if HyperbookLM is accessible"""
        try:
            session = await self._get_session()
            # Next.js apps usually respond on root, or we can check a specific health endpoint if added
            async with session.get(f"{self.base_url}") as response:
                if response.status == 200:
                    return {"status": "healthy"}
                return {"status": "unhealthy", "code": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}


# Alias for backward compatibility
HyperbookLMAdapter = HyperbookLMService


async def check_hyperbooklm_health(hyperbooklm_url: str = "http://hyperbooklm:3002") -> Dict[str, Any]:
    """
    Check HyperbookLM service health
    
    Args:
        hyperbooklm_url: Base URL for HyperbookLM service
        
    Returns:
        Health check result
    """
    service = HyperbookLMService(base_url=hyperbooklm_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
