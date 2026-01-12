"""
Gitea Service Adapter
Integration with Gitea for repository and issue management
"""

import aiohttp
import logging
from typing import Dict, List, Any, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class GiteaService:
    """
    Adapter for communicating with Gitea API
    """

    def __init__(self, base_url: str = "http://localhost:3000", token: str = ""):
        self.base_url = base_url.rstrip('/')
        self.token = token
        self.session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self.session is None or self.session.closed:
            headers = {"Authorization": f"token {self.token}"} if self.token else {}
            self.session = aiohttp.ClientSession(headers=headers)
        return self.session

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def health_check(self) -> Dict[str, Any]:
        """Check if Gitea is running"""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/api/v1/version") as response:
                if response.status == 200:
                    return {"status": "healthy", "version": (await response.json()).get("version")}
                return {"status": "unhealthy", "code": response.status}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    async def create_repo(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new repository"""
        try:
            session = await self._get_session()
            payload = {"name": name, "description": description, "private": True}
            async with session.post(f"{self.base_url}/api/v1/user/repos", json=payload) as response:
                if response.status == 201:
                    return await response.json()
                return {"error": f"Failed to create repo: {response.status}"}
        except Exception as e:
            logger.error(f"Error creating repo: {e}")
            return {"error": str(e)}


# Alias for backward compatibility
GiteaAdapter = GiteaService


async def check_gitea_health(gitea_url: str = "http://gitea:3000") -> Dict[str, Any]:
    """
    Check Gitea service health
    
    Args:
        gitea_url: Base URL for Gitea service
        
    Returns:
        Health check result
    """
    service = GiteaService(base_url=gitea_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
