"""
APISIX API Gateway Adapter
Integration layer for Apache APISIX API Gateway.
Provides route management, upstream configuration, and plugin control.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp


class UpstreamType(Enum):
    """Load balancing types for upstreams."""
    ROUNDROBIN = "roundrobin"
    CHASH = "chash"
    EWMA = "ewma"
    LEAST_CONN = "least_conn"


@dataclass
class UpstreamNode:
    """Represents an upstream node."""
    host: str
    port: int
    weight: int = 1


@dataclass
class Route:
    """Represents an APISIX route."""
    id: Optional[str] = None
    uri: str = ""
    methods: List[str] = field(default_factory=lambda: ["GET", "POST"])
    upstream_id: Optional[str] = None
    upstream_nodes: Dict[str, int] = field(default_factory=dict)
    plugins: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class APISIXHealthStatus:
    """Health status of the APISIX service."""
    healthy: bool
    status: str
    version: Optional[str] = None
    error: Optional[str] = None


class APISIXAdapter:
    """
    Adapter for interacting with APISIX API Gateway.

    Provides methods for:
    - Route management (CRUD)
    - Upstream configuration
    - Plugin management
    - Health monitoring
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        admin_key: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url or os.getenv("APISIX_URL", "http://localhost:9080")
        self.admin_url = f"{self.base_url.replace(':9080', ':9180')}/apisix/admin"
        self.admin_key = admin_key or os.getenv("APISIX_ADMIN_KEY", "ai_nucleus_admin_key")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            headers = {"X-API-KEY": self.admin_key}
            self._session = aiohttp.ClientSession(timeout=self.timeout, headers=headers)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> APISIXHealthStatus:
        """Check the health of the APISIX service."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/apisix/status") as response:
                if response.status == 200:
                    return APISIXHealthStatus(healthy=True, status="healthy")
                return APISIXHealthStatus(healthy=False, status="unhealthy")
        except Exception as e:
            return APISIXHealthStatus(healthy=False, status="unavailable", error=str(e))

    async def list_routes(self) -> List[Route]:
        """List all configured routes."""
        session = await self._get_session()
        async with session.get(f"{self.admin_url}/routes") as response:
            data = await response.json()
            routes = []
            for item in data.get("list", []):
                value = item.get("value", {})
                routes.append(Route(
                    id=item.get("key", "").split("/")[-1],
                    uri=value.get("uri", ""),
                    methods=value.get("methods", []),
                    plugins=value.get("plugins", {}),
                    enabled=value.get("status", 1) == 1
                ))
            return routes

    async def create_route(self, route: Route) -> Dict[str, Any]:
        """Create a new route."""
        session = await self._get_session()
        payload = {
            "uri": route.uri,
            "methods": route.methods,
            "plugins": route.plugins,
            "status": 1 if route.enabled else 0
        }
        if route.upstream_nodes:
            payload["upstream"] = {
                "type": "roundrobin",
                "nodes": route.upstream_nodes
            }

        url = f"{self.admin_url}/routes/{route.id}" if route.id else f"{self.admin_url}/routes"
        method = "PUT" if route.id else "POST"

        async with session.request(method, url, json=payload) as response:
            return await response.json()

    async def delete_route(self, route_id: str) -> bool:
        """Delete a route by ID."""
        session = await self._get_session()
        async with session.delete(f"{self.admin_url}/routes/{route_id}") as response:
            return response.status == 200

    async def get_upstreams(self) -> List[Dict[str, Any]]:
        """List all upstreams."""
        session = await self._get_session()
        async with session.get(f"{self.admin_url}/upstreams") as response:
            data = await response.json()
            return data.get("list", [])

    async def reload_plugins(self) -> bool:
        """Reload APISIX plugins."""
        session = await self._get_session()
        async with session.put(f"{self.admin_url}/plugins/reload") as response:
            return response.status == 200


async def check_apisix_health(base_url: Optional[str] = None) -> APISIXHealthStatus:
    """Quick health check for APISIX service."""
    adapter = APISIXAdapter(base_url=base_url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()


# Alias for backward compatibility
# APISIXAdapter - no main class exists, adapter provides utility functions


async def check_apisix_health(apisix_url: str = "http://apisix:9080") -> Dict[str, Any]:
    """
    Check APISIX service health
    
    Args:
        apisix_url: Base URL for APISIX service
        
    Returns:
        Health check result
    """
    service = APISIXService(base_url=apisix_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
