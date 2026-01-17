"""
Keycloak Adapter
Integration layer for Keycloak Identity and Access Management.
Provides authentication, authorization, and user management.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp


@dataclass
class KeycloakUser:
    """Represents a Keycloak user."""
    id: str
    username: str
    email: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    enabled: bool = True
    realm_roles: List[str] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenResponse:
    """OAuth2 token response."""
    access_token: str
    refresh_token: Optional[str] = None
    token_type: str = "Bearer"
    expires_in: int = 0
    scope: Optional[str] = None


@dataclass
class KeycloakHealthStatus:
    """Health status of the Keycloak service."""
    healthy: bool
    status: str
    version: Optional[str] = None
    error: Optional[str] = None


class KeycloakAdapter:
    """
    Adapter for interacting with Keycloak IAM.

    Provides methods for:
    - Authentication (OAuth2/OIDC)
    - User management
    - Role management
    - Token validation
    - Health monitoring
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        realm: str = "master",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        admin_username: Optional[str] = None,
        admin_password: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url or os.getenv("KEYCLOAK_URL", "http://localhost:8180")
        self.realm = realm or os.getenv("KEYCLOAK_REALM", "master")
        self.client_id = client_id or os.getenv("KEYCLOAK_CLIENT_ID", "admin-cli")
        self.client_secret = client_secret or os.getenv("KEYCLOAK_CLIENT_SECRET")
        self.admin_username = admin_username or os.getenv("KEYCLOAK_ADMIN", "admin")
        self.admin_password = admin_password or os.getenv("KEYCLOAK_ADMIN_PASSWORD", "admin")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
        self._admin_token: Optional[str] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def _get_admin_token(self) -> str:
        """Get admin access token."""
        if self._admin_token:
            return self._admin_token

        token_response = await self.get_token(
            username=self.admin_username,
            password=self.admin_password,
            client_id="admin-cli"
        )
        self._admin_token = token_response.access_token
        return self._admin_token

    async def health_check(self) -> KeycloakHealthStatus:
        """Check the health of Keycloak."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.base_url}/health/ready") as response:
                if response.status == 200:
                    return KeycloakHealthStatus(healthy=True, status="healthy")
                return KeycloakHealthStatus(healthy=False, status="unhealthy")
        except Exception as e:
            return KeycloakHealthStatus(healthy=False, status="unavailable", error=str(e))

    async def get_token(
        self,
        username: str,
        password: str,
        client_id: Optional[str] = None,
        scope: str = "openid"
    ) -> TokenResponse:
        """Get access token using password grant."""
        session = await self._get_session()
        url = f"{self.base_url}/realms/{self.realm}/protocol/openid-connect/token"

        data = {
            "grant_type": "password",
            "client_id": client_id or self.client_id,
            "username": username,
            "password": password,
            "scope": scope
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with session.post(url, data=data) as response:
            result = await response.json()
            return TokenResponse(
                access_token=result["access_token"],
                refresh_token=result.get("refresh_token"),
                token_type=result.get("token_type", "Bearer"),
                expires_in=result.get("expires_in", 0),
                scope=result.get("scope")
            )

    async def refresh_token(self, refresh_token: str) -> TokenResponse:
        """Refresh an access token."""
        session = await self._get_session()
        url = f"{self.base_url}/realms/{self.realm}/protocol/openid-connect/token"

        data = {
            "grant_type": "refresh_token",
            "client_id": self.client_id,
            "refresh_token": refresh_token
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with session.post(url, data=data) as response:
            result = await response.json()
            return TokenResponse(
                access_token=result["access_token"],
                refresh_token=result.get("refresh_token"),
                expires_in=result.get("expires_in", 0)
            )

    async def validate_token(self, token: str) -> Dict[str, Any]:
        """Validate and introspect a token."""
        session = await self._get_session()
        url = f"{self.base_url}/realms/{self.realm}/protocol/openid-connect/token/introspect"

        data = {
            "token": token,
            "client_id": self.client_id
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with session.post(url, data=data) as response:
            return await response.json()

    async def get_user_info(self, token: str) -> Dict[str, Any]:
        """Get user info from token."""
        session = await self._get_session()
        url = f"{self.base_url}/realms/{self.realm}/protocol/openid-connect/userinfo"
        headers = {"Authorization": f"Bearer {token}"}

        async with session.get(url, headers=headers) as response:
            return await response.json()

    async def list_users(self, search: Optional[str] = None) -> List[KeycloakUser]:
        """List users in the realm."""
        admin_token = await self._get_admin_token()
        session = await self._get_session()
        url = f"{self.base_url}/admin/realms/{self.realm}/users"
        headers = {"Authorization": f"Bearer {admin_token}"}
        params = {}
        if search:
            params["search"] = search

        async with session.get(url, headers=headers, params=params) as response:
            users = await response.json()
            return [
                KeycloakUser(
                    id=u["id"],
                    username=u["username"],
                    email=u.get("email"),
                    first_name=u.get("firstName"),
                    last_name=u.get("lastName"),
                    enabled=u.get("enabled", True)
                )
                for u in users
            ]

    async def create_user(self, user: KeycloakUser, password: str) -> str:
        """Create a new user."""
        admin_token = await self._get_admin_token()
        session = await self._get_session()
        url = f"{self.base_url}/admin/realms/{self.realm}/users"
        headers = {"Authorization": f"Bearer {admin_token}"}

        payload = {
            "username": user.username,
            "email": user.email,
            "firstName": user.first_name,
            "lastName": user.last_name,
            "enabled": user.enabled,
            "credentials": [{
                "type": "password",
                "value": password,
                "temporary": False
            }]
        }

        async with session.post(url, headers=headers, json=payload) as response:
            location = response.headers.get("Location", "")
            return location.split("/")[-1]

    async def logout(self, refresh_token: str) -> bool:
        """Logout user by invalidating refresh token."""
        session = await self._get_session()
        url = f"{self.base_url}/realms/{self.realm}/protocol/openid-connect/logout"

        data = {
            "client_id": self.client_id,
            "refresh_token": refresh_token
        }

        if self.client_secret:
            data["client_secret"] = self.client_secret

        async with session.post(url, data=data) as response:
            return response.status == 204


async def check_keycloak_health(base_url: Optional[str] = None) -> KeycloakHealthStatus:
    """Quick health check for Keycloak."""
    adapter = KeycloakAdapter(base_url=base_url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()


# Alias for backward compatibility
# KeycloakAdapter - no main class exists, adapter provides utility functions


async def check_keycloak_health(keycloak_url: str = "http://keycloak:8080") -> Dict[str, Any]:
    """
    Check Keycloak service health
    
    Args:
        keycloak_url: Base URL for Keycloak service
        
    Returns:
        Health check result
    """
    service = KeycloakService(base_url=keycloak_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
