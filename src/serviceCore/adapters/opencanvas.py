"""
Open Canvas Adapter
Integration layer for the Open Canvas (LangChain) visual artifact editor.
Provides artifact management, content generation, and real-time collaboration.
"""

import asyncio
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp


class ArtifactType(Enum):
    """Types of artifacts supported by Open Canvas."""
    TEXT = "text"
    CODE = "code"
    MARKDOWN = "markdown"
    HTML = "html"
    REACT = "react"


class ArtifactStatus(Enum):
    """Status of an artifact."""
    DRAFT = "draft"
    PUBLISHED = "published"
    SHARED = "shared"
    ARCHIVED = "archived"


@dataclass
class Artifact:
    """Represents an Open Canvas artifact."""
    id: str
    type: ArtifactType
    title: str
    content: str
    language: str = "en"
    status: ArtifactStatus = ArtifactStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    version: int = 1


@dataclass
class GenerationRequest:
    """Request for content generation."""
    prompt: str
    artifact_type: ArtifactType = ArtifactType.TEXT
    context: List[str] = field(default_factory=list)
    language: str = "en"
    style: str = "default"
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RewriteRequest:
    """Request for content rewriting."""
    artifact_id: str
    content: str
    instruction: str
    artifact_type: ArtifactType = ArtifactType.TEXT


@dataclass
class OpenCanvasHealthStatus:
    """Health status of the Open Canvas service."""
    healthy: bool
    status: str
    version: Optional[str] = None
    uptime: Optional[float] = None
    error: Optional[str] = None


class OpenCanvasAdapter:
    """
    Adapter for interacting with the Open Canvas service.

    Provides methods for:
    - Artifact CRUD operations
    - Content generation and rewriting
    - Translation
    - Sharing and collaboration
    - Health monitoring
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize the Open Canvas adapter.

        Args:
            base_url: Base URL of the Open Canvas service
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
        """
        self.base_url = base_url or os.getenv("OPENCANVAS_URL", "http://localhost:3003")
        self.api_url = urljoin(self.base_url, "/api")
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create an aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Make an HTTP request with retry logic.

        Args:
            method: HTTP method
            endpoint: API endpoint
            data: Request body data
            params: Query parameters

        Returns:
            Response data as dictionary
        """
        session = await self._get_session()
        url = f"{self.api_url}{endpoint}"
        last_error = None

        for attempt in range(self.max_retries):
            try:
                async with session.request(
                    method,
                    url,
                    json=data,
                    params=params
                ) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(1 * (attempt + 1))

        raise last_error or Exception("Request failed after retries")

    async def health_check(self) -> OpenCanvasHealthStatus:
        """
        Check the health of the Open Canvas service.

        Returns:
            Health status object
        """
        try:
            session = await self._get_session()
            url = f"{self.api_url}/health"

            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    return OpenCanvasHealthStatus(
                        healthy=True,
                        status="healthy",
                        version=data.get("version"),
                        uptime=data.get("uptime")
                    )
                else:
                    return OpenCanvasHealthStatus(
                        healthy=False,
                        status="unhealthy",
                        error=f"HTTP {response.status}"
                    )
        except Exception as e:
            return OpenCanvasHealthStatus(
                healthy=False,
                status="unavailable",
                error=str(e)
            )

    async def create_artifact(self, artifact: Artifact) -> Dict[str, Any]:
        """
        Create a new artifact.

        Args:
            artifact: Artifact to create

        Returns:
            Created artifact data
        """
        payload = {
            "type": artifact.type.value,
            "title": artifact.title,
            "content": artifact.content,
            "language": artifact.language,
            "metadata": artifact.metadata
        }
        return await self._request("POST", "/store/put", data=payload)

    async def get_artifact(self, artifact_id: str) -> Optional[Artifact]:
        """
        Retrieve an artifact by ID.

        Args:
            artifact_id: Artifact ID

        Returns:
            Artifact object or None if not found
        """
        try:
            data = await self._request(
                "GET",
                "/store/get",
                params={"id": artifact_id}
            )
            return Artifact(
                id=data.get("id", artifact_id),
                type=ArtifactType(data.get("type", "text")),
                title=data.get("title", ""),
                content=data.get("content", ""),
                language=data.get("language", "en"),
                status=ArtifactStatus(data.get("status", "draft")),
                metadata=data.get("metadata", {}),
                version=data.get("version", 1)
            )
        except Exception:
            return None

    async def delete_artifact(self, artifact_id: str) -> bool:
        """
        Delete an artifact.

        Args:
            artifact_id: Artifact ID

        Returns:
            True if deleted successfully
        """
        try:
            await self._request(
                "DELETE",
                "/store/delete/id",
                params={"id": artifact_id}
            )
            return True
        except Exception:
            return False

    async def generate_content(
        self,
        request: GenerationRequest
    ) -> Dict[str, Any]:
        """
        Generate new content using AI.

        Args:
            request: Generation request parameters

        Returns:
            Generated content data
        """
        payload = {
            "prompt": request.prompt,
            "artifactType": request.artifact_type.value,
            "context": request.context,
            "options": {
                "language": request.language,
                "style": request.style,
                **request.options
            }
        }
        return await self._request("POST", "/runs/generate", data=payload)

    async def rewrite_content(self, request: RewriteRequest) -> Dict[str, Any]:
        """
        Rewrite or transform existing content.

        Args:
            request: Rewrite request parameters

        Returns:
            Rewritten content data
        """
        payload = {
            "artifactId": request.artifact_id,
            "content": request.content,
            "instruction": request.instruction,
            "type": request.artifact_type.value
        }
        return await self._request("POST", "/runs/rewrite", data=payload)

    async def translate_content(
        self,
        content: str,
        target_language: str,
        source_language: str = "auto"
    ) -> Dict[str, Any]:
        """
        Translate content to another language.

        Args:
            content: Content to translate
            target_language: Target language code
            source_language: Source language code (auto-detect if 'auto')

        Returns:
            Translation result
        """
        payload = {
            "content": content,
            "targetLanguage": target_language,
            "sourceLanguage": source_language
        }
        return await self._request("POST", "/runs/translate", data=payload)

    async def share_artifact(self, artifact_id: str) -> Dict[str, Any]:
        """
        Share an artifact and get a shareable link.

        Args:
            artifact_id: Artifact ID to share

        Returns:
            Share details including URL
        """
        return await self._request(
            "POST",
            "/runs/share",
            data={"artifactId": artifact_id}
        )

    async def submit_feedback(
        self,
        run_id: str,
        score: int,
        comment: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Submit feedback for a generation run.

        Args:
            run_id: Run ID to provide feedback for
            score: Feedback score (1-5)
            comment: Optional feedback comment

        Returns:
            Feedback submission result
        """
        payload = {
            "runId": run_id,
            "score": score,
            "comment": comment
        }
        return await self._request("POST", "/runs/feedback", data=payload)

    async def list_artifacts(
        self,
        limit: int = 50,
        offset: int = 0,
        artifact_type: Optional[ArtifactType] = None
    ) -> List[Artifact]:
        """
        List artifacts with pagination.

        Args:
            limit: Maximum number of artifacts to return
            offset: Pagination offset
            artifact_type: Filter by artifact type

        Returns:
            List of artifacts
        """
        params = {"limit": str(limit), "offset": str(offset)}
        if artifact_type:
            params["type"] = artifact_type.value

        data = await self._request("GET", "/store/list", params=params)

        return [
            Artifact(
                id=item.get("id"),
                type=ArtifactType(item.get("type", "text")),
                title=item.get("title", ""),
                content=item.get("content", ""),
                language=item.get("language", "en"),
                status=ArtifactStatus(item.get("status", "draft")),
                metadata=item.get("metadata", {}),
                version=item.get("version", 1)
            )
            for item in data.get("items", [])
        ]


# Convenience function for health checks
async def check_opencanvas_health(
    base_url: Optional[str] = None
) -> OpenCanvasHealthStatus:
    """
    Quick health check for Open Canvas service.

    Args:
        base_url: Optional custom base URL

    Returns:
        Health status
    """
    adapter = OpenCanvasAdapter(base_url=base_url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()
