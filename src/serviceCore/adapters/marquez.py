"""
Marquez Adapter
Integration layer for Marquez data lineage and metadata service.
Provides data lineage tracking, job/dataset management, and OpenLineage integration.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import aiohttp


class RunState(Enum):
    """State of a job run."""
    NEW = "NEW"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


@dataclass
class Namespace:
    """Represents a Marquez namespace."""
    name: str
    owner_name: str
    description: Optional[str] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None


@dataclass
class Dataset:
    """Represents a Marquez dataset."""
    name: str
    namespace: str
    source_name: str
    physical_name: Optional[str] = None
    description: Optional[str] = None
    fields: List[Dict[str, Any]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_at: Optional[datetime] = None


@dataclass
class Job:
    """Represents a Marquez job."""
    name: str
    namespace: str
    type: str = "BATCH"
    description: Optional[str] = None
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    location: Optional[str] = None
    created_at: Optional[datetime] = None


@dataclass
class Run:
    """Represents a job run."""
    id: str
    job_name: str
    namespace: str
    state: RunState = RunState.NEW
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None


@dataclass
class MarquezHealthStatus:
    """Health status of the Marquez service."""
    healthy: bool
    status: str
    version: Optional[str] = None
    error: Optional[str] = None


class MarquezAdapter:
    """
    Adapter for interacting with Marquez data lineage service.

    Provides methods for:
    - Namespace management
    - Dataset tracking
    - Job management
    - Run tracking
    - Lineage queries
    - Health monitoring
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = 30
    ):
        self.base_url = base_url or os.getenv("MARQUEZ_URL", "http://localhost:5001")
        self.api_url = f"{self.base_url}/api/v1"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    async def health_check(self) -> MarquezHealthStatus:
        """Check the health of Marquez."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.api_url}/namespaces") as response:
                if response.status == 200:
                    return MarquezHealthStatus(healthy=True, status="healthy")
                return MarquezHealthStatus(healthy=False, status="unhealthy")
        except Exception as e:
            return MarquezHealthStatus(healthy=False, status="unavailable", error=str(e))

    # Namespace Operations
    async def list_namespaces(self) -> List[Namespace]:
        """List all namespaces."""
        session = await self._get_session()
        async with session.get(f"{self.api_url}/namespaces") as response:
            data = await response.json()
            return [
                Namespace(
                    name=ns["name"],
                    owner_name=ns.get("ownerName", ""),
                    description=ns.get("description"),
                    created_at=ns.get("createdAt"),
                    updated_at=ns.get("updatedAt")
                )
                for ns in data.get("namespaces", [])
            ]

    async def create_namespace(self, namespace: Namespace) -> Namespace:
        """Create a new namespace."""
        session = await self._get_session()
        payload = {
            "ownerName": namespace.owner_name,
            "description": namespace.description
        }
        async with session.put(
            f"{self.api_url}/namespaces/{namespace.name}",
            json=payload
        ) as response:
            data = await response.json()
            return Namespace(
                name=data["name"],
                owner_name=data["ownerName"],
                description=data.get("description")
            )

    # Dataset Operations
    async def list_datasets(self, namespace: str) -> List[Dataset]:
        """List datasets in a namespace."""
        session = await self._get_session()
        async with session.get(
            f"{self.api_url}/namespaces/{namespace}/datasets"
        ) as response:
            data = await response.json()
            return [
                Dataset(
                    name=ds["name"],
                    namespace=ds["namespace"],
                    source_name=ds.get("sourceName", ""),
                    physical_name=ds.get("physicalName"),
                    description=ds.get("description"),
                    fields=ds.get("fields", []),
                    tags=ds.get("tags", [])
                )
                for ds in data.get("datasets", [])
            ]

    async def create_dataset(self, dataset: Dataset) -> Dataset:
        """Create or update a dataset."""
        session = await self._get_session()
        payload = {
            "type": "DB_TABLE",
            "physicalName": dataset.physical_name or dataset.name,
            "sourceName": dataset.source_name,
            "description": dataset.description,
            "fields": dataset.fields,
            "tags": dataset.tags
        }
        async with session.put(
            f"{self.api_url}/namespaces/{dataset.namespace}/datasets/{dataset.name}",
            json=payload
        ) as response:
            data = await response.json()
            return Dataset(
                name=data["name"],
                namespace=data["namespace"],
                source_name=data.get("sourceName", ""),
                description=data.get("description")
            )

    # Job Operations
    async def list_jobs(self, namespace: str) -> List[Job]:
        """List jobs in a namespace."""
        session = await self._get_session()
        async with session.get(
            f"{self.api_url}/namespaces/{namespace}/jobs"
        ) as response:
            data = await response.json()
            return [
                Job(
                    name=job["name"],
                    namespace=job["namespace"],
                    type=job.get("type", "BATCH"),
                    description=job.get("description"),
                    location=job.get("location")
                )
                for job in data.get("jobs", [])
            ]

    async def create_job(self, job: Job) -> Job:
        """Create or update a job."""
        session = await self._get_session()
        payload = {
            "type": job.type,
            "inputs": [{"namespace": job.namespace, "name": i} for i in job.inputs],
            "outputs": [{"namespace": job.namespace, "name": o} for o in job.outputs],
            "description": job.description,
            "location": job.location
        }
        async with session.put(
            f"{self.api_url}/namespaces/{job.namespace}/jobs/{job.name}",
            json=payload
        ) as response:
            data = await response.json()
            return Job(
                name=data["name"],
                namespace=data["namespace"],
                type=data.get("type", "BATCH"),
                description=data.get("description")
            )

    # Run Operations
    async def start_run(self, namespace: str, job_name: str) -> Run:
        """Start a new job run."""
        session = await self._get_session()
        async with session.post(
            f"{self.api_url}/namespaces/{namespace}/jobs/{job_name}/runs"
        ) as response:
            data = await response.json()
            return Run(
                id=data["id"],
                job_name=job_name,
                namespace=namespace,
                state=RunState.RUNNING
            )

    async def complete_run(self, run_id: str) -> Run:
        """Mark a run as completed."""
        session = await self._get_session()
        async with session.post(
            f"{self.api_url}/jobs/runs/{run_id}/complete"
        ) as response:
            data = await response.json()
            return Run(
                id=data["id"],
                job_name=data.get("jobName", ""),
                namespace=data.get("namespace", ""),
                state=RunState.COMPLETED
            )

    async def fail_run(self, run_id: str) -> Run:
        """Mark a run as failed."""
        session = await self._get_session()
        async with session.post(
            f"{self.api_url}/jobs/runs/{run_id}/fail"
        ) as response:
            data = await response.json()
            return Run(
                id=data["id"],
                job_name=data.get("jobName", ""),
                namespace=data.get("namespace", ""),
                state=RunState.FAILED
            )

    # Lineage Queries
    async def get_lineage(
        self,
        namespace: str,
        dataset_or_job: str,
        node_type: str = "DATASET",
        depth: int = 5
    ) -> Dict[str, Any]:
        """Get lineage graph for a dataset or job."""
        session = await self._get_session()
        params = {
            "nodeId": f"{node_type}:{namespace}:{dataset_or_job}",
            "depth": depth
        }
        async with session.get(
            f"{self.api_url}/lineage",
            params=params
        ) as response:
            return await response.json()


async def check_marquez_health(base_url: Optional[str] = None) -> MarquezHealthStatus:
    """Quick health check for Marquez."""
    adapter = MarquezAdapter(base_url=base_url)
    try:
        return await adapter.health_check()
    finally:
        await adapter.close()


# Alias for backward compatibility
# MarquezAdapter - no main class exists, adapter provides utility functions


async def check_marquez_health(marquez_url: str = "http://marquez:5000") -> Dict[str, Any]:
    """
    Check Marquez service health
    
    Args:
        marquez_url: Base URL for Marquez service
        
    Returns:
        Health check result
    """
    service = MarquezService(base_url=marquez_url)
    try:
        result = await service.health_check()
        return result
    finally:
        await service.close()
