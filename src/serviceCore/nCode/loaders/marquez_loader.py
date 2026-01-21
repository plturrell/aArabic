"""
Marquez Loader for SCIP Index

Tracks SCIP indexing as data lineage in Marquez using OpenLineage.

Lineage Model:
- Namespace: Project/repository
- Datasets: Source files (inputs) and SCIP index (output)
- Jobs: Indexing runs
- Runs: Individual indexing executions

Usage:
    loader = MarquezLoader(url="http://localhost:5000")
    await loader.track_indexing_run("index.scip", "my-project")
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from .scip_parser import load_scip_file, ScipIndex

logger = logging.getLogger(__name__)


@dataclass
class OpenLineageDataset:
    """OpenLineage dataset representation"""
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenLineageJob:
    """OpenLineage job representation"""
    namespace: str
    name: str
    facets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpenLineageRun:
    """OpenLineage run representation"""
    run_id: str
    facets: Dict[str, Any] = field(default_factory=dict)


class MarquezLoader:
    """Tracks SCIP indexing as data lineage in Marquez"""
    
    def __init__(
        self,
        url: str = "http://localhost:5000",
        timeout: int = 30
    ):
        if not AIOHTTP_AVAILABLE:
            raise ImportError("aiohttp not installed. Run: pip install aiohttp")
        
        self.base_url = url
        self.api_url = f"{url}/api/v1"
        self.lineage_url = f"{url}/api/v1/lineage"
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self.timeout)
        return self._session
    
    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def create_namespace(self, namespace: str, owner: str = "ncode") -> Dict:
        """Create or update a namespace"""
        session = await self._get_session()
        payload = {"ownerName": owner}
        
        async with session.put(
            f"{self.api_url}/namespaces/{namespace}",
            json=payload
        ) as response:
            return await response.json()
    
    async def emit_lineage_event(self, event: Dict[str, Any]) -> bool:
        """Emit an OpenLineage event"""
        session = await self._get_session()
        
        try:
            async with session.post(
                self.lineage_url,
                json=event,
                headers={"Content-Type": "application/json"}
            ) as response:
                if response.status in (200, 201, 204):
                    logger.info(f"Emitted lineage event: {event.get('eventType')}")
                    return True
                else:
                    text = await response.text()
                    logger.error(f"Failed to emit event: {response.status} - {text}")
                    return False
        except Exception as e:
            logger.error(f"Error emitting lineage event: {e}")
            return False
    
    def _create_run_event(
        self,
        event_type: str,
        run_id: str,
        job_namespace: str,
        job_name: str,
        inputs: List[OpenLineageDataset],
        outputs: List[OpenLineageDataset],
        job_facets: Dict[str, Any] = None,
        run_facets: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Create an OpenLineage run event"""
        return {
            "eventType": event_type,
            "eventTime": datetime.now(timezone.utc).isoformat(),
            "producer": "ncode-scip-indexer",
            "schemaURL": "https://openlineage.io/spec/1-0-5/OpenLineage.json#/definitions/RunEvent",
            "run": {
                "runId": run_id,
                "facets": run_facets or {}
            },
            "job": {
                "namespace": job_namespace,
                "name": job_name,
                "facets": job_facets or {}
            },
            "inputs": [
                {
                    "namespace": ds.namespace,
                    "name": ds.name,
                    "facets": ds.facets
                }
                for ds in inputs
            ],
            "outputs": [
                {
                    "namespace": ds.namespace,
                    "name": ds.name,
                    "facets": ds.facets
                }
                for ds in outputs
            ]
        }

    async def track_indexing_run(
        self,
        scip_path: str,
        project_name: str,
        namespace: Optional[str] = None
    ) -> Dict[str, Any]:
        """Track a SCIP indexing run as data lineage"""
        # Parse SCIP file
        logger.info(f"Loading SCIP index from {scip_path}")
        index = load_scip_file(scip_path)

        # Determine namespace
        ns = namespace or index.metadata.project_root or project_name

        # Create namespace
        await self.create_namespace(ns)

        # Generate run ID
        run_id = str(uuid.uuid4())
        job_name = f"scip-index-{project_name}"

        # Create input datasets (source files)
        inputs = []
        for doc in index.documents:
            inputs.append(OpenLineageDataset(
                namespace=ns,
                name=doc.relative_path,
                facets={
                    "schema": {
                        "_producer": "ncode",
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                        "fields": [
                            {"name": "language", "type": doc.language},
                            {"name": "symbols", "type": f"count:{len(doc.symbols)}"},
                            {"name": "occurrences", "type": f"count:{len(doc.occurrences)}"}
                        ]
                    }
                }
            ))

        # Create output dataset (SCIP index)
        outputs = [
            OpenLineageDataset(
                namespace=ns,
                name=Path(scip_path).name,
                facets={
                    "schema": {
                        "_producer": "ncode",
                        "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SchemaDatasetFacet.json",
                        "fields": [
                            {"name": "documents", "type": f"count:{len(index.documents)}"},
                            {"name": "external_symbols", "type": f"count:{len(index.external_symbols)}"},
                            {"name": "version", "type": f"scip-v{index.metadata.version}"}
                        ]
                    }
                }
            )
        ]

        # Job facets
        job_facets = {
            "sourceCodeLocation": {
                "_producer": "ncode",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/SourceCodeLocationJobFacet.json",
                "type": "git",
                "url": index.metadata.project_root
            },
            "documentation": {
                "_producer": "ncode",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/DocumentationJobFacet.json",
                "description": f"SCIP indexing job for {project_name} using {index.metadata.tool_info.name}"
            }
        }

        # Run facets
        run_facets = {
            "nominalTime": {
                "_producer": "ncode",
                "_schemaURL": "https://openlineage.io/spec/facets/1-0-0/NominalTimeRunFacet.json",
                "nominalStartTime": datetime.now(timezone.utc).isoformat()
            }
        }

        # Emit START event
        start_event = self._create_run_event(
            "START", run_id, ns, job_name,
            inputs, outputs, job_facets, run_facets
        )
        await self.emit_lineage_event(start_event)

        # Emit COMPLETE event
        complete_event = self._create_run_event(
            "COMPLETE", run_id, ns, job_name,
            inputs, outputs, job_facets, run_facets
        )
        await self.emit_lineage_event(complete_event)

        return {
            "run_id": run_id,
            "namespace": ns,
            "job_name": job_name,
            "inputs": len(inputs),
            "outputs": len(outputs),
            "indexer": index.metadata.tool_info.name,
            "indexer_version": index.metadata.tool_info.version
        }

    async def get_lineage(self, namespace: str, dataset: str) -> Dict[str, Any]:
        """Get lineage for a dataset"""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/namespaces/{namespace}/datasets/{dataset}/lineage"
        ) as response:
            return await response.json()

    async def get_job_runs(self, namespace: str, job_name: str) -> List[Dict]:
        """Get all runs for a job"""
        session = await self._get_session()

        async with session.get(
            f"{self.api_url}/namespaces/{namespace}/jobs/{job_name}/runs"
        ) as response:
            data = await response.json()
            return data.get("runs", [])

