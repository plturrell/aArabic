"""
Proof Domains API Service

FastAPI service providing:
1. REST API for 7 proof domains
2. TOON serialization endpoints
3. Kafka streaming integration
4. Cross-domain proof linking

Port: 8011
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from enum import Enum
import logging
import uuid

from proof_domains import (
    # Core types
    ProofStatus, Evidence, EvidenceType, Confidence, Lineage,
    # Domain proofs
    MathProof, MathProofType, MathDomain,
    CodeProof, VerificationType, ProgrammingLanguage,
    LanguageProof, LogicType, SemanticType,
    DataProof, IntegrityType, TransformationType,
    ProcessProof, ProcessProperty, ProcessState,
    InsightProof, DerivationType, InsightCategory,
    RegulationProof, RegulationType, ComplianceStatus,
    # Serialization
    TOONEncoder, TOONDecoder,
    # Events
    ProofEvent, ProofEventType, create_proof_event,
    # Type unions
    UnifiedProof, DOMAIN_MAP
)

from kafka_integration import (
    KafkaConfig, ProofStreamingService,
    N8nIntegration, N8nWebhookConfig,
    create_streaming_service
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Proof Domains API",
    description="""
    Unified Lean4 Proof Framework API

    **7 Proof Domains:**
    1. Mathematics - Formal mathematical proofs
    2. Code - Program verification
    3. Language - Natural language logic
    4. Data - Data integrity proofs
    5. Processes - Workflow verification
    6. Insights - Analytical derivations
    7. Regulations - Compliance proofs

    **Features:**
    - TOON serialization (40% fewer tokens than JSON)
    - Kafka streaming integration
    - Cross-domain proof linking
    - n8n workflow integration
    """,
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# Global Services
# =============================================================================

toon_encoder = TOONEncoder()
toon_decoder = TOONDecoder()
streaming_service = create_streaming_service(use_mock=True)
n8n_integration = N8nIntegration(streaming_service)

# In-memory proof storage (replace with Qdrant/Memgraph in production)
proof_store: Dict[str, Dict[str, Any]] = {
    "mathematics": {},
    "code": {},
    "language": {},
    "data": {},
    "processes": {},
    "insights": {},
    "regulations": {}
}


# =============================================================================
# Pydantic Models
# =============================================================================

class EvidenceCreate(BaseModel):
    evidence_type: str = "axiom"
    content: str
    sources: List[str] = []


class LineageCreate(BaseModel):
    source_id: str
    derived_from: List[str] = []
    transformation: str = ""


class BaseProofCreate(BaseModel):
    title: str
    description: str
    status: str = "draft"
    evidence: List[EvidenceCreate] = []
    metadata: Dict[str, str] = {}
    lineage: Optional[LineageCreate] = None


# Domain-specific create models
class MathProofCreate(BaseProofCreate):
    proof_type: str = "theorem"
    math_domain: str = "algebra"
    statement: str
    formal_proof: str = ""
    dependencies: List[str] = []


class CodeProofCreate(BaseProofCreate):
    verification_type: str = "functional"
    language: str = "python"
    source_code: str = ""
    specification: str
    invariants: List[str] = []


class LanguageProofCreate(BaseProofCreate):
    logic_type: str = "first_order"
    semantic_type: str = "entailment"
    premises: List[str]
    conclusion: str
    natural_language: str = ""
    formal_representation: str = ""


class DataProofCreate(BaseProofCreate):
    integrity_type: str = "schema"
    schema_before: str
    schema_after: str = ""
    transformation: Optional[str] = None
    constraints: List[str] = []


class ProcessProofCreate(BaseProofCreate):
    property: str = "liveness"
    states: List[str] = []
    transitions: List[Dict[str, str]] = []
    invariants: List[str] = []


class InsightProofCreate(BaseProofCreate):
    derivation_type: str = "statistical"
    category: str = "pattern"
    data_sources: List[str]
    methodology: str
    conclusion: str
    confidence_value: float = 0.95


class RegulationProofCreate(BaseProofCreate):
    regulation_type: str = "policy"
    regulation_id: str
    regulation_text: str
    implementation: str
    compliance_status: str = "pending"
    gaps: List[str] = []
    controls: List[str] = []


# Response models
class ProofResponse(BaseModel):
    id: str
    domain: str
    title: str
    description: str
    status: str
    created_at: str
    updated_at: str
    toon: Optional[str] = None


class ProofListResponse(BaseModel):
    domain: str
    proofs: List[ProofResponse]
    count: int


class TOONResponse(BaseModel):
    toon: str
    token_count: int
    json_token_count: int
    savings_percent: float


class LinkRequest(BaseModel):
    source_id: str
    target_id: str
    relationship: str
    metadata: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    domains: List[str]
    kafka_connected: bool
    timestamp: str


# =============================================================================
# Helper Functions
# =============================================================================

def generate_id() -> str:
    """Generate a unique proof ID."""
    return str(uuid.uuid4())[:8]


def create_evidence(ev: EvidenceCreate) -> Evidence:
    """Convert Pydantic model to Evidence."""
    return Evidence(
        evidence_type=EvidenceType(ev.evidence_type),
        content=ev.content,
        sources=ev.sources
    )


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation)."""
    return len(text.split()) + len(text) // 4


def proof_to_response(proof: UnifiedProof, include_toon: bool = False) -> ProofResponse:
    """Convert proof to response model."""
    response = ProofResponse(
        id=proof.id,
        domain=proof.domain,
        title=proof.title,
        description=proof.description,
        status=proof.status.value,
        created_at=proof.created_at.isoformat(),
        updated_at=proof.updated_at.isoformat()
    )
    if include_toon:
        response.toon = toon_encoder.encode(proof)
    return response


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        service="proof-domains",
        version="1.0.0",
        domains=list(DOMAIN_MAP.keys()),
        kafka_connected=True,  # Mock client always "connected"
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/domains")
async def list_domains():
    """List all proof domains."""
    return {
        "domains": [
            {"name": "mathematics", "description": "Formal mathematical proofs"},
            {"name": "code", "description": "Program verification and correctness"},
            {"name": "language", "description": "Natural language logic and semantics"},
            {"name": "data", "description": "Data integrity and transformations"},
            {"name": "processes", "description": "Workflow and process correctness"},
            {"name": "insights", "description": "Analytical derivations and conclusions"},
            {"name": "regulations", "description": "Compliance and policy verification"}
        ]
    }


# -----------------------------------------------------------------------------
# Mathematics Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/mathematics", response_model=ProofResponse)
async def create_math_proof(proof_data: MathProofCreate, background_tasks: BackgroundTasks):
    """Create a mathematical proof."""
    proof_id = generate_id()

    proof = MathProof(
        id=proof_id,
        domain="mathematics",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        proof_type=MathProofType(proof_data.proof_type),
        math_domain=MathDomain(proof_data.math_domain),
        statement=proof_data.statement,
        formal_proof=proof_data.formal_proof,
        dependencies=proof_data.dependencies
    )

    proof_store["mathematics"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/mathematics", response_model=ProofListResponse)
async def list_math_proofs():
    """List all mathematical proofs."""
    proofs = list(proof_store["mathematics"].values())
    return ProofListResponse(
        domain="mathematics",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Code Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/code", response_model=ProofResponse)
async def create_code_proof(proof_data: CodeProofCreate, background_tasks: BackgroundTasks):
    """Create a code verification proof."""
    proof_id = generate_id()

    proof = CodeProof(
        id=proof_id,
        domain="code",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        verification_type=VerificationType(proof_data.verification_type),
        language=ProgrammingLanguage(proof_data.language),
        source_code=proof_data.source_code,
        specification=proof_data.specification,
        invariants=proof_data.invariants
    )

    proof_store["code"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/code", response_model=ProofListResponse)
async def list_code_proofs():
    """List all code proofs."""
    proofs = list(proof_store["code"].values())
    return ProofListResponse(
        domain="code",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Language Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/language", response_model=ProofResponse)
async def create_language_proof(proof_data: LanguageProofCreate, background_tasks: BackgroundTasks):
    """Create a natural language logic proof."""
    proof_id = generate_id()

    proof = LanguageProof(
        id=proof_id,
        domain="language",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        logic_type=LogicType(proof_data.logic_type),
        semantic_type=SemanticType(proof_data.semantic_type),
        premises=proof_data.premises,
        conclusion=proof_data.conclusion,
        natural_language=proof_data.natural_language,
        formal_representation=proof_data.formal_representation
    )

    proof_store["language"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/language", response_model=ProofListResponse)
async def list_language_proofs():
    """List all language proofs."""
    proofs = list(proof_store["language"].values())
    return ProofListResponse(
        domain="language",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Data Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/data", response_model=ProofResponse)
async def create_data_proof(proof_data: DataProofCreate, background_tasks: BackgroundTasks):
    """Create a data integrity proof."""
    proof_id = generate_id()

    proof = DataProof(
        id=proof_id,
        domain="data",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        integrity_type=IntegrityType(proof_data.integrity_type),
        schema_before=proof_data.schema_before,
        schema_after=proof_data.schema_after,
        transformation=proof_data.transformation,
        constraints=proof_data.constraints
    )

    proof_store["data"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/data", response_model=ProofListResponse)
async def list_data_proofs():
    """List all data proofs."""
    proofs = list(proof_store["data"].values())
    return ProofListResponse(
        domain="data",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Processes Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/processes", response_model=ProofResponse)
async def create_process_proof(proof_data: ProcessProofCreate, background_tasks: BackgroundTasks):
    """Create a process verification proof."""
    proof_id = generate_id()

    proof = ProcessProof(
        id=proof_id,
        domain="processes",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        property=ProcessProperty(proof_data.property),
        states=[ProcessState(s) for s in proof_data.states] if proof_data.states else [],
        transitions=proof_data.transitions,
        invariants=proof_data.invariants
    )

    proof_store["processes"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/processes", response_model=ProofListResponse)
async def list_process_proofs():
    """List all process proofs."""
    proofs = list(proof_store["processes"].values())
    return ProofListResponse(
        domain="processes",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Insights Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/insights", response_model=ProofResponse)
async def create_insight_proof(proof_data: InsightProofCreate, background_tasks: BackgroundTasks):
    """Create an analytical insight proof."""
    proof_id = generate_id()

    proof = InsightProof(
        id=proof_id,
        domain="insights",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        derivation_type=DerivationType(proof_data.derivation_type),
        category=InsightCategory(proof_data.category),
        data_sources=proof_data.data_sources,
        methodology=proof_data.methodology,
        conclusion=proof_data.conclusion,
        confidence=Confidence(
            value=proof_data.confidence_value,
            lower_bound=proof_data.confidence_value - 0.05,
            upper_bound=min(1.0, proof_data.confidence_value + 0.05),
            method="statistical"
        )
    )

    proof_store["insights"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/insights", response_model=ProofListResponse)
async def list_insight_proofs():
    """List all insight proofs."""
    proofs = list(proof_store["insights"].values())
    return ProofListResponse(
        domain="insights",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Regulations Domain
# -----------------------------------------------------------------------------

@app.post("/proofs/regulations", response_model=ProofResponse)
async def create_regulation_proof(proof_data: RegulationProofCreate, background_tasks: BackgroundTasks):
    """Create a compliance/regulation proof."""
    proof_id = generate_id()

    proof = RegulationProof(
        id=proof_id,
        domain="regulations",
        title=proof_data.title,
        description=proof_data.description,
        status=ProofStatus(proof_data.status),
        evidence=[create_evidence(e) for e in proof_data.evidence],
        metadata=proof_data.metadata,
        regulation_type=RegulationType(proof_data.regulation_type),
        regulation_id=proof_data.regulation_id,
        regulation_text=proof_data.regulation_text,
        implementation=proof_data.implementation,
        compliance_status=ComplianceStatus(proof_data.compliance_status),
        gaps=proof_data.gaps,
        controls=proof_data.controls
    )

    proof_store["regulations"][proof_id] = proof
    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.CREATED)

    return proof_to_response(proof, include_toon=True)


@app.get("/proofs/regulations", response_model=ProofListResponse)
async def list_regulation_proofs():
    """List all regulation proofs."""
    proofs = list(proof_store["regulations"].values())
    return ProofListResponse(
        domain="regulations",
        proofs=[proof_to_response(p) for p in proofs],
        count=len(proofs)
    )


# -----------------------------------------------------------------------------
# Generic Proof Endpoints
# -----------------------------------------------------------------------------

@app.get("/proofs/{domain}/{proof_id}")
async def get_proof(domain: str, proof_id: str, format: str = "json"):
    """Get a specific proof by ID."""
    if domain not in proof_store:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")

    proof = proof_store[domain].get(proof_id)
    if not proof:
        raise HTTPException(status_code=404, detail=f"Proof not found: {proof_id}")

    if format == "toon":
        return {"toon": toon_encoder.encode(proof)}
    else:
        return proof_to_response(proof, include_toon=True)


@app.delete("/proofs/{domain}/{proof_id}")
async def delete_proof(domain: str, proof_id: str, background_tasks: BackgroundTasks):
    """Delete a proof."""
    if domain not in proof_store:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")

    proof = proof_store[domain].pop(proof_id, None)
    if not proof:
        raise HTTPException(status_code=404, detail=f"Proof not found: {proof_id}")

    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.DEPRECATED)

    return {"deleted": True, "id": proof_id}


@app.patch("/proofs/{domain}/{proof_id}/verify")
async def verify_proof(domain: str, proof_id: str, background_tasks: BackgroundTasks):
    """Mark a proof as verified."""
    if domain not in proof_store:
        raise HTTPException(status_code=404, detail=f"Unknown domain: {domain}")

    proof = proof_store[domain].get(proof_id)
    if not proof:
        raise HTTPException(status_code=404, detail=f"Proof not found: {proof_id}")

    proof.status = ProofStatus.VERIFIED
    proof.updated_at = datetime.utcnow()

    background_tasks.add_task(streaming_service.publish_proof, proof, ProofEventType.VERIFIED)

    return proof_to_response(proof)


# -----------------------------------------------------------------------------
# Cross-Domain Linking
# -----------------------------------------------------------------------------

@app.post("/proofs/link")
async def link_proofs(link: LinkRequest, background_tasks: BackgroundTasks):
    """Create a cross-domain link between proofs."""
    # Find source proof
    source_proof = None
    for domain, proofs in proof_store.items():
        if link.source_id in proofs:
            source_proof = proofs[link.source_id]
            break

    if not source_proof:
        raise HTTPException(status_code=404, detail=f"Source proof not found: {link.source_id}")

    # Find target proof
    target_proof = None
    for domain, proofs in proof_store.items():
        if link.target_id in proofs:
            target_proof = proofs[link.target_id]
            break

    if not target_proof:
        raise HTTPException(status_code=404, detail=f"Target proof not found: {link.target_id}")

    # Create link via Kafka
    background_tasks.add_task(
        streaming_service.link_proofs,
        source_proof, target_proof,
        link.relationship, link.metadata
    )

    return {
        "linked": True,
        "source": {"id": link.source_id, "domain": source_proof.domain},
        "target": {"id": link.target_id, "domain": target_proof.domain},
        "relationship": link.relationship
    }


# -----------------------------------------------------------------------------
# TOON Serialization
# -----------------------------------------------------------------------------

@app.post("/toon/encode", response_model=TOONResponse)
async def encode_to_toon(data: Dict[str, Any]):
    """Encode JSON data to TOON format."""
    import json

    toon_str = toon_encoder.encode(data)
    json_str = json.dumps(data)

    toon_tokens = estimate_tokens(toon_str)
    json_tokens = estimate_tokens(json_str)
    savings = ((json_tokens - toon_tokens) / json_tokens * 100) if json_tokens > 0 else 0

    return TOONResponse(
        toon=toon_str,
        token_count=toon_tokens,
        json_token_count=json_tokens,
        savings_percent=round(savings, 1)
    )


@app.post("/toon/decode")
async def decode_from_toon(toon_data: Dict[str, str]):
    """Decode TOON format to JSON."""
    toon_str = toon_data.get("toon", "")
    if not toon_str:
        raise HTTPException(status_code=400, detail="Missing 'toon' field")

    try:
        decoded = toon_decoder.decode(toon_str)
        return {"data": decoded}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode TOON: {str(e)}")


# -----------------------------------------------------------------------------
# Kafka Integration
# -----------------------------------------------------------------------------

@app.get("/kafka/config")
async def get_kafka_config():
    """Get Kafka configuration for n8n integration."""
    return n8n_integration.create_n8n_kafka_node_config()


@app.get("/kafka/topics")
async def list_kafka_topics():
    """List available Kafka topics."""
    return {"topics": streaming_service.config.topics}


@app.get("/kafka/messages/{topic}")
async def get_kafka_messages(topic: str, limit: int = 10):
    """Get recent messages from a Kafka topic (mock client only)."""
    if hasattr(streaming_service.client, 'get_messages'):
        messages = streaming_service.client.get_messages(topic)
        return {"topic": topic, "messages": messages[-limit:], "count": len(messages)}
    return {"topic": topic, "messages": [], "count": 0}


# =============================================================================
# Startup/Shutdown
# =============================================================================

@app.on_event("startup")
async def startup():
    logger.info("Starting Proof Domains API service...")
    logger.info(f"Available domains: {list(DOMAIN_MAP.keys())}")


@app.on_event("shutdown")
async def shutdown():
    logger.info("Shutting down Proof Domains API service...")
    streaming_service.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011, log_level="info")
