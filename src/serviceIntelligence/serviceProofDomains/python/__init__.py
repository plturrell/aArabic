"""
Proof Domains - Unified Lean4 Proof Framework

7 Proof Domains aligned with TOON serialization and Kafka streaming:
1. Mathematics - Formal mathematical proofs
2. Code - Program verification and correctness
3. Language - Natural language logic and semantics
4. Data - Data integrity and transformations
5. Processes - Workflow and process correctness
6. Insights - Analytical derivations and conclusions
7. Regulations - Compliance and policy verification
"""

from .proof_domains import (
    # Core types
    ProofStatus, Evidence, EvidenceType, Confidence, Lineage, BaseProof,
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

from .kafka_integration import (
    KafkaConfig, ProofStreamingService,
    N8nIntegration, N8nWebhookConfig,
    create_streaming_service
)

__version__ = "1.0.0"
__all__ = [
    # Core
    "ProofStatus", "Evidence", "EvidenceType", "Confidence", "Lineage", "BaseProof",
    # Mathematics
    "MathProof", "MathProofType", "MathDomain",
    # Code
    "CodeProof", "VerificationType", "ProgrammingLanguage",
    # Language
    "LanguageProof", "LogicType", "SemanticType",
    # Data
    "DataProof", "IntegrityType", "TransformationType",
    # Processes
    "ProcessProof", "ProcessProperty", "ProcessState",
    # Insights
    "InsightProof", "DerivationType", "InsightCategory",
    # Regulations
    "RegulationProof", "RegulationType", "ComplianceStatus",
    # Serialization
    "TOONEncoder", "TOONDecoder",
    # Events
    "ProofEvent", "ProofEventType", "create_proof_event",
    # Kafka
    "KafkaConfig", "ProofStreamingService", "N8nIntegration", "create_streaming_service",
    # Types
    "UnifiedProof", "DOMAIN_MAP"
]
