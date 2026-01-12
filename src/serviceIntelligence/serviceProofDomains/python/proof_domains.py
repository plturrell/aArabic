"""
Proof Domains - Python Bridge for Lean4 Proofs

Provides:
1. Python representations of 7 proof domains
2. TOON serialization for LLM-efficient encoding
3. Kafka integration for proof streaming
4. Cross-domain proof linking

Port: 8011
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import json
import re


# =============================================================================
# Core Types
# =============================================================================

class ProofStatus(Enum):
    DRAFT = "draft"
    PENDING = "pending"
    VERIFIED = "verified"
    FAILED = "failed"
    DEPRECATED = "deprecated"


class EvidenceType(Enum):
    AXIOM = "axiom"
    DERIVATION = "derivation"
    REFERENCE = "reference"
    EMPIRICAL = "empirical"


@dataclass
class Evidence:
    evidence_type: EvidenceType
    content: str
    sources: List[str] = field(default_factory=list)


@dataclass
class Confidence:
    value: float
    lower_bound: float
    upper_bound: float
    method: str


@dataclass
class Lineage:
    source_id: str
    derived_from: List[str]
    transformation: str
    timestamp: datetime


@dataclass
class BaseProof:
    id: str
    domain: str
    title: str
    description: str
    status: ProofStatus
    evidence: List[Evidence] = field(default_factory=list)
    lineage: Optional[Lineage] = None
    metadata: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


# =============================================================================
# Domain 1: Mathematics
# =============================================================================

class MathProofType(Enum):
    THEOREM = "theorem"
    LEMMA = "lemma"
    COROLLARY = "corollary"
    PROPOSITION = "proposition"
    CONJECTURE = "conjecture"


class MathDomain(Enum):
    ALGEBRA = "algebra"
    ANALYSIS = "analysis"
    GEOMETRY = "geometry"
    TOPOLOGY = "topology"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    PROBABILITY = "probability"
    STATISTICS = "statistics"
    DISCRETE_MATH = "discrete_math"


@dataclass
class MathProof(BaseProof):
    proof_type: MathProofType = MathProofType.THEOREM
    math_domain: MathDomain = MathDomain.ALGEBRA
    statement: str = ""
    formal_proof: str = ""
    dependencies: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.domain = "mathematics"


# =============================================================================
# Domain 2: Code
# =============================================================================

class VerificationType(Enum):
    TYPE_CORRECTNESS = "type_correctness"
    MEMORY_SAFETY = "memory_safety"
    TERMINATES = "terminates"
    FUNCTIONAL = "functional"
    INVARIANT = "invariant"
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"


class ProgrammingLanguage(Enum):
    LEAN4 = "lean4"
    RUST = "rust"
    PYTHON = "python"
    TYPESCRIPT = "typescript"
    SOLIDITY = "solidity"
    OTHER = "other"


@dataclass
class CodeProof(BaseProof):
    verification_type: VerificationType = VerificationType.FUNCTIONAL
    language: ProgrammingLanguage = ProgrammingLanguage.PYTHON
    source_code: str = ""
    specification: str = ""
    invariants: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.domain = "code"


# =============================================================================
# Domain 3: Language
# =============================================================================

class LogicType(Enum):
    PROPOSITIONAL = "propositional"
    FIRST_ORDER = "first_order"
    HIGHER_ORDER = "higher_order"
    MODAL = "modal"
    TEMPORAL = "temporal"
    DEONTIC = "deontic"


class SemanticType(Enum):
    ENTAILMENT = "entailment"
    CONTRADICTION = "contradiction"
    EQUIVALENCE = "equivalence"
    IMPLICATION = "implication"
    CONSISTENCY = "consistency"


@dataclass
class LanguageProof(BaseProof):
    logic_type: LogicType = LogicType.FIRST_ORDER
    semantic_type: SemanticType = SemanticType.ENTAILMENT
    premises: List[str] = field(default_factory=list)
    conclusion: str = ""
    natural_language: str = ""
    formal_representation: str = ""

    def __post_init__(self):
        self.domain = "language"


# =============================================================================
# Domain 4: Data
# =============================================================================

class IntegrityType(Enum):
    SCHEMA = "schema"
    REFERENTIAL = "referential"
    DOMAIN = "domain"
    ENTITY = "entity"
    USER_DEFINED = "user_defined"


class TransformationType(Enum):
    MAP = "map"
    FILTER = "filter"
    AGGREGATE = "aggregate"
    JOIN = "join"
    NORMALIZE = "normalize"
    DENORMALIZE = "denormalize"


@dataclass
class DataProof(BaseProof):
    integrity_type: IntegrityType = IntegrityType.SCHEMA
    schema_before: str = ""
    schema_after: str = ""
    transformation: Optional[str] = None
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.domain = "data"


# =============================================================================
# Domain 5: Processes
# =============================================================================

class ProcessProperty(Enum):
    LIVENESS = "liveness"
    SAFETY = "safety"
    FAIRNESS = "fairness"
    DEADLOCK_FREE = "deadlock_free"
    TERMINATES = "terminates"
    DETERMINISTIC = "deterministic"


class ProcessState(Enum):
    INITIAL = "initial"
    RUNNING = "running"
    WAITING = "waiting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessProof(BaseProof):
    property: ProcessProperty = ProcessProperty.LIVENESS
    states: List[ProcessState] = field(default_factory=list)
    transitions: List[Dict[str, str]] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.domain = "processes"


# =============================================================================
# Domain 6: Insights
# =============================================================================

class DerivationType(Enum):
    INDUCTIVE = "inductive"
    DEDUCTIVE = "deductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    STATISTICAL = "statistical"


class InsightCategory(Enum):
    PATTERN = "pattern"
    ANOMALY = "anomaly"
    TREND = "trend"
    CORRELATION = "correlation"
    CAUSATION = "causation"
    PREDICTION = "prediction"


@dataclass
class InsightProof(BaseProof):
    derivation_type: DerivationType = DerivationType.STATISTICAL
    category: InsightCategory = InsightCategory.PATTERN
    data_sources: List[str] = field(default_factory=list)
    methodology: str = ""
    conclusion: str = ""
    confidence: Optional[Confidence] = None

    def __post_init__(self):
        self.domain = "insights"


# =============================================================================
# Domain 7: Regulations
# =============================================================================

class RegulationType(Enum):
    LAW = "law"
    POLICY = "policy"
    STANDARD = "standard"
    GUIDELINE = "guideline"
    CONTRACT = "contract"
    SLA = "sla"


class ComplianceStatus(Enum):
    COMPLIANT = "compliant"
    NON_COMPLIANT = "non_compliant"
    PARTIALLY_COMPLIANT = "partially_compliant"
    NOT_APPLICABLE = "not_applicable"
    PENDING = "pending"


@dataclass
class RegulationProof(BaseProof):
    regulation_type: RegulationType = RegulationType.POLICY
    regulation_id: str = ""
    regulation_text: str = ""
    implementation: str = ""
    compliance_status: ComplianceStatus = ComplianceStatus.PENDING
    gaps: List[str] = field(default_factory=list)
    controls: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.domain = "regulations"


# =============================================================================
# Unified Proof Type
# =============================================================================

UnifiedProof = Union[
    MathProof,
    CodeProof,
    LanguageProof,
    DataProof,
    ProcessProof,
    InsightProof,
    RegulationProof
]

DOMAIN_MAP = {
    "mathematics": MathProof,
    "code": CodeProof,
    "language": LanguageProof,
    "data": DataProof,
    "processes": ProcessProof,
    "insights": InsightProof,
    "regulations": RegulationProof
}


# =============================================================================
# TOON Serialization
# =============================================================================

class TOONEncoder:
    """
    Token-Oriented Object Notation encoder for proof domains.
    Produces LLM-efficient encoding with ~40% fewer tokens than JSON.
    """

    def __init__(self, indent: int = 2, delimiter: str = ","):
        self.indent = indent
        self.delimiter = delimiter

    def encode(self, obj: Any) -> str:
        """Encode object to TOON format."""
        if isinstance(obj, (MathProof, CodeProof, LanguageProof, DataProof,
                           ProcessProof, InsightProof, RegulationProof)):
            return self._encode_proof(obj)
        elif isinstance(obj, list):
            return self._encode_array(obj)
        elif isinstance(obj, dict):
            return self._encode_object(obj)
        else:
            return self._encode_primitive(obj)

    def _encode_proof(self, proof: BaseProof) -> str:
        """Encode a proof object to TOON."""
        lines = []
        data = self._proof_to_dict(proof)

        # Header with domain
        lines.append(f"proof[{proof.domain}]:")

        # Core fields
        lines.append(f"  id: {data['id']}")
        lines.append(f"  title: {self._escape_string(data['title'])}")
        lines.append(f"  status: {data['status']}")

        # Description (may be multiline)
        if data.get('description'):
            lines.append(f"  description: {self._escape_string(data['description'])}")

        # Domain-specific fields
        domain_fields = self._get_domain_fields(proof)
        for key, value in domain_fields.items():
            lines.append(f"  {key}: {self._encode_value(value)}")

        # Evidence array (tabular format)
        if data.get('evidence'):
            evidence = data['evidence']
            if evidence:
                lines.append(f"  evidence[{len(evidence)}]{{type,content}}:")
                for e in evidence:
                    etype = e.get('evidence_type', 'axiom')
                    content = self._escape_string(e.get('content', ''))
                    lines.append(f"    {etype},{content}")

        # Metadata
        if data.get('metadata'):
            lines.append("  metadata:")
            for k, v in data['metadata'].items():
                lines.append(f"    {k}: {v}")

        # Lineage
        if data.get('lineage'):
            lineage = data['lineage']
            lines.append("  lineage:")
            lines.append(f"    source_id: {lineage['source_id']}")
            if lineage.get('derived_from'):
                lines.append(f"    derived_from: [{','.join(lineage['derived_from'])}]")

        return "\n".join(lines)

    def _get_domain_fields(self, proof: BaseProof) -> Dict[str, Any]:
        """Extract domain-specific fields from proof."""
        fields = {}

        if isinstance(proof, MathProof):
            fields['proof_type'] = proof.proof_type.value
            fields['math_domain'] = proof.math_domain.value
            fields['statement'] = proof.statement
            if proof.formal_proof:
                fields['formal_proof'] = proof.formal_proof
            if proof.dependencies:
                fields['dependencies'] = proof.dependencies

        elif isinstance(proof, CodeProof):
            fields['verification_type'] = proof.verification_type.value
            fields['language'] = proof.language.value
            if proof.specification:
                fields['specification'] = proof.specification
            if proof.invariants:
                fields['invariants'] = proof.invariants

        elif isinstance(proof, LanguageProof):
            fields['logic_type'] = proof.logic_type.value
            fields['semantic_type'] = proof.semantic_type.value
            if proof.premises:
                fields['premises'] = proof.premises
            fields['conclusion'] = proof.conclusion

        elif isinstance(proof, DataProof):
            fields['integrity_type'] = proof.integrity_type.value
            if proof.constraints:
                fields['constraints'] = proof.constraints

        elif isinstance(proof, ProcessProof):
            fields['property'] = proof.property.value
            if proof.invariants:
                fields['invariants'] = proof.invariants

        elif isinstance(proof, InsightProof):
            fields['derivation_type'] = proof.derivation_type.value
            fields['category'] = proof.category.value
            fields['conclusion'] = proof.conclusion
            if proof.confidence:
                fields['confidence'] = proof.confidence.value

        elif isinstance(proof, RegulationProof):
            fields['regulation_type'] = proof.regulation_type.value
            fields['regulation_id'] = proof.regulation_id
            fields['compliance_status'] = proof.compliance_status.value
            if proof.gaps:
                fields['gaps'] = proof.gaps

        return fields

    def _proof_to_dict(self, proof: BaseProof) -> Dict[str, Any]:
        """Convert proof to dictionary."""
        data = {
            'id': proof.id,
            'domain': proof.domain,
            'title': proof.title,
            'description': proof.description,
            'status': proof.status.value,
            'evidence': [
                {
                    'evidence_type': e.evidence_type.value,
                    'content': e.content,
                    'sources': e.sources
                }
                for e in proof.evidence
            ],
            'metadata': proof.metadata,
            'created_at': proof.created_at.isoformat() if proof.created_at else None,
            'updated_at': proof.updated_at.isoformat() if proof.updated_at else None
        }

        if proof.lineage:
            data['lineage'] = {
                'source_id': proof.lineage.source_id,
                'derived_from': proof.lineage.derived_from,
                'transformation': proof.lineage.transformation,
                'timestamp': proof.lineage.timestamp.isoformat()
            }

        return data

    def _encode_object(self, obj: Dict) -> str:
        """Encode dictionary to TOON."""
        lines = []
        for key, value in obj.items():
            encoded = self._encode_value(value)
            if '\n' in str(encoded):
                lines.append(f"{key}:")
                for line in str(encoded).split('\n'):
                    lines.append(f"  {line}")
            else:
                lines.append(f"{key}: {encoded}")
        return "\n".join(lines)

    def _encode_array(self, arr: List) -> str:
        """Encode array to TOON tabular format if uniform."""
        if not arr:
            return "[]"

        # Check if all items have same keys (uniform array)
        if all(isinstance(item, dict) for item in arr):
            keys = set(arr[0].keys())
            if all(set(item.keys()) == keys for item in arr):
                # Use tabular format
                header = self.delimiter.join(keys)
                rows = []
                for item in arr:
                    row = self.delimiter.join(str(item[k]) for k in keys)
                    rows.append(row)
                return f"[{len(arr)}]{{{header}}}:\n" + "\n".join(f"  {r}" for r in rows)

        # Fall back to list format
        items = [self._encode_value(item) for item in arr]
        return "[" + self.delimiter.join(items) + "]"

    def _encode_value(self, value: Any) -> str:
        """Encode a single value."""
        if value is None:
            return "null"
        elif isinstance(value, bool):
            return "true" if value else "false"
        elif isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            return self._escape_string(value)
        elif isinstance(value, list):
            if len(value) <= 5 and all(isinstance(v, (str, int, float)) for v in value):
                return "[" + self.delimiter.join(str(v) for v in value) + "]"
            return self._encode_array(value)
        elif isinstance(value, dict):
            return "{" + ",".join(f"{k}:{self._encode_value(v)}" for k, v in value.items()) + "}"
        elif isinstance(value, Enum):
            return value.value
        elif isinstance(value, datetime):
            return value.isoformat()
        else:
            return str(value)

    def _escape_string(self, s: str) -> str:
        """Escape string for TOON."""
        if not s:
            return '""'
        # Only quote if contains special characters
        if any(c in s for c in [',', ':', '\n', '"', '[', ']', '{', '}']):
            escaped = s.replace('\\', '\\\\').replace('"', '\\"')
            return f'"{escaped}"'
        return s


class TOONDecoder:
    """Decode TOON format back to proof objects."""

    def decode(self, toon_str: str) -> UnifiedProof:
        """Decode TOON string to proof object."""
        lines = toon_str.strip().split('\n')

        # Parse header
        header = lines[0]
        domain_match = re.match(r'proof\[(\w+)\]:', header)
        if not domain_match:
            raise ValueError(f"Invalid TOON proof header: {header}")

        domain = domain_match.group(1)
        proof_class = DOMAIN_MAP.get(domain)
        if not proof_class:
            raise ValueError(f"Unknown domain: {domain}")

        # Parse fields
        data = self._parse_body(lines[1:])

        # Create proof object
        return self._create_proof(proof_class, data)

    def _parse_body(self, lines: List[str]) -> Dict[str, Any]:
        """Parse TOON body into dictionary."""
        data = {}
        current_key = None
        current_indent = 0

        for line in lines:
            if not line.strip():
                continue

            indent = len(line) - len(line.lstrip())
            content = line.strip()

            if ':' in content and not content.startswith('-'):
                key, _, value = content.partition(':')
                key = key.strip()
                value = value.strip()

                if value:
                    data[key] = self._parse_value(value)
                else:
                    data[key] = {}
                    current_key = key
                    current_indent = indent
            elif current_key and indent > current_indent:
                # Nested value
                if isinstance(data[current_key], dict):
                    if ':' in content:
                        k, _, v = content.partition(':')
                        data[current_key][k.strip()] = self._parse_value(v.strip())
                    elif content.startswith('-'):
                        if not isinstance(data[current_key], list):
                            data[current_key] = []
                        data[current_key].append(self._parse_value(content[1:].strip()))

        return data

    def _parse_value(self, value: str) -> Any:
        """Parse a TOON value."""
        if value == 'null':
            return None
        elif value == 'true':
            return True
        elif value == 'false':
            return False
        elif value.startswith('"') and value.endswith('"'):
            return value[1:-1].replace('\\"', '"').replace('\\\\', '\\')
        elif value.startswith('[') and value.endswith(']'):
            inner = value[1:-1]
            if not inner:
                return []
            return [self._parse_value(v.strip()) for v in inner.split(',')]
        else:
            try:
                if '.' in value:
                    return float(value)
                return int(value)
            except ValueError:
                return value

    def _create_proof(self, proof_class, data: Dict) -> UnifiedProof:
        """Create proof object from parsed data."""
        # Convert status
        status = ProofStatus(data.get('status', 'draft'))

        # Build evidence
        evidence = []
        if 'evidence' in data:
            for e in data['evidence']:
                evidence.append(Evidence(
                    evidence_type=EvidenceType(e.get('evidence_type', 'axiom')),
                    content=e.get('content', ''),
                    sources=e.get('sources', [])
                ))

        # Create base proof
        proof = proof_class(
            id=data.get('id', ''),
            domain=data.get('domain', ''),
            title=data.get('title', ''),
            description=data.get('description', ''),
            status=status,
            evidence=evidence,
            metadata=data.get('metadata', {})
        )

        return proof


# =============================================================================
# Proof Event (for Kafka streaming)
# =============================================================================

class ProofEventType(Enum):
    CREATED = "created"
    UPDATED = "updated"
    VERIFIED = "verified"
    FAILED = "failed"
    DEPRECATED = "deprecated"
    LINKED = "linked"


@dataclass
class ProofEvent:
    event_id: str
    event_type: ProofEventType
    proof_id: str
    domain: str
    timestamp: datetime
    payload: str  # TOON-encoded proof

    def to_kafka_message(self) -> Dict[str, Any]:
        """Convert to Kafka message format."""
        return {
            'key': self.proof_id,
            'value': {
                'event_id': self.event_id,
                'event_type': self.event_type.value,
                'proof_id': self.proof_id,
                'domain': self.domain,
                'timestamp': self.timestamp.isoformat(),
                'payload': self.payload
            },
            'headers': {
                'domain': self.domain,
                'event_type': self.event_type.value
            }
        }


def create_proof_event(
    proof: UnifiedProof,
    event_type: ProofEventType,
    encoder: TOONEncoder = None
) -> ProofEvent:
    """Create a proof event for Kafka streaming."""
    encoder = encoder or TOONEncoder()
    timestamp = datetime.utcnow()

    return ProofEvent(
        event_id=f"{proof.id}-{int(timestamp.timestamp())}",
        event_type=event_type,
        proof_id=proof.id,
        domain=proof.domain,
        timestamp=timestamp,
        payload=encoder.encode(proof)
    )
