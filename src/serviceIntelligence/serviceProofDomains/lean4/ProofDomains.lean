/-
  ProofDomains - Unified Lean4 Proof Framework

  7 Proof Domains:
  1. Mathematics - Formal mathematical proofs
  2. Code - Program verification and correctness
  3. Language - Natural language logic and semantics
  4. Data - Data integrity and transformations
  5. Processes - Workflow and process correctness
  6. Insights - Analytical derivations and conclusions
  7. Regulations - Compliance and policy verification

  Aligned with TOON (Token-Oriented Object Notation) for LLM-efficient serialization
  Integrated with Kafka for streaming proof events
-/

namespace ProofDomains

-- =============================================================================
-- Core Proof Types
-- =============================================================================

/-- Base evidence type for all proofs -/
inductive Evidence where
  | axiom : String → Evidence
  | derivation : String → List Evidence → Evidence
  | reference : String → Evidence
  | empirical : String → String → Evidence  -- source, data
  deriving Repr, DecidableEq

/-- Proof status for tracking verification state -/
inductive ProofStatus where
  | draft
  | pending
  | verified
  | failed
  | deprecated
  deriving Repr, DecidableEq

/-- Confidence level for probabilistic proofs -/
structure Confidence where
  value : Float
  lower_bound : Float
  upper_bound : Float
  method : String
  deriving Repr

/-- Lineage tracking for proof provenance -/
structure Lineage where
  source_id : String
  derived_from : List String
  transformation : String
  timestamp : Nat
  deriving Repr

/-- Base proof structure all domains extend -/
structure BaseProof where
  id : String
  domain : String
  title : String
  description : String
  status : ProofStatus
  evidence : List Evidence
  lineage : Option Lineage
  metadata : List (String × String)
  deriving Repr

-- =============================================================================
-- Domain 1: Mathematics
-- =============================================================================

namespace Mathematics

/-- Mathematical proof types -/
inductive MathProofType where
  | theorem
  | lemma
  | corollary
  | proposition
  | conjecture
  deriving Repr, DecidableEq

/-- Mathematical domain (area of mathematics) -/
inductive MathDomain where
  | algebra
  | analysis
  | geometry
  | topology
  | numberTheory
  | combinatorics
  | probability
  | statistics
  | discreteMath
  deriving Repr, DecidableEq

/-- Formal mathematical proof -/
structure MathProof extends BaseProof where
  proof_type : MathProofType
  math_domain : MathDomain
  statement : String
  formal_proof : String
  dependencies : List String
  deriving Repr

/-- Mathematical axiom -/
structure MathAxiom where
  id : String
  name : String
  statement : String
  domain : MathDomain
  deriving Repr

/-- Theorem with formal statement and proof -/
def mkTheorem (id name statement proof : String) (domain : MathDomain) : MathProof :=
  { id := id
  , domain := "mathematics"
  , title := name
  , description := statement
  , status := ProofStatus.draft
  , evidence := [Evidence.derivation proof []]
  , lineage := none
  , metadata := [("math_domain", toString domain)]
  , proof_type := MathProofType.theorem
  , math_domain := domain
  , statement := statement
  , formal_proof := proof
  , dependencies := []
  }

end Mathematics

-- =============================================================================
-- Domain 2: Code
-- =============================================================================

namespace Code

/-- Code verification types -/
inductive VerificationType where
  | typeCorrectness
  | memorySafety
  | terminates
  | functional
  | invariant
  | precondition
  | postcondition
  deriving Repr, DecidableEq

/-- Programming language -/
inductive Language where
  | lean4
  | rust
  | python
  | typescript
  | solidity
  | other : String → Language
  deriving Repr, DecidableEq

/-- Code proof for program verification -/
structure CodeProof extends BaseProof where
  verification_type : VerificationType
  language : Language
  source_code : String
  specification : String
  invariants : List String
  deriving Repr

/-- Function contract (pre/post conditions) -/
structure FunctionContract where
  function_name : String
  preconditions : List String
  postconditions : List String
  invariants : List String
  deriving Repr

/-- Create a code verification proof -/
def mkCodeProof (id name : String) (code spec : String) (lang : Language) : CodeProof :=
  { id := id
  , domain := "code"
  , title := name
  , description := s!"Verification of {name}"
  , status := ProofStatus.draft
  , evidence := []
  , lineage := none
  , metadata := [("language", toString lang)]
  , verification_type := VerificationType.functional
  , language := lang
  , source_code := code
  , specification := spec
  , invariants := []
  }

end Code

-- =============================================================================
-- Domain 3: Language
-- =============================================================================

namespace Language

/-- Language logic types -/
inductive LogicType where
  | propositional
  | firstOrder
  | higherOrder
  | modal
  | temporal
  | deontic
  deriving Repr, DecidableEq

/-- Semantic analysis type -/
inductive SemanticType where
  | entailment
  | contradiction
  | equivalence
  | implication
  | consistency
  deriving Repr, DecidableEq

/-- Natural language proof -/
structure LanguageProof extends BaseProof where
  logic_type : LogicType
  semantic_type : SemanticType
  premises : List String
  conclusion : String
  natural_language : String
  formal_representation : String
  deriving Repr

/-- Semantic relation between statements -/
structure SemanticRelation where
  statement_a : String
  statement_b : String
  relation : SemanticType
  confidence : Confidence
  deriving Repr

/-- Create a language entailment proof -/
def mkEntailment (id : String) (premises : List String) (conclusion : String) : LanguageProof :=
  { id := id
  , domain := "language"
  , title := s!"Entailment: {conclusion}"
  , description := "Natural language entailment proof"
  , status := ProofStatus.draft
  , evidence := premises.map Evidence.axiom
  , lineage := none
  , metadata := []
  , logic_type := LogicType.firstOrder
  , semantic_type := SemanticType.entailment
  , premises := premises
  , conclusion := conclusion
  , natural_language := String.intercalate ", " premises ++ " ⊨ " ++ conclusion
  , formal_representation := ""
  }

end Language

-- =============================================================================
-- Domain 4: Data
-- =============================================================================

namespace Data

/-- Data integrity types -/
inductive IntegrityType where
  | schema
  | referential
  | domain
  | entity
  | user_defined
  deriving Repr, DecidableEq

/-- Data transformation types -/
inductive TransformationType where
  | map
  | filter
  | aggregate
  | join
  | normalize
  | denormalize
  deriving Repr, DecidableEq

/-- Data proof for integrity and transformations -/
structure DataProof extends BaseProof where
  integrity_type : IntegrityType
  schema_before : String
  schema_after : String
  transformation : Option String
  constraints : List String
  deriving Repr

/-- Schema definition -/
structure Schema where
  name : String
  fields : List (String × String × Bool)  -- name, type, nullable
  primary_key : List String
  foreign_keys : List (String × String × String)  -- field, ref_table, ref_field
  deriving Repr

/-- Data transformation proof -/
structure TransformationProof where
  id : String
  input_schema : Schema
  output_schema : Schema
  transformation_type : TransformationType
  preserves_integrity : Bool
  proof : String
  deriving Repr

/-- Create a schema integrity proof -/
def mkSchemaProof (id name : String) (schema : String) (constraints : List String) : DataProof :=
  { id := id
  , domain := "data"
  , title := name
  , description := s!"Schema integrity proof for {name}"
  , status := ProofStatus.draft
  , evidence := constraints.map Evidence.axiom
  , lineage := none
  , metadata := []
  , integrity_type := IntegrityType.schema
  , schema_before := schema
  , schema_after := schema
  , transformation := none
  , constraints := constraints
  }

end Data

-- =============================================================================
-- Domain 5: Processes
-- =============================================================================

namespace Processes

/-- Process property types -/
inductive ProcessProperty where
  | liveness
  | safety
  | fairness
  | deadlockFree
  | terminates
  | deterministic
  deriving Repr, DecidableEq

/-- Process state -/
inductive ProcessState where
  | initial
  | running
  | waiting
  | completed
  | failed
  deriving Repr, DecidableEq

/-- Process proof for workflow verification -/
structure ProcessProof extends BaseProof where
  property : ProcessProperty
  states : List ProcessState
  transitions : List (ProcessState × String × ProcessState)
  invariants : List String
  deriving Repr

/-- Workflow definition -/
structure Workflow where
  id : String
  name : String
  steps : List String
  transitions : List (String × String × String)  -- from, condition, to
  initial_state : String
  final_states : List String
  deriving Repr

/-- Create a process liveness proof -/
def mkLivenessProof (id name : String) (workflow : Workflow) : ProcessProof :=
  { id := id
  , domain := "processes"
  , title := s!"Liveness: {name}"
  , description := s!"Process liveness proof for {workflow.name}"
  , status := ProofStatus.draft
  , evidence := []
  , lineage := none
  , metadata := [("workflow", workflow.name)]
  , property := ProcessProperty.liveness
  , states := [ProcessState.initial, ProcessState.completed]
  , transitions := []
  , invariants := []
  }

end Processes

-- =============================================================================
-- Domain 6: Insights
-- =============================================================================

namespace Insights

/-- Insight derivation types -/
inductive DerivationType where
  | inductive
  | deductive
  | abductive
  | analogical
  | statistical
  deriving Repr, DecidableEq

/-- Insight category -/
inductive InsightCategory where
  | pattern
  | anomaly
  | trend
  | correlation
  | causation
  | prediction
  deriving Repr, DecidableEq

/-- Insight proof for analytical conclusions -/
structure InsightProof extends BaseProof where
  derivation_type : DerivationType
  category : InsightCategory
  data_sources : List String
  methodology : String
  conclusion : String
  confidence : Confidence
  deriving Repr

/-- Statistical evidence -/
structure StatisticalEvidence where
  test_name : String
  statistic : Float
  p_value : Float
  effect_size : Float
  sample_size : Nat
  deriving Repr

/-- Create an insight proof -/
def mkInsight (id name conclusion : String) (conf : Float) (sources : List String) : InsightProof :=
  { id := id
  , domain := "insights"
  , title := name
  , description := conclusion
  , status := ProofStatus.draft
  , evidence := sources.map (Evidence.empirical "analysis")
  , lineage := none
  , metadata := []
  , derivation_type := DerivationType.statistical
  , category := InsightCategory.pattern
  , data_sources := sources
  , methodology := "statistical_analysis"
  , conclusion := conclusion
  , confidence := { value := conf, lower_bound := conf - 0.1, upper_bound := conf + 0.1, method := "bootstrap" }
  }

end Insights

-- =============================================================================
-- Domain 7: Regulations
-- =============================================================================

namespace Regulations

/-- Regulation types -/
inductive RegulationType where
  | law
  | policy
  | standard
  | guideline
  | contract
  | sla
  deriving Repr, DecidableEq

/-- Compliance status -/
inductive ComplianceStatus where
  | compliant
  | nonCompliant
  | partiallyCompliant
  | notApplicable
  | pending
  deriving Repr, DecidableEq

/-- Regulation proof for compliance verification -/
structure RegulationProof extends BaseProof where
  regulation_type : RegulationType
  regulation_id : String
  regulation_text : String
  implementation : String
  compliance_status : ComplianceStatus
  gaps : List String
  controls : List String
  deriving Repr

/-- Compliance control -/
structure Control where
  id : String
  name : String
  description : String
  implementation : String
  evidence : List String
  deriving Repr

/-- Create a compliance proof -/
def mkComplianceProof (id name reg_id reg_text impl : String) : RegulationProof :=
  { id := id
  , domain := "regulations"
  , title := name
  , description := s!"Compliance proof for {reg_id}"
  , status := ProofStatus.draft
  , evidence := [Evidence.reference reg_id]
  , lineage := none
  , metadata := [("regulation_id", reg_id)]
  , regulation_type := RegulationType.policy
  , regulation_id := reg_id
  , regulation_text := reg_text
  , implementation := impl
  , compliance_status := ComplianceStatus.pending
  , gaps := []
  , controls := []
  }

end Regulations

-- =============================================================================
-- Unified Proof Type
-- =============================================================================

/-- Union type for all proof domains -/
inductive UnifiedProof where
  | math : Mathematics.MathProof → UnifiedProof
  | code : Code.CodeProof → UnifiedProof
  | language : Language.LanguageProof → UnifiedProof
  | data : Data.DataProof → UnifiedProof
  | process : Processes.ProcessProof → UnifiedProof
  | insight : Insights.InsightProof → UnifiedProof
  | regulation : Regulations.RegulationProof → UnifiedProof
  deriving Repr

/-- Get base proof from unified proof -/
def UnifiedProof.toBase : UnifiedProof → BaseProof
  | .math p => p.toBaseProof
  | .code p => p.toBaseProof
  | .language p => p.toBaseProof
  | .data p => p.toBaseProof
  | .process p => p.toBaseProof
  | .insight p => p.toBaseProof
  | .regulation p => p.toBaseProof

/-- Get domain name from unified proof -/
def UnifiedProof.domainName : UnifiedProof → String
  | .math _ => "mathematics"
  | .code _ => "code"
  | .language _ => "language"
  | .data _ => "data"
  | .process _ => "processes"
  | .insight _ => "insights"
  | .regulation _ => "regulations"

-- =============================================================================
-- Proof Events (for Kafka streaming)
-- =============================================================================

/-- Event types for proof lifecycle -/
inductive ProofEventType where
  | created
  | updated
  | verified
  | failed
  | deprecated
  | linked
  deriving Repr, DecidableEq

/-- Proof event for streaming -/
structure ProofEvent where
  event_id : String
  event_type : ProofEventType
  proof_id : String
  domain : String
  timestamp : Nat
  payload : String  -- TOON-encoded proof data
  deriving Repr

/-- Create a proof event -/
def mkProofEvent (proof : UnifiedProof) (eventType : ProofEventType) (ts : Nat) : ProofEvent :=
  let base := proof.toBase
  { event_id := s!"{base.id}-{ts}"
  , event_type := eventType
  , proof_id := base.id
  , domain := proof.domainName
  , timestamp := ts
  , payload := toString (repr proof)
  }

end ProofDomains
