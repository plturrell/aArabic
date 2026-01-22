"""
Mojo port of tau2 evaluator base.
Provides the base trait for all evaluators.
Migrated from evaluator_base.py to pure Mojo.
"""

from collections import Dict, List


# ============================================================================
# Reward Types - mirrors Python RewardType enum
# ============================================================================

@value
struct RewardType:
    """Reward type identifiers."""
    var value: String
    
    alias DB = RewardType("DB")
    alias ENV_ASSERTION = RewardType("ENV_ASSERTION")
    alias NL_ASSERTION = RewardType("NL_ASSERTION")
    alias ACTION = RewardType("ACTION")
    alias COMMUNICATE = RewardType("COMMUNICATE")
    
    fn __init__(out self, value: String):
        self.value = value
    
    fn __eq__(self, other: RewardType) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: RewardType) -> Bool:
        return self.value != other.value
    
    fn __hash__(self) -> UInt:
        return hash(self.value)


# ============================================================================
# Termination Reasons
# ============================================================================

@value
struct TerminationReason:
    """Reason for simulation termination."""
    var value: String
    
    alias USER_STOP = TerminationReason("user_stop")
    alias AGENT_STOP = TerminationReason("agent_stop")
    alias MAX_STEPS = TerminationReason("max_steps")
    alias TOO_MANY_ERRORS = TerminationReason("too_many_errors")
    
    fn __init__(out self, value: String):
        self.value = value
    
    fn __eq__(self, other: TerminationReason) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: TerminationReason) -> Bool:
        return self.value != other.value


# ============================================================================
# Evaluation Check Structures
# ============================================================================

@value
struct NLAssertionCheck:
    """A natural language assertion check result."""
    var nl_assertion: String
    var met: Bool
    var justification: String
    
    fn __init__(out self, nl_assertion: String = "", met: Bool = False, justification: String = ""):
        self.nl_assertion = nl_assertion
        self.met = met
        self.justification = justification


@value
struct CommunicateCheck:
    """A communication check result."""
    var info: String
    var met: Bool
    var justification: String
    
    fn __init__(out self, info: String = "", met: Bool = False, justification: String = ""):
        self.info = info
        self.met = met
        self.justification = justification


@value
struct DBCheck:
    """A database check result."""
    var db_match: Bool
    var db_reward: Float64
    
    fn __init__(out self, db_match: Bool = False, db_reward: Float64 = 0.0):
        self.db_match = db_match
        self.db_reward = db_reward


@value
struct EnvAssertionCheck:
    """An environment assertion check result."""
    var env_assertion: String
    var met: Bool
    var reward: Float64
    
    fn __init__(out self, env_assertion: String = "", met: Bool = False, reward: Float64 = 0.0):
        self.env_assertion = env_assertion
        self.met = met
        self.reward = reward


@value
struct ActionCheck:
    """An action check result."""
    var action_name: String
    var action_id: String
    var action_match: Bool
    var action_reward: Float64
    
    fn __init__(out self, action_name: String = "", action_id: String = "", 
                action_match: Bool = False, action_reward: Float64 = 0.0):
        self.action_name = action_name
        self.action_id = action_id
        self.action_match = action_match
        self.action_reward = action_reward


# ============================================================================
# Reward Info Structure
# ============================================================================

@value
struct RewardInfo:
    """The reward information from evaluation."""
    var reward: Float64
    var db_check: DBCheck
    var has_db_check: Bool
    var env_assertions: List[EnvAssertionCheck]
    var action_checks: List[ActionCheck]
    var nl_assertions: List[NLAssertionCheck]
    var communicate_checks: List[CommunicateCheck]
    var reward_basis: List[RewardType]
    var reward_breakdown: Dict[String, Float64]
    var info: Dict[String, String]
    
    fn __init__(out self, reward: Float64 = 0.0):
        self.reward = reward
        self.db_check = DBCheck()
        self.has_db_check = False
        self.env_assertions = List[EnvAssertionCheck]()
        self.action_checks = List[ActionCheck]()
        self.nl_assertions = List[NLAssertionCheck]()
        self.communicate_checks = List[CommunicateCheck]()
        self.reward_basis = List[RewardType]()
        self.reward_breakdown = Dict[String, Float64]()
        self.info = Dict[String, String]()

    fn with_db_check(owned self, check: DBCheck) -> RewardInfo:
        """Set DB check and return self for chaining."""
        self.db_check = check
        self.has_db_check = True
        return self^

    fn with_note(owned self, note: String) -> RewardInfo:
        """Add a note to info and return self for chaining."""
        self.info["note"] = note
        return self^

    fn add_reward_breakdown(mut self, reward_type: RewardType, value: Float64):
        """Add a reward breakdown entry."""
        self.reward_breakdown[reward_type.value] = value

    fn set_action_checks(mut self, checks: List[ActionCheck]):
        """Set the action checks list."""
        self.action_checks = checks

    fn set_communicate_checks(mut self, checks: List[CommunicateCheck]):
        """Set the communicate checks list."""
        self.communicate_checks = checks

    fn set_nl_assertions(mut self, assertions: List[NLAssertionCheck]):
        """Set the NL assertions list."""
        self.nl_assertions = assertions

    fn set_env_assertions(mut self, assertions: List[EnvAssertionCheck]):
        """Set the env assertions list."""
        self.env_assertions = assertions


# ============================================================================
# Base Evaluator Trait
# ============================================================================

trait Evaluator:
    """
    Base trait for all evaluators.
    Evaluators calculate rewards based on simulation trajectories.
    """
    pass

