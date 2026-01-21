"""
Mojo port of tau2 main evaluator module.
Provides unified evaluation of simulations across all evaluation types.
Migrated from evaluator.py to pure Mojo.
"""

from collections import Dict, List
from .evaluator_base import (
    RewardInfo,
    RewardType,
    TerminationReason,
    DBCheck,
    ActionCheck,
    CommunicateCheck,
    NLAssertionCheck,
    EnvAssertionCheck,
)
from .evaluator_action import ActionEvaluator, Action, ToolCallInfo
from .evaluator_communicate import CommunicateEvaluator, MessageInfo
from .evaluator_env import EnvironmentEvaluator, EvaluationContext, EnvironmentState, EnvAssertion
from .evaluator_nl_assertions import NLAssertionsEvaluator, NLEvaluationResult


# ============================================================================
# Evaluation Type
# ============================================================================

@value
struct EvaluationType:
    """Type of evaluation to perform."""
    var value: String
    
    alias ENV = EvaluationType("env")
    alias NL_ASSERTIONS = EvaluationType("nl_assertions")
    alias COMMUNICATE = EvaluationType("communicate")
    alias ACTION = EvaluationType("action")
    alias ALL = EvaluationType("all")
    
    fn __init__(out self, value: String):
        self.value = value
    
    fn __eq__(self, other: EvaluationType) -> Bool:
        return self.value == other.value
    
    fn __ne__(self, other: EvaluationType) -> Bool:
        return self.value != other.value


# ============================================================================
# Simulation Context for Evaluation
# ============================================================================

@value
struct SimulationContext:
    """Context containing all data needed for evaluation."""
    var termination_reason: TerminationReason
    var messages: List[MessageInfo]
    var tool_calls: List[ToolCallInfo]
    
    # Task evaluation criteria
    var has_evaluation_criteria: Bool
    var golden_actions: List[Action]
    var communicate_info: List[String]
    var nl_assertions: List[String]
    var env_assertions: List[EnvAssertion]
    var reward_basis: List[RewardType]
    
    # Environment states
    var predicted_env_state: EnvironmentState
    var gold_env_state: EnvironmentState
    var env_assertion_results: List[Bool]
    
    # NL evaluation results (pre-computed via LLM)
    var nl_evaluation_results: List[NLEvaluationResult]
    
    fn __init__(out self):
        self.termination_reason = TerminationReason.AGENT_STOP
        self.messages = List[MessageInfo]()
        self.tool_calls = List[ToolCallInfo]()
        self.has_evaluation_criteria = False
        self.golden_actions = List[Action]()
        self.communicate_info = List[String]()
        self.nl_assertions = List[String]()
        self.env_assertions = List[EnvAssertion]()
        self.reward_basis = List[RewardType]()
        self.predicted_env_state = EnvironmentState()
        self.gold_env_state = EnvironmentState()
        self.env_assertion_results = List[Bool]()
        self.nl_evaluation_results = List[NLEvaluationResult]()


# ============================================================================
# Main Evaluation Function
# ============================================================================

fn evaluate_simulation(
    ctx: SimulationContext,
    evaluation_type: EvaluationType
) -> RewardInfo:
    """
    Evaluate a simulation based on the evaluation type.
    
    Args:
        ctx: Simulation context with all evaluation data
        evaluation_type: Type of evaluation to perform
        
    Returns:
        RewardInfo with evaluation results
    """
    # Check for premature termination
    if ctx.termination_reason == TerminationReason.TOO_MANY_ERRORS or \
       ctx.termination_reason == TerminationReason.MAX_STEPS:
        var reward_info = RewardInfo(0.0)
        return reward_info.with_note(
            "Simulation terminated prematurely. Termination reason: " + ctx.termination_reason.value
        )
    
    # Check for no evaluation criteria
    if not ctx.has_evaluation_criteria:
        var reward_info = RewardInfo(1.0)
        return reward_info.with_note("No evaluation criteria")
    
    # Perform evaluation based on type
    if evaluation_type == EvaluationType.ENV:
        return _evaluate_env(ctx)
    elif evaluation_type == EvaluationType.NL_ASSERTIONS:
        return _evaluate_nl_assertions(ctx)
    elif evaluation_type == EvaluationType.COMMUNICATE:
        return _evaluate_communicate(ctx)
    elif evaluation_type == EvaluationType.ACTION:
        return _evaluate_action(ctx)
    elif evaluation_type == EvaluationType.ALL:
        return _evaluate_all(ctx)
    else:
        var reward_info = RewardInfo(0.0)
        return reward_info.with_note("Unknown evaluation type: " + evaluation_type.value)


fn _evaluate_env(ctx: SimulationContext) -> RewardInfo:
    """Evaluate environment state."""
    var eval_ctx = EvaluationContext()
    eval_ctx.predicted_state = ctx.predicted_env_state
    eval_ctx.gold_state = ctx.gold_env_state
    eval_ctx.env_assertions = ctx.env_assertions
    eval_ctx.env_assertion_results = ctx.env_assertion_results
    eval_ctx.reward_basis = ctx.reward_basis
    return EnvironmentEvaluator.calculate_reward(eval_ctx)


fn _evaluate_nl_assertions(ctx: SimulationContext) -> RewardInfo:
    """Evaluate NL assertions."""
    return NLAssertionsEvaluator.calculate_reward(
        ctx.nl_assertions,
        ctx.nl_evaluation_results
    )


fn _evaluate_communicate(ctx: SimulationContext) -> RewardInfo:
    """Evaluate communicate info."""
    return CommunicateEvaluator.calculate_reward(
        ctx.communicate_info,
        ctx.messages
    )


fn _evaluate_action(ctx: SimulationContext) -> RewardInfo:
    """Evaluate actions."""
    return ActionEvaluator.calculate_reward(
        ctx.golden_actions,
        ctx.tool_calls
    )


fn _evaluate_all(ctx: SimulationContext) -> RewardInfo:
    """
    Evaluate all criteria and combine rewards.

    Combines environment, action, communicate, and NL assertion evaluations
    based on the task's reward basis.
    """
    # Get individual evaluation results
    var env_reward_info = _evaluate_env(ctx)
    var action_reward_info = _evaluate_action(ctx)
    var communicate_reward_info = _evaluate_communicate(ctx)
    var nl_reward_info = _evaluate_nl_assertions(ctx)

    # Combine rewards based on reward basis
    var reward = 1.0
    var reward_breakdown = Dict[String, Float64]()

    # Check which reward types are in the basis
    var env_bases = List[RewardType]()
    env_bases.append(RewardType.DB)
    env_bases.append(RewardType.ENV_ASSERTION)

    var action_bases = List[RewardType]()
    action_bases.append(RewardType.ACTION)

    var nl_bases = List[RewardType]()
    nl_bases.append(RewardType.NL_ASSERTION)

    var comm_bases = List[RewardType]()
    comm_bases.append(RewardType.COMMUNICATE)

    # Check and apply env bases
    var has_env_basis = False
    for i in range(len(ctx.reward_basis)):
        for j in range(len(env_bases)):
            if ctx.reward_basis[i] == env_bases[j]:
                has_env_basis = True
                break
        if has_env_basis:
            break

    if has_env_basis:
        # Copy env reward breakdown
        for key in env_reward_info.reward_breakdown.keys():
            reward_breakdown[key] = env_reward_info.reward_breakdown[key]
        reward *= env_reward_info.reward

    # Check and apply action bases
    var has_action_basis = False
    for i in range(len(ctx.reward_basis)):
        if ctx.reward_basis[i] == RewardType.ACTION:
            has_action_basis = True
            break

    if has_action_basis:
        for key in action_reward_info.reward_breakdown.keys():
            reward_breakdown[key] = action_reward_info.reward_breakdown[key]
        reward *= action_reward_info.reward

    # Check and apply NL bases
    var has_nl_basis = False
    for i in range(len(ctx.reward_basis)):
        if ctx.reward_basis[i] == RewardType.NL_ASSERTION:
            has_nl_basis = True
            break

    if has_nl_basis:
        for key in nl_reward_info.reward_breakdown.keys():
            reward_breakdown[key] = nl_reward_info.reward_breakdown[key]
        reward *= nl_reward_info.reward

    # Check and apply communicate bases
    var has_comm_basis = False
    for i in range(len(ctx.reward_basis)):
        if ctx.reward_basis[i] == RewardType.COMMUNICATE:
            has_comm_basis = True
            break

    if has_comm_basis:
        for key in communicate_reward_info.reward_breakdown.keys():
            reward_breakdown[key] = communicate_reward_info.reward_breakdown[key]
        reward *= communicate_reward_info.reward

    # Build combined reward info
    var reward_info = RewardInfo(reward)

    # Set DB check from env evaluation
    if env_reward_info.has_db_check:
        reward_info = reward_info.with_db_check(env_reward_info.db_check)

    # Set all check lists
    reward_info.set_env_assertions(env_reward_info.env_assertions)
    reward_info.set_action_checks(action_reward_info.action_checks)
    reward_info.set_nl_assertions(nl_reward_info.nl_assertions)
    reward_info.set_communicate_checks(communicate_reward_info.communicate_checks)

    # Set reward basis and breakdown
    reward_info.reward_basis = ctx.reward_basis
    reward_info.reward_breakdown = reward_breakdown

    # Combine info dicts
    reward_info.info["env"] = "env evaluation complete"
    reward_info.info["nl"] = "nl evaluation complete"
    reward_info.info["communicate"] = "communicate evaluation complete"
    reward_info.info["action"] = "action evaluation complete"

    return reward_info


# ============================================================================
# Utility Functions
# ============================================================================

fn has_reward_type(reward_basis: List[RewardType], target: RewardType) -> Bool:
    """Check if a reward type is in the reward basis."""
    for i in range(len(reward_basis)):
        if reward_basis[i] == target:
            return True
    return False


fn create_premature_termination_reward(reason: TerminationReason) -> RewardInfo:
    """Create reward info for premature termination."""
    var reward_info = RewardInfo(0.0)
    return reward_info.with_note(
        "Simulation terminated prematurely. Termination reason: " + reason.value
    )


fn create_no_criteria_reward() -> RewardInfo:
    """Create reward info when no evaluation criteria."""
    var reward_info = RewardInfo(1.0)
    return reward_info.with_note("No evaluation criteria")

