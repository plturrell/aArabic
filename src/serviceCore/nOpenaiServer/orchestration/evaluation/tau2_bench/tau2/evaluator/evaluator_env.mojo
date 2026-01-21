"""
Mojo port of tau2 environment evaluator.
Evaluates the end-state of the simulation environment.
Migrated from evaluator_env.py to pure Mojo.
"""

from collections import Dict, List
from .evaluator_base import (
    RewardInfo,
    RewardType,
    DBCheck,
    EnvAssertionCheck,
    Evaluator,
)
from .evaluator_action import Action, ToolCallInfo


# ============================================================================
# Environment State Structure
# ============================================================================

@value
struct EnvironmentState:
    """Represents the state of an environment for comparison."""
    var db_hash: String
    var user_db_hash: String
    
    fn __init__(out self, db_hash: String = "", user_db_hash: String = ""):
        self.db_hash = db_hash
        self.user_db_hash = user_db_hash
    
    fn matches(self, other: EnvironmentState) -> Bool:
        """Check if two environment states match."""
        return self.db_hash == other.db_hash and self.user_db_hash == other.user_db_hash


# ============================================================================
# Environment Assertion Structure
# ============================================================================

@value
struct EnvAssertion:
    """An environment assertion to check."""
    var assertion_id: String
    var assertion_type: String
    var expression: String
    
    fn __init__(out self, assertion_id: String = "", assertion_type: String = "", expression: String = ""):
        self.assertion_id = assertion_id
        self.assertion_type = assertion_type
        self.expression = expression


# ============================================================================
# Evaluation Context
# ============================================================================

@value
struct EvaluationContext:
    """Context for environment evaluation."""
    var predicted_state: EnvironmentState
    var gold_state: EnvironmentState
    var env_assertions: List[EnvAssertion]
    var env_assertion_results: List[Bool]  # Results of assertion checks
    var reward_basis: List[RewardType]
    
    fn __init__(out self):
        self.predicted_state = EnvironmentState()
        self.gold_state = EnvironmentState()
        self.env_assertions = List[EnvAssertion]()
        self.env_assertion_results = List[Bool]()
        self.reward_basis = List[RewardType]()


# ============================================================================
# Environment Evaluator
# ============================================================================

struct EnvironmentEvaluator(Evaluator):
    """
    Evaluates the end-state of the simulation environment.
    Compares predicted environment with expected (gold) environment.
    """
    
    @staticmethod
    fn calculate_reward(ctx: EvaluationContext) -> RewardInfo:
        """
        Calculate reward based on environment state comparison.
        
        Args:
            ctx: Evaluation context with predicted and gold states
            
        Returns:
            RewardInfo with environment evaluation results
        """
        # Compare database states
        var agent_db_match = ctx.predicted_state.db_hash == ctx.gold_state.db_hash
        var user_db_match = ctx.predicted_state.user_db_hash == ctx.gold_state.user_db_hash
        
        var db_match = agent_db_match and user_db_match
        var db_reward = 1.0 if db_match else 0.0
        
        var db_check = DBCheck(db_match=db_match, db_reward=db_reward)
        
        # Evaluate environment assertions
        var env_assertion_checks = List[EnvAssertionCheck]()
        var env_assertion_reward = 1.0
        
        for i in range(len(ctx.env_assertions)):
            var assertion = ctx.env_assertions[i]
            var success = False
            if i < len(ctx.env_assertion_results):
                success = ctx.env_assertion_results[i]
            
            var assertion_reward = 1.0 if success else 0.0
            env_assertion_checks.append(EnvAssertionCheck(
                env_assertion=assertion.expression,
                met=success,
                reward=assertion_reward
            ))
            env_assertion_reward *= assertion_reward
        
        # Calculate combined reward based on reward basis
        var reward = 1.0
        var reward_info = RewardInfo(reward)
        reward_info = reward_info.with_db_check(db_check)
        reward_info.set_env_assertions(env_assertion_checks)
        
        # Apply reward basis
        var has_db = False
        var has_env_assertion = False
        
        for i in range(len(ctx.reward_basis)):
            if ctx.reward_basis[i] == RewardType.DB:
                has_db = True
            elif ctx.reward_basis[i] == RewardType.ENV_ASSERTION:
                has_env_assertion = True
        
        if has_db:
            reward_info.add_reward_breakdown(RewardType.DB, db_reward)
            reward *= db_reward
        
        if has_env_assertion:
            reward_info.add_reward_breakdown(RewardType.ENV_ASSERTION, env_assertion_reward)
            reward *= env_assertion_reward
        
        reward_info.reward = reward
        
        # Set reward basis
        reward_info.reward_basis = ctx.reward_basis
        
        return reward_info
    
    @staticmethod
    fn calculate_reward_no_criteria() -> RewardInfo:
        """Return reward info for cases with no evaluation criteria."""
        var reward_info = RewardInfo(1.0)
        return reward_info.with_note("No evaluation criteria")
    
    @staticmethod
    fn calculate_reward_no_actions() -> RewardInfo:
        """Return reward info when no expected actions or env assertions."""
        var reward_info = RewardInfo(1.0)
        reward_info = reward_info.with_db_check(DBCheck(db_match=True, db_reward=1.0))
        return reward_info.with_note("No expected actions or env assertions")

