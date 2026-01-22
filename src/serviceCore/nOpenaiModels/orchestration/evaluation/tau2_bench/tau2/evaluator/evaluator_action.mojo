"""
Mojo port of tau2 action evaluator.
Evaluates whether the agent performed required actions.
Migrated from evaluator_action.py to pure Mojo.
"""

from collections import Dict, List
from .evaluator_base import (
    RewardInfo,
    RewardType,
    ActionCheck,
    Evaluator,
)


# ============================================================================
# Action Structure
# ============================================================================

@value
struct Action:
    """An expected action to be performed."""
    var action_id: String
    var name: String
    var requestor: String  # "user" or "assistant"
    var arguments: Dict[String, String]
    var compare_args: List[String]
    
    fn __init__(out self, action_id: String = "", name: String = "", requestor: String = "assistant"):
        self.action_id = action_id
        self.name = name
        self.requestor = requestor
        self.arguments = Dict[String, String]()
        self.compare_args = List[String]()
    
    fn compare_with_tool_call(self, tool_name: String, tool_args: Dict[String, String]) -> Bool:
        """Compare this action with a tool call."""
        if self.name != tool_name:
            return False
        
        # If no compare_args specified, check all arguments
        if len(self.compare_args) == 0:
            # Check all arguments match
            for key in self.arguments.keys():
                if key not in tool_args:
                    return False
                if self.arguments[key] != tool_args[key]:
                    return False
            return True
        
        # Only compare specified args
        for i in range(len(self.compare_args)):
            var key = self.compare_args[i]
            if key not in self.arguments:
                continue
            if key not in tool_args:
                return False
            if self.arguments[key] != tool_args[key]:
                return False
        
        return True


# ============================================================================
# Tool Call Structure (simplified for evaluation)
# ============================================================================

@value
struct ToolCallInfo:
    """Information about a tool call for evaluation."""
    var id: String
    var name: String
    var arguments: Dict[String, String]
    var requestor: String
    
    fn __init__(out self, id: String = "", name: String = "", requestor: String = "assistant"):
        self.id = id
        self.name = name
        self.requestor = requestor
        self.arguments = Dict[String, String]()


# ============================================================================
# Action Evaluator
# ============================================================================

struct ActionEvaluator(Evaluator):
    """
    Evaluates whether the agent performed the required actions.
    """
    
    @staticmethod
    fn calculate_reward(
        golden_actions: List[Action],
        predicted_tool_calls: List[ToolCallInfo]
    ) -> RewardInfo:
        """
        Calculate reward based on whether the agent performed required actions.
        
        Args:
            golden_actions: List of expected actions
            predicted_tool_calls: List of tool calls from trajectory
            
        Returns:
            RewardInfo with action evaluation results
        """
        # Handle empty golden actions
        if len(golden_actions) == 0:
            var reward_info = RewardInfo(1.0)
            reward_info.add_reward_breakdown(RewardType.ACTION, 1.0)
            return reward_info.with_note("No actions to evaluate")
        
        # Evaluate each golden action
        var action_checks = ActionEvaluator.evaluate_actions(
            golden_actions,
            predicted_tool_calls
        )
        
        # Calculate reward: 1 if all actions matched, 0 otherwise
        var all_matched = True
        for i in range(len(action_checks)):
            if not action_checks[i].action_match:
                all_matched = False
                break
        
        var reward = 1.0 if all_matched else 0.0
        
        var reward_info = RewardInfo(reward)
        reward_info.set_action_checks(action_checks)
        reward_info.add_reward_breakdown(RewardType.ACTION, reward)
        
        return reward_info
    
    @staticmethod
    fn evaluate_actions(
        golden_actions: List[Action],
        predicted_tool_calls: List[ToolCallInfo]
    ) -> List[ActionCheck]:
        """
        Evaluate whether predicted tool calls match golden actions.
        
        Args:
            golden_actions: Expected actions
            predicted_tool_calls: Actual tool calls from trajectory
            
        Returns:
            List of ActionCheck results
        """
        var checks = List[ActionCheck]()
        
        for i in range(len(golden_actions)):
            var gold_action = golden_actions[i]
            var found = False
            
            # Check if this golden action was performed
            for j in range(len(predicted_tool_calls)):
                var pred_call = predicted_tool_calls[j]
                if gold_action.compare_with_tool_call(pred_call.name, pred_call.arguments):
                    found = True
                    break
            
            var action_reward = 1.0 if found else 0.0
            checks.append(ActionCheck(
                action_name=gold_action.name,
                action_id=gold_action.action_id,
                action_match=found,
                action_reward=action_reward
            ))
        
        return checks

