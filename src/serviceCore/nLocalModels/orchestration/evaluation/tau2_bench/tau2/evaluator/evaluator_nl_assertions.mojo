"""
Mojo port of tau2 NL assertions evaluator.
Evaluates whether trajectories adhere to natural-language assertions.
Migrated from evaluator_nl_assertions.py to pure Mojo.

Note: The original Python implementation uses LLM calls for evaluation.
This Mojo version provides the structure and requires external LLM integration.
"""

from collections import Dict, List
from .evaluator_base import (
    RewardInfo,
    RewardType,
    NLAssertionCheck,
    Evaluator,
)
from .evaluator_communicate import MessageInfo


# ============================================================================
# NL Evaluation Request/Response Structures
# ============================================================================

@value
struct NLEvaluationRequest:
    """Request structure for NL assertion evaluation."""
    var trajectory_str: String
    var nl_assertions: List[String]
    var system_prompt: String
    var user_prompt: String
    
    fn __init__(out self):
        self.trajectory_str = String()
        self.nl_assertions = List[String]()
        self.system_prompt = String()
        self.user_prompt = String()


@value
struct NLEvaluationResult:
    """Result from NL assertion evaluation."""
    var expected_outcome: String
    var met: Bool
    var reasoning: String
    
    fn __init__(out self, expected_outcome: String = "", met: Bool = False, reasoning: String = ""):
        self.expected_outcome = expected_outcome
        self.met = met
        self.reasoning = reasoning


# ============================================================================
# NL Assertions Evaluator
# ============================================================================

struct NLAssertionsEvaluator(Evaluator):
    """
    Evaluates whether a trajectory adheres to natural-language assertions.
    Uses LLM to evaluate the assertions against conversation history.
    """
    
    @staticmethod
    fn calculate_reward(
        nl_assertions: List[String],
        nl_results: List[NLEvaluationResult]
    ) -> RewardInfo:
        """
        Calculate reward based on NL assertion evaluation results.
        
        Args:
            nl_assertions: List of natural language assertions
            nl_results: Pre-computed evaluation results from LLM
            
        Returns:
            RewardInfo with NL assertion evaluation results
        """
        # Handle empty assertions
        if len(nl_assertions) == 0:
            var reward_info = RewardInfo(1.0)
            reward_info.add_reward_breakdown(RewardType.NL_ASSERTION, 1.0)
            return reward_info.with_note("No nl_assertions to evaluate")
        
        # Convert results to checks
        var nl_checks = List[NLAssertionCheck]()
        for i in range(len(nl_results)):
            var result = nl_results[i]
            nl_checks.append(NLAssertionCheck(
                nl_assertion=result.expected_outcome,
                met=result.met,
                justification=result.reasoning
            ))
        
        # Calculate reward: 1 if all expectations met, 0 otherwise
        var all_met = True
        for i in range(len(nl_checks)):
            if not nl_checks[i].met:
                all_met = False
                break
        
        var reward = 1.0 if all_met else 0.0
        
        var reward_info = RewardInfo(reward)
        reward_info.set_nl_assertions(nl_checks)
        reward_info.add_reward_breakdown(RewardType.NL_ASSERTION, reward)
        
        return reward_info
    
    @staticmethod
    fn build_evaluation_request(
        messages: List[MessageInfo],
        nl_assertions: List[String]
    ) -> NLEvaluationRequest:
        """
        Build the evaluation request for LLM.
        
        Args:
            messages: List of trajectory messages
            nl_assertions: List of NL assertions to evaluate
            
        Returns:
            NLEvaluationRequest ready to be sent to LLM
        """
        # Build trajectory string
        var trajectory_str = String()
        for i in range(len(messages)):
            var msg = messages[i]
            if i > 0:
                trajectory_str += "\n"
            trajectory_str += msg.role + ": " + msg.content
        
        # Build system prompt
        var system_prompt = """
        TASK
        - You will be given a list of expected outcomes and a conversation that was collected during a test case run.
        - The conversation is between an agent and a customer.
        - Your job is to evaluate whether the agent satisfies each of the expected outcomes.
        - Grade each expected outcome individually.

        FORMAT
        - Your response should be a JSON object with the following fields:
        - `reasoning`: a short explanation for your classification
        - `metExpectation`: `true` if the agent satisfies the expected outcomes, `false` otherwise
        - `expectedOutcome`: repeat the expectation from the input that you are grading
        
        Example response structure:
        {
            "results": [
                {
                    "expectedOutcome": "<one of the expected outcomes from the input>",
                    "reasoning": "<reasoning trace>",
                    "metExpectation": <false or true>,
                }
            ]
        }
        """
        
        # Build user prompt  
        var user_prompt = String()
        user_prompt += "conversation:\n"
        user_prompt += trajectory_str
        user_prompt += "\n\nexpectedOutcomes:\n"
        for i in range(len(nl_assertions)):
            if i > 0:
                user_prompt += ", "
            user_prompt += nl_assertions[i]
        
        var request = NLEvaluationRequest()
        request.trajectory_str = trajectory_str
        request.nl_assertions = nl_assertions
        request.system_prompt = system_prompt
        request.user_prompt = user_prompt
        
        return request

