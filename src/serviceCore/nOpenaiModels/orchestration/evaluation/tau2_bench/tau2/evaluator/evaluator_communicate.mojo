"""
Mojo port of tau2 communicate evaluator.
Evaluates whether the agent communicated required information.
Migrated from evaluator_communicate.py to pure Mojo.
"""

from collections import Dict, List
from .evaluator_base import (
    RewardInfo,
    RewardType,
    CommunicateCheck,
    Evaluator,
)


# ============================================================================
# Message Structure for Evaluation
# ============================================================================

@value
struct MessageInfo:
    """Message information for evaluation."""
    var role: String
    var content: String
    var is_text_content: Bool
    
    fn __init__(out self, role: String = "", content: String = ""):
        self.role = role
        self.content = content
        self.is_text_content = len(content.strip()) > 0
    
    fn has_text_content(self) -> Bool:
        """Check if message has text content."""
        return self.is_text_content and len(self.content.strip()) > 0


# ============================================================================
# String Utilities for Matching
# ============================================================================

fn to_lower(s: String) -> String:
    """Convert string to lowercase."""
    var result = String()
    for i in range(len(s)):
        var c = s[i]
        var code = ord(c)
        # ASCII A-Z range: 65-90
        if code >= 65 and code <= 90:
            result += chr(code + 32)
        else:
            result += c
    return result


fn remove_commas(s: String) -> String:
    """Remove commas from string."""
    var result = String()
    for i in range(len(s)):
        if s[i] != ",":
            result += s[i]
    return result


fn contains_substring(haystack: String, needle: String) -> Bool:
    """Check if haystack contains needle (case-insensitive, commas removed)."""
    var lower_haystack = to_lower(remove_commas(haystack))
    var lower_needle = to_lower(needle)
    
    var needle_len = len(lower_needle)
    var haystack_len = len(lower_haystack)
    
    if needle_len == 0:
        return True
    if needle_len > haystack_len:
        return False
    
    for i in range(haystack_len - needle_len + 1):
        var found = True
        for j in range(needle_len):
            if lower_haystack[i + j] != lower_needle[j]:
                found = False
                break
        if found:
            return True
    
    return False


# ============================================================================
# Communicate Evaluator
# ============================================================================

struct CommunicateEvaluator(Evaluator):
    """
    Evaluates whether the agent communicated required information.
    """
    
    @staticmethod
    fn calculate_reward(
        communicate_info: List[String],
        messages: List[MessageInfo]
    ) -> RewardInfo:
        """
        Calculate reward based on whether required info was communicated.
        
        Args:
            communicate_info: List of required information strings
            messages: List of messages from trajectory
            
        Returns:
            RewardInfo with communicate evaluation results
        """
        # Handle empty communicate_info
        if len(communicate_info) == 0:
            var reward_info = RewardInfo(1.0)
            reward_info.add_reward_breakdown(RewardType.COMMUNICATE, 1.0)
            return reward_info.with_note("No communicate_info to evaluate")
        
        # Evaluate each required piece of information
        var communicate_checks = CommunicateEvaluator.evaluate_communicate_info(
            communicate_info,
            messages
        )
        
        # Calculate reward: 1 if all expectations met, 0 otherwise
        var all_met = True
        for i in range(len(communicate_checks)):
            if not communicate_checks[i].met:
                all_met = False
                break
        
        var reward = 1.0 if all_met else 0.0
        
        var reward_info = RewardInfo(reward)
        reward_info.set_communicate_checks(communicate_checks)
        reward_info.add_reward_breakdown(RewardType.COMMUNICATE, reward)
        
        return reward_info
    
    @staticmethod
    fn evaluate_communicate_info(
        communicate_info: List[String],
        messages: List[MessageInfo]
    ) -> List[CommunicateCheck]:
        """
        Evaluate whether required information was communicated.
        
        Args:
            communicate_info: Required info strings
            messages: Trajectory messages
            
        Returns:
            List of CommunicateCheck results
        """
        var checks = List[CommunicateCheck]()
        
        for i in range(len(communicate_info)):
            var info_str = communicate_info[i]
            var found = False
            var found_content = String()
            
            # Search for this info in assistant messages
            for j in range(len(messages)):
                var message = messages[j]
                # Only check assistant messages with text content
                if message.role != "assistant":
                    continue
                if not message.has_text_content():
                    continue
                
                if contains_substring(message.content, info_str):
                    found = True
                    found_content = message.content
                    break
            
            var justification: String
            if found:
                justification = "Information '" + info_str + "' communicated in the message: '" + found_content + "'"
            else:
                justification = "Information '" + info_str + "' not communicated."
            
            checks.append(CommunicateCheck(
                info=info_str,
                met=found,
                justification=justification
            ))
        
        return checks

