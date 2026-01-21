# tau2/agent/base.mojo
# Migrated from tau2/agent/base.py
# Base agent class for TAU2-Bench

from collections import List, Optional
from tau2.data_model.message import (
    AssistantMessage,
    UserMessage,
    ToolMessage,
)

# Agent state is represented as a string (JSON serialized state)
alias AgentState = String

struct ValidAgentInputMessage:
    """Valid input messages for agents: UserMessage or ToolMessage."""
    var message_type: String  # "user" or "tool"
    var content: String
    var tool_calls: List[String]  # JSON representation
    var turn_idx: Int
    
    fn __init__(inout self):
        self.message_type = ""
        self.content = ""
        self.tool_calls = List[String]()
        self.turn_idx = -1

fn is_valid_agent_history_message(message_type: String, has_tool_calls: Bool, requestor: String) -> Bool:
    """
    Check if the message is a valid agent history message.
    
    Args:
        message_type: Type of message ("assistant", "user", "tool")
        has_tool_calls: Whether message has tool calls
        requestor: Who requested the tool call
        
    Returns:
        True if valid agent history message
    """
    if message_type == "assistant":
        return True
    if message_type == "user" and not has_tool_calls:
        return True
    if message_type == "tool" and requestor == "assistant":
        return True
    return False

trait BaseAgent:
    """
    Base agent interface that defines the common methods for all agents.
    
    All agent implementations must provide:
    - generate_next_message: Generate next response from input
    - get_init_state: Get initial agent state
    - is_stop: Check if conversation should stop
    - set_seed: Set random seed for reproducibility
    """
    
    fn generate_next_message(
        inout self,
        message: ValidAgentInputMessage,
        state: String
    ) raises -> (AssistantMessage, String):
        """
        Generate the next message from a user/tool message and agent state.
        
        Args:
            message: The user message or tool message
            state: The current agent state
            
        Returns:
            Tuple of (assistant_message, new_state)
        """
        ...
    
    fn get_init_state(
        inout self,
        message_history: List[String]
    ) raises -> String:
        """
        Get the initial state of the agent.
        Required to rerun agent from any point in conversation.
        
        Args:
            message_history: The message history (JSON strings)
            
        Returns:
            Initial agent state
        """
        ...
    
    fn is_stop(self, message: AssistantMessage) -> Bool:
        """
        Check if the message is a stop message.
        By default the agent does not stop.
        
        Args:
            message: The assistant message to check
            
        Returns:
            True if conversation should stop
        """
        return False
    
    fn set_seed(inout self, seed: Int):
        """
        Set the seed for the agent for reproducibility.
        
        Args:
            seed: Random seed value
        """
        print("Warning: Setting seed not implemented for this agent")

struct LocalAgent:
    """
    Local agent implementation.
    
    Agent developers should implement:
    - generate_next_message: Generate next message (text or tool call)
    - get_init_state: Get initial state (optional, for conversation replay)
    """
    var tools: List[String]  # JSON representation of tools
    var domain_policy: String
    var seed: Int
    
    fn __init__(inout self, tools: List[String], domain_policy: String):
        """
        Initialize local agent with tools and domain policy.
        
        Args:
            tools: Available tools (as JSON strings)
            domain_policy: Domain-specific policy/instructions
        """
        self.tools = tools
        self.domain_policy = domain_policy
        self.seed = -1
    
    fn generate_next_message(
        inout self,
        message: ValidAgentInputMessage,
        state: String
    ) raises -> (AssistantMessage, String):
        """
        Generate next message based on input and state.
        
        Args:
            message: Input message from user or tool
            state: Current agent state
            
        Returns:
            Tuple of (response_message, updated_state)
        """
        # TODO: Implement using LLM_CALL.mojo
        var response = AssistantMessage(
            content="Agent response placeholder",
            turn_idx=message.turn_idx + 1
        )
        return (response, state)
    
    fn get_init_state(
        inout self,
        message_history: List[String]
    ) raises -> String:
        """
        Get initial agent state from message history.
        
        Args:
            message_history: Previous messages
            
        Returns:
            Initial state (JSON string)
        """
        return "{}"
    
    fn set_seed(inout self, seed: Int):
        """Set random seed for reproducibility."""
        self.seed = seed
