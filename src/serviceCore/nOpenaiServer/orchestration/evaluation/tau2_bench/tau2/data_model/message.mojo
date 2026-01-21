# tau2/data_model/message.mojo
# Migrated from tau2/data_model/message.py

from collections import List, Optional
from tau2.utils.utils import get_now

# Role type aliases
alias SystemRole = String
alias UserRole = String
alias AssistantRole = String
alias ToolRole = String
alias ToolRequestor = String

struct ToolCall:
    """A tool call made by an agent."""
    var id: String
    var name: String
    var arguments: String  # JSON string representation
    var requestor: String
    
    fn __init__(
        inout self,
        id: String = "",
        name: String = "",
        arguments: String = "{}",
        requestor: String = "assistant"
    ):
        self.id = id
        self.name = name
        self.arguments = arguments
        self.requestor = requestor
    
    fn to_string(self) -> String:
        var result = "ToolCall (from " + self.requestor + ")\n"
        if self.id != "":
            result += "id: " + self.id + "\n"
        result += "name: " + self.name + "\n"
        result += "arguments: " + self.arguments
        return result


struct SystemMessage:
    """A system message."""
    var role: String
    var content: String
    var turn_idx: Int
    var timestamp: String
    
    fn __init__(
        inout self,
        content: String = "",
        turn_idx: Int = -1,
        timestamp: String = ""
    ):
        self.role = "system"
        self.content = content
        self.turn_idx = turn_idx
        self.timestamp = timestamp if timestamp != "" else get_now()
    
    fn to_string(self) -> String:
        var result = "SystemMessage\n"
        if self.turn_idx >= 0:
            result += "turn_idx: " + String(self.turn_idx) + "\n"
        result += "timestamp: " + self.timestamp + "\n"
        if self.content != "":
            result += "content: " + self.content
        return result


struct AssistantMessage:
    """A message from the assistant."""
    var role: String
    var content: String
    var tool_calls: List[ToolCall]
    var turn_idx: Int
    var timestamp: String
    var cost: Float64
    var usage_prompt_tokens: Int
    var usage_completion_tokens: Int
    var raw_data: String  # JSON string
    
    fn __init__(
        inout self,
        content: String = "",
        tool_calls: List[ToolCall] = List[ToolCall](),
        turn_idx: Int = -1,
        timestamp: String = "",
        cost: Float64 = 0.0,
        usage_prompt_tokens: Int = 0,
        usage_completion_tokens: Int = 0,
        raw_data: String = "{}"
    ):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls
        self.turn_idx = turn_idx
        self.timestamp = timestamp if timestamp != "" else get_now()
        self.cost = cost
        self.usage_prompt_tokens = usage_prompt_tokens
        self.usage_completion_tokens = usage_completion_tokens
        self.raw_data = raw_data
    
    fn has_text_content(self) -> Bool:
        """Check if the message has text content."""
        if self.content == "":
            return False
        # Check if content is all whitespace
        var has_non_space = False
        for i in range(len(self.content)):
            if self.content[i] != " " and self.content[i] != "\t" and self.content[i] != "\n":
                has_non_space = True
                break
        return has_non_space
    
    fn is_tool_call(self) -> Bool:
        """Check if the message is a tool call."""
        return len(self.tool_calls) > 0
    
    fn to_string(self) -> String:
        var result = "AssistantMessage\n"
        if self.turn_idx >= 0:
            result += "turn_idx: " + String(self.turn_idx) + "\n"
        result += "timestamp: " + self.timestamp + "\n"
        if self.content != "":
            result += "content: " + self.content + "\n"
        if len(self.tool_calls) > 0:
            result += "ToolCalls:\n"
            for i in range(len(self.tool_calls)):
                result += self.tool_calls[i].to_string() + "\n"
        if self.cost > 0:
            result += "cost: " + String(self.cost)
        return result


struct UserMessage:
    """A message from the user."""
    var role: String
    var content: String
    var tool_calls: List[ToolCall]
    var turn_idx: Int
    var timestamp: String
    var cost: Float64
    var usage_prompt_tokens: Int
    var usage_completion_tokens: Int
    var raw_data: String
    
    fn __init__(
        inout self,
        content: String = "",
        tool_calls: List[ToolCall] = List[ToolCall](),
        turn_idx: Int = -1,
        timestamp: String = "",
        cost: Float64 = 0.0,
        usage_prompt_tokens: Int = 0,
        usage_completion_tokens: Int = 0,
        raw_data: String = "{}"
    ):
        self.role = "user"
        self.content = content
        self.tool_calls = tool_calls
        self.turn_idx = turn_idx
        self.timestamp = timestamp if timestamp != "" else get_now()
        self.cost = cost
        self.usage_prompt_tokens = usage_prompt_tokens
        self.usage_completion_tokens = usage_completion_tokens
        self.raw_data = raw_data
    
    fn has_text_content(self) -> Bool:
        """Check if the message has text content."""
        if self.content == "":
            return False
        var has_non_space = False
        for i in range(len(self.content)):
            if self.content[i] != " " and self.content[i] != "\t" and self.content[i] != "\n":
                has_non_space = True
                break
        return has_non_space
    
    fn is_tool_call(self) -> Bool:
        """Check if the message is a tool call."""
        return len(self.tool_calls) > 0
    
    fn to_string(self) -> String:
        var result = "UserMessage\n"
        if self.turn_idx >= 0:
            result += "turn_idx: " + String(self.turn_idx) + "\n"
        result += "timestamp: " + self.timestamp + "\n"
        if self.content != "":
            result += "content: " + self.content + "\n"
        if len(self.tool_calls) > 0:
            result += "ToolCalls:\n"
            for i in range(len(self.tool_calls)):
                result += self.tool_calls[i].to_string() + "\n"
        if self.cost > 0:
            result += "cost: " + String(self.cost)
        return result


struct ToolMessage:
    """A message from a tool execution."""
    var id: String
    var role: String
    var content: String
    var requestor: String
    var error: Bool
    var turn_idx: Int
    var timestamp: String
    
    fn __init__(
        inout self,
        id: String,
        content: String = "",
        requestor: String = "assistant",
        error: Bool = False,
        turn_idx: Int = -1,
        timestamp: String = ""
    ):
        self.id = id
        self.role = "tool"
        self.content = content
        self.requestor = requestor
        self.error = error
        self.turn_idx = turn_idx
        self.timestamp = timestamp if timestamp != "" else get_now()
    
    fn to_string(self) -> String:
        var result = "ToolMessage (responding to " + self.requestor + ")\n"
        if self.turn_idx >= 0:
            result += "turn_idx: " + String(self.turn_idx) + "\n"
        result += "timestamp: " + self.timestamp + "\n"
        if self.content != "":
            result += "content: " + self.content + "\n"
        if self.error:
            result += "Error"
        return result


# Message type union representation
# In Mojo, we can use a variant or tagged union pattern
struct MessageType:
    """Enum-like structure for message types."""
    alias SYSTEM = 0
    alias ASSISTANT = 1
    alias USER = 2
    alias TOOL = 3
