"""
Pure Mojo implementation of the legacy LLM_CALL helper.

This module intentionally avoids Python so toolorchestra can run inside the
Shimmy Mojo runtime without CPython bindings.  Instead of reaching out to
remote providers, we synthesize deterministic responses locally so higher
level orchestration logic (evaluation, training scripts, etc.) can be ported
incrementally without blocking on external inference.
"""

from collections import Dict, List
from math import max
from time import now

from mojo_sdk.runtime.startup import Env


# ============================================================================
# Data Models
# ============================================================================

@value
struct ToolSchema:
    name: String
    description: String
    input_schema: String

    fn __init__(inout self, name: String, description: String = "", input_schema: String = "{}"):
        self.name = name
        self.description = description
        self.input_schema = input_schema


@value
struct ToolCall:
    id: String
    name: String
    arguments: String

    fn __init__(inout self, id: String, name: String, arguments: String = "{}"):
        self.id = id
        self.name = name
        self.arguments = arguments


@value
struct ChatMessage:
    role: String
    content: String

    fn __init__(inout self, role: String, content: String):
        self.role = role
        self.content = content


@value
struct Usage:
    prompt_tokens: Int
    completion_tokens: Int

    fn __init__(inout self, prompt_tokens: Int = 0, completion_tokens: Int = 0):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

    fn total(self) -> Int:
        return self.prompt_tokens + self.completion_tokens


@value
struct Choice:
    message: ChatMessage
    finish_reason: String
    tool_calls: List[ToolCall]

    fn __init__(
        inout self,
        message: ChatMessage,
        finish_reason: String = "stop",
        tool_calls: List[ToolCall] = List[ToolCall]()
    ):
        self.message = message
        self.finish_reason = finish_reason
        self.tool_calls = tool_calls


@value
struct LLMResponse:
    choices: List[Choice]
    usage: Usage

    fn __init__(inout self):
        self.choices = List[Choice]()
        self.usage = Usage()


@value
struct LLMRequest:
    model: String
    messages: List[ChatMessage]
    temperature: Float32
    max_tokens: Int
    tools: List[ToolSchema]

    fn __init__(
        inout self,
        model: String,
        messages: List[ChatMessage],
        temperature: Float32 = 0.7,
        max_tokens: Int = 1024,
        tools: List[ToolSchema] = List[ToolSchema]()
    ):
        self.model = model
        self.messages = messages
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools


# ============================================================================
# Helpers
# ============================================================================

fn convert_openai_tools_to_claude(tools: List[ToolSchema]) -> List[ToolSchema]:
    """Compatibility shim so existing orchestration logic can keep the same call."""
    return tools


fn normalize_messages_for_tools(messages: List[ChatMessage], tools: List[ToolSchema]) -> List[ChatMessage]:
    # For the mock generator we only need to guarantee role/content ordering.
    # Future extensions can reintroduce the detailed normalization logic.
    _ = tools
    return messages


fn default_model_id() -> String:
    if (Env.get("SHIMMY_TOOL_MODEL")) |value| {
        return String(value);
    }
    return "local-mock-model"


fn estimate_tokens(text: String) -> Int:
    return max(1, Int(len(text) / 4))


fn synthesize_reply(request: LLMRequest) -> String:
    if len(request.messages) == 0:
        return "[toolorchestra] No input provided."

    let last = request.messages[len(request.messages) - 1]
    var response = String("[")
    response += request.model
    response += "] "
    response += "Echoing "
    response += last.role
    response += " request: "
    response += last.content
    response += "\n\n"
    response += "Temperature="
    response += String(request.temperature)
    response += ", max_tokens="
    response += String(request.max_tokens)
    if len(request.tools) > 0:
        response += "\nAvailable tools: "
        for i in range(len(request.tools)):
            response += request.tools[i].name
            if i < len(request.tools) - 1:
                response += ", "
    return response


# ============================================================================
# Generator
# ============================================================================

struct ToolorchestraGenerator:
    var default_model: String

    fn __init__(inout self, default_model: String):
        self.default_model = default_model

    fn handle(self, request: LLMRequest) -> LLMResponse:
        var normalized = normalize_messages_for_tools(request.messages, request.tools)
        var effective_request = LLMRequest(
            model=request.model if request.model != "" else self.default_model,
            messages=normalized,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            tools=request.tools
        )

        var reply = synthesize_reply(effective_request)

        var response = LLMResponse()
        response.choices.append(
            Choice(
                message=ChatMessage(role="assistant", content=reply),
                finish_reason="stop",
                tool_calls=List[ToolCall]()
            )
        )

        var usage = Usage()
        var prompt_join = String("")
        for message in effective_request.messages:
            prompt_join += message.content
        usage.prompt_tokens = estimate_tokens(prompt_join)
        usage.completion_tokens = estimate_tokens(reply)
        response.usage = usage
        return response


var GLOBAL_GENERATOR = ToolorchestraGenerator(default_model=default_model_id())


# ============================================================================
# Public API (matches legacy signature pattern as closely as possible)
# ============================================================================

fn make_message_list(input: String) -> List[ChatMessage]:
    var messages = List[ChatMessage]()
    messages.append(ChatMessage(role="user", content=input))
    return messages


fn get_llm_response(
    model: String,
    messages: String,
    temperature: Float32 = 1.0,
    return_raw_response: Bool = true,
    tools: List[ToolSchema] = List[ToolSchema](),
    max_length: Int = 1024,
) -> LLMResponse:
    _ = return_raw_response
    let request = LLMRequest(
        model=model,
        messages=make_message_list(messages),
        temperature=temperature,
        max_tokens=max_length,
        tools=tools
    )
    return GLOBAL_GENERATOR.handle(request)


fn get_llm_response(
    model: String,
    messages: List[ChatMessage],
    temperature: Float32,
    return_raw_response: Bool,
    tools: List[ToolSchema],
    max_length: Int,
) -> LLMResponse:
    _ = return_raw_response
    let request = LLMRequest(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_length,
        tools=tools
    )
    return GLOBAL_GENERATOR.handle(request)


fn get_llm_response(
    model: String,
    messages: List[ChatMessage],
    temperature: Float32,
    return_raw_response: Bool,
) -> LLMResponse:
    return get_llm_response(
        model=model,
        messages=messages,
        temperature=temperature,
        return_raw_response=return_raw_response,
        tools=List[ToolSchema](),
        max_length=1024
    )


fn get_llm_response(model: String, messages: String) -> LLMResponse:
    return get_llm_response(
        model=model,
        messages=messages,
        temperature=1.0,
        return_raw_response=true,
        tools=List[ToolSchema](),
        max_length=1024
    )
