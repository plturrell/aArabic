"""
Regression tests for the Mojo-based LLM_CALL shim.
Ensure deterministic behavior so toolorchestra workflows can rely on
stable responses during Python migration.
"""

from collections import List
import sys
sys.path.append("src/serviceCore/nOpenaiServer")

from tools.toolorchestra.LLM_CALL import (
    ChatMessage,
    ToolSchema,
    get_llm_response,
)


fn expect(condition: Bool, message: String) raises:
    if not condition:
        raise Error(message)


fn test_basic_string_input() raises:
    print("🧪 test_basic_string_input")
    var response = get_llm_response("unit-model", "Hello there!", temperature=0.5, max_length=128)
    expect(len(response.choices) == 1, "expected exactly one choice")
    var content = response.choices[0].message.content
    expect("unit-model" in content, "content should echo model id")
    expect("Hello there" in content, "content should echo user prompt")
    expect(response.usage.prompt_tokens > 0, "prompt token estimate should be > 0")
    expect(response.usage.completion_tokens > 0, "completion token estimate should be > 0")


fn test_structured_messages_and_tools() raises:
    print("🧪 test_structured_messages_and_tools")
    var messages = List[ChatMessage]()
    messages.append(ChatMessage(role="system", content="system prompt"))
    messages.append(ChatMessage(role="user", content="Use a tool"))

    var tools = List[ToolSchema]()
    tools.append(ToolSchema(name="search", description="run a search"))
    tools.append(ToolSchema(name="calculator", description="math helper"))

    var response = get_llm_response(
        model="tool-model",
        messages=messages,
        temperature=0.1,
        return_raw_response=true,
        tools=tools,
        max_length=256,
    )
    expect(len(response.choices) == 1, "expected one choice with tools request")
    var content = response.choices[0].message.content
    expect("tool-model" in content, "content should echo model id when tools supplied")
    expect("Available tools" in content, "content should mention tools list")


fn test_message_struct_conversion() raises:
    print("🧪 test_message_struct_conversion")
    var messages = List[ChatMessage]()
    messages.append(ChatMessage(role="user", content="Please respond"))
    messages.append(ChatMessage(role="assistant", content="Previous answer"))

    var response = get_llm_response(
        model="struct-model",
        messages=messages,
        temperature=0.9,
        return_raw_response=false,
        tools=List[ToolSchema](),
        max_length=64,
    )
    expect(len(response.choices) == 1, "struct-model should return one choice")
    expect(response.choices[0].finish_reason == "stop", "finish reason should default to stop")


fn main() raises:
    print("═══════════════════════════════════════════════════════════════════════")
    print("  toolorchestra LLM_CALL regression tests")
    print("═══════════════════════════════════════════════════════════════════════")

    test_basic_string_input()
    print("✅ basic string input passed")

    test_structured_messages_and_tools()
    print("✅ structured messages + tools passed")

    test_message_struct_conversion()
    print("✅ message struct conversion passed")

    print("🎉 All LLM_CALL Mojo tests passed")
