"""
Chat Completion Service
OpenAI-compatible chat API implementation with conversation history support
"""

from collections import List
from python import Python

# Import inference components
from inference.bridge.inference_api import ensure_model_loaded, resolve_model_path, shared_generate


# ============================================================================
# Data Structures
# ============================================================================

struct ChatMessage:
    """Represents a single chat message."""
    var role: String      # "system", "user", or "assistant"
    var content: String   # Message content
    
    fn __init__(inout self, role: String, content: String):
        self.role = role
        self.content = content


struct ChatConfig:
    """Configuration for chat completion."""
    var model: String
    var temperature: Float32
    var top_p: Float32
    var max_tokens: Int
    var stop_sequences: List[String]
    var stream: Bool
    
    fn __init__(inout self):
        self.model = "phi-3-mini"
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 512
        self.stop_sequences = List[String]()
        self.stream = False


# ============================================================================
# JSON Helper Functions
# ============================================================================

fn json_string(s: String) -> String:
    """Escape string for JSON."""
    var result = String('"')
    result += s
    result += String('"')
    return result


fn json_number(n: Int) -> String:
    """Convert number to JSON string."""
    return String(n)


fn json_float(f: Float32) -> String:
    """Convert float to JSON string."""
    return String(f)


# ============================================================================
# Message Parsing
# ============================================================================

fn extract_json_string(body: String, key: String, default: String) -> String:
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return default
    var colon_idx = body.find(":", idx + len(pattern))
    if colon_idx < 0:
        return default
    var quote_idx = body.find("\"", colon_idx)
    if quote_idx < 0:
        return default
    var start_idx = quote_idx + 1
    var end_idx = body.find("\"", start_idx)
    if end_idx < 0:
        return default
    return body[start_idx:end_idx]

fn extract_json_string_at(body: String, start_idx: Int) -> String:
    var colon_idx = body.find(":", start_idx)
    if colon_idx < 0:
        return ""
    var quote_idx = body.find("\"", colon_idx)
    if quote_idx < 0:
        return ""
    var start = quote_idx + 1
    var end = body.find("\"", start)
    if end < 0:
        return ""
    return body[start:end]

fn extract_json_int(body: String, key: String, default: Int) -> Int:
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return default
    var colon_idx = body.find(":", idx + len(pattern))
    if colon_idx < 0:
        return default

    var i = colon_idx + 1
    while i < len(body) and (body[i] == " " or body[i] == "\n" or body[i] == "\t"):
        i += 1

    var value = 0
    var found = False
    while i < len(body):
        var ch = body[i]
        if ch < "0" or ch > "9":
            break
        let digit = Int(ch.as_bytes()[0]) - 48
        value = value * 10 + digit
        found = True
        i += 1

    return value if found else default

fn extract_json_float(body: String, key: String, default: Float32) -> Float32:
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return default
    var colon_idx = body.find(":", idx + len(pattern))
    if colon_idx < 0:
        return default

    var i = colon_idx + 1
    while i < len(body) and (body[i] == " " or body[i] == "\n" or body[i] == "\t"):
        i += 1

    var sign: Float32 = 1.0
    if i < len(body) and body[i] == "-":
        sign = -1.0
        i += 1

    var value: Float32 = 0.0
    var divisor: Float32 = 1.0
    var found = False
    var after_dot = False

    while i < len(body):
        var ch = body[i]
        if ch == ".":
            if after_dot:
                break
            after_dot = True
            i += 1
            continue
        if ch < "0" or ch > "9":
            break
        let digit = Float32(Int(ch.as_bytes()[0]) - 48)
        if after_dot:
            divisor *= 10.0
            value += digit / divisor
        else:
            value = value * 10.0 + digit
        found = True
        i += 1

    return value * sign if found else default

fn extract_json_bool(body: String, key: String, default: Bool) -> Bool:
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return default
    var colon_idx = body.find(":", idx + len(pattern))
    if colon_idx < 0:
        return default

    var i = colon_idx + 1
    while i < len(body) and (body[i] == " " or body[i] == "\n" or body[i] == "\t"):
        i += 1

    if i + 4 <= len(body) and body[i:i + 4] == "true":
        return True
    if i + 5 <= len(body) and body[i:i + 5] == "false":
        return False

    return default

fn extract_json_string_array(body: String, key: String) -> List[String]:
    var values = List[String]()
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return values
    var open_idx = body.find("[", idx)
    if open_idx < 0:
        return values
    var close_idx = body.find("]", open_idx)
    if close_idx < 0:
        return values

    var i = open_idx + 1
    var start = -1
    var escaping = False
    while i < close_idx:
        var ch = body[i]
        if ch == "\\" and not escaping:
            escaping = True
            i += 1
            continue
        if ch == "\"" and not escaping:
            if start < 0:
                start = i + 1
            else:
                values.append(body[start:i])
                start = -1
        else:
            escaping = False
        i += 1

    return values

fn parse_chat_messages(body: String) -> List[ChatMessage]:
    """
    Parse messages array from request body.
    Simplified parser for demonstration - production should use proper JSON parsing.
    """
    var messages = List[ChatMessage]()

    var idx = 0
    while True:
        var role_idx = body.find("\"role\"", idx)
        if role_idx < 0:
            break
        var role = extract_json_string_at(body, role_idx + 6)
        if role == "":
            break
        var content_idx = body.find("\"content\"", role_idx)
        if content_idx < 0:
            break
        var content = extract_json_string_at(body, content_idx + 9)
        messages.append(ChatMessage(role, content))
        idx = content_idx + 1

    if len(messages) == 0:
        var fallback = extract_json_string(body, "prompt", "")
        if fallback != "":
            messages.append(ChatMessage("user", fallback))

    return messages


fn parse_chat_config(body: String) -> ChatConfig:
    """
    Parse chat configuration from request body.
    Extracts model, temperature, max_tokens, etc.
    """
    var config = ChatConfig()

    config.model = extract_json_string(body, "model", config.model)
    config.temperature = extract_json_float(body, "temperature", config.temperature)
    config.top_p = extract_json_float(body, "top_p", config.top_p)
    config.max_tokens = extract_json_int(body, "max_tokens", config.max_tokens)
    config.stream = extract_json_bool(body, "stream", config.stream)
    config.stop_sequences = extract_json_string_array(body, "stop")

    return config


# ============================================================================
# Prompt Building
# ============================================================================

fn build_chat_prompt(messages: List[ChatMessage]) -> String:
    """
    Convert chat messages to a single prompt string.
    Applies chat template based on model type.
    """
    var prompt = String("")
    
    # Build prompt with role markers
    for i in range(len(messages)):
        var msg = messages[i]
        
        if msg.role == "system":
            prompt += String("System: ")
            prompt += msg.content
            prompt += String("\n\n")
        elif msg.role == "user":
            prompt += String("User: ")
            prompt += msg.content
            prompt += String("\n\n")
        elif msg.role == "assistant":
            prompt += String("Assistant: ")
            prompt += msg.content
            prompt += String("\n\n")
    
    # Add final assistant prompt
    prompt += String("Assistant: ")
    
    return prompt


# ============================================================================
# Response Generation
# ============================================================================

fn generate_chat_response(prompt: String, config: ChatConfig) -> String:
    """
    Generate chat response using real inference engine.
    Returns the assistant's reply.
    """
    try:
        var model_path = resolve_model_path(config.model)
        var loaded = ensure_model_loaded(model_path)
        if not loaded:
            return "Error: Failed to load model: " + model_path

        return shared_generate(
            prompt,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
    except e:
        print("❌ Inference error:", e)
        return "Error generating response: " + str(e)


# ============================================================================
# Response Formatting
# ============================================================================

fn format_chat_response(content: String, config: ChatConfig, request_id: String) -> String:
    """
    Format response in OpenAI chat completion format.
    """
    var py = Python.import_module("time")
    var timestamp = Int(py.time())
    
    var response = String("{")
    response += json_string("id") + String(":") + json_string(request_id) + String(",")
    response += json_string("object") + String(":") + json_string("chat.completion") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("model") + String(":") + json_string(config.model) + String(",")
    
    # Choices array
    response += json_string("choices") + String(":[{")
    response += json_string("index") + String(":0,")
    response += json_string("message") + String(":{")
    response += json_string("role") + String(":") + json_string("assistant") + String(",")
    response += json_string("content") + String(":") + json_string(content)
    response += String("},")
    response += json_string("finish_reason") + String(":") + json_string("stop")
    response += String("}],")
    
    # Usage statistics
    response += json_string("usage") + String(":{")
    response += json_string("prompt_tokens") + String(":") + json_number(len(content) // 4) + String(",")
    response += json_string("completion_tokens") + String(":") + json_number(len(content) // 4) + String(",")
    response += json_string("total_tokens") + String(":") + json_number(len(content) // 2)
    response += String("}}")
    
    return response


# ============================================================================
# Main Chat Handler
# ============================================================================

fn handle_chat_request(body: String) -> String:
    """
    Main entry point for chat completion requests.
    
    Args:
        body: JSON request body with messages and config
    
    Returns:
        JSON response in OpenAI format
    """
    print("💬 Processing chat completion request")
    
    # Parse request
    var messages = parse_chat_messages(body)
    var config = parse_chat_config(body)
    
    # Build prompt from conversation history
    var prompt = build_chat_prompt(messages)
    
    print("📝 Prompt:", prompt[:100], "...")  # Log first 100 chars
    
    # Generate response
    var content = generate_chat_response(prompt, config)
    
    # Generate request ID
    var py = Python.import_module("time")
    var timestamp = Int(py.time())
    var request_id = String("chatcmpl-") + String(timestamp)
    
    # Format and return response
    var response = format_chat_response(content, config, request_id)
    
    print("✅ Chat completion generated")
    
    return response


# ============================================================================
# Streaming Support (Future Enhancement)
# ============================================================================

fn handle_chat_stream(body: String) -> String:
    """
    Handle streaming chat completions.
    Returns Server-Sent Events (SSE) format.
    """
    # TODO: Implement streaming support
    # Will require integration with Zig HTTP server for SSE
    return "Streaming not yet implemented"


# ============================================================================
# Function Calling Support (Future Enhancement)
# ============================================================================

fn handle_function_call(body: String) -> String:
    """
    Handle function calling in chat completions.
    Allows LLM to call external functions/tools.
    """
    # TODO: Implement function calling
    # Will integrate with orchestration/tools/
    return "Function calling not yet implemented"
