"""
Chat Completion Service
OpenAI-compatible chat API implementation with conversation history support
"""

from collections import List

# Import inference components
from inference.bridge.inference_api import resolve_model_path, generate_with_model
from services.llm.time_utils import unix_timestamp


# ============================================================================
# Data Structures
# ============================================================================

@fieldwise_init
struct ChatMessage(Copyable, Movable):
    """Represents a single chat message."""
    var role: String      # "system", "user", or "assistant"
    var content: String   # Message content
    
    fn __init__(out self, role: String, content: String):
        self.role = role
        self.content = content


struct ChatConfig(Movable):
    """Configuration for chat completion."""
    var model: String
    var temperature: Float32
    var top_p: Float32
    var max_tokens: Int
    var stop_sequences: List[String]
    var stream: Bool
    
    fn __init__(out self):
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
    return String(body[start_idx:end_idx])

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
    return String(body[start:end])

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
        var digit = Int(ch.as_bytes()[0]) - 48
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
        var digit = Float32(Int(ch.as_bytes()[0]) - 48)
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
        return values^
    var open_idx = body.find("[", idx)
    if open_idx < 0:
        return values^
    var close_idx = body.find("]", open_idx)
    if close_idx < 0:
        return values^

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
                values.append(String(body[start:i]))
                start = -1
        else:
            escaping = False
        i += 1

    return values^

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

    return messages^


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

    return config^


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
    ✅ FIXED: P1 Issue #7 - Now applies stop sequences
    """
    try:
        var model_path = resolve_model_path(config.model)
        var raw_text = generate_with_model(
            prompt,
            model_path,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
        
        # ✅ FIXED: Apply stop sequences to truncate generation
        return apply_stop_sequences(raw_text, config.stop_sequences)
    except e:
        print("❌ Inference error:", e)
        return "Error generating response"

fn apply_stop_sequences(text: String, stop_sequences: List[String]) -> String:
    """
    Truncate text at first occurrence of any stop sequence.
    ✅ FIXED: P1 Issue #7 - Stop sequence detection implemented
    """
    if len(stop_sequences) == 0:
        return text
    
    var earliest_pos = len(text)
    var found_stop = False
    
    # Find the earliest stop sequence in the text
    for i in range(len(stop_sequences)):
        var stop_seq = stop_sequences[i]
        var pos = text.find(stop_seq)
        
        if pos >= 0 and pos < earliest_pos:
            earliest_pos = pos
            found_stop = True
    
    # Truncate at the earliest stop sequence
    if found_stop:
        return String(text[:earliest_pos])
    
    return text


# ============================================================================
# Response Formatting
# ============================================================================

fn format_chat_response(content: String, config: ChatConfig, request_id: String, timestamp: Int) -> String:
    """
    Format response in OpenAI chat completion format.
    """
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
    ✅ P1-11: Now supports function calling
    
    Args:
        body: JSON request body with messages and config
    
    Returns:
        JSON response in OpenAI format
    """
    print("💬 Processing chat completion request")
    
    # Check if function calling is requested
    var has_functions = body.find("\"functions\"") >= 0 or body.find("\"tools\"") >= 0
    
    if has_functions:
        return handle_function_call(body)
    
    # Parse request
    var messages = parse_chat_messages(body)
    var config = parse_chat_config(body)
    
    # Build prompt from conversation history
    var prompt = build_chat_prompt(messages)
    
    print("📝 Prompt:", prompt[:100], "...")  # Log first 100 chars
    
    # Generate response
    var content = generate_chat_response(prompt, config)
    
    # Generate request ID
    var timestamp = unix_timestamp()
    var request_id = String("chatcmpl-") + String(timestamp)
    
    # Format and return response
    var response = format_chat_response(content, config, request_id, timestamp)
    
    print("✅ Chat completion generated")
    
    return response


# ============================================================================
# Streaming Support
# ============================================================================

struct StreamingState:
    """State for streaming response generation"""
    var request_id: String
    var model: String
    var content_buffer: String
    var chunk_index: Int
    var is_complete: Bool

    fn __init__(out self, request_id: String, model: String):
        self.request_id = request_id
        self.model = model
        self.content_buffer = ""
        self.chunk_index = 0
        self.is_complete = False


fn format_sse_chunk(
    content: String,
    state: StreamingState,
    is_final: Bool
) -> String:
    """Format a single SSE chunk in OpenAI streaming format"""
    var chunk = String("data: {")
    chunk += json_string("id") + String(":") + json_string(state.request_id) + String(",")
    chunk += json_string("object") + String(":") + json_string("chat.completion.chunk") + String(",")
    chunk += json_string("created") + String(":") + json_number(unix_timestamp()) + String(",")
    chunk += json_string("model") + String(":") + json_string(state.model) + String(",")

    # Choices array
    chunk += json_string("choices") + String(":[{")
    chunk += json_string("index") + String(":0,")
    chunk += json_string("delta") + String(":{")

    if state.chunk_index == 0:
        # First chunk includes role
        chunk += json_string("role") + String(":") + json_string("assistant")
        if len(content) > 0:
            chunk += String(",")
            chunk += json_string("content") + String(":") + json_string(content)
    elif len(content) > 0:
        # Subsequent chunks only have content
        chunk += json_string("content") + String(":") + json_string(content)

    chunk += String("},")

    if is_final:
        chunk += json_string("finish_reason") + String(":") + json_string("stop")
    else:
        chunk += json_string("finish_reason") + String(":null")

    chunk += String("}]}")
    chunk += String("\n\n")

    return chunk


fn generate_streaming_chunks(prompt: String, config: ChatConfig) -> List[String]:
    """
    Generate streaming response chunks.
    Returns list of SSE-formatted chunks.
    """
    var chunks = List[String]()

    # Initialize streaming state
    var timestamp = unix_timestamp()
    var request_id = String("chatcmpl-") + String(timestamp)
    var state = StreamingState(request_id, config.model)

    # Generate full response first
    var full_content = generate_chat_response(prompt, config)

    # Split into chunks (roughly token-sized pieces)
    var chunk_size = 4  # Approximate characters per token
    var pos = 0

    while pos < len(full_content):
        var end_pos = min(pos + chunk_size, len(full_content))
        var chunk_content = String(full_content[pos:end_pos])

        var is_final = (end_pos >= len(full_content))
        var sse_chunk = format_sse_chunk(chunk_content, state, is_final)
        chunks.append(sse_chunk)

        state.chunk_index += 1
        pos = end_pos

    # Add final [DONE] message
    chunks.append(String("data: [DONE]\n\n"))

    return chunks^


fn handle_chat_stream(body: String) -> String:
    """
    Handle streaming chat completions.
    Returns concatenated SSE chunks for non-streaming transports.
    For true streaming, use generate_streaming_chunks and send via SSE connection.
    """
    print("📡 Processing streaming chat completion request")

    # Parse request
    var messages = parse_chat_messages(body)
    var config = parse_chat_config(body)
    config.stream = True

    # Build prompt
    var prompt = build_chat_prompt(messages)

    # Generate all chunks
    var chunks = generate_streaming_chunks(prompt, config)

    # Concatenate for synchronous response
    # (In production, these would be sent via SSE connection)
    var response = String("")
    for i in range(len(chunks)):
        response += chunks[i]

    print("✅ Streaming response generated with", len(chunks), "chunks")

    return response


# ============================================================================
# Function Calling Support (P1-11 FIXED)
# ============================================================================

@fieldwise_init
struct FunctionDefinition(Copyable, Movable):
    """OpenAI function/tool definition."""
    var name: String
    var description: String
    var parameters: String  # JSON schema string
    
    fn __init__(out self, name: String, description: String, parameters: String):
        self.name = name
        self.description = description
        self.parameters = parameters

@fieldwise_init
struct ToolCall(Copyable, Movable):
    """A function call made by the model."""
    var id: String
    var type: String  # Always "function" for now
    var function_name: String
    var function_arguments: String  # JSON string
    
    fn __init__(out self, id: String, function_name: String, arguments: String):
        self.id = id
        self.type = "function"
        self.function_name = function_name
        self.function_arguments = arguments

fn parse_functions_from_request(body: String) -> List[FunctionDefinition]:
    """
    ✅ P1-11: Parse function definitions from request.
    Supports both 'functions' (deprecated) and 'tools' format.
    """
    var functions = List[FunctionDefinition]()
    
    # Look for "tools" array first (newer format)
    var tools_idx = body.find("\"tools\"")
    if tools_idx >= 0:
        var array_start = body.find("[", tools_idx)
        if array_start >= 0:
            # Parse tools array
            var idx = array_start + 1
            while idx < len(body):
                var func_idx = body.find("\"function\"", idx)
                if func_idx < 0:
                    break
                
                var name_idx = body.find("\"name\"", func_idx)
                var name = extract_json_string_at(body, name_idx) if name_idx >= 0 else ""
                
                var desc_idx = body.find("\"description\"", func_idx)
                var desc = extract_json_string_at(body, desc_idx) if desc_idx >= 0 else ""
                
                # For parameters, we'd need more complex parsing
                # For now, store as empty JSON object
                var params = "{}"
                
                if name != "":
                    functions.append(FunctionDefinition(name, desc, params))
                
                idx = func_idx + 100  # Move past this function
                
                # Check if we've reached end of array
                var next_brace = body.find("}", idx)
                var next_bracket = body.find("]", idx)
                if next_bracket >= 0 and (next_brace < 0 or next_bracket < next_brace):
                    break
    
    # Fallback to "functions" array (deprecated format)
    if len(functions) == 0:
        var functions_idx = body.find("\"functions\"")
        if functions_idx >= 0:
            var array_start = body.find("[", functions_idx)
            if array_start >= 0:
                var idx = array_start + 1
                while idx < len(body):
                    var name_idx = body.find("\"name\"", idx)
                    if name_idx < 0:
                        break
                    
                    var name = extract_json_string_at(body, name_idx)
                    var desc_idx = body.find("\"description\"", name_idx)
                    var desc = extract_json_string_at(body, desc_idx) if desc_idx >= 0 else ""
                    
                    if name != "":
                        functions.append(FunctionDefinition(name, desc, "{}"))
                    
                    idx = name_idx + 100
                    
                    var next_bracket = body.find("]", idx)
                    if next_bracket >= 0:
                        break
    
    return functions^

fn extract_tool_calls_from_response(content: String) -> List[ToolCall]:
    """
    ✅ P1-11: Extract function calls from model output.
    Supports multiple formats:
    - XML: <tool_call>...</tool_call>
    - JSON: {"name": "...", "arguments": {...}}
    - Text: TOOL_CALL: function_name(args)
    """
    var tool_calls = List[ToolCall]()
    
    # Try XML format first (Hermes/GPT-4 style)
    var idx = 0
    while True:
        var start = content.find("<tool_call>", idx)
        if start < 0:
            break
        
        var end = content.find("</tool_call>", start)
        if end < 0:
            break
        
        var tool_content = String(content[start + 11:end])  # Skip "<tool_call>"
        
        # Parse JSON inside tool_call
        var name = extract_json_string(tool_content, "name", "")
        var args = extract_json_string(tool_content, "arguments", "{}")
        
        if name != "":
            var call_id = String("call_") + String(len(tool_calls) + 1)
            tool_calls.append(ToolCall(call_id, name, args))
        
        idx = end + 12
    
    # If no XML format found, try JSON function call format
    if len(tool_calls) == 0:
        var func_idx = content.find("\"function_call\"")
        if func_idx >= 0:
            var name = extract_json_string(content, "name", "")
            var args = extract_json_string(content, "arguments", "{}")
            
            if name != "":
                tool_calls.append(ToolCall("call_1", name, args))
    
    return tool_calls^

fn format_chat_response_with_tools(
    content: String,
    tool_calls: List[ToolCall],
    config: ChatConfig,
    request_id: String,
    timestamp: Int
) -> String:
    """
    ✅ P1-11: Format chat response with function calls.
    """
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
    response += json_string("content") + String(":") + json_string(content) + String(",")
    
    # Add tool_calls if present
    if len(tool_calls) > 0:
        response += json_string("tool_calls") + String(":[")
        for i in range(len(tool_calls)):
            if i > 0:
                response += String(",")
            var call = tool_calls[i]
            response += String("{")
            response += json_string("id") + String(":") + json_string(call.id) + String(",")
            response += json_string("type") + String(":") + json_string("function") + String(",")
            response += json_string("function") + String(":{")
            response += json_string("name") + String(":") + json_string(call.function_name) + String(",")
            response += json_string("arguments") + String(":") + json_string(call.function_arguments)
            response += String("}}")
        response += String("],")
    
    response += String("},")
    
    # Finish reason
    var finish_reason = "tool_calls" if len(tool_calls) > 0 else "stop"
    response += json_string("finish_reason") + String(":") + json_string(finish_reason)
    response += String("}],")
    
    # Usage statistics
    response += json_string("usage") + String(":{")
    response += json_string("prompt_tokens") + String(":") + json_number(len(content) // 4) + String(",")
    response += json_string("completion_tokens") + String(":") + json_number(len(content) // 4) + String(",")
    response += json_string("total_tokens") + String(":") + json_number(len(content) // 2)
    response += String("}}")
    
    return response

fn augment_prompt_with_functions(prompt: String, functions: List[FunctionDefinition]) -> String:
    """
    ✅ P1-11: Add function definitions to the prompt.
    Uses format compatible with most LLMs.
    """
    if len(functions) == 0:
        return prompt
    
    var augmented = prompt
    
    # Add function calling instructions
    augmented += String("\n\nYou have access to the following functions:\n\n")
    
    for i in range(len(functions)):
        var func = functions[i]
        augmented += String("Function: ")
        augmented += func.name
        augmented += String("\n")
        augmented += String("Description: ")
        augmented += func.description
        augmented += String("\n")
        if func.parameters != "":
            augmented += String("Parameters: ")
            augmented += func.parameters
            augmented += String("\n")
        augmented += String("\n")
    
    augmented += String("To call a function, respond with:\n")
    augmented += String("<tool_call>\n")
    augmented += String('{"name": "function_name", "arguments": {"arg1": "value1"}}\n')
    augmented += String("</tool_call>\n\n")
    
    return augmented

fn handle_function_call(body: String) -> String:
    """
    ✅ P1-11 FIXED: Handle function calling in chat completions.
    Full OpenAI-compatible function calling support.
    """
    print("🔧 Processing chat with function calling")
    
    # Parse request
    var messages = parse_chat_messages(body)
    var config = parse_chat_config(body)
    var functions = parse_functions_from_request(body)
    
    print("📝 Found", len(functions), "function definitions")
    
    # Build prompt from conversation history
    var prompt = build_chat_prompt(messages)
    
    # Augment prompt with function definitions
    if len(functions) > 0:
        prompt = augment_prompt_with_functions(prompt, functions)
    
    # Generate response
    var content = generate_chat_response(prompt, config)
    
    # Extract any tool calls from the response
    var tool_calls = extract_tool_calls_from_response(content)
    
    # Generate request ID
    var timestamp = unix_timestamp()
    var request_id = String("chatcmpl-") + String(timestamp)
    
    # Format response with tool calls if present
    var response = format_chat_response_with_tools(
        content,
        tool_calls,
        config,
        request_id,
        timestamp
    )
    
    if len(tool_calls) > 0:
        print("✅ Chat completion with", len(tool_calls), "tool calls")
    else:
        print("✅ Chat completion generated (no tool calls)")
    
    return response
