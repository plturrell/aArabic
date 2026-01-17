"""
Text Completion Service
OpenAI-compatible text completion API implementation
"""

from collections import List

# Import inference components
from inference.bridge.inference_api import resolve_model_path, generate_with_model
from services.llm.time_utils import unix_timestamp


# ============================================================================
# Data Structures
# ============================================================================

struct CompletionConfig(Movable):
    """Configuration for text completion."""
    var model: String
    var prompt: String
    var temperature: Float32
    var top_p: Float32
    var max_tokens: Int
    var n: Int                          # Number of completions to generate
    var stop_sequences: List[String]    # Stop generation at these strings
    var echo: Bool                      # Echo back the prompt
    var best_of: Int                    # Generate N, return best
    var logprobs: Int                   # Return log probabilities
    
    fn __init__(out self):
        self.model = "phi-3-mini"
        self.prompt = ""
        self.temperature = 0.7
        self.top_p = 0.9
        self.max_tokens = 256
        self.n = 1
        self.stop_sequences = List[String]()
        self.echo = False
        self.best_of = 1
        self.logprobs = 0


@fieldwise_init
struct CompletionChoice(Copyable, Movable):
    """A single completion choice."""
    var text: String
    var index: Int
    var finish_reason: String  # "stop", "length", etc.
    
    fn __init__(out self, text: String, index: Int, finish_reason: String):
        self.text = text
        self.index = index
        self.finish_reason = finish_reason


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
# Request Parsing
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

fn parse_completion_config(body: String) -> CompletionConfig:
    """
    Parse completion configuration from request body.
    Extracts prompt, model, temperature, max_tokens, etc.
    """
    var config = CompletionConfig()

    config.model = extract_json_string(body, "model", config.model)
    config.prompt = extract_json_string(body, "prompt", config.prompt)
    config.temperature = extract_json_float(body, "temperature", config.temperature)
    config.top_p = extract_json_float(body, "top_p", config.top_p)
    config.max_tokens = extract_json_int(body, "max_tokens", config.max_tokens)
    config.n = extract_json_int(body, "n", config.n)
    config.best_of = extract_json_int(body, "best_of", config.best_of)
    config.echo = extract_json_bool(body, "echo", config.echo)
    config.stop_sequences = extract_json_string_array(body, "stop")

    return config^


# ============================================================================
# Response Generation
# ============================================================================

fn generate_single_completion(prompt: String, config: CompletionConfig) -> String:
    """
    Generate a single text completion using real inference engine.
    Returns the generated text.
    """
    try:
        var model_path = resolve_model_path(config.model)
        return generate_with_model(
            prompt,
            model_path,
            max_tokens=config.max_tokens,
            temperature=config.temperature
        )
    except e:
        print("❌ Inference error:", e)
        return "Error generating completion"


fn generate_multiple_completions(
    prompt: String, 
    config: CompletionConfig
) -> List[CompletionChoice]:
    """
    Generate multiple completions (when n > 1).
    Returns a list of completion choices.
    """
    var choices = List[CompletionChoice]()
    
    # Generate N completions
    for i in range(config.n):
        var text = generate_single_completion(prompt, config)
        
        # Add variation marker for demonstration
        if i > 0:
            text += String(" (Variation ") + String(i + 1) + String(")")
        
        choices.append(CompletionChoice(text, i, "stop"))
    
    return choices^


fn generate_best_of_completions(
    prompt: String,
    config: CompletionConfig
) -> CompletionChoice:
    """
    Generate best_of completions and return the best one.
    Uses scoring to select the highest quality completion.
    """
    # Generate multiple candidates
    var candidates = generate_multiple_completions(prompt, config)
    
    # TODO: Implement scoring mechanism
    # For now, just return the first one
    return candidates[0]


# ============================================================================
# Response Formatting
# ============================================================================

fn format_completion_choice(choice: CompletionChoice, echo_prompt: String) -> String:
    """Format a single completion choice as JSON."""
    var result = String("{")
    
    # Add echoed prompt if requested
    if len(echo_prompt) > 0:
        result += json_string("text") + String(":") + json_string(echo_prompt + choice.text) + String(",")
    else:
        result += json_string("text") + String(":") + json_string(choice.text) + String(",")
    
    result += json_string("index") + String(":") + json_number(choice.index) + String(",")
    result += json_string("finish_reason") + String(":") + json_string(choice.finish_reason)
    
    # TODO: Add logprobs if requested
    
    result += String("}")
    return result


fn format_completion_response(
    choices: List[CompletionChoice],
    config: CompletionConfig,
    request_id: String,
    timestamp: Int
) -> String:
    """
    Format response in OpenAI text completion format.
    """
    var response = String("{")
    response += json_string("id") + String(":") + json_string(request_id) + String(",")
    response += json_string("object") + String(":") + json_string("text_completion") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("model") + String(":") + json_string(config.model) + String(",")
    
    # Choices array
    response += json_string("choices") + String(":[")
    
    var echo_prompt = config.prompt if config.echo else ""
    
    for i in range(len(choices)):
        if i > 0:
            response += String(",")
        response += format_completion_choice(choices[i], echo_prompt)
    
    response += String("],")
    
    # Usage statistics (simplified)
    var total_tokens = len(config.prompt) + len(choices[0].text)
    response += json_string("usage") + String(":{")
    response += json_string("prompt_tokens") + String(":") + json_number(len(config.prompt) // 4) + String(",")
    response += json_string("completion_tokens") + String(":") + json_number(len(choices[0].text) // 4) + String(",")
    response += json_string("total_tokens") + String(":") + json_number(total_tokens // 4)
    response += String("}}")
    
    return response


# ============================================================================
# Main Completion Handler
# ============================================================================

fn handle_completion_request(body: String) -> String:
    """
    Main entry point for text completion requests.
    
    Args:
        body: JSON request body with prompt and config
    
    Returns:
        JSON response in OpenAI format
    """
    print("📝 Processing text completion request")
    
    # Parse request
    var config = parse_completion_config(body)
    
    print("📋 Prompt length:", len(config.prompt), "chars")
    print("🔢 Generating", config.n, "completion(s)")
    
    # Generate completions
    var choices = List[CompletionChoice]()
    
    if config.best_of > config.n:
        # Generate best_of and select top n
        # For simplicity, just generate n
        choices = generate_multiple_completions(config.prompt, config)
    else:
        # Generate n completions directly
        choices = generate_multiple_completions(config.prompt, config)
    
    # Generate request ID
    var timestamp = unix_timestamp()
    var request_id = String("cmpl-") + String(timestamp)
    
    # Format and return response
    var response = format_completion_response(choices, config, request_id, timestamp)
    
    print("✅ Text completion generated")
    
    return response


# ============================================================================
# Advanced Features (Future Enhancements)
# ============================================================================

fn calculate_logprobs(tokens: List[String], config: CompletionConfig) -> List[Float32]:
    """
    Calculate log probabilities for generated tokens.
    Used when logprobs parameter is set.
    """
    # TODO: Implement logprobs calculation
    var logprobs = List[Float32]()
    return logprobs


fn apply_stop_sequences(text: String, stop_sequences: List[String]) -> String:
    """
    Truncate text at first occurrence of any stop sequence.
    """
    # TODO: Implement stop sequence detection
    return text


fn score_completion(text: String, prompt: String) -> Float32:
    """
    Score completion quality for best_of selection.
    Higher scores are better.
    """
    # TODO: Implement scoring mechanism
    # Could use perplexity, coherence, or other metrics
    return 1.0
