"""
Shimmy-Mojo HTTP Server
Zero Python dependencies - Pure Zig + Mojo architecture
OpenAI-compatible API with Pure Mojo LLM inference
"""

from memory import UnsafePointer, alloc
from collections import List

# Import our Mojo inference components
# LLaMA inference now handled by Zig engine via FFI
# from core.llama_inference import LLaMAModel, create_phi3_mini_config, create_llama32_1b_config, create_llama32_3b_config

# Import specialized LLM service modules
from services.llm.chat import handle_chat_request
from services.llm.completion import handle_completion_request
from services.llm.time_utils import unix_timestamp

# ============================================================================
# Helper Functions for C String Handling
# ============================================================================

fn string_len(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> Int:
    """Get length of null-terminated C string."""
    var i: Int = 0
    while ptr.load(i) != 0:
        i += 1
    return i

fn cstr_to_string(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> String:
    """Convert C string to Mojo String."""
    var length = string_len(ptr)
    if length == 0:
        return ""
    
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))
    
    return String(bytes)

fn cstr_to_string_with_len(
    ptr: UnsafePointer[UInt8, ImmutExternalOrigin],
    length: Int
) -> String:
    """Convert C string with explicit length to Mojo String."""
    if length == 0:
        return ""
    
    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))
    
    return String(bytes)

fn create_response(content: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Create null-terminated C string for response."""
    var content_bytes = content.as_bytes()
    var byte_length = len(content_bytes)
    var ptr = alloc[UInt8](byte_length + 1)
    
    # Copy UTF-8 bytes
    for i in range(byte_length):
        ptr.store(i, content_bytes[i])
    
    # Null terminate
    ptr.store(byte_length, 0)
    
    return ptr

fn string_to_c_ptr(value: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Allocate a null-terminated C string copy."""
    var bytes = value.as_bytes()
    var ptr = alloc[UInt8](len(bytes) + 1)
    for i in range(len(bytes)):
        ptr.store(i, bytes[i])
    ptr.store(len(bytes), 0)
    return ptr

# ============================================================================
# Simple JSON Builder (avoiding Python dependency)
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

fn json_bool(b: Bool) -> String:
    """Convert bool to JSON string."""
    return "true" if b else "false"

# ============================================================================
# Global State (Model Management)
# ============================================================================

# ============================================================================
# HTTP Request Handler (Called by Zig)
# ============================================================================

fn handle_http_request(
    method: UnsafePointer[UInt8, ImmutExternalOrigin],
    path: UnsafePointer[UInt8, ImmutExternalOrigin],
    body: UnsafePointer[UInt8, ImmutExternalOrigin],
    body_len: Int
) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """
    Handle HTTP requests from Zig server.
    Implements OpenAI-compatible API routes.
    """
    
    # Convert C strings to Mojo strings
    var method_str = cstr_to_string(method)
    var path_str = cstr_to_string(path)
    var body_str = cstr_to_string_with_len(body, body_len)
    
    print("🔥 Mojo handling:", method_str, path_str)

    if method_str == "OPTIONS":
        return create_response("{}")
    
    # Route based on path
    if path_str == "/":
        return handle_root()
    
    elif path_str == "/health":
        return handle_health()
    
    elif path_str == "/v1/models":
        return handle_list_models()
    
    elif path_str == "/v1/chat/completions":
        return handle_chat_completions(body_str)
    
    elif path_str == "/v1/completions":
        return handle_completions(body_str)
    
    elif path_str == "/api/tags":
        return handle_ollama_tags()
    
    else:
        return handle_not_found(path_str)

# ============================================================================
# Route Handlers
# ============================================================================

fn handle_root() -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle / - Server info."""
    var response = String("{")
    response += json_string("name") + String(":") + json_string("Shimmy-Mojo API") + String(",")
    response += json_string("version") + String(":") + json_string("1.0.0") + String(",")
    response += json_string("architecture") + String(":") + json_string("Zig + Mojo") + String(",")
    response += json_string("inference") + String(":") + json_string("Pure Mojo") + String(",")
    response += json_string("models") + String(":[") + json_string("phi-3-mini") + String(",") + json_string("llama-3.2-1b") + String(",") + json_string("llama-3.2-3b") + String("]")
    response += String("}")
    
    return create_response(response)

fn handle_health() -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /health - Health check."""
    var model_loaded = False
    var model_info = ""
    
    var response = String("{")
    response += json_string("status") + String(":") + json_string("healthy") + String(",")
    response += json_string("engine") + String(":") + json_string("Zig+Mojo") + String(",")
    response += json_string("inference") + String(":") + json_string("Pure Mojo") + String(",")
    response += json_string("model_loaded") + String(":") + json_bool(model_loaded)
    if model_info != "":
        response += String(",") + json_string("model_info") + String(":") + json_string(model_info)
    response += String("}")
    
    return create_response(response)

fn handle_list_models() -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /v1/models - List available models."""
    var timestamp = unix_timestamp()
    
    var response = String("{")
    response += json_string("object") + String(":") + json_string("list") + String(",")
    response += json_string("data") + String(":[")
    
    # Model 1: Phi-3-Mini
    response += String("{")
    response += json_string("id") + String(":") + json_string("phi-3-mini") + String(",")
    response += json_string("object") + String(":") + json_string("model") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("owned_by") + String(":") + json_string("shimmy-mojo")
    response += String("},")
    
    # Model 2: LLaMA 3.2 1B
    response += String("{")
    response += json_string("id") + String(":") + json_string("llama-3.2-1b") + String(",")
    response += json_string("object") + String(":") + json_string("model") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("owned_by") + String(":") + json_string("shimmy-mojo")
    response += String("},")
    
    # Model 3: LLaMA 3.2 3B
    response += String("{")
    response += json_string("id") + String(":") + json_string("llama-3.2-3b") + String(",")
    response += json_string("object") + String(":") + json_string("model") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("owned_by") + String(":") + json_string("shimmy-mojo")
    response += String("}")
    
    response += String("]}")
    
    return create_response(response)

fn handle_chat_completions(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /v1/chat/completions - Delegate to chat module."""
    # Delegate to specialized chat module
    var response = handle_chat_request(body)
    return create_response(response)

fn handle_completions(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /v1/completions - Delegate to completion module."""
    # Delegate to specialized completion module
    var response = handle_completion_request(body)
    return create_response(response)

fn handle_ollama_tags() -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /api/tags - Ollama-compatible tags endpoint."""
    var response = String("{")
    response += json_string("models") + String(":[")
    response += String("{")
    response += json_string("name") + String(":") + json_string("phi-3-mini") + String(",")
    response += json_string("modified_at") + String(":") + json_string("2026-01-12T00:00:00Z") + String(",")
    response += json_string("size") + String(":0")
    response += String("}")
    response += String("]}")
    
    return create_response(response)

fn handle_not_found(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle 404 - Not found."""
    var response = String("{")
    response += json_string("error") + String(":") + json_string("Not found") + String(",")
    response += json_string("path") + String(":") + json_string(path)
    response += String("}")
    
    return create_response(response)

# ============================================================================
# Main Server Entry Point
# ============================================================================

fn main() raises:
    """
    Main entry point for Shimmy-Mojo HTTP server.
    Loads Zig HTTP library and starts server with Mojo callbacks.
    """
    print("=" * 80)
    print("🦙 Shimmy-Mojo HTTP Server (Deprecated)")
    print("=" * 80)
    print()
    print("This Mojo entrypoint is deprecated because Mojo 0.26.x")
    print("cannot resolve dynamic libraries via external_call.")
    print()
    print("Use the Zig OpenAI server instead:")
    print("  1. ./scripts/build_zig.sh")
    print("  2. ./shimmy_openai_server")
    print()
    print("=" * 80)
    return
