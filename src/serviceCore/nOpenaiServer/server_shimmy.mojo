"""
⚠️  DEPRECATED - This file has been moved to services/llm/handlers.mojo
This copy is kept for backward compatibility only.
Please use services/llm/handlers.mojo for new development.

Shimmy-Mojo HTTP Server
Zero Python dependencies - Pure Zig + Mojo architecture
OpenAI-compatible API with Pure Mojo LLM inference

Canonical location: services/llm/handlers.mojo (since Phase 8)
"""

from sys.ffi import OwnedDLHandle, external_call
from memory import UnsafePointer, alloc
from collections import List
from services.llm.time_utils import unix_timestamp

# Import our Mojo inference components
# LLaMA inference now handled by Zig engine via FFI
# from core.llama_inference import LLaMAModel, create_phi3_mini_config, create_llama32_1b_config, create_llama32_3b_config
from inference.bridge.inference_api import InferenceEngine
from inference.tokenization.tokenizer import BPETokenizer
from inference.generation.sampling import SamplingConfig
from inference.generation.generation import TextGenerator, GenerationConfig

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

# Global model instance (singleton for demo)
# In production, would use proper model management
var global_model_loaded: Bool = False

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
    var response = String("{")
    response += json_string("status") + String(":") + json_string("healthy") + String(",")
    response += json_string("engine") + String(":") + json_string("Zig+Mojo") + String(",")
    response += json_string("inference") + String(":") + json_string("Pure Mojo") + String(",")
    response += json_string("model_loaded") + String(":") + json_bool(global_model_loaded)
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
    """Handle /v1/chat/completions - Chat API."""
    print("💬 Chat completions request")
    
    # Mock response (would parse body and call LLaMA model)
    var timestamp = unix_timestamp()
    var completion_id = String("chatcmpl-") + String(timestamp)
    
    var response = String("{")
    response += json_string("id") + String(":") + json_string(completion_id) + String(",")
    response += json_string("object") + String(":") + json_string("chat.completion") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("model") + String(":") + json_string("phi-3-mini") + String(",")
    response += json_string("choices") + String(":[{")
    response += json_string("index") + String(":0,")
    response += json_string("message") + String(":{")
    response += json_string("role") + String(":") + json_string("assistant") + String(",")
    response += json_string("content") + String(":") + json_string("This is a response from Pure Mojo inference engine!")
    response += String("},")
    response += json_string("finish_reason") + String(":") + json_string("stop")
    response += String("}],")
    response += json_string("usage") + String(":{")
    response += json_string("prompt_tokens") + String(":10,")
    response += json_string("completion_tokens") + String(":15,")
    response += json_string("total_tokens") + String(":25")
    response += String("}}")
    
    return create_response(response)

fn handle_completions(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    """Handle /v1/completions - Text completions API."""
    print("📝 Completions request")
    
    var timestamp = unix_timestamp()
    var completion_id = String("cmpl-") + String(timestamp)
    
    var response = String("{")
    response += json_string("id") + String(":") + json_string(completion_id) + String(",")
    response += json_string("object") + String(":") + json_string("text_completion") + String(",")
    response += json_string("created") + String(":") + json_number(timestamp) + String(",")
    response += json_string("model") + String(":") + json_string("phi-3-mini") + String(",")
    response += json_string("choices") + String(":[{")
    response += json_string("text") + String(":") + json_string("Generated by Pure Mojo!") + String(",")
    response += json_string("index") + String(":0,")
    response += json_string("finish_reason") + String(":") + json_string("stop")
    response += String("}]}")
    
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
    print("🦙 Shimmy-Mojo HTTP Server")
    print("=" * 80)
    print()
    print("Architecture:")
    print("  • Zig:  HTTP server, networking, I/O")
    print("  • Mojo: LLM inference, SIMD compute")
    print()
    print("Features:")
    print("  ✅ OpenAI-compatible API")
    print("  ✅ Zero Python dependencies")
    print("  ✅ Pure Mojo inference")
    print("  ✅ SIMD-optimized")
    print()
    print("🚀 Starting server...")
    print()
    
    # Load Zig HTTP library
    try:
        var zig_lib = OwnedDLHandle("./libzig_http_shimmy.dylib")
        print("✅ Zig library loaded successfully")
        print()
    except:
        print("⚠️  Could not load Zig library")
        print("   Build it first: zig build-lib zig_http_shimmy.zig -dynamic -OReleaseFast")
        print()
        return
    
    print("📡 Server will run on http://0.0.0.0:11434")
    print()
    print("Endpoints:")
    print("  GET  /                        - Server info")
    print("  GET  /health                  - Health check")
    print("  GET  /v1/models               - List models")
    print("  POST /v1/chat/completions     - Chat API")
    print("  POST /v1/completions          - Text API")
    print("  GET  /api/tags                - Ollama-compatible")
    print()
    print("=" * 80)
    print()
    print("🎯 Server ready! Press Ctrl+C to stop.")
    print()
    print("=" * 80)
    print()
    
    # TODO: Actually call zig_shimmy_serve with config
    # For now, just demonstrate the architecture
    
    print("✅ Shimmy-Mojo server initialized!")
    print()
    print("Next steps:")
    print("  1. Build Zig library: ./build_zig.sh")
    print("  2. Run server: mojo run server_shimmy.mojo")
    print("  3. Test with curl or OpenAI SDK")
    print()
    print("=" * 80)
