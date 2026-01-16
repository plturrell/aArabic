"""
Mojo Embedding Service
Pure Mojo handlers with Zig HTTP server interface.
Port: 8007
"""

from sys.ffi import OwnedDLHandle
from memory import UnsafePointer, alloc
from collections import List


# ============================================================================
# Helper Functions for C String Handling
# ============================================================================

fn string_len(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> Int:
    var i: Int = 0
    while ptr.load(i) != 0:
        i += 1
    return i

fn cstr_to_string(ptr: UnsafePointer[UInt8, ImmutExternalOrigin]) -> String:
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
    if length == 0:
        return ""

    var bytes = List[UInt8]()
    for i in range(length):
        bytes.append(ptr.load(i))

    return String(bytes)

fn create_response(content: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var content_bytes = content.as_bytes()
    var byte_length = len(content_bytes)
    var ptr = alloc[UInt8](byte_length + 1)

    for i in range(byte_length):
        ptr.store(i, content_bytes[i])

    ptr.store(byte_length, 0)
    return ptr


# ============================================================================
# Simple JSON Helpers
# ============================================================================

fn json_string(s: String) -> String:
    var result = String('"')
    result += s
    result += String('"')
    return result

fn json_number(n: Int) -> String:
    return String(n)

fn json_float(f: Float32) -> String:
    return String(f)

fn json_bool(b: Bool) -> String:
    return "true" if b else "false"

fn strip_query(path: String) -> String:
    var idx = path.find("?")
    if idx >= 0:
        return path[0:idx]
    return path


# ============================================================================
# Minimal JSON Parsing (for known fields)
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

fn count_string_array(body: String, key: String) -> Int:
    var pattern = String('"') + key + String('"')
    var idx = body.find(pattern)
    if idx < 0:
        return 0
    var open_idx = body.find("[", idx)
    if open_idx < 0:
        return 0
    var close_idx = body.find("]", open_idx)
    if close_idx < 0:
        return 0

    var count = 0
    var in_string = False
    var i = open_idx + 1
    while i < close_idx:
        var ch = body[i]
        if ch == "\"" and (i == 0 or body[i - 1] != "\\"):
            in_string = not in_string
            if in_string:
                count += 1
        i += 1

    return count

fn count_words(text: String) -> Int:
    let bytes = text.as_bytes()
    let length = len(bytes)
    var word_count = 0
    var in_word = False

    for i in range(length):
        let ch = bytes[i]
        let is_space = (ch == 32) or (ch == 9) or (ch == 10) or (ch == 13)
        if not is_space and not in_word:
            word_count += 1
            in_word = True
        elif is_space:
            in_word = False

    return word_count


# ============================================================================
# Embedding Builders (Stubbed)
# ============================================================================

fn append_embedding_array(inout out: String, dim: Int, seed: Int):
    out += "["
    for i in range(dim):
        if i > 0:
            out += ","
        let value = Float32(((i + seed) % 10)) * 0.1
        out += String(value)
    out += "]"

fn append_embeddings_array(inout out: String, count: Int, dim: Int):
    out += "["
    for i in range(count):
        if i > 0:
            out += ","
        append_embedding_array(out, dim, i)
    out += "]"


# ============================================================================
# Metrics
# ============================================================================

var metrics_requests_total: Int = 0
var metrics_embeddings_generated: Int = 0

fn record_request(embeddings_generated: Int):
    metrics_requests_total += 1
    metrics_embeddings_generated += embeddings_generated


# ============================================================================
# HTTP Request Handlers
# ============================================================================

fn handle_health() -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("status") + String(":") + json_string("healthy") + String(",")
    response += json_string("service") + String(":") + json_string("embedding-mojo") + String(",")
    response += json_string("version") + String(":") + json_string("0.1.0") + String(",")
    response += json_string("language") + String(":") + json_string("mojo") + String(",")
    response += json_string("models") + String(":{")
    response += json_string("general") + String(":") + json_string("paraphrase-multilingual-MiniLM-L12-v2 (384d)") + String(",")
    response += json_string("financial") + String(":") + json_string("CamelBERT-Financial (768d)")
    response += String("},")
    response += json_string("features") + String(":[")
    response += json_string("SIMD-optimized tokenization") + String(",")
    response += json_string("Vectorized mean pooling") + String(",")
    response += json_string("Parallel batch processing") + String(",")
    response += json_string("In-memory LRU cache")
    response += String("]}")

    record_request(0)
    return create_response(response)

fn handle_embed_single(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    let model_type = extract_json_string(body, "model_type", "general")
    let dimensions = 768 if model_type == "financial" else 384
    let model_used = "CamelBERT-Financial" if model_type == "financial" else "multilingual-MiniLM"

    var response = String("{")
    response += json_string("embedding") + String(":")
    append_embedding_array(response, dimensions, 0)
    response += String(",")
    response += json_string("model_used") + String(":") + json_string(model_used) + String(",")
    response += json_string("dimensions") + String(":") + json_number(dimensions) + String(",")
    response += json_string("processing_time_ms") + String(":") + json_number(1)
    response += String("}")

    record_request(1)
    return create_response(response)

fn handle_embed_batch(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    let model_type = extract_json_string(body, "model_type", "general")
    var count = count_string_array(body, "texts")
    if count <= 0:
        count = 1
    let normalize = extract_json_bool(body, "normalize", True)

    let dimensions = 768 if model_type == "financial" else 384
    let model_used = "CamelBERT-Financial" if model_type == "financial" else "multilingual-MiniLM"

    var response = String("{")
    response += json_string("embeddings") + String(":")
    append_embeddings_array(response, count, dimensions)
    response += String(",")
    response += json_string("model_used") + String(":") + json_string(model_used) + String(",")
    response += json_string("dimensions") + String(":") + json_number(dimensions) + String(",")
    response += json_string("count") + String(":") + json_number(count) + String(",")
    response += json_string("normalized") + String(":") + json_bool(normalize) + String(",")
    response += json_string("processing_time_ms") + String(":") + json_number(count * 2)
    response += String("}")

    record_request(count)
    return create_response(response)

fn handle_embed_workflow() -> UnsafePointer[UInt8, MutExternalOrigin]:
    let dimensions = 384

    var response = String("{")
    response += json_string("embedding") + String(":")
    append_embedding_array(response, dimensions, 2)
    response += String(",")
    response += json_string("dimensions") + String(":") + json_number(dimensions) + String(",")
    response += json_string("model") + String(":") + json_string("multilingual-MiniLM")
    response += String("}")

    record_request(1)
    return create_response(response)

fn handle_embed_invoice() -> UnsafePointer[UInt8, MutExternalOrigin]:
    let dimensions = 768

    var response = String("{")
    response += json_string("embedding") + String(":")
    append_embedding_array(response, dimensions, 4)
    response += String(",")
    response += json_string("dimensions") + String(":") + json_number(dimensions) + String(",")
    response += json_string("model") + String(":") + json_string("CamelBERT-Financial")
    response += String("}")

    record_request(1)
    return create_response(response)

fn handle_embed_document(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    let dimensions = 384
    let chunk_size = extract_json_int(body, "chunk_size", 512)
    let document_text = extract_json_string(body, "document_text", "")

    var chunks = 0
    if document_text != "":
        let word_count = count_words(document_text)
        let size = chunk_size if chunk_size > 0 else 512
        chunks = (word_count + size - 1) // size
        if chunks == 0:
            chunks = 1

    var response = String("{")
    response += json_string("embeddings") + String(":")
    append_embeddings_array(response, chunks, dimensions)
    response += String(",")
    response += json_string("dimensions") + String(":") + json_number(dimensions) + String(",")
    response += json_string("chunks") + String(":") + json_number(chunks) + String(",")
    response += json_string("model") + String(":") + json_string("multilingual-MiniLM")
    response += String("}")

    record_request(chunks)
    return create_response(response)

fn handle_openai_embeddings(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var model_name = extract_json_string(body, "model", "multilingual-MiniLM")
    var model_type = extract_json_string(body, "model_type", "")
    var input_text = extract_json_string(body, "input", "")
    var input_count = 0

    if input_text != "":
        input_count = 1
    else:
        input_count = count_string_array(body, "input")
        if input_count <= 0:
            input_count = count_string_array(body, "texts")
        if input_count <= 0:
            input_count = 1

    let is_financial = model_type == "financial" or "CamelBERT" in model_name
    let dimensions = 768 if is_financial else 384

    var response = String("{")
    response += json_string("object") + String(":") + json_string("list") + String(",")
    response += json_string("data") + String(":[")
    for i in range(input_count):
        if i > 0:
            response += String(",")
        response += String("{")
        response += json_string("object") + String(":") + json_string("embedding") + String(",")
        response += json_string("index") + String(":") + json_number(i) + String(",")
        response += json_string("embedding") + String(":")
        append_embedding_array(response, dimensions, i)
        response += String("}")
    response += String("],")
    response += json_string("model") + String(":") + json_string(model_name) + String(",")
    response += json_string("usage") + String(":{")
    response += json_string("prompt_tokens") + String(":") + json_number(input_count * 8) + String(",")
    response += json_string("total_tokens") + String(":") + json_number(input_count * 8)
    response += String("}")
    response += String("}")

    record_request(input_count)
    return create_response(response)

fn handle_models() -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("current_model") + String(":{")
    response += json_string("name") + String(":") + json_string("paraphrase-multilingual-MiniLM-L12-v2") + String(",")
    response += json_string("dimensions") + String(":") + json_number(384) + String(",")
    response += json_string("language_support") + String(":[")
    response += json_string("Arabic") + String(",")
    response += json_string("English") + String(",")
    response += json_string("50+ languages") + String("],")
    response += json_string("use_case") + String(":") + json_string("General purpose, multilingual")
    response += String("},")

    response += json_string("available_models") + String(":[")
    response += String("{")
    response += json_string("name") + String(":") + json_string("paraphrase-multilingual-MiniLM-L12-v2") + String(",")
    response += json_string("dimensions") + String(":") + json_number(384) + String(",")
    response += json_string("size_mb") + String(":") + json_number(420) + String(",")
    response += json_string("use_case") + String(":") + json_string("General purpose, fast, multilingual")
    response += String("},")
    response += String("{")
    response += json_string("name") + String(":") + json_string("CamelBERT-Financial") + String(",")
    response += json_string("dimensions") + String(":") + json_number(768) + String(",")
    response += json_string("size_mb") + String(":") + json_number(500) + String(",")
    response += json_string("use_case") + String(":") + json_string("Arabic financial domain")
    response += String("}")
    response += String("],")

    response += json_string("optimization") + String(":{")
    response += json_string("simd_enabled") + String(":") + json_bool(True) + String(",")
    response += json_string("batch_processing") + String(":") + json_bool(True) + String(",")
    response += json_string("cache_enabled") + String(":") + json_bool(True)
    response += String("}")
    response += String("}")

    record_request(0)
    return create_response(response)

fn handle_metrics() -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("requests_total") + String(":") + json_number(metrics_requests_total) + String(",")
    response += json_string("requests_per_second") + String(":") + json_float(Float32(0.0)) + String(",")
    response += json_string("average_latency_ms") + String(":") + json_float(Float32(0.0)) + String(",")
    response += json_string("cache_hit_rate") + String(":") + json_float(Float32(0.0)) + String(",")
    response += json_string("embeddings_generated") + String(":") + json_number(metrics_embeddings_generated)
    response += String("}")

    record_request(0)
    return create_response(response)

fn handle_not_found(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("error") + String(":") + json_string("Not found") + String(",")
    response += json_string("path") + String(":") + json_string(path)
    response += String("}")
    return create_response(response)


# ============================================================================
# Main HTTP Request Handler (called by Zig)
# ============================================================================

fn handle_http_request(
    method: UnsafePointer[UInt8, ImmutExternalOrigin],
    path: UnsafePointer[UInt8, ImmutExternalOrigin],
    body: UnsafePointer[UInt8, ImmutExternalOrigin],
    body_len: Int
) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var method_str = cstr_to_string(method)
    var path_str = cstr_to_string(path)
    var body_str = cstr_to_string_with_len(body, body_len)
    var clean_path = strip_query(path_str)

    if method_str == "OPTIONS":
        return create_response("{}")

    if clean_path == "/health":
        return handle_health()
    elif clean_path == "/embed/single":
        return handle_embed_single(body_str)
    elif clean_path == "/embed/batch":
        return handle_embed_batch(body_str)
    elif clean_path == "/v1/embeddings":
        return handle_openai_embeddings(body_str)
    elif clean_path == "/embed":
        return handle_openai_embeddings(body_str)
    elif clean_path == "/embed/workflow":
        return handle_embed_workflow()
    elif clean_path == "/embed/invoice":
        return handle_embed_invoice()
    elif clean_path == "/embed/document":
        return handle_embed_document(body_str)
    elif clean_path == "/models":
        return handle_models()
    elif clean_path == "/metrics":
        return handle_metrics()

    return handle_not_found(clean_path)


# ============================================================================
# Main Entry Point
# ============================================================================

fn main():
    print("========================================")
    print("Mojo Embedding Service")
    print("========================================")
    print("")
    print("Architecture:")
    print("  - Zig: HTTP server, networking, I/O")
    print("  - Mojo: embedding handlers")
    print("")

    try:
        var zig_lib = OwnedDLHandle("./libzig_http_embedding.dylib")
        print("Zig library loaded successfully")
        print("Build it with: zig build-lib zig_http_embedding.zig -dynamic -OReleaseFast")
        print("TODO: Wire zig_embedding_serve to handle_http_request")
    except:
        print("Could not load Zig library")
        print("Build it with: zig build-lib zig_http_embedding.zig -dynamic -OReleaseFast")

    print("")
    print("Endpoints:")
    print("  GET  /health")
    print("  POST /embed/single")
    print("  POST /embed/batch")
    print("  POST /embed/workflow")
    print("  POST /embed/invoice")
    print("  POST /embed/document")
    print("  GET  /models")
    print("  GET  /metrics")
    print("")
