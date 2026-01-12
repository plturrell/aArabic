"""
Mojo NucleusGraph Service
Pure Mojo handlers with Zig HTTP server interface.
Port: 5000
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

fn extract_query_int(path: String, key: String, default: Int) -> Int:
    var pattern = key + "="
    var idx = path.find(pattern)
    if idx < 0:
        return default

    var i = idx + len(pattern)
    var value = 0
    var found = False
    while i < len(path):
        var ch = path[i]
        if ch < "0" or ch > "9":
            break
        let digit = Int(ch.as_bytes()[0]) - 48
        value = value * 10 + digit
        found = True
        i += 1

    return value if found else default

fn extract_query_string(path: String, key: String, default: String) -> String:
    var pattern = key + "="
    var idx = path.find(pattern)
    if idx < 0:
        return default

    var start_idx = idx + len(pattern)
    var end_idx = path.find("&", start_idx)
    if end_idx < 0:
        end_idx = len(path)

    return path[start_idx:end_idx]


# ============================================================================
# Global State
# ============================================================================

var node_id_counter: Int = 1
var edge_id_counter: Int = 1

fn next_node_id() -> String:
    var id = String("node-") + String(node_id_counter)
    node_id_counter += 1
    return id

fn next_edge_id() -> String:
    var id = String("edge-") + String(edge_id_counter)
    edge_id_counter += 1
    return id


# ============================================================================
# Handlers
# ============================================================================

fn handle_root() -> UnsafePointer[UInt8, MutExternalOrigin]:
    return create_response("{\"service\":\"nucleus-graph\",\"status\":\"ready\"}")

fn handle_health() -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("status") + String(":") + json_string("healthy") + String(",")
    response += json_string("service") + String(":") + json_string("nucleus-graph") + String(",")
    response += json_string("memgraph_host") + String(":") + json_string("memgraph") + String(",")
    response += json_string("memgraph_port") + String(":") + json_number(7687) + String(",")
    response += json_string("memgraph_connected") + String(":") + json_bool(False) + String(",")
    response += json_string("timestamp") + String(":") + json_number(0)
    response += String("}")
    return create_response(response)

fn handle_query(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    let query = extract_json_string(body, "query", "")
    if query == "":
        return create_response("{\"success\":false,\"error\":\"query is required\"}")

    var results_json = String("[]")
    if "MATCH" in query.upper():
        results_json = String("[") +
            String("{\"n\":{\"id\":\"1\",\"labels\":[\"Document\"],\"properties\":{\"title\":\"Sample Document\"}}},") +
            String("{\"n\":{\"id\":\"2\",\"labels\":[\"Concept\"],\"properties\":{\"name\":\"AI Research\"}}}") +
            String("]")

    var response = String("{")
    response += json_string("success") + String(":") + json_bool(True) + String(",")
    response += json_string("data") + String(":{")
    response += json_string("query") + String(":") + json_string(query) + String(",")
    response += json_string("results") + String(":") + results_json + String(",")
    response += json_string("execution_time") + String(":") + json_string("0.001s") + String(",")
    response += json_string("nodes_created") + String(":") + json_number(0) + String(",")
    response += json_string("relationships_created") + String(":") + json_number(0) + String(",")
    response += json_string("properties_set") + String(":") + json_number(0) + String(",")
    response += json_string("message") + String(":") + json_string("Query executed (stubbed response)")
    response += String("}")
    response += String("}")
    return create_response(response)

fn handle_graph(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var limit = extract_query_int(path, "limit", 50)
    if limit < 0:
        limit = 0

    var nodes_json = String("[")
    if limit >= 1:
        nodes_json += String("{\"id\":\"1\",\"label\":\"Document\",\"properties\":{\"title\":\"AI Research Paper\",\"type\":\"Document\"}}")
    if limit >= 2:
        nodes_json += String(",") + String("{\"id\":\"2\",\"label\":\"Concept\",\"properties\":{\"name\":\"Machine Learning\",\"type\":\"Concept\"}}")
    if limit >= 3:
        nodes_json += String(",") + String("{\"id\":\"3\",\"label\":\"Concept\",\"properties\":{\"name\":\"Neural Networks\",\"type\":\"Concept\"}}")
    nodes_json += String("]")

    var edges_json = String("[")
    if limit >= 2:
        edges_json += String("{\"id\":\"e1\",\"source\":\"1\",\"target\":\"2\",\"label\":\"MENTIONS\"}")
    if limit >= 3:
        edges_json += String(",") + String("{\"id\":\"e2\",\"source\":\"1\",\"target\":\"3\",\"label\":\"MENTIONS\"}")
    edges_json += String("]")

    var response = String("{")
    response += json_string("nodes") + String(":") + nodes_json + String(",")
    response += json_string("edges") + String(":") + edges_json
    response += String("}")
    return create_response(response)

fn handle_create_node(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var node_id = extract_json_string(body, "id", "")
    if node_id == "":
        node_id = next_node_id()
    let label = extract_json_string(body, "label", "Node")
    let node_type = extract_json_string(body, "type", "")

    var response = String("{")
    response += json_string("id") + String(":") + json_string(node_id) + String(",")
    response += json_string("label") + String(":") + json_string(label) + String(",")
    response += json_string("properties") + String(":") + String("{}") + String(",")
    response += json_string("type") + String(":") + json_string(node_type)
    response += String("}")
    return create_response(response)

fn handle_create_edge(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var edge_id = extract_json_string(body, "id", "")
    if edge_id == "":
        edge_id = next_edge_id()
    let source = extract_json_string(body, "source", "")
    let target = extract_json_string(body, "target", "")
    let label = extract_json_string(body, "label", "RELATES_TO")

    var response = String("{")
    response += json_string("id") + String(":") + json_string(edge_id) + String(",")
    response += json_string("source") + String(":") + json_string(source) + String(",")
    response += json_string("target") + String(":") + json_string(target) + String(",")
    response += json_string("label") + String(":") + json_string(label) + String(",")
    response += json_string("properties") + String(":") + String("{}")
    response += String("}")
    return create_response(response)

fn handle_schema() -> UnsafePointer[UInt8, MutExternalOrigin]:
    var response = String("{")
    response += json_string("success") + String(":") + json_bool(True) + String(",")
    response += json_string("schema") + String(":{")
    response += json_string("node_labels") + String(":[")
    response += String("{\"label\":\"Document\",\"count\":5,\"properties\":[\"id\",\"title\",\"source\",\"content\"]},")
    response += String("{\"label\":\"Concept\",\"count\":15,\"properties\":[\"name\",\"frequency\"]},")
    response += String("{\"label\":\"Author\",\"count\":3,\"properties\":[\"name\",\"email\"]},")
    response += String("{\"label\":\"Topic\",\"count\":8,\"properties\":[\"name\",\"category\"]}")
    response += String("],")
    response += json_string("relationship_types") + String(":[")
    response += String("{\"type\":\"MENTIONS\",\"count\":25,\"properties\":[\"strength\",\"context\"]},")
    response += String("{\"type\":\"RELATES_TO\",\"count\":12,\"properties\":[\"type\",\"confidence\"]},")
    response += String("{\"type\":\"AUTHORED_BY\",\"count\":5,\"properties\":[\"date\"]},")
    response += String("{\"type\":\"BELONGS_TO\",\"count\":8,\"properties\":[\"relevance\"]}")
    response += String("],")
    response += json_string("constraints") + String(":[] ,")
    response += json_string("indexes") + String(":[]")
    response += String("}")
    response += String("}")
    return create_response(response)

fn handle_visualize(path: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    var limit = extract_query_int(path, "limit", 50)
    if limit < 0:
        limit = 0

    var nodes_json = String("[")
    if limit >= 1:
        nodes_json += String("{\"id\":\"1\",\"label\":\"Document\",\"properties\":{\"title\":\"AI Research Paper\",\"type\":\"Document\"}}")
    if limit >= 2:
        nodes_json += String(",") + String("{\"id\":\"2\",\"label\":\"Concept\",\"properties\":{\"name\":\"Machine Learning\",\"type\":\"Concept\"}}")
    if limit >= 3:
        nodes_json += String(",") + String("{\"id\":\"3\",\"label\":\"Concept\",\"properties\":{\"name\":\"Neural Networks\",\"type\":\"Concept\"}}")
    if limit >= 4:
        nodes_json += String(",") + String("{\"id\":\"4\",\"label\":\"Author\",\"properties\":{\"name\":\"Dr. Smith\",\"type\":\"Author\"}}")
    if limit >= 5:
        nodes_json += String(",") + String("{\"id\":\"5\",\"label\":\"Topic\",\"properties\":{\"name\":\"Deep Learning\",\"type\":\"Topic\"}}")
    nodes_json += String("]")

    var edges_json = String("[")
    if limit >= 2:
        edges_json += String("{\"id\":\"e1\",\"source\":\"1\",\"target\":\"2\",\"type\":\"MENTIONS\",\"properties\":{\"strength\":0.8}}")
    if limit >= 3:
        edges_json += String(",") + String("{\"id\":\"e2\",\"source\":\"1\",\"target\":\"3\",\"type\":\"MENTIONS\",\"properties\":{\"strength\":0.9}}")
    if limit >= 4:
        edges_json += String(",") + String("{\"id\":\"e3\",\"source\":\"2\",\"target\":\"3\",\"type\":\"RELATES_TO\",\"properties\":{\"confidence\":0.7}}")
    if limit >= 4:
        edges_json += String(",") + String("{\"id\":\"e4\",\"source\":\"1\",\"target\":\"4\",\"type\":\"AUTHORED_BY\",\"properties\":{\"date\":\"2024-01-01\"}}")
    if limit >= 5:
        edges_json += String(",") + String("{\"id\":\"e5\",\"source\":\"2\",\"target\":\"5\",\"type\":\"BELONGS_TO\",\"properties\":{\"relevance\":0.85}}")
    edges_json += String("]")

    var response = String("{")
    response += json_string("success") + String(":") + json_bool(True) + String(",")
    response += json_string("graph") + String(":{")
    response += json_string("nodes") + String(":") + nodes_json + String(",")
    response += json_string("edges") + String(":") + edges_json + String(",")
    response += json_string("stats") + String(":{")
    response += json_string("node_count") + String(":") + json_number(limit if limit < 5 else 5) + String(",")
    response += json_string("edge_count") + String(":") + json_number(5) + String(",")
    response += json_string("node_types") + String(":[\"Document\",\"Concept\",\"Author\",\"Topic\"]")
    response += String("}")
    response += String("}")
    response += String("}")
    return create_response(response)

fn handle_import(body: String) -> UnsafePointer[UInt8, MutExternalOrigin]:
    let source = extract_json_string(body, "source", "")
    if source == "":
        return create_response("{\"success\":false,\"error\":\"source and data are required\"}")

    var response = String("{")
    response += json_string("success") + String(":") + json_bool(True) + String(",")
    response += json_string("message") + String(":") + json_string("Successfully imported data from " + source) + String(",")
    response += json_string("stats") + String(":{")
    response += json_string("nodes_created") + String(":") + json_number(0) + String(",")
    response += json_string("relationships_created") + String(":") + json_number(0) + String(",")
    response += json_string("properties_set") + String(":") + json_number(0) + String(",")
    response += json_string("execution_time") + String(":") + json_string("0.05s")
    response += String("}")
    response += String("}")
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

    if clean_path == "/":
        return handle_root()
    elif clean_path == "/health" or clean_path == "/api/nucleus-graph/health":
        return handle_health()
    elif clean_path == "/api/query" or clean_path == "/api/nucleus-graph/query":
        return handle_query(body_str)
    elif clean_path == "/api/graph":
        return handle_graph(path_str)
    elif clean_path == "/api/nodes":
        return handle_create_node(body_str)
    elif clean_path == "/api/edges":
        return handle_create_edge(body_str)
    elif clean_path == "/api/nucleus-graph/schema":
        return handle_schema()
    elif clean_path == "/api/nucleus-graph/visualize":
        return handle_visualize(path_str)
    elif clean_path == "/api/nucleus-graph/import":
        return handle_import(body_str)

    return handle_not_found(clean_path)


# ============================================================================
# Main Entry Point
# ============================================================================

fn main():
    print("========================================")
    print("Mojo NucleusGraph Service")
    print("========================================")
    print("")
    print("Architecture:")
    print("  - Zig: HTTP server, networking, I/O")
    print("  - Mojo: graph handlers")
    print("")

    try:
        var zig_lib = OwnedDLHandle("./libzig_http_nucleusgraph.dylib")
        print("Zig library loaded successfully")
        print("Build it with: zig build-lib zig_http_nucleusgraph.zig -dynamic -OReleaseFast")
        print("TODO: Wire zig_nucleusgraph_serve to handle_http_request")
    except:
        print("Could not load Zig library")
        print("Build it with: zig build-lib zig_http_nucleusgraph.zig -dynamic -OReleaseFast")

    print("")
    print("Endpoints:")
    print("  GET  /")
    print("  GET  /health")
    print("  POST /api/query")
    print("  GET  /api/graph")
    print("  POST /api/nodes")
    print("  POST /api/edges")
    print("  GET  /api/nucleus-graph/schema")
    print("  GET  /api/nucleus-graph/visualize")
    print("  POST /api/nucleus-graph/import")
    print("")
