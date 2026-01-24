// nWorkflow HTTP Server
// Serves SAPUI5 webapp and provides REST API for workflow management
// Port: 8090 (configurable)

const std = @import("std");
const net = std.net;
const mem = std.mem;
const fs = std.fs;
const json = std.json;

const Config = struct {
    host: []const u8 = "0.0.0.0",
    port: u16 = 8090,
    webapp_path: []const u8 = "webapp",
    // OpenAI API proxy configuration
    llm_endpoint: []const u8 = "localhost:11434",
    llm_api_key: ?[]const u8 = null,
    openai_proxy_enabled: bool = true,
};

// OpenAI API Request/Response structures
const OpenAIChatRequest = struct {
    model: []const u8 = "phi-3-mini",
    messages: []const ChatMessage = &[_]ChatMessage{},
    temperature: f32 = 0.7,
    max_tokens: u32 = 1000,
    stream: bool = false,
};

const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

const OpenAIEmbeddingRequest = struct {
    model: []const u8 = "text-embedding-ada-002",
    input: []const u8 = "",
};

const OpenAICompletionRequest = struct {
    model: []const u8 = "phi-3-mini",
    prompt: []const u8 = "",
    temperature: f32 = 0.7,
    max_tokens: u32 = 1000,
};

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Load configuration from environment variables
    var config = Config{};
    if (std.posix.getenv("LLM_ENDPOINT")) |endpoint| {
        config.llm_endpoint = endpoint;
    }
    if (std.posix.getenv("LLM_API_KEY")) |api_key| {
        config.llm_api_key = api_key;
    }
    if (std.posix.getenv("OPENAI_PROXY_ENABLED")) |enabled| {
        config.openai_proxy_enabled = mem.eql(u8, enabled, "true") or mem.eql(u8, enabled, "1");
    }

    const addr = try net.Address.parseIp(config.host, config.port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║           nWorkflow Server - Enterprise Workflow Engine      ║\n", .{});
    std.debug.print("╠══════════════════════════════════════════════════════════════╣\n", .{});
    std.debug.print("║  🌐 Web UI:  http://localhost:{d:<5}                          ║\n", .{config.port});
    std.debug.print("║  📡 API:    http://localhost:{d:<5}/api/v1                    ║\n", .{config.port});
    std.debug.print("║  🤖 OpenAI: http://localhost:{d:<5}/v1                        ║\n", .{config.port});
    std.debug.print("║  📂 Webapp: {s:<30}                   ║\n", .{config.webapp_path});
    std.debug.print("╚══════════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});

    while (true) {
        const conn = server.accept() catch |err| {
            std.debug.print("Accept error: {any}\n", .{err});
            continue;
        };
        handleConnection(allocator, conn, config) catch |err| {
            std.debug.print("Connection error: {any}\n", .{err});
        };
    }
}

fn handleConnection(allocator: mem.Allocator, conn: net.Server.Connection, config: Config) !void {
    defer conn.stream.close();

    var buffer: [8192]u8 = undefined;
    const bytes_read = conn.stream.read(&buffer) catch return;
    if (bytes_read == 0) return;

    const request = buffer[0..bytes_read];

    // Parse method and path
    var lines = mem.splitSequence(u8, request, "\r\n");
    const request_line = lines.first();
    var parts = mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;

    std.debug.print("{s} {s}\n", .{ method, path });

    // Route request
    const response = try routeRequest(allocator, method, path, request, config);
    defer allocator.free(response.body);

    // Send response
    try sendResponse(conn.stream, response);
}

const Response = struct {
    status: u16,
    content_type: []const u8,
    body: []const u8,
};

fn routeRequest(allocator: mem.Allocator, method: []const u8, path: []const u8, request: []const u8, config: Config) !Response {
    // Handle CORS preflight requests
    if (mem.eql(u8, method, "OPTIONS")) {
        return Response{
            .status = 204,
            .content_type = "text/plain",
            .body = try allocator.dupe(u8, ""),
        };
    }

    // OpenAI-compatible API Routes
    if (mem.startsWith(u8, path, "/v1/")) {
        return routeOpenAI(allocator, method, path, request, config);
    }

    // Internal API Routes
    if (mem.startsWith(u8, path, "/api/")) {
        return routeApi(allocator, method, path);
    }

    // Static file serving
    return serveStaticFile(allocator, path, config);
}

// ============================================================================
// OpenAI-Compatible API Routes
// ============================================================================

fn routeOpenAI(allocator: mem.Allocator, method: []const u8, path: []const u8, request: []const u8, config: Config) !Response {
    // Check if OpenAI proxy is enabled
    if (!config.openai_proxy_enabled) {
        return Response{
            .status = 403,
            .content_type = "application/json",
            .body = try allocator.dupe(u8, "{\"error\":{\"message\":\"OpenAI proxy is disabled\",\"type\":\"forbidden\",\"code\":403}}"),
        };
    }

    std.debug.print("🤖 OpenAI API: {s} {s}\n", .{ method, path });

    // GET /v1/models - List available models
    if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/v1/models")) {
        return handleGetModels(allocator);
    }

    // POST /v1/chat/completions - Chat completions
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/chat/completions")) {
        return handleChatCompletions(allocator, request, config);
    }

    // POST /v1/embeddings - Generate embeddings
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/embeddings")) {
        return handleEmbeddings(allocator, request, config);
    }

    // POST /v1/completions - Legacy completions
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/v1/completions")) {
        return handleCompletions(allocator, request, config);
    }

    // 404 for unknown OpenAI routes
    return Response{
        .status = 404,
        .content_type = "application/json",
        .body = try std.fmt.allocPrint(allocator, "{{\"error\":{{\"message\":\"Unknown endpoint: {s}\",\"type\":\"invalid_request_error\",\"code\":404}}}}", .{path}),
    };
}

// GET /v1/models - List available models
fn handleGetModels(allocator: mem.Allocator) !Response {
    const timestamp = @divFloor(std.time.timestamp(), 1);
    const response_body = try std.fmt.allocPrint(allocator,
        \\{{"object":"list","data":[
        \\{{"id":"phi-3-mini","object":"model","created":{d},"owned_by":"nworkflow"}},
        \\{{"id":"llama-3.2-1b","object":"model","created":{d},"owned_by":"nworkflow"}},
        \\{{"id":"llama-3.2-3b","object":"model","created":{d},"owned_by":"nworkflow"}},
        \\{{"id":"text-embedding-ada-002","object":"model","created":{d},"owned_by":"nworkflow"}}
        \\]}}
    , .{ timestamp, timestamp, timestamp, timestamp });

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = response_body,
    };
}

// POST /v1/chat/completions - Chat completions endpoint
fn handleChatCompletions(allocator: mem.Allocator, request: []const u8, config: Config) !Response {
    // Parse request body
    const body = getRequestBody(request) orelse {
        return errorResponse(allocator, 400, "Missing request body");
    };

    // Parse the OpenAI request
    const parsed_request = parseOpenAIChatRequest(allocator, body) catch {
        return errorResponse(allocator, 400, "Invalid JSON in request body");
    };

    // Forward to LLM backend
    const llm_response = forwardToLLM(allocator, config.llm_endpoint, "/v1/chat/completions", body, config.llm_api_key) catch |err| {
        std.debug.print("LLM forward error: {any}\n", .{err});
        // Return a mock response if LLM is unavailable
        return buildMockChatResponse(allocator, parsed_request.model);
    };

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = llm_response,
    };
}

// POST /v1/embeddings - Embeddings endpoint
fn handleEmbeddings(allocator: mem.Allocator, request: []const u8, config: Config) !Response {
    const body = getRequestBody(request) orelse {
        return errorResponse(allocator, 400, "Missing request body");
    };

    // Forward to LLM backend
    const llm_response = forwardToLLM(allocator, config.llm_endpoint, "/v1/embeddings", body, config.llm_api_key) catch |err| {
        std.debug.print("LLM embedding forward error: {any}\n", .{err});
        // Return a mock embedding response
        return buildMockEmbeddingResponse(allocator);
    };

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = llm_response,
    };
}

// POST /v1/completions - Legacy completions endpoint
fn handleCompletions(allocator: mem.Allocator, request: []const u8, config: Config) !Response {
    const body = getRequestBody(request) orelse {
        return errorResponse(allocator, 400, "Missing request body");
    };

    // Parse legacy completion request and convert to chat format
    const chat_body = convertLegacyToChat(allocator, body) catch {
        return errorResponse(allocator, 400, "Invalid completion request format");
    };
    defer allocator.free(chat_body);

    // Forward to chat completions endpoint
    const llm_response = forwardToLLM(allocator, config.llm_endpoint, "/v1/chat/completions", chat_body, config.llm_api_key) catch |err| {
        std.debug.print("LLM completion forward error: {any}\n", .{err});
        return buildMockCompletionResponse(allocator);
    };

    // Convert chat response back to legacy format
    const legacy_response = convertChatToLegacy(allocator, llm_response) catch {
        return Response{
            .status = 200,
            .content_type = "application/json",
            .body = llm_response,
        };
    };
    allocator.free(llm_response);

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = legacy_response,
    };
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract HTTP request body from raw request
fn getRequestBody(request: []const u8) ?[]const u8 {
    // Find the blank line separating headers from body
    const body_start = mem.indexOf(u8, request, "\r\n\r\n") orelse return null;
    const body = request[body_start + 4 ..];
    if (body.len == 0) return null;
    return body;
}

/// Parse OpenAI chat request JSON
fn parseOpenAIChatRequest(allocator: mem.Allocator, body: []const u8) !OpenAIChatRequest {
    _ = allocator;
    var parsed = OpenAIChatRequest{};

    // Simple JSON parsing for key fields
    if (extractJsonString(body, "model")) |model| {
        parsed.model = model;
    }

    if (extractJsonNumber(body, "temperature")) |temp| {
        parsed.temperature = @floatCast(temp);
    }

    if (extractJsonInt(body, "max_tokens")) |max| {
        parsed.max_tokens = @intCast(max);
    }

    return parsed;
}

/// Forward request to LLM backend
fn forwardToLLM(allocator: mem.Allocator, endpoint: []const u8, path: []const u8, body: []const u8, api_key: ?[]const u8) ![]const u8 {
    _ = api_key;

    // Parse host and port from endpoint
    var host: []const u8 = endpoint;
    var port: u16 = 80;

    if (mem.indexOf(u8, endpoint, ":")) |colon_idx| {
        host = endpoint[0..colon_idx];
        port = std.fmt.parseInt(u16, endpoint[colon_idx + 1 ..], 10) catch 80;
    }

    // Connect to LLM backend
    const addr = try net.Address.parseIp(host, port);
    const stream = try net.tcpConnectToAddress(addr);
    defer stream.close();

    // Build HTTP request
    var request_buf: [16384]u8 = undefined;
    const http_request = try std.fmt.bufPrint(&request_buf,
        "POST {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{ path, endpoint, body.len, body },
    );

    // Send request
    _ = try stream.write(http_request);

    // Read response
    var response_buf = std.ArrayList(u8){};
    defer response_buf.deinit(allocator);

    var read_buf: [4096]u8 = undefined;
    while (true) {
        const bytes_read = stream.read(&read_buf) catch break;
        if (bytes_read == 0) break;
        try response_buf.appendSlice(allocator, read_buf[0..bytes_read]);
    }

    // Extract response body
    const response = response_buf.items;
    const body_start = mem.indexOf(u8, response, "\r\n\r\n") orelse return error.InvalidResponse;
    return try allocator.dupe(u8, response[body_start + 4 ..]);
}

/// Build mock chat completion response when LLM is unavailable
fn buildMockChatResponse(allocator: mem.Allocator, model: []const u8) !Response {
    const timestamp = @divFloor(std.time.timestamp(), 1);
    const response_body = try std.fmt.allocPrint(allocator,
        \\{{"id":"chatcmpl-mock-{d}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"I'm currently unable to connect to the LLM backend. Please ensure the LLM service is running at the configured endpoint."}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}
    , .{ timestamp, timestamp, model });

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = response_body,
    };
}

/// Build mock embedding response when LLM is unavailable
fn buildMockEmbeddingResponse(allocator: mem.Allocator) !Response {
    const response_body = try allocator.dupe(u8,
        \\{"object":"list","data":[{"object":"embedding","embedding":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"index":0}],"model":"text-embedding-ada-002","usage":{"prompt_tokens":0,"total_tokens":0}}
    );

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = response_body,
    };
}

/// Build mock legacy completion response
fn buildMockCompletionResponse(allocator: mem.Allocator) !Response {
    const timestamp = @divFloor(std.time.timestamp(), 1);
    const response_body = try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-mock-{d}","object":"text_completion","created":{d},"model":"phi-3-mini","choices":[{{"text":"LLM backend unavailable.","index":0,"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}
    , .{ timestamp, timestamp });

    return Response{
        .status = 200,
        .content_type = "application/json",
        .body = response_body,
    };
}

/// Convert legacy completion request to chat format
fn convertLegacyToChat(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    // Extract prompt from legacy format
    const prompt = extractJsonString(body, "prompt") orelse "";
    const model = extractJsonString(body, "model") orelse "phi-3-mini";
    const max_tokens = extractJsonInt(body, "max_tokens") orelse 1000;
    const temperature = extractJsonNumber(body, "temperature") orelse 0.7;

    return try std.fmt.allocPrint(allocator,
        \\{{"model":"{s}","messages":[{{"role":"user","content":"{s}"}}],"max_tokens":{d},"temperature":{d:.2}}}
    , .{ model, prompt, max_tokens, temperature });
}

/// Convert chat response to legacy completion format
fn convertChatToLegacy(allocator: mem.Allocator, chat_response: []const u8) ![]const u8 {
    // Simple conversion - extract content and wrap in legacy format
    const content_start = mem.indexOf(u8, chat_response, "\"content\":\"") orelse return error.InvalidResponse;
    const content_begin = content_start + 11;
    const content_end = mem.indexOf(u8, chat_response[content_begin..], "\"") orelse return error.InvalidResponse;
    const content = chat_response[content_begin .. content_begin + content_end];

    const timestamp = @divFloor(std.time.timestamp(), 1);
    return try std.fmt.allocPrint(allocator,
        \\{{"id":"cmpl-{d}","object":"text_completion","created":{d},"model":"phi-3-mini","choices":[{{"text":"{s}","index":0,"finish_reason":"stop"}}],"usage":{{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}}}
    , .{ timestamp, timestamp, content });
}

/// Extract string value from JSON (simple parser)
fn extractJsonString(json_str: []const u8, key: []const u8) ?[]const u8 {
    // Look for "key":"value" pattern
    var search_buf: [256]u8 = undefined;
    const search_pattern = std.fmt.bufPrint(&search_buf, "\"{s}\":\"", .{key}) catch return null;

    const key_start = mem.indexOf(u8, json_str, search_pattern) orelse return null;
    const value_start = key_start + search_pattern.len;
    const value_end = mem.indexOf(u8, json_str[value_start..], "\"") orelse return null;

    return json_str[value_start .. value_start + value_end];
}

/// Extract number value from JSON (simple parser)
fn extractJsonNumber(json_str: []const u8, key: []const u8) ?f64 {
    var search_buf: [256]u8 = undefined;
    const search_pattern = std.fmt.bufPrint(&search_buf, "\"{s}\":", .{key}) catch return null;

    const key_start = mem.indexOf(u8, json_str, search_pattern) orelse return null;
    const value_start = key_start + search_pattern.len;

    // Skip whitespace
    var i: usize = 0;
    while (i < json_str[value_start..].len and (json_str[value_start + i] == ' ' or json_str[value_start + i] == '\t')) : (i += 1) {}

    // Find end of number
    var end: usize = i;
    while (end < json_str[value_start..].len) : (end += 1) {
        const c = json_str[value_start + end];
        if (c != '.' and c != '-' and (c < '0' or c > '9')) break;
    }

    const num_str = json_str[value_start + i .. value_start + end];
    return std.fmt.parseFloat(f64, num_str) catch return null;
}

/// Extract integer value from JSON
fn extractJsonInt(json_str: []const u8, key: []const u8) ?i64 {
    const num = extractJsonNumber(json_str, key) orelse return null;
    return @intFromFloat(num);
}

/// Build error response in OpenAI format
fn errorResponse(allocator: mem.Allocator, status: u16, message: []const u8) !Response {
    const error_type = switch (status) {
        400 => "invalid_request_error",
        401 => "authentication_error",
        403 => "forbidden",
        404 => "not_found",
        500 => "internal_server_error",
        else => "unknown_error",
    };

    return Response{
        .status = status,
        .content_type = "application/json",
        .body = try std.fmt.allocPrint(allocator, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\",\"code\":{d}}}}}", .{ message, error_type, status }),
    };
}

fn routeApi(allocator: mem.Allocator, method: []const u8, path: []const u8) !Response {
    // Health check
    if (mem.eql(u8, path, "/api/health") or mem.eql(u8, path, "/api/v1/health")) {
        return Response{
            .status = 200,
            .content_type = "application/json",
            .body = try allocator.dupe(u8, "{\"status\":\"healthy\",\"service\":\"nWorkflow\",\"version\":\"1.0.0\"}"),
        };
    }

    // Info endpoint
    if (mem.eql(u8, path, "/api/v1/info") or mem.eql(u8, path, "/api/info")) {
        return Response{
            .status = 200,
            .content_type = "application/json",
            .body = try allocator.dupe(u8,
                \\{"name":"nWorkflow","version":"1.0.0","description":"Enterprise Workflow Automation Platform","architecture":"Zig + SAPUI5","features":["Petri Net Engine","Visual Editor","AI Integration","Multi-Database Support"]}
            ),
        };
    }

    // Workflows list
    if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/api/v1/workflows")) {
        return Response{
            .status = 200,
            .content_type = "application/json",
            .body = try allocator.dupe(u8, "{\"workflows\":[],\"total\":0}"),
        };
    }

    // Node types
    if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/api/v1/node-types")) {
        return Response{
            .status = 200,
            .content_type = "application/json",
            .body = try allocator.dupe(u8,
                \\{"nodeTypes":[{"id":"start","name":"Start","category":"flow"},{"id":"end","name":"End","category":"flow"},{"id":"task","name":"Task","category":"action"},{"id":"decision","name":"Decision","category":"flow"},{"id":"llm","name":"LLM","category":"ai"},{"id":"http","name":"HTTP Request","category":"action"},{"id":"database","name":"Database","category":"data"},{"id":"transform","name":"Transform","category":"data"},{"id":"filter","name":"Filter","category":"data"},{"id":"aggregate","name":"Aggregate","category":"data"}]}
            ),
        };
    }

    // 404 for unknown API routes
    return Response{
        .status = 404,
        .content_type = "application/json",
        .body = try std.fmt.allocPrint(allocator, "{{\"error\":\"Not found\",\"path\":\"{s}\"}}", .{path}),
    };
}

fn serveStaticFile(allocator: mem.Allocator, path: []const u8, config: Config) !Response {
    // Map URL path to file path
    const file_path = if (mem.eql(u8, path, "/") or mem.eql(u8, path, "/index.html"))
        try std.fmt.allocPrint(allocator, "{s}/index.html", .{config.webapp_path})
    else if (mem.startsWith(u8, path, "/"))
        try std.fmt.allocPrint(allocator, "{s}{s}", .{ config.webapp_path, path })
    else
        try std.fmt.allocPrint(allocator, "{s}/{s}", .{ config.webapp_path, path });
    defer allocator.free(file_path);

    // Read file
    const file = fs.cwd().openFile(file_path, .{}) catch {
        return Response{
            .status = 404,
            .content_type = "text/html",
            .body = try allocator.dupe(u8, "<!DOCTYPE html><html><body><h1>404 Not Found</h1></body></html>"),
        };
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const content = try file.readToEndAlloc(allocator, file_size);

    return Response{
        .status = 200,
        .content_type = getContentType(path),
        .body = content,
    };
}

fn getContentType(path: []const u8) []const u8 {
    if (mem.endsWith(u8, path, ".html")) return "text/html; charset=utf-8";
    if (mem.endsWith(u8, path, ".css")) return "text/css; charset=utf-8";
    if (mem.endsWith(u8, path, ".js")) return "application/javascript; charset=utf-8";
    if (mem.endsWith(u8, path, ".json")) return "application/json; charset=utf-8";
    if (mem.endsWith(u8, path, ".xml")) return "application/xml; charset=utf-8";
    if (mem.endsWith(u8, path, ".properties")) return "text/plain; charset=utf-8";
    if (mem.endsWith(u8, path, ".png")) return "image/png";
    if (mem.endsWith(u8, path, ".jpg") or mem.endsWith(u8, path, ".jpeg")) return "image/jpeg";
    if (mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    if (mem.endsWith(u8, path, ".ico")) return "image/x-icon";
    if (mem.endsWith(u8, path, ".woff")) return "font/woff";
    if (mem.endsWith(u8, path, ".woff2")) return "font/woff2";
    return "application/octet-stream";
}

fn sendResponse(stream: net.Stream, response: Response) !void {
    const status_text = switch (response.status) {
        200 => "OK",
        201 => "Created",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        else => "Unknown",
    };

    var header_buf: [1024]u8 = undefined;
    const header = try std.fmt.bufPrint(&header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{ response.status, status_text, response.content_type, response.body.len },
    );

    _ = try stream.write(header);
    _ = try stream.write(response.body);
}

