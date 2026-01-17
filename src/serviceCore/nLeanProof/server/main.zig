const std = @import("std");
const inference = @import("inference.zig");
const streaming = @import("streaming.zig");

const json = std.json;
const mem = std.mem;
const net = std.net;
const atomic = std.atomic;

const default_host = "0.0.0.0";
const default_port: u16 = 8001;
const default_max_request_bytes: usize = 1024 * 1024;
const default_max_prompt_bytes: usize = 16 * 1024;
const default_max_tokens_cap: u32 = 4096;
const default_pool_size: usize = 2;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

const ErrorResponse = struct {
    @"error": struct {
        message: []const u8,
        code: []const u8,
        type: []const u8 = "invalid_request_error",
    },
};

const Response = struct {
    status: u16,
    body: []u8,
    content_type: []const u8 = "application/json",
};

const HandlerResult = union(enum) {
    response: Response,
    streamed: void,
};

const Limits = struct {
    max_request_bytes: usize,
    max_prompt_bytes: usize,
    max_tokens_cap: u32,
};

const EnginePool = struct {
    allocator: std.mem.Allocator,
    engines: []inference.InferenceEngine,
    next: atomic.Value(usize) = atomic.Value(usize).init(0),

    fn init(alloc: std.mem.Allocator, size: usize) !EnginePool {
        const count = if (size == 0) default_pool_size else size;
        const engs = try alloc.alloc(inference.InferenceEngine, count);
        var pool = EnginePool{
            .allocator = alloc,
            .engines = engs,
        };
        var i: usize = 0;
        while (i < count) : (i += 1) {
            pool.engines[i] = inference.InferenceEngine.init(alloc);
        }
        return pool;
    }

    fn setupAll(self: *EnginePool, lib_path: ?[]const u8, model_path: ?[]const u8) void {
        for (self.engines) |*eng| {
            eng.applyEnvOverrides();
            if (lib_path) |lp| {
                var paths = [_][]const u8{lp};
                eng.loadLibrary(&paths);
            }
            if (model_path) |mp| {
                eng.loadModel(mp) catch {};
            }
        }
    }

    fn acquire(self: *EnginePool) *inference.InferenceEngine {
        const idx = self.next.fetchAdd(1, .monotonic) % self.engines.len;
        return &self.engines[idx];
    }

    fn deinit(self: *EnginePool) void {
        for (self.engines) |*eng| {
            eng.deinit();
        }
        self.allocator.free(self.engines);
    }
};

fn getenv(name: []const u8) ?[]const u8 {
    if (std.posix.getenv(name)) |value| {
        return value[0..value.len];
    }
    return null;
}

fn parseEnvU16(name: []const u8, default_value: u16) u16 {
    if (getenv(name)) |value| {
        return std.fmt.parseInt(u16, value, 10) catch default_value;
    }
    return default_value;
}

fn parseEnvUsize(name: []const u8, default_value: usize) usize {
    if (getenv(name)) |value| {
        return std.fmt.parseInt(usize, value, 10) catch default_value;
    }
    return default_value;
}

fn parseEnvU32(name: []const u8, default_value: u32) u32 {
    if (getenv(name)) |value| {
        return std.fmt.parseInt(u32, value, 10) catch default_value;
    }
    return default_value;
}

fn encodeJson(value: anytype) ![]u8 {
    return json.Stringify.valueAlloc(allocator, value, .{});
}

fn errorResponse(code: []const u8, message: []const u8, status: u16) !Response {
    const payload = try encodeJson(ErrorResponse{ .@"error" = .{ .message = message, .code = code } });
    return Response{ .status = status, .body = payload };
}

fn parseJson(comptime T: type, body: []const u8) !T {
    const parsed = try json.parseFromSlice(T, allocator, body, .{ .ignore_unknown_fields = true });
    // NOTE: parsed owns allocations; we intentionally leak here to keep returned slices alive.
    return parsed.value;
}

fn sendResponse(stream: net.Stream, status: u16, content_type: []const u8, body: []const u8) !void {
    const reason = switch (status) {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        500 => "Internal Server Error",
        else => "OK",
    };

    var header_buf: [512]u8 = undefined;
    const header = try std.fmt.bufPrint(
        &header_buf,
        "HTTP/1.1 {d} {s}\r\n" ++
            "Content-Type: {s}\r\n" ++
            "Content-Length: {d}\r\n" ++
            "Access-Control-Allow-Origin: *\r\n" ++
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
            "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
            "Server: leanShimmy/0.1 (Zig)\r\n" ++
            "\r\n",
        .{ status, reason, content_type, body.len },
    );

    _ = try stream.writeAll(header);
    if (body.len > 0) {
        _ = try stream.writeAll(body);
    }
}

fn badRequest(code: []const u8, msg: []const u8) !Response {
    return errorResponse(code, msg, 400);
}

fn parseContentLength(header: []const u8) ?usize {
    var lines = mem.splitSequence(u8, header, "\r\n");
    while (lines.next()) |line| {
        if (std.ascii.startsWithIgnoreCase(line, "Content-Length:")) {
            const value = mem.trim(u8, line["Content-Length:".len..], " \t");
            return std.fmt.parseInt(usize, value, 10) catch null;
        }
    }
    return null;
}

fn readRequest(stream: net.Stream, max_request_bytes: usize) ![]u8 {
    var buffer = std.ArrayList(u8).empty;
    errdefer buffer.deinit(allocator);

    var temp: [4096]u8 = undefined;
    var header_end: ?usize = null;
    var content_length: usize = 0;

    while (true) {
        const n = try stream.read(&temp);
        if (n == 0) break;
        try buffer.appendSlice(allocator, temp[0..n]);
        if (buffer.items.len > max_request_bytes) {
            return error.RequestTooLarge;
        }

        if (header_end == null) {
            if (mem.indexOf(u8, buffer.items, "\r\n\r\n")) |idx| {
                header_end = idx + 4;
                const header = buffer.items[0..idx];
                content_length = parseContentLength(header) orelse 0;
            }
        }

        if (header_end != null and buffer.items.len >= header_end.? + content_length) {
            break;
        }
    }

    return buffer.toOwnedSlice(allocator);
}

fn handleHealth(engine: *inference.InferenceEngine) !Response {
    const payload = std.fmt.allocPrint(
        allocator,
        "{{\"status\":\"ok\",\"service\":\"leanShimmy\",\"target_lean_version\":\"4.26.0\",\"model_loaded\":{s}}}",
        .{if (engine.isLoaded()) "true" else "false"},
    ) catch return errorResponse("health_failed", "Health check failed", 500);
    return Response{ .status = 200, .body = payload };
}

fn handleVersion(engine: *inference.InferenceEngine) !Response {
    const payload = std.fmt.allocPrint(
        allocator,
        "{{\"service\":\"leanShimmy\",\"version\":\"0.1.0\",\"target_lean_version\":\"4.26.0\",\"model\":\"{s}\"}}",
        .{engine.modelId()},
    ) catch return errorResponse("version_failed", "Version lookup failed", 500);
    return Response{ .status = 200, .body = payload };
}

fn handleNotImplemented() !Response {
    return errorResponse("not_implemented", "Not implemented yet", 501);
}

fn streamChat(
    engine: *inference.InferenceEngine,
    conn_stream: net.Stream,
    prompt: []const u8,
    max_tokens: u32,
    temperature: f32,
) !void {
    const gen = try engine.generate(prompt, max_tokens, temperature);
    defer allocator.free(gen.buf);
    const content = gen.buf[0..gen.used];

    const created = std.time.timestamp();
    const model = engine.modelId();
    var writer = streaming.StreamWriter.init(conn_stream);
    try writer.sendHeaders();

    const Delta = struct {
        role: ?[]const u8 = null,
        content: ?[]const u8 = null,
    };
    const Choice = struct {
        index: u32,
        delta: Delta,
        finish_reason: ?[]const u8 = null,
    };
    const Chunk = struct {
        id: []const u8,
        object: []const u8 = "chat.completion.chunk",
        created: i64,
        model: []const u8,
        choices: []const Choice,
    };

    const stream_id = "chatcmpl-stream";

    const role_chunk = Chunk{
        .id = stream_id,
        .created = created,
        .model = model,
        .choices = &[_]Choice{.{ .index = 0, .delta = .{ .role = "assistant" }, .finish_reason = null }},
    };
    var payload = try encodeJson(role_chunk);
    try writer.sendEvent(payload);
    allocator.free(payload);

    var idx: usize = 0;
    while (idx < content.len) {
        const end = @min(idx + 64, content.len);
        const piece = content[idx..end];
        idx = end;

        const chunk = Chunk{
            .id = stream_id,
            .created = created,
            .model = model,
            .choices = &[_]Choice{.{ .index = 0, .delta = .{ .content = piece }, .finish_reason = null }},
        };
        payload = try encodeJson(chunk);
        try writer.sendEvent(payload);
        allocator.free(payload);
    }

    const done_chunk = Chunk{
        .id = stream_id,
        .created = created,
        .model = model,
        .choices = &[_]Choice{.{ .index = 0, .delta = .{}, .finish_reason = "stop" }},
    };
    payload = try encodeJson(done_chunk);
    try writer.sendEvent(payload);
    allocator.free(payload);
    try writer.sendDone();
}

fn streamCompletion(
    engine: *inference.InferenceEngine,
    conn_stream: net.Stream,
    prompt: []const u8,
    max_tokens: u32,
    temperature: f32,
) !void {
    const gen = try engine.generate(prompt, max_tokens, temperature);
    defer allocator.free(gen.buf);
    const content = gen.buf[0..gen.used];

    const created = std.time.timestamp();
    const model = engine.modelId();
    var writer = streaming.StreamWriter.init(conn_stream);
    try writer.sendHeaders();

    const Choice = struct {
        text: []const u8,
        index: u32,
        finish_reason: ?[]const u8 = null,
    };
    const Chunk = struct {
        id: []const u8,
        object: []const u8 = "text_completion.chunk",
        created: i64,
        model: []const u8,
        choices: []const Choice,
    };

    const stream_id = "cmpl-stream";

    var idx: usize = 0;
    while (idx < content.len) {
        const end = @min(idx + 64, content.len);
        const piece = content[idx..end];
        idx = end;

        const chunk = Chunk{
            .id = stream_id,
            .created = created,
            .model = model,
            .choices = &[_]Choice{.{ .text = piece, .index = 0, .finish_reason = null }},
        };
        const payload = try encodeJson(chunk);
        try writer.sendEvent(payload);
        allocator.free(payload);
    }

    const done_chunk = Chunk{
        .id = stream_id,
        .created = created,
        .model = model,
        .choices = &[_]Choice{.{ .text = "", .index = 0, .finish_reason = "stop" }},
    };
    const payload = try encodeJson(done_chunk);
    try writer.sendEvent(payload);
    allocator.free(payload);
    try writer.sendDone();
}

fn handleChat(engine: *inference.InferenceEngine, body: []const u8, conn_stream: net.Stream, limits: Limits) !HandlerResult {
    const Message = struct {
        role: []const u8 = "",
        content: []const u8 = "",
    };
    const ChatReq = struct {
        model: ?[]const u8 = null,
        prompt: ?[]const u8 = null,
        messages: ?[]const Message = null,
        max_tokens: ?u32 = null,
        temperature: ?f32 = null,
        stream: ?bool = null,
    };

    const req = parseJson(ChatReq, body) catch return .{ .response = try badRequest("invalid_json", "invalid json") };

    var prompt: []const u8 = "";
    var model: []const u8 = engine.modelId();
    var max_tokens: u32 = 256;
    var temperature: f32 = 0.7;
    const stream_requested = req.stream orelse false;
    if (req.model) |m| model = m;
    if (req.prompt) |p| prompt = p;
    if (req.messages) |msgs| {
        if (msgs.len > 0) {
            prompt = msgs[msgs.len - 1].content;
        }
    }
    if (req.max_tokens) |mt| max_tokens = mt;
    if (req.temperature) |t| temperature = t;

    if (prompt.len > limits.max_prompt_bytes) return .{ .response = try badRequest("prompt_too_large", "prompt too large") };
    if (prompt.len == 0) return .{ .response = try badRequest("prompt_required", "prompt required") };
    if (max_tokens == 0 or max_tokens > limits.max_tokens_cap) return .{ .response = try badRequest("invalid_max_tokens", "invalid max_tokens") };
    if (temperature < 0.0 or temperature > 5.0) return .{ .response = try badRequest("invalid_temperature", "invalid temperature") };

    if (!engine.isLoaded() and model.len > 0) {
        engine.loadModel(model) catch {};
    }
    if (!engine.isLoaded()) return .{ .response = try badRequest("model_not_loaded", "model not loaded") };

    if (stream_requested) {
        try streamChat(engine, conn_stream, prompt, max_tokens, temperature);
        return .streamed;
    }

    const gen = try engine.generate(prompt, max_tokens, temperature);
    defer allocator.free(gen.buf);
    const content = gen.buf[0..gen.used];

    const ChatChoice = struct {
        index: u32,
        message: Message,
        finish_reason: []const u8,
    };
    const ChatResponse = struct {
        id: []const u8,
        object: []const u8 = "chat.completion",
        created: i64,
        model: []const u8,
        choices: []const ChatChoice,
    };

    const choice = ChatChoice{
        .index = 0,
        .message = .{ .role = "assistant", .content = content },
        .finish_reason = "stop",
    };
    const choices = [_]ChatChoice{choice};
    const resp = ChatResponse{
        .id = "chatcmpl-stub",
        .created = std.time.timestamp(),
        .model = engine.modelId(),
        .choices = &choices,
    };
    const payload = try encodeJson(resp);
    return .{ .response = Response{ .status = 200, .body = payload } };
}

fn handleCompletion(engine: *inference.InferenceEngine, body: []const u8, conn_stream: net.Stream, limits: Limits) !HandlerResult {
    const CompletionReq = struct {
        model: ?[]const u8 = null,
        prompt: ?[]const u8 = null,
        max_tokens: ?u32 = null,
        temperature: ?f32 = null,
        stream: ?bool = null,
    };

    const req = parseJson(CompletionReq, body) catch return .{ .response = try badRequest("invalid_json", "invalid json") };

    var prompt: []const u8 = "";
    var model: []const u8 = engine.modelId();
    var max_tokens: u32 = 256;
    var temperature: f32 = 0.7;
    const stream_requested = req.stream orelse false;
    if (req.model) |m| model = m;
    if (req.prompt) |p| prompt = p;
    if (req.max_tokens) |mt| max_tokens = mt;
    if (req.temperature) |t| temperature = t;

    if (prompt.len > limits.max_prompt_bytes) return .{ .response = try badRequest("prompt_too_large", "prompt too large") };
    if (prompt.len == 0) return .{ .response = try badRequest("prompt_required", "prompt required") };
    if (max_tokens == 0 or max_tokens > limits.max_tokens_cap) return .{ .response = try badRequest("invalid_max_tokens", "invalid max_tokens") };
    if (temperature < 0.0 or temperature > 5.0) return .{ .response = try badRequest("invalid_temperature", "invalid temperature") };

    if (!engine.isLoaded() and model.len > 0) {
        engine.loadModel(model) catch {};
    }
    if (!engine.isLoaded()) return .{ .response = try badRequest("model_not_loaded", "model not loaded") };

    if (stream_requested) {
        try streamCompletion(engine, conn_stream, prompt, max_tokens, temperature);
        return .streamed;
    }

    const gen = try engine.generate(prompt, max_tokens, temperature);
    defer allocator.free(gen.buf);
    const content = gen.buf[0..gen.used];

    const CompletionChoice = struct {
        index: u32,
        text: []const u8,
        finish_reason: []const u8,
    };
    const CompletionResponse = struct {
        id: []const u8,
        object: []const u8 = "text_completion",
        created: i64,
        model: []const u8,
        choices: []const CompletionChoice,
    };
    const choice = CompletionChoice{
        .index = 0,
        .text = content,
        .finish_reason = "stop",
    };
    const choices = [_]CompletionChoice{choice};
    const resp = CompletionResponse{
        .id = "cmpl-stub",
        .created = std.time.timestamp(),
        .model = engine.modelId(),
        .choices = &choices,
    };
    const payload = try encodeJson(resp);
    return .{ .response = Response{ .status = 200, .body = payload } };
}

fn handleEmbeddings(engine: *inference.InferenceEngine, body: []const u8, limits: Limits) !Response {
    const EmbedReq = struct {
        model: ?[]const u8 = null,
        input: ?[]const u8 = null,
        dimensions: ?u32 = null,
    };

    const req = parseJson(EmbedReq, body) catch return badRequest("invalid_json", "invalid json");

    var input: []const u8 = "";
    var model: []const u8 = engine.modelId();
    if (req.model) |m| model = m;
    if (req.input) |i| input = i;

    if (input.len > limits.max_prompt_bytes) return badRequest("input_too_large", "input too large");
    if (input.len == 0) return badRequest("input_required", "input required");

    if (!engine.isLoaded() and model.len > 0) {
        engine.loadModel(model) catch {};
    }
    if (!engine.isLoaded()) return badRequest("model_not_loaded", "model not loaded");

    const embed_result = try engine.embed(input);
    const embedding = embed_result.buf[0..embed_result.used];
    var buf = std.ArrayList(u8).empty;
    defer buf.deinit(allocator);

    try buf.appendSlice(allocator, "{\"object\":\"list\",\"model\":\"");
    try buf.appendSlice(allocator, engine.modelId());
    try buf.appendSlice(allocator, "\",\"data\":[{\"object\":\"embedding\",\"index\":0,\"embedding\":[");
    var i: usize = 0;
    while (i < embedding.len) : (i += 1) {
        if (i != 0) try buf.append(allocator, ',');
        try std.fmt.format(buf.writer(allocator), "{d}", .{embedding[i]});
    }
    try buf.appendSlice(allocator, "]}]}");

    const payload = try buf.toOwnedSlice(allocator);
    allocator.free(embed_result.buf);
    return Response{ .status = 200, .body = payload };
}

fn handleConnection(connection: net.Server.Connection, pool: *EnginePool, limits: Limits) !void {
    defer connection.stream.close();

    const engine = pool.acquire();
    const request_data = readRequest(connection.stream, limits.max_request_bytes) catch return;
    defer allocator.free(request_data);

    const first_line_end = mem.indexOf(u8, request_data, "\r\n") orelse return;
    const header_end_idx = mem.indexOf(u8, request_data, "\r\n\r\n") orelse return;
    const request_line = request_data[0..first_line_end];

    var parts = mem.splitSequence(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    const clean_path = if (mem.indexOf(u8, path, "?")) |idx| path[0..idx] else path;

    if (mem.eql(u8, method, "OPTIONS")) {
        try sendResponse(connection.stream, 204, "application/json", "");
        return;
    }

    var maybe_response: ?Response = null;
    if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/health")) {
        maybe_response = try handleHealth(engine);
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/version")) {
        maybe_response = try handleVersion(engine);
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/limits")) {
        const payload = try encodeJson(.{
            .max_request_bytes = limits.max_request_bytes,
            .max_prompt_bytes = limits.max_prompt_bytes,
            .max_tokens_cap = limits.max_tokens_cap,
            .max_output_bytes = engine.max_output_bytes,
            .embedding_dims = engine.embedding_dims,
            .engine_pool_size = pool.engines.len,
        });
        maybe_response = Response{ .status = 200, .body = payload };
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/lean4/check")) {
        maybe_response = try handleNotImplemented();
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/lean4/run")) {
        maybe_response = try handleNotImplemented();
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/chat/completions")) {
        const body = request_data[header_end_idx + 4 ..];
        const result = try handleChat(engine, body, connection.stream, limits);
        maybe_response = switch (result) {
            .response => |resp| resp,
            .streamed => null,
        };
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/completions")) {
        const body = request_data[header_end_idx + 4 ..];
        const result = try handleCompletion(engine, body, connection.stream, limits);
        maybe_response = switch (result) {
            .response => |resp| resp,
            .streamed => null,
        };
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/embeddings")) {
        const body = request_data[header_end_idx + 4 ..];
        maybe_response = try handleEmbeddings(engine, body, limits);
    } else {
        maybe_response = try errorResponse("not_found", "Not found", 404);
    }

    if (maybe_response) |response| {
        defer allocator.free(response.body);
        try sendResponse(connection.stream, response.status, response.content_type, response.body);
    }
}

pub fn main() !void {
    defer _ = gpa.deinit();

    const host = getenv("LEANSHIMMY_HOST") orelse default_host;
    const port = parseEnvU16("LEANSHIMMY_PORT", default_port);
    const limits = Limits{
        .max_request_bytes = parseEnvUsize("LEANSHIMMY_MAX_REQUEST_BYTES", default_max_request_bytes),
        .max_prompt_bytes = parseEnvUsize("LEANSHIMMY_MAX_PROMPT_BYTES", default_max_prompt_bytes),
        .max_tokens_cap = parseEnvU32("LEANSHIMMY_MAX_TOKENS", default_max_tokens_cap),
    };
    const pool_size = parseEnvUsize("LEANSHIMMY_ENGINE_POOL_SIZE", default_pool_size);
    var pool = try EnginePool.init(allocator, pool_size);
    defer pool.deinit();

    const lib_path = getenv("LEANSHIMMY_INFERENCE_LIB");
    const model_path = getenv("LEANSHIMMY_MODEL_PATH");
    pool.setupAll(lib_path, model_path);

    const addr = try net.Address.parseIp(host, port);
    var server = try addr.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("leanShimmy server listening on {s}:{d}\n", .{ host, port });
    std.debug.print("engine pool size: {d}\n", .{pool.engines.len});
    std.debug.print("limits -> request: {d} bytes, prompt: {d} bytes, max_tokens: {d}\n", .{ limits.max_request_bytes, limits.max_prompt_bytes, limits.max_tokens_cap });

    while (true) {
        const conn = try server.accept();
        _ = std.Thread.spawn(.{}, handleConnection, .{ conn, &pool, limits }) catch {
            // Fallback to sync handling if thread spawn fails.
            handleConnection(conn, &pool, limits) catch {};
        };
    }
}

test "parseEnvU16 uses default on invalid input" {
    try std.testing.expectEqual(@as(u16, 1234), parseEnvU16("LEANSHIMMY_TEST_PORT", 1234));
}
