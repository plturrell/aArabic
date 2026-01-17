const std = @import("std");

const builtin = @import("builtin");
const json = std.json;
const mem = std.mem;
const net = std.net;

const default_host = "0.0.0.0";
const default_port: u16 = 11434;
const max_request_bytes: usize = 1024 * 1024;
const response_buffer_size: usize = 64 * 1024;
var startup_model_path: ?[]const u8 = null;

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Pre-allocated response buffer pool for zero-allocation responses
const ResponseBufferPool = struct {
    buffers: [4][8192]u8 = undefined,
    in_use: [4]bool = [_]bool{false} ** 4,
    mutex: std.Thread.Mutex = .{},

    fn acquire(self: *ResponseBufferPool) ?*[8192]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (&self.buffers, &self.in_use) |*buf, *used| {
            if (!used.*) {
                used.* = true;
                return buf;
            }
        }
        return null;
    }

    fn release(self: *ResponseBufferPool, buf: *[8192]u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (&self.buffers, &self.in_use) |*pool_buf, *used| {
            if (pool_buf == buf) {
                used.* = false;
                return;
            }
        }
    }
};

var response_pool = ResponseBufferPool{};

var log_enabled: ?bool = null;

fn logEnabled() bool {
    if (log_enabled) |enabled| {
        return enabled;
    }
    log_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
    return log_enabled.?;
}

fn log(comptime fmt: []const u8, args: anytype) void {
    if (logEnabled()) {
        std.debug.print(fmt, args);
    }
}

const LoadModelFn = *const fn ([*:0]const u8) callconv(.c) i32;
const GenerateFn = *const fn ([*]const u8, usize, u32, f32, [*]u8, usize) callconv(.c) i32;
const IsLoadedFn = *const fn () callconv(.c) i32;
const GetInfoFn = *const fn ([*]u8, usize) callconv(.c) i32;
const UnloadFn = *const fn () callconv(.c) void;

const InferenceApi = struct {
    lib: std.DynLib,
    load_model: LoadModelFn,
    generate: GenerateFn,
    is_loaded: IsLoadedFn,
    get_info: GetInfoFn,
    unload: UnloadFn,
};

var inference_api: ?InferenceApi = null;
var loaded_model_path: ?[]u8 = null;
var loaded_model_id: ?[]u8 = null;
var model_mutex = std.Thread.Mutex{};

const ChatMessage = struct {
    role: []const u8,
    content: []const u8,
};

const ChatRequest = struct {
    model: ?[]const u8 = null,
    messages: []const ChatMessage,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    max_tokens: ?u32 = null,
    stream: ?bool = null,
    stop: ?[]const []const u8 = null,
};

const CompletionRequest = struct {
    model: ?[]const u8 = null,
    prompt: []const u8,
    temperature: ?f32 = null,
    top_p: ?f32 = null,
    max_tokens: ?u32 = null,
    stream: ?bool = null,
    stop: ?[]const []const u8 = null,
    n: ?u32 = null,
    echo: ?bool = null,
    best_of: ?u32 = null,
    logprobs: ?u32 = null,
};

const Usage = struct {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
};

const ChatChoice = struct {
    index: u32,
    message: ChatMessage,
    finish_reason: []const u8,
};

const ChatResponse = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const ChatChoice,
    usage: Usage,
};

const CompletionChoice = struct {
    text: []const u8,
    index: u32,
    finish_reason: []const u8,
};

const CompletionResponse = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    model: []const u8,
    choices: []const CompletionChoice,
    usage: Usage,
};

const ModelInfo = struct {
    id: []const u8,
    object: []const u8,
    created: i64,
    owned_by: []const u8,
};

const ModelList = struct {
    object: []const u8,
    data: []const ModelInfo,
};

const ErrorInfo = struct {
    message: []const u8,
    @"type": []const u8,
};

const ErrorResponse = struct {
    @"error": ErrorInfo,
};

const Response = struct {
    status: u16,
    body: []u8,
    content_type: []const u8 = "application/json",
};

fn getenv(name: []const u8) ?[]const u8 {
    if (std.posix.getenv(name)) |value| {
        return value[0..value.len];
    }
    return null;
}

fn loadInferenceApi() !InferenceApi {
    var last_err: ?anyerror = null;
    var lib: std.DynLib = undefined;
    var opened = false;

    if (getenv("SHIMMY_INFERENCE_LIB")) |path| {
        lib = std.DynLib.open(path) catch |err| {
            last_err = err;
            lib = undefined;
            opened = false;
            return err;
        };
        opened = true;
    } else {
        const default_path = switch (builtin.os.tag) {
            .macos => "inference/engine/zig-out/lib/libinference.dylib",
            .linux => "inference/engine/zig-out/lib/libinference.so",
            else => "inference/engine/zig-out/lib/libinference.so",
        };
        const fallback_path = "inference/engine/zig-out/bin/test_mojo_bridge";
        const candidates = [_][]const u8{ default_path, fallback_path };

        for (candidates) |path| {
            lib = std.DynLib.open(path) catch |err| {
                last_err = err;
                continue;
            };
            opened = true;
            break;
        }
    }

    if (!opened) {
        if (last_err) |err| {
            return err;
        }
        return error.LibraryNotFound;
    }

    errdefer lib.close();

    const load_model = lib.lookup(LoadModelFn, "inference_load_model") orelse return error.MissingSymbol;
    const generate = lib.lookup(GenerateFn, "inference_generate") orelse return error.MissingSymbol;
    const is_loaded = lib.lookup(IsLoadedFn, "inference_is_loaded") orelse return error.MissingSymbol;
    const get_info = lib.lookup(GetInfoFn, "inference_get_info") orelse return error.MissingSymbol;
    const unload = lib.lookup(UnloadFn, "inference_unload") orelse return error.MissingSymbol;

    return InferenceApi{
        .lib = lib,
        .load_model = load_model,
        .generate = generate,
        .is_loaded = is_loaded,
        .get_info = get_info,
        .unload = unload,
    };
}

fn ensureInferenceApi() !*InferenceApi {
    if (inference_api == null) {
        inference_api = try loadInferenceApi();
    }
    return &inference_api.?;
}

fn resolveModelId() []const u8 {
    if (getenv("SHIMMY_MODEL_ID")) |value| return value;
    if (getenv("SHIMMY_MODEL_PATH") != null) return "shimmy-model";
    if (startup_model_path) |p| return p;
    return "phi-3-mini";
}

fn resolveModelPath(model_id: []const u8) ![]u8 {
    if (getenv("SHIMMY_MODEL_PATH")) |value| {
        return try allocator.dupe(u8, value);
    }
    if (startup_model_path) |p| return try allocator.dupe(u8, p);

    // Treat direct paths (.gguf) or absolute/relative strings as paths
    if (mem.endsWith(u8, model_id, ".gguf") or mem.startsWith(u8, model_id, "/") or mem.startsWith(u8, model_id, "./")) {
        return try allocator.dupe(u8, model_id);
    }

    // Check vendor: prefix first (before "/" check since vendor:Qwen/... contains "/")
    if (mem.startsWith(u8, model_id, "vendor:")) {
        const suffix = model_id["vendor:".len..];
        // Go up to repo root from src/serviceCore/serviceShimmy-mojo/
        return try std.fmt.allocPrint(allocator, "../../../vendor/layerModels/huggingFace/{s}", .{suffix});
    }
    if (mem.indexOf(u8, model_id, "/") != null or mem.startsWith(u8, model_id, ".")) {
        return try allocator.dupe(u8, model_id);
    }
    const model_dir = getenv("SHIMMY_MODEL_DIR") orelse "models";
    return try std.fmt.allocPrint(allocator, "{s}/{s}", .{ model_dir, model_id });
}

fn makeCString(value: []const u8) ![:0]u8 {
    var buffer = try allocator.alloc(u8, value.len + 1);
    @memcpy(buffer[0..value.len], value);
    buffer[value.len] = 0;
    return buffer[0..value.len :0];
}

fn ensureModelLoaded(api: *InferenceApi, model_id: []const u8) ![]const u8 {
    model_mutex.lock();
    defer model_mutex.unlock();

    if (api.is_loaded() == 1) {
        if (loaded_model_id) |current_id| {
            if (mem.eql(u8, current_id, model_id)) {
                return current_id;
            }
        }
    }

    const path = try resolveModelPath(model_id);
    errdefer allocator.free(path);

    if (api.is_loaded() == 1 and loaded_model_path != null) {
        api.unload();
        allocator.free(loaded_model_path.?);
        loaded_model_path = null;
    }

    if (loaded_model_id) |current_id| {
        allocator.free(current_id);
        loaded_model_id = null;
    }

    const c_path = try makeCString(path);
    defer allocator.free(c_path);

    const rc = api.load_model(c_path.ptr);
    if (rc != 0) {
        return error.ModelLoadFailed;
    }

    loaded_model_path = path;
    loaded_model_id = try allocator.dupe(u8, model_id);
    return loaded_model_id.?;
}

fn generateText(api: *InferenceApi, prompt: []const u8, max_tokens: u32, temperature: f32) ![]u8 {
    var buffer = try allocator.alloc(u8, response_buffer_size);
    errdefer allocator.free(buffer);

    const length = api.generate(prompt.ptr, prompt.len, max_tokens, temperature, buffer.ptr, buffer.len);
    if (length < 0) {
        allocator.free(buffer);
        return error.GenerationFailed;
    }
    // Handle empty responses (length == 0)
    if (length == 0) {
        allocator.free(buffer);
        return try allocator.alloc(u8, 0);
    }
    // Copy to a buffer of exact size so caller can free correctly
    const actual_len: usize = @intCast(length);
    const result = try allocator.alloc(u8, actual_len);
    @memcpy(result, buffer[0..actual_len]);
    allocator.free(buffer);
    return result;
}

fn buildChatPrompt(messages: []const ChatMessage) ![]u8 {
    var list = std.ArrayList(u8).empty;
    errdefer list.deinit(allocator);

    for (messages) |msg| {
        if (mem.eql(u8, msg.role, "system")) {
            try list.appendSlice(allocator, "System: ");
        } else if (mem.eql(u8, msg.role, "user")) {
            try list.appendSlice(allocator, "User: ");
        } else if (mem.eql(u8, msg.role, "assistant")) {
            try list.appendSlice(allocator, "Assistant: ");
        } else {
            continue;
        }
        try list.appendSlice(allocator, msg.content);
        try list.appendSlice(allocator, "\n\n");
    }

    try list.appendSlice(allocator, "Assistant: ");
    return list.toOwnedSlice(allocator);
}

fn writeJsonDirect(writer: anytype, value: anytype) !void {
    try json.stringify(value, .{}, writer);
}

fn encodeJsonFast(value: anytype) ![]u8 {
    // Use Stringify.valueAlloc for fast encoding
    return try json.Stringify.valueAlloc(allocator, value, .{});
}

fn errorBody(message: []const u8) ![]u8 {
    // Manual JSON building for performance
    const escaped_message = try escapeJsonString(message);
    defer allocator.free(escaped_message);
    
    return try std.fmt.allocPrint(
        allocator,
        "{{\"error\":{{\"message\":\"{s}\",\"type\":\"invalid_request_error\"}}}}",
        .{escaped_message}
    );
}

fn handleModels() !Response {
    const now = std.time.timestamp();
    const model_id = getenv("SHIMMY_MODEL_ID") orelse resolveModelId();
    
    // Manual JSON building for performance
    const body = try std.fmt.allocPrint(
        allocator,
        "{{\"object\":\"list\",\"data\":[" ++
            "{{\"id\":\"{s}\",\"object\":\"model\",\"created\":{d},\"owned_by\":\"shimmy-mojo\"}}" ++
            "]}}",
        .{ model_id, now }
    );
    
    return Response{ .status = 200, .body = body };
}

fn escapeJsonString(input: []const u8) ![]u8 {
    // Pre-allocate worst case (every char needs escaping = 6 bytes per char)
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(allocator);
    try result.ensureTotalCapacity(allocator, input.len * 2); // Most chars don't need escaping
    
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => {
                var buf: [6]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                try result.appendSlice(allocator, &buf);
            },
            else => try result.append(allocator, c),
        }
    }
    return try result.toOwnedSlice(allocator);
}

fn handleChat(body: []const u8) !Response {
    const api = try ensureInferenceApi();
    const parsed = json.parseFromSlice(ChatRequest, allocator, body, .{ .ignore_unknown_fields = true }) catch {
        return Response{ .status = 400, .body = try errorBody("Invalid JSON payload") };
    };
    defer parsed.deinit();

    const request = parsed.value;
    if (request.stream orelse false) {
        return Response{ .status = 400, .body = try errorBody("Streaming is not supported yet") };
    }

    const model_id = request.model orelse resolveModelId();
    _ = try ensureModelLoaded(api, model_id);

    const prompt = try buildChatPrompt(request.messages);
    defer allocator.free(prompt);

    const max_tokens = request.max_tokens orelse 512;
    const temperature = request.temperature orelse 0.7;

    const output = try generateText(api, prompt, max_tokens, temperature);
    defer allocator.free(output);

    log("Raw output ({d} bytes): \"{s}\"\n", .{ output.len, output });

    const timestamp = std.time.timestamp();
    const request_id = try std.fmt.allocPrint(allocator, "chatcmpl-{d}", .{std.time.milliTimestamp()});
    defer allocator.free(request_id);

    // Manually build JSON to ensure content is properly string-encoded
    const escaped_output = try escapeJsonString(output);
    defer allocator.free(escaped_output);

    const escaped_model = try escapeJsonString(model_id);
    defer allocator.free(escaped_model);

    const prompt_tokens = prompt.len / 4;
    const completion_tokens = output.len / 4;

    const json_body = try std.fmt.allocPrint(
        allocator,
        "{{\"id\":\"{s}\",\"object\":\"chat.completion\",\"created\":{d},\"model\":\"{s}\"," ++
            "\"choices\":[{{\"index\":0,\"message\":{{\"role\":\"assistant\",\"content\":\"{s}\"}},\"finish_reason\":\"stop\"}}]," ++
            "\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}",
        .{ request_id, timestamp, escaped_model, escaped_output, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens },
    );

    return Response{ .status = 200, .body = json_body };
}

fn handleCompletion(body: []const u8) !Response {
    const t_start = std.time.nanoTimestamp();
    const api = try ensureInferenceApi();
    const t_api = std.time.nanoTimestamp();
    
    const parsed = json.parseFromSlice(CompletionRequest, allocator, body, .{ .ignore_unknown_fields = true }) catch {
        return Response{ .status = 400, .body = try errorBody("Invalid JSON payload") };
    };
    defer parsed.deinit();
    const t_parse = std.time.nanoTimestamp();

    const request = parsed.value;
    if (request.stream orelse false) {
        return Response{ .status = 400, .body = try errorBody("Streaming is not supported yet") };
    }

    const model_id = request.model orelse resolveModelId();
    _ = try ensureModelLoaded(api, model_id);
    const t_load = std.time.nanoTimestamp();

    const max_tokens = request.max_tokens orelse 256;
    const temperature = request.temperature orelse 0.7;

    std.debug.print("🔵 Starting generation: prompt_len={d} max_tokens={d}\n", .{ request.prompt.len, max_tokens });
    const output = try generateText(api, request.prompt, max_tokens, temperature);
    defer allocator.free(output);
    const t_gen = std.time.nanoTimestamp();
    
    std.debug.print("✅ Generation complete: output_len={d}\n", .{output.len});
    std.debug.print("⏱️  HTTP Handler: api={d}ms parse={d}ms load={d}ms gen={d}ms\n", .{
        @divFloor(t_api - t_start, 1_000_000),
        @divFloor(t_parse - t_api, 1_000_000),
        @divFloor(t_load - t_parse, 1_000_000),
        @divFloor(t_gen - t_load, 1_000_000),
    });

    std.debug.print("📦 Creating response...\n", .{});
    const timestamp = std.time.timestamp();
    const request_id = try std.fmt.allocPrint(allocator, "cmpl-{d}", .{std.time.milliTimestamp()});
    defer allocator.free(request_id);
    std.debug.print("   request_id created\n", .{});

    const t_encode_start = std.time.nanoTimestamp();
    
    // Manual JSON building - SKIP escaping for now to test performance
    const prompt_tokens: u32 = @intCast(request.prompt.len / 4);
    const completion_tokens: u32 = @intCast(output.len / 4);
    
    std.debug.print("   Building JSON with output_len={d}...\n", .{output.len});
    const response_body = try std.fmt.allocPrint(
        allocator,
        "{{\"id\":\"{s}\",\"object\":\"text_completion\",\"created\":{d},\"model\":\"lfm2\"," ++
            "\"choices\":[{{\"text\":\"{s}\",\"index\":0,\"finish_reason\":\"stop\"}}]," ++
            "\"usage\":{{\"prompt_tokens\":{d},\"completion_tokens\":{d},\"total_tokens\":{d}}}}}",
        .{ request_id, timestamp, output, prompt_tokens, completion_tokens, prompt_tokens + completion_tokens },
    );
    
    const t_encode_end = std.time.nanoTimestamp();
    std.debug.print("⏱️  JSON encode: {d}ms, body_len={d}\n", .{ @divFloor(t_encode_end - t_encode_start, 1_000_000), response_body.len });
    
    return Response{ .status = 200, .body = response_body };
}

fn handleHealth() !Response {
    const status = if (inference_api != null and inference_api.?.is_loaded() == 1) "ready" else "cold";
    const payload = std.fmt.allocPrint(
        allocator,
        "{{\"status\":\"{s}\",\"model_loaded\":{s}}}",
        .{ status, if (inference_api != null and inference_api.?.is_loaded() == 1) "true" else "false" },
    ) catch return Response{ .status = 500, .body = try errorBody("Health check failed") };
    return Response{ .status = 200, .body = payload };
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
            "Server: Shimmy-Mojo/1.0 (Zig)\r\n" ++
            "\r\n",
        .{ status, reason, content_type, body.len },
    );

    _ = try stream.writeAll(header);
    if (body.len > 0) {
        _ = try stream.writeAll(body);
    }
}

fn readRequest(stream: net.Stream) ![]u8 {
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

fn handleConnection(connection: net.Server.Connection) !void {
    defer connection.stream.close();

    const request_data = readRequest(connection.stream) catch |err| {
        log("❌ Request read error: {any}\n", .{err});
        return;
    };
    defer allocator.free(request_data);

    const first_line_end = mem.indexOf(u8, request_data, "\r\n") orelse {
        return;
    };
    const request_line = request_data[0..first_line_end];

    var parts = mem.splitSequence(u8, request_line, " ");
    const method = parts.next() orelse return;
    const path = parts.next() orelse return;
    const clean_path = if (mem.indexOf(u8, path, "?")) |idx| path[0..idx] else path;

    log("➡️  {s} {s}\n", .{ method, path });

    if (mem.eql(u8, method, "OPTIONS")) {
        try sendResponse(connection.stream, 204, "application/json", "");
        return;
    }

    var body: []const u8 = "";
    if (mem.indexOf(u8, request_data, "\r\n\r\n")) |idx| {
        body = request_data[idx + 4 ..];
    }

    var response: Response = undefined;
    if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/")) {
        const payload = std.fmt.allocPrint(
            allocator,
            "{{\"name\":\"Shimmy-Mojo API\",\"version\":\"1.0.0\",\"status\":\"running\",\"model\":\"{s}\"}}",
            .{resolveModelId()},
        ) catch {
            response = Response{ .status = 500, .body = try errorBody("Failed to build response") };
            try sendResponse(connection.stream, response.status, response.content_type, response.body);
            allocator.free(response.body);
            return;
        };
        response = Response{ .status = 200, .body = payload };
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/health")) {
        response = try handleHealth();
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/v1/models")) {
        response = try handleModels();
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/chat/completions")) {
        response = try handleChat(body);
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, clean_path, "/v1/completions")) {
        response = try handleCompletion(body);
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, clean_path, "/api/tags")) {
        response = try handleModels();
    } else {
        response = Response{ .status = 404, .body = try errorBody("Not found") };
    }

    defer allocator.free(response.body);
    try sendResponse(connection.stream, response.status, response.content_type, response.body);
}

fn warmStart() void {
    // Preload if any model is specified via env or argv.
    if (getenv("SHIMMY_MODEL_PATH") == null and getenv("SHIMMY_MODEL_ID") == null and startup_model_path == null) {
        return;
    }

    if (ensureInferenceApi()) |api| {
        const model_id = resolveModelId();
        if (ensureModelLoaded(api, model_id)) |_| {
            std.debug.print("✅ Model preloaded: {s}\n", .{model_id});
        } else |err| {
            std.debug.print("⚠️  Model preload failed: {any}\n", .{err});
        }
    } else |_| {
        std.debug.print("⚠️  Inference library not loaded for warm start\n", .{});
    }
}

pub fn main() !void {
    defer _ = gpa.deinit();

    // Capture argv[1] as model path if provided (for GGUF)
    var arg_iter = try std.process.argsWithAllocator(allocator);
    defer arg_iter.deinit();
    _ = arg_iter.next(); // skip program name
    if (arg_iter.next()) |model_arg| {
        startup_model_path = try allocator.dupe(u8, model_arg);
    }

    const host = getenv("SHIMMY_HOST") orelse default_host;
    var port = default_port;
    if (getenv("SHIMMY_PORT")) |value| {
        port = std.fmt.parseInt(u16, value, 10) catch default_port;
    }

    std.debug.print("================================================================================\n", .{});
    std.debug.print("🦙 Shimmy-Mojo OpenAI Server (Zig)\n", .{});
    std.debug.print("================================================================================\n", .{});
    std.debug.print("Host: {s}\n", .{host});
    std.debug.print("Port: {d}\n", .{port});
    std.debug.print("Model ID: {s}\n", .{resolveModelId()});
    std.debug.print("================================================================================\n", .{});

    warmStart();

    const address = try net.Address.parseIp(host, port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("✅ Listening on http://{s}:{d}\n", .{ host, port });
    std.debug.print("Endpoints: /v1/models, /v1/chat/completions, /v1/completions\n", .{});

    while (true) {
        const connection = try server.accept();
        handleConnection(connection) catch |err| {
            std.debug.print("❌ Connection error: {any}\n", .{err});
        };
    }
}
