//! Server-Sent Events (SSE) Streaming for LLM Responses
//! Provides SSE streaming support for nWorkflow's LLM integration
//!
//! Features:
//! - SSE message formatting (OpenAI-compatible)
//! - StreamWriter for sending SSE events
//! - ChunkParser for parsing streaming responses
//! - Token counting during streaming
//! - Error handling for upstream/client disconnects

const std = @import("std");
const net = std.net;
const mem = std.mem;
const Allocator = mem.Allocator;
const json = std.json;

/// SSE-related errors
pub const StreamingError = error{
    ConnectionClosed,
    UpstreamDisconnect,
    ClientDisconnect,
    Timeout,
    InvalidChunk,
    ParseError,
    WriteFailed,
    BufferOverflow,
    InvalidJsonResponse,
    RetryExhausted,
};

/// Token usage statistics for streaming
pub const StreamingTokenUsage = struct {
    prompt_tokens: usize = 0,
    completion_tokens: usize = 0,
    total_tokens: usize = 0,

    /// Estimate tokens from text (approximate: ~4 chars per token)
    pub fn estimateFromText(text: []const u8) usize {
        return (text.len + 3) / 4;
    }

    /// Add completion tokens
    pub fn addCompletionTokens(self: *StreamingTokenUsage, count: usize) void {
        self.completion_tokens += count;
        self.total_tokens = self.prompt_tokens + self.completion_tokens;
    }

    /// Create JSON representation
    pub fn toJson(self: *const StreamingTokenUsage, allocator: Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator,
            \\{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}
        , .{ self.prompt_tokens, self.completion_tokens, self.total_tokens });
    }
};

/// Delta content in a streaming chunk
pub const DeltaContent = struct {
    role: ?[]const u8 = null,
    content: ?[]const u8 = null,
};

/// Choice in a streaming chunk
pub const StreamChoice = struct {
    index: u32 = 0,
    delta: DeltaContent = .{},
    finish_reason: ?[]const u8 = null,
};

/// Chat completion chunk (OpenAI-compatible format)
pub const ChatChunk = struct {
    id: []const u8,
    object: []const u8 = "chat.completion.chunk",
    created: i64,
    model: []const u8,
    choices: []const StreamChoice,
    usage: ?StreamingTokenUsage = null,

    /// Serialize chunk to JSON
    pub fn toJson(self: *const ChatChunk, allocator: Allocator) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        try result.appendSlice("{\"id\":\"");
        try result.appendSlice(self.id);
        try result.appendSlice("\",\"object\":\"");
        try result.appendSlice(self.object);
        try result.appendSlice("\",\"created\":");

        var num_buf: [32]u8 = undefined;
        const created_str = std.fmt.bufPrint(&num_buf, "{d}", .{self.created}) catch "0";
        try result.appendSlice(created_str);

        try result.appendSlice(",\"model\":\"");
        try result.appendSlice(self.model);
        try result.appendSlice("\",\"choices\":[");

        for (self.choices, 0..) |choice, i| {
            if (i > 0) try result.append(',');
            try result.appendSlice("{\"index\":");
            const idx_str = std.fmt.bufPrint(&num_buf, "{d}", .{choice.index}) catch "0";
            try result.appendSlice(idx_str);
            try result.appendSlice(",\"delta\":{");

            var has_content = false;
            if (choice.delta.role) |role| {
                try result.appendSlice("\"role\":\"");
                try result.appendSlice(role);
                try result.append('"');
                has_content = true;
            }
            if (choice.delta.content) |content| {
                if (has_content) try result.append(',');
                try result.appendSlice("\"content\":\"");
                // Escape special characters in content
                for (content) |c| {
                    switch (c) {
                        '"' => try result.appendSlice("\\\""),
                        '\\' => try result.appendSlice("\\\\"),
                        '\n' => try result.appendSlice("\\n"),
                        '\r' => try result.appendSlice("\\r"),
                        '\t' => try result.appendSlice("\\t"),
                        else => try result.append(c),
                    }
                }
                try result.append('"');
            }
            try result.appendSlice("},\"finish_reason\":");
            if (choice.finish_reason) |reason| {
                try result.append('"');
                try result.appendSlice(reason);
                try result.append('"');
            } else {
                try result.appendSlice("null");
            }
            try result.append('}');
        }

        try result.appendSlice("]");

        // Add usage if present (typically in final chunk)
        if (self.usage) |usage| {
            try result.appendSlice(",\"usage\":{\"prompt_tokens\":");
            const pt_str = std.fmt.bufPrint(&num_buf, "{d}", .{usage.prompt_tokens}) catch "0";
            try result.appendSlice(pt_str);
            try result.appendSlice(",\"completion_tokens\":");
            const ct_str = std.fmt.bufPrint(&num_buf, "{d}", .{usage.completion_tokens}) catch "0";
            try result.appendSlice(ct_str);
            try result.appendSlice(",\"total_tokens\":");
            const tt_str = std.fmt.bufPrint(&num_buf, "{d}", .{usage.total_tokens}) catch "0";
            try result.appendSlice(tt_str);
            try result.appendSlice("}");
        }

        try result.append('}');
        return result.toOwnedSlice();
    }
};

/// SSE Headers for streaming responses
pub const SSEHeaders = struct {
    /// Standard SSE response headers
    pub const response_headers =
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: text/event-stream\r\n" ++
        "Cache-Control: no-cache\r\n" ++
        "Connection: keep-alive\r\n" ++
        "X-Accel-Buffering: no\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";

    /// Generate custom SSE headers with additional fields
    pub fn generateHeaders(allocator: Allocator, extra_headers: ?[]const [2][]const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        errdefer result.deinit();

        try result.appendSlice("HTTP/1.1 200 OK\r\n");
        try result.appendSlice("Content-Type: text/event-stream\r\n");
        try result.appendSlice("Cache-Control: no-cache\r\n");
        try result.appendSlice("Connection: keep-alive\r\n");
        try result.appendSlice("X-Accel-Buffering: no\r\n");
        try result.appendSlice("Access-Control-Allow-Origin: *\r\n");
        try result.appendSlice("Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n");
        try result.appendSlice("Access-Control-Allow-Headers: Content-Type, Authorization\r\n");

        if (extra_headers) |headers| {
            for (headers) |header| {
                try result.appendSlice(header[0]);
                try result.appendSlice(": ");
                try result.appendSlice(header[1]);
                try result.appendSlice("\r\n");
            }
        }

        try result.appendSlice("\r\n");
        return result.toOwnedSlice();
    }
};

/// StreamWriter for sending SSE events
pub const StreamWriter = struct {
    stream: net.Stream,
    allocator: Allocator,
    bytes_written: usize = 0,
    events_sent: usize = 0,
    last_error: ?StreamingError = null,

    /// Initialize a new StreamWriter
    pub fn init(stream: net.Stream, allocator: Allocator) StreamWriter {
        return .{
            .stream = stream,
            .allocator = allocator,
        };
    }

    /// Send SSE headers to start the stream
    pub fn sendHeaders(self: *StreamWriter) !void {
        self.stream.writeAll(SSEHeaders.response_headers) catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };
        self.bytes_written += SSEHeaders.response_headers.len;
    }

    /// Send custom headers
    pub fn sendCustomHeaders(self: *StreamWriter, extra_headers: ?[]const [2][]const u8) !void {
        const headers = try SSEHeaders.generateHeaders(self.allocator, extra_headers);
        defer self.allocator.free(headers);

        self.stream.writeAll(headers) catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };
        self.bytes_written += headers.len;
    }

    /// Send a data event with the given payload
    pub fn sendEvent(self: *StreamWriter, data: []const u8) !void {
        // Format: "data: {payload}\n\n"
        self.stream.writeAll("data: ") catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };

        self.stream.writeAll(data) catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };

        self.stream.writeAll("\n\n") catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };

        self.bytes_written += 6 + data.len + 2; // "data: " + data + "\n\n"
        self.events_sent += 1;
    }

    /// Send a named event with data
    pub fn sendNamedEvent(self: *StreamWriter, event_name: []const u8, data: []const u8) !void {
        // Format: "event: {name}\ndata: {payload}\n\n"
        self.stream.writeAll("event: ") catch return StreamingError.WriteFailed;
        self.stream.writeAll(event_name) catch return StreamingError.WriteFailed;
        self.stream.writeAll("\n") catch return StreamingError.WriteFailed;
        try self.sendEvent(data);
    }

    /// Send a chat completion chunk
    pub fn sendChunk(self: *StreamWriter, chunk: *const ChatChunk) !void {
        const json_data = try chunk.toJson(self.allocator);
        defer self.allocator.free(json_data);
        try self.sendEvent(json_data);
    }

    /// Send the [DONE] marker to signal end of stream
    pub fn sendDone(self: *StreamWriter) !void {
        self.stream.writeAll("data: [DONE]\n\n") catch |err| {
            self.last_error = .WriteFailed;
            return switch (err) {
                error.BrokenPipe, error.ConnectionResetByPeer => StreamingError.ClientDisconnect,
                else => StreamingError.WriteFailed,
            };
        };
        self.bytes_written += 14; // "data: [DONE]\n\n"
        self.events_sent += 1;
    }

    /// Send a comment (for keep-alive)
    pub fn sendComment(self: *StreamWriter, comment: []const u8) !void {
        self.stream.writeAll(": ") catch return StreamingError.WriteFailed;
        self.stream.writeAll(comment) catch return StreamingError.WriteFailed;
        self.stream.writeAll("\n") catch return StreamingError.WriteFailed;
        self.bytes_written += 2 + comment.len + 1;
    }

    /// Flush the stream (no-op for net.Stream but included for interface consistency)
    pub fn flush(self: *StreamWriter) !void {
        _ = self;
        // net.Stream doesn't have explicit flush, writes are immediate
    }

    /// Get statistics about the stream
    pub fn getStats(self: *const StreamWriter) struct { bytes: usize, events: usize } {
        return .{ .bytes = self.bytes_written, .events = self.events_sent };
    }
};


/// ChunkParser for parsing streaming response chunks
pub const ChunkParser = struct {
    allocator: Allocator,
    buffer: std.ArrayList(u8),
    chunks_parsed: usize = 0,

    /// Initialize a new ChunkParser
    pub fn init(allocator: Allocator) ChunkParser {
        return .{
            .allocator = allocator,
            .buffer = std.ArrayList(u8).init(allocator),
        };
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *ChunkParser) void {
        self.buffer.deinit();
    }

    /// Parse a stream chunk from SSE data line
    /// Returns null if this is a [DONE] marker
    pub fn parseStreamChunk(self: *ChunkParser, data: []const u8) !?ChatChunk {
        // Check for [DONE] marker
        if (mem.eql(u8, mem.trim(u8, data, " \t\r\n"), "[DONE]")) {
            return null;
        }

        // Remove "data: " prefix if present
        const json_data = if (mem.startsWith(u8, data, "data: "))
            data[6..]
        else
            data;

        // Parse JSON
        return try self.parseChunkJson(json_data);
    }

    /// Parse chunk from JSON string
    fn parseChunkJson(self: *ChunkParser, json_str: []const u8) !ChatChunk {
        const parsed = json.parseFromSlice(json.Value, self.allocator, json_str, .{}) catch {
            return StreamingError.InvalidJsonResponse;
        };
        defer parsed.deinit();

        const root = parsed.value;
        if (root != .object) return StreamingError.InvalidChunk;

        // Extract fields
        const id = if (root.object.get("id")) |v| switch (v) {
            .string => |s| s,
            else => "unknown",
        } else "unknown";

        const object_type = if (root.object.get("object")) |v| switch (v) {
            .string => |s| s,
            else => "chat.completion.chunk",
        } else "chat.completion.chunk";

        const created: i64 = if (root.object.get("created")) |v| switch (v) {
            .integer => |i| i,
            else => std.time.timestamp(),
        } else std.time.timestamp();

        const model = if (root.object.get("model")) |v| switch (v) {
            .string => |s| s,
            else => "unknown",
        } else "unknown";

        // Parse choices array
        var choices_list = std.ArrayList(StreamChoice).init(self.allocator);
        defer choices_list.deinit();

        if (root.object.get("choices")) |choices_val| {
            if (choices_val == .array) {
                for (choices_val.array.items) |choice_val| {
                    if (choice_val == .object) {
                        const choice = try self.parseChoice(choice_val.object);
                        try choices_list.append(choice);
                    }
                }
            }
        }

        self.chunks_parsed += 1;

        return ChatChunk{
            .id = try self.allocator.dupe(u8, id),
            .object = try self.allocator.dupe(u8, object_type),
            .created = created,
            .model = try self.allocator.dupe(u8, model),
            .choices = try choices_list.toOwnedSlice(),
        };
    }

    /// Parse a single choice from JSON object
    fn parseChoice(self: *ChunkParser, obj: json.ObjectMap) !StreamChoice {
        const index: u32 = if (obj.get("index")) |v| switch (v) {
            .integer => |i| @intCast(i),
            else => 0,
        } else 0;

        var delta = DeltaContent{};

        if (obj.get("delta")) |delta_val| {
            if (delta_val == .object) {
                if (delta_val.object.get("role")) |role_val| {
                    if (role_val == .string) {
                        delta.role = try self.allocator.dupe(u8, role_val.string);
                    }
                }
                if (delta_val.object.get("content")) |content_val| {
                    if (content_val == .string) {
                        delta.content = try self.allocator.dupe(u8, content_val.string);
                    }
                }
            }
        }

        var finish_reason: ?[]const u8 = null;
        if (obj.get("finish_reason")) |fr_val| {
            if (fr_val == .string) {
                finish_reason = try self.allocator.dupe(u8, fr_val.string);
            }
        }

        return StreamChoice{
            .index = index,
            .delta = delta,
            .finish_reason = finish_reason,
        };
    }

    /// Extract delta content from a chunk
    pub fn extractDeltaContent(chunk: *const ChatChunk) ?[]const u8 {
        if (chunk.choices.len > 0) {
            return chunk.choices[0].delta.content;
        }
        return null;
    }

    /// Check if chunk indicates completion
    pub fn isFinished(chunk: *const ChatChunk) bool {
        if (chunk.choices.len > 0) {
            return chunk.choices[0].finish_reason != null;
        }
        return false;
    }

    /// Get finish reason from chunk
    pub fn getFinishReason(chunk: *const ChatChunk) ?[]const u8 {
        if (chunk.choices.len > 0) {
            return chunk.choices[0].finish_reason;
        }
        return null;
    }
};


/// Configuration for streaming chat handler
pub const StreamingConfig = struct {
    /// Upstream LLM endpoint (e.g., "localhost:11434")
    upstream_host: []const u8 = "localhost",
    upstream_port: u16 = 11434,
    /// API path for chat completions
    api_path: []const u8 = "/v1/chat/completions",
    /// API key for authentication
    api_key: ?[]const u8 = null,
    /// Timeout in milliseconds
    timeout_ms: u32 = 60000,
    /// Maximum retries for transient failures
    max_retries: u32 = 3,
    /// Retry backoff base in milliseconds
    retry_backoff_ms: u32 = 1000,
    /// Chunk size for reading upstream response
    chunk_size: usize = 4096,
};

/// StreamingChatHandler - handles streaming chat completions
pub const StreamingChatHandler = struct {
    allocator: Allocator,
    config: StreamingConfig,
    token_usage: StreamingTokenUsage,
    is_connected: bool = false,

    /// Initialize a new StreamingChatHandler
    pub fn init(allocator: Allocator, config: StreamingConfig) StreamingChatHandler {
        return .{
            .allocator = allocator,
            .config = config,
            .token_usage = .{},
        };
    }

    /// Handle a streaming chat completion request
    pub fn handleStreamingChat(
        self: *StreamingChatHandler,
        request_body: []const u8,
        client_stream: net.Stream,
    ) !void {
        var writer = StreamWriter.init(client_stream, self.allocator);
        var parser = ChunkParser.init(self.allocator);
        defer parser.deinit();

        // Send SSE headers to client
        try writer.sendHeaders();
        self.is_connected = true;

        // Connect to upstream with retry logic
        var retry_count: u32 = 0;
        var upstream_conn: ?net.Stream = null;

        while (retry_count < self.config.max_retries) : (retry_count += 1) {
            upstream_conn = self.connectToUpstream() catch |err| {
                if (retry_count + 1 < self.config.max_retries) {
                    // Exponential backoff
                    const backoff = self.config.retry_backoff_ms * (@as(u32, 1) << @intCast(retry_count));
                    std.time.sleep(backoff * std.time.ns_per_ms);
                    continue;
                }
                // Send error event to client
                try writer.sendEvent("{\"error\":\"Failed to connect to upstream LLM service\"}");
                try writer.sendDone();
                return err;
            };
            break;
        }

        const upstream = upstream_conn orelse return StreamingError.UpstreamDisconnect;
        defer upstream.close();

        // Send request to upstream
        try self.sendUpstreamRequest(upstream, request_body);

        // Read and forward response chunks
        try self.forwardStreamingResponse(upstream, &writer, &parser);

        // Send final usage stats if available
        if (self.token_usage.total_tokens > 0) {
            const usage_json = try self.token_usage.toJson(self.allocator);
            defer self.allocator.free(usage_json);
            // Include usage in a final metadata event
            var meta_buf: [256]u8 = undefined;
            const meta = std.fmt.bufPrint(&meta_buf, "{{\"type\":\"usage\",\"data\":{s}}}", .{usage_json}) catch "";
            if (meta.len > 0) {
                writer.sendEvent(meta) catch {};
            }
        }

        // Send [DONE] marker
        try writer.sendDone();
        self.is_connected = false;
    }

    /// Connect to upstream LLM service
    fn connectToUpstream(self: *StreamingChatHandler) !net.Stream {
        const address = net.Address.parseIp(self.config.upstream_host, self.config.upstream_port) catch {
            // Try resolving hostname
            const addresses = net.getAddressList(self.allocator, self.config.upstream_host, self.config.upstream_port) catch {
                return StreamingError.UpstreamDisconnect;
            };
            defer addresses.deinit();

            if (addresses.addrs.len == 0) {
                return StreamingError.UpstreamDisconnect;
            }
            return net.tcpConnectToAddress(addresses.addrs[0]) catch {
                return StreamingError.UpstreamDisconnect;
            };
        };

        return net.tcpConnectToAddress(address) catch {
            return StreamingError.UpstreamDisconnect;
        };
    }

    /// Send HTTP request to upstream
    fn sendUpstreamRequest(self: *StreamingChatHandler, upstream: net.Stream, body: []const u8) !void {
        // Build HTTP request
        var request_buf: [8192]u8 = undefined;
        var pos: usize = 0;

        // Request line
        const req_line = std.fmt.bufPrint(request_buf[pos..], "POST {s} HTTP/1.1\r\n", .{self.config.api_path}) catch return StreamingError.BufferOverflow;
        pos += req_line.len;

        // Host header
        const host_hdr = std.fmt.bufPrint(request_buf[pos..], "Host: {s}:{d}\r\n", .{ self.config.upstream_host, self.config.upstream_port }) catch return StreamingError.BufferOverflow;
        pos += host_hdr.len;

        // Content headers
        const content_type = "Content-Type: application/json\r\n";
        @memcpy(request_buf[pos..][0..content_type.len], content_type);
        pos += content_type.len;

        const content_len = std.fmt.bufPrint(request_buf[pos..], "Content-Length: {d}\r\n", .{body.len}) catch return StreamingError.BufferOverflow;
        pos += content_len.len;

        // Authorization if provided
        if (self.config.api_key) |key| {
            const auth = std.fmt.bufPrint(request_buf[pos..], "Authorization: Bearer {s}\r\n", .{key}) catch return StreamingError.BufferOverflow;
            pos += auth.len;
        }

        // End of headers
        const end_hdr = "\r\n";
        @memcpy(request_buf[pos..][0..end_hdr.len], end_hdr);
        pos += end_hdr.len;

        // Send headers
        upstream.writeAll(request_buf[0..pos]) catch {
            return StreamingError.UpstreamDisconnect;
        };

        // Send body
        upstream.writeAll(body) catch {
            return StreamingError.UpstreamDisconnect;
        };
    }

    /// Forward streaming response from upstream to client
    fn forwardStreamingResponse(
        self: *StreamingChatHandler,
        upstream: net.Stream,
        writer: *StreamWriter,
        parser: *ChunkParser,
    ) !void {
        var read_buf: [4096]u8 = undefined;
        var line_buf = std.ArrayList(u8).init(self.allocator);
        defer line_buf.deinit();

        var headers_done = false;

        while (true) {
            const bytes_read = upstream.read(&read_buf) catch |err| {
                return switch (err) {
                    error.ConnectionResetByPeer, error.BrokenPipe => StreamingError.UpstreamDisconnect,
                    else => StreamingError.UpstreamDisconnect,
                };
            };

            if (bytes_read == 0) break; // EOF

            // Process received data
            var data = read_buf[0..bytes_read];

            // Skip HTTP headers if not done yet
            if (!headers_done) {
                if (mem.indexOf(u8, data, "\r\n\r\n")) |header_end| {
                    headers_done = true;
                    data = data[header_end + 4 ..];
                } else {
                    continue;
                }
            }

            // Process SSE lines
            try line_buf.appendSlice(data);

            // Process complete lines
            while (mem.indexOf(u8, line_buf.items, "\n")) |nl_pos| {
                const line = line_buf.items[0..nl_pos];

                // Skip empty lines
                if (line.len == 0 or (line.len == 1 and line[0] == '\r')) {
                    _ = line_buf.orderedRemove(0);
                    continue;
                }

                // Process data line
                if (mem.startsWith(u8, line, "data: ")) {
                    const event_data = mem.trimRight(u8, line[6..], "\r");

                    // Check for [DONE]
                    if (mem.eql(u8, event_data, "[DONE]")) {
                        // Clear processed line
                        for (0..nl_pos + 1) |_| {
                            _ = line_buf.orderedRemove(0);
                        }
                        return;
                    }

                    // Parse and forward chunk
                    if (parser.parseStreamChunk(event_data)) |maybe_chunk| {
                        if (maybe_chunk) |chunk| {
                            // Track tokens
                            if (ChunkParser.extractDeltaContent(&chunk)) |content| {
                                self.token_usage.addCompletionTokens(
                                    StreamingTokenUsage.estimateFromText(content),
                                );
                            }

                            // Forward to client
                            try writer.sendEvent(event_data);
                        }
                    } else |_| {
                        // Parse error, skip this chunk
                    }
                }

                // Clear processed line
                for (0..nl_pos + 1) |_| {
                    _ = line_buf.orderedRemove(0);
                }
            }
        }
    }

    /// Check if request has stream=true
    pub fn isStreamingRequest(request_body: []const u8) bool {
        // Simple check for "stream":true or "stream": true
        if (mem.indexOf(u8, request_body, "\"stream\":true")) |_| return true;
        if (mem.indexOf(u8, request_body, "\"stream\": true")) |_| return true;
        return false;
    }

    /// Get token usage statistics
    pub fn getTokenUsage(self: *const StreamingChatHandler) StreamingTokenUsage {
        return self.token_usage;
    }

    /// Reset token usage
    pub fn resetTokenUsage(self: *StreamingChatHandler) void {
        self.token_usage = .{};
    }
};
