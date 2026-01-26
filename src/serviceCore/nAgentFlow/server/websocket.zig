//! WebSocket Support for Real-time Execution Updates
//!
//! Provides WebSocket protocol implementation for streaming workflow
//! execution events to connected clients in real-time.
//!
//! Features:
//! - RFC 6455 compliant WebSocket protocol
//! - Subscription-based workflow event streaming
//! - Multi-tenant support with authentication
//! - Thread-safe connection management

const std = @import("std");
const net = std.net;
const mem = std.mem;
const Allocator = std.mem.Allocator;
const crypto = std.crypto;
const base64 = std.base64;

// ============================================================================
// Constants
// ============================================================================

/// WebSocket magic GUID for handshake (RFC 6455)
const WS_MAGIC_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

/// Maximum frame payload size (16 MB)
const MAX_PAYLOAD_SIZE: usize = 16 * 1024 * 1024;

/// Maximum message size for text messages (1 MB)
const MAX_MESSAGE_SIZE: usize = 1024 * 1024;

/// Ping interval in milliseconds
const PING_INTERVAL_MS: u64 = 30_000;

// ============================================================================
// WebSocket Opcodes (RFC 6455)
// ============================================================================

pub const Opcode = enum(u4) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    // 0x3-0x7 reserved for non-control frames
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
    // 0xB-0xF reserved for control frames

    pub fn isControl(self: Opcode) bool {
        return @intFromEnum(self) >= 0x8;
    }
};

// ============================================================================
// WebSocket Frame Structure
// ============================================================================

/// WebSocket frame as defined in RFC 6455
pub const WebSocketFrame = struct {
    /// Final fragment flag
    fin: bool,
    /// Reserved bits (should be 0)
    rsv1: bool = false,
    rsv2: bool = false,
    rsv3: bool = false,
    /// Frame opcode
    opcode: Opcode,
    /// Masking flag (required for client-to-server frames)
    masked: bool,
    /// Masking key (4 bytes, only present if masked=true)
    mask_key: ?[4]u8 = null,
    /// Payload data (unmasked)
    payload: []const u8,

    /// Parse a WebSocket frame from raw bytes
    pub fn parse(allocator: Allocator, data: []const u8) !struct { frame: WebSocketFrame, bytes_consumed: usize } {
        if (data.len < 2) return error.InsufficientData;

        const byte0 = data[0];
        const byte1 = data[1];

        const fin = (byte0 & 0x80) != 0;
        const rsv1 = (byte0 & 0x40) != 0;
        const rsv2 = (byte0 & 0x20) != 0;
        const rsv3 = (byte0 & 0x10) != 0;
        const opcode_val = @as(u4, @truncate(byte0 & 0x0F));

        // Validate opcode
        const opcode = std.meta.intToEnum(Opcode, opcode_val) catch return error.InvalidOpcode;

        const masked = (byte1 & 0x80) != 0;
        var payload_len: u64 = byte1 & 0x7F;
        var offset: usize = 2;

        // Extended payload length
        if (payload_len == 126) {
            if (data.len < offset + 2) return error.InsufficientData;
            payload_len = std.mem.readInt(u16, data[offset..][0..2], .big);
            offset += 2;
        } else if (payload_len == 127) {
            if (data.len < offset + 8) return error.InsufficientData;
            payload_len = std.mem.readInt(u64, data[offset..][0..8], .big);
            offset += 8;
        }

        // Validate payload size
        if (payload_len > MAX_PAYLOAD_SIZE) return error.PayloadTooLarge;

        // Read masking key if present
        var mask_key: ?[4]u8 = null;
        if (masked) {
            if (data.len < offset + 4) return error.InsufficientData;
            mask_key = data[offset..][0..4].*;
            offset += 4;
        }

        // Read payload
        const payload_len_usize: usize = @intCast(payload_len);
        if (data.len < offset + payload_len_usize) return error.InsufficientData;

        const raw_payload = data[offset .. offset + payload_len_usize];
        const payload = try allocator.dupe(u8, raw_payload);
        errdefer allocator.free(payload);

        // Unmask payload if necessary
        if (masked) {
            const key = mask_key.?;
            for (payload, 0..) |*byte, i| {
                byte.* ^= key[i % 4];
            }
        }

        return .{
            .frame = WebSocketFrame{
                .fin = fin,
                .rsv1 = rsv1,
                .rsv2 = rsv2,
                .rsv3 = rsv3,
                .opcode = opcode,
                .masked = masked,
                .mask_key = mask_key,
                .payload = payload,
            },
            .bytes_consumed = offset + payload_len_usize,
        };
    }

    /// Serialize frame to bytes (for server-to-client, no masking)
    pub fn serialize(self: *const WebSocketFrame, allocator: Allocator) ![]u8 {
        var size: usize = 2; // Base header

        // Calculate extended payload length size
        if (self.payload.len >= 126 and self.payload.len <= 65535) {
            size += 2;
        } else if (self.payload.len > 65535) {
            size += 8;
        }

        size += self.payload.len;

        var buffer = try allocator.alloc(u8, size);
        errdefer allocator.free(buffer);

        var offset: usize = 0;

        // Byte 0: FIN + RSV + Opcode
        buffer[offset] = (@as(u8, if (self.fin) 0x80 else 0)) |
            (@as(u8, if (self.rsv1) 0x40 else 0)) |
            (@as(u8, if (self.rsv2) 0x20 else 0)) |
            (@as(u8, if (self.rsv3) 0x10 else 0)) |
            @intFromEnum(self.opcode);
        offset += 1;

        // Byte 1: Mask flag (0 for server) + Payload length
        if (self.payload.len < 126) {
            buffer[offset] = @intCast(self.payload.len);
            offset += 1;
        } else if (self.payload.len <= 65535) {
            buffer[offset] = 126;
            offset += 1;
            std.mem.writeInt(u16, buffer[offset..][0..2], @intCast(self.payload.len), .big);
            offset += 2;
        } else {
            buffer[offset] = 127;
            offset += 1;
            std.mem.writeInt(u64, buffer[offset..][0..8], self.payload.len, .big);
            offset += 8;
        }

        // Payload
        @memcpy(buffer[offset..], self.payload);

        return buffer;
    }

    /// Free payload memory
    pub fn deinit(self: *WebSocketFrame, allocator: Allocator) void {
        allocator.free(self.payload);
    }
};

// ============================================================================
// WebSocket Handshake
// ============================================================================

/// Result of parsing WebSocket upgrade request
pub const WebSocketKey = struct {
    key: []const u8,
    path: []const u8,
    protocol: ?[]const u8,
};

/// Parse WebSocket upgrade request and extract the Sec-WebSocket-Key
pub fn parseUpgradeRequest(allocator: Allocator, request: []const u8) !WebSocketKey {
    var key: ?[]const u8 = null;
    var path: ?[]const u8 = null;
    var protocol: ?[]const u8 = null;
    var is_upgrade: bool = false;
    var is_websocket: bool = false;

    var lines = mem.splitSequence(u8, request, "\r\n");

    // Parse request line
    if (lines.next()) |request_line| {
        var parts = mem.splitScalar(u8, request_line, ' ');
        const method = parts.next() orelse return error.InvalidRequest;
        if (!mem.eql(u8, method, "GET")) return error.InvalidMethod;

        const req_path = parts.next() orelse return error.InvalidRequest;
        path = try allocator.dupe(u8, req_path);
    } else {
        return error.InvalidRequest;
    }
    errdefer if (path) |p| allocator.free(p);

    // Parse headers
    while (lines.next()) |line| {
        if (line.len == 0) break; // End of headers

        // Find header separator
        if (mem.indexOf(u8, line, ": ")) |sep_idx| {
            const header_name = line[0..sep_idx];
            const header_value = line[sep_idx + 2 ..];

            if (mem.eql(u8, header_name, "Upgrade") or mem.eql(u8, header_name, "upgrade")) {
                is_upgrade = mem.eql(u8, header_value, "websocket");
            } else if (mem.eql(u8, header_name, "Connection") or mem.eql(u8, header_name, "connection")) {
                is_websocket = mem.indexOf(u8, header_value, "Upgrade") != null or
                    mem.indexOf(u8, header_value, "upgrade") != null;
            } else if (mem.eql(u8, header_name, "Sec-WebSocket-Key")) {
                key = try allocator.dupe(u8, header_value);
            } else if (mem.eql(u8, header_name, "Sec-WebSocket-Protocol")) {
                protocol = try allocator.dupe(u8, header_value);
            }
        }
    }
    errdefer if (key) |k| allocator.free(k);
    errdefer if (protocol) |p| allocator.free(p);

    if (!is_upgrade or !is_websocket) return error.NotWebSocketUpgrade;
    if (key == null) return error.MissingWebSocketKey;

    return WebSocketKey{
        .key = key.?,
        .path = path.?,
        .protocol = protocol,
    };
}

/// Generate Sec-WebSocket-Accept key from client key
/// Formula: base64(sha1(key + GUID))
pub fn generateAcceptKey(client_key: []const u8) ![28]u8 {
    var concat_buf: [60 + WS_MAGIC_GUID.len]u8 = undefined;

    if (client_key.len > 60) return error.KeyTooLong;

    @memcpy(concat_buf[0..client_key.len], client_key);
    @memcpy(concat_buf[client_key.len..][0..WS_MAGIC_GUID.len], WS_MAGIC_GUID);

    const concat = concat_buf[0 .. client_key.len + WS_MAGIC_GUID.len];

    // SHA1 hash
    var hash: [20]u8 = undefined;
    crypto.hash.Sha1.hash(concat, &hash, .{});

    // Base64 encode (20 bytes -> 28 chars with padding)
    var accept_key: [28]u8 = undefined;
    _ = base64.standard.Encoder.encode(&accept_key, &hash);

    return accept_key;
}

/// Send WebSocket handshake response
pub fn sendHandshakeResponse(stream: net.Stream, accept_key: *const [28]u8) !void {
    const response =
        "HTTP/1.1 101 Switching Protocols\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Accept: " ++ "{s}" ++ "\r\n" ++
        "\r\n";

    var buf: [256]u8 = undefined;
    const formatted = try std.fmt.bufPrint(&buf,
        "HTTP/1.1 101 Switching Protocols\r\n" ++
            "Upgrade: websocket\r\n" ++
            "Connection: Upgrade\r\n" ++
            "Sec-WebSocket-Accept: {s}\r\n" ++
            "\r\n", .{accept_key});

    _ = response;
    _ = try stream.writeAll(formatted);
}

// ============================================================================
// WebSocket Connection
// ============================================================================

/// Represents a single WebSocket connection
pub const WebSocketConnection = struct {
    /// Network stream
    stream: net.Stream,
    /// Memory allocator
    allocator: Allocator,
    /// Unique client identifier
    client_id: []const u8,
    /// Subscribed workflow IDs
    subscriptions: std.StringHashMap(void),
    /// Whether client is authenticated
    is_authenticated: bool,
    /// Tenant ID for multi-tenancy (optional)
    tenant_id: ?[]const u8,
    /// Connection state
    is_connected: bool,
    /// Last activity timestamp
    last_activity: i64,

    /// Initialize a new WebSocket connection
    pub fn init(allocator: Allocator, stream: net.Stream, client_id: []const u8) !*WebSocketConnection {
        const conn = try allocator.create(WebSocketConnection);
        errdefer allocator.destroy(conn);

        const id_copy = try allocator.dupe(u8, client_id);
        errdefer allocator.free(id_copy);

        conn.* = WebSocketConnection{
            .stream = stream,
            .allocator = allocator,
            .client_id = id_copy,
            .subscriptions = std.StringHashMap(void).init(allocator),
            .is_authenticated = false,
            .tenant_id = null,
            .is_connected = true,
            .last_activity = std.time.timestamp(),
        };

        return conn;
    }

    /// Clean up connection resources
    pub fn deinit(self: *WebSocketConnection) void {
        // Free subscription keys
        var iter = self.subscriptions.keyIterator();
        while (iter.next()) |key| {
            self.allocator.free(key.*);
        }
        self.subscriptions.deinit();

        if (self.tenant_id) |tid| {
            self.allocator.free(tid);
        }

        self.allocator.free(self.client_id);
        self.stream.close();
        self.allocator.destroy(self);
    }

    /// Subscribe to workflow events
    pub fn subscribe(self: *WebSocketConnection, workflow_id: []const u8) !void {
        if (self.subscriptions.contains(workflow_id)) return;

        const id_copy = try self.allocator.dupe(u8, workflow_id);
        errdefer self.allocator.free(id_copy);

        try self.subscriptions.put(id_copy, {});
    }

    /// Unsubscribe from workflow events
    pub fn unsubscribe(self: *WebSocketConnection, workflow_id: []const u8) void {
        if (self.subscriptions.fetchRemove(workflow_id)) |entry| {
            self.allocator.free(entry.key);
        }
    }

    /// Check if subscribed to a workflow
    pub fn isSubscribed(self: *const WebSocketConnection, workflow_id: []const u8) bool {
        return self.subscriptions.contains(workflow_id);
    }

    /// Send a text message to the client
    pub fn sendText(self: *WebSocketConnection, message: []const u8) !void {
        const frame = WebSocketFrame{
            .fin = true,
            .opcode = .text,
            .masked = false,
            .payload = message,
        };

        const data = try frame.serialize(self.allocator);
        defer self.allocator.free(data);

        _ = try self.stream.writeAll(data);
        self.last_activity = std.time.timestamp();
    }

    /// Send a binary message to the client
    pub fn sendBinary(self: *WebSocketConnection, data: []const u8) !void {
        const frame = WebSocketFrame{
            .fin = true,
            .opcode = .binary,
            .masked = false,
            .payload = data,
        };

        const frame_data = try frame.serialize(self.allocator);
        defer self.allocator.free(frame_data);

        _ = try self.stream.writeAll(frame_data);
        self.last_activity = std.time.timestamp();
    }

    /// Send a ping frame
    pub fn sendPing(self: *WebSocketConnection) !void {
        const frame = WebSocketFrame{
            .fin = true,
            .opcode = .ping,
            .masked = false,
            .payload = "",
        };

        const data = try frame.serialize(self.allocator);
        defer self.allocator.free(data);

        _ = try self.stream.writeAll(data);
    }

    /// Send a pong frame
    pub fn sendPong(self: *WebSocketConnection, payload: []const u8) !void {
        const frame = WebSocketFrame{
            .fin = true,
            .opcode = .pong,
            .masked = false,
            .payload = payload,
        };

        const data = try frame.serialize(self.allocator);
        defer self.allocator.free(data);

        _ = try self.stream.writeAll(data);
    }

    /// Send a close frame
    pub fn sendClose(self: *WebSocketConnection, code: u16, reason: []const u8) !void {
        var payload_buf: [125]u8 = undefined;
        std.mem.writeInt(u16, payload_buf[0..2], code, .big);

        const reason_len = @min(reason.len, 123);
        @memcpy(payload_buf[2..][0..reason_len], reason[0..reason_len]);

        const frame = WebSocketFrame{
            .fin = true,
            .opcode = .close,
            .masked = false,
            .payload = payload_buf[0 .. 2 + reason_len],
        };

        const data = try frame.serialize(self.allocator);
        defer self.allocator.free(data);

        _ = try self.stream.writeAll(data);
        self.is_connected = false;
    }

    /// Read a frame from the connection
    pub fn readFrame(self: *WebSocketConnection) !WebSocketFrame {
        var buffer: [MAX_PAYLOAD_SIZE + 14]u8 = undefined;
        const bytes_read = try self.stream.read(&buffer);

        if (bytes_read == 0) {
            self.is_connected = false;
            return error.ConnectionClosed;
        }

        const result = try WebSocketFrame.parse(self.allocator, buffer[0..bytes_read]);
        self.last_activity = std.time.timestamp();
        return result.frame;
    }

    /// Set authentication state
    pub fn setAuthenticated(self: *WebSocketConnection, authenticated: bool, tenant: ?[]const u8) !void {
        self.is_authenticated = authenticated;
        if (self.tenant_id) |old_tid| {
            self.allocator.free(old_tid);
        }
        if (tenant) |t| {
            self.tenant_id = try self.allocator.dupe(u8, t);
        } else {
            self.tenant_id = null;
        }
    }
};


// ============================================================================
// Message Types (JSON Protocol)
// ============================================================================

/// Message type enumeration for WebSocket protocol
pub const MessageType = enum {
    // Client -> Server messages
    subscribe,
    unsubscribe,
    ping,
    authenticate,

    // Server -> Client messages
    execution_started,
    node_started,
    node_completed,
    node_failed,
    execution_completed,
    execution_failed,
    pong,
    subscribed,
    unsubscribed,
    authenticated,
    @"error",

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .subscribe => "subscribe",
            .unsubscribe => "unsubscribe",
            .ping => "ping",
            .authenticate => "authenticate",
            .execution_started => "execution_started",
            .node_started => "node_started",
            .node_completed => "node_completed",
            .node_failed => "node_failed",
            .execution_completed => "execution_completed",
            .execution_failed => "execution_failed",
            .pong => "pong",
            .subscribed => "subscribed",
            .unsubscribed => "unsubscribed",
            .authenticated => "authenticated",
            .@"error" => "error",
        };
    }

    pub fn fromString(str: []const u8) ?MessageType {
        const map = std.StaticStringMap(MessageType).initComptime(.{
            .{ "subscribe", .subscribe },
            .{ "unsubscribe", .unsubscribe },
            .{ "ping", .ping },
            .{ "authenticate", .authenticate },
            .{ "execution_started", .execution_started },
            .{ "node_started", .node_started },
            .{ "node_completed", .node_completed },
            .{ "node_failed", .node_failed },
            .{ "execution_completed", .execution_completed },
            .{ "execution_failed", .execution_failed },
            .{ "pong", .pong },
            .{ "subscribed", .subscribed },
            .{ "unsubscribed", .unsubscribed },
            .{ "authenticated", .authenticated },
            .{ "error", .@"error" },
        });
        return map.get(str);
    }
};

/// JSON message builder for WebSocket protocol
pub const MessageBuilder = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) MessageBuilder {
        return .{ .allocator = allocator };
    }

    /// Build subscription confirmation message
    pub fn subscribed(self: *const MessageBuilder, workflow_id: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"subscribed","workflowId":"{s}"}}
        , .{workflow_id});
    }

    /// Build unsubscription confirmation message
    pub fn unsubscribed(self: *const MessageBuilder, workflow_id: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"unsubscribed","workflowId":"{s}"}}
        , .{workflow_id});
    }

    /// Build pong message
    pub fn pong(self: *const MessageBuilder) ![]const u8 {
        return try self.allocator.dupe(u8, "{\"type\":\"pong\"}");
    }

    /// Build error message
    pub fn errorMsg(self: *const MessageBuilder, error_text: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"error","error":"{s}"}}
        , .{error_text});
    }

    /// Build execution started message
    pub fn executionStarted(
        self: *const MessageBuilder,
        execution_id: []const u8,
        workflow_id: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"execution_started","executionId":"{s}","workflowId":"{s}"}}
        , .{ execution_id, workflow_id });
    }

    /// Build node started message
    pub fn nodeStarted(
        self: *const MessageBuilder,
        execution_id: []const u8,
        node_id: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"node_started","executionId":"{s}","nodeId":"{s}"}}
        , .{ execution_id, node_id });
    }

    /// Build node completed message
    pub fn nodeCompleted(
        self: *const MessageBuilder,
        execution_id: []const u8,
        node_id: []const u8,
        output_json: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"node_completed","executionId":"{s}","nodeId":"{s}","output":{s}}}
        , .{ execution_id, node_id, output_json });
    }

    /// Build node failed message
    pub fn nodeFailed(
        self: *const MessageBuilder,
        execution_id: []const u8,
        node_id: []const u8,
        error_text: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"node_failed","executionId":"{s}","nodeId":"{s}","error":"{s}"}}
        , .{ execution_id, node_id, error_text });
    }

    /// Build execution completed message
    pub fn executionCompleted(
        self: *const MessageBuilder,
        execution_id: []const u8,
        output_json: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"execution_completed","executionId":"{s}","output":{s}}}
        , .{ execution_id, output_json });
    }

    /// Build execution failed message
    pub fn executionFailed(
        self: *const MessageBuilder,
        execution_id: []const u8,
        error_text: []const u8,
    ) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"execution_failed","executionId":"{s}","error":"{s}"}}
        , .{ execution_id, error_text });
    }

    /// Build authenticated message
    pub fn authenticated(self: *const MessageBuilder, client_id: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.allocator,
            \\{{"type":"authenticated","clientId":"{s}"}}
        , .{client_id});
    }
};


// ============================================================================
// WebSocket Server
// ============================================================================

/// WebSocket server managing multiple connections
pub const WebSocketServer = struct {
    /// Memory allocator
    allocator: Allocator,
    /// All active connections
    connections: std.ArrayList(*WebSocketConnection),
    /// Subscription map: workflow_id -> list of connections
    subscriptions: std.StringHashMap(std.ArrayList(*WebSocketConnection)),
    /// Mutex for thread-safe access
    mutex: std.Thread.Mutex,
    /// Message builder for creating JSON messages
    message_builder: MessageBuilder,
    /// Next client ID counter
    next_client_id: u64,

    /// Initialize a new WebSocket server
    pub fn init(allocator: Allocator) WebSocketServer {
        return WebSocketServer{
            .allocator = allocator,
            .connections = std.ArrayList(*WebSocketConnection){},
            .subscriptions = std.StringHashMap(std.ArrayList(*WebSocketConnection)){},
            .mutex = .{},
            .message_builder = MessageBuilder.init(allocator),
            .next_client_id = 1,
        };
    }

    /// Clean up all server resources
    pub fn deinit(self: *WebSocketServer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Clean up all connections
        for (self.connections.items) |conn| {
            conn.deinit();
        }
        self.connections.deinit();

        // Clean up subscription lists (but not keys - they're owned by connections)
        var iter = self.subscriptions.valueIterator();
        while (iter.next()) |list| {
            list.deinit();
        }
        self.subscriptions.deinit();
    }

    /// Generate a unique client ID
    fn generateClientId(self: *WebSocketServer) ![]const u8 {
        const id = self.next_client_id;
        self.next_client_id += 1;
        return std.fmt.allocPrint(self.allocator, "client-{d}", .{id});
    }

    /// Handle WebSocket upgrade request
    pub fn handleUpgrade(self: *WebSocketServer, request: []const u8, stream: net.Stream) !*WebSocketConnection {
        // Parse upgrade request
        const ws_key = try parseUpgradeRequest(self.allocator, request);
        defer {
            self.allocator.free(ws_key.key);
            self.allocator.free(ws_key.path);
            if (ws_key.protocol) |p| self.allocator.free(p);
        }

        // Generate accept key
        const accept_key = try generateAcceptKey(ws_key.key);

        // Send handshake response
        try sendHandshakeResponse(stream, &accept_key);

        // Generate client ID
        const client_id = try self.generateClientId();
        defer self.allocator.free(client_id);

        // Create connection
        const conn = try WebSocketConnection.init(self.allocator, stream, client_id);
        errdefer conn.deinit();

        // Add to connections list
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.connections.append(conn);

        return conn;
    }

    /// Broadcast message to all connections subscribed to a workflow
    pub fn broadcast(self: *WebSocketServer, workflow_id: []const u8, message: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Get subscribers for this workflow
        if (self.subscriptions.get(workflow_id)) |subscribers| {
            for (subscribers.items) |conn| {
                if (conn.is_connected) {
                    conn.sendText(message) catch {
                        // Connection error - will be cleaned up later
                        conn.is_connected = false;
                    };
                }
            }
        }
    }

    /// Broadcast message to all connected clients
    pub fn broadcastToAll(self: *WebSocketServer, message: []const u8) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (self.connections.items) |conn| {
            if (conn.is_connected) {
                conn.sendText(message) catch {
                    conn.is_connected = false;
                };
            }
        }
    }

    /// Remove a connection from the server
    pub fn removeConnection(self: *WebSocketServer, conn: *WebSocketConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Remove from subscription lists
        var sub_iter = conn.subscriptions.keyIterator();
        while (sub_iter.next()) |workflow_id| {
            if (self.subscriptions.getPtr(workflow_id.*)) |subscribers| {
                // Find and remove connection from subscribers list
                var i: usize = 0;
                while (i < subscribers.items.len) {
                    if (subscribers.items[i] == conn) {
                        _ = subscribers.swapRemove(i);
                    } else {
                        i += 1;
                    }
                }
            }
        }

        // Remove from connections list
        var i: usize = 0;
        while (i < self.connections.items.len) {
            if (self.connections.items[i] == conn) {
                _ = self.connections.swapRemove(i);
            } else {
                i += 1;
            }
        }

        conn.deinit();
    }

    /// Handle incoming message from a connection
    pub fn handleMessage(self: *WebSocketServer, conn: *WebSocketConnection, message: []const u8) !void {
        // Parse JSON message
        const parsed = std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            message,
            .{},
        ) catch {
            const err_msg = try self.message_builder.errorMsg("Invalid JSON");
            defer self.allocator.free(err_msg);
            try conn.sendText(err_msg);
            return;
        };
        defer parsed.deinit();

        const obj = parsed.value.object;

        // Get message type
        const type_val = obj.get("type") orelse {
            const err_msg = try self.message_builder.errorMsg("Missing 'type' field");
            defer self.allocator.free(err_msg);
            try conn.sendText(err_msg);
            return;
        };

        const type_str = type_val.string;
        const msg_type = MessageType.fromString(type_str) orelse {
            const err_msg = try self.message_builder.errorMsg("Unknown message type");
            defer self.allocator.free(err_msg);
            try conn.sendText(err_msg);
            return;
        };

        switch (msg_type) {
            .subscribe => {
                const workflow_id_val = obj.get("workflowId") orelse {
                    const err_msg = try self.message_builder.errorMsg("Missing 'workflowId'");
                    defer self.allocator.free(err_msg);
                    try conn.sendText(err_msg);
                    return;
                };
                const workflow_id = workflow_id_val.string;

                // Add subscription
                try conn.subscribe(workflow_id);
                try self.addSubscription(workflow_id, conn);

                // Send confirmation
                const confirm_msg = try self.message_builder.subscribed(workflow_id);
                defer self.allocator.free(confirm_msg);
                try conn.sendText(confirm_msg);
            },
            .unsubscribe => {
                const workflow_id_val = obj.get("workflowId") orelse {
                    const err_msg = try self.message_builder.errorMsg("Missing 'workflowId'");
                    defer self.allocator.free(err_msg);
                    try conn.sendText(err_msg);
                    return;
                };
                const workflow_id = workflow_id_val.string;

                // Remove subscription
                conn.unsubscribe(workflow_id);
                self.removeSubscription(workflow_id, conn);

                // Send confirmation
                const confirm_msg = try self.message_builder.unsubscribed(workflow_id);
                defer self.allocator.free(confirm_msg);
                try conn.sendText(confirm_msg);
            },
            .ping => {
                const pong_msg = try self.message_builder.pong();
                defer self.allocator.free(pong_msg);
                try conn.sendText(pong_msg);
            },
            .authenticate => {
                // Authentication would validate token here
                // For now, just mark as authenticated
                try conn.setAuthenticated(true, null);
                const auth_msg = try self.message_builder.authenticated(conn.client_id);
                defer self.allocator.free(auth_msg);
                try conn.sendText(auth_msg);
            },
            else => {
                const err_msg = try self.message_builder.errorMsg("Unsupported client message type");
                defer self.allocator.free(err_msg);
                try conn.sendText(err_msg);
            },
        }
    }

    /// Add a connection to a workflow's subscriber list
    fn addSubscription(self: *WebSocketServer, workflow_id: []const u8, conn: *WebSocketConnection) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        const result = self.subscriptions.getOrPut(workflow_id) catch return;
        if (!result.found_existing) {
            result.value_ptr.* = std.ArrayList(*WebSocketConnection){};
        }
        try result.value_ptr.append(conn);
    }

    /// Remove a connection from a workflow's subscriber list
    fn removeSubscription(self: *WebSocketServer, workflow_id: []const u8, conn: *WebSocketConnection) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.subscriptions.getPtr(workflow_id)) |subscribers| {
            var i: usize = 0;
            while (i < subscribers.items.len) {
                if (subscribers.items[i] == conn) {
                    _ = subscribers.swapRemove(i);
                } else {
                    i += 1;
                }
            }
        }
    }

    /// Get number of active connections
    pub fn getConnectionCount(self: *WebSocketServer) usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.connections.items.len;
    }

    /// Clean up disconnected connections
    pub fn cleanupDisconnected(self: *WebSocketServer) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var i: usize = 0;
        while (i < self.connections.items.len) {
            const conn = self.connections.items[i];
            if (!conn.is_connected) {
                // Remove from subscription lists
                var sub_iter = conn.subscriptions.keyIterator();
                while (sub_iter.next()) |workflow_id| {
                    if (self.subscriptions.getPtr(workflow_id.*)) |subscribers| {
                        var j: usize = 0;
                        while (j < subscribers.items.len) {
                            if (subscribers.items[j] == conn) {
                                _ = subscribers.swapRemove(j);
                            } else {
                                j += 1;
                            }
                        }
                    }
                }
                _ = self.connections.swapRemove(i);
                conn.deinit();
            } else {
                i += 1;
            }
        }
    }
};


// ============================================================================
// Integration Helpers
// ============================================================================

/// High-level integration helpers for workflow execution notifications
pub const ExecutionNotifier = struct {
    server: *WebSocketServer,
    allocator: Allocator,

    pub fn init(server: *WebSocketServer) ExecutionNotifier {
        return .{
            .server = server,
            .allocator = server.allocator,
        };
    }

    /// Notify subscribers that a workflow execution has started
    pub fn notifyExecutionStarted(self: *const ExecutionNotifier, workflow_id: []const u8, execution_id: []const u8) void {
        const msg = self.server.message_builder.executionStarted(execution_id, workflow_id) catch return;
        defer self.allocator.free(msg);
        self.server.broadcast(workflow_id, msg);
    }

    /// Notify subscribers about node execution progress
    pub fn notifyNodeProgress(
        self: *const ExecutionNotifier,
        workflow_id: []const u8,
        execution_id: []const u8,
        node_id: []const u8,
        status: NodeStatus,
        output_or_error: ?[]const u8,
    ) void {
        const msg = switch (status) {
            .started => self.server.message_builder.nodeStarted(execution_id, node_id) catch return,
            .completed => self.server.message_builder.nodeCompleted(
                execution_id,
                node_id,
                output_or_error orelse "{}",
            ) catch return,
            .failed => self.server.message_builder.nodeFailed(
                execution_id,
                node_id,
                output_or_error orelse "Unknown error",
            ) catch return,
        };
        defer self.allocator.free(msg);
        self.server.broadcast(workflow_id, msg);
    }

    /// Notify subscribers that a workflow execution has completed
    pub fn notifyExecutionCompleted(
        self: *const ExecutionNotifier,
        workflow_id: []const u8,
        execution_id: []const u8,
        output: []const u8,
    ) void {
        const msg = self.server.message_builder.executionCompleted(execution_id, output) catch return;
        defer self.allocator.free(msg);
        self.server.broadcast(workflow_id, msg);
    }

    /// Notify subscribers that a workflow execution has failed
    pub fn notifyExecutionFailed(
        self: *const ExecutionNotifier,
        workflow_id: []const u8,
        execution_id: []const u8,
        error_msg: []const u8,
    ) void {
        const msg = self.server.message_builder.executionFailed(execution_id, error_msg) catch return;
        defer self.allocator.free(msg);
        self.server.broadcast(workflow_id, msg);
    }
};

/// Node execution status
pub const NodeStatus = enum {
    started,
    completed,
    failed,
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Check if an HTTP request is a WebSocket upgrade request
pub fn isWebSocketUpgrade(request: []const u8) bool {
    var lines = mem.splitSequence(u8, request, "\r\n");
    var has_upgrade = false;
    var has_websocket = false;

    while (lines.next()) |line| {
        if (line.len == 0) break;

        if (mem.indexOf(u8, line, ": ")) |sep_idx| {
            const header_name = line[0..sep_idx];
            const header_value = line[sep_idx + 2 ..];

            if (mem.eql(u8, header_name, "Upgrade") or mem.eql(u8, header_name, "upgrade")) {
                has_upgrade = mem.eql(u8, header_value, "websocket");
            } else if (mem.eql(u8, header_name, "Connection") or mem.eql(u8, header_name, "connection")) {
                has_websocket = mem.indexOf(u8, header_value, "Upgrade") != null or
                    mem.indexOf(u8, header_value, "upgrade") != null;
            }
        }
    }

    return has_upgrade and has_websocket;
}

// ============================================================================
// Tests
// ============================================================================

test "WebSocketFrame - parse simple text frame" {
    const allocator = std.testing.allocator;

    // Text frame with "Hello" payload (unmasked server frame format for testing)
    // FIN=1, Opcode=1 (text), Length=5, Payload="Hello"
    const frame_data = [_]u8{ 0x81, 0x05, 'H', 'e', 'l', 'l', 'o' };

    const result = try WebSocketFrame.parse(allocator, &frame_data);
    var frame = result.frame;
    defer frame.deinit(allocator);

    try std.testing.expect(frame.fin == true);
    try std.testing.expect(frame.opcode == .text);
    try std.testing.expect(frame.masked == false);
    try std.testing.expectEqualStrings("Hello", frame.payload);
    try std.testing.expect(result.bytes_consumed == 7);
}

test "WebSocketFrame - parse masked client frame" {
    const allocator = std.testing.allocator;

    // Masked text frame "Hi" from client
    // FIN=1, Opcode=1, Mask=1, Length=2, MaskKey=[0x01,0x02,0x03,0x04], Payload (masked)
    const mask_key = [_]u8{ 0x01, 0x02, 0x03, 0x04 };
    const payload = "Hi";
    var masked_payload: [2]u8 = undefined;
    for (payload, 0..) |c, i| {
        masked_payload[i] = c ^ mask_key[i % 4];
    }

    const frame_data = [_]u8{ 0x81, 0x82 } ++ mask_key ++ masked_payload;

    const result = try WebSocketFrame.parse(allocator, &frame_data);
    var frame = result.frame;
    defer frame.deinit(allocator);

    try std.testing.expect(frame.fin == true);
    try std.testing.expect(frame.opcode == .text);
    try std.testing.expect(frame.masked == true);
    try std.testing.expectEqualStrings("Hi", frame.payload);
}

test "WebSocketFrame - serialize text frame" {
    const allocator = std.testing.allocator;

    const frame = WebSocketFrame{
        .fin = true,
        .opcode = .text,
        .masked = false,
        .payload = "Test",
    };

    const data = try frame.serialize(allocator);
    defer allocator.free(data);

    try std.testing.expect(data[0] == 0x81); // FIN + text opcode
    try std.testing.expect(data[1] == 0x04); // Length = 4
    try std.testing.expectEqualStrings("Test", data[2..6]);
}

test "WebSocketFrame - extended payload length 126" {
    const allocator = std.testing.allocator;

    // Create payload of 200 bytes
    var payload: [200]u8 = undefined;
    @memset(&payload, 'A');

    const frame = WebSocketFrame{
        .fin = true,
        .opcode = .text,
        .masked = false,
        .payload = &payload,
    };

    const data = try frame.serialize(allocator);
    defer allocator.free(data);

    try std.testing.expect(data[0] == 0x81);
    try std.testing.expect(data[1] == 126); // Extended length marker
    try std.testing.expect(std.mem.readInt(u16, data[2..4], .big) == 200);
}

test "generateAcceptKey - RFC 6455 example" {
    // From RFC 6455 section 1.3
    const client_key = "dGhlIHNhbXBsZSBub25jZQ==";
    const accept_key = try generateAcceptKey(client_key);

    try std.testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &accept_key);
}

test "Opcode - isControl" {
    try std.testing.expect(!Opcode.text.isControl());
    try std.testing.expect(!Opcode.binary.isControl());
    try std.testing.expect(Opcode.close.isControl());
    try std.testing.expect(Opcode.ping.isControl());
    try std.testing.expect(Opcode.pong.isControl());
}

test "MessageType - fromString and toString" {
    try std.testing.expect(MessageType.fromString("subscribe") == .subscribe);
    try std.testing.expect(MessageType.fromString("ping") == .ping);
    try std.testing.expect(MessageType.fromString("invalid") == null);

    try std.testing.expectEqualStrings("subscribe", MessageType.subscribe.toString());
    try std.testing.expectEqualStrings("execution_started", MessageType.execution_started.toString());
}

test "MessageBuilder - build messages" {
    const allocator = std.testing.allocator;
    const builder = MessageBuilder.init(allocator);

    const sub_msg = try builder.subscribed("wf-123");
    defer allocator.free(sub_msg);
    try std.testing.expect(mem.indexOf(u8, sub_msg, "subscribed") != null);
    try std.testing.expect(mem.indexOf(u8, sub_msg, "wf-123") != null);

    const pong_msg = try builder.pong();
    defer allocator.free(pong_msg);
    try std.testing.expectEqualStrings("{\"type\":\"pong\"}", pong_msg);

    const exec_msg = try builder.executionStarted("exec-456", "wf-123");
    defer allocator.free(exec_msg);
    try std.testing.expect(mem.indexOf(u8, exec_msg, "execution_started") != null);
    try std.testing.expect(mem.indexOf(u8, exec_msg, "exec-456") != null);
}

test "isWebSocketUpgrade - valid upgrade request" {
    const request =
        "GET /ws HTTP/1.1\r\n" ++
        "Host: localhost:8090\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" ++
        "Sec-WebSocket-Version: 13\r\n" ++
        "\r\n";

    try std.testing.expect(isWebSocketUpgrade(request));
}

test "isWebSocketUpgrade - regular HTTP request" {
    const request =
        "GET /api/v1/workflows HTTP/1.1\r\n" ++
        "Host: localhost:8090\r\n" ++
        "Content-Type: application/json\r\n" ++
        "\r\n";

    try std.testing.expect(!isWebSocketUpgrade(request));
}

test "parseUpgradeRequest - valid request" {
    const allocator = std.testing.allocator;

    const request =
        "GET /ws/workflow HTTP/1.1\r\n" ++
        "Host: localhost:8090\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n" ++
        "Sec-WebSocket-Version: 13\r\n" ++
        "\r\n";

    const ws_key = try parseUpgradeRequest(allocator, request);
    defer {
        allocator.free(ws_key.key);
        allocator.free(ws_key.path);
        if (ws_key.protocol) |p| allocator.free(p);
    }

    try std.testing.expectEqualStrings("dGhlIHNhbXBsZSBub25jZQ==", ws_key.key);
    try std.testing.expectEqualStrings("/ws/workflow", ws_key.path);
}

test "parseUpgradeRequest - missing key" {
    const allocator = std.testing.allocator;

    const request =
        "GET /ws HTTP/1.1\r\n" ++
        "Host: localhost:8090\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "\r\n";

    const result = parseUpgradeRequest(allocator, request);
    try std.testing.expectError(error.MissingWebSocketKey, result);
}

test "parseUpgradeRequest - not an upgrade request" {
    const allocator = std.testing.allocator;

    const request =
        "GET /api HTTP/1.1\r\n" ++
        "Host: localhost:8090\r\n" ++
        "Content-Type: application/json\r\n" ++
        "\r\n";

    const result = parseUpgradeRequest(allocator, request);
    try std.testing.expectError(error.NotWebSocketUpgrade, result);
}
