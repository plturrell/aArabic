// JSON-RPC 2.0 Protocol Implementation
// Day 71: JSON-RPC message types, parsing, serialization, and routing

const std = @import("std");
const json = std.json;
const Allocator = std.mem.Allocator;

// ============================================================================
// JSON-RPC 2.0 Specification Types
// ============================================================================

/// JSON-RPC 2.0 version string
pub const JSONRPC_VERSION = "2.0";

/// Request/Notification ID type (can be string, number, or null)
pub const RequestId = union(enum) {
    string: []const u8,
    number: i64,
    null_id,

    pub fn format(
        self: RequestId,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .string => |s| try writer.print("\"{s}\"", .{s}),
            .number => |n| try writer.print("{d}", .{n}),
            .null_id => try writer.writeAll("null"),
        }
    }

    pub fn eql(self: RequestId, other: RequestId) bool {
        return switch (self) {
            .string => |s1| switch (other) {
                .string => |s2| std.mem.eql(u8, s1, s2),
                else => false,
            },
            .number => |n1| switch (other) {
                .number => |n2| n1 == n2,
                else => false,
            },
            .null_id => switch (other) {
                .null_id => true,
                else => false,
            },
        };
    }
};

/// JSON-RPC Error codes as defined in the specification
pub const ErrorCode = enum(i32) {
    // JSON-RPC defined errors
    ParseError = -32700,
    InvalidRequest = -32600,
    MethodNotFound = -32601,
    InvalidParams = -32602,
    InternalError = -32603,

    // Server errors (reserved -32000 to -32099)
    ServerErrorStart = -32099,
    ServerErrorEnd = -32000,

    pub fn toInt(self: ErrorCode) i32 {
        return @intFromEnum(self);
    }
};

/// JSON-RPC Error object
pub const JsonRpcError = struct {
    code: i32,
    message: []const u8,
    data: ?json.Value = null,

    pub fn init(code: ErrorCode, message: []const u8) JsonRpcError {
        return JsonRpcError{
            .code = code.toInt(),
            .message = message,
            .data = null,
        };
    }

    pub fn initWithData(code: ErrorCode, message: []const u8, data: json.Value) JsonRpcError {
        return JsonRpcError{
            .code = code.toInt(),
            .message = message,
            .data = data,
        };
    }
};

// ============================================================================
// JSON-RPC Message Types
// ============================================================================

/// JSON-RPC Request
pub const Request = struct {
    jsonrpc: []const u8 = JSONRPC_VERSION,
    id: RequestId,
    method: []const u8,
    params: ?json.Value = null,

    pub fn init(id: RequestId, method: []const u8) Request {
        return Request{
            .id = id,
            .method = method,
        };
    }

    pub fn initWithParams(id: RequestId, method: []const u8, params: json.Value) Request {
        return Request{
            .id = id,
            .method = method,
            .params = params,
        };
    }
};

/// JSON-RPC Response (success)
pub const Response = struct {
    jsonrpc: []const u8 = JSONRPC_VERSION,
    id: RequestId,
    result: json.Value,

    pub fn init(id: RequestId, result: json.Value) Response {
        return Response{
            .id = id,
            .result = result,
        };
    }
};

/// JSON-RPC Error Response
pub const ErrorResponse = struct {
    jsonrpc: []const u8 = JSONRPC_VERSION,
    id: RequestId,
    @"error": JsonRpcError,

    pub fn init(id: RequestId, err: JsonRpcError) ErrorResponse {
        return ErrorResponse{
            .id = id,
            .@"error" = err,
        };
    }
};

/// JSON-RPC Notification (no response expected)
pub const Notification = struct {
    jsonrpc: []const u8 = JSONRPC_VERSION,
    method: []const u8,
    params: ?json.Value = null,

    pub fn init(method: []const u8) Notification {
        return Notification{
            .method = method,
        };
    }

    pub fn initWithParams(method: []const u8, params: json.Value) Notification {
        return Notification{
            .method = method,
            .params = params,
        };
    }
};

/// Union of all JSON-RPC message types
pub const JsonRpcMessage = union(enum) {
    request: Request,
    response: Response,
    error_response: ErrorResponse,
    notification: Notification,

    pub fn isRequest(self: JsonRpcMessage) bool {
        return switch (self) {
            .request => true,
            else => false,
        };
    }

    pub fn isResponse(self: JsonRpcMessage) bool {
        return switch (self) {
            .response, .error_response => true,
            else => false,
        };
    }

    pub fn isNotification(self: JsonRpcMessage) bool {
        return switch (self) {
            .notification => true,
            else => false,
        };
    }

    pub fn getId(self: JsonRpcMessage) ?RequestId {
        return switch (self) {
            .request => |r| r.id,
            .response => |r| r.id,
            .error_response => |r| r.id,
            .notification => null,
        };
    }
};

// ============================================================================
// Message Parser
// ============================================================================

pub const ParsedMessage = struct {
    message: JsonRpcMessage,
    parsed: json.Parsed(json.Value),

    pub fn deinit(self: ParsedMessage) void {
        self.parsed.deinit();
    }
};

pub const MessageParser = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) MessageParser {
        return MessageParser{
            .allocator = allocator,
        };
    }

    /// Parse a JSON-RPC message from string
    pub fn parse(self: *MessageParser, message: []const u8) !ParsedMessage {
        const parsed = try json.parseFromSlice(json.Value, self.allocator, message, .{});
        errdefer parsed.deinit();

        const obj = parsed.value.object;

        // Validate jsonrpc version
        const version = obj.get("jsonrpc") orelse return error.InvalidRequest;
        if (!std.mem.eql(u8, version.string, JSONRPC_VERSION)) {
            return error.InvalidRequest;
        }

        // Determine message type based on presence of fields
        const has_id = obj.get("id") != null;
        const has_method = obj.get("method") != null;
        const has_result = obj.get("result") != null;
        const has_error = obj.get("error") != null;

        var rpc_message: JsonRpcMessage = undefined;

        if (has_method and has_id) {
            // Request
            rpc_message = try self.parseRequest(obj);
        } else if (has_method and !has_id) {
            // Notification
            rpc_message = try self.parseNotification(obj);
        } else if (has_result and has_id) {
            // Response
            rpc_message = try self.parseResponse(obj);
        } else if (has_error and has_id) {
            // Error Response
            rpc_message = try self.parseErrorResponse(obj);
        } else {
            return error.InvalidRequest;
        }

        return ParsedMessage{
            .message = rpc_message,
            .parsed = parsed,
        };
    }

    fn parseRequest(self: *MessageParser, obj: json.ObjectMap) !JsonRpcMessage {
        _ = self;
        const id = try parseId(obj.get("id").?);
        const method = obj.get("method").?.string;
        const params = obj.get("params");

        if (params) |p| {
            return JsonRpcMessage{ .request = Request.initWithParams(id, method, p) };
        }

        return JsonRpcMessage{ .request = Request.init(id, method) };
    }

    fn parseNotification(self: *MessageParser, obj: json.ObjectMap) !JsonRpcMessage {
        _ = self;
        const method = obj.get("method").?.string;
        const params = obj.get("params");

        if (params) |p| {
            return JsonRpcMessage{ .notification = Notification.initWithParams(method, p) };
        }

        return JsonRpcMessage{ .notification = Notification.init(method) };
    }

    fn parseResponse(self: *MessageParser, obj: json.ObjectMap) !JsonRpcMessage {
        _ = self;
        const id = try parseId(obj.get("id").?);
        const result = obj.get("result").?;

        return JsonRpcMessage{ .response = Response.init(id, result) };
    }

    fn parseErrorResponse(self: *MessageParser, obj: json.ObjectMap) !JsonRpcMessage {
        _ = self;
        const id = try parseId(obj.get("id").?);
        const err_obj = obj.get("error").?.object;

        const code = @as(i32, @intCast(err_obj.get("code").?.integer));
        const message = err_obj.get("message").?.string;
        const data = err_obj.get("data");

        const err = if (data) |d|
            JsonRpcError.initWithData(@enumFromInt(code), message, d)
        else
            JsonRpcError.init(@enumFromInt(code), message);

        return JsonRpcMessage{ .error_response = ErrorResponse.init(id, err) };
    }

    fn parseId(value: json.Value) !RequestId {
        return switch (value) {
            .string => |s| RequestId{ .string = s },
            .integer => |i| RequestId{ .number = i },
            .null => RequestId.null_id,
            else => error.InvalidRequest,
        };
    }
};

// ============================================================================
// Message Serializer
// ============================================================================

pub const MessageSerializer = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) MessageSerializer {
        return MessageSerializer{
            .allocator = allocator,
        };
    }

    /// Serialize a JSON-RPC message to string
    pub fn serialize(self: *MessageSerializer, message: JsonRpcMessage) ![]const u8 {
        var string = try std.ArrayList(u8).initCapacity(self.allocator, 256);
        defer string.deinit(self.allocator);

        try string.appendSlice(self.allocator, "{\"jsonrpc\":\"2.0\"");

        switch (message) {
            .request => |r| {
                try string.appendSlice(self.allocator, ",\"id\":");
                try appendId(&string, self.allocator, r.id);
                try string.appendSlice(self.allocator, ",\"method\":\"");
                try string.appendSlice(self.allocator, r.method);
                try string.append(self.allocator, '"');
                if (r.params) |p| {
                    try string.appendSlice(self.allocator, ",\"params\":");
                    try appendJsonValue(&string, self.allocator, p);
                }
            },
            .notification => |n| {
                try string.appendSlice(self.allocator, ",\"method\":\"");
                try string.appendSlice(self.allocator, n.method);
                try string.append(self.allocator, '"');
                if (n.params) |p| {
                    try string.appendSlice(self.allocator, ",\"params\":");
                    try appendJsonValue(&string, self.allocator, p);
                }
            },
            .response => |r| {
                try string.appendSlice(self.allocator, ",\"id\":");
                try appendId(&string, self.allocator, r.id);
                try string.appendSlice(self.allocator, ",\"result\":");
                try appendJsonValue(&string, self.allocator, r.result);
            },
            .error_response => |e| {
                try string.appendSlice(self.allocator, ",\"id\":");
                try appendId(&string, self.allocator, e.id);
                try string.appendSlice(self.allocator, ",\"error\":{\"code\":");
                try std.fmt.format(string.writer(self.allocator), "{d}", .{e.@"error".code});
                try string.appendSlice(self.allocator, ",\"message\":\"");
                try string.appendSlice(self.allocator, e.@"error".message);
                try string.append(self.allocator, '"');
                if (e.@"error".data) |d| {
                    try string.appendSlice(self.allocator, ",\"data\":");
                    try appendJsonValue(&string, self.allocator, d);
                }
                try string.append(self.allocator, '}');
            },
        }

        try string.append(self.allocator, '}');
        return try string.toOwnedSlice(self.allocator);
    }

    fn appendId(string: *std.ArrayList(u8), allocator: Allocator, id: RequestId) !void {
        switch (id) {
            .string => |s| {
                try string.append(allocator, '"');
                try string.appendSlice(allocator, s);
                try string.append(allocator, '"');
            },
            .number => |n| {
                try std.fmt.format(string.writer(allocator), "{d}", .{n});
            },
            .null_id => {
                try string.appendSlice(allocator, "null");
            },
        }
    }

    fn appendJsonValue(string: *std.ArrayList(u8), allocator: Allocator, value: json.Value) !void {
        switch (value) {
            .null => try string.appendSlice(allocator, "null"),
            .bool => |b| {
                if (b) {
                    try string.appendSlice(allocator, "true");
                } else {
                    try string.appendSlice(allocator, "false");
                }
            },
            .integer => |i| try std.fmt.format(string.writer(allocator), "{d}", .{i}),
            .float => |f| try std.fmt.format(string.writer(allocator), "{d}", .{f}),
            .number_string => |ns| try string.appendSlice(allocator, ns),
            .string => |s| {
                try string.append(allocator, '"');
                try string.appendSlice(allocator, s);
                try string.append(allocator, '"');
            },
            .array => |arr| {
                try string.append(allocator, '[');
                for (arr.items, 0..) |item, i| {
                    if (i > 0) try string.append(allocator, ',');
                    try appendJsonValue(string, allocator, item);
                }
                try string.append(allocator, ']');
            },
            .object => |obj| {
                try string.append(allocator, '{');
                var iter = obj.iterator();
                var first = true;
                while (iter.next()) |entry| {
                    if (!first) try string.append(allocator, ',');
                    first = false;
                    try string.append(allocator, '"');
                    try string.appendSlice(allocator, entry.key_ptr.*);
                    try string.appendSlice(allocator, "\":");
                    try appendJsonValue(string, allocator, entry.value_ptr.*);
                }
                try string.append(allocator, '}');
            },
        }
    }
};

// ============================================================================
// Message Handler Interface
// ============================================================================

pub const MessageHandler = struct {
    ptr: *anyopaque,
    handleRequestFn: *const fn (ptr: *anyopaque, request: Request) anyerror!json.Value,
    handleNotificationFn: *const fn (ptr: *anyopaque, notification: Notification) anyerror!void,

    pub fn handleRequest(self: MessageHandler, request: Request) !json.Value {
        return self.handleRequestFn(self.ptr, request);
    }

    pub fn handleNotification(self: MessageHandler, notification: Notification) !void {
        return self.handleNotificationFn(self.ptr, notification);
    }
};

// ============================================================================
// Message Router
// ============================================================================

pub const MessageRouter = struct {
    allocator: Allocator,
    handlers: std.StringHashMap(MessageHandler),

    pub fn init(allocator: Allocator) MessageRouter {
        return MessageRouter{
            .allocator = allocator,
            .handlers = std.StringHashMap(MessageHandler).init(allocator),
        };
    }

    pub fn deinit(self: *MessageRouter) void {
        self.handlers.deinit();
    }

    /// Register a handler for a specific method
    pub fn registerHandler(self: *MessageRouter, method: []const u8, handler: MessageHandler) !void {
        try self.handlers.put(method, handler);
    }

    /// Route a message to the appropriate handler
    pub fn route(self: *MessageRouter, message: JsonRpcMessage) !?JsonRpcMessage {
        switch (message) {
            .request => |req| {
                if (self.handlers.get(req.method)) |handler| {
                    const result = handler.handleRequest(req) catch |err| {
                        const error_msg = JsonRpcError.init(.InternalError, @errorName(err));
                        return JsonRpcMessage{ .error_response = ErrorResponse.init(req.id, error_msg) };
                    };
                    return JsonRpcMessage{ .response = Response.init(req.id, result) };
                } else {
                    const error_msg = JsonRpcError.init(.MethodNotFound, "Method not found");
                    return JsonRpcMessage{ .error_response = ErrorResponse.init(req.id, error_msg) };
                }
            },
            .notification => |notif| {
                if (self.handlers.get(notif.method)) |handler| {
                    try handler.handleNotification(notif);
                }
                return null; // Notifications don't get responses
            },
            .response, .error_response => {
                // Responses are handled by the caller, not routed
                return null;
            },
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "RequestId: equality" {
    const id1 = RequestId{ .string = "test" };
    const id2 = RequestId{ .string = "test" };
    const id3 = RequestId{ .string = "other" };
    const id4 = RequestId{ .number = 42 };
    const id5 = RequestId{ .number = 42 };
    const id6 = RequestId{ .null_id = {} };
    const id7 = RequestId{ .null_id = {} };

    try std.testing.expect(id1.eql(id2));
    try std.testing.expect(!id1.eql(id3));
    try std.testing.expect(!id1.eql(id4));
    try std.testing.expect(id4.eql(id5));
    try std.testing.expect(id6.eql(id7));
}

test "Request: creation" {
    const req = Request.init(RequestId{ .number = 1 }, "test_method");

    try std.testing.expectEqualStrings(JSONRPC_VERSION, req.jsonrpc);
    try std.testing.expectEqualStrings("test_method", req.method);
    try std.testing.expect(req.params == null);
}

test "MessageSerializer: serialize request" {
    var serializer = MessageSerializer.init(std.testing.allocator);

    const req = Request.init(RequestId{ .number = 1 }, "test_method");
    const message = JsonRpcMessage{ .request = req };

    const serialized = try serializer.serialize(message);
    defer std.testing.allocator.free(serialized);

    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"jsonrpc\":\"2.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"id\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"method\":\"test_method\"") != null);
}

test "MessageSerializer: serialize notification" {
    var serializer = MessageSerializer.init(std.testing.allocator);

    const notif = Notification.init("test_notification");
    const message = JsonRpcMessage{ .notification = notif };

    const serialized = try serializer.serialize(message);
    defer std.testing.allocator.free(serialized);

    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"jsonrpc\":\"2.0\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"method\":\"test_notification\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"id\"") == null);
}

test "MessageSerializer: serialize error response" {
    var serializer = MessageSerializer.init(std.testing.allocator);

    const err = JsonRpcError.init(.MethodNotFound, "Method not found");
    const err_resp = ErrorResponse.init(RequestId{ .number = 1 }, err);
    const message = JsonRpcMessage{ .error_response = err_resp };

    const serialized = try serializer.serialize(message);
    defer std.testing.allocator.free(serialized);

    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"error\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"code\":-32601") != null);
    try std.testing.expect(std.mem.indexOf(u8, serialized, "\"message\":\"Method not found\"") != null);
}

test "MessageParser: parse request" {
    var parser = MessageParser.init(std.testing.allocator);

    const json_str = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"test\"}";
    const parsed_msg = try parser.parse(json_str);
    defer parsed_msg.deinit();

    try std.testing.expect(parsed_msg.message.isRequest());
}

test "MessageParser: parse notification" {
    var parser = MessageParser.init(std.testing.allocator);

    const json_str = "{\"jsonrpc\":\"2.0\",\"method\":\"notify\"}";
    const parsed_msg = try parser.parse(json_str);
    defer parsed_msg.deinit();

    try std.testing.expect(parsed_msg.message.isNotification());
}

test "MessageRouter: method not found" {
    var router = MessageRouter.init(std.testing.allocator);
    defer router.deinit();

    const req = Request.init(RequestId{ .number = 1 }, "unknown_method");
    const message = JsonRpcMessage{ .request = req };

    const response = try router.route(message);
    try std.testing.expect(response != null);
    try std.testing.expect(response.?.error_response.@"error".code == -32601);
}
