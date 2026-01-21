//! DragonflyDB Client for nWorkflow Caching
//!
//! This module provides a Redis-compatible (RESP protocol) client specifically
//! designed for nWorkflow caching operations including workflow state,
//! session management, and execution tracking.
//!
//! DragonflyDB is a modern in-memory datastore that is fully compatible with
//! the Redis protocol (RESP - REdis Serialization Protocol).

const std = @import("std");
const net = std.net;
const mem = std.mem;
const Allocator = mem.Allocator;

// ============================================================================
// Key Prefixes for nWorkflow
// ============================================================================

/// Key prefix for workflow state storage
pub const WORKFLOW_STATE_PREFIX = "nwf:state:";
/// Key prefix for session storage
pub const SESSION_PREFIX = "nwf:session:";
/// Key prefix for execution tracking
pub const EXECUTION_PREFIX = "nwf:exec:";

// ============================================================================
// Error Types
// ============================================================================

/// Errors that can occur during DragonflyDB operations
pub const DragonflyError = error{
    /// Connection to server failed
    ConnectionFailed,
    /// Already connected to server
    AlreadyConnected,
    /// Not connected to server
    NotConnected,
    /// Authentication failed
    AuthenticationFailed,
    /// Server returned an error
    ServerError,
    /// Invalid response from server
    InvalidResponse,
    /// Invalid RESP protocol type
    InvalidRespType,
    /// Invalid bulk string length
    InvalidBulkStringLength,
    /// Invalid array length
    InvalidArrayLength,
    /// Recursion depth exceeded
    RecursionTooDeep,
    /// Unexpected end of stream
    EndOfStream,
    /// Operation timed out
    Timeout,
    /// Key not found
    KeyNotFound,
    /// Unexpected response type
    UnexpectedResponse,
};

// ============================================================================
// RESP Protocol Types
// ============================================================================

/// RESP protocol value types
pub const RespType = enum {
    simple_string,
    error_msg,
    integer,
    bulk_string,
    array,
    null_value,
};

/// RESP protocol value representation
pub const RespValue = union(RespType) {
    simple_string: []const u8,
    error_msg: []const u8,
    integer: i64,
    bulk_string: []const u8,
    array: []RespValue,
    null_value: void,

    /// Free all memory associated with this value
    pub fn deinit(self: *RespValue, allocator: Allocator) void {
        switch (self.*) {
            .simple_string => |s| allocator.free(s),
            .error_msg => |e| allocator.free(e),
            .bulk_string => |s| allocator.free(s),
            .array => |arr| {
                for (arr) |*item| {
                    item.deinit(allocator);
                }
                allocator.free(arr);
            },
            .integer, .null_value => {},
        }
    }
};

// ============================================================================
// RESP Protocol Helpers
// ============================================================================

/// Format arguments as a RESP array command
/// Caller owns returned memory
pub fn formatCommand(allocator: Allocator, args: []const []const u8) ![]const u8 {
    var list = std.ArrayList(u8).initCapacity(allocator, 128) catch return error.OutOfMemory;
    errdefer list.deinit(allocator);

    // Array header: *<count>\r\n
    list.writer(allocator).print("*{d}\r\n", .{args.len}) catch return error.OutOfMemory;

    // Each argument as bulk string: $<length>\r\n<data>\r\n
    for (args) |arg| {
        list.writer(allocator).print("${d}\r\n", .{arg.len}) catch return error.OutOfMemory;
        list.appendSlice(allocator, arg) catch return error.OutOfMemory;
        list.appendSlice(allocator, "\r\n") catch return error.OutOfMemory;
    }

    return list.toOwnedSlice(allocator) catch return error.OutOfMemory;
}

/// Parse a RESP response from raw data
/// Caller owns returned memory within RespValue
pub fn parseResponse(allocator: Allocator, data: []const u8) !RespValue {
    var parser = RespParser{ .allocator = allocator, .data = data, .pos = 0 };
    return parser.parse();
}

/// Internal RESP parser state
const RespParser = struct {
    allocator: Allocator,
    data: []const u8,
    pos: usize,

    fn parse(self: *RespParser) !RespValue {
        return self.parseValue(0);
    }

    fn parseValue(self: *RespParser, depth: u32) DragonflyError!RespValue {
        if (depth > 64) return DragonflyError.RecursionTooDeep;
        if (self.pos >= self.data.len) return DragonflyError.EndOfStream;

        const type_byte = self.data[self.pos];
        self.pos += 1;

        return switch (type_byte) {
            '+' => self.parseSimpleString(),
            '-' => self.parseError(),
            ':' => self.parseInteger(),
            '$' => self.parseBulkString(),
            '*' => self.parseArray(depth),
            else => DragonflyError.InvalidRespType,
        };
    }

    fn readLine(self: *RespParser) DragonflyError![]const u8 {
        const start = self.pos;
        while (self.pos < self.data.len) {
            if (self.data[self.pos] == '\r' and
                self.pos + 1 < self.data.len and
                self.data[self.pos + 1] == '\n')
            {
                const line = self.data[start..self.pos];
                self.pos += 2; // Skip \r\n
                return line;
            }
            self.pos += 1;
        }
        return DragonflyError.EndOfStream;
    }

    fn parseSimpleString(self: *RespParser) DragonflyError!RespValue {
        const line = self.readLine() catch return DragonflyError.EndOfStream;
        return RespValue{ .simple_string = self.allocator.dupe(u8, line) catch return DragonflyError.InvalidResponse };
    }

    fn parseError(self: *RespParser) DragonflyError!RespValue {
        const line = self.readLine() catch return DragonflyError.EndOfStream;
        return RespValue{ .error_msg = self.allocator.dupe(u8, line) catch return DragonflyError.InvalidResponse };
    }

    fn parseInteger(self: *RespParser) DragonflyError!RespValue {
        const line = self.readLine() catch return DragonflyError.EndOfStream;
        const value = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidResponse;
        return RespValue{ .integer = value };
    }

    fn parseBulkString(self: *RespParser) DragonflyError!RespValue {
        const line = self.readLine() catch return DragonflyError.EndOfStream;
        const length = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidBulkStringLength;

        if (length == -1) {
            return RespValue{ .null_value = {} };
        }

        if (length < 0) {
            return DragonflyError.InvalidBulkStringLength;
        }

        const len: usize = @intCast(length);
        if (self.pos + len + 2 > self.data.len) {
            return DragonflyError.EndOfStream;
        }

        const data = self.allocator.dupe(u8, self.data[self.pos .. self.pos + len]) catch return DragonflyError.InvalidResponse;
        self.pos += len + 2; // Skip data + \r\n
        return RespValue{ .bulk_string = data };
    }

    fn parseArray(self: *RespParser, depth: u32) DragonflyError!RespValue {
        const line = self.readLine() catch return DragonflyError.EndOfStream;
        const count = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidArrayLength;

        if (count == -1) {
            return RespValue{ .null_value = {} };
        }

        if (count < 0) {
            return DragonflyError.InvalidArrayLength;
        }

        const arr = self.allocator.alloc(RespValue, @intCast(count)) catch return DragonflyError.InvalidResponse;
        errdefer {
            for (arr) |*item| {
                item.deinit(self.allocator);
            }
            self.allocator.free(arr);
        }

        for (arr) |*item| {
            item.* = self.parseValueInner(depth + 1) catch return DragonflyError.InvalidResponse;
        }

        return RespValue{ .array = arr };
    }

    fn parseValueInner(self: *RespParser, depth: u32) DragonflyError!RespValue {
        if (depth > 64) return DragonflyError.RecursionTooDeep;
        if (self.pos >= self.data.len) return DragonflyError.EndOfStream;

        const type_byte = self.data[self.pos];
        self.pos += 1;

        return switch (type_byte) {
            '+' => self.parseSimpleString(),
            '-' => self.parseError(),
            ':' => self.parseInteger(),
            '$' => self.parseBulkString(),
            '*' => self.parseArray(depth),
            else => DragonflyError.InvalidRespType,
        };
    }
};


// ============================================================================
// Connection Struct
// ============================================================================

/// DragonflyDB connection for nWorkflow caching
pub const Connection = struct {
    /// Server hostname
    host: []const u8,
    /// Server port
    port: u16,
    /// Active TCP stream connection
    stream: ?net.Stream,
    /// Memory allocator
    allocator: Allocator,
    /// Connection state
    is_connected: bool,

    const Self = @This();

    /// Default host for DragonflyDB
    pub const DEFAULT_HOST = "127.0.0.1";
    /// Default port for DragonflyDB
    pub const DEFAULT_PORT: u16 = 6379;

    /// Initialize a new connection with default settings
    pub fn init(allocator: Allocator) Self {
        return initWithConfig(allocator, DEFAULT_HOST, DEFAULT_PORT);
    }

    /// Initialize a new connection with custom host and port
    pub fn initWithConfig(allocator: Allocator, host: []const u8, port: u16) Self {
        return Self{
            .host = host,
            .port = port,
            .stream = null,
            .allocator = allocator,
            .is_connected = false,
        };
    }

    /// Clean up connection resources
    pub fn deinit(self: *Self) void {
        self.disconnect();
    }

    // ========================================================================
    // Connection Management
    // ========================================================================

    /// Establish connection to DragonflyDB server
    pub fn connect(self: *Self) !void {
        if (self.is_connected) {
            return DragonflyError.AlreadyConnected;
        }

        const address = net.Address.parseIp(self.host, self.port) catch {
            return DragonflyError.ConnectionFailed;
        };

        self.stream = net.tcpConnectToAddress(address) catch {
            return DragonflyError.ConnectionFailed;
        };

        self.is_connected = true;
    }

    /// Close the connection to DragonflyDB server
    pub fn disconnect(self: *Self) void {
        if (self.stream) |*s| {
            s.close();
            self.stream = null;
            self.is_connected = false;
        }
    }

    // ========================================================================
    // Internal RESP Communication
    // ========================================================================

    fn sendCommand(self: *Self, args: []const []const u8) !void {
        if (self.stream == null) return DragonflyError.NotConnected;
        const stream = self.stream.?;

        // Array header: *<count>\r\n
        var header_buf: [32]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf, "*{d}\r\n", .{args.len}) catch unreachable;
        stream.writeAll(header) catch return DragonflyError.ConnectionFailed;

        // Each argument as bulk string: $<length>\r\n<data>\r\n
        for (args) |arg| {
            var len_buf: [32]u8 = undefined;
            const len_str = std.fmt.bufPrint(&len_buf, "${d}\r\n", .{arg.len}) catch unreachable;
            stream.writeAll(len_str) catch return DragonflyError.ConnectionFailed;
            stream.writeAll(arg) catch return DragonflyError.ConnectionFailed;
            stream.writeAll("\r\n") catch return DragonflyError.ConnectionFailed;
        }
    }

    fn readByte(self: *Self) !u8 {
        var buf: [1]u8 = undefined;
        const n = self.stream.?.read(&buf) catch return DragonflyError.ConnectionFailed;
        if (n == 0) return DragonflyError.EndOfStream;
        return buf[0];
    }

    fn readLine(self: *Self) ![]const u8 {
        var line = std.ArrayList(u8).initCapacity(self.allocator, 64) catch return DragonflyError.ConnectionFailed;
        errdefer line.deinit(self.allocator);

        while (true) {
            const byte = try self.readByte();
            if (byte == '\n') break;
            if (byte != '\r') {
                line.append(self.allocator, byte) catch return DragonflyError.ConnectionFailed;
            }
        }

        return line.toOwnedSlice(self.allocator) catch return DragonflyError.ConnectionFailed;
    }

    fn readResponse(self: *Self) !RespValue {
        if (self.stream == null) return DragonflyError.NotConnected;

        const type_byte = try self.readByte();

        return switch (type_byte) {
            '+' => try self.readSimpleString(),
            '-' => try self.readErrorMsg(),
            ':' => try self.readInteger(),
            '$' => try self.readBulkString(),
            '*' => try self.readArray(0),
            else => DragonflyError.InvalidRespType,
        };
    }

    fn readSimpleString(self: *Self) !RespValue {
        const line = try self.readLine();
        return RespValue{ .simple_string = line };
    }

    fn readErrorMsg(self: *Self) !RespValue {
        const line = try self.readLine();
        return RespValue{ .error_msg = line };
    }

    fn readInteger(self: *Self) !RespValue {
        const line = try self.readLine();
        defer self.allocator.free(line);
        const value = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidResponse;
        return RespValue{ .integer = value };
    }

    fn readBulkString(self: *Self) !RespValue {
        const line = try self.readLine();
        defer self.allocator.free(line);
        const length = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidBulkStringLength;

        if (length == -1) {
            return RespValue{ .null_value = {} };
        }

        if (length < 0) {
            return DragonflyError.InvalidBulkStringLength;
        }

        const data = self.allocator.alloc(u8, @intCast(length)) catch return DragonflyError.ConnectionFailed;
        errdefer self.allocator.free(data);

        var total_read: usize = 0;
        while (total_read < data.len) {
            const n = self.stream.?.read(data[total_read..]) catch return DragonflyError.ConnectionFailed;
            if (n == 0) return DragonflyError.EndOfStream;
            total_read += n;
        }

        // Read trailing \r\n
        _ = try self.readByte();
        _ = try self.readByte();

        return RespValue{ .bulk_string = data };
    }

    fn readArray(self: *Self, depth: u32) anyerror!RespValue {
        if (depth > 64) return DragonflyError.RecursionTooDeep;

        const line = try self.readLine();
        defer self.allocator.free(line);
        const count = std.fmt.parseInt(i64, line, 10) catch return DragonflyError.InvalidArrayLength;

        if (count < 0) {
            return RespValue{ .null_value = {} };
        }

        const array = self.allocator.alloc(RespValue, @intCast(count)) catch return DragonflyError.ConnectionFailed;
        errdefer {
            for (array) |*item| {
                item.deinit(self.allocator);
            }
            self.allocator.free(array);
        }

        for (array) |*item| {
            item.* = try self.readResponse();
        }

        return RespValue{ .array = array };
    }

    // ========================================================================
    // Basic Commands
    // ========================================================================

    /// Set a key-value pair with optional TTL in seconds
    pub fn set(self: *Self, key: []const u8, value: []const u8, ttl_seconds: ?u32) !void {
        if (ttl_seconds) |ttl_val| {
            var ttl_buf: [32]u8 = undefined;
            const ttl_str = std.fmt.bufPrint(&ttl_buf, "{d}", .{ttl_val}) catch unreachable;
            try self.sendCommand(&.{ "SET", key, value, "EX", ttl_str });
        } else {
            try self.sendCommand(&.{ "SET", key, value });
        }

        var response = try self.readResponse();
        defer response.deinit(self.allocator);

        switch (response) {
            .simple_string => {},
            .error_msg => return DragonflyError.ServerError,
            else => return DragonflyError.UnexpectedResponse,
        }
    }

    /// Get a value by key. Returns null if key doesn't exist
    /// Caller owns returned memory
    pub fn get(self: *Self, key: []const u8) !?[]const u8 {
        try self.sendCommand(&.{ "GET", key });

        var response = try self.readResponse();

        return switch (response) {
            .bulk_string => |s| s, // Transfer ownership
            .null_value => null,
            .error_msg => {
                response.deinit(self.allocator);
                return DragonflyError.ServerError;
            },
            else => {
                response.deinit(self.allocator);
                return DragonflyError.UnexpectedResponse;
            },
        };
    }

    /// Delete a key. Returns true if key was deleted
    pub fn del(self: *Self, key: []const u8) !bool {
        try self.sendCommand(&.{ "DEL", key });

        var response = try self.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => DragonflyError.ServerError,
            else => DragonflyError.UnexpectedResponse,
        };
    }

    /// Check if a key exists
    pub fn exists(self: *Self, key: []const u8) !bool {
        try self.sendCommand(&.{ "EXISTS", key });

        var response = try self.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => DragonflyError.ServerError,
            else => DragonflyError.UnexpectedResponse,
        };
    }

    /// Set expiration time on a key. Returns true if timeout was set
    pub fn expire(self: *Self, key: []const u8, seconds: u32) !bool {
        var seconds_buf: [32]u8 = undefined;
        const seconds_str = std.fmt.bufPrint(&seconds_buf, "{d}", .{seconds}) catch unreachable;

        try self.sendCommand(&.{ "EXPIRE", key, seconds_str });

        var response = try self.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => DragonflyError.ServerError,
            else => DragonflyError.UnexpectedResponse,
        };
    }

    /// Get TTL of a key. Returns null if key doesn't exist or has no TTL
    /// Returns -1 if key exists but has no expiration
    /// Returns -2 if key doesn't exist
    pub fn ttl(self: *Self, key: []const u8) !?i64 {
        try self.sendCommand(&.{ "TTL", key });

        var response = try self.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .integer => |i| if (i < 0) null else i,
            .error_msg => DragonflyError.ServerError,
            else => DragonflyError.UnexpectedResponse,
        };
    }


    // ========================================================================
    // Workflow-Specific Helpers
    // ========================================================================

    /// Cache workflow state with TTL
    pub fn cacheWorkflowState(self: *Self, workflow_id: []const u8, state_json: []const u8, ttl_val: u32) !void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ WORKFLOW_STATE_PREFIX, workflow_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        try self.set(key, state_json, ttl_val);
    }

    /// Get cached workflow state
    /// Caller owns returned memory
    pub fn getWorkflowState(self: *Self, workflow_id: []const u8) !?[]const u8 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ WORKFLOW_STATE_PREFIX, workflow_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        return self.get(key);
    }

    /// Cache session data with TTL
    pub fn cacheSession(self: *Self, session_id: []const u8, user_data: []const u8, ttl_val: u32) !void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ SESSION_PREFIX, session_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        try self.set(key, user_data, ttl_val);
    }

    /// Get cached session data
    /// Caller owns returned memory
    pub fn getSession(self: *Self, session_id: []const u8) !?[]const u8 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ SESSION_PREFIX, session_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        return self.get(key);
    }

    /// Invalidate (delete) a session
    pub fn invalidateSession(self: *Self, session_id: []const u8) !void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ SESSION_PREFIX, session_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        _ = try self.del(key);
    }

    /// Cache execution data with TTL
    pub fn cacheExecution(self: *Self, execution_id: []const u8, exec_data: []const u8, ttl_val: u32) !void {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ EXECUTION_PREFIX, execution_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        try self.set(key, exec_data, ttl_val);
    }

    /// Get cached execution data
    /// Caller owns returned memory
    pub fn getExecution(self: *Self, execution_id: []const u8) !?[]const u8 {
        var key_buf: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&key_buf, "{s}{s}", .{ EXECUTION_PREFIX, execution_id }) catch {
            return DragonflyError.InvalidResponse;
        };
        return self.get(key);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "formatCommand creates valid RESP array" {
    const allocator = std.testing.allocator;

    const cmd = try formatCommand(allocator, &.{ "SET", "key", "value" });
    defer allocator.free(cmd);

    try std.testing.expectEqualStrings("*3\r\n$3\r\nSET\r\n$3\r\nkey\r\n$5\r\nvalue\r\n", cmd);
}

test "formatCommand handles empty args" {
    const allocator = std.testing.allocator;

    const cmd = try formatCommand(allocator, &.{});
    defer allocator.free(cmd);

    try std.testing.expectEqualStrings("*0\r\n", cmd);
}

test "parseResponse handles simple string" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "+OK\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .simple_string);
    try std.testing.expectEqualStrings("OK", result.simple_string);
}

test "parseResponse handles integer" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, ":42\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .integer);
    try std.testing.expectEqual(@as(i64, 42), result.integer);
}

test "parseResponse handles bulk string" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "$5\r\nhello\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .bulk_string);
    try std.testing.expectEqualStrings("hello", result.bulk_string);
}

test "parseResponse handles null bulk string" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "$-1\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .null_value);
}

test "parseResponse handles error" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "-ERR unknown command\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .error_msg);
    try std.testing.expectEqualStrings("ERR unknown command", result.error_msg);
}

test "parseResponse handles array" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "*2\r\n$3\r\nfoo\r\n$3\r\nbar\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .array);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expectEqualStrings("foo", result.array[0].bulk_string);
    try std.testing.expectEqualStrings("bar", result.array[1].bulk_string);
}

test "parseResponse handles nested array" {
    const allocator = std.testing.allocator;

    var result = try parseResponse(allocator, "*2\r\n*1\r\n:1\r\n*1\r\n:2\r\n");
    defer result.deinit(allocator);

    try std.testing.expect(result == .array);
    try std.testing.expectEqual(@as(usize, 2), result.array.len);
    try std.testing.expect(result.array[0] == .array);
    try std.testing.expect(result.array[1] == .array);
}

test "Connection init with defaults" {
    const allocator = std.testing.allocator;
    var conn = Connection.init(allocator);
    defer conn.deinit();

    try std.testing.expectEqualStrings("127.0.0.1", conn.host);
    try std.testing.expectEqual(@as(u16, 6379), conn.port);
    try std.testing.expect(conn.stream == null);
    try std.testing.expect(!conn.is_connected);
}

test "Connection init with custom config" {
    const allocator = std.testing.allocator;
    var conn = Connection.initWithConfig(allocator, "localhost", 6380);
    defer conn.deinit();

    try std.testing.expectEqualStrings("localhost", conn.host);
    try std.testing.expectEqual(@as(u16, 6380), conn.port);
}

test "key prefixes are correct" {
    try std.testing.expectEqualStrings("nwf:state:", WORKFLOW_STATE_PREFIX);
    try std.testing.expectEqualStrings("nwf:session:", SESSION_PREFIX);
    try std.testing.expectEqualStrings("nwf:exec:", EXECUTION_PREFIX);
}