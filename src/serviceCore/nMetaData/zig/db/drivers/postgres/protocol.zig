const std = @import("std");

/// PostgreSQL wire protocol v3.0 implementation
/// Reference: https://www.postgresql.org/docs/current/protocol.html

/// Message type tags (single byte identifiers)
pub const MessageType = enum(u8) {
    // Frontend messages (client to server)
    bind = 'B',
    close = 'C',
    copy_data = 'd',
    copy_done = 'c',
    copy_fail = 'f',
    describe = 'D',
    execute = 'E',
    flush = 'H',
    function_call = 'F',
    parse = 'P',
    password = 'p',
    query = 'Q',
    sync = 'S',
    terminate = 'X',

    // Backend messages (server to client)
    authentication = 'R',
    backend_key_data = 'K',
    bind_complete = '2',
    close_complete = '3',
    command_complete = 'C',
    copy_in_response = 'G',
    copy_out_response = 'H',
    copy_both_response = 'W',
    data_row = 'D',
    empty_query_response = 'I',
    error_response = 'E',
    function_call_response = 'V',
    negotiate_protocol_version = 'v',
    no_data = 'n',
    notice_response = 'N',
    notification_response = 'A',
    parameter_description = 't',
    parameter_status = 'S',
    parse_complete = '1',
    portal_suspended = 's',
    ready_for_query = 'Z',
    row_description = 'T',

    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .query => "Query",
            .authentication => "Authentication",
            .ready_for_query => "ReadyForQuery",
            .command_complete => "CommandComplete",
            .data_row => "DataRow",
            .error_response => "ErrorResponse",
            .parameter_status => "ParameterStatus",
            .backend_key_data => "BackendKeyData",
            .row_description => "RowDescription",
            .parse => "Parse",
            .bind => "Bind",
            .execute => "Execute",
            .sync => "Sync",
            .terminate => "Terminate",
            else => "Unknown",
        };
    }
};

/// Authentication request types
pub const AuthType = enum(i32) {
    ok = 0,
    kerberos_v5 = 2,
    cleartext_password = 3,
    md5_password = 5,
    scm_credential = 6,
    gss = 7,
    gss_continue = 8,
    sspi = 9,
    sasl = 10,
    sasl_continue = 11,
    sasl_final = 12,

    pub fn toString(self: AuthType) []const u8 {
        return switch (self) {
            .ok => "OK",
            .cleartext_password => "CleartextPassword",
            .md5_password => "MD5Password",
            .sasl => "SASL",
            .sasl_continue => "SASLContinue",
            .sasl_final => "SASLFinal",
            else => "Unknown",
        };
    }
};

/// Transaction status
pub const TransactionStatus = enum(u8) {
    idle = 'I', // Not in a transaction block
    in_transaction = 'T', // In a transaction block
    failed_transaction = 'E', // In a failed transaction block

    pub fn toString(self: TransactionStatus) []const u8 {
        return switch (self) {
            .idle => "Idle",
            .in_transaction => "InTransaction",
            .failed_transaction => "FailedTransaction",
        };
    }
};

/// Message builder for constructing protocol messages
pub const MessageBuilder = struct {
    buffer: std.ArrayList(u8),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) MessageBuilder {
        return MessageBuilder{
            .buffer = std.ArrayList(u8).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MessageBuilder) void {
        self.buffer.deinit();
    }

    pub fn reset(self: *MessageBuilder) void {
        self.buffer.clearRetainingCapacity();
    }

    /// Start a new message with type tag
    pub fn startMessage(self: *MessageBuilder, msg_type: MessageType) !void {
        try self.buffer.append(@intFromEnum(msg_type));
        // Reserve space for length (will be filled in endMessage)
        try self.buffer.appendNTimes(0, 4);
    }

    /// Write a null-terminated string
    pub fn writeString(self: *MessageBuilder, str: []const u8) !void {
        try self.buffer.appendSlice(str);
        try self.buffer.append(0); // null terminator
    }

    /// Write a 32-bit integer (big-endian)
    pub fn writeInt32(self: *MessageBuilder, value: i32) !void {
        var bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, &bytes, value, .big);
        try self.buffer.appendSlice(&bytes);
    }

    /// Write a 16-bit integer (big-endian)
    pub fn writeInt16(self: *MessageBuilder, value: i16) !void {
        var bytes: [2]u8 = undefined;
        std.mem.writeInt(i16, &bytes, value, .big);
        try self.buffer.appendSlice(&bytes);
    }

    /// Write raw bytes
    pub fn writeBytes(self: *MessageBuilder, bytes: []const u8) !void {
        try self.buffer.appendSlice(bytes);
    }

    /// Write a single byte
    pub fn writeByte(self: *MessageBuilder, byte: u8) !void {
        try self.buffer.append(byte);
    }

    /// End message and update length field
    pub fn endMessage(self: *MessageBuilder) ![]const u8 {
        const total_len = self.buffer.items.len;
        const msg_len = total_len - 1; // Exclude type byte
        
        // Write length into reserved space (bytes 1-4)
        var len_bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, &len_bytes, @intCast(msg_len), .big);
        @memcpy(self.buffer.items[1..5], &len_bytes);
        
        return self.buffer.items;
    }

    /// Build a startup message (no type byte, just length + params)
    pub fn buildStartupMessage(self: *MessageBuilder, params: []const []const u8) ![]const u8 {
        self.reset();
        
        // Reserve space for length
        try self.buffer.appendNTimes(0, 4);
        
        // Protocol version (3.0 = 196608)
        try self.writeInt32(196608);
        
        // Write parameters (must be pairs: key, value, key, value, ..., 0)
        for (params) |param| {
            try self.writeString(param);
        }
        
        // Final null terminator
        try self.buffer.append(0);
        
        // Update length
        const total_len = self.buffer.items.len;
        var len_bytes: [4]u8 = undefined;
        std.mem.writeInt(i32, &len_bytes, @intCast(total_len), .big);
        @memcpy(self.buffer.items[0..4], &len_bytes);
        
        return self.buffer.items;
    }
};

/// Message parser for reading protocol messages
pub const MessageParser = struct {
    buffer: []const u8,
    pos: usize,

    pub fn init(buffer: []const u8) MessageParser {
        return MessageParser{
            .buffer = buffer,
            .pos = 0,
        };
    }

    /// Read message type
    pub fn readMessageType(self: *MessageParser) !MessageType {
        if (self.pos >= self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const byte = self.buffer[self.pos];
        self.pos += 1;
        return @enumFromInt(byte);
    }

    /// Read message length (excluding type byte)
    pub fn readLength(self: *MessageParser) !i32 {
        if (self.pos + 4 > self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const length = std.mem.readInt(i32, self.buffer[self.pos..][0..4], .big);
        self.pos += 4;
        return length;
    }

    /// Read a null-terminated string
    pub fn readString(self: *MessageParser, allocator: std.mem.Allocator) ![]const u8 {
        const start = self.pos;
        while (self.pos < self.buffer.len) {
            if (self.buffer[self.pos] == 0) {
                const str = self.buffer[start..self.pos];
                self.pos += 1; // Skip null terminator
                return try allocator.dupe(u8, str);
            }
            self.pos += 1;
        }
        return error.UnterminatedString;
    }

    /// Read a 32-bit integer (big-endian)
    pub fn readInt32(self: *MessageParser) !i32 {
        if (self.pos + 4 > self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const value = std.mem.readInt(i32, self.buffer[self.pos..][0..4], .big);
        self.pos += 4;
        return value;
    }

    /// Read a 16-bit integer (big-endian)
    pub fn readInt16(self: *MessageParser) !i16 {
        if (self.pos + 2 > self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const value = std.mem.readInt(i16, self.buffer[self.pos..][0..2], .big);
        self.pos += 2;
        return value;
    }

    /// Read raw bytes of specified length
    pub fn readBytes(self: *MessageParser, len: usize) ![]const u8 {
        if (self.pos + len > self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const bytes = self.buffer[self.pos .. self.pos + len];
        self.pos += len;
        return bytes;
    }

    /// Read a single byte
    pub fn readByte(self: *MessageParser) !u8 {
        if (self.pos >= self.buffer.len) {
            return error.UnexpectedEndOfMessage;
        }
        const byte = self.buffer[self.pos];
        self.pos += 1;
        return byte;
    }

    /// Check if we've reached the end
    pub fn isAtEnd(self: MessageParser) bool {
        return self.pos >= self.buffer.len;
    }

    /// Get remaining bytes count
    pub fn remaining(self: MessageParser) usize {
        if (self.pos >= self.buffer.len) return 0;
        return self.buffer.len - self.pos;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "MessageType - enum values" {
    try std.testing.expectEqual(@as(u8, 'Q'), @intFromEnum(MessageType.query));
    try std.testing.expectEqual(@as(u8, 'R'), @intFromEnum(MessageType.authentication));
    try std.testing.expectEqual(@as(u8, 'Z'), @intFromEnum(MessageType.ready_for_query));
}

test "MessageType - toString" {
    try std.testing.expectEqualStrings("Query", MessageType.query.toString());
    try std.testing.expectEqualStrings("Authentication", MessageType.authentication.toString());
}

test "AuthType - values and strings" {
    try std.testing.expectEqual(@as(i32, 0), @intFromEnum(AuthType.ok));
    try std.testing.expectEqual(@as(i32, 10), @intFromEnum(AuthType.sasl));
    try std.testing.expectEqualStrings("OK", AuthType.ok.toString());
    try std.testing.expectEqualStrings("SASL", AuthType.sasl.toString());
}

test "TransactionStatus - values and strings" {
    try std.testing.expectEqual(@as(u8, 'I'), @intFromEnum(TransactionStatus.idle));
    try std.testing.expectEqual(@as(u8, 'T'), @intFromEnum(TransactionStatus.in_transaction));
    try std.testing.expectEqualStrings("Idle", TransactionStatus.idle.toString());
}

test "MessageBuilder - basic message" {
    const allocator = std.testing.allocator;
    var builder = MessageBuilder.init(allocator);
    defer builder.deinit();

    try builder.startMessage(.query);
    try builder.writeString("SELECT 1");
    const msg = try builder.endMessage();

    try std.testing.expectEqual(@as(u8, 'Q'), msg[0]); // Type
    try std.testing.expect(msg.len > 5); // Has content
}

test "MessageBuilder - writeInt32" {
    const allocator = std.testing.allocator;
    var builder = MessageBuilder.init(allocator);
    defer builder.deinit();

    try builder.writeInt32(42);
    try builder.writeInt32(-1);

    try std.testing.expectEqual(@as(usize, 8), builder.buffer.items.len);
}

test "MessageBuilder - writeString" {
    const allocator = std.testing.allocator;
    var builder = MessageBuilder.init(allocator);
    defer builder.deinit();

    try builder.writeString("test");

    try std.testing.expectEqual(@as(usize, 5), builder.buffer.items.len); // "test\0"
    try std.testing.expectEqual(@as(u8, 0), builder.buffer.items[4]); // Null terminator
}

test "MessageBuilder - startup message" {
    const allocator = std.testing.allocator;
    var builder = MessageBuilder.init(allocator);
    defer builder.deinit();

    const params = [_][]const u8{ "user", "postgres", "database", "test" };
    const msg = try builder.buildStartupMessage(&params);

    // Check length field
    const len = std.mem.readInt(i32, msg[0..4], .big);
    try std.testing.expectEqual(@as(i32, @intCast(msg.len)), len);

    // Check protocol version
    const version = std.mem.readInt(i32, msg[4..8], .big);
    try std.testing.expectEqual(@as(i32, 196608), version);
}

test "MessageParser - readMessageType" {
    const buffer = [_]u8{ 'Q', 0, 0, 0, 10 };
    var parser = MessageParser.init(&buffer);

    const msg_type = try parser.readMessageType();
    try std.testing.expectEqual(MessageType.query, msg_type);
}

test "MessageParser - readLength" {
    var buffer: [5]u8 = undefined;
    buffer[0] = 'Q';
    std.mem.writeInt(i32, buffer[1..5], 100, .big);

    var parser = MessageParser.init(&buffer);
    _ = try parser.readMessageType(); // Skip type
    const len = try parser.readLength();
    try std.testing.expectEqual(@as(i32, 100), len);
}

test "MessageParser - readString" {
    const allocator = std.testing.allocator;
    const buffer = [_]u8{ 't', 'e', 's', 't', 0 };
    var parser = MessageParser.init(&buffer);

    const str = try parser.readString(allocator);
    defer allocator.free(str);

    try std.testing.expectEqualStrings("test", str);
}

test "MessageParser - readInt32" {
    var buffer: [4]u8 = undefined;
    std.mem.writeInt(i32, &buffer, 12345, .big);

    var parser = MessageParser.init(&buffer);
    const value = try parser.readInt32();
    try std.testing.expectEqual(@as(i32, 12345), value);
}

test "MessageParser - readInt16" {
    var buffer: [2]u8 = undefined;
    std.mem.writeInt(i16, &buffer, 999, .big);

    var parser = MessageParser.init(&buffer);
    const value = try parser.readInt16();
    try std.testing.expectEqual(@as(i16, 999), value);
}

test "MessageParser - isAtEnd and remaining" {
    const buffer = [_]u8{ 1, 2, 3, 4, 5 };
    var parser = MessageParser.init(&buffer);

    try std.testing.expect(!parser.isAtEnd());
    try std.testing.expectEqual(@as(usize, 5), parser.remaining());

    _ = try parser.readBytes(3);
    try std.testing.expectEqual(@as(usize, 2), parser.remaining());

    _ = try parser.readBytes(2);
    try std.testing.expect(parser.isAtEnd());
    try std.testing.expectEqual(@as(usize, 0), parser.remaining());
}

test "MessageParser - error on overflow" {
    const buffer = [_]u8{ 1, 2 };
    var parser = MessageParser.init(&buffer);

    const result = parser.readInt32();
    try std.testing.expectError(error.UnexpectedEndOfMessage, result);
}
