const std = @import("std");

/// SAP HANA SQL Command Network Protocol
/// Based on SAP HANA SQL Command Network Protocol Reference
/// Protocol version: 2.0

/// HANA protocol uses a message-based communication model over TCP/TLS
/// Default port: 30013 (SQL), 443 (Cloud with TLS)

/// Message part header size
pub const PART_HEADER_SIZE: usize = 16;

/// Segment header size  
pub const SEGMENT_HEADER_SIZE: usize = 24;

/// Protocol version
pub const PROTOCOL_VERSION: i32 = 2;

/// Message types
pub const MessageType = enum(u8) {
    // Connection & Session
    connect = 1,
    disconnect = 2,
    authenticate = 65,
    
    // SQL Execution
    execute_direct = 3,
    execute_prepared = 4,
    prepare = 5,
    
    // Transaction
    commit = 16,
    rollback = 17,
    
    // Result handling
    fetch = 6,
    close_result_set = 7,
    
    // Error
    error_msg = 128,
    
    pub fn toString(self: MessageType) []const u8 {
        return switch (self) {
            .connect => "Connect",
            .disconnect => "Disconnect",
            .authenticate => "Authenticate",
            .execute_direct => "ExecuteDirect",
            .execute_prepared => "ExecutePrepared",
            .prepare => "Prepare",
            .commit => "Commit",
            .rollback => "Rollback",
            .fetch => "Fetch",
            .close_result_set => "CloseResultSet",
            .error_msg => "Error",
        };
    }
};

/// Part kind - identifies the content type of a message part
pub const PartKind = enum(u8) {
    // Connection
    authentication = 33,
    connect_options = 34,
    
    // SQL
    command = 3,
    parameters = 32,
    result_set = 5,
    result_set_id = 6,
    
    // Transaction
    transaction_flags = 40,
    
    // Error
    error_info = 11,
    
    // Metadata
    table_location = 45,
    parameter_metadata = 47,
    result_set_metadata = 48,
    
    pub fn toString(self: PartKind) []const u8 {
        return switch (self) {
            .authentication => "Authentication",
            .connect_options => "ConnectOptions",
            .command => "Command",
            .parameters => "Parameters",
            .result_set => "ResultSet",
            .result_set_id => "ResultSetId",
            .transaction_flags => "TransactionFlags",
            .error_info => "ErrorInfo",
            .table_location => "TableLocation",
            .parameter_metadata => "ParameterMetadata",
            .result_set_metadata => "ResultSetMetadata",
        };
    }
};

/// Authentication method
pub const AuthMethod = enum(u8) {
    scramsha256 = 4,    // SCRAM-SHA-256 (recommended)
    jwt = 7,            // JSON Web Token
    saml = 6,           // SAML assertion
    
    pub fn toString(self: AuthMethod) []const u8 {
        return switch (self) {
            .scramsha256 => "SCRAMSHA256",
            .jwt => "JWT",
            .saml => "SAML",
        };
    }
};

/// Data type codes
pub const TypeCode = enum(u8) {
    tinyint = 1,
    smallint = 2,
    integer = 3,
    bigint = 4,
    decimal = 5,
    real = 6,
    double = 7,
    char = 8,
    varchar = 9,
    nchar = 10,
    nvarchar = 11,
    binary = 12,
    varbinary = 13,
    date = 14,
    time = 15,
    timestamp = 16,
    clob = 25,
    nclob = 26,
    blob = 27,
    boolean = 28,
    string = 29,
    nstring = 30,
    
    pub fn toString(self: TypeCode) []const u8 {
        return switch (self) {
            .tinyint => "TINYINT",
            .smallint => "SMALLINT",
            .integer => "INTEGER",
            .bigint => "BIGINT",
            .decimal => "DECIMAL",
            .real => "REAL",
            .double => "DOUBLE",
            .char => "CHAR",
            .varchar => "VARCHAR",
            .nchar => "NCHAR",
            .nvarchar => "NVARCHAR",
            .binary => "BINARY",
            .varbinary => "VARBINARY",
            .date => "DATE",
            .time => "TIME",
            .timestamp => "TIMESTAMP",
            .clob => "CLOB",
            .nclob => "NCLOB",
            .blob => "BLOB",
            .boolean => "BOOLEAN",
            .string => "STRING",
            .nstring => "NSTRING",
        };
    }
};

/// Segment header
pub const SegmentHeader = struct {
    segment_length: i32,
    segment_ofs: i32,
    no_of_parts: i16,
    segment_no: i16,
    segment_kind: u8,
    message_type: MessageType,
    commit: u8,
    command_sequence: u8,
    reserved: [8]u8,
    
    pub fn init(message_type: MessageType, no_of_parts: i16) SegmentHeader {
        return SegmentHeader{
            .segment_length = 0,  // Will be calculated
            .segment_ofs = 0,
            .no_of_parts = no_of_parts,
            .segment_no = 0,
            .segment_kind = 1,  // Request
            .message_type = message_type,
            .commit = 0,
            .command_sequence = 0,
            .reserved = [_]u8{0} ** 8,
        };
    }
    
    pub fn encode(self: SegmentHeader, writer: anytype) !void {
        try writer.writeIntLittle(i32, self.segment_length);
        try writer.writeIntLittle(i32, self.segment_ofs);
        try writer.writeIntLittle(i16, self.no_of_parts);
        try writer.writeIntLittle(i16, self.segment_no);
        try writer.writeByte(self.segment_kind);
        try writer.writeByte(@intFromEnum(self.message_type));
        try writer.writeByte(self.commit);
        try writer.writeByte(self.command_sequence);
        try writer.writeAll(&self.reserved);
    }
    
    pub fn decode(reader: anytype) !SegmentHeader {
        return SegmentHeader{
            .segment_length = try reader.readIntLittle(i32),
            .segment_ofs = try reader.readIntLittle(i32),
            .no_of_parts = try reader.readIntLittle(i16),
            .segment_no = try reader.readIntLittle(i16),
            .segment_kind = try reader.readByte(),
            .message_type = @enumFromInt(try reader.readByte()),
            .commit = try reader.readByte(),
            .command_sequence = try reader.readByte(),
            .reserved = blk: {
                var buf: [8]u8 = undefined;
                _ = try reader.readAll(&buf);
                break :blk buf;
            },
        };
    }
};

/// Part header
pub const PartHeader = struct {
    part_kind: PartKind,
    part_attributes: u8,
    argument_count: i16,
    big_argument_count: i32,
    buffer_length: i32,
    buffer_size: i32,
    
    pub fn init(part_kind: PartKind, argument_count: i16) PartHeader {
        return PartHeader{
            .part_kind = part_kind,
            .part_attributes = 0,
            .argument_count = argument_count,
            .big_argument_count = 0,
            .buffer_length = 0,
            .buffer_size = 0,
        };
    }
    
    pub fn encode(self: PartHeader, writer: anytype) !void {
        try writer.writeByte(@intFromEnum(self.part_kind));
        try writer.writeByte(self.part_attributes);
        try writer.writeIntLittle(i16, self.argument_count);
        try writer.writeIntLittle(i32, self.big_argument_count);
        try writer.writeIntLittle(i32, self.buffer_length);
        try writer.writeIntLittle(i32, self.buffer_size);
    }
    
    pub fn decode(reader: anytype) !PartHeader {
        return PartHeader{
            .part_kind = @enumFromInt(try reader.readByte()),
            .part_attributes = try reader.readByte(),
            .argument_count = try reader.readIntLittle(i16),
            .big_argument_count = try reader.readIntLittle(i32),
            .buffer_length = try reader.readIntLittle(i32),
            .buffer_size = try reader.readIntLittle(i32),
        };
    }
};

/// Connect options
pub const ConnectOption = enum(u8) {
    connection_id = 1,
    complete_array_execution = 2,
    client_locale = 3,
    supports_large_bulk_operations = 4,
    distribution_enabled = 5,
    primary_connection_id = 6,
    client_distribution_mode = 10,
    client_version = 12,
    client_type = 13,
    
    pub fn toString(self: ConnectOption) []const u8 {
        return switch (self) {
            .connection_id => "CONNECTION_ID",
            .complete_array_execution => "COMPLETE_ARRAY_EXECUTION",
            .client_locale => "CLIENT_LOCALE",
            .supports_large_bulk_operations => "SUPPORTS_LARGE_BULK_OPERATIONS",
            .distribution_enabled => "DISTRIBUTION_ENABLED",
            .primary_connection_id => "PRIMARY_CONNECTION_ID",
            .client_distribution_mode => "CLIENT_DISTRIBUTION_MODE",
            .client_version => "CLIENT_VERSION",
            .client_type => "CLIENT_TYPE",
        };
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "MessageType - enum values" {
    try std.testing.expectEqual(@as(u8, 1), @intFromEnum(MessageType.connect));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(MessageType.execute_direct));
    try std.testing.expectEqual(@as(u8, 65), @intFromEnum(MessageType.authenticate));
}

test "MessageType - toString" {
    try std.testing.expectEqualStrings("Connect", MessageType.connect.toString());
    try std.testing.expectEqualStrings("ExecuteDirect", MessageType.execute_direct.toString());
}

test "PartKind - enum values" {
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(PartKind.command));
    try std.testing.expectEqual(@as(u8, 33), @intFromEnum(PartKind.authentication));
}

test "TypeCode - toString" {
    try std.testing.expectEqualStrings("INTEGER", TypeCode.integer.toString());
    try std.testing.expectEqualStrings("VARCHAR", TypeCode.varchar.toString());
}

test "SegmentHeader - init" {
    const header = SegmentHeader.init(.connect, 2);
    
    try std.testing.expectEqual(MessageType.connect, header.message_type);
    try std.testing.expectEqual(@as(i16, 2), header.no_of_parts);
    try std.testing.expectEqual(@as(u8, 1), header.segment_kind);
}

test "PartHeader - init" {
    const header = PartHeader.init(.command, 1);
    
    try std.testing.expectEqual(PartKind.command, header.part_kind);
    try std.testing.expectEqual(@as(i16, 1), header.argument_count);
}

test "SegmentHeader - encode/decode" {
    const allocator = std.testing.allocator;
    
    const original = SegmentHeader.init(.execute_direct, 1);
    
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit();
    
    try original.encode(buffer.writer());
    
    var fbs = std.io.fixedBufferStream(buffer.items);
    const decoded = try SegmentHeader.decode(fbs.reader());
    
    try std.testing.expectEqual(original.message_type, decoded.message_type);
    try std.testing.expectEqual(original.no_of_parts, decoded.no_of_parts);
}

test "PartHeader - encode/decode" {
    const allocator = std.testing.allocator;
    
    const original = PartHeader.init(.parameters, 3);
    
    var buffer = std.ArrayList(u8){};
    defer buffer.deinit();
    
    try original.encode(buffer.writer());
    
    var fbs = std.io.fixedBufferStream(buffer.items);
    const decoded = try PartHeader.decode(fbs.reader());
    
    try std.testing.expectEqual(original.part_kind, decoded.part_kind);
    try std.testing.expectEqual(original.argument_count, decoded.argument_count);
}
