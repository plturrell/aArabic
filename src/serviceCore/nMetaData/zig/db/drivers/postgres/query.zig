const std = @import("std");
const protocol = @import("protocol.zig");
const client_types = @import("../../client.zig");
const MessageBuilder = protocol.MessageBuilder;
const MessageParser = protocol.MessageParser;
const MessageType = protocol.MessageType;
const Value = client_types.Value;
const Column = client_types.Column;
const Row = client_types.Row;
const ResultSet = client_types.ResultSet;

/// PostgreSQL type OIDs (Object Identifiers)
pub const TypeOid = enum(i32) {
    // Boolean
    bool = 16,
    
    // Integer types
    int2 = 21, // smallint (16-bit)
    int4 = 23, // integer (32-bit)
    int8 = 20, // bigint (64-bit)
    
    // Floating point types
    float4 = 700, // real (32-bit)
    float8 = 701, // double precision (64-bit)
    
    // Character types
    text = 25, // text
    varchar = 1043, // varchar(n)
    char = 1042, // char(n)
    bpchar = 1042, // char(n) alternate
    name = 19, // system identifier
    
    // Binary data
    bytea = 17, // byte array
    
    // Date/time types
    timestamp = 1114, // timestamp without timezone
    timestamptz = 1184, // timestamp with timezone
    date = 1082, // date
    time = 1083, // time
    
    // UUID
    uuid = 2950,
    
    // JSON
    json = 114,
    jsonb = 3802,
    
    // Null (used in some contexts)
    unknown = 0,
    
    _,
    
    /// Convert OID to Column.Type
    pub fn toColumnType(self: TypeOid) Column.Type {
        return switch (self) {
            .bool => .boolean,
            .int2, .int4 => .int32,
            .int8 => .int64,
            .float4 => .float32,
            .float8 => .float64,
            .text, .varchar, .char, .bpchar, .name, .json, .jsonb => .string,
            .bytea => .bytes,
            .timestamp, .timestamptz, .date, .time => .timestamp,
            .uuid => .uuid,
            else => .string, // Default to string for unknown types
        };
    }
};

/// Format code for parameter and result data
pub const FormatCode = enum(i16) {
    text = 0,
    binary = 1,
    
    pub fn fromInt(value: i16) FormatCode {
        return switch (value) {
            0 => .text,
            1 => .binary,
            else => .text,
        };
    }
};

/// Query result containing rows and column metadata
pub const QueryResult = struct {
    columns: []Column,
    rows: std.ArrayList(Row),
    command_tag: []const u8,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) QueryResult {
        return QueryResult{
            .columns = &[_]Column{},
            .rows = std.ArrayList(Row).init(allocator),
            .command_tag = "",
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *QueryResult) void {
        // Free columns
        for (self.columns) |col| {
            self.allocator.free(col.name);
        }
        if (self.columns.len > 0) {
            self.allocator.free(self.columns);
        }
        
        // Free rows
        for (self.rows.items) |*row| {
            row.deinit();
        }
        self.rows.deinit();
        
        // Free command tag
        if (self.command_tag.len > 0) {
            self.allocator.free(self.command_tag);
        }
    }
    
    /// Convert to ResultSet
    pub fn toResultSet(self: *QueryResult) !ResultSet {
        // Transfer ownership of columns and rows to ResultSet
        const columns = self.columns;
        const rows = try self.rows.toOwnedSlice();
        
        // Clear our references (ownership transferred)
        self.columns = &[_]Column{};
        
        return ResultSet{
            .columns = columns,
            .rows = rows,
            .allocator = self.allocator,
        };
    }
};

/// Query executor for PostgreSQL
pub const QueryExecutor = struct {
    allocator: std.mem.Allocator,
    stream: std.net.Stream,
    message_builder: MessageBuilder,
    
    pub fn init(allocator: std.mem.Allocator, stream: std.net.Stream) QueryExecutor {
        return QueryExecutor{
            .allocator = allocator,
            .stream = stream,
            .message_builder = MessageBuilder.init(allocator),
        };
    }
    
    pub fn deinit(self: *QueryExecutor) void {
        self.message_builder.deinit();
    }
    
    /// Execute simple query protocol (for queries without parameters)
    pub fn executeSimple(self: *QueryExecutor, sql: []const u8) !ResultSet {
        // Send Query message
        try self.sendQuery(sql);
        
        // Receive and parse response
        var result = QueryResult.init(self.allocator);
        errdefer result.deinit();
        
        try self.receiveQueryResponse(&result);
        
        return try result.toResultSet();
    }
    
    /// Execute extended query protocol (for queries with parameters)
    pub fn executeExtended(
        self: *QueryExecutor,
        sql: []const u8,
        params: []const Value,
    ) !ResultSet {
        // Generate statement name (empty for unnamed)
        const statement_name = "";
        const portal_name = "";
        
        // Send Parse message
        try self.sendParse(statement_name, sql);
        
        // Send Bind message
        try self.sendBind(portal_name, statement_name, params);
        
        // Send Describe message (portal)
        try self.sendDescribe('P', portal_name);
        
        // Send Execute message
        try self.sendExecute(portal_name, 0); // 0 = retrieve all rows
        
        // Send Sync message
        try self.sendSync();
        
        // Receive and parse response
        var result = QueryResult.init(self.allocator);
        errdefer result.deinit();
        
        try self.receiveExtendedResponse(&result);
        
        return try result.toResultSet();
    }
    
    // ========================================================================
    // Message Sending
    // ========================================================================
    
    /// Send Query message (simple query protocol)
    fn sendQuery(self: *QueryExecutor, sql: []const u8) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.query);
        try self.message_builder.writeString(sql);
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Send Parse message (extended query protocol)
    fn sendParse(self: *QueryExecutor, statement_name: []const u8, sql: []const u8) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.parse);
        try self.message_builder.writeString(statement_name);
        try self.message_builder.writeString(sql);
        
        // Parameter type OIDs (0 = infer)
        try self.message_builder.writeInt16(0); // No explicit type OIDs
        
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Send Bind message (extended query protocol)
    fn sendBind(
        self: *QueryExecutor,
        portal_name: []const u8,
        statement_name: []const u8,
        params: []const Value,
    ) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.bind);
        try self.message_builder.writeString(portal_name);
        try self.message_builder.writeString(statement_name);
        
        // Parameter format codes (0 = text, 1 = binary)
        try self.message_builder.writeInt16(1); // All formats the same
        try self.message_builder.writeInt16(0); // Use text format
        
        // Parameter values
        try self.message_builder.writeInt16(@intCast(params.len));
        for (params) |param| {
            try self.writeParameter(param);
        }
        
        // Result format codes (0 = text)
        try self.message_builder.writeInt16(1); // All formats the same
        try self.message_builder.writeInt16(0); // Use text format
        
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Send Describe message
    fn sendDescribe(self: *QueryExecutor, target_type: u8, name: []const u8) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.describe);
        try self.message_builder.writeByte(target_type); // 'S' = statement, 'P' = portal
        try self.message_builder.writeString(name);
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Send Execute message
    fn sendExecute(self: *QueryExecutor, portal_name: []const u8, max_rows: i32) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.execute);
        try self.message_builder.writeString(portal_name);
        try self.message_builder.writeInt32(max_rows);
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Send Sync message
    fn sendSync(self: *QueryExecutor) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.sync);
        const msg = try self.message_builder.endMessage();
        try self.stream.writeAll(msg);
    }
    
    /// Write parameter value in text format
    fn writeParameter(self: *QueryExecutor, param: Value) !void {
        switch (param) {
            .null => {
                try self.message_builder.writeInt32(-1); // NULL indicator
            },
            .bool => |v| {
                const text = if (v) "true" else "false";
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .int32 => |v| {
                var buf: [12]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "{d}", .{v});
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .int64 => |v| {
                var buf: [20]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "{d}", .{v});
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .float32 => |v| {
                var buf: [32]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "{d}", .{v});
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .float64 => |v| {
                var buf: [32]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "{d}", .{v});
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .string => |v| {
                try self.message_builder.writeInt32(@intCast(v.len));
                try self.message_builder.writeBytes(v);
            },
            .bytes => |v| {
                // Encode as hex (\x format)
                const hex_len = v.len * 2 + 2; // \x prefix
                try self.message_builder.writeInt32(@intCast(hex_len));
                try self.message_builder.writeBytes("\\x");
                for (v) |byte| {
                    var hex_buf: [2]u8 = undefined;
                    _ = try std.fmt.bufPrint(&hex_buf, "{x:0>2}", .{byte});
                    try self.message_builder.writeBytes(&hex_buf);
                }
            },
            .timestamp => |v| {
                // Format as ISO 8601: YYYY-MM-DD HH:MM:SS
                var buf: [32]u8 = undefined;
                const text = try std.fmt.bufPrint(&buf, "{d}", .{v});
                try self.message_builder.writeInt32(@intCast(text.len));
                try self.message_builder.writeBytes(text);
            },
            .uuid => |v| {
                // Format as: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                var buf: [36]u8 = undefined;
                _ = try std.fmt.bufPrint(
                    &buf,
                    "{x:0>2}{x:0>2}{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}-{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}{x:0>2}",
                    .{
                        v[0],  v[1],  v[2],  v[3],
                        v[4],  v[5],  v[6],  v[7],
                        v[8],  v[9],  v[10], v[11],
                        v[12], v[13], v[14], v[15],
                    },
                );
                try self.message_builder.writeInt32(36);
                try self.message_builder.writeBytes(&buf);
            },
        }
    }
    
    // ========================================================================
    // Response Parsing
    // ========================================================================
    
    /// Receive and parse simple query response
    fn receiveQueryResponse(self: *QueryExecutor, result: *QueryResult) !void {
        var buffer: [8192]u8 = undefined;
        
        while (true) {
            const bytes_read = try self.stream.read(&buffer);
            if (bytes_read == 0) return error.UnexpectedEOF;
            
            var parser = MessageParser.init(buffer[0..bytes_read]);
            const msg_type = try parser.readMessageType();
            const length = try parser.readLength();
            
            switch (msg_type) {
                .row_description => {
                    try self.parseRowDescription(&parser, result);
                },
                .data_row => {
                    try self.parseDataRow(&parser, result);
                },
                .command_complete => {
                    const tag = try parser.readString(self.allocator);
                    result.command_tag = tag;
                },
                .ready_for_query => {
                    _ = try parser.readByte(); // Transaction status
                    return; // Query complete
                },
                .error_response => {
                    return try self.parseError(&parser);
                },
                .notice_response => {
                    // Skip notice messages for now
                    _ = try parser.readBytes(@intCast(length - 4));
                },
                .empty_query_response => {
                    // Empty query, continue
                },
                else => {
                    // Skip unknown messages
                    _ = try parser.readBytes(@intCast(length - 4));
                },
            }
        }
    }
    
    /// Receive and parse extended query response
    fn receiveExtendedResponse(self: *QueryExecutor, result: *QueryResult) !void {
        var buffer: [8192]u8 = undefined;
        
        while (true) {
            const bytes_read = try self.stream.read(&buffer);
            if (bytes_read == 0) return error.UnexpectedEOF;
            
            var parser = MessageParser.init(buffer[0..bytes_read]);
            const msg_type = try parser.readMessageType();
            const length = try parser.readLength();
            
            switch (msg_type) {
                .parse_complete => {
                    // Parse successful
                },
                .bind_complete => {
                    // Bind successful
                },
                .row_description => {
                    try self.parseRowDescription(&parser, result);
                },
                .data_row => {
                    try self.parseDataRow(&parser, result);
                },
                .command_complete => {
                    const tag = try parser.readString(self.allocator);
                    result.command_tag = tag;
                },
                .ready_for_query => {
                    _ = try parser.readByte(); // Transaction status
                    return; // Query complete
                },
                .error_response => {
                    return try self.parseError(&parser);
                },
                .notice_response => {
                    // Skip notice messages
                    _ = try parser.readBytes(@intCast(length - 4));
                },
                .no_data => {
                    // No data (e.g., for a statement with no result)
                },
                else => {
                    // Skip unknown messages
                    _ = try parser.readBytes(@intCast(length - 4));
                },
            }
        }
    }
    
    /// Parse RowDescription message
    fn parseRowDescription(self: *QueryExecutor, parser: *MessageParser, result: *QueryResult) !void {
        const field_count = try parser.readInt16();
        
        var columns = try self.allocator.alloc(Column, @intCast(field_count));
        errdefer self.allocator.free(columns);
        
        for (columns) |*col| {
            const name = try parser.readString(self.allocator);
            const table_oid = try parser.readInt32();
            _ = table_oid;
            const column_attr = try parser.readInt16();
            _ = column_attr;
            const type_oid_int = try parser.readInt32();
            const type_size = try parser.readInt16();
            _ = type_size;
            const type_modifier = try parser.readInt32();
            _ = type_modifier;
            const format_code = try parser.readInt16();
            _ = format_code;
            
            const type_oid: TypeOid = @enumFromInt(type_oid_int);
            
            col.* = Column{
                .name = name,
                .type = type_oid.toColumnType(),
            };
        }
        
        result.columns = columns;
    }
    
    /// Parse DataRow message
    fn parseDataRow(self: *QueryExecutor, parser: *MessageParser, result: *QueryResult) !void {
        const field_count = try parser.readInt16();
        
        var values = try self.allocator.alloc(Value, @intCast(field_count));
        errdefer self.allocator.free(values);
        
        for (values, 0..) |*val, i| {
            const field_length = try parser.readInt32();
            
            if (field_length == -1) {
                // NULL value
                val.* = .null;
            } else {
                const field_data = try parser.readBytes(@intCast(field_length));
                
                // Convert based on column type
                if (i < result.columns.len) {
                    val.* = try self.parseFieldValue(field_data, result.columns[i].type);
                } else {
                    // Fallback to string
                    const str = try self.allocator.dupe(u8, field_data);
                    val.* = Value{ .string = str };
                }
            }
        }
        
        const row = Row{
            .values = values,
            .allocator = self.allocator,
        };
        
        try result.rows.append(row);
    }
    
    /// Parse field value based on column type
    fn parseFieldValue(self: *QueryExecutor, data: []const u8, col_type: Column.Type) !Value {
        return switch (col_type) {
            .boolean => blk: {
                if (std.mem.eql(u8, data, "t") or std.mem.eql(u8, data, "true")) {
                    break :blk Value{ .bool = true };
                } else {
                    break :blk Value{ .bool = false };
                }
            },
            .int32 => blk: {
                const val = try std.fmt.parseInt(i32, data, 10);
                break :blk Value{ .int32 = val };
            },
            .int64 => blk: {
                const val = try std.fmt.parseInt(i64, data, 10);
                break :blk Value{ .int64 = val };
            },
            .float32 => blk: {
                const val = try std.fmt.parseFloat(f32, data);
                break :blk Value{ .float32 = val };
            },
            .float64 => blk: {
                const val = try std.fmt.parseFloat(f64, data);
                break :blk Value{ .float64 = val };
            },
            .string => blk: {
                const str = try self.allocator.dupe(u8, data);
                break :blk Value{ .string = str };
            },
            .bytes => blk: {
                // Parse hex format: \x...
                if (data.len >= 2 and data[0] == '\\' and data[1] == 'x') {
                    const hex_data = data[2..];
                    const bytes = try self.allocator.alloc(u8, hex_data.len / 2);
                    errdefer self.allocator.free(bytes);
                    
                    var i: usize = 0;
                    while (i < hex_data.len) : (i += 2) {
                        bytes[i / 2] = try std.fmt.parseInt(u8, hex_data[i..i+2], 16);
                    }
                    break :blk Value{ .bytes = bytes };
                } else {
                    const bytes = try self.allocator.dupe(u8, data);
                    break :blk Value{ .bytes = bytes };
                }
            },
            .timestamp => blk: {
                // For now, parse as Unix timestamp
                const val = try std.fmt.parseInt(i64, data, 10);
                break :blk Value{ .timestamp = val };
            },
            .uuid => blk: {
                // Parse UUID: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
                if (data.len != 36) return error.InvalidUUID;
                
                var uuid: [16]u8 = undefined;
                var uuid_idx: usize = 0;
                var i: usize = 0;
                
                while (i < data.len) : (i += 1) {
                    if (data[i] == '-') continue;
                    uuid[uuid_idx] = try std.fmt.parseInt(u8, data[i..i+2], 16);
                    uuid_idx += 1;
                    i += 1;
                }
                
                break :blk Value{ .uuid = uuid };
            },
        };
    }
    
    /// Parse error response
    fn parseError(self: *QueryExecutor, parser: *MessageParser) !void {
        _ = self;
        
        // Read error fields until null terminator
        while (true) {
            const field_type = try parser.readByte();
            if (field_type == 0) break;
            
            const field_value = try parser.readStringNoCopy();
            _ = field_value;
            // In real implementation, build error message from fields
        }
        
        return error.QueryError;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TypeOid - toColumnType conversions" {
    try std.testing.expectEqual(Column.Type.boolean, TypeOid.bool.toColumnType());
    try std.testing.expectEqual(Column.Type.int32, TypeOid.int2.toColumnType());
    try std.testing.expectEqual(Column.Type.int32, TypeOid.int4.toColumnType());
    try std.testing.expectEqual(Column.Type.int64, TypeOid.int8.toColumnType());
    try std.testing.expectEqual(Column.Type.float32, TypeOid.float4.toColumnType());
    try std.testing.expectEqual(Column.Type.float64, TypeOid.float8.toColumnType());
    try std.testing.expectEqual(Column.Type.string, TypeOid.text.toColumnType());
    try std.testing.expectEqual(Column.Type.string, TypeOid.varchar.toColumnType());
    try std.testing.expectEqual(Column.Type.bytes, TypeOid.bytea.toColumnType());
    try std.testing.expectEqual(Column.Type.timestamp, TypeOid.timestamp.toColumnType());
    try std.testing.expectEqual(Column.Type.uuid, TypeOid.uuid.toColumnType());
}

test "FormatCode - fromInt" {
    try std.testing.expectEqual(FormatCode.text, FormatCode.fromInt(0));
    try std.testing.expectEqual(FormatCode.binary, FormatCode.fromInt(1));
    try std.testing.expectEqual(FormatCode.text, FormatCode.fromInt(99)); // Unknown defaults to text
}

test "QueryResult - init and deinit" {
    const allocator = std.testing.allocator;
    
    var result = QueryResult.init(allocator);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), result.columns.len);
    try std.testing.expectEqual(@as(usize, 0), result.rows.items.len);
}

test "QueryResult - add columns and rows" {
    const allocator = std.testing.allocator;
    
    var result = QueryResult.init(allocator);
    defer result.deinit();
    
    // Add columns
    var columns = try allocator.alloc(Column, 2);
    columns[0] = Column{ .name = try allocator.dupe(u8, "id"), .type = .int64 };
    columns[1] = Column{ .name = try allocator.dupe(u8, "name"), .type = .string };
    result.columns = columns;
    
    // Add row
    var values = try allocator.alloc(Value, 2);
    values[0] = Value{ .int64 = 1 };
    values[1] = Value{ .string = try allocator.dupe(u8, "test") };
    const row = Row{ .values = values, .allocator = allocator };
    try result.rows.append(row);
    
    try std.testing.expectEqual(@as(usize, 2), result.columns.len);
    try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
}

test "QueryResult - toResultSet" {
    const allocator = std.testing.allocator;
    
    var result = QueryResult.init(allocator);
    
    // Add columns
    var columns = try allocator.alloc(Column, 1);
    columns[0] = Column{ .name = try allocator.dupe(u8, "id"), .type = .int64 };
    result.columns = columns;
    
    // Add row
    var values = try allocator.alloc(Value, 1);
    values[0] = Value{ .int64 = 42 };
    const row = Row{ .values = values, .allocator = allocator };
    try result.rows.append(row);
    
    // Convert to ResultSet
    var rs = try result.toResultSet();
    defer rs.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), rs.len());
    try std.testing.expectEqual(@as(usize, 1), rs.columns.len);
    
    // Original result should be empty (ownership transferred)
    try std.testing.expectEqual(@as(usize, 0), result.columns.len);
    try std.testing.expectEqual(@as(usize, 0), result.rows.items.len);
    
    result.deinit();
}
