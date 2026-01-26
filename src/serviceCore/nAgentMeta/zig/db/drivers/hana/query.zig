const std = @import("std");
const protocol = @import("protocol.zig");
const connection_mod = @import("connection.zig");

/// Query result set
pub const ResultSet = struct {
    allocator: std.mem.Allocator,
    rows: std.ArrayList(Row),
    column_count: usize,
    row_count: usize,
    current_row: usize,
    
    pub fn init(allocator: std.mem.Allocator) ResultSet {
        return ResultSet{
            .allocator = allocator,
            .rows = std.ArrayList(Row){},
            .column_count = 0,
            .row_count = 0,
            .current_row = 0,
        };
    }
    
    pub fn deinit(self: *ResultSet) void {
        for (self.rows.items) |*row| {
            row.deinit();
        }
        self.rows.deinit();
    }
    
    /// Add a row to the result set
    pub fn addRow(self: *ResultSet, row: Row) !void {
        try self.rows.append(row);
        self.row_count += 1;
        if (self.column_count == 0) {
            self.column_count = row.values.items.len;
        }
    }
    
    /// Get next row
    pub fn next(self: *ResultSet) ?*Row {
        if (self.current_row >= self.row_count) {
            return null;
        }
        const row = &self.rows.items[self.current_row];
        self.current_row += 1;
        return row;
    }
    
    /// Reset cursor to beginning
    pub fn reset(self: *ResultSet) void {
        self.current_row = 0;
    }
};

/// Query result row
pub const Row = struct {
    allocator: std.mem.Allocator,
    values: std.ArrayList(Value),
    
    pub fn init(allocator: std.mem.Allocator) Row {
        return Row{
            .allocator = allocator,
            .values = std.ArrayList(Value){},
        };
    }
    
    pub fn deinit(self: *Row) void {
        for (self.values.items) |*value| {
            value.deinit(self.allocator);
        }
        self.values.deinit();
    }
    
    /// Add a value to the row
    pub fn addValue(self: *Row, value: Value) !void {
        try self.values.append(value);
    }
    
    /// Get value by index
    pub fn getValue(self: Row, index: usize) ?Value {
        if (index >= self.values.items.len) {
            return null;
        }
        return self.values.items[index];
    }
};

/// Query value (supports all HANA types)
pub const Value = union(protocol.TypeCode) {
    tinyint: i8,
    smallint: i16,
    integer: i32,
    bigint: i64,
    decimal: f64,
    real: f32,
    double: f64,
    char: []const u8,
    varchar: []const u8,
    nchar: []const u8,
    nvarchar: []const u8,
    binary: []const u8,
    varbinary: []const u8,
    date: i32,
    time: i64,
    timestamp: i64,
    clob: []const u8,
    nclob: []const u8,
    blob: []const u8,
    boolean: bool,
    string: []const u8,
    nstring: []const u8,
    
    pub fn deinit(self: Value, allocator: std.mem.Allocator) void {
        switch (self) {
            .char, .varchar, .nchar, .nvarchar,
            .binary, .varbinary, .clob, .nclob,
            .blob, .string, .nstring => |data| {
                allocator.free(data);
            },
            else => {},
        }
    }
    
    /// Get integer value
    pub fn asInt(self: Value) !i64 {
        return switch (self) {
            .tinyint => |v| @as(i64, v),
            .smallint => |v| @as(i64, v),
            .integer => |v| @as(i64, v),
            .bigint => |v| v,
            else => error.TypeMismatch,
        };
    }
    
    /// Get float value
    pub fn asFloat(self: Value) !f64 {
        return switch (self) {
            .real => |v| @as(f64, v),
            .double => |v| v,
            .decimal => |v| v,
            else => error.TypeMismatch,
        };
    }
    
    /// Get string value
    pub fn asString(self: Value) ![]const u8 {
        return switch (self) {
            .char, .varchar, .nchar, .nvarchar,
            .string, .nstring => |v| v,
            else => error.TypeMismatch,
        };
    }
    
    /// Get boolean value
    pub fn asBool(self: Value) !bool {
        return switch (self) {
            .boolean => |v| v,
            else => error.TypeMismatch,
        };
    }
};

/// Query executor
pub const QueryExecutor = struct {
    allocator: std.mem.Allocator,
    connection: *connection_mod.HanaConnection,
    
    pub fn init(allocator: std.mem.Allocator, connection: *connection_mod.HanaConnection) QueryExecutor {
        return QueryExecutor{
            .allocator = allocator,
            .connection = connection,
        };
    }
    
    /// Execute a simple query
    pub fn executeQuery(self: *QueryExecutor, sql: []const u8) !ResultSet {
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Create execute_direct segment
        var segment = protocol.SegmentHeader.init(.execute_direct, 1);
        
        // In real implementation:
        // 1. Add command part with SQL
        // 2. Send segment
        // 3. Receive response
        // 4. Parse result set
        
        _ = sql;
        
        var result = ResultSet.init(self.allocator);
        
        // Simplified: return empty result
        return result;
    }
    
    /// Execute a prepared statement
    pub fn executePrepared(
        self: *QueryExecutor,
        statement_id: i64,
        params: []const Value,
    ) !ResultSet {
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Create execute_prepared segment
        var segment = protocol.SegmentHeader.init(.execute_prepared, 2);
        
        // In real implementation:
        // 1. Add result_set_id part
        // 2. Add parameters part
        // 3. Send segment
        // 4. Receive response
        // 5. Parse result set
        
        _ = statement_id;
        _ = params;
        
        var result = ResultSet.init(self.allocator);
        return result;
    }
    
    /// Prepare a statement
    pub fn prepareStatement(self: *QueryExecutor, sql: []const u8) !i64 {
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Create prepare segment
        var segment = protocol.SegmentHeader.init(.prepare, 1);
        
        // In real implementation:
        // 1. Add command part with SQL
        // 2. Send segment
        // 3. Receive response
        // 4. Extract statement_id
        
        _ = sql;
        _ = segment;
        
        return 1; // Placeholder statement ID
    }
    
    /// Fetch more rows from a result set
    pub fn fetch(self: *QueryExecutor, result_set_id: i64, num_rows: usize) !ResultSet {
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Create fetch segment
        var segment = protocol.SegmentHeader.init(.fetch, 1);
        
        // In real implementation:
        // 1. Add result_set_id part
        // 2. Send segment with num_rows
        // 3. Receive response
        // 4. Parse result rows
        
        _ = result_set_id;
        _ = num_rows;
        _ = segment;
        
        var result = ResultSet.init(self.allocator);
        return result;
    }
    
    /// Close a result set
    pub fn closeResultSet(self: *QueryExecutor, result_set_id: i64) !void {
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Create close_result_set segment
        var segment = protocol.SegmentHeader.init(.close_result_set, 1);
        
        // In real implementation:
        // 1. Add result_set_id part
        // 2. Send segment
        // 3. Receive acknowledgment
        
        _ = result_set_id;
        _ = segment;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Value - integer conversion" {
    const v1 = Value{ .integer = 42 };
    const i = try v1.asInt();
    try std.testing.expectEqual(@as(i64, 42), i);
    
    const v2 = Value{ .varchar = "hello" };
    try std.testing.expectError(error.TypeMismatch, v2.asInt());
}

test "Value - float conversion" {
    const v1 = Value{ .double = 3.14 };
    const f = try v1.asFloat();
    try std.testing.expectApproxEqAbs(3.14, f, 0.001);
    
    const v2 = Value{ .integer = 42 };
    try std.testing.expectError(error.TypeMismatch, v2.asFloat());
}

test "Value - string conversion" {
    const v1 = Value{ .varchar = "test" };
    const s = try v1.asString();
    try std.testing.expectEqualStrings("test", s);
    
    const v2 = Value{ .integer = 42 };
    try std.testing.expectError(error.TypeMismatch, v2.asString());
}

test "Value - boolean conversion" {
    const v1 = Value{ .boolean = true };
    const b = try v1.asBool();
    try std.testing.expect(b);
    
    const v2 = Value{ .integer = 1 };
    try std.testing.expectError(error.TypeMismatch, v2.asBool());
}

test "Row - init and deinit" {
    const allocator = std.testing.allocator;
    
    var row = Row.init(allocator);
    defer row.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), row.values.items.len);
}

test "Row - add and get values" {
    const allocator = std.testing.allocator;
    
    var row = Row.init(allocator);
    defer row.deinit();
    
    try row.addValue(Value{ .integer = 42 });
    try row.addValue(Value{ .double = 3.14 });
    
    try std.testing.expectEqual(@as(usize, 2), row.values.items.len);
    
    const v1 = row.getValue(0);
    try std.testing.expect(v1 != null);
    try std.testing.expectEqual(@as(i32, 42), v1.?.integer);
}

test "ResultSet - init and deinit" {
    const allocator = std.testing.allocator;
    
    var result = ResultSet.init(allocator);
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), result.row_count);
}

test "ResultSet - add rows and iterate" {
    const allocator = std.testing.allocator;
    
    var result = ResultSet.init(allocator);
    defer result.deinit();
    
    var row1 = Row.init(allocator);
    try row1.addValue(Value{ .integer = 1 });
    try result.addRow(row1);
    
    var row2 = Row.init(allocator);
    try row2.addValue(Value{ .integer = 2 });
    try result.addRow(row2);
    
    try std.testing.expectEqual(@as(usize, 2), result.row_count);
    
    var count: usize = 0;
    while (result.next()) |_| {
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 2), count);
}
