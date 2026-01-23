const std = @import("std");

/// Database dialect enum
pub const Dialect = enum {
    postgres,
    hana,
    sqlite,

    pub fn format(self: Dialect, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.writeAll(@tagName(self));
    }
};

/// Cross-database value type
pub const Value = union(enum) {
    null,
    bool: bool,
    int32: i32,
    int64: i64,
    float32: f32,
    float64: f64,
    string: []const u8,
    bytes: []const u8,
    timestamp: i64, // Unix timestamp in milliseconds
    uuid: [16]u8,

    /// Format value for debugging
    pub fn format(self: Value, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        switch (self) {
            .null => try writer.writeAll("null"),
            .bool => |v| try writer.print("{}", .{v}),
            .int32 => |v| try writer.print("{}", .{v}),
            .int64 => |v| try writer.print("{}", .{v}),
            .float32 => |v| try writer.print("{d}", .{v}),
            .float64 => |v| try writer.print("{d}", .{v}),
            .string => |v| try writer.print("\"{s}\"", .{v}),
            .bytes => |v| try writer.print("<{d} bytes>", .{v.len}),
            .timestamp => |v| try writer.print("timestamp({d})", .{v}),
            .uuid => |v| {
                try writer.writeAll("uuid(");
                for (v, 0..) |byte, i| {
                    if (i == 4 or i == 6 or i == 8 or i == 10) try writer.writeAll("-");
                    try writer.print("{x:0>2}", .{byte});
                }
                try writer.writeAll(")");
            },
        }
    }

    /// Check if value is null
    pub fn isNull(self: Value) bool {
        return self == .null;
    }

    /// Get as boolean (errors if wrong type)
    pub fn asBool(self: Value) !bool {
        return switch (self) {
            .bool => |v| v,
            else => error.TypeMismatch,
        };
    }

    /// Get as int64 (coerce from int32 if needed)
    pub fn asInt64(self: Value) !i64 {
        return switch (self) {
            .int32 => |v| @as(i64, v),
            .int64 => |v| v,
            else => error.TypeMismatch,
        };
    }

    /// Get as string
    pub fn asString(self: Value) ![]const u8 {
        return switch (self) {
            .string => |v| v,
            else => error.TypeMismatch,
        };
    }
};

/// Column metadata
pub const Column = struct {
    name: []const u8,
    type: Type,

    pub const Type = enum {
        boolean,
        int32,
        int64,
        float32,
        float64,
        string,
        bytes,
        timestamp,
        uuid,
    };
};

/// Query result row
pub const Row = struct {
    values: []Value,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *Row) void {
        // Free any allocated strings or bytes in values
        for (self.values) |val| {
            switch (val) {
                .string => |s| self.allocator.free(s),
                .bytes => |b| self.allocator.free(b),
                else => {},
            }
        }
        self.allocator.free(self.values);
    }

    /// Get value by index
    pub fn get(self: Row, index: usize) !Value {
        if (index >= self.values.len) return error.IndexOutOfBounds;
        return self.values[index];
    }

    /// Get value by column name (requires columns metadata)
    pub fn getByName(self: Row, columns: []const Column, name: []const u8) !Value {
        for (columns, 0..) |col, i| {
            if (std.mem.eql(u8, col.name, name)) {
                return self.get(i);
            }
        }
        return error.ColumnNotFound;
    }
};

/// Query result set
pub const ResultSet = struct {
    columns: []Column,
    rows: []Row,
    allocator: std.mem.Allocator,

    pub fn deinit(self: *ResultSet) void {
        // Free rows
        for (self.rows) |*row| {
            row.deinit();
        }
        self.allocator.free(self.rows);

        // Free column names
        for (self.columns) |col| {
            self.allocator.free(col.name);
        }
        self.allocator.free(self.columns);
    }

    /// Get number of rows
    pub fn len(self: ResultSet) usize {
        return self.rows.len;
    }

    /// Get row by index
    pub fn getRow(self: ResultSet, index: usize) !Row {
        if (index >= self.rows.len) return error.IndexOutOfBounds;
        return self.rows[index];
    }

    /// Iterate over rows
    pub fn iterator(self: *ResultSet) ResultSetIterator {
        return ResultSetIterator{
            .result_set = self,
            .index = 0,
        };
    }
};

/// Iterator for result set rows
pub const ResultSetIterator = struct {
    result_set: *ResultSet,
    index: usize,

    pub fn next(self: *ResultSetIterator) ?Row {
        if (self.index >= self.result_set.rows.len) return null;
        const row = self.result_set.rows[self.index];
        self.index += 1;
        return row;
    }
};

/// Prepared statement interface
pub const PreparedStatement = struct {
    vtable: *const VTable,
    context: *anyopaque,

    pub const VTable = struct {
        execute: *const fn (*anyopaque, []const Value) anyerror!ResultSet,
        close: *const fn (*anyopaque) void,
    };

    pub fn execute(self: *PreparedStatement, params: []const Value) !ResultSet {
        return self.vtable.execute(self.context, params);
    }

    pub fn close(self: *PreparedStatement) void {
        self.vtable.close(self.context);
    }
};

/// Transaction interface
pub const Transaction = struct {
    vtable: *const VTable,
    context: *anyopaque,

    pub const VTable = struct {
        commit: *const fn (*anyopaque) anyerror!void,
        rollback: *const fn (*anyopaque) anyerror!void,
        savepoint: *const fn (*anyopaque, []const u8) anyerror!void,
        rollback_to: *const fn (*anyopaque, []const u8) anyerror!void,
        execute: *const fn (*anyopaque, []const u8, []const Value) anyerror!ResultSet,
        prepare: *const fn (*anyopaque, []const u8) anyerror!*PreparedStatement,
    };

    pub fn commit(self: *Transaction) !void {
        return self.vtable.commit(self.context);
    }

    pub fn rollback(self: *Transaction) !void {
        return self.vtable.rollback(self.context);
    }

    pub fn savepoint(self: *Transaction, name: []const u8) !void {
        return self.vtable.savepoint(self.context, name);
    }

    pub fn rollbackTo(self: *Transaction, name: []const u8) !void {
        return self.vtable.rollback_to(self.context, name);
    }

    pub fn execute(self: *Transaction, sql: []const u8, params: []const Value) !ResultSet {
        return self.vtable.execute(self.context, sql, params);
    }

    pub fn prepare(self: *Transaction, sql: []const u8) !*PreparedStatement {
        return self.vtable.prepare(self.context, sql);
    }
};

/// Isolation level for transactions
pub const IsolationLevel = enum {
    read_uncommitted,
    read_committed,
    repeatable_read,
    serializable,

    pub fn toSQL(self: IsolationLevel) []const u8 {
        return switch (self) {
            .read_uncommitted => "READ UNCOMMITTED",
            .read_committed => "READ COMMITTED",
            .repeatable_read => "REPEATABLE READ",
            .serializable => "SERIALIZABLE",
        };
    }
};

/// Database client interface (trait-like using VTable pattern)
pub const DbClient = struct {
    vtable: *const VTable,
    context: *anyopaque,
    allocator: std.mem.Allocator,

    /// VTable for polymorphic database operations
    pub const VTable = struct {
        /// Connect to database with connection string
        connect: *const fn (*anyopaque, []const u8) anyerror!void,

        /// Disconnect from database
        disconnect: *const fn (*anyopaque) void,

        /// Execute SQL query and return results
        execute: *const fn (*anyopaque, []const u8, []const Value) anyerror!ResultSet,

        /// Prepare SQL statement for repeated execution
        prepare: *const fn (*anyopaque, []const u8) anyerror!*PreparedStatement,

        /// Begin transaction
        begin: *const fn (*anyopaque, IsolationLevel) anyerror!*Transaction,

        /// Ping database to check connection
        ping: *const fn (*anyopaque) anyerror!bool,

        /// Get database dialect
        get_dialect: *const fn (*anyopaque) Dialect,

        /// Get last error message
        get_last_error: *const fn (*anyopaque) ?[]const u8,
    };

    /// Connect to database
    pub fn connect(self: *DbClient, connection_string: []const u8) !void {
        return self.vtable.connect(self.context, connection_string);
    }

    /// Disconnect from database
    pub fn disconnect(self: *DbClient) void {
        self.vtable.disconnect(self.context);
    }

    /// Execute SQL query with parameters
    pub fn execute(self: *DbClient, sql: []const u8, params: []const Value) !ResultSet {
        return self.vtable.execute(self.context, sql, params);
    }

    /// Prepare SQL statement
    pub fn prepare(self: *DbClient, sql: []const u8) !*PreparedStatement {
        return self.vtable.prepare(self.context, sql);
    }

    /// Begin transaction with default isolation level
    pub fn begin(self: *DbClient) !*Transaction {
        return self.beginWithIsolation(.read_committed);
    }

    /// Begin transaction with specific isolation level
    pub fn beginWithIsolation(self: *DbClient, level: IsolationLevel) !*Transaction {
        return self.vtable.begin(self.context, level);
    }

    /// Ping database
    pub fn ping(self: *DbClient) !bool {
        return self.vtable.ping(self.context);
    }

    /// Get database dialect
    pub fn getDialect(self: DbClient) Dialect {
        return self.vtable.get_dialect(self.context);
    }

    /// Get last error message
    pub fn getLastError(self: DbClient) ?[]const u8 {
        return self.vtable.get_last_error(self.context);
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "Value - basic types" {
    const v_null = Value{ .null = {} };
    try std.testing.expect(v_null.isNull());

    const v_bool = Value{ .bool = true };
    try std.testing.expect(!v_bool.isNull());
    try std.testing.expectEqual(true, try v_bool.asBool());

    const v_int32 = Value{ .int32 = 42 };
    try std.testing.expectEqual(@as(i64, 42), try v_int32.asInt64());

    const v_int64 = Value{ .int64 = 1234567890 };
    try std.testing.expectEqual(@as(i64, 1234567890), try v_int64.asInt64());

    const v_string = Value{ .string = "hello" };
    try std.testing.expectEqualStrings("hello", try v_string.asString());
}

test "Value - type mismatch errors" {
    const v_int = Value{ .int32 = 42 };
    try std.testing.expectError(error.TypeMismatch, v_int.asBool());
    try std.testing.expectError(error.TypeMismatch, v_int.asString());

    const v_bool = Value{ .bool = true };
    try std.testing.expectError(error.TypeMismatch, v_bool.asInt64());
}

test "Column - metadata" {
    const col = Column{
        .name = "id",
        .type = .int64,
    };
    try std.testing.expectEqualStrings("id", col.name);
    try std.testing.expectEqual(Column.Type.int64, col.type);
}

test "Row - basic operations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var values = try allocator.alloc(Value, 3);
    values[0] = Value{ .int64 = 1 };
    values[1] = Value{ .string = "test" };
    values[2] = Value{ .bool = true };

    var row = Row{
        .values = values,
        .allocator = allocator,
    };

    try std.testing.expectEqual(@as(i64, 1), try (try row.get(0)).asInt64());
    try std.testing.expectEqualStrings("test", try (try row.get(1)).asString());
    try std.testing.expectEqual(true, try (try row.get(2)).asBool());

    try std.testing.expectError(error.IndexOutOfBounds, row.get(3));
}

test "ResultSet - basic operations" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create columns
    var columns = try allocator.alloc(Column, 2);
    columns[0] = Column{ .name = try allocator.dupe(u8, "id"), .type = .int64 };
    columns[1] = Column{ .name = try allocator.dupe(u8, "name"), .type = .string };

    // Create rows
    var rows = try allocator.alloc(Row, 2);
    
    var values1 = try allocator.alloc(Value, 2);
    values1[0] = Value{ .int64 = 1 };
    values1[1] = Value{ .string = "Alice" };
    rows[0] = Row{ .values = values1, .allocator = allocator };

    var values2 = try allocator.alloc(Value, 2);
    values2[0] = Value{ .int64 = 2 };
    values2[1] = Value{ .string = "Bob" };
    rows[1] = Row{ .values = values2, .allocator = allocator };

    var result = ResultSet{
        .columns = columns,
        .rows = rows,
        .allocator = allocator,
    };

    try std.testing.expectEqual(@as(usize, 2), result.len());

    const row1 = try result.getRow(0);
    try std.testing.expectEqual(@as(i64, 1), try (try row1.get(0)).asInt64());
    try std.testing.expectEqualStrings("Alice", try (try row1.get(1)).asString());
}

test "ResultSet - iterator" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var columns = try allocator.alloc(Column, 1);
    columns[0] = Column{ .name = try allocator.dupe(u8, "id"), .type = .int64 };

    const rows = try allocator.alloc(Row, 3);
    for (rows, 0..) |*row, i| {
        var values = try allocator.alloc(Value, 1);
        values[0] = Value{ .int64 = @as(i64, @intCast(i)) };
        row.* = Row{ .values = values, .allocator = allocator };
    }

    var result = ResultSet{
        .columns = columns,
        .rows = rows,
        .allocator = allocator,
    };

    var iter = result.iterator();
    var count: usize = 0;
    while (iter.next()) |_| {
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 3), count);
}

test "Dialect - format" {
    const postgres = Dialect.postgres;
    const hana = Dialect.hana;
    const sqlite = Dialect.sqlite;

    var buf: [20]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    
    try fbs.writer().print("{}", .{postgres});
    try std.testing.expectEqualStrings("postgres", fbs.getWritten());

    fbs.reset();
    try fbs.writer().print("{}", .{hana});
    try std.testing.expectEqualStrings("hana", fbs.getWritten());

    fbs.reset();
    try fbs.writer().print("{}", .{sqlite});
    try std.testing.expectEqualStrings("sqlite", fbs.getWritten());
}

test "IsolationLevel - toSQL" {
    try std.testing.expectEqualStrings("READ UNCOMMITTED", IsolationLevel.read_uncommitted.toSQL());
    try std.testing.expectEqualStrings("READ COMMITTED", IsolationLevel.read_committed.toSQL());
    try std.testing.expectEqualStrings("REPEATABLE READ", IsolationLevel.repeatable_read.toSQL());
    try std.testing.expectEqualStrings("SERIALIZABLE", IsolationLevel.serializable.toSQL());
}
