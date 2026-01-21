const std = @import("std");
const protocol = @import("protocol.zig");
const connection_mod = @import("connection.zig");
const client_types = @import("../../client.zig");

const SqliteConnection = connection_mod.SqliteConnection;
const SqliteResult = protocol.SqliteResult;
const SqliteType = protocol.SqliteType;
const Row = protocol.Row;
const Value = client_types.Value;
const DbError = @import("../../errors.zig").DbError;

/// SQLite prepared statement
pub const SqliteStatement = struct {
    allocator: std.mem.Allocator,
    connection: *SqliteConnection,
    sql: []const u8,
    param_count: usize,
    
    pub fn init(
        allocator: std.mem.Allocator,
        connection: *SqliteConnection,
        sql: []const u8,
    ) !SqliteStatement {
        return SqliteStatement{
            .allocator = allocator,
            .connection = connection,
            .sql = sql,
            .param_count = countParameters(sql),
        };
    }
    
    pub fn deinit(self: *SqliteStatement) void {
        _ = self;
        // In production: finalize statement
    }
    
    /// Count ? parameters in SQL
    fn countParameters(sql: []const u8) usize {
        var count: usize = 0;
        for (sql) |c| {
            if (c == '?') count += 1;
        }
        return count;
    }
    
    /// Bind parameters to statement
    pub fn bind(self: *SqliteStatement, params: []const Value) !void {
        if (params.len != self.param_count) {
            return DbError.InvalidParameterCount;
        }
        
        // In production: bind each parameter via sqlite3_bind_*
        _ = self;
    }
    
    /// Execute statement and return row count
    pub fn execute(self: *SqliteStatement) !usize {
        _ = self;
        // In production: sqlite3_step until DONE
        return 0;
    }
    
    /// Query and return results
    pub fn query(self: *SqliteStatement, allocator: std.mem.Allocator) !QueryResult {
        _ = self;
        _ = allocator;
        // In production: sqlite3_step, collect rows
        return QueryResult{
            .rows = &[_]Row{},
            .column_count = 0,
        };
    }
    
    /// Reset statement for reuse
    pub fn reset(self: *SqliteStatement) !void {
        _ = self;
        // In production: sqlite3_reset
    }
};

/// Query result set
pub const QueryResult = struct {
    rows: []Row,
    column_count: usize,
    
    pub fn deinit(self: *QueryResult, allocator: std.mem.Allocator) void {
        for (self.rows) |*row| {
            row.deinit();
        }
        allocator.free(self.rows);
    }
};

/// Query executor
pub const QueryExecutor = struct {
    allocator: std.mem.Allocator,
    connection: *SqliteConnection,
    
    pub fn init(allocator: std.mem.Allocator, connection: *SqliteConnection) QueryExecutor {
        return QueryExecutor{
            .allocator = allocator,
            .connection = connection,
        };
    }
    
    /// Execute simple query (no parameters)
    pub fn executeSimple(self: QueryExecutor, sql: []const u8) !usize {
        try self.connection.exec(sql);
        return 0; // Would return affected rows
    }
    
    /// Execute query with parameters
    pub fn executeWithParams(
        self: QueryExecutor,
        sql: []const u8,
        params: []const Value,
    ) !usize {
        var stmt = try SqliteStatement.init(self.allocator, self.connection, sql);
        defer stmt.deinit();
        
        try stmt.bind(params);
        return try stmt.execute();
    }
    
    /// Query and return results
    pub fn query(
        self: QueryExecutor,
        sql: []const u8,
        params: []const Value,
    ) !QueryResult {
        var stmt = try SqliteStatement.init(self.allocator, self.connection, sql);
        defer stmt.deinit();
        
        if (params.len > 0) {
            try stmt.bind(params);
        }
        
        return try stmt.query(self.allocator);
    }
    
    /// Prepare statement for multiple executions
    pub fn prepare(self: QueryExecutor, sql: []const u8) !SqliteStatement {
        return try SqliteStatement.init(self.allocator, self.connection, sql);
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "SqliteStatement - countParameters" {
    try std.testing.expectEqual(@as(usize, 0), SqliteStatement.countParameters("SELECT 1"));
    try std.testing.expectEqual(@as(usize, 1), SqliteStatement.countParameters("SELECT ?"));
    try std.testing.expectEqual(@as(usize, 2), SqliteStatement.countParameters("SELECT ?, ?"));
    try std.testing.expectEqual(@as(usize, 3), SqliteStatement.countParameters("INSERT INTO t VALUES (?, ?, ?)"));
}

test "SqliteStatement - init" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    var stmt = try SqliteStatement.init(allocator, &conn, "SELECT ?");
    defer stmt.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), stmt.param_count);
}

test "QueryExecutor - init" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    const executor = QueryExecutor.init(allocator, &conn);
    try std.testing.expect(executor.connection == &conn);
}

test "SqliteStatement - bind parameter count mismatch" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    var stmt = try SqliteStatement.init(allocator, &conn, "SELECT ?, ?");
    defer stmt.deinit();
    
    // Wrong number of parameters
    const params = [_]Value{Value{ .int32 = 42 }};
    try std.testing.expectError(DbError.InvalidParameterCount, stmt.bind(&params));
}
