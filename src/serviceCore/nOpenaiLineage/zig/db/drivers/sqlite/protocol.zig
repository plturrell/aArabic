const std = @import("std");
const client_types = @import("../../client.zig");

const Value = client_types.Value;
const DbError = @import("../../errors.zig").DbError;

/// SQLite type codes
pub const SqliteType = enum(u8) {
    integer = 1,
    float = 2,
    text = 3,
    blob = 4,
    null = 5,
    
    pub fn fromValue(value: Value) SqliteType {
        return switch (value) {
            .null => .null,
            .bool, .int32, .int64 => .integer,
            .float32, .float64 => .float,
            .string => .text,
            .bytes, .uuid => .blob,
            .timestamp => .integer,
        };
    }
    
    pub fn toValueType(self: SqliteType) type {
        return switch (self) {
            .integer => i64,
            .float => f64,
            .text => []const u8,
            .blob => []const u8,
            .null => void,
        };
    }
};

/// SQLite result codes
pub const SqliteResult = enum(c_int) {
    ok = 0,
    error_generic = 1,
    internal = 2,
    perm = 3,
    abort = 4,
    busy = 5,
    locked = 6,
    nomem = 7,
    readonly = 8,
    interrupt = 9,
    ioerr = 10,
    corrupt = 11,
    notfound = 12,
    full = 13,
    cantopen = 14,
    protocol = 15,
    empty = 16,
    schema = 17,
    toobig = 18,
    constraint = 19,
    mismatch = 20,
    misuse = 21,
    nolfs = 22,
    auth = 23,
    format = 24,
    range = 25,
    notadb = 26,
    notice = 27,
    warning = 28,
    row = 100,
    done = 101,
    
    pub fn isError(self: SqliteResult) bool {
        return @intFromEnum(self) != @intFromEnum(SqliteResult.ok) and
               @intFromEnum(self) != @intFromEnum(SqliteResult.row) and
               @intFromEnum(self) != @intFromEnum(SqliteResult.done);
    }
    
    pub fn toDbError(self: SqliteResult) DbError {
        return switch (self) {
            .ok, .row, .done => DbError.Success,
            .busy, .locked => DbError.ConnectionBusy,
            .nomem => DbError.OutOfMemory,
            .constraint => DbError.ConstraintViolation,
            .corrupt, .notadb => DbError.DatabaseCorrupt,
            .cantopen => DbError.ConnectionFailed,
            .readonly => DbError.ReadOnlyDatabase,
            .interrupt => DbError.QueryInterrupted,
            else => DbError.UnknownError,
        };
    }
    
    pub fn errorMessage(self: SqliteResult) []const u8 {
        return switch (self) {
            .ok => "Success",
            .error_generic => "SQL error or missing database",
            .internal => "Internal logic error in SQLite",
            .perm => "Access permission denied",
            .abort => "Callback routine requested an abort",
            .busy => "The database file is locked",
            .locked => "A table in the database is locked",
            .nomem => "Memory allocation failed",
            .readonly => "Attempt to write a readonly database",
            .interrupt => "Operation terminated by interrupt",
            .ioerr => "Disk I/O error",
            .corrupt => "The database disk image is malformed",
            .notfound => "Unknown opcode or table/index not found",
            .full => "Insertion failed because database is full",
            .cantopen => "Unable to open the database file",
            .protocol => "Database lock protocol error",
            .empty => "Database is empty",
            .schema => "The database schema changed",
            .toobig => "String or BLOB exceeds size limit",
            .constraint => "Abort due to constraint violation",
            .mismatch => "Data type mismatch",
            .misuse => "Library used incorrectly",
            .nolfs => "Uses OS features not supported on host",
            .auth => "Authorization denied",
            .format => "Auxiliary database format error",
            .range => "Bind parameter out of range",
            .notadb => "File opened that is not a database file",
            .notice => "Notifications from sqlite3_log()",
            .warning => "Warnings from sqlite3_log()",
            .row => "Step has another row ready",
            .done => "Step has finished executing",
        };
    }
};

/// SQLite open flags
pub const SqliteOpenFlags = packed struct(c_int) {
    readonly: bool = false,
    readwrite: bool = true,
    create: bool = true,
    deleteonclose: bool = false,
    exclusive: bool = false,
    autoproxy: bool = false,
    uri: bool = false,
    memory: bool = false,
    main_db: bool = false,
    temp_db: bool = false,
    transient_db: bool = false,
    main_journal: bool = false,
    temp_journal: bool = false,
    subjournal: bool = false,
    super_journal: bool = false,
    nomutex: bool = false,
    fullmutex: bool = false,
    sharedcache: bool = false,
    privatecache: bool = false,
    wal: bool = false,
    nofollow: bool = false,
    exrescode: bool = false,
    _padding: u10 = 0,
    
    pub fn default() SqliteOpenFlags {
        return SqliteOpenFlags{
            .readwrite = true,
            .create = true,
        };
    }
    
    pub fn inMemory() SqliteOpenFlags {
        return SqliteOpenFlags{
            .readwrite = true,
            .create = true,
            .memory = true,
        };
    }
    
    pub fn readOnly() SqliteOpenFlags {
        return SqliteOpenFlags{
            .readonly = true,
        };
    }
};

/// SQLite prepared statement lifecycle
pub const StatementState = enum {
    prepared,
    bound,
    executed,
    reset,
    finalized,
};

/// SQLite transaction state
pub const TransactionState = enum {
    none,
    deferred,
    immediate,
    exclusive,
};

/// Value binding for prepared statements
pub const Binding = struct {
    index: usize,
    value: Value,
    
    pub fn init(index: usize, value: Value) Binding {
        return Binding{
            .index = index,
            .value = value,
        };
    }
};

/// Column metadata
pub const ColumnMeta = struct {
    name: []const u8,
    type_code: SqliteType,
    nullable: bool,
    
    pub fn init(name: []const u8, type_code: SqliteType, nullable: bool) ColumnMeta {
        return ColumnMeta{
            .name = name,
            .type_code = type_code,
            .nullable = nullable,
        };
    }
};

/// Query result row
pub const Row = struct {
    columns: []Value,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, column_count: usize) !Row {
        const columns = try allocator.alloc(Value, column_count);
        return Row{
            .columns = columns,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Row) void {
        for (self.columns) |*col| {
            switch (col.*) {
                .string => |s| self.allocator.free(s),
                .bytes => |b| self.allocator.free(b),
                else => {},
            }
        }
        self.allocator.free(self.columns);
    }
    
    pub fn get(self: Row, index: usize) ?Value {
        if (index >= self.columns.len) return null;
        return self.columns[index];
    }
};

/// SQLite connection configuration
pub const SqliteConfig = struct {
    path: []const u8,
    flags: SqliteOpenFlags = SqliteOpenFlags.default(),
    busy_timeout_ms: u32 = 5000,
    journal_mode: JournalMode = .wal,
    synchronous: Synchronous = .normal,
    cache_size: i32 = -2000, // 2MB default
    
    pub const JournalMode = enum {
        delete,
        truncate,
        persist,
        memory,
        wal,
        off,
        
        pub fn toSql(self: JournalMode) []const u8 {
            return switch (self) {
                .delete => "DELETE",
                .truncate => "TRUNCATE",
                .persist => "PERSIST",
                .memory => "MEMORY",
                .wal => "WAL",
                .off => "OFF",
            };
        }
    };
    
    pub const Synchronous = enum {
        off,
        normal,
        full,
        extra,
        
        pub fn toSql(self: Synchronous) []const u8 {
            return switch (self) {
                .off => "OFF",
                .normal => "NORMAL",
                .full => "FULL",
                .extra => "EXTRA",
            };
        }
    };
    
    pub fn inMemory() SqliteConfig {
        return SqliteConfig{
            .path = ":memory:",
            .flags = SqliteOpenFlags.inMemory(),
        };
    }
    
    pub fn file(path: []const u8) SqliteConfig {
        return SqliteConfig{
            .path = path,
        };
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "SqliteType - fromValue" {
    try std.testing.expectEqual(SqliteType.null, SqliteType.fromValue(Value.null));
    try std.testing.expectEqual(SqliteType.integer, SqliteType.fromValue(Value{ .int32 = 42 }));
    try std.testing.expectEqual(SqliteType.float, SqliteType.fromValue(Value{ .float64 = 3.14 }));
    try std.testing.expectEqual(SqliteType.text, SqliteType.fromValue(Value{ .string = "test" }));
    try std.testing.expectEqual(SqliteType.blob, SqliteType.fromValue(Value{ .bytes = &[_]u8{1, 2, 3} }));
}

test "SqliteResult - isError" {
    try std.testing.expect(!SqliteResult.ok.isError());
    try std.testing.expect(!SqliteResult.row.isError());
    try std.testing.expect(!SqliteResult.done.isError());
    try std.testing.expect(SqliteResult.error_generic.isError());
    try std.testing.expect(SqliteResult.busy.isError());
}

test "SqliteResult - toDbError" {
    try std.testing.expectEqual(DbError.Success, SqliteResult.ok.toDbError());
    try std.testing.expectEqual(DbError.ConnectionBusy, SqliteResult.busy.toDbError());
    try std.testing.expectEqual(DbError.OutOfMemory, SqliteResult.nomem.toDbError());
    try std.testing.expectEqual(DbError.ConstraintViolation, SqliteResult.constraint.toDbError());
}

test "SqliteOpenFlags - default" {
    const flags = SqliteOpenFlags.default();
    try std.testing.expect(!flags.readonly);
    try std.testing.expect(flags.readwrite);
    try std.testing.expect(flags.create);
}

test "SqliteOpenFlags - inMemory" {
    const flags = SqliteOpenFlags.inMemory();
    try std.testing.expect(flags.memory);
    try std.testing.expect(flags.readwrite);
    try std.testing.expect(flags.create);
}

test "SqliteConfig - inMemory" {
    const config = SqliteConfig.inMemory();
    try std.testing.expectEqualStrings(":memory:", config.path);
    try std.testing.expect(config.flags.memory);
}

test "SqliteConfig - file" {
    const config = SqliteConfig.file("test.db");
    try std.testing.expectEqualStrings("test.db", config.path);
    try std.testing.expect(!config.flags.memory);
}

test "Binding - init" {
    const binding = Binding.init(1, Value{ .int32 = 42 });
    try std.testing.expectEqual(@as(usize, 1), binding.index);
    try std.testing.expectEqual(@as(i32, 42), binding.value.int32);
}

test "Row - init and deinit" {
    const allocator = std.testing.allocator;
    
    var row = try Row.init(allocator, 3);
    defer row.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), row.columns.len);
}
