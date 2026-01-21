const std = @import("std");
const connection_mod = @import("connection.zig");
const protocol = @import("protocol.zig");
const client_types = @import("../../client.zig");

const SqliteConnection = connection_mod.SqliteConnection;
const TransactionState = protocol.TransactionState;
const IsolationLevel = client_types.IsolationLevel;
const DbError = @import("../../errors.zig").DbError;

/// SQLite transaction manager
pub const SqliteTransaction = struct {
    allocator: std.mem.Allocator,
    connection: *SqliteConnection,
    state: TransactionState,
    isolation_level: IsolationLevel,
    savepoints: std.ArrayList([]const u8),
    
    pub fn init(
        allocator: std.mem.Allocator,
        connection: *SqliteConnection,
        isolation_level: IsolationLevel,
    ) !SqliteTransaction {
        return SqliteTransaction{
            .allocator = allocator,
            .connection = connection,
            .state = .none,
            .isolation_level = isolation_level,
            .savepoints = std.ArrayList([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *SqliteTransaction) void {
        // Clean up any remaining savepoints
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp);
        }
        self.savepoints.deinit();
    }
    
    /// Begin transaction
    pub fn begin(self: *SqliteTransaction) !void {
        if (self.state != .none) {
            return DbError.TransactionInProgress;
        }
        
        // SQLite transaction types based on isolation level
        const tx_type = switch (self.isolation_level) {
            .read_uncommitted, .read_committed => "DEFERRED",
            .repeatable_read => "IMMEDIATE",
            .serializable => "EXCLUSIVE",
        };
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "BEGIN {s} TRANSACTION",
            .{tx_type},
        );
        defer self.allocator.free(sql);
        
        try self.connection.exec(sql);
        
        self.state = switch (self.isolation_level) {
            .read_uncommitted, .read_committed => .deferred,
            .repeatable_read => .immediate,
            .serializable => .exclusive,
        };
    }
    
    /// Commit transaction
    pub fn commit(self: *SqliteTransaction) !void {
        if (self.state == .none) {
            return DbError.NoActiveTransaction;
        }
        
        try self.connection.exec("COMMIT");
        self.state = .none;
        
        // Clear savepoints on commit
        self.clearSavepoints();
    }
    
    /// Rollback transaction
    pub fn rollback(self: *SqliteTransaction) !void {
        if (self.state == .none) {
            return DbError.NoActiveTransaction;
        }
        
        try self.connection.exec("ROLLBACK");
        self.state = .none;
        
        // Clear savepoints on rollback
        self.clearSavepoints();
    }
    
    /// Create savepoint
    pub fn savepoint(self: *SqliteTransaction, name: []const u8) !void {
        if (self.state == .none) {
            return DbError.NoActiveTransaction;
        }
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "SAVEPOINT {s}",
            .{name},
        );
        defer self.allocator.free(sql);
        
        try self.connection.exec(sql);
        
        // Store savepoint name
        const name_copy = try self.allocator.dupe(u8, name);
        try self.savepoints.append(name_copy);
    }
    
    /// Rollback to savepoint
    pub fn rollbackToSavepoint(self: *SqliteTransaction, name: []const u8) !void {
        if (self.state == .none) {
            return DbError.NoActiveTransaction;
        }
        
        // Check if savepoint exists
        var found = false;
        var remove_from: usize = 0;
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp, name)) {
                found = true;
                remove_from = i;
                break;
            }
        }
        
        if (!found) {
            return DbError.SavepointNotFound;
        }
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "ROLLBACK TO SAVEPOINT {s}",
            .{name},
        );
        defer self.allocator.free(sql);
        
        try self.connection.exec(sql);
        
        // Remove savepoints created after this one
        while (self.savepoints.items.len > remove_from + 1) {
            const sp = self.savepoints.pop();
            self.allocator.free(sp);
        }
    }
    
    /// Release savepoint
    pub fn releaseSavepoint(self: *SqliteTransaction, name: []const u8) !void {
        if (self.state == .none) {
            return DbError.NoActiveTransaction;
        }
        
        const sql = try std.fmt.allocPrint(
            self.allocator,
            "RELEASE SAVEPOINT {s}",
            .{name},
        );
        defer self.allocator.free(sql);
        
        try self.connection.exec(sql);
        
        // Remove savepoint from list
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp, name)) {
                _ = self.savepoints.orderedRemove(i);
                self.allocator.free(sp);
                break;
            }
        }
    }
    
    /// Clear all savepoints
    fn clearSavepoints(self: *SqliteTransaction) void {
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp);
        }
        self.savepoints.clearRetainingCapacity();
    }
    
    /// Get current transaction state
    pub fn getState(self: SqliteTransaction) TransactionState {
        return self.state;
    }
    
    /// Check if transaction is active
    pub fn isActive(self: SqliteTransaction) bool {
        return self.state != .none;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "SqliteTransaction - init and deinit" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx.deinit();
    
    try std.testing.expectEqual(TransactionState.none, tx.state);
    try std.testing.expectEqual(@as(usize, 0), tx.savepoints.items.len);
}

test "SqliteTransaction - begin and commit" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    try conn.connect();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx.deinit();
    
    try tx.begin();
    try std.testing.expect(tx.isActive());
    
    try tx.commit();
    try std.testing.expect(!tx.isActive());
}

test "SqliteTransaction - begin and rollback" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    try conn.connect();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .serializable);
    defer tx.deinit();
    
    try tx.begin();
    try std.testing.expectEqual(TransactionState.exclusive, tx.state);
    
    try tx.rollback();
    try std.testing.expectEqual(TransactionState.none, tx.state);
}

test "SqliteTransaction - savepoint operations" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    try conn.connect();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx.deinit();
    
    try tx.begin();
    
    try tx.savepoint("sp1");
    try std.testing.expectEqual(@as(usize, 1), tx.savepoints.items.len);
    
    try tx.savepoint("sp2");
    try std.testing.expectEqual(@as(usize, 2), tx.savepoints.items.len);
    
    try tx.rollbackToSavepoint("sp1");
    try std.testing.expectEqual(@as(usize, 1), tx.savepoints.items.len);
}

test "SqliteTransaction - isolation levels" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    try conn.connect();
    
    // Test different isolation levels
    var tx1 = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx1.deinit();
    try tx1.begin();
    try std.testing.expectEqual(TransactionState.deferred, tx1.state);
    try tx1.rollback();
    
    var tx2 = try SqliteTransaction.init(allocator, &conn, .repeatable_read);
    defer tx2.deinit();
    try tx2.begin();
    try std.testing.expectEqual(TransactionState.immediate, tx2.state);
    try tx2.rollback();
    
    var tx3 = try SqliteTransaction.init(allocator, &conn, .serializable);
    defer tx3.deinit();
    try tx3.begin();
    try std.testing.expectEqual(TransactionState.exclusive, tx3.state);
    try tx3.rollback();
}

test "SqliteTransaction - double begin error" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    try conn.connect();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx.deinit();
    
    try tx.begin();
    try std.testing.expectError(DbError.TransactionInProgress, tx.begin());
}

test "SqliteTransaction - commit without begin error" {
    const allocator = std.testing.allocator;
    const config = protocol.SqliteConfig.inMemory();
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = try SqliteTransaction.init(allocator, &conn, .read_committed);
    defer tx.deinit();
    
    try std.testing.expectError(DbError.NoActiveTransaction, tx.commit());
}
