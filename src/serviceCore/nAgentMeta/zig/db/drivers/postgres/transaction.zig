const std = @import("std");
const query_mod = @import("query.zig");
const client_types = @import("../../client.zig");
const QueryExecutor = query_mod.QueryExecutor;
const Value = client_types.Value;
const ResultSet = client_types.ResultSet;
const IsolationLevel = client_types.IsolationLevel;

/// Transaction state
pub const TransactionState = enum {
    idle,           // No transaction active
    active,         // Transaction in progress
    failed,         // Transaction failed (must rollback)
    committed,      // Transaction committed
    rolled_back,    // Transaction rolled back
    
    pub fn isActive(self: TransactionState) bool {
        return self == .active;
    }
    
    pub fn canExecute(self: TransactionState) bool {
        return self == .active;
    }
};

/// Savepoint information
pub const Savepoint = struct {
    name: []const u8,
    id: u32,
    
    pub fn format(self: Savepoint, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("sp_{s}_{d}", .{ self.name, self.id });
    }
};

/// PostgreSQL transaction manager
pub const PgTransaction = struct {
    allocator: std.mem.Allocator,
    executor: *QueryExecutor,
    state: TransactionState,
    isolation_level: IsolationLevel,
    savepoints: std.ArrayList(Savepoint),
    savepoint_counter: u32,
    is_read_only: bool,
    
    pub fn init(
        allocator: std.mem.Allocator,
        executor: *QueryExecutor,
        isolation_level: IsolationLevel,
    ) PgTransaction {
        return PgTransaction{
            .allocator = allocator,
            .executor = executor,
            .state = .idle,
            .isolation_level = isolation_level,
            .savepoints = std.ArrayList(Savepoint).init(allocator),
            .savepoint_counter = 0,
            .is_read_only = false,
        };
    }
    
    pub fn deinit(self: *PgTransaction) void {
        // Free savepoint names
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp.name);
        }
        self.savepoints.deinit();
    }
    
    /// Begin transaction
    pub fn begin(self: *PgTransaction) !void {
        if (self.state.isActive()) {
            return error.TransactionAlreadyActive;
        }
        
        // Build BEGIN statement with isolation level
        var sql_buf: [256]u8 = undefined;
        const sql = try std.fmt.bufPrint(
            &sql_buf,
            "BEGIN TRANSACTION ISOLATION LEVEL {s}",
            .{self.isolation_level.toSQL()},
        );
        
        // Execute BEGIN
        var result = try self.executor.executeSimple(sql);
        defer result.deinit();
        
        self.state = .active;
    }
    
    /// Begin read-only transaction
    pub fn beginReadOnly(self: *PgTransaction) !void {
        if (self.state.isActive()) {
            return error.TransactionAlreadyActive;
        }
        
        // Build BEGIN statement with read-only mode
        var sql_buf: [256]u8 = undefined;
        const sql = try std.fmt.bufPrint(
            &sql_buf,
            "BEGIN TRANSACTION ISOLATION LEVEL {s} READ ONLY",
            .{self.isolation_level.toSQL()},
        );
        
        // Execute BEGIN
        var result = try self.executor.executeSimple(sql);
        defer result.deinit();
        
        self.state = .active;
        self.is_read_only = true;
    }
    
    /// Commit transaction
    pub fn commit(self: *PgTransaction) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        if (self.state == .failed) {
            return error.TransactionFailed;
        }
        
        // Execute COMMIT
        var result = try self.executor.executeSimple("COMMIT");
        defer result.deinit();
        
        self.state = .committed;
        
        // Clear savepoints
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp.name);
        }
        self.savepoints.clearRetainingCapacity();
        self.savepoint_counter = 0;
    }
    
    /// Rollback transaction
    pub fn rollback(self: *PgTransaction) !void {
        if (!self.state.isActive() and self.state != .failed) {
            return error.NoActiveTransaction;
        }
        
        // Execute ROLLBACK
        var result = try self.executor.executeSimple("ROLLBACK");
        defer result.deinit();
        
        self.state = .rolled_back;
        
        // Clear savepoints
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp.name);
        }
        self.savepoints.clearRetainingCapacity();
        self.savepoint_counter = 0;
    }
    
    /// Create savepoint
    pub fn savepoint(self: *PgTransaction, name: []const u8) !void {
        if (!self.state.canExecute()) {
            return error.TransactionNotActive;
        }
        
        // Generate unique savepoint identifier
        self.savepoint_counter += 1;
        const sp = Savepoint{
            .name = try self.allocator.dupe(u8, name),
            .id = self.savepoint_counter,
        };
        
        // Build SAVEPOINT statement
        var sql_buf: [256]u8 = undefined;
        const sql = try std.fmt.bufPrint(
            &sql_buf,
            "SAVEPOINT {s}",
            .{sp},
        );
        
        // Execute SAVEPOINT
        var result = try self.executor.executeSimple(sql);
        defer result.deinit();
        
        try self.savepoints.append(sp);
    }
    
    /// Rollback to savepoint
    pub fn rollbackTo(self: *PgTransaction, name: []const u8) !void {
        if (!self.state.canExecute() and self.state != .failed) {
            return error.TransactionNotActive;
        }
        
        // Find savepoint
        var found_sp: ?Savepoint = null;
        var sp_index: ?usize = null;
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp.name, name)) {
                found_sp = sp;
                sp_index = i;
                break;
            }
        }
        
        if (found_sp == null) {
            return error.SavepointNotFound;
        }
        
        // Build ROLLBACK TO SAVEPOINT statement
        var sql_buf: [256]u8 = undefined;
        const sql = try std.fmt.bufPrint(
            &sql_buf,
            "ROLLBACK TO SAVEPOINT {s}",
            .{found_sp.?},
        );
        
        // Execute ROLLBACK TO SAVEPOINT
        var result = try self.executor.executeSimple(sql);
        defer result.deinit();
        
        // Remove all savepoints after this one
        if (sp_index) |idx| {
            while (self.savepoints.items.len > idx + 1) {
                const removed = self.savepoints.pop();
                self.allocator.free(removed.name);
            }
        }
        
        // If transaction was failed, restore to active
        if (self.state == .failed) {
            self.state = .active;
        }
    }
    
    /// Release savepoint (remove it without rolling back)
    pub fn releaseSavepoint(self: *PgTransaction, name: []const u8) !void {
        if (!self.state.canExecute()) {
            return error.TransactionNotActive;
        }
        
        // Find savepoint
        var found_sp: ?Savepoint = null;
        var sp_index: ?usize = null;
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp.name, name)) {
                found_sp = sp;
                sp_index = i;
                break;
            }
        }
        
        if (found_sp == null) {
            return error.SavepointNotFound;
        }
        
        // Build RELEASE SAVEPOINT statement
        var sql_buf: [256]u8 = undefined;
        const sql = try std.fmt.bufPrint(
            &sql_buf,
            "RELEASE SAVEPOINT {s}",
            .{found_sp.?},
        );
        
        // Execute RELEASE SAVEPOINT
        var result = try self.executor.executeSimple(sql);
        defer result.deinit();
        
        // Remove this savepoint and all after it
        if (sp_index) |idx| {
            while (self.savepoints.items.len > idx) {
                const removed = self.savepoints.pop();
                self.allocator.free(removed.name);
            }
        }
    }
    
    /// Execute SQL within transaction
    pub fn execute(self: *PgTransaction, sql: []const u8, params: []const Value) !ResultSet {
        if (!self.state.canExecute()) {
            return error.TransactionNotActive;
        }
        
        // Execute query
        const result = if (params.len > 0)
            try self.executor.executeExtended(sql, params)
        else
            try self.executor.executeSimple(sql);
        
        return result;
    }
    
    /// Check if transaction is active
    pub fn isActive(self: PgTransaction) bool {
        return self.state.isActive();
    }
    
    /// Check if transaction is read-only
    pub fn isReadOnly(self: PgTransaction) bool {
        return self.is_read_only;
    }
    
    /// Get current transaction state
    pub fn getState(self: PgTransaction) TransactionState {
        return self.state;
    }
    
    /// Get isolation level
    pub fn getIsolationLevel(self: PgTransaction) IsolationLevel {
        return self.isolation_level;
    }
    
    /// Get number of active savepoints
    pub fn savepointCount(self: PgTransaction) usize {
        return self.savepoints.items.len;
    }
    
    /// Mark transaction as failed (for error handling)
    pub fn markFailed(self: *PgTransaction) void {
        if (self.state.isActive()) {
            self.state = .failed;
        }
    }
};

/// Transaction manager for creating and managing transactions
pub const TransactionManager = struct {
    allocator: std.mem.Allocator,
    executor: *QueryExecutor,
    current_transaction: ?*PgTransaction,
    
    pub fn init(allocator: std.mem.Allocator, executor: *QueryExecutor) TransactionManager {
        return TransactionManager{
            .allocator = allocator,
            .executor = executor,
            .current_transaction = null,
        };
    }
    
    pub fn deinit(self: *TransactionManager) void {
        if (self.current_transaction) |txn| {
            txn.deinit();
            self.allocator.destroy(txn);
            self.current_transaction = null;
        }
    }
    
    /// Begin a new transaction
    pub fn beginTransaction(
        self: *TransactionManager,
        isolation_level: IsolationLevel,
    ) !*PgTransaction {
        if (self.current_transaction != null) {
            return error.TransactionAlreadyActive;
        }
        
        var txn = try self.allocator.create(PgTransaction);
        txn.* = PgTransaction.init(self.allocator, self.executor, isolation_level);
        
        try txn.begin();
        
        self.current_transaction = txn;
        return txn;
    }
    
    /// Begin a read-only transaction
    pub fn beginReadOnlyTransaction(
        self: *TransactionManager,
        isolation_level: IsolationLevel,
    ) !*PgTransaction {
        if (self.current_transaction != null) {
            return error.TransactionAlreadyActive;
        }
        
        var txn = try self.allocator.create(PgTransaction);
        txn.* = PgTransaction.init(self.allocator, self.executor, isolation_level);
        
        try txn.beginReadOnly();
        
        self.current_transaction = txn;
        return txn;
    }
    
    /// End current transaction (cleanup)
    pub fn endTransaction(self: *TransactionManager) void {
        if (self.current_transaction) |txn| {
            txn.deinit();
            self.allocator.destroy(txn);
            self.current_transaction = null;
        }
    }
    
    /// Check if there's an active transaction
    pub fn hasActiveTransaction(self: TransactionManager) bool {
        if (self.current_transaction) |txn| {
            return txn.isActive();
        }
        return false;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TransactionState - isActive" {
    try std.testing.expect(!TransactionState.idle.isActive());
    try std.testing.expect(TransactionState.active.isActive());
    try std.testing.expect(!TransactionState.failed.isActive());
    try std.testing.expect(!TransactionState.committed.isActive());
    try std.testing.expect(!TransactionState.rolled_back.isActive());
}

test "TransactionState - canExecute" {
    try std.testing.expect(!TransactionState.idle.canExecute());
    try std.testing.expect(TransactionState.active.canExecute());
    try std.testing.expect(!TransactionState.failed.canExecute());
    try std.testing.expect(!TransactionState.committed.canExecute());
    try std.testing.expect(!TransactionState.rolled_back.canExecute());
}

test "Savepoint - format" {
    const sp = Savepoint{
        .name = "test",
        .id = 42,
    };
    
    var buf: [64]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try fbs.writer().print("{s}", .{sp});
    try std.testing.expectEqualStrings("sp_test_42", fbs.getWritten());
}

test "PgTransaction - init and deinit" {
    const allocator = std.testing.allocator;
    
    // Create mock executor (we won't actually use it in this test)
    var executor: QueryExecutor = undefined;
    
    var txn = PgTransaction.init(allocator, &executor, .read_committed);
    defer txn.deinit();
    
    try std.testing.expectEqual(TransactionState.idle, txn.state);
    try std.testing.expectEqual(IsolationLevel.read_committed, txn.isolation_level);
    try std.testing.expectEqual(@as(usize, 0), txn.savepoints.items.len);
    try std.testing.expect(!txn.is_read_only);
}

test "PgTransaction - state tracking" {
    const allocator = std.testing.allocator;
    var executor: QueryExecutor = undefined;
    
    var txn = PgTransaction.init(allocator, &executor, .serializable);
    defer txn.deinit();
    
    try std.testing.expect(!txn.isActive());
    try std.testing.expect(!txn.isReadOnly());
    try std.testing.expectEqual(IsolationLevel.serializable, txn.getIsolationLevel());
}

test "PgTransaction - markFailed" {
    const allocator = std.testing.allocator;
    var executor: QueryExecutor = undefined;
    
    var txn = PgTransaction.init(allocator, &executor, .read_committed);
    defer txn.deinit();
    
    // Start transaction (manually set state since we can't actually execute)
    txn.state = .active;
    try std.testing.expect(txn.isActive());
    
    // Mark as failed
    txn.markFailed();
    try std.testing.expectEqual(TransactionState.failed, txn.state);
    try std.testing.expect(!txn.isActive());
}

test "TransactionManager - init and deinit" {
    const allocator = std.testing.allocator;
    var executor: QueryExecutor = undefined;
    
    var mgr = TransactionManager.init(allocator, &executor);
    defer mgr.deinit();
    
    try std.testing.expect(!mgr.hasActiveTransaction());
    try std.testing.expect(mgr.current_transaction == null);
}

test "IsolationLevel - toSQL" {
    try std.testing.expectEqualStrings("READ UNCOMMITTED", IsolationLevel.read_uncommitted.toSQL());
    try std.testing.expectEqualStrings("READ COMMITTED", IsolationLevel.read_committed.toSQL());
    try std.testing.expectEqualStrings("REPEATABLE READ", IsolationLevel.repeatable_read.toSQL());
    try std.testing.expectEqualStrings("SERIALIZABLE", IsolationLevel.serializable.toSQL());
}
