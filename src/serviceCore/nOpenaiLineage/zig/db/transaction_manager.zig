const std = @import("std");
const client = @import("client.zig");
const DbClient = client.DbClient;
const Transaction = client.Transaction;
const IsolationLevel = client.IsolationLevel;
const Value = client.Value;
const ResultSet = client.ResultSet;

/// Transaction state
pub const TransactionState = enum {
    active, // Transaction is active
    committed, // Transaction has been committed
    rolled_back, // Transaction has been rolled back
    failed, // Transaction failed with error

    pub fn isTerminal(self: TransactionState) bool {
        return self == .committed or self == .rolled_back or self == .failed;
    }
};

/// Savepoint information
pub const Savepoint = struct {
    name: []const u8,
    created_at: i64, // Unix timestamp in milliseconds
};

/// Transaction context with automatic rollback
pub const TransactionContext = struct {
    transaction: *Transaction,
    state: TransactionState,
    isolation_level: IsolationLevel,
    savepoints: std.ArrayList(Savepoint),
    started_at: i64,
    allocator: std.mem.Allocator,
    auto_rollback: bool, // Rollback on error or scope exit if not committed

    pub fn init(
        allocator: std.mem.Allocator,
        transaction: *Transaction,
        isolation_level: IsolationLevel,
    ) TransactionContext {
        return TransactionContext{
            .transaction = transaction,
            .state = .active,
            .isolation_level = isolation_level,
            .savepoints = std.ArrayList(Savepoint).init(allocator),
            .started_at = std.time.milliTimestamp(),
            .allocator = allocator,
            .auto_rollback = true,
        };
    }

    pub fn deinit(self: *TransactionContext) void {
        // Automatic rollback if transaction not explicitly committed
        if (self.auto_rollback and self.state == .active) {
            self.rollback() catch |err| {
                std.log.err("Failed to auto-rollback transaction: {}", .{err});
            };
        }

        // Clean up savepoint names
        for (self.savepoints.items) |sp| {
            self.allocator.free(sp.name);
        }
        self.savepoints.deinit();
    }

    /// Commit the transaction
    pub fn commit(self: *TransactionContext) !void {
        if (self.state.isTerminal()) {
            return error.TransactionAlreadyTerminated;
        }

        try self.transaction.commit();
        self.state = .committed;
        self.auto_rollback = false; // Don't rollback after successful commit
    }

    /// Rollback the transaction
    pub fn rollback(self: *TransactionContext) !void {
        if (self.state.isTerminal()) {
            return error.TransactionAlreadyTerminated;
        }

        try self.transaction.rollback();
        self.state = .rolled_back;
        self.auto_rollback = false; // Already rolled back
    }

    /// Create a savepoint
    pub fn savepoint(self: *TransactionContext, name: []const u8) !void {
        if (self.state != .active) {
            return error.TransactionNotActive;
        }

        try self.transaction.savepoint(name);

        const sp = Savepoint{
            .name = try self.allocator.dupe(u8, name),
            .created_at = std.time.milliTimestamp(),
        };
        try self.savepoints.append(sp);
    }

    /// Rollback to a savepoint
    pub fn rollbackTo(self: *TransactionContext, name: []const u8) !void {
        if (self.state != .active) {
            return error.TransactionNotActive;
        }

        // Find savepoint
        var found = false;
        for (self.savepoints.items) |sp| {
            if (std.mem.eql(u8, sp.name, name)) {
                found = true;
                break;
            }
        }

        if (!found) {
            return error.SavepointNotFound;
        }

        try self.transaction.rollback_to(name);
    }

    /// Execute SQL within the transaction
    pub fn execute(self: *TransactionContext, sql: []const u8, params: []const Value) !ResultSet {
        if (self.state != .active) {
            return error.TransactionNotActive;
        }

        return self.transaction.execute(sql, params) catch |err| {
            self.state = .failed;
            return err;
        };
    }

    /// Get transaction duration in milliseconds
    pub fn getDuration(self: TransactionContext) i64 {
        return std.time.milliTimestamp() - self.started_at;
    }

    /// Check if transaction is active
    pub fn isActive(self: TransactionContext) bool {
        return self.state == .active;
    }
};

/// Transaction manager for managing database transactions
pub const TransactionManager = struct {
    allocator: std.mem.Allocator,
    client: *DbClient,
    active_transactions: std.ArrayList(*TransactionContext),
    mutex: std.Thread.Mutex,
    
    // Metrics
    total_transactions: u64,
    committed_transactions: u64,
    rolled_back_transactions: u64,
    failed_transactions: u64,
    total_transaction_time_ms: u64,

    pub fn init(allocator: std.mem.Allocator, db_client: *DbClient) TransactionManager {
        return TransactionManager{
            .allocator = allocator,
            .client = db_client,
            .active_transactions = std.ArrayList(*TransactionContext).init(allocator),
            .mutex = std.Thread.Mutex{},
            .total_transactions = 0,
            .committed_transactions = 0,
            .rolled_back_transactions = 0,
            .failed_transactions = 0,
            .total_transaction_time_ms = 0,
        };
    }

    pub fn deinit(self: *TransactionManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Rollback any active transactions
        for (self.active_transactions.items) |tx_ctx| {
            tx_ctx.rollback() catch |err| {
                std.log.err("Failed to rollback transaction on shutdown: {}", .{err});
            };
            tx_ctx.deinit();
            self.allocator.destroy(tx_ctx);
        }
        self.active_transactions.deinit();
    }

    /// Begin a new transaction with default isolation level
    pub fn begin(self: *TransactionManager) !*TransactionContext {
        return self.beginWithIsolation(.read_committed);
    }

    /// Begin a new transaction with specific isolation level
    pub fn beginWithIsolation(self: *TransactionManager, level: IsolationLevel) !*TransactionContext {
        self.mutex.lock();
        defer self.mutex.unlock();

        self.total_transactions += 1;

        const transaction = try self.client.beginWithIsolation(level);
        const tx_ctx = try self.allocator.create(TransactionContext);
        tx_ctx.* = TransactionContext.init(self.allocator, transaction, level);

        try self.active_transactions.append(tx_ctx);

        return tx_ctx;
    }

    /// End a transaction (commit or rollback already called)
    pub fn end(self: *TransactionManager, tx_ctx: *TransactionContext) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Update metrics
        const duration = tx_ctx.getDuration();
        self.total_transaction_time_ms += @intCast(duration);

        switch (tx_ctx.state) {
            .committed => self.committed_transactions += 1,
            .rolled_back => self.rolled_back_transactions += 1,
            .failed => self.failed_transactions += 1,
            .active => {
                // Transaction still active, will be rolled back by deinit
                self.rolled_back_transactions += 1;
            },
        }

        // Remove from active list
        for (self.active_transactions.items, 0..) |active_tx, i| {
            if (active_tx == tx_ctx) {
                _ = self.active_transactions.orderedRemove(i);
                break;
            }
        }

        tx_ctx.deinit();
        self.allocator.destroy(tx_ctx);
    }

    /// Get transaction manager metrics
    pub fn getMetrics(self: *TransactionManager) TransactionMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();

        const avg_duration = if (self.total_transactions > 0)
            @as(f64, @floatFromInt(self.total_transaction_time_ms)) / @as(f64, @floatFromInt(self.total_transactions))
        else
            0.0;

        return TransactionMetrics{
            .active_transactions = self.active_transactions.items.len,
            .total_transactions = self.total_transactions,
            .committed = self.committed_transactions,
            .rolled_back = self.rolled_back_transactions,
            .failed = self.failed_transactions,
            .avg_duration_ms = avg_duration,
        };
    }
};

/// Transaction metrics
pub const TransactionMetrics = struct {
    active_transactions: usize,
    total_transactions: u64,
    committed: u64,
    rolled_back: u64,
    failed: u64,
    avg_duration_ms: f64,

    pub fn format(self: TransactionMetrics, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "TransactionMetrics{{ active={d}, total={d}, committed={d}, rolled_back={d}, failed={d}, avg_duration={d:.2}ms }}",
            .{
                self.active_transactions,
                self.total_transactions,
                self.committed,
                self.rolled_back,
                self.failed,
                self.avg_duration_ms,
            },
        );
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "TransactionState - terminal states" {
    try std.testing.expect(!TransactionState.active.isTerminal());
    try std.testing.expect(TransactionState.committed.isTerminal());
    try std.testing.expect(TransactionState.rolled_back.isTerminal());
    try std.testing.expect(TransactionState.failed.isTerminal());
}

test "TransactionContext - basic lifecycle" {
    const allocator = std.testing.allocator;

    var mock_transaction = Transaction{
        .vtable = undefined,
        .context = undefined,
    };

    var tx_ctx = TransactionContext.init(allocator, &mock_transaction, .read_committed);
    defer tx_ctx.deinit();

    try std.testing.expectEqual(TransactionState.active, tx_ctx.state);
    try std.testing.expect(tx_ctx.isActive());
    try std.testing.expectEqual(IsolationLevel.read_committed, tx_ctx.isolation_level);
}

test "TransactionContext - duration tracking" {
    const allocator = std.testing.allocator;

    var mock_transaction = Transaction{
        .vtable = undefined,
        .context = undefined,
    };

    var tx_ctx = TransactionContext.init(allocator, &mock_transaction, .read_committed);
    defer tx_ctx.deinit();

    std.time.sleep(10 * std.time.ns_per_ms); // Sleep 10ms

    const duration = tx_ctx.getDuration();
    try std.testing.expect(duration >= 10);
}

test "TransactionManager - init and deinit" {
    const allocator = std.testing.allocator;

    var mock_client = DbClient{
        .vtable = undefined,
        .context = undefined,
        .allocator = allocator,
    };

    var manager = TransactionManager.init(allocator, &mock_client);
    defer manager.deinit();

    try std.testing.expectEqual(@as(u64, 0), manager.total_transactions);
    try std.testing.expectEqual(@as(usize, 0), manager.active_transactions.items.len);
}

test "TransactionManager - metrics" {
    const allocator = std.testing.allocator;

    var mock_client = DbClient{
        .vtable = undefined,
        .context = undefined,
        .allocator = allocator,
    };

    var manager = TransactionManager.init(allocator, &mock_client);
    defer manager.deinit();

    const metrics = manager.getMetrics();
    try std.testing.expectEqual(@as(usize, 0), metrics.active_transactions);
    try std.testing.expectEqual(@as(u64, 0), metrics.total_transactions);
    try std.testing.expectEqual(@as(f64, 0.0), metrics.avg_duration_ms);
}

test "TransactionMetrics - format" {
    const metrics = TransactionMetrics{
        .active_transactions = 3,
        .total_transactions = 100,
        .committed = 95,
        .rolled_back = 4,
        .failed = 1,
        .avg_duration_ms = 15.5,
    };

    var buf: [200]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try fbs.writer().print("{}", .{metrics});

    const result = fbs.getWritten();
    try std.testing.expect(std.mem.indexOf(u8, result, "active=3") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "total=100") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "committed=95") != null);
}

test "Savepoint - structure" {
    const allocator = std.testing.allocator;

    const name = try allocator.dupe(u8, "sp1");
    defer allocator.free(name);

    const sp = Savepoint{
        .name = name,
        .created_at = std.time.milliTimestamp(),
    };

    try std.testing.expectEqualStrings("sp1", sp.name);
    try std.testing.expect(sp.created_at > 0);
}
