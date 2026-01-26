const std = @import("std");
const protocol = @import("protocol.zig");
const connection_mod = @import("connection.zig");

/// Transaction isolation level
pub const IsolationLevel = enum {
    read_uncommitted,
    read_committed,
    repeatable_read,
    serializable,
    
    pub fn toString(self: IsolationLevel) []const u8 {
        return switch (self) {
            .read_uncommitted => "READ UNCOMMITTED",
            .read_committed => "READ COMMITTED",
            .repeatable_read => "REPEATABLE READ",
            .serializable => "SERIALIZABLE",
        };
    }
};

/// Transaction state
pub const TransactionState = enum {
    inactive,
    active,
    committed,
    rolled_back,
    failed,
    
    pub fn isActive(self: TransactionState) bool {
        return self == .active;
    }
};

/// Transaction manager
pub const TransactionManager = struct {
    allocator: std.mem.Allocator,
    connection: *connection_mod.HanaConnection,
    state: TransactionState,
    isolation_level: IsolationLevel,
    auto_commit: bool,
    savepoints: std.ArrayList([]const u8),
    
    pub fn init(
        allocator: std.mem.Allocator,
        connection: *connection_mod.HanaConnection,
    ) TransactionManager {
        return TransactionManager{
            .allocator = allocator,
            .connection = connection,
            .state = .inactive,
            .isolation_level = .read_committed,
            .auto_commit = true,
            .savepoints = std.ArrayList([]const u8){},
        };
    }
    
    pub fn deinit(self: *TransactionManager) void {
        for (self.savepoints.items) |savepoint| {
            self.allocator.free(savepoint);
        }
        self.savepoints.deinit();
    }
    
    /// Begin a new transaction
    pub fn begin(self: *TransactionManager) !void {
        if (self.state.isActive()) {
            return error.TransactionAlreadyActive;
        }
        
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Send COMMIT message with transaction start flag
        var segment = protocol.SegmentHeader.init(.commit, 0);
        
        // In real implementation:
        // 1. Set transaction_flags
        // 2. Send segment
        // 3. Receive acknowledgment
        
        _ = segment;
        
        self.state = .active;
        self.auto_commit = false;
    }
    
    /// Commit the current transaction
    pub fn commit(self: *TransactionManager) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Send COMMIT message
        var segment = protocol.SegmentHeader.init(.commit, 0);
        
        // In real implementation:
        // 1. Send commit segment
        // 2. Receive acknowledgment
        // 3. Clear savepoints
        
        _ = segment;
        
        self.state = .committed;
        
        // Clear savepoints
        for (self.savepoints.items) |savepoint| {
            self.allocator.free(savepoint);
        }
        self.savepoints.clearRetainingCapacity();
        
        self.state = .inactive;
        self.auto_commit = true;
    }
    
    /// Rollback the current transaction
    pub fn rollback(self: *TransactionManager) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        if (!self.connection.isConnected()) {
            return error.NotConnected;
        }
        
        // Send ROLLBACK message
        var segment = protocol.SegmentHeader.init(.rollback, 0);
        
        // In real implementation:
        // 1. Send rollback segment
        // 2. Receive acknowledgment
        // 3. Clear savepoints
        
        _ = segment;
        
        self.state = .rolled_back;
        
        // Clear savepoints
        for (self.savepoints.items) |savepoint| {
            self.allocator.free(savepoint);
        }
        self.savepoints.clearRetainingCapacity();
        
        self.state = .inactive;
        self.auto_commit = true;
    }
    
    /// Create a savepoint
    pub fn savepoint(self: *TransactionManager, name: []const u8) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        // Store savepoint name
        const name_copy = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(name_copy);
        
        try self.savepoints.append(name_copy);
        
        // In real implementation:
        // 1. Send SQL: SAVEPOINT <name>
        // 2. Receive acknowledgment
    }
    
    /// Rollback to a savepoint
    pub fn rollbackToSavepoint(self: *TransactionManager, name: []const u8) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        // Find savepoint
        var found = false;
        var index: usize = 0;
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp, name)) {
                found = true;
                index = i;
                break;
            }
        }
        
        if (!found) {
            return error.SavepointNotFound;
        }
        
        // In real implementation:
        // 1. Send SQL: ROLLBACK TO SAVEPOINT <name>
        // 2. Receive acknowledgment
        
        // Remove savepoints after this one
        while (self.savepoints.items.len > index + 1) {
            const sp = self.savepoints.pop();
            self.allocator.free(sp);
        }
    }
    
    /// Release a savepoint
    pub fn releaseSavepoint(self: *TransactionManager, name: []const u8) !void {
        if (!self.state.isActive()) {
            return error.NoActiveTransaction;
        }
        
        // Find and remove savepoint
        var found = false;
        var index: usize = 0;
        for (self.savepoints.items, 0..) |sp, i| {
            if (std.mem.eql(u8, sp, name)) {
                found = true;
                index = i;
                break;
            }
        }
        
        if (!found) {
            return error.SavepointNotFound;
        }
        
        // In real implementation:
        // 1. Send SQL: RELEASE SAVEPOINT <name>
        // 2. Receive acknowledgment
        
        const sp = self.savepoints.orderedRemove(index);
        self.allocator.free(sp);
    }
    
    /// Set isolation level
    pub fn setIsolationLevel(self: *TransactionManager, level: IsolationLevel) !void {
        if (self.state.isActive()) {
            return error.CannotChangeIsolationInTransaction;
        }
        
        self.isolation_level = level;
        
        // In real implementation:
        // 1. Send SQL: SET TRANSACTION ISOLATION LEVEL <level>
        // 2. Receive acknowledgment
    }
    
    /// Set auto-commit mode
    pub fn setAutoCommit(self: *TransactionManager, enabled: bool) !void {
        self.auto_commit = enabled;
        
        // In real implementation:
        // 1. Send connection property update
        // 2. Receive acknowledgment
    }
    
    /// Get transaction state
    pub fn getState(self: TransactionManager) TransactionState {
        return self.state;
    }
    
    /// Check if transaction is active
    pub fn isActive(self: TransactionManager) bool {
        return self.state.isActive();
    }
    
    /// Get isolation level
    pub fn getIsolationLevel(self: TransactionManager) IsolationLevel {
        return self.isolation_level;
    }
    
    /// Check if auto-commit is enabled
    pub fn isAutoCommit(self: TransactionManager) bool {
        return self.auto_commit;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "IsolationLevel - toString" {
    try std.testing.expectEqualStrings("READ UNCOMMITTED", IsolationLevel.read_uncommitted.toString());
    try std.testing.expectEqualStrings("READ COMMITTED", IsolationLevel.read_committed.toString());
    try std.testing.expectEqualStrings("REPEATABLE READ", IsolationLevel.repeatable_read.toString());
    try std.testing.expectEqualStrings("SERIALIZABLE", IsolationLevel.serializable.toString());
}

test "TransactionState - isActive" {
    try std.testing.expect(!TransactionState.inactive.isActive());
    try std.testing.expect(TransactionState.active.isActive());
    try std.testing.expect(!TransactionState.committed.isActive());
    try std.testing.expect(!TransactionState.rolled_back.isActive());
}

test "TransactionManager - init and deinit" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = TransactionManager.init(allocator, &conn);
    defer tx.deinit();
    
    try std.testing.expectEqual(TransactionState.inactive, tx.state);
    try std.testing.expectEqual(IsolationLevel.read_committed, tx.isolation_level);
    try std.testing.expect(tx.auto_commit);
}

test "TransactionManager - isolation level" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = TransactionManager.init(allocator, &conn);
    defer tx.deinit();
    
    try tx.setIsolationLevel(.serializable);
    try std.testing.expectEqual(IsolationLevel.serializable, tx.getIsolationLevel());
}

test "TransactionManager - auto commit" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = TransactionManager.init(allocator, &conn);
    defer tx.deinit();
    
    try std.testing.expect(tx.isAutoCommit());
    
    try tx.setAutoCommit(false);
    try std.testing.expect(!tx.isAutoCommit());
}

test "TransactionManager - state tracking" {
    const allocator = std.testing.allocator;
    
    const config = connection_mod.HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "password",
    };
    
    var conn = try connection_mod.HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    var tx = TransactionManager.init(allocator, &conn);
    defer tx.deinit();
    
    try std.testing.expectEqual(TransactionState.inactive, tx.getState());
    try std.testing.expect(!tx.isActive());
}
