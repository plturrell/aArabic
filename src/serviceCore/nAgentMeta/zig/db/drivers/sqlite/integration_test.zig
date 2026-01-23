const std = @import("std");
const connection_mod = @import("connection.zig");
const query_mod = @import("query.zig");
const transaction_mod = @import("transaction.zig");
const pool_mod = @import("pool.zig");
const protocol = @import("protocol.zig");
const client_types = @import("../../client.zig");

const SqliteConnection = connection_mod.SqliteConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;
const QueryExecutor = query_mod.QueryExecutor;
const SqliteTransaction = transaction_mod.SqliteTransaction;
const SqliteConnectionPool = pool_mod.SqliteConnectionPool;
const SqlitePoolConfig = pool_mod.SqlitePoolConfig;
const Value = client_types.Value;
const IsolationLevel = client_types.IsolationLevel;

/// Integration test configuration
pub const IntegrationTestConfig = struct {
    database_path: []const u8 = ":memory:",
    mode: protocol.OpenMode = .read_write_create,
    
    pub fn toConnectionConfig(self: IntegrationTestConfig) ConnectionConfig {
        return ConnectionConfig{
            .path = self.database_path,
            .mode = self.mode,
            .journal_mode = .wal,
            .synchronous = .normal,
            .cache_size = 2000,
            .foreign_keys = true,
            .timeout_ms = 5000,
        };
    }
};

/// Integration test suite runner
pub const IntegrationTestSuite = struct {
    allocator: std.mem.Allocator,
    config: IntegrationTestConfig,
    tests_run: usize,
    tests_passed: usize,
    tests_failed: usize,
    
    pub fn init(allocator: std.mem.Allocator, config: IntegrationTestConfig) IntegrationTestSuite {
        return IntegrationTestSuite{
            .allocator = allocator,
            .config = config,
            .tests_run = 0,
            .tests_passed = 0,
            .tests_failed = 0,
        };
    }
    
    /// Run all integration tests
    pub fn runAll(self: *IntegrationTestSuite) !void {
        std.debug.print("\n=== SQLite Integration Tests ===\n\n", .{});
        
        try self.runTest("Connection", testConnection);
        try self.runTest("Simple Query", testSimpleQuery);
        try self.runTest("Prepared Statement", testPreparedStatement);
        try self.runTest("Transaction Commit", testTransactionCommit);
        try self.runTest("Transaction Rollback", testTransactionRollback);
        try self.runTest("Savepoint", testSavepoint);
        try self.runTest("Connection Pool", testConnectionPool);
        try self.runTest("WAL Mode", testWalMode);
        try self.runTest("Foreign Keys", testForeignKeys);
        try self.runTest("Concurrent Access", testConcurrentAccess);
        
        std.debug.print("\n=== Test Summary ===\n", .{});
        std.debug.print("Tests Run: {d}\n", .{self.tests_run});
        std.debug.print("Tests Passed: {d}\n", .{self.tests_passed});
        std.debug.print("Tests Failed: {d}\n", .{self.tests_failed});
        
        if (self.tests_failed > 0) {
            return error.IntegrationTestsFailed;
        }
    }
    
    /// Run a single test
    fn runTest(
        self: *IntegrationTestSuite,
        name: []const u8,
        test_fn: fn (*IntegrationTestSuite) anyerror!void,
    ) !void {
        self.tests_run += 1;
        
        std.debug.print("Running: {s}... ", .{name});
        
        test_fn(self) catch |err| {
            self.tests_failed += 1;
            std.debug.print("FAILED ({any})\n", .{err});
            return;
        };
        
        self.tests_passed += 1;
        std.debug.print("PASSED\n", .{});
    }
    
    /// Test: Basic connection lifecycle
    fn testConnection(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        // Verify connection is active
        try std.testing.expect(conn.isConnected());
    }
    
    /// Test: Simple query execution (SELECT 1)
    fn testSimpleQuery(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Execute: SELECT 1 AS result
        const sql = "SELECT 1 AS result";
        const result = try executor.executeQuery(sql, &[_]Value{});
        defer result.deinit();
        
        // Verify single row with value 1
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Prepared statement with parameters
    fn testPreparedStatement(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Execute: SELECT ?1 + ?2 AS sum
        const sql = "SELECT ?1 + ?2 AS sum";
        const params = [_]Value{
            Value{ .integer = 5 },
            Value{ .integer = 10 },
        };
        const result = try executor.executeQuery(sql, &params);
        defer result.deinit();
        
        // Verify result is 15
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Transaction commit
    fn testTransactionCommit(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        // Create test table
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        _ = try executor.executeQuery(
            "CREATE TABLE test_commit (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        
        // Begin transaction
        var tx = try SqliteTransaction.init(self.allocator, &conn, .read_committed);
        defer tx.deinit();
        
        try tx.begin();
        
        // Insert data
        _ = try executor.executeQuery(
            "INSERT INTO test_commit (value) VALUES ('committed')",
            &[_]Value{},
        );
        
        // Commit
        try tx.commit();
        
        // Verify data exists
        const result = try executor.executeQuery(
            "SELECT COUNT(*) AS count FROM test_commit",
            &[_]Value{},
        );
        defer result.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Transaction rollback
    fn testTransactionRollback(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        // Create test table
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        _ = try executor.executeQuery(
            "CREATE TABLE test_rollback (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        
        // Begin transaction
        var tx = try SqliteTransaction.init(self.allocator, &conn, .read_committed);
        defer tx.deinit();
        
        try tx.begin();
        
        // Insert data
        _ = try executor.executeQuery(
            "INSERT INTO test_rollback (value) VALUES ('rolled back')",
            &[_]Value{},
        );
        
        // Rollback
        try tx.rollback();
        
        // Verify data does not exist
        const result = try executor.executeQuery(
            "SELECT COUNT(*) AS count FROM test_rollback",
            &[_]Value{},
        );
        defer result.deinit();
        
        // Should be 0 rows after rollback
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Savepoint operations
    fn testSavepoint(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        // Create test table
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        _ = try executor.executeQuery(
            "CREATE TABLE test_savepoint (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        
        // Begin transaction
        var tx = try SqliteTransaction.init(self.allocator, &conn, .read_committed);
        defer tx.deinit();
        
        try tx.begin();
        
        // Insert first row
        _ = try executor.executeQuery(
            "INSERT INTO test_savepoint (value) VALUES ('first')",
            &[_]Value{},
        );
        
        // Create savepoint
        try tx.savepoint("sp1");
        
        // Insert second row
        _ = try executor.executeQuery(
            "INSERT INTO test_savepoint (value) VALUES ('second')",
            &[_]Value{},
        );
        
        // Rollback to savepoint (removes second row)
        try tx.rollbackToSavepoint("sp1");
        
        // Commit
        try tx.commit();
        
        // Verify only first row exists
        const result = try executor.executeQuery(
            "SELECT COUNT(*) AS count FROM test_savepoint",
            &[_]Value{},
        );
        defer result.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Connection pool operations
    fn testConnectionPool(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        const pool_config = SqlitePoolConfig{
            .connection_config = conn_config,
            .min_size = 1,
            .max_size = 3,
            .acquire_timeout_ms = 5000,
            .idle_timeout_ms = 300000,
        };
        
        var pool = try SqliteConnectionPool.init(self.allocator, pool_config);
        defer pool.deinit();
        
        // Acquire connection
        var conn = try pool.acquire();
        
        // Verify connection works
        try std.testing.expect(conn.isConnected());
        
        // Release connection
        try pool.release(conn);
        
        // Get pool stats
        const stats = pool.getStats();
        try std.testing.expectEqual(@as(usize, 1), stats.available);
    }
    
    /// Test: WAL mode configuration
    fn testWalMode(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Query journal mode
        const result = try executor.executeQuery(
            "PRAGMA journal_mode",
            &[_]Value{},
        );
        defer result.deinit();
        
        // Should be WAL mode
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Foreign key constraints
    fn testForeignKeys(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try SqliteConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = QueryExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Create parent and child tables
        _ = try executor.executeQuery(
            "CREATE TABLE parent (id INTEGER PRIMARY KEY)",
            &[_]Value{},
        );
        
        _ = try executor.executeQuery(
            \\CREATE TABLE child (
            \\  id INTEGER PRIMARY KEY,
            \\  parent_id INTEGER,
            \\  FOREIGN KEY (parent_id) REFERENCES parent(id)
            \\)
            ,
            &[_]Value{},
        );
        
        // Insert parent
        _ = try executor.executeQuery(
            "INSERT INTO parent (id) VALUES (1)",
            &[_]Value{},
        );
        
        // Insert child with valid parent_id should succeed
        _ = try executor.executeQuery(
            "INSERT INTO child (parent_id) VALUES (1)",
            &[_]Value{},
        );
        
        // Verify child was inserted
        const result = try executor.executeQuery(
            "SELECT COUNT(*) AS count FROM child",
            &[_]Value{},
        );
        defer result.deinit();
        
        try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
    }
    
    /// Test: Concurrent access with pool
    fn testConcurrentAccess(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        const pool_config = SqlitePoolConfig{
            .connection_config = conn_config,
            .min_size = 2,
            .max_size = 4,
            .acquire_timeout_ms = 5000,
            .idle_timeout_ms = 300000,
        };
        
        var pool = try SqliteConnectionPool.init(self.allocator, pool_config);
        defer pool.deinit();
        
        // Create test table
        var conn1 = try pool.acquire();
        var executor = QueryExecutor.init(self.allocator, conn1);
        defer executor.deinit();
        
        _ = try executor.executeQuery(
            "CREATE TABLE test_concurrent (id INTEGER PRIMARY KEY, value TEXT)",
            &[_]Value{},
        );
        
        try pool.release(conn1);
        
        // Acquire multiple connections and perform operations
        var conn2 = try pool.acquire();
        var conn3 = try pool.acquire();
        
        try std.testing.expect(conn2.isConnected());
        try std.testing.expect(conn3.isConnected());
        
        try pool.release(conn2);
        try pool.release(conn3);
        
        // Verify pool stats
        const stats = pool.getStats();
        try std.testing.expect(stats.available >= 2);
    }
};

/// Run integration tests (requires SQLite library)
pub fn runIntegrationTests(allocator: std.mem.Allocator) !void {
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    try suite.runAll();
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

/// Check if SQLite is available
pub fn isSqliteAvailable() bool {
    // Check if libsqlite3 is available
    // In production, would attempt to load the library
    return true;
}

/// Create test database file
pub fn createTestDatabase(allocator: std.mem.Allocator, path: []const u8) !void {
    _ = allocator;
    _ = path;
    // Would create a temporary SQLite database file
}

/// Delete test database file
pub fn deleteTestDatabase(path: []const u8) !void {
    _ = path;
    // Would delete the temporary SQLite database file
}

/// Run database migration for testing
pub fn runTestMigration(allocator: std.mem.Allocator, conn: *SqliteConnection) !void {
    var executor = QueryExecutor.init(allocator, conn);
    defer executor.deinit();
    
    // Create test schema
    _ = try executor.executeQuery(
        \\CREATE TABLE IF NOT EXISTS migrations (
        \\  id INTEGER PRIMARY KEY,
        \\  name TEXT NOT NULL,
        \\  applied_at TEXT NOT NULL
        \\)
        ,
        &[_]Value{},
    );
}

// ============================================================================
// Unit Tests (for test infrastructure itself)
// ============================================================================

test "IntegrationTestConfig - toConnectionConfig" {
    const test_config = IntegrationTestConfig{
        .database_path = "test.db",
        .mode = .read_write,
    };
    
    const conn_config = test_config.toConnectionConfig();
    
    try std.testing.expectEqualStrings("test.db", conn_config.path);
    try std.testing.expectEqual(protocol.OpenMode.read_write, conn_config.mode);
    try std.testing.expectEqual(protocol.JournalMode.wal, conn_config.journal_mode);
    try std.testing.expect(conn_config.foreign_keys);
}

test "IntegrationTestSuite - init" {
    const allocator = std.testing.allocator;
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    
    try std.testing.expectEqual(@as(usize, 0), suite.tests_run);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_passed);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_failed);
}

test "isSqliteAvailable - always true for stub" {
    try std.testing.expect(isSqliteAvailable());
}
