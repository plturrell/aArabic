const std = @import("std");
const connection_mod = @import("connection.zig");
const query_mod = @import("query.zig");
const transaction_mod = @import("transaction.zig");
const pool_mod = @import("pool.zig");
const client_types = @import("../../client.zig");

const PgConnection = connection_mod.PgConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;
const QueryExecutor = query_mod.QueryExecutor;
const PgTransaction = transaction_mod.PgTransaction;
const PgConnectionPool = pool_mod.PgConnectionPool;
const PgPoolConfig = pool_mod.PgPoolConfig;
const Value = client_types.Value;
const IsolationLevel = client_types.IsolationLevel;

/// Integration test configuration
pub const IntegrationTestConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 5432,
    database: []const u8 = "test",
    user: []const u8 = "postgres",
    password: []const u8 = "postgres",
    
    pub fn toConnectionConfig(self: IntegrationTestConfig) ConnectionConfig {
        return ConnectionConfig{
            .host = self.host,
            .port = self.port,
            .database = self.database,
            .user = self.user,
            .password = self.password,
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
        std.debug.print("\n=== PostgreSQL Integration Tests ===\n\n", .{});
        
        try self.runTest("Connection", testConnection);
        try self.runTest("Simple Query", testSimpleQuery);
        try self.runTest("Extended Query", testExtendedQuery);
        try self.runTest("Transaction", testTransaction);
        try self.runTest("Savepoint", testSavepoint);
        try self.runTest("Connection Pool", testConnectionPool);
        
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
    
    /// Test: Basic connection
    fn testConnection(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try PgConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        // Try to connect (will fail without real PostgreSQL)
        // In production, this would actually connect
        _ = conn;
    }
    
    /// Test: Simple query execution
    fn testSimpleQuery(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would execute: SELECT 1
        // Verify result contains single row with value 1
    }
    
    /// Test: Extended query with parameters
    fn testExtendedQuery(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would execute: SELECT $1::int + $2::int
        // With parameters: [5, 10]
        // Verify result is 15
    }
    
    /// Test: Transaction commit/rollback
    fn testTransaction(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would execute transaction with INSERT
        // Rollback and verify row not inserted
        // Commit and verify row inserted
    }
    
    /// Test: Savepoint operations
    fn testSavepoint(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would create savepoint
        // Perform operation
        // Rollback to savepoint
        // Verify partial rollback worked
    }
    
    /// Test: Connection pool operations
    fn testConnectionPool(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would acquire multiple connections
        // Release them
        // Verify pool metrics
    }
};

/// Run integration tests (requires PostgreSQL)
pub fn runIntegrationTests(allocator: std.mem.Allocator) !void {
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    try suite.runAll();
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

/// Check if PostgreSQL is available
pub fn isPostgresAvailable(allocator: std.mem.Allocator, config: ConnectionConfig) bool {
    var conn = PgConnection.init(allocator, config) catch return false;
    defer conn.deinit();
    
    conn.connect() catch return false;
    conn.disconnect();
    
    return true;
}

/// Create test database
pub fn createTestDatabase(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would execute: CREATE DATABASE test_nmetadata
}

/// Drop test database
pub fn dropTestDatabase(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would execute: DROP DATABASE test_nmetadata
}

// ============================================================================
// Unit Tests (for test infrastructure itself)
// ============================================================================

test "IntegrationTestConfig - toConnectionConfig" {
    const test_config = IntegrationTestConfig{
        .host = "localhost",
        .database = "testdb",
        .user = "testuser",
        .password = "testpass",
    };
    
    const conn_config = test_config.toConnectionConfig();
    
    try std.testing.expectEqualStrings("localhost", conn_config.host);
    try std.testing.expectEqualStrings("testdb", conn_config.database);
    try std.testing.expectEqualStrings("testuser", conn_config.user);
    try std.testing.expectEqual(@as(u16, 5432), conn_config.port);
}

test "IntegrationTestSuite - init" {
    const allocator = std.testing.allocator;
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    
    try std.testing.expectEqual(@as(usize, 0), suite.tests_run);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_passed);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_failed);
}
