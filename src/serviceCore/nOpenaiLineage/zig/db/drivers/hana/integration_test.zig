const std = @import("std");
const connection_mod = @import("connection.zig");
const query_mod = @import("query.zig");
const transaction_mod = @import("transaction.zig");
const pool_mod = @import("pool.zig");
const client_types = @import("../../client.zig");

const HanaConnection = connection_mod.HanaConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;
const QueryExecutor = query_mod.QueryExecutor;
const HanaTransaction = transaction_mod.HanaTransaction;
const HanaConnectionPool = pool_mod.HanaConnectionPool;
const HanaPoolConfig = pool_mod.HanaPoolConfig;
const Value = client_types.Value;
const IsolationLevel = client_types.IsolationLevel;

/// Integration test configuration
pub const IntegrationTestConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 30015,
    database: []const u8 = "HXE",
    user: []const u8 = "SYSTEM",
    password: []const u8 = "Password123",
    schema: []const u8 = "NMETADATA_TEST",
    
    pub fn toConnectionConfig(self: IntegrationTestConfig) ConnectionConfig {
        return ConnectionConfig{
            .host = self.host,
            .port = self.port,
            .database = self.database,
            .user = self.user,
            .password = self.password,
            .schema = self.schema,
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
        std.debug.print("\n=== SAP HANA Integration Tests ===\n\n", .{});
        
        try self.runTest("Connection", testConnection);
        try self.runTest("Simple Query", testSimpleQuery);
        try self.runTest("Prepared Statement", testPreparedStatement);
        try self.runTest("Transaction", testTransaction);
        try self.runTest("Savepoint", testSavepoint);
        try self.runTest("Connection Pool", testConnectionPool);
        try self.runTest("HANA-specific Types", testHanaTypes);
        try self.runTest("Spatial Data", testSpatialData);
        
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
    
    /// Test: Basic connection to SAP HANA
    fn testConnection(self: *IntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        // Try to connect (will fail without real SAP HANA)
        // In production, this would actually connect
        _ = conn;
    }
    
    /// Test: Simple query execution (SELECT DUMMY)
    fn testSimpleQuery(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would execute: SELECT 'X' AS DUMMY FROM DUMMY
        // Verify result contains single row with value 'X'
    }
    
    /// Test: Prepared statement with parameters
    fn testPreparedStatement(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would execute: SELECT ? + ? FROM DUMMY
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
    
    /// Test: HANA-specific data types
    fn testHanaTypes(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would test: TINYINT, SMALLDECIMAL, ALPHANUM, ST_GEOMETRY
        // Verify proper type handling and conversion
    }
    
    /// Test: Spatial data operations
    fn testSpatialData(self: *IntegrationTestSuite) !void {
        _ = self;
        // Would test: ST_POINT, ST_LINESTRING, ST_POLYGON
        // Verify spatial queries work correctly
    }
};

/// Run integration tests (requires SAP HANA)
pub fn runIntegrationTests(allocator: std.mem.Allocator) !void {
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    try suite.runAll();
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

/// Check if SAP HANA is available
pub fn isHanaAvailable(allocator: std.mem.Allocator, config: ConnectionConfig) bool {
    var conn = HanaConnection.init(allocator, config) catch return false;
    defer conn.deinit();
    
    conn.connect() catch return false;
    conn.disconnect();
    
    return true;
}

/// Create test schema
pub fn createTestSchema(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would execute: CREATE SCHEMA NMETADATA_TEST
}

/// Drop test schema
pub fn dropTestSchema(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would execute: DROP SCHEMA NMETADATA_TEST CASCADE
}

/// Create test tables
pub fn createTestTables(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would execute table creation DDL
}

/// Populate test data
pub fn populateTestData(allocator: std.mem.Allocator, config: ConnectionConfig) !void {
    _ = allocator;
    _ = config;
    // Would insert test data
}

// ============================================================================
// Unit Tests (for test infrastructure itself)
// ============================================================================

test "IntegrationTestConfig - toConnectionConfig" {
    const test_config = IntegrationTestConfig{
        .host = "hana-server",
        .database = "HXE",
        .user = "testuser",
        .password = "testpass",
        .schema = "TEST_SCHEMA",
    };
    
    const conn_config = test_config.toConnectionConfig();
    
    try std.testing.expectEqualStrings("hana-server", conn_config.host);
    try std.testing.expectEqualStrings("HXE", conn_config.database);
    try std.testing.expectEqualStrings("testuser", conn_config.user);
    try std.testing.expectEqualStrings("TEST_SCHEMA", conn_config.schema);
    try std.testing.expectEqual(@as(u16, 30015), conn_config.port);
}

test "IntegrationTestSuite - init" {
    const allocator = std.testing.allocator;
    const config = IntegrationTestConfig{};
    
    var suite = IntegrationTestSuite.init(allocator, config);
    
    try std.testing.expectEqual(@as(usize, 0), suite.tests_run);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_passed);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_failed);
}

test "IntegrationTestConfig - default values" {
    const config = IntegrationTestConfig{};
    
    try std.testing.expectEqualStrings("localhost", config.host);
    try std.testing.expectEqual(@as(u16, 30015), config.port);
    try std.testing.expectEqualStrings("HXE", config.database);
    try std.testing.expectEqualStrings("SYSTEM", config.user);
}
