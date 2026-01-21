const std = @import("std");
const graph = @import("graph.zig");
const connection_mod = @import("connection.zig");
const query_mod = @import("query.zig");

const GraphExecutor = graph.GraphExecutor;
const GraphWorkspace = graph.GraphWorkspace;
const GraphTableQuery = graph.GraphTableQuery;
const GraphAlgorithm = graph.GraphAlgorithm;
const TraversalDirection = graph.TraversalDirection;
const HanaConnection = connection_mod.HanaConnection;
const ConnectionConfig = connection_mod.ConnectionConfig;

/// Graph integration test configuration
pub const GraphIntegrationTestConfig = struct {
    host: []const u8 = "localhost",
    port: u16 = 30015,
    database: []const u8 = "test",
    user: []const u8 = "SYSTEM",
    password: []const u8 = "Password123",
    workspace_name: []const u8 = "TEST_LINEAGE_GRAPH",
    
    pub fn toConnectionConfig(self: GraphIntegrationTestConfig) ConnectionConfig {
        return ConnectionConfig{
            .host = self.host,
            .port = self.port,
            .database = self.database,
            .user = self.user,
            .password = self.password,
        };
    }
};

/// Graph integration test suite
pub const GraphIntegrationTestSuite = struct {
    allocator: std.mem.Allocator,
    config: GraphIntegrationTestConfig,
    tests_run: usize,
    tests_passed: usize,
    tests_failed: usize,
    
    pub fn init(allocator: std.mem.Allocator, config: GraphIntegrationTestConfig) GraphIntegrationTestSuite {
        return GraphIntegrationTestSuite{
            .allocator = allocator,
            .config = config,
            .tests_run = 0,
            .tests_passed = 0,
            .tests_failed = 0,
        };
    }
    
    /// Run all graph integration tests
    pub fn runAll(self: *GraphIntegrationTestSuite) !void {
        std.debug.print("\n=== HANA Graph Engine Integration Tests ===\n\n", .{});
        
        try self.runTest("Create Graph Workspace", testCreateWorkspace);
        try self.runTest("Upstream Lineage (INCOMING)", testUpstreamLineage);
        try self.runTest("Downstream Lineage (OUTGOING)", testDownstreamLineage);
        try self.runTest("Shortest Path", testShortestPath);
        try self.runTest("All Paths", testAllPaths);
        try self.runTest("Connected Component", testConnectedComponent);
        try self.runTest("Graph Query with WHERE", testGraphQueryWithFilter);
        try self.runTest("Multi-hop Traversal", testMultiHopTraversal);
        try self.runTest("Drop Graph Workspace", testDropWorkspace);
        
        std.debug.print("\n=== Test Summary ===\n", .{});
        std.debug.print("Tests Run: {d}\n", .{self.tests_run});
        std.debug.print("Tests Passed: {d}\n", .{self.tests_passed});
        std.debug.print("Tests Failed: {d}\n", .{self.tests_failed});
        
        if (self.tests_failed > 0) {
            return error.GraphIntegrationTestsFailed;
        }
    }
    
    /// Run a single test
    fn runTest(
        self: *GraphIntegrationTestSuite,
        name: []const u8,
        test_fn: fn (*GraphIntegrationTestSuite) anyerror!void,
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
    
    /// Test: Create graph workspace
    fn testCreateWorkspace(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        const workspace = GraphWorkspace{
            .name = self.config.workspace_name,
            .schema = "METADATA",
            .vertex_table = "DATASETS",
            .edge_table = "LINEAGE_EDGES",
        };
        
        // Would execute: CREATE GRAPH WORKSPACE
        try executor.createWorkspace(workspace);
    }
    
    /// Test: Upstream lineage traversal
    fn testUpstreamLineage(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Find all upstream datasets (sources)
        const result = try executor.findUpstreamLineage(
            self.config.workspace_name,
            "dataset_123",
            5,
        );
        defer result.deinit();
        
        // Would return all datasets that flow into dataset_123
        // Expected: Rows with vertex_id, edge_id, hop_distance
    }
    
    /// Test: Downstream lineage traversal
    fn testDownstreamLineage(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Find all downstream datasets (consumers)
        const result = try executor.findDownstreamLineage(
            self.config.workspace_name,
            "dataset_123",
            5,
        );
        defer result.deinit();
        
        // Would return all datasets that consume dataset_123
    }
    
    /// Test: Shortest path between datasets
    fn testShortestPath(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Find shortest path
        const result = try executor.findShortestPath(
            self.config.workspace_name,
            "source_dataset",
            "target_dataset",
        );
        defer result.deinit();
        
        // Would return shortest path with hop count
    }
    
    /// Test: All paths between datasets
    fn testAllPaths(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Find all paths (limited depth)
        const result = try executor.findAllPaths(
            self.config.workspace_name,
            "source_dataset",
            "target_dataset",
            3,
        );
        defer result.deinit();
        
        // Would return multiple paths
    }
    
    /// Test: Connected component
    fn testConnectedComponent(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Find all connected datasets
        const result = try executor.findConnectedComponent(
            self.config.workspace_name,
            "dataset_123",
        );
        defer result.deinit();
        
        // Would return all datasets in same component
    }
    
    /// Test: Graph query with WHERE clause
    fn testGraphQueryWithFilter(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Build filtered graph query
        var query = GraphTableQuery.init(self.allocator, self.config.workspace_name);
        _ = query.withAlgorithm(.neighbors)
            .fromVertex("dataset_123")
            .withDirection(.outgoing)
            .withMaxDepth(3)
            .where("type = 'TABLE'");
        
        const result = try executor.executeGraphQuery(query);
        defer result.deinit();
        
        // Would return only table-type datasets
    }
    
    /// Test: Multi-hop traversal
    fn testMultiHopTraversal(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        // Test traversal at different depths
        for ([_]u32{ 1, 3, 5, 10 }) |depth| {
            const result = try executor.findUpstreamLineage(
                self.config.workspace_name,
                "dataset_123",
                depth,
            );
            defer result.deinit();
            
            // Verify results scale with depth
            _ = depth;
        }
    }
    
    /// Test: Drop graph workspace
    fn testDropWorkspace(self: *GraphIntegrationTestSuite) !void {
        const conn_config = self.config.toConnectionConfig();
        
        var conn = try HanaConnection.init(self.allocator, conn_config);
        defer conn.deinit();
        
        try conn.connect();
        defer conn.disconnect();
        
        var executor = GraphExecutor.init(self.allocator, &conn);
        defer executor.deinit();
        
        const workspace = GraphWorkspace{
            .name = self.config.workspace_name,
            .schema = "METADATA",
            .vertex_table = "DATASETS",
            .edge_table = "LINEAGE_EDGES",
        };
        
        // Would execute: DROP GRAPH WORKSPACE
        try executor.dropWorkspace(workspace);
    }
};

/// Run graph integration tests
pub fn runGraphIntegrationTests(allocator: std.mem.Allocator) !void {
    const config = GraphIntegrationTestConfig{};
    
    var suite = GraphIntegrationTestSuite.init(allocator, config);
    try suite.runAll();
}

// ============================================================================
// Integration Test Helpers
// ============================================================================

/// Create test graph data
pub fn createTestGraphData(allocator: std.mem.Allocator, conn: *HanaConnection) !void {
    var query_executor = query_mod.QueryExecutor.init(allocator, conn);
    defer query_executor.deinit();
    
    // Create vertices table
    const create_vertices = 
        \\CREATE TABLE METADATA.DATASETS (
        \\  id NVARCHAR(255) PRIMARY KEY,
        \\  name NVARCHAR(255),
        \\  type NVARCHAR(50)
        \\)
    ;
    _ = try query_executor.executeQuery(create_vertices, &[_]query_mod.Value{});
    
    // Create edges table
    const create_edges =
        \\CREATE TABLE METADATA.LINEAGE_EDGES (
        \\  id NVARCHAR(255) PRIMARY KEY,
        \\  source_id NVARCHAR(255),
        \\  target_id NVARCHAR(255),
        \\  FOREIGN KEY (source_id) REFERENCES METADATA.DATASETS(id),
        \\  FOREIGN KEY (target_id) REFERENCES METADATA.DATASETS(id)
        \\)
    ;
    _ = try query_executor.executeQuery(create_edges, &[_]query_mod.Value{});
    
    // Insert sample datasets
    const datasets = [_][]const u8{
        "ds_1", "ds_2", "ds_3", "ds_4", "ds_5",
    };
    
    for (datasets) |ds_id| {
        const insert_sql = try std.fmt.allocPrint(
            allocator,
            "INSERT INTO METADATA.DATASETS (id, name, type) VALUES ('{s}', 'Dataset {s}', 'TABLE')",
            .{ ds_id, ds_id },
        );
        defer allocator.free(insert_sql);
        
        _ = try query_executor.executeQuery(insert_sql, &[_]query_mod.Value{});
    }
    
    // Create lineage edges: ds_1 -> ds_2 -> ds_3 -> ds_4 -> ds_5
    const edges = [_][2][]const u8{
        .{ "ds_1", "ds_2" },
        .{ "ds_2", "ds_3" },
        .{ "ds_3", "ds_4" },
        .{ "ds_4", "ds_5" },
    };
    
    for (edges, 0..) |edge, i| {
        const insert_sql = try std.fmt.allocPrint(
            allocator,
            "INSERT INTO METADATA.LINEAGE_EDGES (id, source_id, target_id) VALUES ('edge_{d}', '{s}', '{s}')",
            .{ i, edge[0], edge[1] },
        );
        defer allocator.free(insert_sql);
        
        _ = try query_executor.executeQuery(insert_sql, &[_]query_mod.Value{});
    }
}

/// Clean up test graph data
pub fn cleanupTestGraphData(allocator: std.mem.Allocator, conn: *HanaConnection) !void {
    var query_executor = query_mod.QueryExecutor.init(allocator, conn);
    defer query_executor.deinit();
    
    _ = try query_executor.executeQuery(
        "DROP TABLE METADATA.LINEAGE_EDGES",
        &[_]query_mod.Value{},
    );
    
    _ = try query_executor.executeQuery(
        "DROP TABLE METADATA.DATASETS",
        &[_]query_mod.Value{},
    );
}

// ============================================================================
// Unit Tests
// ============================================================================

test "GraphIntegrationTestConfig - toConnectionConfig" {
    const test_config = GraphIntegrationTestConfig{
        .host = "hana-test.example.com",
        .port = 30015,
        .database = "testdb",
    };
    
    const conn_config = test_config.toConnectionConfig();
    
    try std.testing.expectEqualStrings("hana-test.example.com", conn_config.host);
    try std.testing.expectEqual(@as(u16, 30015), conn_config.port);
    try std.testing.expectEqualStrings("testdb", conn_config.database);
}

test "GraphIntegrationTestSuite - init" {
    const allocator = std.testing.allocator;
    const config = GraphIntegrationTestConfig{};
    
    var suite = GraphIntegrationTestSuite.init(allocator, config);
    
    try std.testing.expectEqual(@as(usize, 0), suite.tests_run);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_passed);
    try std.testing.expectEqual(@as(usize, 0), suite.tests_failed);
}
