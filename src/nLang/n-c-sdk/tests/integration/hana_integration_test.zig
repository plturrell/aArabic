const std = @import("std");
const hana = @import("../../zig-out/lib/zig/hana/client.zig");
const QueryBuilder = hana.QueryBuilder;
const BatchOperations = hana.BatchOperations;
const StreamingQuery = hana.StreamingQuery;
const testing = std.testing;

/// Integration tests for HANA client
/// These tests validate complete database workflows

test "HANA Integration - Client initialization and connection pool" {
    const allocator = testing.allocator;

    const config = hana.HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 3,
        .pool_max = 10,
    };

    const client = try hana.HanaClient.init(allocator, config);
    defer client.deinit();

    // Verify pool initialization
    try testing.expect(client.pool.connections.items.len == 3);
    try testing.expect(client.pool.available.items.len == 3);

    // Verify metrics
    const metrics = client.getMetrics();
    try testing.expect(metrics.total_connections == 3);
    try testing.expect(metrics.idle_connections == 3);
}

test "HANA Integration - Connection acquisition and release cycle" {
    const allocator = testing.allocator;

    const config = hana.HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 2,
        .pool_max = 5,
    };

    const client = try hana.HanaClient.init(allocator, config);
    defer client.deinit();

    // Acquire multiple connections
    const conn1 = try client.getConnection();
    try testing.expect(client.pool.available.items.len == 1);

    const conn2 = try client.getConnection();
    try testing.expect(client.pool.available.items.len == 0);

    // Release connections
    client.releaseConnection(conn1);
    try testing.expect(client.pool.available.items.len == 1);

    client.releaseConnection(conn2);
    try testing.expect(client.pool.available.items.len == 2);
}

test "HANA Integration - Query builder with complex query" {
    const allocator = testing.allocator;

    var qb = QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("u.id")
        .select("u.name")
        .select("COUNT(o.id) as order_count")
        .from("users u")
        .leftJoin("orders o", "o.user_id = u.id")
        .where("u.status", .eq, "'active'")
        .where("u.created_at", .gte, "'2024-01-01'")
        .groupBy("u.id")
        .groupBy("u.name")
        .having("COUNT(o.id)", .gt, "5")
        .orderBy("order_count", .desc)
        .limit(100)
        .offset(0);

    const sql = try qb.build();
    defer allocator.free(sql);

    // Verify SQL structure
    try testing.expect(std.mem.indexOf(u8, sql, "SELECT u.id, u.name, COUNT(o.id) as order_count") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "LEFT JOIN orders o ON o.user_id = u.id") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "WHERE u.status = 'active' AND u.created_at >= '2024-01-01'") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "GROUP BY u.id, u.name") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "HAVING COUNT(o.id) > 5") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "ORDER BY order_count DESC") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "LIMIT 100") != null);
}

test "HANA Integration - Batch operations structure" {
    const allocator = testing.allocator;

    var batch = BatchOperations.init(allocator, .{
        .batch_size = 1000,
    });
    defer batch.deinit();

    // Add multiple operations
    const params1 = [_]hana.Parameter{ .{ .int = 1 }, .{ .string = "Alice" } };
    const params2 = [_]hana.Parameter{ .{ .int = 2 }, .{ .string = "Bob" } };
    const params3 = [_]hana.Parameter{ .{ .int = 3 }, .{ .string = "Charlie" } };

    try batch.add("INSERT INTO users (id, name) VALUES (?, ?)", &params1);
    try batch.add("INSERT INTO users (id, name) VALUES (?, ?)", &params2);
    try batch.add("INSERT INTO users (id, name) VALUES (?, ?)", &params3);

    try testing.expect(batch.operations.items.len == 3);
}

test "HANA Integration - Bulk insert validation" {
    const allocator = testing.allocator;

    const columns = [_][]const u8{ "id", "name", "email", "age" };
    var inserter = hana.BulkInserter.init(allocator, "users", &columns, 1000);
    defer inserter.deinit();

    // Add rows
    for (0..100) |i| {
        const values = [_]hana.Parameter{
            .{ .int = @intCast(i) },
            .{ .string = "User" },
            .{ .string = "user@example.com" },
            .{ .int = 25 },
        };
        try inserter.addRow(&values);
    }

    try testing.expect(inserter.rows.items.len == 100);
}

test "HANA Integration - Parameter type validation" {
    const allocator = testing.allocator;

    // Test different parameter types
    const params = [_]hana.Parameter{
        .{ .int = 42 },
        .{ .float = 3.14 },
        .{ .string = "Hello, HANA!" },
        .{ .bool_value = true },
        .null_value,
    };

    _ = allocator;
    try testing.expect(params.len == 5);

    // Verify parameter access
    switch (params[0]) {
        .int => |val| try testing.expect(val == 42),
        else => try testing.expect(false),
    }

    switch (params[1]) {
        .float => |val| try testing.expect(val == 3.14),
        else => try testing.expect(false),
    }
}

test "HANA Integration - Result parsing and type conversion" {
    const allocator = testing.allocator;

    // Create mock result
    var values = try allocator.alloc(hana.Value, 4);
    values[0] = .{ .int = 1 };
    values[1] = .{ .string = try allocator.dupe(u8, "Alice") };
    values[2] = .{ .float = 30.5 };
    values[3] = .{ .bool_value = true };

    var column_names = try allocator.alloc([]const u8, 4);
    column_names[0] = "id";
    column_names[1] = "name";
    column_names[2] = "score";
    column_names[3] = "active";

    const row = hana.Row{
        .values = values,
        .column_names = column_names,
        .allocator = allocator,
    };
    defer row.deinit();

    // Test type conversions
    try testing.expect(row.getInt("id") == 1);
    try testing.expectEqualStrings("Alice", row.getString("name").?);
    try testing.expect(row.getFloat("score") == 30.5);
    try testing.expect(row.getBool("active").? == true);
}

test "HANA Integration - Connection health check" {
    const allocator = testing.allocator;

    const config = hana.HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 1,
        .pool_max = 5,
    };

    const client = try hana.HanaClient.init(allocator, config);
    defer client.deinit();

    // Health check should succeed for newly created client
    const healthy = try client.healthCheck();
    try testing.expect(healthy);
}

test "HANA Integration - Transaction semantics" {
    const allocator = testing.allocator;
    
    // Test structure for transaction operations
    // In production, would test actual BEGIN/COMMIT/ROLLBACK
    
    const config = hana.HanaClient.HanaConfig{
        .host = "localhost",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
        .pool_min = 1,
        .pool_max = 5,
    };

    const client = try hana.HanaClient.init(allocator, config);
    defer client.deinit();

    const conn = try client.getConnection();
    defer client.releaseConnection(conn);

    // Transaction workflow
    try conn.execute("BEGIN TRANSACTION");
    try conn.execute("INSERT INTO test VALUES (1, 'test')");
    try conn.execute("COMMIT");
}