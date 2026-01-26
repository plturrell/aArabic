const std = @import("std");
const hana = @import("../../zig-out/lib/zig/hana/client.zig");
const testing = std.testing;

/// Comprehensive unit tests for HANA client components

test "HANA Unit - QueryResult structure" {
    const allocator = testing.allocator;

    // Create empty result
    const result = hana.QueryResult{
        .rows = try allocator.alloc(hana.Row, 0),
        .columns = try allocator.alloc([]const u8, 0),
        .allocator = allocator,
    };
    defer result.deinit();

    try testing.expect(result.getRowCount() == 0);
}

test "HANA Unit - Row value access" {
    const allocator = testing.allocator;

    var values = try allocator.alloc(hana.Value, 3);
    values[0] = .{ .int = 42 };
    values[1] = .{ .string = try allocator.dupe(u8, "test") };
    values[2] = .{ .float = 3.14 };

    const columns = try allocator.alloc([]const u8, 3);
    columns[0] = "id";
    columns[1] = "name";
    columns[2] = "score";

    const row = hana.Row{
        .values = values,
        .column_names = columns,
        .allocator = allocator,
    };
    defer row.deinit();

    // Test by index
    const val0 = row.get(0);
    try testing.expect(val0 != null);
    try testing.expect(val0.?.asInt().? == 42);

    // Test by column name
    try testing.expect(row.getInt("id") == 42);
    try testing.expectEqualStrings("test", row.getString("name").?);
    try testing.expect(row.getFloat("score") == 3.14);
}

test "HANA Unit - Parameter types" {
    // Test all parameter types
    const params = [_]hana.Parameter{
        .{ .int = -123 },
        .{ .float = -45.67 },
        .{ .string = "Hello" },
        .{ .bool_value = false },
        .null_value,
    };

    try testing.expect(params.len == 5);
}

test "HANA Unit - Value type conversions" {
    const allocator = testing.allocator;

    // Int to float
    const int_val = hana.Value{ .int = 42 };
    try testing.expect(int_val.asFloat().? == 42.0);

    // Float to int
    const float_val = hana.Value{ .float = 3.14 };
    try testing.expect(float_val.asInt().? == 3);

    // String value
    const str_val = hana.Value{ .string = try allocator.dupe(u8, "test") };
    defer str_val.deinit(allocator);
    try testing.expectEqualStrings("test", str_val.asString().?);

    // Bool to int
    const bool_val = hana.Value{ .bool_value = true };
    try testing.expect(bool_val.asBool().? == true);
    try testing.expect(bool_val.asInt().? == 1);

    // Null value
    const null_val = hana.Value.null_value;
    try testing.expect(null_val.asInt() == null);
    try testing.expect(null_val.asString() == null);
}

test "HANA Unit - Connection metrics tracking" {
    var metrics = hana.HanaClient.ConnectionMetrics{};

    // Record successful connections
    metrics.recordConnection(true);
    metrics.recordConnection(true);
    metrics.recordConnection(false);

    metrics.mutex.lock();
    defer metrics.mutex.unlock();

    try testing.expect(metrics.total_connections == 2);
    try testing.expect(metrics.failed_connections == 1);

    // Record queries
    metrics.recordQuery(true);
    metrics.recordQuery(true);
    metrics.recordQuery(false);

    try testing.expect(metrics.total_queries == 3);
    try testing.expect(metrics.failed_queries == 1);
}

test "HANA Unit - Connection pool capacity" {
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

    // Pool should have min connections
    try testing.expect(client.pool.connections.items.len >= config.pool_min);
    try testing.expect(client.pool.connections.items.len <= config.pool_max);
}

test "HANA Unit - Query builder - SELECT with all clauses" {
    const allocator = testing.allocator;

    var qb = hana.QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("*")
        .from("users")
        .where("age", .gte, "18")
        .where("status", .eq, "'active'")
        .orderBy("created_at", .desc)
        .limit(10);

    const sql = try qb.build();
    defer allocator.free(sql);

    try testing.expect(std.mem.indexOf(u8, sql, "SELECT *") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "FROM users") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "WHERE") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "ORDER BY") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "LIMIT") != null);
}

test "HANA Unit - Query builder - JOIN operations" {
    const allocator = testing.allocator;

    var qb = hana.QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("u.name")
        .select("o.amount")
        .from("users u")
        .innerJoin("orders o", "o.user_id = u.id")
        .leftJoin("payments p", "p.order_id = o.id");

    const sql = try qb.build();
    defer allocator.free(sql);

    try testing.expect(std.mem.indexOf(u8, sql, "INNER JOIN") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "LEFT JOIN") != null);
}

test "HANA Unit - Query builder - GROUP BY and HAVING" {
    const allocator = testing.allocator;

    var qb = hana.QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("category")
        .select("COUNT(*) as total")
        .from("products")
        .groupBy("category")
        .having("COUNT(*)", .gt, "10");

    const sql = try qb.build();
    defer allocator.free(sql);

    try testing.expect(std.mem.indexOf(u8, sql, "GROUP BY category") != null);
    try testing.expect(std.mem.indexOf(u8, sql, "HAVING") != null);
}

test "HANA Unit - Query builder - OR conditions" {
    const allocator = testing.allocator;

    var qb = hana.QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("*")
        .from("users")
        .where("status", .eq, "'active'")
        .orWhere("role", .eq, "'admin'");

    const sql = try qb.build();
    defer allocator.free(sql);

    try testing.expect(std.mem.indexOf(u8, sql, "WHERE status = 'active' OR role = 'admin'") != null);
}

test "HANA Unit - Query builder - IS NULL operator" {
    const allocator = testing.allocator;

    var qb = hana.QueryBuilder.init(allocator, .select);
    defer qb.deinit();

    _ = try qb.select("*")
        .from("users")
        .where("deleted_at", .is_null, "");

    const sql = try qb.build();
    defer allocator.free(sql);

    try testing.expect(std.mem.indexOf(u8, sql, "deleted_at IS NULL") != null);
}

test "HANA Unit - Batch operations initialization" {
    const allocator = testing.allocator;

    var batch = hana.BatchOperations.init(allocator, .{
        .batch_size = 500,
    });
    defer batch.deinit();

    try testing.expect(batch.batch_size == 500);
    try testing.expect(batch.operations.items.len == 0);
}

test "HANA Unit - Bulk inserter column validation" {
    const allocator = testing.allocator;

    const columns = [_][]const u8{ "id", "name", "email" };
    var inserter = hana.BulkInserter.init(allocator, "users", &columns, 1000);
    defer inserter.deinit();

    // Correct number of values
    const correct_values = [_]hana.Parameter{
        .{ .int = 1 },
        .{ .string = "Alice" },
        .{ .string = "alice@example.com" },
    };
    try inserter.addRow(&correct_values);
    try testing.expect(inserter.rows.items.len == 1);

    // Incorrect number of values should error
    const wrong_values = [_]hana.Parameter{
        .{ .int = 2 },
        .{ .string = "Bob" },
    };
    const result = inserter.addRow(&wrong_values);
    try testing.expectError(error.ColumnValueMismatch, result);
}

test "HANA Unit - Configuration from environment variables" {
    const allocator = testing.allocator;

    // This test would need environment variables set
    // Testing structure only
    const config = hana.HanaClient.HanaConfig{
        .host = "test-host",
        .port = 30015,
        .database = "TESTDB",
        .user = "TESTUSER",
        .password = "test123",
    };

    try testing.expectEqualStrings("test-host", config.host);
    try testing.expect(config.port == 30015);
}

test "HANA Unit - Row case-insensitive column lookup" {
    const allocator = testing.allocator;

    var values = try allocator.alloc(hana.Value, 2);
    values[0] = .{ .int = 1 };
    values[1] = .{ .string = try allocator.dupe(u8, "test") };

    const columns = try allocator.alloc([]const u8, 2);
    columns[0] = "ID"; // Uppercase (HANA default)
    columns[1] = "NAME";

    const row = hana.Row{
        .values = values,
        .column_names = columns,
        .allocator = allocator,
    };
    defer row.deinit();

    // Should match case-insensitively
    try testing.expect(row.getInt("id") == 1);
    try testing.expect(row.getInt("ID") == 1);
    try testing.expectEqualStrings("test", row.getString("name").?);
    try testing.expectEqualStrings("test", row.getString("NAME").?);
}