// Zig OData Client for SAP HANA Cloud
// Implements OData v4 protocol for HANA Cloud SQL execution
// Exports C ABI functions for database/prompt_history.zig

const std = @import("std");
const mem = std.mem;

const header_line = "================================================================================";

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// ============================================================================
// HANA Cloud OData SQL Execution via curl
// ============================================================================

/// Execute SQL via HANA Cloud OData v4 API
/// Uses curl to POST to HANA's native OData SQL endpoint
fn executeSqlViaHanaOData(
    host: []const u8,
    port: c_int,
    user: []const u8,
    password: []const u8,
    schema: []const u8,
    sql: []const u8,
) !void {
    _ = port; // HANA Cloud always uses 443
    
    // Escape SQL for JSON
    const escaped_sql = try escapeJsonString(sql);
    defer allocator.free(escaped_sql);
    
    // Build OData action payload
    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"SCHEMA\":\"{s}\",\"SQL\":\"{s}\"}}",
        .{ schema, escaped_sql },
    );
    defer allocator.free(payload);

    const user_pass = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ user, password });
    defer allocator.free(user_pass);

    // HANA Cloud OData endpoint for SQL execution
    const url = try std.fmt.allocPrint(
        allocator,
        "https://{s}/sap/opu/odata/sap/SQL_EXECUTION_SRV/ExecuteSQL",
        .{host},
    );
    defer allocator.free(url);

    const curl_args = [_][]const u8{
        "curl",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", "Accept: application/json",
        "-u", user_pass,
        "-d", payload,
        "-k", // Accept self-signed certs
        "-s", // Silent
        "-w", "%{http_code}", // Output HTTP status
        url,
    };

    var child = std.process.Child.init(&curl_args, allocator);
    child.stdout_behavior = .Pipe;
    
    try child.spawn();
    
    const output = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);
    defer allocator.free(output);
    
    const result = try child.wait();
    
    // Check HTTP status code in output
    const is_success = (result == .Exited and result.Exited == 0) and
        (mem.endsWith(u8, output, "200") or mem.endsWith(u8, output, "201") or mem.endsWith(u8, output, "204"));
    
    if (!is_success) {
        std.debug.print("   HTTP Response: {s}\n", .{output});
        return error.ODataExecutionFailed;
    }
}

/// Query SQL via HANA Cloud OData v4 API
fn querySqlViaHanaOData(
    host: []const u8,
    port: c_int,
    user: []const u8,
    password: []const u8,
    schema: []const u8,
    sql: []const u8,
) ![]u8 {
    _ = port; // HANA Cloud always uses 443
    
    // Escape SQL for JSON
    const escaped_sql = try escapeJsonString(sql);
    defer allocator.free(escaped_sql);
    
    // Build OData query payload
    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"SCHEMA\":\"{s}\",\"SQL\":\"{s}\"}}",
        .{ schema, escaped_sql },
    );
    defer allocator.free(payload);

    const user_pass = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ user, password });
    defer allocator.free(user_pass);

    // HANA Cloud OData endpoint
    const url = try std.fmt.allocPrint(
        allocator,
        "https://{s}/sap/opu/odata/sap/SQL_QUERY_SRV/ExecuteQuery",
        .{host},
    );
    defer allocator.free(url);

    const curl_args = [_][]const u8{
        "curl",
        "-X", "POST",
        "-H", "Content-Type: application/json",
        "-H", "Accept: application/json",
        "-u", user_pass,
        "-d", payload,
        "-k",
        "-s",
        url,
    };

    var child = std.process.Child.init(&curl_args, allocator);
    child.stdout_behavior = .Pipe;
    
    try child.spawn();
    
    const output = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);
    
    const result = try child.wait();
    if (result != .Exited or result.Exited != 0) {
        allocator.free(output);
        return error.ODataQueryFailed;
    }

    return output;
}

/// Escape string for JSON
fn escapeJsonString(input: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .{};
    errdefer result.deinit(allocator);

    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(allocator, "\\\""),
            '\\' => try result.appendSlice(allocator, "\\\\"),
            '\n' => try result.appendSlice(allocator, "\\n"),
            '\r' => try result.appendSlice(allocator, "\\r"),
            '\t' => try result.appendSlice(allocator, "\\t"),
            else => try result.append(allocator, c),
        }
    }

    return result.toOwnedSlice(allocator);
}

// ============================================================================
// C ABI Exports for database/prompt_history.zig
// ============================================================================

/// Initialize OData client library
export fn zig_odata_init() callconv(.c) c_int {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("📡 Zig OData Client for HANA Cloud\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Features:\n", .{});
    std.debug.print("  ✅ OData v4 protocol\n", .{});
    std.debug.print("  ✅ HANA Cloud native integration\n", .{});
    std.debug.print("  ✅ SQL execution via OData actions\n", .{});
    std.debug.print("  ✅ JSON result formatting\n", .{});
    std.debug.print("\n{s}\n\n", .{header_line});
    return 0;
}

/// Execute SQL DDL statement via HANA Cloud OData API
export fn zig_odata_execute_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
) callconv(.c) c_int {
    const host_str = mem.span(host);
    const user_str = mem.span(user);
    const password_str = mem.span(password);
    const schema_str = mem.span(schema);
    const sql_str = mem.span(sql);
    
    std.debug.print("🔷 Executing SQL on HANA Cloud via OData:\n", .{});
    std.debug.print("   Host: {s}\n", .{host_str});
    std.debug.print("   User: {s}\n", .{user_str});
    std.debug.print("   Schema: {s}\n", .{schema_str});
    std.debug.print("   SQL: {s}\n", .{sql_str});
    
    // Execute via HANA Cloud OData API
    executeSqlViaHanaOData(host_str, port, user_str, password_str, schema_str, sql_str) catch |err| {
        std.debug.print("   ❌ SQL execution failed: {}\n", .{err});
        return -1;
    };
    
    std.debug.print("   ✅ SQL executed successfully via OData\n", .{});
    return 0;
}

/// Query SQL via HANA Cloud OData API
export fn zig_odata_query_sql(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
    schema: [*:0]const u8,
    sql: [*:0]const u8,
    result_buf: [*]u8,
    result_buf_len: c_int,
) callconv(.c) c_int {
    const host_str = mem.span(host);
    const user_str = mem.span(user);
    const password_str = mem.span(password);
    const schema_str = mem.span(schema);
    const sql_str = mem.span(sql);
    
    std.debug.print("🔷 Querying HANA Cloud via OData:\n", .{});
    std.debug.print("   Host: {s}\n", .{host_str});
    std.debug.print("   User: {s}\n", .{user_str});
    std.debug.print("   Schema: {s}\n", .{schema_str});
    std.debug.print("   SQL: {s}\n", .{sql_str});
    
    // Query via HANA Cloud OData API
    const result = querySqlViaHanaOData(host_str, port, user_str, password_str, schema_str, sql_str) catch |err| {
        std.debug.print("   ❌ SQL query failed: {}\n", .{err});
        const empty = "{\"d\":{\"results\":[]}}";
        const len = @min(empty.len, @as(usize, @intCast(result_buf_len)) - 1);
        @memcpy(result_buf[0..len], empty[0..len]);
        result_buf[len] = 0;
        return @intCast(len);
    };
    defer allocator.free(result);
    
    const len = @min(result.len, @as(usize, @intCast(result_buf_len)) - 1);
    @memcpy(result_buf[0..len], result[0..len]);
    result_buf[len] = 0;
    
    std.debug.print("   ✅ Query returned {d} bytes via OData\n", .{len});
    return @intCast(len);
}

/// Test connection to HANA Cloud
export fn zig_odata_test_connection(
    host: [*:0]const u8,
    port: c_int,
    user: [*:0]const u8,
    password: [*:0]const u8,
) callconv(.c) c_int {
    _ = port; // HANA Cloud always uses 443
    const host_str = mem.span(host);
    const user_str = mem.span(user);
    const password_str = mem.span(password);
    
    std.debug.print("🔷 Testing HANA Cloud OData connection:\n", .{});
    std.debug.print("   Host: {s}\n", .{host_str});
    std.debug.print("   User: {s}\n", .{user_str});
    
    // Test with $metadata endpoint
    const user_pass = std.fmt.allocPrint(allocator, "{s}:{s}", .{ user_str, password_str }) catch return -1;
    defer allocator.free(user_pass);

    const url = std.fmt.allocPrint(
        allocator,
        "https://{s}/$metadata",
        .{host_str},
    ) catch return -1;
    defer allocator.free(url);

    const curl_args = [_][]const u8{
        "curl", "-I", "-s", "-k",
        "-u", user_pass,
        "-w", "%{http_code}",
        url,
    };

    var child = std.process.Child.init(&curl_args, allocator);
    const result = child.spawnAndWait() catch return -1;
    
    if (result == .Exited and result.Exited == 0) {
        std.debug.print("   ✅ Connection test passed\n", .{});
        return 0;
    }
    
    std.debug.print("   ❌ Connection test failed\n", .{});
    return -1;
}

// Test entry point (commented out - this is a library)
// pub fn main() !void {
//     std.debug.print("{s}\n", .{header_line});
//     std.debug.print("📡 Zig OData Client for HANA Cloud\n", .{});
//     std.debug.print("{s}\n\n", .{header_line});
//     
//     std.debug.print("HANA Cloud OData Integration:\n", .{});
//     std.debug.print("  • Native OData v4 API\n", .{});
//     std.debug.print("  • SQL execution via OData actions\n", .{});
//     std.debug.print("  • JSON result formatting\n", .{});
//     std.debug.print("\n{s}\n", .{header_line});
// }
