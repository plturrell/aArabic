// Zig SQL Client for SAP HANA Cloud
// Implements HANA Cloud SQL API (not OData - HANA Cloud uses different endpoints)
// Exports C ABI functions for database/prompt_history.zig

const std = @import("std");
const mem = std.mem;
const json = std.json;

const header_line = "================================================================================";

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
pub const allocator = gpa.allocator();

// ============================================================================
// HANA Cloud SQL Execution via SQL API
// ============================================================================
// HANA Cloud exposes SQL via:
//   1. /sql/v1 - Direct SQL API (requires JWT or Basic Auth)
//   2. /sap/bc/sql - SQL endpoint on BTP
//   3. Database Cockpit WebSocket API
//
// The /sap/opu/odata/sap/SQL_EXECUTION_SRV paths are for SAP on-premise (S/4HANA),
// NOT for HANA Cloud.
// ============================================================================

fn fetchOAuthToken(alloc: std.mem.Allocator) ?[]u8 {
    const auth_url = std.process.getEnvVarOwned(alloc, "AICORE_AUTH_URL") catch return null;
    const client_id = std.process.getEnvVarOwned(alloc, "AICORE_CLIENT_ID") catch {
        alloc.free(auth_url);
        return null;
    };
    const client_secret = std.process.getEnvVarOwned(alloc, "AICORE_CLIENT_SECRET") catch {
        alloc.free(auth_url);
        alloc.free(client_id);
        return null;
    };

    defer alloc.free(auth_url);
    defer alloc.free(client_id);
    defer alloc.free(client_secret);

    const payload = std.fmt.allocPrint(alloc, "grant_type=client_credentials&client_id={s}&client_secret={s}", .{ client_id, client_secret }) catch return null;
    defer alloc.free(payload);

    const args = [_][]const u8{
        "curl",
        "-s",
        "-X",
        "POST",
        "-H",
        "Content-Type: application/x-www-form-urlencoded",
        "-d",
        payload,
        auth_url,
    };

    var child: std.process.Child = undefined;
    child.allocator = alloc;
    child.argv = args[0..];
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Inherit;
    child.env_map = null;
    child.spawn() catch return null;
    
    const out = child.stdout.?.readToEndAlloc(alloc, 1024 * 1024) catch {
        return null;
    };
    _ = child.wait() catch {};

    var parsed = std.json.parseFromSlice(std.json.Value, alloc, out, .{}) catch {
        alloc.free(out);
        return null;
    };
    defer parsed.deinit();
    alloc.free(out);

    if (parsed.value == .object) {
        if (parsed.value.object.get("access_token")) |tok| {
            switch (tok) {
                .string => |s| return alloc.dupe(u8, s) catch null,
                else => {},
            }
        }
    }
    return null;
}

/// Execute SQL via HANA Cloud SQL API
/// Uses the correct HANA Cloud endpoint at /sql/v1
pub fn executeSqlViaHanaOData(
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

    // HANA Cloud SQL API payload format
    // SET SCHEMA before executing to ensure correct schema context
    const full_sql = try std.fmt.allocPrint(
        allocator,
        "SET SCHEMA {s}; {s}",
        .{ schema, escaped_sql },
    );
    defer allocator.free(full_sql);

    const escaped_full_sql = try escapeJsonString(full_sql);
    defer allocator.free(escaped_full_sql);

    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"sql\":\"{s}\"}}",
        .{escaped_full_sql},
    );
    defer allocator.free(payload);

    const user_pass = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ user, password });
    defer allocator.free(user_pass);

    const bearer = fetchOAuthToken(allocator);
    defer if (bearer) |b| allocator.free(b);

    // Try multiple HANA Cloud SQL endpoints in order of preference
    const endpoints = [_][]const u8{
        "/sql/v1",                    // Primary HANA Cloud SQL API
        "/sap/bc/sql",               // BTP SQL endpoint
        "/api/v1/sql",               // Alternative API path
    };

    for (endpoints) |endpoint| {
        const url = try std.fmt.allocPrint(
            allocator,
            "https://{s}{s}",
            .{ host, endpoint },
        );
        defer allocator.free(url);

        std.debug.print("   Trying endpoint: {s}\n", .{url});

        var args_builder = std.ArrayList([]const u8){};
        defer args_builder.deinit(allocator);

        try args_builder.appendSlice(allocator, &[_][]const u8{
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json",
        });

        if (bearer) |b| {
            const auth_header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{b});
            defer allocator.free(auth_header);
            try args_builder.appendSlice(allocator, &[_][]const u8{"-H", auth_header});
        } else {
            try args_builder.appendSlice(allocator, &[_][]const u8{"-u", user_pass});
        }

        try args_builder.appendSlice(allocator, &[_][]const u8{
            "-d",
            payload,
            "-k", // Accept self-signed certs for dev
            "-s", // Silent
            "-w",
            "\n%{http_code}", // Output HTTP status on new line
            "--connect-timeout",
            "10",
            url,
        });

        const curl_args = try args_builder.toOwnedSlice(allocator);
        defer allocator.free(curl_args);

        var child: std.process.Child = undefined;
        child.allocator = allocator;
        child.argv = curl_args;
        child.stdin_behavior = .Inherit;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;
        child.env_map = null;
        try child.spawn();

        const output = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);
        defer allocator.free(output);

        const result = try child.wait();

        // Parse HTTP status from last line
        if (output.len > 3) {
            const last_newline = mem.lastIndexOf(u8, output, "\n") orelse 0;
            const status_str = output[last_newline + 1 ..];

            if (mem.eql(u8, status_str, "200") or mem.eql(u8, status_str, "201") or mem.eql(u8, status_str, "204")) {
                std.debug.print("   ✅ Success on endpoint: {s}\n", .{endpoint});
                return;
            }

            // 401/403 means endpoint exists but auth failed
            if (mem.eql(u8, status_str, "401") or mem.eql(u8, status_str, "403")) {
                std.debug.print("   ⚠️  Auth failed (HTTP {s}) - check credentials\n", .{status_str});
                return error.AuthenticationFailed;
            }

            // 404 means try next endpoint
            if (mem.eql(u8, status_str, "404")) {
                std.debug.print("   ⚠️  Endpoint not found, trying next...\n", .{});
                continue;
            }

            std.debug.print("   HTTP {s}: {s}\n", .{ status_str, output[0..last_newline] });
        }

        if (result != .Exited or result.Exited != 0) {
            continue; // Try next endpoint
        }
    }

    // All endpoints failed - try hdbsql fallback
    return executeSqlViaHdbsql(host, user, password, schema, sql);
}

/// Fallback: Execute SQL via hdbsql CLI (if installed)
fn executeSqlViaHdbsql(
    host: []const u8,
    user: []const u8,
    password: []const u8,
    schema: []const u8,
    sql: []const u8,
) !void {
    std.debug.print("   Trying hdbsql CLI fallback...\n", .{});

    const hdbsql_args = [_][]const u8{
        "hdbsql",
        "-n",
        host,
        "-u",
        user,
        "-p",
        password,
        "-d",
        "SYSTEMDB",
        "-encrypt",
        "-sslValidateCertificate",
        "false",
        try std.fmt.allocPrint(allocator, "SET SCHEMA {s}; {s}", .{ schema, sql }),
    };

    var child: std.process.Child = undefined;
    child.allocator = allocator;
    child.argv = hdbsql_args[0..];
    child.stdin_behavior = .Inherit;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    child.env_map = null;
    const result = child.spawnAndWait() catch |err| {
        std.debug.print("   ❌ hdbsql not available: {}\n", .{err});
        return error.ODataExecutionFailed;
    };

    if (result == .Exited and result.Exited == 0) {
        std.debug.print("   ✅ Success via hdbsql\n", .{});
        return;
    }

    return error.ODataExecutionFailed;
}

/// Query SQL via HANA Cloud SQL API
pub fn querySqlViaHanaOData(
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

    // Prepend SET SCHEMA for correct context
    const full_sql = try std.fmt.allocPrint(
        allocator,
        "SET SCHEMA {s}; {s}",
        .{ schema, escaped_sql },
    );
    defer allocator.free(full_sql);

    const escaped_full_sql = try escapeJsonString(full_sql);
    defer allocator.free(escaped_full_sql);

    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"sql\":\"{s}\"}}",
        .{escaped_full_sql},
    );
    defer allocator.free(payload);

    const user_pass = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ user, password });
    defer allocator.free(user_pass);

    const bearer = fetchOAuthToken(allocator);
    defer if (bearer) |b| allocator.free(b);

    // Try HANA Cloud SQL endpoints
    const endpoints = [_][]const u8{
        "/sql/v1",
        "/sap/bc/sql",
        "/api/v1/sql",
    };

    for (endpoints) |endpoint| {
        const url = try std.fmt.allocPrint(
            allocator,
            "https://{s}{s}",
            .{ host, endpoint },
        );
        defer allocator.free(url);

        var args_builder = std.ArrayList([]const u8){};
        defer args_builder.deinit(allocator);

        try args_builder.appendSlice(allocator, &[_][]const u8{
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-Type: application/json",
            "-H",
            "Accept: application/json",
        });

        if (bearer) |b| {
            const auth_header = try std.fmt.allocPrint(allocator, "Authorization: Bearer {s}", .{b});
            defer allocator.free(auth_header);
            try args_builder.appendSlice(allocator, &[_][]const u8{"-H", auth_header});
        } else {
            try args_builder.appendSlice(allocator, &[_][]const u8{"-u", user_pass});
        }

        try args_builder.appendSlice(allocator, &[_][]const u8{
            "-d",
            payload,
            "-k",
            "-s",
            "--connect-timeout",
            "10",
            url,
        });

        const curl_args = try args_builder.toOwnedSlice(allocator);
        defer allocator.free(curl_args);

        var child: std.process.Child = undefined;
        child.allocator = allocator;
        child.argv = curl_args;
        child.stdin_behavior = .Inherit;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Inherit;
        child.env_map = null;
        try child.spawn();

        const output = try child.stdout.?.readToEndAlloc(allocator, 1024 * 1024);

        const result = try child.wait();
        if (result == .Exited and result.Exited == 0 and output.len > 0) {
            // Check if response looks like valid JSON
            if (output[0] == '{' or output[0] == '[') {
                return output;
            }
        }

        allocator.free(output);
    }

    return error.ODataQueryFailed;
}

/// Escape string for JSON
fn escapeJsonString(input: []const u8) ![]u8 {
    var result = std.ArrayList(u8){};
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

/// Initialize SQL client library
export fn zig_odata_init() callconv(.c) c_int {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("📡 Zig SQL Client for SAP HANA Cloud\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Supported endpoints (tried in order):\n", .{});
    std.debug.print("  1. /sql/v1        - HANA Cloud SQL API\n", .{});
    std.debug.print("  2. /sap/bc/sql    - BTP SQL endpoint\n", .{});
    std.debug.print("  3. /api/v1/sql    - Alternative API\n", .{});
    std.debug.print("  4. hdbsql CLI     - Fallback if HTTP fails\n", .{});
    std.debug.print("\nAuthentication:\n", .{});
    std.debug.print("  • Basic Auth (user:password)\n", .{});
    std.debug.print("  • Note: Some endpoints may require JWT/XSUAA\n", .{});
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

    std.debug.print("🔷 Testing HANA Cloud connection:\n", .{});
    std.debug.print("   Host: {s}\n", .{host_str});
    std.debug.print("   User: {s}\n", .{user_str});

    const user_pass = std.fmt.allocPrint(allocator, "{s}:{s}", .{ user_str, password_str }) catch return -1;
    defer allocator.free(user_pass);

    // Test various HANA Cloud endpoints to find working one
    const test_endpoints = [_][]const u8{
        "/sql/v1",           // HANA Cloud SQL API
        "/sap/bc/sql",       // BTP SQL endpoint
        "/",                  // Root (basic connectivity)
    };

    for (test_endpoints) |endpoint| {
        const url = std.fmt.allocPrint(
            allocator,
            "https://{s}{s}",
            .{ host_str, endpoint },
        ) catch continue;
        defer allocator.free(url);

        std.debug.print("   Testing: {s}\n", .{url});

        const curl_args = [_][]const u8{
            "curl",
            "-I",
            "-s",
            "-k",
            "-u",
            user_pass,
            "-w",
            "\n%{http_code}",
            "--connect-timeout",
            "10",
            url,
        };

        var child: std.process.Child = undefined;
        child.allocator = allocator;
        child.argv = curl_args[0..];
        child.stdin_behavior = .Inherit;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Inherit;
        child.env_map = null;
        child.spawn() catch continue;

        const output = child.stdout.?.readToEndAlloc(allocator, 8192) catch continue;
        defer allocator.free(output);

        const result = child.wait() catch continue;

        if (result == .Exited and result.Exited == 0 and output.len >= 3) {
            // Get HTTP status from last line
            const last_newline = mem.lastIndexOf(u8, output, "\n") orelse 0;
            const status = output[last_newline + 1 ..];

            std.debug.print("   {s} -> HTTP {s}\n", .{ endpoint, status });

            // 200, 401, 403 all mean endpoint exists
            if (mem.eql(u8, status, "200") or mem.eql(u8, status, "401") or mem.eql(u8, status, "403")) {
                std.debug.print("   ✅ Endpoint reachable: {s}\n", .{endpoint});
                if (mem.eql(u8, status, "401") or mem.eql(u8, status, "403")) {
                    std.debug.print("   ⚠️  Auth required - check credentials\n", .{});
                }
                return 0;
            }
        }
    }

    std.debug.print("   ❌ No working endpoint found\n", .{});
    std.debug.print("   ℹ️  HANA Cloud may require:\n", .{});
    std.debug.print("      - JWT token from XSUAA instead of Basic Auth\n", .{});
    std.debug.print("      - SAP Passport/API Key\n", .{});
    std.debug.print("      - Different SQL API endpoint\n", .{});
    return -1;
}

/// Diagnostic: List available endpoints on HANA Cloud host
export fn zig_odata_diagnose(
    host: [*:0]const u8,
    user: [*:0]const u8,
    password: [*:0]const u8,
) callconv(.c) c_int {
    const host_str = mem.span(host);
    const user_str = mem.span(user);
    const password_str = mem.span(password);

    std.debug.print("\n{s}\n", .{header_line});
    std.debug.print("🔍 HANA Cloud Endpoint Diagnostics\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Host: {s}\n\n", .{host_str});

    const user_pass = std.fmt.allocPrint(allocator, "{s}:{s}", .{ user_str, password_str }) catch return -1;
    defer allocator.free(user_pass);

    // Test all potential endpoints
    const all_endpoints = [_][]const u8{
        "/",
        "/sql/v1",
        "/sap/bc/sql",
        "/api/v1/sql",
        "/sap/opu/odata/sap/",
        "/odata/v4",
        "/$metadata",
    };

    for (all_endpoints) |endpoint| {
        const url = std.fmt.allocPrint(
            allocator,
            "https://{s}{s}",
            .{ host_str, endpoint },
        ) catch continue;
        defer allocator.free(url);

        const curl_args = [_][]const u8{
            "curl",
            "-I",
            "-s",
            "-k",
            "-u",
            user_pass,
            "-w",
            "%{http_code}",
            "--connect-timeout",
            "5",
            "-o",
            "/dev/null",
            url,
        };

        var child: std.process.Child = undefined;
        child.allocator = allocator;
        child.argv = curl_args[0..];
        child.stdin_behavior = .Inherit;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Inherit;
        child.env_map = null;
        child.spawn() catch {
            std.debug.print("  {s: <30} -> FAILED (spawn)\n", .{endpoint});
            continue;
        };

        const output = child.stdout.?.readToEndAlloc(allocator, 256) catch {
            std.debug.print("  {s: <30} -> FAILED (read)\n", .{endpoint});
            continue;
        };
        defer allocator.free(output);

        _ = child.wait() catch continue;

        const status = if (output.len >= 3) output[0..3] else "???";
        const icon = if (mem.eql(u8, status, "200"))
            "✅"
        else if (mem.eql(u8, status, "401") or mem.eql(u8, status, "403"))
            "🔐"
        else if (mem.eql(u8, status, "404"))
            "❌"
        else
            "⚠️";

        std.debug.print("  {s} {s: <30} -> HTTP {s}\n", .{ icon, endpoint, status });
    }

    std.debug.print("\n{s}\n", .{header_line});
    std.debug.print("Legend: ✅=OK  🔐=Auth Required  ❌=Not Found  ⚠️=Other\n", .{});
    std.debug.print("{s}\n\n", .{header_line});

    return 0;
}

// Test entry point (commented out - this is a library)
// pub fn main() !void {
//     std.debug.print("{s}\n", .{header_line});
//     std.debug.print("� Zig OData Client for HANA Cloud\n", .{});
//     std.debug.print("{s}\n\n", .{header_line});
//     
//     std.debug.print("HANA Cloud OData Integration:\n", .{});
//     std.debug.print("  • Native OData v4 API\n", .{});
//     std.debug.print("  • SQL execution via OData actions\n", .{});
//     std.debug.print("  • JSON result formatting\n", .{});
//     std.debug.print("\n{s}\n", .{header_line});
// }
