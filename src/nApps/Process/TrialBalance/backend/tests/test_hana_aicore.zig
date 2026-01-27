//! ============================================================================
//! HANA and AI Core Integration Test
//! Verifies connectivity to SAP HANA Cloud and SAP AI Core
//! ============================================================================
//!
//! [CODE:file=test_hana_aicore.zig]
//! [CODE:module=tests]
//! [CODE:language=zig]
//!
//! Run with: zig build test-hana-aicore
//! Or directly: zig test src/tests/test_hana_aicore.zig

const std = @import("std");
const hana = @import("../integrations/src/hana_client.zig");

pub fn main() !void {
    const stdout = std.io.getStdOut().writer();
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    try stdout.print("\n" ++
        "╔══════════════════════════════════════════════════════════════════╗\n" ++
        "║       SAP HANA Cloud & AI Core Integration Test                 ║\n" ++
        "╚══════════════════════════════════════════════════════════════════╝\n\n", .{});
    
    // ========================================================================
    // Test 1: Environment Variables
    // ========================================================================
    try stdout.print("═══ Test 1: Environment Variables ═══\n", .{});
    
    const hana_host = std.process.getEnvVarOwned(allocator, "HANA_HOST") catch |err| {
        try stdout.print("❌ HANA_HOST not set: {}\n", .{err});
        return;
    };
    defer allocator.free(hana_host);
    try stdout.print("✅ HANA_HOST: {s}...{s}\n", .{ hana_host[0..@min(30, hana_host.len)], if (hana_host.len > 30) "..." else "" });
    
    const aicore_url = std.process.getEnvVarOwned(allocator, "AICORE_BASE_URL") catch |err| {
        try stdout.print("⚠️  AICORE_BASE_URL not set: {}\n", .{err});
        null;
    };
    if (aicore_url) |url| {
        defer allocator.free(url);
        try stdout.print("✅ AICORE_BASE_URL: {s}\n", .{url});
    }
    
    // ========================================================================
    // Test 2: HANA Connection
    // ========================================================================
    try stdout.print("\n═══ Test 2: HANA Cloud Connection ═══\n", .{});
    
    var hana_client = hana.HanaClient.initFromEnv(allocator) catch |err| {
        try stdout.print("❌ Failed to initialize HANA client: {}\n", .{err});
        return;
    };
    defer hana_client.deinit();
    
    try stdout.print("✅ HANA client initialized\n", .{});
    try stdout.print("   Host: {s}\n", .{hana_client.config.host});
    try stdout.print("   Port: {d}\n", .{hana_client.config.port});
    try stdout.print("   Database: {s}\n", .{if (hana_client.config.database.len > 0) hana_client.config.database else "(default)"});
    try stdout.print("   Schema: {s}\n", .{hana_client.config.schema});
    
    // Test connection
    try stdout.print("\n   Testing connection (SELECT 1 FROM DUMMY)...\n", .{});
    const connected = hana_client.testConnection() catch |err| {
        try stdout.print("❌ Connection test failed: {}\n", .{err});
        false;
    };
    
    if (connected) {
        try stdout.print("✅ HANA connection successful!\n", .{});
    } else {
        try stdout.print("❌ HANA connection failed (no error, but no result)\n", .{});
    }
    
    // ========================================================================
    // Test 3: HANA Query (if connected)
    // ========================================================================
    if (connected) {
        try stdout.print("\n═══ Test 3: HANA Table Query ═══\n", .{});
        
        // Try to query DUMMY table
        const sql = "SELECT CURRENT_TIMESTAMP AS NOW, CURRENT_USER AS USER_NAME FROM DUMMY";
        try stdout.print("   Executing: {s}\n", .{sql});
        
        const result = hana_client.executeQuery(sql) catch |err| {
            try stdout.print("❌ Query failed: {}\n", .{err});
            null;
        };
        
        if (result) |res| {
            defer @constCast(&res).deinit();
            try stdout.print("✅ Query returned {d} row(s)\n", .{res.row_count});
            for (res.rows) |row| {
                for (row.columns, 0..) |col, i| {
                    try stdout.print("   {s}: {s}\n", .{ col, row.values[i] });
                }
            }
        }
    }
    
    // ========================================================================
    // Test 4: AI Core Token (OAuth)
    // ========================================================================
    try stdout.print("\n═══ Test 4: AI Core Authentication ═══\n", .{});
    
    const aicore_client_id = std.process.getEnvVarOwned(allocator, "AICORE_CLIENT_ID") catch null;
    const aicore_client_secret = std.process.getEnvVarOwned(allocator, "AICORE_CLIENT_SECRET") catch null;
    const aicore_auth_url = std.process.getEnvVarOwned(allocator, "AICORE_AUTH_URL") catch null;
    
    defer if (aicore_client_id) |id| allocator.free(id);
    defer if (aicore_client_secret) |s| allocator.free(s);
    defer if (aicore_auth_url) |u| allocator.free(u);
    
    if (aicore_client_id != null and aicore_client_secret != null and aicore_auth_url != null) {
        try stdout.print("✅ AI Core credentials found\n", .{});
        try stdout.print("   Client ID: {s}...{s}\n", .{
            aicore_client_id.?[0..@min(20, aicore_client_id.?.len)],
            if (aicore_client_id.?.len > 20) "..." else ""
        });
        try stdout.print("   Auth URL: {s}\n", .{aicore_auth_url.?});
        
        // Try to get OAuth token
        try stdout.print("\n   Requesting OAuth token...\n", .{});
        
        var http_client = std.http.Client.init(allocator, .{});
        defer http_client.deinit();
        
        // Create Basic Auth header
        const creds = try std.fmt.allocPrint(allocator, "{s}:{s}", .{ aicore_client_id.?, aicore_client_secret.? });
        defer allocator.free(creds);
        
        const encoded_len = std.base64.standard.Encoder.calcSize(creds.len);
        const encoded = try allocator.alloc(u8, encoded_len);
        defer allocator.free(encoded);
        _ = std.base64.standard.Encoder.encode(encoded, creds);
        
        const auth_header = try std.fmt.allocPrint(allocator, "Basic {s}", .{encoded});
        defer allocator.free(auth_header);
        
        const uri = try std.Uri.parse(aicore_auth_url.?);
        var header_buffer: [8192]u8 = undefined;
        var request = http_client.open(.POST, uri, .{ .server_header_buffer = &header_buffer }) catch |err| {
            try stdout.print("❌ Failed to create request: {}\n", .{err});
            return;
        };
        defer request.deinit();
        
        request.transfer_encoding = .chunked;
        try request.headers.append("Content-Type", "application/x-www-form-urlencoded");
        try request.headers.append("Authorization", auth_header);
        
        request.send() catch |err| {
            try stdout.print("❌ Failed to send request: {}\n", .{err});
            return;
        };
        try request.writeAll("grant_type=client_credentials");
        try request.finish();
        request.wait() catch |err| {
            try stdout.print("❌ Request failed: {}\n", .{err});
            return;
        };
        
        if (request.response.status == .ok) {
            const body = request.reader().readAllAlloc(allocator, 10 * 1024) catch "";
            defer allocator.free(body);
            
            // Check for access_token in response
            if (std.mem.indexOf(u8, body, "access_token")) |_| {
                try stdout.print("✅ AI Core OAuth token received!\n", .{});
                try stdout.print("   Token type: Bearer\n", .{});
            } else {
                try stdout.print("⚠️  Response received but no access_token found\n", .{});
            }
        } else {
            try stdout.print("❌ OAuth request failed with status: {}\n", .{request.response.status});
        }
    } else {
        try stdout.print("⚠️  AI Core credentials not fully configured\n", .{});
    }
    
    // ========================================================================
    // Summary
    // ========================================================================
    try stdout.print("\n" ++
        "╔══════════════════════════════════════════════════════════════════╗\n" ++
        "║                        Test Summary                             ║\n" ++
        "╚══════════════════════════════════════════════════════════════════╝\n", .{});
    
    try stdout.print("HANA Cloud: {s}\n", .{if (connected) "✅ Connected" else "❌ Not Connected"});
    try stdout.print("AI Core:    {s}\n", .{if (aicore_auth_url != null) "✅ Configured" else "❌ Not Configured"});
}

test "HANA config from env" {
    // This test requires env vars to be set
    const allocator = std.testing.allocator;
    
    // Try to load config - will fail if env vars not set
    const config = hana.HanaConfig.fromEnv(allocator) catch |err| {
        std.debug.print("Config load failed (expected if env not set): {}\n", .{err});
        return;
    };
    
    try std.testing.expect(config.host.len > 0);
    try std.testing.expect(config.user.len > 0);
}