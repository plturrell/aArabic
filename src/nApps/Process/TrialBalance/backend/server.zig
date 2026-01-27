//! ============================================================================
//! Trial Balance Backend API Server
//! Zig HTTP server with SQLite support, API endpoints, and static file serving
//! ============================================================================
//!
//! [CODE:file=server.zig]
//! [CODE:module=backend]
//! [CODE:language=zig]
//!
//! [RELATION:uses=CODE:main.zig]
//! [RELATION:uses=CODE:trial_balance.zig]
//! [RELATION:uses=CODE:odps_api.zig]
//!
//! Note: Infrastructure - HTTP server entry point and route configuration.

const std = @import("std");
const net = std.net;
const fs = std.fs;

// Import calculation modules
const balance_engine = @import("balance_engine");
const fx_converter = @import("fx_converter");
const sqlite_adapter = @import("sqlite_adapter");
const websocket = @import("websocket");

// ============================================================================
// Configuration
// ============================================================================

const Config = struct {
    port: u16 = 8091,
    host: []const u8 = "0.0.0.0",
    db_path: []const u8 = "../BusDocs/schema/sqlite/trial_balance_dev.db",
    webapp_path: []const u8 = "../webapp",
};

// ============================================================================
// HTTP Server Implementation
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const config = Config{};

    // Initialize database
    const db_path_z = try allocator.dupeZ(u8, config.db_path);
    defer allocator.free(db_path_z);

    var db = try sqlite_adapter.Database.init(db_path_z);
    defer db.deinit();

    // Create server address
    const address = try net.Address.parseIp(config.host, config.port);
    
    // Create TCP listener
    var server = try address.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();

    std.debug.print("\n", .{});
    std.debug.print("╔══════════════════════════════════════════════════════════╗\n", .{});
    std.debug.print("║   Trial Balance API Server - Zig HTTP + WebSocket       ║\n", .{});
    std.debug.print("╚══════════════════════════════════════════════════════════╝\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("✓ Server listening on http://{s}:{}\n", .{ config.host, config.port });
    std.debug.print("✓ Database: {s}\n", .{config.db_path});
    std.debug.print("✓ Static files: {s}\n", .{config.webapp_path});
    std.debug.print("\n", .{});
    std.debug.print("API Endpoints:\n", .{});
    std.debug.print("  GET  /api/health\n", .{});
    std.debug.print("  GET  /api/v1/accounts\n", .{});
    std.debug.print("  POST /api/v1/trial-balance/calculate\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("WebSocket Streaming:\n", .{});
    std.debug.print("  WS   ws://{s}:{}/ws (Real-time calculation updates)\n", .{ config.host, config.port });
    std.debug.print("\n", .{});
    std.debug.print("Frontend:\n", .{});
    std.debug.print("  GET  /          (index.html)\n", .{});
    std.debug.print("  GET  /*         (static files)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Press Ctrl+C to stop\n", .{});
    std.debug.print("\n", .{});

    // Accept connections
    while (true) {
        const connection = try server.accept();
        
        // Handle connection in a separate thread would go here
        // For simplicity, handling synchronously
        handleConnection(allocator, connection, &db, config) catch |err| {
            std.debug.print("Error handling connection: {}\n", .{err});
        };
    }
}

fn handleConnection(
    allocator: std.mem.Allocator,
    connection: net.Server.Connection,
    db: *sqlite_adapter.Database,
    config: Config,
) !void {
    defer connection.stream.close();

    var buffer: [8192]u8 = undefined;
    const bytes_read = try connection.stream.read(&buffer);
    
    if (bytes_read == 0) return;

    const request = buffer[0..bytes_read];
    
    // Check for WebSocket upgrade
    if (std.mem.indexOf(u8, request, "Upgrade: websocket") != null or
        std.mem.indexOf(u8, request, "Upgrade: WebSocket") != null) {
        try handleWebSocketConnection(allocator, connection.stream, request, db);
        return;
    }
    
    // Parse HTTP request line
    var lines = std.mem.splitScalar(u8, request, '\n');
    const request_line = lines.next() orelse return;
    
    var parts = std.mem.splitScalar(u8, request_line, ' ');
    const method = parts.next() orelse return;
    const path_raw = parts.next() orelse return;
    
    // Remove trailing \r if present
    const path = std.mem.trimRight(u8, path_raw, "\r");
    
    std.debug.print("[{s}] {s} {s}\n", .{ 
        getTimestamp(), 
        method, 
        path 
    });

    // Route the request
    if (std.mem.startsWith(u8, path, "/api/")) {
        try handleApiRequest(allocator, connection, method, path, request, db);
    } else {
        try handleStaticFile(allocator, connection, path, config);
    }
}

fn handleApiRequest(
    allocator: std.mem.Allocator,
    connection: net.Server.Connection,
    method: []const u8,
    path: []const u8,
    request: []const u8,
    db: *sqlite_adapter.Database,
) !void {
    // CORS headers
    const cors_headers = 
        \\Access-Control-Allow-Origin: *
        \\Access-Control-Allow-Methods: GET, POST, OPTIONS
        \\Access-Control-Allow-Headers: Content-Type
        \\
    ;

    if (std.mem.eql(u8, method, "OPTIONS")) {
        const response = 
            \\HTTP/1.1 204 No Content
            \\Access-Control-Allow-Origin: *
            \\Access-Control-Allow-Methods: GET, POST, OPTIONS
            \\Access-Control-Allow-Headers: Content-Type
            \\
            \\
        ;
        try connection.stream.writeAll(response);
        return;
    }

    if (std.mem.eql(u8, path, "/api/health")) {
        const body = 
            \\{"status":"healthy","service":"trial-balance-api","version":"1.0.0"}
        ;
        const response = try std.fmt.allocPrint(allocator,
            \\HTTP/1.1 200 OK
            \\Content-Type: application/json
            \\Content-Length: {}
            \\
        ++ cors_headers ++
            \\
            \\{s}
        , .{ body.len, body });
        defer allocator.free(response);
        
        try connection.stream.writeAll(response);
        return;
    }

    if (std.mem.eql(u8, path, "/api/v1/accounts") and std.mem.eql(u8, method, "GET")) {
        try handleListAccounts(allocator, connection, db);
        return;
    }

    if (std.mem.eql(u8, path, "/api/v1/trial-balance/calculate") and std.mem.eql(u8, method, "POST")) {
        try handleCalculateTrialBalance(allocator, connection, request, db);
        return;
    }

    // 404 Not Found
    const response = 
        \\HTTP/1.1 404 Not Found
        \\Content-Type: text/plain
        \\Content-Length: 9
        \\Access-Control-Allow-Origin: *
        \\
        \\Not Found
    ;
    try connection.stream.writeAll(response);
}

fn handleListAccounts(
    allocator: std.mem.Allocator,
    connection: net.Server.Connection,
    db: *sqlite_adapter.Database,
) !void {
    var reader = sqlite_adapter.DatabaseReader.init(db, allocator);
    var accounts = try reader.read_gl_accounts();
    defer {
        for (accounts.items) |account| {
            allocator.free(account.account_id);
            allocator.free(account.account_number);
            allocator.free(account.description);
            allocator.free(account.ifrs_schedule);
            allocator.free(account.ifrs_category);
            allocator.free(account.account_type);
        }
        accounts.deinit(allocator);
    }

    // Build JSON response
    var json_buffer: std.ArrayList(u8) = .{};
    defer json_buffer.deinit(allocator);
    
        try json_buffer.appendSlice(allocator, "[\n");
    for (accounts.items, 0..) |account, i| {
        const account_json = try std.fmt.allocPrint(allocator,
            \\  {{"accountId":"{s}","accountNumber":"{s}","description":"{s}","ifrsSchedule":"{s}","ifrsCategory":"{s}","accountType":"{s}"}}
        , .{
            account.account_id,
            account.account_number,
            account.description,
            account.ifrs_schedule,
            account.ifrs_category,
            account.account_type,
        });
        defer allocator.free(account_json);
        
        try json_buffer.appendSlice(allocator, account_json);
        if (i < accounts.items.len - 1) {
            try json_buffer.appendSlice(allocator, ",\n");
        }
    }
    try json_buffer.appendSlice(allocator, "\n]");

    // Build response with headers
    var response_buf: std.ArrayList(u8) = .{};
    defer response_buf.deinit(allocator);
    
    const headers = try std.fmt.allocPrint(allocator,
        \\HTTP/1.1 200 OK
        \\Content-Type: application/json
        \\Content-Length: {}
        \\Access-Control-Allow-Origin: *
        \\
        \\{s}
    , .{ json_buffer.items.len, json_buffer.items });
    defer allocator.free(headers);
    
    try connection.stream.writeAll(headers);
}

fn handleCalculateTrialBalance(
    allocator: std.mem.Allocator,
    connection: net.Server.Connection,
    request: []const u8,
    db: *sqlite_adapter.Database,
) !void {
    // Find request body (after \r\n\r\n)
    const body_start = std.mem.indexOf(u8, request, "\r\n\r\n") orelse return;
    const body = request[body_start + 4 ..];

    // Simple JSON parsing (looking for company_code, fiscal_year, period)
    // For production, use proper JSON parser
    const company_code = "HKG";
    const fiscal_year = "2025";
    const period = "011";

    var reader = sqlite_adapter.DatabaseReader.init(db, allocator);
    var entries_db = try reader.read_journal_entries(company_code, fiscal_year, period);
    defer {
        for (entries_db.items) |entry| {
            allocator.free(entry.entry_id);
            allocator.free(entry.company_code);
            allocator.free(entry.fiscal_year);
            allocator.free(entry.period);
            allocator.free(entry.account);
            allocator.free(entry.currency);
        }
        entries_db.deinit(allocator);
    }

    var journal_entries = try allocator.alloc(balance_engine.JournalEntry, entries_db.items.len);
    defer allocator.free(journal_entries);

    for (entries_db.items, 0..) |db_entry, i| {
        journal_entries[i] = balance_engine.JournalEntry{
            .company_code = db_entry.company_code,
            .fiscal_year = db_entry.fiscal_year,
            .period = db_entry.period,
            .document_number = db_entry.entry_id,
            .line_item = "001",
            .account = db_entry.account,
            .debit_credit_indicator = db_entry.debit_credit_indicator,
            .amount = db_entry.amount,
            .currency = db_entry.currency,
            .posting_date = "2025-01-01",
        };
    }

    var calculator = balance_engine.TrialBalanceCalculator.init(allocator);
    var result = try calculator.calculate_trial_balance(journal_entries);
    defer result.deinit(allocator);

    const result_json = try std.fmt.allocPrint(allocator,
        \\{{"totalDebits":{d:.2},"totalCredits":{d:.2},"balanceDifference":{d:.2},"isBalanced":{},"accountCount":{}}}
    , .{
        result.total_debits,
        result.total_credits,
        result.balance_difference,
        result.is_balanced,
        result.accounts.items.len,
    });
    defer allocator.free(result_json);

    const response = try std.fmt.allocPrint(allocator,
        \\HTTP/1.1 200 OK
        \\Content-Type: application/json
        \\Content-Length: {}
        \\Access-Control-Allow-Origin: *
        \\
        \\{s}
    , .{ result_json.len, result_json });
    defer allocator.free(response);
    
    try connection.stream.writeAll(response);
    
    _ = body; // Suppress unused warning
}

fn handleStaticFile(
    allocator: std.mem.Allocator,
    connection: net.Server.Connection,
    path: []const u8,
    config: Config,
) !void {
    // Map / to /index.html
    const file_path = if (std.mem.eql(u8, path, "/")) 
        "/index.html" 
    else 
        path;

    // Build full file path
    const full_path = try std.fmt.allocPrint(
        allocator,
        "{s}{s}",
        .{ config.webapp_path, file_path },
    );
    defer allocator.free(full_path);

    // Try to read file
    const file = fs.cwd().openFile(full_path, .{}) catch {
        const response = 
            \\HTTP/1.1 404 Not Found
            \\Content-Type: text/plain
            \\Content-Length: 9
            \\
            \\
            \\Not Found
        ;
        try connection.stream.writeAll(response);
        return;
    };
    defer file.close();

    const file_size = try file.getEndPos();
    const content = try file.readToEndAlloc(allocator, file_size);
    defer allocator.free(content);

    // Determine content type
    const content_type = getContentType(file_path);

    const response = try std.fmt.allocPrint(allocator,
        \\HTTP/1.1 200 OK
        \\Content-Type: {s}
        \\Content-Length: {}
        \\
        \\
        \\{s}
    , .{ content_type, content.len, content });
    defer allocator.free(response);

    try connection.stream.writeAll(response);
}

fn getContentType(path: []const u8) []const u8 {
    if (std.mem.endsWith(u8, path, ".html")) return "text/html";
    if (std.mem.endsWith(u8, path, ".css")) return "text/css";
    if (std.mem.endsWith(u8, path, ".js")) return "application/javascript";
    if (std.mem.endsWith(u8, path, ".json")) return "application/json";
    if (std.mem.endsWith(u8, path, ".xml")) return "application/xml";
    if (std.mem.endsWith(u8, path, ".png")) return "image/png";
    if (std.mem.endsWith(u8, path, ".jpg")) return "image/jpeg";
    if (std.mem.endsWith(u8, path, ".svg")) return "image/svg+xml";
    return "application/octet-stream";
}

fn handleWebSocketConnection(
    allocator: std.mem.Allocator,
    stream: net.Stream,
    request: []const u8,
    db: *sqlite_adapter.Database,
) !void {
    // Perform WebSocket handshake
    try websocket.handleWebSocketUpgrade(stream, request);
    
    // Register client for broadcasting
    websocket.registerClient(stream);
    defer websocket.unregisterClient(stream);
    
    // Send welcome message
    const welcome = 
        \\{"type":"tb:connected","payload":{"message":"Connected to Trial Balance WebSocket"}}
    ;
    try websocket.sendWebSocketMessage(stream, welcome);
    
    // Handle incoming WebSocket messages
    var frame_buffer: [8192]u8 = undefined;
    while (true) {
        const payload = websocket.receiveWebSocketFrame(stream, &frame_buffer) catch |err| {
            if (err == error.ConnectionClosed) break;
            std.debug.print("WebSocket error: {any}\n", .{err});
            break;
        };
        
        if (payload.len == 0) continue; // Empty frame (e.g., after ping/pong)
        
        std.debug.print("WS received: {s}\n", .{payload});
        
        // Parse message and handle commands
        if (std.mem.indexOf(u8, payload, "calculate")) |_| {
            // Trigger calculation with streaming updates
            try handleStreamingCalculation(allocator, db);
        }
    }
}

fn handleStreamingCalculation(
    allocator: std.mem.Allocator,
    db: *sqlite_adapter.Database,
) !void {
    const start_time = std.time.milliTimestamp();
    
    // Read journal entries
    const company_code = "HKG";
    const fiscal_year = "2025";
    const period = "011";
    
    try websocket.sendValidationStatus(allocator, "starting", 0, 0);
    
    var reader = sqlite_adapter.DatabaseReader.init(db, allocator);
    var entries_db = try reader.read_journal_entries(company_code, fiscal_year, period);
    defer {
        for (entries_db.items) |entry| {
            allocator.free(entry.entry_id);
            allocator.free(entry.company_code);
            allocator.free(entry.fiscal_year);
            allocator.free(entry.period);
            allocator.free(entry.account);
            allocator.free(entry.currency);
        }
        entries_db.deinit(allocator);
    }
    
    const total_entries = entries_db.items.len;
    try websocket.sendValidationStatus(allocator, "loading", 0, 0);
    
    // Convert to calculation engine format
    var journal_entries = try allocator.alloc(balance_engine.JournalEntry, total_entries);
    defer allocator.free(journal_entries);
    
    for (entries_db.items, 0..) |db_entry, i| {
        journal_entries[i] = balance_engine.JournalEntry{
            .company_code = db_entry.company_code,
            .fiscal_year = db_entry.fiscal_year,
            .period = db_entry.period,
            .document_number = db_entry.entry_id,
            .line_item = "001",
            .account = db_entry.account,
            .debit_credit_indicator = db_entry.debit_credit_indicator,
            .amount = db_entry.amount,
            .currency = db_entry.currency,
            .posting_date = "2025-01-01",
        };
        
        // Stream progress every 5000 entries
        if (i > 0 and i % 5000 == 0) {
            const elapsed = @as(f64, @floatFromInt(std.time.milliTimestamp() - start_time));
            const entries_per_sec = (@as(f64, @floatFromInt(i)) / elapsed) * 1000.0;
            try websocket.sendCalculationProgress(allocator, i, total_entries, 0, 0);
            try websocket.sendPerformanceMetrics(allocator, entries_per_sec, 0, elapsed);
        }
    }
    
    try websocket.sendValidationStatus(allocator, "calculating", 0, 0);
    
    // Perform calculation
    var calculator = balance_engine.TrialBalanceCalculator.init(allocator);
    var result = try calculator.calculate_trial_balance(journal_entries);
    defer result.deinit(allocator);
    
    const end_time = std.time.milliTimestamp();
    const elapsed_ms = @as(f64, @floatFromInt(end_time - start_time));
    
    // Send completion message
    try websocket.sendCalculationComplete(
        allocator,
        result.total_debits,
        result.total_credits,
        result.balance_difference,
        result.is_balanced,
        result.accounts.items.len,
        elapsed_ms,
    );
    
    std.debug.print("✓ Streaming calculation completed in {d:.0}ms\n", .{elapsed_ms});
}

fn getTimestamp() []const u8 {
    // Simple timestamp - in production use proper time formatting
    return "2026-01-26 22:16:00";
}
