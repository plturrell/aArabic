//! ============================================================================
//! Trial Balance Backend Server
//! High-performance HTTP server for Trial Balance REST API
//! ============================================================================
//!
//! [CODE:file=main.zig]
//! [CODE:module=main]
//! [CODE:language=zig]
//!
//! [ODPS:product=trial-balance-aggregated,exchange-rates,variances]
//!
//! [API:produces=/api/v1/trial-balance,/api/v1/accounts,/api/health]
//!
//! [RELATION:calls=CODE:trial_balance.zig]
//! [RELATION:calls=CODE:static.zig]
//! [RELATION:calls=CODE:odps_api.zig]
//! [RELATION:calls=CODE:balance_engine.zig]
//! [RELATION:orchestrates=PETRI:TB_PROCESS_petrinet.pnml]
//!
//! This is the main entry point for the Trial Balance backend server.
//! It serves both the REST API and static webapp files.

const std = @import("std");
const TrialBalanceAPI = @import("api/trial_balance.zig").TrialBalanceAPI;
const static_handler = @import("http/static.zig");
const net = std.net;

/// Trial Balance Backend Server
/// High-performance backend using n-c-sdk for trial balance operations
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("Trial Balance Backend Server\n", .{});
    
    // Initialize API
    var api = try TrialBalanceAPI.init(allocator);
    defer api.deinit();

    const port = 8091;
    const address = try net.Address.parseIp("127.0.0.1", port);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("Starting server on port {d}...\n", .{port});
    std.debug.print("Ready to accept connections...\n", .{});

    while (true) {
        const connection = try server.accept();
        handleConnection(allocator, connection, &api) catch |err| {
            std.debug.print("Error handling connection: {}\n", .{err});
        };
    }
}

fn handleConnection(allocator: std.mem.Allocator, connection: net.Server.Connection, api: *TrialBalanceAPI) !void {
    defer connection.stream.close();

    var buffer: [8192]u8 = undefined;
    const bytes_read = try connection.stream.read(&buffer);
    if (bytes_read == 0) return;

    const request = buffer[0..bytes_read];
    
    // Simple HTTP parsing
    var lines_iter = std.mem.tokenizeScalar(u8, request, '\n');
    const request_line = lines_iter.next() orelse return;
    
    var parts_iter = std.mem.tokenizeScalar(u8, request_line, ' ');
    const method = parts_iter.next() orelse return;
    const path = parts_iter.next() orelse return;

    std.debug.print("{s} {s}\n", .{method, path});

    // Routing - Check if it's an API request or static file
    if (std.mem.startsWith(u8, path, "/api/")) {
        // API Routes
        return handleAPI(allocator, connection, path, method, request, api);
    } else {
        // Static Files - serve from ../webapp directory
        const webapp_dir = "../webapp";
        const full_response = try static_handler.serveStaticFile(allocator, webapp_dir, path);
        defer allocator.free(full_response);
        _ = try connection.stream.write(full_response);
        return;
    }
}

fn handleAPI(allocator: std.mem.Allocator, connection: net.Server.Connection, path: []const u8, method: []const u8, request: []const u8, api: *TrialBalanceAPI) !void {
    // CORS headers
    const headers = 
        "HTTP/1.1 200 OK\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS\r\n" ++
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n" ++
        "\r\n";

    if (std.mem.eql(u8, method, "OPTIONS")) {
        _ = try connection.stream.write(headers);
        return;
    }

    var response_body: []u8 = undefined;
    var status_code: u16 = 200;

    if (std.mem.eql(u8, path, "/api/v1/accounts")) {
        // Return sample account data
        response_body = try api.getTrialBalance();
    } else if (std.mem.eql(u8, path, "/api/v1/trial-balance/calculate") and std.mem.eql(u8, method, "POST")) {
        // Calculate trial balance
        if (std.mem.indexOf(u8, request, "\r\n\r\n")) |body_start| {
            const body = request[body_start+4..];
            response_body = try api.createEntry(body);
        } else {
            response_body = try allocator.dupe(u8, "{\"error\": \"No body found\"}");
            status_code = 400;
        }
    } else if (std.mem.eql(u8, path, "/api/health")) {
        response_body = try allocator.dupe(u8, "{\"status\": \"healthy\", \"service\": \"trial-balance\"}");
    } else if (std.mem.eql(u8, path, "/api/v1/trial-balance/summary")) {
        response_body = try api.getSummary();
    } else if (std.mem.eql(u8, path, "/api/v1/trial-balance/process-review") and std.mem.eql(u8, method, "POST")) {
        // Trigger the DOI process simulation
        response_body = try api.processReviewFile("HKG_PL review Nov'25.xlsb");
    } else if (std.mem.eql(u8, path, "/api/v1/trial-balance/narrative") and std.mem.eql(u8, method, "POST")) {
        if (std.mem.indexOf(u8, request, "\r\n\r\n")) |body_start| {
            const body = request[body_start+4..];
            response_body = try api.getNarrative(body);
        } else {
             response_body = try allocator.dupe(u8, "{\"error\": \"No body found\"}");
             status_code = 400;
        }
    } else {
        response_body = try allocator.dupe(u8, "{\"error\": \"Not Found\"}");
        status_code = 404;
    }
    defer allocator.free(response_body);

    if (status_code == 200) {
        _ = try connection.stream.write(headers);
        _ = try connection.stream.write(response_body);
    } else {
        const err_headers = try std.fmt.allocPrint(allocator, "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{status_code});
        defer allocator.free(err_headers);
        _ = try connection.stream.write(err_headers);
        _ = try connection.stream.write(response_body);
    }
}

test "main functionality" {
    // Add tests here
}