const std = @import("std");
const net = std.net;
const mem = std.mem;
const upload = @import("upload.zig");
const odata_chat = @import("odata_chat.zig");
const odata_summary = @import("odata_summary.zig");

// ============================================================================
// HyperShimmy HTTP + OData V4 Server
// ============================================================================
//
// Day 2 Implementation: Basic HTTP server with OData service root
// Day 16 Implementation: File upload endpoint
// Day 28 Implementation: OData Chat action
// Day 32 Implementation: OData Summary action
// 
// Features:
// - HTTP/1.1 server on port 11434
// - Basic request routing
// - Health check endpoint
// - OData service root endpoint
// - File upload endpoint (POST /api/upload)
// - OData Chat action (POST /odata/v4/research/Chat)
// - OData Summary action (POST /odata/v4/research/GenerateSummary)
// - JSON response formatting
//
// Based on working zig_http_shimmy.zig pattern
// ============================================================================

const ServerConfig = struct {
    host: []const u8,
    port: u16,
};

const default_config = ServerConfig{
    .host = "0.0.0.0",
    .port = 11434,
};

/// Handle a single connection
fn handleConnection(conn: net.Server.Connection, allocator: mem.Allocator) !void {
    defer conn.stream.close();
    
    // Use larger buffer for file uploads
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(allocator);
    
    // Read request in chunks
    var chunk: [4096]u8 = undefined;
    while (true) {
        const bytes_read = try conn.stream.read(&chunk);
        if (bytes_read == 0) break;
        try buffer.appendSlice(allocator, chunk[0..bytes_read]);
        
        // Check if we've read all headers (look for \r\n\r\n)
        if (mem.indexOf(u8, buffer.items, "\r\n\r\n")) |headers_end| {
            // Check Content-Length header
            const headers = buffer.items[0..headers_end];
            if (mem.indexOf(u8, headers, "Content-Length:")) |cl_start| {
                const cl_line_start = cl_start + 15; // "Content-Length:".len
                const cl_line_end = mem.indexOf(u8, headers[cl_line_start..], "\r\n") orelse headers.len - cl_line_start;
                const cl_str = mem.trim(u8, headers[cl_line_start .. cl_line_start + cl_line_end], " \t");
                const content_length = try std.fmt.parseInt(usize, cl_str, 10);
                
                // Read remaining body if needed
                const body_start = headers_end + 4;
                const current_body_len = buffer.items.len - body_start;
                const remaining = content_length - current_body_len;
                
                if (remaining > 0) {
                    var remaining_to_read = remaining;
                    while (remaining_to_read > 0) {
                        const read_size = @min(chunk.len, remaining_to_read);
                        const read = try conn.stream.read(chunk[0..read_size]);
                        if (read == 0) break;
                        try buffer.appendSlice(allocator, chunk[0..read]);
                        remaining_to_read -= read;
                    }
                }
                break;
            } else {
                // No Content-Length, assume we have all data
                break;
            }
        }
        
        // Safety check to prevent infinite reading
        if (buffer.items.len > 100 * 1024 * 1024) { // 100MB limit
            return error.RequestTooLarge;
        }
    }
    
    if (buffer.items.len == 0) return;
    
    const request_data = buffer.items;
    
    // Parse HTTP request
    var lines = mem.splitSequence(u8, request_data, "\r\n");
    const first_line = lines.next() orelse return error.InvalidRequest;
    
    var parts = mem.splitSequence(u8, first_line, " ");
    const method = parts.next() orelse return error.InvalidMethod;
    const path = parts.next() orelse return error.InvalidPath;
    
    // Extract headers and body
    const headers_body_split = mem.indexOf(u8, request_data, "\r\n\r\n") orelse request_data.len;
    const headers = request_data[0..headers_body_split];
    const body = if (headers_body_split + 4 < request_data.len)
        request_data[headers_body_split + 4..]
    else
        &[_]u8{};
    
    // Log request
    std.debug.print("[{s}] {s} ({d} bytes)\n", .{ method, path, body.len });
    
    // Route and generate response
    const response_body = try routeRequest(allocator, method, path, headers, body);
    defer if (response_body.len > 0) allocator.free(response_body);
    
    // Determine status and content type
    const status = if (mem.startsWith(u8, response_body, "{\"error\""))
        "404 Not Found"
    else
        "200 OK";
    
    const content_type = if (mem.eql(u8, path, "/odata/v4/research/$metadata"))
        "application/xml"
    else
        "application/json";
    
    // Send HTTP response
    const http_response = try std.fmt.allocPrint(
        allocator,
        "HTTP/1.1 {s}\r\n" ++
        "Content-Type: {s}\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Access-Control-Allow-Origin: *\r\n" ++
        "Server: HyperShimmy/0.1.0 (Zig+Mojo)\r\n" ++
        "\r\n" ++
        "{s}",
        .{ status, content_type, response_body.len, response_body },
    );
    defer allocator.free(http_response);
    
    _ = try conn.stream.writeAll(http_response);
}

/// Route request to appropriate handler
fn routeRequest(allocator: mem.Allocator, method: []const u8, path: []const u8, headers: []const u8, body: []const u8) ![]const u8 {
    // Handle OData Chat action
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/odata/v4/research/Chat")) {
        return try handleODataChatAction(allocator, body);
    }
    
    // Handle OData Summary action
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/odata/v4/research/GenerateSummary")) {
        return try handleODataSummaryAction(allocator, body);
    }
    
    // Handle file upload endpoint
    if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/api/upload")) {
        return try handleUploadRequest(allocator, headers, body);
    }
    
    // Serve webapp static files
    if (mem.eql(u8, path, "/") or mem.eql(u8, path, "/index.html")) {
        return try getStaticFile(allocator, "webapp/index.html");
    } else if (mem.startsWith(u8, path, "/webapp/") or 
               mem.startsWith(u8, path, "/view/") or
               mem.startsWith(u8, path, "/controller/") or
               mem.startsWith(u8, path, "/css/") or
               mem.startsWith(u8, path, "/i18n/") or
               mem.endsWith(u8, path, ".js") or
               mem.endsWith(u8, path, ".json") or
               mem.endsWith(u8, path, ".xml") or
               mem.endsWith(u8, path, ".css") or
               mem.endsWith(u8, path, ".properties")) {
        // Serve static file
        const file_path = if (mem.startsWith(u8, path, "/webapp/"))
            path[1..] // Remove leading slash
        else
            try std.fmt.allocPrint(allocator, "webapp{s}", .{path});
        defer if (!mem.startsWith(u8, path, "/webapp/")) allocator.free(file_path);
        
        return getStaticFile(allocator, file_path) catch |err| {
            std.debug.print("Static file not found: {s} ({any})\n", .{ file_path, err });
            return try std.fmt.allocPrint(allocator,
                \\{{"error":{{"code":"NotFound","message":"File not found","path":"{s}"}}}}
            , .{path});
        };
    } else if (mem.eql(u8, path, "/api/info")) {
        return try allocator.dupe(u8,
            \\{"name":"HyperShimmy","version":"0.1.0-dev","description":"Pure Mojo/Zig Research Assistant","architecture":"Zig HTTP + OData + Mojo AI","status":"In Development - Week 1, Day 4"}
        );
    } else if (mem.eql(u8, path, "/health")) {
        const timestamp = std.time.timestamp();
        return try std.fmt.allocPrint(allocator,
            \\{{"status":"healthy","service":"HyperShimmy","version":"0.1.0-dev","engine":"Zig+Mojo","timestamp":{d}}}
        , .{timestamp});
    } else if (mem.startsWith(u8, path, "/odata/v4/research")) {
        if (mem.eql(u8, path, "/odata/v4/research") or 
            mem.eql(u8, path, "/odata/v4/research/")) {
            return try allocator.dupe(u8,
                \\{"@odata.context":"/odata/v4/research/$metadata","value":[{"name":"Sources","kind":"EntitySet","url":"Sources"},{"name":"Messages","kind":"EntitySet","url":"Messages"},{"name":"Summaries","kind":"EntitySet","url":"Summaries"},{"name":"MindmapNodes","kind":"EntitySet","url":"MindmapNodes"}]}
            );
        } else if (mem.eql(u8, path, "/odata/v4/research/$metadata")) {
            // Serve OData metadata XML
            return try getMetadataXml(allocator);
        } else {
            return try std.fmt.allocPrint(allocator,
                \\{{"error":{{"code":"ResourceNotFound","message":"OData resource not found","path":"{s}"}}}}
            , .{path});
        }
    } else {
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"code":"NotFound","message":"Resource not found","path":"{s}"}}}}
        , .{path});
    }
}

/// Handle OData Chat action
fn handleODataChatAction(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    return odata_chat.handleODataChatRequest(allocator, body) catch |err| {
        std.debug.print("❌ OData Chat action failed: {any}\n", .{err});
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"code":"InternalError","message":"Chat action failed: {any}"}}}}
        , .{err});
    };
}

/// Handle OData Summary action
fn handleODataSummaryAction(allocator: mem.Allocator, body: []const u8) ![]const u8 {
    return odata_summary.handleODataSummaryRequest(allocator, body) catch |err| {
        std.debug.print("❌ OData Summary action failed: {any}\n", .{err});
        return try std.fmt.allocPrint(allocator,
            \\{{"error":{{"code":"InternalError","message":"Summary action failed: {any}"}}}}
        , .{err});
    };
}

/// Handle file upload request
fn handleUploadRequest(allocator: mem.Allocator, headers: []const u8, body: []const u8) ![]const u8 {
    // Extract Content-Type header
    const content_type = blk: {
        if (mem.indexOf(u8, headers, "Content-Type:")) |ct_start| {
            const ct_line_start = ct_start + 13; // "Content-Type:".len
            const ct_line_end = mem.indexOf(u8, headers[ct_line_start..], "\r\n") orelse headers.len - ct_line_start;
            break :blk mem.trim(u8, headers[ct_line_start .. ct_line_start + ct_line_end], " \t");
        }
        return try std.fmt.allocPrint(allocator,
            \\{{"success":false,"error":"Missing Content-Type header"}}
        , .{});
    };
    
    // Check if it's multipart/form-data
    if (!mem.startsWith(u8, content_type, "multipart/form-data")) {
        return try std.fmt.allocPrint(allocator,
            \\{{"success":false,"error":"Content-Type must be multipart/form-data"}}
        , .{});
    }
    
    // Initialize upload handler
    var handler = try upload.UploadHandler.init(allocator, "uploads");
    
    // Handle upload
    var result = handler.handleUpload(content_type, body) catch |err| {
        return try std.fmt.allocPrint(allocator,
            \\{{"success":false,"error":"Upload failed: {any}"}}
        , .{err});
    };
    defer result.deinit(allocator);
    
    // Format result as JSON
    return try handler.resultToJson(&result);
}

/// Get static file content
fn getStaticFile(allocator: mem.Allocator, file_path: []const u8) ![]const u8 {
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    
    const file_size = try file.getEndPos();
    const content = try file.readToEndAlloc(allocator, file_size);
    return content;
}

/// Get OData metadata XML
fn getMetadataXml(allocator: mem.Allocator) ![]const u8 {
    const metadata_path = "odata/metadata.xml";
    const file = std.fs.cwd().openFile(metadata_path, .{}) catch |err| {
        std.debug.print("Warning: Could not open {s}: {any}\n", .{ metadata_path, err });
        // Return inline minimal metadata if file not found
        return try allocator.dupe(u8,
            \\<?xml version="1.0" encoding="UTF-8"?>
            \\<edmx:Edmx xmlns:edmx="http://docs.oasis-open.org/odata/ns/edmx" Version="4.0">
            \\  <edmx:DataServices>
            \\    <Schema xmlns="http://docs.oasis-open.org/odata/ns/edm" Namespace="HyperShimmy.Research">
            \\      <EntityContainer Name="ResearchService"/>
            \\    </Schema>
            \\  </edmx:DataServices>
            \\</edmx:Edmx>
        );
    };
    defer file.close();
    
    const file_size = try file.getEndPos();
    const content = try file.readToEndAlloc(allocator, file_size);
    return content;
}

/// Start HTTP server
pub fn startServer(config: ServerConfig) !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Parse address
    const addr = try net.Address.parseIp(config.host, config.port);
    
    // Create server
    var server = try addr.listen(.{
        .reuse_address = true,
    });
    defer server.deinit();
    
    std.debug.print("\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("🚀 HyperShimmy Server Started\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Configuration:\n", .{});
    std.debug.print("  • Address:   {s}:{d}\n", .{ config.host, config.port });
    std.debug.print("\n", .{});
    std.debug.print("Endpoints:\n", .{});
    std.debug.print("  • Server Info:    http://localhost:{d}/\n", .{config.port});
    std.debug.print("  • Health Check:   http://localhost:{d}/health\n", .{config.port});
    std.debug.print("  • File Upload:    POST http://localhost:{d}/api/upload\n", .{config.port});
    std.debug.print("  • OData Root:     http://localhost:{d}/odata/v4/research/\n", .{config.port});
    std.debug.print("  • Chat Action:    POST http://localhost:{d}/odata/v4/research/Chat\n", .{config.port});
    std.debug.print("  • Summary Action: POST http://localhost:{d}/odata/v4/research/GenerateSummary\n", .{config.port});
    std.debug.print("\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("✓ Server ready! Press Ctrl+C to stop.\n", .{});
    std.debug.print("======================================================================\n", .{});
    std.debug.print("\n", .{});
    
    // Accept connections loop
    while (true) {
        const conn = try server.accept();
        
        // Handle connection (single-threaded for now)
        handleConnection(conn, allocator) catch |err| {
            std.debug.print("⚠️  Connection error: {any}\n", .{err});
        };
    }
}

pub fn main() !void {
    try startServer(default_config);
}

// ============================================================================
// Tests
// ============================================================================

test "server config" {
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
    };
    
    try std.testing.expectEqualStrings("127.0.0.1", config.host);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
}

test "default config" {
    try std.testing.expectEqualStrings("0.0.0.0", default_config.host);
    try std.testing.expectEqual(@as(u16, 11434), default_config.port);
}
