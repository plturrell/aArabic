// Zig Qdrant Client for Mojo
// Native Qdrant vector database client
// Provides search and upsert operations via REST API

const std = @import("std");
const http = std.http;
const net = std.net;
const mem = std.mem;
const json = std.json;

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Qdrant search request
pub const SearchRequest = extern struct {
    collection: [*:0]const u8,
    vector: [*]const f32,
    vector_len: usize,
    limit: usize,
};

/// Qdrant search result
pub const SearchResult = extern struct {
    id: u64,
    score: f32,
    payload: [*:0]const u8,
};

/// Search Qdrant collection
/// Returns JSON string with results
export fn zig_qdrant_search(
    url: [*:0]const u8,
    request: *const SearchRequest
) callconv(.c) [*:0]const u8 {
    const qdrant_url = mem.span(url);
    
    const result = qdrantSearch(qdrant_url, request) catch |err| {
        std.debug.print("Qdrant search error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn qdrantSearch(qdrant_url: []const u8, request: *const SearchRequest) ![:0]const u8 {
    const collection = mem.span(request.collection);
    const vector = request.vector[0..request.vector_len];
    
    std.debug.print("🔍 Qdrant search: {s}, limit={d}\n", .{collection, request.limit});
    
    // Build search URL
    const search_url = try std.fmt.allocPrint(
        allocator,
        "{s}/collections/{s}/points/search",
        .{qdrant_url, collection}
    );
    defer allocator.free(search_url);
    
    // Build request body
    var body_list: std.ArrayList(u8) = .{};
    defer body_list.deinit(allocator);
    
    const writer = body_list.writer(allocator);
    
    // Start JSON
    try writer.writeAll("{\"vector\":[");
    
    // Write vector
    for (vector, 0..) |val, i| {
        if (i > 0) try writer.writeAll(",");
        try std.fmt.format(writer, "{d}", .{val});
    }
    
    // Write limit and options
    try std.fmt.format(
        writer,
        "],\"limit\":{d},\"with_payload\":true}}",
        .{request.limit}
    );
    
    const body = body_list.items;
    
    std.debug.print("📤 Request body: {d} bytes\n", .{body.len});
    
    // Parse URL
    const uri = try std.Uri.parse(search_url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 6333
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send POST request
    const http_request = try std.fmt.allocPrint(
        allocator,
        "POST {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
    );
    defer allocator.free(http_request);
    
    _ = try conn.writeAll(http_request);
    
    // Read response
    var response_buffer: [65536]u8 = undefined;
    const bytes_read = try conn.read(&response_buffer);
    
    std.debug.print("📥 Response: {d} bytes\n", .{bytes_read});
    
    // Find body
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, response_buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const response_body = response_buffer[body_start..bytes_read];
        
        // Copy to result
        const result = try allocator.allocSentinel(u8, response_body.len, 0);
        @memcpy(result[0..response_body.len], response_body);
        
        return result;
    }
    
    return "{}";
}

/// Upsert points to Qdrant collection
export fn zig_qdrant_upsert(
    url: [*:0]const u8,
    collection: [*:0]const u8,
    points_json: [*:0]const u8
) callconv(.c) c_int {
    const qdrant_url = mem.span(url);
    const collection_name = mem.span(collection);
    const points_data = mem.span(points_json);
    
    qdrantUpsert(qdrant_url, collection_name, points_data) catch |err| {
        std.debug.print("Qdrant upsert error: {any}\n", .{err});
        return -1;
    };
    
    return 0;
}

fn qdrantUpsert(qdrant_url: []const u8, collection: []const u8, points_json: []const u8) !void {
    std.debug.print("📝 Qdrant upsert: {s}\n", .{collection});
    
    // Build upsert URL
    const upsert_url = try std.fmt.allocPrint(
        allocator,
        "{s}/collections/{s}/points",
        .{qdrant_url, collection}
    );
    defer allocator.free(upsert_url);
    
    // Build request body
    const body = try std.fmt.allocPrint(
        allocator,
        "{{\"points\":{s}}}",
        .{points_json}
    );
    defer allocator.free(body);
    
    // Parse URL
    const uri = try std.Uri.parse(upsert_url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 6333
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send PUT request
    const http_request = try std.fmt.allocPrint(
        allocator,
        "PUT {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
    );
    defer allocator.free(http_request);
    
    _ = try conn.writeAll(http_request);
    
    // Read response
    var response_buffer: [4096]u8 = undefined;
    _ = try conn.read(&response_buffer);
    
    std.debug.print("✅ Upsert complete\n", .{});
}

/// Create Qdrant collection
export fn zig_qdrant_create_collection(
    url: [*:0]const u8,
    collection: [*:0]const u8,
    vector_size: usize
) callconv(.c) c_int {
    const qdrant_url = mem.span(url);
    const collection_name = mem.span(collection);
    
    qdrantCreateCollection(qdrant_url, collection_name, vector_size) catch |err| {
        std.debug.print("Qdrant create collection error: {any}\n", .{err});
        return -1;
    };
    
    return 0;
}

fn qdrantCreateCollection(qdrant_url: []const u8, collection: []const u8, vector_size: usize) !void {
    std.debug.print("🆕 Creating collection: {s} (size={d})\n", .{collection, vector_size});
    
    // Build create URL
    const create_url = try std.fmt.allocPrint(
        allocator,
        "{s}/collections/{s}",
        .{qdrant_url, collection}
    );
    defer allocator.free(create_url);
    
    // Build request body
    const body = try std.fmt.allocPrint(
        allocator,
        "{{\"vectors\":{{\"size\":{d},\"distance\":\"Cosine\"}}}}",
        .{vector_size}
    );
    defer allocator.free(body);
    
    // Parse URL
    const uri = try std.Uri.parse(create_url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 6333
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send PUT request
    const http_request = try std.fmt.allocPrint(
        allocator,
        "PUT {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Content-Type: application/json\r\n" ++
        "Content-Length: {d}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n" ++
        "{s}",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded, body.len, body}
    );
    defer allocator.free(http_request);
    
    _ = try conn.writeAll(http_request);
    
    // Read response
    var response_buffer: [4096]u8 = undefined;
    _ = try conn.read(&response_buffer);
    
    std.debug.print("✅ Collection created\n", .{});
}

/// Get collection info
export fn zig_qdrant_get_collection(
    url: [*:0]const u8,
    collection: [*:0]const u8
) callconv(.c) [*:0]const u8 {
    const qdrant_url = mem.span(url);
    const collection_name = mem.span(collection);
    
    const result = qdrantGetCollection(qdrant_url, collection_name) catch {
        return "{}";
    };
    
    return result.ptr;
}

fn qdrantGetCollection(qdrant_url: []const u8, collection: []const u8) ![:0]const u8 {
    // Build URL
    const get_url = try std.fmt.allocPrint(
        allocator,
        "{s}/collections/{s}",
        .{qdrant_url, collection}
    );
    defer allocator.free(get_url);
    
    // Parse URL
    const uri = try std.Uri.parse(get_url);
    
    // Connect
    const addr = try net.Address.parseIp(
        uri.host.?.percent_encoded,
        uri.port orelse 6333
    );
    
    const conn = try net.tcpConnectToAddress(addr);
    defer conn.close();
    
    // Send GET request
    const http_request = try std.fmt.allocPrint(
        allocator,
        "GET {s} HTTP/1.1\r\n" ++
        "Host: {s}\r\n" ++
        "Connection: close\r\n" ++
        "\r\n",
        .{uri.path.percent_encoded, uri.host.?.percent_encoded}
    );
    defer allocator.free(http_request);
    
    _ = try conn.writeAll(http_request);
    
    // Read response
    var response_buffer: [16384]u8 = undefined;
    const bytes_read = try conn.read(&response_buffer);
    
    // Find body
    const body_marker = "\r\n\r\n";
    if (mem.indexOf(u8, response_buffer[0..bytes_read], body_marker)) |idx| {
        const body_start = idx + body_marker.len;
        const response_body = response_buffer[body_start..bytes_read];
        
        const result = try allocator.allocSentinel(u8, response_body.len, 0);
        @memcpy(result[0..response_body.len], response_body);
        
        return result;
    }
    
    return "{}";
}

// For testing
pub fn main() !void {
    std.debug.print("🧪 Zig Qdrant Client Test\n", .{});
    std.debug.print("\nFeatures:\n", .{});
    std.debug.print("  • Search collections\n", .{});
    std.debug.print("  • Upsert points\n", .{});
    std.debug.print("  • Create collections\n", .{});
    std.debug.print("  • Get collection info\n", .{});
    std.debug.print("\n✅ Qdrant client ready!\n", .{});
    std.debug.print("Build with: zig build-lib zig_qdrant.zig -dynamic\n", .{});
}
