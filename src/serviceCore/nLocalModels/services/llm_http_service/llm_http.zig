const std = @import("std");
const net = std.net;
const mem = std.mem;
const json = std.json;

// External C ABI functions from Mojo
extern fn extract_workflow_c(
    markdown_ptr: [*]const u8,
    markdown_len: usize,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32;

extern fn get_health_status_c(
    result_buffer: [*]u8,
    buffer_size: usize,
) i32;

const PORT = 8006;
const BUFFER_SIZE = 1024 * 1024; // 1MB buffer for results

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("🧠 LLM HTTP Service (Zig + Mojo)\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("   Port: {d}\n", .{PORT});
    std.debug.print("   Backend: Mojo RLM + TOON\n", .{});
    std.debug.print("   Endpoints:\n", .{});
    std.debug.print("     - GET  /\n", .{});
    std.debug.print("     - GET  /health\n", .{});
    std.debug.print("     - POST /extract-workflow\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});

    const address = try net.Address.parseIp("127.0.0.1", PORT);
    var server = try address.listen(.{ .reuse_address = true });
    defer server.deinit();

    std.debug.print("✅ Server listening on http://127.0.0.1:{d}\n", .{PORT});
    std.debug.print("   Ready to accept connections...\n\n", .{});

    while (true) {
        const connection = try server.accept();
        std.debug.print("📥 New connection from {}\n", .{connection.address});

        // Handle connection in current thread (can be improved with threading later)
        handleConnection(allocator, connection) catch |err| {
            std.debug.print("❌ Error handling connection: {}\n", .{err});
        };
    }
}

fn handleConnection(allocator: mem.Allocator, connection: net.Server.Connection) !void {
    defer connection.stream.close();

    var buffer: [8192]u8 = undefined;
    const bytes_read = try connection.stream.read(&buffer);

    if (bytes_read == 0) return;

    const request = buffer[0..bytes_read];
    
    std.debug.print("📨 Request ({d} bytes)\n", .{bytes_read});

    // Parse HTTP request
    var lines_iter = mem.tokenizeScalar(u8, request, '\n');
    const request_line = lines_iter.next() orelse return error.InvalidRequest;

    // Parse method and path
    var parts_iter = mem.tokenizeScalar(u8, request_line, ' ');
    const method = parts_iter.next() orelse return error.InvalidRequest;
    const path = parts_iter.next() orelse return error.InvalidRequest;

    std.debug.print("   Method: {s}\n", .{method});
    std.debug.print("   Path: {s}\n", .{path});

    // Route request
    if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/")) {
        try handleRoot(connection.stream);
    } else if (mem.eql(u8, method, "GET") and mem.eql(u8, path, "/health")) {
        try handleHealth(allocator, connection.stream);
    } else if (mem.eql(u8, method, "POST") and mem.eql(u8, path, "/extract-workflow")) {
        try handleExtractWorkflow(allocator, connection.stream, request);
    } else {
        try send404(connection.stream);
    }
}

fn handleRoot(stream: net.Stream) !void {
    const response =
        \\HTTP/1.1 200 OK
        \\Content-Type: application/json
        \\Access-Control-Allow-Origin: *
        \\
        \\{
        \\  "service": "llm-http",
        \\  "version": "1.0.0",
        \\  "description": "LLM HTTP Service - Workflow Extraction",
        \\  "endpoints": [
        \\    "GET /",
        \\    "GET /health",
        \\    "POST /extract-workflow"
        \\  ],
        \\  "backend": "Mojo RLM + TOON"
        \\}
        \\
    ;
    _ = try stream.write(response);
    std.debug.print("✅ Sent root info\n", .{});
}

fn handleHealth(allocator: mem.Allocator, stream: net.Stream) !void {
    var result_buffer = try allocator.alloc(u8, BUFFER_SIZE);
    defer allocator.free(result_buffer);

    // Call Mojo health check
    const rc = get_health_status_c(result_buffer.ptr, BUFFER_SIZE);

    if (rc != 0) {
        try sendError(stream, 500, "Health check failed");
        return;
    }

    // Find null terminator
    var json_len: usize = 0;
    while (json_len < BUFFER_SIZE and result_buffer[json_len] != 0) {
        json_len += 1;
    }

    const json_result = result_buffer[0..json_len];

    // Send response
    const header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    _ = try stream.write(header);
    _ = try stream.write(json_result);

    std.debug.print("✅ Health check OK\n", .{});
}

fn handleExtractWorkflow(allocator: mem.Allocator, stream: net.Stream, request: []const u8) !void {
    // Find request body (after \r\n\r\n)
    const body_start = mem.indexOf(u8, request, "\r\n\r\n") orelse return error.NoBody;
    const body = request[body_start + 4 ..];

    std.debug.print("📄 Request body ({d} bytes)\n", .{body.len});

    // Parse JSON request
    const parsed = json.parseFromSlice(
        struct {
            markdown: []const u8,
            temperature: ?f32 = null,
        },
        allocator,
        body,
        .{},
    ) catch {
        try sendError(stream, 400, "Invalid JSON");
        return;
    };
    defer parsed.deinit();

    const markdown = parsed.value.markdown;
    const temperature = parsed.value.temperature orelse 0.3;

    std.debug.print("   Markdown: {d} chars\n", .{markdown.len});
    std.debug.print("   Temperature: {d}\n", .{temperature});

    // Allocate result buffer
    var result_buffer = try allocator.alloc(u8, BUFFER_SIZE);
    defer allocator.free(result_buffer);

    // Call Mojo workflow extraction
    std.debug.print("🔄 Calling Mojo workflow extraction...\n", .{});
    const rc = extract_workflow_c(
        markdown.ptr,
        markdown.len,
        temperature,
        result_buffer.ptr,
        BUFFER_SIZE,
    );

    if (rc != 0) {
        try sendError(stream, 500, "Workflow extraction failed");
        return;
    }

    // Find null terminator
    var json_len: usize = 0;
    while (json_len < BUFFER_SIZE and result_buffer[json_len] != 0) {
        json_len += 1;
    }

    const json_result = result_buffer[0..json_len];

    // Send response
    const header = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n";
    _ = try stream.write(header);
    _ = try stream.write(json_result);

    std.debug.print("✅ Workflow extracted successfully ({d} bytes)\n", .{json_len});
}

fn send404(stream: net.Stream) !void {
    const response =
        \\HTTP/1.1 404 Not Found
        \\Content-Type: application/json
        \\Access-Control-Allow-Origin: *
        \\
        \\{"error":"Not Found"}
        \\
    ;
    _ = try stream.write(response);
    std.debug.print("❌ Sent 404\n", .{});
}

fn sendError(stream: net.Stream, status: u16, message: []const u8) !void {
    var buffer: [1024]u8 = undefined;
    const response = try std.fmt.bufPrint(
        &buffer,
        "HTTP/1.1 {d} Error\r\nContent-Type: application/json\r\nAccess-Control-Allow-Origin: *\r\n\r\n{{\"error\":\"{s}\"}}\r\n",
        .{ status, message },
    );
    _ = try stream.write(response);
    std.debug.print("❌ Sent error {d}: {s}\n", .{ status, message });
}
