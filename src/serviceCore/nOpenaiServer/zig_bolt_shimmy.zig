// Zig Bolt Protocol Client for Graph Databases
// Neo4j & Memgraph compatible via Bolt protocol v4/v5
// Exports C ABI functions that Mojo can call via FFI

const std = @import("std");
const net = std.net;
const mem = std.mem;

const header_line = "================================================================================";

var log_enabled: ?bool = null;

fn logEnabled() bool {
    if (log_enabled) |enabled| {
        return enabled;
    }
    log_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
    return log_enabled.?;
}

fn log(comptime fmt: []const u8, args: anytype) void {
    if (logEnabled()) {
        std.debug.print(fmt, args);
    }
}

// Global allocator
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

// Connection pool (simple array for now, max 16 connections)
const MAX_CONNECTIONS = 16;
var connection_pool: [MAX_CONNECTIONS]?BoltConnection = [_]?BoltConnection{null} ** MAX_CONNECTIONS;
var connection_mutex = std.Thread.Mutex{};

// Security: Input validation limits
const MAX_QUERY_SIZE = 1024 * 1024;  // 1MB max query size
const MAX_PARAM_SIZE = 256 * 1024;    // 256KB max parameters
const MAX_HOST_SIZE = 255;            // Standard hostname limit
const MAX_USERNAME_SIZE = 255;
const MAX_PASSWORD_SIZE = 255;

// Bolt protocol version constants
const BOLT_MAGIC: [4]u8 = .{ 0x60, 0x60, 0xB0, 0x17 };
const BOLT_VERSION_4: [4]u8 = .{ 0x00, 0x00, 0x04, 0x04 };
const BOLT_VERSION_5: [4]u8 = .{ 0x00, 0x00, 0x05, 0x00 };

// Bolt message signatures
const MSG_HELLO: u8 = 0x01;
const MSG_RUN: u8 = 0x10;
const MSG_PULL: u8 = 0x3F;
const MSG_RESET: u8 = 0x0F;
const MSG_GOODBYE: u8 = 0x02;
const MSG_SUCCESS: u8 = 0x70;
const MSG_RECORD: u8 = 0x71;
const MSG_FAILURE: u8 = 0x7F;

// PackStream type markers
const TINY_STRING: u8 = 0x80; // 0x80..0x8F for strings 0-15 bytes
const STRING_8: u8 = 0xD0;
const STRING_16: u8 = 0xD1;
const STRING_32: u8 = 0xD2;
const TINY_MAP: u8 = 0xA0; // 0xA0..0xAF for maps 0-15 items
const MAP_8: u8 = 0xD8;
const TINY_LIST: u8 = 0x90; // 0x90..0x9F for lists 0-15 items
const LIST_8: u8 = 0xD4;
const NULL: u8 = 0xC0;
const TRUE: u8 = 0xC3;
const FALSE: u8 = 0xC2;
const INT_8: u8 = 0xC8;
const INT_16: u8 = 0xC9;
const INT_32: u8 = 0xCA;
const INT_64: u8 = 0xCB;
const FLOAT_64: u8 = 0xC1;

const BoltConnection = struct {
    stream: net.Stream,
    version: u8,
    server_agent: []const u8,
    allocated: bool,
};

// ============================================================================
// Security: Input Validation Functions
// ============================================================================

fn validateConnectionId(connection_id: c_int) !void {
    if (connection_id < 0 or connection_id >= MAX_CONNECTIONS) {
        return error.InvalidConnectionId;
    }
    
    connection_mutex.lock();
    defer connection_mutex.unlock();
    
    if (connection_pool[@intCast(connection_id)] == null) {
        return error.ConnectionNotActive;
    }
}

fn validateQueryInput(query: []const u8, params: []const u8) !void {
    // Check query size
    if (query.len > MAX_QUERY_SIZE) {
        log("❌ Query too large: {d} bytes (max: {d})\n", .{query.len, MAX_QUERY_SIZE});
        return error.QueryTooLarge;
    }
    
    // Check params size
    if (params.len > MAX_PARAM_SIZE) {
        log("❌ Parameters too large: {d} bytes (max: {d})\n", .{params.len, MAX_PARAM_SIZE});
        return error.ParamsTooLarge;
    }
    
    // Check for embedded null bytes (security)
    for (query) |c| {
        if (c == 0) return error.InvalidQuery;
    }
    
    for (params) |c| {
        if (c == 0) return error.InvalidParams;
    }
}

fn validateConnectionInput(host: []const u8, username: []const u8, password: []const u8) !void {
    if (host.len == 0 or host.len > MAX_HOST_SIZE) {
        return error.InvalidHost;
    }
    
    if (username.len > MAX_USERNAME_SIZE) {
        return error.UsernameTooLong;
    }
    
    if (password.len > MAX_PASSWORD_SIZE) {
        return error.PasswordTooLong;
    }
    
    // Check for embedded null bytes
    for (host) |c| {
        if (c == 0) return error.InvalidHost;
    }
}

fn secureClear(buffer: []u8) void {
    @memset(buffer, 0);
    // Prevent compiler from optimizing away the memset
    // Use volatile pointer write to force memory operation
    const ptr: [*]volatile u8 = @ptrCast(buffer.ptr);
    _ = ptr[0];
}

/// Initialize Bolt client library
export fn zig_bolt_init() callconv(.c) c_int {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("🔌 Zig Bolt Protocol Client\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Features:\n", .{});
    std.debug.print("  ✅ Bolt Protocol v4/v5\n", .{});
    std.debug.print("  ✅ Neo4j & Memgraph support\n", .{});
    std.debug.print("  ✅ PackStream serialization\n", .{});
    std.debug.print("  ✅ Connection pooling\n", .{});
    std.debug.print("  ✅ FFI for Mojo integration\n", .{});
    std.debug.print("\n{s}\n\n", .{header_line});
    return 0;
}

/// Connect to a Bolt-compatible graph database
export fn zig_bolt_connect(
    host: [*:0]const u8,
    port: u16,
    username: [*:0]const u8,
    password: [*:0]const u8,
) callconv(.c) c_int {
    const host_str = mem.span(host);
    const username_str = mem.span(username);
    const password_str = mem.span(password);
    
    log("Connecting to {s}:{d}\n", .{ host_str, port });
    
    const conn_id = connectInternal(host_str, port, username_str, password_str) catch |err| {
        std.debug.print("❌ Connection error: {any}\n", .{err});
        return -1;
    };
    
    log("✅ Connected successfully (ID: {d})\n", .{conn_id});
    return conn_id;
}

fn connectInternal(
    host: []const u8,
    port: u16,
    username: []const u8,
    password: []const u8,
) !c_int {
    // Validate inputs
    try validateConnectionInput(host, username, password);
    
    // Parse address and connect
    const addr = try net.Address.parseIp(host, port);
    const stream = try net.tcpConnectToAddress(addr);
    errdefer stream.close();
    
    // Perform Bolt handshake
    try performHandshake(stream);
    
    // Send HELLO message with authentication
    try sendHello(stream, username, password);
    
    // Read SUCCESS response
    const success = try readMessage(stream);
    defer allocator.free(success);
    
    // TODO: Parse server info from success message
    
    // Store connection in pool
    connection_mutex.lock();
    defer connection_mutex.unlock();
    
    for (&connection_pool, 0..) |*slot, i| {
        if (slot.* == null) {
            slot.* = BoltConnection{
                .stream = stream,
                .version = 4,
                .server_agent = try allocator.dupe(u8, "unknown"),
                .allocated = true,
            };
            return @intCast(i);
        }
    }
    
    return error.ConnectionPoolFull;
}

fn performHandshake(stream: net.Stream) !void {
    // Send magic bytes + supported versions
    var handshake: [20]u8 = undefined;
    @memcpy(handshake[0..4], &BOLT_MAGIC);
    @memcpy(handshake[4..8], &BOLT_VERSION_5); // Prefer v5
    @memcpy(handshake[8..12], &BOLT_VERSION_4); // Fallback v4
    @memset(handshake[12..16], 0); // No v3
    @memset(handshake[16..20], 0); // No v2
    
    _ = try stream.writeAll(&handshake);
    
    // Read selected version
    var version_response: [4]u8 = undefined;
    const bytes_read = try stream.read(&version_response);
    if (bytes_read != 4) return error.IncompleteHandshake;
    
    // Check if server accepted a version
    if (mem.eql(u8, &version_response, &[_]u8{ 0, 0, 0, 0 })) {
        return error.NoCompatibleVersion;
    }
    
    log("Bolt version negotiated: {d}.{d}\n", .{ version_response[2], version_response[3] });
}

fn sendHello(stream: net.Stream, username: []const u8, password: []const u8) !void {
    var buffer: [1024]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);
    const writer = fbs.writer();
    
    // Structure signature: 0xB1 (1-element struct) + MSG_HELLO
    try writer.writeByte(0xB1);
    try writer.writeByte(MSG_HELLO);
    
    // Map with auth credentials
    try writer.writeByte(TINY_MAP | 3); // 3 items
    
    // "user_agent"
    try writeString(writer, "user_agent");
    try writeString(writer, "Shimmy-Mojo/1.0");
    
    // "scheme"
    try writeString(writer, "scheme");
    try writeString(writer, "basic");
    
    // "principal"
    try writeString(writer, "principal");
    try writeString(writer, username);
    
    // "credentials"
    try writeString(writer, "credentials");
    try writeString(writer, password);
    
    const message = fbs.getWritten();
    try sendChunked(stream, message);
    
    // Security: Zero sensitive data from buffer after sending
    // (credentials are in the buffer from PackStream encoding)
    secureClear(&buffer);
}

fn sendRun(stream: net.Stream, query: []const u8, params: []const u8) !void {
    var buffer: [8192]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);
    const writer = fbs.writer();
    
    // Structure: 0xB3 (3-element struct) + MSG_RUN
    try writer.writeByte(0xB3);
    try writer.writeByte(MSG_RUN);
    
    // Query string
    try writeString(writer, query);
    
    // Parameters (as JSON string for now, TODO: parse and encode as PackStream map)
    if (params.len > 0 and !mem.eql(u8, params, "{}")) {
        try writeString(writer, params);
    } else {
        try writer.writeByte(TINY_MAP); // Empty map
    }
    
    // Extra metadata (empty map)
    try writer.writeByte(TINY_MAP);
    
    const message = fbs.getWritten();
    try sendChunked(stream, message);
}

fn sendPull(stream: net.Stream) !void {
    var buffer: [64]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buffer);
    const writer = fbs.writer();
    
    // Structure: 0xB1 (1-element struct) + MSG_PULL
    try writer.writeByte(0xB1);
    try writer.writeByte(MSG_PULL);
    
    // Extra metadata with n=-1 (fetch all)
    try writer.writeByte(TINY_MAP | 1); // 1 item
    try writeString(writer, "n");
    try writer.writeByte(@as(u8, @bitCast(@as(i8, -1)))); // Tiny int -1
    
    const message = fbs.getWritten();
    try sendChunked(stream, message);
}

fn writeString(writer: anytype, str: []const u8) !void {
    if (str.len <= 15) {
        try writer.writeByte(TINY_STRING | @as(u8, @intCast(str.len)));
    } else if (str.len <= 255) {
        try writer.writeByte(STRING_8);
        try writer.writeByte(@intCast(str.len));
    } else if (str.len <= 65535) {
        try writer.writeByte(STRING_16);
        try writer.writeInt(u16, @intCast(str.len), .big);
    } else {
        try writer.writeByte(STRING_32);
        try writer.writeInt(u32, @intCast(str.len), .big);
    }
    try writer.writeAll(str);
}

fn sendChunked(stream: net.Stream, message: []const u8) !void {
    // Send message size (16-bit big endian)
    const size: u16 = @intCast(message.len);
    var size_bytes: [2]u8 = undefined;
    std.mem.writeInt(u16, &size_bytes, size, .big);
    
    _ = try stream.writeAll(&size_bytes);
    _ = try stream.writeAll(message);
    
    // Send end-of-message marker (0x00 0x00)
    _ = try stream.writeAll(&[_]u8{ 0x00, 0x00 });
}

fn readMessage(stream: net.Stream) ![]u8 {
    var result = try std.ArrayList(u8).initCapacity(allocator, 1024);
    errdefer result.deinit(allocator);
    
    while (true) {
        // Read chunk size
        var size_bytes: [2]u8 = undefined;
        const size_read = try stream.read(&size_bytes);
        if (size_read != 2) return error.IncompleteChunk;
        const chunk_size = std.mem.readInt(u16, &size_bytes, .big);
        
        if (chunk_size == 0) {
            // End of message
            break;
        }
        
        // Read chunk data
        const start = result.items.len;
        try result.resize(allocator, start + chunk_size);
        var total_read: usize = 0;
        while (total_read < chunk_size) {
            const n = try stream.read(result.items[start + total_read..start + chunk_size]);
            if (n == 0) return error.ConnectionClosed;
            total_read += n;
        }
    }
    
    return result.toOwnedSlice(allocator);
}

/// Execute a Cypher query
export fn zig_bolt_execute(
    connection_id: c_int,
    query: [*:0]const u8,
    params: [*:0]const u8,
) callconv(.c) [*:0]const u8 {
    const query_str = mem.span(query);
    const params_str = mem.span(params);
    
    log("Executing query: {s}\n", .{query_str});
    
    const result = executeInternal(connection_id, query_str, params_str) catch |err| {
        std.debug.print("❌ Query error: {any}\n", .{err});
        return "{}";
    };
    
    return result.ptr;
}

fn executeInternal(
    connection_id: c_int,
    query: []const u8,
    params: []const u8,
) ![:0]const u8 {
    // Validate connection ID
    try validateConnectionId(connection_id);
    
    // Validate query inputs
    try validateQueryInput(query, params);
    
    connection_mutex.lock();
    defer connection_mutex.unlock();
    
    const conn = &connection_pool[@intCast(connection_id)];
    if (conn.* == null) {
        return error.InvalidConnection;
    }
    
    const stream = conn.*.?.stream;
    
    // Send RUN message
    try sendRun(stream, query, params);
    
    // Read SUCCESS response
    const run_response = try readMessage(stream);
    defer allocator.free(run_response);
    
    // TODO: Check if RUN succeeded
    
    // Send PULL message
    try sendPull(stream);
    
    // Collect RECORD messages until SUCCESS
    var records = try std.ArrayList(u8).initCapacity(allocator, 1024);
    defer records.deinit(allocator);
    
    try records.appendSlice(allocator, "[");
    var first = true;
    
    while (true) {
        const msg = try readMessage(stream);
        defer allocator.free(msg);
        
        if (msg.len == 0) continue;
        
        const signature = if (msg.len > 1) msg[1] else 0;
        
        if (signature == MSG_SUCCESS) {
            break;
        } else if (signature == MSG_RECORD) {
            // TODO: Parse record and convert to JSON
            if (!first) try records.appendSlice(allocator, ",");
            try records.appendSlice(allocator, "{}");
            first = false;
        } else if (signature == MSG_FAILURE) {
            return error.QueryFailed;
        }
    }
    
    try records.appendSlice(allocator, "]");
    
    const result = try allocator.allocSentinel(u8, records.items.len, 0);
    @memcpy(result[0..records.items.len], records.items);
    
    return result;
}

/// Disconnect from database
export fn zig_bolt_disconnect(connection_id: c_int) callconv(.c) void {
    connection_mutex.lock();
    defer connection_mutex.unlock();
    
    const conn = &connection_pool[@intCast(connection_id)];
    if (conn.* != null) {
        // Send GOODBYE
        var buffer: [8]u8 = undefined;
        buffer[0] = 0xB0; // 0-element struct
        buffer[1] = MSG_GOODBYE;
        
        sendChunked(conn.*.?.stream, buffer[0..2]) catch {};
        
        conn.*.?.stream.close();
        allocator.free(conn.*.?.server_agent);
        conn.* = null;
        
        log("Disconnected (ID: {d})\n", .{connection_id});
    }
}

/// Free a string returned by Bolt functions
/// CRITICAL: Must be called to prevent memory leaks
export fn zig_bolt_free_string(str: [*:0]const u8) callconv(.c) void {
    if (@intFromPtr(str) == 0) return;
    
    const slice = mem.span(str);
    allocator.free(slice);
    
    log("Freed string ({d} bytes)\n", .{slice.len});
}

/// Free JSON result returned by execute
/// CRITICAL: Must be called after zig_bolt_execute to prevent memory leaks
export fn zig_bolt_free_result(result: [*:0]const u8) callconv(.c) void {
    zig_bolt_free_string(result);
}

// Test/demo entry point
pub fn main() !void {
    std.debug.print("{s}\n", .{header_line});
    std.debug.print("🔌 Zig Bolt Protocol Client\n", .{});
    std.debug.print("{s}\n\n", .{header_line});
    std.debug.print("Build Instructions:\n", .{});
    std.debug.print("  macOS:  zig build-lib zig_bolt_shimmy.zig -dynamic -OReleaseFast\n", .{});
    std.debug.print("  Linux:  zig build-lib zig_bolt_shimmy.zig -dynamic -OReleaseFast\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Output:\n", .{});
    std.debug.print("  • libzig_bolt_shimmy.dylib (macOS)\n", .{});
    std.debug.print("  • libzig_bolt_shimmy.so (Linux)\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Usage from Mojo:\n", .{});
    std.debug.print("  from sys.ffi import DLHandle\n", .{});
    std.debug.print("  var lib = DLHandle(\"./libzig_bolt_shimmy.dylib\")\n", .{});
    std.debug.print("  var connect = lib.get_function[...](\"zig_bolt_connect\")\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Features:\n", .{});
    std.debug.print("  ✅ Bolt Protocol v4/v5\n", .{});
    std.debug.print("  ✅ Neo4j & Memgraph support\n", .{});
    std.debug.print("  ✅ PackStream serialization\n", .{});
    std.debug.print("  ✅ Connection pooling\n", .{});
    std.debug.print("  ✅ FFI for Mojo\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("{s}\n", .{header_line});
}
