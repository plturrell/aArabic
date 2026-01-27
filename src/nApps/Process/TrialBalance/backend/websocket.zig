// ============================================================================
// WebSocket Module for Trial Balance Real-time Streaming
// Based on n-c-sdk demos WebSocket implementation
// ============================================================================

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;

// WebSocket constants
const WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11";

// WebSocket opcode constants
pub const OpCode = enum(u8) {
    continuation = 0x0,
    text = 0x1,
    binary = 0x2,
    close = 0x8,
    ping = 0x9,
    pong = 0xA,
};

// Client management
const MaxClients = 32;
pub var client_streams: [MaxClients]?net.Stream = [_]?net.Stream{null} ** MaxClients;
pub var clients_mutex = std.Thread.Mutex{};
pub var connected_clients: usize = 0;

// ============================================================================
// WebSocket Handshake
// ============================================================================

pub fn handleWebSocketUpgrade(stream: net.Stream, request: []const u8) !void {
    // Find Sec-WebSocket-Key
    const key_prefix = "Sec-WebSocket-Key: ";
    const key_start = (std.mem.indexOf(u8, request, key_prefix) orelse return error.NoWebSocketKey) + key_prefix.len;
    const key_end = std.mem.indexOfPos(u8, request, key_start, "\r\n") orelse return error.InvalidKey;
    const client_key = request[key_start..key_end];

    // Generate accept key using SHA-1
    var sha1 = std.crypto.hash.Sha1.init(.{});
    sha1.update(client_key);
    sha1.update(WS_GUID);
    const hash = sha1.finalResult();

    // Base64 encode the hash
    var accept_key: [28]u8 = undefined;
    _ = std.base64.standard.Encoder.encode(&accept_key, &hash);

    // Send WebSocket upgrade response
    var resp_buf: [512]u8 = undefined;
    const resp = try std.fmt.bufPrint(&resp_buf,
        "HTTP/1.1 101 Switching Protocols\r\n" ++
        "Upgrade: websocket\r\n" ++
        "Connection: Upgrade\r\n" ++
        "Sec-WebSocket-Accept: {s}\r\n" ++
        "\r\n",
        .{accept_key});
    
    _ = try stream.write(resp);
    
    std.debug.print("✓ WebSocket handshake complete\n", .{});
}

// ============================================================================
// WebSocket Frame Handling
// ============================================================================

pub fn sendWebSocketMessage(stream: net.Stream, message: []const u8) !void {
    var frame: [8192]u8 = undefined;
    
    // Frame header: FIN bit (0x80) + text opcode (0x01) = 0x81
    frame[0] = 0x81;
    
    if (message.len < 126) {
        // Small payload
        frame[1] = @intCast(message.len);
        @memcpy(frame[2..][0..message.len], message);
        _ = try stream.write(frame[0 .. 2 + message.len]);
    } else if (message.len < 65536) {
        // Medium payload (use 16-bit length)
        frame[1] = 126;
        frame[2] = @intCast((message.len >> 8) & 0xFF);
        frame[3] = @intCast(message.len & 0xFF);
        @memcpy(frame[4..][0..message.len], message);
        _ = try stream.write(frame[0 .. 4 + message.len]);
    } else {
        // Large payload (use 64-bit length)
        frame[1] = 127;
        const len = message.len;
        frame[2] = @intCast((len >> 56) & 0xFF);
        frame[3] = @intCast((len >> 48) & 0xFF);
        frame[4] = @intCast((len >> 40) & 0xFF);
        frame[5] = @intCast((len >> 32) & 0xFF);
        frame[6] = @intCast((len >> 24) & 0xFF);
        frame[7] = @intCast((len >> 16) & 0xFF);
        frame[8] = @intCast((len >> 8) & 0xFF);
        frame[9] = @intCast(len & 0xFF);
        
        // For very large messages, send in chunks
        _ = try stream.write(frame[0..10]);
        _ = try stream.write(message);
    }
}

pub fn receiveWebSocketFrame(stream: net.Stream, buffer: []u8) ![]const u8 {
    const header_len = try stream.read(buffer[0..2]);
    if (header_len < 2) return error.InvalidFrame;

    const opcode: u8 = buffer[0] & 0x0F;
    const masked = (buffer[1] & 0x80) != 0;
    var payload_len: usize = buffer[1] & 0x7F;
    var offset: usize = 2;

    // Handle extended payload length
    if (payload_len == 126) {
        _ = try stream.read(buffer[offset..][0..2]);
        payload_len = (@as(usize, buffer[offset]) << 8) | buffer[offset + 1];
        offset += 2;
    } else if (payload_len == 127) {
        _ = try stream.read(buffer[offset..][0..8]);
        payload_len = 0;
        for (0..8) |i| {
            payload_len = (payload_len << 8) | buffer[offset + i];
        }
        offset += 8;
    }

    // Read mask key if present
    var mask: [4]u8 = undefined;
    if (masked) {
        _ = try stream.read(buffer[offset..][0..4]);
        @memcpy(&mask, buffer[offset..][0..4]);
        offset += 4;
    }

    // Read payload
    if (payload_len > buffer.len - offset) return error.PayloadTooLarge;
    const payload_start = offset;
    _ = try stream.read(buffer[offset..][0..payload_len]);

    // Unmask payload if masked
    if (masked) {
        for (buffer[payload_start..][0..payload_len], 0..) |*byte, i| {
            byte.* ^= mask[i % 4];
        }
    }

    // Handle control frames
    if (opcode == @intFromEnum(OpCode.close)) {
        return error.ConnectionClosed;
    } else if (opcode == @intFromEnum(OpCode.ping)) {
        // Respond with pong
        var pong_frame: [2]u8 = .{ 0x8A, 0x00 };
        _ = try stream.write(&pong_frame);
        return &[_]u8{}; // Empty payload
    }

    return buffer[payload_start..][0..payload_len];
}

// ============================================================================
// Client Management
// ============================================================================

pub fn registerClient(stream: net.Stream) void {
    clients_mutex.lock();
    defer clients_mutex.unlock();

    for (&client_streams) |*slot| {
        if (slot.* == null) {
            slot.* = stream;
            connected_clients += 1;
            std.debug.print("📡 WebSocket client connected ({d} total)\n", .{connected_clients});
            return;
        }
    }
    
    std.debug.print("⚠️  Max WebSocket clients reached ({d})\n", .{MaxClients});
}

pub fn unregisterClient(stream: net.Stream) void {
    clients_mutex.lock();
    defer clients_mutex.unlock();

    for (&client_streams) |*slot| {
        if (slot.*) |s| {
            if (s.handle == stream.handle) {
                slot.* = null;
                if (connected_clients > 0) connected_clients -= 1;
                std.debug.print("📡 WebSocket client disconnected ({d} remaining)\n", .{connected_clients});
                return;
            }
        }
    }
}

pub fn broadcastMessage(message: []const u8) void {
    clients_mutex.lock();
    defer clients_mutex.unlock();

    var dead_clients: usize = 0;
    
    for (&client_streams) |*slot| {
        if (slot.*) |stream| {
            sendWebSocketMessage(stream, message) catch {
                slot.* = null;
                dead_clients += 1;
            };
        }
    }
    
    if (dead_clients > 0 and connected_clients >= dead_clients) {
        connected_clients -= dead_clients;
    }
}

// ============================================================================
// Trial Balance Specific Messages
// ============================================================================

pub fn sendCalculationProgress(
    allocator: Allocator,
    entries_processed: usize,
    total_entries: usize,
    accounts_calculated: usize,
    current_balance: f64,
) !void {
    const percentage = (@as(f64, @floatFromInt(entries_processed)) / @as(f64, @floatFromInt(total_entries))) * 100.0;
    
    const message = try std.fmt.allocPrint(allocator,
        \\{{"type":"tb:progress","payload":{{
        \\"entriesProcessed":{d},
        \\"totalEntries":{d},
        \\"percentage":{d:.1},
        \\"accountsCalculated":{d},
        \\"currentBalance":{d:.2}
        \\}}}}
    , .{
        entries_processed,
        total_entries,
        percentage,
        accounts_calculated,
        current_balance,
    });
    defer allocator.free(message);
    
    broadcastMessage(message);
}

pub fn sendValidationStatus(
    allocator: Allocator,
    status: []const u8,
    errors: usize,
    warnings: usize,
) !void {
    const message = try std.fmt.allocPrint(allocator,
        \\{{"type":"tb:validation","payload":{{
        \\"status":"{s}",
        \\"errors":{d},
        \\"warnings":{d}
        \\}}}}
    , .{
        status,
        errors,
        warnings,
    });
    defer allocator.free(message);
    
    broadcastMessage(message);
}

pub fn sendPerformanceMetrics(
    allocator: Allocator,
    entries_per_second: f64,
    memory_mb: f64,
    elapsed_ms: f64,
) !void {
    const message = try std.fmt.allocPrint(allocator,
        \\{{"type":"tb:performance","payload":{{
        \\"entriesPerSecond":{d:.0},
        \\"memoryMb":{d:.1},
        \\"elapsedMs":{d:.0}
        \\}}}}
    , .{
        entries_per_second,
        memory_mb,
        elapsed_ms,
    });
    defer allocator.free(message);
    
    broadcastMessage(message);
}

pub fn sendCalculationComplete(
    allocator: Allocator,
    total_debits: f64,
    total_credits: f64,
    balance_difference: f64,
    is_balanced: bool,
    account_count: usize,
    elapsed_ms: f64,
) !void {
    const message = try std.fmt.allocPrint(allocator,
        \\{{"type":"tb:complete","payload":{{
        \\"totalDebits":{d:.2},
        \\"totalCredits":{d:.2},
        \\"balanceDifference":{d:.2},
        \\"isBalanced":{},
        \\"accountCount":{d},
        \\"elapsedMs":{d:.0}
        \\}}}}
    , .{
        total_debits,
        total_credits,
        balance_difference,
        is_balanced,
        account_count,
        elapsed_ms,
    });
    defer allocator.free(message);
    
    broadcastMessage(message);
}

pub fn sendError(
    allocator: Allocator,
    error_message: []const u8,
) !void {
    const message = try std.fmt.allocPrint(allocator,
        \\{{"type":"tb:error","payload":{{
        \\"message":"{s}"
        \\}}}}
    , .{
        error_message,
    });
    defer allocator.free(message);
    
    broadcastMessage(message);
}