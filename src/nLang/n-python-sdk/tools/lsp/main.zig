// Mojo LSP Server - Main Entry Point
// Day 114: Working mojo-lsp binary with editor integration
// Updated for Zig 0.15.2 compatibility

const std = @import("std");
const jsonrpc = @import("jsonrpc.zig");
const server = @import("server.zig");
const json = std.json;

const Allocator = std.mem.Allocator;
const LspServer = server.LspServer;
const MessageParser = jsonrpc.MessageParser;
const MessageSerializer = jsonrpc.MessageSerializer;
const JsonRpcMessage = jsonrpc.JsonRpcMessage;
const MessageRouter = jsonrpc.MessageRouter;

// ============================================================================
// LSP Transport (stdio) - Zig 0.15.2 compatible
// ============================================================================

pub const StdioTransport = struct {
    allocator: Allocator,

    pub fn init(allocator: Allocator) StdioTransport {
        return StdioTransport{
            .allocator = allocator,
        };
    }

    /// Read a single byte from stdin
    fn readByte() !u8 {
        var buf: [1]u8 = undefined;
        const n = try std.posix.read(std.posix.STDIN_FILENO, &buf);
        if (n == 0) return error.UnexpectedEOF;
        return buf[0];
    }

    /// Read a line from stdin (until \n)
    fn readLine(buffer: []u8) ![]u8 {
        var i: usize = 0;
        while (i < buffer.len) {
            const byte = try readByte();
            if (byte == '\n') {
                return buffer[0..i];
            }
            buffer[i] = byte;
            i += 1;
        }
        return error.BufferTooSmall;
    }

    /// Read a Content-Length delimited message from stdin
    pub fn readMessage(self: *StdioTransport) ![]const u8 {
        var line_buf: [1024]u8 = undefined;
        var content_length: ?usize = null;

        // Read headers
        while (true) {
            const line = try readLine(&line_buf);

            // Trim \r if present
            const trimmed = if (line.len > 0 and line[line.len - 1] == '\r')
                line[0 .. line.len - 1]
            else
                line;

            if (trimmed.len == 0) {
                // Empty line marks end of headers
                break;
            }

            // Parse Content-Length header
            if (std.mem.startsWith(u8, trimmed, "Content-Length: ")) {
                const len_str = trimmed["Content-Length: ".len..];
                content_length = try std.fmt.parseInt(usize, len_str, 10);
            }
        }

        const len = content_length orelse return error.MissingContentLength;

        // Read message body
        const message = try self.allocator.alloc(u8, len);
        errdefer self.allocator.free(message);

        var total_read: usize = 0;
        while (total_read < len) {
            const n = try std.posix.read(std.posix.STDIN_FILENO, message[total_read..]);
            if (n == 0) return error.UnexpectedEOF;
            total_read += n;
        }

        return message;
    }

    /// Write a Content-Length delimited message to stdout
    pub fn writeMessage(self: *StdioTransport, message: []const u8) !void {
        _ = self;

        // Format header
        var header_buf: [64]u8 = undefined;
        const header = std.fmt.bufPrint(&header_buf, "Content-Length: {d}\r\n\r\n", .{message.len}) catch unreachable;

        // Write header
        _ = try std.posix.write(std.posix.STDOUT_FILENO, header);

        // Write message body
        var total_written: usize = 0;
        while (total_written < message.len) {
            const n = try std.posix.write(std.posix.STDOUT_FILENO, message[total_written..]);
            if (n == 0) return error.UnexpectedEOF;
            total_written += n;
        }
    }
};

// ============================================================================
// LSP Request Handlers
// ============================================================================

pub const LspHandlers = struct {
    lsp_server: *LspServer,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, lsp_server: *LspServer) LspHandlers {
        return LspHandlers{
            .lsp_server = lsp_server,
            .allocator = allocator,
        };
    }
    
    /// Handle initialize request
    pub fn handleInitialize(self: *LspHandlers, params: json.Value) !json.Value {
        _ = params;
        
        // Initialize server
        const result = try self.lsp_server.handleInitialize();
        
        return result;
    }
    
    /// Handle textDocument/didOpen notification
    pub fn handleDidOpen(self: *LspHandlers, params: json.Value) !void {
        const obj = params.object;
        
        const text_doc = obj.get("textDocument") orelse return error.InvalidParams;
        const text_doc_obj = text_doc.object;
        
        const uri = text_doc_obj.get("uri") orelse return error.InvalidParams;
        const language_id = text_doc_obj.get("languageId") orelse return error.InvalidParams;
        const version = text_doc_obj.get("version") orelse return error.InvalidParams;
        const text = text_doc_obj.get("text") orelse return error.InvalidParams;
        
        try self.lsp_server.handleDidOpen(
            uri.string,
            language_id.string,
            @intCast(version.integer),
            text.string,
        );
    }
    
    /// Handle textDocument/didChange notification
    pub fn handleDidChange(self: *LspHandlers, params: json.Value) !void {
        const obj = params.object;
        
        const text_doc = obj.get("textDocument") orelse return error.InvalidParams;
        const text_doc_obj = text_doc.object;
        
        const uri = text_doc_obj.get("uri") orelse return error.InvalidParams;
        const version = text_doc_obj.get("version") orelse return error.InvalidParams;
        
        const content_changes = obj.get("contentChanges") orelse return error.InvalidParams;
        const changes_array = content_changes.array;
        
        if (changes_array.items.len > 0) {
            const first_change = changes_array.items[0].object;
            const text = first_change.get("text") orelse return error.InvalidParams;
            
            try self.lsp_server.handleDidChange(
                uri.string,
                @intCast(version.integer),
                text.string,
            );
        }
    }
    
    /// Handle textDocument/didClose notification
    pub fn handleDidClose(self: *LspHandlers, params: json.Value) !void {
        const obj = params.object;
        
        const text_doc = obj.get("textDocument") orelse return error.InvalidParams;
        const text_doc_obj = text_doc.object;
        
        const uri = text_doc_obj.get("uri") orelse return error.InvalidParams;
        
        try self.lsp_server.handleDidClose(uri.string);
    }
};

// ============================================================================
// Main LSP Server Loop
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    // Initialize components
    var transport = StdioTransport.init(allocator);
    var lsp_server = LspServer.init(allocator);
    defer lsp_server.deinit();
    
    var parser = MessageParser.init(allocator);
    _ = MessageSerializer.init(allocator); // Reserved for future use
    
    var handlers = LspHandlers.init(allocator, &lsp_server);
    
    std.debug.print("[mojo-lsp] Starting LSP server...\n", .{});
    
    // Main message loop
    while (true) {
        // Read message from stdin
        const raw_message = transport.readMessage() catch |err| {
            if (err == error.UnexpectedEOF) break;
            std.debug.print("[mojo-lsp] Error reading message: {any}\n", .{err});
            continue;
        };
        defer allocator.free(raw_message);
        
        std.debug.print("[mojo-lsp] Received: {s}\n", .{raw_message});
        
        // Parse JSON-RPC message
        const parsed = parser.parse(raw_message) catch |err| {
            std.debug.print("[mojo-lsp] Parse error: {any}\n", .{err});
            continue;
        };
        defer parsed.deinit();

        // Handle message
        const response = try handleMessage(&handlers, parsed.message, allocator);
        
        // Send response if present
        if (response) |resp| {
            defer allocator.free(resp);
            
            std.debug.print("[mojo-lsp] Sending: {s}\n", .{resp});
            try transport.writeMessage(resp);
        }
    }
    
    std.debug.print("[mojo-lsp] Server stopped.\n", .{});
}

fn handleMessage(handlers: *LspHandlers, message: JsonRpcMessage, allocator: Allocator) !?[]const u8 {
    var serializer = MessageSerializer.init(allocator);
    
    switch (message) {
        .request => |req| {
            std.debug.print("[mojo-lsp] Request: {s} (id={any})\n", .{ req.method, req.id });
            
            // Route request to appropriate handler
            if (std.mem.eql(u8, req.method, "initialize")) {
                if (req.params) |params| {
                    const result = handlers.handleInitialize(params) catch |err| {
                        std.debug.print("[mojo-lsp] Initialize error: {any}\n", .{err});
                        const error_msg = jsonrpc.JsonRpcError.init(.InternalError, @errorName(err));
                        const error_resp = jsonrpc.ErrorResponse.init(req.id, error_msg);
                        return try serializer.serialize(JsonRpcMessage{ .error_response = error_resp });
                    };
                    
                    const response = jsonrpc.Response.init(req.id, result);
                    return try serializer.serialize(JsonRpcMessage{ .response = response });
                }
            } else if (std.mem.eql(u8, req.method, "shutdown")) {
                // Acknowledge shutdown
                const response = jsonrpc.Response.init(req.id, json.Value{ .null = {} });
                return try serializer.serialize(JsonRpcMessage{ .response = response });
            }
            
            // Unknown method
            const error_msg = jsonrpc.JsonRpcError.init(.MethodNotFound, "Method not found");
            const error_resp = jsonrpc.ErrorResponse.init(req.id, error_msg);
            return try serializer.serialize(JsonRpcMessage{ .error_response = error_resp });
        },
        
        .notification => |notif| {
            std.debug.print("[mojo-lsp] Notification: {s}\n", .{notif.method});
            
            // Route notification to appropriate handler
            if (std.mem.eql(u8, notif.method, "initialized")) {
                // Server is now initialized
                std.debug.print("[mojo-lsp] Server initialized!\n", .{});
            } else if (std.mem.eql(u8, notif.method, "textDocument/didOpen")) {
                if (notif.params) |params| {
                    handlers.handleDidOpen(params) catch |err| {
                        std.debug.print("[mojo-lsp] didOpen error: {any}\n", .{err});
                    };
                }
            } else if (std.mem.eql(u8, notif.method, "textDocument/didChange")) {
                if (notif.params) |params| {
                    handlers.handleDidChange(params) catch |err| {
                        std.debug.print("[mojo-lsp] didChange error: {any}\n", .{err});
                    };
                }
            } else if (std.mem.eql(u8, notif.method, "textDocument/didClose")) {
                if (notif.params) |params| {
                    handlers.handleDidClose(params) catch |err| {
                        std.debug.print("[mojo-lsp] didClose error: {any}\n", .{err});
                    };
                }
            } else if (std.mem.eql(u8, notif.method, "exit")) {
                std.debug.print("[mojo-lsp] Exit notification received\n", .{});
                std.process.exit(0);
            }
            
            // Notifications don't get responses
            return null;
        },
        
        .response, .error_response => {
            // Client responses (shouldn't happen in server)
            std.debug.print("[mojo-lsp] Unexpected response from client\n", .{});
            return null;
        },
    }
}

// ============================================================================
// Tests
// ============================================================================

test "StdioTransport: message format" {
    // Test that we can format messages correctly
    const allocator = std.testing.allocator;
    
    var list = try std.ArrayList(u8).initCapacity(allocator, 100);
    defer list.deinit(allocator);
    
    const message = "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"test\"}";
    
    try std.fmt.format(list.writer(allocator), "Content-Length: {d}\r\n\r\n{s}", .{ message.len, message });
    
    const formatted = try list.toOwnedSlice(allocator);
    defer allocator.free(formatted);
    
    try std.testing.expect(std.mem.indexOf(u8, formatted, "Content-Length:") != null);
    try std.testing.expect(std.mem.indexOf(u8, formatted, message) != null);
}

test "LspHandlers: initialize" {
    var lsp_server = LspServer.init(std.testing.allocator);
    defer lsp_server.deinit();
    
    var handlers = LspHandlers.init(std.testing.allocator, &lsp_server);
    
    const params = json.Value{ .null = {} };
    const result = try handlers.handleInitialize(params);
    
    try std.testing.expect(result == .null);
}
