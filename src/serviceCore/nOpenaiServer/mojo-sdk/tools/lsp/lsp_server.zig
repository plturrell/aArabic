// Mojo SDK - LSP Server
// Day 113: LSP Foundation

const std = @import("std");
const jsonrpc = @import("jsonrpc.zig");
const json = std.json;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var server = LspServer.init(allocator);
    defer server.deinit();

    try server.registerHandlers();
    try server.run();
}

const LspServer = struct {
    allocator: std.mem.Allocator,
    router: jsonrpc.MessageRouter,
    running: bool,
    initialized: bool,

    const Self = @This();

    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .router = jsonrpc.MessageRouter.init(allocator),
            .running = true,
            .initialized = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.router.deinit();
    }

    pub fn registerHandlers(self: *Self) !void {
        // We use a wrapper struct to pass 'self' context if needed, but for now strict function pointers
        // The MessageHandler in jsonrpc.zig uses *anyopaque, so we can pass self.

        try self.router.registerHandler("initialize", .{
            .ptr = self,
            .handleRequestFn = handleInitialize,
            .handleNotificationFn = undefined, // Requests don't use this
        });

        try self.router.registerHandler("shutdown", .{
            .ptr = self,
            .handleRequestFn = handleShutdown,
            .handleNotificationFn = undefined,
        });

        try self.router.registerHandler("exit", .{
            .ptr = self,
            .handleRequestFn = undefined, // Notification only
            .handleNotificationFn = handleExit,
        });

        try self.router.registerHandler("initialized", .{
            .ptr = self,
            .handleRequestFn = undefined,
            .handleNotificationFn = handleInitialized,
        });
    }

    pub fn run(self: *Self) !void {
        const stdin = std.fs.File.stdin().reader();
        const stdout = std.fs.File.stdout().writer();

        // Input buffer
        var header_buffer: [1024]u8 = undefined;

        while (self.running) {
            // 1. Read Content-Length header
            const content_length = try readContentLength(stdin, &header_buffer) orelse break; // EOF

            // 2. Read Body
            const body = try self.allocator.alloc(u8, content_length);
            defer self.allocator.free(body);

            try stdin.readNoEof(body);

            // 3. Parse
            var parser = jsonrpc.MessageParser.init(self.allocator);
            var parsed_msg: jsonrpc.ParsedMessage = undefined;

            parsed_msg = parser.parse(body) catch |err| {
                // If parse error, we should send error response if possible, but without ID we can't do much
                // unless we can parse enough to get ID.
                // For now, log to stderr
                std.debug.print("LSP Parse Error: {}\n", .{err});
                continue;
            };
            defer parsed_msg.deinit();

            // 4. Route
            const response_msg = self.router.route(parsed_msg.message) catch |err| {
                std.debug.print("LSP Route Error: {}\n", .{err});
                continue;
            };

            // 5. Send Response
            if (response_msg) |resp| {
                var serializer = jsonrpc.MessageSerializer.init(self.allocator);
                const serialized = try serializer.serialize(resp);
                defer self.allocator.free(serialized);

                try stdout.print("Content-Length: {}\r\n\r\n{s}", .{ serialized.len, serialized });
            }
        }
    }
};

// Handlers ...
fn handleInitialize(ptr: *anyopaque, request: jsonrpc.Request) anyerror!json.Value {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = self;
    _ = request; // Read capabilities if needed

    // Return ServerCapabilities
    // Simple response: { capabilities: { textDocumentSync: 1 } }
    // We construct a JSON value tree using std.json stuff?
    // Or just return a struct that can be serialized.
    // Ensure jsonrpc.Response.result is json.Value.

    // For now, return empty object (dummy)
    // We really need a way to construct json.Value easily.
    return json.Value{ .object = .{} };
}

fn handleShutdown(ptr: *anyopaque, request: jsonrpc.Request) anyerror!json.Value {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = self;
    _ = request;
    return json.Value{ .null = {} };
}

fn handleExit(ptr: *anyopaque, notification: jsonrpc.Notification) anyerror!void {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = notification;
    self.running = false;
}

fn handleInitialized(ptr: *anyopaque, notification: jsonrpc.Notification) anyerror!void {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = notification;
    self.initialized = true;
}

// Helper to read content length
fn readContentLength(reader: anytype, buffer: []u8) !?usize {
    // Read line by line until empty line
    var length: ?usize = null;

    while (true) {
        const line = try reader.readUntilDelimiterOrEof(buffer, '\n') orelse return null;
        // Handle CR
        const trimmed = std.mem.trimRight(u8, line, "\r");

        if (trimmed.len == 0) {
            // Empty line, end of headers
            return length;
        }

        if (std.mem.startsWith(u8, trimmed, "Content-Length: ")) {
            const num_str = trimmed["Content-Length: ".len..];
            length = try std.fmt.parseInt(usize, num_str, 10);
        }
    }
}
