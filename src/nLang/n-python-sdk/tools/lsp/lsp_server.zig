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
        const stdin_file = std.fs.File.stdin();
        const stdout_file = std.fs.File.stdout();

        // Input buffer for headers
        var header_buffer: [1024]u8 = undefined;

        while (self.running) {
            // 1. Read Content-Length header
            const content_length = try readContentLength(stdin_file, &header_buffer) orelse break; // EOF

            // 2. Read Body
            const body = try self.allocator.alloc(u8, content_length);
            defer self.allocator.free(body);

            const bytes_read = try stdin_file.readAll(body);
            if (bytes_read != content_length) {
                break; // EOF or error
            }

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

                // Write header using File.writeAll
                var header_buf: [64]u8 = undefined;
                const header = std.fmt.bufPrint(&header_buf, "Content-Length: {d}\r\n\r\n", .{serialized.len}) catch unreachable;
                try stdout_file.writeAll(header);
                try stdout_file.writeAll(serialized);
            }
        }
    }
};

// Handlers ...
fn handleInitialize(ptr: *anyopaque, request: jsonrpc.Request) anyerror!json.Value {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = request; // Read capabilities if needed

    // Return ServerCapabilities
    // Simple response: { capabilities: { textDocumentSync: 1 } }
    // We construct a JSON value tree using std.json stuff?
    // Or just return a struct that can be serialized.
    // Ensure jsonrpc.Response.result is json.Value.

    // For now, return empty object (dummy)
    // In Zig 0.15, ObjectMap needs allocator
    const obj = json.ObjectMap.init(self.allocator);
    return json.Value{ .object = obj };
}

fn handleShutdown(ptr: *anyopaque, request: jsonrpc.Request) anyerror!json.Value {
    const self: *LspServer = @ptrCast(@alignCast(ptr));
    _ = self;
    _ = request;
    return json.Value.null;
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

// Helper to read content length using File.read() directly
fn readContentLength(file: std.fs.File, buffer: []u8) !?usize {
    // Read line by line until empty line
    var length: ?usize = null;
    var line_start: usize = 0;
    var byte_buf: [1]u8 = undefined;

    while (true) {
        // Read one byte at a time to find newline
        const bytes_read = file.read(&byte_buf) catch |err| {
            return err;
        };

        if (bytes_read == 0) return null; // EOF

        const byte = byte_buf[0];

        if (byte == '\n') {
            // Got a line - trim trailing \r
            var line_end = line_start;
            if (line_end > 0 and buffer[line_end - 1] == '\r') {
                line_end -= 1;
            }

            const line = buffer[0..line_end];

            if (line.len == 0) {
                // Empty line, end of headers
                return length;
            }

            if (std.mem.startsWith(u8, line, "Content-Length: ")) {
                const num_str = line["Content-Length: ".len..];
                length = try std.fmt.parseInt(usize, num_str, 10);
            }

            line_start = 0;
        } else {
            if (line_start < buffer.len) {
                buffer[line_start] = byte;
                line_start += 1;
            }
        }
    }
}
