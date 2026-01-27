//! HTTP Server - Day 29
//!
//! Core HTTP server implementation for nMetaData REST API.
//! Provides the foundation for all API endpoints with request routing,
//! middleware support, and error handling.
//!
//! Key Features:
//! - HTTP/1.1 support
//! - Route-based request handling
//! - Middleware chain execution
//! - JSON request/response handling
//! - Graceful error handling
//! - Connection management
//!
//! Architecture:
//! ```
//! Request → Middleware Chain → Router → Handler → Response
//!             ↓                           ↓
//!         Auth, CORS, etc.          Business Logic
//! ```

const std = @import("std");
const net = std.net;
const http = std.http;
const Allocator = std.mem.Allocator;

const router_mod = @import("router.zig");
const Router = router_mod.Router;
const RouterHandler = router_mod.Handler;
const Request = @import("types.zig").Request;
const Response = @import("types.zig").Response;
const Middleware = @import("middleware.zig").Middleware;
const MiddlewareChain = @import("middleware.zig").MiddlewareChain;

/// HTTP Server configuration
pub const ServerConfig = struct {
    /// Host address to bind to
    host: []const u8 = "127.0.0.1",
    
    /// Port to listen on
    port: u16 = 8080,
    
    /// Maximum request body size (bytes)
    max_body_size: usize = 10 * 1024 * 1024, // 10MB
    
    /// Request timeout (milliseconds)
    request_timeout: u64 = 30000, // 30 seconds
    
    /// Maximum concurrent connections
    max_connections: usize = 1000,
    
    /// Enable request logging
    enable_logging: bool = true,
    
    /// API version prefix
    api_version: []const u8 = "/api/v1",
};

/// HTTP Server state
pub const ServerState = enum {
    stopped,
    starting,
    running,
    stopping,
};

/// HTTP Server instance
pub const Server = struct {
    allocator: Allocator,
    config: ServerConfig,
    router: Router,
    middleware_chain: MiddlewareChain,
    state: ServerState,
    listener: ?net.Server,
    
    /// Initialize new HTTP server
    pub fn init(allocator: Allocator, config: ServerConfig) !*Server {
        const server = try allocator.create(Server);
        errdefer allocator.destroy(server);
        
        server.* = Server{
            .allocator = allocator,
            .config = config,
            .router = Router.init(allocator),
            .middleware_chain = MiddlewareChain.init(allocator),
            .state = .stopped,
            .listener = null,
        };
        
        return server;
    }
    
    /// Clean up server resources
    pub fn deinit(self: *Server) void {
        if (self.listener) |*listener| {
            listener.deinit();
        }
        self.router.deinit();
        self.middleware_chain.deinit();
        self.allocator.destroy(self);
    }
    
    /// Add middleware to the chain
    pub fn use(self: *Server, middleware: Middleware) !void {
        try self.middleware_chain.add(middleware);
    }
    
    /// Register a route handler
    pub fn route(
        self: *Server,
        method: http.Method,
        path: []const u8,
        handler: RouterHandler,
    ) !void {
        try self.router.addRoute(method, path, handler);
    }
    
    /// Start the HTTP server
    pub fn start(self: *Server) !void {
        if (self.state != .stopped) {
            return error.ServerAlreadyRunning;
        }
        
        self.state = .starting;
        
        // Create address
        const address = try net.Address.parseIp(self.config.host, self.config.port);
        
        // Start listening
        self.listener = try address.listen(.{
            .reuse_address = true,
        });
        
        self.state = .running;
        
        if (self.config.enable_logging) {
            std.debug.print("🚀 nMetaData Server running at http://{s}:{d}\n", .{
                self.config.host,
                self.config.port,
            });
            std.debug.print("📋 API Base: {s}\n", .{self.config.api_version});
        }
    }
    
    /// Stop the HTTP server
    pub fn stop(self: *Server) void {
        if (self.state != .running) {
            return;
        }
        
        self.state = .stopping;
        
        if (self.listener) |*listener| {
            listener.deinit();
            self.listener = null;
        }
        
        self.state = .stopped;
        
        if (self.config.enable_logging) {
            std.debug.print("🛑 nMetaData Server stopped\n", .{});
        }
    }
    
    /// Accept and handle incoming connections
    pub fn serve(self: *Server) !void {
        if (self.state != .running or self.listener == null) {
            return error.ServerNotRunning;
        }
        
        while (self.state == .running) {
            // Accept connection
            const connection = self.listener.?.accept() catch |err| {
                if (self.config.enable_logging) {
                    std.debug.print("Failed to accept connection: {any}\n", .{err});
                }
                continue;
            };
            
            // Handle connection (synchronous for now, could spawn threads)
            self.handleConnection(connection) catch |err| {
                if (self.config.enable_logging) {
                    std.debug.print("Error handling connection: {any}\n", .{err});
                }
            };
        }
    }
    
    /// Handle a single connection
    fn handleConnection(self: *Server, connection: net.Server.Connection) !void {
        defer connection.stream.close();
        
        var arena = std.heap.ArenaAllocator.init(self.allocator);
        defer arena.deinit();
        const arena_allocator = arena.allocator();
        
        // Read HTTP request
        var read_buffer: [8192]u8 = undefined;
        var reader = connection.stream.reader(read_buffer[0..]);
        const reader_io = reader.interface();
        var write_buffer: [8192]u8 = undefined;
        var writer = connection.stream.writer(write_buffer[0..]);
        var http_server = http.Server.init(reader_io, &writer.interface);
        var request = http_server.receiveHead() catch |err| {
            try self.sendErrorResponse(connection.stream, 400, "Bad Request");
            return err;
        };
        
        // Parse request
        var req = try self.parseRequest(arena_allocator, &request);
        
        // Create response
        var resp = Response.init(arena_allocator);
        
        // Execute middleware chain
        const should_continue = try self.middleware_chain.execute(&req, &resp);
        
        if (should_continue) {
            // Route to handler
            self.router.handle(&req, &resp) catch |err| {
                if (self.config.enable_logging) {
                    std.debug.print("Handler error: {any}\n", .{err});
                }
                resp.status = 500;
                const ErrorResponse = struct {
                    @"error": []const u8,
                };
                try resp.json(ErrorResponse{ .@"error" = "Internal Server Error" });
            };
        }
        
        // Send response
        try self.sendResponse(connection.stream, &resp);
    }
    
    /// Parse HTTP request into our Request type
    fn parseRequest(
        self: *Server,
        allocator: Allocator,
        http_request: *http.Server.Request,
    ) !Request {
        
        var req = Request.init(allocator);
        
        // Copy method
        req.method = http_request.head.method;
        
        // Copy path
        req.path = try allocator.dupe(u8, http_request.head.target);
        
        // Parse query string
        if (std.mem.indexOf(u8, req.path, "?")) |query_start| {
            const query_string = req.path[query_start + 1 ..];
            req.path = req.path[0..query_start];
            try req.parseQuery(query_string);
        }
        
        // Copy headers
        var header_iter = http_request.iterateHeaders();
        while (header_iter.next()) |header| {
            try req.headers.put(
                try allocator.dupe(u8, header.name),
                try allocator.dupe(u8, header.value),
            );
        }
        
        // Read body if present
        if (http_request.head.content_length) |length| {
            if (length > 0 and length <= self.config.max_body_size) {
                const body_buffer = try allocator.alloc(u8, @intCast(length));
                var transfer_buffer: [4096]u8 = undefined;
                const body_reader = http_request.server.reader.bodyReader(
                    transfer_buffer[0..],
                    http_request.head.transfer_encoding,
                    http_request.head.content_length,
                );
                try body_reader.readSliceAll(body_buffer);
                req.body = body_buffer;
            }
        }
        
        return req;
    }
    
    /// Send HTTP response
    fn sendResponse(self: *Server, stream: net.Stream, response: *Response) !void {
        _ = self;
        
        var buf: [4096]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const writer = fbs.writer();
        
        // Status line
        try writer.print("HTTP/1.1 {d} {s}\r\n", .{
            response.status,
            response.getStatusText(),
        });
        
        // Headers
        var header_iter = response.headers.iterator();
        while (header_iter.next()) |entry| {
            try writer.print("{s}: {s}\r\n", .{ entry.key_ptr.*, entry.value_ptr.* });
        }
        
        // Content-Length header
        if (response.body) |body| {
            try writer.print("Content-Length: {d}\r\n", .{body.len});
        }
        
        // End of headers
        try writer.writeAll("\r\n");
        
        // Write headers
        try stream.writeAll(fbs.getWritten());
        
        // Write body
        if (response.body) |body| {
            try stream.writeAll(body);
        }
    }
    
    /// Send error response
    fn sendErrorResponse(self: *Server, stream: net.Stream, status: u16, message: []const u8) !void {
        _ = self;
        
        var buf: [1024]u8 = undefined;
        const response = try std.fmt.bufPrint(&buf,
            \\HTTP/1.1 {d} {s}
            \\Content-Type: text/plain
            \\Content-Length: {d}
            \\
            \\{s}
        , .{ status, message, message.len, message });
        
        try stream.writeAll(response);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Server: init and deinit" {
    const allocator = std.testing.allocator;
    
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 8888,
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    try std.testing.expectEqual(ServerState.stopped, server.state);
    try std.testing.expectEqualStrings("127.0.0.1", server.config.host);
    try std.testing.expectEqual(@as(u16, 8888), server.config.port);
}

test "Server: configuration defaults" {
    const config = ServerConfig{};
    
    try std.testing.expectEqualStrings("127.0.0.1", config.host);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
    try std.testing.expectEqual(@as(usize, 10 * 1024 * 1024), config.max_body_size);
    try std.testing.expectEqual(@as(u64, 30000), config.request_timeout);
    try std.testing.expect(config.enable_logging);
}

test "Server: add middleware" {
    const allocator = std.testing.allocator;
    
    const config = ServerConfig{};
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    const middleware = Middleware{
        .name = "test",
        .handler = undefined,
    };
    
    try server.use(middleware);
    try std.testing.expectEqual(@as(usize, 1), server.middleware_chain.middlewares.items.len);
}

test "Server: register routes" {
    const allocator = std.testing.allocator;
    
    const config = ServerConfig{};
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
        }
    }.handle;
    
    try server.route(.GET, "/test", handler);
    try std.testing.expectEqual(@as(usize, 1), server.router.routes.items.len);
}
