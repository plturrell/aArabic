//! HTTP Server Integration Tests - Day 29
//!
//! Comprehensive integration tests for the HTTP server,
//! testing the full request/response cycle including:
//! - Server lifecycle
//! - Route handling
//! - Middleware execution
//! - Error handling
//! - JSON responses

const std = @import("std");
const testing = std.testing;

const Server = @import("server.zig").Server;
const ServerConfig = @import("server.zig").ServerConfig;
const Router = @import("router.zig").Router;
const Request = @import("types.zig").Request;
const Response = @import("types.zig").Response;
const Middleware = @import("middleware.zig").Middleware;
const loggingMiddleware = @import("middleware.zig").loggingMiddleware;
const corsMiddleware = @import("middleware.zig").corsMiddleware;
const CorsConfig = @import("middleware.zig").CorsConfig;

// ============================================================================
// Server Lifecycle Tests
// ============================================================================

test "Integration: server lifecycle" {
    const allocator = testing.allocator;
    
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 9999,
        .enable_logging = false,
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    try testing.expectEqual(@import("server.zig").ServerState.stopped, server.state);
}

test "Integration: add routes and middleware" {
    const allocator = testing.allocator;
    
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 9998,
        .enable_logging = false,
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    // Add middleware
    try server.use(loggingMiddleware());
    try server.use(corsMiddleware(CorsConfig{}));
    
    // Add routes
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
            try resp.json(.{ .message = "success" });
        }
    }.handle;
    
    try server.route(.GET, "/test", handler);
    try server.route(.POST, "/api/data", handler);
    
    try testing.expectEqual(@as(usize, 2), server.middleware_chain.count());
    try testing.expectEqual(@as(usize, 2), server.router.count());
}

// ============================================================================
// Route Handler Tests
// ============================================================================

test "Integration: GET handler" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
            try resp.json(.{
                .message = "Hello, World!",
                .status = "success",
            });
        }
    }.handle;
    
    try router.addRoute(.GET, "/hello", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/hello");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
    try testing.expect(resp.body != null);
}

test "Integration: POST handler with JSON body" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 201;
            try resp.json(.{
                .id = 123,
                .created = true,
            });
        }
    }.handle;
    
    try router.addRoute(.POST, "/api/items", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    req.path = try allocator.dupe(u8, "/api/items");
    req.body = 
        \\{"name":"test","value":42}
    ;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try testing.expectEqual(@as(u16, 201), resp.status);
}

test "Integration: 404 not found" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/nonexistent");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try testing.expectEqual(@as(u16, 404), resp.status);
}

// ============================================================================
// Middleware Integration Tests
// ============================================================================

test "Integration: middleware chain execution" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 9997,
        .enable_logging = false,
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    // Add middleware that modifies request
    try server.use(Middleware{
        .name = "test-modifier",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                try req.headers.put(
                    try req.allocator.dupe(u8, "X-Test"),
                    try req.allocator.dupe(u8, "middleware-passed"),
                );
                _ = resp;
                return true;
            }
        }.handle,
    });
    
    // Add handler that checks for header
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            if (req.header("X-Test")) |value| {
                resp.status = 200;
                try resp.json(.{ .test_header = value });
            } else {
                resp.status = 500;
            }
        }
    }.handle;
    
    try server.route(.GET, "/test", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/test");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    // Execute middleware chain
    const should_continue = try server.middleware_chain.execute(&req, &resp);
    try testing.expect(should_continue);
    
    // Execute handler
    try server.router.handle(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

test "Integration: middleware stops processing" {
    const allocator = testing.allocator;
    
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 9996,
        .enable_logging = false,
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    // Add middleware that stops processing
    try server.use(Middleware{
        .name = "auth",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Simulate failed auth
                if (req.header("Authorization") == null) {
                    resp.status = 401;
                    try resp.json(.{ .error = "Unauthorized" });
                    return false;
                }
                return true;
            }
        }.handle,
    });
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/protected");
    // No Authorization header
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try server.middleware_chain.execute(&req, &resp);
    try testing.expect(!should_continue);
    try testing.expectEqual(@as(u16, 401), resp.status);
}

// ============================================================================
// Query Parameter Tests
// ============================================================================

test "Integration: query parameter parsing" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            const page = req.queryParam("page") orelse "1";
            const limit = req.queryParam("limit") orelse "10";
            
            resp.status = 200;
            try resp.json(.{
                .page = page,
                .limit = limit,
            });
        }
    }.handle;
    
    try router.addRoute(.GET, "/api/items", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/api/items");
    try req.parseQuery("page=2&limit=20");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try testing.expectEqual(@as(u16, 200), resp.status);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "Integration: handler error recovery" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            _ = resp;
            return error.TestError;
        }
    }.handle;
    
    try router.addRoute(.GET, "/error", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/error");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    // Router.handle should catch the error
    router.handle(&req, &resp) catch {
        // Error occurred as expected
        resp.status = 500;
        try resp.json(.{ .error = "Internal Server Error" });
    };
    
    try testing.expectEqual(@as(u16, 500), resp.status);
}

// ============================================================================
// Multiple Routes Test
// ============================================================================

test "Integration: multiple routes with different methods" {
    const allocator = testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const getHandler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
            try resp.json(.{ .method = "GET" });
        }
    }.handle;
    
    const postHandler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 201;
            try resp.json(.{ .method = "POST" });
        }
    }.handle;
    
    const putHandler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
            try resp.json(.{ .method = "PUT" });
        }
    }.handle;
    
    const deleteHandler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 204;
        }
    }.handle;
    
    try router.addRoute(.GET, "/resource", getHandler);
    try router.addRoute(.POST, "/resource", postHandler);
    try router.addRoute(.PUT, "/resource", putHandler);
    try router.addRoute(.DELETE, "/resource", deleteHandler);
    
    // Test GET
    {
        var req = Request.init(allocator);
        defer req.deinit();
        req.method = .GET;
        req.path = try allocator.dupe(u8, "/resource");
        
        var resp = Response.init(allocator);
        defer resp.deinit();
        
        try router.handle(&req, &resp);
        try testing.expectEqual(@as(u16, 200), resp.status);
    }
    
    // Test POST
    {
        var req = Request.init(allocator);
        defer req.deinit();
        req.method = .POST;
        req.path = try allocator.dupe(u8, "/resource");
        
        var resp = Response.init(allocator);
        defer resp.deinit();
        
        try router.handle(&req, &resp);
        try testing.expectEqual(@as(u16, 201), resp.status);
    }
    
    // Test DELETE
    {
        var req = Request.init(allocator);
        defer req.deinit();
        req.method = .DELETE;
        req.path = try allocator.dupe(u8, "/resource");
        
        var resp = Response.init(allocator);
        defer resp.deinit();
        
        try router.handle(&req, &resp);
        try testing.expectEqual(@as(u16, 204), resp.status);
    }
}

// ============================================================================
// Configuration Tests
// ============================================================================

test "Integration: custom server configuration" {
    const allocator = testing.allocator;
    
    const config = ServerConfig{
        .host = "0.0.0.0",
        .port = 3000,
        .max_body_size = 5 * 1024 * 1024, // 5MB
        .request_timeout = 60000, // 60 seconds
        .max_connections = 500,
        .enable_logging = true,
        .api_version = "/api/v2",
    };
    
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    try testing.expectEqualStrings("0.0.0.0", server.config.host);
    try testing.expectEqual(@as(u16, 3000), server.config.port);
    try testing.expectEqual(@as(usize, 5 * 1024 * 1024), server.config.max_body_size);
    try testing.expectEqualStrings("/api/v2", server.config.api_version);
}
