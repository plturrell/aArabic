//! Router - Day 29
//!
//! HTTP request router with pattern matching and parameter extraction.
//! Supports dynamic path parameters and method-based routing.
//!
//! Key Features:
//! - Method-based routing (GET, POST, PUT, DELETE, etc.)
//! - Path parameter extraction (/users/:id)
//! - Wildcard matching (/api/*)
//! - Route priority handling
//! - 404 handling
//!
//! Example:
//! ```zig
//! var router = Router.init(allocator);
//! try router.addRoute(.GET, "/users/:id", getUserHandler);
//! try router.addRoute(.POST, "/users", createUserHandler);
//! router.handle(&request, &response);
//! ```

const std = @import("std");
const http = std.http;
const Allocator = std.mem.Allocator;

const Request = @import("types.zig").Request;
const Response = @import("types.zig").Response;

/// Handler function signature
pub const Handler = *const fn (req: *Request, resp: *Response) anyerror!void;

/// Route definition
pub const Route = struct {
    method: http.Method,
    path: []const u8,
    handler: Handler,
    
    /// Check if this route matches the request
    pub fn matches(self: *const Route, method: http.Method, path: []const u8) bool {
        if (self.method != method) {
            return false;
        }
        
        return matchPath(self.path, path);
    }
};

/// Match a path pattern against a request path
fn matchPath(pattern: []const u8, path: []const u8) bool {
    // Simple exact match for now
    // TODO: Implement parameter extraction (:id) and wildcards (*)
    return std.mem.eql(u8, pattern, path);
}

/// Extract parameters from path
fn extractParams(
    allocator: Allocator,
    pattern: []const u8,
    path: []const u8,
) !std.StringHashMap([]const u8) {
    var params = std.StringHashMap([]const u8).init(allocator);
    
    // Split pattern and path by '/'
    var pattern_iter = std.mem.splitScalar(u8, pattern, '/');
    var path_iter = std.mem.splitScalar(u8, path, '/');
    
    while (pattern_iter.next()) |pattern_segment| {
        const path_segment = path_iter.next() orelse break;
        
        // Check for parameter (starts with ':')
        if (pattern_segment.len > 0 and pattern_segment[0] == ':') {
            const param_name = pattern_segment[1..];
            try params.put(
                try allocator.dupe(u8, param_name),
                try allocator.dupe(u8, path_segment),
            );
        }
    }
    
    return params;
}

/// HTTP Router
pub const Router = struct {
    allocator: Allocator,
    routes: std.ArrayList(Route),
    
    /// Initialize new router
    pub fn init(allocator: Allocator) Router {
        return Router{
            .allocator = allocator,
            .routes = std.ArrayList(Route).init(allocator),
        };
    }
    
    /// Clean up router resources
    pub fn deinit(self: *Router) void {
        self.routes.deinit();
    }
    
    /// Add a route to the router
    pub fn addRoute(
        self: *Router,
        method: http.Method,
        path: []const u8,
        handler: Handler,
    ) !void {
        const route = Route{
            .method = method,
            .path = try self.allocator.dupe(u8, path),
            .handler = handler,
        };
        
        try self.routes.append(route);
    }
    
    /// Handle incoming request
    pub fn handle(self: *Router, req: *Request, resp: *Response) !void {
        // Find matching route
        for (self.routes.items) |*route| {
            if (route.matches(req.method, req.path)) {
                // Extract path parameters
                const params = try extractParams(self.allocator, route.path, req.path);
                defer {
                    var iter = params.iterator();
                    while (iter.next()) |entry| {
                        self.allocator.free(entry.key_ptr.*);
                        self.allocator.free(entry.value_ptr.*);
                    }
                    params.deinit();
                }
                
                // Merge params into request
                var param_iter = params.iterator();
                while (param_iter.next()) |entry| {
                    try req.params.put(
                        try self.allocator.dupe(u8, entry.key_ptr.*),
                        try self.allocator.dupe(u8, entry.value_ptr.*),
                    );
                }
                
                // Call handler
                try route.handler(req, resp);
                return;
            }
        }
        
        // No route found - 404
        resp.status = 404;
        const NotFoundResponse = struct {
            @"error": []const u8,
            path: []const u8,
        };
        try resp.json(NotFoundResponse{
            .@"error" = "Not Found",
            .path = req.path,
        });
    }
    
    /// Get route by method and path
    pub fn getRoute(self: *Router, method: http.Method, path: []const u8) ?*Route {
        for (self.routes.items) |*route| {
            if (route.matches(method, path)) {
                return route;
            }
        }
        return null;
    }
    
    /// Remove a route
    pub fn removeRoute(self: *Router, method: http.Method, path: []const u8) bool {
        for (self.routes.items, 0..) |route, i| {
            if (route.method == method and std.mem.eql(u8, route.path, path)) {
                self.allocator.free(route.path);
                _ = self.routes.orderedRemove(i);
                return true;
            }
        }
        return false;
    }
    
    /// Get number of registered routes
    pub fn count(self: *const Router) usize {
        return self.routes.items.len;
    }
    
    /// Check if router has any routes
    pub fn isEmpty(self: *const Router) bool {
        return self.routes.items.len == 0;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Router: init and deinit" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    try std.testing.expect(router.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), router.count());
}

test "Router: add route" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
        }
    }.handle;
    
    try router.addRoute(.GET, "/test", handler);
    
    try std.testing.expectEqual(@as(usize, 1), router.count());
    try std.testing.expect(!router.isEmpty());
}

test "Router: route matching" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
        }
    }.handle;
    
    try router.addRoute(.GET, "/users", handler);
    try router.addRoute(.POST, "/users", handler);
    try router.addRoute(.GET, "/posts", handler);
    
    const route1 = router.getRoute(.GET, "/users");
    try std.testing.expect(route1 != null);
    try std.testing.expectEqual(http.Method.GET, route1.?.method);
    
    const route2 = router.getRoute(.POST, "/users");
    try std.testing.expect(route2 != null);
    try std.testing.expectEqual(http.Method.POST, route2.?.method);
    
    const route3 = router.getRoute(.DELETE, "/users");
    try std.testing.expect(route3 == null);
}

test "Router: handle request" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
            try resp.json(.{ .message = "success" });
        }
    }.handle;
    
    try router.addRoute(.GET, "/test", handler);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/test");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try std.testing.expectEqual(@as(u16, 200), resp.status);
}

test "Router: 404 not found" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/nonexistent");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    try router.handle(&req, &resp);
    
    try std.testing.expectEqual(@as(u16, 404), resp.status);
}

test "Router: remove route" {
    const allocator = std.testing.allocator;
    
    var router = Router.init(allocator);
    defer router.deinit();
    
    const handler = struct {
        fn handle(req: *Request, resp: *Response) !void {
            _ = req;
            resp.status = 200;
        }
    }.handle;
    
    try router.addRoute(.GET, "/test", handler);
    try std.testing.expectEqual(@as(usize, 1), router.count());
    
    const removed = router.removeRoute(.GET, "/test");
    try std.testing.expect(removed);
    try std.testing.expectEqual(@as(usize, 0), router.count());
}

test "Router: path matching exact" {
    try std.testing.expect(matchPath("/users", "/users"));
    try std.testing.expect(!matchPath("/users", "/posts"));
    try std.testing.expect(!matchPath("/users", "/users/123"));
}

test "Router: extract params" {
    const allocator = std.testing.allocator;
    
    const params = try extractParams(allocator, "/users/:id/posts/:postId", "/users/123/posts/456");
    defer {
        var iter = params.iterator();
        while (iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
            allocator.free(entry.value_ptr.*);
        }
        params.deinit();
    }
    
    try std.testing.expectEqualStrings("123", params.get("id").?);
    try std.testing.expectEqualStrings("456", params.get("postId").?);
}
