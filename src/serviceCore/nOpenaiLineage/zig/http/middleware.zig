//! Middleware - Day 29
//!
//! Middleware chain for HTTP request processing.
//! Provides a flexible way to add cross-cutting concerns like:
//! - Request logging
//! - CORS handling
//! - Authentication/Authorization
//! - Error handling
//! - Request timing
//! - Rate limiting
//!
//! Architecture:
//! ```
//! Request → Middleware 1 → Middleware 2 → ... → Handler
//!             ↓                ↓                    ↓
//!         Auth Check      CORS Headers        Business Logic
//! ```
//!
//! Example:
//! ```zig
//! var chain = MiddlewareChain.init(allocator);
//! try chain.add(loggingMiddleware());
//! try chain.add(corsMiddleware());
//! try chain.add(authMiddleware());
//! const should_continue = try chain.execute(&req, &resp);
//! ```

const std = @import("std");
const Allocator = std.mem.Allocator;

const Request = @import("types.zig").Request;
const Response = @import("types.zig").Response;

/// Middleware handler function
/// Returns true if request should continue to next middleware/handler
/// Returns false if request has been handled (e.g., auth failed)
pub const MiddlewareHandler = *const fn (req: *Request, resp: *Response) anyerror!bool;

/// Middleware definition
pub const Middleware = struct {
    name: []const u8,
    handler: MiddlewareHandler,
};

/// Middleware chain executor
pub const MiddlewareChain = struct {
    allocator: Allocator,
    middlewares: std.ArrayList(Middleware),
    
    /// Initialize new middleware chain
    pub fn init(allocator: Allocator) MiddlewareChain {
        return MiddlewareChain{
            .allocator = allocator,
            .middlewares = std.ArrayList(Middleware).init(allocator),
        };
    }
    
    /// Clean up middleware chain
    pub fn deinit(self: *MiddlewareChain) void {
        self.middlewares.deinit();
    }
    
    /// Add middleware to chain
    pub fn add(self: *MiddlewareChain, middleware: Middleware) !void {
        try self.middlewares.append(middleware);
    }
    
    /// Execute middleware chain
    /// Returns true if request should continue to handler
    pub fn execute(self: *MiddlewareChain, req: *Request, resp: *Response) !bool {
        for (self.middlewares.items) |middleware| {
            const should_continue = try middleware.handler(req, resp);
            if (!should_continue) {
                return false;
            }
        }
        return true;
    }
    
    /// Get number of middlewares in chain
    pub fn count(self: *const MiddlewareChain) usize {
        return self.middlewares.items.len;
    }
};

// ============================================================================
// Built-in Middlewares
// ============================================================================

/// Logging middleware - logs all requests
pub fn loggingMiddleware() Middleware {
    return Middleware{
        .name = "logging",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = resp;
                std.debug.print("[{s}] {s}\n", .{ @tagName(req.method), req.path });
                return true;
            }
        }.handle,
    };
}

/// CORS middleware configuration
pub const CorsConfig = struct {
    allow_origin: []const u8 = "*",
    allow_methods: []const u8 = "GET,POST,PUT,DELETE,PATCH,OPTIONS",
    allow_headers: []const u8 = "Content-Type,Authorization",
    max_age: u32 = 86400, // 24 hours
};

/// Create CORS middleware with config
pub fn corsMiddleware(config: CorsConfig) Middleware {
    return Middleware{
        .name = "cors",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Set CORS headers
                try resp.setHeader("Access-Control-Allow-Origin", config.allow_origin);
                try resp.setHeader("Access-Control-Allow-Methods", config.allow_methods);
                try resp.setHeader("Access-Control-Allow-Headers", config.allow_headers);
                
                const max_age_str = try std.fmt.allocPrint(
                    resp.allocator,
                    "{d}",
                    .{config.max_age},
                );
                try resp.setHeader("Access-Control-Max-Age", max_age_str);
                
                // Handle preflight OPTIONS request
                if (req.method == .OPTIONS) {
                    resp.status = 204;
                    return false; // Stop processing
                }
                
                return true;
            }
        }.handle,
    };
}

/// Request timing middleware
pub fn timingMiddleware() Middleware {
    return Middleware{
        .name = "timing",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = req;
                const start_time = std.time.milliTimestamp();
                
                // Store start time in response context (simplified for now)
                _ = start_time;
                _ = resp;
                
                // In a real implementation, we'd add timing info to response headers
                // after the handler completes
                
                return true;
            }
        }.handle,
    };
}

/// Content-Type validation middleware
pub fn contentTypeMiddleware(required_type: []const u8) Middleware {
    return Middleware{
        .name = "content-type",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Skip for GET requests
                if (req.method == .GET or req.method == .HEAD or req.method == .DELETE) {
                    return true;
                }
                
                if (req.header("Content-Type")) |content_type| {
                    if (std.mem.indexOf(u8, content_type, required_type) != null) {
                        return true;
                    }
                }
                
                // Invalid or missing content type
                resp.status = 415; // Unsupported Media Type
                const ErrorResponse = struct {
                    @"error": []const u8,
                    expected: []const u8,
                };
                try resp.json(ErrorResponse{
                    .@"error" = "Unsupported Media Type",
                    .expected = required_type,
                });
                return false;
            }
        }.handle,
    };
}

/// Error recovery middleware
pub fn errorRecoveryMiddleware() Middleware {
    return Middleware{
        .name = "error-recovery",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = req;
                _ = resp;
                // This middleware would wrap subsequent processing in error handling
                // For now, just pass through
                return true;
            }
        }.handle,
    };
}

/// Health check middleware - responds to /health endpoint
pub fn healthCheckMiddleware() Middleware {
    return Middleware{
        .name = "health-check",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                if (std.mem.eql(u8, req.path, "/health")) {
                    resp.status = 200;
                    const HealthResponse = struct {
                        status: []const u8,
                        timestamp: i64,
                    };
                    try resp.json(HealthResponse{
                        .status = "healthy",
                        .timestamp = std.time.timestamp(),
                    });
                    return false; // Stop processing
                }
                return true;
            }
        }.handle,
    };
}

/// Request ID middleware - adds unique ID to each request
pub fn requestIdMiddleware() Middleware {
    return Middleware{
        .name = "request-id",
        .handler = struct {
            var counter: std.atomic.Value(u64) = std.atomic.Value(u64).init(0);
            
            fn handle(req: *Request, resp: *Response) !bool {
                const id = counter.fetchAdd(1, .monotonic);
                
                const id_str = try std.fmt.allocPrint(
                    req.allocator,
                    "req-{d}",
                    .{id},
                );
                
                try req.headers.put(
                    try req.allocator.dupe(u8, "X-Request-ID"),
                    id_str,
                );
                
                try resp.setHeader("X-Request-ID", id_str);
                
                return true;
            }
        }.handle,
    };
}

// ============================================================================
// Tests
// ============================================================================

test "MiddlewareChain: init and deinit" {
    const allocator = std.testing.allocator;
    
    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), chain.count());
}

test "MiddlewareChain: add middleware" {
    const allocator = std.testing.allocator;
    
    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();
    
    try chain.add(loggingMiddleware());
    try chain.add(timingMiddleware());
    
    try std.testing.expectEqual(@as(usize, 2), chain.count());
}

test "MiddlewareChain: execute chain" {
    const allocator = std.testing.allocator;
    
    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();
    
    // Add middleware that continues
    try chain.add(Middleware{
        .name = "test1",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = req;
                _ = resp;
                return true;
            }
        }.handle,
    });
    
    var req = Request.init(allocator);
    defer req.deinit();
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try chain.execute(&req, &resp);
    try std.testing.expect(should_continue);
}

test "MiddlewareChain: early termination" {
    const allocator = std.testing.allocator;
    
    var chain = MiddlewareChain.init(allocator);
    defer chain.deinit();
    
    // Add middleware that stops processing
    try chain.add(Middleware{
        .name = "stopper",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = req;
                resp.status = 403;
                return false; // Stop here
            }
        }.handle,
    });
    
    // This should never execute
    try chain.add(Middleware{
        .name = "never-reached",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = req;
                _ = resp;
                return true;
            }
        }.handle,
    });
    
    var req = Request.init(allocator);
    defer req.deinit();
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try chain.execute(&req, &resp);
    try std.testing.expect(!should_continue);
    try std.testing.expectEqual(@as(u16, 403), resp.status);
}

test "Middleware: logging" {
    const allocator = std.testing.allocator;
    
    const middleware = loggingMiddleware();
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/test");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(should_continue);
}

test "Middleware: CORS" {
    const allocator = std.testing.allocator;
    
    const config = CorsConfig{
        .allow_origin = "https://example.com",
    };
    const middleware = corsMiddleware(config);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(should_continue);
    try std.testing.expectEqualStrings(
        "https://example.com",
        resp.headers.get("Access-Control-Allow-Origin").?,
    );
}

test "Middleware: CORS preflight" {
    const allocator = std.testing.allocator;
    
    const config = CorsConfig{};
    const middleware = corsMiddleware(config);
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .OPTIONS; // Preflight request
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(!should_continue); // Should stop processing
    try std.testing.expectEqual(@as(u16, 204), resp.status);
}

test "Middleware: health check" {
    const allocator = std.testing.allocator;
    
    const middleware = healthCheckMiddleware();
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .GET;
    req.path = try allocator.dupe(u8, "/health");
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(!should_continue);
    try std.testing.expectEqual(@as(u16, 200), resp.status);
}

test "Middleware: content type validation" {
    const allocator = std.testing.allocator;
    
    const middleware = contentTypeMiddleware("application/json");
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    try req.headers.put(
        try allocator.dupe(u8, "Content-Type"),
        try allocator.dupe(u8, "application/json"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(should_continue);
}

test "Middleware: content type validation fail" {
    const allocator = std.testing.allocator;
    
    const middleware = contentTypeMiddleware("application/json");
    
    var req = Request.init(allocator);
    defer req.deinit();
    req.method = .POST;
    try req.headers.put(
        try allocator.dupe(u8, "Content-Type"),
        try allocator.dupe(u8, "text/plain"),
    );
    
    var resp = Response.init(allocator);
    defer resp.deinit();
    
    const should_continue = try middleware.handler(&req, &resp);
    try std.testing.expect(!should_continue);
    try std.testing.expectEqual(@as(u16, 415), resp.status);
}
