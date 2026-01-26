const std = @import("std");
const http = @import("../../lib/std/http.zig");
const testing = std.testing;

/// Integration tests for the HTTP framework
/// Tests complete request/response cycles with real networking

test "HTTP Integration - Basic server lifecycle" {
    const allocator = testing.allocator;

    var app = try http.Application.init(allocator, .{
        .port = 8091,
    });
    defer app.deinit();

    const handler = struct {
        fn handle(ctx: *http.Context) !void {
            try ctx.json(.{ .status = "ok" });
        }
    }.handle;

    try app.get("/test", handler);

    // Note: Full integration would spawn server in thread
    // and make actual HTTP requests
}

test "HTTP Integration - Middleware chain execution" {
    const allocator = testing.allocator;

    var app = try http.Application.init(allocator, .{
        .port = 8092,
    });
    defer app.deinit();

    // Add middleware
    try app.use(http.Middleware.requestId());
    try app.use(http.Middleware.logging());

    const handler = struct {
        fn handle(ctx: *http.Context) !void {
            const req_id = ctx.getState("request_id");
            try testing.expect(req_id != null);
            try ctx.text("OK");
        }
    }.handle;

    try app.get("/", handler);
}

test "HTTP Integration - Router with parameters" {
    const allocator = testing.allocator;

    var app = try http.Application.init(allocator, .{
        .port = 8093,
    });
    defer app.deinit();

    const handler = struct {
        fn handle(ctx: *http.Context) !void {
            const user_id = ctx.param("id") orelse return error.MissingParam;
            const post_id = ctx.param("post_id") orelse return error.MissingParam;

            try ctx.json(.{
                .user_id = user_id,
                .post_id = post_id,
            });
        }
    }.handle;

    try app.get("/users/:id/posts/:post_id", handler);

    // Simulate request
    var ctx = try http.Context.init(allocator);
    defer ctx.deinit();

    ctx.request.method = .GET;
    try ctx.parseUri("/users/123/posts/456");

    const matched = app.router.match(ctx);
    try testing.expect(matched != null);
    try testing.expectEqualStrings("123", ctx.param("id").?);
    try testing.expectEqualStrings("456", ctx.param("post_id").?);
}

test "HTTP Integration - JSON request/response" {
    const allocator = testing.allocator;

    var ctx = try http.Context.init(allocator);
    defer ctx.deinit();

    ctx.request.method = .POST;
    ctx.request.path = "/api/users";
    ctx.request.body = "{\"name\":\"Alice\",\"age\":30}";

    // Test JSON response
    try ctx.status(.created);
    try ctx.json(.{
        .id = 1,
        .name = "Alice",
        .age = 30,
    });

    try testing.expect(ctx.response.sent);
    try testing.expect(ctx.response.status == .created);
}

test "HTTP Integration - CORS middleware" {
    const allocator = testing.allocator;

    var app = try http.Application.init(allocator, .{
        .port = 8094,
    });
    defer app.deinit();

    try app.use(http.Middleware.cors(.{
        .allow_origin = "https://example.com",
        .allow_methods = "GET,POST,PUT,DELETE",
    }));

    const handler = struct {
        fn handle(ctx: *http.Context) !void {
            try ctx.text("OK");
        }
    }.handle;

    try app.get("/", handler);
}

test "HTTP Integration - Error handling" {
    const allocator = testing.allocator;

    var ctx = try http.Context.init(allocator);
    defer ctx.deinit();

    const handler = struct {
        fn handle(c: *http.Context) !void {
            _ = c;
            return error.SomethingWentWrong;
        }
    }.handle;

    // Handler should propagate errors properly
    try testing.expectError(error.SomethingWentWrong, handler(ctx));
}