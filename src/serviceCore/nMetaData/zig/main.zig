//! nMetaData Server - Day 30
//!
//! Main entry point for the nMetaData HTTP REST API server.
//! Provides metadata management with multi-database support.

const std = @import("std");
const http = std.http;

// HTTP server components
const Server = @import("http/server.zig").Server;
const ServerConfig = @import("http/server.zig").ServerConfig;
const Request = @import("http/types.zig").Request;
const Response = @import("http/types.zig").Response;
const loggingMiddleware = @import("http/middleware.zig").loggingMiddleware;
const corsMiddleware = @import("http/middleware.zig").corsMiddleware;
const CorsConfig = @import("http/middleware.zig").CorsConfig;
const requestIdMiddleware = @import("http/middleware.zig").requestIdMiddleware;
const healthCheckMiddleware = @import("http/middleware.zig").healthCheckMiddleware;

// API handlers
const handlers = @import("api/handlers.zig");
const graphql = @import("api/graphql_handler.zig");

// Database components (integrated in Day 30)
// const DatabaseClient = @import("db/client.zig").DatabaseClient;

/// Application version
pub const VERSION = "0.1.0";

/// Root handler
fn handleRoot(req: *Request, resp: *Response) !void {
    _ = req;
    
    resp.status = 200;
    try resp.json(.{
        .message = "Welcome to nMetaData API",
        .version = VERSION,
        .endpoints = .{
            .health = "/health",
            .info = "/api/v1/info",
            .status = "/api/v1/status",
            .datasets = "/api/v1/datasets",
            .lineage = "/api/v1/lineage",
        },
        .documentation = "https://docs.nmetadata.io",
    });
}

/// Main entry point
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("🚀 nMetaData Server v{s}\n", .{VERSION});
    std.debug.print("================================================================================\n", .{});
    
    // Create server configuration
    const config = ServerConfig{
        .host = "127.0.0.1",
        .port = 8080,
        .max_body_size = 10 * 1024 * 1024, // 10MB
        .request_timeout = 30000, // 30 seconds
        .enable_logging = true,
        .api_version = "/api/v1",
    };
    
    // Initialize server
    var server = try Server.init(allocator, config);
    defer server.deinit();
    
    // Add middleware
    try server.use(healthCheckMiddleware());
    try server.use(requestIdMiddleware());
    try server.use(loggingMiddleware());
    try server.use(corsMiddleware(CorsConfig{
        .allow_origin = "*",
        .allow_methods = "GET,POST,PUT,DELETE,PATCH,OPTIONS",
        .allow_headers = "Content-Type,Authorization,X-Request-ID",
    }));
    
    // Register routes
    
    // Root & info
    try server.route(.GET, "/", handleRoot);
    try server.route(.GET, "/api/v1/info", handlers.apiInfo);
    
    // Dataset CRUD
    try server.route(.GET, "/api/v1/datasets", handlers.listDatasets);
    try server.route(.POST, "/api/v1/datasets", handlers.createDataset);
    try server.route(.GET, "/api/v1/datasets/:id", handlers.getDataset);
    try server.route(.PUT, "/api/v1/datasets/:id", handlers.updateDataset);
    try server.route(.DELETE, "/api/v1/datasets/:id", handlers.deleteDataset);
    
    // Lineage queries
    try server.route(.GET, "/api/v1/lineage/upstream/:id", handlers.getUpstreamLineage);
    try server.route(.GET, "/api/v1/lineage/downstream/:id", handlers.getDownstreamLineage);
    try server.route(.POST, "/api/v1/lineage/edges", handlers.createLineageEdge);
    
    // System endpoints
    try server.route(.GET, "/api/v1/status", handlers.systemStatus);
    
    // GraphQL endpoints
    try server.route(.POST, "/api/v1/graphql", graphql.graphqlHandler);
    try server.route(.GET, "/api/v1/graphiql", graphql.graphiqlHandler);
    try server.route(.GET, "/api/v1/schema", graphql.schemaHandler);
    
    std.debug.print("\n", .{});
    std.debug.print("📋 Registered Routes:\n", .{});
    std.debug.print("  GET    /                                - API information\n", .{});
    std.debug.print("  GET    /health                          - Health check\n", .{});
    std.debug.print("  GET    /api/v1/info                     - Server info\n", .{});
    std.debug.print("  GET    /api/v1/status                   - System status\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  📊 Dataset Management:\n", .{});
    std.debug.print("  GET    /api/v1/datasets                 - List datasets\n", .{});
    std.debug.print("  POST   /api/v1/datasets                 - Create dataset\n", .{});
    std.debug.print("  GET    /api/v1/datasets/:id             - Get dataset\n", .{});
    std.debug.print("  PUT    /api/v1/datasets/:id             - Update dataset\n", .{});
    std.debug.print("  DELETE /api/v1/datasets/:id             - Delete dataset\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  🔗 Lineage Tracking:\n", .{});
    std.debug.print("  GET    /api/v1/lineage/upstream/:id     - Get upstream lineage\n", .{});
    std.debug.print("  GET    /api/v1/lineage/downstream/:id   - Get downstream lineage\n", .{});
    std.debug.print("  POST   /api/v1/lineage/edges            - Create lineage edge\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("  🔮 GraphQL:\n", .{});
    std.debug.print("  POST   /api/v1/graphql                  - GraphQL endpoint\n", .{});
    std.debug.print("  GET    /api/v1/graphiql                 - GraphiQL playground\n", .{});
    std.debug.print("  GET    /api/v1/schema                   - Schema introspection\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("🔧 Middleware:\n", .{});
    std.debug.print("  - Health Check\n", .{});
    std.debug.print("  - Request ID\n", .{});
    std.debug.print("  - Logging\n", .{});
    std.debug.print("  - CORS\n", .{});
    std.debug.print("\n", .{});
    
    // Start server
    try server.start();
    
    std.debug.print("✨ Server ready! Try:\n", .{});
    std.debug.print("   # Basic info\n", .{});
    std.debug.print("   curl http://127.0.0.1:8080/\n", .{});
    std.debug.print("   curl http://127.0.0.1:8080/health\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("   # List datasets\n", .{});
    std.debug.print("   curl http://127.0.0.1:8080/api/v1/datasets\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("   # Create dataset\n", .{});
    std.debug.print("   curl -X POST http://127.0.0.1:8080/api/v1/datasets \\\n", .{});
    std.debug.print("     -H 'Content-Type: application/json' \\\n", .{});
    std.debug.print("     -d '{{\"name\":\"test\",\"type\":\"table\"}}'\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("   # Get lineage\n", .{});
    std.debug.print("   curl http://127.0.0.1:8080/api/v1/lineage/upstream/ds-001\n", .{});
    std.debug.print("\n", .{});
    std.debug.print("Press Ctrl+C to stop.\n", .{});
    std.debug.print("================================================================================\n", .{});
    
    // Serve requests (this will block)
    try server.serve();
}

test "main test placeholder" {
    try std.testing.expect(true);
}
