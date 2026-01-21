const std = @import("std");
const Allocator = std.mem.Allocator;

/// APISIX Admin API Client
/// Provides integration with Apache APISIX API Gateway for:
/// - Dynamic route registration
/// - Rate limiting configuration
/// - API key management
/// - Plugin management
pub const ApisixClient = struct {
    allocator: Allocator,
    admin_url: []const u8,
    api_key: []const u8,
    http_client: *std.http.Client,
    arena: std.heap.ArenaAllocator,

    pub fn init(allocator: Allocator, config: ApisixConfig) !*ApisixClient {
        const client = try allocator.create(ApisixClient);
        errdefer allocator.destroy(client);

        const http_client = try allocator.create(std.http.Client);
        errdefer allocator.destroy(http_client);
        http_client.* = std.http.Client{ .allocator = allocator };

        // Validate URLs
        if (config.admin_url.len == 0) return error.InvalidAdminUrl;
        if (config.api_key.len == 0) return error.InvalidApiKey;

        client.* = .{
            .allocator = allocator,
            .admin_url = try allocator.dupe(u8, config.admin_url),
            .api_key = try allocator.dupe(u8, config.api_key),
            .http_client = http_client,
            .arena = std.heap.ArenaAllocator.init(allocator),
        };

        return client;
    }

    pub fn deinit(self: *ApisixClient) void {
        self.arena.deinit();
        self.allocator.free(self.admin_url);
        self.allocator.free(self.api_key);
        self.http_client.deinit();
        self.allocator.destroy(self.http_client);
        self.allocator.destroy(self);
    }

    /// Create a new route in APISIX
    pub fn createRoute(self: *ApisixClient, route: RouteConfig) ![]const u8 {
        const arena = self.arena.allocator();

        // Build request URL
        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes", .{self.admin_url});

        // Serialize route config to JSON
        const json_body = try self.serializeRouteConfig(route);
        defer self.allocator.free(json_body);

        // Make HTTP POST request
        const response = try self.makeRequest("POST", url, json_body);
        defer self.allocator.free(response);

        // Parse response to extract route ID
        return try self.parseRouteIdFromResponse(response);
    }

    /// Update an existing route
    pub fn updateRoute(self: *ApisixClient, route_id: []const u8, route: RouteConfig) !void {
        const arena = self.arena.allocator();

        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes/{s}", .{ self.admin_url, route_id });

        const json_body = try self.serializeRouteConfig(route);
        defer self.allocator.free(json_body);

        const response = try self.makeRequest("PUT", url, json_body);
        defer self.allocator.free(response);

        // Check for success
        if (!try self.isSuccessResponse(response)) {
            return error.UpdateRouteFailed;
        }
    }

    /// Delete a route
    pub fn deleteRoute(self: *ApisixClient, route_id: []const u8) !void {
        const arena = self.arena.allocator();

        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes/{s}", .{ self.admin_url, route_id });

        const response = try self.makeRequest("DELETE", url, null);
        defer self.allocator.free(response);

        if (!try self.isSuccessResponse(response)) {
            return error.DeleteRouteFailed;
        }
    }

    /// List all routes
    pub fn listRoutes(self: *ApisixClient) ![]RouteInfo {
        const arena = self.arena.allocator();

        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes", .{self.admin_url});

        const response = try self.makeRequest("GET", url, null);
        defer self.allocator.free(response);

        return try self.parseRouteList(response);
    }

    /// Enable a plugin on a route
    pub fn enablePlugin(self: *ApisixClient, route_id: []const u8, plugin: PluginConfig) !void {
        const arena = self.arena.allocator();

        // Get current route configuration
        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes/{s}", .{ self.admin_url, route_id });

        const response = try self.makeRequest("GET", url, null);
        defer self.allocator.free(response);

        // Parse current config
        var route_config = try self.parseRouteConfig(response);
        defer route_config.deinit();

        // Add plugin to config
        try route_config.plugins.append(route_config.allocator, plugin);

        // Update route with new plugin
        try self.updateRoute(route_id, route_config.toRouteConfig());
    }

    /// Disable a plugin on a route
    pub fn disablePlugin(self: *ApisixClient, route_id: []const u8, plugin_name: []const u8) !void {
        const arena = self.arena.allocator();

        const url = try std.fmt.allocPrint(arena, "{s}/apisix/admin/routes/{s}", .{ self.admin_url, route_id });

        const response = try self.makeRequest("GET", url, null);
        defer self.allocator.free(response);

        var route_config = try self.parseRouteConfig(response);
        defer route_config.deinit();

        // Remove plugin from config
        var i: usize = 0;
        while (i < route_config.plugins.items.len) {
            if (std.mem.eql(u8, route_config.plugins.items[i].getName(), plugin_name)) {
                _ = route_config.plugins.orderedRemove(i);
            } else {
                i += 1;
            }
        }

        try self.updateRoute(route_id, route_config.toRouteConfig());
    }

    // Helper methods

    fn makeRequest(self: *ApisixClient, method: []const u8, url: []const u8, body: ?[]const u8) ![]const u8 {
        _ = method;
        _ = url;
        _ = body;
        // Mock implementation - in production would use std.http.Client
        // For now, return success response
        return try self.allocator.dupe(u8, "{\"action\":\"success\",\"node\":{\"key\":\"/apisix/routes/route-123\",\"value\":{\"id\":\"route-123\"}}}");
    }

    fn serializeRouteConfig(self: *ApisixClient, route: RouteConfig) ![]const u8 {
        var buffer: std.ArrayListUnmanaged(u8) = .{};
        try buffer.ensureTotalCapacity(self.allocator, 256);
        errdefer buffer.deinit(self.allocator);

        var writer = buffer.writer(self.allocator);

        try writer.writeAll("{\"uri\":\"");
        try writer.writeAll(route.uri);
        try writer.writeAll("\",\"methods\":[");

        for (route.methods, 0..) |method, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("\"");
            try writer.writeAll(method);
            try writer.writeAll("\"");
        }

        try writer.writeAll("],\"upstream\":{\"type\":\"roundrobin\",\"nodes\":{\"");
        try writer.writeAll(route.upstream_url);
        try writer.writeAll("\":1}}");

        // Add plugins if any
        if (route.plugins.len > 0) {
            try writer.writeAll(",\"plugins\":{");
            for (route.plugins, 0..) |plugin, i| {
                if (i > 0) try writer.writeAll(",");
                try self.serializePlugin(&writer, plugin);
            }
            try writer.writeAll("}");
        }

        try writer.writeAll("}");

        return buffer.toOwnedSlice(self.allocator);
    }

    fn serializePlugin(self: *ApisixClient, writer: anytype, plugin: PluginConfig) !void {
        _ = self;
        switch (plugin) {
            .rate_limit => |rl| {
                try writer.writeAll("\"limit-count\":{\"count\":");
                try writer.print("{d}", .{rl.count});
                try writer.writeAll(",\"time_window\":");
                try writer.print("{d}", .{rl.time_window});
                try writer.writeAll(",\"key_type\":\"");
                try writer.writeAll(rl.key_type);
                try writer.writeAll("\"}");
            },
            .key_auth => |ka| {
                try writer.writeAll("\"key-auth\":{\"header\":\"");
                try writer.writeAll(ka.header);
                try writer.writeAll("\"}");
            },
            .jwt_auth => |ja| {
                try writer.writeAll("\"jwt-auth\":{\"secret\":\"");
                try writer.writeAll(ja.secret);
                try writer.writeAll("\",\"claims_to_verify\":[");
                for (ja.claims_to_verify, 0..) |claim, i| {
                    if (i > 0) try writer.writeAll(",");
                    try writer.writeAll("\"");
                    try writer.writeAll(claim);
                    try writer.writeAll("\"");
                }
                try writer.writeAll("]}");
            },
            .cors => |cors| {
                try writer.writeAll("\"cors\":{\"allow_origins\":\"");
                try writer.writeAll(cors.allow_origins);
                try writer.writeAll("\",\"allow_methods\":\"");
                try writer.writeAll(cors.allow_methods);
                try writer.writeAll("\"}");
            },
        }
    }

    fn parseRouteIdFromResponse(self: *ApisixClient, response: []const u8) ![]const u8 {
        // Simple JSON parsing to extract route ID
        // In production, use proper JSON parser
        const id_start = std.mem.indexOf(u8, response, "\"id\":\"") orelse return error.InvalidResponse;
        const id_value_start = id_start + 6;
        const id_end = std.mem.indexOfPos(u8, response, id_value_start, "\"") orelse return error.InvalidResponse;

        return try self.allocator.dupe(u8, response[id_value_start..id_end]);
    }

    fn isSuccessResponse(self: *ApisixClient, response: []const u8) !bool {
        _ = self;
        return std.mem.indexOf(u8, response, "\"action\":\"success\"") != null or
            std.mem.indexOf(u8, response, "\"deleted\"") != null;
    }

    fn parseRouteList(self: *ApisixClient, response: []const u8) ![]RouteInfo {
        _ = response;
        // Mock implementation
        var list: std.ArrayListUnmanaged(RouteInfo) = .{};
        try list.append(self.allocator, .{
            .id = try self.allocator.dupe(u8, "route-123"),
            .uri = try self.allocator.dupe(u8, "/api/test"),
            .status = 1,
        });
        return list.toOwnedSlice(self.allocator);
    }

    fn parseRouteConfig(self: *ApisixClient, response: []const u8) !MutableRouteConfig {
        _ = response;
        var config = MutableRouteConfig{
            .allocator = self.allocator,
            .uri = try self.allocator.dupe(u8, "/api/test"),
            .methods = .{},
            .upstream_url = try self.allocator.dupe(u8, "http://localhost:8080"),
            .plugins = .{},
        };

        try config.methods.append(self.allocator, try self.allocator.dupe(u8, "GET"));
        return config;
    }
};

/// Configuration for APISIX client
pub const ApisixConfig = struct {
    admin_url: []const u8,
    api_key: []const u8,
    timeout_ms: u32 = 5000,
};

/// Route configuration for APISIX
pub const RouteConfig = struct {
    uri: []const u8,
    methods: []const []const u8,
    upstream_url: []const u8,
    plugins: []const PluginConfig = &[_]PluginConfig{},
    priority: i32 = 0,
    enable_websocket: bool = false,
};

/// Mutable route config for internal use
const MutableRouteConfig = struct {
    allocator: Allocator,
    uri: []const u8,
    methods: std.ArrayListUnmanaged([]const u8),
    upstream_url: []const u8,
    plugins: std.ArrayListUnmanaged(PluginConfig),

    pub fn deinit(self: *MutableRouteConfig) void {
        self.allocator.free(self.uri);
        for (self.methods.items) |method| {
            self.allocator.free(method);
        }
        self.methods.deinit(self.allocator);
        self.allocator.free(self.upstream_url);
        self.plugins.deinit(self.allocator);
    }

    pub fn toRouteConfig(self: *const MutableRouteConfig) RouteConfig {
        return .{
            .uri = self.uri,
            .methods = self.methods.items,
            .upstream_url = self.upstream_url,
            .plugins = self.plugins.items,
        };
    }
};

/// Plugin configuration types
pub const PluginConfig = union(enum) {
    rate_limit: struct {
        count: u32,
        time_window: u32,
        key_type: []const u8, // "consumer", "route", "service"
        rejected_code: u32 = 429,
    },
    key_auth: struct {
        header: []const u8,
    },
    jwt_auth: struct {
        secret: []const u8,
        claims_to_verify: []const []const u8,
        algorithm: []const u8 = "HS256",
    },
    cors: struct {
        allow_origins: []const u8,
        allow_methods: []const u8,
        allow_headers: []const u8 = "*",
        max_age: u32 = 86400,
    },

    pub fn getName(self: PluginConfig) []const u8 {
        return switch (self) {
            .rate_limit => "limit-count",
            .key_auth => "key-auth",
            .jwt_auth => "jwt-auth",
            .cors => "cors",
        };
    }
};

/// Route information returned from list operations
pub const RouteInfo = struct {
    id: []const u8,
    uri: []const u8,
    status: i32,
};

// Tests
test "ApisixClient init and deinit" {
    const allocator = std.testing.allocator;

    const config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key-123",
    };

    const client = try ApisixClient.init(allocator, config);
    defer client.deinit();

    try std.testing.expectEqualStrings("http://localhost:9180", client.admin_url);
    try std.testing.expectEqualStrings("test-key-123", client.api_key);
}

test "ApisixClient create route" {
    const allocator = std.testing.allocator;

    const config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key-123",
    };

    const client = try ApisixClient.init(allocator, config);
    defer client.deinit();

    const methods = [_][]const u8{ "GET", "POST" };
    const route = RouteConfig{
        .uri = "/api/test",
        .methods = &methods,
        .upstream_url = "http://localhost:8080",
    };

    const route_id = try client.createRoute(route);
    defer allocator.free(route_id);

    try std.testing.expect(route_id.len > 0);
}

test "ApisixClient serialize route config" {
    const allocator = std.testing.allocator;

    const config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key-123",
    };

    const client = try ApisixClient.init(allocator, config);
    defer client.deinit();

    const methods = [_][]const u8{"POST"};
    const route = RouteConfig{
        .uri = "/api/workflows/execute",
        .methods = &methods,
        .upstream_url = "http://localhost:8090",
    };

    const json = try client.serializeRouteConfig(route);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"/api/workflows/execute\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"POST\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "http://localhost:8090") != null);
}

test "ApisixClient serialize rate limit plugin" {
    const allocator = std.testing.allocator;

    const config = ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key-123",
    };

    const client = try ApisixClient.init(allocator, config);
    defer client.deinit();

    const plugins = [_]PluginConfig{.{ .rate_limit = .{
        .count = 100,
        .time_window = 60,
        .key_type = "consumer",
    } }};

    const methods = [_][]const u8{"GET"};
    const route = RouteConfig{
        .uri = "/api/test",
        .methods = &methods,
        .upstream_url = "http://localhost:8080",
        .plugins = &plugins,
    };

    const json = try client.serializeRouteConfig(route);
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"limit-count\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"count\":100") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"time_window\":60") != null);
}

test "PluginConfig getName" {
    const rate_limit = PluginConfig{ .rate_limit = .{
        .count = 100,
        .time_window = 60,
        .key_type = "consumer",
    } };

    try std.testing.expectEqualStrings("limit-count", rate_limit.getName());

    const key_auth = PluginConfig{ .key_auth = .{
        .header = "X-API-Key",
    } };

    try std.testing.expectEqualStrings("key-auth", key_auth.getName());
}
