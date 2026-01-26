// Load Balancer Configuration for nWorkflow
// Day 32: APISIX Gateway Integration (Continued)

const std = @import("std");
const Allocator = std.mem.Allocator;
const ApisixClient = @import("apisix_client.zig").ApisixClient;
const RouteConfig = @import("apisix_client.zig").RouteConfig;

/// Load balancing types supported by APISIX
pub const LoadBalancerType = enum {
    roundrobin,
    chash, // Consistent hashing
    ewma, // Exponentially weighted moving average
    least_conn, // Least connections

    pub fn toString(self: LoadBalancerType) []const u8 {
        return switch (self) {
            .roundrobin => "roundrobin",
            .chash => "chash",
            .ewma => "ewma",
            .least_conn => "least_conn",
        };
    }
};

/// Health check configuration
pub const HealthCheckConfig = struct {
    active: ?ActiveHealthCheck = null,
    passive: ?PassiveHealthCheck = null,
};

/// Active health check (proactive polling)
pub const ActiveHealthCheck = struct {
    type: []const u8 = "http", // http, https, tcp
    timeout: u32 = 1, // seconds
    http_path: []const u8 = "/health",
    healthy: struct {
        interval: u32 = 2, // seconds
        successes: u32 = 2, // consecutive successes to mark healthy
    } = .{},
    unhealthy: struct {
        interval: u32 = 5, // seconds
        http_failures: u32 = 3, // consecutive failures to mark unhealthy
        timeouts: u32 = 3,
    } = .{},
};

/// Passive health check (based on actual requests)
pub const PassiveHealthCheck = struct {
    type: []const u8 = "http",
    healthy: struct {
        http_statuses: []u32, // e.g., [200, 201, 202]
        successes: u32 = 3,
    },
    unhealthy: struct {
        http_statuses: []u32, // e.g., [429, 500, 502, 503]
        http_failures: u32 = 3,
        timeouts: u32 = 3,
    },
};

/// Upstream node (backend server)
pub const UpstreamNode = struct {
    host: []const u8,
    port: u16,
    weight: i32 = 1, // Higher weight = more traffic
    priority: i32 = 0, // 0 is highest priority
    metadata: ?std.json.Value = null,
};

/// Upstream service configuration
pub const UpstreamConfig = struct {
    type: LoadBalancerType = .roundrobin,
    nodes: []UpstreamNode,
    scheme: []const u8 = "http", // http, https, grpc
    pass_host: []const u8 = "pass", // pass, node, rewrite
    timeout: ?struct {
        connect: u32 = 60,
        send: u32 = 60,
        read: u32 = 60,
    } = null,
    retries: u32 = 1,
    retry_timeout: u32 = 0,
    keepalive_pool: ?struct {
        size: u32 = 320,
        idle_timeout: u32 = 60,
        requests: u32 = 1000,
    } = null,
    checks: ?HealthCheckConfig = null,
    hash_on: ?[]const u8 = null, // For consistent hashing
};

/// Load balancer manager
pub const LoadBalancerManager = struct {
    allocator: Allocator,
    apisix_client: *ApisixClient,
    upstreams: std.StringHashMap(UpstreamInfo),
    arena: std.heap.ArenaAllocator,

    const UpstreamInfo = struct {
        id: []const u8,
        name: []const u8,
        config: UpstreamConfig,
        created_at: i64,
    };

    pub fn init(allocator: Allocator, apisix_client: *ApisixClient) !LoadBalancerManager {
        return LoadBalancerManager{
            .allocator = allocator,
            .apisix_client = apisix_client,
            .upstreams = std.StringHashMap(UpstreamInfo).init(allocator),
            .arena = std.heap.ArenaAllocator.init(allocator),
        };
    }

    pub fn deinit(self: *LoadBalancerManager) void {
        var it = self.upstreams.valueIterator();
        while (it.next()) |info| {
            self.allocator.free(info.id);
            self.allocator.free(info.name);
        }
        self.upstreams.deinit();
        self.arena.deinit();
    }

    /// Create upstream with load balancing configuration
    pub fn createUpstream(
        self: *LoadBalancerManager,
        name: []const u8,
        config: UpstreamConfig,
    ) ![]const u8 {
        // Generate upstream ID
        const upstream_id = try self.generateUpstreamId(name);
        errdefer self.allocator.free(upstream_id);

        // Dupe name once and use for both hashmap key and info.name
        const duped_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(duped_name);

        // Store upstream info - use duped_name for both key and info.name
        const info = UpstreamInfo{
            .id = upstream_id,
            .name = duped_name,
            .config = config,
            .created_at = std.time.timestamp(),
        };

        try self.upstreams.put(duped_name, info);

        return upstream_id;
    }

    /// Update upstream configuration
    pub fn updateUpstream(
        self: *LoadBalancerManager,
        name: []const u8,
        config: UpstreamConfig,
    ) !void {
        var info = self.upstreams.getPtr(name) orelse return error.UpstreamNotFound;
        info.config = config;
    }

    /// Delete upstream
    pub fn deleteUpstream(self: *LoadBalancerManager, name: []const u8) !void {
        if (self.upstreams.fetchRemove(name)) |kv| {
            self.allocator.free(kv.value.id);
            self.allocator.free(kv.value.name);
            self.allocator.free(kv.key);
        } else {
            return error.UpstreamNotFound;
        }
    }

    /// Add node to existing upstream
    pub fn addNode(
        self: *LoadBalancerManager,
        upstream_name: []const u8,
        node: UpstreamNode,
    ) !void {
        var info = self.upstreams.getPtr(upstream_name) orelse return error.UpstreamNotFound;

        // Create new nodes array with additional node
        const arena_allocator = self.arena.allocator();
        const old_nodes = info.config.nodes;
        const new_nodes = try arena_allocator.alloc(UpstreamNode, old_nodes.len + 1);

        @memcpy(new_nodes[0..old_nodes.len], old_nodes);
        new_nodes[old_nodes.len] = node;

        info.config.nodes = new_nodes;
    }

    /// Remove node from upstream
    pub fn removeNode(
        self: *LoadBalancerManager,
        upstream_name: []const u8,
        host: []const u8,
        port: u16,
    ) !void {
        var info = self.upstreams.getPtr(upstream_name) orelse return error.UpstreamNotFound;

        const arena_allocator = self.arena.allocator();
        var new_nodes = std.ArrayListUnmanaged(UpstreamNode){};

        for (info.config.nodes) |node| {
            if (!std.mem.eql(u8, node.host, host) or node.port != port) {
                try new_nodes.append(arena_allocator, node);
            }
        }

        if (new_nodes.items.len == info.config.nodes.len) {
            return error.NodeNotFound;
        }

        info.config.nodes = try new_nodes.toOwnedSlice(arena_allocator);
    }

    /// Update node weight for traffic distribution
    pub fn updateNodeWeight(
        self: *LoadBalancerManager,
        upstream_name: []const u8,
        host: []const u8,
        port: u16,
        weight: i32,
    ) !void {
        const info = self.upstreams.getPtr(upstream_name) orelse return error.UpstreamNotFound;

        for (info.config.nodes) |*node| {
            if (std.mem.eql(u8, node.host, host) and node.port == port) {
                node.weight = weight;
                return;
            }
        }

        return error.NodeNotFound;
    }

    /// Enable health checks for upstream
    pub fn enableHealthChecks(
        self: *LoadBalancerManager,
        upstream_name: []const u8,
        checks: HealthCheckConfig,
    ) !void {
        var info = self.upstreams.getPtr(upstream_name) orelse return error.UpstreamNotFound;
        info.config.checks = checks;
    }

    /// Disable health checks
    pub fn disableHealthChecks(self: *LoadBalancerManager, upstream_name: []const u8) !void {
        var info = self.upstreams.getPtr(upstream_name) orelse return error.UpstreamNotFound;
        info.config.checks = null;
    }

    /// List all upstreams
    pub fn listUpstreams(self: *const LoadBalancerManager) ![]UpstreamInfo {
        const arena_allocator = self.arena.allocator();
        var result = std.ArrayList(UpstreamInfo){};

        var it = self.upstreams.valueIterator();
        while (it.next()) |info| {
            try result.append(info.*);
        }

        return result.toOwnedSlice();
    }

    /// Get upstream info
    pub fn getUpstream(self: *const LoadBalancerManager, name: []const u8) !UpstreamInfo {
        return self.upstreams.get(name) orelse error.UpstreamNotFound;
    }

    /// Serialize upstream config to JSON for APISIX
    pub fn serializeConfig(self: *LoadBalancerManager, config: UpstreamConfig) ![]const u8 {
        const arena_allocator = self.arena.allocator();
        var string = std.ArrayListUnmanaged(u8){};
        const writer = string.writer(arena_allocator);

        try writer.writeAll("{");

        // Load balancer type
        try writer.print("\"type\":\"{s}\"", .{config.type.toString()});

        // Nodes
        try writer.writeAll(",\"nodes\":{");
        for (config.nodes, 0..) |node, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.print("\"{s}:{d}\":{d}", .{ node.host, node.port, node.weight });
        }
        try writer.writeAll("}");

        // Scheme
        try writer.print(",\"scheme\":\"{s}\"", .{config.scheme});

        // Pass host
        try writer.print(",\"pass_host\":\"{s}\"", .{config.pass_host});

        // Timeout
        if (config.timeout) |timeout| {
            try writer.print(",\"timeout\":{{\"connect\":{d},\"send\":{d},\"read\":{d}}}", .{
                timeout.connect,
                timeout.send,
                timeout.read,
            });
        }

        // Retries
        try writer.print(",\"retries\":{d}", .{config.retries});

        // Health checks
        if (config.checks) |checks| {
            try writer.writeAll(",\"checks\":{");

            if (checks.active) |active| {
                try writer.print("\"active\":{{\"type\":\"{s}\",\"timeout\":{d},\"http_path\":\"{s}\"", .{
                    active.type,
                    active.timeout,
                    active.http_path,
                });
                try writer.print(",\"healthy\":{{\"interval\":{d},\"successes\":{d}}}", .{
                    active.healthy.interval,
                    active.healthy.successes,
                });
                try writer.print(",\"unhealthy\":{{\"interval\":{d},\"http_failures\":{d},\"timeouts\":{d}}}", .{
                    active.unhealthy.interval,
                    active.unhealthy.http_failures,
                    active.unhealthy.timeouts,
                });
                try writer.writeAll("}");
            }

            try writer.writeAll("}");
        }

        try writer.writeAll("}");
        return string.toOwnedSlice(arena_allocator);
    }

    fn generateUpstreamId(self: *LoadBalancerManager, name: []const u8) ![]const u8 {
        var hash = std.hash.Wyhash.init(0);
        hash.update(name);
        const hash_value = hash.final();

        const id = try std.fmt.allocPrint(self.allocator, "upstream_{d}", .{hash_value});
        return id;
    }
};

// Tests
test "LoadBalancerManager: init and deinit" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    try std.testing.expect(manager.upstreams.count() == 0);
}

test "LoadBalancerManager: create upstream" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    var nodes = [_]UpstreamNode{
        .{ .host = "127.0.0.1", .port = 8080, .weight = 1 },
        .{ .host = "127.0.0.1", .port = 8081, .weight = 2 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = nodes[0..],
    };

    const upstream_id = try manager.createUpstream("my-service", config);
    // Note: upstream_id is managed by the manager and freed in deinit, so we don't free it here

    try std.testing.expect(manager.upstreams.count() == 1);
    try std.testing.expect(upstream_id.len > 0);
}

test "LoadBalancerManager: add and remove nodes" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    var nodes = [_]UpstreamNode{
        .{ .host = "127.0.0.1", .port = 8080, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = nodes[0..],
    };

    _ = try manager.createUpstream("my-service", config);

    // Add node
    try manager.addNode("my-service", .{ .host = "127.0.0.1", .port = 8081, .weight = 1 });

    const info = try manager.getUpstream("my-service");
    try std.testing.expectEqual(@as(usize, 2), info.config.nodes.len);

    // Remove node
    try manager.removeNode("my-service", "127.0.0.1", 8081);

    const info2 = try manager.getUpstream("my-service");
    try std.testing.expectEqual(@as(usize, 1), info2.config.nodes.len);
}

test "LoadBalancerManager: update node weight" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    var nodes = [_]UpstreamNode{
        .{ .host = "127.0.0.1", .port = 8080, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = nodes[0..],
    };

    _ = try manager.createUpstream("my-service", config);

    try manager.updateNodeWeight("my-service", "127.0.0.1", 8080, 5);

    const info = try manager.getUpstream("my-service");
    try std.testing.expectEqual(@as(i32, 5), info.config.nodes[0].weight);
}

test "LoadBalancerManager: enable health checks" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    var nodes = [_]UpstreamNode{
        .{ .host = "127.0.0.1", .port = 8080, .weight = 1 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = nodes[0..],
    };

    _ = try manager.createUpstream("my-service", config);

    const health_checks = HealthCheckConfig{
        .active = ActiveHealthCheck{
            .http_path = "/healthz",
            .healthy = .{ .interval = 1, .successes = 1 },
            .unhealthy = .{ .interval = 3, .http_failures = 2, .timeouts = 2 },
        },
    };

    try manager.enableHealthChecks("my-service", health_checks);

    const info = try manager.getUpstream("my-service");
    try std.testing.expect(info.config.checks != null);
}

test "LoadBalancerManager: serialize config" {
    const allocator = std.testing.allocator;

    const apisix_config = @import("apisix_client.zig").ApisixConfig{
        .admin_url = "http://localhost:9180",
        .api_key = "test-key",
    };

    var apisix_client = try ApisixClient.init(allocator, apisix_config);
    defer apisix_client.deinit();

    var manager = try LoadBalancerManager.init(allocator, apisix_client);
    defer manager.deinit();

    var nodes = [_]UpstreamNode{
        .{ .host = "127.0.0.1", .port = 8080, .weight = 1 },
        .{ .host = "127.0.0.1", .port = 8081, .weight = 2 },
    };

    const config = UpstreamConfig{
        .type = .roundrobin,
        .nodes = nodes[0..],
        .scheme = "http",
        .retries = 2,
    };

    const json = try manager.serializeConfig(config);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"type\":\"roundrobin\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"127.0.0.1:8080\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"127.0.0.1:8081\":2") != null);
}
