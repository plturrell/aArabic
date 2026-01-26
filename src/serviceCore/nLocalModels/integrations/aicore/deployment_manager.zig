//! SAP AI Core Deployment Manager
//!
//! Manages the full deployment lifecycle for SAP AI Core including:
//! - Creating and deleting deployments
//! - Monitoring deployment status
//! - Scaling replicas
//! - Retrieving inference endpoints
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-21

const std = @import("std");
const http = std.http;
const json = std.json;
const AICoreConfig = @import("aicore_config.zig").AICoreConfig;
const ResourcePlan = @import("aicore_config.zig").ResourcePlan;
const ServingTemplateGenerator = @import("serving_template_generator.zig").ServingTemplateGenerator;
const ServingTemplateConfig = @import("serving_template_generator.zig").ServingTemplateConfig;

/// Deployment status states
pub const DeploymentStatus = enum {
    pending,
    creating,
    running,
    failed,
    stopped,
    deleting,

    pub fn toString(self: DeploymentStatus) []const u8 {
        return switch (self) {
            .pending => "PENDING",
            .creating => "CREATING",
            .running => "RUNNING",
            .failed => "FAILED",
            .stopped => "STOPPED",
            .deleting => "DELETING",
        };
    }

    pub fn fromString(str: []const u8) DeploymentStatus {
        if (std.mem.eql(u8, str, "PENDING")) return .pending;
        if (std.mem.eql(u8, str, "CREATING")) return .creating;
        if (std.mem.eql(u8, str, "RUNNING")) return .running;
        if (std.mem.eql(u8, str, "FAILED")) return .failed;
        if (std.mem.eql(u8, str, "STOPPED")) return .stopped;
        if (std.mem.eql(u8, str, "DELETING")) return .deleting;
        return .pending;
    }
};

/// Deployment information
pub const DeploymentInfo = struct {
    deployment_id: []const u8,
    configuration_id: []const u8,
    status: DeploymentStatus,
    endpoint_url: ?[]const u8,
    model_id: []const u8,
    resource_plan: ResourcePlan,
    replicas: u32,
    created_at: i64,
    updated_at: i64,
    error_message: ?[]const u8,

    pub fn deinit(self: *DeploymentInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.deployment_id);
        allocator.free(self.configuration_id);
        allocator.free(self.model_id);
        if (self.endpoint_url) |url| allocator.free(url);
        if (self.error_message) |msg| allocator.free(msg);
    }
};

/// API error response
pub const ApiError = struct {
    code: []const u8,
    message: []const u8,
};

/// Deployment Manager for SAP AI Core
pub const DeploymentManager = struct {
    allocator: std.mem.Allocator,
    config: AICoreConfig,
    access_token: ?[]const u8,
    token_expires_at: i64,
    deployments: std.StringHashMap(DeploymentInfo),

    const Self = @This();
    const POLL_INTERVAL_MS: u64 = 5000;
    const TOKEN_REFRESH_BUFFER_SECS: i64 = 300;

    /// Initialize the deployment manager
    pub fn init(allocator: std.mem.Allocator, config: AICoreConfig) Self {
        return Self{
            .allocator = allocator,
            .config = config,
            .access_token = null,
            .token_expires_at = 0,
            .deployments = std.StringHashMap(DeploymentInfo).init(allocator),
        };
    }

    /// Cleanup resources
    pub fn deinit(self: *Self) void {
        if (self.access_token) |token| {
            self.allocator.free(token);
        }
        var it = self.deployments.valueIterator();
        while (it.next()) |info| {
            var mutable_info = info.*;
            mutable_info.deinit(self.allocator);
        }
        self.deployments.deinit();
    }

    /// Authenticate with AI Core OAuth2 endpoint
    fn authenticate(self: *Self) !void {
        const current_time = std.time.timestamp();
        if (self.access_token != null and current_time < self.token_expires_at - TOKEN_REFRESH_BUFFER_SECS) {
            return; // Token still valid
        }

        std.log.info("🔐 Authenticating with AI Core...", .{});

        const auth_url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/oauth/token",
            .{self.config.auth_url},
        );
        defer self.allocator.free(auth_url);

        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=client_credentials&client_id={s}&client_secret={s}",
            .{ self.config.client_id, self.config.client_secret },
        );
        defer self.allocator.free(body);

        const response = try self.httpPost(auth_url, body, "application/x-www-form-urlencoded");
        defer self.allocator.free(response);

        // Parse token response
        const parsed = try json.parseFromSlice(json.Value, self.allocator, response, .{});
        defer parsed.deinit();

        const access_token = parsed.value.object.get("access_token") orelse return error.AuthenticationFailed;
        const expires_in = parsed.value.object.get("expires_in") orelse return error.AuthenticationFailed;

        if (self.access_token) |old_token| {
            self.allocator.free(old_token);
        }
        self.access_token = try self.allocator.dupe(u8, access_token.string);
        self.token_expires_at = current_time + expires_in.integer;

        std.log.info("✅ Authentication successful, token expires in {d}s", .{expires_in.integer});
    }

    /// Create a new deployment
    pub fn createDeployment(self: *Self, model_id: []const u8, resource_plan: ResourcePlan) !DeploymentInfo {
        try self.authenticate();

        std.log.info("🚀 Creating deployment for model: {s} with plan: {s}", .{ model_id, resource_plan.toString() });

        // First create configuration
        const config_id = try self.createConfiguration(model_id, resource_plan);
        defer self.allocator.free(config_id);

        // Then create deployment
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/deployments",
            .{self.config.api_url},
        );
        defer self.allocator.free(url);

        const request_body = try std.fmt.allocPrint(
            self.allocator,
            \\{{"configurationId": "{s}"}}
        ,
            .{config_id},
        );
        defer self.allocator.free(request_body);

        const response = try self.httpPostWithAuth(url, request_body);
        defer self.allocator.free(response);

        const parsed = try json.parseFromSlice(json.Value, self.allocator, response, .{});
        defer parsed.deinit();

        const deployment_id = parsed.value.object.get("id") orelse return error.DeploymentCreationFailed;

        const info = DeploymentInfo{
            .deployment_id = try self.allocator.dupe(u8, deployment_id.string),
            .configuration_id = try self.allocator.dupe(u8, config_id),
            .status = .pending,
            .endpoint_url = null,
            .model_id = try self.allocator.dupe(u8, model_id),
            .resource_plan = resource_plan,
            .replicas = 1,
            .created_at = std.time.timestamp(),
            .updated_at = std.time.timestamp(),
            .error_message = null,
        };

        try self.deployments.put(info.deployment_id, info);

        std.log.info("✅ Deployment created: {s}", .{info.deployment_id});
        return info;
    }

    /// Create deployment configuration
    fn createConfiguration(self: *Self, model_id: []const u8, resource_plan: ResourcePlan) ![]const u8 {
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/configurations",
            .{self.config.api_url},
        );
        defer self.allocator.free(url);

        const request_body = try std.fmt.allocPrint(
            self.allocator,
            \\{{"name": "nopena-{s}", "scenarioId": "{s}", "executableId": "serving",
            \\"parameterBindings": [{{"key": "modelId", "value": "{s}"}},
            \\{{"key": "resourcePlan", "value": "{s}"}}]}}
        ,
            .{ model_id, self.config.scenario_id, model_id, resource_plan.toString() },
        );
        defer self.allocator.free(request_body);

        const response = try self.httpPostWithAuth(url, request_body);
        defer self.allocator.free(response);

        const parsed = try json.parseFromSlice(json.Value, self.allocator, response, .{});
        defer parsed.deinit();

        const config_id = parsed.value.object.get("id") orelse return error.ConfigurationCreationFailed;
        return try self.allocator.dupe(u8, config_id.string);
    }

    /// Get deployment status
    pub fn getDeployment(self: *Self, deployment_id: []const u8) !DeploymentInfo {
        try self.authenticate();

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/deployments/{s}",
            .{ self.config.api_url, deployment_id },
        );
        defer self.allocator.free(url);

        const response = try self.httpGetWithAuth(url);
        defer self.allocator.free(response);

        return try self.parseDeploymentResponse(response);
    }

    /// List all deployments
    pub fn listDeployments(self: *Self) ![]DeploymentInfo {
        try self.authenticate();

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/deployments?scenarioId={s}",
            .{ self.config.api_url, self.config.scenario_id },
        );
        defer self.allocator.free(url);

        const response = try self.httpGetWithAuth(url);
        defer self.allocator.free(response);

        const parsed = try json.parseFromSlice(json.Value, self.allocator, response, .{});
        defer parsed.deinit();

        const resources = parsed.value.object.get("resources") orelse return self.allocator.alloc(DeploymentInfo, 0);
        const items = resources.array.items;

        var result = try self.allocator.alloc(DeploymentInfo, items.len);
        for (items, 0..) |item, i| {
            result[i] = try self.parseDeploymentValue(item);
        }
        return result;
    }

    /// Delete a deployment
    pub fn deleteDeployment(self: *Self, deployment_id: []const u8) !void {
        try self.authenticate();

        std.log.info("🗑️  Deleting deployment: {s}", .{deployment_id});

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/deployments/{s}",
            .{ self.config.api_url, deployment_id },
        );
        defer self.allocator.free(url);

        _ = try self.httpDeleteWithAuth(url);

        if (self.deployments.fetchRemove(deployment_id)) |kv| {
            var info = kv.value;
            info.deinit(self.allocator);
        }

        std.log.info("✅ Deployment deleted: {s}", .{deployment_id});
    }

    /// Scale deployment replicas
    pub fn scaleDeployment(self: *Self, deployment_id: []const u8, replicas: u32) !void {
        try self.authenticate();

        std.log.info("📈 Scaling deployment {s} to {d} replicas", .{ deployment_id, replicas });

        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/v2/lm/deployments/{s}",
            .{ self.config.api_url, deployment_id },
        );
        defer self.allocator.free(url);

        const request_body = try std.fmt.allocPrint(
            self.allocator,
            \\{{"minReplicas": {d}, "maxReplicas": {d}}}
        ,
            .{ replicas, replicas },
        );
        defer self.allocator.free(request_body);

        _ = try self.httpPatchWithAuth(url, request_body);

        std.log.info("✅ Deployment scaled to {d} replicas", .{replicas});
    }

    /// Wait for deployment to reach RUNNING state
    pub fn waitForReady(self: *Self, deployment_id: []const u8, timeout_ms: u64) !DeploymentInfo {
        std.log.info("⏳ Waiting for deployment {s} to be ready...", .{deployment_id});

        const start_time = std.time.milliTimestamp();
        const deadline = start_time + @as(i64, @intCast(timeout_ms));

        while (std.time.milliTimestamp() < deadline) {
            const info = try self.getDeployment(deployment_id);

            switch (info.status) {
                .running => {
                    std.log.info("✅ Deployment {s} is ready!", .{deployment_id});
                    return info;
                },
                .failed => {
                    std.log.err("❌ Deployment failed: {s}", .{info.error_message orelse "Unknown error"});
                    return error.DeploymentFailed;
                },
                else => {
                    std.log.debug("Status: {s}, waiting...", .{info.status.toString()});
                },
            }

            std.time.sleep(POLL_INTERVAL_MS * std.time.ns_per_ms);
        }

        std.log.err("⏰ Timeout waiting for deployment {s}", .{deployment_id});
        return error.DeploymentTimeout;
    }

    /// Get inference endpoint URL
    pub fn getEndpointUrl(self: *Self, deployment_id: []const u8) ![]const u8 {
        const info = try self.getDeployment(deployment_id);

        if (info.status != .running) {
            return error.DeploymentNotReady;
        }

        if (info.endpoint_url) |url| {
            return try self.allocator.dupe(u8, url);
        }

        return error.EndpointNotAvailable;
    }

    /// Parse deployment response from API
    fn parseDeploymentResponse(self: *Self, response: []const u8) !DeploymentInfo {
        const parsed = try json.parseFromSlice(json.Value, self.allocator, response, .{});
        defer parsed.deinit();
        return try self.parseDeploymentValue(parsed.value);
    }

    fn parseDeploymentValue(self: *Self, value: json.Value) !DeploymentInfo {
        const obj = value.object;

        const id = obj.get("id") orelse return error.InvalidResponse;
        const config_id = obj.get("configurationId") orelse return error.InvalidResponse;
        const status_str = obj.get("status") orelse return error.InvalidResponse;

        var endpoint_url: ?[]const u8 = null;
        if (obj.get("deploymentUrl")) |url_val| {
            endpoint_url = try self.allocator.dupe(u8, url_val.string);
        }

        var error_msg: ?[]const u8 = null;
        if (obj.get("statusMessage")) |msg_val| {
            error_msg = try self.allocator.dupe(u8, msg_val.string);
        }

        return DeploymentInfo{
            .deployment_id = try self.allocator.dupe(u8, id.string),
            .configuration_id = try self.allocator.dupe(u8, config_id.string),
            .status = DeploymentStatus.fromString(status_str.string),
            .endpoint_url = endpoint_url,
            .model_id = try self.allocator.dupe(u8, ""),
            .resource_plan = .infer_s,
            .replicas = 1,
            .created_at = std.time.timestamp(),
            .updated_at = std.time.timestamp(),
            .error_message = error_msg,
        };
    }

    // ============================================================
    // HTTP Helper Methods
    // ============================================================

    fn httpPost(self: *Self, url: []const u8, body: []const u8, content_type: []const u8) ![]const u8 {
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var buf: [4096]u8 = undefined;
        var req = try client.open(.POST, uri, .{ .server_header_buffer = &buf });
        defer req.deinit();

        req.headers.content_type = .{ .override = content_type };
        req.transfer_encoding = .{ .content_length = body.len };

        try req.send();
        try req.writer().writeAll(body);
        try req.finish();
        try req.wait();

        var response_body = std.ArrayList(u8){};
        try req.reader().readAllArrayList(&response_body, 65536);
        return response_body.toOwnedSlice();
    }

    fn httpPostWithAuth(self: *Self, url: []const u8, body: []const u8) ![]const u8 {
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var buf: [4096]u8 = undefined;
        var req = try client.open(.POST, uri, .{ .server_header_buffer = &buf });
        defer req.deinit();

        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.access_token.?});
        defer self.allocator.free(auth_header);

        req.headers.authorization = .{ .override = auth_header };
        req.headers.content_type = .{ .override = "application/json" };
        try req.extra_headers.append(.{ .name = "AI-Resource-Group", .value = self.config.resource_group });
        req.transfer_encoding = .{ .content_length = body.len };

        try req.send();
        try req.writer().writeAll(body);
        try req.finish();
        try req.wait();

        var response_body = std.ArrayList(u8){};
        try req.reader().readAllArrayList(&response_body, 65536);
        return response_body.toOwnedSlice();
    }

    fn httpGetWithAuth(self: *Self, url: []const u8) ![]const u8 {
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var buf: [4096]u8 = undefined;
        var req = try client.open(.GET, uri, .{ .server_header_buffer = &buf });
        defer req.deinit();

        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.access_token.?});
        defer self.allocator.free(auth_header);

        req.headers.authorization = .{ .override = auth_header };
        try req.extra_headers.append(.{ .name = "AI-Resource-Group", .value = self.config.resource_group });

        try req.send();
        try req.finish();
        try req.wait();

        var response_body = std.ArrayList(u8){};
        try req.reader().readAllArrayList(&response_body, 65536);
        return response_body.toOwnedSlice();
    }

    fn httpDeleteWithAuth(self: *Self, url: []const u8) !void {
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var buf: [4096]u8 = undefined;
        var req = try client.open(.DELETE, uri, .{ .server_header_buffer = &buf });
        defer req.deinit();

        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.access_token.?});
        defer self.allocator.free(auth_header);

        req.headers.authorization = .{ .override = auth_header };
        try req.extra_headers.append(.{ .name = "AI-Resource-Group", .value = self.config.resource_group });

        try req.send();
        try req.finish();
        try req.wait();
    }

    fn httpPatchWithAuth(self: *Self, url: []const u8, body: []const u8) ![]const u8 {
        var client = http.Client{ .allocator = self.allocator };
        defer client.deinit();

        const uri = try std.Uri.parse(url);
        var buf: [4096]u8 = undefined;
        var req = try client.open(.PATCH, uri, .{ .server_header_buffer = &buf });
        defer req.deinit();

        const auth_header = try std.fmt.allocPrint(self.allocator, "Bearer {s}", .{self.access_token.?});
        defer self.allocator.free(auth_header);

        req.headers.authorization = .{ .override = auth_header };
        req.headers.content_type = .{ .override = "application/json" };
        try req.extra_headers.append(.{ .name = "AI-Resource-Group", .value = self.config.resource_group });
        req.transfer_encoding = .{ .content_length = body.len };

        try req.send();
        try req.writer().writeAll(body);
        try req.finish();
        try req.wait();

        var response_body = std.ArrayList(u8){};
        try req.reader().readAllArrayList(&response_body, 65536);
        return response_body.toOwnedSlice();
    }
};

// ============================================================
// Tests
// ============================================================

test "deployment status conversion" {
    const testing = std.testing;
    try testing.expectEqualStrings("RUNNING", DeploymentStatus.running.toString());
    try testing.expectEqual(DeploymentStatus.failed, DeploymentStatus.fromString("FAILED"));
    try testing.expectEqual(DeploymentStatus.pending, DeploymentStatus.fromString("UNKNOWN"));
}

test "deployment manager init and deinit" {
    const testing = std.testing;
    const config = AICoreConfig{};
    var manager = DeploymentManager.init(testing.allocator, config);
    defer manager.deinit();

    try testing.expect(manager.access_token == null);
    try testing.expectEqual(@as(i64, 0), manager.token_expires_at);
}

test "deployment info struct" {
    const testing = std.testing;

    var info = DeploymentInfo{
        .deployment_id = try testing.allocator.dupe(u8, "dep-123"),
        .configuration_id = try testing.allocator.dupe(u8, "cfg-456"),
        .status = .running,
        .endpoint_url = try testing.allocator.dupe(u8, "https://example.com/v2/inference"),
        .model_id = try testing.allocator.dupe(u8, "llama-7b"),
        .resource_plan = .infer_s,
        .replicas = 2,
        .created_at = 1705847200,
        .updated_at = 1705847200,
        .error_message = null,
    };
    defer info.deinit(testing.allocator);

    try testing.expectEqualStrings("dep-123", info.deployment_id);
    try testing.expectEqual(DeploymentStatus.running, info.status);
}