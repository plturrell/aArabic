//! SAP AI Core Serving Template Generator
//!
//! Generates YAML serving templates for SAP AI Core deployments.
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-21

const std = @import("std");
const aicore_config = @import("aicore_config.zig");
const AICoreConfig = aicore_config.AICoreConfig;
const GpuType = aicore_config.GpuType;

/// Resource plan types for AI Core GPU allocation
pub const ResourcePlan = enum {
    starter,
    basic,
    infer_s,
    infer_m,
    infer_l,

    pub fn toString(self: ResourcePlan) []const u8 {
        return switch (self) {
            .starter => "starter",
            .basic => "basic",
            .infer_s => "infer.s",
            .infer_m => "infer.m",
            .infer_l => "infer.l",
        };
    }
};

/// Environment variable configuration
pub const EnvVar = struct {
    name: []const u8,
    value: []const u8,
};

/// SAP AI Core Serving Template
pub const ServingTemplate = struct {
    name: []const u8,
    labels: []const []const u8 = &[_][]const u8{},
    annotations: []const []const u8 = &[_][]const u8{},
    image: []const u8,
    resource_plan: ResourcePlan = .infer_s,
    port: u16 = 11434,
    env_vars: []const EnvVar = &[_]EnvVar{},
    health_check_path: []const u8 = "/health",
    liveness_path: []const u8 = "/health/liveness",

    /// Validate the serving template
    pub fn validate(self: ServingTemplate) !void {
        if (self.name.len == 0) return error.MissingName;
        if (self.image.len == 0) return error.MissingImage;
        if (self.port == 0) return error.InvalidPort;
        if (self.health_check_path.len == 0) return error.MissingHealthCheckPath;
        if (self.liveness_path.len == 0) return error.MissingLivenessPath;
    }

    /// Generate YAML serving template string
    pub fn generateYaml(self: ServingTemplate, allocator: std.mem.Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8).init(allocator);
        errdefer buffer.deinit();

        // Header and metadata
        try buffer.appendSlice("apiVersion: ai.sap.com/v1alpha1\n");
        try buffer.appendSlice("kind: ServingTemplate\n");
        try buffer.appendSlice("metadata:\n");
        try buffer.appendSlice("  name: ");
        try buffer.appendSlice(self.name);
        try buffer.appendSlice("\n");

        // Labels
        if (self.labels.len > 0) {
            try buffer.appendSlice("  labels:\n");
            var i: usize = 0;
            while (i + 1 < self.labels.len) : (i += 2) {
                try buffer.appendSlice("    ");
                try buffer.appendSlice(self.labels[i]);
                try buffer.appendSlice(": \"");
                try buffer.appendSlice(self.labels[i + 1]);
                try buffer.appendSlice("\"\n");
            }
        }

        // Annotations
        if (self.annotations.len > 0) {
            try buffer.appendSlice("  annotations:\n");
            var i: usize = 0;
            while (i + 1 < self.annotations.len) : (i += 2) {
                try buffer.appendSlice("    ");
                try buffer.appendSlice(self.annotations[i]);
                try buffer.appendSlice(": \"");
                try buffer.appendSlice(self.annotations[i + 1]);
                try buffer.appendSlice("\"\n");
            }
        }

        // Spec section
        try buffer.appendSlice("spec:\n");
        try buffer.appendSlice("  template:\n");
        try buffer.appendSlice("    apiVersion: serving.kserve.io/v1beta1\n");
        try buffer.appendSlice("    metadata:\n");
        try buffer.appendSlice("      labels:\n");
        try buffer.appendSlice("        ai.sap.com/resourcePlan: \"");
        try buffer.appendSlice(self.resource_plan.toString());
        try buffer.appendSlice("\"\n");
        try buffer.appendSlice("    spec:\n");
        try buffer.appendSlice("      predictor:\n");
        try buffer.appendSlice("        containers:\n");
        try buffer.appendSlice("        - name: kserve-container\n");
        try buffer.appendSlice("          image: \"");
        try buffer.appendSlice(self.image);
        try buffer.appendSlice("\"\n");
        try buffer.appendSlice("          ports:\n");
        try buffer.appendSlice("          - containerPort: ");
        try appendPort(&buffer, self.port);
        try buffer.appendSlice("\n");
        try buffer.appendSlice("            protocol: TCP\n");

        // Environment variables
        if (self.env_vars.len > 0) {
            try buffer.appendSlice("          env:\n");
            for (self.env_vars) |env| {
                try buffer.appendSlice("          - name: ");
                try buffer.appendSlice(env.name);
                try buffer.appendSlice("\n            value: \"");
                try buffer.appendSlice(env.value);
                try buffer.appendSlice("\"\n");
            }
        }

        // Health probes
        try buffer.appendSlice("          livenessProbe:\n");
        try buffer.appendSlice("            httpGet:\n");
        try buffer.appendSlice("              path: ");
        try buffer.appendSlice(self.liveness_path);
        try buffer.appendSlice("\n");
        try buffer.appendSlice("              port: ");
        try appendPort(&buffer, self.port);
        try buffer.appendSlice("\n");
        try buffer.appendSlice("            initialDelaySeconds: 60\n");
        try buffer.appendSlice("            periodSeconds: 30\n");
        try buffer.appendSlice("          readinessProbe:\n");
        try buffer.appendSlice("            httpGet:\n");
        try buffer.appendSlice("              path: ");
        try buffer.appendSlice(self.health_check_path);
        try buffer.appendSlice("\n");
        try buffer.appendSlice("              port: ");
        try appendPort(&buffer, self.port);
        try buffer.appendSlice("\n");
        try buffer.appendSlice("            initialDelaySeconds: 30\n");
        try buffer.appendSlice("            periodSeconds: 10\n");

        return buffer.toOwnedSlice();
    }
};

/// Helper to append port number to buffer
fn appendPort(buffer: *std.ArrayList(u8), port: u16) !void {
    var port_buf: [5]u8 = undefined;
    const port_str = std.fmt.bufPrint(&port_buf, "{d}", .{port}) catch unreachable;
    try buffer.appendSlice(port_str);
}

/// Generate a ServingTemplate from AICoreConfig
pub fn generateFromConfig(config: AICoreConfig, image: []const u8) ServingTemplate {
    // Map GPU type and count to resource plan
    const resource_plan: ResourcePlan = switch (config.gpu_type) {
        .T4 => if (config.gpu_count >= 2) .infer_m else .infer_s,
        .V100 => .infer_m,
        .A100 => .infer_l,
    };

    return ServingTemplate{
        .name = config.scenario_id,
        .image = image,
        .resource_plan = resource_plan,
        .port = 11434,
        .health_check_path = "/health",
        .liveness_path = "/health/liveness",
    };
}

// ============================================================================
// Tests
// ============================================================================

test "ServingTemplate validate - valid template" {
    const template = ServingTemplate{
        .name = "test-template",
        .image = "docker.io/user/image:latest",
        .port = 8080,
    };
    try template.validate();
}

test "ServingTemplate validate - missing name" {
    const template = ServingTemplate{
        .name = "",
        .image = "docker.io/user/image:latest",
    };
    try std.testing.expectError(error.MissingName, template.validate());
}

test "ServingTemplate validate - missing image" {
    const template = ServingTemplate{
        .name = "test",
        .image = "",
    };
    try std.testing.expectError(error.MissingImage, template.validate());
}

test "ServingTemplate generateYaml - basic template" {
    const testing = std.testing;
    const template = ServingTemplate{
        .name = "nopena-llama",
        .image = "docker.io/nopena/server:1.0.0",
        .resource_plan = .infer_s,
        .port = 11434,
    };

    const yaml = try template.generateYaml(testing.allocator);
    defer testing.allocator.free(yaml);

    try testing.expect(std.mem.indexOf(u8, yaml, "apiVersion: ai.sap.com/v1alpha1") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "name: nopena-llama") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "infer.s") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "containerPort: 11434") != null);
}

test "ServingTemplate generateYaml - with env vars" {
    const testing = std.testing;
    const env_vars = [_]EnvVar{
        .{ .name = "MODEL_ID", .value = "llama-7b" },
        .{ .name = "GPU_ENABLED", .value = "true" },
    };
    const template = ServingTemplate{
        .name = "test-with-env",
        .image = "test:latest",
        .env_vars = &env_vars,
    };

    const yaml = try template.generateYaml(testing.allocator);
    defer testing.allocator.free(yaml);

    try testing.expect(std.mem.indexOf(u8, yaml, "MODEL_ID") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "llama-7b") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "GPU_ENABLED") != null);
}

test "ResourcePlan toString" {
    const testing = std.testing;
    try testing.expectEqualStrings("starter", ResourcePlan.starter.toString());
    try testing.expectEqualStrings("basic", ResourcePlan.basic.toString());
    try testing.expectEqualStrings("infer.s", ResourcePlan.infer_s.toString());
    try testing.expectEqualStrings("infer.m", ResourcePlan.infer_m.toString());
    try testing.expectEqualStrings("infer.l", ResourcePlan.infer_l.toString());
}

test "generateFromConfig - creates valid template" {
    const testing = std.testing;
    const config = AICoreConfig.init(
        testing.allocator,
        "default-group",
        "nopena-scenario",
        "exec-001",
        "llama-7b",
        "1.0.0",
        "https://api.ai.core.sap",
    );
    const template = generateFromConfig(config, "docker.io/nopena/server:latest");

    try template.validate();
    try testing.expectEqualStrings(config.scenario_id, template.name);
    try testing.expectEqualStrings("docker.io/nopena/server:latest", template.image);
}
