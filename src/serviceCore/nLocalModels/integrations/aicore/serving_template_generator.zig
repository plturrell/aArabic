//! SAP AI Core Serving Template Generator
//!
//! Generates YAML serving templates for SAP AI Core deployments with:
//! - Support for T4 GPU resource plans (infer.s)
//! - Configurable replicas, ports, and environment variables
//! - Health probe configuration (liveness, readiness, startup)
//! - GPU resource limits and requests
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-21

const std = @import("std");
const AICoreConfig = @import("aicore_config.zig").AICoreConfig;
const ResourcePlan = @import("aicore_config.zig").ResourcePlan;

/// Health probe configuration
pub const HealthProbeConfig = struct {
    /// Path for health check endpoint
    path: []const u8 = "/health/liveness",
    /// Port for health check
    port: u16 = 11434,
    /// Initial delay before first probe (seconds)
    initial_delay_seconds: u32 = 60,
    /// Period between probes (seconds)
    period_seconds: u32 = 30,
    /// Timeout for probe response (seconds)
    timeout_seconds: u32 = 10,
    /// Number of failures before marking unhealthy
    failure_threshold: u32 = 3,
};

/// Serving template configuration
pub const ServingTemplateConfig = struct {
    /// Template name
    name: []const u8 = "nopena-llama-7b",
    /// Scenario ID
    scenario_id: []const u8 = "nopena",
    /// Version
    version: []const u8 = "1.0.0",
    /// Minimum replicas
    min_replicas: u32 = 1,
    /// Maximum replicas
    max_replicas: u32 = 3,
    /// Container port
    container_port: u16 = 11434,
    /// Resource plan
    resource_plan: ResourcePlan = .infer_s,
    /// GPU count
    gpu_count: u8 = 1,
    /// Memory limit (Gi)
    memory_limit_gi: u32 = 32,
    /// Memory request (Gi)
    memory_request_gi: u32 = 16,
    /// Liveness probe config
    liveness_probe: HealthProbeConfig = .{ .path = "/health/liveness", .initial_delay_seconds = 60, .period_seconds = 30 },
    /// Readiness probe config
    readiness_probe: HealthProbeConfig = .{ .path = "/health/readiness", .initial_delay_seconds = 120, .period_seconds = 10 },
};

/// Serving Template Generator
pub const ServingTemplateGenerator = struct {
    allocator: std.mem.Allocator,
    config: ServingTemplateConfig,
    aicore_config: AICoreConfig,

    pub fn init(allocator: std.mem.Allocator, config: ServingTemplateConfig, aicore_config: AICoreConfig) ServingTemplateGenerator {
        return .{ .allocator = allocator, .config = config, .aicore_config = aicore_config };
    }

    /// Generate the complete serving template YAML
    pub fn generate(self: *ServingTemplateGenerator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        const writer = buffer.writer();

        // Header
        try writer.print(
            \\apiVersion: ai.sap.com/v1alpha1
            \\kind: ServingTemplate
            \\metadata:
            \\  name: {s}
            \\  labels:
            \\    scenarios.ai.sap.com/id: "{s}"
            \\    ai.sap.com/version: "{s}"
            \\spec:
            \\  template:
            \\    apiVersion: serving.kserve.io/v1beta1
            \\    metadata:
            \\      labels:
            \\        ai.sap.com/resourcePlan: "{s}"
            \\    spec:
            \\      predictor:
            \\        minReplicas: {d}
            \\        maxReplicas: {d}
            \\        containers:
            \\        - name: kserve-container
            \\          image: "{{{{dockerImage}}}}"
            \\          ports:
            \\          - containerPort: {d}
            \\            protocol: TCP
            \\
        , .{
            self.config.name, self.config.scenario_id, self.config.version,
            self.config.resource_plan.toString(),
            self.config.min_replicas, self.config.max_replicas, self.config.container_port,
        });

        // Environment variables
        try self.writeEnvVars(writer);

        // Resources
        try writer.print(
            \\          resources:
            \\            limits:
            \\              nvidia.com/gpu: "{d}"
            \\              memory: "{d}Gi"
            \\            requests:
            \\              nvidia.com/gpu: "{d}"
            \\              memory: "{d}Gi"
            \\
        , .{ self.config.gpu_count, self.config.memory_limit_gi, self.config.gpu_count, self.config.memory_request_gi });

        // Health probes
        try self.writeProbe(writer, "livenessProbe", self.config.liveness_probe);
        try self.writeProbe(writer, "readinessProbe", self.config.readiness_probe);

        return buffer.toOwnedSlice();
    }

    fn writeEnvVars(self: *ServingTemplateGenerator, writer: anytype) !void {
        try writer.writeAll("          env:\n");
        try writer.print("          - name: MODEL_ID\n            value: \"{{{{modelId}}}}\"\n", .{});
        try writer.print("          - name: MODEL_PATH\n            value: \"{{{{modelPath}}}}\"\n", .{});
        try writer.print("          - name: GPU_ENABLED\n            value: \"{s}\"\n", .{if (self.aicore_config.gpu_enabled) "true" else "false"});
        try writer.print("          - name: STORAGE_BACKEND\n            value: \"{{{{storageBackend}}}}\"\n", .{});
        try writer.print("          - name: OBJECTSTORE_ENDPOINT\n            value: \"{{{{objectstoreEndpoint}}}}\"\n", .{});
        try writer.print("          - name: OBJECTSTORE_ACCESS_KEY\n            value: \"{{{{objectstoreAccessKey}}}}\"\n", .{});
        try writer.print("          - name: OBJECTSTORE_SECRET_KEY\n            value: \"{{{{objectstoreSecretKey}}}}\"\n", .{});
    }

    fn writeProbe(self: *ServingTemplateGenerator, writer: anytype, name: []const u8, probe: HealthProbeConfig) !void {
        _ = self;
        try writer.print(
            \\          {s}:
            \\            httpGet:
            \\              path: {s}
            \\              port: {d}
            \\            initialDelaySeconds: {d}
            \\            periodSeconds: {d}
            \\            timeoutSeconds: {d}
            \\            failureThreshold: {d}
            \\
        , .{ name, probe.path, probe.port, probe.initial_delay_seconds, probe.period_seconds, probe.timeout_seconds, probe.failure_threshold });
    }
};

test "generate serving template" {
    const testing = std.testing;
    const config = ServingTemplateConfig{};
    const aicore = AICoreConfig{};
    var gen = ServingTemplateGenerator.init(testing.allocator, config, aicore);
    const yaml = try gen.generate();
    defer testing.allocator.free(yaml);
    try testing.expect(std.mem.indexOf(u8, yaml, "apiVersion: ai.sap.com/v1alpha1") != null);
    try testing.expect(std.mem.indexOf(u8, yaml, "infer.s") != null);
}

