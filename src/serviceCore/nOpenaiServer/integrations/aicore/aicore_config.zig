//! SAP AI Core Configuration
//!
//! Configuration struct for AI Core deployments supporting:
//! - Authentication URL and client credentials
//! - Resource group configuration
//! - Docker image and model path settings
//! - Storage backend configuration (objectstore/local)
//!
//! Version: 1.0.0
//! Last Updated: 2026-01-21

const std = @import("std");

/// Resource plan types for AI Core GPU allocation
pub const ResourcePlan = enum {
    /// CPU only (no GPU)
    infer_xs,
    /// T4 GPU - 16GB VRAM (recommended for most workloads)
    infer_s,
    /// A10 GPU - 24GB VRAM
    infer_m,
    /// A100 GPU - 40/80GB VRAM
    infer_l,

    pub fn toString(self: ResourcePlan) []const u8 {
        return switch (self) {
            .infer_xs => "infer.xs",
            .infer_s => "infer.s",
            .infer_m => "infer.m",
            .infer_l => "infer.l",
        };
    }

    pub fn getApproxHourlyCost(self: ResourcePlan) f32 {
        return switch (self) {
            .infer_xs => 0.05,
            .infer_s => 0.50,
            .infer_m => 1.00,
            .infer_l => 3.00,
        };
    }
};

/// Storage backend type
pub const StorageBackend = enum {
    objectstore,
    local,

    pub fn toString(self: StorageBackend) []const u8 {
        return switch (self) {
            .objectstore => "objectstore",
            .local => "local",
        };
    }
};

/// Docker image configuration
pub const DockerConfig = struct {
    /// Docker registry URL (e.g., "docker.io/username")
    registry: []const u8 = "docker.io",
    /// Image name
    image_name: []const u8 = "nopena-server",
    /// Image tag/version
    tag: []const u8 = "latest",

    /// Get full image path
    pub fn getFullImagePath(self: DockerConfig, allocator: std.mem.Allocator) ![]const u8 {
        return std.fmt.allocPrint(allocator, "{s}/{s}:{s}", .{
            self.registry,
            self.image_name,
            self.tag,
        });
    }
};

/// Object store configuration for model storage
pub const ObjectStoreConfig = struct {
    /// S3-compatible endpoint URL
    endpoint: []const u8 = "https://s3.amazonaws.com",
    /// Access key ID
    access_key: []const u8 = "",
    /// Secret access key
    secret_key: []const u8 = "",
    /// Bucket name
    bucket: []const u8 = "models",
    /// Region
    region: []const u8 = "us-east-1",
};

/// AI Core deployment configuration
pub const AICoreConfig = struct {
    /// AI Core authentication URL
    auth_url: []const u8 = "https://your-subdomain.authentication.us10.hana.ondemand.com",
    /// AI Core API URL
    api_url: []const u8 = "https://api.ai.prod.us-east-1.aws.ml.hana.ondemand.com",
    /// OAuth2 client ID
    client_id: []const u8 = "",
    /// OAuth2 client secret
    client_secret: []const u8 = "",
    /// Resource group name
    resource_group: []const u8 = "default",

    /// Model configuration
    model_id: []const u8 = "llama-7b-q4km",
    model_path: []const u8 = "models/llama-7b-q4km.gguf",

    /// Storage backend type
    storage_backend: StorageBackend = .objectstore,
    /// Object store config (when storage_backend = .objectstore)
    objectstore: ObjectStoreConfig = .{},

    /// Docker image configuration
    docker: DockerConfig = .{},

    /// Resource plan (GPU type)
    resource_plan: ResourcePlan = .infer_s,

    /// GPU settings
    gpu_enabled: bool = true,
    gpu_device_id: u8 = 0,
    gpu_max_memory_gb: u16 = 16,

    /// Scenario ID for AI Core
    scenario_id: []const u8 = "nopena",

    /// Create from environment variables
    pub fn fromEnv(allocator: std.mem.Allocator) !AICoreConfig {
        _ = allocator;
        var config = AICoreConfig{};

        if (std.posix.getenv("AICORE_AUTH_URL")) |v| config.auth_url = v;
        if (std.posix.getenv("AICORE_API_URL")) |v| config.api_url = v;
        if (std.posix.getenv("AICORE_CLIENT_ID")) |v| config.client_id = v;
        if (std.posix.getenv("AICORE_CLIENT_SECRET")) |v| config.client_secret = v;
        if (std.posix.getenv("AICORE_RESOURCE_GROUP")) |v| config.resource_group = v;
        if (std.posix.getenv("MODEL_ID")) |v| config.model_id = v;
        if (std.posix.getenv("MODEL_PATH")) |v| config.model_path = v;

        return config;
    }

    /// Validate configuration
    pub fn validate(self: AICoreConfig) !void {
        if (self.client_id.len == 0) return error.MissingClientId;
        if (self.client_secret.len == 0) return error.MissingClientSecret;
        if (self.model_id.len == 0) return error.MissingModelId;
        if (self.storage_backend == .objectstore) {
            if (self.objectstore.access_key.len == 0) return error.MissingObjectStoreAccessKey;
            if (self.objectstore.secret_key.len == 0) return error.MissingObjectStoreSecretKey;
        }
    }
};

test "resource plan strings" {
    const testing = std.testing;
    try testing.expectEqualStrings("infer.s", ResourcePlan.infer_s.toString());
    try testing.expectEqual(@as(f32, 0.50), ResourcePlan.infer_s.getApproxHourlyCost());
}

test "docker config full path" {
    const testing = std.testing;
    var config = DockerConfig{ .registry = "docker.io/user", .image_name = "nopena", .tag = "1.0.0" };
    const path = try config.getFullImagePath(testing.allocator);
    defer testing.allocator.free(path);
    try testing.expectEqualStrings("docker.io/user/nopena:1.0.0", path);
}

