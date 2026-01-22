// SAP AI Core configuration for model deployment and inference
// Handles resource groups, deployments, GPU allocation, and authentication

const std = @import("std");

// ============================================================================
// GPU Type Enum
// ============================================================================

pub const GpuType = enum {
    T4,
    V100,
    A100,

    pub fn toString(self: GpuType) []const u8 {
        return switch (self) {
            .T4 => "T4",
            .V100 => "V100",
            .A100 => "A100",
        };
    }

    pub fn fromString(str: []const u8) ?GpuType {
        if (std.mem.eql(u8, str, "T4")) return .T4;
        if (std.mem.eql(u8, str, "V100")) return .V100;
        if (std.mem.eql(u8, str, "A100")) return .A100;
        return null;
    }
};

// ============================================================================
// Configuration Errors
// ============================================================================

pub const ConfigError = error{
    MissingResourceGroup,
    MissingScenarioId,
    MissingExecutableId,
    MissingModelName,
    MissingModelVersion,
    MissingApiUrl,
    InvalidGpuCount,
    InvalidMemory,
    InvalidReplicaConfig,
    OutOfMemory,
    JsonSerializationError,
};

// ============================================================================
// SAP AI Core Configuration
// ============================================================================

pub const AICoreConfig = struct {
    allocator: std.mem.Allocator,

    // SAP AI Core identifiers
    resource_group: []const u8,
    scenario_id: []const u8,
    executable_id: []const u8,
    deployment_id: ?[]const u8,

    // Model configuration
    model_name: []const u8,
    model_version: []const u8,

    // API configuration
    api_url: []const u8,
    auth_token: ?[]const u8,

    // GPU configuration
    gpu_type: GpuType,
    gpu_count: u8,
    memory_gb: u16,

    // Scaling configuration
    min_replicas: u8,
    max_replicas: u8,

    // Track which strings we own (allocated)
    owned_strings: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    /// Initialize a new AICoreConfig with required fields
    pub fn init(
        allocator: std.mem.Allocator,
        resource_group: []const u8,
        scenario_id: []const u8,
        executable_id: []const u8,
        model_name: []const u8,
        model_version: []const u8,
        api_url: []const u8,
    ) Self {
        return Self{
            .allocator = allocator,
            .resource_group = resource_group,
            .scenario_id = scenario_id,
            .executable_id = executable_id,
            .deployment_id = null,
            .model_name = model_name,
            .model_version = model_version,
            .api_url = api_url,
            .auth_token = null,
            .gpu_type = .T4,
            .gpu_count = 1,
            .memory_gb = 16,
            .min_replicas = 1,
            .max_replicas = 1,
            .owned_strings = .{},
        };
    }

    /// Free all allocated memory
    pub fn deinit(self: *Self) void {
        // Free all owned strings
        for (self.owned_strings.items) |str| {
            self.allocator.free(str);
        }
        self.owned_strings.deinit(self.allocator);
    }

    /// Allocate and track a string copy
    fn allocString(self: *Self, source: []const u8) ![]const u8 {
        const copy = try self.allocator.dupe(u8, source);
        try self.owned_strings.append(copy);
        return copy;
    }

    /// Load configuration from environment variables (AICORE_* prefix)
    pub fn fromEnv(allocator: std.mem.Allocator) !Self {
        var config = Self{
            .allocator = allocator,
            .resource_group = "",
            .scenario_id = "",
            .executable_id = "",
            .deployment_id = null,
            .model_name = "",
            .model_version = "",
            .api_url = "",
            .auth_token = null,
            .gpu_type = .T4,
            .gpu_count = 1,
            .memory_gb = 16,
            .min_replicas = 1,
            .max_replicas = 1,
            .owned_strings = .{},
        };
        errdefer config.deinit();

        // Required fields
        if (std.posix.getenv("AICORE_RESOURCE_GROUP")) |v| {
            config.resource_group = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_SCENARIO_ID")) |v| {
            config.scenario_id = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_EXECUTABLE_ID")) |v| {
            config.executable_id = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_MODEL_NAME")) |v| {
            config.model_name = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_MODEL_VERSION")) |v| {
            config.model_version = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_API_URL")) |v| {
            config.api_url = try config.allocString(v);
        }

        // Optional fields
        if (std.posix.getenv("AICORE_DEPLOYMENT_ID")) |v| {
            config.deployment_id = try config.allocString(v);
        }
        if (std.posix.getenv("AICORE_AUTH_TOKEN")) |v| {
            config.auth_token = try config.allocString(v);
        }

        // GPU configuration
        if (std.posix.getenv("AICORE_GPU_TYPE")) |v| {
            if (GpuType.fromString(v)) |gpu_type| {
                config.gpu_type = gpu_type;
            }
        }
        if (getEnvU8("AICORE_GPU_COUNT")) |v| {
            config.gpu_count = v;
        }
        if (getEnvU16("AICORE_MEMORY_GB")) |v| {
            config.memory_gb = v;
        }

        // Scaling configuration
        if (getEnvU8("AICORE_MIN_REPLICAS")) |v| {
            config.min_replicas = v;
        }
        if (getEnvU8("AICORE_MAX_REPLICAS")) |v| {
            config.max_replicas = v;
        }

        return config;
    }

    /// Validate all required fields are set and values are valid
    pub fn validate(self: *const Self) ConfigError!void {
        if (self.resource_group.len == 0) return ConfigError.MissingResourceGroup;
        if (self.scenario_id.len == 0) return ConfigError.MissingScenarioId;
        if (self.executable_id.len == 0) return ConfigError.MissingExecutableId;
        if (self.model_name.len == 0) return ConfigError.MissingModelName;
        if (self.model_version.len == 0) return ConfigError.MissingModelVersion;
        if (self.api_url.len == 0) return ConfigError.MissingApiUrl;
        if (self.gpu_count == 0) return ConfigError.InvalidGpuCount;
        if (self.memory_gb == 0) return ConfigError.InvalidMemory;
        if (self.min_replicas > self.max_replicas) return ConfigError.InvalidReplicaConfig;
    }

    /// Serialize configuration to JSON string
    pub fn toJson(self: *const Self) ![]u8 {
        var buffer: std.ArrayListUnmanaged(u8) = .{};
        errdefer buffer.deinit(self.allocator);

        const writer = buffer.writer(self.allocator);

        try writer.writeAll("{\n");
        try writer.print("  \"resource_group\": \"{s}\",\n", .{self.resource_group});
        try writer.print("  \"scenario_id\": \"{s}\",\n", .{self.scenario_id});
        try writer.print("  \"executable_id\": \"{s}\",\n", .{self.executable_id});

        if (self.deployment_id) |dep_id| {
            try writer.print("  \"deployment_id\": \"{s}\",\n", .{dep_id});
        } else {
            try writer.writeAll("  \"deployment_id\": null,\n");
        }

        try writer.print("  \"model_name\": \"{s}\",\n", .{self.model_name});
        try writer.print("  \"model_version\": \"{s}\",\n", .{self.model_version});
        try writer.print("  \"api_url\": \"{s}\",\n", .{self.api_url});

        if (self.auth_token) |_| {
            try writer.writeAll("  \"auth_token\": \"***REDACTED***\",\n");
        } else {
            try writer.writeAll("  \"auth_token\": null,\n");
        }

        try writer.print("  \"gpu_type\": \"{s}\",\n", .{self.gpu_type.toString()});
        try writer.print("  \"gpu_count\": {d},\n", .{self.gpu_count});
        try writer.print("  \"memory_gb\": {d},\n", .{self.memory_gb});
        try writer.print("  \"min_replicas\": {d},\n", .{self.min_replicas});
        try writer.print("  \"max_replicas\": {d}\n", .{self.max_replicas});
        try writer.writeAll("}");

        return buffer.toOwnedSlice();
    }
};

// ============================================================================
// Environment Variable Helpers
// ============================================================================

fn getEnvU8(name: []const u8) ?u8 {
    if (std.posix.getenv(name)) |value| {
        return std.fmt.parseInt(u8, value, 10) catch null;
    }
    return null;
}

fn getEnvU16(name: []const u8) ?u16 {
    if (std.posix.getenv(name)) |value| {
        return std.fmt.parseInt(u16, value, 10) catch null;
    }
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "AICoreConfig: init and deinit" {
    const allocator = std.testing.allocator;

    var config = AICoreConfig.init(
        allocator,
        "default",
        "llm-scenario",
        "llm-executable",
        "llama-3",
        "1.0.0",
        "https://api.ai.sap.cloud",
    );
    defer config.deinit();

    try std.testing.expectEqualStrings("default", config.resource_group);
    try std.testing.expectEqualStrings("llm-scenario", config.scenario_id);
    try std.testing.expectEqual(@as(u8, 1), config.gpu_count);
    try std.testing.expectEqual(GpuType.T4, config.gpu_type);
}

test "AICoreConfig: validate passes for valid config" {
    const allocator = std.testing.allocator;

    var config = AICoreConfig.init(
        allocator,
        "default",
        "llm-scenario",
        "llm-executable",
        "llama-3",
        "1.0.0",
        "https://api.ai.sap.cloud",
    );
    defer config.deinit();

    try config.validate();
}

test "AICoreConfig: validate fails for missing resource_group" {
    const allocator = std.testing.allocator;

    var config = AICoreConfig.init(
        allocator,
        "",
        "llm-scenario",
        "llm-executable",
        "llama-3",
        "1.0.0",
        "https://api.ai.sap.cloud",
    );
    defer config.deinit();

    try std.testing.expectError(ConfigError.MissingResourceGroup, config.validate());
}

test "AICoreConfig: validate fails for invalid replica config" {
    const allocator = std.testing.allocator;

    var config = AICoreConfig.init(
        allocator,
        "default",
        "llm-scenario",
        "llm-executable",
        "llama-3",
        "1.0.0",
        "https://api.ai.sap.cloud",
    );
    defer config.deinit();

    config.min_replicas = 5;
    config.max_replicas = 2;

    try std.testing.expectError(ConfigError.InvalidReplicaConfig, config.validate());
}

test "AICoreConfig: toJson serialization" {
    const allocator = std.testing.allocator;

    var config = AICoreConfig.init(
        allocator,
        "default",
        "llm-scenario",
        "llm-executable",
        "llama-3",
        "1.0.0",
        "https://api.ai.sap.cloud",
    );
    defer config.deinit();

    config.gpu_type = .A100;
    config.gpu_count = 4;
    config.memory_gb = 80;

    const json = try config.toJson();
    defer allocator.free(json);

    try std.testing.expect(std.mem.indexOf(u8, json, "\"resource_group\": \"default\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gpu_type\": \"A100\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"gpu_count\": 4") != null);
    try std.testing.expect(std.mem.indexOf(u8, json, "\"memory_gb\": 80") != null);
}

test "GpuType: toString and fromString" {
    try std.testing.expectEqualStrings("T4", GpuType.T4.toString());
    try std.testing.expectEqualStrings("V100", GpuType.V100.toString());
    try std.testing.expectEqualStrings("A100", GpuType.A100.toString());

    try std.testing.expectEqual(GpuType.T4, GpuType.fromString("T4").?);
    try std.testing.expectEqual(GpuType.V100, GpuType.fromString("V100").?);
    try std.testing.expectEqual(GpuType.A100, GpuType.fromString("A100").?);
    try std.testing.expectEqual(@as(?GpuType, null), GpuType.fromString("INVALID"));
}

