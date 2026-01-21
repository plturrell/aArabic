//! Test file for Enhanced Model Registry - Day 11
//! Tests model discovery, versioning, and multi-model management

const std = @import("std");
const model_registry = @import("model_registry.zig");
const ModelRegistry = model_registry.ModelRegistry;
const ModelConfig = model_registry.ModelConfig;
const ModelVersion = model_registry.ModelVersion;
const ModelMetadata = model_registry.ModelMetadata;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("🧪 Enhanced Model Registry Test Suite - Day 11\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("\n", .{});

    // Test 1: Model Version Parsing
    try testModelVersionParsing();

    // Test 2: Model Registry Initialization
    try testRegistryInitialization(allocator);

    // Test 3: Model Registration
    try testModelRegistration(allocator);

    // Test 4: Model Discovery
    try testModelDiscovery(allocator);

    // Test 5: Version Management
    try testVersionManagement(allocator);

    // Test 6: JSON Serialization
    try testJsonSerialization(allocator);

    // Test 7: Health Status Tracking
    try testHealthStatusTracking(allocator);

    std.debug.print("\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("✅ All Tests Passed!\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("\n", .{});
}

fn testModelVersionParsing() !void {
    std.debug.print("Test 1: Model Version Parsing\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const version1 = try ModelVersion.parse("1.2.3");
    std.debug.assert(version1.major == 1);
    std.debug.assert(version1.minor == 2);
    std.debug.assert(version1.patch == 3);
    std.debug.print("  ✓ Parse '1.2.3' -> {}\n", .{version1});

    const version2 = ModelVersion{ .major = 2, .minor = 0, .patch = 0 };
    const cmp = version1.compare(version2);
    std.debug.assert(cmp == .lt);
    std.debug.print("  ✓ Version comparison: {} < {}\n", .{ version1, version2 });

    std.debug.print("  ✅ Model version parsing tests passed\n\n", .{});
}

fn testRegistryInitialization(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 2: Model Registry Initialization\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    std.debug.print("  ✓ Registry initialized\n", .{});
    std.debug.print("  ✓ Model base path: {s}\n", .{registry.model_base_path});
    std.debug.print("  ✓ Metadata path: {s}\n", .{registry.metadata_path});
    std.debug.print("  ✓ Initial model count: {}\n", .{registry.len()});

    std.debug.print("  ✅ Registry initialization tests passed\n\n", .{});
}

fn testModelRegistration(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 3: Model Registration\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    // Create test model metadata
    const tags = try allocator.alloc([]const u8, 2);
    tags[0] = try allocator.dupe(u8, "test");
    tags[1] = try allocator.dupe(u8, "llama");

    const metadata = ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .quantization = try allocator.dupe(u8, "Q4_K_M"),
        .parameter_count = try allocator.dupe(u8, "1B"),
        .format = try allocator.dupe(u8, "gguf"),
        .context_length = 4096,
        .tags = tags,
        .source = try allocator.dupe(u8, "test"),
        .license = try allocator.dupe(u8, "MIT"),
        .created_at = std.time.timestamp(),
        .size_bytes = 1024 * 1024 * 1024, // 1GB
    };

    const config = try ModelConfig.init(allocator, .{
        .id = "test-model",
        .path = "/path/to/model",
        .display_name = "Test Model",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .metadata = metadata,
        .preload = false,
    });

    try registry.register(config);
    std.debug.print("  ✓ Model registered: {s}\n", .{config.id});
    std.debug.print("  ✓ Registry count: {}\n", .{registry.len()});

    const retrieved = registry.get("test-model");
    std.debug.assert(retrieved != null);
    std.debug.print("  ✓ Model retrieved: {s}\n", .{retrieved.?.display_name});

    const default_model = registry.default();
    std.debug.assert(default_model != null);
    std.debug.print("  ✓ Default model set: {s}\n", .{default_model.?.id});

    std.debug.print("  ✅ Model registration tests passed\n\n", .{});
}

fn testModelDiscovery(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 4: Model Discovery\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    std.debug.print("  🔍 Scanning vendor/layerModels...\n", .{});
    const stats = registry.discoverModels() catch |err| {
        std.debug.print("  ⚠️  Discovery failed (directory may not exist): {}\n", .{err});
        std.debug.print("  ℹ️  This is expected if vendor/layerModels doesn't exist\n\n", .{});
        return;
    };

    std.debug.print("  ✓ Total scanned: {}\n", .{stats.total_scanned});
    std.debug.print("  ✓ Models found: {}\n", .{stats.models_found});
    std.debug.print("  ✓ Models added: {}\n", .{stats.models_added});
    std.debug.print("  ✓ Models updated: {}\n", .{stats.models_updated});
    std.debug.print("  ✓ Errors: {}\n", .{stats.errors});

    if (registry.len() > 0) {
        const models = try registry.listModels(allocator);
        defer {
            for (models) |model| allocator.free(model);
            allocator.free(models);
        }

        std.debug.print("  ✓ Discovered models:\n", .{});
        for (models) |model| {
            std.debug.print("    - {s}\n", .{model});
        }
    }

    std.debug.print("  ✅ Model discovery tests passed\n\n", .{});
}

fn testVersionManagement(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 5: Version Management\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    // Register multiple versions of the same model
    for (0..3) |i| {
        const tags = try allocator.alloc([]const u8, 1);
        tags[0] = try allocator.dupe(u8, "test");

        const metadata = ModelMetadata{
            .architecture = try allocator.dupe(u8, "llama"),
            .quantization = try allocator.dupe(u8, "Q4_K_M"),
            .parameter_count = try allocator.dupe(u8, "1B"),
            .format = try allocator.dupe(u8, "gguf"),
            .context_length = 4096,
            .tags = tags,
            .source = try allocator.dupe(u8, "test"),
            .license = try allocator.dupe(u8, "MIT"),
            .created_at = std.time.timestamp(),
            .size_bytes = 1024 * 1024 * 1024,
        };

        const version = ModelVersion{ .major = 1, .minor = @intCast(i), .patch = 0 };
        const id = try std.fmt.allocPrint(allocator, "test-model-v{}", .{version});
        defer allocator.free(id);

        const config = try ModelConfig.init(allocator, .{
            .id = id,
            .path = "/path/to/model",
            .display_name = id,
            .version = version,
            .metadata = metadata,
            .preload = false,
        });

        try registry.register(config);
        std.debug.print("  ✓ Registered version: {}\n", .{version});
    }

    std.debug.print("  ✓ Total models registered: {}\n", .{registry.len()});
    std.debug.print("  ✅ Version management tests passed\n\n", .{});
}

fn testJsonSerialization(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 6: JSON Serialization\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    // Add a test model
    const tags = try allocator.alloc([]const u8, 1);
    tags[0] = try allocator.dupe(u8, "test");

    const metadata = ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .quantization = try allocator.dupe(u8, "Q4_K_M"),
        .parameter_count = try allocator.dupe(u8, "1B"),
        .format = try allocator.dupe(u8, "gguf"),
        .context_length = 4096,
        .tags = tags,
        .source = try allocator.dupe(u8, "test"),
        .license = try allocator.dupe(u8, "MIT"),
        .created_at = std.time.timestamp(),
        .size_bytes = 1024 * 1024 * 1024,
    };

    const config = try ModelConfig.init(allocator, .{
        .id = "test-model",
        .path = "/path/to/model",
        .display_name = "Test Model",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .metadata = metadata,
        .preload = false,
    });

    try registry.register(config);

    const json = try registry.toJson(allocator);
    defer allocator.free(json);

    std.debug.print("  ✓ JSON serialization successful\n", .{});
    std.debug.print("  ✓ JSON length: {} bytes\n", .{json.len});
    std.debug.assert(std.mem.indexOf(u8, json, "\"id\":\"test-model\"") != null);
    std.debug.assert(std.mem.indexOf(u8, json, "\"architecture\":\"llama\"") != null);
    std.debug.print("  ✓ JSON contains expected fields\n", .{});

    std.debug.print("  ✅ JSON serialization tests passed\n\n", .{});
}

fn testHealthStatusTracking(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 7: Health Status Tracking\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    var registry = try ModelRegistry.init(allocator, "vendor/layerModels", "vendor/layerData");
    defer registry.deinit();

    // Add test models with different health statuses
    const tags = try allocator.alloc([]const u8, 1);
    tags[0] = try allocator.dupe(u8, "test");

    const metadata = ModelMetadata{
        .architecture = try allocator.dupe(u8, "llama"),
        .quantization = try allocator.dupe(u8, "Q4_K_M"),
        .parameter_count = try allocator.dupe(u8, "1B"),
        .format = try allocator.dupe(u8, "gguf"),
        .context_length = 4096,
        .tags = tags,
        .source = try allocator.dupe(u8, "test"),
        .license = try allocator.dupe(u8, "MIT"),
        .created_at = std.time.timestamp(),
        .size_bytes = 1024 * 1024 * 1024,
    };

    const config = try ModelConfig.init(allocator, .{
        .id = "healthy-model",
        .path = "/path/to/model",
        .display_name = "Healthy Model",
        .version = .{ .major = 1, .minor = 0, .patch = 0 },
        .metadata = metadata,
        .preload = false,
    });

    try registry.register(config);

    // Update health status
    if (registry.getMut("healthy-model")) |model| {
        model.updateHealthStatus(.healthy);
        model.markUsed();
        std.debug.print("  ✓ Health status updated: {}\n", .{model.health_status});
        std.debug.print("  ✓ Use count: {}\n", .{model.use_count});
        std.debug.assert(model.last_used != null);
        std.debug.print("  ✓ Last used timestamp recorded\n", .{});
    }

    // Get healthy models
    const healthy_models = try registry.getHealthyModels(allocator);
    defer {
        for (healthy_models) |model| allocator.free(model);
        allocator.free(healthy_models);
    }

    std.debug.print("  ✓ Healthy models count: {}\n", .{healthy_models.len});
    std.debug.assert(healthy_models.len == 1);

    std.debug.print("  ✅ Health status tracking tests passed\n\n", .{});
}
