//! Configuration Loader for ModelRegistry
//! Parses config.json and populates ModelRegistry with available models

const std = @import("std");
const json = std.json;
const ModelRegistry = @import("model_registry.zig").ModelRegistry;
const ModelConfig = @import("model_registry.zig").ModelConfig;
const ModelMetadata = @import("model_registry.zig").ModelMetadata;
const ModelVersion = @import("model_registry.zig").ModelVersion;

pub const ConfigModel = struct {
    id: []const u8,
    name: []const u8,
    path: []const u8,
    architecture: []const u8,
    format: []const u8,
    size_mb: u64,
    quantization: []const u8,
    description: []const u8,
    status: []const u8,
    use_cases: [][]const u8 = &[_][]const u8{},
    tier_config: ?TierConfig = null,
    
    pub const TierConfig = struct {
        max_ram_mb: u32,
        kv_cache_ram_mb: u32 = 0,
        max_ssd_mb: u32 = 0,
        enable_distributed: bool = false,
        enable_compression: bool = false,
    };
};

pub const Config = struct {
    available_models: []ConfigModel,
};

/// Load models from config.json into ModelRegistry
pub fn loadModelsFromConfig(
    allocator: std.mem.Allocator,
    registry: *ModelRegistry,
    config_path: []const u8,
) !usize {
    // Read config file
    const file = std.fs.cwd().openFile(config_path, .{}) catch |err| {
        std.debug.print("❌ Failed to open config file '{s}': {}\n", .{ config_path, err });
        return 0;
    };
    defer file.close();
    
    const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024); // 10MB max
    defer allocator.free(content);
    
    // Parse JSON
    const parsed = json.parseFromSlice(json.Value, allocator, content, .{}) catch |err| {
        std.debug.print("❌ Failed to parse config JSON: {}\n", .{err});
        return 0;
    };
    defer parsed.deinit();
    
    const root = parsed.value.object;
    
    // Extract available_models array
    const models_value = root.get("available_models") orelse {
        std.debug.print("⚠️  No 'available_models' found in config\n", .{});
        return 0;
    };
    
    const models_array = models_value.array;
    var loaded_count: usize = 0;
    
    for (models_array.items) |model_value| {
        const model_obj = model_value.object;
        
        // Extract required fields
        const id = model_obj.get("id").?.string;
        const name = model_obj.get("name").?.string;
        const path = model_obj.get("path").?.string;
        const architecture = model_obj.get("architecture").?.string;
        const format = model_obj.get("format").?.string;
        const size_mb = @as(u64, @intFromFloat(model_obj.get("size_mb").?.float));
        const quantization = model_obj.get("quantization").?.string;
        _ = model_obj.get("description"); // Description not used in config creation
        const status = model_obj.get("status").?.string;
        
        // Build full model path
        const full_path = try std.fmt.allocPrint(
            allocator,
            "{s}/{s}",
            .{ registry.model_base_path, path },
        );
        
        // Extract tier config if present
        var max_ram: u32 = 4096; // Default 4GB
        var max_ssd: u32 = 0;
        
        if (model_obj.get("tier_config")) |tier_value| {
            max_ram = @as(u32, @intFromFloat(tier_value.object.get("max_ram_mb").?.float));
            max_ssd = if (tier_value.object.get("max_ssd_mb")) |v| @as(u32, @intFromFloat(v.float)) else 0;
        }
        
        // Create tags array
        const tags = try allocator.alloc([]const u8, 1);
        tags[0] = try allocator.dupe(u8, status);
        
        // Create metadata
        const metadata = ModelMetadata{
            .architecture = try allocator.dupe(u8, architecture),
            .quantization = try allocator.dupe(u8, quantization),
            .parameter_count = try extractParameterCount(allocator, name),
            .format = try allocator.dupe(u8, format),
            .context_length = 4096,
            .tags = tags,
            .source = try allocator.dupe(u8, "config"),
            .license = try allocator.dupe(u8, "unknown"),
            .created_at = std.time.timestamp(),
            .size_bytes = size_mb * 1024 * 1024,
        };
        
        // Determine if model should be preloaded
        const preload = if (model_obj.get("tier_config")) |_| blk: {
            // Preload small models (< 2GB) by default
            break :blk size_mb < 2048 and max_ssd == 0;
        } else false;
        
        // Create model config
        const config = try ModelConfig.init(allocator, .{
            .id = id,
            .path = full_path,
            .display_name = name,
            .version = .{ .major = 1, .minor = 0, .patch = 0 },
            .metadata = metadata,
            .preload = preload,
            .enabled = std.mem.eql(u8, status, "active") or std.mem.eql(u8, status, "ready"),
        });
        
        // Register in registry
        try registry.register(config);
        loaded_count += 1;
        
        std.debug.print("✅ Registered model: {s} ({d}MB, {s})\n", .{ 
            name, 
            size_mb,
            architecture,
        });
    }
    
    return loaded_count;
}

/// Extract parameter count from model name (e.g., "1.2B" from "LFM2.5 1.2B Q4_0")
fn extractParameterCount(allocator: std.mem.Allocator, name: []const u8) ![]const u8 {
    // Look for patterns like "1B", "1.2B", "33B", "70B"
    if (std.mem.indexOf(u8, name, "1.2B") != null) return try allocator.dupe(u8, "1.2B");
    if (std.mem.indexOf(u8, name, "0.5B") != null) return try allocator.dupe(u8, "0.5B");
    if (std.mem.indexOf(u8, name, "33B") != null) return try allocator.dupe(u8, "33B");
    if (std.mem.indexOf(u8, name, "70B") != null) return try allocator.dupe(u8, "70B");
    if (std.mem.indexOf(u8, name, "270M") != null or std.mem.indexOf(u8, name, "270m") != null) 
        return try allocator.dupe(u8, "270M");
    
    return try allocator.dupe(u8, "unknown");
}
