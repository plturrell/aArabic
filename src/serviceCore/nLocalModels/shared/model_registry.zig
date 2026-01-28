//! Enhanced Model Registry - Day 11
//! Multi-model management with versioning, metadata, and filesystem integration
//! Integrates with vendor/layerModels for model files and vendor/layerData for storage

const std = @import("std");
const builtin = @import("builtin");

// ============================================================================
// Model Version
// ============================================================================

pub const ModelVersion = struct {
    major: u32,
    minor: u32,
    patch: u32,

    pub fn format(self: ModelVersion, writer: anytype) !void {
        try writer.print("{d}.{d}.{d}", .{ self.major, self.minor, self.patch });
    }
    
    pub fn compare(self: ModelVersion, other: ModelVersion) std.math.Order {
        if (self.major != other.major) return std.math.order(self.major, other.major);
        if (self.minor != other.minor) return std.math.order(self.minor, other.minor);
        if (self.patch != other.patch) return std.math.order(self.patch, other.patch);
        return .eq;
    }
    
    pub fn parse(str: []const u8) !ModelVersion {
        var iter = std.mem.splitScalar(u8, str, '.');
        const major = try std.fmt.parseInt(u32, iter.next() orelse return error.InvalidVersion, 10);
        const minor = try std.fmt.parseInt(u32, iter.next() orelse return error.InvalidVersion, 10);
        const patch = try std.fmt.parseInt(u32, iter.rest(), 10);
        return ModelVersion{ .major = major, .minor = minor, .patch = patch };
    }
};

// ============================================================================
// Model Metadata
// ============================================================================

pub const ModelMetadata = struct {
    architecture: []const u8, // "llama", "phi", "qwen", "gemma", etc.
    quantization: []const u8, // "Q4_K_M", "Q8_0", "F16", etc.
    parameter_count: []const u8, // "1B", "3B", "7B", "70B", etc.
    format: []const u8, // "gguf", "safetensors", "pytorch", etc.
    context_length: u32,
    tags: []const []const u8,
    source: []const u8, // "huggingface", "local", "ollama"
    license: []const u8,
    created_at: i64, // Unix timestamp
    size_bytes: u64,
    
    pub fn deinit(self: *ModelMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.architecture);
        allocator.free(self.quantization);
        allocator.free(self.parameter_count);
        allocator.free(self.format);
        for (self.tags) |tag| allocator.free(tag);
        allocator.free(self.tags);
        allocator.free(self.source);
        allocator.free(self.license);
    }
};

// ============================================================================
// Model Config (Enhanced)
// ============================================================================

pub const ModelConfig = struct {
    id: []const u8,
    path: []const u8,
    display_name: []const u8,
    version: ModelVersion,
    metadata: ModelMetadata,
    preload: bool = false,
    max_workers: ?u32 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
    enabled: bool = true,
    health_status: HealthStatus = .unknown,
    last_used: ?i64 = null, // Unix timestamp
    use_count: u64 = 0,

    pub const HealthStatus = enum {
        unknown,
        healthy,
        degraded,
        unhealthy,
        loading,
    };

    pub const InitParams = struct {
        id: []const u8,
        path: []const u8,
        display_name: ?[]const u8 = null,
        version: ModelVersion = .{ .major = 1, .minor = 0, .patch = 0 },
        metadata: ModelMetadata,
        preload: bool = false,
        max_workers: ?u32 = null,
        max_tokens: ?u32 = null,
        temperature: ?f32 = null,
        enabled: bool = true,
    };

    pub fn init(allocator: std.mem.Allocator, params: InitParams) !ModelConfig {
        const display = params.display_name orelse params.id;
        return ModelConfig{
            .id = try allocator.dupe(u8, params.id),
            .path = try allocator.dupe(u8, params.path),
            .display_name = try allocator.dupe(u8, display),
            .version = params.version,
            .metadata = params.metadata,
            .preload = params.preload,
            .max_workers = params.max_workers,
            .max_tokens = params.max_tokens,
            .temperature = params.temperature,
            .enabled = params.enabled,
            .health_status = .unknown,
            .last_used = null,
            .use_count = 0,
        };
    }

    pub fn deinit(self: *ModelConfig, allocator: std.mem.Allocator) void {
        if (self.id.len != 0) allocator.free(self.id);
        if (self.path.len != 0) allocator.free(self.path);
        if (self.display_name.len != 0) allocator.free(self.display_name);
        self.metadata.deinit(allocator);
        self.* = undefined;
    }
    
    pub fn markUsed(self: *ModelConfig) void {
        self.use_count += 1;
        self.last_used = std.time.timestamp();
    }
    
    pub fn updateHealthStatus(self: *ModelConfig, status: HealthStatus) void {
        self.health_status = status;
    }
};

// ============================================================================
// Model Registry (Enhanced)
// ============================================================================

pub const ModelRegistry = struct {
    allocator: std.mem.Allocator,
    models: std.StringHashMap(ModelConfig),
    model_versions: std.StringHashMap(std.ArrayList(ModelVersion)),
    default_model_id: ?[]const u8,
    model_base_path: []const u8, // vendor/layerModels
    metadata_path: []const u8, // vendor/layerData
    
    pub const DiscoveryStats = struct {
        total_scanned: u32 = 0,
        models_found: u32 = 0,
        models_added: u32 = 0,
        models_updated: u32 = 0,
        errors: u32 = 0,
    };

    pub fn init(allocator: std.mem.Allocator, model_base_path: []const u8, metadata_path: []const u8) !ModelRegistry {
        return ModelRegistry{
            .allocator = allocator,
            .models = std.StringHashMap(ModelConfig).init(allocator),
            .model_versions = std.StringHashMap(std.ArrayList(ModelVersion)).init(allocator),
            .default_model_id = null,
            .model_base_path = try allocator.dupe(u8, model_base_path),
            .metadata_path = try allocator.dupe(u8, metadata_path),
        };
    }

    pub fn deinit(self: *ModelRegistry) void {
        var iter = self.models.valueIterator();
        while (iter.next()) |config| {
            var mut_config = config.*;
            mut_config.deinit(self.allocator);
        }
        self.models.deinit();
        
        var version_iter = self.model_versions.valueIterator();
        while (version_iter.next()) |versions| {
            versions.deinit();
        }
        self.model_versions.deinit();
        
        if (self.default_model_id) |id| self.allocator.free(id);
        self.allocator.free(self.model_base_path);
        self.allocator.free(self.metadata_path);
    }

    pub fn len(self: *const ModelRegistry) usize {
        return self.models.count();
    }

    pub fn register(self: *ModelRegistry, config: ModelConfig) !void {
        const id_copy = try self.allocator.dupe(u8, config.id);
        errdefer self.allocator.free(id_copy);
        
        try self.models.put(id_copy, config);
        
        // Track version
        const entry = try self.model_versions.getOrPut(id_copy);
        if (!entry.found_existing) {
            entry.value_ptr.* = std.ArrayList(ModelVersion).empty;
        }
        try entry.value_ptr.append(self.allocator, config.version);
        
        // Set as default if first model
        if (self.default_model_id == null) {
            self.default_model_id = try self.allocator.dupe(u8, id_copy);
        }
    }

    pub fn get(self: *const ModelRegistry, id: []const u8) ?*const ModelConfig {
        return self.models.getPtr(id);
    }
    
    pub fn getMut(self: *ModelRegistry, id: []const u8) ?*ModelConfig {
        return self.models.getPtr(id);
    }

    pub fn getByVersion(self: *const ModelRegistry, id: []const u8, version: ModelVersion) ?*const ModelConfig {
        if (self.models.getPtr(id)) |config| {
            if (config.version.compare(version) == .eq) {
                return config;
            }
        }
        return null;
    }

    pub fn default(self: *const ModelRegistry) ?*const ModelConfig {
        if (self.default_model_id) |id| {
            return self.get(id);
        }
        return null;
    }
    
    pub fn setDefault(self: *ModelRegistry, id: []const u8) !void {
        if (self.models.contains(id)) {
            if (self.default_model_id) |old_id| {
                self.allocator.free(old_id);
            }
            self.default_model_id = try self.allocator.dupe(u8, id);
        } else {
            return error.ModelNotFound;
        }
    }

    pub fn listModels(self: *const ModelRegistry, allocator: std.mem.Allocator) ![][]const u8 {
        var list = std.ArrayList([]const u8){};
        errdefer list.deinit();
        
        var iter = self.models.keyIterator();
        while (iter.next()) |key| {
            try list.append(try allocator.dupe(u8, key.*));
        }
        
        return try list.toOwnedSlice();
    }

    pub fn getVersions(self: *const ModelRegistry, id: []const u8) ?[]const ModelVersion {
        if (self.model_versions.get(id)) |versions| {
            return versions.items;
        }
        return null;
    }

    pub fn discoverModels(self: *ModelRegistry) !DiscoveryStats {
        var stats = DiscoveryStats{};
        
        // Open model base directory
        var dir = std.fs.cwd().openDir(self.model_base_path, .{ .iterate = true }) catch |err| {
            std.debug.print("Failed to open model directory '{s}': {}\n", .{ self.model_base_path, err });
            stats.errors += 1;
            return stats;
        };
        defer dir.close();
        
        // Iterate through model directories
        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .directory) continue;
            
            stats.total_scanned += 1;
            
            // Try to discover model in this directory
            self.discoverModelInDirectory(entry.name, &stats) catch |err| {
                std.debug.print("Error discovering model in '{s}': {}\n", .{ entry.name, err });
                stats.errors += 1;
            };
        }
        
        return stats;
    }

    fn discoverModelInDirectory(self: *ModelRegistry, dir_name: []const u8, stats: *DiscoveryStats) !void {
        // Extract metadata from directory name
        // Format: vendor-model-size (e.g., "Llama-3.2-1B", "google-gemma-3-270m-it")
        const metadata = try self.parseModelDirectoryName(dir_name);
        defer {
            self.allocator.free(metadata.architecture);
            self.allocator.free(metadata.parameter_count);
        }
        
        // Check if model already exists
        if (self.models.contains(dir_name)) {
            stats.models_updated += 1;
            return;
        }
        
        // Build full path
        const full_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}",
            .{ self.model_base_path, dir_name },
        );
        defer self.allocator.free(full_path);
        
        // Get directory size
        const size_bytes = try self.getDirectorySize(full_path);
        
        // Create model config
        const tags = try self.allocator.alloc([]const u8, 1);
        tags[0] = try self.allocator.dupe(u8, "local");
        
        const model_metadata = ModelMetadata{
            .architecture = try self.allocator.dupe(u8, metadata.architecture),
            .quantization = try self.allocator.dupe(u8, "unknown"),
            .parameter_count = try self.allocator.dupe(u8, metadata.parameter_count),
            .format = try self.allocator.dupe(u8, "gguf"),
            .context_length = 4096,
            .tags = tags,
            .source = try self.allocator.dupe(u8, "local"),
            .license = try self.allocator.dupe(u8, "unknown"),
            .created_at = std.time.timestamp(),
            .size_bytes = size_bytes,
        };
        
        const config = try ModelConfig.init(self.allocator, .{
            .id = dir_name,
            .path = full_path,
            .display_name = dir_name,
            .version = .{ .major = 1, .minor = 0, .patch = 0 },
            .metadata = model_metadata,
            .preload = false,
        });
        
        try self.register(config);
        stats.models_found += 1;
        stats.models_added += 1;
    }

    fn parseModelDirectoryName(self: *ModelRegistry, name: []const u8) !struct {
        architecture: []const u8,
        parameter_count: []const u8,
    } {
        // Extract architecture and parameter count from directory name
        var arch: []const u8 = "unknown";
        var params: []const u8 = "unknown";
        
        // Check for known architectures
        if (std.mem.indexOf(u8, name, "translategemma") != null or std.mem.indexOf(u8, name, "TranslateGemma") != null) {
            arch = "gemma2";  // TranslateGemma uses Gemma2 architecture
        } else if (std.mem.indexOf(u8, name, "Llama") != null or std.mem.indexOf(u8, name, "llama") != null) {
            arch = "llama";
        } else if (std.mem.indexOf(u8, name, "phi") != null or std.mem.indexOf(u8, name, "Phi") != null) {
            arch = "phi";
        } else if (std.mem.indexOf(u8, name, "Qwen") != null or std.mem.indexOf(u8, name, "qwen") != null) {
            arch = "qwen";
        } else if (std.mem.indexOf(u8, name, "gemma") != null or std.mem.indexOf(u8, name, "Gemma") != null) {
            arch = "gemma";
        } else if (std.mem.indexOf(u8, name, "Nemotron") != null) {
            arch = "nemotron";
        } else if (std.mem.indexOf(u8, name, "deepseek") != null or std.mem.indexOf(u8, name, "DeepSeek") != null) {
            arch = "deepseek";
        } else if (std.mem.indexOf(u8, name, "LFM") != null) {
            arch = "lfm";  // Liquid Foundation Model
        }
        
        // Extract parameter count (look for patterns like "1B", "3B", "270m")
        if (std.mem.indexOf(u8, name, "27b") != null or std.mem.indexOf(u8, name, "27B") != null) {
            params = "27B";
        } else if (std.mem.indexOf(u8, name, "33b") != null or std.mem.indexOf(u8, name, "33B") != null) {
            params = "33B";
        } else if (std.mem.indexOf(u8, name, "70B") != null or std.mem.indexOf(u8, name, "70b") != null) {
            params = "70B";
        } else if (std.mem.indexOf(u8, name, "1B") != null or std.mem.indexOf(u8, name, "1b") != null) {
            params = "1B";
        } else if (std.mem.indexOf(u8, name, "3B") != null or std.mem.indexOf(u8, name, "3b") != null) {
            params = "3B";
        } else if (std.mem.indexOf(u8, name, "7B") != null or std.mem.indexOf(u8, name, "7b") != null) {
            params = "7B";
        } else if (std.mem.indexOf(u8, name, "270m") != null or std.mem.indexOf(u8, name, "270M") != null) {
            params = "270M";
        } else if (std.mem.indexOf(u8, name, "0.5B") != null) {
            params = "0.5B";
        } else if (std.mem.indexOf(u8, name, "1.2B") != null) {
            params = "1.2B";
        }
        
        return .{
            .architecture = try self.allocator.dupe(u8, arch),
            .parameter_count = try self.allocator.dupe(u8, params),
        };
    }

    fn getDirectorySize(self: *ModelRegistry, path: []const u8) !u64 {
        var dir = try std.fs.cwd().openDir(path, .{ .iterate = true });
        defer dir.close();
        
        var total_size: u64 = 0;
        var iter = dir.iterate();
        
        while (try iter.next()) |entry| {
            if (entry.kind == .file) {
                const stat = try dir.statFile(entry.name);
                total_size += stat.size;
            } else if (entry.kind == .directory) {
                // Recursively get subdirectory size
                const subpath = try std.fmt.allocPrint(
                    std.heap.page_allocator,
                    "{s}/{s}",
                    .{ path, entry.name },
                );
                defer std.heap.page_allocator.free(subpath);
                total_size += try self.getDirectorySize(subpath);
            }
        }
        
        return total_size;
    }

    pub fn getHealthyModels(self: *const ModelRegistry, allocator: std.mem.Allocator) ![][]const u8 {
        var list = std.ArrayList([]const u8){};
        errdefer list.deinit();
        
        var iter = self.models.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.health_status == .healthy and entry.value_ptr.enabled) {
                try list.append(try allocator.dupe(u8, entry.key_ptr.*));
            }
        }
        
        return try list.toOwnedSlice();
    }

    pub fn toJson(self: *const ModelRegistry, alloc: std.mem.Allocator) ![]u8 {
        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(alloc);

        try buffer.appendSlice(alloc, "{\"object\":\"list\",\"data\":[");

        var iter = self.models.valueIterator();
        var first = true;
        while (iter.next()) |cfg| {
            if (!first) try buffer.appendSlice(alloc, ",");
            first = false;

            const escaped_id = try escapeJsonString(alloc, cfg.id);
            defer alloc.free(escaped_id);
            const escaped_name = try escapeJsonString(alloc, cfg.display_name);
            defer alloc.free(escaped_name);
            const escaped_path = try escapeJsonString(alloc, cfg.path);
            defer alloc.free(escaped_path);

            try buffer.appendSlice(alloc, "{\"id\":");
            try buffer.appendSlice(alloc, escaped_id);
            try buffer.appendSlice(alloc, ",\"object\":\"model\"");
            try buffer.appendSlice(alloc, ",\"created\":");
            try buffer.writer(alloc).print("{d}", .{cfg.metadata.created_at});
            try buffer.appendSlice(alloc, ",\"owned_by\":\"shimmy-mojo\"");
            try buffer.appendSlice(alloc, ",\"display_name\":");
            try buffer.appendSlice(alloc, escaped_name);
            try buffer.appendSlice(alloc, ",\"path\":");
            try buffer.appendSlice(alloc, escaped_path);
            try buffer.appendSlice(alloc, ",\"architecture\":\"");
            try buffer.appendSlice(alloc, cfg.metadata.architecture);
            try buffer.appendSlice(alloc, "\",\"quantization\":\"");
            try buffer.appendSlice(alloc, cfg.metadata.quantization);
            try buffer.appendSlice(alloc, "\",\"parameter_count\":\"");
            try buffer.appendSlice(alloc, cfg.metadata.parameter_count);
            try buffer.appendSlice(alloc, "\",\"format\":\"");
            try buffer.appendSlice(alloc, cfg.metadata.format);
            try buffer.appendSlice(alloc, "\",\"size_mb\":");
            try buffer.writer(alloc).print("{d}", .{cfg.metadata.size_bytes / (1024 * 1024)});
            try buffer.appendSlice(alloc, ",\"size_bytes\":");
            try buffer.writer(alloc).print("{d}", .{cfg.metadata.size_bytes});
            try buffer.appendSlice(alloc, ",\"enabled\":");
            try buffer.appendSlice(alloc, if (cfg.enabled) "true" else "false");
            try buffer.appendSlice(alloc, ",\"health_status\":\"");
            try buffer.appendSlice(alloc, @tagName(cfg.health_status));
            try buffer.appendSlice(alloc, "\",\"use_count\":");
            try buffer.writer(alloc).print("{d}", .{cfg.use_count});
            try buffer.appendSlice(alloc, ",\"preload\":");
            try buffer.appendSlice(alloc, if (cfg.preload) "true" else "false");

            if (cfg.max_workers) |workers| {
                try buffer.appendSlice(alloc, ",\"max_workers\":");
                try buffer.writer(alloc).print("{d}", .{workers});
            }
            if (cfg.max_tokens) |tokens| {
                try buffer.appendSlice(alloc, ",\"max_tokens\":");
                try buffer.writer(alloc).print("{d}", .{tokens});
            }
            if (cfg.temperature) |temp| {
                try buffer.appendSlice(alloc, ",\"temperature\":");
                try buffer.writer(alloc).print("{d}", .{temp});
            }
            try buffer.appendSlice(alloc, "}");
        }
        try buffer.appendSlice(alloc, "]}");

        return try buffer.toOwnedSlice(alloc);
    }
    
    // Legacy compatibility
    pub fn initLegacy(configs: []const ModelConfig) ModelRegistry {
        _ = configs;
        @panic("Use init() with allocator instead");
    }
};

fn oldToJson(self: *const ModelRegistry, allocator: std.mem.Allocator) ![]u8 {
    _ = self;
    _ = allocator;
    @panic("Use new toJson implementation");
}

fn oldToJsonImpl(configs: []const ModelConfig, allocator: std.mem.Allocator) ![]u8 {
        _ = configs;
        var buffer = std.ArrayList(u8).empty;
        errdefer buffer.deinit(allocator);
        try buffer.appendSlice(allocator, "{\"object\":\"list\",\"data\":[]}");
        return try buffer.toOwnedSlice(allocator);
    }

fn escapeJsonString(alloc: std.mem.Allocator, input: []const u8) ![]u8 {
    var result = std.ArrayList(u8).empty;
    errdefer result.deinit(alloc);

    try result.append(alloc, '"');
    for (input) |c| {
        switch (c) {
            '"' => try result.appendSlice(alloc, "\\\""),
            '\\' => try result.appendSlice(alloc, "\\\\"),
            '\n' => try result.appendSlice(alloc, "\\n"),
            '\r' => try result.appendSlice(alloc, "\\r"),
            '\t' => try result.appendSlice(alloc, "\\t"),
            0x00...0x08, 0x0b, 0x0c, 0x0e...0x1f => {
                var buf: [6]u8 = undefined;
                _ = std.fmt.bufPrint(&buf, "\\u{x:0>4}", .{c}) catch unreachable;
                try result.appendSlice(alloc, &buf);
            },
            else => try result.append(alloc, c),
        }
    }
    try result.append(alloc, '"');
    return try result.toOwnedSlice(alloc);
}
