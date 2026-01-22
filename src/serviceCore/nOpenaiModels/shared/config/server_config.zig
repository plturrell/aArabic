const std = @import("std");
const ModelRegistry = @import("../model_registry.zig");

/// Server Configuration
/// Supports environment variables and config file overrides
pub const ServerConfig = struct {
    // Network settings
    host: []const u8 = "127.0.0.1",
    port: u16 = 11434,
    
    // Thread pool settings
    num_workers: u32 = 4,
    max_connections: u32 = 100,
    
    // Buffer sizes
    max_request_bytes: usize = 16 * 1024 * 1024, // 16MB
    response_buffer_size: usize = 64 * 1024, // 64KB
    read_buffer_size: usize = 4096,
    
    // Timeouts (milliseconds)
    request_timeout_ms: u32 = 30000, // 30s
    generation_timeout_ms: u32 = 300000, // 5min
    keepalive_timeout_ms: u32 = 5000, // 5s
    
    // Model settings
    default_model_id: []const u8 = "shimmy-local",
    model_path: ?[]const u8 = null,
    auto_load_model: bool = true,
    models: []ModelRegistry.ModelConfig = &.{},
    
    // Generation defaults
    default_max_tokens: u32 = 512,
    default_temperature: f32 = 0.7,
    default_top_p: f32 = 0.9,
    default_top_k: u32 = 40,
    
    // Logging
    debug_enabled: bool = false,
    log_requests: bool = true,

    // Authentication
    api_key: ?[]const u8 = null,

    // Rate limiting
    // NOTE: This rate limiter is per-instance. For distributed deployments with
    // multiple instances behind a load balancer, use an external rate limiter:
    //   - Redis/DragonflyDB with INCR + EXPIRE for sliding window
    //   - nginx rate limiting module
    //   - Envoy/Istio rate limiting
    //   - Cloud provider rate limiting (AWS WAF, GCP Cloud Armor)
    // See: distributed_tier.zig for DragonflyDB-based distributed rate limiting
    rate_limit_requests_per_sec: u32 = 100,
    rate_limit_burst: u32 = 200,

    // Memory requirements
    min_memory_mb: u32 = 0, // 0 = no check
    
    /// Load configuration from environment variables
    pub fn fromEnv(allocator: std.mem.Allocator) !ServerConfig {
        var config = ServerConfig{};
        
        // Network
        if (std.posix.getenv("SHIMMY_HOST")) |v| config.host = v;
        if (std.posix.getenv("SHIMMY_PORT")) |v| {
            config.port = std.fmt.parseInt(u16, v, 10) catch config.port;
        }
        
        // Workers
        if (std.posix.getenv("SHIMMY_WORKERS")) |v| {
            config.num_workers = std.fmt.parseInt(u32, v, 10) catch config.num_workers;
        } else {
            config.num_workers = @min(@as(u32, @intCast(std.Thread.getCpuCount() catch 4)), 16);
        }
        
        // Buffers
        if (std.posix.getenv("SHIMMY_MAX_REQUEST_MB")) |v| {
            const mb = std.fmt.parseInt(usize, v, 10) catch 16;
            config.max_request_bytes = mb * 1024 * 1024;
        }
        
        // Timeouts
        if (std.posix.getenv("SHIMMY_REQUEST_TIMEOUT_MS")) |v| {
            config.request_timeout_ms = std.fmt.parseInt(u32, v, 10) catch config.request_timeout_ms;
        }
        if (std.posix.getenv("SHIMMY_GENERATION_TIMEOUT_MS")) |v| {
            config.generation_timeout_ms = std.fmt.parseInt(u32, v, 10) catch config.generation_timeout_ms;
        }
        
        // Model
        if (std.posix.getenv("SHIMMY_MODEL_ID")) |v| config.default_model_id = v;
        if (std.posix.getenv("SHIMMY_MODEL_PATH")) |v| config.model_path = v;
        if (std.posix.getenv("SHIMMY_AUTO_LOAD")) |v| {
            config.auto_load_model = std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "1");
        }
        if (std.posix.getenv("SHIMMY_MODELS_JSON")) |json_text| {
            config.models = try parseModelsJson(allocator, json_text);
            if (config.models.len > 0) {
                config.default_model_id = config.models[0].id;
                config.model_path = config.models[0].path;
            }
        }
        
        // Generation defaults
        if (std.posix.getenv("SHIMMY_DEFAULT_MAX_TOKENS")) |v| {
            config.default_max_tokens = std.fmt.parseInt(u32, v, 10) catch config.default_max_tokens;
        }
        if (std.posix.getenv("SHIMMY_DEFAULT_TEMPERATURE")) |v| {
            config.default_temperature = std.fmt.parseFloat(f32, v) catch config.default_temperature;
        }
        
        // Logging
        config.debug_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
        if (std.posix.getenv("SHIMMY_LOG_REQUESTS")) |v| {
            config.log_requests = std.mem.eql(u8, v, "true") or std.mem.eql(u8, v, "1");
        }

        // Rate limiting
        if (std.posix.getenv("SHIMMY_RATE_LIMIT_RPS")) |v| {
            config.rate_limit_requests_per_sec = std.fmt.parseInt(u32, v, 10) catch config.rate_limit_requests_per_sec;
        }
        if (std.posix.getenv("SHIMMY_RATE_LIMIT_BURST")) |v| {
            config.rate_limit_burst = std.fmt.parseInt(u32, v, 10) catch config.rate_limit_burst;
        }

        // Memory requirements
        if (std.posix.getenv("SHIMMY_MIN_MEMORY_MB")) |v| {
            config.min_memory_mb = std.fmt.parseInt(u32, v, 10) catch config.min_memory_mb;
        }

        // Max connections
        if (std.posix.getenv("SHIMMY_MAX_CONNECTIONS")) |v| {
            config.max_connections = std.fmt.parseInt(u32, v, 10) catch config.max_connections;
        }

        return config;
    }
    
    /// Load from JSON config file (if exists)
    pub fn loadFromFile(allocator: std.mem.Allocator, path: []const u8) !?ServerConfig {
        const file = std.fs.cwd().openFile(path, .{}) catch |err| {
            if (err == error.FileNotFound) return null;
            return err;
        };
        defer file.close();
        
        const content = try file.readToEndAlloc(allocator, 1024 * 1024);
        defer allocator.free(content);
        
        return try parseJson(allocator, content);
    }
    
    fn parseJson(allocator: std.mem.Allocator, content: []const u8) !ServerConfig {
        var config = ServerConfig{};
        const parsed = std.json.parseFromSlice(
            struct {
                host: ?[]const u8 = null,
                port: ?u16 = null,
                num_workers: ?u32 = null,
                max_connections: ?u32 = null,
                max_request_mb: ?usize = null,
                request_timeout_ms: ?u32 = null,
                generation_timeout_ms: ?u32 = null,
                model_id: ?[]const u8 = null,
                model_path: ?[]const u8 = null,
                default_max_tokens: ?u32 = null,
                default_temperature: ?f32 = null,
                debug: ?bool = null,
                rate_limit_requests_per_sec: ?u32 = null,
                rate_limit_burst: ?u32 = null,
                models: ?[]ModelJson = null,
            },
            allocator,
            content,
            .{ .ignore_unknown_fields = true },
        ) catch return config;
        defer parsed.deinit();

        const v = parsed.value;
        if (v.host) |h| config.host = h;
        if (v.port) |p| config.port = p;
        if (v.num_workers) |n| config.num_workers = n;
        if (v.max_connections) |m| config.max_connections = m;
        if (v.max_request_mb) |m| config.max_request_bytes = m * 1024 * 1024;
        if (v.request_timeout_ms) |t| config.request_timeout_ms = t;
        if (v.generation_timeout_ms) |t| config.generation_timeout_ms = t;
        if (v.model_id) |m| config.default_model_id = m;
        if (v.model_path) |p| config.model_path = p;
        if (v.default_max_tokens) |t| config.default_max_tokens = t;
        if (v.default_temperature) |t| config.default_temperature = t;
        if (v.debug) |d| config.debug_enabled = d;
        if (v.rate_limit_requests_per_sec) |r| config.rate_limit_requests_per_sec = r;
        if (v.rate_limit_burst) |r| config.rate_limit_burst = r;
        if (v.models) |models_list| {
            config.models = try buildModelConfigs(allocator, models_list);
            if (config.models.len > 0) {
                config.default_model_id = config.models[0].id;
                config.model_path = config.models[0].path;
            }
        }

        return config;
    }
    
    /// Print configuration summary
    pub fn print(self: ServerConfig) void {
        std.debug.print("Configuration:\n", .{});
        std.debug.print("  Host: {s}:{d}\n", .{ self.host, self.port });
        std.debug.print("  Workers: {d}\n", .{self.num_workers});
        std.debug.print("  Max request: {d} MB\n", .{self.max_request_bytes / (1024 * 1024)});
        std.debug.print("  Model: {s}\n", .{self.default_model_id});
        std.debug.print("  Models configured: {d}\n", .{self.models.len});
        std.debug.print("  Rate limit: {d} req/s, burst: {d}\n", .{ self.rate_limit_requests_per_sec, self.rate_limit_burst });
        std.debug.print("  Debug: {}\n", .{self.debug_enabled});
    }

    pub fn deinit(self: *ServerConfig, allocator: std.mem.Allocator) void {
        if (self.models.len > 0) {
            for (self.models) |*cfg| cfg.deinit(allocator);
            allocator.free(self.models);
        }
        self.models = &.{};
    }
};

const ModelJson = struct {
    id: []const u8,
    path: []const u8,
    display_name: ?[]const u8 = null,
    preload: ?bool = null,
    max_workers: ?u32 = null,
    max_tokens: ?u32 = null,
    temperature: ?f32 = null,
};

fn buildModelConfigs(allocator: std.mem.Allocator, entries: []ModelJson) ![]ModelRegistry.ModelConfig {
    if (entries.len == 0) return &.{};

    const list = try allocator.alloc(ModelRegistry.ModelConfig, entries.len);
    errdefer {
        for (list[0..]) |*cfg| cfg.deinit(allocator);
        allocator.free(list);
    }

    for (entries, 0..) |entry, idx| {
        if (entry.id.len == 0 or entry.path.len == 0) {
            return error.InvalidModelEntry;
        }
        list[idx] = try ModelRegistry.ModelConfig.init(allocator, .{
            .id = entry.id,
            .path = entry.path,
            .display_name = entry.display_name,
            .preload = entry.preload orelse false,
            .max_workers = entry.max_workers,
            .max_tokens = entry.max_tokens,
            .temperature = entry.temperature,
            .metadata = .{
                .architecture = "unknown",
                .quantization = "unknown",
                .parameter_count = "unknown",
                .format = "gguf",
                .context_length = 4096,
                .tags = &.{},
                .source = "local",
                .license = "unknown",
                .created_at = std.time.timestamp(),
                .size_bytes = 0,
            },
        });
    }

    return list;
}

fn parseModelsJson(allocator: std.mem.Allocator, json_text: []const u8) ![]ModelRegistry.ModelConfig {
    const parsed = try std.json.parseFromSlice([]ModelJson, allocator, json_text, .{});
    defer parsed.deinit();
    return try buildModelConfigs(allocator, parsed.value);
}

