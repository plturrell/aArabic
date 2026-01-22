// mHC Configuration Loader - Day 47 Implementation
// Provides JSON loading, environment variable support, validation, and runtime updates
//
// Configuration hierarchy (lowest to highest priority):
// 1. Programmatic defaults
// 2. JSON configuration files
// 3. Environment variables (MHC_* prefix)
// 4. Runtime API updates
//
// Reference: docs/specs/mhc_configuration.md (Day 31)

const std = @import("std");
const config = @import("mhc_configuration.zig");
const MHCConfiguration = config.MHCConfiguration;
const CoreConfig = config.CoreConfig;
const MatrixOpsConfig = config.MatrixOpsConfig;
const TransformerConfig = config.TransformerConfig;
const GGUFConfig = config.GGUFConfig;
const RuntimeConfig = config.RuntimeConfig;
const LayerRange = config.LayerRange;

/// Configuration loader errors
pub const ConfigError = error{
    FileNotFound,
    InvalidJson,
    InvalidValue,
    ValidationFailed,
    IoError,
    OutOfMemory,
    InvalidEnvVar,
    ParseError,
};

/// Configuration source tracking
pub const ConfigSource = enum {
    default,
    json_file,
    env_var,
    runtime_api,
};

/// Configuration update callback type
pub const ConfigUpdateCallback = *const fn (old_config: MHCConfiguration, new_config: MHCConfiguration) void;

/// Configuration loader with runtime update support
pub const MHCConfigLoader = struct {
    allocator: std.mem.Allocator,
    current_config: MHCConfiguration,
    source: ConfigSource,
    json_path: ?[]const u8,
    callbacks: std.ArrayListUnmanaged(ConfigUpdateCallback),
    version: u64,

    // String storage for allocated strings
    owned_strings: std.ArrayListUnmanaged([]const u8),

    const Self = @This();

    /// Initialize config loader with defaults
    pub fn init(allocator: std.mem.Allocator) Self {
        return Self{
            .allocator = allocator,
            .current_config = config.default_config(),
            .source = .default,
            .json_path = null,
            .callbacks = .{},
            .version = 0,
            .owned_strings = .{},
        };
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *Self) void {
        for (self.owned_strings.items) |str| {
            self.allocator.free(str);
        }
        self.owned_strings.deinit(self.allocator);
        self.callbacks.deinit(self.allocator);
        if (self.json_path) |path| {
            self.allocator.free(path);
        }
    }

    /// Load configuration from JSON file
    pub fn loadFromJson(self: *Self, path: []const u8) ConfigError!void {
        // Read file contents
        const file = std.fs.cwd().openFile(path, .{}) catch {
            return ConfigError.FileNotFound;
        };
        defer file.close();

        const content = file.readToEndAlloc(self.allocator, 1024 * 1024) catch {
            return ConfigError.IoError;
        };
        defer self.allocator.free(content);

        // Parse JSON
        const parsed = std.json.parseFromSlice(std.json.Value, self.allocator, content, .{}) catch {
            return ConfigError.InvalidJson;
        };
        defer parsed.deinit();

        // Build config from JSON
        var new_config = config.default_config();
        try self.parseJsonConfig(&new_config, parsed.value);

        // Store path
        if (self.json_path) |old_path| {
            self.allocator.free(old_path);
        }
        self.json_path = self.allocator.dupe(u8, path) catch {
            return ConfigError.OutOfMemory;
        };

        // Update current config
        const old_config = self.current_config;
        self.current_config = new_config;
        self.source = .json_file;
        self.version += 1;

        // Notify callbacks
        self.notifyCallbacks(old_config, new_config);
    }

    /// Parse JSON value into configuration struct
    fn parseJsonConfig(self: *Self, cfg: *MHCConfiguration, json_val: std.json.Value) ConfigError!void {
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        // Parse schema_version
        if (obj.get("schema_version")) |v| {
            if (v == .string) {
                cfg.schema_version = try self.allocString(v.string);
            }
        }

        // Parse core config
        if (obj.get("core")) |core_val| {
            try self.parseCoreConfig(&cfg.core, core_val);
        }

        // Parse matrix_ops config
        if (obj.get("matrix_ops")) |matrix_val| {
            try self.parseMatrixOpsConfig(&cfg.matrix_ops, matrix_val);
        }

        // Parse transformer config
        if (obj.get("transformer")) |trans_val| {
            try self.parseTransformerConfig(&cfg.transformer, trans_val);
        }

        // Parse gguf config
        if (obj.get("gguf")) |gguf_val| {
            try self.parseGGUFConfig(&cfg.gguf, gguf_val);
        }

        // Parse runtime config
        if (obj.get("runtime")) |runtime_val| {
            try self.parseRuntimeConfig(&cfg.runtime, runtime_val);
        }
    }

    /// Parse core configuration section
    fn parseCoreConfig(self: *Self, core: *CoreConfig, json_val: std.json.Value) ConfigError!void {
        _ = self;
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        if (obj.get("enabled")) |v| {
            if (v == .bool) core.enabled = v.bool;
        }
        if (obj.get("sinkhorn_iterations")) |v| {
            if (v == .integer) core.sinkhorn_iterations = @intCast(@as(i64, v.integer));
        }
        if (obj.get("manifold_epsilon")) |v| {
            if (v == .float) core.manifold_epsilon = @floatCast(v.float);
        }
        if (obj.get("stability_threshold")) |v| {
            if (v == .float) core.stability_threshold = @floatCast(v.float);
        }
        if (obj.get("manifold_beta")) |v| {
            if (v == .float) core.manifold_beta = @floatCast(v.float);
        }
        if (obj.get("early_stopping")) |v| {
            if (v == .bool) core.early_stopping = v.bool;
        }
        if (obj.get("log_stability_metrics")) |v| {
            if (v == .bool) core.log_stability_metrics = v.bool;
        }
        if (obj.get("layer_range")) |v| {
            core.layer_range = try parseLayerRange(v);
        }
    }

    /// Parse matrix ops configuration section
    fn parseMatrixOpsConfig(self: *Self, matrix: *MatrixOpsConfig, json_val: std.json.Value) ConfigError!void {
        _ = self;
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        if (obj.get("use_mhc")) |v| {
            if (v == .bool) matrix.use_mhc = v.bool;
        }
        if (obj.get("abort_on_instability")) |v| {
            if (v == .bool) matrix.abort_on_instability = v.bool;
        }
        if (obj.get("use_simd")) |v| {
            if (v == .bool) matrix.use_simd = v.bool;
        }
        if (obj.get("thread_pool_size")) |v| {
            if (v == .integer) matrix.thread_pool_size = @intCast(@as(i64, v.integer));
        }
        if (obj.get("support_quantization")) |v| {
            if (v == .bool) matrix.support_quantization = v.bool;
        }
        if (obj.get("batch_size")) |v| {
            if (v == .integer) matrix.batch_size = @intCast(@as(i64, v.integer));
        }
    }

    /// Parse transformer configuration section
    fn parseTransformerConfig(self: *Self, trans: *TransformerConfig, json_val: std.json.Value) ConfigError!void {
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        if (obj.get("mhc_in_attention")) |v| {
            if (v == .bool) trans.mhc_in_attention = v.bool;
        }
        if (obj.get("mhc_in_ffn")) |v| {
            if (v == .bool) trans.mhc_in_ffn = v.bool;
        }
        if (obj.get("mhc_in_residual")) |v| {
            if (v == .bool) trans.mhc_in_residual = v.bool;
        }
        if (obj.get("track_stability")) |v| {
            if (v == .bool) trans.track_stability = v.bool;
        }
        if (obj.get("layer_selection")) |v| {
            if (v == .string) trans.layer_selection = try self.allocString(v.string);
        }
        if (obj.get("adaptive_threshold")) |v| {
            if (v == .float) trans.adaptive_threshold = @floatCast(v.float);
        }
        if (obj.get("manual_layer_range")) |v| {
            trans.manual_layer_range = try parseLayerRange(v);
        }
    }

    /// Parse GGUF configuration section
    fn parseGGUFConfig(self: *Self, gguf: *GGUFConfig, json_val: std.json.Value) ConfigError!void {
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        if (obj.get("auto_detect")) |v| {
            if (v == .bool) gguf.auto_detect = v.bool;
        }
        if (obj.get("require_metadata")) |v| {
            if (v == .bool) gguf.require_metadata = v.bool;
        }
        if (obj.get("use_fallback")) |v| {
            if (v == .bool) gguf.use_fallback = v.bool;
        }
        if (obj.get("validation_level")) |v| {
            if (v == .string) gguf.validation_level = try self.allocString(v.string);
        }
    }

    /// Parse runtime configuration section
    fn parseRuntimeConfig(self: *Self, runtime: *RuntimeConfig, json_val: std.json.Value) ConfigError!void {
        const obj = switch (json_val) {
            .object => |o| o,
            else => return ConfigError.InvalidJson,
        };

        if (obj.get("hot_reload")) |v| {
            if (v == .bool) runtime.hot_reload = v.bool;
        }
        if (obj.get("watch_interval_sec")) |v| {
            if (v == .integer) runtime.watch_interval_sec = @intCast(@as(i64, v.integer));
        }
        if (obj.get("log_config_changes")) |v| {
            if (v == .bool) runtime.log_config_changes = v.bool;
        }
        if (obj.get("validation_mode")) |v| {
            if (v == .string) runtime.validation_mode = try self.allocString(v.string);
        }
        if (obj.get("config_file_path")) |v| {
            if (v == .string) runtime.config_file_path = try self.allocString(v.string);
        }
        if (obj.get("audit_log_enabled")) |v| {
            if (v == .bool) runtime.audit_log_enabled = v.bool;
        }
        if (obj.get("audit_log_path")) |v| {
            if (v == .string) runtime.audit_log_path = try self.allocString(v.string);
        }
    }

    /// Parse layer range from JSON
    fn parseLayerRange(json_val: std.json.Value) ConfigError!?LayerRange {
        const obj = switch (json_val) {
            .object => |o| o,
            .null => return null,
            else => return ConfigError.InvalidJson,
        };

        const start = obj.get("start") orelse return ConfigError.InvalidJson;
        const end = obj.get("end") orelse return ConfigError.InvalidJson;

        if (start != .integer or end != .integer) {
            return ConfigError.InvalidJson;
        }

        return LayerRange{
            .start = @intCast(@as(i64, start.integer)),
            .end = @intCast(@as(i64, end.integer)),
        };
    }

    /// Allocate and store a string copy
    fn allocString(self: *Self, str: []const u8) ConfigError![]const u8 {
        const copy = self.allocator.dupe(u8, str) catch {
            return ConfigError.OutOfMemory;
        };
        self.owned_strings.append(copy) catch {
            self.allocator.free(copy);
            return ConfigError.OutOfMemory;
        };
        return copy;
    }

    /// Load configuration from environment variables (MHC_* prefix)
    pub fn loadFromEnv(self: *Self) ConfigError!void {
        var new_config = self.current_config;

        // Core configuration
        if (getEnvBool("MHC_ENABLED")) |v| new_config.core.enabled = v;
        if (getEnvU32("MHC_SINKHORN_ITERATIONS")) |v| new_config.core.sinkhorn_iterations = v;
        if (getEnvF32("MHC_MANIFOLD_EPSILON")) |v| new_config.core.manifold_epsilon = v;
        if (getEnvF32("MHC_STABILITY_THRESHOLD")) |v| new_config.core.stability_threshold = v;
        if (getEnvF32("MHC_MANIFOLD_BETA")) |v| new_config.core.manifold_beta = v;
        if (getEnvBool("MHC_EARLY_STOPPING")) |v| new_config.core.early_stopping = v;
        if (getEnvBool("MHC_LOG_STABILITY_METRICS")) |v| new_config.core.log_stability_metrics = v;

        // Matrix ops configuration
        if (getEnvBool("MHC_USE_MHC")) |v| new_config.matrix_ops.use_mhc = v;
        if (getEnvBool("MHC_ABORT_ON_INSTABILITY")) |v| new_config.matrix_ops.abort_on_instability = v;
        if (getEnvBool("MHC_USE_SIMD")) |v| new_config.matrix_ops.use_simd = v;
        if (getEnvU32("MHC_THREAD_POOL_SIZE")) |v| new_config.matrix_ops.thread_pool_size = v;
        if (getEnvU32("MHC_BATCH_SIZE")) |v| new_config.matrix_ops.batch_size = v;

        // Transformer configuration
        if (getEnvBool("MHC_IN_ATTENTION")) |v| new_config.transformer.mhc_in_attention = v;
        if (getEnvBool("MHC_IN_FFN")) |v| new_config.transformer.mhc_in_ffn = v;
        if (getEnvBool("MHC_IN_RESIDUAL")) |v| new_config.transformer.mhc_in_residual = v;
        if (getEnvBool("MHC_TRACK_STABILITY")) |v| new_config.transformer.track_stability = v;
        if (getEnvF32("MHC_ADAPTIVE_THRESHOLD")) |v| new_config.transformer.adaptive_threshold = v;
        if (getEnvStr("MHC_LAYER_SELECTION")) |v| {
            new_config.transformer.layer_selection = try self.allocString(v);
        }

        // GGUF configuration
        if (getEnvBool("MHC_GGUF_AUTO_DETECT")) |v| new_config.gguf.auto_detect = v;
        if (getEnvBool("MHC_GGUF_REQUIRE_METADATA")) |v| new_config.gguf.require_metadata = v;
        if (getEnvBool("MHC_GGUF_USE_FALLBACK")) |v| new_config.gguf.use_fallback = v;
        if (getEnvStr("MHC_GGUF_VALIDATION_LEVEL")) |v| {
            new_config.gguf.validation_level = try self.allocString(v);
        }

        // Runtime configuration
        if (getEnvBool("MHC_HOT_RELOAD")) |v| new_config.runtime.hot_reload = v;
        if (getEnvU32("MHC_WATCH_INTERVAL_SEC")) |v| new_config.runtime.watch_interval_sec = v;
        if (getEnvBool("MHC_LOG_CONFIG_CHANGES")) |v| new_config.runtime.log_config_changes = v;
        if (getEnvStr("MHC_VALIDATION_MODE")) |v| {
            new_config.runtime.validation_mode = try self.allocString(v);
        }
        if (getEnvStr("MHC_CONFIG_FILE_PATH")) |v| {
            new_config.runtime.config_file_path = try self.allocString(v);
        }

        // Update config
        const old_config = self.current_config;
        self.current_config = new_config;
        self.source = .env_var;
        self.version += 1;

        self.notifyCallbacks(old_config, new_config);
    }

    /// Merge two configurations (override takes precedence)
    pub fn merge(base: MHCConfiguration, override: MHCConfiguration) MHCConfiguration {
        var result = base;

        // Merge core config
        result.core.enabled = override.core.enabled;
        result.core.sinkhorn_iterations = override.core.sinkhorn_iterations;
        result.core.manifold_epsilon = override.core.manifold_epsilon;
        result.core.stability_threshold = override.core.stability_threshold;
        result.core.manifold_beta = override.core.manifold_beta;
        result.core.early_stopping = override.core.early_stopping;
        result.core.log_stability_metrics = override.core.log_stability_metrics;
        if (override.core.layer_range) |range| {
            result.core.layer_range = range;
        }

        // Merge matrix_ops config
        result.matrix_ops.use_mhc = override.matrix_ops.use_mhc;
        result.matrix_ops.abort_on_instability = override.matrix_ops.abort_on_instability;
        result.matrix_ops.use_simd = override.matrix_ops.use_simd;
        result.matrix_ops.thread_pool_size = override.matrix_ops.thread_pool_size;
        result.matrix_ops.support_quantization = override.matrix_ops.support_quantization;
        result.matrix_ops.batch_size = override.matrix_ops.batch_size;

        // Merge transformer config
        result.transformer.mhc_in_attention = override.transformer.mhc_in_attention;
        result.transformer.mhc_in_ffn = override.transformer.mhc_in_ffn;
        result.transformer.mhc_in_residual = override.transformer.mhc_in_residual;
        result.transformer.track_stability = override.transformer.track_stability;
        result.transformer.layer_selection = override.transformer.layer_selection;
        result.transformer.adaptive_threshold = override.transformer.adaptive_threshold;
        if (override.transformer.manual_layer_range) |range| {
            result.transformer.manual_layer_range = range;
        }

        // Merge gguf config
        result.gguf.auto_detect = override.gguf.auto_detect;
        result.gguf.require_metadata = override.gguf.require_metadata;
        result.gguf.use_fallback = override.gguf.use_fallback;
        result.gguf.validation_level = override.gguf.validation_level;

        // Merge runtime config
        result.runtime.hot_reload = override.runtime.hot_reload;
        result.runtime.watch_interval_sec = override.runtime.watch_interval_sec;
        result.runtime.log_config_changes = override.runtime.log_config_changes;
        result.runtime.validation_mode = override.runtime.validation_mode;
        result.runtime.config_file_path = override.runtime.config_file_path;
        result.runtime.audit_log_enabled = override.runtime.audit_log_enabled;
        result.runtime.audit_log_path = override.runtime.audit_log_path;

        // Merge optional sections
        if (override.geometric) |geo| {
            result.geometric = geo;
        }
        if (override.monitoring) |mon| {
            result.monitoring = mon;
        }

        return result;
    }

    /// Validate configuration
    pub fn validate(cfg: MHCConfiguration) ConfigError!void {
        cfg.validate() catch {
            return ConfigError.ValidationFailed;
        };
    }

    /// Get current configuration
    pub fn getConfig(self: *const Self) MHCConfiguration {
        return self.current_config;
    }

    /// Update configuration at runtime
    pub fn updateRuntime(self: *Self, new_config: MHCConfiguration) ConfigError!void {
        // Validate new config
        try validate(new_config);

        // Update
        const old_config = self.current_config;
        self.current_config = new_config;
        self.source = .runtime_api;
        self.version += 1;

        self.notifyCallbacks(old_config, new_config);
    }

    /// Update a single core parameter at runtime
    pub fn setCoreEnabled(self: *Self, enabled: bool) void {
        const old_config = self.current_config;
        self.current_config.core.enabled = enabled;
        self.source = .runtime_api;
        self.version += 1;
        self.notifyCallbacks(old_config, self.current_config);
    }

    /// Update sinkhorn iterations at runtime
    pub fn setSinkhornIterations(self: *Self, iterations: u32) ConfigError!void {
        if (iterations < 5 or iterations > 50) {
            return ConfigError.InvalidValue;
        }
        const old_config = self.current_config;
        self.current_config.core.sinkhorn_iterations = iterations;
        self.source = .runtime_api;
        self.version += 1;
        self.notifyCallbacks(old_config, self.current_config);
    }

    /// Register update callback
    pub fn onUpdate(self: *Self, callback: ConfigUpdateCallback) ConfigError!void {
        self.callbacks.append(callback) catch {
            return ConfigError.OutOfMemory;
        };
    }

    /// Notify all registered callbacks
    fn notifyCallbacks(self: *Self, old_config: MHCConfiguration, new_config: MHCConfiguration) void {
        for (self.callbacks.items) |callback| {
            callback(old_config, new_config);
        }
    }

    /// Get configuration version
    pub fn getVersion(self: *const Self) u64 {
        return self.version;
    }

    /// Get configuration source
    pub fn getSource(self: *const Self) ConfigSource {
        return self.source;
    }

    /// Reload from JSON file (if one was loaded)
    pub fn reload(self: *Self) ConfigError!void {
        if (self.json_path) |path| {
            try self.loadFromJson(path);
        }
    }
};

// ============================================================================
// Environment Variable Helpers
// ============================================================================

fn getEnvStr(name: []const u8) ?[]const u8 {
    return std.posix.getenv(name);
}

fn getEnvBool(name: []const u8) ?bool {
    if (getEnvStr(name)) |value| {
        if (std.mem.eql(u8, value, "true") or std.mem.eql(u8, value, "1")) {
            return true;
        }
        if (std.mem.eql(u8, value, "false") or std.mem.eql(u8, value, "0")) {
            return false;
        }
    }
    return null;
}

fn getEnvU32(name: []const u8) ?u32 {
    if (getEnvStr(name)) |value| {
        return std.fmt.parseInt(u32, value, 10) catch null;
    }
    return null;
}

fn getEnvF32(name: []const u8) ?f32 {
    if (getEnvStr(name)) |value| {
        return std.fmt.parseFloat(f32, value) catch null;
    }
    return null;
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Load configuration with full hierarchy: defaults -> JSON -> ENV
pub fn loadConfig(allocator: std.mem.Allocator, json_path: ?[]const u8) ConfigError!MHCConfigLoader {
    var loader = MHCConfigLoader.init(allocator);
    errdefer loader.deinit();

    // Load from JSON if path provided
    if (json_path) |path| {
        loader.loadFromJson(path) catch |err| {
            // JSON file is optional, continue with defaults if not found
            if (err != ConfigError.FileNotFound) {
                return err;
            }
        };
    }

    // Apply environment variable overrides
    try loader.loadFromEnv();

    // Validate final config
    try MHCConfigLoader.validate(loader.current_config);

    return loader;
}

// ============================================================================
// Unit Tests
// ============================================================================

test "MHCConfigLoader.init creates default config" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    try std.testing.expect(!loader.current_config.core.enabled);
    try std.testing.expectEqual(@as(u32, 10), loader.current_config.core.sinkhorn_iterations);
    try std.testing.expectEqual(ConfigSource.default, loader.source);
    try std.testing.expectEqual(@as(u64, 0), loader.version);
}

test "MHCConfigLoader.merge combines configs" {
    const base = config.default_config();
    var override = config.default_config();
    override.core.enabled = true;
    override.core.sinkhorn_iterations = 25;
    override.matrix_ops.thread_pool_size = 8;

    const merged = MHCConfigLoader.merge(base, override);

    try std.testing.expect(merged.core.enabled);
    try std.testing.expectEqual(@as(u32, 25), merged.core.sinkhorn_iterations);
    try std.testing.expectEqual(@as(u32, 8), merged.matrix_ops.thread_pool_size);
}

test "MHCConfigLoader.validate accepts valid config" {
    const cfg = config.default_config();
    try MHCConfigLoader.validate(cfg);
}

test "MHCConfigLoader.validate rejects invalid sinkhorn iterations" {
    var cfg = config.default_config();
    cfg.core.sinkhorn_iterations = 100; // Invalid: > 50

    try std.testing.expectError(ConfigError.ValidationFailed, MHCConfigLoader.validate(cfg));
}

test "MHCConfigLoader.validate rejects invalid epsilon" {
    var cfg = config.default_config();
    cfg.core.manifold_epsilon = 2.0; // Invalid: >= 1

    try std.testing.expectError(ConfigError.ValidationFailed, MHCConfigLoader.validate(cfg));
}

test "MHCConfigLoader runtime update with valid config" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    var new_config = config.default_config();
    new_config.core.enabled = true;
    new_config.core.sinkhorn_iterations = 20;

    try loader.updateRuntime(new_config);

    try std.testing.expect(loader.current_config.core.enabled);
    try std.testing.expectEqual(@as(u32, 20), loader.current_config.core.sinkhorn_iterations);
    try std.testing.expectEqual(ConfigSource.runtime_api, loader.source);
    try std.testing.expectEqual(@as(u64, 1), loader.version);
}

test "MHCConfigLoader runtime update rejects invalid config" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    var new_config = config.default_config();
    new_config.core.sinkhorn_iterations = 100; // Invalid

    try std.testing.expectError(ConfigError.ValidationFailed, loader.updateRuntime(new_config));
}

test "MHCConfigLoader.setCoreEnabled updates enabled flag" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    try std.testing.expect(!loader.current_config.core.enabled);

    loader.setCoreEnabled(true);

    try std.testing.expect(loader.current_config.core.enabled);
    try std.testing.expectEqual(@as(u64, 1), loader.version);
}

test "MHCConfigLoader.setSinkhornIterations validates range" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    // Valid range
    try loader.setSinkhornIterations(25);
    try std.testing.expectEqual(@as(u32, 25), loader.current_config.core.sinkhorn_iterations);

    // Invalid: too low
    try std.testing.expectError(ConfigError.InvalidValue, loader.setSinkhornIterations(3));

    // Invalid: too high
    try std.testing.expectError(ConfigError.InvalidValue, loader.setSinkhornIterations(60));
}

test "MHCConfigLoader callback notification" {
    const allocator = std.testing.allocator;
    var loader = MHCConfigLoader.init(allocator);
    defer loader.deinit();

    const TestState = struct {
        var callback_called: bool = false;
        var old_enabled: bool = false;
        var new_enabled: bool = false;
    };

    const callback = struct {
        fn cb(old_cfg: MHCConfiguration, new_cfg: MHCConfiguration) void {
            TestState.callback_called = true;
            TestState.old_enabled = old_cfg.core.enabled;
            TestState.new_enabled = new_cfg.core.enabled;
        }
    }.cb;

    try loader.onUpdate(callback);

    loader.setCoreEnabled(true);

    try std.testing.expect(TestState.callback_called);
    try std.testing.expect(!TestState.old_enabled);
    try std.testing.expect(TestState.new_enabled);
}

test "getEnvBool parses boolean values" {
    // Note: Cannot easily test env vars in unit tests without setting them
    // These tests verify the function handles null correctly
    try std.testing.expect(getEnvBool("NONEXISTENT_VAR_12345") == null);
}

test "getEnvU32 parses integer values" {
    try std.testing.expect(getEnvU32("NONEXISTENT_VAR_12345") == null);
}

test "getEnvF32 parses float values" {
    try std.testing.expect(getEnvF32("NONEXISTENT_VAR_12345") == null);
}

test "parseLayerRange parses valid range" {
    const allocator = std.testing.allocator;
    const json_str =
        \\{"start": 5, "end": 10}
    ;

    const parsed = std.json.parseFromSlice(std.json.Value, allocator, json_str, .{}) catch unreachable;
    defer parsed.deinit();

    const range = try MHCConfigLoader.parseLayerRange(parsed.value);

    try std.testing.expect(range != null);
    try std.testing.expectEqual(@as(u32, 5), range.?.start);
    try std.testing.expectEqual(@as(u32, 10), range.?.end);
}

test "parseLayerRange returns null for null value" {
    const range = try MHCConfigLoader.parseLayerRange(.null);
    try std.testing.expect(range == null);
}

