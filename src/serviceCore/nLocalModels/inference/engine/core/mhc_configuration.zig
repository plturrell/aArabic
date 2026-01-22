// mHC Configuration System
// Implements multi-source configuration with hot-reload support
//
// Configuration precedence (highest to lowest):
// 1. CLI arguments
// 2. Environment variables
// 3. JSON configuration files
// 4. Programmatic defaults
//
// Reference: docs/specs/mhc_configuration.md (Day 31)

const std = @import("std");
const builtin = @import("builtin");

/// Layer range specification for selective mHC application
pub const LayerRange = struct {
    start: u32,
    end: u32,

    /// Check if layer_id is within this range
    pub fn contains(self: LayerRange, layer_id: u32) bool {
        return layer_id >= self.start and layer_id <= self.end;
    }

    /// Validate layer range (start <= end)
    pub fn validate(self: LayerRange) !void {
        if (self.start > self.end) {
            return error.InvalidLayerRange;
        }
    }
};

/// Core mHC constraint settings (from Day 27 spec)
pub const CoreConfig = struct {
    /// Enable mHC constraints globally
    enabled: bool = false,

    /// Sinkhorn-Knopp iterations (5-50)
    sinkhorn_iterations: u32 = 10,

    /// Convergence threshold (1e-8 to 1e-3)
    manifold_epsilon: f32 = 1e-6,

    /// Stability validation threshold
    stability_threshold: f32 = 1e-4,

    /// Maximum activation bound
    manifold_beta: f32 = 10.0,

    /// Enable early stopping
    early_stopping: bool = true,

    /// Log detailed stability metrics
    log_stability_metrics: bool = false,

    /// Apply to specific layer range (null = all layers)
    layer_range: ?LayerRange = null,

    /// Validate configuration parameters
    pub fn validate(self: CoreConfig) !void {
        if (self.sinkhorn_iterations < 5 or self.sinkhorn_iterations > 50) {
            return error.InvalidIterations;
        }
        if (self.manifold_epsilon <= 0 or self.manifold_epsilon >= 1) {
            return error.InvalidEpsilon;
        }
        if (self.stability_threshold <= 0) {
            return error.InvalidThreshold;
        }
        if (self.manifold_beta <= 0) {
            return error.InvalidBeta;
        }
        if (self.layer_range) |range| {
            try range.validate();
        }
    }
};

/// Matrix operation settings (from Day 28 spec)
pub const MatrixOpsConfig = struct {
    /// Enable mHC in matrix operations
    use_mhc: bool = true,

    /// Abort on instability detection
    abort_on_instability: bool = false,

    /// Enable SIMD optimizations
    use_simd: bool = true,

    /// Thread pool size for parallel operations (0 = auto-detect)
    thread_pool_size: u32 = 0,

    /// Enable quantized matmul support
    support_quantization: bool = true,

    /// Batch size for batch operations
    batch_size: u32 = 32,

    /// Validate configuration parameters
    pub fn validate(self: MatrixOpsConfig) !void {
        if (self.batch_size == 0) {
            return error.InvalidBatchSize;
        }
    }
};

/// Transformer integration settings (from Day 29 spec)
pub const TransformerConfig = struct {
    /// Apply mHC to attention output
    mhc_in_attention: bool = true,

    /// Apply mHC to FFN output
    mhc_in_ffn: bool = true,

    /// Apply mHC to residual connections
    mhc_in_residual: bool = false,

    /// Track stability metrics
    track_stability: bool = true,

    /// Layer selection strategy: "all", "adaptive", "manual"
    layer_selection: []const u8 = "adaptive",

    /// Manual layer range (if layer_selection = "manual")
    manual_layer_range: ?LayerRange = null,

    /// Adaptive selection threshold (layers with α > threshold)
    adaptive_threshold: f32 = 1.05,

    /// Validate configuration parameters
    pub fn validate(self: TransformerConfig) !void {
        // Validate layer_selection enum
        const valid_selections = [_][]const u8{ "all", "adaptive", "manual" };
        var valid = false;
        for (valid_selections) |sel| {
            if (std.mem.eql(u8, self.layer_selection, sel)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return error.InvalidLayerSelection;
        }

        // Validate adaptive_threshold range
        if (self.adaptive_threshold < 0.9 or self.adaptive_threshold > 2.0) {
            return error.InvalidAdaptiveThreshold;
        }

        // Validate manual_layer_range if needed
        if (std.mem.eql(u8, self.layer_selection, "manual")) {
            if (self.manual_layer_range == null) {
                return error.MissingManualLayerRange;
            }
            try self.manual_layer_range.?.validate();
        }
    }
};

/// GGUF loader settings (from Day 30 spec)
pub const GGUFConfig = struct {
    /// Auto-detect mHC from GGUF metadata
    auto_detect: bool = true,

    /// Require mHC metadata in GGUF files
    require_metadata: bool = false,

    /// Fallback to defaults if metadata missing
    use_fallback: bool = true,

    /// Validation level: "strict", "loose", "none"
    validation_level: []const u8 = "loose",

    /// Validate configuration parameters
    pub fn validate(self: GGUFConfig) !void {
        const valid_levels = [_][]const u8{ "strict", "loose", "none" };
        var valid = false;
        for (valid_levels) |level| {
            if (std.mem.eql(u8, self.validation_level, level)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return error.InvalidValidationLevel;
        }
    }
};

/// Hyperbolic manifold settings (Days 54-60, optional)
pub const HyperbolicConfig = struct {
    /// Curvature parameter (negative for hyperbolic)
    curvature: f32 = -1.0,

    /// Use Poincaré ball model (vs hyperboloid)
    use_poincare: bool = true,

    /// Numerical stability epsilon
    epsilon: f32 = 1e-8,
};

/// Spherical manifold settings (Days 54-60, optional)
pub const SphericalConfig = struct {
    /// Sphere radius
    radius: f32 = 1.0,

    /// Use stereographic projection
    use_stereographic: bool = false,
};

/// Product manifold settings (Days 54-60, optional)
pub const ProductManifoldConfig = struct {
    /// Component manifolds
    components: [][]const u8,

    /// Component weights for distance computation
    weights: []f32,
};

/// Geometric extensions settings (Days 54-60, optional)
pub const GeometricConfig = struct {
    /// Enable geometric extensions
    enabled: bool = false,

    /// Manifold type: "euclidean", "hyperbolic", "spherical", "product"
    manifold_type: []const u8 = "euclidean",

    /// Hyperbolic settings
    hyperbolic: ?HyperbolicConfig = null,

    /// Spherical settings
    spherical: ?SphericalConfig = null,

    /// Product manifold settings
    product: ?ProductManifoldConfig = null,

    /// Auto-detect geometry from data
    auto_detect_geometry: bool = false,

    /// Curvature estimation method: "ollivier_ricci", "sectional"
    curvature_method: []const u8 = "ollivier_ricci",
};

/// Alert threshold configuration (Days 61-67, optional)
pub const AlertThresholds = struct {
    /// Maximum allowed instability rate (0.0-1.0)
    max_instability_rate: f32 = 0.05,

    /// Maximum amplification factor
    max_amplification: f32 = 1.5,

    /// Minimum amplification factor
    min_amplification: f32 = 0.5,

    /// Maximum energy spike (relative to baseline)
    max_energy_spike: f32 = 2.0,
};

/// Monitoring and observability settings (Days 61-67, optional)
pub const MonitoringConfig = struct {
    /// Enable uncertainty quantification
    uncertainty_quantification: bool = false,

    /// Bootstrap samples for uncertainty estimation
    bootstrap_samples: u32 = 100,

    /// Enable failure detection
    failure_detection: bool = true,

    /// Alert thresholds
    alert_thresholds: AlertThresholds = .{},

    /// Metrics collection interval (milliseconds)
    metrics_interval_ms: u32 = 1000,

    /// Enable Prometheus metrics export
    prometheus_enabled: bool = true,

    /// Prometheus port
    prometheus_port: u16 = 9090,
};

/// Runtime behavior settings
pub const RuntimeConfig = struct {
    /// Enable hot-reload of configuration
    hot_reload: bool = true,

    /// Configuration file watch interval (seconds)
    watch_interval_sec: u32 = 5,

    /// Log configuration changes
    log_config_changes: bool = true,

    /// Validation mode: "strict", "warn", "silent"
    validation_mode: []const u8 = "warn",

    /// Configuration file path
    config_file_path: []const u8 = "config/mhc_config.json",

    /// Enable configuration audit log
    audit_log_enabled: bool = true,

    /// Audit log file path
    audit_log_path: []const u8 = "logs/mhc_config_audit.log",

    /// Validate configuration parameters
    pub fn validate(self: RuntimeConfig) !void {
        const valid_modes = [_][]const u8{ "strict", "warn", "silent" };
        var valid = false;
        for (valid_modes) |mode| {
            if (std.mem.eql(u8, self.validation_mode, mode)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return error.InvalidValidationMode;
        }

        if (self.watch_interval_sec == 0) {
            return error.InvalidWatchInterval;
        }
    }
};

/// Root configuration structure
pub const MHCConfiguration = struct {
    /// Schema version (semantic versioning)
    schema_version: []const u8 = "1.0.0",

    /// Core mHC settings
    core: CoreConfig = .{},

    /// Matrix operation settings
    matrix_ops: MatrixOpsConfig = .{},

    /// Transformer integration settings
    transformer: TransformerConfig = .{},

    /// GGUF loader settings
    gguf: GGUFConfig = .{},

    /// Geometric extensions (optional, Days 54-60)
    geometric: ?GeometricConfig = null,

    /// Monitoring and observability (optional, Days 61-67)
    monitoring: ?MonitoringConfig = null,

    /// Runtime behavior
    runtime: RuntimeConfig = .{},

    /// Validate entire configuration
    pub fn validate(self: MHCConfiguration) !void {
        try self.core.validate();
        try self.matrix_ops.validate();
        try self.transformer.validate();
        try self.gguf.validate();
        try self.runtime.validate();
    }
};

/// Get default configuration
pub fn default_config() MHCConfiguration {
    return MHCConfiguration{};
}

// ============================================================================
// MHC Transformer Config (used by transformer.zig and gguf_loader.zig)
// ============================================================================

/// Stability callback function type
pub const StabilityCallbackFn = *const fn(layer_id: u32, amplification: f32, is_stable: bool) void;

/// MHC configuration for transformer integration
/// This is used by both transformer.zig and gguf_loader.zig
pub const MHCTransformerConfig = struct {
    // Global enable/disable
    enabled: bool = false,

    // Per-component enable/disable
    attention_enabled: bool = true,
    ffn_enabled: bool = true,
    residual_enabled: bool = false,  // Only for deep instability

    // Layer range (null = all layers)
    layer_range: ?LayerRange = null,

    // Core mHC parameters (simplified version without mhc_constraints dependency)
    core: CoreMHCParams = .{},

    // Stability thresholds
    attention_stability_threshold: f32 = 1e-4,
    ffn_stability_threshold: f32 = 1e-4,
    residual_stability_threshold: f32 = 1e-4,

    // Abort on instability (fail-fast mode)
    abort_on_instability: bool = false,

    // Callback for custom stability handling (simplified)
    stability_callback: ?StabilityCallbackFn = null,
};

/// Core mHC parameters (standalone, no external dependencies)
pub const CoreMHCParams = struct {
    enabled: bool = true,
    sinkhorn_iterations: u32 = 10,
    manifold_epsilon: f32 = 1e-6,
    stability_threshold: f32 = 1e-4,
    manifold_beta: f32 = 10.0,
    log_stability_metrics: bool = false,
    early_stopping: bool = true,
};

// ============================================================================
// Unit Tests
// ============================================================================

test "LayerRange.contains" {
    const range = LayerRange{ .start = 10, .end = 20 };

    try std.testing.expect(range.contains(10));
    try std.testing.expect(range.contains(15));
    try std.testing.expect(range.contains(20));
    try std.testing.expect(!range.contains(9));
    try std.testing.expect(!range.contains(21));
}

test "LayerRange.validate valid range" {
    const range = LayerRange{ .start = 10, .end = 20 };
    try range.validate();
}

test "LayerRange.validate invalid range" {
    const range = LayerRange{ .start = 20, .end = 10 };
    try std.testing.expectError(error.InvalidLayerRange, range.validate());
}

test "CoreConfig.validate valid config" {
    const config = CoreConfig{
        .enabled = true,
        .sinkhorn_iterations = 10,
        .manifold_epsilon = 1e-6,
    };
    try config.validate();
}

test "CoreConfig.validate invalid iterations" {
    const config = CoreConfig{
        .sinkhorn_iterations = 100, // Too high
    };
    try std.testing.expectError(error.InvalidIterations, config.validate());
}

test "CoreConfig.validate invalid epsilon" {
    const config = CoreConfig{
        .manifold_epsilon = 2.0, // Too high
    };
    try std.testing.expectError(error.InvalidEpsilon, config.validate());
}

test "TransformerConfig.validate valid adaptive" {
    const config = TransformerConfig{
        .layer_selection = "adaptive",
        .adaptive_threshold = 1.05,
    };
    try config.validate();
}

test "TransformerConfig.validate invalid selection" {
    const config = TransformerConfig{
        .layer_selection = "invalid",
    };
    try std.testing.expectError(error.InvalidLayerSelection, config.validate());
}

test "TransformerConfig.validate manual without range" {
    const config = TransformerConfig{
        .layer_selection = "manual",
        .manual_layer_range = null, // Missing
    };
    try std.testing.expectError(error.MissingManualLayerRange, config.validate());
}

test "MHCConfiguration default values" {
    const config = default_config();

    try std.testing.expect(!config.core.enabled);
    try std.testing.expectEqual(@as(u32, 10), config.core.sinkhorn_iterations);
    try std.testing.expectEqual(@as(f32, 1e-6), config.core.manifold_epsilon);
    try std.testing.expect(config.matrix_ops.use_mhc);
    try std.testing.expect(config.transformer.mhc_in_attention);
}
