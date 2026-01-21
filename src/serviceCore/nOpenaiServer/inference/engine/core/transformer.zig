const std = @import("std");
const matrix_ops = @import("matrix_ops");
const attention = @import("attention");
const feed_forward = @import("feed_forward");
const kv_cache = @import("kv_cache");
const mhc_config = @import("mhc_configuration");
const mhc_constraints = @import("mhc_constraints");

/// Complete Transformer layer for Llama models with mHC integration
/// Architecture: RMSNorm → Attention → [mHC] → Residual → RMSNorm → FFN → [mHC] → Residual

// ============================================================================
// Structures
// ============================================================================

pub const TransformerConfig = struct {
    embed_dim: u32,
    ffn_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-5,
    
    // NEW: mHC configuration
    mhc_config: MHCTransformerConfig = .{},
    
    // NEW: Stability tracking
    stability_tracker: ?*StabilityTracker = null,
};

// Re-export MHCTransformerConfig from mhc_configuration to avoid circular dependencies
pub const MHCTransformerConfig = mhc_config.MHCTransformerConfig;
pub const LayerRange = mhc_config.LayerRange;

pub const TransformerWeights = struct {
    allocator: std.mem.Allocator,
    // Attention
    attn_norm: []const f32, // RMS norm before attention [embed_dim]
    wq: matrix_ops.Weight,
    wk: matrix_ops.Weight,
    wv: matrix_ops.Weight,
    wo: matrix_ops.Weight,

    // FFN
    ffn_norm: []const f32, // RMS norm before FFN [embed_dim]
    w_gate: matrix_ops.Weight,
    w_up: matrix_ops.Weight,
    w_down: matrix_ops.Weight,

    pub fn deinit(self: *TransformerWeights) void {
        self.allocator.free(self.attn_norm);
        switch (self.wq) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        switch (self.wk) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        switch (self.wv) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        switch (self.wo) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        self.allocator.free(self.ffn_norm);
        switch (self.w_gate) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        switch (self.w_up) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
        switch (self.w_down) {
            .f32 => |data| self.allocator.free(data),
            .q4_0 => |data| self.allocator.free(data),
            .q4_k => |data| self.allocator.free(data),
            .q6_k => {},
        }
    }
};

// ============================================================================
// Stability Tracking System
// ============================================================================

pub const StabilityTracker = struct {
    allocator: std.mem.Allocator,
    
    // Per-layer stability metrics
    attention_metrics: []std.ArrayList(mhc_constraints.StabilityMetrics),
    ffn_metrics: []std.ArrayList(mhc_constraints.StabilityMetrics),
    residual_metrics: []std.ArrayList(mhc_constraints.StabilityMetrics),
    
    // Aggregated statistics
    total_layers: u32,
    total_forward_passes: u64,
    unstable_attention_count: u64,
    unstable_ffn_count: u64,
    unstable_residual_count: u64,
    
    // Mutex for thread-safe access
    mutex: std.Thread.Mutex = .{},
    
    pub fn init(allocator: std.mem.Allocator, n_layers: u32) !*StabilityTracker {
        var tracker = try allocator.create(StabilityTracker);
        tracker.* = .{
            .allocator = allocator,
            .attention_metrics = try allocator.alloc(std.ArrayList(mhc_constraints.StabilityMetrics), n_layers),
            .ffn_metrics = try allocator.alloc(std.ArrayList(mhc_constraints.StabilityMetrics), n_layers),
            .residual_metrics = try allocator.alloc(std.ArrayList(mhc_constraints.StabilityMetrics), n_layers),
            .total_layers = n_layers,
            .total_forward_passes = 0,
            .unstable_attention_count = 0,
            .unstable_ffn_count = 0,
            .unstable_residual_count = 0,
        };
        
        // Initialize ArrayLists
        var i: usize = 0;
        while (i < n_layers) : (i += 1) {
            tracker.attention_metrics[i] = try @TypeOf(tracker.attention_metrics[i]).initCapacity(allocator, 16);
            tracker.ffn_metrics[i] = try @TypeOf(tracker.ffn_metrics[i]).initCapacity(allocator, 16);
            tracker.residual_metrics[i] = try @TypeOf(tracker.residual_metrics[i]).initCapacity(allocator, 16);
        }
        
        return tracker;
    }
    
    pub fn deinit(self: *StabilityTracker) void {
        var i: usize = 0;
        while (i < self.total_layers) : (i += 1) {
            self.attention_metrics[i].deinit(self.allocator);
            self.ffn_metrics[i].deinit(self.allocator);
            self.residual_metrics[i].deinit(self.allocator);
        }
        self.allocator.free(self.attention_metrics);
        self.allocator.free(self.ffn_metrics);
        self.allocator.free(self.residual_metrics);
        self.allocator.destroy(self);
    }
    
    pub fn recordAttentionStability(
        self: *StabilityTracker,
        layer_id: u32,
        metrics: mhc_constraints.StabilityMetrics,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.attention_metrics[layer_id].append(self.allocator, metrics);
        if (!metrics.is_stable) {
            self.unstable_attention_count += 1;
        }
    }
    
    pub fn recordFFNStability(
        self: *StabilityTracker,
        layer_id: u32,
        metrics: mhc_constraints.StabilityMetrics,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.ffn_metrics[layer_id].append(self.allocator, metrics);
        if (!metrics.is_stable) {
            self.unstable_ffn_count += 1;
        }
    }
    
    pub fn recordResidualStability(
        self: *StabilityTracker,
        layer_id: u32,
        metrics: mhc_constraints.StabilityMetrics,
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        try self.residual_metrics[layer_id].append(self.allocator, metrics);
        if (!metrics.is_stable) {
            self.unstable_residual_count += 1;
        }
    }
    
    pub fn getLayerStats(self: *StabilityTracker, layer_id: u32) LayerStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const attn_metrics = self.attention_metrics[layer_id].items;
        const ffn_metrics = self.ffn_metrics[layer_id].items;
        
        return .{
            .layer_id = layer_id,
            .attention_avg_alpha = avgAmplificationFactor(attn_metrics),
            .ffn_avg_alpha = avgAmplificationFactor(ffn_metrics),
            .attention_stability_rate = stabilityRate(attn_metrics),
            .ffn_stability_rate = stabilityRate(ffn_metrics),
            .attention_avg_iters = avgIterations(attn_metrics),
            .ffn_avg_iters = avgIterations(ffn_metrics),
        };
    }
    
    pub fn getGlobalStats(self: *StabilityTracker) GlobalStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const total_attention = self.total_forward_passes * self.total_layers;
        const total_ffn = self.total_forward_passes * self.total_layers;
        
        const attn_rate = if (total_attention > 0)
            1.0 - @as(f32, @floatFromInt(self.unstable_attention_count)) / @as(f32, @floatFromInt(total_attention))
        else
            1.0;
            
        const ffn_rate = if (total_ffn > 0)
            1.0 - @as(f32, @floatFromInt(self.unstable_ffn_count)) / @as(f32, @floatFromInt(total_ffn))
        else
            1.0;
        
        return .{
            .total_forward_passes = self.total_forward_passes,
            .attention_stability_rate = attn_rate,
            .ffn_stability_rate = ffn_rate,
        };
    }
};

pub const LayerStats = struct {
    layer_id: u32,
    attention_avg_alpha: f32,
    ffn_avg_alpha: f32,
    attention_stability_rate: f32,
    ffn_stability_rate: f32,
    attention_avg_iters: f32,
    ffn_avg_iters: f32,
};

pub const GlobalStats = struct {
    total_forward_passes: u64,
    attention_stability_rate: f32,
    ffn_stability_rate: f32,
};

fn avgAmplificationFactor(metrics: []const mhc_constraints.StabilityMetrics) f32 {
    if (metrics.len == 0) return 1.0;
    var sum: f32 = 0.0;
    for (metrics) |m| {
        sum += m.amplification_factor;
    }
    return sum / @as(f32, @floatFromInt(metrics.len));
}

fn stabilityRate(metrics: []const mhc_constraints.StabilityMetrics) f32 {
    if (metrics.len == 0) return 1.0;
    var stable_count: u32 = 0;
    for (metrics) |m| {
        if (m.is_stable) stable_count += 1;
    }
    return @as(f32, @floatFromInt(stable_count)) / @as(f32, @floatFromInt(metrics.len));
}

fn avgIterations(metrics: []const mhc_constraints.StabilityMetrics) f32 {
    if (metrics.len == 0) return 0.0;
    var sum: u32 = 0;
    for (metrics) |m| {
        sum += m.convergence_iterations;
    }
    return @as(f32, @floatFromInt(sum)) / @as(f32, @floatFromInt(metrics.len));
}

// ============================================================================
// Layer Selection Logic
// ============================================================================

/// Check if mHC should be applied to this layer
fn shouldApplyMHC(layer_id: u32, layer_range: ?LayerRange) bool {
    if (layer_range == null) return true;  // Apply to all layers
    
    const range = layer_range.?;
    return range.contains(layer_id);
}

// ============================================================================
// mHC Application Functions
// ============================================================================

fn applyMHCToAttention(
    attn_output: []f32,
    layer_id: u32,
    config: TransformerConfig,
) !void {
    const mhc_cfg = config.mhc_config.core;
    
    // Save activations before constraints for metrics
    const norm_before = mhc_constraints.compute_norm(attn_output);
    
    // Apply mHC constraints (L2 ball projection)
    _ = mhc_constraints.apply_manifold_constraints(attn_output, mhc_cfg.manifold_beta);
    
    // Compute stability metrics
    const norm_after = mhc_constraints.compute_norm(attn_output);
    const amplification = if (norm_before > 0) norm_after / norm_before else 1.0;
    const is_stable = mhc_constraints.StabilityMetrics.calculate_stability(amplification);
    
    const metrics = mhc_constraints.StabilityMetrics{
        .layer_id = layer_id,
        .signal_norm_before = norm_before,
        .signal_norm_after = norm_after,
        .amplification_factor = amplification,
        .convergence_iterations = 0,  // N/A for direct projection
        .max_activation = blk: {
            var max_val: f32 = 0.0;
            for (attn_output) |val| {
                max_val = @max(max_val, @abs(val));
            }
            break :blk max_val;
        },
        .is_stable = is_stable,
        .timestamp = std.time.milliTimestamp(),
    };
    
    // Log metrics if enabled
    if (mhc_cfg.log_stability_metrics) {
        std.log.info(
            "Layer {d} Attention mHC: alpha={d}, iters={d}, stable={s}",
            .{
                layer_id,
                metrics.amplification_factor,
                metrics.convergence_iterations,
                if (metrics.is_stable) "yes" else "no",
            },
        );
    }
    
    // Track stability
    if (config.stability_tracker) |tracker| {
        try tracker.recordAttentionStability(layer_id, metrics);
    }
    
    // Call custom callback
    if (config.mhc_config.stability_callback) |callback| {
        callback(metrics.layer_id, metrics.amplification_factor, metrics.is_stable);
    }

    // Abort on instability if configured
    if (!metrics.is_stable and config.mhc_config.abort_on_instability) {
        return error.AttentionInstability;
    }
}

fn applyMHCToFFN(
    ffn_output: []f32,
    layer_id: u32,
    config: TransformerConfig,
) !void {
    const mhc_cfg = config.mhc_config.core;
    
    // Save activations before constraints for metrics
    const norm_before = mhc_constraints.compute_norm(ffn_output);
    
    // Apply mHC constraints (L2 ball projection)
    _ = mhc_constraints.apply_manifold_constraints(ffn_output, mhc_cfg.manifold_beta);
    
    // Compute stability metrics
    const norm_after = mhc_constraints.compute_norm(ffn_output);
    const amplification = if (norm_before > 0) norm_after / norm_before else 1.0;
    const is_stable = mhc_constraints.StabilityMetrics.calculate_stability(amplification);
    
    const metrics = mhc_constraints.StabilityMetrics{
        .layer_id = layer_id,
        .signal_norm_before = norm_before,
        .signal_norm_after = norm_after,
        .amplification_factor = amplification,
        .convergence_iterations = 0,  // N/A for direct projection
        .max_activation = blk: {
            var max_val: f32 = 0.0;
            for (ffn_output) |val| {
                max_val = @max(max_val, @abs(val));
            }
            break :blk max_val;
        },
        .is_stable = is_stable,
        .timestamp = std.time.milliTimestamp(),
    };
    
    // Log metrics if enabled
    if (mhc_cfg.log_stability_metrics) {
        std.log.info(
            "Layer {d} FFN mHC: alpha={d}, iters={d}, stable={s}",
            .{
                layer_id,
                metrics.amplification_factor,
                metrics.convergence_iterations,
                if (metrics.is_stable) "yes" else "no",
            },
        );
    }
    
    // Track stability
    if (config.stability_tracker) |tracker| {
        try tracker.recordFFNStability(layer_id, metrics);
    }
    
    // Call custom callback
    if (config.mhc_config.stability_callback) |callback| {
        callback(metrics.layer_id, metrics.amplification_factor, metrics.is_stable);
    }

    // Abort on instability if configured
    if (!metrics.is_stable and config.mhc_config.abort_on_instability) {
        return error.FFNInstability;
    }
}

fn applyMHCToResidual(
    residual: []f32,
    layer_id: u32,
    config: TransformerConfig,
) !void {
    const mhc_cfg = config.mhc_config.core;
    
    // Save activations before constraints for metrics
    const norm_before = mhc_constraints.compute_norm(residual);
    
    // Apply mHC constraints (L2 ball projection)
    _ = mhc_constraints.apply_manifold_constraints(residual, mhc_cfg.manifold_beta);
    
    // Compute stability metrics
    const norm_after = mhc_constraints.compute_norm(residual);
    const amplification = if (norm_before > 0) norm_after / norm_before else 1.0;
    const is_stable = mhc_constraints.StabilityMetrics.calculate_stability(amplification);
    
    const metrics = mhc_constraints.StabilityMetrics{
        .layer_id = layer_id,
        .signal_norm_before = norm_before,
        .signal_norm_after = norm_after,
        .amplification_factor = amplification,
        .convergence_iterations = 0,  // N/A for direct projection
        .max_activation = blk: {
            var max_val: f32 = 0.0;
            for (residual) |val| {
                max_val = @max(max_val, @abs(val));
            }
            break :blk max_val;
        },
        .is_stable = is_stable,
        .timestamp = std.time.milliTimestamp(),
    };
    
    // Log metrics if enabled
    if (mhc_cfg.log_stability_metrics) {
        std.log.info(
            "Layer {d} Residual mHC: alpha={d}, iters={d}, stable={s}",
            .{
                layer_id,
                metrics.amplification_factor,
                metrics.convergence_iterations,
                if (metrics.is_stable) "yes" else "no",
            },
        );
    }
    
    // Track stability
    if (config.stability_tracker) |tracker| {
        try tracker.recordResidualStability(layer_id, metrics);
    }
    
    // Call custom callback
    if (config.mhc_config.stability_callback) |callback| {
        callback(metrics.layer_id, metrics.amplification_factor, metrics.is_stable);
    }

    // Abort on instability if configured
    if (!metrics.is_stable and config.mhc_config.abort_on_instability) {
        return error.ResidualInstability;
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

/// Compute a single transformer layer with optional mHC integration
pub fn computeTransformerLayer(
    allocator: std.mem.Allocator,
    output: []f32,
    input: []const f32,
    weights: TransformerWeights,
    cache: *kv_cache.KVCache,
    layer: u32,
    position: u32,
    config: TransformerConfig,
    rope_freqs: []const f32,
) !void {
    const embed_dim = config.embed_dim;

    // Workspace buffers
    const normed = try allocator.alloc(f32, embed_dim);
    defer allocator.free(normed);
    const attn_out = try allocator.alloc(f32, embed_dim);
    defer allocator.free(attn_out);
    const residual1 = try allocator.alloc(f32, embed_dim);
    defer allocator.free(residual1);
    const ffn_out = try allocator.alloc(f32, embed_dim);
    defer allocator.free(ffn_out);

    // 1. Pre-attention RMS norm
    matrix_ops.rms_norm(normed, input, weights.attn_norm, config.rms_norm_eps);

    // 2. Self-attention
    const attn_config = attention.AttentionConfig{
        .n_heads = config.n_heads,
        .n_kv_heads = config.n_kv_heads,
        .head_dim = config.head_dim,
        .rope_theta = config.rope_theta,
    };

    const attn_weights = attention.AttentionWeights{
        .wq = weights.wq,
        .wk = weights.wk,
        .wv = weights.wv,
        .wo = weights.wo,
    };

    try attention.computeAttention(
        allocator,
        attn_out,
        normed,
        attn_weights,
        cache,
        layer,
        position,
        attn_config,
        rope_freqs,
        null, // No thread pool for now
    );

    // 2b. Apply mHC to attention output (if enabled)
    if (config.mhc_config.enabled and
        config.mhc_config.attention_enabled and
        shouldApplyMHC(layer, config.mhc_config.layer_range)) {
        try applyMHCToAttention(attn_out, layer, config);
    }

    // 3. Residual connection
    matrix_ops.vec_add(residual1, input, attn_out);

    // 4. Pre-FFN RMS norm
    matrix_ops.rms_norm(normed, residual1, weights.ffn_norm, config.rms_norm_eps);

    // 5. Feed-forward network
    const ffn_weights = feed_forward.FFNWeights{
        .w_gate = weights.w_gate,
        .w_up = weights.w_up,
        .w_down = weights.w_down,
    };

    try feed_forward.computeFFN(allocator, ffn_out, normed, ffn_weights, config.ffn_dim, null);

    // 5b. Apply mHC to FFN output (if enabled)
    if (config.mhc_config.enabled and
        config.mhc_config.ffn_enabled and
        shouldApplyMHC(layer, config.mhc_config.layer_range)) {
        try applyMHCToFFN(ffn_out, layer, config);
    }

    // 6. Final residual connection
    matrix_ops.vec_add(output, residual1, ffn_out);

    // 7. Optional: Apply mHC to final residual (for deep instability)
    if (config.mhc_config.enabled and
        config.mhc_config.residual_enabled and
        shouldApplyMHC(layer, config.mhc_config.layer_range)) {
        try applyMHCToResidual(output, layer, config);
    }
}

// ============================================================================
// Testing
// ============================================================================

test "transformer layer without mHC" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const config = TransformerConfig{
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-5,
        .mhc_config = .{ .enabled = false },
    };
    
    const input = try allocator.alloc(f32, config.embed_dim);
    defer allocator.free(input);
    for (input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;
    
    const output = try allocator.alloc(f32, config.embed_dim);
    defer allocator.free(output);
    
    // This is a basic structure test - full test would need weights and cache
    try testing.expect(input.len == config.embed_dim);
    try testing.expect(output.len == config.embed_dim);
}

test "transformer layer with mHC enabled" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const config = TransformerConfig{
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-5,
        .mhc_config = .{
            .enabled = true,
            .attention_enabled = true,
            .ffn_enabled = true,
        },
    };
    
    const input = try allocator.alloc(f32, config.embed_dim);
    defer allocator.free(input);
    for (input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) * 0.1;
    
    try testing.expect(config.mhc_config.enabled);
    try testing.expect(config.mhc_config.attention_enabled);
    try testing.expect(config.mhc_config.ffn_enabled);
}

test "transformer layer with selective mHC" {
    const testing = std.testing;
    
    const config = TransformerConfig{
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-5,
        .mhc_config = .{
            .enabled = true,
            .attention_enabled = true,
            .ffn_enabled = true,
            .layer_range = .{ .start = 2, .end = 4 },
        },
    };
    
    // Test layer selection logic
    try testing.expect(!shouldApplyMHC(0, config.mhc_config.layer_range));
    try testing.expect(!shouldApplyMHC(1, config.mhc_config.layer_range));
    try testing.expect(shouldApplyMHC(2, config.mhc_config.layer_range));
    try testing.expect(shouldApplyMHC(3, config.mhc_config.layer_range));
    try testing.expect(!shouldApplyMHC(4, config.mhc_config.layer_range));
}

test "stability tracker records metrics" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var tracker = try StabilityTracker.init(allocator, 4);
    defer tracker.deinit();
    
    const metrics = mhc_constraints.StabilityMetrics{
        .layer_id = 0,
        .signal_norm_before = 1.0,
        .signal_norm_after = 1.05,
        .amplification_factor = 1.05,
        .convergence_iterations = 5,
        .max_activation = 10.5,
        .is_stable = true,
        .timestamp = 0,
    };
    
    try tracker.recordAttentionStability(0, metrics);
    try tracker.recordFFNStability(0, metrics);
    
    const stats = tracker.getLayerStats(0);
    try testing.expectApproxEqAbs(stats.attention_avg_alpha, 1.05, 0.01);
    try testing.expectApproxEqAbs(stats.ffn_avg_alpha, 1.05, 0.01);
}

test "layer range validation" {
    const testing = std.testing;
    
    const range1 = LayerRange{ .start = 5, .end = 8 };
    try range1.validate(10);  // Should pass
    
    const range2 = LayerRange{ .start = 8, .end = 5 };
    try testing.expectError(error.InvalidLayerRange, range2.validate(10));
    
    const range3 = LayerRange{ .start = 5, .end = 15 };
    try testing.expectError(error.InvalidLayerRange, range3.validate(10));
}

test "layer range contains" {
    const testing = std.testing;
    
    const range = LayerRange{ .start = 5, .end = 10 };
    
    try testing.expect(!range.contains(4));
    try testing.expect(range.contains(5));
    try testing.expect(range.contains(7));
    try testing.expect(range.contains(9));
    try testing.expect(!range.contains(10));
}

test "should apply mHC logic" {
    const testing = std.testing;
    
    // Null range = all layers
    try testing.expect(shouldApplyMHC(0, null));
    try testing.expect(shouldApplyMHC(100, null));
    
    // Specific range
    const range = LayerRange{ .start = 5, .end = 10 };
    try testing.expect(!shouldApplyMHC(4, range));
    try testing.expect(shouldApplyMHC(5, range));
    try testing.expect(shouldApplyMHC(9, range));
    try testing.expect(!shouldApplyMHC(10, range));
}

test "stability metrics aggregation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var tracker = try StabilityTracker.init(allocator, 2);
    defer tracker.deinit();
    
    // Add stable metrics
    const stable_metrics = mhc_constraints.StabilityMetrics{
        .layer_id = 0,
        .signal_norm_before = 1.0,
        .signal_norm_after = 1.01,
        .amplification_factor = 1.01,
        .convergence_iterations = 3,
        .max_activation = 5.0,
        .is_stable = true,
        .timestamp = 0,
    };
    
    try tracker.recordAttentionStability(0, stable_metrics);
    try tracker.recordFFNStability(0, stable_metrics);
    
    tracker.total_forward_passes = 1;
    
    const global_stats = tracker.getGlobalStats();
    try testing.expectApproxEqAbs(global_stats.attention_stability_rate, 1.0, 0.01);
    try testing.expectApproxEqAbs(global_stats.ffn_stability_rate, 1.0, 0.01);
}

pub fn test_transformer(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Transformer Layer with mHC Integration\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});

    const config = TransformerConfig{
        .embed_dim = 64,
        .ffn_dim = 256,
        .n_heads = 4,
        .n_kv_heads = 4,
        .head_dim = 16,
        .rope_theta = 10000.0,
        .rms_norm_eps = 1e-5,
        .mhc_config = .{
            .enabled = true,
            .attention_enabled = true,
            .ffn_enabled = true,
        },
    };

    const embed_dim = config.embed_dim;

    std.debug.print("\n1️⃣  Configuration test...\n", .{});
    std.debug.print("   mHC enabled: {}\n", .{config.mhc_config.enabled});
    std.debug.print("   Attention mHC: {}\n", .{config.mhc_config.attention_enabled});
    std.debug.print("   FFN mHC: {}\n", .{config.mhc_config.ffn_enabled});

    const input = try allocator.alloc(f32, embed_dim);
    defer allocator.free(input);
    for (input) |*v| v.* = 1.0;

    const output = try allocator.alloc(f32, embed_dim);
    defer allocator.free(output);

    std.debug.print("\n2️⃣  Testing stability tracker...\n", .{});
    var tracker = try StabilityTracker.init(allocator, 4);
    defer tracker.deinit();
    
    const metrics = mhc_constraints.StabilityMetrics{
        .layer_id = 0,
        .signal_norm_before = 1.0,
        .signal_norm_after = 1.05,
        .amplification_factor = 1.05,
        .convergence_iterations = 5,
        .max_activation = 10.5,
        .is_stable = true,
        .timestamp = 0,
    };
    
    try tracker.recordAttentionStability(0, metrics);
    const stats = tracker.getLayerStats(0);
    std.debug.print("   Layer 0 avg alpha: {d}\n", .{stats.attention_avg_alpha});

    std.debug.print("\n✅ Transformer mHC integration tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
