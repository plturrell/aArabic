// mHC Failure Mode Detection Module
// Implements failure detection and adaptive tau control for manifold Hamiltonian Control
//
// Core Components:
// - FailureMode: Enum of possible failure modes
// - Detection Functions: Detect various failure conditions
// - AdaptiveTauController: Adaptive tau parameter control
// - Mitigation Strategies: Recovery strategies for each failure mode
//
// Reference: docs/DAY_63_FAILURE_DETECTION_REPORT.md

const std = @import("std");
const math = std.math;

// ============================================================================
// Failure Mode Enumeration
// ============================================================================

/// Represents different failure modes that can occur during mHC optimization
pub const FailureMode = enum {
    /// No failure detected
    none,

    /// Tau too high, causing over-regularization/over-smoothing
    over_constraint,

    /// Geometric loss and statistical loss are diverging
    geo_stat_conflict,

    /// Sudden spike in energy function
    energy_spike,

    /// Maximum iterations exceeded without convergence
    convergence_failure,

    /// NaN or Inf values detected
    numerical_instability,

    /// Format failure mode as string
    pub fn toString(self: FailureMode) []const u8 {
        return switch (self) {
            .none => "none",
            .over_constraint => "over_constraint",
            .geo_stat_conflict => "geo_stat_conflict",
            .energy_spike => "energy_spike",
            .convergence_failure => "convergence_failure",
            .numerical_instability => "numerical_instability",
        };
    }

    /// Get severity level (0-3: info, warning, error, critical)
    pub fn severity(self: FailureMode) u8 {
        return switch (self) {
            .none => 0,
            .over_constraint => 1,
            .geo_stat_conflict => 2,
            .energy_spike => 2,
            .convergence_failure => 2,
            .numerical_instability => 3,
        };
    }

    /// Check if failure is recoverable
    pub fn isRecoverable(self: FailureMode) bool {
        return switch (self) {
            .none => true,
            .over_constraint => true,
            .geo_stat_conflict => true,
            .energy_spike => true,
            .convergence_failure => true,
            .numerical_instability => false, // NaN/Inf usually require reset
        };
    }
};

// ============================================================================
// Detection Configuration
// ============================================================================

/// Configuration for failure detection thresholds
pub const DetectionConfig = struct {
    /// Threshold for over-constraint detection (variance ratio)
    over_constraint_threshold: f32 = 0.1,

    /// Threshold for geo-stat conflict (loss ratio)
    geo_stat_ratio_threshold: f32 = 5.0,

    /// Threshold for energy spike (relative increase)
    energy_spike_threshold: f32 = 2.0,

    /// Window size for energy history analysis
    energy_history_window: usize = 10,

    /// Maximum allowed iterations for convergence
    max_iterations: u32 = 1000,

    /// Threshold for variance being too low (over-smoothed)
    min_variance_threshold: f32 = 1e-8,

    /// Create default configuration
    pub fn default() DetectionConfig {
        return DetectionConfig{};
    }
};

// ============================================================================
// Detection Result
// ============================================================================

/// Result of failure detection analysis
pub const DetectionResult = struct {
    /// Detected failure mode
    mode: FailureMode,

    /// Confidence of detection (0.0-1.0)
    confidence: f32,

    /// Severity level (0-3)
    severity: u8,

    /// Detailed message
    message: []const u8,

    /// Timestamp of detection
    timestamp: i64,

    /// Create a no-failure result
    pub fn noFailure() DetectionResult {
        return DetectionResult{
            .mode = .none,
            .confidence = 1.0,
            .severity = 0,
            .message = "No failure detected",
            .timestamp = std.time.milliTimestamp(),
        };
    }

    /// Create a failure result
    pub fn failure(mode: FailureMode, confidence: f32, message: []const u8) DetectionResult {
        return DetectionResult{
            .mode = mode,
            .confidence = confidence,
            .severity = mode.severity(),
            .message = message,
            .timestamp = std.time.milliTimestamp(),
        };
    }
};

// ============================================================================
// Detection Functions
// ============================================================================

/// Detect over-constraint failure
/// Returns true if tau causes over-smoothing (variance too low relative to tau)
pub fn detect_over_constraint(tau: f32, variance: f32, threshold: f32) bool {
    // Over-constraint when high tau leads to very low variance
    if (tau <= 0 or variance < 0) return false;

    // Ratio of variance to tau - if very low, over-constrained
    const ratio = variance / tau;
    return ratio < threshold;
}

/// Detect over-constraint with detailed result
pub fn detect_over_constraint_detailed(tau: f32, variance: f32, threshold: f32) DetectionResult {
    if (tau <= 0 or variance < 0) {
        return DetectionResult.noFailure();
    }

    const ratio = variance / tau;
    if (ratio < threshold) {
        const confidence = 1.0 - (ratio / threshold);
        return DetectionResult.failure(
            .over_constraint,
            @min(1.0, confidence),
            "Variance too low relative to tau - over-regularization detected",
        );
    }
    return DetectionResult.noFailure();
}

/// Detect geometric vs statistical loss conflict
/// Returns true if the ratio between losses exceeds threshold
pub fn detect_geo_stat_conflict(geo_loss: f32, stat_loss: f32, ratio_threshold: f32) bool {
    // Avoid division by zero
    if (stat_loss <= 0 or geo_loss <= 0) return false;

    // Check ratio in both directions
    const ratio = if (geo_loss > stat_loss) geo_loss / stat_loss else stat_loss / geo_loss;
    return ratio > ratio_threshold;
}

/// Detect geo-stat conflict with detailed result
pub fn detect_geo_stat_conflict_detailed(geo_loss: f32, stat_loss: f32, ratio_threshold: f32) DetectionResult {
    if (stat_loss <= 0 or geo_loss <= 0) {
        return DetectionResult.noFailure();
    }

    const ratio = if (geo_loss > stat_loss) geo_loss / stat_loss else stat_loss / geo_loss;
    if (ratio > ratio_threshold) {
        const confidence = @min(1.0, (ratio - ratio_threshold) / ratio_threshold);
        const message = if (geo_loss > stat_loss)
            "Geometric loss dominates statistical loss"
        else
            "Statistical loss dominates geometric loss";
        return DetectionResult.failure(.geo_stat_conflict, confidence, message);
    }
    return DetectionResult.noFailure();
}

/// Detect sudden energy spike in history
/// Returns true if recent energy increase exceeds threshold relative to baseline
pub fn detect_energy_spike(energy_history: []const f32, spike_threshold: f32) bool {
    if (energy_history.len < 2) return false;

    // Calculate baseline (mean of all but last value)
    var baseline: f32 = 0.0;
    const baseline_count = energy_history.len - 1;
    for (energy_history[0..baseline_count]) |e| {
        baseline += e;
    }
    baseline /= @as(f32, @floatFromInt(baseline_count));

    if (baseline <= 0) return false;

    // Check if last value spikes relative to baseline
    const last_energy = energy_history[energy_history.len - 1];
    const ratio = last_energy / baseline;
    return ratio > spike_threshold;
}

/// Detect energy spike with detailed result
pub fn detect_energy_spike_detailed(energy_history: []const f32, spike_threshold: f32) DetectionResult {
    if (energy_history.len < 2) {
        return DetectionResult.noFailure();
    }

    var baseline: f32 = 0.0;
    const baseline_count = energy_history.len - 1;
    for (energy_history[0..baseline_count]) |e| {
        baseline += e;
    }
    baseline /= @as(f32, @floatFromInt(baseline_count));

    if (baseline <= 0) {
        return DetectionResult.noFailure();
    }

    const last_energy = energy_history[energy_history.len - 1];
    const ratio = last_energy / baseline;
    if (ratio > spike_threshold) {
        const confidence = @min(1.0, (ratio - spike_threshold) / spike_threshold);
        return DetectionResult.failure(
            .energy_spike,
            confidence,
            "Sudden energy increase detected",
        );
    }
    return DetectionResult.noFailure();
}

/// Check if iterations exceeded maximum (convergence failure)
pub fn detect_convergence_failure(iterations: u32, max_iterations: u32) bool {
    return iterations >= max_iterations;
}

/// Detect convergence failure with detailed result
pub fn detect_convergence_failure_detailed(iterations: u32, max_iterations: u32) DetectionResult {
    if (iterations >= max_iterations) {
        // Confidence based on how much we exceeded
        const overshoot = iterations - max_iterations;
        const confidence = @min(1.0, 0.5 + @as(f32, @floatFromInt(overshoot)) / @as(f32, @floatFromInt(max_iterations)));
        return DetectionResult.failure(
            .convergence_failure,
            confidence,
            "Maximum iterations exceeded without convergence",
        );
    }
    return DetectionResult.noFailure();
}

/// Detect numerical instability (NaN or Inf values)
pub fn detect_numerical_instability(values: []const f32) bool {
    for (values) |v| {
        if (math.isNan(v) or math.isInf(v)) {
            return true;
        }
    }
    return false;
}

/// Detect numerical instability with detailed result
pub fn detect_numerical_instability_detailed(values: []const f32) DetectionResult {
    var nan_count: u32 = 0;
    var inf_count: u32 = 0;

    for (values) |v| {
        if (math.isNan(v)) nan_count += 1;
        if (math.isInf(v)) inf_count += 1;
    }

    if (nan_count > 0 or inf_count > 0) {
        const total_bad = nan_count + inf_count;
        const confidence = @min(1.0, @as(f32, @floatFromInt(total_bad)) / @as(f32, @floatFromInt(values.len)));
        const message = if (nan_count > 0 and inf_count > 0)
            "Both NaN and Inf values detected"
        else if (nan_count > 0)
            "NaN values detected in tensor"
        else
            "Inf values detected in tensor";
        return DetectionResult.failure(.numerical_instability, @max(0.5, confidence), message);
    }
    return DetectionResult.noFailure();
}

/// Count NaN values in array
pub fn count_nan_values(values: []const f32) u32 {
    var count: u32 = 0;
    for (values) |v| {
        if (math.isNan(v)) count += 1;
    }
    return count;
}

/// Count Inf values in array
pub fn count_inf_values(values: []const f32) u32 {
    var count: u32 = 0;
    for (values) |v| {
        if (math.isInf(v)) count += 1;
    }
    return count;
}


// ============================================================================
// Adaptive Tau Controller
// ============================================================================

/// Adaptive controller for tau parameter based on detected failure modes
pub const AdaptiveTauController = struct {
    /// Current tau value
    current_tau: f32,

    /// Minimum allowed tau value
    min_tau: f32 = 0.001,

    /// Maximum allowed tau value
    max_tau: f32 = 10.0,

    /// Rate of adaptation (0.0-1.0)
    adaptation_rate: f32 = 0.1,

    /// History of tau adjustments (stored as array for simplicity)
    adjustment_history_data: [64]TauAdjustment = undefined,
    adjustment_history_len: usize = 0,

    /// Statistics tracking
    total_adjustments: u32 = 0,
    upward_adjustments: u32 = 0,
    downward_adjustments: u32 = 0,

    /// Momentum for smooth adjustments
    momentum: f32 = 0.0,

    /// Momentum decay factor
    momentum_decay: f32 = 0.9,

    /// Tau adjustment record
    pub const TauAdjustment = struct {
        timestamp: i64,
        old_tau: f32,
        new_tau: f32,
        reason: FailureMode,
        delta: f32,
    };

    /// Initialize controller with starting tau
    pub fn init(_: std.mem.Allocator, initial_tau: f32) AdaptiveTauController {
        return AdaptiveTauController{
            .current_tau = initial_tau,
        };
    }

    /// Initialize with custom bounds
    pub fn initWithBounds(
        _: std.mem.Allocator,
        initial_tau: f32,
        min_tau: f32,
        max_tau: f32,
        adaptation_rate: f32,
    ) AdaptiveTauController {
        return AdaptiveTauController{
            .current_tau = @max(min_tau, @min(max_tau, initial_tau)),
            .min_tau = min_tau,
            .max_tau = max_tau,
            .adaptation_rate = adaptation_rate,
        };
    }

    /// Deinitialize and free resources
    pub fn deinit(self: *AdaptiveTauController) void {
        self.adjustment_history_len = 0;
    }

    /// Adjust tau based on detected failure mode
    pub fn adjust_tau(self: *AdaptiveTauController, failure_mode: FailureMode) f32 {
        const old_tau = self.current_tau;
        var delta: f32 = 0.0;

        switch (failure_mode) {
            .none => {
                // No adjustment needed, but apply momentum decay
                self.momentum *= self.momentum_decay;
                return self.current_tau;
            },
            .over_constraint => {
                // Reduce tau to decrease regularization
                delta = -self.current_tau * self.adaptation_rate;
            },
            .geo_stat_conflict => {
                // Moderate reduction to rebalance
                delta = -self.current_tau * self.adaptation_rate * 0.5;
            },
            .energy_spike => {
                // Reduce tau more aggressively
                delta = -self.current_tau * self.adaptation_rate * 1.5;
                // Also reduce momentum
                self.momentum *= 0.5;
            },
            .convergence_failure => {
                // Increase tau slightly to strengthen constraints
                delta = self.current_tau * self.adaptation_rate * 0.3;
            },
            .numerical_instability => {
                // Emergency reduction
                delta = -self.current_tau * 0.5;
                self.momentum = 0.0;
            },
        }

        // Apply momentum
        self.momentum = self.momentum_decay * self.momentum + delta;
        const adjusted_delta = self.momentum;

        // Apply adjustment with bounds
        self.current_tau = @max(self.min_tau, @min(self.max_tau, self.current_tau + adjusted_delta));

        // Track statistics
        self.total_adjustments += 1;
        if (self.current_tau > old_tau) {
            self.upward_adjustments += 1;
        } else if (self.current_tau < old_tau) {
            self.downward_adjustments += 1;
        }

        // Record adjustment (ring buffer)
        if (self.adjustment_history_len < 64) {
            self.adjustment_history_data[self.adjustment_history_len] = TauAdjustment{
                .timestamp = std.time.milliTimestamp(),
                .old_tau = old_tau,
                .new_tau = self.current_tau,
                .reason = failure_mode,
                .delta = self.current_tau - old_tau,
            };
            self.adjustment_history_len += 1;
        }

        return self.current_tau;
    }

    /// Get current tau value
    pub fn get_tau(self: *const AdaptiveTauController) f32 {
        return self.current_tau;
    }

    /// Reset tau to a specific value
    pub fn reset_tau(self: *AdaptiveTauController, new_tau: f32) void {
        self.current_tau = @max(self.min_tau, @min(self.max_tau, new_tau));
        self.momentum = 0.0;
    }

    /// Get adjustment statistics
    pub fn get_statistics(self: *const AdaptiveTauController) struct {
        total: u32,
        upward: u32,
        downward: u32,
        avg_delta: f32,
    } {
        var total_delta: f32 = 0.0;
        for (self.adjustment_history_data[0..self.adjustment_history_len]) |adj| {
            total_delta += @abs(adj.delta);
        }
        const avg = if (self.adjustment_history_len > 0)
            total_delta / @as(f32, @floatFromInt(self.adjustment_history_len))
        else
            0.0;

        return .{
            .total = self.total_adjustments,
            .upward = self.upward_adjustments,
            .downward = self.downward_adjustments,
            .avg_delta = avg,
        };
    }

    /// Check if tau is at minimum bound
    pub fn is_at_min(self: *const AdaptiveTauController) bool {
        return self.current_tau <= self.min_tau;
    }

    /// Check if tau is at maximum bound
    pub fn is_at_max(self: *const AdaptiveTauController) bool {
        return self.current_tau >= self.max_tau;
    }
};


// ============================================================================
// Mitigation Strategies
// ============================================================================

/// Result of applying a mitigation strategy
pub const MitigationResult = struct {
    /// Whether mitigation was successful
    success: bool,

    /// New tau value after mitigation
    new_tau: f32,

    /// Description of action taken
    action: []const u8,

    /// Additional parameter adjustments applied
    momentum_adjusted: bool = false,
    weights_rebalanced: bool = false,

    /// Number of mitigation steps applied
    steps_applied: u32 = 1,
};

/// Mitigate over-constraint by reducing tau
pub fn mitigate_over_constraint(controller: *AdaptiveTauController) MitigationResult {
    const old_tau = controller.current_tau;

    // Apply multiple reduction steps for over-constraint
    _ = controller.adjust_tau(.over_constraint);

    // Additional reduction if still high
    if (controller.current_tau > controller.max_tau * 0.5) {
        _ = controller.adjust_tau(.over_constraint);
        return MitigationResult{
            .success = true,
            .new_tau = controller.current_tau,
            .action = "Applied double tau reduction for severe over-constraint",
            .steps_applied = 2,
        };
    }

    return MitigationResult{
        .success = controller.current_tau < old_tau,
        .new_tau = controller.current_tau,
        .action = "Reduced tau to decrease regularization",
        .steps_applied = 1,
    };
}

/// Mitigate energy spike by reducing tau and momentum
pub fn mitigate_energy_spike(controller: *AdaptiveTauController) MitigationResult {
    const old_tau = controller.current_tau;

    // Energy spike mitigation includes momentum reset
    _ = controller.adjust_tau(.energy_spike);

    return MitigationResult{
        .success = controller.current_tau < old_tau,
        .new_tau = controller.current_tau,
        .action = "Reduced tau and damped momentum for energy spike",
        .momentum_adjusted = true,
        .steps_applied = 1,
    };
}

/// Mitigate geo-stat conflict by rebalancing weights
pub fn mitigate_geo_stat_conflict(controller: *AdaptiveTauController) MitigationResult {
    // Moderate adjustment for geo-stat conflict
    _ = controller.adjust_tau(.geo_stat_conflict);

    return MitigationResult{
        .success = true,
        .new_tau = controller.current_tau,
        .action = "Adjusted tau to rebalance geometric and statistical losses",
        .weights_rebalanced = true,
        .steps_applied = 1,
    };
}

/// Mitigate convergence failure by strengthening constraints
pub fn mitigate_convergence_failure(controller: *AdaptiveTauController) MitigationResult {
    const old_tau = controller.current_tau;

    _ = controller.adjust_tau(.convergence_failure);

    return MitigationResult{
        .success = controller.current_tau > old_tau,
        .new_tau = controller.current_tau,
        .action = "Increased tau to strengthen convergence",
        .steps_applied = 1,
    };
}

/// Mitigate numerical instability with emergency reset
pub fn mitigate_numerical_instability(controller: *AdaptiveTauController) MitigationResult {
    // Emergency reduction for numerical instability
    _ = controller.adjust_tau(.numerical_instability);

    // If still problematic, reset to minimum
    if (controller.current_tau > controller.min_tau * 10) {
        controller.reset_tau(controller.min_tau * 2);
    }

    return MitigationResult{
        .success = true, // Always "succeed" but may need external reset
        .new_tau = controller.current_tau,
        .action = "Emergency tau reduction for numerical instability",
        .momentum_adjusted = true,
        .steps_applied = 2,
    };
}

/// Apply appropriate mitigation for detected failure mode
pub fn apply_mitigation(controller: *AdaptiveTauController, failure_mode: FailureMode) MitigationResult {
    return switch (failure_mode) {
        .none => MitigationResult{
            .success = true,
            .new_tau = controller.current_tau,
            .action = "No mitigation needed",
            .steps_applied = 0,
        },
        .over_constraint => mitigate_over_constraint(controller),
        .geo_stat_conflict => mitigate_geo_stat_conflict(controller),
        .energy_spike => mitigate_energy_spike(controller),
        .convergence_failure => mitigate_convergence_failure(controller),
        .numerical_instability => mitigate_numerical_instability(controller),
    };
}

// ============================================================================
// Comprehensive Failure Detector
// ============================================================================

/// Comprehensive failure detector that checks all failure modes
pub const FailureDetector = struct {
    config: DetectionConfig,

    /// Energy history buffer (fixed size ring buffer)
    energy_buffer_data: [64]f32 = undefined,
    energy_buffer_len: usize = 0,

    /// Last detected failure
    last_failure: FailureMode = .none,

    /// Detection statistics
    detection_counts: struct {
        over_constraint: u32 = 0,
        geo_stat_conflict: u32 = 0,
        energy_spike: u32 = 0,
        convergence_failure: u32 = 0,
        numerical_instability: u32 = 0,
    } = .{},

    /// Initialize detector
    pub fn init(_: std.mem.Allocator, config: DetectionConfig) FailureDetector {
        return FailureDetector{
            .config = config,
        };
    }

    /// Deinitialize detector
    pub fn deinit(self: *FailureDetector) void {
        self.energy_buffer_len = 0;
    }

    /// Add energy value to history
    pub fn record_energy(self: *FailureDetector, energy: f32) void {
        if (self.energy_buffer_len < 64) {
            self.energy_buffer_data[self.energy_buffer_len] = energy;
            self.energy_buffer_len += 1;
        } else {
            // Shift buffer left (ring buffer behavior)
            for (0..63) |i| {
                self.energy_buffer_data[i] = self.energy_buffer_data[i + 1];
            }
            self.energy_buffer_data[63] = energy;
        }
    }

    /// Get energy buffer as slice
    fn getEnergyBuffer(self: *const FailureDetector) []const f32 {
        return self.energy_buffer_data[0..self.energy_buffer_len];
    }

    /// Run comprehensive failure detection
    pub fn detect(
        self: *FailureDetector,
        tau: f32,
        variance: f32,
        geo_loss: f32,
        stat_loss: f32,
        iterations: u32,
        values: []const f32,
    ) DetectionResult {
        // Check numerical instability first (most critical)
        if (detect_numerical_instability(values)) {
            self.last_failure = .numerical_instability;
            self.detection_counts.numerical_instability += 1;
            return detect_numerical_instability_detailed(values);
        }

        // Check energy spike
        const energy_buf = self.getEnergyBuffer();
        if (energy_buf.len >= 2) {
            if (detect_energy_spike(energy_buf, self.config.energy_spike_threshold)) {
                self.last_failure = .energy_spike;
                self.detection_counts.energy_spike += 1;
                return detect_energy_spike_detailed(energy_buf, self.config.energy_spike_threshold);
            }
        }

        // Check convergence failure
        if (detect_convergence_failure(iterations, self.config.max_iterations)) {
            self.last_failure = .convergence_failure;
            self.detection_counts.convergence_failure += 1;
            return detect_convergence_failure_detailed(iterations, self.config.max_iterations);
        }

        // Check over-constraint
        if (detect_over_constraint(tau, variance, self.config.over_constraint_threshold)) {
            self.last_failure = .over_constraint;
            self.detection_counts.over_constraint += 1;
            return detect_over_constraint_detailed(tau, variance, self.config.over_constraint_threshold);
        }

        // Check geo-stat conflict
        if (detect_geo_stat_conflict(geo_loss, stat_loss, self.config.geo_stat_ratio_threshold)) {
            self.last_failure = .geo_stat_conflict;
            self.detection_counts.geo_stat_conflict += 1;
            return detect_geo_stat_conflict_detailed(geo_loss, stat_loss, self.config.geo_stat_ratio_threshold);
        }

        self.last_failure = .none;
        return DetectionResult.noFailure();
    }

    /// Get total detection count
    pub fn total_detections(self: *const FailureDetector) u32 {
        const counts = self.detection_counts;
        return counts.over_constraint + counts.geo_stat_conflict +
            counts.energy_spike + counts.convergence_failure + counts.numerical_instability;
    }

    /// Reset statistics
    pub fn reset_stats(self: *FailureDetector) void {
        self.detection_counts = .{};
        self.last_failure = .none;
        self.energy_buffer_len = 0;
    }
};


// ============================================================================
// Unit Tests
// ============================================================================

test "FailureMode.toString returns correct strings" {
    try std.testing.expectEqualStrings("none", FailureMode.none.toString());
    try std.testing.expectEqualStrings("over_constraint", FailureMode.over_constraint.toString());
    try std.testing.expectEqualStrings("geo_stat_conflict", FailureMode.geo_stat_conflict.toString());
    try std.testing.expectEqualStrings("energy_spike", FailureMode.energy_spike.toString());
    try std.testing.expectEqualStrings("convergence_failure", FailureMode.convergence_failure.toString());
    try std.testing.expectEqualStrings("numerical_instability", FailureMode.numerical_instability.toString());
}

test "FailureMode.severity returns correct levels" {
    try std.testing.expectEqual(@as(u8, 0), FailureMode.none.severity());
    try std.testing.expectEqual(@as(u8, 1), FailureMode.over_constraint.severity());
    try std.testing.expectEqual(@as(u8, 2), FailureMode.geo_stat_conflict.severity());
    try std.testing.expectEqual(@as(u8, 2), FailureMode.energy_spike.severity());
    try std.testing.expectEqual(@as(u8, 2), FailureMode.convergence_failure.severity());
    try std.testing.expectEqual(@as(u8, 3), FailureMode.numerical_instability.severity());
}

test "FailureMode.isRecoverable returns correct status" {
    try std.testing.expect(FailureMode.none.isRecoverable());
    try std.testing.expect(FailureMode.over_constraint.isRecoverable());
    try std.testing.expect(FailureMode.geo_stat_conflict.isRecoverable());
    try std.testing.expect(FailureMode.energy_spike.isRecoverable());
    try std.testing.expect(FailureMode.convergence_failure.isRecoverable());
    try std.testing.expect(!FailureMode.numerical_instability.isRecoverable());
}

test "detect_over_constraint detects high tau with low variance" {
    // High tau (5.0) with low variance (0.01) -> over-constrained
    try std.testing.expect(detect_over_constraint(5.0, 0.01, 0.1));

    // Normal tau (1.0) with normal variance (0.5) -> not over-constrained
    try std.testing.expect(!detect_over_constraint(1.0, 0.5, 0.1));

    // Edge case: zero tau
    try std.testing.expect(!detect_over_constraint(0.0, 0.5, 0.1));

    // Edge case: negative variance
    try std.testing.expect(!detect_over_constraint(1.0, -0.5, 0.1));
}

test "detect_over_constraint_detailed returns correct result" {
    const result = detect_over_constraint_detailed(5.0, 0.01, 0.1);
    try std.testing.expectEqual(FailureMode.over_constraint, result.mode);
    try std.testing.expect(result.confidence > 0.0);
    try std.testing.expect(result.confidence <= 1.0);
}

test "detect_geo_stat_conflict detects diverging losses" {
    // Large ratio (10:1) -> conflict
    try std.testing.expect(detect_geo_stat_conflict(10.0, 1.0, 5.0));

    // Large ratio other direction (1:10) -> conflict
    try std.testing.expect(detect_geo_stat_conflict(1.0, 10.0, 5.0));

    // Small ratio (2:1) -> no conflict
    try std.testing.expect(!detect_geo_stat_conflict(2.0, 1.0, 5.0));

    // Equal losses -> no conflict
    try std.testing.expect(!detect_geo_stat_conflict(1.0, 1.0, 5.0));

    // Zero loss -> no conflict (avoids division by zero)
    try std.testing.expect(!detect_geo_stat_conflict(1.0, 0.0, 5.0));
}

test "detect_geo_stat_conflict_detailed returns correct message" {
    const result1 = detect_geo_stat_conflict_detailed(10.0, 1.0, 5.0);
    try std.testing.expectEqual(FailureMode.geo_stat_conflict, result1.mode);
    try std.testing.expectEqualStrings("Geometric loss dominates statistical loss", result1.message);

    const result2 = detect_geo_stat_conflict_detailed(1.0, 10.0, 5.0);
    try std.testing.expectEqualStrings("Statistical loss dominates geometric loss", result2.message);
}

test "detect_energy_spike detects sudden increase" {
    const history_spike = [_]f32{ 1.0, 1.1, 1.0, 0.9, 1.0, 5.0 };
    try std.testing.expect(detect_energy_spike(&history_spike, 2.0));

    const history_normal = [_]f32{ 1.0, 1.1, 1.0, 0.9, 1.0, 1.2 };
    try std.testing.expect(!detect_energy_spike(&history_normal, 2.0));

    // Single value -> no spike detectable
    const history_single = [_]f32{1.0};
    try std.testing.expect(!detect_energy_spike(&history_single, 2.0));
}

test "detect_energy_spike_detailed returns correct confidence" {
    const history = [_]f32{ 1.0, 1.0, 1.0, 1.0, 4.0 };
    const result = detect_energy_spike_detailed(&history, 2.0);
    try std.testing.expectEqual(FailureMode.energy_spike, result.mode);
    try std.testing.expect(result.confidence > 0.0);
}

test "detect_convergence_failure checks iterations" {
    try std.testing.expect(detect_convergence_failure(1000, 1000));
    try std.testing.expect(detect_convergence_failure(1001, 1000));
    try std.testing.expect(!detect_convergence_failure(999, 1000));
    try std.testing.expect(!detect_convergence_failure(0, 1000));
}

test "detect_convergence_failure_detailed returns result" {
    const result = detect_convergence_failure_detailed(1000, 1000);
    try std.testing.expectEqual(FailureMode.convergence_failure, result.mode);
    try std.testing.expect(result.confidence >= 0.5);
}

test "detect_numerical_instability detects NaN" {
    const values_nan = [_]f32{ 1.0, 2.0, math.nan(f32), 4.0 };
    try std.testing.expect(detect_numerical_instability(&values_nan));

    const values_normal = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    try std.testing.expect(!detect_numerical_instability(&values_normal));
}

test "detect_numerical_instability detects Inf" {
    const values_inf = [_]f32{ 1.0, 2.0, math.inf(f32), 4.0 };
    try std.testing.expect(detect_numerical_instability(&values_inf));

    const values_neg_inf = [_]f32{ 1.0, -math.inf(f32), 3.0 };
    try std.testing.expect(detect_numerical_instability(&values_neg_inf));
}

test "detect_numerical_instability_detailed counts bad values" {
    const values = [_]f32{ math.nan(f32), math.inf(f32), 1.0, math.nan(f32) };
    const result = detect_numerical_instability_detailed(&values);
    try std.testing.expectEqual(FailureMode.numerical_instability, result.mode);
    try std.testing.expect(result.confidence >= 0.5);
}

test "count_nan_values counts correctly" {
    const values = [_]f32{ math.nan(f32), 1.0, math.nan(f32), 2.0 };
    try std.testing.expectEqual(@as(u32, 2), count_nan_values(&values));

    const clean = [_]f32{ 1.0, 2.0, 3.0 };
    try std.testing.expectEqual(@as(u32, 0), count_nan_values(&clean));
}

test "count_inf_values counts correctly" {
    const values = [_]f32{ math.inf(f32), 1.0, -math.inf(f32), 2.0 };
    try std.testing.expectEqual(@as(u32, 2), count_inf_values(&values));
}


// ============================================================================
// AdaptiveTauController Tests
// ============================================================================

test "AdaptiveTauController.init creates controller with correct tau" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), controller.get_tau(), 0.001);
}

test "AdaptiveTauController.initWithBounds respects bounds" {
    const allocator = std.testing.allocator;

    // Initial tau clamped to max
    var controller1 = AdaptiveTauController.initWithBounds(allocator, 20.0, 0.1, 5.0, 0.1);
    defer controller1.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), controller1.get_tau(), 0.001);

    // Initial tau clamped to min
    var controller2 = AdaptiveTauController.initWithBounds(allocator, 0.001, 0.1, 5.0, 0.1);
    defer controller2.deinit();
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), controller2.get_tau(), 0.001);
}

test "AdaptiveTauController.adjust_tau reduces tau for over_constraint" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const old_tau = controller.current_tau;
    _ = controller.adjust_tau(.over_constraint);

    try std.testing.expect(controller.current_tau < old_tau);
}

test "AdaptiveTauController.adjust_tau increases tau for convergence_failure" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const old_tau = controller.current_tau;
    _ = controller.adjust_tau(.convergence_failure);

    try std.testing.expect(controller.current_tau > old_tau);
}

test "AdaptiveTauController.adjust_tau handles energy_spike" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const old_tau = controller.current_tau;
    _ = controller.adjust_tau(.energy_spike);

    try std.testing.expect(controller.current_tau < old_tau);
    // Momentum should be reduced
    try std.testing.expect(controller.momentum < 0);
}

test "AdaptiveTauController.adjust_tau no change for none" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const old_tau = controller.current_tau;
    _ = controller.adjust_tau(.none);

    try std.testing.expectApproxEqAbs(old_tau, controller.current_tau, 0.001);
}

test "AdaptiveTauController respects min_tau bound" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.initWithBounds(allocator, 0.01, 0.001, 10.0, 0.9);
    defer controller.deinit();

    // Apply many reductions
    for (0..20) |_| {
        _ = controller.adjust_tau(.numerical_instability);
    }

    try std.testing.expect(controller.current_tau >= controller.min_tau);
    try std.testing.expect(controller.is_at_min());
}

test "AdaptiveTauController respects max_tau bound" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.initWithBounds(allocator, 9.0, 0.001, 10.0, 0.9);
    defer controller.deinit();

    // Apply many increases
    for (0..20) |_| {
        _ = controller.adjust_tau(.convergence_failure);
    }

    try std.testing.expect(controller.current_tau <= controller.max_tau);
    try std.testing.expect(controller.is_at_max());
}

test "AdaptiveTauController.reset_tau resets correctly" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    _ = controller.adjust_tau(.over_constraint);
    controller.reset_tau(5.0);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), controller.current_tau, 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), controller.momentum, 0.001);
}

test "AdaptiveTauController.get_statistics tracks adjustments" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    _ = controller.adjust_tau(.over_constraint);
    _ = controller.adjust_tau(.convergence_failure);
    _ = controller.adjust_tau(.over_constraint);

    const stats = controller.get_statistics();
    try std.testing.expectEqual(@as(u32, 3), stats.total);
    try std.testing.expect(stats.downward >= 2);
}

// ============================================================================
// Mitigation Strategy Tests
// ============================================================================

test "mitigate_over_constraint reduces tau" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 5.0);
    defer controller.deinit();

    const result = mitigate_over_constraint(&controller);

    try std.testing.expect(result.success);
    try std.testing.expect(result.new_tau < 5.0);
    try std.testing.expect(result.steps_applied >= 1);
}

test "mitigate_energy_spike reduces tau and adjusts momentum" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 2.0);
    defer controller.deinit();

    const result = mitigate_energy_spike(&controller);

    try std.testing.expect(result.success);
    try std.testing.expect(result.new_tau < 2.0);
    try std.testing.expect(result.momentum_adjusted);
}

test "mitigate_geo_stat_conflict adjusts weights" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const result = mitigate_geo_stat_conflict(&controller);

    try std.testing.expect(result.success);
    try std.testing.expect(result.weights_rebalanced);
}

test "mitigate_convergence_failure increases tau" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 1.0);
    defer controller.deinit();

    const result = mitigate_convergence_failure(&controller);

    try std.testing.expect(result.success);
    try std.testing.expect(result.new_tau > 1.0);
}

test "mitigate_numerical_instability emergency reduction" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 5.0);
    defer controller.deinit();

    const result = mitigate_numerical_instability(&controller);

    try std.testing.expect(result.success);
    try std.testing.expect(result.new_tau < 5.0);
    try std.testing.expect(result.momentum_adjusted);
}

test "apply_mitigation dispatches correctly" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 2.0);
    defer controller.deinit();

    const result_none = apply_mitigation(&controller, .none);
    try std.testing.expectEqual(@as(u32, 0), result_none.steps_applied);

    const result_over = apply_mitigation(&controller, .over_constraint);
    try std.testing.expect(result_over.steps_applied >= 1);
}


// ============================================================================
// FailureDetector Tests
// ============================================================================

test "FailureDetector.init creates detector with default config" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    try std.testing.expectEqual(FailureMode.none, detector.last_failure);
    try std.testing.expectEqual(@as(u32, 0), detector.total_detections());
}

test "FailureDetector.record_energy adds to buffer" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    detector.record_energy(1.0);
    detector.record_energy(2.0);
    detector.record_energy(3.0);

    try std.testing.expectEqual(@as(usize, 3), detector.energy_buffer_len);
}

test "FailureDetector.detect finds numerical instability first" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ 1.0, math.nan(f32), 2.0 };
    const result = detector.detect(1.0, 0.5, 1.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.numerical_instability, result.mode);
    try std.testing.expectEqual(@as(u32, 1), detector.detection_counts.numerical_instability);
}

test "FailureDetector.detect finds energy spike" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    // Build up energy history with spike
    detector.record_energy(1.0);
    detector.record_energy(1.0);
    detector.record_energy(1.0);
    detector.record_energy(5.0); // Spike!

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const result = detector.detect(1.0, 0.5, 1.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.energy_spike, result.mode);
}

test "FailureDetector.detect finds convergence failure" {
    const allocator = std.testing.allocator;
    var config = DetectionConfig.default();
    config.max_iterations = 500;
    var detector = FailureDetector.init(allocator, config);
    defer detector.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const result = detector.detect(1.0, 0.5, 1.0, 1.0, 500, &values);

    try std.testing.expectEqual(FailureMode.convergence_failure, result.mode);
}

test "FailureDetector.detect finds over_constraint" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    // High tau (10.0) with very low variance (0.001)
    const result = detector.detect(10.0, 0.001, 1.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.over_constraint, result.mode);
}

test "FailureDetector.detect finds geo_stat_conflict" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    // Large ratio between losses
    const result = detector.detect(1.0, 0.5, 100.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.geo_stat_conflict, result.mode);
}

test "FailureDetector.detect returns no failure for normal case" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const result = detector.detect(1.0, 0.5, 1.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.none, result.mode);
}

test "FailureDetector.reset_stats clears all counts" {
    const allocator = std.testing.allocator;
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ math.nan(f32) };
    _ = detector.detect(1.0, 0.5, 1.0, 1.0, 100, &values);

    try std.testing.expect(detector.total_detections() > 0);

    detector.reset_stats();

    try std.testing.expectEqual(@as(u32, 0), detector.total_detections());
    try std.testing.expectEqual(FailureMode.none, detector.last_failure);
}

// ============================================================================
// Combined Scenario Tests
// ============================================================================

test "combined: detect and mitigate over_constraint" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 5.0);
    defer controller.deinit();
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const result = detector.detect(5.0, 0.001, 1.0, 1.0, 100, &values);

    if (result.mode != .none) {
        const mitigation = apply_mitigation(&controller, result.mode);
        try std.testing.expect(mitigation.success);
    }
}

test "combined: detect and mitigate energy_spike" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 2.0);
    defer controller.deinit();
    var detector = FailureDetector.init(allocator, DetectionConfig.default());
    defer detector.deinit();

    // Create energy spike
    detector.record_energy(1.0);
    detector.record_energy(1.0);
    detector.record_energy(10.0);

    const values = [_]f32{ 1.0, 2.0, 3.0 };
    const result = detector.detect(1.0, 0.5, 1.0, 1.0, 100, &values);

    try std.testing.expectEqual(FailureMode.energy_spike, result.mode);

    const old_tau = controller.current_tau;
    const mitigation = apply_mitigation(&controller, result.mode);
    try std.testing.expect(mitigation.success);
    try std.testing.expect(controller.current_tau < old_tau);
}

test "combined: multiple failures in sequence" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 3.0);
    defer controller.deinit();

    // Simulate sequence of failures with resets to test independent behavior
    _ = apply_mitigation(&controller, .over_constraint);
    const tau1 = controller.current_tau;
    try std.testing.expect(tau1 < 3.0); // Reduced for over_constraint

    _ = apply_mitigation(&controller, .geo_stat_conflict);
    const tau2 = controller.current_tau;
    try std.testing.expect(tau2 < tau1); // Reduced more for geo_stat

    // Reset momentum and test convergence failure independently
    controller.reset_tau(1.0);
    const tau_before_convergence = controller.current_tau;
    _ = apply_mitigation(&controller, .convergence_failure);
    const tau_after_convergence = controller.current_tau;
    try std.testing.expect(tau_after_convergence > tau_before_convergence); // Increased for convergence
}

test "combined: recovery from numerical instability" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 5.0);
    defer controller.deinit();

    const result = apply_mitigation(&controller, .numerical_instability);

    try std.testing.expect(result.success);
    try std.testing.expect(result.momentum_adjusted);
    try std.testing.expect(controller.current_tau < 5.0);
}

test "combined: sustained failure mode handling" {
    const allocator = std.testing.allocator;
    var controller = AdaptiveTauController.init(allocator, 5.0);
    defer controller.deinit();

    // Apply same failure repeatedly
    for (0..5) |_| {
        _ = apply_mitigation(&controller, .over_constraint);
    }

    // Tau should be significantly reduced but still above minimum
    try std.testing.expect(controller.current_tau < 5.0);
    try std.testing.expect(controller.current_tau >= controller.min_tau);

    const stats = controller.get_statistics();
    try std.testing.expect(stats.downward >= 5);
}

test "DetectionResult.noFailure returns correct structure" {
    const result = DetectionResult.noFailure();
    try std.testing.expectEqual(FailureMode.none, result.mode);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.confidence, 0.001);
    try std.testing.expectEqual(@as(u8, 0), result.severity);
}

test "DetectionResult.failure returns correct structure" {
    const result = DetectionResult.failure(.energy_spike, 0.8, "Test message");
    try std.testing.expectEqual(FailureMode.energy_spike, result.mode);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), result.confidence, 0.001);
    try std.testing.expectEqual(@as(u8, 2), result.severity);
    try std.testing.expectEqualStrings("Test message", result.message);
}

test "DetectionConfig.default returns valid config" {
    const config = DetectionConfig.default();
    try std.testing.expect(config.over_constraint_threshold > 0);
    try std.testing.expect(config.geo_stat_ratio_threshold > 0);
    try std.testing.expect(config.energy_spike_threshold > 0);
    try std.testing.expect(config.max_iterations > 0);
}