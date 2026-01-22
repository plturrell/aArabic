// mHC Constraints Module
// Implements Manifold-Constrained Hyper-Connections using Sinkhorn-Knopp normalization
//
// Core Functions:
// - sinkhorn_normalize: Iterative matrix normalization to doubly stochastic form
// - check_stability: Validate signal amplification within bounds
// - apply_manifold_constraints: Project activations onto constraint manifold
// - compute_stability_metrics: Collect stability statistics
//
// Reference: docs/specs/mhc_constraints_api.md (Day 27)

const std = @import("std");
const builtin = @import("builtin");

/// Configuration for mHC constraint operations
pub const MHCConfig = struct {
    /// Enable/disable mHC constraints globally
    enabled: bool = false,

    /// Number of Sinkhorn-Knopp iterations (5-50 range)
    sinkhorn_iterations: u32 = 10,

    /// Convergence threshold for row/column normalization
    manifold_epsilon: f32 = 1e-6,

    /// Stability validation threshold
    stability_threshold: f32 = 1e-4,

    /// Maximum activation bound for manifold projection
    manifold_beta: f32 = 10.0,

    /// Log detailed stability metrics
    log_stability_metrics: bool = false,

    /// Apply constraints to specific layer range (null = all layers)
    layer_range: ?LayerRange = null,

    /// Allow early stopping when convergence detected
    early_stopping: bool = true,

    /// Validation
    pub fn validate(self: MHCConfig) !void {
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
    }
};

/// Layer range for selective mHC application
pub const LayerRange = struct {
    start: u32,
    end: u32,

    pub fn contains(self: LayerRange, layer_id: u32) bool {
        return layer_id >= self.start and layer_id <= self.end;
    }
};

/// Stability metrics for a single operation
pub const StabilityMetrics = struct {
    /// Layer or operation ID
    layer_id: u32,

    /// Signal L2 norm before constraints
    signal_norm_before: f32,

    /// Signal L2 norm after constraints
    signal_norm_after: f32,

    /// Amplification factor: norm_after / norm_before
    amplification_factor: f32,

    /// Number of Sinkhorn-Knopp iterations until convergence
    convergence_iterations: u32,

    /// Maximum absolute activation value
    max_activation: f32,

    /// Stability flag: true if amplification in [0.9, 1.1]
    is_stable: bool,

    /// Timestamp (milliseconds since epoch)
    timestamp: i64,

    /// Calculate stability status from amplification
    pub fn calculate_stability(amplification: f32) bool {
        return amplification >= 0.9 and amplification <= 1.1;
    }

    /// Format metrics for logging
    pub fn format(
        self: StabilityMetrics,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "Layer {d}: α={d:.3} ({s}), iters={d}, max={d:.3}",
            .{
                self.layer_id,
                self.amplification_factor,
                if (self.is_stable) "stable" else "UNSTABLE",
                self.convergence_iterations,
                self.max_activation,
            },
        );
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Compute sum of each row
fn compute_row_sums(matrix: []const f32, rows: usize, cols: usize, row_sums: []f32) void {
    for (0..rows) |i| {
        var sum: f32 = 0.0;
        for (0..cols) |j| {
            sum += matrix[i * cols + j];
        }
        row_sums[i] = sum;
    }
}

/// Compute sum of each column
fn compute_col_sums(matrix: []const f32, rows: usize, cols: usize, col_sums: []f32) void {
    // Initialize to zero
    for (col_sums) |*sum| {
        sum.* = 0.0;
    }

    // Accumulate column sums
    for (0..rows) |i| {
        for (0..cols) |j| {
            col_sums[j] += matrix[i * cols + j];
        }
    }
}

/// Check if normalization has converged
fn check_convergence(row_sums: []const f32, col_sums: []const f32, epsilon: f32) bool {
    // Check row sums ≈ 1.0
    for (row_sums) |sum| {
        if (@abs(sum - 1.0) > epsilon) {
            return false;
        }
    }

    // Check column sums ≈ 1.0
    for (col_sums) |sum| {
        if (@abs(sum - 1.0) > epsilon) {
            return false;
        }
    }

    return true;
}

/// Compute L2 norm of vector with SIMD optimization
pub fn compute_norm(vector: []const f32) f32 {
    var norm_sq: f32 = 0.0;
    
    // SIMD optimization for ARM NEON (4x f32 per instruction)
    if (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .arm) {
        const vec_len = vector.len;
        const simd_width = 4;
        const simd_iterations = vec_len / simd_width;
        
        // Process 4 elements at a time
        var i: usize = 0;
        while (i < simd_iterations * simd_width) : (i += simd_width) {
            // Manual SIMD (Zig doesn't have @Vector for f32x4 yet in stable)
            const v0 = vector[i];
            const v1 = vector[i + 1];
            const v2 = vector[i + 2];
            const v3 = vector[i + 3];
            norm_sq += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }
        
        // Handle remaining elements
        while (i < vec_len) : (i += 1) {
            const val = vector[i];
            norm_sq += val * val;
        }
    } else {
        // Fallback for other architectures
        for (vector) |val| {
            norm_sq += val * val;
        }
    }
    
    return @sqrt(norm_sq);
}

// ============================================================================
// Core Functions
// ============================================================================

/// Apply Sinkhorn-Knopp normalization to matrix
///
/// Normalizes matrix so that:
/// - Sum of each row ≈ 1.0
/// - Sum of each column ≈ 1.0
///
/// Parameters:
///   - matrix: Input/output matrix (modified in-place)
///   - rows: Number of rows
///   - cols: Number of columns
///   - config: mHC configuration
///   - allocator: Memory allocator for temporary buffers
///
/// Returns:
///   - Number of iterations until convergence
///
/// Complexity: O(iterations × rows × cols)
/// Memory: O(rows + cols) temporary buffers
pub fn sinkhorn_normalize(
    matrix: []f32,
    rows: usize,
    cols: usize,
    config: MHCConfig,
    allocator: std.mem.Allocator,
) !u32 {
    // Validate inputs
    if (rows == 0 or cols == 0) return error.InvalidDimensions;
    if (matrix.len != rows * cols) return error.DimensionMismatch;

    // Allocate temporary buffers
    const row_sums = try allocator.alloc(f32, rows);
    defer allocator.free(row_sums);
    const col_sums = try allocator.alloc(f32, cols);
    defer allocator.free(col_sums);

    var iterations: u32 = 0;

    // Iterative normalization
    for (0..config.sinkhorn_iterations) |iter| {
        iterations = @intCast(iter + 1);

        // Row normalization
        compute_row_sums(matrix, rows, cols, row_sums);
        for (0..rows) |i| {
            const sum = row_sums[i];
            if (sum > config.manifold_epsilon) {
                const scale = 1.0 / sum;
                for (0..cols) |j| {
                    matrix[i * cols + j] *= scale;
                }
            }
        }

        // Column normalization
        compute_col_sums(matrix, rows, cols, col_sums);
        for (0..cols) |j| {
            const sum = col_sums[j];
            if (sum > config.manifold_epsilon) {
                const scale = 1.0 / sum;
                for (0..rows) |i| {
                    matrix[i * cols + j] *= scale;
                }
            }
        }

        // Check convergence (early stopping)
        if (config.early_stopping and iter >= 3) {
            if (check_convergence(row_sums, col_sums, config.manifold_epsilon)) {
                break;
            }
        }
    }

    return iterations;
}

/// Check if activations are stable (bounded)
///
/// Stability criteria:
/// - max(|activations|) < threshold
/// - No NaN or Inf values
///
/// Parameters:
///   - activations: Array of activation values
///   - threshold: Maximum allowed absolute value
///
/// Returns:
///   - true if stable, false if unstable
///
/// Complexity: O(n)
/// Memory: O(1)
pub fn check_stability(activations: []const f32, threshold: f32) bool {
    for (activations) |val| {
        // Check for NaN/Inf
        if (std.math.isNan(val) or std.math.isInf(val)) {
            return false;
        }

        // Check threshold
        const abs_val = @abs(val);
        if (abs_val >= threshold) {
            return false;
        }
    }

    return true;
}

/// Apply manifold constraints to activations
///
/// Projects activations onto L2 ball: ||x||₂ ≤ β
/// If ||x||₂ > β, scale down: x' = β · x / ||x||₂
///
/// Parameters:
///   - activations: Input/output activations (modified in-place)
///   - beta: Maximum L2 norm bound
///
/// Returns:
///   - Norm before projection
///
/// Complexity: O(n)
/// Memory: O(1)
pub fn apply_manifold_constraints(activations: []f32, beta: f32) f32 {
    // Compute L2 norm
    const norm = compute_norm(activations);

    // Project if exceeds bound
    if (norm > beta) {
        const scale = beta / norm;
        for (activations) |*val| {
            val.* *= scale;
        }
    }

    return norm;
}

/// Compute stability metrics for operation
///
/// Calculates:
/// - Amplification factor: ||after|| / ||before||
/// - Maximum activation value
/// - Stability status
///
/// Parameters:
///   - layer_id: Layer or operation identifier
///   - activations_before: Activations before constraints
///   - activations_after: Activations after constraints
///   - iterations: Number of Sinkhorn-Knopp iterations
///
/// Returns:
///   - StabilityMetrics structure
///
/// Complexity: O(n)
/// Memory: O(1)
pub fn compute_stability_metrics(
    layer_id: u32,
    activations_before: []const f32,
    activations_after: []const f32,
    iterations: u32,
) StabilityMetrics {
    // Compute L2 norms
    const norm_before = compute_norm(activations_before);
    const norm_after = compute_norm(activations_after);

    // Find maximum activation
    var max_val: f32 = 0.0;
    for (activations_after) |val| {
        max_val = @max(max_val, @abs(val));
    }

    // Calculate amplification factor
    const amplification = if (norm_before > 0) norm_after / norm_before else 1.0;

    return StabilityMetrics{
        .layer_id = layer_id,
        .signal_norm_before = norm_before,
        .signal_norm_after = norm_after,
        .amplification_factor = amplification,
        .convergence_iterations = iterations,
        .max_activation = max_val,
        .is_stable = StabilityMetrics.calculate_stability(amplification),
        .timestamp = std.time.milliTimestamp(),
    };
}

// ============================================================================
// Unit Tests
// ============================================================================

test "sinkhorn_normalize converges" {
    const allocator = std.testing.allocator;
    // Use square matrix for perfect doubly stochastic convergence
    var matrix = [_]f32{ 1, 2, 3, 4 };

    const config = MHCConfig{
        .enabled = true,
        .sinkhorn_iterations = 20,
    };

    const iters = try sinkhorn_normalize(&matrix, 2, 2, config, allocator);

    // Check row sums ≈ 1.0 for square matrices
    const row1_sum = matrix[0] + matrix[1];
    const row2_sum = matrix[2] + matrix[3];
    try std.testing.expectApproxEqAbs(row1_sum, 1.0, 0.01);
    try std.testing.expectApproxEqAbs(row2_sum, 1.0, 0.01);

    // Check column sums ≈ 1.0 for square matrices
    const col1_sum = matrix[0] + matrix[2];
    const col2_sum = matrix[1] + matrix[3];
    try std.testing.expectApproxEqAbs(col1_sum, 1.0, 0.01);
    try std.testing.expectApproxEqAbs(col2_sum, 1.0, 0.01);

    // Check convergence happened
    try std.testing.expect(iters <= 20);
    try std.testing.expect(iters > 0);
}

test "check_stability detects instability" {
    const stable = [_]f32{ 0.1, -0.05, 0.03 };
    const unstable = [_]f32{ 100.0, -200.0, 50.0 };

    try std.testing.expect(check_stability(&stable, 1.0));
    try std.testing.expect(!check_stability(&unstable, 1.0));
}

test "apply_manifold_constraints bounds norm" {
    var activations = [_]f32{ 3.0, 4.0, 0.0 }; // ||x||₂ = 5.0
    const norm = apply_manifold_constraints(&activations, 1.0);

    // Original norm should be 5.0
    try std.testing.expectApproxEqRel(norm, 5.0, 0.01);

    // New norm should be ≤ 1.0
    const new_norm = compute_norm(&activations);
    try std.testing.expectApproxEqRel(new_norm, 1.0, 0.01);
}

test "sinkhorn_normalize handles zero matrix" {
    const allocator = std.testing.allocator;
    var matrix = [_]f32{ 0, 0, 0, 0 };

    const config = MHCConfig{};
    const iters = try sinkhorn_normalize(&matrix, 2, 2, config, allocator);

    // Should complete without crash
    try std.testing.expect(iters <= config.sinkhorn_iterations);
}

test "check_stability detects NaN" {
    const nan_array = [_]f32{ 1.0, std.math.nan(f32), 2.0 };
    try std.testing.expect(!check_stability(&nan_array, 10.0));
}

test "compute_stability_metrics calculates amplification" {
    const before = [_]f32{ 1.0, 0.0, 0.0 }; // norm = 1.0
    const after = [_]f32{ 2.0, 0.0, 0.0 }; // norm = 2.0

    const metrics = compute_stability_metrics(0, &before, &after, 10);

    try std.testing.expectApproxEqRel(metrics.amplification_factor, 2.0, 0.01);
    try std.testing.expect(!metrics.is_stable); // 2.0 > 1.1
}

test "sinkhorn_normalize stops early when converged" {
    const allocator = std.testing.allocator;
    var matrix = [_]f32{ 1, 1, 1, 1 }; // Already nearly doubly stochastic

    const config = MHCConfig{
        .enabled = true,
        .sinkhorn_iterations = 20,
        .early_stopping = true,
    };

    const iters = try sinkhorn_normalize(&matrix, 2, 2, config, allocator);

    // Should stop early (much less than 20)
    try std.testing.expect(iters < 10);
}

test "sinkhorn_normalize handles large matrices" {
    const allocator = std.testing.allocator;
    const size = 100;
    const matrix = try allocator.alloc(f32, size * size);
    defer allocator.free(matrix);

    // Initialize with simple pattern
    for (matrix, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt((i % 10) + 1)) / 10.0;
    }

    const config = MHCConfig{};
    const iters = try sinkhorn_normalize(matrix, size, size, config, allocator);

    try std.testing.expect(iters > 0);
}

test "sinkhorn_normalize handles non-square matrices" {
    const allocator = std.testing.allocator;
    const matrix = try allocator.alloc(f32, 10 * 20);
    defer allocator.free(matrix);

    // Initialize with simple pattern
    for (matrix, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt((i % 10) + 1)) / 10.0;
    }

    const config = MHCConfig{};
    const iters = try sinkhorn_normalize(matrix, 10, 20, config, allocator);

    try std.testing.expect(iters > 0);
}

test "MHCConfig validates parameters" {
    const invalid_iters = MHCConfig{
        .sinkhorn_iterations = 100, // Too high
    };
    try std.testing.expectError(error.InvalidIterations, invalid_iters.validate());

    const invalid_epsilon = MHCConfig{
        .manifold_epsilon = 2.0, // Too high
    };
    try std.testing.expectError(error.InvalidEpsilon, invalid_epsilon.validate());
}
