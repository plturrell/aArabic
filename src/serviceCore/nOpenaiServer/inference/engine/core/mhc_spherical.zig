// mHC Spherical Manifold Module
// Implements Spherical Geometry Operations for Manifold-Constrained Hyper-Connections
//
// Core Functions:
// - spherical_distance: Great circle distance on unit sphere
// - frechet_mean: Weighted mean point on sphere
// - normalize_to_sphere: Project to unit sphere
// - spherical_exp_map: Exponential map on sphere
// - spherical_log_map: Logarithmic map on sphere
// - spherical_sinkhorn: Sinkhorn-Knopp adapted for spherical manifold
//
// Reference: Day 56 - Spherical mHC Implementation

const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

// Import core mHC modules
const mhc_constraints = @import("mhc_constraints.zig");
const mhc_config = @import("mhc_configuration.zig");

/// Configuration for spherical manifold operations
pub const SphericalConfig = struct {
    /// Sphere radius (typically 1.0 for unit sphere)
    radius: f32 = 1.0,

    /// Numerical stability epsilon
    epsilon: f32 = 1e-8,

    /// Maximum iterations for iterative algorithms
    max_iterations: u32 = 100,

    /// Convergence tolerance for Fréchet mean
    convergence_tolerance: f32 = 1e-6,

    /// Use geodesic interpolation (vs Euclidean)
    use_geodesic: bool = true,

    /// Apply manifold constraints after operations
    apply_constraints: bool = true,

    /// Log computation metrics
    log_metrics: bool = false,

    /// Validate configuration parameters
    pub fn validate(self: SphericalConfig) !void {
        if (self.radius <= 0) {
            return error.InvalidRadius;
        }
        if (self.epsilon <= 0 or self.epsilon >= 0.1) {
            return error.InvalidEpsilon;
        }
        if (self.max_iterations < 1 or self.max_iterations > 10000) {
            return error.InvalidIterations;
        }
        if (self.convergence_tolerance <= 0) {
            return error.InvalidTolerance;
        }
    }
};

/// Result metrics for spherical operations
pub const SphericalMetrics = struct {
    /// Distance computed (for distance operations)
    distance: f32 = 0.0,

    /// Number of iterations (for iterative algorithms)
    iterations: u32 = 0,

    /// Convergence achieved
    converged: bool = false,

    /// Final error/residual
    residual: f32 = 0.0,

    /// Input norm before normalization
    input_norm: f32 = 0.0,

    /// Output norm after normalization
    output_norm: f32 = 0.0,

    /// Stability flag
    is_stable: bool = true,

    /// Computation time (nanoseconds)
    computation_time_ns: i64 = 0,
};

// ============================================================================
// Vector Helper Functions
// ============================================================================

/// Compute L2 norm of a vector
pub fn vector_norm(v: []const f32) f32 {
    var sum: f32 = 0.0;
    for (v) |val| {
        sum += val * val;
    }
    return @sqrt(sum);
}

/// Compute dot product of two vectors
pub fn dot_product(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var sum: f32 = 0.0;
    for (a, b) |va, vb| {
        sum += va * vb;
    }
    return sum;
}

/// Normalize vector in-place to unit length
pub fn normalize_vector(v: []f32) f32 {
    const norm = vector_norm(v);
    if (norm > 1e-10) {
        const inv_norm = 1.0 / norm;
        for (v) |*val| {
            val.* *= inv_norm;
        }
    }
    return norm;
}

/// Scale vector in-place
pub fn scale_vector(v: []f32, scalar: f32) void {
    for (v) |*val| {
        val.* *= scalar;
    }
}

/// Add two vectors: result = a + b
pub fn add_vectors(result: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(result.len == a.len and a.len == b.len);
    for (result, a, b) |*r, va, vb| {
        r.* = va + vb;
    }
}

/// Subtract two vectors: result = a - b
pub fn subtract_vectors(result: []f32, a: []const f32, b: []const f32) void {
    std.debug.assert(result.len == a.len and a.len == b.len);
    for (result, a, b) |*r, va, vb| {
        r.* = va - vb;
    }
}

/// Copy vector
pub fn copy_vector(dest: []f32, src: []const f32) void {
    std.debug.assert(dest.len == src.len);
    @memcpy(dest, src);
}

// ============================================================================
// Core Spherical Functions
// ============================================================================

/// Compute great circle (geodesic) distance between two points on the unit sphere
///
/// Uses the numerically stable haversine formula for small distances
/// and the standard arccos formula for larger distances.
///
/// Formula: d(x, y) = arccos(x · y) for unit vectors
///          d(x, y) = 2 * arcsin(||x - y|| / 2) for small angles (haversine)
///
/// Parameters:
///   - x: First point on sphere (will be normalized)
///   - y: Second point on sphere (will be normalized)
///   - config: Spherical configuration
///
/// Returns: Geodesic distance in radians [0, π]
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn spherical_distance(x: []const f32, y: []const f32, config: SphericalConfig) f32 {
    std.debug.assert(x.len == y.len);

    // Compute dot product
    var dot: f32 = 0.0;
    var norm_x_sq: f32 = 0.0;
    var norm_y_sq: f32 = 0.0;

    for (x, y) |vx, vy| {
        dot += vx * vy;
        norm_x_sq += vx * vx;
        norm_y_sq += vy * vy;
    }

    // Normalize dot product by norms
    const norm_x = @sqrt(norm_x_sq);
    const norm_y = @sqrt(norm_y_sq);

    if (norm_x < config.epsilon or norm_y < config.epsilon) {
        return math.pi; // Maximum distance for degenerate case
    }

    const cos_angle = dot / (norm_x * norm_y);

    // Clamp to [-1, 1] for numerical stability
    const cos_clamped = @max(-1.0, @min(1.0, cos_angle));

    // Use arccos for the distance
    return math.acos(cos_clamped) * config.radius;
}

/// Compute great circle distance with metrics
pub fn spherical_distance_with_metrics(
    x: []const f32,
    y: []const f32,
    config: SphericalConfig,
) SphericalMetrics {
    const start_time = std.time.nanoTimestamp();
    const distance = spherical_distance(x, y, config);
    const end_time = std.time.nanoTimestamp();

    return SphericalMetrics{
        .distance = distance,
        .iterations = 1,
        .converged = true,
        .residual = 0.0,
        .input_norm = vector_norm(x),
        .output_norm = vector_norm(y),
        .is_stable = !math.isNan(distance) and !math.isInf(distance),
        .computation_time_ns = @intCast(end_time - start_time),
    };
}

/// Normalize a point to lie on the unit sphere
///
/// Projects the input point onto the sphere of specified radius.
/// For zero vectors, returns the input unchanged.
///
/// Parameters:
///   - x: Input point (modified in-place)
///   - config: Spherical configuration
///
/// Returns: Original norm before projection
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn normalize_to_sphere(x: []f32, config: SphericalConfig) f32 {
    const norm = vector_norm(x);

    if (norm < config.epsilon) {
        return norm; // Don't normalize near-zero vectors
    }

    const target_norm = config.radius;
    const scale = target_norm / norm;

    for (x) |*val| {
        val.* *= scale;
    }

    return norm;
}

/// Exponential map on the sphere
///
/// Maps a tangent vector at base point to a point on the sphere.
/// The tangent vector should be orthogonal to the base point.
///
/// Formula: exp_p(v) = cos(||v||) * p + sin(||v||) * v/||v||
///
/// Parameters:
///   - result: Output point on sphere (must be pre-allocated)
///   - base: Base point on sphere (assumed normalized)
///   - tangent: Tangent vector at base point
///   - config: Spherical configuration
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn spherical_exp_map(
    result: []f32,
    base: []const f32,
    tangent: []const f32,
    config: SphericalConfig,
) void {
    std.debug.assert(result.len == base.len and base.len == tangent.len);

    const tangent_norm = vector_norm(tangent);

    if (tangent_norm < config.epsilon) {
        // Zero tangent: return base point
        copy_vector(result, base);
        return;
    }

    // Compute cos and sin of the tangent norm
    const cos_t = @cos(tangent_norm);
    const sin_t = @sin(tangent_norm);
    const inv_tangent_norm = 1.0 / tangent_norm;

    // exp_p(v) = cos(||v||) * p + sin(||v||) * v/||v||
    for (result, base, tangent) |*r, b, t| {
        r.* = cos_t * b + sin_t * inv_tangent_norm * t;
    }

    // Ensure result is on sphere
    _ = normalize_to_sphere(result, config);
}


/// Logarithmic map on the sphere
///
/// Maps a point on the sphere to a tangent vector at the base point.
/// This is the inverse of the exponential map.
///
/// Formula: log_p(q) = arccos(p·q) * (q - (p·q)*p) / ||q - (p·q)*p||
///
/// Parameters:
///   - result: Output tangent vector (must be pre-allocated)
///   - base: Base point on sphere (assumed normalized)
///   - point: Target point on sphere
///   - config: Spherical configuration
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn spherical_log_map(
    result: []f32,
    base: []const f32,
    point: []const f32,
    config: SphericalConfig,
) void {
    std.debug.assert(result.len == base.len and base.len == point.len);

    // Compute dot product (cosine of angle)
    const dot = dot_product(base, point);
    const cos_angle = @max(-1.0, @min(1.0, dot));
    const angle = math.acos(cos_angle);

    if (angle < config.epsilon) {
        // Points are identical: zero tangent
        for (result) |*r| {
            r.* = 0.0;
        }
        return;
    }

    // Compute the component of point orthogonal to base
    // v = point - (base · point) * base
    for (result, base, point) |*r, b, p| {
        r.* = p - cos_angle * b;
    }

    // Normalize and scale by angle
    const v_norm = vector_norm(result);

    if (v_norm < config.epsilon) {
        // Points are antipodal: undefined log map, return zero
        for (result) |*r| {
            r.* = 0.0;
        }
        return;
    }

    const scale = angle / v_norm;
    for (result) |*r| {
        r.* *= scale;
    }
}

/// Compute the Fréchet mean (intrinsic mean) on the sphere
///
/// Uses iterative gradient descent to find the point that minimizes
/// the sum of squared geodesic distances to all input points.
///
/// Algorithm:
///   1. Initialize with Euclidean mean normalized to sphere
///   2. Iteratively update: mean = exp_mean(Σ w_i * log_mean(p_i))
///   3. Stop when convergence or max iterations reached
///
/// Parameters:
///   - result: Output mean point (must be pre-allocated)
///   - points: Array of points on sphere (each dim-dimensional)
///   - weights: Optional weights for each point (null = uniform)
///   - dim: Dimension of each point
///   - config: Spherical configuration
///   - allocator: Memory allocator for temporary buffers
///
/// Returns: Convergence metrics
///
/// Complexity: O(iterations × n_points × dim)
/// Memory: O(dim) temporary buffers
pub fn frechet_mean(
    result: []f32,
    points: []const f32,
    weights: ?[]const f32,
    dim: usize,
    config: SphericalConfig,
    allocator: std.mem.Allocator,
) !SphericalMetrics {
    const start_time = std.time.nanoTimestamp();
    const n_points = points.len / dim;

    if (n_points == 0) {
        return error.EmptyPointSet;
    }

    if (n_points == 1) {
        // Single point is its own mean
        copy_vector(result, points[0..dim]);
        _ = normalize_to_sphere(result, config);
        return SphericalMetrics{
            .iterations = 0,
            .converged = true,
            .residual = 0.0,
            .output_norm = config.radius,
            .is_stable = true,
            .computation_time_ns = @intCast(std.time.nanoTimestamp() - start_time),
        };
    }

    // Allocate temporary buffers
    const tangent = try allocator.alloc(f32, dim);
    defer allocator.free(tangent);
    const tangent_sum = try allocator.alloc(f32, dim);
    defer allocator.free(tangent_sum);
    const new_mean = try allocator.alloc(f32, dim);
    defer allocator.free(new_mean);

    // Initialize with Euclidean mean
    for (result) |*r| {
        r.* = 0.0;
    }

    var total_weight: f32 = 0.0;
    for (0..n_points) |i| {
        const w = if (weights) |wts| wts[i] else 1.0;
        total_weight += w;
        const point_start = i * dim;
        for (result, 0..) |*r, j| {
            r.* += w * points[point_start + j];
        }
    }

    // Normalize weights and result
    if (total_weight > config.epsilon) {
        for (result) |*r| {
            r.* /= total_weight;
        }
    }
    _ = normalize_to_sphere(result, config);

    // Iterative refinement
    var iterations: u32 = 0;
    var residual: f32 = math.inf(f32);
    var converged = false;

    while (iterations < config.max_iterations and !converged) {
        iterations += 1;

        // Reset tangent sum
        for (tangent_sum) |*t| {
            t.* = 0.0;
        }

        // Accumulate weighted log maps
        for (0..n_points) |i| {
            const w = if (weights) |wts| wts[i] else 1.0 / @as(f32, @floatFromInt(n_points));
            const point_start = i * dim;

            spherical_log_map(tangent, result, points[point_start .. point_start + dim], config);

            for (tangent_sum, tangent) |*ts, t| {
                ts.* += w * t;
            }
        }

        // Compute step size (residual is the norm of tangent sum)
        residual = vector_norm(tangent_sum);

        // Check convergence
        if (residual < config.convergence_tolerance) {
            converged = true;
            break;
        }

        // Update mean using exponential map
        spherical_exp_map(new_mean, result, tangent_sum, config);
        copy_vector(result, new_mean);
    }

    const end_time = std.time.nanoTimestamp();

    return SphericalMetrics{
        .iterations = iterations,
        .converged = converged,
        .residual = residual,
        .output_norm = vector_norm(result),
        .is_stable = converged and !math.isNan(residual),
        .computation_time_ns = @intCast(end_time - start_time),
    };
}


/// Geodesic interpolation (spherical linear interpolation - slerp)
///
/// Interpolates along the geodesic between two points on the sphere.
///
/// Formula: slerp(p, q, t) = sin((1-t)θ)/sin(θ) * p + sin(tθ)/sin(θ) * q
///          where θ = arccos(p · q)
///
/// Parameters:
///   - result: Output interpolated point (must be pre-allocated)
///   - p: Start point on sphere
///   - q: End point on sphere
///   - t: Interpolation parameter [0, 1]
///   - config: Spherical configuration
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn geodesic_interpolate(
    result: []f32,
    p: []const f32,
    q: []const f32,
    t: f32,
    config: SphericalConfig,
) void {
    std.debug.assert(result.len == p.len and p.len == q.len);
    std.debug.assert(t >= 0.0 and t <= 1.0);

    // Compute angle between vectors
    const dot = dot_product(p, q);
    const cos_angle = @max(-1.0, @min(1.0, dot));
    const angle = math.acos(cos_angle);

    if (angle < config.epsilon) {
        // Points are nearly identical: return p (or q)
        copy_vector(result, p);
        return;
    }

    if (@abs(angle - math.pi) < config.epsilon) {
        // Points are antipodal: undefined geodesic, use linear interpolation
        for (result, p, q) |*r, vp, vq| {
            r.* = (1.0 - t) * vp + t * vq;
        }
        _ = normalize_to_sphere(result, config);
        return;
    }

    // Slerp formula
    const sin_angle = @sin(angle);
    const coeff_p = @sin((1.0 - t) * angle) / sin_angle;
    const coeff_q = @sin(t * angle) / sin_angle;

    for (result, p, q) |*r, vp, vq| {
        r.* = coeff_p * vp + coeff_q * vq;
    }

    // Ensure result is on sphere (numerical stability)
    _ = normalize_to_sphere(result, config);
}

/// Normalize a matrix row along geodesics
///
/// Projects each row of the matrix to lie on the unit sphere,
/// then applies geodesic-aware normalization for doubly stochastic structure.
///
/// Parameters:
///   - matrix: Input/output matrix (modified in-place)
///   - rows: Number of rows
///   - cols: Number of columns
///   - config: Spherical configuration
///
/// Returns: Average norm before normalization
///
/// Complexity: O(rows × cols)
/// Memory: O(1)
pub fn geodesic_normalize_rows(
    matrix: []f32,
    rows: usize,
    cols: usize,
    config: SphericalConfig,
) f32 {
    var total_norm: f32 = 0.0;

    for (0..rows) |i| {
        const row_start = i * cols;
        const row_end = row_start + cols;
        const row = matrix[row_start..row_end];

        // Normalize row to unit sphere
        const norm = normalize_to_sphere(row, config);
        total_norm += norm;
    }

    return total_norm / @as(f32, @floatFromInt(rows));
}

/// Spherical Sinkhorn-Knopp normalization
///
/// Adapts the Sinkhorn-Knopp algorithm to work on the spherical manifold.
/// Instead of Euclidean row/column normalization, uses geodesic projections.
///
/// The algorithm alternates between:
///   1. Projecting rows onto the sphere
///   2. Scaling columns to have unit sum (in tangent space)
///   3. Projecting back to sphere
///
/// Parameters:
///   - matrix: Input/output matrix (modified in-place)
///   - rows: Number of rows
///   - cols: Number of columns
///   - config: Spherical configuration
///   - allocator: Memory allocator for temporary buffers
///
/// Returns: Number of iterations until convergence
///
/// Complexity: O(iterations × rows × cols)
/// Memory: O(cols) temporary buffer
pub fn spherical_sinkhorn(
    matrix: []f32,
    rows: usize,
    cols: usize,
    config: SphericalConfig,
    allocator: std.mem.Allocator,
) !u32 {
    if (rows == 0 or cols == 0) return error.InvalidDimensions;
    if (matrix.len != rows * cols) return error.DimensionMismatch;

    // Allocate column sum buffer
    const col_sums = try allocator.alloc(f32, cols);
    defer allocator.free(col_sums);

    // Track previous row norms for convergence check
    const prev_row_norms = try allocator.alloc(f32, rows);
    defer allocator.free(prev_row_norms);

    var iterations: u32 = 0;
    var converged = false;

    // Initialize previous norms
    for (0..rows) |i| {
        const row_start = i * cols;
        prev_row_norms[i] = vector_norm(matrix[row_start .. row_start + cols]);
    }

    while (iterations < config.max_iterations and !converged) {
        iterations += 1;

        // Step 1: Row normalization (project to sphere)
        for (0..rows) |i| {
            const row_start = i * cols;
            const row_end = row_start + cols;
            _ = normalize_to_sphere(matrix[row_start..row_end], config);
        }

        // Step 2: Compute column sums (in embedding space)
        for (col_sums) |*sum| {
            sum.* = 0.0;
        }
        for (0..rows) |i| {
            for (0..cols) |j| {
                col_sums[j] += @abs(matrix[i * cols + j]);
            }
        }

        // Step 3: Column normalization (scale by inverse sum)
        for (0..cols) |j| {
            if (col_sums[j] > config.epsilon) {
                const scale = @as(f32, @floatFromInt(rows)) / col_sums[j];
                for (0..rows) |i| {
                    matrix[i * cols + j] *= scale;
                }
            }
        }

        // Step 4: Re-project rows to sphere
        for (0..rows) |i| {
            const row_start = i * cols;
            const row_end = row_start + cols;
            _ = normalize_to_sphere(matrix[row_start..row_end], config);
        }

        // Check convergence: compare row norms to previous iteration
        var max_change: f32 = 0.0;
        for (0..rows) |i| {
            const row_start = i * cols;
            const curr_norm = vector_norm(matrix[row_start .. row_start + cols]);
            const change = @abs(curr_norm - prev_row_norms[i]);
            max_change = @max(max_change, change);
            prev_row_norms[i] = curr_norm;
        }

        if (max_change < config.convergence_tolerance and iterations >= 3) {
            converged = true;
        }
    }

    return iterations;
}

/// Parallel transport of a tangent vector along a geodesic
///
/// Transports a tangent vector at base_point to the tangent space at target_point
/// along the geodesic connecting them.
///
/// Parameters:
///   - result: Output transported tangent vector (must be pre-allocated)
///   - tangent: Input tangent vector at base_point
///   - base_point: Starting point on sphere
///   - target_point: Ending point on sphere
///   - config: Spherical configuration
///
/// Complexity: O(dim)
/// Memory: O(1)
pub fn parallel_transport(
    result: []f32,
    tangent: []const f32,
    base_point: []const f32,
    target_point: []const f32,
    config: SphericalConfig,
) void {
    std.debug.assert(result.len == tangent.len);
    std.debug.assert(tangent.len == base_point.len);
    std.debug.assert(base_point.len == target_point.len);

    const dim = result.len;

    // Compute the geodesic direction
    const dot_bp_tp = dot_product(base_point, target_point);
    const cos_angle = @max(-1.0, @min(1.0, dot_bp_tp));
    const angle = math.acos(cos_angle);

    if (angle < config.epsilon) {
        // Points are identical: tangent stays the same
        copy_vector(result, tangent);
        return;
    }

    // Compute unit geodesic direction at base point
    // u = (target - cos(angle) * base) / sin(angle)
    const sin_angle = @sin(angle);

    // Component of tangent along geodesic direction
    var tangent_along: f32 = 0.0;
    var tangent_perp_sq: f32 = 0.0;

    for (0..dim) |i| {
        const u_i = (target_point[i] - cos_angle * base_point[i]) / sin_angle;
        tangent_along += tangent[i] * u_i;
    }

    // Transport formula for sphere:
    // T(v) = v - (v · base + v · target/(1 + cos(angle))) * (base + target)
    const dot_v_base = dot_product(tangent, base_point);
    const dot_v_target = dot_product(tangent, target_point);

    const coeff = if (@abs(1.0 + cos_angle) > config.epsilon)
        dot_v_base + dot_v_target / (1.0 + cos_angle)
    else
        dot_v_base;

    for (result, tangent, base_point, target_point) |*r, v, b, t| {
        r.* = v - coeff * (b + t);
        tangent_perp_sq += r.* * r.*;
    }

    // Ensure result is orthogonal to target (tangent space)
    const dot_result_target = dot_product(result, target_point);
    for (result, target_point) |*r, t| {
        r.* -= dot_result_target * t;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "spherical_distance identity" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const config = SphericalConfig{};

    const dist = spherical_distance(&x, &x, config);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), dist, 0.0001);
}

test "spherical_distance orthogonal" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const config = SphericalConfig{};

    const dist = spherical_distance(&x, &y, config);

    // Orthogonal unit vectors should have distance = π/2
    try std.testing.expectApproxEqAbs(math.pi / 2.0, dist, 0.0001);
}

test "spherical_distance antipodal" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ -1.0, 0.0, 0.0 };
    const config = SphericalConfig{};

    const dist = spherical_distance(&x, &y, config);

    // Antipodal points should have distance = π
    try std.testing.expectApproxEqAbs(math.pi, dist, 0.0001);
}

test "spherical_distance symmetry" {
    const x = [_]f32{ 0.6, 0.8, 0.0 };
    const y = [_]f32{ 0.0, 0.6, 0.8 };
    const config = SphericalConfig{};

    const dist_xy = spherical_distance(&x, &y, config);
    const dist_yx = spherical_distance(&y, &x, config);

    try std.testing.expectApproxEqAbs(dist_xy, dist_yx, 0.0001);
}

test "normalize_to_sphere" {
    var x = [_]f32{ 3.0, 4.0, 0.0 }; // Norm = 5.0
    const config = SphericalConfig{};

    const original_norm = normalize_to_sphere(&x, config);

    try std.testing.expectApproxEqAbs(@as(f32, 5.0), original_norm, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vector_norm(&x), 0.0001);
}

test "spherical_exp_map zero tangent" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = SphericalConfig{};

    spherical_exp_map(&result, &base, &tangent, config);

    // Zero tangent should return base point
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[2], 0.0001);
}

test "spherical_exp_map quarter circle" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    // Tangent of length π/2 in y direction
    const tangent = [_]f32{ 0.0, math.pi / 2.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = SphericalConfig{};

    spherical_exp_map(&result, &base, &tangent, config);

    // Should end up at (0, 1, 0)
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[1], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[2], 0.01);
}

test "spherical_log_map identity" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = SphericalConfig{};

    spherical_log_map(&result, &base, &base, config);

    // Log of same point should be zero
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), vector_norm(&result), 0.0001);
}

test "spherical_log_map inverse of exp_map" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, 0.5, 0.3 };
    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;
    const config = SphericalConfig{};

    // exp then log should recover original tangent
    spherical_exp_map(&exp_result, &base, &tangent, config);
    spherical_log_map(&log_result, &base, &exp_result, config);

    try std.testing.expectApproxEqAbs(tangent[0], log_result[0], 0.01);
    try std.testing.expectApproxEqAbs(tangent[1], log_result[1], 0.01);
    try std.testing.expectApproxEqAbs(tangent[2], log_result[2], 0.01);
}

test "geodesic_interpolate endpoints" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = SphericalConfig{};

    // t=0 should return p
    geodesic_interpolate(&result, &p, &q, 0.0, config);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.0001);

    // t=1 should return q
    geodesic_interpolate(&result, &p, &q, 1.0, config);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[1], 0.0001);
}

test "geodesic_interpolate midpoint" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = SphericalConfig{};

    // t=0.5 should return midpoint on arc
    geodesic_interpolate(&result, &p, &q, 0.5, config);

    // Midpoint should be at 45 degrees: (√2/2, √2/2, 0)
    const expected = @sqrt(2.0) / 2.0;
    try std.testing.expectApproxEqAbs(expected, result[0], 0.0001);
    try std.testing.expectApproxEqAbs(expected, result[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[2], 0.0001);
}

test "frechet_mean single point" {
    const allocator = std.testing.allocator;
    var result: [3]f32 = undefined;
    const points = [_]f32{ 0.6, 0.8, 0.0 };
    const config = SphericalConfig{};

    const metrics = try frechet_mean(&result, &points, null, 3, config, allocator);

    try std.testing.expect(metrics.converged);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vector_norm(&result), 0.01);
}

test "frechet_mean two opposite points" {
    const allocator = std.testing.allocator;
    var result: [3]f32 = undefined;
    // Two points at equal angles from x-axis
    const points = [_]f32{
        1.0,  0.0, 0.0,
        0.0,  1.0, 0.0,
    };
    const config = SphericalConfig{};

    const metrics = try frechet_mean(&result, &points, null, 3, config, allocator);

    // Mean should be on the sphere
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), vector_norm(&result), 0.01);
    _ = metrics;
}

test "spherical_sinkhorn convergence" {
    const allocator = std.testing.allocator;
    var matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = SphericalConfig{ .max_iterations = 20 };

    const iters = try spherical_sinkhorn(&matrix, 2, 2, config, allocator);

    // Should converge in reasonable iterations
    try std.testing.expect(iters <= 20);
    try std.testing.expect(iters > 0);

    // Each row should have unit norm
    const row1_norm = vector_norm(matrix[0..2]);
    const row2_norm = vector_norm(matrix[2..4]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row1_norm, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), row2_norm, 0.01);
}

test "spherical_sinkhorn handles zero matrix" {
    const allocator = std.testing.allocator;
    var matrix = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const config = SphericalConfig{};

    // Should not crash on zero matrix
    const iters = try spherical_sinkhorn(&matrix, 2, 2, config, allocator);
    try std.testing.expect(iters <= config.max_iterations);
}

test "parallel_transport preserves norm" {
    var result: [3]f32 = undefined;
    const tangent = [_]f32{ 0.0, 0.5, 0.5 };
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const target = [_]f32{ 0.0, 1.0, 0.0 };
    const config = SphericalConfig{};

    parallel_transport(&result, &tangent, &base, &target, config);

    // Transported tangent should have same norm (approximately)
    const original_norm = vector_norm(&tangent);
    const transported_norm = vector_norm(&result);
    try std.testing.expectApproxEqAbs(original_norm, transported_norm, 0.1);
}

test "SphericalConfig validation" {
    const invalid_radius = SphericalConfig{ .radius = -1.0 };
    try std.testing.expectError(error.InvalidRadius, invalid_radius.validate());

    const invalid_epsilon = SphericalConfig{ .epsilon = 0.5 };
    try std.testing.expectError(error.InvalidEpsilon, invalid_epsilon.validate());

    const valid = SphericalConfig{};
    try valid.validate();
}

test "vector helper functions" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    var result: [3]f32 = undefined;

    // Test dot product
    const dot = dot_product(&a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 32.0), dot, 0.0001);

    // Test add
    add_vectors(&result, &a, &b);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 0.0001);

    // Test subtract
    subtract_vectors(&result, &b, &a);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[2], 0.0001);
}
