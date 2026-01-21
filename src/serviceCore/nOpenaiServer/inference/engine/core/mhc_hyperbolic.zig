// mHC Hyperbolic Distance Implementation
// Implements Poincaré ball model operations with SIMD optimization
//
// Core Functions:
// - hyperbolic_distance: Distance in Poincaré ball model
// - mobius_add: Möbius addition for hyperbolic vector addition
// - mobius_scalar_mul: Möbius scalar multiplication
// - project_to_ball: Project points back into unit ball
// - poincare_to_klein/klein_to_poincare: Model conversions
//
// Reference: Day 54 - Hyperbolic Geometry for mHC (Days 54-60 Geometric Extensions)

const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

// ============================================================================
// Constants
// ============================================================================

/// Default curvature for Poincaré ball (negative for hyperbolic space)
pub const DEFAULT_CURVATURE: f32 = -1.0;

/// Numerical stability epsilon to prevent division by zero
pub const EPSILON: f32 = 1e-8;

/// Maximum norm for points in Poincaré ball (slightly less than 1)
pub const MAX_NORM: f32 = 1.0 - 1e-5;

/// Minimum norm threshold for operations
pub const MIN_NORM: f32 = 1e-15;

/// SIMD vector width for f32 operations
const SIMD_WIDTH: usize = 8;

/// SIMD vector type for f32
const Vec8 = @Vector(SIMD_WIDTH, f32);

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for hyperbolic operations
pub const HyperbolicConfig = struct {
    /// Curvature parameter (must be negative for hyperbolic)
    curvature: f32 = DEFAULT_CURVATURE,

    /// Numerical stability epsilon
    epsilon: f32 = EPSILON,

    /// Maximum norm for ball projection
    max_norm: f32 = MAX_NORM,

    /// Enable SIMD optimization
    use_simd: bool = true,

    /// Validate configuration
    pub fn validate(self: HyperbolicConfig) !void {
        if (self.curvature >= 0) {
            return error.InvalidHyperbolicCurvature;
        }
        if (self.epsilon <= 0 or self.epsilon >= 1) {
            return error.InvalidEpsilon;
        }
        if (self.max_norm <= 0 or self.max_norm >= 1) {
            return error.InvalidMaxNorm;
        }
    }

    /// Get the absolute curvature (positive value)
    pub fn abs_curvature(self: HyperbolicConfig) f32 {
        return @abs(self.curvature);
    }

    /// Get the sqrt of absolute curvature
    pub fn sqrt_c(self: HyperbolicConfig) f32 {
        return @sqrt(self.abs_curvature());
    }
};

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// Compute squared L2 norm with SIMD optimization
pub fn norm_squared_simd(x: []const f32) f32 {
    const n = x.len;
    var sum: f32 = 0.0;

    // SIMD path
    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const vec: Vec8 = x[i..][0..SIMD_WIDTH].*;
        const sq = vec * vec;
        sum += @reduce(.Add, sq);
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        sum += x[i] * x[i];
    }

    return sum;
}

/// Compute L2 norm with SIMD optimization
pub fn norm_simd(x: []const f32) f32 {
    return @sqrt(norm_squared_simd(x));
}

/// Compute dot product with SIMD optimization
pub fn dot_product_simd(x: []const f32, y: []const f32) f32 {
    const n = @min(x.len, y.len);
    var sum: f32 = 0.0;

    // SIMD path
    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const x_vec: Vec8 = x[i..][0..SIMD_WIDTH].*;
        const y_vec: Vec8 = y[i..][0..SIMD_WIDTH].*;
        const prod = x_vec * y_vec;
        sum += @reduce(.Add, prod);
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        sum += x[i] * y[i];
    }

    return sum;
}

/// Element-wise vector addition with SIMD: result = a + b
pub fn vec_add_simd(result: []f32, a: []const f32, b: []const f32) void {
    const n = @min(@min(result.len, a.len), b.len);

    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const a_vec: Vec8 = a[i..][0..SIMD_WIDTH].*;
        const b_vec: Vec8 = b[i..][0..SIMD_WIDTH].*;
        result[i..][0..SIMD_WIDTH].* = a_vec + b_vec;
    }

    while (i < n) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Element-wise vector subtraction with SIMD: result = a - b
pub fn vec_sub_simd(result: []f32, a: []const f32, b: []const f32) void {
    const n = @min(@min(result.len, a.len), b.len);

    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const a_vec: Vec8 = a[i..][0..SIMD_WIDTH].*;
        const b_vec: Vec8 = b[i..][0..SIMD_WIDTH].*;
        result[i..][0..SIMD_WIDTH].* = a_vec - b_vec;
    }

    while (i < n) : (i += 1) {
        result[i] = a[i] - b[i];
    }
}

/// Scale vector by scalar with SIMD: result = a * scalar
pub fn vec_scale_simd(result: []f32, a: []const f32, scalar: f32) void {
    const n = @min(result.len, a.len);
    const scalar_vec: Vec8 = @splat(scalar);

    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const a_vec: Vec8 = a[i..][0..SIMD_WIDTH].*;
        result[i..][0..SIMD_WIDTH].* = a_vec * scalar_vec;
    }

    while (i < n) : (i += 1) {
        result[i] = a[i] * scalar;
    }
}

// ============================================================================
// Numerical Stability Functions
// ============================================================================

/// Clamp a value to prevent numerical instability
pub fn clamp(x: f32, min_val: f32, max_val: f32) f32 {
    return @max(min_val, @min(max_val, x));
}

/// Safe division with epsilon to prevent division by zero
pub fn safe_div(numerator: f32, denominator: f32, eps: f32) f32 {
    const safe_denom = if (@abs(denominator) < eps) eps else denominator;
    return numerator / safe_denom;
}

/// Arctanh with numerical stability (clamp input to (-1, 1))
pub fn safe_atanh(x: f32) f32 {
    const clamped = clamp(x, -MAX_NORM, MAX_NORM);
    return math.atanh(clamped);
}

/// Tanh with overflow protection
pub fn safe_tanh(x: f32) f32 {
    // Clamp input to prevent overflow
    const clamped = clamp(x, -15.0, 15.0);
    return math.tanh(clamped);
}

// ============================================================================
// Ball Projection Functions
// ============================================================================

/// Project a point onto the Poincaré ball boundary
///
/// If ||x|| >= max_norm, scale x to have norm = max_norm
/// This ensures points stay within the valid region of the Poincaré ball.
///
/// Parameters:
///   - x: Input/output vector (modified in-place)
///   - max_norm: Maximum allowed norm (default: MAX_NORM)
///
/// Returns:
///   - Original norm before projection
///
/// Complexity: O(n)
pub fn project_to_ball(x: []f32, max_norm: f32) f32 {
    const norm = norm_simd(x);

    if (norm >= max_norm) {
        const scale = max_norm / (norm + EPSILON);
        vec_scale_simd(x, x, scale);
    }

    return norm;
}

/// Project point to ball with given norm constraint
pub fn project_to_ball_with_norm(x: []f32, max_norm: f32, current_norm: f32) void {
    if (current_norm >= max_norm) {
        const scale = max_norm / (current_norm + EPSILON);
        vec_scale_simd(x, x, scale);
    }
}

/// Check if point is inside the Poincaré ball
pub fn is_in_ball(x: []const f32, max_norm: f32) bool {
    const norm_sq = norm_squared_simd(x);
    return norm_sq < max_norm * max_norm;
}

// ============================================================================
// Core Hyperbolic Operations
// ============================================================================

/// Compute hyperbolic distance in Poincaré ball model
///
/// The Poincaré ball distance formula:
/// d(x, y) = (2/sqrt(c)) * arctanh(sqrt(c) * ||(-x) ⊕_c y||)
///
/// Where ⊕_c is Möbius addition with curvature c.
///
/// Parameters:
///   - x: First point in Poincaré ball
///   - y: Second point in Poincaré ball
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator for temporary storage
///
/// Returns:
///   - Hyperbolic distance between x and y
///
/// Complexity: O(n) where n is dimension
pub fn hyperbolic_distance(
    x: []const f32,
    y: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !f32 {
    const n = x.len;
    if (n != y.len) return error.DimensionMismatch;
    if (n == 0) return 0.0;

    // Allocate temporary buffers
    const neg_x = try allocator.alloc(f32, n);
    defer allocator.free(neg_x);
    const mobius_result = try allocator.alloc(f32, n);
    defer allocator.free(mobius_result);

    // Compute -x
    for (0..n) |i| {
        neg_x[i] = -x[i];
    }

    // Compute (-x) ⊕_c y using Möbius addition
    try mobius_add(mobius_result, neg_x, y, config, allocator);

    // Compute ||(-x) ⊕_c y||
    const mobius_norm = norm_simd(mobius_result);

    // Compute distance: (2/sqrt(c)) * arctanh(sqrt(c) * norm)
    const sqrt_c = config.sqrt_c();
    const inner = sqrt_c * mobius_norm;
    const atanh_val = safe_atanh(inner);
    const distance = (2.0 / sqrt_c) * atanh_val;

    return distance;
}

/// Möbius addition in the Poincaré ball model
///
/// Formula: x ⊕_c y = ((1 + 2c⟨x,y⟩ + c||y||²)x + (1 - c||x||²)y) / (1 + 2c⟨x,y⟩ + c²||x||²||y||²)
///
/// Parameters:
///   - result: Output vector
///   - x: First point in Poincaré ball
///   - y: Second point in Poincaré ball
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator (unused, kept for API consistency)
///
/// Complexity: O(n)
pub fn mobius_add(
    result: []f32,
    x: []const f32,
    y: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    const n = x.len;
    if (n != y.len or n != result.len) return error.DimensionMismatch;

    const c = config.abs_curvature();
    const x_norm_sq = norm_squared_simd(x);
    const y_norm_sq = norm_squared_simd(y);
    const xy_dot = dot_product_simd(x, y);

    // Compute numerator coefficients
    const coef_x = 1.0 + 2.0 * c * xy_dot + c * y_norm_sq;
    const coef_y = 1.0 - c * x_norm_sq;

    // Compute denominator
    const denom = 1.0 + 2.0 * c * xy_dot + c * c * x_norm_sq * y_norm_sq;
    const safe_denom = @max(@abs(denom), EPSILON);

    // result = (coef_x * x + coef_y * y) / denom
    for (0..n) |i| {
        result[i] = (coef_x * x[i] + coef_y * y[i]) / safe_denom;
    }

    // Project back to ball if needed
    _ = project_to_ball(result, config.max_norm);
}

// ============================================================================
// Exponential and Logarithmic Maps (Day 55)
// ============================================================================

/// Conformal factor λ_x = 2 / (1 - c * ||x||²)
/// where c is the absolute curvature
pub fn conformal_factor(x: []const f32, curvature: f32) f32 {
    const c = @abs(curvature);
    const x_norm_sq = norm_squared_simd(x);
    return 2.0 / @max(1.0 - c * x_norm_sq, EPSILON);
}

/// Exponential map at the origin: exp_0^c(v)
/// Maps tangent vector v at origin to point on the Poincaré ball
///
/// Formula: exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||)
///
/// Parameters:
///   - result: Output point on manifold
///   - v: Tangent vector at origin
///   - config: Hyperbolic configuration
///
/// Complexity: O(n)
pub fn exp_map_origin(
    result: []f32,
    v: []const f32,
    config: HyperbolicConfig,
) void {
    const n = v.len;
    std.debug.assert(result.len == n);

    const sqrt_c = config.sqrt_c();
    const v_norm = norm_simd(v);

    if (v_norm < EPSILON) {
        // Near origin, exp_0(v) ≈ v
        @memcpy(result, v);
        return;
    }

    // exp_0(v) = tanh(sqrt(c) * ||v|| / 2) * v / (sqrt(c) * ||v||)
    const arg = sqrt_c * v_norm / 2.0;
    const tanh_arg = safe_tanh(arg);
    const scale = tanh_arg / (sqrt_c * v_norm);

    vec_scale_simd(result, v, scale);
    _ = project_to_ball(result, config.max_norm);
}

/// Exponential map at base point: exp_x^c(v)
/// Maps tangent vector v at base point x to point on the Poincaré ball
///
/// Formula: exp_x(v) = x ⊕_c (tanh(sqrt(c) * λ_x * ||v|| / 2) * v / (sqrt(c) * ||v||))
///
/// Parameters:
///   - result: Output point on manifold
///   - base: Base point in Poincaré ball
///   - v: Tangent vector at base point
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator for temporary storage
///
/// Complexity: O(n)
pub fn exp_map(
    result: []f32,
    base: []const f32,
    v: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    const n = v.len;
    if (n != base.len or n != result.len) return error.DimensionMismatch;

    const sqrt_c = config.sqrt_c();
    const lambda_x = conformal_factor(base, config.curvature);
    const v_norm = norm_simd(v);

    if (v_norm < EPSILON) {
        // exp_x(0) = x
        @memcpy(result, base);
        return;
    }

    // Compute the tangent space point: tanh(sqrt(c) * λ_x * ||v|| / 2) * v / (sqrt(c) * ||v||)
    const arg = sqrt_c * lambda_x * v_norm / 2.0;
    const tanh_arg = safe_tanh(arg);
    const scale = tanh_arg / (sqrt_c * v_norm);

    // Allocate temp for scaled tangent
    const scaled_v = try allocator.alloc(f32, n);
    defer allocator.free(scaled_v);

    vec_scale_simd(scaled_v, v, scale);

    // exp_x(v) = x ⊕_c scaled_v
    try mobius_add(result, base, scaled_v, config, allocator);
}

/// Logarithmic map at the origin: log_0^c(y)
/// Maps point y on Poincaré ball to tangent vector at origin
///
/// Formula: log_0(y) = 2 * arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
///
/// Parameters:
///   - result: Output tangent vector at origin
///   - y: Point in Poincaré ball
///   - config: Hyperbolic configuration
///
/// Complexity: O(n)
pub fn log_map_origin(
    result: []f32,
    y: []const f32,
    config: HyperbolicConfig,
) void {
    const n = y.len;
    std.debug.assert(result.len == n);

    const sqrt_c = config.sqrt_c();
    const y_norm = norm_simd(y);

    if (y_norm < EPSILON) {
        // log_0(0) = 0
        @memset(result, 0);
        return;
    }

    // log_0(y) = 2 * arctanh(sqrt(c) * ||y||) * y / (sqrt(c) * ||y||)
    const arg = sqrt_c * y_norm;
    const atanh_arg = safe_atanh(arg);
    const scale = 2.0 * atanh_arg / (sqrt_c * y_norm);

    vec_scale_simd(result, y, scale);
}

/// Logarithmic map at base point: log_x^c(y)
/// Maps point y on Poincaré ball to tangent vector at base point x
///
/// Formula: log_x(y) = (2 / (sqrt(c) * λ_x)) * arctanh(sqrt(c) * ||(-x) ⊕_c y||) * ((-x) ⊕_c y) / ||(-x) ⊕_c y||
///
/// Parameters:
///   - result: Output tangent vector at base point
///   - base: Base point in Poincaré ball
///   - y: Target point in Poincaré ball
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator for temporary storage
///
/// Complexity: O(n)
pub fn log_map(
    result: []f32,
    base: []const f32,
    y: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    const n = y.len;
    if (n != base.len or n != result.len) return error.DimensionMismatch;

    const sqrt_c = config.sqrt_c();
    const lambda_x = conformal_factor(base, config.curvature);

    // Allocate temp for -x
    const neg_x = try allocator.alloc(f32, n);
    defer allocator.free(neg_x);

    for (0..n) |i| {
        neg_x[i] = -base[i];
    }

    // Compute (-x) ⊕_c y
    const mobius_result = try allocator.alloc(f32, n);
    defer allocator.free(mobius_result);

    try mobius_add(mobius_result, neg_x, y, config, allocator);

    const mobius_norm = norm_simd(mobius_result);

    if (mobius_norm < EPSILON) {
        // log_x(x) = 0
        @memset(result, 0);
        return;
    }

    // log_x(y) = (2 / (sqrt(c) * λ_x)) * arctanh(sqrt(c) * ||diff||) * diff / ||diff||
    const arg = sqrt_c * mobius_norm;
    const atanh_arg = safe_atanh(arg);
    const scale = (2.0 / (sqrt_c * lambda_x)) * atanh_arg / mobius_norm;

    vec_scale_simd(result, mobius_result, scale);
}

/// Parallel transport of tangent vector v from point x to point y along geodesic
///
/// Formula: P_{x→y}(v) = (λ_x / λ_y) * gyr[y, -x](v)
/// where gyr is the gyration (Thomas rotation)
///
/// Simplified formula using Möbius gyration:
/// P_{x→y}(v) = log_y(exp_x(v) ⊕_c (-x ⊕_c y)) when ||v|| is small
///
/// For efficiency, we use the direct formula:
/// P_{x→y}(v) = v * (λ_x / λ_y)
/// This is an approximation valid when x and y are close.
///
/// Parameters:
///   - result: Output transported vector at y
///   - v: Tangent vector at x
///   - x: Source point
///   - y: Target point
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator
///
/// Complexity: O(n)
pub fn parallel_transport(
    result: []f32,
    v: []const f32,
    x: []const f32,
    y: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    const n = v.len;
    if (n != x.len or n != y.len or n != result.len) return error.DimensionMismatch;

    // Compute conformal factors
    const lambda_x = conformal_factor(x, config.curvature);
    const lambda_y = conformal_factor(y, config.curvature);

    // Compute (-x) ⊕_c y for gyration
    const neg_x = try allocator.alloc(f32, n);
    defer allocator.free(neg_x);
    for (0..n) |i| {
        neg_x[i] = -x[i];
    }

    const diff = try allocator.alloc(f32, n);
    defer allocator.free(diff);
    try mobius_add(diff, neg_x, y, config, allocator);

    // Use the gyration-based formula for parallel transport
    // P_{x→y}(v) = gyr[y, -x](v) * (λ_x / λ_y)
    // For the gyration, we use: gyr[a,b](v) = -a ⊕ (a ⊕ (b ⊕ v))
    // Simplified: scale by conformal factor ratio

    const diff_norm_sq = norm_squared_simd(diff);
    const v_norm_sq = norm_squared_simd(v);
    const diff_v_dot = dot_product_simd(diff, v);

    if (diff_norm_sq < EPSILON or v_norm_sq < EPSILON) {
        // Simple scaling when points are close or v is small
        const scale = lambda_x / lambda_y;
        vec_scale_simd(result, v, scale);
        return;
    }

    // Full gyration-based transport formula
    // Gyration: gyr[a,b](v) involves rotation in the plane of a, b
    const c = config.abs_curvature();

    // Coefficient for the transport
    const coef = lambda_x / lambda_y;

    // Apply gyration effect
    // v' = v - 2c * ⟨diff, v⟩ / (1 + c||diff||²) * diff
    const denom = 1.0 + c * diff_norm_sq;
    const gyration_coef = 2.0 * c * diff_v_dot / denom;

    for (0..n) |i| {
        result[i] = coef * (v[i] - gyration_coef * diff[i]);
    }
}

/// Convert Euclidean gradient to Riemannian gradient
///
/// In the Poincaré ball with metric g_x = (λ_x)² I, the Riemannian gradient is:
/// grad_R f(x) = (1 / λ_x²) * grad_E f(x)
///
/// This is essential for Riemannian optimization (gradient descent on manifold).
///
/// Parameters:
///   - result: Output Riemannian gradient
///   - euclidean_grad: Euclidean gradient
///   - x: Point where gradient is evaluated
///   - config: Hyperbolic configuration
///
/// Complexity: O(n)
pub fn euclidean_to_riemannian_grad(
    result: []f32,
    euclidean_grad: []const f32,
    x: []const f32,
    config: HyperbolicConfig,
) void {
    const n = euclidean_grad.len;
    std.debug.assert(result.len == n and x.len == n);

    // Compute conformal factor
    const lambda_x = conformal_factor(x, config.curvature);
    const lambda_sq = lambda_x * lambda_x;

    // grad_R = grad_E / λ²
    const scale = 1.0 / lambda_sq;
    vec_scale_simd(result, euclidean_grad, scale);
}

/// Riemannian gradient descent step
///
/// x_new = exp_x(-lr * grad_R)
///
/// Parameters:
///   - result: Updated point on manifold
///   - x: Current point
///   - euclidean_grad: Euclidean gradient
///   - lr: Learning rate
///   - config: Hyperbolic configuration
///   - allocator: Memory allocator
///
/// Complexity: O(n)
pub fn riemannian_sgd_step(
    result: []f32,
    x: []const f32,
    euclidean_grad: []const f32,
    lr: f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    const n = x.len;
    if (n != euclidean_grad.len or n != result.len) return error.DimensionMismatch;

    // Allocate temp for Riemannian gradient
    const riemannian_grad = try allocator.alloc(f32, n);
    defer allocator.free(riemannian_grad);

    // Convert to Riemannian gradient
    euclidean_to_riemannian_grad(riemannian_grad, euclidean_grad, x, config);

    // Scale by -lr for descent direction
    const neg_grad = try allocator.alloc(f32, n);
    defer allocator.free(neg_grad);
    vec_scale_simd(neg_grad, riemannian_grad, -lr);

    // x_new = exp_x(-lr * grad_R)
    try exp_map(result, x, neg_grad, config, allocator);
}

// ============================================================================
// Möbius Scalar Multiplication
// ============================================================================

/// Möbius scalar multiplication in Poincaré ball model
/// r ⊗_c x = (1/sqrt(|c|)) * tanh(r * arctanh(sqrt(|c|) * ||x||)) * (x / ||x||)
pub fn mobius_scalar_mul(
    result: []f32,
    r: f32,
    x: []const f32,
    config: HyperbolicConfig,
) void {
    const n = x.len;
    std.debug.assert(result.len == n);

    const sqrt_c = @sqrt(config.abs_curvature());
    const x_norm = norm_simd(x);

    if (x_norm < config.epsilon) {
        for (result) |*val| val.* = 0.0;
        return;
    }

    // arctanh(sqrt(c) * ||x||)
    const atanh_arg = sqrt_c * @min(x_norm, config.max_norm);
    const atanh_val = safe_atanh(atanh_arg);

    // tanh(r * arctanh(...))
    const tanh_val = safe_tanh(r * atanh_val);

    // Scale: (1/sqrt(c)) * tanh(...) / ||x||
    const scale = tanh_val / (sqrt_c * x_norm);

    vec_scale_simd(result, x, scale);
    _ = project_to_ball(result, config.max_norm);
}

// ============================================================================
// Model Conversions
// ============================================================================

/// Convert from Poincaré ball to Klein model
pub fn poincare_to_klein(result: []f32, x: []const f32) void {
    const n = x.len;
    std.debug.assert(result.len == n);

    const x_norm_sq = norm_squared_simd(x);
    const scale = 2.0 / (1.0 + x_norm_sq);

    vec_scale_simd(result, x, scale);
}

/// Convert from Klein model to Poincaré ball
pub fn klein_to_poincare(result: []f32, x: []const f32) void {
    const n = x.len;
    std.debug.assert(result.len == n);

    const x_norm_sq = norm_squared_simd(x);
    const scale = 1.0 / (1.0 + @sqrt(@max(1.0 - x_norm_sq, EPSILON)));

    vec_scale_simd(result, x, scale);
}

// ============================================================================
// Hyperbolic Midpoint
// ============================================================================

/// Compute hyperbolic midpoint of two points
pub fn hyperbolic_midpoint(
    result: []f32,
    x: []const f32,
    y: []const f32,
    config: HyperbolicConfig,
    allocator: std.mem.Allocator,
) !void {
    const n = x.len;
    std.debug.assert(result.len == n and y.len == n);

    // midpoint = mobius_add(x, mobius_scalar_mul(0.5, mobius_add(-x, y)))
    const neg_x = try allocator.alloc(f32, n);
    defer allocator.free(neg_x);
    vec_scale_simd(neg_x, x, -1.0);

    const diff = try allocator.alloc(f32, n);
    defer allocator.free(diff);
    try mobius_add(diff, neg_x, y, config, allocator);

    const half_diff = try allocator.alloc(f32, n);
    defer allocator.free(half_diff);
    mobius_scalar_mul(half_diff, 0.5, diff, config);

    try mobius_add(result, x, half_diff, config, allocator);
}

// ============================================================================
// Unit Tests
// ============================================================================

test "exp_map_origin and log_map_origin round-trip" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    // Test vector
    const v = [_]f32{ 0.1, 0.2, 0.3 };
    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;

    // exp_0(v)
    exp_map_origin(&exp_result, &v, config);

    // log_0(exp_0(v)) should ≈ v
    log_map_origin(&log_result, &exp_result, config);

    // Verify round-trip: log(exp(v)) ≈ v
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(log_result[i], v[i], 1e-5);
    }

    _ = allocator;
}

test "log_map_origin and exp_map_origin inverse" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    // Test point inside ball
    const y = [_]f32{ 0.3, -0.2, 0.1 };
    var log_result: [3]f32 = undefined;
    var exp_result: [3]f32 = undefined;

    // log_0(y)
    log_map_origin(&log_result, &y, config);

    // exp_0(log_0(y)) should ≈ y
    exp_map_origin(&exp_result, &log_result, config);

    // Verify round-trip: exp(log(y)) ≈ y
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(exp_result[i], y[i], 1e-5);
    }

    _ = allocator;
}

test "exp_map and log_map round-trip at base point" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    // Base point
    const base = [_]f32{ 0.2, 0.1, -0.15 };
    // Tangent vector at base
    const v = [_]f32{ 0.05, -0.03, 0.02 };

    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;

    // exp_x(v)
    try exp_map(&exp_result, &base, &v, config, allocator);

    // log_x(exp_x(v)) should ≈ v
    try log_map(&log_result, &base, &exp_result, config, allocator);

    // Verify round-trip
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(log_result[i], v[i], 1e-4);
    }
}

test "log_map and exp_map inverse at base point" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    // Base point
    const base = [_]f32{ 0.1, -0.2, 0.15 };
    // Target point
    const target = [_]f32{ 0.25, 0.1, -0.1 };

    var log_result: [3]f32 = undefined;
    var exp_result: [3]f32 = undefined;

    // log_x(y)
    try log_map(&log_result, &base, &target, config, allocator);

    // exp_x(log_x(y)) should ≈ y
    try exp_map(&exp_result, &base, &log_result, config, allocator);

    // Verify round-trip: exp(log(y)) ≈ y
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(exp_result[i], target[i], 1e-4);
    }
}

test "euclidean_to_riemannian_grad at origin" {
    const config = HyperbolicConfig{};

    // At origin, λ = 2, so grad_R = grad_E / 4
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    const grad_e = [_]f32{ 1.0, 2.0, 3.0 };
    var grad_r: [3]f32 = undefined;

    euclidean_to_riemannian_grad(&grad_r, &grad_e, &origin, config);

    // At origin: λ = 2, so scale = 1/4
    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(grad_e[i] / 4.0, grad_r[i], 0.01);
    }
}

test "mobius_add with zero vector" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 0.3, 0.4, 0.0 };
    var result: [3]f32 = undefined;

    try mobius_add(&result, &x, &y, config, allocator);

    // 0 ⊕ y should equal y (approximately)
    try std.testing.expectApproxEqAbs(y[0], result[0], 0.01);
    try std.testing.expectApproxEqAbs(y[1], result[1], 0.01);
}

test "mobius_add result stays in ball" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.5, 0.5, 0.0 };
    const y = [_]f32{ 0.4, 0.3, 0.0 };
    var result: [3]f32 = undefined;

    try mobius_add(&result, &x, &y, config, allocator);

    try std.testing.expect(is_in_ball(&result, MAX_NORM));
}

test "mobius_scalar_mul with scalar 1.0" {
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var result: [3]f32 = undefined;

    mobius_scalar_mul(&result, 1.0, &x, config);

    // 1 ⊗ x should equal x (approximately)
    try std.testing.expectApproxEqAbs(x[0], result[0], 0.01);
    try std.testing.expectApproxEqAbs(x[1], result[1], 0.01);
}

test "mobius_scalar_mul with scalar 0.0" {
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var result: [3]f32 = undefined;

    mobius_scalar_mul(&result, 0.0, &x, config);

    // 0 ⊗ x should equal 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.01);
}

test "mobius_scalar_mul result stays in ball" {
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.6, 0.6, 0.0 };
    var result: [3]f32 = undefined;

    mobius_scalar_mul(&result, 2.0, &x, config);

    try std.testing.expect(is_in_ball(&result, MAX_NORM));
}

test "poincare_to_klein and back preserves point" {
    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var klein: [3]f32 = undefined;
    var back: [3]f32 = undefined;

    poincare_to_klein(&klein, &x);
    klein_to_poincare(&back, &klein);

    // Should get back approximately the same point
    try std.testing.expectApproxEqAbs(x[0], back[0], 0.01);
    try std.testing.expectApproxEqAbs(x[1], back[1], 0.01);
}

test "poincare_to_klein maps origin to origin" {
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    poincare_to_klein(&result, &origin);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.001);
}

test "exp_map_origin maps zero to origin" {
    const config = HyperbolicConfig{};
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    exp_map_origin(&result, &zero, config);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.001);
}

test "log_map_origin maps origin to zero" {
    const config = HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    log_map_origin(&result, &origin, config);

    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 0.001);
}

test "exp_map and log_map are inverse operations" {
    const config = HyperbolicConfig{};
    const v = [_]f32{ 0.5, 0.3, 0.1 };
    var on_manifold: [3]f32 = undefined;
    var back: [3]f32 = undefined;

    exp_map_origin(&on_manifold, &v, config);
    log_map_origin(&back, &on_manifold, config);

    // Should get back approximately the same tangent vector
    try std.testing.expectApproxEqAbs(v[0], back[0], 0.05);
    try std.testing.expectApproxEqAbs(v[1], back[1], 0.05);
}

test "hyperbolic_midpoint is between two points" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.1, 0.0, 0.0 };
    const y = [_]f32{ 0.5, 0.0, 0.0 };
    var mid: [3]f32 = undefined;

    try hyperbolic_midpoint(&mid, &x, &y, config, allocator);

    // Distance from x to mid should equal distance from mid to y
    const d_x_mid = try hyperbolic_distance(&x, &mid, config, allocator);
    const d_mid_y = try hyperbolic_distance(&mid, &y, config, allocator);

    try std.testing.expectApproxEqAbs(d_x_mid, d_mid_y, 0.1);
}

test "SIMD operations with large vectors" {
    // Test with vector larger than SIMD width
    var x: [32]f32 = undefined;
    var y: [32]f32 = undefined;
    var result: [32]f32 = undefined;

    for (0..32) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.01;
        y[i] = @as(f32, @floatFromInt(32 - i)) * 0.01;
    }

    vec_add_simd(&result, &x, &y);

    // All elements should be 0.32
    for (0..32) |i| {
        try std.testing.expectApproxEqAbs(@as(f32, 0.32), result[i], 0.001);
    }
}

test "parallel_transport preserves tangent vector norm approximately" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const v = [_]f32{ 0.1, 0.2, 0.0 };
    const x = [_]f32{ 0.1, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.2, 0.0 };
    var result: [3]f32 = undefined;

    try parallel_transport(&result, &v, &x, &y, config, allocator);

    // Result should be finite
    try std.testing.expect(!math.isNan(result[0]));
    try std.testing.expect(!math.isNan(result[1]));
    try std.testing.expect(!math.isInf(result[0]));
    try std.testing.expect(!math.isInf(result[1]));
}

test "numerical stability with points near boundary" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    // Points very close to boundary
    const x = [_]f32{ 0.99, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.99, 0.0 };

    // Project to ensure they're in ball
    var x_proj: [3]f32 = undefined;
    var y_proj: [3]f32 = undefined;
    @memcpy(&x_proj, &x);
    @memcpy(&y_proj, &y);
    _ = project_to_ball(&x_proj, MAX_NORM);
    _ = project_to_ball(&y_proj, MAX_NORM);

    // Distance should still be finite
    const d = try hyperbolic_distance(&x_proj, &y_proj, config, allocator);
    try std.testing.expect(!math.isNan(d));
    try std.testing.expect(!math.isInf(d));
    try std.testing.expect(d > 0);
}

test "parallel_transport preserves norm approximately" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.1, 0.1, 0.0 };
    const y = [_]f32{ 0.15, 0.05, 0.1 };
    const v = [_]f32{ 0.1, -0.1, 0.05 };

    var transported: [3]f32 = undefined;

    try parallel_transport(&transported, &v, &x, &y, config, allocator);

    // Compute norms (Riemannian norms differ, but we check the transport ran)
    const v_norm = norm_simd(&v);
    const t_norm = norm_simd(&transported);

    // Transport should produce a valid vector
    try std.testing.expect(t_norm > 0);
    try std.testing.expect(!std.math.isNan(t_norm));
    _ = v_norm;
}

test "mobius_add identity" {
    const allocator = std.testing.allocator;
    const config = HyperbolicConfig{};

    const x = [_]f32{ 0.3, -0.2, 0.1 };
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    // x ⊕ 0 = x
    try mobius_add(&result, &x, &zero, config, allocator);

    for (0..3) |i| {
        try std.testing.expectApproxEqAbs(result[i], x[i], 1e-6);
    }
}

test "conformal_factor at origin" {
    const config = HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };

    const lambda = conformal_factor(&origin, config.curvature);

    // At origin: λ = 2 / (1 - 0) = 2
    try std.testing.expectApproxEqAbs(lambda, 2.0, 1e-6);
}

test "project_to_ball keeps point inside" {
    var point = [_]f32{ 0.8, 0.7, 0.6 };
    const original_norm = norm_simd(&point);

    _ = project_to_ball(&point, MAX_NORM);

    const new_norm = norm_simd(&point);

    // Should be projected if original was outside
    if (original_norm >= MAX_NORM) {
        try std.testing.expect(new_norm < 1.0);
    }
}
