// mHC Product Manifold Support Module - Day 57
// Implements product manifold operations for mixed geometry spaces
//
// Features:
// 1. ProductManifoldConfig - Configure component manifolds
// 2. Component-wise Distance - Combine distances from each component
// 3. Manifold Type Per Dimension - Assign Euclidean/Hyperbolic/Spherical per dimension range
// 4. Weighted Constraint Combination - Combine constraints with weights
// 5. Code-switching Support - Mixed Arabic-English handling
//
// Reference: mhc_configuration.zig for base types

const std = @import("std");
const math = std.math;

// ============================================================================
// Core Types
// ============================================================================

/// Manifold type enumeration
pub const ManifoldType = enum {
    Euclidean,
    Hyperbolic,
    Spherical,

    pub fn getName(self: ManifoldType) []const u8 {
        return switch (self) {
            .Euclidean => "Euclidean",
            .Hyperbolic => "Hyperbolic",
            .Spherical => "Spherical",
        };
    }
};

/// Component of a product manifold with dimension range and weight
pub const ManifoldComponent = struct {
    /// Type of manifold for this component
    manifold_type: ManifoldType,

    /// Start dimension index (inclusive)
    dim_start: u32,

    /// End dimension index (exclusive)
    dim_end: u32,

    /// Weight for distance/constraint combination
    weight: f32 = 1.0,

    /// Curvature parameter (negative for hyperbolic, positive for spherical)
    curvature: f32 = 1.0,

    /// Numerical stability epsilon
    epsilon: f32 = 1e-8,

    /// Get the number of dimensions in this component
    pub fn dims(self: ManifoldComponent) u32 {
        return self.dim_end - self.dim_start;
    }

    /// Validate component configuration
    pub fn validate(self: ManifoldComponent) bool {
        if (self.dim_end <= self.dim_start) return false;
        if (self.weight <= 0) return false;
        if (self.manifold_type == .Hyperbolic and self.curvature >= 0) return false;
        if (self.manifold_type == .Spherical and self.curvature <= 0) return false;
        return true;
    }
};

/// Product manifold configuration with multiple components
pub const ProductManifoldConfig = struct {
    /// Array of manifold components
    components: []const ManifoldComponent,

    /// Total dimension of product space
    total_dims: u32,

    /// Enable code-switching optimization for mixed Arabic-English
    code_switching_enabled: bool = false,

    /// Arabic dimension range for code-switching (typically first half)
    arabic_dim_start: u32 = 0,
    arabic_dim_end: u32 = 0,

    /// English dimension range for code-switching (typically second half)
    english_dim_start: u32 = 0,
    english_dim_end: u32 = 0,

    /// Validate the entire product manifold configuration
    pub fn validate(self: ProductManifoldConfig) bool {
        if (self.components.len == 0) return false;

        var covered_dims: u32 = 0;
        var prev_end: u32 = 0;

        for (self.components) |comp| {
            if (!comp.validate()) return false;
            if (comp.dim_start != prev_end) return false; // Must be contiguous
            covered_dims += comp.dims();
            prev_end = comp.dim_end;
        }

        return covered_dims == self.total_dims;
    }

    /// Get component containing a specific dimension
    pub fn getComponentForDim(self: ProductManifoldConfig, dim: u32) ?ManifoldComponent {
        for (self.components) |comp| {
            if (dim >= comp.dim_start and dim < comp.dim_end) {
                return comp;
            }
        }
        return null;
    }

    /// Calculate total weight sum for normalization
    pub fn totalWeight(self: ProductManifoldConfig) f32 {
        var sum: f32 = 0.0;
        for (self.components) |comp| {
            sum += comp.weight;
        }
        return sum;
    }
};

/// Code-switching context for mixed language handling
pub const CodeSwitchContext = struct {
    /// Ratio of Arabic tokens in current context (0.0 to 1.0)
    arabic_ratio: f32,

    /// Whether to apply language-specific geometric constraints
    apply_constraints: bool = true,

    /// Smoothing factor for language transition boundaries
    transition_smoothing: f32 = 0.1,
};

// ============================================================================
// Distance Functions
// ============================================================================

/// Euclidean distance between two points
pub fn euclidean_distance(x: []const f32, y: []const f32) f32 {
    std.debug.assert(x.len == y.len);
    var sum: f32 = 0.0;
    for (x, y) |xi, yi| {
        const diff = xi - yi;
        sum += diff * diff;
    }
    return @sqrt(sum);
}

/// Hyperbolic distance in Poincaré ball model
/// d(x, y) = arcosh(1 + 2 * ||x - y||² / ((1 - ||x||²)(1 - ||y||²)))
pub fn hyperbolic_distance(x: []const f32, y: []const f32, curvature: f32, epsilon: f32) f32 {
    std.debug.assert(x.len == y.len);
    const c = -curvature; // Convert to positive for calculations

    var norm_x_sq: f32 = 0.0;
    var norm_y_sq: f32 = 0.0;
    var diff_sq: f32 = 0.0;

    for (x, y) |xi, yi| {
        norm_x_sq += xi * xi;
        norm_y_sq += yi * yi;
        const diff = xi - yi;
        diff_sq += diff * diff;
    }

    // Clamp norms for numerical stability (must be < 1 for Poincaré ball)
    norm_x_sq = @min(norm_x_sq, 1.0 - epsilon);
    norm_y_sq = @min(norm_y_sq, 1.0 - epsilon);

    const denom = (1.0 - norm_x_sq) * (1.0 - norm_y_sq);
    const ratio = 2.0 * diff_sq / @max(denom, epsilon);

    // arcosh(x) = ln(x + sqrt(x² - 1))
    const cosh_dist = 1.0 + ratio;
    const dist = math.acosh(cosh_dist);

    return dist / @sqrt(c);
}

/// Spherical distance (great circle distance)
/// d(x, y) = arccos(x · y / (||x|| ||y||)) / sqrt(curvature)
pub fn spherical_distance(x: []const f32, y: []const f32, curvature: f32, epsilon: f32) f32 {
    std.debug.assert(x.len == y.len);

    var dot: f32 = 0.0;
    var norm_x_sq: f32 = 0.0;
    var norm_y_sq: f32 = 0.0;

    for (x, y) |xi, yi| {
        dot += xi * yi;
        norm_x_sq += xi * xi;
        norm_y_sq += yi * yi;
    }

    const norm_x = @sqrt(norm_x_sq);
    const norm_y = @sqrt(norm_y_sq);
    const denom = @max(norm_x * norm_y, epsilon);

    // Clamp to [-1, 1] for numerical stability
    var cos_angle = dot / denom;
    cos_angle = @max(-1.0, @min(1.0, cos_angle));

    return math.acos(cos_angle) / @sqrt(curvature);
}

/// Combined product distance across all manifold components
pub fn product_distance(x: []const f32, y: []const f32, config: ProductManifoldConfig) f32 {
    std.debug.assert(x.len == y.len);
    std.debug.assert(x.len == config.total_dims);

    var total_dist_sq: f32 = 0.0;
    const weight_sum = config.totalWeight();

    for (config.components) |comp| {
        const start = comp.dim_start;
        const end = comp.dim_end;
        const x_slice = x[start..end];
        const y_slice = y[start..end];

        const dist = switch (comp.manifold_type) {
            .Euclidean => euclidean_distance(x_slice, y_slice),
            .Hyperbolic => hyperbolic_distance(x_slice, y_slice, comp.curvature, comp.epsilon),
            .Spherical => spherical_distance(x_slice, y_slice, comp.curvature, comp.epsilon),
        };

        // Weighted squared distance for Riemannian product
        const normalized_weight = comp.weight / weight_sum;
        total_dist_sq += normalized_weight * dist * dist;
    }

    return @sqrt(total_dist_sq);
}

// ============================================================================
// Projection Functions
// ============================================================================

/// Project point onto Euclidean ball of radius r
fn project_euclidean(point: []f32, radius: f32) void {
    var norm_sq: f32 = 0.0;
    for (point) |val| {
        norm_sq += val * val;
    }
    const norm = @sqrt(norm_sq);

    if (norm > radius) {
        const scale = radius / norm;
        for (point) |*val| {
            val.* *= scale;
        }
    }
}

/// Project point onto Poincaré ball (||x|| < 1)
fn project_hyperbolic(point: []f32, epsilon: f32) void {
    var norm_sq: f32 = 0.0;
    for (point) |val| {
        norm_sq += val * val;
    }
    const norm = @sqrt(norm_sq);
    const max_norm = 1.0 - epsilon;

    if (norm >= max_norm) {
        const scale = max_norm / norm;
        for (point) |*val| {
            val.* *= scale;
        }
    }
}

/// Project point onto unit sphere (||x|| = 1)
fn project_spherical(point: []f32, epsilon: f32) void {
    var norm_sq: f32 = 0.0;
    for (point) |val| {
        norm_sq += val * val;
    }
    const norm = @sqrt(norm_sq);

    if (norm > epsilon) {
        const scale = 1.0 / norm;
        for (point) |*val| {
            val.* *= scale;
        }
    }
}

/// Project point onto product manifold (each component separately)
pub fn product_project(point: []f32, config: ProductManifoldConfig) void {
    std.debug.assert(point.len == config.total_dims);

    for (config.components) |comp| {
        const slice = point[comp.dim_start..comp.dim_end];

        switch (comp.manifold_type) {
            .Euclidean => project_euclidean(slice, 10.0), // Default L2 ball
            .Hyperbolic => project_hyperbolic(slice, comp.epsilon),
            .Spherical => project_spherical(slice, comp.epsilon),
        }
    }
}


// ============================================================================
// Exponential and Logarithmic Maps
// ============================================================================

/// Euclidean exponential map (trivial: base + v)
fn exp_euclidean(base: []const f32, v: []const f32, result: []f32) void {
    for (base, v, result) |b, vi, *r| {
        r.* = b + vi;
    }
}

/// Poincaré ball exponential map
/// exp_x(v) = x ⊕ tanh(λ_x ||v|| / 2) * (v / ||v||)
fn exp_hyperbolic(base: []const f32, v: []const f32, result: []f32, curvature: f32, epsilon: f32) void {
    const c = -curvature;

    var norm_x_sq: f32 = 0.0;
    var norm_v_sq: f32 = 0.0;

    for (base) |xi| {
        norm_x_sq += xi * xi;
    }
    for (v) |vi| {
        norm_v_sq += vi * vi;
    }

    const norm_v = @sqrt(norm_v_sq);
    const lambda_x = 2.0 / @max(1.0 - norm_x_sq, epsilon);

    if (norm_v < epsilon) {
        // Zero tangent vector, return base point
        @memcpy(result, base);
        return;
    }

    const t = math.tanh(@sqrt(c) * lambda_x * norm_v / 2.0);

    // Compute Möbius addition: x ⊕ (t * v/||v||)
    var scaled_v: [256]f32 = undefined;
    const n = base.len;
    for (0..n) |i| {
        scaled_v[i] = t * v[i] / norm_v;
    }

    // Möbius addition formula
    var dot_x_sv: f32 = 0.0;
    var norm_sv_sq: f32 = 0.0;
    for (0..n) |i| {
        dot_x_sv += base[i] * scaled_v[i];
        norm_sv_sq += scaled_v[i] * scaled_v[i];
    }

    const num_coeff = 1.0 + 2.0 * c * dot_x_sv + c * norm_sv_sq;
    const denom = 1.0 + 2.0 * c * dot_x_sv + c * c * norm_x_sq * norm_sv_sq;

    for (0..n) |i| {
        result[i] = (num_coeff * base[i] + (1.0 - c * norm_x_sq) * scaled_v[i]) / @max(denom, epsilon);
    }

    // Project to ensure we stay in the ball
    project_hyperbolic(result, epsilon);
}

/// Spherical exponential map
/// exp_x(v) = cos(||v||) * x + sin(||v||) * (v / ||v||)
fn exp_spherical(base: []const f32, v: []const f32, result: []f32, epsilon: f32) void {
    var norm_v_sq: f32 = 0.0;
    for (v) |vi| {
        norm_v_sq += vi * vi;
    }
    const norm_v = @sqrt(norm_v_sq);

    if (norm_v < epsilon) {
        @memcpy(result, base);
        return;
    }

    const cos_t = @cos(norm_v);
    const sin_t = @sin(norm_v);

    for (base, v, result) |b, vi, *r| {
        r.* = cos_t * b + sin_t * (vi / norm_v);
    }

    // Ensure result is on sphere
    project_spherical(result, epsilon);
}

/// Product manifold exponential map
pub fn product_exp_map(base: []const f32, v: []const f32, result: []f32, config: ProductManifoldConfig) void {
    std.debug.assert(base.len == config.total_dims);
    std.debug.assert(v.len == config.total_dims);
    std.debug.assert(result.len == config.total_dims);

    for (config.components) |comp| {
        const start = comp.dim_start;
        const end = comp.dim_end;

        switch (comp.manifold_type) {
            .Euclidean => exp_euclidean(base[start..end], v[start..end], result[start..end]),
            .Hyperbolic => exp_hyperbolic(base[start..end], v[start..end], result[start..end], comp.curvature, comp.epsilon),
            .Spherical => exp_spherical(base[start..end], v[start..end], result[start..end], comp.epsilon),
        }
    }
}

/// Euclidean logarithmic map (trivial: y - base)
fn log_euclidean(base: []const f32, y: []const f32, result: []f32) void {
    for (base, y, result) |b, yi, *r| {
        r.* = yi - b;
    }
}

/// Poincaré ball logarithmic map
fn log_hyperbolic(base: []const f32, y: []const f32, result: []f32, curvature: f32, epsilon: f32) void {
    const c = -curvature;
    const n = base.len;

    // Compute -x ⊕ y (Möbius addition of -x and y)
    var norm_x_sq: f32 = 0.0;
    var norm_y_sq: f32 = 0.0;
    var dot_neg_x_y: f32 = 0.0;

    for (0..n) |i| {
        norm_x_sq += base[i] * base[i];
        norm_y_sq += y[i] * y[i];
        dot_neg_x_y += (-base[i]) * y[i];
    }

    const num_coeff = 1.0 + 2.0 * c * dot_neg_x_y + c * norm_y_sq;
    const denom = 1.0 + 2.0 * c * dot_neg_x_y + c * c * norm_x_sq * norm_y_sq;

    // Compute -x ⊕ y
    var mob_add: [256]f32 = undefined;
    for (0..n) |i| {
        mob_add[i] = (num_coeff * (-base[i]) + (1.0 - c * norm_x_sq) * y[i]) / @max(denom, epsilon);
    }

    var norm_mob_sq: f32 = 0.0;
    for (0..n) |i| {
        norm_mob_sq += mob_add[i] * mob_add[i];
    }
    const norm_mob = @sqrt(norm_mob_sq);

    if (norm_mob < epsilon) {
        @memset(result, 0.0);
        return;
    }

    const lambda_x = 2.0 / @max(1.0 - norm_x_sq, epsilon);
    const coeff = (2.0 / (@sqrt(c) * lambda_x)) * math.atanh(@sqrt(c) * norm_mob);

    for (0..n) |i| {
        result[i] = coeff * mob_add[i] / norm_mob;
    }
}

/// Spherical logarithmic map
fn log_spherical(base: []const f32, y: []const f32, result: []f32, epsilon: f32) void {
    var dot: f32 = 0.0;
    for (base, y) |b, yi| {
        dot += b * yi;
    }

    // Clamp for numerical stability
    dot = @max(-1.0, @min(1.0, dot));
    const angle = math.acos(dot);

    if (angle < epsilon) {
        @memset(result, 0.0);
        return;
    }

    const sin_angle = @sin(angle);
    if (@abs(sin_angle) < epsilon) {
        @memset(result, 0.0);
        return;
    }

    for (base, y, result) |b, yi, *r| {
        r.* = (angle / sin_angle) * (yi - dot * b);
    }
}

/// Product manifold logarithmic map
pub fn product_log_map(base: []const f32, y: []const f32, result: []f32, config: ProductManifoldConfig) void {
    std.debug.assert(base.len == config.total_dims);
    std.debug.assert(y.len == config.total_dims);
    std.debug.assert(result.len == config.total_dims);

    for (config.components) |comp| {
        const start = comp.dim_start;
        const end = comp.dim_end;

        switch (comp.manifold_type) {
            .Euclidean => log_euclidean(base[start..end], y[start..end], result[start..end]),
            .Hyperbolic => log_hyperbolic(base[start..end], y[start..end], result[start..end], comp.curvature, comp.epsilon),
            .Spherical => log_spherical(base[start..end], y[start..end], result[start..end], comp.epsilon),
        }
    }
}



// ============================================================================
// Code-Switching Support (Arabic-English)
// ============================================================================

/// Apply code-switching aware constraints to embeddings
/// Adjusts weights based on detected language ratio
pub fn apply_code_switch_constraints(
    embeddings: []f32,
    config: ProductManifoldConfig,
    context: CodeSwitchContext,
) void {
    if (!config.code_switching_enabled) {
        // Just apply standard projection
        product_project(embeddings, config);
        return;
    }

    // Apply language-specific geometric constraints
    const arabic_weight = context.arabic_ratio;
    const english_weight = 1.0 - context.arabic_ratio;

    // Apply differential constraints based on language regions
    if (config.arabic_dim_end > config.arabic_dim_start) {
        const arabic_slice = embeddings[config.arabic_dim_start..config.arabic_dim_end];
        // Scale Arabic dimensions by their contribution
        for (arabic_slice) |*val| {
            val.* *= (1.0 + context.transition_smoothing * arabic_weight);
        }
    }

    if (config.english_dim_end > config.english_dim_start) {
        const english_slice = embeddings[config.english_dim_start..config.english_dim_end];
        // Scale English dimensions by their contribution
        for (english_slice) |*val| {
            val.* *= (1.0 + context.transition_smoothing * english_weight);
        }
    }

    // Finally project onto the product manifold
    product_project(embeddings, config);
}

/// Combined weighted distance for code-switched content
pub fn code_switch_distance(
    x: []const f32,
    y: []const f32,
    config: ProductManifoldConfig,
    context: CodeSwitchContext,
) f32 {
    if (!config.code_switching_enabled) {
        return product_distance(x, y, config);
    }

    // Compute separate distances for Arabic and English regions
    var arabic_dist: f32 = 0.0;
    var english_dist: f32 = 0.0;

    if (config.arabic_dim_end > config.arabic_dim_start) {
        arabic_dist = euclidean_distance(
            x[config.arabic_dim_start..config.arabic_dim_end],
            y[config.arabic_dim_start..config.arabic_dim_end],
        );
    }

    if (config.english_dim_end > config.english_dim_start) {
        english_dist = euclidean_distance(
            x[config.english_dim_start..config.english_dim_end],
            y[config.english_dim_start..config.english_dim_end],
        );
    }

    // Weight by language ratio
    const weighted_dist = context.arabic_ratio * arabic_dist +
        (1.0 - context.arabic_ratio) * english_dist;

    // Add the base product distance for other dimensions
    const base_dist = product_distance(x, y, config);

    return (weighted_dist + base_dist) / 2.0;
}

// ============================================================================
// Weighted Constraint Combination
// ============================================================================

/// Apply weighted combination of manifold constraints
pub fn apply_weighted_constraints(
    embeddings: []f32,
    config: ProductManifoldConfig,
    constraint_weights: []const f32,
) void {
    std.debug.assert(constraint_weights.len == config.components.len);

    for (config.components, constraint_weights) |comp, weight| {
        if (weight <= 0) continue;

        const slice = embeddings[comp.dim_start..comp.dim_end];

        // Apply weighted projection based on constraint type
        switch (comp.manifold_type) {
            .Euclidean => {
                // L2 ball projection with weighted radius
                const radius = 10.0 * weight;
                project_euclidean(slice, radius);
            },
            .Hyperbolic => {
                // Tighter or looser Poincaré ball based on weight
                const adjusted_eps = comp.epsilon / weight;
                project_hyperbolic(slice, adjusted_eps);
            },
            .Spherical => {
                // Standard sphere projection (weight affects subsequent scaling)
                project_spherical(slice, comp.epsilon);
                // Apply weight as scaling
                for (slice) |*val| {
                    val.* *= weight;
                }
                // Re-normalize to sphere
                project_spherical(slice, comp.epsilon);
            },
        }
    }
}

// ============================================================================
// Factory Functions
// ============================================================================

/// Create a standard Arabic-English code-switching configuration
pub fn createArabicEnglishConfig(total_dims: u32, allocator: std.mem.Allocator) !ProductManifoldConfig {
    const half = total_dims / 2;

    const components = try allocator.alloc(ManifoldComponent, 2);
    components[0] = ManifoldComponent{
        .manifold_type = .Hyperbolic, // Arabic - hierarchical morphology
        .dim_start = 0,
        .dim_end = half,
        .weight = 1.0,
        .curvature = -1.0, // Standard hyperbolic
        .epsilon = 1e-8,
    };
    components[1] = ManifoldComponent{
        .manifold_type = .Euclidean, // English - flatter semantic space
        .dim_start = half,
        .dim_end = total_dims,
        .weight = 1.0,
        .curvature = 0.0,
        .epsilon = 1e-8,
    };

    return ProductManifoldConfig{
        .components = components,
        .total_dims = total_dims,
        .code_switching_enabled = true,
        .arabic_dim_start = 0,
        .arabic_dim_end = half,
        .english_dim_start = half,
        .english_dim_end = total_dims,
    };
}

/// Create a triple product: Hyperbolic × Euclidean × Spherical
pub fn createTripleProductConfig(
    hyperbolic_dims: u32,
    euclidean_dims: u32,
    spherical_dims: u32,
    allocator: std.mem.Allocator,
) !ProductManifoldConfig {
    const components = try allocator.alloc(ManifoldComponent, 3);

    const hyp_end = hyperbolic_dims;
    const euc_end = hyp_end + euclidean_dims;
    const sph_end = euc_end + spherical_dims;

    components[0] = ManifoldComponent{
        .manifold_type = .Hyperbolic,
        .dim_start = 0,
        .dim_end = hyp_end,
        .weight = 1.0,
        .curvature = -1.0,
        .epsilon = 1e-8,
    };
    components[1] = ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = hyp_end,
        .dim_end = euc_end,
        .weight = 1.0,
        .curvature = 0.0,
        .epsilon = 1e-8,
    };
    components[2] = ManifoldComponent{
        .manifold_type = .Spherical,
        .dim_start = euc_end,
        .dim_end = sph_end,
        .weight = 1.0,
        .curvature = 1.0,
        .epsilon = 1e-8,
    };

    return ProductManifoldConfig{
        .components = components,
        .total_dims = sph_end,
        .code_switching_enabled = false,
        .arabic_dim_start = 0,
        .arabic_dim_end = 0,
        .english_dim_start = 0,
        .english_dim_end = 0,
    };
}


// ============================================================================
// Tests
// ============================================================================

test "ManifoldComponent validation" {
    const valid_euclidean = ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = 0,
        .dim_end = 64,
        .weight = 1.0,
        .curvature = 0.0,
    };
    try std.testing.expect(valid_euclidean.validate());
    try std.testing.expectEqual(@as(u32, 64), valid_euclidean.dims());

    const valid_hyperbolic = ManifoldComponent{
        .manifold_type = .Hyperbolic,
        .dim_start = 0,
        .dim_end = 32,
        .weight = 1.0,
        .curvature = -1.0, // Must be negative
    };
    try std.testing.expect(valid_hyperbolic.validate());

    const invalid_hyperbolic = ManifoldComponent{
        .manifold_type = .Hyperbolic,
        .dim_start = 0,
        .dim_end = 32,
        .curvature = 1.0, // Invalid: should be negative
    };
    try std.testing.expect(!invalid_hyperbolic.validate());

    const valid_spherical = ManifoldComponent{
        .manifold_type = .Spherical,
        .dim_start = 64,
        .dim_end = 128,
        .curvature = 1.0, // Must be positive
    };
    try std.testing.expect(valid_spherical.validate());
}

test "euclidean_distance" {
    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist = euclidean_distance(&x, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dist, 1e-5);
}

test "spherical_distance" {
    // Two orthogonal unit vectors on sphere
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const dist = spherical_distance(&x, &y, 1.0, 1e-8);
    // Should be π/2 for orthogonal vectors
    try std.testing.expectApproxEqAbs(math.pi / 2.0, dist, 1e-5);
}

test "hyperbolic_distance origin" {
    // Distance from origin in Poincaré ball
    const origin = [_]f32{ 0.0, 0.0 };
    const point = [_]f32{ 0.5, 0.0 };
    const dist = hyperbolic_distance(&origin, &point, -1.0, 1e-8);
    // Distance should be positive
    try std.testing.expect(dist > 0);
}

test "product_distance simple" {
    const components = [_]ManifoldComponent{
        ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 3,
            .weight = 1.0,
            .curvature = 0.0,
        },
    };

    const config = ProductManifoldConfig{
        .components = &components,
        .total_dims = 3,
    };

    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist = product_distance(&x, &y, config);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dist, 1e-5);
}

test "project_euclidean" {
    var point = [_]f32{ 6.0, 8.0 }; // norm = 10
    project_euclidean(&point, 5.0); // project to ball of radius 5
    const norm = @sqrt(point[0] * point[0] + point[1] * point[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 1e-5);
}

test "project_spherical" {
    var point = [_]f32{ 3.0, 4.0 }; // norm = 5
    project_spherical(&point, 1e-8);
    const norm = @sqrt(point[0] * point[0] + point[1] * point[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), norm, 1e-5);
}

test "project_hyperbolic" {
    var point = [_]f32{ 0.8, 0.6 }; // norm = 1.0, at boundary
    project_hyperbolic(&point, 1e-5);
    const norm = @sqrt(point[0] * point[0] + point[1] * point[1]);
    try std.testing.expect(norm < 1.0);
}

test "exp_euclidean" {
    const base = [_]f32{ 1.0, 2.0 };
    const v = [_]f32{ 0.5, -0.5 };
    var result: [2]f32 = undefined;
    exp_euclidean(&base, &v, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), result[1], 1e-5);
}

test "log_euclidean" {
    const base = [_]f32{ 1.0, 2.0 };
    const y = [_]f32{ 1.5, 1.5 };
    var result: [2]f32 = undefined;
    log_euclidean(&base, &y, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), result[1], 1e-5);
}

test "CodeSwitchContext defaults" {
    const ctx = CodeSwitchContext{
        .arabic_ratio = 0.6,
    };
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), ctx.arabic_ratio, 1e-5);
    try std.testing.expect(ctx.apply_constraints);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1), ctx.transition_smoothing, 1e-5);
}

test "ManifoldType names" {
    try std.testing.expectEqualStrings("Euclidean", ManifoldType.Euclidean.getName());
    try std.testing.expectEqualStrings("Hyperbolic", ManifoldType.Hyperbolic.getName());
    try std.testing.expectEqualStrings("Spherical", ManifoldType.Spherical.getName());
}

test "ProductManifoldConfig validation" {
    const components = [_]ManifoldComponent{
        ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
        ManifoldComponent{
            .manifold_type = .Hyperbolic,
            .dim_start = 32,
            .dim_end = 64,
            .weight = 0.5,
            .curvature = -1.0,
        },
    };

    const config = ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };

    try std.testing.expect(config.validate());
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), config.totalWeight(), 1e-5);

    const comp = config.getComponentForDim(40);
    try std.testing.expect(comp != null);
    try std.testing.expectEqual(ManifoldType.Hyperbolic, comp.?.manifold_type);
}