// mHC Geometry Detector - Day 58
// Automatic Geometry Detection using Ollivier-Ricci Curvature
//
// Features:
// 1. k-NN Graph Construction - Build nearest neighbor graph with SIMD distances
// 2. Ollivier-Ricci Curvature - Estimate discrete curvature from k-NN graph
// 3. Curvature-based Classification - Classify as Euclidean/Hyperbolic/Spherical
// 4. Confidence Scoring - Score confidence of geometry detection
// 5. Auto-detection Pipeline - Automatically select best manifold type
//
// Reference: mhc_product_manifold.zig for ManifoldType

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// Import ManifoldType from product manifold module
const product_manifold = @import("mhc_product_manifold.zig");
pub const ManifoldType = product_manifold.ManifoldType;

// ============================================================================
// Constants and Configuration
// ============================================================================

/// SIMD vector width for distance computations
const SIMD_WIDTH: usize = 8;
const Vec8 = @Vector(SIMD_WIDTH, f32);

/// Classification thresholds for curvature
pub const CurvatureThresholds = struct {
    /// Curvature below this is Hyperbolic (negative curvature)
    hyperbolic_threshold: f32 = -0.1,
    /// Curvature above this is Spherical (positive curvature)
    spherical_threshold: f32 = 0.1,
    /// Minimum confidence to trust detection
    min_confidence: f32 = 0.5,
};

/// Configuration for geometry detection
pub const GeometryDetectorConfig = struct {
    /// Number of nearest neighbors for k-NN graph
    k_neighbors: u32 = 10,
    /// Number of edges to sample for curvature estimation
    sample_edges: u32 = 100,
    /// Numerical stability epsilon
    epsilon: f32 = 1e-8,
    /// Curvature classification thresholds
    thresholds: CurvatureThresholds = .{},
    /// Random seed for edge sampling
    random_seed: u64 = 42,
};

// ============================================================================
// Core Types
// ============================================================================

/// Result of geometry detection
pub const GeometryDetectionResult = struct {
    /// Detected manifold type
    manifold_type: ManifoldType,
    /// Mean Ollivier-Ricci curvature
    mean_curvature: f32,
    /// Standard deviation of curvature
    std_curvature: f32,
    /// Confidence score [0, 1]
    confidence: f32,
    /// Number of edges sampled
    edges_sampled: u32,
    /// Computation time in nanoseconds
    computation_time_ns: i64,

    /// Check if detection is reliable
    pub fn isReliable(self: GeometryDetectionResult, min_confidence: f32) bool {
        return self.confidence >= min_confidence;
    }

    /// Get recommended curvature for the detected manifold
    pub fn getRecommendedCurvature(self: GeometryDetectionResult) f32 {
        return switch (self.manifold_type) {
            .Euclidean => 0.0,
            .Hyperbolic => @min(self.mean_curvature, -0.1),
            .Spherical => @max(self.mean_curvature, 0.1),
        };
    }
};

/// Edge in k-NN graph
pub const KNNEdge = struct {
    /// Source point index
    source: u32,
    /// Target point index
    target: u32,
    /// Distance between points
    distance: f32,
};

/// k-NN Graph representation
pub const KNNGraph = struct {
    /// Number of points
    num_points: u32,
    /// Number of neighbors per point
    k: u32,
    /// Neighbor indices [num_points * k]
    neighbors: []u32,
    /// Distances to neighbors [num_points * k]
    distances: []f32,
    /// Allocator used
    allocator: Allocator,

    /// Get neighbors of a point
    pub fn getNeighbors(self: *const KNNGraph, point_idx: u32) []const u32 {
        const start = point_idx * self.k;
        const end = start + self.k;
        return self.neighbors[start..end];
    }

    /// Get distances to neighbors of a point
    pub fn getDistances(self: *const KNNGraph, point_idx: u32) []const f32 {
        const start = point_idx * self.k;
        const end = start + self.k;
        return self.distances[start..end];
    }

    /// Free graph memory
    pub fn deinit(self: *KNNGraph) void {
        self.allocator.free(self.neighbors);
        self.allocator.free(self.distances);
    }
};

// ============================================================================
// SIMD Distance Computation
// ============================================================================

/// Compute squared Euclidean distance with SIMD optimization
pub fn squared_distance_simd(x: []const f32, y: []const f32) f32 {
    const n = @min(x.len, y.len);
    var sum: f32 = 0.0;

    // SIMD path
    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const x_vec: Vec8 = x[i..][0..SIMD_WIDTH].*;
        const y_vec: Vec8 = y[i..][0..SIMD_WIDTH].*;
        const diff = x_vec - y_vec;
        const sq = diff * diff;
        sum += @reduce(.Add, sq);
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        const diff = x[i] - y[i];
        sum += diff * diff;
    }

    return sum;
}

/// Compute Euclidean distance
pub fn euclidean_distance(x: []const f32, y: []const f32) f32 {
    return @sqrt(squared_distance_simd(x, y));
}

// ============================================================================
// k-NN Graph Construction
// ============================================================================

/// Build k-nearest neighbors graph from point cloud
/// Uses brute-force search with SIMD distance computation
///
/// Parameters:
///   - points: Point cloud as flat array [num_points * dim]
///   - num_points: Number of points
///   - dim: Dimension of each point
///   - k: Number of neighbors per point
///   - allocator: Memory allocator
///
/// Returns: KNNGraph structure
pub fn build_knn_graph(
    points: []const f32,
    num_points: u32,
    dim: u32,
    k: u32,
    allocator: Allocator,
) !KNNGraph {
    std.debug.assert(points.len == num_points * dim);
    std.debug.assert(k < num_points);

    // Allocate output arrays
    const neighbors = try allocator.alloc(u32, num_points * k);
    errdefer allocator.free(neighbors);
    const distances = try allocator.alloc(f32, num_points * k);
    errdefer allocator.free(distances);

    // Temporary buffer for all distances from one point
    const temp_dists = try allocator.alloc(f32, num_points);
    defer allocator.free(temp_dists);
    const temp_indices = try allocator.alloc(u32, num_points);
    defer allocator.free(temp_indices);

    // For each point, find k nearest neighbors
    for (0..num_points) |i| {
        const point_i = points[i * dim .. (i + 1) * dim];

        // Compute distances to all other points
        for (0..num_points) |j| {
            temp_indices[j] = @intCast(j);
            if (i == j) {
                temp_dists[j] = math.floatMax(f32); // Exclude self
            } else {
                const point_j = points[j * dim .. (j + 1) * dim];
                temp_dists[j] = euclidean_distance(point_i, point_j);
            }
        }

        // Partial sort to find k smallest
        // Simple selection for now - could use quickselect for large n
        for (0..k) |kk| {
            var min_idx: usize = kk;
            var min_dist = temp_dists[kk];
            for (kk + 1..num_points) |j| {
                if (temp_dists[j] < min_dist) {
                    min_dist = temp_dists[j];
                    min_idx = j;
                }
            }
            // Swap
            if (min_idx != kk) {
                const tmp_d = temp_dists[kk];
                const tmp_i = temp_indices[kk];
                temp_dists[kk] = temp_dists[min_idx];
                temp_indices[kk] = temp_indices[min_idx];
                temp_dists[min_idx] = tmp_d;
                temp_indices[min_idx] = tmp_i;
            }
        }

        // Copy k nearest to output
        const out_start = i * k;
        for (0..k) |kk| {
            neighbors[out_start + kk] = temp_indices[kk];
            distances[out_start + kk] = temp_dists[kk];
        }
    }

    return KNNGraph{
        .num_points = num_points,
        .k = k,
        .neighbors = neighbors,
        .distances = distances,
        .allocator = allocator,
    };
}

// ============================================================================
// Ollivier-Ricci Curvature Computation
// ============================================================================

/// Compute 1-Wasserstein distance between uniform distributions on neighbors
/// This is the core of Ollivier-Ricci curvature estimation
fn wasserstein_neighbor_distance(
    graph: *const KNNGraph,
    points: []const f32,
    dim: u32,
    i: u32,
    j: u32,
    epsilon: f32,
) f32 {
    const neighbors_i = graph.getNeighbors(i);
    const neighbors_j = graph.getNeighbors(j);

    // Simplified Wasserstein: average distance between neighbor sets
    // Full optimal transport would require solving assignment problem
    var total_dist: f32 = 0.0;
    var count: u32 = 0;

    for (neighbors_i) |ni| {
        const point_ni = points[ni * dim .. (ni + 1) * dim];
        var min_dist: f32 = math.floatMax(f32);

        for (neighbors_j) |nj| {
            const point_nj = points[nj * dim .. (nj + 1) * dim];
            const dist = euclidean_distance(point_ni, point_nj);
            min_dist = @min(min_dist, dist);
        }

        if (min_dist < math.floatMax(f32)) {
            total_dist += min_dist;
            count += 1;
        }
    }

    if (count == 0) return 0.0;
    return total_dist / @as(f32, @floatFromInt(count)) + epsilon;
}

/// Compute Ollivier-Ricci curvature for a single edge
/// κ(x, y) = 1 - W(μ_x, μ_y) / d(x, y)
/// where W is Wasserstein distance and μ_x is uniform on neighbors of x
pub fn compute_edge_curvature(
    graph: *const KNNGraph,
    points: []const f32,
    dim: u32,
    source: u32,
    target: u32,
    config: GeometryDetectorConfig,
) f32 {
    const point_s = points[source * dim .. (source + 1) * dim];
    const point_t = points[target * dim .. (target + 1) * dim];
    const edge_dist = euclidean_distance(point_s, point_t);

    if (edge_dist < config.epsilon) return 0.0;

    const wasserstein = wasserstein_neighbor_distance(
        graph,
        points,
        dim,
        source,
        target,
        config.epsilon,
    );

    // Ollivier-Ricci curvature formula
    return 1.0 - wasserstein / edge_dist;
}

/// Compute mean Ollivier-Ricci curvature over sampled edges
pub fn compute_ollivier_ricci(
    graph: *const KNNGraph,
    points: []const f32,
    dim: u32,
    config: GeometryDetectorConfig,
) struct { mean: f32, std: f32, count: u32 } {
    const num_points = graph.num_points;
    const k = graph.k;

    // Collect all edges
    var curvatures: [1024]f32 = undefined; // Max edges to sample
    var count: u32 = 0;

    // Sample edges deterministically based on seed
    var prng = std.Random.DefaultPrng.init(config.random_seed);
    const random = prng.random();

    const max_edges = @min(config.sample_edges, num_points * k);

    for (0..max_edges) |_| {
        if (count >= 1024) break;

        const i = random.intRangeAtMost(u32, 0, num_points - 1);
        const neighbor_idx = random.intRangeAtMost(u32, 0, k - 1);
        const j = graph.neighbors[i * k + neighbor_idx];

        const curv = compute_edge_curvature(graph, points, dim, i, j, config);
        if (!math.isNan(curv) and !math.isInf(curv)) {
            curvatures[count] = curv;
            count += 1;
        }
    }

    if (count == 0) {
        return .{ .mean = 0.0, .std = 0.0, .count = 0 };
    }

    // Compute mean
    var sum: f32 = 0.0;
    for (0..count) |i| {
        sum += curvatures[i];
    }
    const mean = sum / @as(f32, @floatFromInt(count));

    // Compute std
    var variance_sum: f32 = 0.0;
    for (0..count) |i| {
        const diff = curvatures[i] - mean;
        variance_sum += diff * diff;
    }
    const variance = variance_sum / @as(f32, @floatFromInt(count));
    const std_dev = @sqrt(variance);

    return .{ .mean = mean, .std = std_dev, .count = count };
}

// ============================================================================
// Geometry Classification
// ============================================================================

/// Classify geometry based on mean curvature
pub fn classify_geometry(mean_curvature: f32, thresholds: CurvatureThresholds) ManifoldType {
    if (mean_curvature < thresholds.hyperbolic_threshold) {
        return .Hyperbolic;
    } else if (mean_curvature > thresholds.spherical_threshold) {
        return .Spherical;
    } else {
        return .Euclidean;
    }
}

/// Compute confidence score for detection
/// Higher confidence when curvature is far from thresholds and std is low
pub fn get_detection_confidence(
    mean_curvature: f32,
    std_curvature: f32,
    manifold_type: ManifoldType,
    thresholds: CurvatureThresholds,
) f32 {
    // Distance from decision boundary
    const boundary_dist = switch (manifold_type) {
        .Hyperbolic => @abs(mean_curvature - thresholds.hyperbolic_threshold),
        .Spherical => @abs(mean_curvature - thresholds.spherical_threshold),
        .Euclidean => @min(
            @abs(mean_curvature - thresholds.hyperbolic_threshold),
            @abs(mean_curvature - thresholds.spherical_threshold),
        ),
    };

    // Normalize boundary distance (sigmoid-like)
    const boundary_conf = boundary_dist / (boundary_dist + 0.1);

    // Low std means consistent curvature -> higher confidence
    const std_conf = 1.0 / (1.0 + std_curvature);

    // Combined confidence
    return boundary_conf * std_conf;
}

// ============================================================================
// Full Auto-Detection Pipeline
// ============================================================================

/// Perform full geometry auto-detection on point cloud
///
/// This is the main entry point for automatic geometry detection.
/// It builds a k-NN graph, computes Ollivier-Ricci curvature,
/// classifies the geometry, and returns confidence scores.
///
/// Parameters:
///   - points: Point cloud as flat array [num_points * dim]
///   - num_points: Number of points
///   - dim: Dimension of each point
///   - config: Detection configuration
///   - allocator: Memory allocator
///
/// Returns: GeometryDetectionResult with manifold type and confidence
pub fn detect_geometry(
    points: []const f32,
    num_points: u32,
    dim: u32,
    config: GeometryDetectorConfig,
    allocator: Allocator,
) !GeometryDetectionResult {
    const start_time = std.time.nanoTimestamp();

    // Validate inputs
    if (num_points < config.k_neighbors + 1) {
        return error.InsufficientPoints;
    }
    if (points.len != num_points * dim) {
        return error.InvalidPointCloud;
    }

    // Step 1: Build k-NN graph
    var graph = try build_knn_graph(
        points,
        num_points,
        dim,
        config.k_neighbors,
        allocator,
    );
    defer graph.deinit();

    // Step 2: Compute Ollivier-Ricci curvature
    const curvature_result = compute_ollivier_ricci(&graph, points, dim, config);

    // Step 3: Classify geometry
    const manifold_type = classify_geometry(curvature_result.mean, config.thresholds);

    // Step 4: Compute confidence
    const confidence = get_detection_confidence(
        curvature_result.mean,
        curvature_result.std,
        manifold_type,
        config.thresholds,
    );

    const end_time = std.time.nanoTimestamp();

    return GeometryDetectionResult{
        .manifold_type = manifold_type,
        .mean_curvature = curvature_result.mean,
        .std_curvature = curvature_result.std,
        .confidence = confidence,
        .edges_sampled = curvature_result.count,
        .computation_time_ns = @intCast(end_time - start_time),
    };
}

/// Detect geometry with default configuration
pub fn detect_geometry_auto(
    points: []const f32,
    num_points: u32,
    dim: u32,
    allocator: Allocator,
) !GeometryDetectionResult {
    return detect_geometry(points, num_points, dim, .{}, allocator);
}

// ============================================================================
// Helper Functions for Point Generation
// ============================================================================

/// Generate points on a sphere (positive curvature)
fn generate_spherical_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        // Generate random direction
        for (0..dim) |d| {
            const val = random.floatNorm(f32);
            buffer[i * dim + d] = val;
            norm_sq += val * val;
        }
        // Normalize to unit sphere
        const norm = @sqrt(norm_sq);
        if (norm > 1e-8) {
            for (0..dim) |d| {
                buffer[i * dim + d] /= norm;
            }
        }
    }
}

/// Generate points in Euclidean space (zero curvature)
fn generate_euclidean_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        for (0..dim) |d| {
            // Uniform distribution in [-1, 1]
            buffer[i * dim + d] = random.float(f32) * 2.0 - 1.0;
        }
    }
}

/// Generate points in hyperbolic-like space (negative curvature)
/// Simulates Poincaré ball with concentration toward boundary
fn generate_hyperbolic_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        // Generate random direction
        for (0..dim) |d| {
            const val = random.floatNorm(f32);
            buffer[i * dim + d] = val;
            norm_sq += val * val;
        }
        // Normalize and scale toward boundary (r ~ 0.8-0.95)
        const norm = @sqrt(norm_sq);
        const target_radius = 0.8 + random.float(f32) * 0.15;
        if (norm > 1e-8) {
            for (0..dim) |d| {
                buffer[i * dim + d] = buffer[i * dim + d] * target_radius / norm;
            }
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

test "squared_distance_simd basic" {
    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist_sq = squared_distance_simd(&x, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), dist_sq, 1e-5);
}

test "euclidean_distance basic" {
    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist = euclidean_distance(&x, &y);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), dist, 1e-5);
}

test "build_knn_graph small" {
    const allocator = std.testing.allocator;

    // 4 points in 2D forming a square
    const points = [_]f32{
        0.0, 0.0, // point 0
        1.0, 0.0, // point 1
        1.0, 1.0, // point 2
        0.0, 1.0, // point 3
    };

    var graph = try build_knn_graph(&points, 4, 2, 2, allocator);
    defer graph.deinit();

    try std.testing.expectEqual(@as(u32, 4), graph.num_points);
    try std.testing.expectEqual(@as(u32, 2), graph.k);

    // Each point should have 2 neighbors
    const neighbors_0 = graph.getNeighbors(0);
    try std.testing.expectEqual(@as(usize, 2), neighbors_0.len);
}

test "classify_geometry thresholds" {
    const thresholds = CurvatureThresholds{};

    try std.testing.expectEqual(ManifoldType.Hyperbolic, classify_geometry(-0.5, thresholds));
    try std.testing.expectEqual(ManifoldType.Spherical, classify_geometry(0.5, thresholds));
    try std.testing.expectEqual(ManifoldType.Euclidean, classify_geometry(0.0, thresholds));
    try std.testing.expectEqual(ManifoldType.Euclidean, classify_geometry(0.05, thresholds));
    try std.testing.expectEqual(ManifoldType.Euclidean, classify_geometry(-0.05, thresholds));
}

test "get_detection_confidence varies with curvature" {
    const thresholds = CurvatureThresholds{};

    // Strong hyperbolic signal should have high confidence
    const conf_strong = get_detection_confidence(-0.5, 0.1, .Hyperbolic, thresholds);
    // Weak hyperbolic signal should have lower confidence
    const conf_weak = get_detection_confidence(-0.12, 0.1, .Hyperbolic, thresholds);

    try std.testing.expect(conf_strong > conf_weak);
}

test "get_detection_confidence varies with std" {
    const thresholds = CurvatureThresholds{};

    // Low std should have higher confidence
    const conf_low_std = get_detection_confidence(-0.5, 0.05, .Hyperbolic, thresholds);
    const conf_high_std = get_detection_confidence(-0.5, 0.5, .Hyperbolic, thresholds);

    try std.testing.expect(conf_low_std > conf_high_std);
}

test "GeometryDetectionResult isReliable" {
    const result = GeometryDetectionResult{
        .manifold_type = .Hyperbolic,
        .mean_curvature = -0.5,
        .std_curvature = 0.1,
        .confidence = 0.7,
        .edges_sampled = 50,
        .computation_time_ns = 1000,
    };

    try std.testing.expect(result.isReliable(0.5));
    try std.testing.expect(!result.isReliable(0.8));
}

test "GeometryDetectionResult getRecommendedCurvature" {
    const hyperbolic = GeometryDetectionResult{
        .manifold_type = .Hyperbolic,
        .mean_curvature = -0.3,
        .std_curvature = 0.1,
        .confidence = 0.7,
        .edges_sampled = 50,
        .computation_time_ns = 1000,
    };
    try std.testing.expect(hyperbolic.getRecommendedCurvature() < 0);

    const spherical = GeometryDetectionResult{
        .manifold_type = .Spherical,
        .mean_curvature = 0.3,
        .std_curvature = 0.1,
        .confidence = 0.7,
        .edges_sampled = 50,
        .computation_time_ns = 1000,
    };
    try std.testing.expect(spherical.getRecommendedCurvature() > 0);

    const euclidean = GeometryDetectionResult{
        .manifold_type = .Euclidean,
        .mean_curvature = 0.0,
        .std_curvature = 0.1,
        .confidence = 0.7,
        .edges_sampled = 50,
        .computation_time_ns = 1000,
    };
    try std.testing.expectEqual(@as(f32, 0.0), euclidean.getRecommendedCurvature());
}

test "detect_geometry insufficient points" {
    const allocator = std.testing.allocator;
    const points = [_]f32{ 0.0, 0.0, 1.0, 1.0 }; // Only 2 points

    const config = GeometryDetectorConfig{ .k_neighbors = 5 };

    const result = detect_geometry(&points, 2, 2, config, allocator);
    try std.testing.expectError(error.InsufficientPoints, result);
}

test "detect_geometry full pipeline euclidean" {
    const allocator = std.testing.allocator;

    // Generate 20 points in 3D Euclidean space
    const num_points: u32 = 20;
    const dim: u32 = 3;
    var points: [num_points * dim]f32 = undefined;
    generate_euclidean_points(&points, num_points, dim, 12345);

    const config = GeometryDetectorConfig{
        .k_neighbors = 5,
        .sample_edges = 50,
    };

    const result = try detect_geometry(&points, num_points, dim, config, allocator);

    // Should detect some geometry (may not be perfectly Euclidean due to randomness)
    try std.testing.expect(result.edges_sampled > 0);
    try std.testing.expect(result.computation_time_ns > 0);
    try std.testing.expect(result.confidence >= 0.0 and result.confidence <= 1.0);
}

test "detect_geometry full pipeline spherical" {
    const allocator = std.testing.allocator;

    // Generate 20 points on a sphere
    const num_points: u32 = 20;
    const dim: u32 = 3;
    var points: [num_points * dim]f32 = undefined;
    generate_spherical_points(&points, num_points, dim, 54321);

    const config = GeometryDetectorConfig{
        .k_neighbors = 5,
        .sample_edges = 50,
    };

    const result = try detect_geometry(&points, num_points, dim, config, allocator);

    try std.testing.expect(result.edges_sampled > 0);
    try std.testing.expect(!math.isNan(result.mean_curvature));
    try std.testing.expect(!math.isNan(result.std_curvature));
}

test "detect_geometry_auto convenience function" {
    const allocator = std.testing.allocator;

    const num_points: u32 = 15;
    const dim: u32 = 4;
    var points: [num_points * dim]f32 = undefined;
    generate_euclidean_points(&points, num_points, dim, 99999);

    const result = try detect_geometry_auto(&points, num_points, dim, allocator);

    try std.testing.expect(result.edges_sampled > 0);
}

test "compute_edge_curvature basic" {
    const allocator = std.testing.allocator;

    // Simple 4-point setup
    const points = [_]f32{
        0.0, 0.0, // point 0
        1.0, 0.0, // point 1
        1.0, 1.0, // point 2
        0.0, 1.0, // point 3
    };

    var graph = try build_knn_graph(&points, 4, 2, 2, allocator);
    defer graph.deinit();

    const config = GeometryDetectorConfig{};
    const curv = compute_edge_curvature(&graph, &points, 2, 0, 1, config);

    try std.testing.expect(!math.isNan(curv));
    try std.testing.expect(!math.isInf(curv));
}

test "compute_ollivier_ricci returns valid stats" {
    const allocator = std.testing.allocator;

    const num_points: u32 = 10;
    const dim: u32 = 2;
    var points: [num_points * dim]f32 = undefined;
    generate_euclidean_points(&points, num_points, dim, 11111);

    var graph = try build_knn_graph(&points, num_points, dim, 3, allocator);
    defer graph.deinit();

    const config = GeometryDetectorConfig{ .sample_edges = 20 };
    const result = compute_ollivier_ricci(&graph, &points, dim, config);

    try std.testing.expect(result.count > 0);
    try std.testing.expect(!math.isNan(result.mean));
    try std.testing.expect(!math.isNan(result.std));
    try std.testing.expect(result.std >= 0.0);
}

test "KNNGraph getters" {
    const allocator = std.testing.allocator;

    const points = [_]f32{
        0.0, 0.0,
        1.0, 0.0,
        2.0, 0.0,
        3.0, 0.0,
    };

    var graph = try build_knn_graph(&points, 4, 2, 2, allocator);
    defer graph.deinit();

    const neighbors = graph.getNeighbors(1);
    const distances = graph.getDistances(1);

    try std.testing.expectEqual(@as(usize, 2), neighbors.len);
    try std.testing.expectEqual(@as(usize, 2), distances.len);

    // Point 1 at (1,0) should have neighbors at (0,0) and (2,0)
    // Both at distance 1.0
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distances[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), distances[1], 1e-5);
}

test "squared_distance_simd with SIMD width data" {
    // Test with data length >= SIMD_WIDTH (8)
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0 };
    const y = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };

    const dist_sq = squared_distance_simd(&x, &y);
    // Expected: 1 + 4 + 9 + 16 + 25 + 36 + 49 + 64 + 81 + 100 = 385
    try std.testing.expectApproxEqAbs(@as(f32, 385.0), dist_sq, 1e-5);
}

test "generate_spherical_points normalized" {
    const num_points: u32 = 5;
    const dim: u32 = 3;
    var points: [num_points * dim]f32 = undefined;
    generate_spherical_points(&points, num_points, dim, 42);

    // All points should have unit norm
    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        for (0..dim) |d| {
            const val = points[i * dim + d];
            norm_sq += val * val;
        }
        try std.testing.expectApproxEqAbs(@as(f32, 1.0), @sqrt(norm_sq), 1e-5);
    }
}

test "generate_hyperbolic_points in ball" {
    const num_points: u32 = 5;
    const dim: u32 = 3;
    var points: [num_points * dim]f32 = undefined;
    generate_hyperbolic_points(&points, num_points, dim, 42);

    // All points should have norm < 1 (inside Poincaré ball)
    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        for (0..dim) |d| {
            const val = points[i * dim + d];
            norm_sq += val * val;
        }
        try std.testing.expect(@sqrt(norm_sq) < 1.0);
    }
}

test "CurvatureThresholds default values" {
    const thresholds = CurvatureThresholds{};
    try std.testing.expectEqual(@as(f32, -0.1), thresholds.hyperbolic_threshold);
    try std.testing.expectEqual(@as(f32, 0.1), thresholds.spherical_threshold);
    try std.testing.expectEqual(@as(f32, 0.5), thresholds.min_confidence);
}

test "GeometryDetectorConfig default values" {
    const config = GeometryDetectorConfig{};
    try std.testing.expectEqual(@as(u32, 10), config.k_neighbors);
    try std.testing.expectEqual(@as(u32, 100), config.sample_edges);
    try std.testing.expectEqual(@as(f32, 1e-8), config.epsilon);
}
