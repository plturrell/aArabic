// mHC Uncertainty Quantification - Day 61
// Uncertainty-aware geometry detection using bootstrap resampling
//
// Features:
// 1. Bootstrap Resampling - Estimate curvature distribution via resampling
// 2. Confidence Intervals - Percentile-based confidence interval computation
// 3. Vote-based Classification - Majority voting across bootstrap samples
// 4. Uncertainty Metrics - Quantify detection reliability
//
// Reference: mhc_geometry_detector.zig for base geometry detection

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// Import from geometry detector module
const geometry_detector = @import("mhc_geometry_detector.zig");
pub const ManifoldType = geometry_detector.ManifoldType;
pub const CurvatureThresholds = geometry_detector.CurvatureThresholds;
pub const GeometryDetectorConfig = geometry_detector.GeometryDetectorConfig;
pub const GeometryDetectionResult = geometry_detector.GeometryDetectionResult;
pub const classify_geometry = geometry_detector.classify_geometry;

// ============================================================================
// Constants and Configuration
// ============================================================================

/// Maximum number of bootstrap samples supported
const MAX_BOOTSTRAP_SAMPLES: usize = 500;

/// Maximum number of points for bootstrap resampling
const MAX_POINTS: usize = 10000;

/// Default random seed for reproducibility
const DEFAULT_SEED: u64 = 42;

// ============================================================================
// Core Configuration Types
// ============================================================================

/// Configuration for uncertainty-aware geometry detection
pub const UncertaintyConfig = struct {
    /// Number of bootstrap resamples to generate
    bootstrap_samples: usize = 100,
    /// Confidence level for interval computation (0.0 to 1.0)
    confidence_level: f32 = 0.95,
    /// Detection threshold for vote confidence
    detection_threshold: f32 = 0.6,
    /// Random seed for reproducible resampling
    random_seed: u64 = DEFAULT_SEED,
    /// Minimum samples required for valid statistics
    min_valid_samples: usize = 10,
    /// Base geometry detector configuration
    geometry_config: GeometryDetectorConfig = .{},
};

/// Confidence interval representation
pub const ConfidenceInterval = struct {
    /// Lower bound of the interval
    lower: f32,
    /// Upper bound of the interval
    upper: f32,
    /// Point estimate (mean)
    mean: f32,
    /// Standard deviation
    std: f32,
    /// Confidence level used
    confidence: f32,
    /// Number of valid samples
    sample_count: usize,

    /// Check if a value falls within the confidence interval
    pub fn contains(self: ConfidenceInterval, value: f32) bool {
        return value >= self.lower and value <= self.upper;
    }

    /// Get the width of the confidence interval
    pub fn width(self: ConfidenceInterval) f32 {
        return self.upper - self.lower;
    }

    /// Check if interval is narrow (high precision)
    pub fn isNarrow(self: ConfidenceInterval, threshold: f32) bool {
        return self.width() < threshold;
    }
};

/// Vote result for geometry classification
pub const VoteResult = struct {
    /// Winning manifold type
    manifold_type: ManifoldType,
    /// Proportion of votes for winning type (0.0 to 1.0)
    vote_proportion: f32,
    /// Number of votes for Euclidean
    euclidean_votes: u32,
    /// Number of votes for Hyperbolic
    hyperbolic_votes: u32,
    /// Number of votes for Spherical
    spherical_votes: u32,
    /// Total votes cast
    total_votes: u32,

    /// Check if vote is confident (above threshold)
    pub fn isConfident(self: VoteResult, threshold: f32) bool {
        return self.vote_proportion >= threshold;
    }

    /// Get vote proportions for all types
    pub fn getProportions(self: VoteResult) struct { euclidean: f32, hyperbolic: f32, spherical: f32 } {
        const total = @as(f32, @floatFromInt(self.total_votes));
        if (total < 1.0) {
            return .{ .euclidean = 0.0, .hyperbolic = 0.0, .spherical = 0.0 };
        }
        return .{
            .euclidean = @as(f32, @floatFromInt(self.euclidean_votes)) / total,
            .hyperbolic = @as(f32, @floatFromInt(self.hyperbolic_votes)) / total,
            .spherical = @as(f32, @floatFromInt(self.spherical_votes)) / total,
        };
    }
};

/// Full uncertainty quantification result
pub const UncertaintyResult = struct {
    /// Vote-based classification result
    vote_result: VoteResult,
    /// Confidence interval for mean curvature
    curvature_ci: ConfidenceInterval,
    /// Bootstrap curvature samples (for diagnostics)
    bootstrap_means: []f32,
    /// Whether detection is reliable
    is_reliable: bool,
    /// Computation time in nanoseconds
    computation_time_ns: i64,
    /// Allocator used (for freeing bootstrap_means)
    allocator: Allocator,

    /// Free allocated memory
    pub fn deinit(self: *UncertaintyResult) void {
        self.allocator.free(self.bootstrap_means);
    }

    /// Get recommended geometry type with uncertainty awareness
    pub fn getRecommendedGeometry(self: UncertaintyResult) ?ManifoldType {
        if (!self.is_reliable) return null;
        return self.vote_result.manifold_type;
    }

    /// Compute entropy of vote distribution (uncertainty measure)
    pub fn getVoteEntropy(self: UncertaintyResult) f32 {
        const props = self.vote_result.getProportions();
        var entropy: f32 = 0.0;

        // H = -sum(p * log(p))
        inline for ([_]f32{ props.euclidean, props.hyperbolic, props.spherical }) |p| {
            if (p > 1e-10) {
                entropy -= p * @log(p);
            }
        }

        // Normalize by max entropy (log(3) for 3 classes)
        return entropy / @log(@as(f32, 3.0));
    }
};

// ============================================================================
// UncertaintyAwareGeometryDetector
// ============================================================================

/// Uncertainty-aware geometry detector using bootstrap resampling
/// This extends the base geometry detector with statistical uncertainty quantification
pub const UncertaintyAwareGeometryDetector = struct {
    /// Configuration
    config: UncertaintyConfig,
    /// Memory allocator
    allocator: Allocator,
    /// PRNG state
    prng: std.Random.DefaultPrng,

    const Self = @This();

    /// Initialize a new uncertainty-aware detector
    pub fn init(allocator: Allocator, config: UncertaintyConfig) Self {
        return .{
            .config = config,
            .allocator = allocator,
            .prng = std.Random.DefaultPrng.init(config.random_seed),
        };
    }

    /// Reset PRNG to initial seed (for reproducibility)
    pub fn resetSeed(self: *Self) void {
        self.prng = std.Random.DefaultPrng.init(self.config.random_seed);
    }

    /// Detect geometry with uncertainty quantification
    pub fn detectWithUncertainty(
        self: *Self,
        points: []const f32,
        num_points: u32,
        dim: u32,
    ) !UncertaintyResult {
        const start_time = std.time.nanoTimestamp();

        // Validate inputs
        if (num_points < self.config.geometry_config.k_neighbors + 1) {
            return error.InsufficientPoints;
        }
        if (points.len != num_points * dim) {
            return error.InvalidPointCloud;
        }

        // Allocate bootstrap results
        const bootstrap_count = @min(self.config.bootstrap_samples, MAX_BOOTSTRAP_SAMPLES);
        const bootstrap_means = try self.allocator.alloc(f32, bootstrap_count);
        errdefer self.allocator.free(bootstrap_means);

        // Perform bootstrap resampling
        const bootstrap_result = try self.bootstrapCurvature(
            points,
            num_points,
            dim,
            bootstrap_count,
            bootstrap_means,
        );

        // Compute confidence interval
        const curvature_ci = self.computeConfidenceInterval(
            bootstrap_means[0..bootstrap_result.valid_samples],
            self.config.confidence_level,
        );

        // Perform vote-based classification
        const vote_result = self.voteClassification(
            bootstrap_means[0..bootstrap_result.valid_samples],
        );

        // Determine reliability
        const is_reliable = vote_result.isConfident(self.config.detection_threshold) and
            curvature_ci.sample_count >= self.config.min_valid_samples;

        const end_time = std.time.nanoTimestamp();

        return UncertaintyResult{
            .vote_result = vote_result,
            .curvature_ci = curvature_ci,
            .bootstrap_means = bootstrap_means,
            .is_reliable = is_reliable,
            .computation_time_ns = @intCast(end_time - start_time),
            .allocator = self.allocator,
        };
    }

    /// Perform bootstrap resampling to estimate curvature distribution
    fn bootstrapCurvature(
        self: *Self,
        points: []const f32,
        num_points: u32,
        dim: u32,
        n_samples: usize,
        output: []f32,
    ) !struct { valid_samples: usize, total_attempts: usize } {
        const random = self.prng.random();

        // Allocate temporary buffer for resampled indices
        const resample_indices = try self.allocator.alloc(u32, num_points);
        defer self.allocator.free(resample_indices);

        // Allocate temporary buffer for resampled points
        const resampled_points = try self.allocator.alloc(f32, num_points * dim);
        defer self.allocator.free(resampled_points);

        var valid_count: usize = 0;

        for (0..n_samples) |sample_idx| {
            // Generate bootstrap sample indices (with replacement)
            for (0..num_points) |i| {
                resample_indices[i] = random.intRangeAtMost(u32, 0, num_points - 1);
            }

            // Copy resampled points
            for (0..num_points) |i| {
                const src_idx = resample_indices[i];
                const src_start = src_idx * dim;
                const dst_start = i * dim;
                for (0..dim) |d| {
                    resampled_points[dst_start + d] = points[src_start + d];
                }
            }

            // Detect geometry on resampled data
            const result = geometry_detector.detect_geometry(
                resampled_points,
                num_points,
                dim,
                self.config.geometry_config,
                self.allocator,
            ) catch {
                continue; // Skip failed samples
            };

            if (!math.isNan(result.mean_curvature) and !math.isInf(result.mean_curvature)) {
                output[valid_count] = result.mean_curvature;
                valid_count += 1;
            }

            _ = sample_idx;
        }

        return .{ .valid_samples = valid_count, .total_attempts = n_samples };
    }

    /// Compute percentile-based confidence interval
    fn computeConfidenceInterval(
        self: *Self,
        samples: []f32,
        confidence: f32,
    ) ConfidenceInterval {
        _ = self;

        if (samples.len == 0) {
            return ConfidenceInterval{
                .lower = 0.0,
                .upper = 0.0,
                .mean = 0.0,
                .std = 0.0,
                .confidence = confidence,
                .sample_count = 0,
            };
        }

        // Create sorted copy for percentile computation (note: sorts in-place)
        const sorted = samples;
        std.mem.sort(f32, sorted, {}, std.sort.asc(f32));

        // Compute mean
        var sum: f32 = 0.0;
        for (sorted) |s| {
            sum += s;
        }
        const mean = sum / @as(f32, @floatFromInt(sorted.len));

        // Compute std
        var variance_sum: f32 = 0.0;
        for (sorted) |s| {
            const diff = s - mean;
            variance_sum += diff * diff;
        }
        const std_dev = @sqrt(variance_sum / @as(f32, @floatFromInt(sorted.len)));

        // Compute percentile indices
        // Lower = (1-confidence)/2 percentile
        // Upper = (1+confidence)/2 percentile
        const alpha = 1.0 - confidence;
        const lower_pct = alpha / 2.0;
        const upper_pct = 1.0 - alpha / 2.0;

        const n = sorted.len;
        const lower_idx = @min(@as(usize, @intFromFloat(lower_pct * @as(f32, @floatFromInt(n)))), n - 1);
        const upper_idx = @min(@as(usize, @intFromFloat(upper_pct * @as(f32, @floatFromInt(n)))), n - 1);

        return ConfidenceInterval{
            .lower = sorted[lower_idx],
            .upper = sorted[upper_idx],
            .mean = mean,
            .std = std_dev,
            .confidence = confidence,
            .sample_count = sorted.len,
        };
    }

    /// Classify each bootstrap sample and return majority vote
    fn voteClassification(self: *Self, curvatures: []const f32) VoteResult {
        var euclidean_votes: u32 = 0;
        var hyperbolic_votes: u32 = 0;
        var spherical_votes: u32 = 0;

        const thresholds = self.config.geometry_config.thresholds;

        for (curvatures) |curv| {
            const manifold = classify_geometry(curv, thresholds);
            switch (manifold) {
                .Euclidean => euclidean_votes += 1,
                .Hyperbolic => hyperbolic_votes += 1,
                .Spherical => spherical_votes += 1,
            }
        }

        const total_votes = euclidean_votes + hyperbolic_votes + spherical_votes;

        // Find winner
        var winner: ManifoldType = .Euclidean;
        var max_votes = euclidean_votes;

        if (hyperbolic_votes > max_votes) {
            winner = .Hyperbolic;
            max_votes = hyperbolic_votes;
        }
        if (spherical_votes > max_votes) {
            winner = .Spherical;
            max_votes = spherical_votes;
        }

        const vote_proportion = if (total_votes > 0)
            @as(f32, @floatFromInt(max_votes)) / @as(f32, @floatFromInt(total_votes))
        else
            0.0;

        return VoteResult{
            .manifold_type = winner,
            .vote_proportion = vote_proportion,
            .euclidean_votes = euclidean_votes,
            .hyperbolic_votes = hyperbolic_votes,
            .spherical_votes = spherical_votes,
            .total_votes = total_votes,
        };
    }
};


// ============================================================================
// Standalone Helper Functions
// ============================================================================

/// Bootstrap curvature computation (standalone version)
/// Returns mean, std, and confidence interval from resampled curvatures
pub fn bootstrap_curvature(
    points: []const f32,
    num_points: u32,
    dim: u32,
    n_samples: usize,
    allocator: Allocator,
) !struct { mean: f32, std: f32, ci_lower: f32, ci_upper: f32, valid_count: usize } {
    var detector = UncertaintyAwareGeometryDetector.init(allocator, .{
        .bootstrap_samples = n_samples,
    });

    const result = try detector.detectWithUncertainty(points, num_points, dim);
    defer @constCast(&result).deinit();

    return .{
        .mean = result.curvature_ci.mean,
        .std = result.curvature_ci.std,
        .ci_lower = result.curvature_ci.lower,
        .ci_upper = result.curvature_ci.upper,
        .valid_count = result.curvature_ci.sample_count,
    };
}

/// Compute percentile-based confidence interval (standalone version)
pub fn compute_confidence_interval(
    samples: []f32,
    confidence: f32,
) ConfidenceInterval {
    if (samples.len == 0) {
        return ConfidenceInterval{
            .lower = 0.0,
            .upper = 0.0,
            .mean = 0.0,
            .std = 0.0,
            .confidence = confidence,
            .sample_count = 0,
        };
    }

    // Sort samples for percentile computation
    std.mem.sort(f32, samples, {}, std.sort.asc(f32));

    // Compute mean
    var sum: f32 = 0.0;
    for (samples) |s| {
        sum += s;
    }
    const mean = sum / @as(f32, @floatFromInt(samples.len));

    // Compute std
    var variance_sum: f32 = 0.0;
    for (samples) |s| {
        const diff = s - mean;
        variance_sum += diff * diff;
    }
    const std_dev = @sqrt(variance_sum / @as(f32, @floatFromInt(samples.len)));

    // Compute percentile indices
    const alpha = 1.0 - confidence;
    const lower_pct = alpha / 2.0;
    const upper_pct = 1.0 - alpha / 2.0;

    const n = samples.len;
    const lower_idx = @min(@as(usize, @intFromFloat(lower_pct * @as(f32, @floatFromInt(n)))), n - 1);
    const upper_idx = @min(@as(usize, @intFromFloat(upper_pct * @as(f32, @floatFromInt(n)))), n - 1);

    return ConfidenceInterval{
        .lower = samples[lower_idx],
        .upper = samples[upper_idx],
        .mean = mean,
        .std = std_dev,
        .confidence = confidence,
        .sample_count = samples.len,
    };
}

/// Vote-based classification (standalone version)
pub fn vote_classification(
    curvatures: []const f32,
    thresholds: CurvatureThresholds,
) VoteResult {
    var euclidean_votes: u32 = 0;
    var hyperbolic_votes: u32 = 0;
    var spherical_votes: u32 = 0;

    for (curvatures) |curv| {
        const manifold = classify_geometry(curv, thresholds);
        switch (manifold) {
            .Euclidean => euclidean_votes += 1,
            .Hyperbolic => hyperbolic_votes += 1,
            .Spherical => spherical_votes += 1,
        }
    }

    const total_votes = euclidean_votes + hyperbolic_votes + spherical_votes;

    // Find winner
    var winner: ManifoldType = .Euclidean;
    var max_votes = euclidean_votes;

    if (hyperbolic_votes > max_votes) {
        winner = .Hyperbolic;
        max_votes = hyperbolic_votes;
    }
    if (spherical_votes > max_votes) {
        winner = .Spherical;
        max_votes = spherical_votes;
    }

    const vote_proportion = if (total_votes > 0)
        @as(f32, @floatFromInt(max_votes)) / @as(f32, @floatFromInt(total_votes))
    else
        0.0;

    return VoteResult{
        .manifold_type = winner,
        .vote_proportion = vote_proportion,
        .euclidean_votes = euclidean_votes,
        .hyperbolic_votes = hyperbolic_votes,
        .spherical_votes = spherical_votes,
        .total_votes = total_votes,
    };
}

/// Compute required sample size for desired confidence interval width
pub fn compute_required_sample_size(
    current_std: f32,
    desired_margin: f32,
    confidence_level: f32,
) u32 {
    // Using z-score approximation for confidence interval
    // n = (z * sigma / margin)^2
    // z_95 ≈ 1.96, z_99 ≈ 2.576

    const z_score: f32 = if (confidence_level >= 0.99)
        2.576
    else if (confidence_level >= 0.95)
        1.96
    else if (confidence_level >= 0.90)
        1.645
    else
        1.28; // 80% confidence

    if (desired_margin < 1e-10) return MAX_POINTS;

    const n_float = math.pow(f32, z_score * current_std / desired_margin, 2);
    const n = @as(u32, @intFromFloat(@ceil(n_float)));

    return @min(n, @as(u32, MAX_POINTS));
}

/// Compute calibration error (|P(true|predicted) - confidence|)
pub fn compute_calibration_error(
    predictions: []const ManifoldType,
    ground_truth: []const ManifoldType,
    confidence_levels: []const f32,
) f32 {
    if (predictions.len == 0 or predictions.len != ground_truth.len or
        predictions.len != confidence_levels.len)
    {
        return 1.0; // Maximum error for invalid input
    }

    // Group predictions by confidence bucket
    const num_buckets: usize = 10;
    var bucket_correct: [num_buckets]u32 = [_]u32{0} ** num_buckets;
    var bucket_total: [num_buckets]u32 = [_]u32{0} ** num_buckets;
    var bucket_conf_sum: [num_buckets]f32 = [_]f32{0.0} ** num_buckets;

    for (0..predictions.len) |i| {
        const conf = confidence_levels[i];
        const bucket_idx = @min(@as(usize, @intFromFloat(conf * @as(f32, num_buckets))), num_buckets - 1);

        bucket_total[bucket_idx] += 1;
        bucket_conf_sum[bucket_idx] += conf;

        if (predictions[i] == ground_truth[i]) {
            bucket_correct[bucket_idx] += 1;
        }
    }

    // Compute Expected Calibration Error (ECE)
    var ece: f32 = 0.0;
    var total_samples: f32 = 0.0;

    for (0..num_buckets) |b| {
        if (bucket_total[b] > 0) {
            const accuracy = @as(f32, @floatFromInt(bucket_correct[b])) /
                @as(f32, @floatFromInt(bucket_total[b]));
            const avg_conf = bucket_conf_sum[b] / @as(f32, @floatFromInt(bucket_total[b]));
            const weight = @as(f32, @floatFromInt(bucket_total[b]));

            ece += weight * @abs(accuracy - avg_conf);
            total_samples += weight;
        }
    }

    if (total_samples > 0) {
        ece /= total_samples;
    }

    return ece;
}


// ============================================================================
// Point Generation Helpers (for testing)
// ============================================================================

/// Generate points in Euclidean space
fn generate_euclidean_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        for (0..dim) |d| {
            buffer[i * dim + d] = random.float(f32) * 2.0 - 1.0;
        }
    }
}

/// Generate points on a sphere (positive curvature)
fn generate_spherical_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        for (0..dim) |d| {
            const val = random.floatNorm(f32);
            buffer[i * dim + d] = val;
            norm_sq += val * val;
        }
        const norm = @sqrt(norm_sq);
        if (norm > 1e-8) {
            for (0..dim) |d| {
                buffer[i * dim + d] /= norm;
            }
        }
    }
}

/// Generate points in hyperbolic-like space (Poincaré ball)
fn generate_hyperbolic_points(buffer: []f32, num_points: u32, dim: u32, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (0..num_points) |i| {
        var norm_sq: f32 = 0.0;
        for (0..dim) |d| {
            const val = random.floatNorm(f32);
            buffer[i * dim + d] = val;
            norm_sq += val * val;
        }
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

// Test 1: UncertaintyConfig default values
test "UncertaintyConfig default values" {
    const config = UncertaintyConfig{};
    try std.testing.expectEqual(@as(usize, 100), config.bootstrap_samples);
    try std.testing.expectApproxEqAbs(@as(f32, 0.95), config.confidence_level, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), config.detection_threshold, 1e-6);
}

// Test 2: ConfidenceInterval contains method
test "ConfidenceInterval contains" {
    const ci = ConfidenceInterval{
        .lower = 0.1,
        .upper = 0.5,
        .mean = 0.3,
        .std = 0.1,
        .confidence = 0.95,
        .sample_count = 100,
    };

    try std.testing.expect(ci.contains(0.3));
    try std.testing.expect(ci.contains(0.1));
    try std.testing.expect(ci.contains(0.5));
    try std.testing.expect(!ci.contains(0.05));
    try std.testing.expect(!ci.contains(0.6));
}

// Test 3: ConfidenceInterval width
test "ConfidenceInterval width" {
    const ci = ConfidenceInterval{
        .lower = 0.1,
        .upper = 0.5,
        .mean = 0.3,
        .std = 0.1,
        .confidence = 0.95,
        .sample_count = 100,
    };

    try std.testing.expectApproxEqAbs(@as(f32, 0.4), ci.width(), 1e-6);
}

// Test 4: ConfidenceInterval isNarrow
test "ConfidenceInterval isNarrow" {
    const narrow_ci = ConfidenceInterval{
        .lower = 0.29,
        .upper = 0.31,
        .mean = 0.3,
        .std = 0.01,
        .confidence = 0.95,
        .sample_count = 100,
    };

    const wide_ci = ConfidenceInterval{
        .lower = 0.0,
        .upper = 1.0,
        .mean = 0.5,
        .std = 0.3,
        .confidence = 0.95,
        .sample_count = 100,
    };

    try std.testing.expect(narrow_ci.isNarrow(0.1));
    try std.testing.expect(!wide_ci.isNarrow(0.5));
}

// Test 5: VoteResult isConfident
test "VoteResult isConfident" {
    const confident = VoteResult{
        .manifold_type = .Euclidean,
        .vote_proportion = 0.8,
        .euclidean_votes = 80,
        .hyperbolic_votes = 10,
        .spherical_votes = 10,
        .total_votes = 100,
    };

    const uncertain = VoteResult{
        .manifold_type = .Euclidean,
        .vote_proportion = 0.4,
        .euclidean_votes = 40,
        .hyperbolic_votes = 30,
        .spherical_votes = 30,
        .total_votes = 100,
    };

    try std.testing.expect(confident.isConfident(0.6));
    try std.testing.expect(!uncertain.isConfident(0.6));
}

// Test 6: VoteResult getProportions
test "VoteResult getProportions" {
    const result = VoteResult{
        .manifold_type = .Hyperbolic,
        .vote_proportion = 0.5,
        .euclidean_votes = 20,
        .hyperbolic_votes = 50,
        .spherical_votes = 30,
        .total_votes = 100,
    };

    const props = result.getProportions();
    try std.testing.expectApproxEqAbs(@as(f32, 0.2), props.euclidean, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), props.hyperbolic, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), props.spherical, 1e-6);
}


// Test 7: VoteResult with zero votes
test "VoteResult zero votes" {
    const result = VoteResult{
        .manifold_type = .Euclidean,
        .vote_proportion = 0.0,
        .euclidean_votes = 0,
        .hyperbolic_votes = 0,
        .spherical_votes = 0,
        .total_votes = 0,
    };

    const props = result.getProportions();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), props.euclidean, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), props.hyperbolic, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), props.spherical, 1e-6);
}

// Test 8: compute_confidence_interval with synthetic data
test "compute_confidence_interval basic" {
    var samples = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    const ci = compute_confidence_interval(&samples, 0.95);

    try std.testing.expect(ci.sample_count == 10);
    try std.testing.expect(ci.lower <= ci.mean);
    try std.testing.expect(ci.mean <= ci.upper);
    try std.testing.expect(ci.std >= 0.0);

    // Mean of 0.1 to 1.0 should be 0.55
    try std.testing.expectApproxEqAbs(@as(f32, 0.55), ci.mean, 1e-5);
}

// Test 9: compute_confidence_interval with empty array
test "compute_confidence_interval empty" {
    const empty_array: []f32 = &[_]f32{};
    const ci = compute_confidence_interval(@constCast(empty_array), 0.95);

    try std.testing.expectEqual(@as(usize, 0), ci.sample_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ci.mean, 1e-6);
}

// Test 10: compute_confidence_interval with single sample
test "compute_confidence_interval single sample" {
    var samples = [_]f32{0.5};
    const ci = compute_confidence_interval(&samples, 0.95);

    try std.testing.expectEqual(@as(usize, 1), ci.sample_count);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ci.mean, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ci.lower, 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), ci.upper, 1e-6);
}

// Test 11: vote_classification with all Euclidean
test "vote_classification all euclidean" {
    const curvatures = [_]f32{ 0.0, 0.01, -0.01, 0.05, -0.05 };
    const result = vote_classification(&curvatures, CurvatureThresholds{});

    try std.testing.expectEqual(ManifoldType.Euclidean, result.manifold_type);
    try std.testing.expectEqual(@as(u32, 5), result.euclidean_votes);
    try std.testing.expectEqual(@as(u32, 0), result.hyperbolic_votes);
    try std.testing.expectEqual(@as(u32, 0), result.spherical_votes);
}

// Test 12: vote_classification with all Hyperbolic
test "vote_classification all hyperbolic" {
    const curvatures = [_]f32{ -0.5, -0.3, -0.2, -0.4, -0.6 };
    const result = vote_classification(&curvatures, CurvatureThresholds{});

    try std.testing.expectEqual(ManifoldType.Hyperbolic, result.manifold_type);
    try std.testing.expectEqual(@as(u32, 5), result.hyperbolic_votes);
}

// Test 13: vote_classification with all Spherical
test "vote_classification all spherical" {
    const curvatures = [_]f32{ 0.5, 0.3, 0.2, 0.4, 0.6 };
    const result = vote_classification(&curvatures, CurvatureThresholds{});

    try std.testing.expectEqual(ManifoldType.Spherical, result.manifold_type);
    try std.testing.expectEqual(@as(u32, 5), result.spherical_votes);
}

// Test 14: vote_classification with mixed votes
test "vote_classification mixed" {
    const curvatures = [_]f32{ -0.5, -0.3, 0.0, 0.5, 0.01 };
    const result = vote_classification(&curvatures, CurvatureThresholds{});

    try std.testing.expectEqual(@as(u32, 2), result.hyperbolic_votes);
    try std.testing.expectEqual(@as(u32, 2), result.euclidean_votes);
    try std.testing.expectEqual(@as(u32, 1), result.spherical_votes);
    try std.testing.expectEqual(@as(u32, 5), result.total_votes);
}

// Test 15: vote_classification empty array
test "vote_classification empty" {
    const curvatures: []const f32 = &[_]f32{};
    const result = vote_classification(curvatures, CurvatureThresholds{});

    try std.testing.expectEqual(@as(u32, 0), result.total_votes);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result.vote_proportion, 1e-6);
}

// Test 16: compute_required_sample_size
test "compute_required_sample_size" {
    // With std=0.1, margin=0.05, 95% confidence
    const n1 = compute_required_sample_size(0.1, 0.05, 0.95);
    // n = (1.96 * 0.1 / 0.05)^2 = (3.92)^2 ≈ 15.4 -> 16
    try std.testing.expect(n1 >= 15 and n1 <= 20);

    // Smaller margin should require more samples
    const n2 = compute_required_sample_size(0.1, 0.01, 0.95);
    try std.testing.expect(n2 > n1);

    // Higher confidence should require more samples
    const n3 = compute_required_sample_size(0.1, 0.05, 0.99);
    try std.testing.expect(n3 > n1);
}

// Test 17: compute_required_sample_size with tiny margin
test "compute_required_sample_size tiny margin" {
    const n = compute_required_sample_size(0.1, 1e-12, 0.95);
    try std.testing.expectEqual(@as(u32, MAX_POINTS), n);
}

// Test 18: compute_calibration_error perfect predictions
test "compute_calibration_error perfect" {
    const preds = [_]ManifoldType{ .Euclidean, .Hyperbolic, .Spherical };
    const truth = [_]ManifoldType{ .Euclidean, .Hyperbolic, .Spherical };
    const confs = [_]f32{ 1.0, 1.0, 1.0 };

    const ece = compute_calibration_error(&preds, &truth, &confs);
    // Perfect predictions at 100% confidence should have near-zero ECE
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ece, 1e-5);
}

// Test 19: compute_calibration_error empty arrays
test "compute_calibration_error empty" {
    const preds: []const ManifoldType = &[_]ManifoldType{};
    const truth: []const ManifoldType = &[_]ManifoldType{};
    const confs: []const f32 = &[_]f32{};

    const ece = compute_calibration_error(preds, truth, confs);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ece, 1e-5);
}

// Test 20: compute_calibration_error mismatched lengths
test "compute_calibration_error mismatched" {
    const preds = [_]ManifoldType{ .Euclidean, .Hyperbolic };
    const truth = [_]ManifoldType{.Euclidean};
    const confs = [_]f32{ 0.9, 0.8 };

    const ece = compute_calibration_error(&preds, &truth, &confs);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), ece, 1e-5);
}


// Test 21: UncertaintyAwareGeometryDetector init
test "UncertaintyAwareGeometryDetector init" {
    const allocator = std.testing.allocator;
    const config = UncertaintyConfig{
        .bootstrap_samples = 50,
        .confidence_level = 0.90,
    };

    const detector = UncertaintyAwareGeometryDetector.init(allocator, config);
    try std.testing.expectEqual(@as(usize, 50), detector.config.bootstrap_samples);
    try std.testing.expectApproxEqAbs(@as(f32, 0.90), detector.config.confidence_level, 1e-6);
}

// Test 22: UncertaintyAwareGeometryDetector insufficient points
test "UncertaintyAwareGeometryDetector insufficient points" {
    const allocator = std.testing.allocator;
    var detector = UncertaintyAwareGeometryDetector.init(allocator, .{});

    const points = [_]f32{ 0.0, 0.0, 1.0, 1.0 }; // Only 2 points
    const result = detector.detectWithUncertainty(&points, 2, 2);

    try std.testing.expectError(error.InsufficientPoints, result);
}

// Test 23: UncertaintyAwareGeometryDetector resetSeed
test "UncertaintyAwareGeometryDetector resetSeed" {
    const allocator = std.testing.allocator;
    var detector = UncertaintyAwareGeometryDetector.init(allocator, .{ .random_seed = 12345 });

    // Generate some random numbers
    _ = detector.prng.random().int(u64);
    _ = detector.prng.random().int(u64);

    // Reset seed
    detector.resetSeed();

    // First number after reset should be deterministic
    const first = detector.prng.random().int(u64);
    detector.resetSeed();
    const second = detector.prng.random().int(u64);

    try std.testing.expectEqual(first, second);
}

// Test 24: UncertaintyResult getVoteEntropy
test "UncertaintyResult getVoteEntropy uniform" {
    const allocator = std.testing.allocator;
    const bootstrap_means = try allocator.alloc(f32, 1);
    bootstrap_means[0] = 0.0;

    var result = UncertaintyResult{
        .vote_result = VoteResult{
            .manifold_type = .Euclidean,
            .vote_proportion = 0.333,
            .euclidean_votes = 33,
            .hyperbolic_votes = 33,
            .spherical_votes = 34,
            .total_votes = 100,
        },
        .curvature_ci = ConfidenceInterval{
            .lower = 0.0,
            .upper = 0.0,
            .mean = 0.0,
            .std = 0.0,
            .confidence = 0.95,
            .sample_count = 1,
        },
        .bootstrap_means = bootstrap_means,
        .is_reliable = true,
        .computation_time_ns = 0,
        .allocator = allocator,
    };
    defer result.deinit();

    // Near-uniform distribution should have high entropy (close to 1.0)
    const entropy = result.getVoteEntropy();
    try std.testing.expect(entropy > 0.9);
}

// Test 25: UncertaintyResult getVoteEntropy concentrated
test "UncertaintyResult getVoteEntropy concentrated" {
    const allocator = std.testing.allocator;
    const bootstrap_means = try allocator.alloc(f32, 1);
    bootstrap_means[0] = 0.0;

    var result = UncertaintyResult{
        .vote_result = VoteResult{
            .manifold_type = .Euclidean,
            .vote_proportion = 1.0,
            .euclidean_votes = 100,
            .hyperbolic_votes = 0,
            .spherical_votes = 0,
            .total_votes = 100,
        },
        .curvature_ci = ConfidenceInterval{
            .lower = 0.0,
            .upper = 0.0,
            .mean = 0.0,
            .std = 0.0,
            .confidence = 0.95,
            .sample_count = 1,
        },
        .bootstrap_means = bootstrap_means,
        .is_reliable = true,
        .computation_time_ns = 0,
        .allocator = allocator,
    };
    defer result.deinit();

    // Concentrated distribution should have low entropy (close to 0.0)
    const entropy = result.getVoteEntropy();
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), entropy, 1e-5);
}

// Test 26: UncertaintyResult getRecommendedGeometry reliable
test "UncertaintyResult getRecommendedGeometry reliable" {
    const allocator = std.testing.allocator;
    const bootstrap_means = try allocator.alloc(f32, 1);
    bootstrap_means[0] = 0.0;

    var result = UncertaintyResult{
        .vote_result = VoteResult{
            .manifold_type = .Hyperbolic,
            .vote_proportion = 0.8,
            .euclidean_votes = 10,
            .hyperbolic_votes = 80,
            .spherical_votes = 10,
            .total_votes = 100,
        },
        .curvature_ci = ConfidenceInterval{
            .lower = -0.5,
            .upper = -0.1,
            .mean = -0.3,
            .std = 0.1,
            .confidence = 0.95,
            .sample_count = 100,
        },
        .bootstrap_means = bootstrap_means,
        .is_reliable = true,
        .computation_time_ns = 0,
        .allocator = allocator,
    };
    defer result.deinit();

    const recommended = result.getRecommendedGeometry();
    try std.testing.expectEqual(ManifoldType.Hyperbolic, recommended.?);
}

// Test 27: UncertaintyResult getRecommendedGeometry unreliable
test "UncertaintyResult getRecommendedGeometry unreliable" {
    const allocator = std.testing.allocator;
    const bootstrap_means = try allocator.alloc(f32, 1);
    bootstrap_means[0] = 0.0;

    var result = UncertaintyResult{
        .vote_result = VoteResult{
            .manifold_type = .Euclidean,
            .vote_proportion = 0.4,
            .euclidean_votes = 40,
            .hyperbolic_votes = 30,
            .spherical_votes = 30,
            .total_votes = 100,
        },
        .curvature_ci = ConfidenceInterval{
            .lower = -1.0,
            .upper = 1.0,
            .mean = 0.0,
            .std = 0.5,
            .confidence = 0.95,
            .sample_count = 100,
        },
        .bootstrap_means = bootstrap_means,
        .is_reliable = false,
        .computation_time_ns = 0,
        .allocator = allocator,
    };
    defer result.deinit();

    const recommended = result.getRecommendedGeometry();
    try std.testing.expect(recommended == null);
}