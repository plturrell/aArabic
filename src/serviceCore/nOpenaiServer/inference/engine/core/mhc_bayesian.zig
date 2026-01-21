// mHC Bayesian Curvature Estimation - Day 62
// Implements Bayesian inference for curvature estimation in mHC
//
// Core Components:
// - BayesianCurvatureEstimator: Main struct for Bayesian curvature inference
// - Gaussian Prior/Likelihood: Log probability functions
// - Posterior Update: Conjugate Gaussian updates
// - Credible Intervals: Compute confidence intervals
// - Calibration Metrics: ECE, MCE, Brier score
//
// Reference: docs/DAY_62_BAYESIAN_REPORT.md

const std = @import("std");
const math = std.math;
const Allocator = std.mem.Allocator;

// ============================================================================
// Constants
// ============================================================================

/// Numerical stability epsilon
pub const EPSILON: f32 = 1e-10;

/// Default prior mean (uninformative prior centered at 0)
pub const DEFAULT_PRIOR_MEAN: f32 = 0.0;

/// Default prior variance (diffuse prior)
pub const DEFAULT_PRIOR_VARIANCE: f32 = 1.0;

/// Default likelihood variance (observation noise)
pub const DEFAULT_LIKELIHOOD_VARIANCE: f32 = 0.1;

/// Log(2π) constant for Gaussian calculations
pub const LOG_2PI: f32 = 1.8378770664093453;

/// Number of bins for calibration metrics
pub const DEFAULT_CALIBRATION_BINS: u32 = 10;

// ============================================================================
// Error Types
// ============================================================================

pub const BayesianError = error{
    InvalidVariance,
    InvalidObservations,
    InvalidConfidenceLevel,
    EmptyObservations,
    InvalidBinCount,
    InvalidPredictions,
    OutOfMemory,
};

// ============================================================================
// Interval Type
// ============================================================================

/// Interval bounds for credible/confidence intervals
pub const Interval = struct {
    lower: f32,
    upper: f32,
};

// ============================================================================
// BayesianCurvatureEstimator Struct
// ============================================================================

/// Bayesian estimator for manifold curvature using conjugate Gaussian priors
pub const BayesianCurvatureEstimator = struct {
    /// Prior mean for curvature
    prior_mean: f32 = DEFAULT_PRIOR_MEAN,

    /// Prior variance for curvature
    prior_variance: f32 = DEFAULT_PRIOR_VARIANCE,

    /// Likelihood variance (observation noise)
    likelihood_variance: f32 = DEFAULT_LIKELIHOOD_VARIANCE,

    /// Current posterior mean (updated after observations)
    posterior_mean: f32 = DEFAULT_PRIOR_MEAN,

    /// Current posterior variance (updated after observations)
    posterior_variance: f32 = DEFAULT_PRIOR_VARIANCE,

    /// Number of observations incorporated
    observation_count: u32 = 0,

    /// Sum of observations (for incremental updates)
    observation_sum: f32 = 0.0,

    /// Create a new estimator with default parameters
    pub fn init() BayesianCurvatureEstimator {
        return BayesianCurvatureEstimator{};
    }

    /// Create a new estimator with custom parameters
    pub fn initWithParams(
        prior_mean: f32,
        prior_variance: f32,
        likelihood_variance: f32,
    ) BayesianError!BayesianCurvatureEstimator {
        if (prior_variance <= 0) return BayesianError.InvalidVariance;
        if (likelihood_variance <= 0) return BayesianError.InvalidVariance;

        return BayesianCurvatureEstimator{
            .prior_mean = prior_mean,
            .prior_variance = prior_variance,
            .likelihood_variance = likelihood_variance,
            .posterior_mean = prior_mean,
            .posterior_variance = prior_variance,
        };
    }

    /// Validate the estimator configuration
    pub fn validate(self: BayesianCurvatureEstimator) BayesianError!void {
        if (self.prior_variance <= 0) return BayesianError.InvalidVariance;
        if (self.likelihood_variance <= 0) return BayesianError.InvalidVariance;
        if (self.posterior_variance <= 0) return BayesianError.InvalidVariance;
    }

    /// Reset the estimator to prior state
    pub fn reset(self: *BayesianCurvatureEstimator) void {
        self.posterior_mean = self.prior_mean;
        self.posterior_variance = self.prior_variance;
        self.observation_count = 0;
        self.observation_sum = 0.0;
    }

    // ========================================================================
    // Gaussian Prior/Likelihood Functions
    // ========================================================================

    /// Compute log prior probability: log N(curvature | prior_mean, prior_variance)
    pub fn log_prior(self: BayesianCurvatureEstimator, curvature: f32) f32 {
        return log_gaussian(curvature, self.prior_mean, self.prior_variance);
    }

    /// Compute log likelihood: sum of log N(obs_i | curvature, likelihood_variance)
    pub fn log_likelihood(self: BayesianCurvatureEstimator, curvature: f32, observations: []const f32) f32 {
        if (observations.len == 0) return 0.0;

        var log_lik: f32 = 0.0;
        for (observations) |obs| {
            log_lik += log_gaussian(obs, curvature, self.likelihood_variance);
        }
        return log_lik;
    }

    /// Compute unnormalized log posterior: log_prior + log_likelihood
    pub fn log_posterior(self: BayesianCurvatureEstimator, curvature: f32, observations: []const f32) f32 {
        return self.log_prior(curvature) + self.log_likelihood(curvature, observations);
    }

    // ========================================================================
    // Posterior Update Functions
    // ========================================================================

    /// Update posterior given new observations using conjugate Gaussian formula
    /// posterior_variance = 1 / (1/prior_variance + n/likelihood_variance)
    /// posterior_mean = posterior_variance * (prior_mean/prior_variance + sum(obs)/likelihood_variance)
    pub fn update_posterior(self: *BayesianCurvatureEstimator, observations: []const f32) BayesianError!void {
        if (observations.len == 0) return BayesianError.EmptyObservations;

        const n: f32 = @floatFromInt(observations.len);
        var obs_sum: f32 = 0.0;
        for (observations) |obs| {
            obs_sum += obs;
        }

        // Conjugate Gaussian update
        const prior_precision = 1.0 / self.prior_variance;
        const likelihood_precision = n / self.likelihood_variance;

        const posterior_precision = prior_precision + likelihood_precision;
        self.posterior_variance = 1.0 / posterior_precision;
        self.posterior_mean = self.posterior_variance * (self.prior_mean * prior_precision + obs_sum / self.likelihood_variance);

        // Track observation statistics
        self.observation_count += @intCast(observations.len);
        self.observation_sum += obs_sum;
    }

    /// Incremental update with a single observation
    pub fn update_single(self: *BayesianCurvatureEstimator, observation: f32) void {
        // Update using current posterior as prior
        const prior_precision = 1.0 / self.posterior_variance;
        const likelihood_precision = 1.0 / self.likelihood_variance;

        const new_precision = prior_precision + likelihood_precision;
        const new_variance = 1.0 / new_precision;
        const new_mean = new_variance * (self.posterior_mean * prior_precision + observation * likelihood_precision);

        self.posterior_mean = new_mean;
        self.posterior_variance = new_variance;
        self.observation_count += 1;
        self.observation_sum += observation;
    }

    /// Get the Maximum A Posteriori (MAP) estimate
    pub fn get_map_estimate(self: BayesianCurvatureEstimator) f32 {
        return self.posterior_mean;
    }

    /// Get the posterior standard deviation
    pub fn get_posterior_std(self: BayesianCurvatureEstimator) f32 {
        return @sqrt(self.posterior_variance);
    }

    // ========================================================================
    // Credible Intervals
    // ========================================================================

    /// Compute credible interval for the posterior
    /// Returns (lower_bound, upper_bound) for the given confidence level
    /// Uses normal distribution quantiles
    pub fn credible_interval(
        self: BayesianCurvatureEstimator,
        level: f32,
    ) BayesianError!Interval {
        if (level <= 0.0 or level >= 1.0) return BayesianError.InvalidConfidenceLevel;

        const alpha = 1.0 - level;
        const z = normal_quantile(1.0 - alpha / 2.0);
        const std_dev = @sqrt(self.posterior_variance);

        return Interval{
            .lower = self.posterior_mean - z * std_dev,
            .upper = self.posterior_mean + z * std_dev,
        };
    }

    /// Compute highest density interval (HDI) - for symmetric Gaussian, same as credible interval
    pub fn highest_density_interval(
        self: BayesianCurvatureEstimator,
        level: f32,
    ) BayesianError!Interval {
        // For Gaussian, HDI equals symmetric credible interval
        return self.credible_interval(level);
    }
};

// ============================================================================
// Helper Functions for Gaussian Calculations
// ============================================================================

/// Compute log of Gaussian PDF: log N(x | mean, variance)
pub fn log_gaussian(x: f32, mean: f32, variance: f32) f32 {
    const diff = x - mean;
    return -0.5 * (LOG_2PI + @log(@max(variance, EPSILON)) + (diff * diff) / @max(variance, EPSILON));
}

/// Compute Gaussian PDF: N(x | mean, variance)
pub fn gaussian_pdf(x: f32, mean: f32, variance: f32) f32 {
    return @exp(log_gaussian(x, mean, variance));
}

/// Standard normal CDF approximation using Abramowitz and Stegun
pub fn standard_normal_cdf(x: f32) f32 {
    const a1: f32 = 0.254829592;
    const a2: f32 = -0.284496736;
    const a3: f32 = 1.421413741;
    const a4: f32 = -1.453152027;
    const a5: f32 = 1.061405429;
    const p: f32 = 0.3275911;

    const sign: f32 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    const t = 1.0 / (1.0 + p * abs_x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-abs_x * abs_x / 2.0);

    return 0.5 * (1.0 + sign * y);
}

/// Normal quantile (inverse CDF) approximation using Beasley-Springer-Moro algorithm
pub fn normal_quantile(p: f32) f32 {
    if (p <= 0.0) return -math.inf(f32);
    if (p >= 1.0) return math.inf(f32);

    // Coefficients for rational approximation
    const a = [_]f32{
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    };

    const b = [_]f32{
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    };

    const c = [_]f32{
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    };

    const d = [_]f32{
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    };

    const p_low: f32 = 0.02425;
    const p_high: f32 = 1.0 - p_low;

    var q: f32 = undefined;
    var r: f32 = undefined;

    if (p < p_low) {
        // Lower region
        q = @sqrt(-2.0 * @log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        // Central region
        q = p - 0.5;
        r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    } else {
        // Upper region
        q = @sqrt(-2.0 * @log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

// ============================================================================
// Calibration Metrics
// ============================================================================

/// Result of calibration error calculation
pub const CalibrationResult = struct {
    error_value: f32,
    bin_counts: [DEFAULT_CALIBRATION_BINS]u32,
    bin_accuracies: [DEFAULT_CALIBRATION_BINS]f32,
    bin_confidences: [DEFAULT_CALIBRATION_BINS]f32,
};

/// Expected Calibration Error (ECE)
/// Measures the weighted average of calibration error across confidence bins
/// ECE = Σ (|bin_i| / n) * |accuracy_i - confidence_i|
pub fn expected_calibration_error(
    predictions: []const f32,
    outcomes: []const f32,
    n_bins: u32,
) BayesianError!f32 {
    if (predictions.len == 0) return BayesianError.EmptyObservations;
    if (predictions.len != outcomes.len) return BayesianError.InvalidPredictions;
    if (n_bins == 0 or n_bins > 100) return BayesianError.InvalidBinCount;

    const n: f32 = @floatFromInt(predictions.len);
    const bin_width: f32 = 1.0 / @as(f32, @floatFromInt(n_bins));

    var bin_counts = [_]u32{0} ** 100;
    var bin_correct = [_]f32{0.0} ** 100;
    var bin_confidence_sum = [_]f32{0.0} ** 100;

    // Accumulate statistics per bin
    for (predictions, outcomes) |pred, outcome| {
        const clamped_pred = @max(0.0, @min(1.0, pred));
        var bin_idx: u32 = @intFromFloat(clamped_pred / bin_width);
        bin_idx = @min(bin_idx, n_bins - 1);

        bin_counts[bin_idx] += 1;
        bin_confidence_sum[bin_idx] += clamped_pred;

        // Binary outcome: 1.0 if correct, 0.0 if incorrect
        if (outcome > 0.5) {
            bin_correct[bin_idx] += 1.0;
        }
    }

    // Calculate weighted ECE
    var ece: f32 = 0.0;
    for (0..n_bins) |i| {
        if (bin_counts[i] > 0) {
            const count: f32 = @floatFromInt(bin_counts[i]);
            const accuracy = bin_correct[i] / count;
            const avg_confidence = bin_confidence_sum[i] / count;
            ece += (count / n) * @abs(accuracy - avg_confidence);
        }
    }

    return ece;
}

/// Maximum Calibration Error (MCE)
/// Returns the maximum calibration error across all bins
/// MCE = max_i |accuracy_i - confidence_i|
pub fn maximum_calibration_error(
    predictions: []const f32,
    outcomes: []const f32,
    n_bins: u32,
) BayesianError!f32 {
    if (predictions.len == 0) return BayesianError.EmptyObservations;
    if (predictions.len != outcomes.len) return BayesianError.InvalidPredictions;
    if (n_bins == 0 or n_bins > 100) return BayesianError.InvalidBinCount;

    const bin_width: f32 = 1.0 / @as(f32, @floatFromInt(n_bins));

    var bin_counts = [_]u32{0} ** 100;
    var bin_correct = [_]f32{0.0} ** 100;
    var bin_confidence_sum = [_]f32{0.0} ** 100;

    // Accumulate statistics per bin
    for (predictions, outcomes) |pred, outcome| {
        const clamped_pred = @max(0.0, @min(1.0, pred));
        var bin_idx: u32 = @intFromFloat(clamped_pred / bin_width);
        bin_idx = @min(bin_idx, n_bins - 1);

        bin_counts[bin_idx] += 1;
        bin_confidence_sum[bin_idx] += clamped_pred;

        if (outcome > 0.5) {
            bin_correct[bin_idx] += 1.0;
        }
    }

    // Find maximum calibration error
    var mce: f32 = 0.0;
    for (0..n_bins) |i| {
        if (bin_counts[i] > 0) {
            const count: f32 = @floatFromInt(bin_counts[i]);
            const accuracy = bin_correct[i] / count;
            const avg_confidence = bin_confidence_sum[i] / count;
            mce = @max(mce, @abs(accuracy - avg_confidence));
        }
    }

    return mce;
}

/// Brier Score
/// Measures the mean squared error of probabilistic predictions
/// Brier = (1/n) * Σ (prediction_i - outcome_i)²
pub fn brier_score(predictions: []const f32, outcomes: []const f32) BayesianError!f32 {
    if (predictions.len == 0) return BayesianError.EmptyObservations;
    if (predictions.len != outcomes.len) return BayesianError.InvalidPredictions;

    var sum_sq_error: f32 = 0.0;
    for (predictions, outcomes) |pred, outcome| {
        const diff = pred - outcome;
        sum_sq_error += diff * diff;
    }

    return sum_sq_error / @as(f32, @floatFromInt(predictions.len));
}

/// Reliability diagram data for visualization
pub const ReliabilityDiagram = struct {
    bin_accuracies: []f32,
    bin_confidences: []f32,
    bin_counts: []u32,
    n_bins: u32,
    allocator: Allocator,

    pub fn deinit(self: *ReliabilityDiagram) void {
        self.allocator.free(self.bin_accuracies);
        self.allocator.free(self.bin_confidences);
        self.allocator.free(self.bin_counts);
    }
};

/// Compute reliability diagram data
pub fn compute_reliability_diagram(
    predictions: []const f32,
    outcomes: []const f32,
    n_bins: u32,
    allocator: Allocator,
) BayesianError!ReliabilityDiagram {
    if (predictions.len == 0) return BayesianError.EmptyObservations;
    if (predictions.len != outcomes.len) return BayesianError.InvalidPredictions;
    if (n_bins == 0 or n_bins > 100) return BayesianError.InvalidBinCount;

    const bin_width: f32 = 1.0 / @as(f32, @floatFromInt(n_bins));

    var bin_counts = allocator.alloc(u32, n_bins) catch return BayesianError.OutOfMemory;
    errdefer allocator.free(bin_counts);
    var bin_accuracies = allocator.alloc(f32, n_bins) catch return BayesianError.OutOfMemory;
    errdefer allocator.free(bin_accuracies);
    var bin_confidences = allocator.alloc(f32, n_bins) catch return BayesianError.OutOfMemory;
    errdefer allocator.free(bin_confidences);

    // Initialize
    @memset(bin_counts, 0);
    var bin_correct = allocator.alloc(f32, n_bins) catch return BayesianError.OutOfMemory;
    defer allocator.free(bin_correct);
    var bin_conf_sum = allocator.alloc(f32, n_bins) catch return BayesianError.OutOfMemory;
    defer allocator.free(bin_conf_sum);
    @memset(bin_correct, 0.0);
    @memset(bin_conf_sum, 0.0);

    // Accumulate statistics per bin
    for (predictions, outcomes) |pred, outcome| {
        const clamped_pred = @max(0.0, @min(1.0, pred));
        var bin_idx: u32 = @intFromFloat(clamped_pred / bin_width);
        bin_idx = @min(bin_idx, n_bins - 1);

        bin_counts[bin_idx] += 1;
        bin_conf_sum[bin_idx] += clamped_pred;

        if (outcome > 0.5) {
            bin_correct[bin_idx] += 1.0;
        }
    }

    // Compute averages
    for (0..n_bins) |i| {
        if (bin_counts[i] > 0) {
            const count: f32 = @floatFromInt(bin_counts[i]);
            bin_accuracies[i] = bin_correct[i] / count;
            bin_confidences[i] = bin_conf_sum[i] / count;
        } else {
            bin_accuracies[i] = 0.0;
            bin_confidences[i] = (@as(f32, @floatFromInt(i)) + 0.5) * bin_width;
        }
    }

    return ReliabilityDiagram{
        .bin_accuracies = bin_accuracies,
        .bin_confidences = bin_confidences,
        .bin_counts = bin_counts,
        .n_bins = n_bins,
        .allocator = allocator,
    };
}

// ============================================================================
// Synthetic Data Generation for Testing
// ============================================================================

/// Generate synthetic curvature observations from a Gaussian distribution
pub fn generate_synthetic_observations(
    buffer: []f32,
    true_curvature: f32,
    noise_variance: f32,
    seed: u64,
) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (buffer) |*val| {
        // Box-Muller transform for Gaussian sampling
        const rand1 = random.float(f32);
        const rand2 = random.float(f32);
        const z = @sqrt(-2.0 * @log(@max(rand1, EPSILON))) * @cos(2.0 * math.pi * rand2);
        val.* = true_curvature + z * @sqrt(noise_variance);
    }
}

/// Generate calibrated predictions (for testing calibration metrics)
pub fn generate_calibrated_predictions(
    predictions: []f32,
    outcomes: []f32,
    calibration: f32, // 0 = random, 1 = perfect
    seed: u64,
) void {
    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    for (predictions, outcomes) |*pred, *out| {
        const true_prob = random.float(f32);
        pred.* = true_prob * calibration + random.float(f32) * (1.0 - calibration);
        out.* = if (random.float(f32) < true_prob) 1.0 else 0.0;
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

test "BayesianCurvatureEstimator init default" {
    const estimator = BayesianCurvatureEstimator.init();

    try std.testing.expectEqual(@as(f32, 0.0), estimator.prior_mean);
    try std.testing.expectEqual(@as(f32, 1.0), estimator.prior_variance);
    try std.testing.expectEqual(@as(f32, 0.1), estimator.likelihood_variance);
    try std.testing.expectEqual(@as(f32, 0.0), estimator.posterior_mean);
    try std.testing.expectEqual(@as(f32, 1.0), estimator.posterior_variance);
}

test "BayesianCurvatureEstimator init with params" {
    const estimator = try BayesianCurvatureEstimator.initWithParams(1.0, 2.0, 0.5);

    try std.testing.expectEqual(@as(f32, 1.0), estimator.prior_mean);
    try std.testing.expectEqual(@as(f32, 2.0), estimator.prior_variance);
    try std.testing.expectEqual(@as(f32, 0.5), estimator.likelihood_variance);
}

test "BayesianCurvatureEstimator invalid variance" {
    const result = BayesianCurvatureEstimator.initWithParams(0.0, -1.0, 0.1);
    try std.testing.expectError(BayesianError.InvalidVariance, result);

    const result2 = BayesianCurvatureEstimator.initWithParams(0.0, 1.0, 0.0);
    try std.testing.expectError(BayesianError.InvalidVariance, result2);
}

test "log_prior at prior mean" {
    const estimator = BayesianCurvatureEstimator.init();
    const log_p = estimator.log_prior(0.0);

    // At mean, log prior should be maximum
    const log_p_offset = estimator.log_prior(0.5);
    try std.testing.expect(log_p > log_p_offset);
}

test "log_prior symmetry" {
    const estimator = try BayesianCurvatureEstimator.initWithParams(0.0, 1.0, 0.1);

    const log_p_pos = estimator.log_prior(0.5);
    const log_p_neg = estimator.log_prior(-0.5);

    try std.testing.expectApproxEqAbs(log_p_pos, log_p_neg, 1e-5);
}

test "log_likelihood single observation" {
    const estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{0.5};

    const log_lik = estimator.log_likelihood(0.5, &obs);

    // Likelihood should be maximum when curvature equals observation
    const log_lik_offset = estimator.log_likelihood(1.0, &obs);
    try std.testing.expect(log_lik > log_lik_offset);
}

test "log_likelihood multiple observations" {
    const estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{ 0.4, 0.5, 0.6 };

    const log_lik = estimator.log_likelihood(0.5, &obs);

    // Should be finite and negative
    try std.testing.expect(!math.isNan(log_lik));
    try std.testing.expect(!math.isInf(log_lik));
}

test "log_posterior combines prior and likelihood" {
    const estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{0.5};

    const log_post = estimator.log_posterior(0.3, &obs);
    const log_prior = estimator.log_prior(0.3);
    const log_lik = estimator.log_likelihood(0.3, &obs);

    try std.testing.expectApproxEqAbs(log_post, log_prior + log_lik, 1e-5);
}

test "update_posterior single observation" {
    var estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{0.5};

    try estimator.update_posterior(&obs);

    // Posterior mean should move towards observation
    try std.testing.expect(estimator.posterior_mean > 0.0);
    try std.testing.expect(estimator.posterior_mean < 0.5);

    // Posterior variance should decrease
    try std.testing.expect(estimator.posterior_variance < estimator.prior_variance);
}

test "update_posterior multiple observations" {
    var estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{ 0.8, 0.9, 1.0, 0.85, 0.95 };

    try estimator.update_posterior(&obs);

    // Posterior mean should be close to observation mean (0.9)
    try std.testing.expectApproxEqAbs(@as(f32, 0.9), estimator.posterior_mean, 0.15);

    // Observation count should be updated
    try std.testing.expectEqual(@as(u32, 5), estimator.observation_count);
}

test "update_posterior empty observations" {
    var estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{};

    const result = estimator.update_posterior(&obs);
    try std.testing.expectError(BayesianError.EmptyObservations, result);
}

test "update_single incremental" {
    var estimator = BayesianCurvatureEstimator.init();

    estimator.update_single(1.0);
    const mean1 = estimator.posterior_mean;

    estimator.update_single(1.0);
    const mean2 = estimator.posterior_mean;

    // Mean should move further towards 1.0
    try std.testing.expect(mean2 > mean1);
    try std.testing.expectEqual(@as(u32, 2), estimator.observation_count);
}

test "credible_interval 95 percent" {
    var estimator = BayesianCurvatureEstimator.init();
    estimator.posterior_mean = 0.0;
    estimator.posterior_variance = 1.0;

    const ci = try estimator.credible_interval(0.95);

    // 95% CI for N(0,1) should be approximately (-1.96, 1.96)
    try std.testing.expectApproxEqAbs(@as(f32, -1.96), ci.lower, 0.05);
    try std.testing.expectApproxEqAbs(@as(f32, 1.96), ci.upper, 0.05);
}

test "credible_interval 68 percent" {
    var estimator = BayesianCurvatureEstimator.init();
    estimator.posterior_mean = 2.0;
    estimator.posterior_variance = 0.25; // std = 0.5

    const ci = try estimator.credible_interval(0.68);

    // 68% CI should be approximately mean ± std
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), ci.lower, 0.1);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), ci.upper, 0.1);
}

test "credible_interval invalid level" {
    const estimator = BayesianCurvatureEstimator.init();

    const result1 = estimator.credible_interval(0.0);
    try std.testing.expectError(BayesianError.InvalidConfidenceLevel, result1);

    const result2 = estimator.credible_interval(1.0);
    try std.testing.expectError(BayesianError.InvalidConfidenceLevel, result2);
}

test "reset estimator" {
    var estimator = BayesianCurvatureEstimator.init();
    const obs = [_]f32{ 1.0, 1.5, 2.0 };
    try estimator.update_posterior(&obs);

    estimator.reset();

    try std.testing.expectEqual(estimator.prior_mean, estimator.posterior_mean);
    try std.testing.expectEqual(estimator.prior_variance, estimator.posterior_variance);
    try std.testing.expectEqual(@as(u32, 0), estimator.observation_count);
}

test "log_gaussian at mean" {
    const log_p = log_gaussian(0.0, 0.0, 1.0);

    // log N(0|0,1) = -0.5 * log(2π)
    try std.testing.expectApproxEqAbs(-0.5 * LOG_2PI, log_p, 1e-4);
}

test "gaussian_pdf standard normal" {
    const pdf = gaussian_pdf(0.0, 0.0, 1.0);

    // N(0|0,1) = 1/sqrt(2π) ≈ 0.3989
    try std.testing.expectApproxEqAbs(@as(f32, 0.3989), pdf, 0.001);
}

test "standard_normal_cdf at zero" {
    const cdf = standard_normal_cdf(0.0);

    // Φ(0) = 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), cdf, 0.01);
}

test "standard_normal_cdf symmetry" {
    const cdf_pos = standard_normal_cdf(1.0);
    const cdf_neg = standard_normal_cdf(-1.0);

    // Φ(-x) = 1 - Φ(x)
    try std.testing.expectApproxEqAbs(cdf_pos, 1.0 - cdf_neg, 0.01);
}

test "normal_quantile at 0.5" {
    const q = normal_quantile(0.5);

    // Φ⁻¹(0.5) = 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), q, 0.01);
}

test "normal_quantile at 0.975" {
    const q = normal_quantile(0.975);

    // Φ⁻¹(0.975) ≈ 1.96
    try std.testing.expectApproxEqAbs(@as(f32, 1.96), q, 0.02);
}

test "normal_quantile at 0.025" {
    const q = normal_quantile(0.025);

    // Φ⁻¹(0.025) ≈ -1.96
    try std.testing.expectApproxEqAbs(@as(f32, -1.96), q, 0.02);
}

test "expected_calibration_error perfect calibration" {
    // Perfect calibration: predictions match outcomes
    const predictions = [_]f32{ 0.0, 0.0, 1.0, 1.0 };
    const outcomes = [_]f32{ 0.0, 0.0, 1.0, 1.0 };

    const ece = try expected_calibration_error(&predictions, &outcomes, 10);

    // Perfect calibration should have ECE close to 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ece, 0.15);
}

test "expected_calibration_error overconfident" {
    // Overconfident: always predict 0.9 but only 50% accuracy
    const predictions = [_]f32{ 0.9, 0.9, 0.9, 0.9 };
    const outcomes = [_]f32{ 1.0, 1.0, 0.0, 0.0 };

    const ece = try expected_calibration_error(&predictions, &outcomes, 10);

    // Should have non-zero ECE (pred avg 0.9, actual 0.5)
    try std.testing.expect(ece > 0.3);
}

test "expected_calibration_error empty predictions" {
    const predictions = [_]f32{};
    const outcomes = [_]f32{};

    const result = expected_calibration_error(&predictions, &outcomes, 10);
    try std.testing.expectError(BayesianError.EmptyObservations, result);
}

test "maximum_calibration_error" {
    const predictions = [_]f32{ 0.1, 0.1, 0.9, 0.9 };
    const outcomes = [_]f32{ 0.0, 0.0, 1.0, 1.0 };

    const mce = try maximum_calibration_error(&predictions, &outcomes, 10);

    // Well calibrated predictions should have low MCE
    try std.testing.expect(mce < 0.2);
}

test "maximum_calibration_error higher than ECE" {
    const predictions = [_]f32{ 0.2, 0.5, 0.8 };
    const outcomes = [_]f32{ 0.0, 1.0, 0.0 };

    const ece = try expected_calibration_error(&predictions, &outcomes, 10);
    const mce = try maximum_calibration_error(&predictions, &outcomes, 10);

    // MCE should always be >= ECE
    try std.testing.expect(mce >= ece - 0.01);
}

test "brier_score perfect predictions" {
    const predictions = [_]f32{ 0.0, 1.0, 0.0, 1.0 };
    const outcomes = [_]f32{ 0.0, 1.0, 0.0, 1.0 };

    const brier = try brier_score(&predictions, &outcomes);

    // Perfect predictions should have Brier score of 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), brier, 1e-5);
}

test "brier_score worst predictions" {
    const predictions = [_]f32{ 1.0, 0.0, 1.0, 0.0 };
    const outcomes = [_]f32{ 0.0, 1.0, 0.0, 1.0 };

    const brier = try brier_score(&predictions, &outcomes);

    // Completely wrong predictions should have Brier score of 1
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), brier, 1e-5);
}

test "brier_score uncertainty" {
    const predictions = [_]f32{ 0.5, 0.5, 0.5, 0.5 };
    const outcomes = [_]f32{ 0.0, 1.0, 0.0, 1.0 };

    const brier = try brier_score(&predictions, &outcomes);

    // 50/50 predictions on balanced outcomes should have Brier ≈ 0.25
    try std.testing.expectApproxEqAbs(@as(f32, 0.25), brier, 1e-5);
}

test "brier_score empty" {
    const predictions = [_]f32{};
    const outcomes = [_]f32{};

    const result = brier_score(&predictions, &outcomes);
    try std.testing.expectError(BayesianError.EmptyObservations, result);
}

test "compute_reliability_diagram basic" {
    const allocator = std.testing.allocator;
    const predictions = [_]f32{ 0.1, 0.3, 0.5, 0.7, 0.9 };
    const outcomes = [_]f32{ 0.0, 0.0, 1.0, 1.0, 1.0 };

    var diagram = try compute_reliability_diagram(&predictions, &outcomes, 10, allocator);
    defer diagram.deinit();

    try std.testing.expectEqual(@as(u32, 10), diagram.n_bins);
    try std.testing.expect(diagram.bin_accuracies.len == 10);
    try std.testing.expect(diagram.bin_confidences.len == 10);
}

test "synthetic observations generation" {
    var buffer: [100]f32 = undefined;
    generate_synthetic_observations(&buffer, 0.5, 0.1, 12345);

    // Compute mean of samples
    var sum: f32 = 0.0;
    for (buffer) |val| {
        sum += val;
        // Check no NaN or Inf
        try std.testing.expect(!math.isNan(val));
        try std.testing.expect(!math.isInf(val));
    }
    const mean = sum / 100.0;

    // Mean should be close to true curvature
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), mean, 0.15);
}

test "posterior update with synthetic data" {
    var estimator = try BayesianCurvatureEstimator.initWithParams(0.0, 1.0, 0.1);
    var observations: [50]f32 = undefined;
    generate_synthetic_observations(&observations, 0.3, 0.1, 42);

    try estimator.update_posterior(&observations);

    // Posterior mean should be close to true curvature
    try std.testing.expectApproxEqAbs(@as(f32, 0.3), estimator.posterior_mean, 0.15);

    // Posterior variance should be small with 50 observations
    try std.testing.expect(estimator.posterior_variance < 0.05);
}

test "get_map_estimate returns posterior mean" {
    var estimator = BayesianCurvatureEstimator.init();
    estimator.posterior_mean = 1.5;

    const map = estimator.get_map_estimate();
    try std.testing.expectEqual(@as(f32, 1.5), map);
}

test "get_posterior_std returns sqrt of variance" {
    var estimator = BayesianCurvatureEstimator.init();
    estimator.posterior_variance = 4.0;

    const std_dev = estimator.get_posterior_std();
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), std_dev, 1e-5);
}

test "validate estimator" {
    var estimator = BayesianCurvatureEstimator.init();
    try estimator.validate();

    estimator.posterior_variance = -1.0;
    const result = estimator.validate();
    try std.testing.expectError(BayesianError.InvalidVariance, result);
}

test "highest_density_interval equals credible interval for Gaussian" {
    const estimator = BayesianCurvatureEstimator.init();

    const ci = try estimator.credible_interval(0.9);
    const hdi = try estimator.highest_density_interval(0.9);

    try std.testing.expectApproxEqAbs(ci.lower, hdi.lower, 1e-5);
    try std.testing.expectApproxEqAbs(ci.upper, hdi.upper, 1e-5);
}
