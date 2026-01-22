// mHC Speculative Decoding Integration Module
// Implements Geometric Validation for Speculative Decoding with mHC Constraints
//
// Core Components:
// - GeometricValidator: Configurable validator with curvature, distance, stability weights
// - Speculative Acceptance Functions: Score computation for draft candidates
// - Speculation Pipeline: Batch validation and candidate selection
// - Integration with Speculative Attention: Context management for speculation
//
// Reference: Day 65 - Speculative mHC Integration

const std = @import("std");
const math = std.math;
const builtin = @import("builtin");

// Import core mHC modules
const mhc_constraints = @import("mhc_constraints.zig");
const mhc_hyperbolic = @import("mhc_hyperbolic.zig");
const mhc_spherical = @import("mhc_spherical.zig");

// ============================================================================
// Constants
// ============================================================================

/// Default curvature weight for geometric validation
pub const DEFAULT_CURVATURE_WEIGHT: f32 = 0.3;

/// Default distance weight for geometric validation
pub const DEFAULT_DISTANCE_WEIGHT: f32 = 0.4;

/// Default stability weight for geometric validation
pub const DEFAULT_STABILITY_WEIGHT: f32 = 0.3;

/// Default acceptance threshold
pub const DEFAULT_ACCEPTANCE_THRESHOLD: f32 = 0.5;

/// Numerical stability epsilon
pub const EPSILON: f32 = 1e-8;

/// SIMD vector width for f32 operations
const SIMD_WIDTH: usize = 8;
const Vec8 = @Vector(SIMD_WIDTH, f32);

// ============================================================================
// GeometricValidator Configuration
// ============================================================================

/// Configuration for geometric validation in speculative decoding
pub const GeometricValidator = struct {
    /// Weight for curvature score in combined acceptance (0-1)
    curvature_weight: f32 = DEFAULT_CURVATURE_WEIGHT,

    /// Weight for distance score in combined acceptance (0-1)
    distance_weight: f32 = DEFAULT_DISTANCE_WEIGHT,

    /// Weight for stability score in combined acceptance (0-1)
    stability_weight: f32 = DEFAULT_STABILITY_WEIGHT,

    /// Threshold for acceptance (0-1), candidates scoring above are accepted
    acceptance_threshold: f32 = DEFAULT_ACCEPTANCE_THRESHOLD,

    /// Temperature for acceptance probability (higher = more lenient)
    temperature: f32 = 1.0,

    /// Maximum allowed curvature deviation
    max_curvature_deviation: f32 = 0.5,

    /// Maximum allowed distance for similarity
    max_distance: f32 = 2.0,

    /// Maximum allowed energy delta for stability
    max_energy_delta: f32 = 1.0,

    /// Enable adaptive thresholds based on sequence position
    adaptive_thresholds: bool = false,

    /// Validate configuration parameters
    pub fn validate(self: GeometricValidator) !void {
        if (self.curvature_weight < 0 or self.curvature_weight > 1) {
            return error.InvalidCurvatureWeight;
        }
        if (self.distance_weight < 0 or self.distance_weight > 1) {
            return error.InvalidDistanceWeight;
        }
        if (self.stability_weight < 0 or self.stability_weight > 1) {
            return error.InvalidStabilityWeight;
        }

        // Weights should sum to 1.0 (with tolerance)
        const weight_sum = self.curvature_weight + self.distance_weight + self.stability_weight;
        if (@abs(weight_sum - 1.0) > 0.01) {
            return error.WeightsSumNotOne;
        }

        if (self.acceptance_threshold < 0 or self.acceptance_threshold > 1) {
            return error.InvalidAcceptanceThreshold;
        }
        if (self.temperature <= 0) {
            return error.InvalidTemperature;
        }
    }

    /// Create validator with default weights
    pub fn default() GeometricValidator {
        return GeometricValidator{};
    }

    /// Create validator with custom weights (auto-normalized)
    pub fn withWeights(curvature: f32, distance: f32, stability: f32) GeometricValidator {
        const sum = curvature + distance + stability;
        const norm = if (sum > EPSILON) sum else 1.0;
        return GeometricValidator{
            .curvature_weight = curvature / norm,
            .distance_weight = distance / norm,
            .stability_weight = stability / norm,
        };
    }

    /// Create a strict validator with high acceptance threshold
    pub fn strict() GeometricValidator {
        return GeometricValidator{
            .acceptance_threshold = 0.7,
            .max_curvature_deviation = 0.3,
            .max_distance = 1.5,
            .max_energy_delta = 0.5,
        };
    }

    /// Create a lenient validator with low acceptance threshold
    pub fn lenient() GeometricValidator {
        return GeometricValidator{
            .acceptance_threshold = 0.3,
            .max_curvature_deviation = 1.0,
            .max_distance = 3.0,
            .max_energy_delta = 2.0,
        };
    }
};

// ============================================================================
// SpeculativeCandidate
// ============================================================================

/// Maximum embedding dimension for stack allocation
pub const MAX_EMBEDDING_DIM: usize = 512;

/// Represents a speculative decoding candidate with geometric properties
pub const SpeculativeCandidate = struct {
    /// Token ID of the candidate
    token_id: u32,

    /// Embedding vector of the candidate (slice into external buffer)
    embedding: []const f32,

    /// Curvature at this point in the sequence
    curvature: f32,

    /// Energy level of the candidate
    energy: f32,

    /// Log probability from the draft model
    log_prob: f32,

    /// Position in the speculative sequence
    position: u32,

    /// Create a candidate from raw data
    pub fn init(
        token_id: u32,
        embedding: []const f32,
        curvature: f32,
        energy: f32,
        log_prob: f32,
        position: u32,
    ) SpeculativeCandidate {
        return SpeculativeCandidate{
            .token_id = token_id,
            .embedding = embedding,
            .curvature = curvature,
            .energy = energy,
            .log_prob = log_prob,
            .position = position,
        };
    }
};

/// Target context for validation
pub const TargetContext = struct {
    /// Target embedding for comparison
    embedding: []const f32,

    /// Target curvature
    curvature: f32,

    /// Target energy level
    energy: f32,

    /// Baseline energy for delta computation
    baseline_energy: f32,
};

/// Validation result for a candidate
pub const ValidationResult = struct {
    /// Whether the candidate is accepted
    accepted: bool,

    /// Combined acceptance score (0-1)
    score: f32,

    /// Individual curvature score (0-1)
    curvature_score: f32,

    /// Individual distance score (0-1)
    distance_score: f32,

    /// Individual stability score (0-1)
    stability_score: f32,

    /// Create an accepted result
    pub fn accept(score: f32, curvature: f32, distance: f32, stability: f32) ValidationResult {
        return ValidationResult{
            .accepted = true,
            .score = score,
            .curvature_score = curvature,
            .distance_score = distance,
            .stability_score = stability,
        };
    }

    /// Create a rejected result
    pub fn reject(score: f32, curvature: f32, distance: f32, stability: f32) ValidationResult {
        return ValidationResult{
            .accepted = false,
            .score = score,
            .curvature_score = curvature,
            .distance_score = distance,
            .stability_score = stability,
        };
    }
};

// ============================================================================
// Speculative Acceptance Functions
// ============================================================================

/// Compute curvature score based on how close curvatures match
///
/// Score = 1 - (|draft_curvature - target_curvature| / max_deviation)
/// Clamped to [0, 1]
///
/// Parameters:
///   - draft_curvature: Curvature of the draft candidate
///   - target_curvature: Expected target curvature
///   - max_deviation: Maximum allowed curvature deviation
///
/// Returns: Score in [0, 1], where 1 = perfect match
pub fn compute_curvature_score(
    draft_curvature: f32,
    target_curvature: f32,
    max_deviation: f32,
) f32 {
    const deviation = @abs(draft_curvature - target_curvature);
    const safe_max = @max(max_deviation, EPSILON);
    const normalized = deviation / safe_max;
    return @max(0.0, 1.0 - normalized);
}

/// Compute distance score based on embedding similarity
///
/// Score = exp(-distance / temperature) or 1 - (distance / max_distance)
/// Uses cosine similarity for normalized embeddings.
///
/// Parameters:
///   - draft_embedding: Embedding of the draft candidate
///   - target_embedding: Target embedding for comparison
///   - max_distance: Maximum distance for normalization
///
/// Returns: Score in [0, 1], where 1 = identical embeddings
pub fn compute_distance_score(
    draft_embedding: []const f32,
    target_embedding: []const f32,
    max_distance: f32,
) f32 {
    if (draft_embedding.len != target_embedding.len or draft_embedding.len == 0) {
        return 0.0;
    }

    // Compute Euclidean distance with SIMD
    const distance = euclidean_distance_simd(draft_embedding, target_embedding);

    // Convert to similarity score
    const safe_max = @max(max_distance, EPSILON);
    const normalized = distance / safe_max;
    return @max(0.0, 1.0 - normalized);
}

/// Compute stability score based on energy delta
///
/// Score = exp(-|energy_delta| / temperature) or 1 - (|delta| / max_delta)
/// Lower energy changes indicate more stable predictions.
///
/// Parameters:
///   - energy_delta: Change in energy from baseline
///   - max_energy_delta: Maximum expected energy change
///
/// Returns: Score in [0, 1], where 1 = perfectly stable
pub fn compute_stability_score(energy_delta: f32, max_energy_delta: f32) f32 {
    const abs_delta = @abs(energy_delta);
    const safe_max = @max(max_energy_delta, EPSILON);
    const normalized = abs_delta / safe_max;
    return @max(0.0, 1.0 - normalized);
}

/// Compute combined acceptance score using weighted combination
///
/// Combined = w_c * curvature_score + w_d * distance_score + w_s * stability_score
///
/// Parameters:
///   - candidate: Draft candidate to evaluate
///   - target: Target context for comparison
///   - validator: Geometric validator configuration
///
/// Returns: Combined score in [0, 1]
pub fn compute_combined_acceptance(
    candidate: SpeculativeCandidate,
    target: TargetContext,
    validator: GeometricValidator,
) f32 {
    // Compute individual scores
    const curvature_score = compute_curvature_score(
        candidate.curvature,
        target.curvature,
        validator.max_curvature_deviation,
    );

    const distance_score = compute_distance_score(
        candidate.embedding,
        target.embedding,
        validator.max_distance,
    );

    const energy_delta = candidate.energy - target.baseline_energy;
    const stability_score = compute_stability_score(
        energy_delta,
        validator.max_energy_delta,
    );

    // Weighted combination
    return validator.curvature_weight * curvature_score +
        validator.distance_weight * distance_score +
        validator.stability_weight * stability_score;
}

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// Compute Euclidean distance between two vectors with SIMD optimization
pub fn euclidean_distance_simd(a: []const f32, b: []const f32) f32 {
    const n = @min(a.len, b.len);
    var sum_sq: f32 = 0.0;

    // SIMD path
    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const a_vec: Vec8 = a[i..][0..SIMD_WIDTH].*;
        const b_vec: Vec8 = b[i..][0..SIMD_WIDTH].*;
        const diff = a_vec - b_vec;
        const sq = diff * diff;
        sum_sq += @reduce(.Add, sq);
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        const diff = a[i] - b[i];
        sum_sq += diff * diff;
    }

    return @sqrt(sum_sq);
}

/// Compute cosine similarity between two vectors with SIMD
pub fn cosine_similarity_simd(a: []const f32, b: []const f32) f32 {
    const n = @min(a.len, b.len);
    var dot: f32 = 0.0;
    var norm_a_sq: f32 = 0.0;
    var norm_b_sq: f32 = 0.0;

    // SIMD path
    var i: usize = 0;
    while (i + SIMD_WIDTH <= n) : (i += SIMD_WIDTH) {
        const a_vec: Vec8 = a[i..][0..SIMD_WIDTH].*;
        const b_vec: Vec8 = b[i..][0..SIMD_WIDTH].*;
        dot += @reduce(.Add, a_vec * b_vec);
        norm_a_sq += @reduce(.Add, a_vec * a_vec);
        norm_b_sq += @reduce(.Add, b_vec * b_vec);
    }

    // Scalar remainder
    while (i < n) : (i += 1) {
        dot += a[i] * b[i];
        norm_a_sq += a[i] * a[i];
        norm_b_sq += b[i] * b[i];
    }

    const denom = @sqrt(norm_a_sq) * @sqrt(norm_b_sq);
    if (denom < EPSILON) return 0.0;
    return dot / denom;
}

/// Compute L2 norm with SIMD
pub fn norm_simd(x: []const f32) f32 {
    var sum_sq: f32 = 0.0;

    var i: usize = 0;
    while (i + SIMD_WIDTH <= x.len) : (i += SIMD_WIDTH) {
        const vec: Vec8 = x[i..][0..SIMD_WIDTH].*;
        sum_sq += @reduce(.Add, vec * vec);
    }

    while (i < x.len) : (i += 1) {
        sum_sq += x[i] * x[i];
    }

    return @sqrt(sum_sq);
}

// ============================================================================
// Speculation Pipeline
// ============================================================================

/// Validate a single candidate against target context
///
/// Returns (accepted, score) tuple indicating if candidate passes threshold
///
/// Parameters:
///   - candidate: Draft candidate to validate
///   - target: Target context for comparison
///   - validator: Geometric validator configuration
///
/// Returns: ValidationResult with acceptance status and scores
pub fn validate_candidate(
    candidate: SpeculativeCandidate,
    target: TargetContext,
    validator: GeometricValidator,
) ValidationResult {
    // Compute individual scores
    const curvature_score = compute_curvature_score(
        candidate.curvature,
        target.curvature,
        validator.max_curvature_deviation,
    );

    const distance_score = compute_distance_score(
        candidate.embedding,
        target.embedding,
        validator.max_distance,
    );

    const energy_delta = candidate.energy - target.baseline_energy;
    const stability_score = compute_stability_score(
        energy_delta,
        validator.max_energy_delta,
    );

    // Compute combined score
    const combined = validator.curvature_weight * curvature_score +
        validator.distance_weight * distance_score +
        validator.stability_weight * stability_score;

    // Apply temperature scaling
    const scaled_score = if (validator.temperature != 1.0)
        @min(1.0, combined * validator.temperature)
    else
        combined;

    // Check acceptance threshold
    const accepted = scaled_score >= validator.acceptance_threshold;

    return if (accepted)
        ValidationResult.accept(scaled_score, curvature_score, distance_score, stability_score)
    else
        ValidationResult.reject(scaled_score, curvature_score, distance_score, stability_score);
}

/// Batch validation result
pub const BatchValidationResult = struct {
    /// Number of accepted candidates
    num_accepted: u32,

    /// Total number of candidates validated
    num_total: u32,

    /// Acceptance rate (0-1)
    acceptance_rate: f32,

    /// Average score across all candidates
    avg_score: f32,

    /// Index of best candidate
    best_idx: u32,

    /// Score of best candidate
    best_score: f32,

    /// Array of individual results (external buffer)
    results: []ValidationResult,
};

/// Batch validate multiple candidates
///
/// Parameters:
///   - candidates: Array of draft candidates
///   - target: Target context for comparison
///   - validator: Geometric validator configuration
///   - results: Pre-allocated array for results (must be >= candidates.len)
///
/// Returns: BatchValidationResult with aggregate statistics
pub fn batch_validate(
    candidates: []const SpeculativeCandidate,
    target: TargetContext,
    validator: GeometricValidator,
    results: []ValidationResult,
) BatchValidationResult {
    if (candidates.len == 0 or results.len < candidates.len) {
        return BatchValidationResult{
            .num_accepted = 0,
            .num_total = 0,
            .acceptance_rate = 0.0,
            .avg_score = 0.0,
            .best_idx = 0,
            .best_score = 0.0,
            .results = results[0..0],
        };
    }

    var num_accepted: u32 = 0;
    var total_score: f32 = 0.0;
    var best_idx: u32 = 0;
    var best_score: f32 = 0.0;

    for (candidates, 0..) |candidate, i| {
        const result = validate_candidate(candidate, target, validator);
        results[i] = result;

        if (result.accepted) {
            num_accepted += 1;
        }

        total_score += result.score;

        if (result.score > best_score) {
            best_score = result.score;
            best_idx = @intCast(i);
        }
    }

    const num_total: u32 = @intCast(candidates.len);
    const acceptance_rate = @as(f32, @floatFromInt(num_accepted)) / @as(f32, @floatFromInt(num_total));
    const avg_score = total_score / @as(f32, @floatFromInt(num_total));

    return BatchValidationResult{
        .num_accepted = num_accepted,
        .num_total = num_total,
        .acceptance_rate = acceptance_rate,
        .avg_score = avg_score,
        .best_idx = best_idx,
        .best_score = best_score,
        .results = results[0..candidates.len],
    };
}

/// Find the best candidate from a batch
///
/// Returns the candidate with the highest score that meets the threshold.
/// If no candidate meets the threshold, returns null.
///
/// Parameters:
///   - candidates: Array of draft candidates
///   - target: Target context for comparison
///   - validator: Geometric validator configuration
///
/// Returns: Optional index of best candidate, or null if none accepted
pub fn find_best_candidate(
    candidates: []const SpeculativeCandidate,
    target: TargetContext,
    validator: GeometricValidator,
) ?u32 {
    if (candidates.len == 0) return null;

    var best_idx: ?u32 = null;
    var best_score: f32 = 0.0;

    for (candidates, 0..) |candidate, i| {
        const result = validate_candidate(candidate, target, validator);

        if (result.accepted and result.score > best_score) {
            best_score = result.score;
            best_idx = @intCast(i);
        }
    }

    return best_idx;
}

/// Find the longest accepted prefix in a sequence of candidates
///
/// Returns the length of the longest prefix where all candidates are accepted.
/// Useful for speculative decoding where we want consecutive accepted tokens.
///
/// Parameters:
///   - candidates: Array of draft candidates in sequence order
///   - targets: Array of target contexts (one per candidate)
///   - validator: Geometric validator configuration
///
/// Returns: Length of longest accepted prefix
pub fn find_longest_accepted_prefix(
    candidates: []const SpeculativeCandidate,
    targets: []const TargetContext,
    validator: GeometricValidator,
) u32 {
    const n = @min(candidates.len, targets.len);
    var prefix_len: u32 = 0;

    for (0..n) |i| {
        const result = validate_candidate(candidates[i], targets[i], validator);
        if (!result.accepted) break;
        prefix_len = @intCast(i + 1);
    }

    return prefix_len;
}

// ============================================================================
// GeometricSpeculationContext
// ============================================================================

/// Context for managing geometric state during speculative decoding
pub const GeometricSpeculationContext = struct {
    /// Current curvature estimate
    current_curvature: f32,

    /// Current energy level
    current_energy: f32,

    /// Running mean of accepted curvatures
    mean_curvature: f32,

    /// Running variance of curvatures
    curvature_variance: f32,

    /// Number of samples used for statistics
    num_samples: u32,

    /// Baseline energy for stability computation
    baseline_energy: f32,

    /// Accumulated curvature for sequence
    accumulated_curvature: f32,

    /// Total tokens processed
    total_tokens: u32,

    /// Last accepted position
    last_accepted_position: u32,

    /// Validator configuration
    validator: GeometricValidator,

    /// Allocator for internal buffers
    allocator: std.mem.Allocator,

    /// Initialize a new speculation context
    pub fn init(allocator: std.mem.Allocator, validator: GeometricValidator) GeometricSpeculationContext {
        return GeometricSpeculationContext{
            .current_curvature = 0.0,
            .current_energy = 0.0,
            .mean_curvature = 0.0,
            .curvature_variance = 0.0,
            .num_samples = 0,
            .baseline_energy = 0.0,
            .accumulated_curvature = 0.0,
            .total_tokens = 0,
            .last_accepted_position = 0,
            .validator = validator,
            .allocator = allocator,
        };
    }

    /// Reset the context to initial state
    pub fn reset(self: *GeometricSpeculationContext) void {
        self.current_curvature = 0.0;
        self.current_energy = 0.0;
        self.mean_curvature = 0.0;
        self.curvature_variance = 0.0;
        self.num_samples = 0;
        self.baseline_energy = 0.0;
        self.accumulated_curvature = 0.0;
        self.total_tokens = 0;
        self.last_accepted_position = 0;
    }

    /// Prepare geometric context for speculation
    ///
    /// Updates internal state based on draft tokens and computes
    /// expected geometric properties for validation.
    ///
    /// Parameters:
    ///   - draft_curvatures: Curvatures of draft tokens
    ///   - draft_energies: Energy levels of draft tokens
    ///
    /// Returns: TargetContext for validation
    pub fn prepare_speculation(
        self: *GeometricSpeculationContext,
        draft_curvatures: []const f32,
        draft_energies: []const f32,
    ) TargetContext {
        // Update current estimates based on draft tokens
        if (draft_curvatures.len > 0) {
            var sum_curvature: f32 = 0.0;
            for (draft_curvatures) |c| {
                sum_curvature += c;
            }
            self.current_curvature = sum_curvature / @as(f32, @floatFromInt(draft_curvatures.len));
        }

        if (draft_energies.len > 0) {
            self.current_energy = draft_energies[draft_energies.len - 1];
        }

        // Return target context for validation
        return TargetContext{
            .embedding = &.{}, // Will be set by caller
            .curvature = self.mean_curvature,
            .energy = self.current_energy,
            .baseline_energy = self.baseline_energy,
        };
    }

    /// Finalize speculation after acceptance decisions
    ///
    /// Updates running statistics and geometric state based on
    /// which tokens were accepted.
    ///
    /// Parameters:
    ///   - accepted_curvatures: Curvatures of accepted tokens
    ///   - accepted_energies: Energies of accepted tokens
    ///   - num_accepted: Number of tokens accepted
    pub fn finalize_speculation(
        self: *GeometricSpeculationContext,
        accepted_curvatures: []const f32,
        accepted_energies: []const f32,
        num_accepted: u32,
    ) void {
        // Update running statistics with Welford's online algorithm
        for (accepted_curvatures) |curvature| {
            self.num_samples += 1;
            const delta = curvature - self.mean_curvature;
            self.mean_curvature += delta / @as(f32, @floatFromInt(self.num_samples));
            const delta2 = curvature - self.mean_curvature;
            self.curvature_variance += delta * delta2;
            self.accumulated_curvature += curvature;
        }

        // Update energy baseline
        if (accepted_energies.len > 0) {
            const last_energy = accepted_energies[accepted_energies.len - 1];
            // Exponential moving average for baseline
            const alpha: f32 = 0.1;
            self.baseline_energy = alpha * last_energy + (1.0 - alpha) * self.baseline_energy;
        }

        // Update position tracking
        self.total_tokens += num_accepted;
        self.last_accepted_position = self.total_tokens;
    }

    /// Get current curvature variance (unbiased estimator)
    pub fn get_curvature_variance(self: *const GeometricSpeculationContext) f32 {
        if (self.num_samples < 2) return 0.0;
        return self.curvature_variance / @as(f32, @floatFromInt(self.num_samples - 1));
    }

    /// Get current curvature standard deviation
    pub fn get_curvature_std(self: *const GeometricSpeculationContext) f32 {
        return @sqrt(self.get_curvature_variance());
    }

    /// Check if speculation should be terminated early
    ///
    /// Based on accumulated curvature deviation and energy instability.
    pub fn should_terminate_speculation(
        self: *const GeometricSpeculationContext,
        current_curvature: f32,
        current_energy: f32,
    ) bool {
        // Check curvature deviation
        const curvature_deviation = @abs(current_curvature - self.mean_curvature);
        if (curvature_deviation > self.validator.max_curvature_deviation * 2.0) {
            return true;
        }

        // Check energy spike
        const energy_delta = @abs(current_energy - self.baseline_energy);
        if (energy_delta > self.validator.max_energy_delta * 2.0) {
            return true;
        }

        return false;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "GeometricValidator default configuration" {
    const validator = GeometricValidator.default();
    try std.testing.expectApproxEqAbs(validator.curvature_weight, 0.3, 0.001);
    try std.testing.expectApproxEqAbs(validator.distance_weight, 0.4, 0.001);
    try std.testing.expectApproxEqAbs(validator.stability_weight, 0.3, 0.001);
    try std.testing.expectApproxEqAbs(validator.acceptance_threshold, 0.5, 0.001);
}

test "GeometricValidator validation passes for valid config" {
    const validator = GeometricValidator.default();
    try validator.validate();
}

test "GeometricValidator validation fails for invalid weights" {
    const invalid = GeometricValidator{
        .curvature_weight = 0.5,
        .distance_weight = 0.5,
        .stability_weight = 0.5, // Sum = 1.5, not 1.0
    };
    try std.testing.expectError(error.WeightsSumNotOne, invalid.validate());
}

test "GeometricValidator withWeights normalizes" {
    const validator = GeometricValidator.withWeights(1.0, 2.0, 1.0);
    try std.testing.expectApproxEqAbs(validator.curvature_weight, 0.25, 0.001);
    try std.testing.expectApproxEqAbs(validator.distance_weight, 0.5, 0.001);
    try std.testing.expectApproxEqAbs(validator.stability_weight, 0.25, 0.001);
}

test "GeometricValidator strict has higher threshold" {
    const strict = GeometricValidator.strict();
    const default_val = GeometricValidator.default();
    try std.testing.expect(strict.acceptance_threshold > default_val.acceptance_threshold);
}

test "GeometricValidator lenient has lower threshold" {
    const lenient = GeometricValidator.lenient();
    const default_val = GeometricValidator.default();
    try std.testing.expect(lenient.acceptance_threshold < default_val.acceptance_threshold);
}

test "compute_curvature_score perfect match" {
    const score = compute_curvature_score(0.5, 0.5, 1.0);
    try std.testing.expectApproxEqAbs(score, 1.0, 0.001);
}

test "compute_curvature_score max deviation" {
    const score = compute_curvature_score(1.0, 0.0, 1.0);
    try std.testing.expectApproxEqAbs(score, 0.0, 0.001);
}

test "compute_curvature_score partial match" {
    const score = compute_curvature_score(0.5, 0.0, 1.0);
    try std.testing.expectApproxEqAbs(score, 0.5, 0.001);
}

test "compute_distance_score identical embeddings" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const score = compute_distance_score(&a, &b, 2.0);
    try std.testing.expectApproxEqAbs(score, 1.0, 0.001);
}

test "compute_distance_score maximum distance" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ -1.0, 0.0, 0.0 };
    const score = compute_distance_score(&a, &b, 2.0);
    try std.testing.expectApproxEqAbs(score, 0.0, 0.001);
}

test "compute_distance_score partial distance" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 0.0, 0.0 };
    const score = compute_distance_score(&a, &b, 2.0);
    try std.testing.expectApproxEqAbs(score, 0.5, 0.001);
}

test "compute_stability_score zero delta" {
    const score = compute_stability_score(0.0, 1.0);
    try std.testing.expectApproxEqAbs(score, 1.0, 0.001);
}

test "compute_stability_score max delta" {
    const score = compute_stability_score(1.0, 1.0);
    try std.testing.expectApproxEqAbs(score, 0.0, 0.001);
}

test "compute_stability_score negative delta" {
    const score = compute_stability_score(-0.5, 1.0);
    try std.testing.expectApproxEqAbs(score, 0.5, 0.001);
}

test "euclidean_distance_simd zero distance" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const dist = euclidean_distance_simd(&a, &b);
    try std.testing.expectApproxEqAbs(dist, 0.0, 0.001);
}

test "euclidean_distance_simd unit distance" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 1.0, 0.0, 0.0 };
    const dist = euclidean_distance_simd(&a, &b);
    try std.testing.expectApproxEqAbs(dist, 1.0, 0.001);
}

test "cosine_similarity_simd identical" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    const sim = cosine_similarity_simd(&a, &b);
    try std.testing.expectApproxEqAbs(sim, 1.0, 0.001);
}

test "cosine_similarity_simd orthogonal" {
    const a = [_]f32{ 1.0, 0.0, 0.0 };
    const b = [_]f32{ 0.0, 1.0, 0.0 };
    const sim = cosine_similarity_simd(&a, &b);
    try std.testing.expectApproxEqAbs(sim, 0.0, 0.001);
}

test "norm_simd unit vector" {
    const v = [_]f32{ 1.0, 0.0, 0.0 };
    const n = norm_simd(&v);
    try std.testing.expectApproxEqAbs(n, 1.0, 0.001);
}

test "norm_simd pythagorean" {
    const v = [_]f32{ 3.0, 4.0 };
    const n = norm_simd(&v);
    try std.testing.expectApproxEqAbs(n, 5.0, 0.001);
}

test "validate_candidate accepts high scoring candidate" {
    const embedding = [_]f32{ 1.0, 0.0, 0.0 };
    const candidate = SpeculativeCandidate.init(1, &embedding, 0.5, 1.0, -0.5, 0);
    const target = TargetContext{
        .embedding = &embedding,
        .curvature = 0.5,
        .energy = 1.0,
        .baseline_energy = 1.0,
    };
    const validator = GeometricValidator.default();
    const result = validate_candidate(candidate, target, validator);

    try std.testing.expect(result.accepted);
    try std.testing.expect(result.score >= validator.acceptance_threshold);
}

test "validate_candidate rejects low scoring candidate" {
    const draft_emb = [_]f32{ 1.0, 0.0, 0.0 };
    const target_emb = [_]f32{ -1.0, 0.0, 0.0 };
    const candidate = SpeculativeCandidate.init(1, &draft_emb, 1.0, 10.0, -0.5, 0);
    const target = TargetContext{
        .embedding = &target_emb,
        .curvature = -1.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };
    const validator = GeometricValidator.strict();
    const result = validate_candidate(candidate, target, validator);

    try std.testing.expect(!result.accepted);
}

test "compute_combined_acceptance weighted correctly" {
    const embedding = [_]f32{ 1.0, 0.0, 0.0 };
    const candidate = SpeculativeCandidate.init(1, &embedding, 0.0, 0.0, 0.0, 0);
    const target = TargetContext{
        .embedding = &embedding,
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };
    const validator = GeometricValidator.default();
    const score = compute_combined_acceptance(candidate, target, validator);

    // All scores should be 1.0, so combined should be 1.0
    try std.testing.expectApproxEqAbs(score, 1.0, 0.001);
}

test "batch_validate counts correctly" {
    const emb1 = [_]f32{ 1.0, 0.0, 0.0 };
    const emb2 = [_]f32{ 0.0, 1.0, 0.0 };
    const emb3 = [_]f32{ -1.0, 0.0, 0.0 };

    const candidates = [_]SpeculativeCandidate{
        SpeculativeCandidate.init(1, &emb1, 0.0, 0.0, 0.0, 0),
        SpeculativeCandidate.init(2, &emb2, 0.5, 0.5, -0.5, 1),
        SpeculativeCandidate.init(3, &emb3, 1.0, 1.0, -1.0, 2),
    };

    const target = TargetContext{
        .embedding = &emb1,
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };

    const validator = GeometricValidator.default();
    var results: [3]ValidationResult = undefined;

    const batch_result = batch_validate(&candidates, target, validator, &results);

    try std.testing.expect(batch_result.num_total == 3);
    try std.testing.expect(batch_result.num_accepted >= 1); // At least first should be accepted
    try std.testing.expect(batch_result.best_idx == 0); // First candidate is best match
}

test "find_best_candidate returns best" {
    const emb1 = [_]f32{ 1.0, 0.0, 0.0 };
    const emb2 = [_]f32{ 0.9, 0.1, 0.0 };

    const candidates = [_]SpeculativeCandidate{
        SpeculativeCandidate.init(1, &emb2, 0.1, 0.1, 0.0, 0), // Good but not perfect
        SpeculativeCandidate.init(2, &emb1, 0.0, 0.0, 0.0, 1), // Perfect match
    };

    const target = TargetContext{
        .embedding = &emb1,
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };

    const validator = GeometricValidator.default();
    const best = find_best_candidate(&candidates, target, validator);

    try std.testing.expect(best != null);
    try std.testing.expect(best.? == 1); // Second candidate is best
}

test "find_best_candidate returns null when none accepted" {
    const emb1 = [_]f32{ 1.0, 0.0, 0.0 };
    const emb2 = [_]f32{ -1.0, 0.0, 0.0 };

    const candidates = [_]SpeculativeCandidate{
        SpeculativeCandidate.init(1, &emb2, 2.0, 5.0, 0.0, 0), // Very different
    };

    const target = TargetContext{
        .embedding = &emb1,
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };

    // Use strict validator
    const validator = GeometricValidator.strict();
    const best = find_best_candidate(&candidates, target, validator);

    try std.testing.expect(best == null);
}

test "find_longest_accepted_prefix all accepted" {
    const emb = [_]f32{ 1.0, 0.0, 0.0 };

    const candidates = [_]SpeculativeCandidate{
        SpeculativeCandidate.init(1, &emb, 0.0, 0.0, 0.0, 0),
        SpeculativeCandidate.init(2, &emb, 0.0, 0.0, 0.0, 1),
        SpeculativeCandidate.init(3, &emb, 0.0, 0.0, 0.0, 2),
    };

    const target = TargetContext{
        .embedding = &emb,
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };

    const targets = [_]TargetContext{ target, target, target };
    const validator = GeometricValidator.lenient();

    const prefix_len = find_longest_accepted_prefix(&candidates, &targets, validator);
    try std.testing.expect(prefix_len == 3);
}

test "GeometricSpeculationContext init and reset" {
    const allocator = std.testing.allocator;
    const validator = GeometricValidator.default();
    var ctx = GeometricSpeculationContext.init(allocator, validator);

    try std.testing.expectApproxEqAbs(ctx.current_curvature, 0.0, 0.001);
    try std.testing.expectApproxEqAbs(ctx.mean_curvature, 0.0, 0.001);
    try std.testing.expect(ctx.num_samples == 0);

    // Modify and reset
    ctx.current_curvature = 1.0;
    ctx.num_samples = 10;
    ctx.reset();

    try std.testing.expectApproxEqAbs(ctx.current_curvature, 0.0, 0.001);
    try std.testing.expect(ctx.num_samples == 0);
}

test "GeometricSpeculationContext finalize_speculation updates stats" {
    const allocator = std.testing.allocator;
    const validator = GeometricValidator.default();
    var ctx = GeometricSpeculationContext.init(allocator, validator);

    const curvatures = [_]f32{ 0.1, 0.2, 0.3 };
    const energies = [_]f32{ 1.0, 1.1, 1.2 };

    ctx.finalize_speculation(&curvatures, &energies, 3);

    try std.testing.expect(ctx.num_samples == 3);
    try std.testing.expect(ctx.total_tokens == 3);
    try std.testing.expectApproxEqAbs(ctx.mean_curvature, 0.2, 0.001);
}

test "GeometricSpeculationContext get_curvature_variance" {
    const allocator = std.testing.allocator;
    const validator = GeometricValidator.default();
    var ctx = GeometricSpeculationContext.init(allocator, validator);

    // Before any samples, variance should be 0
    try std.testing.expectApproxEqAbs(ctx.get_curvature_variance(), 0.0, 0.001);

    // Add samples with known variance
    const curvatures = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const energies = [_]f32{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    ctx.finalize_speculation(&curvatures, &energies, 5);

    // Mean should be 3.0, variance should be 2.5
    try std.testing.expectApproxEqAbs(ctx.mean_curvature, 3.0, 0.001);
    try std.testing.expectApproxEqAbs(ctx.get_curvature_variance(), 2.5, 0.01);
}

test "GeometricSpeculationContext should_terminate_speculation" {
    const allocator = std.testing.allocator;
    const validator = GeometricValidator.default();
    var ctx = GeometricSpeculationContext.init(allocator, validator);

    // Set some baseline values
    ctx.mean_curvature = 0.5;
    ctx.baseline_energy = 1.0;

    // Normal values should not terminate
    try std.testing.expect(!ctx.should_terminate_speculation(0.6, 1.1));

    // Extreme curvature should terminate
    try std.testing.expect(ctx.should_terminate_speculation(5.0, 1.0));

    // Extreme energy should terminate
    try std.testing.expect(ctx.should_terminate_speculation(0.5, 10.0));
}

test "SpeculativeCandidate init" {
    const emb = [_]f32{ 1.0, 2.0, 3.0 };
    const candidate = SpeculativeCandidate.init(42, &emb, 0.5, 1.0, -0.3, 7);

    try std.testing.expect(candidate.token_id == 42);
    try std.testing.expectApproxEqAbs(candidate.curvature, 0.5, 0.001);
    try std.testing.expectApproxEqAbs(candidate.energy, 1.0, 0.001);
    try std.testing.expectApproxEqAbs(candidate.log_prob, -0.3, 0.001);
    try std.testing.expect(candidate.position == 7);
}

test "ValidationResult accept and reject" {
    const accepted = ValidationResult.accept(0.8, 0.7, 0.9, 0.75);
    try std.testing.expect(accepted.accepted);
    try std.testing.expectApproxEqAbs(accepted.score, 0.8, 0.001);

    const rejected = ValidationResult.reject(0.3, 0.2, 0.4, 0.3);
    try std.testing.expect(!rejected.accepted);
    try std.testing.expectApproxEqAbs(rejected.score, 0.3, 0.001);
}

test "batch_validate empty candidates" {
    const candidates: []const SpeculativeCandidate = &.{};
    const target = TargetContext{
        .embedding = &.{},
        .curvature = 0.0,
        .energy = 0.0,
        .baseline_energy = 0.0,
    };
    const validator = GeometricValidator.default();
    var results: [0]ValidationResult = .{};

    const batch_result = batch_validate(candidates, target, validator, &results);

    try std.testing.expect(batch_result.num_total == 0);
    try std.testing.expect(batch_result.num_accepted == 0);
    try std.testing.expectApproxEqAbs(batch_result.acceptance_rate, 0.0, 0.001);
}

test "SIMD operations with large vectors" {
    // Test with vectors larger than SIMD width
    var a: [32]f32 = undefined;
    var b: [32]f32 = undefined;

    for (0..32) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(i)) * 0.1;
    }

    const dist = euclidean_distance_simd(&a, &b);
    try std.testing.expectApproxEqAbs(dist, 0.0, 0.001);

    const sim = cosine_similarity_simd(&a, &b);
    try std.testing.expectApproxEqAbs(sim, 1.0, 0.001);
}

test "GeometricSpeculationContext prepare_speculation" {
    const allocator = std.testing.allocator;
    const validator = GeometricValidator.default();
    var ctx = GeometricSpeculationContext.init(allocator, validator);

    const curvatures = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    const energies = [_]f32{ 1.0, 1.1, 1.2, 1.3 };

    const target_ctx = ctx.prepare_speculation(&curvatures, &energies);

    // Current curvature should be mean of draft curvatures
    try std.testing.expectApproxEqAbs(ctx.current_curvature, 0.25, 0.001);
    // Current energy should be last draft energy
    try std.testing.expectApproxEqAbs(ctx.current_energy, 1.3, 0.001);
    // Target curvature should be mean curvature (initially 0)
    try std.testing.expectApproxEqAbs(target_ctx.curvature, 0.0, 0.001);
}

