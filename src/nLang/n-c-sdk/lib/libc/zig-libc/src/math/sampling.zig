// Probabilistic sampling algorithms for language model inference
// Pure Zig implementations without external dependencies

const std = @import("std");
const math = std.math;

// =============================================================================
// Configuration
// =============================================================================

/// Sampling configuration for language model inference
pub const SamplingConfig = extern struct {
    temperature: f32 = 1.0,
    top_k: u32 = 0, // 0 = disabled
    top_p: f32 = 1.0, // 1.0 = disabled
    repetition_penalty: f32 = 1.0, // 1.0 = disabled
    frequency_penalty: f32 = 0.0, // 0.0 = disabled
    presence_penalty: f32 = 0.0, // 0.0 = disabled
};

// =============================================================================
// Core Sampling Functions
// =============================================================================

/// Greedy sampling: return index of maximum value (argmax)
pub fn sample_greedy(logits: [*]const f32, n: usize) usize {
    if (n == 0) return 0;

    var max_idx: usize = 0;
    var max_val = logits[0];

    for (1..n) |i| {
        if (logits[i] > max_val) {
            max_val = logits[i];
            max_idx = i;
        }
    }

    return max_idx;
}

/// Temperature scaling: scale logits by temperature, output to result
/// Higher temperature = more random, lower = more deterministic
pub fn sample_temperature(
    logits: [*]const f32,
    n: usize,
    temperature: f32,
    result: [*]f32,
) void {
    if (n == 0) return;

    // Clamp temperature to avoid division by zero or negative
    const temp = @max(1e-7, temperature);

    for (0..n) |i| {
        result[i] = logits[i] / temp;
    }
}

/// Top-k sampling: keep only top-k highest values, zero out rest, renormalize
pub fn sample_top_k(
    logits: [*]const f32,
    n: usize,
    k: usize,
    result: [*]f32,
) void {
    if (n == 0) return;
    if (k == 0 or k >= n) {
        // Copy all logits as-is
        for (0..n) |i| {
            result[i] = logits[i];
        }
        return;
    }

    // Find the k-th largest value using partial selection
    // First, copy logits to result
    for (0..n) |i| {
        result[i] = logits[i];
    }

    // Find threshold: k-th largest value
    // Simple approach: find top-k values
    var threshold: f32 = -math.inf(f32);

    // Use a simple O(n*k) approach for small k
    var found_count: usize = 0;
    var temp_threshold: f32 = math.inf(f32);

    for (0..k) |_| {
        var current_max: f32 = -math.inf(f32);
        for (0..n) |i| {
            if (logits[i] < temp_threshold and logits[i] > current_max) {
                current_max = logits[i];
            } else if (logits[i] == temp_threshold and found_count == 0) {
                current_max = logits[i];
            }
        }
        if (current_max > -math.inf(f32)) {
            threshold = current_max;
            temp_threshold = current_max;
            found_count = 0;
            // Count how many have this value
            for (0..n) |i| {
                if (logits[i] == current_max) found_count += 1;
            }
        }
    }

    // Zero out values below threshold
    for (0..n) |i| {
        if (result[i] < threshold) {
            result[i] = -math.inf(f32);
        }
    }
}

/// Top-p (nucleus) sampling: keep smallest set of tokens with cumulative prob >= p
pub fn sample_top_p(
    logits: [*]const f32,
    n: usize,
    p: f32,
    result: [*]f32,
) void {
    if (n == 0) return;
    if (p >= 1.0) {
        for (0..n) |i| {
            result[i] = logits[i];
        }
        return;
    }

    // First compute softmax probabilities
    softmax_inplace_internal(logits, n, result);

    // Sort indices by probability (descending)
    // Simple O(n^2) sort for correctness
    var sorted_indices: [8192]usize = undefined;
    const max_n = @min(n, 8192);
    for (0..max_n) |i| {
        sorted_indices[i] = i;
    }

    // Bubble sort by probability (descending)
    for (0..max_n) |i| {
        for (i + 1..max_n) |j| {
            if (result[sorted_indices[j]] > result[sorted_indices[i]]) {
                const tmp = sorted_indices[i];
                sorted_indices[i] = sorted_indices[j];
                sorted_indices[j] = tmp;
            }
        }
    }

    // Find cutoff point
    var cumsum: f32 = 0.0;
    var cutoff_idx: usize = max_n;
    for (0..max_n) |i| {
        cumsum += result[sorted_indices[i]];
        if (cumsum >= p) {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Zero out tokens below cutoff, restore logits for kept tokens
    for (0..max_n) |i| {
        var keep = false;
        for (0..cutoff_idx) |j| {
            if (sorted_indices[j] == i) {
                keep = true;
                break;
            }
        }
        if (keep) {
            result[i] = logits[i];
        } else {
            result[i] = -math.inf(f32);
        }
    }

    // Handle remaining indices beyond max_n
    for (max_n..n) |i| {
        result[i] = -math.inf(f32);
    }
}

/// Sample an index from a probability distribution using a random value in [0, 1)
pub fn sample_from_probs(probs: [*]const f32, n: usize, random_val: f32) usize {
    if (n == 0) return 0;

    var cumsum: f32 = 0.0;
    for (0..n) |i| {
        cumsum += probs[i];
        if (random_val < cumsum) {
            return i;
        }
    }

    // Fallback to last index
    return n - 1;
}

// =============================================================================
// Penalty Functions
// =============================================================================

/// Apply repetition penalty: penalize tokens that have appeared before
/// penalty > 1.0 reduces probability of repeated tokens
pub fn apply_repetition_penalty(
    logits: [*]f32,
    n: usize,
    token_counts: [*]const u32,
    penalty: f32,
) void {
    if (n == 0 or penalty == 1.0) return;

    for (0..n) |i| {
        if (token_counts[i] > 0) {
            if (logits[i] > 0) {
                logits[i] /= penalty;
            } else {
                logits[i] *= penalty;
            }
        }
    }
}

/// Apply frequency penalty: penalize based on how often a token appeared
/// Subtracts penalty * count from logit
pub fn apply_frequency_penalty(
    logits: [*]f32,
    n: usize,
    token_counts: [*]const u32,
    penalty: f32,
) void {
    if (n == 0 or penalty == 0.0) return;

    for (0..n) |i| {
        const count: f32 = @floatFromInt(token_counts[i]);
        logits[i] -= penalty * count;
    }
}

/// Apply presence penalty: penalize if token has appeared at all
/// Subtracts penalty from logit if token count > 0
pub fn apply_presence_penalty(
    logits: [*]f32,
    n: usize,
    token_counts: [*]const u32,
    penalty: f32,
) void {
    if (n == 0 or penalty == 0.0) return;

    for (0..n) |i| {
        if (token_counts[i] > 0) {
            logits[i] -= penalty;
        }
    }
}

// =============================================================================
// Utility Functions
// =============================================================================

/// Internal softmax helper that writes to result array
fn softmax_inplace_internal(logits: [*]const f32, n: usize, result: [*]f32) void {
    if (n == 0) return;

    // Find max for numerical stability
    var max_val = logits[0];
    for (1..n) |i| {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (0..n) |i| {
        result[i] = @exp(logits[i] - max_val);
        sum += result[i];
    }

    // Normalize
    if (sum > 0) {
        for (0..n) |i| {
            result[i] /= sum;
        }
    }
}

/// Softmax: convert logits to probabilities in-place
pub fn softmax_inplace(logits: [*]f32, n: usize) void {
    if (n == 0) return;

    // Find max for numerical stability
    var max_val = logits[0];
    for (1..n) |i| {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute exp(x - max) and sum
    var sum: f32 = 0.0;
    for (0..n) |i| {
        logits[i] = @exp(logits[i] - max_val);
        sum += logits[i];
    }

    // Normalize
    if (sum > 0) {
        for (0..n) |i| {
            logits[i] /= sum;
        }
    }
}

/// Log-softmax: compute log(softmax(x)) with numerical stability
/// log_softmax(x_i) = x_i - max - log(sum(exp(x_j - max)))
pub fn log_softmax(logits: [*]const f32, n: usize, result: [*]f32) void {
    if (n == 0) return;

    // Find max for numerical stability
    var max_val = logits[0];
    for (1..n) |i| {
        if (logits[i] > max_val) {
            max_val = logits[i];
        }
    }

    // Compute log(sum(exp(x - max)))
    var sum: f32 = 0.0;
    for (0..n) |i| {
        sum += @exp(logits[i] - max_val);
    }
    const log_sum = @log(sum);

    // Compute log-softmax
    for (0..n) |i| {
        result[i] = logits[i] - max_val - log_sum;
    }
}

/// Temperature scaling: divide logits by temperature
pub fn temperature_scale(logits: [*]const f32, n: usize, temperature: f32, result: [*]f32) void {
    sample_temperature(logits, n, temperature, result);
}

// =============================================================================
// C-Compatible Exports
// =============================================================================

/// C-compatible: Greedy sampling
pub export fn n_sample_greedy(logits: ?[*]const f32, n: usize) usize {
    if (logits == null) return 0;
    return sample_greedy(logits.?, n);
}

/// C-compatible: Temperature sampling
pub export fn n_sample_temperature(
    logits: ?[*]const f32,
    n: usize,
    temperature: f32,
    result: ?[*]f32,
) void {
    if (logits == null or result == null) return;
    sample_temperature(logits.?, n, temperature, result.?);
}

/// C-compatible: Top-k sampling
pub export fn n_sample_top_k(
    logits: ?[*]const f32,
    n: usize,
    k: usize,
    result: ?[*]f32,
) void {
    if (logits == null or result == null) return;
    sample_top_k(logits.?, n, k, result.?);
}

/// C-compatible: Top-p (nucleus) sampling
pub export fn n_sample_top_p(
    logits: ?[*]const f32,
    n: usize,
    p: f32,
    result: ?[*]f32,
) void {
    if (logits == null or result == null) return;
    sample_top_p(logits.?, n, p, result.?);
}

/// C-compatible: Sample from probability distribution
pub export fn n_sample_from_probs(probs: ?[*]const f32, n: usize, random_val: f32) usize {
    if (probs == null) return 0;
    return sample_from_probs(probs.?, n, random_val);
}

/// C-compatible: Apply repetition penalty
pub export fn n_apply_repetition_penalty(
    logits: ?[*]f32,
    n: usize,
    token_counts: ?[*]const u32,
    penalty: f32,
) void {
    if (logits == null or token_counts == null) return;
    apply_repetition_penalty(logits.?, n, token_counts.?, penalty);
}

/// C-compatible: Apply frequency penalty
pub export fn n_apply_frequency_penalty(
    logits: ?[*]f32,
    n: usize,
    token_counts: ?[*]const u32,
    penalty: f32,
) void {
    if (logits == null or token_counts == null) return;
    apply_frequency_penalty(logits.?, n, token_counts.?, penalty);
}

/// C-compatible: Apply presence penalty
pub export fn n_apply_presence_penalty(
    logits: ?[*]f32,
    n: usize,
    token_counts: ?[*]const u32,
    penalty: f32,
) void {
    if (logits == null or token_counts == null) return;
    apply_presence_penalty(logits.?, n, token_counts.?, penalty);
}

/// C-compatible: Softmax in-place
pub export fn n_softmax_inplace(logits: ?[*]f32, n: usize) void {
    if (logits == null) return;
    softmax_inplace(logits.?, n);
}

/// C-compatible: Log-softmax
pub export fn n_log_softmax(logits: ?[*]const f32, n: usize, result: ?[*]f32) void {
    if (logits == null or result == null) return;
    log_softmax(logits.?, n, result.?);
}

/// C-compatible: Temperature scale
pub export fn n_temperature_scale(
    logits: ?[*]const f32,
    n: usize,
    temperature: f32,
    result: ?[*]f32,
) void {
    if (logits == null or result == null) return;
    temperature_scale(logits.?, n, temperature, result.?);
}

// =============================================================================
// Tests
// =============================================================================

test "sample_greedy finds max" {
    const logits = [_]f32{ 1.0, 5.0, 3.0, 2.0 };
    const result = sample_greedy(&logits, 4);
    try std.testing.expectEqual(@as(usize, 1), result);
}

test "sample_greedy first element" {
    const logits = [_]f32{ 10.0, 5.0, 3.0, 2.0 };
    const result = sample_greedy(&logits, 4);
    try std.testing.expectEqual(@as(usize, 0), result);
}

test "sample_greedy last element" {
    const logits = [_]f32{ 1.0, 2.0, 3.0, 100.0 };
    const result = sample_greedy(&logits, 4);
    try std.testing.expectEqual(@as(usize, 3), result);
}

test "sample_temperature scales correctly" {
    const logits = [_]f32{ 2.0, 4.0, 6.0, 8.0 };
    var result: [4]f32 = undefined;
    sample_temperature(&logits, 4, 2.0, &result);

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[3], 1e-6);
}

test "softmax_inplace produces valid probabilities" {
    var logits = [_]f32{ 1.0, 2.0, 3.0 };
    softmax_inplace(&logits, 3);

    // Sum should be 1.0
    const sum = logits[0] + logits[1] + logits[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);

    // All probabilities should be positive
    try std.testing.expect(logits[0] > 0);
    try std.testing.expect(logits[1] > 0);
    try std.testing.expect(logits[2] > 0);

    // Higher logit should have higher probability
    try std.testing.expect(logits[2] > logits[1]);
    try std.testing.expect(logits[1] > logits[0]);
}

test "sample_from_probs basic" {
    const probs = [_]f32{ 0.1, 0.2, 0.3, 0.4 };

    // Random value 0.05 should select index 0
    try std.testing.expectEqual(@as(usize, 0), sample_from_probs(&probs, 4, 0.05));

    // Random value 0.15 should select index 1 (cumsum 0.1-0.3)
    try std.testing.expectEqual(@as(usize, 1), sample_from_probs(&probs, 4, 0.15));

    // Random value 0.5 should select index 2 (cumsum 0.3-0.6)
    try std.testing.expectEqual(@as(usize, 2), sample_from_probs(&probs, 4, 0.5));

    // Random value 0.95 should select index 3
    try std.testing.expectEqual(@as(usize, 3), sample_from_probs(&probs, 4, 0.95));
}

test "apply_repetition_penalty" {
    var logits = [_]f32{ 2.0, -1.0, 3.0, 0.5 };
    const token_counts = [_]u32{ 1, 0, 2, 1 };

    apply_repetition_penalty(&logits, 4, &token_counts, 1.5);

    // Positive logits with count > 0 should be divided by penalty
    try std.testing.expectApproxEqAbs(@as(f32, 2.0 / 1.5), logits[0], 1e-6);

    // Token with count 0 should be unchanged
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), logits[1], 1e-6);

    // Positive logit with count > 0
    try std.testing.expectApproxEqAbs(@as(f32, 3.0 / 1.5), logits[2], 1e-6);
}

test "apply_frequency_penalty" {
    var logits = [_]f32{ 5.0, 3.0, 1.0 };
    const token_counts = [_]u32{ 0, 1, 3 };

    apply_frequency_penalty(&logits, 3, &token_counts, 0.5);

    // 5.0 - 0.5 * 0 = 5.0
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[0], 1e-6);
    // 3.0 - 0.5 * 1 = 2.5
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), logits[1], 1e-6);
    // 1.0 - 0.5 * 3 = -0.5
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), logits[2], 1e-6);
}

test "apply_presence_penalty" {
    var logits = [_]f32{ 5.0, 3.0, 1.0 };
    const token_counts = [_]u32{ 0, 1, 5 };

    apply_presence_penalty(&logits, 3, &token_counts, 1.0);

    // No count, unchanged
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), logits[0], 1e-6);
    // Count > 0, subtract penalty
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), logits[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), logits[2], 1e-6);
}

test "log_softmax numerical stability" {
    const logits = [_]f32{ 1000.0, 1001.0, 1002.0 };
    var result: [3]f32 = undefined;
    log_softmax(&logits, 3, &result);

    // Should not overflow or produce NaN
    try std.testing.expect(!math.isNan(result[0]));
    try std.testing.expect(!math.isNan(result[1]));
    try std.testing.expect(!math.isNan(result[2]));

    // log-softmax values should be negative (log of probability)
    try std.testing.expect(result[0] < 0);
    try std.testing.expect(result[1] < 0);
    try std.testing.expect(result[2] < 0);

    // exp(log_softmax) should sum to 1
    const sum = @exp(result[0]) + @exp(result[1]) + @exp(result[2]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), sum, 1e-5);
}

test "sample_top_k basic" {
    const logits = [_]f32{ 1.0, 5.0, 3.0, 2.0, 4.0 };
    var result: [5]f32 = undefined;
    sample_top_k(&logits, 5, 2, &result);

    // Top 2 are indices 1 (5.0) and 4 (4.0)
    // Others should be -inf
    try std.testing.expect(result[1] > -math.inf(f32));
    try std.testing.expect(result[4] > -math.inf(f32));
}

