// Monte Carlo methods and time-series utilities
// Uses RNG from stdlib/random.zig for all random operations

const std = @import("std");
const math = std.math;
const random = @import("../stdlib/random.zig");

// =============================================================================
// Monte Carlo Methods
// =============================================================================

/// 1D Monte Carlo integration: estimates ∫[a,b] f(x) dx using n_samples random points
/// Returns (b-a) * average(f(x)) for uniformly sampled x in [a, b)
pub fn monte_carlo_integrate(
    f: *const fn (f64) f64,
    a: f64,
    b: f64,
    n_samples: usize,
) f64 {
    if (n_samples == 0) return 0.0;
    if (a >= b) return 0.0;

    const width = b - a;
    var sum: f64 = 0.0;

    for (0..n_samples) |_| {
        const x = a + width * random.rand_f64();
        sum += f(x);
    }

    return width * sum / @as(f64, @floatFromInt(n_samples));
}

/// Estimate π using Monte Carlo method: sample random points in unit square
/// and count fraction falling inside quarter circle of radius 1
pub fn monte_carlo_pi(n_samples: usize) f64 {
    if (n_samples == 0) return 0.0;

    var inside: usize = 0;

    for (0..n_samples) |_| {
        const x = random.rand_f64();
        const y = random.rand_f64();
        if (x * x + y * y <= 1.0) {
            inside += 1;
        }
    }

    return 4.0 * @as(f64, @floatFromInt(inside)) / @as(f64, @floatFromInt(n_samples));
}

/// Importance sampling helper: estimates E[f(x)] using proposal distribution
/// f: target function, proposal_pdf: PDF of proposal distribution
/// proposal_sample: function that returns a sample from proposal distribution
/// Result: sum(f(x) / proposal_pdf(x)) / n_samples where x ~ proposal
pub fn importance_sampling(
    f: *const fn (f64) f64,
    proposal_pdf: *const fn (f64) f64,
    proposal_sample: *const fn () f64,
    n_samples: usize,
) f64 {
    if (n_samples == 0) return 0.0;

    var sum: f64 = 0.0;

    for (0..n_samples) |_| {
        const x = proposal_sample();
        const pdf_val = proposal_pdf(x);
        if (pdf_val > 0.0) {
            sum += f(x) / pdf_val;
        }
    }

    return sum / @as(f64, @floatFromInt(n_samples));
}

/// Bootstrap estimate of mean with confidence interval
/// Returns: BootstrapResult with mean, lower and upper confidence bounds
pub const BootstrapResult = struct {
    mean: f64,
    lower: f64,
    upper: f64,
};

pub fn bootstrap_mean(data: [*]const f64, n: usize, n_bootstrap: usize) BootstrapResult {
    if (n == 0 or n_bootstrap == 0) {
        return BootstrapResult{ .mean = 0.0, .lower = 0.0, .upper = 0.0 };
    }

    // Calculate original mean
    var original_mean: f64 = 0.0;
    for (0..n) |i| {
        original_mean += data[i];
    }
    original_mean /= @as(f64, @floatFromInt(n));

    // Bootstrap resampling - store means in a fixed buffer
    // For simplicity, compute running statistics for percentiles
    var sum_means: f64 = 0.0;
    var min_mean: f64 = math.inf(f64);
    var max_mean: f64 = -math.inf(f64);
    var sum_sq: f64 = 0.0;

    for (0..n_bootstrap) |_| {
        var bootstrap_sum: f64 = 0.0;
        // Resample with replacement
        for (0..n) |_| {
            const idx = random.rand_u64() % n;
            bootstrap_sum += data[idx];
        }
        const bootstrap_mean_val = bootstrap_sum / @as(f64, @floatFromInt(n));
        sum_means += bootstrap_mean_val;
        sum_sq += bootstrap_mean_val * bootstrap_mean_val;
        if (bootstrap_mean_val < min_mean) min_mean = bootstrap_mean_val;
        if (bootstrap_mean_val > max_mean) max_mean = bootstrap_mean_val;
    }

    // Use approximate 95% CI: mean ± 1.96 * std_error
    const mean_of_means = sum_means / @as(f64, @floatFromInt(n_bootstrap));
    const variance = (sum_sq / @as(f64, @floatFromInt(n_bootstrap))) - mean_of_means * mean_of_means;
    const std_error = math.sqrt(@max(0.0, variance));
    const z = 1.96; // 95% confidence

    return BootstrapResult{
        .mean = original_mean,
        .lower = mean_of_means - z * std_error,
        .upper = mean_of_means + z * std_error,
    };
}

// =============================================================================
// Time Series Functions
// =============================================================================

/// Simple moving average: computes SMA with given window size
/// result must have space for n elements; first (window-1) elements set to NaN
pub fn moving_average(data: [*]const f64, n: usize, window: usize, result: [*]f64) void {
    if (n == 0 or window == 0) return;

    // First (window-1) elements are undefined
    for (0..@min(window - 1, n)) |i| {
        result[i] = math.nan(f64);
    }

    if (window > n) return;

    // Compute initial sum for first window
    var sum: f64 = 0.0;
    for (0..window) |i| {
        sum += data[i];
    }
    result[window - 1] = sum / @as(f64, @floatFromInt(window));

    // Sliding window
    for (window..n) |i| {
        sum += data[i] - data[i - window];
        result[i] = sum / @as(f64, @floatFromInt(window));
    }
}

/// Exponential moving average with smoothing factor alpha in (0, 1]
/// EMA_t = alpha * x_t + (1 - alpha) * EMA_{t-1}
/// result[0] = data[0], then apply EMA recursively
pub fn exponential_moving_average(data: [*]const f64, n: usize, alpha: f64, result: [*]f64) void {
    if (n == 0) return;

    // Clamp alpha to valid range
    const a = @max(0.0, @min(1.0, alpha));
    const one_minus_a = 1.0 - a;

    result[0] = data[0];
    for (1..n) |i| {
        result[i] = a * data[i] + one_minus_a * result[i - 1];
    }
}

/// Autocorrelation at given lag: corr(x_t, x_{t+lag})
/// Returns Pearson correlation between x[0..n-lag] and x[lag..n]
pub fn autocorrelation(data: [*]const f64, n: usize, lag: usize) f64 {
    if (n == 0 or lag >= n) return 0.0;

    const effective_n = n - lag;
    if (effective_n < 2) return 0.0;

    // Compute means
    var mean_x: f64 = 0.0;
    var mean_y: f64 = 0.0;
    for (0..effective_n) |i| {
        mean_x += data[i];
        mean_y += data[i + lag];
    }
    mean_x /= @as(f64, @floatFromInt(effective_n));
    mean_y /= @as(f64, @floatFromInt(effective_n));

    // Compute covariance and variances
    var cov: f64 = 0.0;
    var var_x: f64 = 0.0;
    var var_y: f64 = 0.0;

    for (0..effective_n) |i| {
        const dx = data[i] - mean_x;
        const dy = data[i + lag] - mean_y;
        cov += dx * dy;
        var_x += dx * dx;
        var_y += dy * dy;
    }

    const denom = math.sqrt(var_x * var_y);
    if (denom == 0.0) return 0.0;

    return cov / denom;
}

/// First difference: result[i] = data[i+1] - data[i]
/// result must have space for (n-1) elements
pub fn diff(data: [*]const f64, n: usize, result: [*]f64) void {
    if (n < 2) return;

    for (0..n - 1) |i| {
        result[i] = data[i + 1] - data[i];
    }
}

/// Cumulative sum: result[i] = sum(data[0..i+1])
/// result must have space for n elements
pub fn cumsum(data: [*]const f64, n: usize, result: [*]f64) void {
    if (n == 0) return;

    result[0] = data[0];
    for (1..n) |i| {
        result[i] = result[i - 1] + data[i];
    }
}

// =============================================================================
// Simulation Helpers
// =============================================================================

/// 1D random walk: starting at 0, each step adds ±step_size with equal probability
/// result must have space for n_steps elements; result[0] = first step
pub fn random_walk(n_steps: usize, step_size: f64, result: [*]f64) void {
    if (n_steps == 0) return;

    var position: f64 = 0.0;
    for (0..n_steps) |i| {
        // Random direction: +1 or -1 with equal probability
        const direction: f64 = if (random.rand_u64() & 1 == 0) 1.0 else -1.0;
        position += direction * step_size;
        result[i] = position;
    }
}

/// Brownian motion (Wiener process): W_{t+dt} = W_t + N(0, sigma * sqrt(dt))
/// result must have space for n_steps elements; result[0] = first increment
pub fn brownian_motion(n_steps: usize, dt: f64, sigma: f64, result: [*]f64) void {
    if (n_steps == 0) return;

    const scale = sigma * math.sqrt(dt);
    var position: f64 = 0.0;

    for (0..n_steps) |i| {
        position += random.normal() * scale;
        result[i] = position;
    }
}

/// Geometric Brownian Motion for stock price simulation
/// S_{t+dt} = S_t * exp((mu - sigma^2/2)*dt + sigma*sqrt(dt)*Z)
/// where Z ~ N(0,1). Models log-normal stock price dynamics.
/// result must have space for n_steps elements; result[0] = s0
pub fn geometric_brownian_motion(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    n_steps: usize,
    result: [*]f64,
) void {
    if (n_steps == 0) return;

    const drift = (mu - 0.5 * sigma * sigma) * dt;
    const vol = sigma * math.sqrt(dt);

    var price = s0;
    result[0] = price;

    for (1..n_steps) |i| {
        const z = random.normal();
        price *= math.exp(drift + vol * z);
        result[i] = price;
    }
}

// =============================================================================
// C-Compatible Exports
// =============================================================================

/// C-compatible: Monte Carlo integration
pub export fn n_monte_carlo_integrate(
    f: *const fn (f64) callconv(.C) f64,
    a: f64,
    b: f64,
    n_samples: usize,
) f64 {
    // Wrapper to handle C calling convention
    const wrapper = struct {
        var c_func: *const fn (f64) callconv(.C) f64 = undefined;
        fn call(x: f64) f64 {
            return c_func(x);
        }
    };
    wrapper.c_func = f;
    return monte_carlo_integrate(&wrapper.call, a, b, n_samples);
}

/// C-compatible: Estimate π using Monte Carlo
pub export fn n_monte_carlo_pi(n_samples: usize) f64 {
    return monte_carlo_pi(n_samples);
}

/// C-compatible: Bootstrap mean estimation
pub export fn n_bootstrap_mean(
    data: ?[*]const f64,
    n: usize,
    n_bootstrap: usize,
    out_mean: ?*f64,
    out_lower: ?*f64,
    out_upper: ?*f64,
) void {
    if (data == null) return;
    const result = bootstrap_mean(data.?, n, n_bootstrap);
    if (out_mean) |p| p.* = result.mean;
    if (out_lower) |p| p.* = result.lower;
    if (out_upper) |p| p.* = result.upper;
}

/// C-compatible: Simple moving average
pub export fn n_moving_average(
    data: ?[*]const f64,
    n: usize,
    window: usize,
    result: ?[*]f64,
) void {
    if (data == null or result == null) return;
    moving_average(data.?, n, window, result.?);
}

/// C-compatible: Exponential moving average
pub export fn n_exponential_moving_average(
    data: ?[*]const f64,
    n: usize,
    alpha: f64,
    result: ?[*]f64,
) void {
    if (data == null or result == null) return;
    exponential_moving_average(data.?, n, alpha, result.?);
}

/// C-compatible: Autocorrelation at given lag
pub export fn n_autocorrelation(data: ?[*]const f64, n: usize, lag: usize) f64 {
    if (data == null) return 0.0;
    return autocorrelation(data.?, n, lag);
}

/// C-compatible: First difference
pub export fn n_diff(data: ?[*]const f64, n: usize, result: ?[*]f64) void {
    if (data == null or result == null) return;
    diff(data.?, n, result.?);
}

/// C-compatible: Cumulative sum
pub export fn n_cumsum(data: ?[*]const f64, n: usize, result: ?[*]f64) void {
    if (data == null or result == null) return;
    cumsum(data.?, n, result.?);
}

/// C-compatible: 1D random walk
pub export fn n_random_walk(n_steps: usize, step_size: f64, result: ?[*]f64) void {
    if (result == null) return;
    random_walk(n_steps, step_size, result.?);
}

/// C-compatible: Brownian motion
pub export fn n_brownian_motion(n_steps: usize, dt: f64, sigma: f64, result: ?[*]f64) void {
    if (result == null) return;
    brownian_motion(n_steps, dt, sigma, result.?);
}

/// C-compatible: Geometric Brownian motion
pub export fn n_geometric_brownian_motion(
    s0: f64,
    mu: f64,
    sigma: f64,
    dt: f64,
    n_steps: usize,
    result: ?[*]f64,
) void {
    if (result == null) return;
    geometric_brownian_motion(s0, mu, sigma, dt, n_steps, result.?);
}

// =============================================================================
// Tests
// =============================================================================

test "monte_carlo_pi convergence" {
    // With many samples, should be close to π
    const estimate = monte_carlo_pi(10000);
    try std.testing.expect(@abs(estimate - math.pi) < 0.1);
}

test "moving_average basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var result: [5]f64 = undefined;
    moving_average(&data, 5, 3, &result);

    // First 2 should be NaN
    try std.testing.expect(math.isNan(result[0]));
    try std.testing.expect(math.isNan(result[1]));
    // SMA of [1,2,3] = 2, [2,3,4] = 3, [3,4,5] = 4
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[4], 1e-10);
}

test "diff basic" {
    const data = [_]f64{ 1.0, 3.0, 6.0, 10.0 };
    var result: [3]f64 = undefined;
    diff(&data, 4, &result);

    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[2], 1e-10);
}

test "cumsum basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    var result: [4]f64 = undefined;
    cumsum(&data, 4, &result);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), result[3], 1e-10);
}

test "exponential_moving_average basic" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    var result: [5]f64 = undefined;
    exponential_moving_average(&data, 5, 0.5, &result);

    // EMA with alpha=0.5: result[0] = 1.0
    // result[1] = 0.5*2 + 0.5*1 = 1.5
    // result[2] = 0.5*3 + 0.5*1.5 = 2.25
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), result[2], 1e-10);
}

test "autocorrelation at lag 0 is 1" {
    const data = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const result = autocorrelation(&data, 5, 0);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result, 1e-10);
}

