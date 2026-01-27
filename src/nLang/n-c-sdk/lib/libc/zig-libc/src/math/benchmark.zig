// Benchmarks and robustness tests for math library
// Pure Zig implementations - no external dependencies

const std = @import("std");
const math = std.math;
const random = @import("../stdlib/random.zig");
const blas = @import("blas.zig");
const optim = @import("optim.zig");
const lib = @import("lib.zig");

// =============================================================================
// Result Types
// =============================================================================

/// Result structure for benchmark functions
pub const BenchResult = extern struct {
    success: u8, // 1 if successful, 0 if failed
    elapsed_ns: u64, // Elapsed time in nanoseconds
    samples_per_sec: f64, // Throughput (samples/second)
    extra_value: f64, // Additional metric (e.g., quality score)
};

/// Result structure for statistical tests
pub const TestResult = extern struct {
    success: u8, // 1 if passed, 0 if failed
    statistic: f64, // Test statistic value
    threshold: f64, // Threshold for pass/fail
    elapsed_ns: u64, // Elapsed time in nanoseconds
};

// =============================================================================
// Timer Helper
// =============================================================================

fn getTimestamp() u64 {
    return @bitCast(std.time.nanoTimestamp());
}

// =============================================================================
// RNG Benchmarks
// =============================================================================

/// Benchmark xoshiro256** and PCG64 throughput
pub fn bench_rng_throughput(n_samples: usize) BenchResult {
    if (n_samples == 0) return .{ .success = 0, .elapsed_ns = 0, .samples_per_sec = 0, .extra_value = 0 };

    // Benchmark xoshiro256**
    const start_xo = getTimestamp();
    var sum_xo: u64 = 0;
    for (0..n_samples) |_| {
        sum_xo +%= random.rand_u64();
    }
    const end_xo = getTimestamp();
    const elapsed_xo = end_xo - start_xo;

    // Benchmark PCG64
    const start_pcg = getTimestamp();
    var sum_pcg: u64 = 0;
    for (0..n_samples) |_| {
        sum_pcg +%= random.pcg64_next();
    }
    const end_pcg = getTimestamp();
    const elapsed_pcg = end_pcg - start_pcg;

    // Use sums to prevent optimization (volatile store)
    const total_sum = sum_xo +% sum_pcg;
    _ = total_sum;

    // Average throughput
    const total_elapsed = elapsed_xo + elapsed_pcg;
    const total_samples: f64 = @floatFromInt(n_samples * 2);
    const samples_per_sec = if (total_elapsed > 0)
        total_samples * 1e9 / @as(f64, @floatFromInt(total_elapsed))
    else
        0;

    return .{
        .success = 1,
        .elapsed_ns = total_elapsed,
        .samples_per_sec = samples_per_sec,
        .extra_value = @as(f64, @floatFromInt(elapsed_xo)) / @as(f64, @floatFromInt(@max(elapsed_pcg, 1))),
    };
}

/// Basic RNG quality test - uniformity check
pub fn bench_rng_quality(n_samples: usize) BenchResult {
    if (n_samples == 0) return .{ .success = 0, .elapsed_ns = 0, .samples_per_sec = 0, .extra_value = 0 };

    const start = getTimestamp();

    // Count samples in 10 bins
    const n_bins = 10;
    var bins: [n_bins]u64 = [_]u64{0} ** n_bins;

    for (0..n_samples) |_| {
        const u = random.rand_f64();
        const bin_idx: usize = @intFromFloat(@min(u * @as(f64, n_bins), @as(f64, n_bins - 1)));
        bins[bin_idx] += 1;
    }

    // Calculate chi-squared statistic
    const expected: f64 = @as(f64, @floatFromInt(n_samples)) / @as(f64, n_bins);
    var chi_sq: f64 = 0;
    for (bins) |count| {
        const observed: f64 = @floatFromInt(count);
        const diff = observed - expected;
        chi_sq += (diff * diff) / expected;
    }

    const end = getTimestamp();
    const elapsed = end - start;

    // Chi-squared critical value for df=9, alpha=0.05 is 16.919
    const critical = 16.919;
    const passed = chi_sq < critical;

    return .{
        .success = if (passed) 1 else 0,
        .elapsed_ns = elapsed,
        .samples_per_sec = if (elapsed > 0) @as(f64, @floatFromInt(n_samples)) * 1e9 / @as(f64, @floatFromInt(elapsed)) else 0,
        .extra_value = chi_sq,
    };
}

// =============================================================================
// Distribution Accuracy Tests
// =============================================================================

/// Kolmogorov-Smirnov test for uniform distribution
pub fn test_uniform_ks(n_samples: usize) TestResult {
    if (n_samples == 0) return .{ .success = 0, .statistic = 0, .threshold = 0, .elapsed_ns = 0 };
    if (n_samples > 10000) return .{ .success = 0, .statistic = 0, .threshold = 0, .elapsed_ns = 0 };

    const start = getTimestamp();

    // Generate samples and sort using simple bubble sort (for small n)
    var samples: [10000]f64 = undefined;
    const n = @min(n_samples, 10000);
    for (0..n) |i| {
        samples[i] = random.rand_f64();
    }

    // Simple insertion sort
    for (1..n) |i| {
        const key = samples[i];
        var j: usize = i;
        while (j > 0 and samples[j - 1] > key) : (j -= 1) {
            samples[j] = samples[j - 1];
        }
        samples[j] = key;
    }

    // Calculate KS statistic
    var d_max: f64 = 0;
    const nf: f64 = @floatFromInt(n);
    for (0..n) |i| {
        const cdf = samples[i]; // Uniform CDF = x for x in [0,1]
        const ecdf = @as(f64, @floatFromInt(i + 1)) / nf;
        const ecdf_prev = @as(f64, @floatFromInt(i)) / nf;
        d_max = @max(d_max, @abs(ecdf - cdf));
        d_max = @max(d_max, @abs(ecdf_prev - cdf));
    }

    const end = getTimestamp();

    // Critical value for alpha=0.05: 1.36 / sqrt(n)
    const critical = 1.36 / math.sqrt(nf);
    const passed = d_max < critical;

    return .{ .success = if (passed) 1 else 0, .statistic = d_max, .threshold = critical, .elapsed_ns = end - start };
}

/// Kolmogorov-Smirnov test for normal distribution
pub fn test_normal_ks(n_samples: usize) TestResult {
    if (n_samples == 0 or n_samples > 10000) return .{ .success = 0, .statistic = 0, .threshold = 0, .elapsed_ns = 0 };

    const start = getTimestamp();

    var samples: [10000]f64 = undefined;
    const n = @min(n_samples, 10000);
    for (0..n) |i| {
        samples[i] = random.normal();
    }

    // Insertion sort
    for (1..n) |i| {
        const key = samples[i];
        var j: usize = i;
        while (j > 0 and samples[j - 1] > key) : (j -= 1) {
            samples[j] = samples[j - 1];
        }
        samples[j] = key;
    }

    // Calculate KS statistic against standard normal CDF
    var d_max: f64 = 0;
    const nf: f64 = @floatFromInt(n);
    for (0..n) |i| {
        const cdf = random.normal_cdf(samples[i], 0.0, 1.0);
        const ecdf = @as(f64, @floatFromInt(i + 1)) / nf;
        const ecdf_prev = @as(f64, @floatFromInt(i)) / nf;
        d_max = @max(d_max, @abs(ecdf - cdf));
        d_max = @max(d_max, @abs(ecdf_prev - cdf));
    }

    const end = getTimestamp();
    const critical = 1.36 / math.sqrt(nf);
    const passed = d_max < critical;

    return .{ .success = if (passed) 1 else 0, .statistic = d_max, .threshold = critical, .elapsed_ns = end - start };
}

/// Chi-squared goodness of fit test for uniform distribution
pub fn test_chi_squared_uniformity(n_samples: usize, n_bins: usize) TestResult {
    if (n_samples == 0 or n_bins == 0 or n_bins > 100) return .{ .success = 0, .statistic = 0, .threshold = 0, .elapsed_ns = 0 };

    const start = getTimestamp();

    var bins: [100]u64 = [_]u64{0} ** 100;
    const actual_bins = @min(n_bins, 100);

    for (0..n_samples) |_| {
        const u = random.rand_f64();
        const bin_idx: usize = @intFromFloat(@min(u * @as(f64, @floatFromInt(actual_bins)), @as(f64, actual_bins - 1)));
        bins[bin_idx] += 1;
    }

    const expected: f64 = @as(f64, @floatFromInt(n_samples)) / @as(f64, @floatFromInt(actual_bins));
    var chi_sq: f64 = 0;
    for (0..actual_bins) |i| {
        const observed: f64 = @floatFromInt(bins[i]);
        const diff = observed - expected;
        chi_sq += (diff * diff) / expected;
    }

    const end = getTimestamp();

    // Approximate critical value: use chi-squared 0.05 quantile
    // For df = n_bins - 1, critical ≈ df + 1.645 * sqrt(2*df)
    const df: f64 = @floatFromInt(actual_bins - 1);
    const critical = df + 1.645 * math.sqrt(2.0 * df);
    const passed = chi_sq < critical;

    return .{ .success = if (passed) 1 else 0, .statistic = chi_sq, .threshold = critical, .elapsed_ns = end - start };
}

// =============================================================================
// BLAS Microbenchmarks
// =============================================================================

/// Benchmark BLAS daxpy operation
pub fn bench_daxpy(n: usize, iterations: usize) BenchResult {
    if (n == 0 or iterations == 0 or n > 4096) return .{ .success = 0, .elapsed_ns = 0, .samples_per_sec = 0, .extra_value = 0 };

    var x_buf: [4096]f64 = undefined;
    var y_buf: [4096]f64 = undefined;

    // Initialize vectors
    for (0..n) |i| {
        x_buf[i] = @floatFromInt(i);
        y_buf[i] = @floatFromInt(i * 2);
    }

    const start = getTimestamp();
    for (0..iterations) |_| {
        blas.blas_daxpy(@intCast(n), 2.0, &x_buf, 1, &y_buf, 1);
    }
    const end = getTimestamp();
    const elapsed = end - start;

    const ops_per_iter: f64 = @floatFromInt(n * 2); // 1 mul + 1 add per element
    const total_ops = ops_per_iter * @as(f64, @floatFromInt(iterations));
    const flops = if (elapsed > 0) total_ops * 1e9 / @as(f64, @floatFromInt(elapsed)) else 0;

    return .{ .success = 1, .elapsed_ns = elapsed, .samples_per_sec = flops, .extra_value = y_buf[0] };
}

/// Benchmark matrix multiply (n×n)
pub fn bench_dgemm(n: usize, iterations: usize) BenchResult {
    if (n == 0 or iterations == 0 or n > 64) return .{ .success = 0, .elapsed_ns = 0, .samples_per_sec = 0, .extra_value = 0 };

    var a_buf: [4096]f64 = undefined; // 64x64 max
    var b_buf: [4096]f64 = undefined;
    var c_buf: [4096]f64 = undefined;

    // Initialize matrices
    for (0..n * n) |i| {
        a_buf[i] = 1.0;
        b_buf[i] = 1.0;
        c_buf[i] = 0.0;
    }

    const start = getTimestamp();
    for (0..iterations) |_| {
        blas.blas_dgemm(blas.BLAS_NO_TRANS, blas.BLAS_NO_TRANS, @intCast(n), @intCast(n), @intCast(n), 1.0, &a_buf, @intCast(n), &b_buf, @intCast(n), 0.0, &c_buf, @intCast(n));
    }
    const end = getTimestamp();
    const elapsed = end - start;

    // 2*n^3 flops for matrix multiply
    const ops_per_iter: f64 = 2.0 * @as(f64, @floatFromInt(n * n * n));
    const total_ops = ops_per_iter * @as(f64, @floatFromInt(iterations));
    const flops = if (elapsed > 0) total_ops * 1e9 / @as(f64, @floatFromInt(elapsed)) else 0;

    return .{ .success = 1, .elapsed_ns = elapsed, .samples_per_sec = flops, .extra_value = c_buf[0] };
}

/// Benchmark dot product
pub fn bench_dot_product(n: usize, iterations: usize) BenchResult {
    if (n == 0 or iterations == 0 or n > 4096) return .{ .success = 0, .elapsed_ns = 0, .samples_per_sec = 0, .extra_value = 0 };

    var x_buf: [4096]f64 = undefined;
    var y_buf: [4096]f64 = undefined;

    for (0..n) |i| {
        x_buf[i] = @floatFromInt(i);
        y_buf[i] = @floatFromInt(i);
    }

    const start = getTimestamp();
    var result: f64 = 0;
    for (0..iterations) |_| {
        result = blas.blas_ddot(@intCast(n), &x_buf, 1, &y_buf, 1);
    }
    const end = getTimestamp();
    const elapsed = end - start;

    const ops_per_iter: f64 = @floatFromInt(n * 2); // 1 mul + 1 add per element
    const total_ops = ops_per_iter * @as(f64, @floatFromInt(iterations));
    const flops = if (elapsed > 0) total_ops * 1e9 / @as(f64, @floatFromInt(elapsed)) else 0;

    return .{ .success = 1, .elapsed_ns = elapsed, .samples_per_sec = flops, .extra_value = result };
}


// =============================================================================
// Optimizer Convergence Tests
// =============================================================================

/// Test gradient descent on x² (should converge to 0)
pub fn test_gd_quadratic() TestResult {
    const start = getTimestamp();

    // Define f(x) = x² and f'(x) = 2x as C-compatible functions
    const funcs = struct {
        fn f(x: f64) callconv(.C) f64 {
            return x * x;
        }
        fn df(x: f64) callconv(.C) f64 {
            return 2.0 * x;
        }
    };

    const result = optim.gradient_descent_1d(&funcs.f, &funcs.df, 5.0, 0.1, 1e-8, 1000);

    const end = getTimestamp();

    const converged = result.converged == 1;
    const near_zero = @abs(result.value) < 1e-4;
    const passed = converged and near_zero;

    return .{
        .success = if (passed) 1 else 0,
        .statistic = result.value,
        .threshold = 1e-4,
        .elapsed_ns = end - start,
    };
}

/// Test Adam optimizer on Rosenbrock function
pub fn test_adam_rosenbrock() TestResult {
    const start = getTimestamp();

    // Rosenbrock gradient: ∇f = (∂f/∂x, ∂f/∂y)
    // f(x,y) = (1-x)² + 100(y-x²)²
    // ∂f/∂x = -2(1-x) - 400x(y-x²)
    // ∂f/∂y = 200(y-x²)
    const grad_fn = struct {
        fn gradient(grad: [*]f64, x: [*]const f64, n: usize) callconv(.C) void {
            if (n < 2) return;
            const x0 = x[0];
            const x1 = x[1];
            grad[0] = -2.0 * (1.0 - x0) - 400.0 * x0 * (x1 - x0 * x0);
            grad[1] = 200.0 * (x1 - x0 * x0);
        }
    };

    var x = [_]f64{ -1.0, 1.0 };
    _ = optim.adam(&grad_fn.gradient, &x, 2, 0.01, 0.9, 0.999, 1e-8, 5000);

    const end = getTimestamp();

    // Rosenbrock minimum is at (1, 1)
    const dist = math.sqrt((x[0] - 1.0) * (x[0] - 1.0) + (x[1] - 1.0) * (x[1] - 1.0));
    const passed = dist < 0.5; // Rosenbrock is hard, allow some tolerance

    return .{
        .success = if (passed) 1 else 0,
        .statistic = dist,
        .threshold = 0.5,
        .elapsed_ns = end - start,
    };
}

/// Test Newton's method on known roots
pub fn test_newton_roots() TestResult {
    const start = getTimestamp();

    // Test on f(x) = x² - 2 (root at sqrt(2))
    const funcs = struct {
        fn f(x: f64) callconv(.C) f64 {
            return x * x - 2.0;
        }
        fn df(x: f64) callconv(.C) f64 {
            return 2.0 * x;
        }
    };

    const result = optim.newton(&funcs.f, &funcs.df, 1.0, 1e-12, 100);

    const end = getTimestamp();

    const expected = math.sqrt(2.0);
    const error_val = @abs(result.value - expected);
    const passed = result.converged == 1 and error_val < 1e-10;

    return .{
        .success = if (passed) 1 else 0,
        .statistic = error_val,
        .threshold = 1e-10,
        .elapsed_ns = end - start,
    };
}

// =============================================================================
// Robustness Tests
// =============================================================================

/// Test gamma function with extreme inputs
pub fn test_overflow_gamma() TestResult {
    const start = getTimestamp();
    var all_passed = true;

    // Test large positive input (should return inf)
    const large = lib.tgamma(200.0);
    if (!math.isInf(large) or math.signbit(large)) all_passed = false;

    // Test small positive input
    const small = lib.tgamma(0.001);
    if (!math.isFinite(small) or small <= 0) all_passed = false;

    // Test negative non-integer (should be finite)
    const neg = lib.tgamma(-0.5);
    if (!math.isFinite(neg)) all_passed = false;

    // Test negative integer (should be NaN or inf - pole)
    const neg_int = lib.tgamma(-1.0);
    if (math.isFinite(neg_int)) all_passed = false; // Should be undefined at poles

    // Test zero (should be inf)
    const zero = lib.tgamma(0.0);
    if (!math.isInf(zero)) all_passed = false;

    const end = getTimestamp();

    return .{ .success = if (all_passed) 1 else 0, .statistic = if (all_passed) 1.0 else 0.0, .threshold = 1.0, .elapsed_ns = end - start };
}

/// Test exp with very negative inputs
pub fn test_underflow_exp() TestResult {
    const start = getTimestamp();
    var all_passed = true;

    // Test very negative input (should underflow to 0)
    const very_neg = lib.exp(-1000.0);
    if (very_neg != 0.0) all_passed = false;

    // Test moderately negative input (should be small but finite)
    const mod_neg = lib.exp(-100.0);
    if (!math.isFinite(mod_neg) or mod_neg < 0) all_passed = false;

    // Test large positive input (should overflow to inf)
    const large_pos = lib.exp(1000.0);
    if (!math.isInf(large_pos)) all_passed = false;

    // Test zero (should be 1)
    const zero_exp = lib.exp(0.0);
    if (@abs(zero_exp - 1.0) > 1e-15) all_passed = false;

    // Test negative zero
    const neg_zero_exp = lib.exp(-0.0);
    if (@abs(neg_zero_exp - 1.0) > 1e-15) all_passed = false;

    const end = getTimestamp();

    return .{ .success = if (all_passed) 1 else 0, .statistic = if (all_passed) 1.0 else 0.0, .threshold = 1.0, .elapsed_ns = end - start };
}

/// Verify RNG produces same sequence given same seed
pub fn test_determinism(seed: u64) TestResult {
    const start = getTimestamp();

    // Generate first sequence
    random.xoshiro_seed_u64(seed);
    var seq1: [100]u64 = undefined;
    for (0..100) |i| {
        seq1[i] = random.rand_u64();
    }

    // Reset and generate second sequence
    random.xoshiro_seed_u64(seed);
    var seq2: [100]u64 = undefined;
    for (0..100) |i| {
        seq2[i] = random.rand_u64();
    }

    // Compare sequences
    var all_match = true;
    for (0..100) |i| {
        if (seq1[i] != seq2[i]) {
            all_match = false;
            break;
        }
    }

    const end = getTimestamp();

    return .{ .success = if (all_match) 1 else 0, .statistic = if (all_match) 1.0 else 0.0, .threshold = 1.0, .elapsed_ns = end - start };
}

/// Test handling of NaN, Inf, -Inf, 0, -0
pub fn test_special_values() TestResult {
    const start = getTimestamp();
    var all_passed = true;

    const nan_val = math.nan(f64);
    const pos_inf = math.inf(f64);
    const neg_inf = -math.inf(f64);
    const zero: f64 = 0.0;
    const neg_zero: f64 = -0.0;

    // sin(NaN) = NaN
    if (!math.isNan(lib.sin(nan_val))) all_passed = false;

    // sin(inf) = NaN
    if (!math.isNan(lib.sin(pos_inf))) all_passed = false;

    // exp(inf) = inf
    if (!math.isInf(lib.exp(pos_inf))) all_passed = false;

    // exp(-inf) = 0
    if (lib.exp(neg_inf) != 0.0) all_passed = false;

    // log(0) = -inf
    if (!math.isNegativeInf(lib.log(zero))) all_passed = false;

    // sqrt(-0) = -0
    const sqrt_neg_zero = lib.sqrt(neg_zero);
    if (sqrt_neg_zero != 0.0 or !math.signbit(sqrt_neg_zero)) all_passed = false;

    // isnan correctly identifies NaN
    if (lib.isnan(nan_val) != 1) all_passed = false;
    if (lib.isnan(1.0) != 0) all_passed = false;

    // isinf correctly identifies infinity
    if (lib.isinf(pos_inf) != 1) all_passed = false;
    if (lib.isinf(neg_inf) != 1) all_passed = false;
    if (lib.isinf(1.0) != 0) all_passed = false;

    const end = getTimestamp();

    return .{ .success = if (all_passed) 1 else 0, .statistic = if (all_passed) 1.0 else 0.0, .threshold = 1.0, .elapsed_ns = end - start };
}

// =============================================================================
// C-Compatible Exports
// =============================================================================

pub export fn n_bench_rng_throughput(n_samples: usize) BenchResult {
    return bench_rng_throughput(n_samples);
}

pub export fn n_bench_rng_quality(n_samples: usize) BenchResult {
    return bench_rng_quality(n_samples);
}

pub export fn n_test_uniform_ks(n_samples: usize) TestResult {
    return test_uniform_ks(n_samples);
}

pub export fn n_test_normal_ks(n_samples: usize) TestResult {
    return test_normal_ks(n_samples);
}

pub export fn n_test_chi_squared_uniformity(n_samples: usize, n_bins: usize) TestResult {
    return test_chi_squared_uniformity(n_samples, n_bins);
}

pub export fn n_bench_daxpy(n: usize, iterations: usize) BenchResult {
    return bench_daxpy(n, iterations);
}

pub export fn n_bench_dgemm(n: usize, iterations: usize) BenchResult {
    return bench_dgemm(n, iterations);
}

pub export fn n_bench_dot_product(n: usize, iterations: usize) BenchResult {
    return bench_dot_product(n, iterations);
}

pub export fn n_test_gd_quadratic() TestResult {
    return test_gd_quadratic();
}

pub export fn n_test_adam_rosenbrock() TestResult {
    return test_adam_rosenbrock();
}

pub export fn n_test_newton_roots() TestResult {
    return test_newton_roots();
}

pub export fn n_test_overflow_gamma() TestResult {
    return test_overflow_gamma();
}

pub export fn n_test_underflow_exp() TestResult {
    return test_underflow_exp();
}

pub export fn n_test_determinism(seed: u64) TestResult {
    return test_determinism(seed);
}

pub export fn n_test_special_values() TestResult {
    return test_special_values();
}
