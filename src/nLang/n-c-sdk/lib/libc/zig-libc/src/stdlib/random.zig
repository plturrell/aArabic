// Random number generation
// Upgraded RNG with xoshiro256**, PCG64, distribution helpers, and jump functions
const std = @import("std");

// =============================================================================
// === Xoshiro256** - Primary high-quality RNG ===
// =============================================================================

threadlocal var xoshiro_state: [4]u64 = .{ 0x123456789abcdef0, 0xfedcba9876543210, 0x0f1e2d3c4b5a6978, 0x8877665544332211 };

fn rotl(x: u64, k: u6) u64 {
    return std.math.rotl(u64, x, k);
}

fn splitmix64(seed: u64) u64 {
    var z = seed +% 0x9E3779B97F4A7C15;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn splitmix64_state(state: *u64) u64 {
    state.* +%= 0x9E3779B97F4A7C15;
    var z = state.*;
    z = (z ^ (z >> 30)) *% 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) *% 0x94D049BB133111EB;
    return z ^ (z >> 31);
}

fn xoshiro_next() u64 {
    var s = &xoshiro_state;
    const result = rotl(s[1] *% 5, 7) *% 9;
    const t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

/// Jump constants for xoshiro256** - equivalent to 2^128 calls
const XOSHIRO_JUMP: [4]u64 = .{
    0x180ec6d33cfd0aba,
    0xd5a61266f0c9392c,
    0xa9582618e03fc9aa,
    0x39abdc4529b1661c,
};

/// Long jump constants for xoshiro256** - equivalent to 2^192 calls
const XOSHIRO_LONG_JUMP: [4]u64 = .{
    0x76e15d3efefdcbbf,
    0xc5004e441c522fb3,
    0x77710069854ee241,
    0x39109bb02acbe635,
};

/// Jump ahead 2^128 steps in the xoshiro256** sequence.
/// Useful for generating non-overlapping subsequences for parallel computation.
pub fn xoshiro_jump() void {
    xoshiro_jump_with_constants(&XOSHIRO_JUMP);
}

/// Jump ahead 2^192 steps in the xoshiro256** sequence.
/// Useful for generating widely separated subsequences.
pub fn xoshiro_long_jump() void {
    xoshiro_jump_with_constants(&XOSHIRO_LONG_JUMP);
}

fn xoshiro_jump_with_constants(jump_constants: *const [4]u64) void {
    var s0: u64 = 0;
    var s1: u64 = 0;
    var s2: u64 = 0;
    var s3: u64 = 0;

    for (jump_constants) |jc| {
        var b: u6 = 0;
        while (true) : (b += 1) {
            if ((jc & (@as(u64, 1) << b)) != 0) {
                s0 ^= xoshiro_state[0];
                s1 ^= xoshiro_state[1];
                s2 ^= xoshiro_state[2];
                s3 ^= xoshiro_state[3];
            }
            _ = xoshiro_next();
            if (b == 63) break;
        }
    }

    xoshiro_state[0] = s0;
    xoshiro_state[1] = s1;
    xoshiro_state[2] = s2;
    xoshiro_state[3] = s3;
}

/// C-compatible: Jump ahead 2^128 steps
pub export fn n_xoshiro_jump() void {
    xoshiro_jump();
}

/// C-compatible: Jump ahead 2^192 steps
pub export fn n_xoshiro_long_jump() void {
    xoshiro_long_jump();
}

/// Seed the RNG with a 32-bit value (expand via splitmix64).
pub export fn srandom(seed: c_uint) void {
    const x: u64 = seed;
    xoshiro_state[0] = splitmix64(x);
    xoshiro_state[1] = splitmix64(xoshiro_state[0]);
    xoshiro_state[2] = splitmix64(xoshiro_state[1]);
    xoshiro_state[3] = splitmix64(xoshiro_state[2]);
}

/// Seed rand() interface (alias to srandom)
pub export fn srand(seed: c_uint) void {
    srandom(seed);
}

/// Seed xoshiro256** with a 64-bit value
pub fn xoshiro_seed_u64(seed: u64) void {
    var sm_state = seed;
    xoshiro_state[0] = splitmix64_state(&sm_state);
    xoshiro_state[1] = splitmix64_state(&sm_state);
    xoshiro_state[2] = splitmix64_state(&sm_state);
    xoshiro_state[3] = splitmix64_state(&sm_state);
}

/// Seed xoshiro256** with an arbitrary byte array.
/// Uses splitmix64 mixing to expand the seed material.
pub fn xoshiro_seed_bytes(seed_bytes: []const u8) void {
    // Mix seed bytes into initial state using FNV-1a hash combined with splitmix64
    var hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for (seed_bytes) |b| {
        hash ^= b;
        hash *%= 0x100000001b3; // FNV prime
    }
    xoshiro_seed_u64(hash);
}

/// C-compatible: Seed xoshiro256** with bytes
pub export fn n_xoshiro_seed(seed_ptr: ?[*]const u8, seed_len: usize) void {
    if (seed_ptr) |ptr| {
        xoshiro_seed_bytes(ptr[0..seed_len]);
    } else {
        xoshiro_seed_u64(0);
    }
}

/// Get direct access to xoshiro state for advanced usage (e.g., serialization)
pub fn xoshiro_get_state() [4]u64 {
    return xoshiro_state;
}

/// Set xoshiro state directly (for deserialization or testing)
pub fn xoshiro_set_state(state: [4]u64) void {
    xoshiro_state = state;
}

// =============================================================================
// === PCG64 - Permuted Congruential Generator ===
// =============================================================================

/// PCG64 state: 128-bit state internally (two u64s for state and increment)
const Pcg64 = struct {
    state: u128,
    increment: u128,

    const MULTIPLIER: u128 = 0x2360ED051FC65DA44385DF649FCCF645;
    const DEFAULT_INCREMENT: u128 = 0x5851F42D4C957F2D14057B7EF767814F;

    fn init(seed: u128, stream: u128) Pcg64 {
        // Increment must be odd
        const inc = (stream << 1) | 1;
        var pcg = Pcg64{ .state = 0, .increment = inc };
        // Advance state twice with seed addition
        pcg.state = pcg.state *% MULTIPLIER +% inc;
        pcg.state +%= seed;
        pcg.state = pcg.state *% MULTIPLIER +% inc;
        return pcg;
    }

    fn next(self: *Pcg64) u64 {
        const old_state = self.state;
        self.state = old_state *% MULTIPLIER +% self.increment;
        // XSL-RR output function for 128-bit state -> 64-bit output
        const xored: u64 = @truncate((old_state ^ (old_state >> 64)) >> 58);
        const rot: u6 = @truncate(old_state >> 122);
        const out: u64 = @truncate((old_state ^ (old_state >> 64)) >> 6);
        _ = xored;
        return std.math.rotr(u64, out, rot);
    }
};

threadlocal var pcg64_instance: Pcg64 = Pcg64.init(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);

/// Initialize PCG64 with a seed and stream.
/// Different streams produce independent, non-overlapping sequences.
pub fn pcg64_seed(seed: u128, stream: u128) void {
    pcg64_instance = Pcg64.init(seed, stream);
}

/// Initialize PCG64 with two 64-bit values for seed and stream
pub fn pcg64_seed_u64(seed_hi: u64, seed_lo: u64, stream_hi: u64, stream_lo: u64) void {
    const seed: u128 = (@as(u128, seed_hi) << 64) | seed_lo;
    const stream: u128 = (@as(u128, stream_hi) << 64) | stream_lo;
    pcg64_seed(seed, stream);
}

/// Initialize PCG64 with an arbitrary byte array
pub fn pcg64_seed_bytes(seed_bytes: []const u8) void {
    // Hash seed bytes to create 128-bit seed
    var hash_lo: u64 = 0xcbf29ce484222325;
    var hash_hi: u64 = 0x6c62272e07bb0142;
    for (seed_bytes, 0..) |b, i| {
        if (i % 2 == 0) {
            hash_lo ^= b;
            hash_lo *%= 0x100000001b3;
        } else {
            hash_hi ^= b;
            hash_hi *%= 0x100000001b3;
        }
    }
    const seed: u128 = (@as(u128, hash_hi) << 64) | hash_lo;
    // Use different mixing for stream to ensure independence
    const stream: u128 = (@as(u128, hash_lo) << 64) | hash_hi;
    pcg64_seed(seed, stream);
}

/// Generate next PCG64 random u64
pub fn pcg64_next() u64 {
    return pcg64_instance.next();
}

/// C-compatible: Seed PCG64 with bytes
pub export fn n_pcg64_seed(seed_ptr: ?[*]const u8, seed_len: usize) void {
    if (seed_ptr) |ptr| {
        pcg64_seed_bytes(ptr[0..seed_len]);
    } else {
        pcg64_seed(0, 0);
    }
}

/// C-compatible: Seed PCG64 with stream selection (for parallel sequences)
pub export fn n_pcg64_seed_stream(seed_hi: u64, seed_lo: u64, stream_hi: u64, stream_lo: u64) void {
    pcg64_seed_u64(seed_hi, seed_lo, stream_hi, stream_lo);
}

/// C-compatible: Get next PCG64 value
pub export fn n_pcg64_random() u64 {
    return pcg64_next();
}

// Backward compatibility: keep rng_state as alias
const rng_state = &xoshiro_state;

/// Uniform u64
pub fn rand_u64() u64 {
    return xoshiro_next();
}

/// Uniform u32
pub fn rand_u32() u32 {
    return @intCast(xoshiro_next() & 0xFFFFFFFF);
}

/// Uniform double in [0,1)
pub fn rand_f64() f64 {
    const v = xoshiro_next() >> 11; // 53 bits
    return @as(f64, @floatFromInt(v)) * (1.0 / @as(f64, @floatFromInt(@as(u64, 1) << 53)));
}

/// Uniform float in [0,1)
pub fn rand_f32() f32 {
    const v = xoshiro_next() >> 40; // 24 bits
    return @as(f32, @floatFromInt(v)) * (1.0 / (1 << 24));
}

/// C-compatible rand() in [0, RAND_MAX]
pub export fn rand() c_int {
    return @intCast(rand_u32() % (@as(u32, RAND_MAX) + 1));
}

/// C-compatible random() returning long in [0, 2^31)
pub export fn random() c_long {
    return @intCast(rand_u32() & 0x7FFFFFFF);
}

/// RAND_MAX constant - maximum value returned by rand()
pub const RAND_MAX: c_int = 32767;

// === Distributions ===

/// Uniform integer in [0, max] inclusive
pub fn uniform_u64(max: u64) u64 {
    const bound = max + 1;
    const threshold = (~@as(u64, 0)) % bound;
    var r: u64 = undefined;
    while (true) {
        r = rand_u64();
        if (r >= threshold) break;
    }
    return r % bound;
}

/// Uniform float in [a, b)
pub fn uniform_f64(a: f64, b: f64) f64 {
    return a + (b - a) * rand_f64();
}

/// Standard normal (mean 0, stddev 1) via Box-Muller
threadlocal var spare_normal: f64 = 0.0;
threadlocal var has_spare: bool = false;

pub fn normal() f64 {
    if (has_spare) {
        has_spare = false;
        return spare_normal;
    }

    var u: f64 = 0;
    var v: f64 = 0;
    var s: f64 = 0;
    while (s == 0 or s >= 1) {
        u = rand_f64() * 2.0 - 1.0;
        v = rand_f64() * 2.0 - 1.0;
        s = u * u + v * v;
    }
    const m = std.math.sqrt(-2.0 * std.math.log(f64, std.math.e, s) / s);
    spare_normal = v * m;
    has_spare = true;
    return u * m;
}

pub fn normal_params(mean: f64, stddev: f64) f64 {
    return mean + stddev * normal();
}

pub fn lognormal(mean: f64, stddev: f64) f64 {
    return std.math.exp(normal_params(mean, stddev));
}

pub fn exponential(lambda: f64) f64 {
    const u = rand_f64();
    return -std.math.log(f64, std.math.e, 1.0 - u) / lambda;
}

pub fn gamma(shape: f64, scale: f64) f64 {
    if (shape < 1.0) {
        // Boost: gamma(k, theta) = gamma(k+1, theta) * U^(1/k)
        const g = gamma(shape + 1.0, scale);
        const u = rand_f64();
        return g * std.math.pow(f64, u, 1.0 / shape);
    }

    const d = shape - 1.0 / 3.0;
    const c = 1.0 / std.math.sqrt(9.0 * d);
    while (true) {
        const x = normal();
        const v = 1.0 + c * x;
        if (v <= 0) continue;
        const v3 = v * v * v;
        const u = rand_f64();
        if (u < 1.0 - 0.0331 * (x * x) * (x * x)) {
            return scale * d * v3;
        }
        if (std.math.log(f64, std.math.e, u) < 0.5 * x * x + d * (1.0 - v3 + std.math.log(f64, std.math.e, v3))) {
            return scale * d * v3;
        }
    }
}

pub fn poisson(lambda: f64) c_long {
    if (lambda <= 0) return 0;
    if (lambda < 30.0) {
        // Knuth for small lambda
        const L = std.math.exp(-lambda);
        var k: c_long = 0;
        var p: f64 = 1.0;
        while (p > L) {
            k += 1;
            p *= rand_f64();
        }
        return k - 1;
    }
    // Normal approximation fallback
    const g = std.math.sqrt(lambda);
    while (true) {
        const x = normal();
        const k = @as(c_long, @intFromFloat(std.math.floor(lambda + g * x + 0.5)));
        if (k >= 0) return k;
    }
}

// === Additional Distributions ===

/// Binomial distribution: number of successes in n trials with probability p
/// Uses exact algorithm for small n, normal approximation for large n
pub fn binomial(n: u64, p: f64) u64 {
    if (n == 0 or p <= 0.0) return 0;
    if (p >= 1.0) return n;

    // Use normal approximation for large n (n*p >= 10 and n*(1-p) >= 10)
    const np = @as(f64, @floatFromInt(n)) * p;
    const nq = @as(f64, @floatFromInt(n)) * (1.0 - p);

    if (np >= 10.0 and nq >= 10.0) {
        // Normal approximation with continuity correction
        const mean = np;
        const stddev = std.math.sqrt(np * (1.0 - p));
        while (true) {
            const x = normal_params(mean, stddev);
            const rounded = std.math.floor(x + 0.5);
            if (rounded >= 0.0 and rounded <= @as(f64, @floatFromInt(n))) {
                return @intFromFloat(rounded);
            }
        }
    }

    // Exact algorithm for small n: count successes in n Bernoulli trials
    var successes: u64 = 0;
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        if (rand_f64() < p) {
            successes += 1;
        }
    }
    return successes;
}

/// Beta distribution using gamma sampling: beta(a,b) = gamma(a,1) / (gamma(a,1) + gamma(b,1))
pub fn beta(alpha: f64, beta_param: f64) f64 {
    if (alpha <= 0.0 or beta_param <= 0.0) return 0.0;

    const x = gamma(alpha, 1.0);
    const y = gamma(beta_param, 1.0);

    // Handle edge case where both are very small
    if (x + y == 0.0) return 0.5;

    return x / (x + y);
}

/// Chi-squared distribution with k degrees of freedom
/// Chi-squared(k) = Gamma(k/2, 2)
pub fn chi_squared(k: f64) f64 {
    if (k <= 0.0) return 0.0;
    return gamma(k / 2.0, 2.0);
}

/// Student's t distribution with df degrees of freedom
/// t = Z / sqrt(V/df) where Z ~ N(0,1) and V ~ Chi-squared(df)
pub fn students_t(df: f64) f64 {
    if (df <= 0.0) return 0.0;

    const z = normal();
    const v = chi_squared(df);

    // Handle edge case
    if (v == 0.0) return 0.0;

    return z / std.math.sqrt(v / df);
}

// === Probability Density Functions (PDF) ===

/// Normal (Gaussian) probability density function
pub fn normal_pdf(x: f64, mean: f64, stddev: f64) f64 {
    if (stddev <= 0.0) return 0.0;

    const z = (x - mean) / stddev;
    const sqrt_2pi = std.math.sqrt(2.0 * std.math.pi);
    return std.math.exp(-0.5 * z * z) / (stddev * sqrt_2pi);
}

/// Exponential probability density function
pub fn exponential_pdf(x: f64, lambda: f64) f64 {
    if (lambda <= 0.0 or x < 0.0) return 0.0;
    return lambda * std.math.exp(-lambda * x);
}

/// Gamma probability density function
/// f(x; k, theta) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k))
pub fn gamma_pdf(x: f64, shape: f64, scale: f64) f64 {
    if (shape <= 0.0 or scale <= 0.0 or x < 0.0) return 0.0;
    if (x == 0.0) {
        if (shape < 1.0) return std.math.inf(f64);
        if (shape == 1.0) return 1.0 / scale;
        return 0.0;
    }

    // Use log-space to avoid overflow: log(f) = (k-1)*log(x) - x/theta - k*log(theta) - log(Gamma(k))
    const log_pdf = (shape - 1.0) * std.math.log(f64, std.math.e, x) -
        x / scale -
        shape * std.math.log(f64, std.math.e, scale) -
        lgamma(shape);
    return std.math.exp(log_pdf);
}

// === Cumulative Distribution Functions (CDF) ===

/// Normal (Gaussian) cumulative distribution function using error function approximation
pub fn normal_cdf(x: f64, mean: f64, stddev: f64) f64 {
    if (stddev <= 0.0) {
        if (x < mean) return 0.0;
        return 1.0;
    }

    const z = (x - mean) / (stddev * std.math.sqrt(2.0));
    return 0.5 * (1.0 + erf(z));
}

/// Exponential cumulative distribution function
pub fn exponential_cdf(x: f64, lambda: f64) f64 {
    if (lambda <= 0.0 or x < 0.0) return 0.0;
    return 1.0 - std.math.exp(-lambda * x);
}

/// Gamma cumulative distribution function (lower incomplete gamma ratio)
/// Uses series expansion for small x, continued fraction for large x
pub fn gamma_cdf(x: f64, shape: f64, scale: f64) f64 {
    if (shape <= 0.0 or scale <= 0.0 or x <= 0.0) return 0.0;

    const z = x / scale;
    return lower_incomplete_gamma_ratio(shape, z);
}

// === Quantile Functions (Inverse CDF) ===

/// Normal quantile function (inverse CDF) using rational approximation
/// Returns x such that P(X <= x) = p for X ~ N(mean, stddev)
pub fn normal_quantile(p: f64, mean: f64, stddev: f64) f64 {
    if (p <= 0.0) return -std.math.inf(f64);
    if (p >= 1.0) return std.math.inf(f64);
    if (stddev <= 0.0) return mean;

    // Use Abramowitz and Stegun rational approximation for standard normal
    const z = standard_normal_quantile(p);
    return mean + stddev * z;
}

/// Exponential quantile function (inverse CDF)
/// Returns x such that P(X <= x) = p for X ~ Exp(lambda)
pub fn exponential_quantile(p: f64, lambda: f64) f64 {
    if (p <= 0.0) return 0.0;
    if (p >= 1.0) return std.math.inf(f64);
    if (lambda <= 0.0) return 0.0;

    return -std.math.log(f64, std.math.e, 1.0 - p) / lambda;
}

// === Helper Functions for Special Mathematical Functions ===

/// Log-gamma function using Lanczos approximation
fn lgamma(x: f64) f64 {
    if (x <= 0.0) return std.math.inf(f64);

    // Lanczos approximation coefficients (g=7, n=9)
    const g: f64 = 7.0;
    const c = [_]f64{
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    };

    if (x < 0.5) {
        // Use reflection formula: Gamma(x) * Gamma(1-x) = pi / sin(pi*x)
        return std.math.log(f64, std.math.e, std.math.pi / @sin(std.math.pi * x)) - lgamma(1.0 - x);
    }

    const z = x - 1.0;
    var ag = c[0];
    for (1..9) |i| {
        ag += c[i] / (z + @as(f64, @floatFromInt(i)));
    }

    const sqrt_2pi = std.math.sqrt(2.0 * std.math.pi);
    return 0.5 * std.math.log(f64, std.math.e, 2.0 * std.math.pi) +
        (z + 0.5) * std.math.log(f64, std.math.e, z + g + 0.5) -
        (z + g + 0.5) +
        std.math.log(f64, std.math.e, ag / sqrt_2pi * sqrt_2pi);
}

/// Error function approximation using Horner's method
fn erf(x: f64) f64 {
    // Abramowitz and Stegun approximation 7.1.26
    const a1: f64 = 0.254829592;
    const a2: f64 = -0.284496736;
    const a3: f64 = 1.421413741;
    const a4: f64 = -1.453152027;
    const a5: f64 = 1.061405429;
    const p: f64 = 0.3275911;

    const sign: f64 = if (x < 0) -1.0 else 1.0;
    const abs_x = @abs(x);

    const t = 1.0 / (1.0 + p * abs_x);
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;

    const y = 1.0 - (a1 * t + a2 * t2 + a3 * t3 + a4 * t4 + a5 * t5) * std.math.exp(-abs_x * abs_x);
    return sign * y;
}

/// Lower incomplete gamma ratio P(a, x) = gamma(a, x) / Gamma(a)
/// Uses series expansion for x < a+1, continued fraction otherwise
fn lower_incomplete_gamma_ratio(a: f64, x: f64) f64 {
    if (x < 0.0) return 0.0;
    if (x == 0.0) return 0.0;

    if (x < a + 1.0) {
        // Series expansion: P(a,x) = e^(-x) * x^a * sum(x^n / Gamma(a+n+1))
        return gamma_series(a, x);
    } else {
        // Continued fraction: Q(a,x) = 1 - P(a,x)
        return 1.0 - gamma_continued_fraction(a, x);
    }
}

/// Series expansion for lower incomplete gamma
fn gamma_series(a: f64, x: f64) f64 {
    const max_iter: usize = 200;
    const eps: f64 = 1e-15;

    var sum: f64 = 1.0 / a;
    var term: f64 = 1.0 / a;
    var ap = a;

    for (0..max_iter) |_| {
        ap += 1.0;
        term *= x / ap;
        sum += term;
        if (@abs(term) < @abs(sum) * eps) break;
    }

    return sum * std.math.exp(-x + a * std.math.log(f64, std.math.e, x) - lgamma(a));
}

/// Continued fraction for upper incomplete gamma Q(a,x) = 1 - P(a,x)
fn gamma_continued_fraction(a: f64, x: f64) f64 {
    const max_iter: usize = 200;
    const eps: f64 = 1e-15;
    const tiny: f64 = 1e-30;

    // Modified Lentz's method
    var b = x + 1.0 - a;
    var c = 1.0 / tiny;
    var d = 1.0 / b;
    var h = d;

    for (1..max_iter + 1) |i| {
        const an = -@as(f64, @floatFromInt(i)) * (@as(f64, @floatFromInt(i)) - a);
        b += 2.0;
        d = an * d + b;
        if (@abs(d) < tiny) d = tiny;
        c = b + an / c;
        if (@abs(c) < tiny) c = tiny;
        d = 1.0 / d;
        const delta = d * c;
        h *= delta;
        if (@abs(delta - 1.0) < eps) break;
    }

    return std.math.exp(-x + a * std.math.log(f64, std.math.e, x) - lgamma(a)) * h;
}

/// Standard normal quantile function (inverse CDF for N(0,1))
/// Uses Abramowitz and Stegun rational approximation
fn standard_normal_quantile(p: f64) f64 {
    if (p <= 0.0) return -std.math.inf(f64);
    if (p >= 1.0) return std.math.inf(f64);

    // Rational approximation for central region
    if (p > 0.5) {
        return -standard_normal_quantile(1.0 - p);
    }

    // Approximation for lower tail (p <= 0.5)
    if (p < 1e-10) {
        // Very small p: use asymptotic expansion
        const t = std.math.sqrt(-2.0 * std.math.log(f64, std.math.e, p));
        return -t;
    }

    // Rational approximation coefficients
    const a = [_]f64{
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    };

    const b = [_]f64{
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    };

    const c = [_]f64{
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    };

    const d = [_]f64{
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    };

    const p_low: f64 = 0.02425;

    if (p < p_low) {
        // Lower region: rational approximation
        const q = std.math.sqrt(-2.0 * std.math.log(f64, std.math.e, p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else {
        // Central region: rational approximation
        const q = p - 0.5;
        const r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
            (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0);
    }
}

// =============================================================================
// === Legacy LCG - DEPRECATED ===
// =============================================================================
//
// WARNING: This simple LCG (Linear Congruential Generator) is provided only
// for legacy compatibility. It has poor statistical properties and should NOT
// be used for new code. Use xoshiro256** (via rand_u64) or PCG64 instead.
//
// Known issues with LCG:
// - Short period (2^64)
// - Predictable lower bits
// - Fails many statistical tests
// - Not suitable for simulations, cryptography, or any serious application

threadlocal var lcg_state: u64 = 0x123456789ABCDEF;

/// DEPRECATED: Simple LCG, kept for legacy compatibility only.
/// Use xoshiro_next() or pcg64_next() for new code.
fn lcg_next() u32 {
    lcg_state = lcg_state *% 6364136223846793005 +% 1442695040888963407;
    return @truncate(lcg_state >> 32);
}

/// DEPRECATED: Seed the legacy LCG generator.
/// For new code, use srandom() or xoshiro_seed_bytes()/pcg64_seed_bytes().
pub fn lcg_seed(seed: u64) void {
    lcg_state = seed;
}

/// C-compatible: Seed legacy LCG (deprecated)
pub export fn n_lcg_seed(seed: u64) void {
    lcg_seed(seed);
}

/// C-compatible: Get next legacy LCG value (deprecated)
pub export fn n_lcg_random() u32 {
    return lcg_next();
}

// Arc4random-compatible interface using the upgraded xoshiro256** backend
// These functions now use the high-quality xoshiro256** generator internally.

fn arc4_next() u32 {
    // Use xoshiro for better quality while maintaining interface compatibility
    return @truncate(xoshiro_next());
}

pub export fn arc4random() u32 {
    return arc4_next();
}

pub export fn arc4random_uniform(upper_bound: u32) u32 {
    if (upper_bound < 2) return 0;
    const min = (-%upper_bound) % upper_bound;
    var r: u32 = undefined;
    while (true) {
        r = arc4random();
        if (r >= min) break;
    }
    return r % upper_bound;
}

pub export fn arc4random_buf(buf: ?*anyopaque, nbytes: usize) void {
    if (buf == null) return;
    const bytes = @as([*]u8, @ptrCast(buf));
    var i: usize = 0;
    while (i < nbytes) {
        const r = arc4random();
        const copy_len = @min(4, nbytes - i);
        const r_bytes = std.mem.asBytes(&r);
        @memcpy(bytes[i..][0..copy_len], r_bytes[0..copy_len]);
        i += copy_len;
    }
}
