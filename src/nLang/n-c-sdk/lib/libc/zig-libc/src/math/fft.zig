// FFT and Probability utilities for zig-libc
// Pure Zig implementations of FFT, DFT, convolution, and probability functions

const std = @import("std");
const math = std.math;

// ============================================================================
// Complex Number Type
// ============================================================================

/// Complex number with f64 real and imaginary parts
pub const Complex = extern struct {
    re: f64,
    im: f64,

    /// Add two complex numbers
    pub fn add(a: Complex, b: Complex) Complex {
        return .{ .re = a.re + b.re, .im = a.im + b.im };
    }

    /// Subtract two complex numbers
    pub fn sub(a: Complex, b: Complex) Complex {
        return .{ .re = a.re - b.re, .im = a.im - b.im };
    }

    /// Multiply two complex numbers
    pub fn mul(a: Complex, b: Complex) Complex {
        return .{
            .re = a.re * b.re - a.im * b.im,
            .im = a.re * b.im + a.im * b.re,
        };
    }

    /// Divide two complex numbers
    pub fn div(a: Complex, b: Complex) Complex {
        const denom = b.re * b.re + b.im * b.im;
        if (denom == 0.0) {
            return .{ .re = math.nan(f64), .im = math.nan(f64) };
        }
        return .{
            .re = (a.re * b.re + a.im * b.im) / denom,
            .im = (a.im * b.re - a.re * b.im) / denom,
        };
    }

    /// Complex conjugate
    pub fn conj(a: Complex) Complex {
        return .{ .re = a.re, .im = -a.im };
    }

    /// Absolute value (magnitude)
    pub fn abs(a: Complex) f64 {
        return math.hypot(a.re, a.im);
    }

    /// e^(i*theta) = cos(theta) + i*sin(theta)
    pub fn exp_i(theta: f64) Complex {
        return .{ .re = math.cos(theta), .im = math.sin(theta) };
    }
};

// ============================================================================
// FFT/DFT Functions
// ============================================================================

/// Direct DFT - O(n²) complexity
/// X[k] = sum_{n=0}^{N-1} x[n] * e^(-2*pi*i*k*n/N)
pub export fn dft(input: [*]const Complex, output: [*]Complex, n: usize) void {
    if (n == 0) return;
    const pi2 = 2.0 * math.pi;

    for (0..n) |k| {
        var sum = Complex{ .re = 0.0, .im = 0.0 };
        for (0..n) |j| {
            const angle = -pi2 * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(n));
            const twiddle = Complex.exp_i(angle);
            sum = Complex.add(sum, Complex.mul(input[j], twiddle));
        }
        output[k] = sum;
    }
}

/// Inverse DFT - O(n²) complexity
/// x[n] = (1/N) * sum_{k=0}^{N-1} X[k] * e^(2*pi*i*k*n/N)
pub export fn idft(input: [*]const Complex, output: [*]Complex, n: usize) void {
    if (n == 0) return;
    const pi2 = 2.0 * math.pi;
    const inv_n = 1.0 / @as(f64, @floatFromInt(n));

    for (0..n) |j| {
        var sum = Complex{ .re = 0.0, .im = 0.0 };
        for (0..n) |k| {
            const angle = pi2 * @as(f64, @floatFromInt(k * j)) / @as(f64, @floatFromInt(n));
            const twiddle = Complex.exp_i(angle);
            sum = Complex.add(sum, Complex.mul(input[k], twiddle));
        }
        output[j] = Complex{ .re = sum.re * inv_n, .im = sum.im * inv_n };
    }
}

/// Cooley-Tukey FFT - O(n log n) complexity
/// n must be a power of 2
pub export fn fft(input: [*]const Complex, output: [*]Complex, n: usize) void {
    if (n == 0) return;
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    // Check if n is power of 2
    if ((n & (n - 1)) != 0) {
        // Fall back to DFT for non-power-of-2
        dft(input, output, n);
        return;
    }

    // Copy input to output for in-place processing
    for (0..n) |i| {
        output[i] = input[i];
    }

    // Bit-reversal permutation
    bitReverse(output, n);

    // Cooley-Tukey iterative FFT
    var len: usize = 2;
    while (len <= n) : (len *= 2) {
        const angle = -2.0 * math.pi / @as(f64, @floatFromInt(len));
        const wlen = Complex.exp_i(angle);

        var i: usize = 0;
        while (i < n) : (i += len) {
            var w = Complex{ .re = 1.0, .im = 0.0 };
            for (0..len / 2) |j| {
                const u = output[i + j];
                const t = Complex.mul(w, output[i + j + len / 2]);
                output[i + j] = Complex.add(u, t);
                output[i + j + len / 2] = Complex.sub(u, t);
                w = Complex.mul(w, wlen);
            }
        }
    }
}

/// Bit-reversal permutation helper
fn bitReverse(data: [*]Complex, n: usize) void {
    const bits = @ctz(n);
    for (0..n) |i| {
        const rev = @bitReverse(@as(usize, i)) >> (@bitSizeOf(usize) - bits);
        if (i < rev) {
            const tmp = data[i];
            data[i] = data[rev];
            data[rev] = tmp;
        }
    }
}

/// Inverse FFT - O(n log n) complexity
/// n must be a power of 2
pub export fn ifft(input: [*]const Complex, output: [*]Complex, n: usize) void {
    if (n == 0) return;
    if (n == 1) {
        output[0] = input[0];
        return;
    }

    // Check if n is power of 2
    if ((n & (n - 1)) != 0) {
        idft(input, output, n);
        return;
    }

    // Copy input to output for in-place processing
    for (0..n) |i| {
        output[i] = input[i];
    }

    // Bit-reversal permutation
    bitReverse(output, n);

    // Cooley-Tukey iterative FFT with positive angle (inverse)
    var len: usize = 2;
    while (len <= n) : (len *= 2) {
        const angle = 2.0 * math.pi / @as(f64, @floatFromInt(len));
        const wlen = Complex.exp_i(angle);

        var i: usize = 0;
        while (i < n) : (i += len) {
            var w = Complex{ .re = 1.0, .im = 0.0 };
            for (0..len / 2) |j| {
                const u = output[i + j];
                const t = Complex.mul(w, output[i + j + len / 2]);
                output[i + j] = Complex.add(u, t);
                output[i + j + len / 2] = Complex.sub(u, t);
                w = Complex.mul(w, wlen);
            }
        }
    }

    // Scale by 1/n
    const inv_n = 1.0 / @as(f64, @floatFromInt(n));
    for (0..n) |i| {
        output[i].re *= inv_n;
        output[i].im *= inv_n;
    }
}

/// FFT of real-valued input
/// Converts real input to complex and performs FFT
pub export fn fft_real(input: [*]const f64, output: [*]Complex, n: usize) void {
    if (n == 0) return;

    // Convert real to complex
    for (0..n) |i| {
        output[i] = Complex{ .re = input[i], .im = 0.0 };
    }

    // Create temp copy for input to fft
    var temp: [4096]Complex = undefined;
    if (n > 4096) {
        // For large n, use output as both input and output
        fftInPlace(output, n);
        return;
    }

    for (0..n) |i| {
        temp[i] = output[i];
    }
    fft(&temp, output, n);
}

/// In-place FFT helper for fft_real with large arrays
fn fftInPlace(data: [*]Complex, n: usize) void {
    if ((n & (n - 1)) != 0) return; // Must be power of 2

    bitReverse(data, n);

    var len: usize = 2;
    while (len <= n) : (len *= 2) {
        const angle = -2.0 * math.pi / @as(f64, @floatFromInt(len));
        const wlen = Complex.exp_i(angle);

        var i: usize = 0;
        while (i < n) : (i += len) {
            var w = Complex{ .re = 1.0, .im = 0.0 };
            for (0..len / 2) |j| {
                const u = data[i + j];
                const t = Complex.mul(w, data[i + j + len / 2]);
                data[i + j] = Complex.add(u, t);
                data[i + j + len / 2] = Complex.sub(u, t);
                w = Complex.mul(w, wlen);
            }
        }
    }
}

/// Compute power spectrum |X[k]|²
pub export fn power_spectrum(fft_output: [*]const Complex, result: [*]f64, n: usize) void {
    for (0..n) |k| {
        const re = fft_output[k].re;
        const im = fft_output[k].im;
        result[k] = re * re + im * im;
    }
}

// ============================================================================
// Convolution Functions
// ============================================================================

/// Find next power of 2 >= n
fn nextPow2(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if (@bitSizeOf(usize) > 32) {
        v |= v >> 32;
    }
    return v + 1;
}

/// Linear convolution using FFT
/// result should have size at least n_a + n_b - 1
/// Uses internal buffer - limited to 2048 elements per input
pub export fn convolve(a: [*]const f64, n_a: usize, b: [*]const f64, n_b: usize, result: [*]f64) void {
    if (n_a == 0 or n_b == 0) return;

    const conv_len = n_a + n_b - 1;
    const fft_len = nextPow2(conv_len);

    // Static buffers for FFT operations
    var buf_a: [4096]Complex = undefined;
    var buf_b: [4096]Complex = undefined;
    var fft_a: [4096]Complex = undefined;
    var fft_b: [4096]Complex = undefined;
    var ifft_out: [4096]Complex = undefined;

    if (fft_len > 4096) {
        // For very large convolutions, fall back to direct method
        directConvolve(a, n_a, b, n_b, result);
        return;
    }

    // Zero-pad and copy inputs
    for (0..fft_len) |i| {
        buf_a[i] = if (i < n_a) Complex{ .re = a[i], .im = 0.0 } else Complex{ .re = 0.0, .im = 0.0 };
        buf_b[i] = if (i < n_b) Complex{ .re = b[i], .im = 0.0 } else Complex{ .re = 0.0, .im = 0.0 };
    }

    // FFT both signals
    fft(&buf_a, &fft_a, fft_len);
    fft(&buf_b, &fft_b, fft_len);

    // Multiply in frequency domain
    for (0..fft_len) |i| {
        fft_a[i] = Complex.mul(fft_a[i], fft_b[i]);
    }

    // Inverse FFT
    ifft(&fft_a, &ifft_out, fft_len);

    // Copy real part to result
    for (0..conv_len) |i| {
        result[i] = ifft_out[i].re;
    }
}

/// Direct convolution fallback for large arrays
fn directConvolve(a: [*]const f64, n_a: usize, b: [*]const f64, n_b: usize, result: [*]f64) void {
    const conv_len = n_a + n_b - 1;

    // Zero initialize result
    for (0..conv_len) |i| {
        result[i] = 0.0;
    }

    // Direct convolution: (a * b)[n] = sum_k a[k] * b[n-k]
    for (0..n_a) |i| {
        for (0..n_b) |j| {
            result[i + j] += a[i] * b[j];
        }
    }
}

/// Cross-correlation using FFT
/// result should have size at least n_a + n_b - 1
/// Computes correlation: (a ⋆ b)[n] = sum_k a[k] * b[k+n]
pub export fn correlate(a: [*]const f64, n_a: usize, b: [*]const f64, n_b: usize, result: [*]f64) void {
    if (n_a == 0 or n_b == 0) return;

    const corr_len = n_a + n_b - 1;
    const fft_len = nextPow2(corr_len);

    var buf_a: [4096]Complex = undefined;
    var buf_b: [4096]Complex = undefined;
    var fft_a: [4096]Complex = undefined;
    var fft_b: [4096]Complex = undefined;
    var ifft_out: [4096]Complex = undefined;

    if (fft_len > 4096) {
        directCorrelate(a, n_a, b, n_b, result);
        return;
    }

    // Zero-pad and copy inputs
    for (0..fft_len) |i| {
        buf_a[i] = if (i < n_a) Complex{ .re = a[i], .im = 0.0 } else Complex{ .re = 0.0, .im = 0.0 };
        buf_b[i] = if (i < n_b) Complex{ .re = b[i], .im = 0.0 } else Complex{ .re = 0.0, .im = 0.0 };
    }

    // FFT both signals
    fft(&buf_a, &fft_a, fft_len);
    fft(&buf_b, &fft_b, fft_len);

    // Multiply A by conjugate of B in frequency domain
    for (0..fft_len) |i| {
        fft_a[i] = Complex.mul(fft_a[i], Complex.conj(fft_b[i]));
    }

    // Inverse FFT
    ifft(&fft_a, &ifft_out, fft_len);

    // Copy real part to result (with proper shift)
    for (0..corr_len) |i| {
        result[i] = ifft_out[i].re;
    }
}

/// Direct correlation fallback for large arrays
fn directCorrelate(a: [*]const f64, n_a: usize, b: [*]const f64, n_b: usize, result: [*]f64) void {
    const corr_len = n_a + n_b - 1;

    for (0..corr_len) |i| {
        result[i] = 0.0;
    }

    // Direct correlation
    for (0..n_a) |i| {
        for (0..n_b) |j| {
            const idx = i + (n_b - 1 - j);
            result[idx] += a[i] * b[j];
        }
    }
}

// ============================================================================
// Probability Utilities
// ============================================================================

/// Numerically stable log(sum(exp(a_i)))
/// Uses the log-sum-exp trick: log(sum(exp(a_i))) = max_a + log(sum(exp(a_i - max_a)))
pub export fn log_sum_exp(a: [*]const f64, n: usize) f64 {
    if (n == 0) return -math.inf(f64);
    if (n == 1) return a[0];

    // Find maximum
    var max_val = a[0];
    for (1..n) |i| {
        if (a[i] > max_val) max_val = a[i];
    }

    // Handle -inf case
    if (max_val == -math.inf(f64)) return -math.inf(f64);

    // Compute sum of exp(a_i - max)
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += math.exp(a[i] - max_val);
    }

    return max_val + math.log(sum);
}

/// Softmax: exp(a_i) / sum(exp(a_j))
/// Uses log-sum-exp for numerical stability
pub export fn softmax(a: [*]const f64, result: [*]f64, n: usize) void {
    if (n == 0) return;

    const lse = log_sum_exp(a, n);

    for (0..n) |i| {
        result[i] = math.exp(a[i] - lse);
    }
}

/// Log-softmax: log(softmax(a_i)) = a_i - log(sum(exp(a_j)))
/// More numerically stable than log(softmax(a))
pub export fn log_softmax(a: [*]const f64, result: [*]f64, n: usize) void {
    if (n == 0) return;

    const lse = log_sum_exp(a, n);

    for (0..n) |i| {
        result[i] = a[i] - lse;
    }
}

/// KL divergence D(P||Q) = sum(p_i * log(p_i / q_i))
/// Returns inf if q_i = 0 where p_i > 0
pub export fn kl_divergence(p: [*]const f64, q: [*]const f64, n: usize) f64 {
    if (n == 0) return 0.0;

    var sum: f64 = 0.0;
    for (0..n) |i| {
        if (p[i] > 0.0) {
            if (q[i] <= 0.0) {
                return math.inf(f64);
            }
            sum += p[i] * math.log(p[i] / q[i]);
        }
    }
    return sum;
}

/// Shannon entropy: -sum(p_i * log(p_i))
/// Treats 0 * log(0) as 0
pub export fn entropy(p: [*]const f64, n: usize) f64 {
    if (n == 0) return 0.0;

    var sum: f64 = 0.0;
    for (0..n) |i| {
        if (p[i] > 0.0) {
            sum -= p[i] * math.log(p[i]);
        }
    }
    return sum;
}

