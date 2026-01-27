// math module for zig-libc - Pure Zig implementations
const std = @import("std");
const math = std.math;
const bessel = @import("bessel.zig");
pub const linalg = @import("linalg.zig");
pub const montecarlo = @import("montecarlo.zig");
pub const blas = @import("blas.zig");
pub const fft = @import("fft.zig");
pub const benchmark = @import("benchmark.zig");
pub const attention = @import("attention.zig");
pub const graph = @import("graph.zig");
pub const quantization = @import("quantization.zig");
pub const sampling = @import("sampling.zig");
pub const simd = @import("simd.zig");

// --- Trigonometric ---
pub fn sin(x: f64) callconv(.C) f64 { return math.sin(x); }
pub fn cos(x: f64) callconv(.C) f64 { return math.cos(x); }
pub fn tan(x: f64) callconv(.C) f64 { return math.tan(x); }
pub fn asin(x: f64) callconv(.C) f64 { return math.asin(x); }
pub fn acos(x: f64) callconv(.C) f64 { return math.acos(x); }
pub fn atan(x: f64) callconv(.C) f64 { return math.atan(x); }
pub fn atan2(y: f64, x: f64) callconv(.C) f64 { return math.atan2(y, x); }

pub fn sinh(x: f64) callconv(.C) f64 { return math.sinh(x); }
pub fn cosh(x: f64) callconv(.C) f64 { return math.cosh(x); }
pub fn tanh(x: f64) callconv(.C) f64 { return math.tanh(x); }
pub fn asinh(x: f64) callconv(.C) f64 { return math.asinh(x); }
pub fn acosh(x: f64) callconv(.C) f64 { return math.acosh(x); }
pub fn atanh(x: f64) callconv(.C) f64 { return math.atanh(x); }

pub fn sinf(x: f32) callconv(.C) f32 { return math.sin(x); }
pub fn cosf(x: f32) callconv(.C) f32 { return math.cos(x); }
pub fn tanf(x: f32) callconv(.C) f32 { return math.tan(x); }
pub fn asinf(x: f32) callconv(.C) f32 { return math.asin(x); }
pub fn acosf(x: f32) callconv(.C) f32 { return math.acos(x); }
pub fn atanf(x: f32) callconv(.C) f32 { return math.atan(x); }
pub fn atan2f(y: f32, x: f32) callconv(.C) f32 { return math.atan2(y, x); }

pub fn sinhf(x: f32) callconv(.C) f32 { return math.sinh(x); }
pub fn coshf(x: f32) callconv(.C) f32 { return math.cosh(x); }
pub fn tanhf(x: f32) callconv(.C) f32 { return math.tanh(x); }
pub fn asinhf(x: f32) callconv(.C) f32 { return math.asinh(x); }
pub fn acoshf(x: f32) callconv(.C) f32 { return math.acosh(x); }
pub fn atanhf(x: f32) callconv(.C) f32 { return math.atanh(x); }

pub fn sinl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.sin(@as(f128, @floatCast(x)))); }
pub fn cosl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.cos(@as(f128, @floatCast(x)))); }
pub fn tanl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.tan(@as(f128, @floatCast(x)))); }
pub fn asinl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.asin(@as(f128, @floatCast(x)))); }
pub fn acosl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.acos(@as(f128, @floatCast(x)))); }
pub fn atanl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.atan(@as(f128, @floatCast(x)))); }
pub fn atan2l(y: c_longdouble, x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.atan2(@as(f128, @floatCast(y)), @as(f128, @floatCast(x)))); }

pub fn sinhl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.sinh(@as(f128, @floatCast(x)))); }
pub fn coshl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.cosh(@as(f128, @floatCast(x)))); }
pub fn tanhl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.tanh(@as(f128, @floatCast(x)))); }
pub fn asinhl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.asinh(@as(f128, @floatCast(x)))); }
pub fn acoshl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.acosh(@as(f128, @floatCast(x)))); }
pub fn atanhl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.atanh(@as(f128, @floatCast(x)))); }

// --- Exponential / Logarithmic ---
pub fn exp(x: f64) callconv(.C) f64 { return math.exp(x); }
pub fn exp2(x: f64) callconv(.C) f64 { return math.exp2(x); }
pub fn expm1(x: f64) callconv(.C) f64 { return math.expm1(x); }
pub fn log(x: f64) callconv(.C) f64 { return math.log(x); }
pub fn log10(x: f64) callconv(.C) f64 { return math.log10(x); }
pub fn log2(x: f64) callconv(.C) f64 { return math.log2(x); }
pub fn log1p(x: f64) callconv(.C) f64 { return math.log1p(x); }
pub fn pow(x: f64, y: f64) callconv(.C) f64 { return math.pow(f64, x, y); }
pub fn sqrt(x: f64) callconv(.C) f64 { return math.sqrt(x); }
pub fn cbrt(x: f64) callconv(.C) f64 { return math.cbrt(x); }
pub fn hypot(x: f64, y: f64) callconv(.C) f64 { return math.hypot(x, y); }

pub fn expf(x: f32) callconv(.C) f32 { return math.exp(x); }
pub fn exp2f(x: f32) callconv(.C) f32 { return math.exp2(x); }
pub fn expm1f(x: f32) callconv(.C) f32 { return math.expm1(x); }
pub fn logf(x: f32) callconv(.C) f32 { return math.log(x); }
pub fn log10f(x: f32) callconv(.C) f32 { return math.log10(x); }
pub fn log2f(x: f32) callconv(.C) f32 { return math.log2(x); }
pub fn log1pf(x: f32) callconv(.C) f32 { return math.log1p(x); }
pub fn powf(x: f32, y: f32) callconv(.C) f32 { return math.pow(f32, x, y); }
pub fn sqrtf(x: f32) callconv(.C) f32 { return math.sqrt(x); }
pub fn cbrtf(x: f32) callconv(.C) f32 { return math.cbrt(x); }
pub fn hypotf(x: f32, y: f32) callconv(.C) f32 { return math.hypot(x, y); }

pub fn expl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.exp(@as(f128, @floatCast(x)))); }
pub fn exp2l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.exp2(@as(f128, @floatCast(x)))); }
pub fn expm1l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.expm1(@as(f128, @floatCast(x)))); }
pub fn logl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.log(@as(f128, @floatCast(x)))); }
pub fn log10l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.log10(@as(f128, @floatCast(x)))); }
pub fn log2l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.log2(@as(f128, @floatCast(x)))); }
pub fn log1pl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.log1p(@as(f128, @floatCast(x)))); }
pub fn powl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.pow(f128, @as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }
pub fn sqrtl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.sqrt(@as(f128, @floatCast(x)))); }
pub fn cbrtl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.cbrt(@as(f128, @floatCast(x)))); }
pub fn hypotl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.hypot(@as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }

// --- Rounding / Remainder ---
pub fn ceil(x: f64) callconv(.C) f64 { return math.ceil(x); }
pub fn floor(x: f64) callconv(.C) f64 { return math.floor(x); }
pub fn trunc(x: f64) callconv(.C) f64 { return math.trunc(x); }
pub fn round(x: f64) callconv(.C) f64 { return math.round(x); }
pub fn nearbyint(x: f64) callconv(.C) f64 { return math.round(x); } // Approximation
pub fn fmod(x: f64, y: f64) callconv(.C) f64 { return math.mod(f64, x, y); } // Note: math.mod might differ from fmod
pub fn remainder(x: f64, y: f64) callconv(.C) f64 { return math.rem(f64, x, y); }
pub fn remquo(x: f64, y: f64, quo: *c_int) callconv(.C) f64 {
    if (!math.isFinite(x) or !math.isFinite(y) or y == 0.0) {
        quo.* = 0;
        if (math.isNan(x) or math.isNan(y)) return math.nan(f64);
        return math.nan(f64);
    }

    // Compute quotient as nearest integer
    const q = x / y;
    const n = math.round(q);

    // Extract sign and low 3 bits of quotient
    const sign: c_int = if ((x < 0) != (y < 0)) -1 else 1;
    const abs_n: u64 = @intFromFloat(@abs(n));
    const low_bits: c_int = @intCast(abs_n & 0x7);
    quo.* = sign * low_bits;

    // Compute remainder: x - n*y
    return x - n * y;
}

pub fn ceilf(x: f32) callconv(.C) f32 { return math.ceil(x); }
pub fn floorf(x: f32) callconv(.C) f32 { return math.floor(x); }
pub fn truncf(x: f32) callconv(.C) f32 { return math.trunc(x); }
pub fn roundf(x: f32) callconv(.C) f32 { return math.round(x); }
pub fn nearbyintf(x: f32) callconv(.C) f32 { return math.round(x); }
pub fn fmodf(x: f32, y: f32) callconv(.C) f32 { return math.mod(f32, x, y); }
pub fn remainderf(x: f32, y: f32) callconv(.C) f32 { return math.rem(f32, x, y); }
pub fn remquof(x: f32, y: f32, quo: *c_int) callconv(.C) f32 {
    if (!math.isFinite(x) or !math.isFinite(y) or y == 0.0) {
        quo.* = 0;
        if (math.isNan(x) or math.isNan(y)) return math.nan(f32);
        return math.nan(f32);
    }

    const q = x / y;
    const n = math.round(q);
    const sign: c_int = if ((x < 0) != (y < 0)) -1 else 1;
    const abs_n: u32 = @intFromFloat(@abs(n));
    const low_bits: c_int = @intCast(abs_n & 0x7);
    quo.* = sign * low_bits;

    return x - n * y;
}

pub fn ceill(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.ceil(@as(f128, @floatCast(x)))); }
pub fn floorl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.floor(@as(f128, @floatCast(x)))); }
pub fn truncl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.trunc(@as(f128, @floatCast(x)))); }
pub fn roundl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.round(@as(f128, @floatCast(x)))); }
pub fn nearbyintl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.round(@as(f128, @floatCast(x)))); }
pub fn fmodl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.mod(f128, @as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }
pub fn remainderl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.rem(f128, @as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }
pub fn remquol(x: c_longdouble, y: c_longdouble, quo: *c_int) callconv(.C) c_longdouble {
    if (!math.isFinite(x) or !math.isFinite(y) or y == 0.0) {
        quo.* = 0;
        if (math.isNan(x) or math.isNan(y)) return @as(c_longdouble, math.nan(f64));
        return @as(c_longdouble, math.nan(f64));
    }

    const q = x / y;
    const n = math.round(q);
    const sign: c_int = if ((x < 0) != (y < 0)) -1 else 1;
    const abs_n: u64 = @intFromFloat(@abs(n));
    const low_bits: c_int = @intCast(abs_n & 0x7);
    quo.* = sign * low_bits;

    return x - n * y;
}

// --- Special Functions ---
pub fn erf(x: f64) callconv(.C) f64 { return math.erf(x); }
pub fn erfc(x: f64) callconv(.C) f64 { return math.erfc(x); }
pub fn tgamma(x: f64) callconv(.C) f64 { return math.gamma(f64, x); }
pub fn lgamma(x: f64) callconv(.C) f64 { return math.lgamma(f64, x); }
pub fn j0(x: f64) callconv(.C) f64 { return bessel.j0_impl(x); }
pub fn j1(x: f64) callconv(.C) f64 { return bessel.j1_impl(x); }
pub fn jn(n: c_int, x: f64) callconv(.C) f64 { return bessel.jn_impl(@intCast(n), x); }
pub fn y0(x: f64) callconv(.C) f64 { return bessel.y0_impl(x); }
pub fn y1(x: f64) callconv(.C) f64 { return bessel.y1_impl(x); }
pub fn yn(n: c_int, x: f64) callconv(.C) f64 { return bessel.yn_impl(@intCast(n), x); }

// Modified Bessel functions (i0/i1/in shadow Zig primitives, use @"..." syntax)
pub fn @"i0"(x: f64) callconv(.C) f64 { return bessel.bessel_i0_impl(x); }
pub fn @"i1"(x: f64) callconv(.C) f64 { return bessel.bessel_i1_impl(x); }
pub fn @"in"(n: c_int, x: f64) callconv(.C) f64 { return bessel.in_impl(@intCast(n), x); }
pub fn k0(x: f64) callconv(.C) f64 { return bessel.k0_impl(x); }
pub fn k1(x: f64) callconv(.C) f64 { return bessel.k1_impl(x); }
pub fn kn(n: c_int, x: f64) callconv(.C) f64 { return bessel.kn_impl(@intCast(n), x); }

pub fn erff(x: f32) callconv(.C) f32 { return math.erf(x); }
pub fn erfcf(x: f32) callconv(.C) f32 { return math.erfc(x); }
pub fn tgammaf(x: f32) callconv(.C) f32 { return math.gamma(f32, x); }
pub fn lgammaf(x: f32) callconv(.C) f32 { return math.lgamma(f32, x); }
pub fn j0f(x: f32) callconv(.C) f32 { return bessel.j0f_impl(x); }
pub fn j1f(x: f32) callconv(.C) f32 { return bessel.j1f_impl(x); }
pub fn jnf(n: c_int, x: f32) callconv(.C) f32 { return bessel.jnf_impl(@intCast(n), x); }
pub fn y0f(x: f32) callconv(.C) f32 { return bessel.y0f_impl(x); }
pub fn y1f(x: f32) callconv(.C) f32 { return bessel.y1f_impl(x); }
pub fn ynf(n: c_int, x: f32) callconv(.C) f32 { return bessel.ynf_impl(@intCast(n), x); }

// Modified Bessel f32 versions
pub fn i0f(x: f32) callconv(.C) f32 { return bessel.bessel_i0f_impl(x); }
pub fn i1f(x: f32) callconv(.C) f32 { return bessel.bessel_i1f_impl(x); }
pub fn inf(n: c_int, x: f32) callconv(.C) f32 { return bessel.inf_impl(@intCast(n), x); }
pub fn k0f(x: f32) callconv(.C) f32 { return bessel.k0f_impl(x); }
pub fn k1f(x: f32) callconv(.C) f32 { return bessel.k1f_impl(x); }
pub fn knf(n: c_int, x: f32) callconv(.C) f32 { return bessel.knf_impl(@intCast(n), x); }

pub fn erfl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.erf(@as(f128, @floatCast(x)))); }
pub fn erfcl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.erfc(@as(f128, @floatCast(x)))); }
pub fn tgammal(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.gamma(f128, @as(f128, @floatCast(x)))); }
pub fn lgammal(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.lgamma(f128, @as(f128, @floatCast(x)))); }
pub fn j0l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.j0_impl(@floatCast(x))); }
pub fn j1l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.j1_impl(@floatCast(x))); }
pub fn jnl(n: c_int, x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.jn_impl(@intCast(n), @floatCast(x))); }
pub fn y0l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.y0_impl(@floatCast(x))); }
pub fn y1l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.y1_impl(@floatCast(x))); }
pub fn ynl(n: c_int, x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.yn_impl(@intCast(n), @floatCast(x))); }

// Modified Bessel long double versions
pub fn i0l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.bessel_i0_impl(@floatCast(x))); }
pub fn i1l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.bessel_i1_impl(@floatCast(x))); }
pub fn inl(n: c_int, x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.in_impl(@intCast(n), @floatCast(x))); }
pub fn k0l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.k0_impl(@floatCast(x))); }
pub fn k1l(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.k1_impl(@floatCast(x))); }
pub fn knl(n: c_int, x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(bessel.kn_impl(@intCast(n), @floatCast(x))); }

// --- Extended Special Functions ---

// Beta function: B(a,b) = Gamma(a)*Gamma(b)/Gamma(a+b)
fn beta_impl(a: f64, b: f64) f64 {
    // Handle edge cases
    if (!math.isFinite(a) or !math.isFinite(b)) {
        if (math.isNan(a) or math.isNan(b)) return math.nan(f64);
        return 0.0;
    }
    if (a <= 0.0 or b <= 0.0) {
        // For non-positive integers, gamma has poles
        if (a == @trunc(a) or b == @trunc(b)) return math.nan(f64);
    }
    // Use log-beta for numerical stability: exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    return @exp(lbeta_impl(a, b));
}

// Log-beta function: ln(B(a,b)) = lgamma(a) + lgamma(b) - lgamma(a+b)
fn lbeta_impl(a: f64, b: f64) f64 {
    if (!math.isFinite(a) or !math.isFinite(b)) {
        if (math.isNan(a) or math.isNan(b)) return math.nan(f64);
        return -math.inf(f64);
    }
    const lga = math.lgamma(f64, a);
    const lgb = math.lgamma(f64, b);
    const lgab = math.lgamma(f64, a + b);
    return lga + lgb - lgab;
}

pub export fn beta(a: f64, b: f64) f64 { return beta_impl(a, b); }
pub export fn lbeta(a: f64, b: f64) f64 { return lbeta_impl(a, b); }
pub export fn betaf(a: f32, b: f32) f32 { return @floatCast(beta_impl(@floatCast(a), @floatCast(b))); }
pub export fn lbetaf(a: f32, b: f32) f32 { return @floatCast(lbeta_impl(@floatCast(a), @floatCast(b))); }
pub export fn betal(a: c_longdouble, b: c_longdouble) c_longdouble { return @floatCast(beta_impl(@floatCast(a), @floatCast(b))); }
pub export fn lbetal(a: c_longdouble, b: c_longdouble) c_longdouble { return @floatCast(lbeta_impl(@floatCast(a), @floatCast(b))); }

// Lower incomplete gamma function: γ(a, x) = ∫₀ˣ t^(a-1) * e^(-t) dt
// Returns the regularized form: P(a,x) = γ(a,x) / Γ(a)
fn lower_incomplete_gamma_impl(a: f64, x: f64) f64 {
    if (!math.isFinite(a) or !math.isFinite(x)) {
        if (math.isNan(a) or math.isNan(x)) return math.nan(f64);
        if (x == math.inf(f64)) return 1.0;
        return 0.0;
    }
    if (x < 0.0) return math.nan(f64);
    if (x == 0.0) return 0.0;
    if (a <= 0.0 and a == @trunc(a)) return math.nan(f64);

    // Use series expansion for x < a+1
    if (x < a + 1.0) {
        return gamma_series(a, x);
    }
    // Use continued fraction for x >= a+1
    return 1.0 - gamma_cf(a, x);
}

// Upper incomplete gamma function: Γ(a, x) = ∫ₓ^∞ t^(a-1) * e^(-t) dt
// Returns the regularized form: Q(a,x) = Γ(a,x) / Γ(a) = 1 - P(a,x)
fn upper_incomplete_gamma_impl(a: f64, x: f64) f64 {
    if (!math.isFinite(a) or !math.isFinite(x)) {
        if (math.isNan(a) or math.isNan(x)) return math.nan(f64);
        if (x == math.inf(f64)) return 0.0;
        return 1.0;
    }
    if (x < 0.0) return math.nan(f64);
    if (x == 0.0) return 1.0;
    if (a <= 0.0 and a == @trunc(a)) return math.nan(f64);

    if (x < a + 1.0) {
        return 1.0 - gamma_series(a, x);
    }
    return gamma_cf(a, x);
}

// Series expansion for lower incomplete gamma (regularized)
fn gamma_series(a: f64, x: f64) f64 {
    const max_iter: usize = 200;
    const eps: f64 = 1e-15;

    var ap = a;
    var sum = 1.0 / a;
    var del = sum;

    for (0..max_iter) |_| {
        ap += 1.0;
        del *= x / ap;
        sum += del;
        if (@abs(del) < @abs(sum) * eps) break;
    }

    // Return regularized: sum * x^a * exp(-x) / Gamma(a)
    const lga = math.lgamma(f64, a);
    return sum * @exp(a * @log(x) - x - lga);
}

// Continued fraction for upper incomplete gamma (regularized)
fn gamma_cf(a: f64, x: f64) f64 {
    const max_iter: usize = 200;
    const eps: f64 = 1e-15;
    const fpmin: f64 = 1e-300;

    var b = x + 1.0 - a;
    var c = 1.0 / fpmin;
    var d = 1.0 / b;
    var h = d;

    for (1..max_iter + 1) |i| {
        const fi: f64 = @floatFromInt(i);
        const an = -fi * (fi - a);
        b += 2.0;
        d = an * d + b;
        if (@abs(d) < fpmin) d = fpmin;
        c = b + an / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const del = d * c;
        h *= del;
        if (@abs(del - 1.0) < eps) break;
    }

    const lga = math.lgamma(f64, a);
    return @exp(a * @log(x) - x - lga) * h;
}

pub export fn lower_incomplete_gamma(a: f64, x: f64) f64 { return lower_incomplete_gamma_impl(a, x); }
pub export fn upper_incomplete_gamma(a: f64, x: f64) f64 { return upper_incomplete_gamma_impl(a, x); }
pub export fn lower_incomplete_gammaf(a: f32, x: f32) f32 { return @floatCast(lower_incomplete_gamma_impl(@floatCast(a), @floatCast(x))); }
pub export fn upper_incomplete_gammaf(a: f32, x: f32) f32 { return @floatCast(upper_incomplete_gamma_impl(@floatCast(a), @floatCast(x))); }
pub export fn lower_incomplete_gammal(a: c_longdouble, x: c_longdouble) c_longdouble { return @floatCast(lower_incomplete_gamma_impl(@floatCast(a), @floatCast(x))); }
pub export fn upper_incomplete_gammal(a: c_longdouble, x: c_longdouble) c_longdouble { return @floatCast(upper_incomplete_gamma_impl(@floatCast(a), @floatCast(x))); }

// Regularized incomplete beta function: I_x(a,b) = B(x;a,b) / B(a,b)
// where B(x;a,b) = ∫₀ˣ t^(a-1) * (1-t)^(b-1) dt
fn incomplete_beta_impl(x: f64, a: f64, b: f64) f64 {
    if (!math.isFinite(x) or !math.isFinite(a) or !math.isFinite(b)) {
        return math.nan(f64);
    }
    if (x < 0.0 or x > 1.0) return math.nan(f64);
    if (a <= 0.0 or b <= 0.0) return math.nan(f64);
    if (x == 0.0) return 0.0;
    if (x == 1.0) return 1.0;

    // Use symmetry relation: I_x(a,b) = 1 - I_{1-x}(b,a)
    // Choose the form with better convergence
    if (x > (a + 1.0) / (a + b + 2.0)) {
        return 1.0 - incomplete_beta_impl(1.0 - x, b, a);
    }

    // Use continued fraction expansion
    return beta_cf(x, a, b);
}

// Continued fraction for incomplete beta
fn beta_cf(x: f64, a: f64, b: f64) f64 {
    const max_iter: usize = 200;
    const eps: f64 = 1e-15;
    const fpmin: f64 = 1e-300;

    const qab = a + b;
    const qap = a + 1.0;
    const qam = a - 1.0;

    var c: f64 = 1.0;
    var d = 1.0 - qab * x / qap;
    if (@abs(d) < fpmin) d = fpmin;
    d = 1.0 / d;
    var h = d;

    for (1..max_iter + 1) |m| {
        const mf: f64 = @floatFromInt(m);
        const m2: f64 = 2.0 * mf;

        // Even step
        var aa = mf * (b - mf) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        h *= d * c;

        // Odd step
        aa = -(a + mf) * (qab + mf) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if (@abs(d) < fpmin) d = fpmin;
        c = 1.0 + aa / c;
        if (@abs(c) < fpmin) c = fpmin;
        d = 1.0 / d;
        const del = d * c;
        h *= del;

        if (@abs(del - 1.0) < eps) break;
    }

    // Multiply by the prefactor
    const lbeta_val = lbeta_impl(a, b);
    const prefactor = @exp(a * @log(x) + b * @log(1.0 - x) - lbeta_val) / a;
    return prefactor * h;
}

pub export fn incomplete_beta(x: f64, a: f64, b: f64) f64 { return incomplete_beta_impl(x, a, b); }
pub export fn incomplete_betaf(x: f32, a: f32, b: f32) f32 { return @floatCast(incomplete_beta_impl(@floatCast(x), @floatCast(a), @floatCast(b))); }
pub export fn incomplete_betal(x: c_longdouble, a: c_longdouble, b: c_longdouble) c_longdouble { return @floatCast(incomplete_beta_impl(@floatCast(x), @floatCast(a), @floatCast(b))); }

// Digamma (psi) function: ψ(x) = d/dx ln(Γ(x)) = Γ'(x)/Γ(x)
fn digamma_impl(x: f64) f64 {
    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return math.inf(f64);
        return math.nan(f64);
    }

    // Handle negative values using reflection formula:
    // ψ(1-x) - ψ(x) = π*cot(πx)
    if (x <= 0.0) {
        if (x == @trunc(x)) return math.nan(f64); // poles at non-positive integers
        return digamma_impl(1.0 - x) - math.pi / @tan(math.pi * x);
    }

    // Use recurrence relation to shift x to larger values: ψ(x+1) = ψ(x) + 1/x
    var result: f64 = 0.0;
    var xv = x;
    while (xv < 6.0) {
        result -= 1.0 / xv;
        xv += 1.0;
    }

    // Asymptotic expansion for large x:
    // ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶) + ...
    result += @log(xv) - 0.5 / xv;

    const x2 = 1.0 / (xv * xv);
    // Bernoulli numbers: B2=1/6, B4=-1/30, B6=1/42, B8=-1/30, B10=5/66
    const coeffs = [_]f64{
        1.0 / 12.0, // B2/2
        -1.0 / 120.0, // B4/4
        1.0 / 252.0, // B6/6
        -1.0 / 240.0, // B8/8
        5.0 / 660.0, // B10/10
        -691.0 / 32760.0, // B12/12
        1.0 / 12.0, // B14/14 (approx)
    };

    var term = x2;
    for (coeffs) |c| {
        result -= c * term;
        term *= x2;
    }

    return result;
}

pub export fn digamma(x: f64) f64 { return digamma_impl(x); }
pub export fn digammaf(x: f32) f32 { return @floatCast(digamma_impl(@floatCast(x))); }
pub export fn digammal(x: c_longdouble) c_longdouble { return @floatCast(digamma_impl(@floatCast(x))); }

// --- Manipulation ---
pub fn frexp(x: f64, exp_ptr: *c_int) callconv(.C) f64 { 
    const result = math.frexp(x);
    exp_ptr.* = @intCast(result.exponent);
    return result.significand;
}
pub fn ldexp(x: f64, exp_val: c_int) callconv(.C) f64 { return math.ldexp(x, @intCast(exp_val)); }
pub fn modf(x: f64, iptr: *f64) callconv(.C) f64 {
    const result = math.modf(x);
    iptr.* = result.ipart;
    return result.fpart;
}
pub fn scalbn(x: f64, n: c_int) callconv(.C) f64 { return math.scalbn(x, @intCast(n)); }
pub fn logb(x: f64) callconv(.C) f64 { return math.logb(x); }
pub fn nextafter(x: f64, y: f64) callconv(.C) f64 { return math.nextafter(x, y); }
pub fn copysign(x: f64, y: f64) callconv(.C) f64 { return math.copysign(x, y); }
pub fn nan(tagp: [*:0]const u8) callconv(.C) f64 { _ = tagp; return math.nan(f64); }

pub fn frexpf(x: f32, exp_ptr: *c_int) callconv(.C) f32 {
    const result = math.frexp(x);
    exp_ptr.* = @intCast(result.exponent);
    return result.significand;
}
pub fn ldexpf(x: f32, exp_val: c_int) callconv(.C) f32 { return math.ldexp(x, @intCast(exp_val)); }
pub fn modff(x: f32, iptr: *f32) callconv(.C) f32 {
    const result = math.modf(x);
    iptr.* = result.ipart;
    return result.fpart;
}
pub fn scalbnf(x: f32, n: c_int) callconv(.C) f32 { return math.scalbn(x, @intCast(n)); }
pub fn logbf(x: f32) callconv(.C) f32 { return math.logb(x); }
pub fn nextafterf(x: f32, y: f32) callconv(.C) f32 { return math.nextafter(x, y); }
pub fn copysignf(x: f32, y: f32) callconv(.C) f32 { return math.copysign(x, y); }
pub fn nanf(tagp: [*:0]const u8) callconv(.C) f32 { _ = tagp; return math.nan(f32); }

pub fn frexpl(x: c_longdouble, exp_ptr: *c_int) callconv(.C) c_longdouble {
    const result = math.frexp(@as(f128, @floatCast(x)));
    exp_ptr.* = @intCast(result.exponent);
    return @floatCast(result.significand);
}
pub fn ldexpl(x: c_longdouble, exp_val: c_int) callconv(.C) c_longdouble { return @floatCast(math.ldexp(@as(f128, @floatCast(x)), @intCast(exp_val))); }
pub fn modfl(x: c_longdouble, iptr: *c_longdouble) callconv(.C) c_longdouble {
    const result = math.modf(@as(f128, @floatCast(x)));
    iptr.* = @floatCast(result.ipart);
    return @floatCast(result.fpart);
}
pub fn scalbnl(x: c_longdouble, n: c_int) callconv(.C) c_longdouble { return @floatCast(math.scalbn(@as(f128, @floatCast(x)), @intCast(n))); }
pub fn logbl(x: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.logb(@as(f128, @floatCast(x)))); }
pub fn nextafterl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.nextafter(@as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }
pub fn copysignl(x: c_longdouble, y: c_longdouble) callconv(.C) c_longdouble { return @floatCast(math.copysign(@as(f128, @floatCast(x)), @as(f128, @floatCast(y)))); }
pub fn nanl(tagp: [*:0]const u8) callconv(.C) c_longdouble { _ = tagp; return @floatCast(math.nan(f128)); }


// --- Classification helpers ---
pub const FP_NAN: c_int = 0;
pub const FP_INFINITE: c_int = 1;
pub const FP_ZERO: c_int = 2;
pub const FP_SUBNORMAL: c_int = 3;
pub const FP_NORMAL: c_int = 4;

fn classify(comptime T: type, x: T) c_int {
    if (std.math.isNan(x)) return FP_NAN;
    if (std.math.isInf(x)) return FP_INFINITE;
    if (x == 0) return FP_ZERO;
    if (std.math.isNormal(x)) return FP_NORMAL;
    return FP_SUBNORMAL;
}

pub export fn fpclassify(x: f64) c_int { return classify(f64, x); }
pub export fn fpclassifyf(x: f32) c_int { return classify(f32, x); }
pub export fn fpclassifyl(x: c_longdouble) c_int {
    return classify(c_longdouble, x);
}

pub export fn isfinite(x: f64) c_int { return if (std.math.isFinite(x)) 1 else 0; }
pub export fn isfinitef(x: f32) c_int { return if (std.math.isFinite(x)) 1 else 0; }
pub export fn isfinitel(x: c_longdouble) c_int { return if (std.math.isFinite(x)) 1 else 0; }

pub export fn isinf(x: f64) c_int { return if (std.math.isInf(x)) 1 else 0; }
pub export fn isinff(x: f32) c_int { return if (std.math.isInf(x)) 1 else 0; }
pub export fn isinfl(x: c_longdouble) c_int { return if (std.math.isInf(x)) 1 else 0; }

pub export fn isnan(x: f64) c_int { return if (std.math.isNan(x)) 1 else 0; }
pub export fn isnanf(x: f32) c_int { return if (std.math.isNan(x)) 1 else 0; }
pub export fn isnanl(x: c_longdouble) c_int { return if (std.math.isNan(x)) 1 else 0; }

pub export fn isnormal(x: f64) c_int { return if (std.math.isNormal(x)) 1 else 0; }
pub export fn isnormalf(x: f32) c_int { return if (std.math.isNormal(x)) 1 else 0; }
pub export fn isnormall(x: c_longdouble) c_int { return if (std.math.isNormal(x)) 1 else 0; }

pub export fn signbit(x: f64) c_int { return if (std.math.signbit(x)) 1 else 0; }
pub export fn signbitf(x: f32) c_int { return if (std.math.signbit(x)) 1 else 0; }
pub export fn signbitl(x: c_longdouble) c_int { return if (std.math.signbit(x)) 1 else 0; }

inline fn cmpOrdered(a: anytype, b: anytype) bool {
    return !(std.math.isNan(a) or std.math.isNan(b));
}

pub export fn isgreater(x: f64, y: f64) c_int { return if (cmpOrdered(x, y) and x > y) 1 else 0; }
pub export fn isgreaterequal(x: f64, y: f64) c_int { return if (cmpOrdered(x, y) and x >= y) 1 else 0; }
pub export fn isless(x: f64, y: f64) c_int { return if (cmpOrdered(x, y) and x < y) 1 else 0; }
pub export fn islessequal(x: f64, y: f64) c_int { return if (cmpOrdered(x, y) and x <= y) 1 else 0; }
pub export fn islessgreater(x: f64, y: f64) c_int { return if (cmpOrdered(x, y) and (x < y or x > y)) 1 else 0; }
pub export fn isunordered(x: f64, y: f64) c_int { return if (std.math.isNan(x) or std.math.isNan(y)) 1 else 0; }

pub export fn isgreaterf(x: f32, y: f32) c_int { return if (cmpOrdered(x, y) and x > y) 1 else 0; }
pub export fn isgreaterequalf(x: f32, y: f32) c_int { return if (cmpOrdered(x, y) and x >= y) 1 else 0; }
pub export fn islessf(x: f32, y: f32) c_int { return if (cmpOrdered(x, y) and x < y) 1 else 0; }
pub export fn islessequalf(x: f32, y: f32) c_int { return if (cmpOrdered(x, y) and x <= y) 1 else 0; }
pub export fn islessgreaterf(x: f32, y: f32) c_int { return if (cmpOrdered(x, y) and (x < y or x > y)) 1 else 0; }
pub export fn isunorderedf(x: f32, y: f32) c_int { return if (std.math.isNan(x) or std.math.isNan(y)) 1 else 0; }

pub export fn isgreaterl(x: c_longdouble, y: c_longdouble) c_int { return if (cmpOrdered(x, y) and x > y) 1 else 0; }
pub export fn isgreaterequall(x: c_longdouble, y: c_longdouble) c_int { return if (cmpOrdered(x, y) and x >= y) 1 else 0; }
pub export fn islessl(x: c_longdouble, y: c_longdouble) c_int { return if (cmpOrdered(x, y) and x < y) 1 else 0; }
pub export fn islessequall(x: c_longdouble, y: c_longdouble) c_int { return if (cmpOrdered(x, y) and x <= y) 1 else 0; }
pub export fn islessgreaterl(x: c_longdouble, y: c_longdouble) c_int { return if (cmpOrdered(x, y) and (x < y or x > y)) 1 else 0; }
pub export fn isunorderedl(x: c_longdouble, y: c_longdouble) c_int { return if (std.math.isNan(x) or std.math.isNan(y)) 1 else 0; }
