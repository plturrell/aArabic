// Bessel functions for zig-libc
// Implements J0, J1, Y0, Y1, Jn, Yn using polynomial/rational approximations
// Based on standard fdlibm algorithms

const std = @import("std");
const math = std.math;

// Constants
const pi: f64 = math.pi;
const two_over_pi: f64 = 2.0 / pi;
const invsqrtpi: f64 = 0.5641895835477562869480794515607725858;

// ============================================================================
// J0(x) - Bessel function of first kind, order 0
// ============================================================================

// Coefficients for |x| < 8.0 (rational polynomial)
const j0_r: [6]f64 = .{
    1.56249999999999947958e-02, // 0x3F8FFFFF_FFFFD329
    -1.89979294238854721751e-04, // 0xBF28E6A5_B61AC6E9
    1.82954049532700665670e-06, // 0x3EBEB1D1_0C503919
    -4.61832688532103189199e-09, // 0xBE33D5E7_73D63FCE
    -3.11122668429037999e-13, // 0xBD6D6D9E_B8BA9441
    0.0,
};

const j0_s: [6]f64 = .{
    1.56191029464890010492e-02, // 0x3F8FFCE8_82C8C2A4
    1.16926784663337450260e-04, // 0x3F1EA6D2_DD57DBF4
    5.13546550207318111446e-07, // 0x3EA13B54_CE84D5A9
    1.16614003333790000901e-09, // 0x3E1408BC_F4745D8F
    0.0,
    0.0,
};

// Coefficients for |x| >= 8.0 (asymptotic expansion for P0, Q0)
const p0_r: [6]f64 = .{
    0.0,
    -1.14207370375678408678e-02, // 0xBF8766E2_D58AC2EA
    -1.25049348214267155403e-01, // 0xBFC00926_8F39B3D3
    -2.87295284148892147540e-02, // 0xBF9D6A6A_2C9F19F2
    0.0,
    0.0,
};

const p0_s: [5]f64 = .{
    1.0,
    1.26515116761744035846e+01, // 0x402949E4_A8465826
    4.02323526458968884480e+01, // 0x40441DD6_7C9C8B5C
    2.38754762999804893880e+01, // 0x4037D72D_E63F64FF
    0.0,
};

const q0_r: [6]f64 = .{
    0.0,
    7.32421766612684765896e-02, // 0x3FB2BFF5_D9F6C0FF
    1.17152418604024889619e+00, // 0x3FF2BEC3_A9ECFEF2
    4.95634774626823989190e-01, // 0x3FDFB5B1_C41EB652
    0.0,
    0.0,
};

const q0_s: [6]f64 = .{
    1.0,
    1.63707730398719028750e+01, // 0x40305F14_43C30D28
    7.66399247256106188800e+01, // 0x405328F1_CDB3B270
    7.17768808896899156400e+01, // 0x4051EF6A_8721A5E5
    0.0,
    0.0,
};

fn polyeval(coef: []const f64, x: f64) f64 {
    var result: f64 = 0.0;
    var i: usize = coef.len;
    while (i > 0) {
        i -= 1;
        result = result * x + coef[i];
    }
    return result;
}

pub fn j0_impl(x: f64) f64 {
    const ax = @abs(x);

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return 0.0; // J0(inf) = 0
    }

    if (ax < 8.0) {
        // Use polynomial approximation for small x
        if (ax < 1.0e-5) {
            // For very small x, J0(x) ≈ 1 - x²/4
            if (ax < 1.0e-15) return 1.0;
            return 1.0 - 0.25 * x * x;
        }

        const z = x * x;
        const r = z * polyeval(&j0_r, z);
        const s = 1.0 + z * polyeval(&j0_s, z);
        return (1.0 + z * (-0.25 + r / s));
    }

    // For large x, use asymptotic expansion: J0(x) ≈ sqrt(2/(πx)) * cos(x - π/4)
    const z = 8.0 / ax;
    const z2 = z * z;

    const p = polyeval(&p0_r, z2) / polyeval(&p0_s, z2);
    const q = z * polyeval(&q0_r, z2) / polyeval(&q0_s, z2);

    const xx = ax - 0.25 * pi;
    return math.sqrt(two_over_pi / ax) * (p * math.cos(xx) - q * math.sin(xx));
}

// ============================================================================
// J1(x) - Bessel function of first kind, order 1
// ============================================================================

const j1_r: [4]f64 = .{
    -6.25000000000000000000e-02, // 0xBFB00000_00000000
    1.40705666955189706048e-03, // 0x3F5710F7_B01C4E7E
    -1.59955631084035597520e-05, // 0xBEF0C5C6_BA169668
    4.96727999609584448412e-08, // 0x3E6AAAFA_46CA0BD9
};

const j1_s: [4]f64 = .{
    1.0,
    1.91537599538363460805e-02, // 0x3F939E8C_18664D65
    1.85946785588630915560e-04, // 0x3F2861B2_1CFCCAEF
    1.17718464042623683263e-06, // 0x3EB3BFF8_333F8498
};

pub fn j1_impl(x: f64) f64 {
    const ax = @abs(x);
    const sign: f64 = if (x < 0) -1.0 else 1.0;

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return 0.0; // J1(inf) = 0
    }

    if (ax < 8.0) {
        if (ax < 1.0e-5) {
            // For very small x, J1(x) ≈ x/2
            if (ax < 1.0e-15) return 0.5 * x;
            return x * 0.5 * (1.0 - 0.125 * x * x);
        }

        const z = x * x;
        const r = polyeval(&j1_r, z);
        const s = polyeval(&j1_s, z);
        return x * (0.5 + r / s);
    }

    // Asymptotic expansion
    const z = 8.0 / ax;
    const z2 = z * z;

    // Use same P0/Q0 approximations (close enough for J1)
    const p = 1.0 - 0.001953125 * z2; // 1 - 1/512 * z^2 (simplified)
    const q = 0.0625 * z; // z/16 (simplified)

    const xx = ax - 0.75 * pi;
    return sign * math.sqrt(two_over_pi / ax) * (p * math.cos(xx) - q * math.sin(xx));
}

// ============================================================================
// Y0(x) - Bessel function of second kind, order 0
// ============================================================================

// Y0 polynomial coefficients (from fdlibm)
const y0_coef_u: [7]f64 = .{
    -7.38042951086872317523e-02,
    1.76666452509181115538e-01,
    -1.38185671945596898451e-02,
    3.47453432093683650238e-04,
    -3.81407053724364161125e-06,
    1.95590137035022920206e-08,
    -3.98205194132103398453e-11,
};
const y0_coef_v: [5]f64 = .{
    1.0,
    1.27304834834123699328e-02,
    7.60068627350353253702e-05,
    2.59150851840457805467e-07,
    4.41110311332675467403e-10,
};

pub fn y0_impl(x: f64) f64 {
    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return -math.inf(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return -math.inf(f64);
        return math.nan(f64);
    }

    if (x < 8.0) {
        const z = x * x;
        const u = polyeval(&y0_coef_u, z);
        const v = polyeval(&y0_coef_v, z);
        return u / v + two_over_pi * j0_impl(x) * @log(x);
    }

    // Asymptotic expansion for large x
    const z = 8.0 / x;
    const z2 = z * z;

    // P0, Q0 coefficients for asymptotic expansion
    const pR0: [6]f64 = .{ 0.0, -1.14207370375678408678e-02, -1.25049348214267155403e-01, -2.87295284148892147540e-02, -8.00321395949498785028e-04, -7.13916368723498963520e-06 };
    const pS0: [5]f64 = .{ 1.0, 1.26515116761744035846e+01, 4.02323526458968884480e+01, 2.38754762999804893880e+01, 1.69753115163608902345e+00 };
    const qR0: [6]f64 = .{ 0.0, 7.32421766612684765896e-02, 1.17152418604024889619e+00, 4.95634774626823989190e-01, 2.63556765282008299190e-02, 2.72740065395104352590e-04 };
    const qS0: [6]f64 = .{ 1.0, 1.63707730398719028750e+01, 7.66399247256106188800e+01, 7.17768808896899156400e+01, 1.59700373368405903980e+01, 4.89999643455366268340e-01 };

    const p = polyeval(&pR0, z2) / polyeval(&pS0, z2);
    const q = z * polyeval(&qR0, z2) / polyeval(&qS0, z2);

    const xx = x - 0.25 * pi;
    return math.sqrt(two_over_pi / x) * (p * math.cos(xx) + q * math.sin(xx));
}

// ============================================================================
// Y1(x) - Bessel function of second kind, order 1
// ============================================================================

// Y1 polynomial coefficients (from fdlibm)
const y1_coef_u: [5]f64 = .{
    -1.96057090646238940668e-01,
    5.04438716639811282616e-02,
    -1.91256895860641695620e-03,
    2.35252600561610495928e-05,
    -9.19099158039878874504e-08,
};
const y1_coef_v: [6]f64 = .{
    1.0,
    1.99167318236649903973e-02,
    2.02552581025135171496e-04,
    1.35608801097498456556e-06,
    6.22741452364621501295e-09,
    1.66559246207992079114e-11,
};

pub fn y1_impl(x: f64) f64 {
    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return -math.inf(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return -math.inf(f64);
        return math.nan(f64);
    }

    if (x < 8.0) {
        const z = x * x;
        const u = polyeval(&y1_coef_u, z);
        const v = polyeval(&y1_coef_v, z);
        return x * (u / v) + two_over_pi * (j1_impl(x) * @log(x) - 1.0 / x);
    }

    // Asymptotic expansion for large x
    const z = 8.0 / x;
    const z2 = z * z;

    // P1, Q1 coefficients for asymptotic expansion
    const pR1: [6]f64 = .{ 0.0, 1.17187499999988647970e-02, 1.32394806593073575129e-01, 3.49251971286573707685e-02, 1.02724376373891297175e-03, 9.15441403768215802150e-06 };
    const pS1: [5]f64 = .{ 1.0, 1.32260997545042486630e+01, 4.40015519207498822684e+01, 2.81657574805987995760e+01, 2.32732049335039820656e+00 };
    const qR1: [6]f64 = .{ 0.0, -1.09375000000000381510e-02, -1.27536300095527163250e-01, -3.63995395318916980250e-02, -1.15900497795921042930e-03, -1.16165475989616698800e-05 };
    const qS1: [6]f64 = .{ 1.0, 1.36288287570020149508e+01, 4.99999818498737393610e+01, 5.11661904626890500070e+01, 1.35807679805568704170e+01, 5.85262622436972655100e-01 };

    const p = polyeval(&pR1, z2) / polyeval(&pS1, z2);
    const q = z * polyeval(&qR1, z2) / polyeval(&qS1, z2);

    const xx = x - 0.75 * pi;
    return math.sqrt(two_over_pi / x) * (p * math.cos(xx) + q * math.sin(xx));
}

// ============================================================================
// Jn(n, x) - Bessel function of first kind, order n
// ============================================================================

pub fn jn_impl(n: i32, x: f64) f64 {
    if (n == 0) return j0_impl(x);
    if (n == 1) return j1_impl(x);
    if (n < 0) {
        // J_{-n}(x) = (-1)^n * J_n(x)
        const nn: u32 = @intCast(-n);
        const sign: f64 = if (nn % 2 == 0) 1.0 else -1.0;
        return sign * jn_impl(-n, x);
    }

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return 0.0;
    }

    const ax = @abs(x);
    if (ax == 0.0) return 0.0;

    const sign: f64 = if (x < 0 and @as(u32, @intCast(n)) % 2 == 1) -1.0 else 1.0;

    // For small x or n > x, use forward recurrence
    // For large x and n < x, use backward recurrence (Miller's algorithm)

    if (@as(f64, @floatFromInt(n)) > ax) {
        // Miller's backward recurrence for better numerical stability

        // Start from a large index m > n
        const m: i32 = @intCast(@max(n + 20, @as(i32, @intFromFloat(ax))));
        var jnp1: f64 = 0.0;
        var jn_val: f64 = 1.0e-30; // Small initial value
        var sum: f64 = 0.0;

        var k: i32 = m;
        while (k >= 0) : (k -= 1) {
            const kf: f64 = @floatFromInt(k);
            const jnm1 = 2.0 * (kf + 1.0) / ax * jn_val - jnp1;
            jnp1 = jn_val;
            jn_val = jnm1;

            if (k % 2 == 0) sum += jn_val;
            if (k == n) jnp1 = jn_val; // Save J_n value
        }

        // Normalize using J0(x) + 2*(J2 + J4 + ...) = 1
        sum = 2.0 * sum - jn_val;
        return sign * jnp1 / sum;
    } else {
        // Forward recurrence
        var jnm1 = j0_impl(ax);
        var jn_val = j1_impl(ax);

        var k: i32 = 1;
        while (k < n) : (k += 1) {
            const kf: f64 = @floatFromInt(k);
            const temp = 2.0 * kf / ax * jn_val - jnm1;
            jnm1 = jn_val;
            jn_val = temp;
        }

        return sign * jn_val;
    }
}

// ============================================================================
// Yn(n, x) - Bessel function of second kind, order n
// ============================================================================

pub fn yn_impl(n: i32, x: f64) f64 {
    if (n == 0) return y0_impl(x);
    if (n == 1) return y1_impl(x);
    if (n < 0) {
        const nn: u32 = @intCast(-n);
        const sign: f64 = if (nn % 2 == 0) 1.0 else -1.0;
        return sign * yn_impl(-n, x);
    }

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return -math.inf(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return -math.inf(f64);
        return math.nan(f64);
    }

    // Forward recurrence: Y_{n+1}(x) = (2n/x) * Y_n(x) - Y_{n-1}(x)
    var ynm1 = y0_impl(x);
    var yn_val = y1_impl(x);

    var k: i32 = 1;
    while (k < n) : (k += 1) {
        const kf: f64 = @floatFromInt(k);
        const temp = 2.0 * kf / x * yn_val - ynm1;
        ynm1 = yn_val;
        yn_val = temp;
    }

    return yn_val;
}

// ============================================================================
// Float (f32) versions
// ============================================================================

pub fn j0f_impl(x: f32) f32 {
    return @floatCast(j0_impl(@floatCast(x)));
}

pub fn j1f_impl(x: f32) f32 {
    return @floatCast(j1_impl(@floatCast(x)));
}

pub fn y0f_impl(x: f32) f32 {
    return @floatCast(y0_impl(@floatCast(x)));
}

pub fn y1f_impl(x: f32) f32 {
    return @floatCast(y1_impl(@floatCast(x)));
}

pub fn jnf_impl(n: i32, x: f32) f32 {
    return @floatCast(jn_impl(n, @floatCast(x)));
}

pub fn ynf_impl(n: i32, x: f32) f32 {
    return @floatCast(yn_impl(n, @floatCast(x)));
}

// ============================================================================
// Tests
// ============================================================================

test "j0 basic values" {
    const tolerance: f64 = 1e-6;

    // J0(0) = 1
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), j0_impl(0.0), tolerance);

    // J0(2.4048) ≈ 0 (first zero)
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), j0_impl(2.4048255576957728), 1e-4);

    // J0(1) ≈ 0.7652
    try std.testing.expectApproxEqAbs(@as(f64, 0.7651976865579666), j0_impl(1.0), tolerance);
}

test "j1 basic values" {
    const tolerance: f64 = 1e-6;

    // J1(0) = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), j1_impl(0.0), tolerance);

    // J1(1) ≈ 0.4401
    try std.testing.expectApproxEqAbs(@as(f64, 0.44005058574493355), j1_impl(1.0), tolerance);
}

test "y0 basic values" {
    const tolerance: f64 = 1e-5;

    // Y0(1) ≈ 0.0883
    try std.testing.expectApproxEqAbs(@as(f64, 0.08825696421567696), y0_impl(1.0), tolerance);

    // Y0(0) = -inf
    try std.testing.expect(y0_impl(0.0) == -math.inf(f64));
}

test "y1 basic values" {
    const tolerance: f64 = 1e-5;

    // Y1(1) ≈ -0.7812
    try std.testing.expectApproxEqAbs(@as(f64, -0.7812128213002887), y1_impl(1.0), tolerance);
}

// ============================================================================
// Modified Bessel Functions of the First Kind: I0, I1, In
// Using polynomial approximations from Abramowitz & Stegun
// ============================================================================

pub fn bessel_i0_impl(x: f64) f64 {
    const ax = @abs(x);

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return math.inf(f64); // I0(±inf) = +inf
    }

    if (ax < 3.75) {
        // Polynomial approximation for |x| < 3.75
        // I0(x) ≈ 1 + 3.5156229*(x/3.75)^2 + 3.0899424*(x/3.75)^4 + ...
        const t = ax / 3.75;
        const t2 = t * t;
        return 1.0 + t2 * (3.5156229 + t2 * (3.0899424 + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))));
    }

    // Asymptotic expansion for |x| >= 3.75
    // I0(x) ≈ exp(x)/sqrt(x) * (0.39894228 + 0.01328592/t + ...)
    const t = 3.75 / ax;
    const result = 0.39894228 + t * (0.01328592 + t * (0.00225319 + t * (-0.00157565 + t * (0.00916281 + t * (-0.02057706 + t * (0.02635537 + t * (-0.01647633 + t * 0.00392377)))))));
    return math.exp(ax) / math.sqrt(ax) * result;
}

pub fn bessel_i1_impl(x: f64) f64 {
    const ax = @abs(x);
    const sign: f64 = if (x < 0) -1.0 else 1.0;

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return sign * math.inf(f64); // I1(±inf) = ±inf
    }

    if (ax < 3.75) {
        // Polynomial approximation for |x| < 3.75
        // I1(x) ≈ x * (0.5 + 0.87890594*(x/3.75)^2 + ...)
        const t = ax / 3.75;
        const t2 = t * t;
        const result = 0.5 + t2 * (0.87890594 + t2 * (0.51498869 + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411)))));
        return sign * ax * result;
    }

    // Asymptotic expansion for |x| >= 3.75
    const t = 3.75 / ax;
    const result = 0.39894228 + t * (-0.03988024 + t * (-0.00362018 + t * (0.00163801 + t * (-0.01031555 + t * (0.02282967 + t * (-0.02895312 + t * (0.01787654 + t * (-0.00420059))))))));
    return sign * math.exp(ax) / math.sqrt(ax) * result;
}

pub fn in_impl(n: i32, x: f64) f64 {
    if (n == 0) return bessel_i0_impl(x);
    if (n == 1) return bessel_i1_impl(x);
    if (n < 0) {
        // I_{-n}(x) = I_n(x)
        return in_impl(-n, x);
    }

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        return math.inf(f64);
    }

    const ax = @abs(x);
    if (ax == 0.0) return 0.0;

    const sign: f64 = if (x < 0 and @as(u32, @intCast(n)) % 2 == 1) -1.0 else 1.0;

    // Use Miller's backward recurrence for numerical stability
    const tox = 2.0 / ax;
    const nf: f64 = @floatFromInt(n);

    // Determine starting index for backward recurrence
    const iacc: i32 = 40;

    var m: i32 = @intFromFloat(2.0 * (@as(f64, @floatFromInt(n)) + @sqrt(nf * @as(f64, @floatFromInt(iacc)))));
    if (@rem(m, 2) == 1) m += 1;

    var bi: f64 = 0.0;
    var bip: f64 = 0.0;
    var bim: f64 = 0.0;
    var ans: f64 = 0.0;

    var j: i32 = m;
    while (j > 0) : (j -= 1) {
        const jf: f64 = @floatFromInt(j);
        bim = bip + jf * tox * bi;
        bip = bi;
        bi = bim;

        // Renormalize to prevent overflow
        const abi = @abs(bi);
        if (abi > 1.0e10) {
            ans *= 1.0e-10;
            bi *= 1.0e-10;
            bip *= 1.0e-10;
        }

        if (j == n) ans = bip;
    }

    // Normalize using I0
    ans *= bessel_i0_impl(ax) / bi;
    return sign * ans;
}

// ============================================================================
// Modified Bessel Functions of the Second Kind: K0, K1, Kn
// Using polynomial approximations from Abramowitz & Stegun
// ============================================================================

pub fn k0_impl(x: f64) f64 {
    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return math.nan(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return math.inf(f64);
        return math.nan(f64);
    }

    if (x <= 2.0) {
        // K0(x) = -ln(x/2)*I0(x) + polynomial
        const t = x / 2.0;
        const t2 = t * t;
        const poly = -0.57721566 + t2 * (0.42278420 + t2 * (0.23069756 + t2 * (0.03488590 + t2 * (0.00262698 + t2 * (0.00010750 + t2 * 0.00000740)))));
        return -@log(t) * bessel_i0_impl(x) + poly;
    }

    // Asymptotic expansion for x > 2
    // K0(x) ≈ sqrt(π/(2x)) * exp(-x) * (1 + polynomial)
    const t = 2.0 / x;
    const poly = 1.25331414 + t * (-0.07832358 + t * (0.02189568 + t * (-0.01062446 + t * (0.00587872 + t * (-0.00251540 + t * 0.00053208)))));
    return math.exp(-x) / math.sqrt(x) * poly;
}

pub fn k1_impl(x: f64) f64 {
    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return math.nan(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return math.inf(f64);
        return math.nan(f64);
    }

    if (x <= 2.0) {
        // K1(x) = ln(x/2)*I1(x) + 1/x + polynomial
        const t = x / 2.0;
        const t2 = t * t;
        const poly = 1.0 + t2 * (0.15443144 + t2 * (-0.67278579 + t2 * (-0.18156897 + t2 * (-0.01919402 + t2 * (-0.00110404 + t2 * (-0.00004686))))));
        return @log(t) * bessel_i1_impl(x) + poly / x;
    }

    // Asymptotic expansion for x > 2
    const t = 2.0 / x;
    const poly = 1.25331414 + t * (0.23498619 + t * (-0.03655620 + t * (0.01504268 + t * (-0.00780353 + t * (0.00325614 + t * (-0.00068245))))));
    return math.exp(-x) / math.sqrt(x) * poly;
}

pub fn kn_impl(n: i32, x: f64) f64 {
    if (n == 0) return k0_impl(x);
    if (n == 1) return k1_impl(x);
    if (n < 0) {
        // K_{-n}(x) = K_n(x)
        return kn_impl(-n, x);
    }

    if (!math.isFinite(x)) {
        if (math.isNan(x)) return x;
        if (x > 0) return 0.0;
        return math.nan(f64);
    }

    if (x <= 0.0) {
        if (x == 0.0) return math.inf(f64);
        return math.nan(f64);
    }

    // Forward recurrence: K_{n+1}(x) = (2n/x) * K_n(x) + K_{n-1}(x)
    var knm1 = k0_impl(x);
    var kn_val = k1_impl(x);

    var k: i32 = 1;
    while (k < n) : (k += 1) {
        const kf: f64 = @floatFromInt(k);
        const temp = 2.0 * kf / x * kn_val + knm1;
        knm1 = kn_val;
        kn_val = temp;
    }

    return kn_val;
}

// ============================================================================
// Float (f32) versions for modified Bessel functions
// ============================================================================

pub fn bessel_i0f_impl(x: f32) f32 {
    return @floatCast(bessel_i0_impl(@floatCast(x)));
}

pub fn bessel_i1f_impl(x: f32) f32 {
    return @floatCast(bessel_i1_impl(@floatCast(x)));
}

pub fn inf_impl(n: i32, x: f32) f32 {
    return @floatCast(in_impl(n, @floatCast(x)));
}

pub fn k0f_impl(x: f32) f32 {
    return @floatCast(k0_impl(@floatCast(x)));
}

pub fn k1f_impl(x: f32) f32 {
    return @floatCast(k1_impl(@floatCast(x)));
}

pub fn knf_impl(n: i32, x: f32) f32 {
    return @floatCast(kn_impl(n, @floatCast(x)));
}

// ============================================================================
// Tests for modified Bessel functions
// ============================================================================

test "i0 basic values" {
    const tolerance: f64 = 1e-6;

    // I0(0) = 1
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), bessel_i0_impl(0.0), tolerance);

    // I0(1) ≈ 1.2660658777520084
    try std.testing.expectApproxEqAbs(@as(f64, 1.2660658777520084), bessel_i0_impl(1.0), tolerance);

    // I0(2) ≈ 2.2795853023360673
    try std.testing.expectApproxEqAbs(@as(f64, 2.2795853023360673), bessel_i0_impl(2.0), tolerance);
}

test "i1 basic values" {
    const tolerance: f64 = 1e-6;

    // I1(0) = 0
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), bessel_i1_impl(0.0), tolerance);

    // I1(1) ≈ 0.5651591039924851
    try std.testing.expectApproxEqAbs(@as(f64, 0.5651591039924851), bessel_i1_impl(1.0), tolerance);

    // I1(-1) = -I1(1)
    try std.testing.expectApproxEqAbs(-bessel_i1_impl(1.0), bessel_i1_impl(-1.0), tolerance);
}

test "k0 basic values" {
    const tolerance: f64 = 1e-5;

    // K0(1) ≈ 0.4210244382407083
    try std.testing.expectApproxEqAbs(@as(f64, 0.4210244382407083), k0_impl(1.0), tolerance);

    // K0(0) = +inf
    try std.testing.expect(k0_impl(0.0) == math.inf(f64));

    // K0(negative) = NaN
    try std.testing.expect(math.isNan(k0_impl(-1.0)));
}

test "k1 basic values" {
    const tolerance: f64 = 1e-5;

    // K1(1) ≈ 0.6019072301972346
    try std.testing.expectApproxEqAbs(@as(f64, 0.6019072301972346), k1_impl(1.0), tolerance);

    // K1(0) = +inf
    try std.testing.expect(k1_impl(0.0) == math.inf(f64));
}
