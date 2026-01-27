// Optimization and Numerical Methods module
// Pure Zig implementations for root finding, optimization, and autodiff

const std = @import("std");
const math = std.math;

// ============================================================================
// Dual Number Type for Forward-Mode Automatic Differentiation
// ============================================================================

/// Dual number for forward-mode automatic differentiation
/// Represents a value and its derivative: x + ε*dx where ε² = 0
pub const Dual = extern struct {
    val: f64, // Function value
    deriv: f64, // Derivative value

    pub fn init(val: f64, deriv: f64) Dual {
        return .{ .val = val, .deriv = deriv };
    }

    pub fn constant(val: f64) Dual {
        return .{ .val = val, .deriv = 0.0 };
    }

    pub fn variable(val: f64) Dual {
        return .{ .val = val, .deriv = 1.0 };
    }

    pub fn add(a: Dual, b: Dual) Dual {
        return .{ .val = a.val + b.val, .deriv = a.deriv + b.deriv };
    }

    pub fn sub(a: Dual, b: Dual) Dual {
        return .{ .val = a.val - b.val, .deriv = a.deriv - b.deriv };
    }

    pub fn mul(a: Dual, b: Dual) Dual {
        // (a + εa')(b + εb') = ab + ε(ab' + a'b)
        return .{ .val = a.val * b.val, .deriv = a.val * b.deriv + a.deriv * b.val };
    }

    pub fn div(a: Dual, b: Dual) Dual {
        // (a + εa')/(b + εb') = a/b + ε(a'b - ab')/b²
        const denom = b.val * b.val;
        return .{ .val = a.val / b.val, .deriv = (a.deriv * b.val - a.val * b.deriv) / denom };
    }

    pub fn sin_d(d: Dual) Dual {
        return .{ .val = math.sin(d.val), .deriv = d.deriv * math.cos(d.val) };
    }

    pub fn cos_d(d: Dual) Dual {
        return .{ .val = math.cos(d.val), .deriv = -d.deriv * math.sin(d.val) };
    }

    pub fn exp_d(d: Dual) Dual {
        const e = math.exp(d.val);
        return .{ .val = e, .deriv = d.deriv * e };
    }

    pub fn log_d(d: Dual) Dual {
        return .{ .val = @log(d.val), .deriv = d.deriv / d.val };
    }

    pub fn sqrt_d(d: Dual) Dual {
        const s = math.sqrt(d.val);
        return .{ .val = s, .deriv = d.deriv / (2.0 * s) };
    }

    pub fn pow_d(d: Dual, n: f64) Dual {
        const p = math.pow(d.val, n);
        return .{ .val = p, .deriv = d.deriv * n * math.pow(d.val, n - 1.0) };
    }
};

// ============================================================================
// Optimization Result Type
// ============================================================================

/// Result of optimization/root-finding algorithms
pub const OptimResult = extern struct {
    value: f64, // Solution or minimum point
    f_value: f64, // Function value at solution
    iterations: u32, // Number of iterations used
    converged: u8, // 1 if converged, 0 otherwise
};

// ============================================================================
// Root Finding Methods
// ============================================================================

/// Bisection method for finding root of f(x) = 0
/// Requires f(a) and f(b) to have opposite signs
pub fn bisect(
    f: *const fn (f64) callconv(.C) f64,
    a_init: f64,
    b_init: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    var a = a_init;
    var b = b_init;
    var fa = f(a);
    var fb = f(b);

    // Check for sign change
    if (fa * fb > 0) {
        return .{ .value = math.nan(f64), .f_value = math.nan(f64), .iterations = 0, .converged = 0 };
    }

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        const mid = (a + b) / 2.0;
        const fm = f(mid);

        if (@abs(fm) < tol or (b - a) / 2.0 < tol) {
            return .{ .value = mid, .f_value = fm, .iterations = iter + 1, .converged = 1 };
        }

        if (fa * fm < 0) {
            b = mid;
            fb = fm;
        } else {
            a = mid;
            fa = fm;
        }
    }

    const final = (a + b) / 2.0;
    return .{ .value = final, .f_value = f(final), .iterations = max_iter, .converged = 0 };
}

/// Newton-Raphson method for finding root of f(x) = 0
pub fn newton(
    f: *const fn (f64) callconv(.C) f64,
    df: *const fn (f64) callconv(.C) f64,
    x0: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    var x = x0;

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        const fx = f(x);
        const dfx = df(x);

        if (@abs(dfx) < 1e-15) {
            return .{ .value = x, .f_value = fx, .iterations = iter + 1, .converged = 0 };
        }

        const x_new = x - fx / dfx;

        if (@abs(x_new - x) < tol) {
            return .{ .value = x_new, .f_value = f(x_new), .iterations = iter + 1, .converged = 1 };
        }
        x = x_new;
    }

    return .{ .value = x, .f_value = f(x), .iterations = max_iter, .converged = 0 };
}

/// Secant method for finding root of f(x) = 0
pub fn secant(
    f: *const fn (f64) callconv(.C) f64,
    x0: f64,
    x1: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    var x_prev = x0;
    var x_curr = x1;
    var f_prev = f(x_prev);

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        const f_curr = f(x_curr);

        const denom = f_curr - f_prev;
        if (@abs(denom) < 1e-15) {
            return .{ .value = x_curr, .f_value = f_curr, .iterations = iter + 1, .converged = 0 };
        }

        const x_new = x_curr - f_curr * (x_curr - x_prev) / denom;

        if (@abs(x_new - x_curr) < tol) {
            return .{ .value = x_new, .f_value = f(x_new), .iterations = iter + 1, .converged = 1 };
        }

        x_prev = x_curr;
        f_prev = f_curr;
        x_curr = x_new;
    }

    return .{ .value = x_curr, .f_value = f(x_curr), .iterations = max_iter, .converged = 0 };
}

/// Brent's method for finding root of f(x) = 0
/// Combines bisection, secant, and inverse quadratic interpolation
pub fn brent(
    f: *const fn (f64) callconv(.C) f64,
    a_init: f64,
    b_init: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    var a = a_init;
    var b = b_init;
    var fa = f(a);
    var fb = f(b);

    // Ensure f(a) and f(b) have opposite signs
    if (fa * fb > 0) {
        return .{ .value = math.nan(f64), .f_value = math.nan(f64), .iterations = 0, .converged = 0 };
    }

    // Ensure |f(a)| >= |f(b)|
    if (@abs(fa) < @abs(fb)) {
        const tmp = a;
        a = b;
        b = tmp;
        const ftmp = fa;
        fa = fb;
        fb = ftmp;
    }

    var c = a;
    var fc = fa;
    var mflag = true;
    var d: f64 = 0;

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        if (@abs(fb) < tol) {
            return .{ .value = b, .f_value = fb, .iterations = iter + 1, .converged = 1 };
        }

        var s: f64 = undefined;

        if (fa != fc and fb != fc) {
            // Inverse quadratic interpolation
            s = a * fb * fc / ((fa - fb) * (fa - fc)) +
                b * fa * fc / ((fb - fa) * (fb - fc)) +
                c * fa * fb / ((fc - fa) * (fc - fb));
        } else {
            // Secant method
            s = b - fb * (b - a) / (fb - fa);
        }

        // Conditions for bisection
        const cond1 = !((s > (3 * a + b) / 4 and s < b) or (s < (3 * a + b) / 4 and s > b));
        const cond2 = mflag and @abs(s - b) >= @abs(b - c) / 2;
        const cond3 = !mflag and @abs(s - b) >= @abs(c - d) / 2;
        const cond4 = mflag and @abs(b - c) < tol;
        const cond5 = !mflag and @abs(c - d) < tol;

        if (cond1 or cond2 or cond3 or cond4 or cond5) {
            s = (a + b) / 2;
            mflag = true;
        } else {
            mflag = false;
        }

        const fs = f(s);
        d = c;
        c = b;
        fc = fb;

        if (fa * fs < 0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }

        if (@abs(fa) < @abs(fb)) {
            const tmp = a;
            a = b;
            b = tmp;
            const ftmp = fa;
            fa = fb;
            fb = ftmp;
        }

        if (@abs(b - a) < tol) {
            return .{ .value = b, .f_value = fb, .iterations = iter + 1, .converged = 1 };
        }
    }

    return .{ .value = b, .f_value = fb, .iterations = max_iter, .converged = 0 };
}

// ============================================================================
// 1D Optimization Methods
// ============================================================================

/// Golden section search for finding minimum of unimodal function
pub fn golden_section_min(
    f: *const fn (f64) callconv(.C) f64,
    a_init: f64,
    b_init: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    const phi: f64 = (1.0 + math.sqrt(5.0)) / 2.0; // Golden ratio
    const resphi = 2.0 - phi;

    var a = a_init;
    var b = b_init;
    var x1 = a + resphi * (b - a);
    var x2 = b - resphi * (b - a);
    var f1 = f(x1);
    var f2 = f(x2);

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        if (@abs(b - a) < tol) {
            const x_min = (a + b) / 2.0;
            return .{ .value = x_min, .f_value = f(x_min), .iterations = iter + 1, .converged = 1 };
        }

        if (f1 < f2) {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = a + resphi * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = b - resphi * (b - a);
            f2 = f(x2);
        }
    }

    const x_min = (a + b) / 2.0;
    return .{ .value = x_min, .f_value = f(x_min), .iterations = max_iter, .converged = 0 };
}

/// 1D gradient descent for minimization
pub fn gradient_descent_1d(
    f: *const fn (f64) callconv(.C) f64,
    df: *const fn (f64) callconv(.C) f64,
    x0: f64,
    learning_rate: f64,
    tol: f64,
    max_iter: u32,
) OptimResult {
    var x = x0;

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        const grad = df(x);

        if (@abs(grad) < tol) {
            return .{ .value = x, .f_value = f(x), .iterations = iter + 1, .converged = 1 };
        }

        const x_new = x - learning_rate * grad;

        if (@abs(x_new - x) < tol) {
            return .{ .value = x_new, .f_value = f(x_new), .iterations = iter + 1, .converged = 1 };
        }

        x = x_new;
    }

    return .{ .value = x, .f_value = f(x), .iterations = max_iter, .converged = 0 };
}

// ============================================================================
// Multi-dimensional Optimization Methods
// ============================================================================

/// Result for multi-dimensional optimization
pub const MultiDimResult = extern struct {
    iterations: u32,
    converged: u8,
};

/// Multi-dimensional gradient descent
/// grad_fn: function that computes gradient into output array
/// x: mutable array of current position (modified in place)
/// n: dimension of the problem
pub fn gradient_descent(
    grad_fn: *const fn ([*]f64, [*]const f64, usize) callconv(.C) void,
    x: [*]f64,
    n: usize,
    learning_rate: f64,
    tol: f64,
    max_iter: u32,
) MultiDimResult {
    // Stack buffer for small dimensions, otherwise we can't allocate
    var grad_buf: [64]f64 = undefined;
    const grad: [*]f64 = if (n <= 64) &grad_buf else return .{ .iterations = 0, .converged = 0 };

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        grad_fn(grad, x, n);

        // Compute gradient norm
        var grad_norm: f64 = 0.0;
        for (0..n) |i| {
            grad_norm += grad[i] * grad[i];
        }
        grad_norm = math.sqrt(grad_norm);

        if (grad_norm < tol) {
            return .{ .iterations = iter + 1, .converged = 1 };
        }

        // Update x
        var max_change: f64 = 0.0;
        for (0..n) |i| {
            const delta = learning_rate * grad[i];
            x[i] -= delta;
            if (@abs(delta) > max_change) max_change = @abs(delta);
        }

        if (max_change < tol) {
            return .{ .iterations = iter + 1, .converged = 1 };
        }
    }

    return .{ .iterations = max_iter, .converged = 0 };
}

/// Adam optimizer for multi-dimensional optimization
/// grad_fn: function that computes gradient into output array
/// x: mutable array of current position (modified in place)
/// n: dimension of the problem
pub fn adam(
    grad_fn: *const fn ([*]f64, [*]const f64, usize) callconv(.C) void,
    x: [*]f64,
    n: usize,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    max_iter: u32,
) MultiDimResult {
    // Stack buffers for small dimensions
    var grad_buf: [64]f64 = undefined;
    var m_buf: [64]f64 = undefined; // First moment
    var v_buf: [64]f64 = undefined; // Second moment

    if (n > 64) return .{ .iterations = 0, .converged = 0 };

    const grad: [*]f64 = &grad_buf;
    const m: [*]f64 = &m_buf;
    const v: [*]f64 = &v_buf;

    // Initialize moments to zero
    for (0..n) |i| {
        m[i] = 0.0;
        v[i] = 0.0;
    }

    var iter: u32 = 0;
    while (iter < max_iter) : (iter += 1) {
        grad_fn(grad, x, n);

        const t: f64 = @floatFromInt(iter + 1);

        // Bias correction terms
        const bc1 = 1.0 - math.pow(beta1, t);
        const bc2 = 1.0 - math.pow(beta2, t);

        var max_change: f64 = 0.0;

        for (0..n) |i| {
            // Update biased first moment estimate
            m[i] = beta1 * m[i] + (1.0 - beta1) * grad[i];
            // Update biased second raw moment estimate
            v[i] = beta2 * v[i] + (1.0 - beta2) * grad[i] * grad[i];

            // Compute bias-corrected estimates
            const m_hat = m[i] / bc1;
            const v_hat = v[i] / bc2;

            // Update parameters
            const delta = learning_rate * m_hat / (math.sqrt(v_hat) + epsilon);
            x[i] -= delta;
            if (@abs(delta) > max_change) max_change = @abs(delta);
        }

        if (max_change < epsilon) {
            return .{ .iterations = iter + 1, .converged = 1 };
        }
    }

    return .{ .iterations = max_iter, .converged = 0 };
}

// ============================================================================
// C Export Functions
// ============================================================================

/// C function pointer type for scalar functions
const CScalarFn = *const fn (f64) callconv(.C) f64;

/// C function pointer type for gradient functions (multi-dim)
const CGradFn = *const fn ([*]f64, [*]const f64, usize) callconv(.C) void;

// --- Dual number C exports ---

pub fn dual_new(val: f64, deriv: f64) callconv(.C) Dual {
    return Dual.init(val, deriv);
}

pub fn dual_constant(val: f64) callconv(.C) Dual {
    return Dual.constant(val);
}

pub fn dual_variable(val: f64) callconv(.C) Dual {
    return Dual.variable(val);
}

pub fn dual_add(a: Dual, b: Dual) callconv(.C) Dual {
    return Dual.add(a, b);
}

pub fn dual_sub(a: Dual, b: Dual) callconv(.C) Dual {
    return Dual.sub(a, b);
}

pub fn dual_mul(a: Dual, b: Dual) callconv(.C) Dual {
    return Dual.mul(a, b);
}

pub fn dual_div(a: Dual, b: Dual) callconv(.C) Dual {
    return Dual.div(a, b);
}

pub fn dual_sin(d: Dual) callconv(.C) Dual {
    return Dual.sin_d(d);
}

pub fn dual_cos(d: Dual) callconv(.C) Dual {
    return Dual.cos_d(d);
}

pub fn dual_exp(d: Dual) callconv(.C) Dual {
    return Dual.exp_d(d);
}

pub fn dual_log(d: Dual) callconv(.C) Dual {
    return Dual.log_d(d);
}

pub fn dual_sqrt(d: Dual) callconv(.C) Dual {
    return Dual.sqrt_d(d);
}

pub fn dual_pow(d: Dual, n: f64) callconv(.C) Dual {
    return Dual.pow_d(d, n);
}

// --- Root finding C exports ---

pub fn optim_bisect(f: CScalarFn, a: f64, b: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return bisect(f, a, b, tol, max_iter);
}

pub fn optim_newton(f: CScalarFn, df: CScalarFn, x0: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return newton(f, df, x0, tol, max_iter);
}

pub fn optim_secant(f: CScalarFn, x0: f64, x1: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return secant(f, x0, x1, tol, max_iter);
}

pub fn optim_brent(f: CScalarFn, a: f64, b: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return brent(f, a, b, tol, max_iter);
}

// --- 1D optimization C exports ---

pub fn optim_golden_section(f: CScalarFn, a: f64, b: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return golden_section_min(f, a, b, tol, max_iter);
}

pub fn optim_gd_1d(f: CScalarFn, df: CScalarFn, x0: f64, lr: f64, tol: f64, max_iter: u32) callconv(.C) OptimResult {
    return gradient_descent_1d(f, df, x0, lr, tol, max_iter);
}

// --- Multi-dimensional optimization C exports ---

pub fn optim_gradient_descent(
    grad_fn: CGradFn,
    x: [*]f64,
    n: usize,
    lr: f64,
    tol: f64,
    max_iter: u32,
) callconv(.C) MultiDimResult {
    return gradient_descent(grad_fn, x, n, lr, tol, max_iter);
}

pub fn optim_adam(
    grad_fn: CGradFn,
    x: [*]f64,
    n: usize,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    max_iter: u32,
) callconv(.C) MultiDimResult {
    return adam(grad_fn, x, n, lr, beta1, beta2, eps, max_iter);
}
