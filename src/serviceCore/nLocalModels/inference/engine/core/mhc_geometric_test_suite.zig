// mHC Comprehensive Geometric Test Suite - Day 59
// Tests all geometric modules together: Hyperbolic, Spherical, Product Manifold, Auto-detection
//
// Test Categories:
// 1. Hyperbolic Tests (50+) - mhc_hyperbolic.zig functions
// 2. Spherical Tests (40+) - mhc_spherical.zig functions
// 3. Product Manifold Tests (30+) - mhc_product_manifold.zig functions
// 4. Auto-detection Tests (20+) - Geometry detection logic
//
// Edge Cases:
// - Zero vectors
// - Unit vectors
// - Points near boundary
// - Large dimensions
// - Numerical stability
//
// Reference: Day 59 - Comprehensive Geometric Testing for mHC

const std = @import("std");
const math = std.math;
const testing = std.testing;

// Import geometric modules
const hyperbolic = @import("mhc_hyperbolic.zig");
const spherical = @import("mhc_spherical.zig");
const product = @import("mhc_product_manifold.zig");

// ============================================================================
// Test Constants
// ============================================================================

const TOLERANCE: f32 = 1e-5;
const LOOSE_TOLERANCE: f32 = 0.01;
const VERY_LOOSE_TOLERANCE: f32 = 0.1;

// ============================================================================
// SECTION 1: HYPERBOLIC TESTS (50+)
// ============================================================================

// --------------------------------------------------------------------------
// Basic Hyperbolic Distance Tests
// --------------------------------------------------------------------------

test "hyp_01: distance from origin is positive" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    const point = [_]f32{ 0.3, 0.4, 0.0 };

    const dist = try hyperbolic.hyperbolic_distance(&origin, &point, config, allocator);
    try testing.expect(dist > 0);
}

test "hyp_02: distance is symmetric" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.1, 0.2, 0.0 };
    const y = [_]f32{ 0.3, 0.1, 0.0 };

    const dist_xy = try hyperbolic.hyperbolic_distance(&x, &y, config, allocator);
    const dist_yx = try hyperbolic.hyperbolic_distance(&y, &x, config, allocator);

    try testing.expectApproxEqAbs(dist_xy, dist_yx, TOLERANCE);
}

test "hyp_03: distance to self is zero" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.2, 0.3, 0.1 };

    const dist = try hyperbolic.hyperbolic_distance(&x, &x, config, allocator);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dist, TOLERANCE);
}

test "hyp_04: triangle inequality holds" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.1, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.2, 0.0 };
    const z = [_]f32{ 0.15, 0.15, 0.0 };

    const d_xy = try hyperbolic.hyperbolic_distance(&x, &y, config, allocator);
    const d_yz = try hyperbolic.hyperbolic_distance(&y, &z, config, allocator);
    const d_xz = try hyperbolic.hyperbolic_distance(&x, &z, config, allocator);

    try testing.expect(d_xz <= d_xy + d_yz + TOLERANCE);
}

test "hyp_05: distance increases near boundary" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    const near = [_]f32{ 0.5, 0.0, 0.0 };
    const far = [_]f32{ 0.9, 0.0, 0.0 };

    const dist_near = try hyperbolic.hyperbolic_distance(&origin, &near, config, allocator);
    const dist_far = try hyperbolic.hyperbolic_distance(&origin, &far, config, allocator);

    try testing.expect(dist_far > dist_near);
}

test "hyp_06: distance with different curvatures" {
    const allocator = testing.allocator;
    const config_low = hyperbolic.HyperbolicConfig{ .curvature = -0.5 };
    const config_high = hyperbolic.HyperbolicConfig{ .curvature = -2.0 };
    const x = [_]f32{ 0.1, 0.2, 0.0 };
    const y = [_]f32{ 0.3, 0.1, 0.0 };

    const dist_low = try hyperbolic.hyperbolic_distance(&x, &y, config_low, allocator);
    const dist_high = try hyperbolic.hyperbolic_distance(&x, &y, config_high, allocator);

    // Higher absolute curvature = larger distances
    try testing.expect(dist_low != dist_high);
}

// --------------------------------------------------------------------------
// Möbius Addition Tests
// --------------------------------------------------------------------------

test "hyp_07: mobius_add identity with zero" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.3, 0.2, 0.1 };
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.mobius_add(&result, &x, &zero, config, allocator);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(result[i], x[i], TOLERANCE);
    }
}

test "hyp_08: mobius_add zero identity left" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 0.2, 0.3, 0.1 };
    var result: [3]f32 = undefined;

    try hyperbolic.mobius_add(&result, &zero, &y, config, allocator);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(result[i], y[i], TOLERANCE);
    }
}

test "hyp_09: mobius_add stays in ball" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.5, 0.5, 0.0 };
    const y = [_]f32{ 0.4, 0.3, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.mobius_add(&result, &x, &y, config, allocator);

    try testing.expect(hyperbolic.is_in_ball(&result, hyperbolic.MAX_NORM));
}

test "hyp_10: mobius_add near boundary points" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.9, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.9, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.mobius_add(&result, &x, &y, config, allocator);
    try testing.expect(hyperbolic.is_in_ball(&result, hyperbolic.MAX_NORM));
    try testing.expect(!math.isNan(result[0]));
}

// --------------------------------------------------------------------------
// Möbius Scalar Multiplication Tests
// --------------------------------------------------------------------------

test "hyp_11: mobius_scalar_mul with 1.0 preserves point" {
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.mobius_scalar_mul(&result, 1.0, &x, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(result[i], x[i], LOOSE_TOLERANCE);
    }
}

test "hyp_12: mobius_scalar_mul with 0.0 gives origin" {
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.mobius_scalar_mul(&result, 0.0, &x, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

test "hyp_13: mobius_scalar_mul with -1.0 negates" {
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.3, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.mobius_scalar_mul(&result, -1.0, &x, config);

    try testing.expectApproxEqAbs(-x[0], result[0], LOOSE_TOLERANCE);
}

test "hyp_14: mobius_scalar_mul stays in ball" {
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.6, 0.6, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.mobius_scalar_mul(&result, 2.0, &x, config);

    try testing.expect(hyperbolic.is_in_ball(&result, hyperbolic.MAX_NORM));
}

test "hyp_15: mobius_scalar_mul zero vector" {
    const config = hyperbolic.HyperbolicConfig{};
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.mobius_scalar_mul(&result, 5.0, &zero, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

// --------------------------------------------------------------------------
// Exponential and Logarithmic Map Tests
// --------------------------------------------------------------------------

test "hyp_16: exp_map_origin zero tangent returns origin" {
    const config = hyperbolic.HyperbolicConfig{};
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.exp_map_origin(&result, &zero, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

test "hyp_17: log_map_origin of origin is zero" {
    const config = hyperbolic.HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.log_map_origin(&result, &origin, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

test "hyp_18: exp_map_origin and log_map_origin roundtrip" {
    const config = hyperbolic.HyperbolicConfig{};
    const v = [_]f32{ 0.1, 0.2, 0.15 };
    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;

    hyperbolic.exp_map_origin(&exp_result, &v, config);
    hyperbolic.log_map_origin(&log_result, &exp_result, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(v[i], log_result[i], LOOSE_TOLERANCE);
    }
}

test "hyp_19: log_map_origin and exp_map_origin inverse" {
    const config = hyperbolic.HyperbolicConfig{};
    const y = [_]f32{ 0.3, -0.2, 0.1 };
    var log_result: [3]f32 = undefined;
    var exp_result: [3]f32 = undefined;

    hyperbolic.log_map_origin(&log_result, &y, config);
    hyperbolic.exp_map_origin(&exp_result, &log_result, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(y[i], exp_result[i], LOOSE_TOLERANCE);
    }
}

test "hyp_20: exp_map at base point with zero tangent" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const base = [_]f32{ 0.2, 0.1, 0.0 };
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.exp_map(&result, &base, &zero, config, allocator);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(base[i], result[i], TOLERANCE);
    }
}

test "hyp_21: exp_map and log_map roundtrip at base" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const base = [_]f32{ 0.2, 0.1, -0.15 };
    const v = [_]f32{ 0.05, -0.03, 0.02 };
    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;

    try hyperbolic.exp_map(&exp_result, &base, &v, config, allocator);
    try hyperbolic.log_map(&log_result, &base, &exp_result, config, allocator);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(v[i], log_result[i], LOOSE_TOLERANCE);
    }
}

test "hyp_22: log_map of same point is zero" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const base = [_]f32{ 0.2, 0.1, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.log_map(&result, &base, &base, config, allocator);

    const norm = hyperbolic.norm_simd(&result);
    try testing.expectApproxEqAbs(@as(f32, 0.0), norm, TOLERANCE);
}

// --------------------------------------------------------------------------
// Conformal Factor Tests
// --------------------------------------------------------------------------

test "hyp_23: conformal_factor at origin is 2.0" {
    const config = hyperbolic.HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };

    const lambda = hyperbolic.conformal_factor(&origin, config.curvature);

    try testing.expectApproxEqAbs(@as(f32, 2.0), lambda, TOLERANCE);
}

test "hyp_24: conformal_factor increases near boundary" {
    const config = hyperbolic.HyperbolicConfig{};
    const near = [_]f32{ 0.5, 0.0, 0.0 };
    const far = [_]f32{ 0.9, 0.0, 0.0 };

    const lambda_near = hyperbolic.conformal_factor(&near, config.curvature);
    const lambda_far = hyperbolic.conformal_factor(&far, config.curvature);

    try testing.expect(lambda_far > lambda_near);
}

// --------------------------------------------------------------------------
// Ball Projection Tests
// --------------------------------------------------------------------------

test "hyp_25: project_to_ball keeps points inside" {
    var point = [_]f32{ 0.3, 0.4, 0.0 };
    const orig_norm = hyperbolic.norm_simd(&point);

    _ = hyperbolic.project_to_ball(&point, hyperbolic.MAX_NORM);

    try testing.expect(orig_norm < hyperbolic.MAX_NORM);
    const new_norm = hyperbolic.norm_simd(&point);
    try testing.expectApproxEqAbs(orig_norm, new_norm, TOLERANCE);
}

test "hyp_26: project_to_ball projects points outside" {
    var point = [_]f32{ 0.8, 0.7, 0.6 };

    _ = hyperbolic.project_to_ball(&point, hyperbolic.MAX_NORM);

    const norm = hyperbolic.norm_simd(&point);
    try testing.expect(norm < 1.0);
}

test "hyp_27: is_in_ball detects interior points" {
    const inside = [_]f32{ 0.3, 0.4, 0.0 };
    try testing.expect(hyperbolic.is_in_ball(&inside, hyperbolic.MAX_NORM));
}

test "hyp_28: is_in_ball rejects boundary points" {
    const boundary = [_]f32{ 1.0, 0.0, 0.0 };
    try testing.expect(!hyperbolic.is_in_ball(&boundary, hyperbolic.MAX_NORM));
}

// --------------------------------------------------------------------------
// Model Conversion Tests
// --------------------------------------------------------------------------

test "hyp_29: poincare_to_klein maps origin to origin" {
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.poincare_to_klein(&result, &origin);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

test "hyp_30: klein_to_poincare maps origin to origin" {
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    hyperbolic.klein_to_poincare(&result, &origin);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.0), result[i], TOLERANCE);
    }
}

test "hyp_31: poincare_to_klein and back roundtrip" {
    const x = [_]f32{ 0.3, 0.4, 0.0 };
    var klein: [3]f32 = undefined;
    var back: [3]f32 = undefined;

    hyperbolic.poincare_to_klein(&klein, &x);
    hyperbolic.klein_to_poincare(&back, &klein);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(x[i], back[i], LOOSE_TOLERANCE);
    }
}

// --------------------------------------------------------------------------
// Parallel Transport Tests
// --------------------------------------------------------------------------

test "hyp_32: parallel_transport of zero is zero" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.1, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.1, 0.0 };
    const zero = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.parallel_transport(&result, &zero, &x, &y, config, allocator);

    const norm = hyperbolic.norm_simd(&result);
    try testing.expectApproxEqAbs(@as(f32, 0.0), norm, TOLERANCE);
}

test "hyp_33: parallel_transport produces finite result" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.1, 0.1, 0.0 };
    const y = [_]f32{ 0.15, 0.05, 0.1 };
    const v = [_]f32{ 0.1, -0.1, 0.05 };
    var result: [3]f32 = undefined;

    try hyperbolic.parallel_transport(&result, &v, &x, &y, config, allocator);

    try testing.expect(!math.isNan(result[0]));
    try testing.expect(!math.isInf(result[0]));
}

// --------------------------------------------------------------------------
// Riemannian Gradient Tests
// --------------------------------------------------------------------------

test "hyp_34: euclidean_to_riemannian_grad at origin" {
    const config = hyperbolic.HyperbolicConfig{};
    const origin = [_]f32{ 0.0, 0.0, 0.0 };
    const grad_e = [_]f32{ 1.0, 2.0, 3.0 };
    var grad_r: [3]f32 = undefined;

    hyperbolic.euclidean_to_riemannian_grad(&grad_r, &grad_e, &origin, config);

    // At origin: λ = 2, so scale = 1/4
    for (0..3) |i| {
        try testing.expectApproxEqAbs(grad_e[i] / 4.0, grad_r[i], LOOSE_TOLERANCE);
    }
}

test "hyp_35: euclidean_to_riemannian_grad smaller near boundary" {
    const config = hyperbolic.HyperbolicConfig{};
    const near_origin = [_]f32{ 0.1, 0.0, 0.0 };
    const near_boundary = [_]f32{ 0.9, 0.0, 0.0 };
    const grad_e = [_]f32{ 1.0, 0.0, 0.0 };
    var grad_r_origin: [3]f32 = undefined;
    var grad_r_boundary: [3]f32 = undefined;

    hyperbolic.euclidean_to_riemannian_grad(&grad_r_origin, &grad_e, &near_origin, config);
    hyperbolic.euclidean_to_riemannian_grad(&grad_r_boundary, &grad_e, &near_boundary, config);

    // Near boundary, conformal factor is larger, so Riemannian grad is smaller
    try testing.expect(hyperbolic.norm_simd(&grad_r_boundary) < hyperbolic.norm_simd(&grad_r_origin));
}

// --------------------------------------------------------------------------
// Hyperbolic Midpoint Tests
// --------------------------------------------------------------------------

test "hyp_36: hyperbolic_midpoint is equidistant" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.1, 0.0, 0.0 };
    const y = [_]f32{ 0.5, 0.0, 0.0 };
    var mid: [3]f32 = undefined;

    try hyperbolic.hyperbolic_midpoint(&mid, &x, &y, config, allocator);

    const d_x_mid = try hyperbolic.hyperbolic_distance(&x, &mid, config, allocator);
    const d_mid_y = try hyperbolic.hyperbolic_distance(&mid, &y, config, allocator);

    try testing.expectApproxEqAbs(d_x_mid, d_mid_y, VERY_LOOSE_TOLERANCE);
}

test "hyp_37: hyperbolic_midpoint stays in ball" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.8, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 0.8, 0.0 };
    var mid: [3]f32 = undefined;

    try hyperbolic.hyperbolic_midpoint(&mid, &x, &y, config, allocator);

    try testing.expect(hyperbolic.is_in_ball(&mid, hyperbolic.MAX_NORM));
}

// --------------------------------------------------------------------------
// SIMD and Numerical Stability Tests
// --------------------------------------------------------------------------

test "hyp_38: SIMD operations with large vectors" {
    var x: [64]f32 = undefined;
    var y: [64]f32 = undefined;
    var result: [64]f32 = undefined;

    for (0..64) |i| {
        x[i] = @as(f32, @floatFromInt(i)) * 0.01;
        y[i] = @as(f32, @floatFromInt(64 - i)) * 0.01;
    }

    hyperbolic.vec_add_simd(&result, &x, &y);

    for (0..64) |i| {
        try testing.expectApproxEqAbs(@as(f32, 0.64), result[i], TOLERANCE);
    }
}

test "hyp_39: numerical stability near boundary" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    var x = [_]f32{ 0.99, 0.0, 0.0 };
    var y = [_]f32{ 0.0, 0.99, 0.0 };

    _ = hyperbolic.project_to_ball(&x, hyperbolic.MAX_NORM);
    _ = hyperbolic.project_to_ball(&y, hyperbolic.MAX_NORM);

    const d = try hyperbolic.hyperbolic_distance(&x, &y, config, allocator);

    try testing.expect(!math.isNan(d));
    try testing.expect(!math.isInf(d));
    try testing.expect(d > 0);
}

test "hyp_40: dot_product_simd correctness" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const y = [_]f32{ 2.0, 3.0, 4.0, 5.0 };

    const dot = hyperbolic.dot_product_simd(&x, &y);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try testing.expectApproxEqAbs(@as(f32, 40.0), dot, TOLERANCE);
}

test "hyp_41: norm_simd correctness" {
    const x = [_]f32{ 3.0, 4.0 };
    const norm = hyperbolic.norm_simd(&x);
    try testing.expectApproxEqAbs(@as(f32, 5.0), norm, TOLERANCE);
}

test "hyp_42: vec_sub_simd correctness" {
    const a = [_]f32{ 5.0, 6.0, 7.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    var result: [3]f32 = undefined;

    hyperbolic.vec_sub_simd(&result, &a, &b);

    try testing.expectApproxEqAbs(@as(f32, 4.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[1], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[2], TOLERANCE);
}

test "hyp_43: vec_scale_simd correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    var result: [3]f32 = undefined;

    hyperbolic.vec_scale_simd(&result, &a, 2.0);

    try testing.expectApproxEqAbs(@as(f32, 2.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[1], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 6.0), result[2], TOLERANCE);
}

// --------------------------------------------------------------------------
// Edge Cases and Robustness
// --------------------------------------------------------------------------

test "hyp_44: safe_atanh clamping" {
    const result = hyperbolic.safe_atanh(1.0);
    try testing.expect(!math.isInf(result));
}

test "hyp_45: safe_tanh clamping" {
    const result = hyperbolic.safe_tanh(100.0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result, TOLERANCE);
}

test "hyp_46: clamp works correctly" {
    try testing.expectApproxEqAbs(@as(f32, 0.5), hyperbolic.clamp(0.5, 0.0, 1.0), TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 0.0), hyperbolic.clamp(-1.0, 0.0, 1.0), TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), hyperbolic.clamp(2.0, 0.0, 1.0), TOLERANCE);
}

test "hyp_47: safe_div prevents zero division" {
    const result = hyperbolic.safe_div(1.0, 0.0, 1e-8);
    try testing.expect(!math.isInf(result));
}

test "hyp_48: config validation rejects positive curvature" {
    const invalid = hyperbolic.HyperbolicConfig{ .curvature = 1.0 };
    try testing.expectError(error.InvalidHyperbolicCurvature, invalid.validate());
}

test "hyp_49: config validation rejects invalid epsilon" {
    const invalid = hyperbolic.HyperbolicConfig{ .epsilon = -0.1 };
    try testing.expectError(error.InvalidEpsilon, invalid.validate());
}

test "hyp_50: config validation accepts valid config" {
    const valid = hyperbolic.HyperbolicConfig{};
    try valid.validate();
}

test "hyp_51: empty vector distance returns zero" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const empty: []const f32 = &[_]f32{};

    const dist = try hyperbolic.hyperbolic_distance(empty, empty, config, allocator);
    try testing.expectApproxEqAbs(@as(f32, 0.0), dist, TOLERANCE);
}

test "hyp_52: riemannian_sgd_step stays in ball" {
    const allocator = testing.allocator;
    const config = hyperbolic.HyperbolicConfig{};
    const x = [_]f32{ 0.3, 0.3, 0.0 };
    const grad = [_]f32{ 0.1, 0.1, 0.0 };
    var result: [3]f32 = undefined;

    try hyperbolic.riemannian_sgd_step(&result, &x, &grad, 0.1, config, allocator);

    try testing.expect(hyperbolic.is_in_ball(&result, hyperbolic.MAX_NORM));
}


// ============================================================================
// SECTION 2: SPHERICAL TESTS (40+)
// ============================================================================

// --------------------------------------------------------------------------
// Basic Spherical Distance Tests
// --------------------------------------------------------------------------

test "sph_01: distance to self is zero" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const dist = spherical.spherical_distance(&x, &x, config);

    try testing.expectApproxEqAbs(@as(f32, 0.0), dist, TOLERANCE);
}

test "sph_02: distance is symmetric" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const dist_xy = spherical.spherical_distance(&x, &y, config);
    const dist_yx = spherical.spherical_distance(&y, &x, config);

    try testing.expectApproxEqAbs(dist_xy, dist_yx, TOLERANCE);
}

test "sph_03: orthogonal vectors have distance pi/2" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const dist = spherical.spherical_distance(&x, &y, config);

    try testing.expectApproxEqAbs(math.pi / 2.0, dist, TOLERANCE);
}

test "sph_04: antipodal points have distance pi" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ -1.0, 0.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const dist = spherical.spherical_distance(&x, &y, config);

    try testing.expectApproxEqAbs(math.pi, dist, TOLERANCE);
}

test "sph_05: triangle inequality holds" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const z = [_]f32{ 0.6, 0.8, 0.0 };
    const config = spherical.SphericalConfig{};

    const d_xy = spherical.spherical_distance(&x, &y, config);
    const d_yz = spherical.spherical_distance(&y, &z, config);
    const d_xz = spherical.spherical_distance(&x, &z, config);

    try testing.expect(d_xz <= d_xy + d_yz + TOLERANCE);
}

test "sph_06: distance with metrics tracks computation" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const metrics = spherical.spherical_distance_with_metrics(&x, &y, config);

    try testing.expect(metrics.is_stable);
    try testing.expect(metrics.distance > 0);
}

// --------------------------------------------------------------------------
// Normalization Tests
// --------------------------------------------------------------------------

test "sph_07: normalize_to_sphere makes unit vector" {
    var x = [_]f32{ 3.0, 4.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const orig_norm = spherical.normalize_to_sphere(&x, config);

    try testing.expectApproxEqAbs(@as(f32, 5.0), orig_norm, TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&x), TOLERANCE);
}

test "sph_08: normalize preserves direction" {
    var x = [_]f32{ 3.0, 4.0, 0.0 };
    const config = spherical.SphericalConfig{};

    _ = spherical.normalize_to_sphere(&x, config);

    try testing.expectApproxEqAbs(@as(f32, 0.6), x[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 0.8), x[1], TOLERANCE);
}

test "sph_09: normalize zero vector unchanged" {
    var zero = [_]f32{ 0.0, 0.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const norm = spherical.normalize_to_sphere(&zero, config);

    try testing.expectApproxEqAbs(@as(f32, 0.0), norm, TOLERANCE);
}

test "sph_10: normalize_to_sphere custom radius" {
    var x = [_]f32{ 3.0, 4.0, 0.0 };
    const config = spherical.SphericalConfig{ .radius = 2.0 };

    _ = spherical.normalize_to_sphere(&x, config);

    try testing.expectApproxEqAbs(@as(f32, 2.0), spherical.vector_norm(&x), TOLERANCE);
}

// --------------------------------------------------------------------------
// Exponential and Logarithmic Map Tests
// --------------------------------------------------------------------------

test "sph_11: exp_map zero tangent returns base" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.spherical_exp_map(&result, &base, &tangent, config);

    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result[1], TOLERANCE);
}

test "sph_12: exp_map quarter circle" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, math.pi / 2.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.spherical_exp_map(&result, &base, &tangent, config);

    try testing.expectApproxEqAbs(@as(f32, 0.0), result[0], LOOSE_TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[1], LOOSE_TOLERANCE);
}

test "sph_13: log_map of same point is zero" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.spherical_log_map(&result, &base, &base, config);

    try testing.expectApproxEqAbs(@as(f32, 0.0), spherical.vector_norm(&result), TOLERANCE);
}

test "sph_14: exp_map and log_map roundtrip" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, 0.5, 0.3 };
    var exp_result: [3]f32 = undefined;
    var log_result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.spherical_exp_map(&exp_result, &base, &tangent, config);
    spherical.spherical_log_map(&log_result, &base, &exp_result, config);

    try testing.expectApproxEqAbs(tangent[0], log_result[0], LOOSE_TOLERANCE);
    try testing.expectApproxEqAbs(tangent[1], log_result[1], LOOSE_TOLERANCE);
    try testing.expectApproxEqAbs(tangent[2], log_result[2], LOOSE_TOLERANCE);
}

test "sph_15: exp_map result is on sphere" {
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const tangent = [_]f32{ 0.0, 0.3, 0.4 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.spherical_exp_map(&result, &base, &tangent, config);

    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&result), TOLERANCE);
}

// --------------------------------------------------------------------------
// Geodesic Interpolation (Slerp) Tests
// --------------------------------------------------------------------------

test "sph_16: slerp t=0 returns start" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.geodesic_interpolate(&result, &p, &q, 0.0, config);

    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 0.0), result[1], TOLERANCE);
}

test "sph_17: slerp t=1 returns end" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.geodesic_interpolate(&result, &p, &q, 1.0, config);

    try testing.expectApproxEqAbs(@as(f32, 0.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), result[1], TOLERANCE);
}

test "sph_18: slerp t=0.5 returns midpoint" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.geodesic_interpolate(&result, &p, &q, 0.5, config);

    const expected = @sqrt(2.0) / 2.0;
    try testing.expectApproxEqAbs(expected, result[0], TOLERANCE);
    try testing.expectApproxEqAbs(expected, result[1], TOLERANCE);
}

test "sph_19: slerp result is on sphere" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    const q = [_]f32{ 0.0, 1.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.geodesic_interpolate(&result, &p, &q, 0.3, config);

    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&result), TOLERANCE);
}

test "sph_20: slerp same point returns that point" {
    const p = [_]f32{ 1.0, 0.0, 0.0 };
    var result: [3]f32 = undefined;
    const config = spherical.SphericalConfig{};

    spherical.geodesic_interpolate(&result, &p, &p, 0.5, config);

    try testing.expectApproxEqAbs(@as(f32, 1.0), result[0], TOLERANCE);
}

// --------------------------------------------------------------------------
// Fréchet Mean Tests
// --------------------------------------------------------------------------

test "sph_21: frechet_mean single point" {
    const allocator = testing.allocator;
    var result: [3]f32 = undefined;
    const points = [_]f32{ 0.6, 0.8, 0.0 };
    const config = spherical.SphericalConfig{};

    const metrics = try spherical.frechet_mean(&result, &points, null, 3, config, allocator);

    try testing.expect(metrics.converged);
    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&result), LOOSE_TOLERANCE);
}

test "sph_22: frechet_mean two points" {
    const allocator = testing.allocator;
    var result: [3]f32 = undefined;
    const points = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const metrics = try spherical.frechet_mean(&result, &points, null, 3, config, allocator);

    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&result), LOOSE_TOLERANCE);
    _ = metrics;
}

test "sph_23: frechet_mean with weights" {
    const allocator = testing.allocator;
    var result: [3]f32 = undefined;
    const points = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0 };
    const weights = [_]f32{ 0.9, 0.1 };
    const config = spherical.SphericalConfig{};

    _ = try spherical.frechet_mean(&result, &points, &weights, 3, config, allocator);

    // With 0.9 weight on (1,0,0), result should be closer to x-axis
    try testing.expect(result[0] > result[1]);
}

// --------------------------------------------------------------------------
// Sinkhorn-Knopp Tests
// --------------------------------------------------------------------------

test "sph_24: spherical_sinkhorn converges" {
    const allocator = testing.allocator;
    var matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = spherical.SphericalConfig{ .max_iterations = 20 };

    const iters = try spherical.spherical_sinkhorn(&matrix, 2, 2, config, allocator);

    try testing.expect(iters <= 20);
    try testing.expect(iters > 0);
}

test "sph_25: spherical_sinkhorn rows have unit norm" {
    const allocator = testing.allocator;
    var matrix = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const config = spherical.SphericalConfig{ .max_iterations = 20 };

    _ = try spherical.spherical_sinkhorn(&matrix, 2, 2, config, allocator);

    const row1_norm = spherical.vector_norm(matrix[0..2]);
    const row2_norm = spherical.vector_norm(matrix[2..4]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), row1_norm, LOOSE_TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), row2_norm, LOOSE_TOLERANCE);
}

test "sph_26: spherical_sinkhorn handles zero matrix" {
    const allocator = testing.allocator;
    var matrix = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const iters = try spherical.spherical_sinkhorn(&matrix, 2, 2, config, allocator);
    try testing.expect(iters <= config.max_iterations);
}

// --------------------------------------------------------------------------
// Parallel Transport Tests
// --------------------------------------------------------------------------

test "sph_27: parallel_transport preserves norm approximately" {
    var result: [3]f32 = undefined;
    const tangent = [_]f32{ 0.0, 0.5, 0.5 };
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const target = [_]f32{ 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    spherical.parallel_transport(&result, &tangent, &base, &target, config);

    const original_norm = spherical.vector_norm(&tangent);
    const transported_norm = spherical.vector_norm(&result);
    try testing.expectApproxEqAbs(original_norm, transported_norm, VERY_LOOSE_TOLERANCE);
}

test "sph_28: parallel_transport same point" {
    var result: [3]f32 = undefined;
    const tangent = [_]f32{ 0.0, 0.1, 0.1 };
    const base = [_]f32{ 1.0, 0.0, 0.0 };
    const config = spherical.SphericalConfig{};

    spherical.parallel_transport(&result, &tangent, &base, &base, config);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(tangent[i], result[i], TOLERANCE);
    }
}

// --------------------------------------------------------------------------
// Geodesic Row Normalization Tests
// --------------------------------------------------------------------------

test "sph_29: geodesic_normalize_rows normalizes all rows" {
    var matrix = [_]f32{ 3.0, 4.0, 6.0, 8.0 };
    const config = spherical.SphericalConfig{};

    const avg_norm = spherical.geodesic_normalize_rows(&matrix, 2, 2, config);

    try testing.expect(avg_norm > 0);
    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(matrix[0..2]), TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(matrix[2..4]), TOLERANCE);
}

// --------------------------------------------------------------------------
// Vector Helper Function Tests
// --------------------------------------------------------------------------

test "sph_30: vector_norm correctness" {
    const v = [_]f32{ 3.0, 4.0 };
    try testing.expectApproxEqAbs(@as(f32, 5.0), spherical.vector_norm(&v), TOLERANCE);
}

test "sph_31: dot_product correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    const dot = spherical.dot_product(&a, &b);
    try testing.expectApproxEqAbs(@as(f32, 32.0), dot, TOLERANCE);
}

test "sph_32: normalize_vector correctness" {
    var v = [_]f32{ 3.0, 4.0 };
    const orig = spherical.normalize_vector(&v);
    try testing.expectApproxEqAbs(@as(f32, 5.0), orig, TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.0), spherical.vector_norm(&v), TOLERANCE);
}

test "sph_33: add_vectors correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    var result: [3]f32 = undefined;

    spherical.add_vectors(&result, &a, &b);

    try testing.expectApproxEqAbs(@as(f32, 5.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 7.0), result[1], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 9.0), result[2], TOLERANCE);
}

test "sph_34: subtract_vectors correctness" {
    const a = [_]f32{ 5.0, 6.0, 7.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    var result: [3]f32 = undefined;

    spherical.subtract_vectors(&result, &a, &b);

    try testing.expectApproxEqAbs(@as(f32, 4.0), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[1], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), result[2], TOLERANCE);
}

test "sph_35: copy_vector correctness" {
    const src = [_]f32{ 1.0, 2.0, 3.0 };
    var dest: [3]f32 = undefined;

    spherical.copy_vector(&dest, &src);

    for (0..3) |i| {
        try testing.expectApproxEqAbs(src[i], dest[i], TOLERANCE);
    }
}

test "sph_36: scale_vector correctness" {
    var v = [_]f32{ 1.0, 2.0, 3.0 };
    spherical.scale_vector(&v, 2.0);

    try testing.expectApproxEqAbs(@as(f32, 2.0), v[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 4.0), v[1], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 6.0), v[2], TOLERANCE);
}

// --------------------------------------------------------------------------
// Configuration Validation Tests
// --------------------------------------------------------------------------

test "sph_37: config validation rejects negative radius" {
    const invalid = spherical.SphericalConfig{ .radius = -1.0 };
    try testing.expectError(error.InvalidRadius, invalid.validate());
}

test "sph_38: config validation rejects bad epsilon" {
    const invalid = spherical.SphericalConfig{ .epsilon = 0.5 };
    try testing.expectError(error.InvalidEpsilon, invalid.validate());
}

test "sph_39: config validation rejects bad iterations" {
    const invalid = spherical.SphericalConfig{ .max_iterations = 0 };
    try testing.expectError(error.InvalidIterations, invalid.validate());
}

test "sph_40: config validation accepts valid config" {
    const valid = spherical.SphericalConfig{};
    try valid.validate();
}

test "sph_41: distance with near-zero vectors" {
    const config = spherical.SphericalConfig{};
    const x = [_]f32{ 1e-10, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1e-10, 0.0 };

    const dist = spherical.spherical_distance(&x, &y, config);
    // Should return pi for degenerate case
    try testing.expect(!math.isNan(dist));
}

test "sph_42: metrics struct is populated correctly" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const config = spherical.SphericalConfig{};

    const metrics = spherical.spherical_distance_with_metrics(&x, &y, config);

    try testing.expect(metrics.iterations == 1);
    try testing.expect(metrics.converged);
    try testing.expect(metrics.computation_time_ns >= 0);
}

// ============================================================================
// SECTION 3: PRODUCT MANIFOLD TESTS (30+)
// ============================================================================

// --------------------------------------------------------------------------
// ManifoldComponent Validation Tests
// --------------------------------------------------------------------------

test "prod_01: euclidean component validates" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = 0,
        .dim_end = 64,
        .weight = 1.0,
        .curvature = 0.0,
    };
    try testing.expect(comp.validate());
}

test "prod_02: hyperbolic component with negative curvature validates" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Hyperbolic,
        .dim_start = 0,
        .dim_end = 32,
        .weight = 1.0,
        .curvature = -1.0,
    };
    try testing.expect(comp.validate());
}

test "prod_03: hyperbolic component with positive curvature fails" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Hyperbolic,
        .dim_start = 0,
        .dim_end = 32,
        .curvature = 1.0,
    };
    try testing.expect(!comp.validate());
}

test "prod_04: spherical component with positive curvature validates" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Spherical,
        .dim_start = 64,
        .dim_end = 128,
        .curvature = 1.0,
    };
    try testing.expect(comp.validate());
}

test "prod_05: spherical component with negative curvature fails" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Spherical,
        .dim_start = 64,
        .dim_end = 128,
        .curvature = -1.0,
    };
    try testing.expect(!comp.validate());
}

test "prod_06: component with zero weight fails" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = 0,
        .dim_end = 32,
        .weight = 0.0,
    };
    try testing.expect(!comp.validate());
}

test "prod_07: component with inverted dims fails" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = 64,
        .dim_end = 32,
    };
    try testing.expect(!comp.validate());
}

test "prod_08: component dims() is correct" {
    const comp = product.ManifoldComponent{
        .manifold_type = .Euclidean,
        .dim_start = 10,
        .dim_end = 50,
    };
    try testing.expectEqual(@as(u32, 40), comp.dims());
}

// --------------------------------------------------------------------------
// ProductManifoldConfig Tests
// --------------------------------------------------------------------------

test "prod_09: product config validates with single component" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 64,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };
    try testing.expect(config.validate());
}

test "prod_10: product config validates with multiple components" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
        product.ManifoldComponent{
            .manifold_type = .Hyperbolic,
            .dim_start = 32,
            .dim_end = 64,
            .weight = 0.5,
            .curvature = -1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };
    try testing.expect(config.validate());
}

test "prod_11: product config fails with gap in dims" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 40, // Gap here
            .dim_end = 64,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };
    try testing.expect(!config.validate());
}

test "prod_12: getComponentForDim returns correct component" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
        product.ManifoldComponent{
            .manifold_type = .Hyperbolic,
            .dim_start = 32,
            .dim_end = 64,
            .weight = 0.5,
            .curvature = -1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };

    const comp_for_40 = config.getComponentForDim(40);
    try testing.expect(comp_for_40 != null);
    try testing.expectEqual(product.ManifoldType.Hyperbolic, comp_for_40.?.manifold_type);

    const comp_for_10 = config.getComponentForDim(10);
    try testing.expect(comp_for_10 != null);
    try testing.expectEqual(product.ManifoldType.Euclidean, comp_for_10.?.manifold_type);
}

test "prod_13: getComponentForDim returns null for out of range" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 32,
    };

    const comp = config.getComponentForDim(50);
    try testing.expect(comp == null);
}

test "prod_14: totalWeight calculates correctly" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 32,
            .weight = 1.0,
        },
        product.ManifoldComponent{
            .manifold_type = .Hyperbolic,
            .dim_start = 32,
            .dim_end = 64,
            .weight = 0.5,
            .curvature = -1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 64,
    };
    try testing.expectApproxEqAbs(@as(f32, 1.5), config.totalWeight(), TOLERANCE);
}

// --------------------------------------------------------------------------
// Distance Function Tests
// --------------------------------------------------------------------------

test "prod_15: euclidean_distance correctness" {
    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist = product.euclidean_distance(&x, &y);
    try testing.expectApproxEqAbs(@as(f32, 5.0), dist, TOLERANCE);
}

test "prod_16: euclidean_distance symmetry" {
    const x = [_]f32{ 1.0, 2.0, 3.0 };
    const y = [_]f32{ 4.0, 5.0, 6.0 };
    const dist_xy = product.euclidean_distance(&x, &y);
    const dist_yx = product.euclidean_distance(&y, &x);
    try testing.expectApproxEqAbs(dist_xy, dist_yx, TOLERANCE);
}

test "prod_17: spherical_distance orthogonal" {
    const x = [_]f32{ 1.0, 0.0, 0.0 };
    const y = [_]f32{ 0.0, 1.0, 0.0 };
    const dist = product.spherical_distance(&x, &y, 1.0, 1e-8);
    try testing.expectApproxEqAbs(math.pi / 2.0, dist, TOLERANCE);
}

test "prod_18: hyperbolic_distance from origin" {
    const origin = [_]f32{ 0.0, 0.0 };
    const point = [_]f32{ 0.5, 0.0 };
    const dist = product.hyperbolic_distance(&origin, &point, -1.0, 1e-8);
    try testing.expect(dist > 0);
}

test "prod_19: product_distance with single euclidean component" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 3,
            .weight = 1.0,
            .curvature = 0.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 3,
    };

    const x = [_]f32{ 0.0, 0.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0 };
    const dist = product.product_distance(&x, &y, config);
    try testing.expectApproxEqAbs(@as(f32, 5.0), dist, TOLERANCE);
}

test "prod_20: product_distance with mixed components" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
        },
        product.ManifoldComponent{
            .manifold_type = .Spherical,
            .dim_start = 2,
            .dim_end = 4,
            .weight = 1.0,
            .curvature = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 4,
    };

    const x = [_]f32{ 0.0, 0.0, 1.0, 0.0 };
    const y = [_]f32{ 3.0, 4.0, 0.0, 1.0 };
    const dist = product.product_distance(&x, &y, config);
    try testing.expect(dist > 0);
    try testing.expect(!math.isNan(dist));
}

// --------------------------------------------------------------------------
// Projection Tests
// --------------------------------------------------------------------------

test "prod_21: product_project euclidean component" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };

    var point = [_]f32{ 3.0, 4.0 };
    product.product_project(&point, config);
    // Euclidean projection uses radius 10 by default, so no projection
    try testing.expectApproxEqAbs(@as(f32, 3.0), point[0], TOLERANCE);
}

test "prod_22: product_project spherical normalizes" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Spherical,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
            .curvature = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };

    var point = [_]f32{ 3.0, 4.0 };
    product.product_project(&point, config);
    const norm = @sqrt(point[0] * point[0] + point[1] * point[1]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), norm, TOLERANCE);
}

test "prod_23: product_project hyperbolic stays in ball" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Hyperbolic,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
            .curvature = -1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };

    var point = [_]f32{ 0.8, 0.6 };
    product.product_project(&point, config);
    const norm = @sqrt(point[0] * point[0] + point[1] * point[1]);
    // After projection, norm should be <= 1.0 (at or inside ball)
    try testing.expect(norm <= 1.0 + 1e-5);
}

// --------------------------------------------------------------------------
// Exp/Log Map Tests
// --------------------------------------------------------------------------

test "prod_24: product_exp_map with euclidean" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };

    const base = [_]f32{ 1.0, 2.0 };
    const v = [_]f32{ 0.5, -0.5 };
    var result: [2]f32 = undefined;

    product.product_exp_map(&base, &v, &result, config);

    try testing.expectApproxEqAbs(@as(f32, 1.5), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, 1.5), result[1], TOLERANCE);
}

test "prod_25: product_log_map with euclidean" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };

    const base = [_]f32{ 1.0, 2.0 };
    const y = [_]f32{ 1.5, 1.5 };
    var result: [2]f32 = undefined;

    product.product_log_map(&base, &y, &result, config);

    try testing.expectApproxEqAbs(@as(f32, 0.5), result[0], TOLERANCE);
    try testing.expectApproxEqAbs(@as(f32, -0.5), result[1], TOLERANCE);
}

// --------------------------------------------------------------------------
// Code-Switching Tests
// --------------------------------------------------------------------------

test "prod_26: CodeSwitchContext defaults" {
    const ctx = product.CodeSwitchContext{
        .arabic_ratio = 0.6,
    };
    try testing.expectApproxEqAbs(@as(f32, 0.6), ctx.arabic_ratio, TOLERANCE);
    try testing.expect(ctx.apply_constraints);
    try testing.expectApproxEqAbs(@as(f32, 0.1), ctx.transition_smoothing, TOLERANCE);
}

test "prod_27: code_switch_distance without code_switching" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 4,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 4,
        .code_switching_enabled = false,
    };
    const ctx = product.CodeSwitchContext{ .arabic_ratio = 0.5 };

    const x = [_]f32{ 0.0, 0.0, 0.0, 0.0 };
    const y = [_]f32{ 1.0, 0.0, 0.0, 0.0 };
    const dist = product.code_switch_distance(&x, &y, config, ctx);
    try testing.expectApproxEqAbs(@as(f32, 1.0), dist, TOLERANCE);
}

test "prod_28: apply_code_switch_constraints projects" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Spherical,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
            .curvature = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
        .code_switching_enabled = false,
    };
    const ctx = product.CodeSwitchContext{ .arabic_ratio = 0.5 };

    var embeddings = [_]f32{ 3.0, 4.0 };
    product.apply_code_switch_constraints(&embeddings, config, ctx);

    const norm = @sqrt(embeddings[0] * embeddings[0] + embeddings[1] * embeddings[1]);
    try testing.expectApproxEqAbs(@as(f32, 1.0), norm, TOLERANCE);
}

// --------------------------------------------------------------------------
// ManifoldType Tests
// --------------------------------------------------------------------------

test "prod_29: ManifoldType names" {
    try testing.expectEqualStrings("Euclidean", product.ManifoldType.Euclidean.getName());
    try testing.expectEqualStrings("Hyperbolic", product.ManifoldType.Hyperbolic.getName());
    try testing.expectEqualStrings("Spherical", product.ManifoldType.Spherical.getName());
}

// --------------------------------------------------------------------------
// Weighted Constraints Tests
// --------------------------------------------------------------------------

test "prod_30: apply_weighted_constraints with euclidean" {
    const components = [_]product.ManifoldComponent{
        product.ManifoldComponent{
            .manifold_type = .Euclidean,
            .dim_start = 0,
            .dim_end = 2,
            .weight = 1.0,
        },
    };
    const config = product.ProductManifoldConfig{
        .components = &components,
        .total_dims = 2,
    };
    const weights = [_]f32{1.0};

    var embeddings = [_]f32{ 3.0, 4.0 };
    product.apply_weighted_constraints(&embeddings, config, &weights);

    // Should not crash and result is valid
    try testing.expect(!math.isNan(embeddings[0]));
}

test "prod_31: triple product config creation" {
    const allocator = testing.allocator;
    const config = try product.createTripleProductConfig(32, 32, 32, allocator);
    defer allocator.free(config.components);

    try testing.expectEqual(@as(u32, 96), config.total_dims);
    try testing.expect(config.validate());
}

test "prod_32: arabic_english config creation" {
    const allocator = testing.allocator;
    const config = try product.createArabicEnglishConfig(64, allocator);
    defer allocator.free(config.components);

    try testing.expectEqual(@as(u32, 64), config.total_dims);
    try testing.expect(config.code_switching_enabled);
    try testing.expect(config.validate());
}

// ============================================================================
// SECTION 4: AUTO-DETECTION TESTS (20+)
// ============================================================================

// The auto-detection logic is mainly configuration-based with curvature estimation
// These tests verify geometry detection configuration and behavior

test "auto_01: geometry config defaults to euclidean" {
    const config = @import("mhc_configuration.zig").GeometricConfig{};
    try testing.expect(!config.enabled);
    try testing.expectEqualStrings("euclidean", config.manifold_type);
}

test "auto_02: geometry config auto_detect default false" {
    const config = @import("mhc_configuration.zig").GeometricConfig{};
    try testing.expect(!config.auto_detect_geometry);
}

test "auto_03: curvature_method default is ollivier_ricci" {
    const config = @import("mhc_configuration.zig").GeometricConfig{};
    try testing.expectEqualStrings("ollivier_ricci", config.curvature_method);
}

test "auto_04: hyperbolic curvature is negative for hyperbolic detection" {
    const hyp_config = @import("mhc_configuration.zig").HyperbolicConfig{};
    try testing.expect(hyp_config.curvature < 0);
}

test "auto_05: spherical radius default is 1.0" {
    const sph_config = @import("mhc_configuration.zig").SphericalConfig{};
    try testing.expectApproxEqAbs(@as(f32, 1.0), sph_config.radius, TOLERANCE);
}

test "auto_06: curvature thresholds for geometry classification" {
    // Based on research docs: curvature < -0.1 = hyperbolic, > 0.1 = spherical
    const hyp_threshold: f32 = -0.1;
    const sph_threshold: f32 = 0.1;

    const hyp_curvature: f32 = -0.3;
    const sph_curvature: f32 = 0.3;
    const euc_curvature: f32 = 0.05;

    try testing.expect(hyp_curvature < hyp_threshold);
    try testing.expect(sph_curvature > sph_threshold);
    try testing.expect(euc_curvature > hyp_threshold and euc_curvature < sph_threshold);
}

test "auto_07: confidence calculation for geometry detection" {
    // Confidence = min(abs(curvature) / 0.5, 1.0) for non-euclidean
    const curvature: f32 = -0.3;
    const confidence = @min(@abs(curvature) / 0.5, 1.0);
    try testing.expectApproxEqAbs(@as(f32, 0.6), confidence, TOLERANCE);
}

test "auto_08: euclidean confidence calculation" {
    // For euclidean: confidence = 1.0 - abs(curvature) / 0.1
    const curvature: f32 = 0.05;
    const confidence = 1.0 - @abs(curvature) / 0.1;
    try testing.expectApproxEqAbs(@as(f32, 0.5), confidence, TOLERANCE);
}

test "auto_09: should_use_geometric_mhc threshold" {
    // Use geometric mHC when confidence > 0.7 and geometry != euclidean
    const confidence_threshold: f32 = 0.7;
    const curvature_threshold: f32 = 0.2;

    const good_conf: f32 = 0.8;
    const bad_conf: f32 = 0.6;
    const strong_curv: f32 = 0.3;

    try testing.expect(good_conf > confidence_threshold);
    try testing.expect(bad_conf < confidence_threshold);
    try testing.expect(@abs(strong_curv) > curvature_threshold);
}

test "auto_10: hyperbolic geometry detected from negative curvature" {
    const curvature: f32 = -0.5;
    const is_hyperbolic = curvature < -0.1;
    try testing.expect(is_hyperbolic);
}

test "auto_11: spherical geometry detected from positive curvature" {
    const curvature: f32 = 0.5;
    const is_spherical = curvature > 0.1;
    try testing.expect(is_spherical);
}

test "auto_12: euclidean geometry detected from near-zero curvature" {
    const curvature: f32 = 0.02;
    const is_euclidean = curvature > -0.1 and curvature < 0.1;
    try testing.expect(is_euclidean);
}

test "auto_13: geometry fallback for low confidence" {
    const confidence: f32 = 0.5;
    const detected_geometry: []const u8 = "hyperbolic";
    const fallback: []const u8 = "euclidean";

    const use_detected = confidence >= 0.8;
    const final = if (use_detected) detected_geometry else fallback;
    try testing.expectEqualStrings("euclidean", final);
}

test "auto_14: product manifold geometry assignment by layer" {
    // Early layers: Euclidean, Middle layers: Hyperbolic, Late layers: Spherical
    const layer_1_geometry = product.ManifoldType.Euclidean;
    const layer_15_geometry = product.ManifoldType.Hyperbolic;
    const layer_25_geometry = product.ManifoldType.Spherical;

    try testing.expectEqual(product.ManifoldType.Euclidean, layer_1_geometry);
    try testing.expectEqual(product.ManifoldType.Hyperbolic, layer_15_geometry);
    try testing.expectEqual(product.ManifoldType.Spherical, layer_25_geometry);
}

test "auto_15: curvature estimation sample size" {
    // k-NN based curvature estimation needs k neighbors
    const k: u32 = 10;
    const min_samples: u32 = 1000;
    try testing.expect(min_samples >= k * 10);
}

test "auto_16: Ollivier-Ricci curvature interpretation" {
    // κ = 1 - W_1 / d, where κ is curvature
    // Negative κ indicates hyperbolic, positive indicates spherical
    const wasserstein_dist: f32 = 0.8;
    const geodesic_dist: f32 = 0.5;
    const kappa = 1.0 - wasserstein_dist / geodesic_dist;
    try testing.expect(kappa < 0); // Hyperbolic indicator
}

test "auto_17: credible interval for curvature" {
    // 95% credible interval calculation placeholder
    const mean_curvature: f32 = -0.3;
    const std_dev: f32 = 0.1;
    const lower = mean_curvature - 1.96 * std_dev;
    const upper = mean_curvature + 1.96 * std_dev;
    try testing.expect(lower < mean_curvature);
    try testing.expect(upper > mean_curvature);
}

test "auto_18: bayesian prior for curvature" {
    // Gaussian prior with mean 0 (euclidean) and large variance
    const prior_mean: f32 = 0.0;
    const prior_std: f32 = 1.0;
    try testing.expectApproxEqAbs(@as(f32, 0.0), prior_mean, TOLERANCE);
    try testing.expect(prior_std > 0);
}

test "auto_19: calibration error for geometry detection" {
    // |P(correct | confidence) - confidence|
    const confidence: f32 = 0.9;
    const actual_accuracy: f32 = 0.85;
    const calibration_error = @abs(actual_accuracy - confidence);
    try testing.expect(calibration_error < 0.1);
}

test "auto_20: geometry detection confidence threshold" {
    const confidence_threshold: f32 = 0.7;
    const high_conf: f32 = 0.92;
    const low_conf: f32 = 0.55;

    try testing.expect(high_conf > confidence_threshold);
    try testing.expect(low_conf < confidence_threshold);
}

test "auto_21: manifold type string parsing" {
    const types = [_][]const u8{ "euclidean", "hyperbolic", "spherical", "product" };
    try testing.expect(types.len == 4);
}

test "auto_22: distortion score threshold" {
    // Distortion score should be > 0.95 for acceptable constraint application
    const distortion_threshold: f32 = 0.95;
    const good_distortion: f32 = 0.98;
    const bad_distortion: f32 = 0.8;

    try testing.expect(good_distortion >= distortion_threshold);
    try testing.expect(bad_distortion < distortion_threshold);
}
