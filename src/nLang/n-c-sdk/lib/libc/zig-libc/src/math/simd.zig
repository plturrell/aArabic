// SIMD-optimized vector operations for zig-libc
// Uses Zig's @Vector type for SIMD operations where possible

const std = @import("std");
const math = std.math;

// SIMD vector width for f32 (8 floats = 256-bit AVX)
const VEC_SIZE = 8;
const Vec8f = @Vector(VEC_SIZE, f32);

// ============================================================================
// Helper Functions
// ============================================================================

/// Reduce vector to scalar sum
inline fn vecSum(v: Vec8f) f32 {
    return @reduce(.Add, v);
}

/// Reduce vector to max value
inline fn vecMax(v: Vec8f) f32 {
    return @reduce(.Max, v);
}

/// Load vector from pointer, handling alignment
inline fn loadVec(ptr: [*]const f32) Vec8f {
    return ptr[0..VEC_SIZE].*;
}

/// Store vector to pointer
inline fn storeVec(ptr: [*]f32, v: Vec8f) void {
    ptr[0..VEC_SIZE].* = v;
}

// ============================================================================
// Similarity Metrics
// ============================================================================

/// Dot product using SIMD vectors
pub export fn n_dot_product(a: [*]const f32, b: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    // Process VEC_SIZE elements at a time
    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        const vb = loadVec(b + i);
        sum += va * vb;
    }

    var result = vecSum(sum);

    // Handle remaining elements
    while (i < nu) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// Cosine similarity between vectors: dot(a,b) / (|a| * |b|)
pub export fn n_cosine_similarity(a: [*]const f32, b: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var dot_sum: Vec8f = @splat(0.0);
    var a_sq_sum: Vec8f = @splat(0.0);
    var b_sq_sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        const vb = loadVec(b + i);
        dot_sum += va * vb;
        a_sq_sum += va * va;
        b_sq_sum += vb * vb;
    }

    var dot = vecSum(dot_sum);
    var a_sq = vecSum(a_sq_sum);
    var b_sq = vecSum(b_sq_sum);

    // Handle remaining elements
    while (i < nu) : (i += 1) {
        dot += a[i] * b[i];
        a_sq += a[i] * a[i];
        b_sq += b[i] * b[i];
    }

    const norm = @sqrt(a_sq) * @sqrt(b_sq);
    if (norm == 0.0) return 0.0;
    return dot / norm;
}

/// Euclidean distance (L2): sqrt(sum((a[i] - b[i])^2))
pub export fn n_euclidean_distance(a: [*]const f32, b: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        const vb = loadVec(b + i);
        const diff = va - vb;
        sum += diff * diff;
    }

    var result = vecSum(sum);

    while (i < nu) : (i += 1) {
        const diff = a[i] - b[i];
        result += diff * diff;
    }

    return @sqrt(result);
}

/// Manhattan distance (L1): sum(|a[i] - b[i]|)
pub export fn n_manhattan_distance(a: [*]const f32, b: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        const vb = loadVec(b + i);
        const diff = va - vb;
        sum += @abs(diff);
    }

    var result = vecSum(sum);

    while (i < nu) : (i += 1) {
        result += @abs(a[i] - b[i]);
    }

    return result;
}

// ============================================================================
// Norms
// ============================================================================

/// L2 norm (Euclidean): sqrt(sum(a[i]^2))
pub export fn n_l2_norm(a: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        sum += va * va;
    }

    var result = vecSum(sum);

    while (i < nu) : (i += 1) {
        result += a[i] * a[i];
    }

    return @sqrt(result);
}

/// L1 norm (Manhattan): sum(|a[i]|)
pub export fn n_l1_norm(a: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var sum: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        sum += @abs(va);
    }

    var result = vecSum(sum);

    while (i < nu) : (i += 1) {
        result += @abs(a[i]);
    }

    return result;
}

/// L-infinity norm: max(|a[i]|)
pub export fn n_linf_norm(a: [*]const f32, n: c_int) f32 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);

    var max_vec: Vec8f = @splat(0.0);
    var i: usize = 0;

    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        max_vec = @max(max_vec, @abs(va));
    }

    var result = vecMax(max_vec);

    while (i < nu) : (i += 1) {
        result = @max(result, @abs(a[i]));
    }

    return result;
}

// ============================================================================
// Pooling Operations
// ============================================================================

/// Mean pooling: average of n_vectors embeddings
/// embeddings: row-major matrix [n_vectors x dim]
/// result: output vector [dim]
pub export fn n_mean_pool(
    embeddings: [*]const f32,
    n_vectors: c_int,
    dim: c_int,
    result: [*]f32,
) void {
    if (n_vectors <= 0 or dim <= 0) return;
    const nv: usize = @intCast(n_vectors);
    const d: usize = @intCast(dim);

    // Initialize result to zero
    var j: usize = 0;
    while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
        storeVec(result + j, @splat(0.0));
    }
    while (j < d) : (j += 1) {
        result[j] = 0.0;
    }

    // Sum all vectors
    for (0..nv) |v| {
        const row = embeddings + v * d;
        j = 0;
        while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
            const current = loadVec(result + j);
            const add = loadVec(row + j);
            storeVec(result + j, current + add);
        }
        while (j < d) : (j += 1) {
            result[j] += row[j];
        }
    }

    // Divide by n_vectors
    const scale: Vec8f = @splat(1.0 / @as(f32, @floatFromInt(nv)));
    const scale_scalar: f32 = 1.0 / @as(f32, @floatFromInt(nv));
    j = 0;
    while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
        const current = loadVec(result + j);
        storeVec(result + j, current * scale);
    }
    while (j < d) : (j += 1) {
        result[j] *= scale_scalar;
    }
}

/// Max pooling: element-wise maximum across n_vectors embeddings
pub export fn n_max_pool(
    embeddings: [*]const f32,
    n_vectors: c_int,
    dim: c_int,
    result: [*]f32,
) void {
    if (n_vectors <= 0 or dim <= 0) return;
    const nv: usize = @intCast(n_vectors);
    const d: usize = @intCast(dim);

    // Initialize with first vector
    var j: usize = 0;
    while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
        storeVec(result + j, loadVec(embeddings + j));
    }
    while (j < d) : (j += 1) {
        result[j] = embeddings[j];
    }

    // Take max with remaining vectors
    for (1..nv) |v| {
        const row = embeddings + v * d;
        j = 0;
        while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
            const current = loadVec(result + j);
            const candidate = loadVec(row + j);
            storeVec(result + j, @max(current, candidate));
        }
        while (j < d) : (j += 1) {
            result[j] = @max(result[j], row[j]);
        }
    }
}

/// Weighted pooling: weighted average of embeddings
/// weights: [n_vectors] weights for each embedding
pub export fn n_weighted_pool(
    embeddings: [*]const f32,
    weights: [*]const f32,
    n_vectors: c_int,
    dim: c_int,
    result: [*]f32,
) void {
    if (n_vectors <= 0 or dim <= 0) return;
    const nv: usize = @intCast(n_vectors);
    const d: usize = @intCast(dim);

    // Initialize result to zero
    var j: usize = 0;
    while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
        storeVec(result + j, @splat(0.0));
    }
    while (j < d) : (j += 1) {
        result[j] = 0.0;
    }

    // Compute weighted sum and total weight
    var total_weight: f32 = 0.0;
    for (0..nv) |v| {
        const w = weights[v];
        total_weight += w;
        const w_vec: Vec8f = @splat(w);
        const row = embeddings + v * d;
        j = 0;
        while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
            const current = loadVec(result + j);
            const add = loadVec(row + j) * w_vec;
            storeVec(result + j, current + add);
        }
        while (j < d) : (j += 1) {
            result[j] += row[j] * w;
        }
    }

    // Normalize by total weight
    if (total_weight != 0.0) {
        const inv_weight: Vec8f = @splat(1.0 / total_weight);
        const inv_weight_scalar = 1.0 / total_weight;
        j = 0;
        while (j + VEC_SIZE <= d) : (j += VEC_SIZE) {
            const current = loadVec(result + j);
            storeVec(result + j, current * inv_weight);
        }
        while (j < d) : (j += 1) {
            result[j] *= inv_weight_scalar;
        }
    }
}

// ============================================================================
// Batch Operations
// ============================================================================

/// Batch normalize: result[i] = (data[i] - mean) / std
pub export fn n_batch_normalize(
    data: [*]const f32,
    n: c_int,
    mean: f32,
    std_val: f32,
    result: [*]f32,
) void {
    if (n <= 0 or std_val == 0.0) return;
    const nu: usize = @intCast(n);

    const mean_vec: Vec8f = @splat(mean);
    const inv_std: f32 = 1.0 / std_val;
    const inv_std_vec: Vec8f = @splat(inv_std);

    var i: usize = 0;
    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const v = loadVec(data + i);
        storeVec(result + i, (v - mean_vec) * inv_std_vec);
    }

    while (i < nu) : (i += 1) {
        result[i] = (data[i] - mean) * inv_std;
    }
}

/// Batch scale: result[i] = data[i] * scale
pub export fn n_batch_scale(
    data: [*]const f32,
    n: c_int,
    scale: f32,
    result: [*]f32,
) void {
    if (n <= 0) return;
    const nu: usize = @intCast(n);

    const scale_vec: Vec8f = @splat(scale);

    var i: usize = 0;
    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const v = loadVec(data + i);
        storeVec(result + i, v * scale_vec);
    }

    while (i < nu) : (i += 1) {
        result[i] = data[i] * scale;
    }
}

/// Batch add: result[i] = a[i] + b[i]
pub export fn n_batch_add(
    a: [*]const f32,
    b: [*]const f32,
    n: c_int,
    result: [*]f32,
) void {
    if (n <= 0) return;
    const nu: usize = @intCast(n);

    var i: usize = 0;
    while (i + VEC_SIZE <= nu) : (i += VEC_SIZE) {
        const va = loadVec(a + i);
        const vb = loadVec(b + i);
        storeVec(result + i, va + vb);
    }

    while (i < nu) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

// ============================================================================
// Tests
// ============================================================================

test "n_dot_product" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    const result = n_dot_product(&a, &b, 4);
    // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
    try std.testing.expectApproxEqAbs(@as(f32, 40.0), result, 1e-5);
}

test "n_cosine_similarity" {
    // Test with identical vectors (should be 1.0)
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0 };
    const result = n_cosine_similarity(&a, &b, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result, 1e-5);

    // Test with orthogonal vectors (should be 0.0)
    const c = [_]f32{ 1.0, 0.0 };
    const d = [_]f32{ 0.0, 1.0 };
    const result2 = n_cosine_similarity(&c, &d, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result2, 1e-5);
}

test "n_euclidean_distance" {
    const a = [_]f32{ 0.0, 0.0, 0.0 };
    const b = [_]f32{ 3.0, 4.0, 0.0 };
    const result = n_euclidean_distance(&a, &b, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-5);
}

test "n_manhattan_distance" {
    const a = [_]f32{ 0.0, 0.0 };
    const b = [_]f32{ 3.0, 4.0 };
    const result = n_manhattan_distance(&a, &b, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result, 1e-5);
}

test "n_l2_norm" {
    const a = [_]f32{ 3.0, 4.0 };
    const result = n_l2_norm(&a, 2);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-5);
}

test "n_l1_norm" {
    const a = [_]f32{ -1.0, 2.0, -3.0 };
    const result = n_l1_norm(&a, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result, 1e-5);
}

test "n_linf_norm" {
    const a = [_]f32{ 1.0, -5.0, 3.0 };
    const result = n_linf_norm(&a, 3);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result, 1e-5);
}

test "n_mean_pool" {
    // 2 vectors of dim 3
    const embeddings = [_]f32{ 1.0, 2.0, 3.0, 3.0, 4.0, 5.0 };
    var result: [3]f32 = undefined;
    n_mean_pool(&embeddings, 2, 3, &result);
    // Mean: [(1+3)/2, (2+4)/2, (3+5)/2] = [2, 3, 4]
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[2], 1e-5);
}

test "n_max_pool" {
    const embeddings = [_]f32{ 1.0, 4.0, 3.0, 3.0, 2.0, 5.0 };
    var result: [3]f32 = undefined;
    n_max_pool(&embeddings, 2, 3, &result);
    // Max: [max(1,3), max(4,2), max(3,5)] = [3, 4, 5]
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[2], 1e-5);
}

test "n_batch_normalize" {
    const data = [_]f32{ 2.0, 4.0, 6.0 };
    var result: [3]f32 = undefined;
    n_batch_normalize(&data, 3, 4.0, 2.0, &result);
    // (2-4)/2=-1, (4-4)/2=0, (6-4)/2=1
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[2], 1e-5);
}

test "n_batch_scale" {
    const data = [_]f32{ 1.0, 2.0, 3.0 };
    var result: [3]f32 = undefined;
    n_batch_scale(&data, 3, 2.5, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.5), result[2], 1e-5);
}

test "n_batch_add" {
    const a = [_]f32{ 1.0, 2.0, 3.0 };
    const b = [_]f32{ 4.0, 5.0, 6.0 };
    var result: [3]f32 = undefined;
    n_batch_add(&a, &b, 3, &result);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[0], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), result[1], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), result[2], 1e-5);
}

