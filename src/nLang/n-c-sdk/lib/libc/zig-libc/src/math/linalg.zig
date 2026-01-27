// Linear Algebra and Statistics functions for zig-libc
// Pure Zig implementations of vector, matrix, and statistical operations

const std = @import("std");
const math = std.math;

// ============================================================================
// Vector Operations (for f64 arrays)
// ============================================================================

/// Element-wise vector addition: result[i] = a[i] + b[i]
pub export fn vec_add(a: [*]const f64, b: [*]const f64, result: [*]f64, n: usize) void {
    for (0..n) |i| {
        result[i] = a[i] + b[i];
    }
}

/// Element-wise vector subtraction: result[i] = a[i] - b[i]
pub export fn vec_sub(a: [*]const f64, b: [*]const f64, result: [*]f64, n: usize) void {
    for (0..n) |i| {
        result[i] = a[i] - b[i];
    }
}

/// Scalar multiplication: result[i] = a[i] * scalar
pub export fn vec_scale(a: [*]const f64, scalar: f64, result: [*]f64, n: usize) void {
    for (0..n) |i| {
        result[i] = a[i] * scalar;
    }
}

/// Dot product: sum(a[i] * b[i])
pub export fn vec_dot(a: [*]const f64, b: [*]const f64, n: usize) f64 {
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += a[i] * b[i];
    }
    return sum;
}

/// Euclidean norm (L2): sqrt(sum(a[i]^2))
pub export fn vec_norm(a: [*]const f64, n: usize) f64 {
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += a[i] * a[i];
    }
    return math.sqrt(sum);
}

/// Normalize to unit vector: result[i] = a[i] / ||a||
pub export fn vec_normalize(a: [*]const f64, result: [*]f64, n: usize) void {
    const norm = vec_norm(a, n);
    if (norm == 0.0) {
        for (0..n) |i| {
            result[i] = 0.0;
        }
        return;
    }
    const inv_norm = 1.0 / norm;
    for (0..n) |i| {
        result[i] = a[i] * inv_norm;
    }
}

// ============================================================================
// Matrix Operations (row-major storage)
// ============================================================================

/// Element-wise matrix addition: result[i,j] = a[i,j] + b[i,j]
pub export fn mat_add(a: [*]const f64, b: [*]const f64, result: [*]f64, rows: usize, cols: usize) void {
    const size = rows * cols;
    for (0..size) |i| {
        result[i] = a[i] + b[i];
    }
}

/// Scalar matrix multiplication: result[i,j] = a[i,j] * scalar
pub export fn mat_scale(a: [*]const f64, scalar: f64, result: [*]f64, rows: usize, cols: usize) void {
    const size = rows * cols;
    for (0..size) |i| {
        result[i] = a[i] * scalar;
    }
}

/// Matrix multiplication: C = A * B where A is m×k, B is k×n, C is m×n
pub export fn mat_mul(a: [*]const f64, b: [*]const f64, result: [*]f64, m: usize, k: usize, n: usize) void {
    for (0..m) |i| {
        for (0..n) |j| {
            var sum: f64 = 0.0;
            for (0..k) |p| {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

/// Matrix transpose: result[j,i] = a[i,j]
pub export fn mat_transpose(a: [*]const f64, result: [*]f64, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        for (0..cols) |j| {
            result[j * rows + i] = a[i * cols + j];
        }
    }
}

/// Matrix-vector product: result = A * x where A is rows×cols, x is cols×1
pub export fn mat_vec_mul(a: [*]const f64, x: [*]const f64, result: [*]f64, rows: usize, cols: usize) void {
    for (0..rows) |i| {
        var sum: f64 = 0.0;
        for (0..cols) |j| {
            sum += a[i * cols + j] * x[j];
        }
        result[i] = sum;
    }
}

// ============================================================================
// Statistics Functions
// ============================================================================

/// Arithmetic mean: sum(data) / n
pub export fn mean(data: [*]const f64, n: usize) f64 {
    if (n == 0) return 0.0;
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += data[i];
    }
    return sum / @as(f64, @floatFromInt(n));
}

/// Sample variance: sum((data[i] - mean)^2) / (n-1)
pub export fn variance(data: [*]const f64, n: usize) f64 {
    if (n <= 1) return 0.0;
    const m = mean(data, n);
    var sum_sq: f64 = 0.0;
    for (0..n) |i| {
        const diff = data[i] - m;
        sum_sq += diff * diff;
    }
    return sum_sq / @as(f64, @floatFromInt(n - 1));
}

/// Sample standard deviation: sqrt(variance)
pub export fn stddev(data: [*]const f64, n: usize) f64 {
    return math.sqrt(variance(data, n));
}

/// Sample covariance: sum((x[i] - mean_x) * (y[i] - mean_y)) / (n-1)
pub export fn covariance(x: [*]const f64, y: [*]const f64, n: usize) f64 {
    if (n <= 1) return 0.0;
    const mean_x = mean(x, n);
    const mean_y = mean(y, n);
    var sum: f64 = 0.0;
    for (0..n) |i| {
        sum += (x[i] - mean_x) * (y[i] - mean_y);
    }
    return sum / @as(f64, @floatFromInt(n - 1));
}

/// Pearson correlation coefficient: cov(x,y) / (stddev(x) * stddev(y))
pub export fn correlation(x: [*]const f64, y: [*]const f64, n: usize) f64 {
    if (n <= 1) return 0.0;
    const cov = covariance(x, y, n);
    const std_x = stddev(x, n);
    const std_y = stddev(y, n);
    if (std_x == 0.0 or std_y == 0.0) return 0.0;
    return cov / (std_x * std_y);
}

/// Median (modifies array for sorting)
/// Uses in-place quickselect algorithm
pub export fn median(data: [*]f64, n: usize) f64 {
    if (n == 0) return 0.0;
    if (n == 1) return data[0];

    // Sort the array using insertion sort (simple for C interop)
    insertionSort(data, n);

    // Return middle element(s)
    if (n % 2 == 1) {
        return data[n / 2];
    } else {
        return (data[n / 2 - 1] + data[n / 2]) / 2.0;
    }
}

/// Helper: insertion sort for median calculation
fn insertionSort(data: [*]f64, n: usize) void {
    if (n <= 1) return;

    for (1..n) |i| {
        const key = data[i];
        var j: usize = i;
        while (j > 0 and data[j - 1] > key) {
            data[j] = data[j - 1];
            j -= 1;
        }
        data[j] = key;
    }
}

// ============================================================================
// Tests
// ============================================================================

test "vec_add" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const b = [_]f64{ 4.0, 5.0, 6.0 };
    var result: [3]f64 = undefined;

    vec_add(&a, &b, &result, 3);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), result[2], 1e-10);
}

test "vec_dot" {
    const a = [_]f64{ 1.0, 2.0, 3.0 };
    const b = [_]f64{ 4.0, 5.0, 6.0 };

    const dot = vec_dot(&a, &b, 3);

    try std.testing.expectApproxEqAbs(@as(f64, 32.0), dot, 1e-10);
}

test "vec_norm" {
    const a = [_]f64{ 3.0, 4.0 };

    const norm = vec_norm(&a, 2);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), norm, 1e-10);
}

test "mat_mul" {
    // A = [[1, 2], [3, 4]] (2x2)
    // B = [[5, 6], [7, 8]] (2x2)
    // C = [[19, 22], [43, 50]]
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f64{ 5.0, 6.0, 7.0, 8.0 };
    var result: [4]f64 = undefined;

    mat_mul(&a, &b, &result, 2, 2, 2);

    try std.testing.expectApproxEqAbs(@as(f64, 19.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 22.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 43.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 50.0), result[3], 1e-10);
}

test "mat_transpose" {
    // A = [[1, 2, 3], [4, 5, 6]] (2x3)
    // A^T = [[1, 4], [2, 5], [3, 6]] (3x2)
    const a = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    var result: [6]f64 = undefined;

    mat_transpose(&a, &result, 2, 3);

    try std.testing.expectApproxEqAbs(@as(f64, 1.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), result[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), result[3], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), result[4], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result[5], 1e-10);
}

test "mean and variance" {
    const data = [_]f64{ 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };

    const m = mean(&data, 8);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), m, 1e-10);

    const v = variance(&data, 8);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), v, 1e-10);
}

test "correlation" {
    // Perfect positive correlation
    const x = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const y = [_]f64{ 2.0, 4.0, 6.0, 8.0, 10.0 };

    const corr = correlation(&x, &y, 5);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), corr, 1e-10);
}

test "median" {
    var data_odd = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0 };
    const med_odd = median(&data_odd, 5);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), med_odd, 1e-10);

    var data_even = [_]f64{ 5.0, 1.0, 3.0, 2.0, 4.0, 6.0 };
    const med_even = median(&data_even, 6);
    try std.testing.expectApproxEqAbs(@as(f64, 3.5), med_even, 1e-10);
}

