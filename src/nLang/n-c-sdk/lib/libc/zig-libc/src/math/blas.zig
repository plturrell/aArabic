// BLAS-like operations and matrix decompositions for zig-libc
// Pure Zig implementations following BLAS conventions

const std = @import("std");
const math = std.math;

// ============================================================================
// Transpose flags (compatible with CBLAS)
// ============================================================================

pub const BLAS_NO_TRANS: c_int = 111;
pub const BLAS_TRANS: c_int = 112;
pub const BLAS_CONJ_TRANS: c_int = 113;

// ============================================================================
// BLAS Level 1: Vector Operations
// ============================================================================

/// Scale vector: x = alpha * x
/// n: number of elements
/// alpha: scalar multiplier
/// x: vector (modified in place)
/// incx: stride between elements
pub export fn blas_dscal(n: c_int, alpha: f64, x: [*]f64, incx: c_int) void {
    if (n <= 0 or incx <= 0) return;
    const nu: usize = @intCast(n);
    const incu: usize = @intCast(incx);

    var ix: usize = 0;
    for (0..nu) |_| {
        x[ix] *= alpha;
        ix += incu;
    }
}

/// Copy vector: y = x
/// n: number of elements
/// x: source vector
/// incx: source stride
/// y: destination vector (modified)
/// incy: destination stride
pub export fn blas_dcopy(n: c_int, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) void {
    if (n <= 0) return;
    const nu: usize = @intCast(n);
    const incxu: usize = @intCast(if (incx > 0) incx else -incx);
    const incyu: usize = @intCast(if (incy > 0) incy else -incy);

    var ix: usize = if (incx < 0) (nu - 1) * incxu else 0;
    var iy: usize = if (incy < 0) (nu - 1) * incyu else 0;

    for (0..nu) |_| {
        y[iy] = x[ix];
        if (incx > 0) ix += incxu else if (ix >= incxu) ix -= incxu;
        if (incy > 0) iy += incyu else if (iy >= incyu) iy -= incyu;
    }
}

/// Axpy: y = alpha*x + y
/// n: number of elements
/// alpha: scalar multiplier
/// x: source vector
/// incx: source stride
/// y: destination vector (modified)
/// incy: destination stride
pub export fn blas_daxpy(n: c_int, alpha: f64, x: [*]const f64, incx: c_int, y: [*]f64, incy: c_int) void {
    if (n <= 0 or alpha == 0.0) return;
    const nu: usize = @intCast(n);
    const incxu: usize = @intCast(if (incx > 0) incx else -incx);
    const incyu: usize = @intCast(if (incy > 0) incy else -incy);

    var ix: usize = if (incx < 0) (nu - 1) * incxu else 0;
    var iy: usize = if (incy < 0) (nu - 1) * incyu else 0;

    for (0..nu) |_| {
        y[iy] += alpha * x[ix];
        if (incx > 0) ix += incxu else if (ix >= incxu) ix -= incxu;
        if (incy > 0) iy += incyu else if (iy >= incyu) iy -= incyu;
    }
}

/// Dot product: result = sum(x[i] * y[i])
/// n: number of elements
/// x: first vector
/// incx: first vector stride
/// y: second vector
/// incy: second vector stride
pub export fn blas_ddot(n: c_int, x: [*]const f64, incx: c_int, y: [*]const f64, incy: c_int) f64 {
    if (n <= 0) return 0.0;
    const nu: usize = @intCast(n);
    const incxu: usize = @intCast(if (incx > 0) incx else -incx);
    const incyu: usize = @intCast(if (incy > 0) incy else -incy);

    var ix: usize = if (incx < 0) (nu - 1) * incxu else 0;
    var iy: usize = if (incy < 0) (nu - 1) * incyu else 0;

    var result: f64 = 0.0;
    for (0..nu) |_| {
        result += x[ix] * y[iy];
        if (incx > 0) ix += incxu else if (ix >= incxu) ix -= incxu;
        if (incy > 0) iy += incyu else if (iy >= incyu) iy -= incyu;
    }
    return result;
}

/// Euclidean norm: sqrt(sum(x[i]^2))
/// n: number of elements
/// x: vector
/// incx: stride
pub export fn blas_dnrm2(n: c_int, x: [*]const f64, incx: c_int) f64 {
    if (n <= 0 or incx <= 0) return 0.0;
    const nu: usize = @intCast(n);
    const incu: usize = @intCast(incx);

    // Use scaled sum to avoid overflow/underflow
    var scale: f64 = 0.0;
    var ssq: f64 = 1.0;

    var ix: usize = 0;
    for (0..nu) |_| {
        if (x[ix] != 0.0) {
            const absxi = @abs(x[ix]);
            if (scale < absxi) {
                ssq = 1.0 + ssq * (scale / absxi) * (scale / absxi);
                scale = absxi;
            } else {
                ssq += (absxi / scale) * (absxi / scale);
            }
        }
        ix += incu;
    }
    return scale * math.sqrt(ssq);
}

/// Sum of absolute values: sum(|x[i]|)
/// n: number of elements
/// x: vector
/// incx: stride
pub export fn blas_dasum(n: c_int, x: [*]const f64, incx: c_int) f64 {
    if (n <= 0 or incx <= 0) return 0.0;
    const nu: usize = @intCast(n);
    const incu: usize = @intCast(incx);

    var result: f64 = 0.0;
    var ix: usize = 0;
    for (0..nu) |_| {
        result += @abs(x[ix]);
        ix += incu;
    }
    return result;
}

/// Index of max absolute value (1-based, FORTRAN convention)
/// n: number of elements
/// x: vector
/// incx: stride
/// Returns: 1-based index of element with max absolute value
pub export fn blas_idamax(n: c_int, x: [*]const f64, incx: c_int) c_int {
    if (n <= 0 or incx <= 0) return 0;
    if (n == 1) return 1;

    const nu: usize = @intCast(n);
    const incu: usize = @intCast(incx);

    var max_val: f64 = @abs(x[0]);
    var max_idx: usize = 0;
    var ix: usize = incu;

    for (1..nu) |i| {
        const absval = @abs(x[ix]);
        if (absval > max_val) {
            max_val = absval;
            max_idx = i;
        }
        ix += incu;
    }
    return @intCast(max_idx + 1); // 1-based index
}

// ============================================================================
// BLAS Level 2: Matrix-Vector Operations
// ============================================================================

/// General matrix-vector multiply: y = alpha*op(A)*x + beta*y
/// trans: BLAS_NO_TRANS or BLAS_TRANS
/// m: number of rows of A
/// n: number of columns of A
/// alpha: scalar multiplier for A*x
/// a: matrix A (m x n, column-major)
/// lda: leading dimension of A
/// x: input vector
/// incx: stride of x
/// beta: scalar multiplier for y
/// y: output vector (modified)
/// incy: stride of y
pub export fn blas_dgemv(
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    x: [*]const f64,
    incx: c_int,
    beta: f64,
    y: [*]f64,
    incy: c_int,
) void {
    if (m <= 0 or n <= 0) return;

    const mu: usize = @intCast(m);
    const nu: usize = @intCast(n);
    const ldau: usize = @intCast(lda);
    const incxu: usize = @intCast(if (incx > 0) incx else -incx);
    const incyu: usize = @intCast(if (incy > 0) incy else -incy);

    // Determine dimensions based on transpose
    const len_x: usize = if (trans == BLAS_NO_TRANS) nu else mu;
    const len_y: usize = if (trans == BLAS_NO_TRANS) mu else nu;

    // Scale y by beta
    if (beta == 0.0) {
        var iy: usize = 0;
        for (0..len_y) |_| {
            y[iy] = 0.0;
            iy += incyu;
        }
    } else if (beta != 1.0) {
        var iy: usize = 0;
        for (0..len_y) |_| {
            y[iy] *= beta;
            iy += incyu;
        }
    }

    if (alpha == 0.0) return;

    if (trans == BLAS_NO_TRANS) {
        // y = alpha * A * x + y
        var ix: usize = if (incx < 0) (len_x - 1) * incxu else 0;
        for (0..nu) |j| {
            const temp = alpha * x[ix];
            var iy: usize = if (incy < 0) (len_y - 1) * incyu else 0;
            for (0..mu) |i| {
                y[iy] += temp * a[j * ldau + i];
                iy += incyu;
            }
            if (incx > 0) ix += incxu else if (ix >= incxu) ix -= incxu;
        }
    } else {
        // y = alpha * A^T * x + y
        var iy: usize = if (incy < 0) (len_y - 1) * incyu else 0;
        for (0..nu) |j| {
            var temp: f64 = 0.0;
            var ix: usize = if (incx < 0) (len_x - 1) * incxu else 0;
            for (0..mu) |i| {
                temp += a[j * ldau + i] * x[ix];
                if (incx > 0) ix += incxu else if (ix >= incxu) ix -= incxu;
            }
            y[iy] += alpha * temp;
            if (incy > 0) iy += incyu else if (iy >= incyu) iy -= incyu;
        }
    }
}

// ============================================================================
// BLAS Level 3: Matrix-Matrix Operations
// ============================================================================

/// General matrix-matrix multiply: C = alpha*op(A)*op(B) + beta*C
/// transa: transpose flag for A (BLAS_NO_TRANS or BLAS_TRANS)
/// transb: transpose flag for B (BLAS_NO_TRANS or BLAS_TRANS)
/// m: number of rows of op(A) and C
/// n: number of columns of op(B) and C
/// k: number of columns of op(A) and rows of op(B)
/// alpha: scalar multiplier for A*B
/// a: matrix A (column-major)
/// lda: leading dimension of A
/// b: matrix B (column-major)
/// ldb: leading dimension of B
/// beta: scalar multiplier for C
/// c: matrix C (column-major, modified)
/// ldc: leading dimension of C
pub export fn blas_dgemm(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: f64,
    a: [*]const f64,
    lda: c_int,
    b: [*]const f64,
    ldb: c_int,
    beta: f64,
    c: [*]f64,
    ldc: c_int,
) void {
    if (m <= 0 or n <= 0 or k <= 0) return;

    const mu: usize = @intCast(m);
    const nu: usize = @intCast(n);
    const ku: usize = @intCast(k);
    const ldau: usize = @intCast(lda);
    const ldbu: usize = @intCast(ldb);
    const ldcu: usize = @intCast(ldc);

    // Scale C by beta
    if (beta == 0.0) {
        for (0..nu) |j| {
            for (0..mu) |i| {
                c[j * ldcu + i] = 0.0;
            }
        }
    } else if (beta != 1.0) {
        for (0..nu) |j| {
            for (0..mu) |i| {
                c[j * ldcu + i] *= beta;
            }
        }
    }

    if (alpha == 0.0) return;

    const nota = (transa == BLAS_NO_TRANS);
    const notb = (transb == BLAS_NO_TRANS);

    if (nota and notb) {
        // C = alpha * A * B + C
        for (0..nu) |j| {
            for (0..ku) |l| {
                const temp = alpha * b[j * ldbu + l];
                for (0..mu) |i| {
                    c[j * ldcu + i] += temp * a[l * ldau + i];
                }
            }
        }
    } else if (!nota and notb) {
        // C = alpha * A^T * B + C
        for (0..nu) |j| {
            for (0..mu) |i| {
                var temp: f64 = 0.0;
                for (0..ku) |l| {
                    temp += a[i * ldau + l] * b[j * ldbu + l];
                }
                c[j * ldcu + i] += alpha * temp;
            }
        }
    } else if (nota and !notb) {
        // C = alpha * A * B^T + C
        for (0..nu) |j| {
            for (0..ku) |l| {
                const temp = alpha * b[l * ldbu + j];
                for (0..mu) |i| {
                    c[j * ldcu + i] += temp * a[l * ldau + i];
                }
            }
        }
    } else {
        // C = alpha * A^T * B^T + C
        for (0..nu) |j| {
            for (0..mu) |i| {
                var temp: f64 = 0.0;
                for (0..ku) |l| {
                    temp += a[i * ldau + l] * b[l * ldbu + j];
                }
                c[j * ldcu + i] += alpha * temp;
            }
        }
    }
}

// ============================================================================
// Matrix Decompositions
// ============================================================================

/// LU decomposition with partial pivoting
/// Decomposes A into P*A = L*U where P is a permutation matrix
/// a: input matrix (n x n, row-major) - NOT modified
/// n: matrix dimension
/// pivot: output pivot indices array (n elements)
/// result: output LU matrix (n x n, row-major) - L and U packed together
/// Returns: 0 on success, -1 if singular
pub export fn lu_decompose(
    a: [*]const f64,
    n: c_int,
    pivot: [*]c_int,
    result: [*]f64,
) c_int {
    if (n <= 0) return -1;
    const nu: usize = @intCast(n);

    // Copy a to result
    for (0..nu * nu) |i| {
        result[i] = a[i];
    }

    // Initialize pivot array
    for (0..nu) |i| {
        pivot[i] = @intCast(i);
    }

    for (0..nu) |k| {
        // Find pivot
        var max_val: f64 = @abs(result[k * nu + k]);
        var max_row: usize = k;

        for (k + 1..nu) |i| {
            const absval = @abs(result[i * nu + k]);
            if (absval > max_val) {
                max_val = absval;
                max_row = i;
            }
        }

        // Check for singularity
        if (max_val < 1e-15) {
            return -1; // Singular matrix
        }

        // Swap rows if needed
        if (max_row != k) {
            // Swap pivot entries
            const tmp_pivot = pivot[k];
            pivot[k] = pivot[max_row];
            pivot[max_row] = tmp_pivot;

            // Swap rows in result
            for (0..nu) |j| {
                const tmp = result[k * nu + j];
                result[k * nu + j] = result[max_row * nu + j];
                result[max_row * nu + j] = tmp;
            }
        }

        // Elimination
        const diag = result[k * nu + k];
        for (k + 1..nu) |i| {
            result[i * nu + k] /= diag;
            const factor = result[i * nu + k];
            for (k + 1..nu) |j| {
                result[i * nu + j] -= factor * result[k * nu + j];
            }
        }
    }

    return 0;
}

/// Solve Ax=b using LU decomposition
/// lu: LU factorization from lu_decompose (n x n, row-major)
/// pivot: pivot indices from lu_decompose
/// n: matrix dimension
/// b: right-hand side vector (n elements)
/// x: solution vector (n elements, modified)
/// Returns: 0 on success
pub export fn lu_solve(
    lu: [*]const f64,
    pivot: [*]const c_int,
    n: c_int,
    b: [*]const f64,
    x: [*]f64,
) c_int {
    if (n <= 0) return -1;
    const nu: usize = @intCast(n);

    // Apply permutation and forward substitution (Ly = Pb)
    for (0..nu) |i| {
        const pi: usize = @intCast(pivot[i]);
        var sum: f64 = b[pi];
        for (0..i) |j| {
            sum -= lu[i * nu + j] * x[j];
        }
        x[i] = sum;
    }

    // Back substitution (Ux = y)
    var i: usize = nu;
    while (i > 0) {
        i -= 1;
        var sum: f64 = x[i];
        for (i + 1..nu) |j| {
            sum -= lu[i * nu + j] * x[j];
        }
        x[i] = sum / lu[i * nu + i];
    }

    return 0;
}

/// Cholesky decomposition for symmetric positive-definite matrices
/// Decomposes A into A = L * L^T where L is lower triangular
/// a: input matrix (n x n, row-major) - symmetric positive-definite
/// n: matrix dimension
/// result: output lower triangular matrix L (n x n, row-major)
/// Returns: 0 on success, -1 if not positive-definite
pub export fn cholesky_decompose(
    a: [*]const f64,
    n: c_int,
    result: [*]f64,
) c_int {
    if (n <= 0) return -1;
    const nu: usize = @intCast(n);

    // Initialize result to zero
    for (0..nu * nu) |i| {
        result[i] = 0.0;
    }

    for (0..nu) |i| {
        for (0..i + 1) |j| {
            var sum: f64 = 0.0;

            if (j == i) {
                // Diagonal element
                for (0..j) |k| {
                    sum += result[j * nu + k] * result[j * nu + k];
                }
                const diag = a[j * nu + j] - sum;
                if (diag <= 0.0) {
                    return -1; // Not positive-definite
                }
                result[j * nu + j] = math.sqrt(diag);
            } else {
                // Off-diagonal element
                for (0..j) |k| {
                    sum += result[i * nu + k] * result[j * nu + k];
                }
                result[i * nu + j] = (a[i * nu + j] - sum) / result[j * nu + j];
            }
        }
    }

    return 0;
}

/// QR decomposition using Gram-Schmidt process
/// Decomposes A into A = Q * R where Q is orthogonal and R is upper triangular
/// a: input matrix (m x n, row-major)
/// m: number of rows
/// n: number of columns (m >= n required)
/// q: output orthogonal matrix Q (m x n, row-major)
/// r: output upper triangular matrix R (n x n, row-major)
/// Returns: 0 on success, -1 on error
pub export fn qr_decompose(
    a: [*]const f64,
    m: c_int,
    n: c_int,
    q: [*]f64,
    r: [*]f64,
) c_int {
    if (m <= 0 or n <= 0 or m < n) return -1;
    const mu: usize = @intCast(m);
    const nu: usize = @intCast(n);

    // Initialize R to zero
    for (0..nu * nu) |i| {
        r[i] = 0.0;
    }

    // Copy A to Q
    for (0..mu * nu) |i| {
        q[i] = a[i];
    }

    // Modified Gram-Schmidt
    for (0..nu) |j| {
        // Compute norm of column j
        var norm: f64 = 0.0;
        for (0..mu) |i| {
            norm += q[i * nu + j] * q[i * nu + j];
        }
        norm = math.sqrt(norm);

        if (norm < 1e-15) {
            return -1; // Linearly dependent columns
        }

        r[j * nu + j] = norm;

        // Normalize column j
        const inv_norm = 1.0 / norm;
        for (0..mu) |i| {
            q[i * nu + j] *= inv_norm;
        }

        // Orthogonalize remaining columns
        for (j + 1..nu) |k| {
            // Compute dot product of column j and column k
            var dot: f64 = 0.0;
            for (0..mu) |i| {
                dot += q[i * nu + j] * q[i * nu + k];
            }

            r[j * nu + k] = dot;

            // Subtract projection
            for (0..mu) |i| {
                q[i * nu + k] -= dot * q[i * nu + j];
            }
        }
    }

    return 0;
}

// ============================================================================
// Sparse Matrix (COO Format)
// ============================================================================

/// Sparse matrix in COO (Coordinate) format
pub const SparseMatrix = extern struct {
    /// Row indices (0-based)
    row_indices: [*]c_int,
    /// Column indices (0-based)
    col_indices: [*]c_int,
    /// Non-zero values
    values: [*]f64,
    /// Number of non-zero elements
    nnz: c_int,
    /// Number of rows
    nrows: c_int,
    /// Number of columns
    ncols: c_int,
};

/// Sparse matrix-vector multiply: result = sparse * x
/// sparse: pointer to SparseMatrix structure
/// x: input vector (ncols elements)
/// result: output vector (nrows elements, modified)
/// n: size of result vector (should be >= sparse.nrows)
/// Returns: 0 on success
pub export fn sparse_matvec(
    sparse: *const SparseMatrix,
    x: [*]const f64,
    result: [*]f64,
    n: c_int,
) c_int {
    if (n <= 0) return -1;
    const nu: usize = @intCast(n);
    const nnzu: usize = @intCast(sparse.nnz);

    // Initialize result to zero
    for (0..nu) |i| {
        result[i] = 0.0;
    }

    // Accumulate contributions from non-zero elements
    for (0..nnzu) |k| {
        const row: usize = @intCast(sparse.row_indices[k]);
        const col: usize = @intCast(sparse.col_indices[k]);
        const val = sparse.values[k];
        result[row] += val * x[col];
    }

    return 0;
}

// ============================================================================
// Tests
// ============================================================================

test "blas_dscal" {
    var x = [_]f64{ 1.0, 2.0, 3.0, 4.0 };
    blas_dscal(4, 2.0, &x, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), x[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 4.0), x[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), x[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 8.0), x[3], 1e-10);
}

test "blas_dcopy" {
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    var y: [3]f64 = undefined;
    blas_dcopy(3, &x, 1, &y, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), y[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), y[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), y[2], 1e-10);
}

test "blas_daxpy" {
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    var y = [_]f64{ 4.0, 5.0, 6.0 };
    blas_daxpy(3, 2.0, &x, 1, &y, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), y[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 9.0), y[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 12.0), y[2], 1e-10);
}

test "blas_ddot" {
    const x = [_]f64{ 1.0, 2.0, 3.0 };
    const y = [_]f64{ 4.0, 5.0, 6.0 };
    const dot = blas_ddot(3, &x, 1, &y, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 32.0), dot, 1e-10);
}

test "blas_dnrm2" {
    const x = [_]f64{ 3.0, 4.0 };
    const nrm = blas_dnrm2(2, &x, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), nrm, 1e-10);
}

test "blas_dasum" {
    const x = [_]f64{ -1.0, 2.0, -3.0 };
    const asum = blas_dasum(3, &x, 1);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), asum, 1e-10);
}

test "blas_idamax" {
    const x = [_]f64{ 1.0, -5.0, 3.0, 2.0 };
    const idx = blas_idamax(4, &x, 1);
    try std.testing.expectEqual(@as(c_int, 2), idx); // 1-based index
}

test "lu_decompose and lu_solve" {
    // A = [[4, 3], [6, 3]]
    const a = [_]f64{ 4.0, 3.0, 6.0, 3.0 };
    var lu: [4]f64 = undefined;
    var pivot: [2]c_int = undefined;

    const result = lu_decompose(&a, 2, &pivot, &lu);
    try std.testing.expectEqual(@as(c_int, 0), result);

    // Solve Ax = b where b = [7, 9]
    const b = [_]f64{ 7.0, 9.0 };
    var x: [2]f64 = undefined;
    const solve_result = lu_solve(&lu, &pivot, 2, &b, &x);
    try std.testing.expectEqual(@as(c_int, 0), solve_result);

    // Verify solution: x should be [1, 1]
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), x[1], 1e-10);
}

test "cholesky_decompose" {
    // A = [[4, 2], [2, 10]] (symmetric positive-definite)
    const a = [_]f64{ 4.0, 2.0, 2.0, 10.0 };
    var l: [4]f64 = undefined;

    const result = cholesky_decompose(&a, 2, &l);
    try std.testing.expectEqual(@as(c_int, 0), result);

    // L should be [[2, 0], [1, 3]]
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), l[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 0.0), l[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 1.0), l[2], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), l[3], 1e-10);
}

test "sparse_matvec" {
    // Sparse matrix: [[1, 0, 2], [0, 3, 0], [4, 0, 5]]
    var rows = [_]c_int{ 0, 0, 1, 2, 2 };
    var cols = [_]c_int{ 0, 2, 1, 0, 2 };
    var vals = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };

    const sparse = SparseMatrix{
        .row_indices = &rows,
        .col_indices = &cols,
        .values = &vals,
        .nnz = 5,
        .nrows = 3,
        .ncols = 3,
    };

    const x = [_]f64{ 1.0, 2.0, 3.0 };
    var result: [3]f64 = undefined;

    const ret = sparse_matvec(&sparse, &x, &result, 3);
    try std.testing.expectEqual(@as(c_int, 0), ret);

    // result = [1*1 + 2*3, 3*2, 4*1 + 5*3] = [7, 6, 19]
    try std.testing.expectApproxEqAbs(@as(f64, 7.0), result[0], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 6.0), result[1], 1e-10);
    try std.testing.expectApproxEqAbs(@as(f64, 19.0), result[2], 1e-10);
}

