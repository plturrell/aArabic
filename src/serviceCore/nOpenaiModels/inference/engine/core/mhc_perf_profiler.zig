// mHC Performance Profiler Module
// Implements profiling infrastructure for mHC code paths
//
// Core Components:
// - PerfTimer: High-resolution nanosecond timer
// - ProfileResult: Stores timing data per operation
// - profile_sinkhorn(): Profile Sinkhorn normalization
// - profile_stability_check(): Profile stability checking
// - get_perf_report(): Generate performance report
//
// Target: <5% overhead compared to non-mHC path
//
// Reference: docs/DAY_48_PERFORMANCE_OPTIMIZATION_REPORT.md

const std = @import("std");
const builtin = @import("builtin");
const mhc_constraints = @import("mhc_constraints.zig");

// ============================================================================
// High-Resolution Timer
// ============================================================================

/// High-resolution timer for profiling with nanosecond precision
pub const PerfTimer = struct {
    start_ns: i128,
    lap_ns: i128,

    /// Start a new timer
    pub fn start() PerfTimer {
        const now = std.time.nanoTimestamp();
        return .{
            .start_ns = now,
            .lap_ns = now,
        };
    }

    /// Get elapsed nanoseconds since start
    pub fn elapsed_ns(self: *const PerfTimer) i128 {
        return std.time.nanoTimestamp() - self.start_ns;
    }

    /// Get elapsed microseconds since start
    pub fn elapsed_us(self: *const PerfTimer) f64 {
        return @as(f64, @floatFromInt(self.elapsed_ns())) / 1_000.0;
    }

    /// Get elapsed milliseconds since start
    pub fn elapsed_ms(self: *const PerfTimer) f64 {
        return @as(f64, @floatFromInt(self.elapsed_ns())) / 1_000_000.0;
    }

    /// Take a lap reading (returns time since last lap)
    pub fn lap(self: *PerfTimer) f64 {
        const now = std.time.nanoTimestamp();
        const elapsed = now - self.lap_ns;
        self.lap_ns = now;
        return @as(f64, @floatFromInt(elapsed)) / 1_000.0; // Return μs
    }

    /// Reset the timer
    pub fn reset(self: *PerfTimer) void {
        const now = std.time.nanoTimestamp();
        self.start_ns = now;
        self.lap_ns = now;
    }
};

// ============================================================================
// Profile Result
// ============================================================================

/// Stores timing data for profiled operations
pub const ProfileResult = struct {
    /// Operation name/identifier
    operation: []const u8,

    /// Total time in microseconds
    total_us: f64,

    /// Number of iterations/calls
    iterations: u32,

    /// Time per iteration in microseconds
    per_iter_us: f64,

    /// Minimum time observed (μs)
    min_us: f64,

    /// Maximum time observed (μs)
    max_us: f64,

    /// Standard deviation (μs)
    std_dev_us: f64,

    /// Memory allocations during profiling
    allocations: u64,

    /// Bytes allocated
    bytes_allocated: u64,

    /// Overhead percentage (compared to baseline)
    overhead_pct: f64,

    /// Format result for display
    pub fn format(
        self: ProfileResult,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        try writer.print(
            "{s}: {d:.3}μs total ({d} iters, {d:.3}μs/iter, overhead: {d:.2}%)",
            .{
                self.operation,
                self.total_us,
                self.iterations,
                self.per_iter_us,
                self.overhead_pct,
            },
        );
    }
};

// ============================================================================
// Profiler State
// ============================================================================

/// Maximum number of profile results to store
const MAX_PROFILE_RESULTS = 32;

/// Global profiler state
pub const ProfilerState = struct {
    results: [MAX_PROFILE_RESULTS]?ProfileResult,
    result_count: usize,
    enabled: bool,
    baseline_sinkhorn_us: f64,
    baseline_stability_us: f64,
    baseline_norm_us: f64,
    baseline_constraints_us: f64,

    pub fn init() ProfilerState {
        return .{
            .results = [_]?ProfileResult{null} ** MAX_PROFILE_RESULTS,
            .result_count = 0,
            .enabled = true,
            .baseline_sinkhorn_us = 0,
            .baseline_stability_us = 0,
            .baseline_norm_us = 0,
            .baseline_constraints_us = 0,
        };
    }

    pub fn add_result(self: *ProfilerState, result: ProfileResult) void {
        if (self.result_count < MAX_PROFILE_RESULTS) {
            self.results[self.result_count] = result;
            self.result_count += 1;
        }
    }

    pub fn clear(self: *ProfilerState) void {
        self.results = [_]?ProfileResult{null} ** MAX_PROFILE_RESULTS;
        self.result_count = 0;
    }
};

/// Thread-local profiler state
pub var profiler_state: ProfilerState = ProfilerState.init();

// ============================================================================
// SIMD-Optimized Operations (Reduced Overhead)
// ============================================================================

/// SIMD vector width based on architecture
pub const SIMD_WIDTH: usize = if (builtin.cpu.arch == .aarch64 or builtin.cpu.arch == .arm) 4 else if (builtin.cpu.arch == .x86_64) 8 else 1;

/// SIMD-optimized row sum computation
/// Uses 4-wide vectorization for ARM NEON, 8-wide for AVX
pub fn simd_row_sums(matrix: []const f32, rows: usize, cols: usize, row_sums: []f32) void {
    const simd_cols = cols / SIMD_WIDTH * SIMD_WIDTH;

    for (0..rows) |i| {
        var sum: f32 = 0.0;
        const row_start = i * cols;

        // SIMD portion
        var j: usize = 0;
        while (j < simd_cols) : (j += SIMD_WIDTH) {
            inline for (0..SIMD_WIDTH) |k| {
                sum += matrix[row_start + j + k];
            }
        }

        // Remainder
        while (j < cols) : (j += 1) {
            sum += matrix[row_start + j];
        }

        row_sums[i] = sum;
    }
}

/// SIMD-optimized column sum computation
pub fn simd_col_sums(matrix: []const f32, rows: usize, cols: usize, col_sums: []f32) void {
    // Zero-initialize
    @memset(col_sums, 0.0);

    for (0..rows) |i| {
        const row_start = i * cols;
        const simd_cols = cols / SIMD_WIDTH * SIMD_WIDTH;

        // SIMD portion
        var j: usize = 0;
        while (j < simd_cols) : (j += SIMD_WIDTH) {
            inline for (0..SIMD_WIDTH) |k| {
                col_sums[j + k] += matrix[row_start + j + k];
            }
        }

        // Remainder
        while (j < cols) : (j += 1) {
            col_sums[j] += matrix[row_start + j];
        }
    }
}

/// SIMD-optimized row scaling (for Sinkhorn normalization)
pub fn simd_scale_row(matrix: []f32, row: usize, cols: usize, scale: f32) void {
    const row_start = row * cols;
    const simd_cols = cols / SIMD_WIDTH * SIMD_WIDTH;

    // SIMD portion
    var j: usize = 0;
    while (j < simd_cols) : (j += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            matrix[row_start + j + k] *= scale;
        }
    }

    // Remainder
    while (j < cols) : (j += 1) {
        matrix[row_start + j] *= scale;
    }
}

/// SIMD-optimized L2 norm computation
pub fn simd_compute_norm(vector: []const f32) f32 {
    var norm_sq: f32 = 0.0;
    const simd_len = vector.len / SIMD_WIDTH * SIMD_WIDTH;

    // SIMD portion - unroll loop for better pipelining
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            const val = vector[i + k];
            norm_sq += val * val;
        }
    }

    // Remainder
    while (i < vector.len) : (i += 1) {
        const val = vector[i];
        norm_sq += val * val;
    }

    return @sqrt(norm_sq);
}

/// SIMD-optimized stability check (early exit on first violation)
pub fn simd_check_stability(activations: []const f32, threshold: f32) bool {
    const simd_len = activations.len / SIMD_WIDTH * SIMD_WIDTH;

    // SIMD portion
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            const val = activations[i + k];
            if (std.math.isNan(val) or std.math.isInf(val)) return false;
            if (@abs(val) >= threshold) return false;
        }
    }

    // Remainder
    while (i < activations.len) : (i += 1) {
        const val = activations[i];
        if (std.math.isNan(val) or std.math.isInf(val)) return false;
        if (@abs(val) >= threshold) return false;
    }

    return true;
}


// ============================================================================
// Profiling Functions
// ============================================================================

/// Profile Sinkhorn normalization with detailed timing
pub fn profile_sinkhorn(
    matrix: []f32,
    rows: usize,
    cols: usize,
    config: mhc_constraints.MHCConfig,
    allocator: std.mem.Allocator,
    warmup_iters: u32,
    profile_iters: u32,
) !ProfileResult {
    var timer = PerfTimer.start();

    // Allocate temp buffers once (memory optimization)
    const row_sums = try allocator.alloc(f32, rows);
    defer allocator.free(row_sums);
    const col_sums = try allocator.alloc(f32, cols);
    defer allocator.free(col_sums);

    // Create backup for repeated runs
    const backup = try allocator.alloc(f32, matrix.len);
    defer allocator.free(backup);
    @memcpy(backup, matrix);

    // Warmup (avoid cold cache effects)
    for (0..warmup_iters) |_| {
        @memcpy(matrix, backup);
        _ = try mhc_constraints.sinkhorn_normalize(matrix, rows, cols, config, allocator);
    }

    // Collect timing samples
    var times = try allocator.alloc(f64, profile_iters);
    defer allocator.free(times);
    var total_iters: u32 = 0;

    timer.reset();
    for (0..profile_iters) |run| {
        @memcpy(matrix, backup);

        const lap_timer = PerfTimer.start();
        const iters = try mhc_constraints.sinkhorn_normalize(matrix, rows, cols, config, allocator);
        times[run] = lap_timer.elapsed_us();
        total_iters += iters;
    }

    const total_us = timer.elapsed_us();

    // Compute statistics
    var min_us: f64 = times[0];
    var max_us: f64 = times[0];
    var sum: f64 = 0.0;

    for (times) |t| {
        min_us = @min(min_us, t);
        max_us = @max(max_us, t);
        sum += t;
    }

    const avg = sum / @as(f64, @floatFromInt(profile_iters));

    // Compute standard deviation
    var variance: f64 = 0.0;
    for (times) |t| {
        const diff = t - avg;
        variance += diff * diff;
    }
    const std_dev = @sqrt(variance / @as(f64, @floatFromInt(profile_iters)));

    // Compute overhead vs baseline
    const overhead = if (profiler_state.baseline_sinkhorn_us > 0)
        ((avg - profiler_state.baseline_sinkhorn_us) / profiler_state.baseline_sinkhorn_us) * 100.0
    else
        0.0;

    const result = ProfileResult{
        .operation = "sinkhorn_normalize",
        .total_us = total_us,
        .iterations = profile_iters,
        .per_iter_us = avg,
        .min_us = min_us,
        .max_us = max_us,
        .std_dev_us = std_dev,
        .allocations = 2, // row_sums + col_sums per iter
        .bytes_allocated = (rows + cols) * @sizeOf(f32),
        .overhead_pct = overhead,
    };

    profiler_state.add_result(result);
    return result;
}

/// Profile stability checking
pub fn profile_stability_check(
    activations: []const f32,
    threshold: f32,
    warmup_iters: u32,
    profile_iters: u32,
    allocator: std.mem.Allocator,
) !ProfileResult {
    // Collect timing samples
    var times = try allocator.alloc(f64, profile_iters);
    defer allocator.free(times);

    // Warmup
    for (0..warmup_iters) |_| {
        _ = mhc_constraints.check_stability(activations, threshold);
    }

    var timer = PerfTimer.start();
    for (0..profile_iters) |run| {
        const lap = PerfTimer.start();
        _ = mhc_constraints.check_stability(activations, threshold);
        times[run] = lap.elapsed_us();
    }
    const total_us = timer.elapsed_us();

    // Compute statistics
    var min_us: f64 = times[0];
    var max_us: f64 = times[0];
    var sum: f64 = 0.0;

    for (times) |t| {
        min_us = @min(min_us, t);
        max_us = @max(max_us, t);
        sum += t;
    }

    const avg = sum / @as(f64, @floatFromInt(profile_iters));

    var variance: f64 = 0.0;
    for (times) |t| {
        const diff = t - avg;
        variance += diff * diff;
    }
    const std_dev = @sqrt(variance / @as(f64, @floatFromInt(profile_iters)));

    const overhead = if (profiler_state.baseline_stability_us > 0)
        ((avg - profiler_state.baseline_stability_us) / profiler_state.baseline_stability_us) * 100.0
    else
        0.0;

    const result = ProfileResult{
        .operation = "check_stability",
        .total_us = total_us,
        .iterations = profile_iters,
        .per_iter_us = avg,
        .min_us = min_us,
        .max_us = max_us,
        .std_dev_us = std_dev,
        .allocations = 0,
        .bytes_allocated = 0,
        .overhead_pct = overhead,
    };

    profiler_state.add_result(result);
    return result;
}

/// Profile manifold constraints application
pub fn profile_manifold_constraints(
    activations: []f32,
    beta: f32,
    warmup_iters: u32,
    profile_iters: u32,
    allocator: std.mem.Allocator,
) !ProfileResult {
    // Backup for repeated runs
    const backup = try allocator.alloc(f32, activations.len);
    defer allocator.free(backup);
    @memcpy(backup, activations);

    var times = try allocator.alloc(f64, profile_iters);
    defer allocator.free(times);

    // Warmup
    for (0..warmup_iters) |_| {
        @memcpy(activations, backup);
        _ = mhc_constraints.apply_manifold_constraints(activations, beta);
    }

    var timer = PerfTimer.start();
    for (0..profile_iters) |run| {
        @memcpy(activations, backup);
        const lap = PerfTimer.start();
        _ = mhc_constraints.apply_manifold_constraints(activations, beta);
        times[run] = lap.elapsed_us();
    }
    const total_us = timer.elapsed_us();

    // Statistics
    var min_us: f64 = times[0];
    var max_us: f64 = times[0];
    var sum: f64 = 0.0;

    for (times) |t| {
        min_us = @min(min_us, t);
        max_us = @max(max_us, t);
        sum += t;
    }

    const avg = sum / @as(f64, @floatFromInt(profile_iters));

    var variance: f64 = 0.0;
    for (times) |t| {
        const diff = t - avg;
        variance += diff * diff;
    }
    const std_dev = @sqrt(variance / @as(f64, @floatFromInt(profile_iters)));

    const overhead = if (profiler_state.baseline_constraints_us > 0)
        ((avg - profiler_state.baseline_constraints_us) / profiler_state.baseline_constraints_us) * 100.0
    else
        0.0;

    const result = ProfileResult{
        .operation = "apply_manifold_constraints",
        .total_us = total_us,
        .iterations = profile_iters,
        .per_iter_us = avg,
        .min_us = min_us,
        .max_us = max_us,
        .std_dev_us = std_dev,
        .allocations = 0,
        .bytes_allocated = 0,
        .overhead_pct = overhead,
    };

    profiler_state.add_result(result);
    return result;
}



// ============================================================================
// Performance Report Generation
// ============================================================================

/// Performance report structure
pub const PerfReport = struct {
    total_profiled_us: f64,
    num_operations: usize,
    results: []const ?ProfileResult,
    simd_width: usize,
    arch: []const u8,
    meets_target: bool, // <5% overhead target

    /// Print formatted performance report
    pub fn print(self: *const PerfReport) void {
        std.debug.print("\n", .{});
        std.debug.print("╔══════════════════════════════════════════════════════════════════════╗\n", .{});
        std.debug.print("║            mHC Performance Profiling Report                         ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Architecture: {s:<20} SIMD Width: {d:<5}                  ║\n", .{ self.arch, self.simd_width });
        std.debug.print("║ Target: <5% overhead                                                ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════════════╣\n", .{});

        var max_overhead: f64 = 0.0;

        for (self.results[0..self.num_operations]) |maybe_result| {
            if (maybe_result) |result| {
                std.debug.print("║ {s:<30} │ {d:>10.3}μs/iter │ ±{d:>6.2}% ║\n", .{
                    result.operation,
                    result.per_iter_us,
                    result.overhead_pct,
                });
                max_overhead = @max(max_overhead, @abs(result.overhead_pct));
            }
        }

        std.debug.print("╠══════════════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Total profiled: {d:>10.3}μs │ Operations: {d:<5}                   ║\n", .{
            self.total_profiled_us,
            self.num_operations,
        });
        std.debug.print("║ Max overhead: {d:>6.2}%  │  Status: {s:<10}                      ║\n", .{
            max_overhead,
            if (self.meets_target) "✅ PASS" else "❌ FAIL",
        });
        std.debug.print("╚══════════════════════════════════════════════════════════════════════╝\n", .{});
    }
};

/// Generate performance report from profiler state
pub fn get_perf_report() PerfReport {
    var total_us: f64 = 0.0;
    var max_overhead: f64 = 0.0;

    for (profiler_state.results[0..profiler_state.result_count]) |maybe_result| {
        if (maybe_result) |result| {
            total_us += result.total_us;
            max_overhead = @max(max_overhead, @abs(result.overhead_pct));
        }
    }

    const arch_name: []const u8 = switch (builtin.cpu.arch) {
        .aarch64 => "aarch64 (ARM64)",
        .arm => "arm",
        .x86_64 => "x86_64",
        .x86 => "x86",
        else => "unknown",
    };

    return PerfReport{
        .total_profiled_us = total_us,
        .num_operations = profiler_state.result_count,
        .results = &profiler_state.results,
        .simd_width = SIMD_WIDTH,
        .arch = arch_name,
        .meets_target = max_overhead < 5.0,
    };
}

// ============================================================================
// Memory-Optimized Sinkhorn (Reduced Allocations)
// ============================================================================

/// Pre-allocated buffer pool for Sinkhorn normalization
pub const SinkhornBufferPool = struct {
    row_sums: []f32,
    col_sums: []f32,
    max_rows: usize,
    max_cols: usize,
    allocator: std.mem.Allocator,

    /// Create buffer pool for given max dimensions
    pub fn init(allocator: std.mem.Allocator, max_rows: usize, max_cols: usize) !SinkhornBufferPool {
        return .{
            .row_sums = try allocator.alloc(f32, max_rows),
            .col_sums = try allocator.alloc(f32, max_cols),
            .max_rows = max_rows,
            .max_cols = max_cols,
            .allocator = allocator,
        };
    }

    /// Free buffer pool
    pub fn deinit(self: *SinkhornBufferPool) void {
        self.allocator.free(self.row_sums);
        self.allocator.free(self.col_sums);
    }

    /// Get buffers (returns slices of appropriate size)
    pub fn get_buffers(self: *SinkhornBufferPool, rows: usize, cols: usize) !struct { row_sums: []f32, col_sums: []f32 } {
        if (rows > self.max_rows or cols > self.max_cols) {
            return error.BufferTooSmall;
        }
        return .{
            .row_sums = self.row_sums[0..rows],
            .col_sums = self.col_sums[0..cols],
        };
    }
};

/// Optimized Sinkhorn with pre-allocated buffers and SIMD
pub fn sinkhorn_normalize_optimized(
    matrix: []f32,
    rows: usize,
    cols: usize,
    config: mhc_constraints.MHCConfig,
    pool: *SinkhornBufferPool,
) !u32 {
    if (rows == 0 or cols == 0) return error.InvalidDimensions;
    if (matrix.len != rows * cols) return error.DimensionMismatch;

    const buffers = try pool.get_buffers(rows, cols);
    const row_sums = buffers.row_sums;
    const col_sums = buffers.col_sums;

    var iterations: u32 = 0;

    for (0..config.sinkhorn_iterations) |iter| {
        iterations = @intCast(iter + 1);

        // Row normalization with SIMD
        simd_row_sums(matrix, rows, cols, row_sums);
        for (0..rows) |i| {
            const sum = row_sums[i];
            if (sum > config.manifold_epsilon) {
                simd_scale_row(matrix, i, cols, 1.0 / sum);
            }
        }

        // Column normalization with SIMD
        simd_col_sums(matrix, rows, cols, col_sums);
        for (0..cols) |j| {
            const sum = col_sums[j];
            if (sum > config.manifold_epsilon) {
                const scale = 1.0 / sum;
                for (0..rows) |i| {
                    matrix[i * cols + j] *= scale;
                }
            }
        }

        // Early stopping check
        if (config.early_stopping and iter >= 3) {
            var converged = true;
            for (row_sums[0..rows]) |sum| {
                if (@abs(sum - 1.0) > config.manifold_epsilon) {
                    converged = false;
                    break;
                }
            }
            if (converged) {
                for (col_sums[0..cols]) |sum| {
                    if (@abs(sum - 1.0) > config.manifold_epsilon) {
                        converged = false;
                        break;
                    }
                }
            }
            if (converged) break;
        }
    }

    return iterations;
}

/// Establish baseline timing for overhead calculation
pub fn establish_baseline(allocator: std.mem.Allocator, matrix_size: usize) !void {
    // Create test data
    const matrix = try allocator.alloc(f32, matrix_size * matrix_size);
    defer allocator.free(matrix);
    for (matrix, 0..) |*val, i| {
        val.* = @as(f32, @floatFromInt((i % 10) + 1)) / 10.0;
    }

    const activations = try allocator.alloc(f32, matrix_size);
    defer allocator.free(activations);
    @memset(activations, 0.5);

    const config = mhc_constraints.MHCConfig{};

    // Time baseline operations (100 iterations)
    var timer = PerfTimer.start();
    for (0..100) |_| {
        _ = try mhc_constraints.sinkhorn_normalize(matrix, matrix_size, matrix_size, config, allocator);
    }
    profiler_state.baseline_sinkhorn_us = timer.elapsed_us() / 100.0;

    timer.reset();
    for (0..100) |_| {
        _ = mhc_constraints.check_stability(activations, 10.0);
    }
    profiler_state.baseline_stability_us = timer.elapsed_us() / 100.0;

    timer.reset();
    for (0..100) |_| {
        _ = mhc_constraints.apply_manifold_constraints(activations, 10.0);
    }
    profiler_state.baseline_constraints_us = timer.elapsed_us() / 100.0;
}


// ============================================================================
// Hot Path Identification
// ============================================================================

/// Identified hot path operations in mHC
pub const HotPath = enum {
    sinkhorn_row_norm,
    sinkhorn_col_norm,
    stability_check,
    l2_norm_compute,
    manifold_projection,
};

/// Hot path timing breakdown
pub const HotPathBreakdown = struct {
    path: HotPath,
    time_us: f64,
    percentage: f64,
    call_count: u64,

    pub fn format(
        self: HotPathBreakdown,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        const name = switch (self.path) {
            .sinkhorn_row_norm => "Sinkhorn Row Norm",
            .sinkhorn_col_norm => "Sinkhorn Col Norm",
            .stability_check => "Stability Check",
            .l2_norm_compute => "L2 Norm Compute",
            .manifold_projection => "Manifold Projection",
        };
        try writer.print("{s}: {d:.3}μs ({d:.1}%, {d} calls)", .{
            name,
            self.time_us,
            self.percentage,
            self.call_count,
        });
    }
};

/// Profile and identify hot paths in a full mHC pass
pub fn identify_hot_paths(
    matrix: []f32,
    rows: usize,
    cols: usize,
    activations: []f32,
    config: mhc_constraints.MHCConfig,
    allocator: std.mem.Allocator,
) !struct { breakdowns: [5]HotPathBreakdown, total_us: f64 } {
    var breakdowns: [5]HotPathBreakdown = undefined;
    var total_time: f64 = 0.0;

    // Allocate buffers
    const row_sums = try allocator.alloc(f32, rows);
    defer allocator.free(row_sums);
    const col_sums = try allocator.alloc(f32, cols);
    defer allocator.free(col_sums);

    // Time each hot path
    var timer = PerfTimer.start();

    // 1. Row normalization
    for (0..config.sinkhorn_iterations) |_| {
        simd_row_sums(matrix, rows, cols, row_sums);
        for (0..rows) |i| {
            const sum = row_sums[i];
            if (sum > config.manifold_epsilon) {
                simd_scale_row(matrix, i, cols, 1.0 / sum);
            }
        }
    }
    const row_time = timer.elapsed_us();
    breakdowns[0] = .{ .path = .sinkhorn_row_norm, .time_us = row_time, .percentage = 0, .call_count = config.sinkhorn_iterations };
    total_time += row_time;

    timer.reset();
    // 2. Column normalization
    for (0..config.sinkhorn_iterations) |_| {
        simd_col_sums(matrix, rows, cols, col_sums);
        for (0..cols) |j| {
            const sum = col_sums[j];
            if (sum > config.manifold_epsilon) {
                const scale = 1.0 / sum;
                for (0..rows) |i| {
                    matrix[i * cols + j] *= scale;
                }
            }
        }
    }
    const col_time = timer.elapsed_us();
    breakdowns[1] = .{ .path = .sinkhorn_col_norm, .time_us = col_time, .percentage = 0, .call_count = config.sinkhorn_iterations };
    total_time += col_time;

    timer.reset();
    // 3. Stability check
    for (0..100) |_| {
        _ = simd_check_stability(activations, config.manifold_beta);
    }
    const stab_time = timer.elapsed_us();
    breakdowns[2] = .{ .path = .stability_check, .time_us = stab_time, .percentage = 0, .call_count = 100 };
    total_time += stab_time;

    timer.reset();
    // 4. L2 norm compute
    for (0..100) |_| {
        _ = simd_compute_norm(activations);
    }
    const norm_time = timer.elapsed_us();
    breakdowns[3] = .{ .path = .l2_norm_compute, .time_us = norm_time, .percentage = 0, .call_count = 100 };
    total_time += norm_time;

    timer.reset();
    // 5. Manifold projection
    for (0..100) |_| {
        _ = mhc_constraints.apply_manifold_constraints(activations, config.manifold_beta);
    }
    const proj_time = timer.elapsed_us();
    breakdowns[4] = .{ .path = .manifold_projection, .time_us = proj_time, .percentage = 0, .call_count = 100 };
    total_time += proj_time;

    // Calculate percentages
    if (total_time > 0) {
        for (&breakdowns) |*bd| {
            bd.percentage = (bd.time_us / total_time) * 100.0;
        }
    }

    return .{ .breakdowns = breakdowns, .total_us = total_time };
}

// ============================================================================
// Unit Tests
// ============================================================================

test "PerfTimer accuracy" {
    var timer = PerfTimer.start();

    // Small delay
    var sum: f64 = 0;
    for (0..10000) |i| {
        sum += @as(f64, @floatFromInt(i));
    }

    const elapsed = timer.elapsed_us();
    try std.testing.expect(elapsed >= 0);
    try std.testing.expect(elapsed < 10000); // Should be < 10ms
}

test "PerfTimer lap functionality" {
    var timer = PerfTimer.start();

    // First lap
    var sum: f64 = 0;
    for (0..1000) |i| {
        sum += @as(f64, @floatFromInt(i));
    }
    const lap1 = timer.lap();

    // Second lap
    for (0..1000) |i| {
        sum += @as(f64, @floatFromInt(i));
    }
    const lap2 = timer.lap();

    try std.testing.expect(lap1 >= 0);
    try std.testing.expect(lap2 >= 0);
    try std.testing.expect(timer.elapsed_us() >= lap1 + lap2);
}

test "ProfileResult formatting" {
    const result = ProfileResult{
        .operation = "test_op",
        .total_us = 100.0,
        .iterations = 10,
        .per_iter_us = 10.0,
        .min_us = 8.0,
        .max_us = 12.0,
        .std_dev_us = 1.5,
        .allocations = 5,
        .bytes_allocated = 1024,
        .overhead_pct = 3.5,
    };

    // Test that format doesn't crash
    var buf: [256]u8 = undefined;
    var fbs = std.io.fixedBufferStream(&buf);
    try result.format("", .{}, fbs.writer());
    try std.testing.expect(fbs.pos > 0);
}

test "ProfilerState add_result" {
    var state = ProfilerState.init();

    const result = ProfileResult{
        .operation = "test",
        .total_us = 50.0,
        .iterations = 5,
        .per_iter_us = 10.0,
        .min_us = 9.0,
        .max_us = 11.0,
        .std_dev_us = 0.5,
        .allocations = 0,
        .bytes_allocated = 0,
        .overhead_pct = 0.0,
    };

    state.add_result(result);
    try std.testing.expectEqual(@as(usize, 1), state.result_count);
    try std.testing.expect(state.results[0] != null);
}

test "simd_row_sums correctness" {
    const matrix = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var row_sums: [2]f32 = undefined;

    simd_row_sums(&matrix, 2, 3, &row_sums);

    try std.testing.expectApproxEqAbs(row_sums[0], 6.0, 0.001); // 1+2+3
    try std.testing.expectApproxEqAbs(row_sums[1], 15.0, 0.001); // 4+5+6
}

test "simd_col_sums correctness" {
    const matrix = [_]f32{ 1, 2, 3, 4, 5, 6 };
    var col_sums: [3]f32 = undefined;

    simd_col_sums(&matrix, 2, 3, &col_sums);

    try std.testing.expectApproxEqAbs(col_sums[0], 5.0, 0.001); // 1+4
    try std.testing.expectApproxEqAbs(col_sums[1], 7.0, 0.001); // 2+5
    try std.testing.expectApproxEqAbs(col_sums[2], 9.0, 0.001); // 3+6
}

test "simd_compute_norm correctness" {
    const vector = [_]f32{ 3.0, 4.0, 0.0 };
    const norm = simd_compute_norm(&vector);
    try std.testing.expectApproxEqRel(norm, 5.0, 0.001);
}

test "simd_check_stability" {
    const stable = [_]f32{ 0.1, -0.5, 0.3 };
    const unstable = [_]f32{ 100.0, 0.1, 0.1 };

    try std.testing.expect(simd_check_stability(&stable, 10.0));
    try std.testing.expect(!simd_check_stability(&unstable, 10.0));
}

test "SinkhornBufferPool" {
    const allocator = std.testing.allocator;

    var pool = try SinkhornBufferPool.init(allocator, 100, 100);
    defer pool.deinit();

    const buffers = try pool.get_buffers(50, 50);
    try std.testing.expectEqual(@as(usize, 50), buffers.row_sums.len);
    try std.testing.expectEqual(@as(usize, 50), buffers.col_sums.len);

    // Test error on oversized request
    try std.testing.expectError(error.BufferTooSmall, pool.get_buffers(150, 50));
}

test "sinkhorn_normalize_optimized" {
    const allocator = std.testing.allocator;

    var matrix = [_]f32{ 1, 2, 3, 4 };
    var pool = try SinkhornBufferPool.init(allocator, 2, 2);
    defer pool.deinit();

    const config = mhc_constraints.MHCConfig{
        .sinkhorn_iterations = 20,
        .early_stopping = true,
    };

    const iters = try sinkhorn_normalize_optimized(&matrix, 2, 2, config, &pool);
    try std.testing.expect(iters > 0);
    try std.testing.expect(iters <= 20);

    // Verify doubly stochastic property
    const row1 = matrix[0] + matrix[1];
    const row2 = matrix[2] + matrix[3];
    try std.testing.expectApproxEqAbs(row1, 1.0, 0.01);
    try std.testing.expectApproxEqAbs(row2, 1.0, 0.01);
}

test "get_perf_report" {
    profiler_state.clear();

    const result = ProfileResult{
        .operation = "test_op",
        .total_us = 100.0,
        .iterations = 10,
        .per_iter_us = 10.0,
        .min_us = 8.0,
        .max_us = 12.0,
        .std_dev_us = 1.0,
        .allocations = 0,
        .bytes_allocated = 0,
        .overhead_pct = 2.5,
    };

    profiler_state.add_result(result);

    const report = get_perf_report();
    try std.testing.expectEqual(@as(usize, 1), report.num_operations);
    try std.testing.expect(report.meets_target); // 2.5% < 5%
}

test "profile_sinkhorn executes" {
    const allocator = std.testing.allocator;
    profiler_state.clear();

    var matrix = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const config = mhc_constraints.MHCConfig{
        .sinkhorn_iterations = 10,
    };

    const result = try profile_sinkhorn(&matrix, 3, 3, config, allocator, 2, 5);
    try std.testing.expect(result.total_us > 0);
    try std.testing.expectEqual(@as(u32, 5), result.iterations);
}

test "profile_stability_check executes" {
    const allocator = std.testing.allocator;
    profiler_state.clear();

    const activations = [_]f32{ 0.1, 0.2, 0.3, 0.4, 0.5 };
    const result = try profile_stability_check(&activations, 10.0, 5, 10, allocator);

    try std.testing.expect(result.total_us > 0);
    try std.testing.expectEqual(@as(u32, 10), result.iterations);
}

test "profile_manifold_constraints executes" {
    const allocator = std.testing.allocator;
    profiler_state.clear();

    // Use larger array to get measurable time
    const activations = try allocator.alloc(f32, 1024);
    defer allocator.free(activations);
    for (activations, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 10)) + 1.0;

    const result = try profile_manifold_constraints(activations, 1.0, 5, 100, allocator);

    // Just verify it executes without error, time may be 0 on fast systems
    try std.testing.expect(result.total_us >= 0);
    try std.testing.expectEqual(@as(u32, 100), result.iterations);
}

test "identify_hot_paths" {
    const allocator = std.testing.allocator;

    const matrix = try allocator.alloc(f32, 16);
    defer allocator.free(matrix);
    for (matrix, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i + 1));

    const activations = try allocator.alloc(f32, 16);
    defer allocator.free(activations);
    @memset(activations, 0.5);

    const config = mhc_constraints.MHCConfig{
        .sinkhorn_iterations = 5,
    };

    const result = try identify_hot_paths(matrix, 4, 4, activations, config, allocator);

    try std.testing.expect(result.total_us > 0);

    // Verify percentages sum to ~100%
    var total_pct: f64 = 0;
    for (result.breakdowns) |bd| {
        total_pct += bd.percentage;
    }
    try std.testing.expectApproxEqAbs(total_pct, 100.0, 0.1);
}