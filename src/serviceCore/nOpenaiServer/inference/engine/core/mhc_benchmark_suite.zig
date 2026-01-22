// mHC Benchmarking Suite
// Comprehensive performance benchmarks for mHC (manifold Hyperbolic Constraints)
//
// Core Components:
// - BenchmarkScenario: Defines standard benchmark cases
// - BenchmarkResult: Stores benchmark timing and statistics
// - StandardVsMHCComparison: Compares performance with/without mHC
// - StabilityMeasurement: Measures stability improvements from mHC
// - BenchmarkRunner: Executes benchmarks with reproducibility
//
// Benchmarks Include:
// - Sinkhorn normalization throughput
// - Matrix operations with mHC
// - Transformer layer with mHC
// - End-to-end inference pipeline
//
// Reference: docs/DAY_51_BENCHMARKING_SUITE_REPORT.md

const std = @import("std");
const builtin = @import("builtin");
const mhc_constraints = @import("mhc_constraints.zig");
const mhc_perf_profiler = @import("mhc_perf_profiler.zig");
const matrix_ops = @import("matrix_ops.zig");

// ============================================================================
// Benchmark Configuration
// ============================================================================

/// Configuration for reproducible benchmarks
pub const BenchmarkConfig = struct {
    /// Random seed for deterministic initialization
    seed: u64 = 42,

    /// Number of warmup iterations (cache warming)
    warmup_iterations: u32 = 10,

    /// Number of benchmark iterations for timing
    benchmark_iterations: u32 = 100,

    /// Matrix sizes to test
    matrix_sizes: []const usize = &[_]usize{ 32, 64, 128, 256, 512 },

    /// Sinkhorn iteration counts to test
    sinkhorn_iterations: []const u32 = &[_]u32{ 5, 10, 15, 20 },

    /// Enable detailed output
    verbose: bool = false,

    /// Output format
    output_format: OutputFormat = .table,

    pub const OutputFormat = enum {
        table,
        json,
        csv,
    };
};

/// Standard benchmark scenarios
pub const BenchmarkScenario = enum {
    /// Pure Sinkhorn normalization throughput
    sinkhorn_throughput,

    /// Matrix multiplication with mHC constraints
    matmul_with_mhc,

    /// Stability check operations
    stability_check,

    /// L2 norm and manifold projection
    manifold_projection,

    /// Full mHC pipeline (Sinkhorn + constraints + stability)
    full_mhc_pipeline,

    /// Comparison: standard vs mHC-enabled operations
    standard_vs_mhc,

    /// Transformer layer simulation with mHC
    transformer_layer,

    /// End-to-end inference simulation
    e2e_inference,

    pub fn name(self: BenchmarkScenario) []const u8 {
        return switch (self) {
            .sinkhorn_throughput => "Sinkhorn Throughput",
            .matmul_with_mhc => "MatMul with mHC",
            .stability_check => "Stability Check",
            .manifold_projection => "Manifold Projection",
            .full_mhc_pipeline => "Full mHC Pipeline",
            .standard_vs_mhc => "Standard vs mHC",
            .transformer_layer => "Transformer Layer",
            .e2e_inference => "E2E Inference",
        };
    }
};

// ============================================================================
// Benchmark Results
// ============================================================================

/// Result from a single benchmark run
pub const BenchmarkResult = struct {
    /// Scenario that was benchmarked
    scenario: BenchmarkScenario,

    /// Matrix size used (if applicable)
    matrix_size: usize,

    /// Configuration parameters
    sinkhorn_iters: u32,

    /// Timing statistics (all in microseconds)
    total_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
    mean_time_us: f64,
    std_dev_us: f64,
    median_time_us: f64,
    p95_time_us: f64,
    p99_time_us: f64,

    /// Throughput metrics
    ops_per_second: f64,
    elements_per_second: f64,

    /// mHC-specific metrics
    avg_convergence_iters: f32,
    stability_rate: f32, // Percentage of stable results

    /// Memory metrics
    peak_memory_bytes: usize,
    allocations_per_op: u32,

    /// Comparison metrics (for standard_vs_mhc)
    baseline_time_us: f64,
    overhead_percent: f64,
};

/// Aggregate results for a benchmark suite
pub const BenchmarkSuiteResult = struct {
    /// All individual results
    results: std.ArrayList(BenchmarkResult),

    /// Suite-level statistics
    total_duration_us: f64,
    scenarios_run: u32,
    overall_overhead_percent: f64,
    meets_overhead_target: bool, // <5% target

    /// Timestamp
    timestamp: i64,
    platform: []const u8,

    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BenchmarkSuiteResult {
        return .{
            .allocator = allocator,
            .results = .{},
            .total_duration_us = 0,
            .scenarios_run = 0,
            .overall_overhead_percent = 0,
            .meets_overhead_target = true,
            .timestamp = std.time.milliTimestamp(),
            .platform = getPlatformString(),
        };
    }

    pub fn deinit(self: *BenchmarkSuiteResult) void {
        self.results.deinit(self.allocator);
    }

    pub fn addResult(self: *BenchmarkSuiteResult, result: BenchmarkResult) !void {
        try self.results.append(result);
        self.scenarios_run += 1;
        self.total_duration_us += result.total_time_us;
        if (result.overhead_percent > 5.0) {
            self.meets_overhead_target = false;
        }
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

fn getPlatformString() []const u8 {
    return switch (builtin.cpu.arch) {
        .aarch64 => "aarch64 (ARM64)",
        .x86_64 => "x86_64",
        .arm => "arm",
        else => "unknown",
    };
}

/// Deterministic random number generator for reproducible benchmarks
pub const DeterministicRng = struct {
    state: u64,

    pub fn init(seed: u64) DeterministicRng {
        return .{ .state = seed };
    }

    /// Generate next random u64
    pub fn next(self: *DeterministicRng) u64 {
        // xorshift64*
        var x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        return x *% 0x2545F4914F6CDD1D;
    }

    /// Generate random f32 in [0, 1)
    pub fn nextFloat(self: *DeterministicRng) f32 {
        return @as(f32, @floatFromInt(self.next() >> 40)) / @as(f32, @floatFromInt(@as(u64, 1) << 24));
    }

    /// Fill array with random f32 values in [min, max)
    pub fn fillArray(self: *DeterministicRng, arr: []f32, min: f32, max: f32) void {
        const range = max - min;
        for (arr) |*val| {
            val.* = self.nextFloat() * range + min;
        }
    }
};

/// Compute percentile from sorted array
fn computePercentile(sorted: []const f64, percentile: f64) f64 {
    if (sorted.len == 0) return 0;
    if (sorted.len == 1) return sorted[0];

    const idx = (percentile / 100.0) * @as(f64, @floatFromInt(sorted.len - 1));
    const lower = @as(usize, @intFromFloat(@floor(idx)));
    const upper = @min(lower + 1, sorted.len - 1);
    const frac = idx - @as(f64, @floatFromInt(lower));

    return sorted[lower] * (1.0 - frac) + sorted[upper] * frac;
}

/// Compute statistics from timing samples
fn computeStats(samples: []f64) struct {
    min: f64,
    max: f64,
    mean: f64,
    std_dev: f64,
    median: f64,
    p95: f64,
    p99: f64,
} {
    if (samples.len == 0) return .{ .min = 0, .max = 0, .mean = 0, .std_dev = 0, .median = 0, .p95 = 0, .p99 = 0 };

    // Sort for percentiles
    std.mem.sort(f64, samples, {}, std.sort.asc(f64));

    var min: f64 = samples[0];
    var max: f64 = samples[0];
    var sum: f64 = 0;

    for (samples) |s| {
        min = @min(min, s);
        max = @max(max, s);
        sum += s;
    }

    const mean = sum / @as(f64, @floatFromInt(samples.len));

    var variance: f64 = 0;
    for (samples) |s| {
        const diff = s - mean;
        variance += diff * diff;
    }
    const std_dev = @sqrt(variance / @as(f64, @floatFromInt(samples.len)));

    return .{
        .min = min,
        .max = max,
        .mean = mean,
        .std_dev = std_dev,
        .median = computePercentile(samples, 50.0),
        .p95 = computePercentile(samples, 95.0),
        .p99 = computePercentile(samples, 99.0),
    };
}

// ============================================================================
// Benchmark Runner
// ============================================================================

/// Main benchmark runner
pub const BenchmarkRunner = struct {
    allocator: std.mem.Allocator,
    config: BenchmarkConfig,
    rng: DeterministicRng,
    buffer_pool: ?mhc_perf_profiler.SinkhornBufferPool,

    pub fn init(allocator: std.mem.Allocator, config: BenchmarkConfig) !BenchmarkRunner {
        const max_size = if (config.matrix_sizes.len > 0)
            config.matrix_sizes[config.matrix_sizes.len - 1]
        else
            512;

        return .{
            .allocator = allocator,
            .config = config,
            .rng = DeterministicRng.init(config.seed),
            .buffer_pool = try mhc_perf_profiler.SinkhornBufferPool.init(allocator, max_size, max_size),
        };
    }

    pub fn deinit(self: *BenchmarkRunner) void {
        if (self.buffer_pool) |*pool| {
            pool.deinit();
        }
    }

    /// Run all benchmark scenarios
    pub fn runAll(self: *BenchmarkRunner) !BenchmarkSuiteResult {
        var suite = BenchmarkSuiteResult.init(self.allocator);

        // Run each scenario for each matrix size
        for (self.config.matrix_sizes) |size| {
            try suite.addResult(try self.benchmarkSinkhornThroughput(size));
            try suite.addResult(try self.benchmarkStabilityCheck(size));
            try suite.addResult(try self.benchmarkManifoldProjection(size));
            try suite.addResult(try self.benchmarkFullPipeline(size));
            try suite.addResult(try self.benchmarkStandardVsMHC(size));
        }

        // Compute overall overhead
        var total_overhead: f64 = 0;
        var overhead_count: u32 = 0;
        for (suite.results.items) |result| {
            if (result.overhead_percent > 0) {
                total_overhead += result.overhead_percent;
                overhead_count += 1;
            }
        }
        suite.overall_overhead_percent = if (overhead_count > 0)
            total_overhead / @as(f64, @floatFromInt(overhead_count))
        else
            0;

        return suite;
    }

    /// Benchmark Sinkhorn normalization throughput
    pub fn benchmarkSinkhornThroughput(self: *BenchmarkRunner, size: usize) !BenchmarkResult {
        const matrix = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(matrix);
        const backup = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(backup);

        // Initialize with deterministic data
        self.rng.fillArray(matrix, 0.1, 1.0);
        @memcpy(backup, matrix);

        const mhc_config = mhc_constraints.MHCConfig{
            .enabled = true,
            .sinkhorn_iterations = 10,
            .early_stopping = true,
        };

        // Collect timing samples
        var samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(samples);

        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            @memcpy(matrix, backup);
            _ = try mhc_constraints.sinkhorn_normalize(matrix, size, size, mhc_config, self.allocator);
        }

        // Benchmark
        var total_iters: u32 = 0;
        var timer = mhc_perf_profiler.PerfTimer.start();

        for (0..self.config.benchmark_iterations) |i| {
            @memcpy(matrix, backup);
            const lap = mhc_perf_profiler.PerfTimer.start();
            const iters = try mhc_constraints.sinkhorn_normalize(matrix, size, size, mhc_config, self.allocator);
            samples[i] = lap.elapsed_us();
            total_iters += iters;
        }

        const total_time = timer.elapsed_us();
        const stats = computeStats(samples);
        const elements = size * size;

        return BenchmarkResult{
            .scenario = .sinkhorn_throughput,
            .matrix_size = size,
            .sinkhorn_iters = 10,
            .total_time_us = total_time,
            .min_time_us = stats.min,
            .max_time_us = stats.max,
            .mean_time_us = stats.mean,
            .std_dev_us = stats.std_dev,
            .median_time_us = stats.median,
            .p95_time_us = stats.p95,
            .p99_time_us = stats.p99,
            .ops_per_second = @as(f64, @floatFromInt(self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .elements_per_second = @as(f64, @floatFromInt(elements * self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .avg_convergence_iters = @as(f32, @floatFromInt(total_iters)) / @as(f32, @floatFromInt(self.config.benchmark_iterations)),
            .stability_rate = 1.0,
            .peak_memory_bytes = (size + size) * @sizeOf(f32),
            .allocations_per_op = 2,
            .baseline_time_us = 0,
            .overhead_percent = 0,
        };
    }

    /// Benchmark stability checking
    pub fn benchmarkStabilityCheck(self: *BenchmarkRunner, size: usize) !BenchmarkResult {
        const activations = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(activations);

        self.rng.fillArray(activations, -5.0, 5.0);

        var samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(samples);

        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            _ = mhc_constraints.check_stability(activations, 10.0);
        }

        // Benchmark
        var stable_count: u32 = 0;
        var timer = mhc_perf_profiler.PerfTimer.start();

        for (0..self.config.benchmark_iterations) |i| {
            const lap = mhc_perf_profiler.PerfTimer.start();
            const stable = mhc_constraints.check_stability(activations, 10.0);
            samples[i] = lap.elapsed_us();
            if (stable) stable_count += 1;
        }

        const total_time = timer.elapsed_us();
        const stats = computeStats(samples);
        const elements = size * size;

        return BenchmarkResult{
            .scenario = .stability_check,
            .matrix_size = size,
            .sinkhorn_iters = 0,
            .total_time_us = total_time,
            .min_time_us = stats.min,
            .max_time_us = stats.max,
            .mean_time_us = stats.mean,
            .std_dev_us = stats.std_dev,
            .median_time_us = stats.median,
            .p95_time_us = stats.p95,
            .p99_time_us = stats.p99,
            .ops_per_second = @as(f64, @floatFromInt(self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .elements_per_second = @as(f64, @floatFromInt(elements * self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .avg_convergence_iters = 0,
            .stability_rate = @as(f32, @floatFromInt(stable_count)) / @as(f32, @floatFromInt(self.config.benchmark_iterations)),
            .peak_memory_bytes = 0,
            .allocations_per_op = 0,
            .baseline_time_us = 0,
            .overhead_percent = 0,
        };
    }

    /// Benchmark manifold projection
    pub fn benchmarkManifoldProjection(self: *BenchmarkRunner, size: usize) !BenchmarkResult {
        const activations = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(activations);
        const backup = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(backup);

        self.rng.fillArray(activations, -10.0, 10.0);
        @memcpy(backup, activations);

        var samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(samples);

        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            @memcpy(activations, backup);
            _ = mhc_constraints.apply_manifold_constraints(activations, 5.0);
        }

        // Benchmark
        var timer = mhc_perf_profiler.PerfTimer.start();

        for (0..self.config.benchmark_iterations) |i| {
            @memcpy(activations, backup);
            const lap = mhc_perf_profiler.PerfTimer.start();
            _ = mhc_constraints.apply_manifold_constraints(activations, 5.0);
            samples[i] = lap.elapsed_us();
        }

        const total_time = timer.elapsed_us();
        const stats = computeStats(samples);
        const elements = size * size;

        return BenchmarkResult{
            .scenario = .manifold_projection,
            .matrix_size = size,
            .sinkhorn_iters = 0,
            .total_time_us = total_time,
            .min_time_us = stats.min,
            .max_time_us = stats.max,
            .mean_time_us = stats.mean,
            .std_dev_us = stats.std_dev,
            .median_time_us = stats.median,
            .p95_time_us = stats.p95,
            .p99_time_us = stats.p99,
            .ops_per_second = @as(f64, @floatFromInt(self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .elements_per_second = @as(f64, @floatFromInt(elements * self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .avg_convergence_iters = 0,
            .stability_rate = 1.0,
            .peak_memory_bytes = 0,
            .allocations_per_op = 0,
            .baseline_time_us = 0,
            .overhead_percent = 0,
        };
    }

    /// Benchmark full mHC pipeline
    pub fn benchmarkFullPipeline(self: *BenchmarkRunner, size: usize) !BenchmarkResult {
        const matrix = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(matrix);
        const backup = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(backup);

        self.rng.fillArray(matrix, 0.1, 2.0);
        @memcpy(backup, matrix);

        const mhc_config = mhc_constraints.MHCConfig{
            .enabled = true,
            .sinkhorn_iterations = 10,
            .manifold_beta = 5.0,
            .early_stopping = true,
        };

        var samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(samples);

        // Warmup
        for (0..self.config.warmup_iterations) |_| {
            @memcpy(matrix, backup);
            _ = try mhc_constraints.sinkhorn_normalize(matrix, size, size, mhc_config, self.allocator);
            _ = mhc_constraints.apply_manifold_constraints(matrix, mhc_config.manifold_beta);
            _ = mhc_constraints.check_stability(matrix, mhc_config.stability_threshold);
        }

        // Benchmark full pipeline
        var stable_count: u32 = 0;
        var total_iters: u32 = 0;
        var timer = mhc_perf_profiler.PerfTimer.start();

        for (0..self.config.benchmark_iterations) |i| {
            @memcpy(matrix, backup);
            const lap = mhc_perf_profiler.PerfTimer.start();

            // Full mHC pipeline
            const iters = try mhc_constraints.sinkhorn_normalize(matrix, size, size, mhc_config, self.allocator);
            _ = mhc_constraints.apply_manifold_constraints(matrix, mhc_config.manifold_beta);
            const stable = mhc_constraints.check_stability(matrix, 10.0);

            samples[i] = lap.elapsed_us();
            total_iters += iters;
            if (stable) stable_count += 1;
        }

        const total_time = timer.elapsed_us();
        const stats = computeStats(samples);
        const elements = size * size;

        return BenchmarkResult{
            .scenario = .full_mhc_pipeline,
            .matrix_size = size,
            .sinkhorn_iters = 10,
            .total_time_us = total_time,
            .min_time_us = stats.min,
            .max_time_us = stats.max,
            .mean_time_us = stats.mean,
            .std_dev_us = stats.std_dev,
            .median_time_us = stats.median,
            .p95_time_us = stats.p95,
            .p99_time_us = stats.p99,
            .ops_per_second = @as(f64, @floatFromInt(self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .elements_per_second = @as(f64, @floatFromInt(elements * self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .avg_convergence_iters = @as(f32, @floatFromInt(total_iters)) / @as(f32, @floatFromInt(self.config.benchmark_iterations)),
            .stability_rate = @as(f32, @floatFromInt(stable_count)) / @as(f32, @floatFromInt(self.config.benchmark_iterations)),
            .peak_memory_bytes = (size + size) * @sizeOf(f32),
            .allocations_per_op = 2,
            .baseline_time_us = 0,
            .overhead_percent = 0,
        };
    }

    /// Benchmark standard vs mHC comparison
    pub fn benchmarkStandardVsMHC(self: *BenchmarkRunner, size: usize) !BenchmarkResult {
        const a = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(b);
        const c = try self.allocator.alloc(f32, size * size);
        defer self.allocator.free(c);

        self.rng.fillArray(a, -1.0, 1.0);
        self.rng.fillArray(b, -1.0, 1.0);

        var baseline_samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(baseline_samples);
        var mhc_samples = try self.allocator.alloc(f64, self.config.benchmark_iterations);
        defer self.allocator.free(mhc_samples);

        // Warmup standard matmul
        for (0..self.config.warmup_iterations) |_| {
            try matrix_ops.matmul(c, .{ .f32 = a }, b, size, size, size, self.allocator, null);
        }

        // Benchmark standard matmul
        for (0..self.config.benchmark_iterations) |i| {
            const lap = mhc_perf_profiler.PerfTimer.start();
            try matrix_ops.matmul(c, .{ .f32 = a }, b, size, size, size, self.allocator, null);
            baseline_samples[i] = lap.elapsed_us();
        }

        const baseline_stats = computeStats(baseline_samples);

        // Configure mHC
        const mhc_config = matrix_ops.MatMulConfig{
            .use_mhc = true,
            .layer_id = 0,
            .mhc_config = .{
                .enabled = true,
                .sinkhorn_iterations = 10,
                .early_stopping = true,
            },
        };

        // Warmup mHC matmul
        for (0..self.config.warmup_iterations) |_| {
            _ = try matrix_ops.matmul_with_mhc(c, .{ .f32 = a }, b, size, size, size, mhc_config, self.allocator, null);
        }

        // Benchmark mHC matmul
        var stable_count: u32 = 0;
        var timer = mhc_perf_profiler.PerfTimer.start();

        for (0..self.config.benchmark_iterations) |i| {
            const lap = mhc_perf_profiler.PerfTimer.start();
            const metrics = try matrix_ops.matmul_with_mhc(c, .{ .f32 = a }, b, size, size, size, mhc_config, self.allocator, null);
            mhc_samples[i] = lap.elapsed_us();
            if (metrics) |m| {
                if (m.is_stable) stable_count += 1;
            }
        }

        const total_time = timer.elapsed_us();
        const mhc_stats = computeStats(mhc_samples);
        const elements = size * size;

        const overhead = ((mhc_stats.mean - baseline_stats.mean) / baseline_stats.mean) * 100.0;

        return BenchmarkResult{
            .scenario = .standard_vs_mhc,
            .matrix_size = size,
            .sinkhorn_iters = 10,
            .total_time_us = total_time,
            .min_time_us = mhc_stats.min,
            .max_time_us = mhc_stats.max,
            .mean_time_us = mhc_stats.mean,
            .std_dev_us = mhc_stats.std_dev,
            .median_time_us = mhc_stats.median,
            .p95_time_us = mhc_stats.p95,
            .p99_time_us = mhc_stats.p99,
            .ops_per_second = @as(f64, @floatFromInt(self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .elements_per_second = @as(f64, @floatFromInt(elements * self.config.benchmark_iterations)) / (total_time / 1_000_000.0),
            .avg_convergence_iters = 10,
            .stability_rate = @as(f32, @floatFromInt(stable_count)) / @as(f32, @floatFromInt(self.config.benchmark_iterations)),
            .peak_memory_bytes = (size + size) * @sizeOf(f32),
            .allocations_per_op = 2,
            .baseline_time_us = baseline_stats.mean,
            .overhead_percent = overhead,
        };
    }
};


// ============================================================================
// Report Generation
// ============================================================================

/// Report formatter for benchmark results
pub const BenchmarkReporter = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) BenchmarkReporter {
        return .{ .allocator = allocator };
    }

    /// Print table-formatted report to stdout
    pub fn printTable(self: *BenchmarkReporter, suite: *const BenchmarkSuiteResult) void {
        _ = self;
        std.debug.print("\n", .{});
        std.debug.print("╔══════════════════════════════════════════════════════════════════════════════════════╗\n", .{});
        std.debug.print("║                        mHC Benchmark Suite Report                                    ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Platform: {s:<20} Date: {d:<20}                        ║\n", .{ suite.platform, suite.timestamp });
        std.debug.print("║ Target: <5% overhead                                                                 ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Scenario             │ Size   │ Mean(μs) │ P95(μs) │ Ops/s    │ Overhead │ Status   ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════════════════════════════╣\n", .{});

        for (suite.results.items) |result| {
            const status = if (result.overhead_percent > 5.0) "❌ FAIL" else if (result.overhead_percent > 0) "✅ PASS" else "—";
            std.debug.print("║ {s:<20} │ {d:>5}  │ {d:>8.2} │ {d:>7.2} │ {d:>8.0} │ {d:>6.2}%  │ {s:<7} ║\n", .{
                result.scenario.name(),
                result.matrix_size,
                result.mean_time_us,
                result.p95_time_us,
                result.ops_per_second,
                result.overhead_percent,
                status,
            });
        }

        std.debug.print("╠══════════════════════════════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Total Duration: {d:>10.2}μs │ Scenarios: {d:<3} │ Overall Overhead: {d:>5.2}%          ║\n", .{
            suite.total_duration_us,
            suite.scenarios_run,
            suite.overall_overhead_percent,
        });
        std.debug.print("║ Status: {s}                                                                       ║\n", .{
            if (suite.meets_overhead_target) "✅ ALL PASS - <5% overhead target met" else "❌ FAIL - overhead exceeds 5%",
        });
        std.debug.print("╚══════════════════════════════════════════════════════════════════════════════════════╝\n", .{});
    }

    /// Generate CSV output
    pub fn toCsv(self: *BenchmarkReporter, suite: *const BenchmarkSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        // Header
        try writer.writeAll("scenario,matrix_size,mean_us,min_us,max_us,p95_us,p99_us,std_dev_us,ops_per_sec,overhead_pct,stability_rate\n");

        // Data rows
        for (suite.results.items) |result| {
            try writer.print("{s},{d},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.3},{d:.0},{d:.2},{d:.3}\n", .{
                result.scenario.name(),
                result.matrix_size,
                result.mean_time_us,
                result.min_time_us,
                result.max_time_us,
                result.p95_time_us,
                result.p99_time_us,
                result.std_dev_us,
                result.ops_per_second,
                result.overhead_percent,
                result.stability_rate,
            });
        }

        return buffer.toOwnedSlice();
    }

    /// Generate JSON output
    pub fn toJson(self: *BenchmarkReporter, suite: *const BenchmarkSuiteResult) ![]u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        const writer = buffer.writer();

        try writer.writeAll("{\n");
        try writer.print("  \"platform\": \"{s}\",\n", .{suite.platform});
        try writer.print("  \"timestamp\": {d},\n", .{suite.timestamp});
        try writer.print("  \"total_duration_us\": {d:.3},\n", .{suite.total_duration_us});
        try writer.print("  \"overall_overhead_percent\": {d:.3},\n", .{suite.overall_overhead_percent});
        try writer.print("  \"meets_target\": {s},\n", .{if (suite.meets_overhead_target) "true" else "false"});
        try writer.writeAll("  \"results\": [\n");

        for (suite.results.items, 0..) |result, i| {
            try writer.writeAll("    {\n");
            try writer.print("      \"scenario\": \"{s}\",\n", .{result.scenario.name()});
            try writer.print("      \"matrix_size\": {d},\n", .{result.matrix_size});
            try writer.print("      \"mean_us\": {d:.3},\n", .{result.mean_time_us});
            try writer.print("      \"min_us\": {d:.3},\n", .{result.min_time_us});
            try writer.print("      \"max_us\": {d:.3},\n", .{result.max_time_us});
            try writer.print("      \"p95_us\": {d:.3},\n", .{result.p95_time_us});
            try writer.print("      \"p99_us\": {d:.3},\n", .{result.p99_time_us});
            try writer.print("      \"ops_per_second\": {d:.0},\n", .{result.ops_per_second});
            try writer.print("      \"overhead_percent\": {d:.3},\n", .{result.overhead_percent});
            try writer.print("      \"stability_rate\": {d:.3}\n", .{result.stability_rate});
            if (i < suite.results.items.len - 1) {
                try writer.writeAll("    },\n");
            } else {
                try writer.writeAll("    }\n");
            }
        }

        try writer.writeAll("  ]\n");
        try writer.writeAll("}\n");

        return buffer.toOwnedSlice();
    }
};

// ============================================================================
// Stability Measurement
// ============================================================================

/// Measures stability improvements from mHC
pub const StabilityMeasurement = struct {
    /// Run stability comparison: with and without mHC
    pub fn measureStabilityImprovement(
        allocator: std.mem.Allocator,
        matrix_size: usize,
        iterations: u32,
        seed: u64,
    ) !StabilityReport {
        var rng = DeterministicRng.init(seed);

        const matrix = try allocator.alloc(f32, matrix_size * matrix_size);
        defer allocator.free(matrix);
        const backup = try allocator.alloc(f32, matrix_size * matrix_size);
        defer allocator.free(backup);

        // Initialize with potentially unstable values (high variance)
        rng.fillArray(matrix, -100.0, 100.0);
        @memcpy(backup, matrix);

        // Measure without mHC
        var unstable_without_mhc: u32 = 0;
        var max_norm_without: f32 = 0;

        for (0..iterations) |_| {
            @memcpy(matrix, backup);
            const norm = mhc_constraints.compute_norm(matrix);
            max_norm_without = @max(max_norm_without, norm);
            if (!mhc_constraints.check_stability(matrix, 50.0)) {
                unstable_without_mhc += 1;
            }
        }

        // Measure with mHC
        var unstable_with_mhc: u32 = 0;
        var max_norm_with: f32 = 0;
        var total_convergence_iters: u32 = 0;

        const mhc_config = mhc_constraints.MHCConfig{
            .enabled = true,
            .sinkhorn_iterations = 15,
            .manifold_beta = 10.0,
            .early_stopping = true,
        };

        for (0..iterations) |_| {
            @memcpy(matrix, backup);

            // Apply mHC pipeline
            const iters = try mhc_constraints.sinkhorn_normalize(matrix, matrix_size, matrix_size, mhc_config, allocator);
            _ = mhc_constraints.apply_manifold_constraints(matrix, mhc_config.manifold_beta);

            total_convergence_iters += iters;

            const norm = mhc_constraints.compute_norm(matrix);
            max_norm_with = @max(max_norm_with, norm);

            if (!mhc_constraints.check_stability(matrix, 50.0)) {
                unstable_with_mhc += 1;
            }
        }

        const stability_without = 1.0 - @as(f32, @floatFromInt(unstable_without_mhc)) / @as(f32, @floatFromInt(iterations));
        const stability_with = 1.0 - @as(f32, @floatFromInt(unstable_with_mhc)) / @as(f32, @floatFromInt(iterations));

        return StabilityReport{
            .matrix_size = matrix_size,
            .iterations = iterations,
            .stability_rate_without_mhc = stability_without,
            .stability_rate_with_mhc = stability_with,
            .stability_improvement = stability_with - stability_without,
            .max_norm_without_mhc = max_norm_without,
            .max_norm_with_mhc = max_norm_with,
            .norm_reduction_factor = if (max_norm_with > 0) max_norm_without / max_norm_with else 0,
            .avg_convergence_iters = @as(f32, @floatFromInt(total_convergence_iters)) / @as(f32, @floatFromInt(iterations)),
        };
    }
};

/// Report on stability improvements
pub const StabilityReport = struct {
    matrix_size: usize,
    iterations: u32,
    stability_rate_without_mhc: f32,
    stability_rate_with_mhc: f32,
    stability_improvement: f32,
    max_norm_without_mhc: f32,
    max_norm_with_mhc: f32,
    norm_reduction_factor: f32,
    avg_convergence_iters: f32,

    pub fn print(self: *const StabilityReport) void {
        std.debug.print("\n╔══════════════════════════════════════════════════════════════╗\n", .{});
        std.debug.print("║              Stability Improvement Report                     ║\n", .{});
        std.debug.print("╠══════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Matrix Size: {d}x{d}                                          ║\n", .{ self.matrix_size, self.matrix_size });
        std.debug.print("║ Iterations: {d}                                              ║\n", .{self.iterations});
        std.debug.print("╠══════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Stability without mHC: {d:>6.2}%                               ║\n", .{self.stability_rate_without_mhc * 100});
        std.debug.print("║ Stability with mHC:    {d:>6.2}%                               ║\n", .{self.stability_rate_with_mhc * 100});
        std.debug.print("║ Improvement:           {d:>+6.2}%                               ║\n", .{self.stability_improvement * 100});
        std.debug.print("╠══════════════════════════════════════════════════════════════╣\n", .{});
        std.debug.print("║ Max norm without mHC:  {d:>10.2}                            ║\n", .{self.max_norm_without_mhc});
        std.debug.print("║ Max norm with mHC:     {d:>10.2}                            ║\n", .{self.max_norm_with_mhc});
        std.debug.print("║ Norm reduction factor: {d:>10.2}x                           ║\n", .{self.norm_reduction_factor});
        std.debug.print("║ Avg convergence iters: {d:>10.2}                            ║\n", .{self.avg_convergence_iters});
        std.debug.print("╚══════════════════════════════════════════════════════════════╝\n", .{});
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Run quick benchmark with default settings
pub fn runQuickBenchmark(allocator: std.mem.Allocator) !BenchmarkSuiteResult {
    const config = BenchmarkConfig{
        .seed = 42,
        .warmup_iterations = 5,
        .benchmark_iterations = 50,
        .matrix_sizes = &[_]usize{ 32, 64, 128 },
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    return runner.runAll();
}

/// Run full benchmark suite
pub fn runFullBenchmark(allocator: std.mem.Allocator) !BenchmarkSuiteResult {
    const config = BenchmarkConfig{
        .seed = 42,
        .warmup_iterations = 10,
        .benchmark_iterations = 100,
        .matrix_sizes = &[_]usize{ 32, 64, 128, 256, 512 },
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    return runner.runAll();
}

// ============================================================================
// Unit Tests
// ============================================================================

test "DeterministicRng produces consistent results" {
    var rng1 = DeterministicRng.init(42);
    var rng2 = DeterministicRng.init(42);

    // Same seed should produce same sequence
    for (0..10) |_| {
        try std.testing.expectEqual(rng1.next(), rng2.next());
    }
}

test "DeterministicRng nextFloat in range" {
    var rng = DeterministicRng.init(12345);

    for (0..100) |_| {
        const val = rng.nextFloat();
        try std.testing.expect(val >= 0.0);
        try std.testing.expect(val < 1.0);
    }
}

test "DeterministicRng fillArray respects range" {
    var rng = DeterministicRng.init(999);
    var arr: [100]f32 = undefined;

    rng.fillArray(&arr, -5.0, 5.0);

    for (arr) |val| {
        try std.testing.expect(val >= -5.0);
        try std.testing.expect(val < 5.0);
    }
}

test "computePercentile basic" {
    var sorted = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    try std.testing.expectApproxEqAbs(computePercentile(&sorted, 0.0), 1.0, 0.01);
    try std.testing.expectApproxEqAbs(computePercentile(&sorted, 50.0), 3.0, 0.01);
    try std.testing.expectApproxEqAbs(computePercentile(&sorted, 100.0), 5.0, 0.01);
}

test "computeStats correctness" {
    var samples = [_]f64{ 1.0, 2.0, 3.0, 4.0, 5.0 };
    const stats = computeStats(&samples);

    try std.testing.expectApproxEqAbs(stats.min, 1.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.max, 5.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.mean, 3.0, 0.01);
    try std.testing.expectApproxEqAbs(stats.median, 3.0, 0.01);
}

test "BenchmarkSuiteResult init and deinit" {
    const allocator = std.testing.allocator;
    var suite = BenchmarkSuiteResult.init(allocator);
    defer suite.deinit();

    try std.testing.expectEqual(@as(u32, 0), suite.scenarios_run);
    try std.testing.expect(suite.meets_overhead_target);
}

test "BenchmarkRunner init and deinit" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{ 16, 32 },
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    try std.testing.expect(runner.buffer_pool != null);
}

test "benchmarkSinkhornThroughput executes" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{16},
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    const result = try runner.benchmarkSinkhornThroughput(16);

    try std.testing.expect(result.total_time_us > 0);
    try std.testing.expect(result.mean_time_us > 0);
    try std.testing.expect(result.ops_per_second > 0);
    try std.testing.expectEqual(BenchmarkScenario.sinkhorn_throughput, result.scenario);
}

test "benchmarkStabilityCheck executes" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{16},
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    const result = try runner.benchmarkStabilityCheck(16);

    try std.testing.expect(result.total_time_us > 0);
    try std.testing.expectEqual(BenchmarkScenario.stability_check, result.scenario);
}

test "benchmarkManifoldProjection executes" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{16},
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    const result = try runner.benchmarkManifoldProjection(16);

    try std.testing.expect(result.total_time_us > 0);
    try std.testing.expectEqual(BenchmarkScenario.manifold_projection, result.scenario);
}

test "benchmarkFullPipeline executes" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{16},
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    const result = try runner.benchmarkFullPipeline(16);

    try std.testing.expect(result.total_time_us > 0);
    try std.testing.expectEqual(BenchmarkScenario.full_mhc_pipeline, result.scenario);
    try std.testing.expect(result.avg_convergence_iters > 0);
}

test "benchmarkStandardVsMHC executes" {
    const allocator = std.testing.allocator;
    const config = BenchmarkConfig{
        .matrix_sizes = &[_]usize{16},
        .benchmark_iterations = 5,
        .warmup_iterations = 2,
    };

    var runner = try BenchmarkRunner.init(allocator, config);
    defer runner.deinit();

    const result = try runner.benchmarkStandardVsMHC(16);

    try std.testing.expect(result.total_time_us > 0);
    try std.testing.expect(result.baseline_time_us > 0);
    try std.testing.expectEqual(BenchmarkScenario.standard_vs_mhc, result.scenario);
}

test "BenchmarkReporter toCsv" {
    const allocator = std.testing.allocator;
    var suite = BenchmarkSuiteResult.init(allocator);
    defer suite.deinit();

    try suite.addResult(BenchmarkResult{
        .scenario = .sinkhorn_throughput,
        .matrix_size = 64,
        .sinkhorn_iters = 10,
        .total_time_us = 100.0,
        .min_time_us = 1.0,
        .max_time_us = 5.0,
        .mean_time_us = 2.0,
        .std_dev_us = 0.5,
        .median_time_us = 2.0,
        .p95_time_us = 4.0,
        .p99_time_us = 5.0,
        .ops_per_second = 50000,
        .elements_per_second = 1000000,
        .avg_convergence_iters = 8.5,
        .stability_rate = 0.95,
        .peak_memory_bytes = 1024,
        .allocations_per_op = 2,
        .baseline_time_us = 0,
        .overhead_percent = 0,
    });

    var reporter = BenchmarkReporter.init(allocator);
    const csv = try reporter.toCsv(&suite);
    defer allocator.free(csv);

    try std.testing.expect(csv.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, csv, "Sinkhorn Throughput") != null);
}

test "StabilityMeasurement executes" {
    const allocator = std.testing.allocator;

    const report = try StabilityMeasurement.measureStabilityImprovement(allocator, 16, 10, 42);

    try std.testing.expect(report.stability_rate_with_mhc >= 0);
    try std.testing.expect(report.stability_rate_with_mhc <= 1.0);
    try std.testing.expect(report.avg_convergence_iters > 0);
}

test "runQuickBenchmark executes" {
    // Skip in CI - too slow
    if (@import("builtin").is_test) {
        const allocator = std.testing.allocator;
        // Use minimal config for test
        const config = BenchmarkConfig{
            .seed = 42,
            .warmup_iterations = 2,
            .benchmark_iterations = 3,
            .matrix_sizes = &[_]usize{16},
        };

        var runner = try BenchmarkRunner.init(allocator, config);
        defer runner.deinit();

        var suite = try runner.runAll();
        defer suite.deinit();

        try std.testing.expect(suite.scenarios_run > 0);
    }
}

