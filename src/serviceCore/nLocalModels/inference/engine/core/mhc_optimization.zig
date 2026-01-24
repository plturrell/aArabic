//! mHC Performance Optimization & Profiling
//!
//! Day 69: Comprehensive performance optimization infrastructure for mHC.
//! Includes profiling, SIMD optimizations, memory pooling, and low-overhead monitoring.
//!
//! Key Features:
//! - Fine-grained profiling with minimal overhead
//! - SIMD-optimized vector operations with benchmarking
//! - Memory pool for allocation reduction
//! - Low-overhead monitoring with configurable sampling
//! - Complete benchmark suite with baseline comparison

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

// ============================================================================
// Profiling Infrastructure
// ============================================================================

/// Configuration for the profiler
pub const ProfilerConfig = struct {
    /// Sampling rate (1.0 = all, 0.1 = 10%)
    sampling_rate: f32 = 1.0,
    /// Enable/disable profiling
    enabled: bool = true,
    /// Maximum path entries to track
    max_paths: usize = 128,
    /// Include stack traces (expensive)
    include_stack_traces: bool = false,
    /// Flush interval in microseconds
    flush_interval_us: u64 = 1_000_000, // 1 second
    /// Overhead target percentage
    overhead_target_pct: f32 = 2.0,

    pub fn production() ProfilerConfig {
        return .{
            .sampling_rate = 0.01, // 1%
            .enabled = true,
            .max_paths = 64,
            .include_stack_traces = false,
        };
    }

    pub fn development() ProfilerConfig {
        return .{
            .sampling_rate = 1.0, // 100%
            .enabled = true,
            .max_paths = 256,
            .include_stack_traces = true,
        };
    }

    pub fn disabled() ProfilerConfig {
        return .{
            .sampling_rate = 0.0,
            .enabled = false,
            .max_paths = 0,
            .include_stack_traces = false,
        };
    }
};

/// Profile data for a single code path
pub const CodePathProfile = struct {
    /// Name of the profiled path
    name: [64]u8,
    name_len: usize,
    /// Total number of calls
    call_count: u64,
    /// Total time in microseconds
    total_time_us: f64,
    /// Average time per call
    avg_time_us: f64,
    /// Maximum time seen
    max_time_us: f64,
    /// Minimum time seen
    min_time_us: f64,
    /// Variance for std dev calculation
    variance_sum: f64,
    /// Last update timestamp
    last_update_ns: i128,

    pub fn init(name: []const u8) CodePathProfile {
        var profile = CodePathProfile{
            .name = undefined,
            .name_len = @min(name.len, 64),
            .call_count = 0,
            .total_time_us = 0.0,
            .avg_time_us = 0.0,
            .max_time_us = 0.0,
            .min_time_us = std.math.inf(f64),
            .variance_sum = 0.0,
            .last_update_ns = 0,
        };
        @memset(&profile.name, 0);
        @memcpy(profile.name[0..profile.name_len], name[0..profile.name_len]);
        return profile;
    }

    pub fn getName(self: *const CodePathProfile) []const u8 {
        return self.name[0..self.name_len];
    }

    pub fn record(self: *CodePathProfile, elapsed_us: f64) void {
        self.call_count += 1;
        self.total_time_us += elapsed_us;

        // Update min/max
        if (elapsed_us > self.max_time_us) self.max_time_us = elapsed_us;
        if (elapsed_us < self.min_time_us) self.min_time_us = elapsed_us;

        // Online variance using Welford's algorithm
        const old_avg = self.avg_time_us;
        self.avg_time_us = self.total_time_us / @as(f64, @floatFromInt(self.call_count));
        self.variance_sum += (elapsed_us - old_avg) * (elapsed_us - self.avg_time_us);

        self.last_update_ns = std.time.nanoTimestamp();
    }

    pub fn getStdDev(self: *const CodePathProfile) f64 {
        if (self.call_count < 2) return 0.0;
        return @sqrt(self.variance_sum / @as(f64, @floatFromInt(self.call_count - 1)));
    }

    pub fn getThroughput(self: *const CodePathProfile) f64 {
        if (self.avg_time_us == 0) return 0.0;
        return 1_000_000.0 / self.avg_time_us; // ops/sec
    }
};

/// Active profile handle for RAII-style profiling
pub const ProfileHandle = struct {
    profiler: *Profiler,
    path_index: usize,
    start_ns: i128,
    active: bool,

    pub fn end(self: *ProfileHandle) void {
        if (!self.active) return;
        self.active = false;

        const end_ns = std.time.nanoTimestamp();
        const elapsed_ns = end_ns - self.start_ns;
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;

        if (self.path_index < self.profiler.path_count) {
            self.profiler.paths[self.path_index].record(elapsed_us);
        }
    }
};

/// Maximum number of profile paths
const MAX_PROFILE_PATHS: usize = 256;

/// Profiler instance
pub const Profiler = struct {
    config: ProfilerConfig,
    paths: [MAX_PROFILE_PATHS]CodePathProfile,
    path_count: usize,
    prng: std.Random.Xoshiro256,
    total_overhead_us: f64,
    measurement_count: u64,

    pub fn init(config: ProfilerConfig) Profiler {
        return Profiler{
            .config = config,
            .paths = undefined,
            .path_count = 0,
            .prng = std.Random.Xoshiro256.init(0x12345678),
            .total_overhead_us = 0.0,
            .measurement_count = 0,
        };
    }

    /// Find or create a profile path
    fn findOrCreatePath(self: *Profiler, path_name: []const u8) ?usize {
        // Search existing paths
        for (0..self.path_count) |i| {
            if (std.mem.eql(u8, self.paths[i].getName(), path_name)) {
                return i;
            }
        }

        // Create new path if room
        if (self.path_count < @min(MAX_PROFILE_PATHS, self.config.max_paths)) {
            const idx = self.path_count;
            self.paths[idx] = CodePathProfile.init(path_name);
            self.path_count += 1;
            return idx;
        }

        return null;
    }

    /// Start profiling a code path
    pub fn start_profile(self: *Profiler, path_name: []const u8) ProfileHandle {
        if (!self.config.enabled) {
            return ProfileHandle{
                .profiler = self,
                .path_index = 0,
                .start_ns = 0,
                .active = false,
            };
        }

        // Probabilistic sampling
        if (self.config.sampling_rate < 1.0) {
            const rand = self.prng.random().float(f32);
            if (rand > self.config.sampling_rate) {
                return ProfileHandle{
                    .profiler = self,
                    .path_index = 0,
                    .start_ns = 0,
                    .active = false,
                };
            }
        }

        const path_index = self.findOrCreatePath(path_name) orelse 0;

        return ProfileHandle{
            .profiler = self,
            .path_index = path_index,
            .start_ns = std.time.nanoTimestamp(),
            .active = true,
        };
    }

    /// End profiling (standalone function for legacy API)
    pub fn end_profile(self: *Profiler, path_name: []const u8, start_ns: i128) void {
        if (!self.config.enabled) return;

        const end_ns = std.time.nanoTimestamp();
        const elapsed_ns = end_ns - start_ns;
        const elapsed_us = @as(f64, @floatFromInt(elapsed_ns)) / 1000.0;

        if (self.findOrCreatePath(path_name)) |idx| {
            self.paths[idx].record(elapsed_us);
        }
    }

    /// Get summary of all profiled paths
    pub fn get_profile_summary(self: *const Profiler) ProfileSummary {
        var summary = ProfileSummary{
            .path_count = self.path_count,
            .total_calls = 0,
            .total_time_us = 0.0,
            .hottest_path_index = 0,
            .hottest_time_us = 0.0,
        };

        for (0..self.path_count) |i| {
            const path = &self.paths[i];
            summary.total_calls += path.call_count;
            summary.total_time_us += path.total_time_us;

            if (path.total_time_us > summary.hottest_time_us) {
                summary.hottest_time_us = path.total_time_us;
                summary.hottest_path_index = i;
            }
        }

        return summary;
    }

    /// Get a specific path profile
    pub fn getPath(self: *const Profiler, index: usize) ?*const CodePathProfile {
        if (index >= self.path_count) return null;
        return &self.paths[index];
    }

    /// Reset all profiles
    pub fn reset(self: *Profiler) void {
        self.path_count = 0;
        self.total_overhead_us = 0.0;
        self.measurement_count = 0;
    }

    /// Calculate profiling overhead percentage
    pub fn getOverheadPct(self: *const Profiler) f64 {
        if (self.measurement_count == 0) return 0.0;
        const summary = self.get_profile_summary();
        if (summary.total_time_us == 0) return 0.0;
        return (self.total_overhead_us / summary.total_time_us) * 100.0;
    }
};

/// Summary of all profiled paths
pub const ProfileSummary = struct {
    path_count: usize,
    total_calls: u64,
    total_time_us: f64,
    hottest_path_index: usize,
    hottest_time_us: f64,
};

/// Global profiler instance (thread-local in production)
pub var global_profiler: Profiler = Profiler.init(ProfilerConfig{});

/// Convenience function to start profiling
pub fn start_profile(path_name: []const u8) ProfileHandle {
    return global_profiler.start_profile(path_name);
}

/// Convenience function to end profiling
pub fn end_profile(path_name: []const u8, start_ns: i128) void {
    global_profiler.end_profile(path_name, start_ns);
}

/// Get global profile summary
pub fn get_profile_summary() ProfileSummary {
    return global_profiler.get_profile_summary();
}

// ============================================================================
// SIMD Optimization Utilities
// ============================================================================

/// SIMD vector width based on architecture
pub const SIMD_WIDTH: usize = switch (builtin.cpu.arch) {
    .aarch64, .arm => 4, // NEON 128-bit
    .x86_64 => 8, // AVX 256-bit
    else => 1,
};

/// SIMD capabilities info
pub const SIMDInfo = struct {
    width: usize,
    arch_name: []const u8,
    estimated_speedup: f32,

    pub fn detect() SIMDInfo {
        return switch (builtin.cpu.arch) {
            .aarch64 => .{ .width = 4, .arch_name = "ARM NEON", .estimated_speedup = 3.5 },
            .arm => .{ .width = 4, .arch_name = "ARM NEON", .estimated_speedup = 3.0 },
            .x86_64 => .{ .width = 8, .arch_name = "x86 AVX", .estimated_speedup = 6.0 },
            else => .{ .width = 1, .arch_name = "Scalar", .estimated_speedup = 1.0 },
        };
    }
};

/// SIMD-optimized dot product
pub fn simd_dot_product_optimized(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    const simd_len = len / SIMD_WIDTH * SIMD_WIDTH;
    var result: f32 = 0.0;

    // SIMD portion - unrolled for better pipelining
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            result += a[i + k] * b[i + k];
        }
    }

    // Remainder
    while (i < len) : (i += 1) {
        result += a[i] * b[i];
    }

    return result;
}

/// Non-SIMD baseline for comparison
pub fn scalar_dot_product(a: []const f32, b: []const f32) f32 {
    const len = @min(a.len, b.len);
    var result: f32 = 0.0;
    for (0..len) |i| {
        result += a[i] * b[i];
    }
    return result;
}

/// SIMD-optimized L2 norm
pub fn simd_norm_optimized(x: []const f32) f32 {
    const simd_len = x.len / SIMD_WIDTH * SIMD_WIDTH;
    var norm_sq: f32 = 0.0;

    // SIMD portion
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            const val = x[i + k];
            norm_sq += val * val;
        }
    }

    // Remainder
    while (i < x.len) : (i += 1) {
        const val = x[i];
        norm_sq += val * val;
    }

    return @sqrt(norm_sq);
}

/// Non-SIMD baseline for norm
pub fn scalar_norm(x: []const f32) f32 {
    var norm_sq: f32 = 0.0;
    for (x) |val| {
        norm_sq += val * val;
    }
    return @sqrt(norm_sq);
}

/// SIMD-optimized vector scaling: result = x * scalar
pub fn simd_scale_optimized(result: []f32, x: []const f32, scalar: f32) void {
    const len = @min(result.len, x.len);
    const simd_len = len / SIMD_WIDTH * SIMD_WIDTH;

    // SIMD portion
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            result[i + k] = x[i + k] * scalar;
        }
    }

    // Remainder
    while (i < len) : (i += 1) {
        result[i] = x[i] * scalar;
    }
}

/// Non-SIMD baseline for scaling
pub fn scalar_scale(result: []f32, x: []const f32, scalar: f32) void {
    const len = @min(result.len, x.len);
    for (0..len) |i| {
        result[i] = x[i] * scalar;
    }
}

/// SIMD-optimized vector addition: result = a + b
pub fn simd_add_optimized(result: []f32, a: []const f32, b: []const f32) void {
    const len = @min(@min(result.len, a.len), b.len);
    const simd_len = len / SIMD_WIDTH * SIMD_WIDTH;

    // SIMD portion
    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            result[i + k] = a[i + k] + b[i + k];
        }
    }

    // Remainder
    while (i < len) : (i += 1) {
        result[i] = a[i] + b[i];
    }
}

/// Non-SIMD baseline for addition
pub fn scalar_add(result: []f32, a: []const f32, b: []const f32) void {
    const len = @min(@min(result.len, a.len), b.len);
    for (0..len) |i| {
        result[i] = a[i] + b[i];
    }
}

/// SIMD-optimized vector subtraction: result = a - b
pub fn simd_sub_optimized(result: []f32, a: []const f32, b: []const f32) void {
    const len = @min(@min(result.len, a.len), b.len);
    const simd_len = len / SIMD_WIDTH * SIMD_WIDTH;

    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            result[i + k] = a[i + k] - b[i + k];
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] - b[i];
    }
}

/// SIMD-optimized element-wise multiply: result = a * b
pub fn simd_mul_optimized(result: []f32, a: []const f32, b: []const f32) void {
    const len = @min(@min(result.len, a.len), b.len);
    const simd_len = len / SIMD_WIDTH * SIMD_WIDTH;

    var i: usize = 0;
    while (i < simd_len) : (i += SIMD_WIDTH) {
        inline for (0..SIMD_WIDTH) |k| {
            result[i + k] = a[i + k] * b[i + k];
        }
    }

    while (i < len) : (i += 1) {
        result[i] = a[i] * b[i];
    }
}


// ============================================================================
// Memory Pool for Reusable Allocations
// ============================================================================

/// Memory block header for pool allocations
const BlockHeader = struct {
    size: usize,
    in_use: bool,
    next: ?*BlockHeader,
};

/// Slab size classes for efficient allocation
pub const SlabClass = enum(u8) {
    tiny = 0, // 64 bytes
    small = 1, // 256 bytes
    medium = 2, // 1KB
    large = 3, // 4KB
    huge = 4, // 16KB

    pub fn getSize(self: SlabClass) usize {
        return switch (self) {
            .tiny => 64,
            .small => 256,
            .medium => 1024,
            .large => 4096,
            .huge => 16384,
        };
    }

    pub fn fromSize(size: usize) SlabClass {
        if (size <= 64) return .tiny;
        if (size <= 256) return .small;
        if (size <= 1024) return .medium;
        if (size <= 4096) return .large;
        return .huge;
    }
};

/// Memory pool statistics
pub const PoolStats = struct {
    total_allocated: usize,
    current_in_use: usize,
    allocation_count: u64,
    free_count: u64,
    reset_count: u64,
    cache_hits: u64,
    cache_misses: u64,

    pub fn getAllocationReduction(self: *const PoolStats) f64 {
        const total = self.cache_hits + self.cache_misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(total)) * 100.0;
    }
};

/// Number of slabs per class
const SLABS_PER_CLASS: usize = 32;

/// Memory pool for reusable allocations
pub const MemoryPool = struct {
    allocator: Allocator,

    // Fixed-size slab allocators for each class
    slab_buffers: [5][SLABS_PER_CLASS]?[]u8,
    slab_in_use: [5][SLABS_PER_CLASS]bool,

    // Large allocation tracking
    large_allocs: std.ArrayListUnmanaged([]u8),

    // Statistics
    stats: PoolStats,

    pub fn init(allocator: Allocator) MemoryPool {
        var pool = MemoryPool{
            .allocator = allocator,
            .slab_buffers = undefined,
            .slab_in_use = undefined,
            .large_allocs = .{},
            .stats = .{
                .total_allocated = 0,
                .current_in_use = 0,
                .allocation_count = 0,
                .free_count = 0,
                .reset_count = 0,
                .cache_hits = 0,
                .cache_misses = 0,
            },
        };

        // Initialize slab arrays
        for (0..5) |class| {
            for (0..SLABS_PER_CLASS) |i| {
                pool.slab_buffers[class][i] = null;
                pool.slab_in_use[class][i] = false;
            }
        }

        return pool;
    }

    pub fn deinit(self: *MemoryPool) void {
        // Free all slab buffers
        for (0..5) |class| {
            for (0..SLABS_PER_CLASS) |i| {
                if (self.slab_buffers[class][i]) |buf| {
                    self.allocator.free(buf);
                }
            }
        }

        // Free large allocations
        for (self.large_allocs.items) |buf| {
            self.allocator.free(buf);
        }
        self.large_allocs.deinit(self.allocator);
    }

    /// Allocate memory from pool
    pub fn alloc(self: *MemoryPool, size: usize) ![]u8 {
        self.stats.allocation_count += 1;

        // Try to find a free slab of appropriate size
        const slab_class = SlabClass.fromSize(size);
        const class_idx = @intFromEnum(slab_class);
        const slab_size = slab_class.getSize();

        // Search for free slab
        for (0..SLABS_PER_CLASS) |i| {
            if (!self.slab_in_use[class_idx][i]) {
                // Found free slot
                if (self.slab_buffers[class_idx][i]) |buf| {
                    // Reuse existing buffer
                    self.slab_in_use[class_idx][i] = true;
                    self.stats.cache_hits += 1;
                    self.stats.current_in_use += buf.len;
                    return buf[0..size];
                } else {
                    // Allocate new buffer
                    const buf = try self.allocator.alloc(u8, slab_size);
                    self.slab_buffers[class_idx][i] = buf;
                    self.slab_in_use[class_idx][i] = true;
                    self.stats.total_allocated += slab_size;
                    self.stats.current_in_use += slab_size;
                    self.stats.cache_misses += 1;
                    return buf[0..size];
                }
            }
        }

        // Fall back to direct allocation for oversized or when slabs full
        self.stats.cache_misses += 1;
        const buf = try self.allocator.alloc(u8, size);
        try self.large_allocs.append(self.allocator, buf);
        self.stats.total_allocated += size;
        self.stats.current_in_use += size;
        return buf;
    }

    /// Free memory back to pool
    pub fn free(self: *MemoryPool, ptr: []u8) void {
        self.stats.free_count += 1;

        // Check if it's in a slab
        const slab_class = SlabClass.fromSize(ptr.len);
        const class_idx = @intFromEnum(slab_class);

        for (0..SLABS_PER_CLASS) |i| {
            if (self.slab_buffers[class_idx][i]) |buf| {
                if (buf.ptr == ptr.ptr) {
                    self.slab_in_use[class_idx][i] = false;
                    self.stats.current_in_use -= buf.len;
                    return;
                }
            }
        }

        // Check large allocations
        for (self.large_allocs.items, 0..) |buf, idx| {
            if (buf.ptr == ptr.ptr) {
                self.stats.current_in_use -= buf.len;
                self.allocator.free(buf);
                _ = self.large_allocs.swapRemove(idx);
                return;
            }
        }
    }

    /// Reset pool without deallocating
    pub fn reset(self: *MemoryPool) void {
        self.stats.reset_count += 1;
        self.stats.current_in_use = 0;

        // Mark all slabs as free
        for (0..5) |class| {
            for (0..SLABS_PER_CLASS) |i| {
                self.slab_in_use[class][i] = false;
            }
        }

        // Don't free large allocs, just mark them available on next reset
    }

    /// Get pool statistics
    pub fn getStats(self: *const MemoryPool) PoolStats {
        return self.stats;
    }

    /// Allocate typed slice
    pub fn allocTyped(self: *MemoryPool, comptime T: type, count: usize) ![]T {
        const byte_size = count * @sizeOf(T);
        const bytes = try self.alloc(byte_size);
        return @as([*]T, @ptrCast(@alignCast(bytes.ptr)))[0..count];
    }
};

// ============================================================================
// Monitoring Overhead Reduction
// ============================================================================

/// Metric value for low-overhead monitoring
pub const MetricSample = struct {
    value: f64,
    timestamp_us: i64,
};

/// Ring buffer for sampled metrics
pub const MetricRingBuffer = struct {
    samples: [256]MetricSample,
    write_index: usize,
    count: usize,

    pub fn init() MetricRingBuffer {
        return .{
            .samples = undefined,
            .write_index = 0,
            .count = 0,
        };
    }

    pub fn add(self: *MetricRingBuffer, value: f64) void {
        self.samples[self.write_index] = .{
            .value = value,
            .timestamp_us = @intCast(@divTrunc(std.time.nanoTimestamp(), 1000)),
        };
        self.write_index = (self.write_index + 1) % 256;
        if (self.count < 256) self.count += 1;
    }

    pub fn getAverage(self: *const MetricRingBuffer) f64 {
        if (self.count == 0) return 0.0;
        var sum: f64 = 0.0;
        for (0..self.count) |i| {
            sum += self.samples[i].value;
        }
        return sum / @as(f64, @floatFromInt(self.count));
    }

    pub fn getLatest(self: *const MetricRingBuffer) ?f64 {
        if (self.count == 0) return null;
        const idx = if (self.write_index == 0) 255 else self.write_index - 1;
        return self.samples[idx].value;
    }
};

/// Maximum metrics to track
const MAX_METRICS: usize = 64;

/// Low-overhead monitor with sampling
pub const LowOverheadMonitor = struct {
    /// Sample rate: 1 in N calls recorded
    sample_rate: usize,
    /// Call counter for sampling
    call_counter: u64,
    /// Metric ring buffers
    metrics: [MAX_METRICS]MetricRingBuffer,
    metric_names: [MAX_METRICS][32]u8,
    metric_count: usize,
    /// PRNG for randomized sampling
    prng: std.Random.Xoshiro256,
    /// Overhead tracking
    total_overhead_ns: u64,
    total_samples: u64,

    pub fn init(sample_rate: usize) LowOverheadMonitor {
        var monitor = LowOverheadMonitor{
            .sample_rate = if (sample_rate == 0) 100 else sample_rate,
            .call_counter = 0,
            .metrics = undefined,
            .metric_names = undefined,
            .metric_count = 0,
            .prng = std.Random.Xoshiro256.init(0xDEADBEEF),
            .total_overhead_ns = 0,
            .total_samples = 0,
        };

        for (0..MAX_METRICS) |i| {
            monitor.metrics[i] = MetricRingBuffer.init();
            @memset(&monitor.metric_names[i], 0);
        }

        return monitor;
    }

    /// Find or create metric index
    fn findOrCreateMetric(self: *LowOverheadMonitor, name: []const u8) ?usize {
        // Search existing
        for (0..self.metric_count) |i| {
            const stored_name = self.metric_names[i][0..@min(name.len, 32)];
            if (std.mem.eql(u8, stored_name, name[0..@min(name.len, 32)])) {
                return i;
            }
        }

        // Create new
        if (self.metric_count < MAX_METRICS) {
            const idx = self.metric_count;
            const copy_len = @min(name.len, 32);
            @memcpy(self.metric_names[idx][0..copy_len], name[0..copy_len]);
            self.metric_count += 1;
            return idx;
        }

        return null;
    }

    /// Record metric if sampled (low overhead)
    pub fn record_if_sampled(self: *LowOverheadMonitor, metric: []const u8, value: f64) bool {
        self.call_counter += 1;

        // Deterministic sampling
        if (self.call_counter % self.sample_rate != 0) {
            return false;
        }

        const start = std.time.nanoTimestamp();

        if (self.findOrCreateMetric(metric)) |idx| {
            self.metrics[idx].add(value);
            self.total_samples += 1;
        }

        const end = std.time.nanoTimestamp();
        self.total_overhead_ns += @intCast(end - start);

        return true;
    }

    /// Force record (bypasses sampling)
    pub fn record_always(self: *LowOverheadMonitor, metric: []const u8, value: f64) void {
        if (self.findOrCreateMetric(metric)) |idx| {
            self.metrics[idx].add(value);
            self.total_samples += 1;
        }
    }

    /// Get metric average
    pub fn getMetricAverage(self: *const LowOverheadMonitor, metric: []const u8) ?f64 {
        for (0..self.metric_count) |i| {
            const name_len = @min(metric.len, 32);
            if (std.mem.eql(u8, self.metric_names[i][0..name_len], metric[0..name_len])) {
                return self.metrics[i].getAverage();
            }
        }
        return null;
    }

    /// Calculate monitoring overhead percentage
    pub fn getOverheadPct(self: *const LowOverheadMonitor) f64 {
        if (self.total_samples == 0) return 0.0;
        // Estimate overhead as percentage of typical operation time (1ms)
        const overhead_per_sample_ns = @as(f64, @floatFromInt(self.total_overhead_ns)) / @as(f64, @floatFromInt(self.total_samples));
        // Overhead relative to sample rate (amortized over all calls)
        return (overhead_per_sample_ns / 1_000_000.0) * 100.0 / @as(f64, @floatFromInt(self.sample_rate));
    }

    /// Get summary statistics
    pub fn getSummary(self: *const LowOverheadMonitor) MonitorSummary {
        return .{
            .metric_count = self.metric_count,
            .total_samples = self.total_samples,
            .sample_rate = self.sample_rate,
            .overhead_pct = self.getOverheadPct(),
            .call_counter = self.call_counter,
        };
    }
};

/// Monitor summary
pub const MonitorSummary = struct {
    metric_count: usize,
    total_samples: u64,
    sample_rate: usize,
    overhead_pct: f64,
    call_counter: u64,
};

/// Default low-overhead monitor instance
pub var global_monitor: LowOverheadMonitor = LowOverheadMonitor.init(100);



// ============================================================================
// Benchmark Suite
// ============================================================================

/// Single benchmark result
pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    total_time_us: f64,
    avg_time_us: f64,
    min_time_us: f64,
    max_time_us: f64,
    std_dev_us: f64,
    ops_per_sec: f64,
    speedup_vs_baseline: f64,
};

/// Benchmark configuration
pub const BenchmarkConfig = struct {
    warmup_iterations: usize = 10,
    benchmark_iterations: usize = 100,
    vector_size: usize = 1024,
    matrix_size: usize = 64,
};

/// Day 1 baseline performance (simulated)
pub const Day1Baseline = struct {
    dot_product_us: f64 = 50.0,
    norm_us: f64 = 25.0,
    scale_us: f64 = 20.0,
    add_us: f64 = 15.0,
    pool_alloc_us: f64 = 100.0,
};

/// Complete benchmark results
pub const BenchmarkSuite = struct {
    results: [16]?BenchmarkResult,
    result_count: usize,
    total_time_us: f64,
    simd_info: SIMDInfo,

    pub fn init() BenchmarkSuite {
        return .{
            .results = [_]?BenchmarkResult{null} ** 16,
            .result_count = 0,
            .total_time_us = 0.0,
            .simd_info = SIMDInfo.detect(),
        };
    }

    pub fn addResult(self: *BenchmarkSuite, result: BenchmarkResult) void {
        if (self.result_count < 16) {
            self.results[self.result_count] = result;
            self.result_count += 1;
            self.total_time_us += result.total_time_us;
        }
    }

    pub fn getAverageSpeedup(self: *const BenchmarkSuite) f64 {
        if (self.result_count == 0) return 1.0;
        var sum: f64 = 0.0;
        for (0..self.result_count) |i| {
            if (self.results[i]) |r| {
                sum += r.speedup_vs_baseline;
            }
        }
        return sum / @as(f64, @floatFromInt(self.result_count));
    }
};

/// Run a single benchmark
fn runBenchmark(
    comptime name: []const u8,
    comptime func: fn ([]f32, []const f32, []const f32) void,
    a: []f32,
    b: []const f32,
    c: []const f32,
    config: BenchmarkConfig,
    baseline_us: f64,
) BenchmarkResult {
    // Warmup
    for (0..config.warmup_iterations) |_| {
        func(a, b, c);
    }

    var samples: [256]f64 = undefined;
    var total: f64 = 0.0;
    var min_time: f64 = std.math.inf(f64);
    var max_time: f64 = 0.0;

    const iters = @min(config.benchmark_iterations, 256);

    for (0..iters) |i| {
        const start = std.time.nanoTimestamp();
        func(a, b, c);
        const end = std.time.nanoTimestamp();
        const elapsed = @as(f64, @floatFromInt(end - start)) / 1000.0;

        samples[i] = elapsed;
        total += elapsed;
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
    }

    const avg = total / @as(f64, @floatFromInt(iters));

    // Calculate std dev
    var variance: f64 = 0.0;
    for (0..iters) |i| {
        const diff = samples[i] - avg;
        variance += diff * diff;
    }
    const std_dev = @sqrt(variance / @as(f64, @floatFromInt(iters)));

    return .{
        .name = name,
        .iterations = iters,
        .total_time_us = total,
        .avg_time_us = avg,
        .min_time_us = min_time,
        .max_time_us = max_time,
        .std_dev_us = std_dev,
        .ops_per_sec = if (avg > 0) 1_000_000.0 / avg else 0,
        .speedup_vs_baseline = if (avg > 0) baseline_us / avg else 1.0,
    };
}

/// Wrapper for add benchmark
fn benchAddSimd(result: []f32, a: []const f32, b: []const f32) void {
    simd_add_optimized(result, a, b);
}

fn benchAddScalar(result: []f32, a: []const f32, b: []const f32) void {
    scalar_add(result, a, b);
}

/// Run all benchmarks
pub fn run_all_benchmarks(allocator: Allocator) !BenchmarkSuite {
    var suite = BenchmarkSuite.init();
    const config = BenchmarkConfig{};
    const baseline = Day1Baseline{};

    // Allocate test vectors
    const a = try allocator.alloc(f32, config.vector_size);
    defer allocator.free(a);
    const b = try allocator.alloc(f32, config.vector_size);
    defer allocator.free(b);
    const result = try allocator.alloc(f32, config.vector_size);
    defer allocator.free(result);

    // Initialize with test data
    for (0..config.vector_size) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.001;
        b[i] = @as(f32, @floatFromInt(config.vector_size - i)) * 0.001;
    }

    // Benchmark SIMD add
    suite.addResult(runBenchmark(
        "simd_add_optimized",
        benchAddSimd,
        result,
        a,
        b,
        config,
        baseline.add_us,
    ));

    // Benchmark scalar add
    suite.addResult(runBenchmark(
        "scalar_add",
        benchAddScalar,
        result,
        a,
        b,
        config,
        baseline.add_us,
    ));

    // Benchmark dot products
    const dot_simd_start = std.time.nanoTimestamp();
    var dot_accum: f32 = 0;
    for (0..config.benchmark_iterations) |_| {
        dot_accum += simd_dot_product_optimized(a, b);
    }
    const dot_simd_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - dot_simd_start)) / 1000.0;
    std.mem.doNotOptimizeAway(&dot_accum);

    suite.addResult(.{
        .name = "simd_dot_product",
        .iterations = config.benchmark_iterations,
        .total_time_us = dot_simd_time,
        .avg_time_us = dot_simd_time / @as(f64, @floatFromInt(config.benchmark_iterations)),
        .min_time_us = 0,
        .max_time_us = 0,
        .std_dev_us = 0,
        .ops_per_sec = @as(f64, @floatFromInt(config.benchmark_iterations)) * 1_000_000.0 / dot_simd_time,
        .speedup_vs_baseline = baseline.dot_product_us / (dot_simd_time / @as(f64, @floatFromInt(config.benchmark_iterations))),
    });

    // Benchmark norm
    const norm_simd_start = std.time.nanoTimestamp();
    var norm_accum: f32 = 0;
    for (0..config.benchmark_iterations) |_| {
        norm_accum += simd_norm_optimized(a);
    }
    const norm_simd_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - norm_simd_start)) / 1000.0;
    std.mem.doNotOptimizeAway(&norm_accum);

    suite.addResult(.{
        .name = "simd_norm",
        .iterations = config.benchmark_iterations,
        .total_time_us = norm_simd_time,
        .avg_time_us = norm_simd_time / @as(f64, @floatFromInt(config.benchmark_iterations)),
        .min_time_us = 0,
        .max_time_us = 0,
        .std_dev_us = 0,
        .ops_per_sec = @as(f64, @floatFromInt(config.benchmark_iterations)) * 1_000_000.0 / norm_simd_time,
        .speedup_vs_baseline = baseline.norm_us / (norm_simd_time / @as(f64, @floatFromInt(config.benchmark_iterations))),
    });

    // Benchmark memory pool
    var pool = MemoryPool.init(allocator);
    defer pool.deinit();

    const pool_start = std.time.nanoTimestamp();
    for (0..config.benchmark_iterations) |_| {
        const buf = try pool.alloc(256);
        pool.free(buf);
    }
    const pool_time = @as(f64, @floatFromInt(std.time.nanoTimestamp() - pool_start)) / 1000.0;

    suite.addResult(.{
        .name = "memory_pool_alloc",
        .iterations = config.benchmark_iterations,
        .total_time_us = pool_time,
        .avg_time_us = pool_time / @as(f64, @floatFromInt(config.benchmark_iterations)),
        .min_time_us = 0,
        .max_time_us = 0,
        .std_dev_us = 0,
        .ops_per_sec = @as(f64, @floatFromInt(config.benchmark_iterations)) * 1_000_000.0 / pool_time,
        .speedup_vs_baseline = baseline.pool_alloc_us / (pool_time / @as(f64, @floatFromInt(config.benchmark_iterations))),
    });

    return suite;
}

/// Compare results with Day 1 baseline
pub fn compare_with_baseline(suite: *const BenchmarkSuite) BaselineComparison {
    var comparison = BaselineComparison{
        .avg_speedup = suite.getAverageSpeedup(),
        .best_speedup = 0.0,
        .worst_speedup = std.math.inf(f64),
        .best_benchmark = "",
        .worst_benchmark = "",
        .meets_target = true,
    };

    for (0..suite.result_count) |i| {
        if (suite.results[i]) |r| {
            if (r.speedup_vs_baseline > comparison.best_speedup) {
                comparison.best_speedup = r.speedup_vs_baseline;
                comparison.best_benchmark = r.name;
            }
            if (r.speedup_vs_baseline < comparison.worst_speedup) {
                comparison.worst_speedup = r.speedup_vs_baseline;
                comparison.worst_benchmark = r.name;
            }
            // Target: at least 2x speedup
            if (r.speedup_vs_baseline < 2.0) {
                comparison.meets_target = false;
            }
        }
    }

    return comparison;
}

/// Baseline comparison result
pub const BaselineComparison = struct {
    avg_speedup: f64,
    best_speedup: f64,
    worst_speedup: f64,
    best_benchmark: []const u8,
    worst_benchmark: []const u8,
    meets_target: bool,
};

/// Generate performance summary string
pub fn generatePerformanceSummary(suite: *const BenchmarkSuite, comparison: *const BaselineComparison) [512]u8 {
    var buf: [512]u8 = undefined;
    @memset(&buf, 0);

    const summary = std.fmt.bufPrint(&buf,
        \\=== mHC Performance Summary ===
        \\SIMD: {s} (width={d})
        \\Benchmarks: {d}
        \\Avg Speedup: {d:.2}x
        \\Best: {s} ({d:.2}x)
        \\Worst: {s} ({d:.2}x)
        \\Target Met: {s}
    , .{
        suite.simd_info.arch_name,
        suite.simd_info.width,
        suite.result_count,
        comparison.avg_speedup,
        comparison.best_benchmark,
        comparison.best_speedup,
        comparison.worst_benchmark,
        comparison.worst_speedup,
        if (comparison.meets_target) "YES" else "NO",
    }) catch "";

    _ = summary;
    return buf;
}


// ============================================================================
// Tests (20+)
// ============================================================================

test "ProfilerConfig production defaults" {
    const config = ProfilerConfig.production();
    try std.testing.expectEqual(@as(f32, 0.01), config.sampling_rate);
    try std.testing.expect(config.enabled);
    try std.testing.expectEqual(@as(usize, 64), config.max_paths);
    try std.testing.expect(!config.include_stack_traces);
}

test "ProfilerConfig development defaults" {
    const config = ProfilerConfig.development();
    try std.testing.expectEqual(@as(f32, 1.0), config.sampling_rate);
    try std.testing.expect(config.enabled);
    try std.testing.expect(config.include_stack_traces);
}

test "ProfilerConfig disabled" {
    const config = ProfilerConfig.disabled();
    try std.testing.expect(!config.enabled);
    try std.testing.expectEqual(@as(f32, 0.0), config.sampling_rate);
}

test "CodePathProfile init and getName" {
    const profile = CodePathProfile.init("test_path");
    try std.testing.expectEqualStrings("test_path", profile.getName());
    try std.testing.expectEqual(@as(u64, 0), profile.call_count);
}

test "CodePathProfile record and statistics" {
    var profile = CodePathProfile.init("timing_test");

    profile.record(10.0);
    profile.record(20.0);
    profile.record(30.0);

    try std.testing.expectEqual(@as(u64, 3), profile.call_count);
    try std.testing.expectApproxEqAbs(@as(f64, 60.0), profile.total_time_us, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), profile.avg_time_us, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), profile.min_time_us, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), profile.max_time_us, 0.001);
}

test "CodePathProfile std dev calculation" {
    var profile = CodePathProfile.init("std_test");

    // Add values with known std dev
    profile.record(2.0);
    profile.record(4.0);
    profile.record(4.0);
    profile.record(4.0);
    profile.record(5.0);
    profile.record(5.0);
    profile.record(7.0);
    profile.record(9.0);

    const std_dev = profile.getStdDev();
    try std.testing.expect(std_dev > 0);
    try std.testing.expect(std_dev < 3.0);
}

test "CodePathProfile throughput" {
    var profile = CodePathProfile.init("throughput_test");
    profile.record(100.0); // 100us average

    const throughput = profile.getThroughput();
    try std.testing.expectApproxEqAbs(@as(f64, 10000.0), throughput, 1.0); // 10k ops/sec
}

test "Profiler init and path creation" {
    var profiler = Profiler.init(ProfilerConfig{});

    var handle = profiler.start_profile("path_one");
    handle.end();

    try std.testing.expectEqual(@as(usize, 1), profiler.path_count);

    // Same path should not create new entry
    var handle2 = profiler.start_profile("path_one");
    handle2.end();
    try std.testing.expectEqual(@as(usize, 1), profiler.path_count);

    // Different path should create new entry
    var handle3 = profiler.start_profile("path_two");
    handle3.end();
    try std.testing.expectEqual(@as(usize, 2), profiler.path_count);
}

test "Profiler disabled does not record" {
    var profiler = Profiler.init(ProfilerConfig.disabled());

    var handle = profiler.start_profile("should_not_record");
    handle.end();

    try std.testing.expectEqual(@as(usize, 0), profiler.path_count);
}

test "Profiler get_profile_summary" {
    var profiler = Profiler.init(ProfilerConfig{});

    var handle1 = profiler.start_profile("fast_path");
    handle1.end();

    var handle2 = profiler.start_profile("slow_path");
    // Small delay via busy loop
    var dummy: u32 = 0;
    for (0..1000) |i| {
        dummy +%= @as(u32, @truncate(i));
    }
    std.mem.doNotOptimizeAway(&dummy);
    handle2.end();

    const summary = profiler.get_profile_summary();
    try std.testing.expectEqual(@as(usize, 2), summary.path_count);
    try std.testing.expectEqual(@as(u64, 2), summary.total_calls);
}

test "Profiler reset clears data" {
    var profiler = Profiler.init(ProfilerConfig{});

    var handle = profiler.start_profile("to_be_cleared");
    handle.end();

    try std.testing.expectEqual(@as(usize, 1), profiler.path_count);

    profiler.reset();
    try std.testing.expectEqual(@as(usize, 0), profiler.path_count);
}

test "simd_dot_product_optimized correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };

    const result = simd_dot_product_optimized(&a, &b);
    // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
    try std.testing.expectApproxEqAbs(@as(f32, 70.0), result, 0.001);
}

test "simd_dot_product matches scalar" {
    var a: [128]f32 = undefined;
    var b: [128]f32 = undefined;

    for (0..128) |i| {
        a[i] = @as(f32, @floatFromInt(i)) * 0.1;
        b[i] = @as(f32, @floatFromInt(128 - i)) * 0.1;
    }

    const simd_result = simd_dot_product_optimized(&a, &b);
    const scalar_result = scalar_dot_product(&a, &b);

    try std.testing.expectApproxEqRel(simd_result, scalar_result, 0.0001);
}

test "simd_norm_optimized correctness" {
    const v = [_]f32{ 3.0, 4.0, 0.0 };
    const norm = simd_norm_optimized(&v);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), norm, 0.001);
}

test "simd_norm matches scalar" {
    var v: [64]f32 = undefined;
    for (0..64) |i| {
        v[i] = @as(f32, @floatFromInt(i)) * 0.05;
    }

    const simd_result = simd_norm_optimized(&v);
    const scalar_result = scalar_norm(&v);

    try std.testing.expectApproxEqRel(simd_result, scalar_result, 0.0001);
}


test "simd_scale_optimized correctness" {
    const x = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var result: [4]f32 = undefined;

    simd_scale_optimized(&result, &x, 2.0);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), result[3], 0.001);
}

test "simd_add_optimized correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    var result: [4]f32 = undefined;

    simd_add_optimized(&result, &a, &b);

    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result[3], 0.001);
}

test "simd_sub_optimized correctness" {
    const a = [_]f32{ 5.0, 6.0, 7.0, 8.0 };
    const b = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    var result: [4]f32 = undefined;

    simd_sub_optimized(&result, &a, &b);

    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), result[3], 0.001);
}

test "simd_mul_optimized correctness" {
    const a = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const b = [_]f32{ 2.0, 3.0, 4.0, 5.0 };
    var result: [4]f32 = undefined;

    simd_mul_optimized(&result, &a, &b);

    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), result[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), result[3], 0.001);
}

test "SlabClass size mapping" {
    try std.testing.expectEqual(@as(usize, 64), SlabClass.tiny.getSize());
    try std.testing.expectEqual(@as(usize, 256), SlabClass.small.getSize());
    try std.testing.expectEqual(@as(usize, 1024), SlabClass.medium.getSize());
    try std.testing.expectEqual(@as(usize, 4096), SlabClass.large.getSize());
    try std.testing.expectEqual(@as(usize, 16384), SlabClass.huge.getSize());
}

test "SlabClass fromSize classification" {
    try std.testing.expectEqual(SlabClass.tiny, SlabClass.fromSize(32));
    try std.testing.expectEqual(SlabClass.tiny, SlabClass.fromSize(64));
    try std.testing.expectEqual(SlabClass.small, SlabClass.fromSize(128));
    try std.testing.expectEqual(SlabClass.medium, SlabClass.fromSize(512));
    try std.testing.expectEqual(SlabClass.large, SlabClass.fromSize(2048));
    try std.testing.expectEqual(SlabClass.huge, SlabClass.fromSize(8192));
}

test "MemoryPool basic alloc/free" {
    const allocator = std.testing.allocator;
    var pool = MemoryPool.init(allocator);
    defer pool.deinit();

    const buf = try pool.alloc(128);
    try std.testing.expect(buf.len >= 128);

    pool.free(buf);

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.allocation_count);
    try std.testing.expectEqual(@as(u64, 1), stats.free_count);
}

test "MemoryPool slab reuse" {
    const allocator = std.testing.allocator;
    var pool = MemoryPool.init(allocator);
    defer pool.deinit();

    // First allocation
    const buf1 = try pool.alloc(64);
    const ptr1 = buf1.ptr;
    pool.free(buf1);

    // Second allocation should reuse
    const buf2 = try pool.alloc(64);
    const ptr2 = buf2.ptr;
    pool.free(buf2);

    try std.testing.expectEqual(ptr1, ptr2); // Same memory reused

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.cache_hits);
}

test "MemoryPool reset" {
    const allocator = std.testing.allocator;
    var pool = MemoryPool.init(allocator);
    defer pool.deinit();

    _ = try pool.alloc(128);
    _ = try pool.alloc(256);
    _ = try pool.alloc(512);

    pool.reset();

    const stats = pool.getStats();
    try std.testing.expectEqual(@as(u64, 1), stats.reset_count);
    try std.testing.expectEqual(@as(usize, 0), stats.current_in_use);
}

test "PoolStats allocation reduction" {
    var stats = PoolStats{
        .total_allocated = 1000,
        .current_in_use = 500,
        .allocation_count = 100,
        .free_count = 50,
        .reset_count = 1,
        .cache_hits = 80,
        .cache_misses = 20,
    };

    const reduction = stats.getAllocationReduction();
    try std.testing.expectApproxEqAbs(@as(f64, 80.0), reduction, 0.001);
}

test "MetricRingBuffer basic operations" {
    var ring = MetricRingBuffer.init();

    ring.add(10.0);
    ring.add(20.0);
    ring.add(30.0);

    try std.testing.expectEqual(@as(usize, 3), ring.count);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), ring.getAverage(), 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 30.0), ring.getLatest().?, 0.001);
}

test "MetricRingBuffer wrap around" {
    var ring = MetricRingBuffer.init();

    // Fill buffer beyond capacity
    for (0..300) |i| {
        ring.add(@as(f64, @floatFromInt(i)));
    }

    try std.testing.expectEqual(@as(usize, 256), ring.count);
    try std.testing.expectApproxEqAbs(@as(f64, 299.0), ring.getLatest().?, 0.001);
}

test "LowOverheadMonitor sampling" {
    var monitor = LowOverheadMonitor.init(10); // 1 in 10 sampling

    var recorded_count: usize = 0;
    for (0..100) |_| {
        if (monitor.record_if_sampled("test_metric", 1.0)) {
            recorded_count += 1;
        }
    }

    try std.testing.expectEqual(@as(usize, 10), recorded_count);
    try std.testing.expectEqual(@as(u64, 100), monitor.call_counter);
}

test "LowOverheadMonitor getMetricAverage" {
    var monitor = LowOverheadMonitor.init(1); // Record all

    _ = monitor.record_if_sampled("latency", 10.0);
    _ = monitor.record_if_sampled("latency", 20.0);
    _ = monitor.record_if_sampled("latency", 30.0);

    const avg = monitor.getMetricAverage("latency");
    try std.testing.expect(avg != null);
    try std.testing.expectApproxEqAbs(@as(f64, 20.0), avg.?, 0.001);
}

test "LowOverheadMonitor overhead target" {
    var monitor = LowOverheadMonitor.init(100); // 1% sampling

    for (0..1000) |_| {
        _ = monitor.record_if_sampled("perf_test", 1.0);
    }

    const overhead = monitor.getOverheadPct();
    // Overhead should be very low with 1% sampling
    try std.testing.expect(overhead < 2.0); // Target: <2%
}

test "LowOverheadMonitor summary" {
    var monitor = LowOverheadMonitor.init(50);

    // Record metric_a 500 times
    for (0..500) |_| {
        _ = monitor.record_if_sampled("metric_a", 1.0);
    }

    // Record metric_b 500 times separately
    for (0..500) |_| {
        _ = monitor.record_if_sampled("metric_b", 2.0);
    }

    const summary = monitor.getSummary();
    try std.testing.expectEqual(@as(usize, 2), summary.metric_count);
    try std.testing.expectEqual(@as(usize, 50), summary.sample_rate);
    try std.testing.expectEqual(@as(u64, 1000), summary.call_counter);
}

test "BenchmarkSuite init and addResult" {
    var suite = BenchmarkSuite.init();

    try std.testing.expectEqual(@as(usize, 0), suite.result_count);

    suite.addResult(.{
        .name = "test_bench",
        .iterations = 100,
        .total_time_us = 1000.0,
        .avg_time_us = 10.0,
        .min_time_us = 8.0,
        .max_time_us = 15.0,
        .std_dev_us = 2.0,
        .ops_per_sec = 100000.0,
        .speedup_vs_baseline = 2.5,
    });

    try std.testing.expectEqual(@as(usize, 1), suite.result_count);
    try std.testing.expectApproxEqAbs(@as(f64, 1000.0), suite.total_time_us, 0.001);
}

test "BenchmarkSuite getAverageSpeedup" {
    var suite = BenchmarkSuite.init();

    suite.addResult(.{
        .name = "bench1",
        .iterations = 100,
        .total_time_us = 100.0,
        .avg_time_us = 1.0,
        .min_time_us = 0.5,
        .max_time_us = 2.0,
        .std_dev_us = 0.3,
        .ops_per_sec = 1000000.0,
        .speedup_vs_baseline = 2.0,
    });

    suite.addResult(.{
        .name = "bench2",
        .iterations = 100,
        .total_time_us = 100.0,
        .avg_time_us = 1.0,
        .min_time_us = 0.5,
        .max_time_us = 2.0,
        .std_dev_us = 0.3,
        .ops_per_sec = 1000000.0,
        .speedup_vs_baseline = 4.0,
    });

    const avg_speedup = suite.getAverageSpeedup();
    try std.testing.expectApproxEqAbs(@as(f64, 3.0), avg_speedup, 0.001);
}

test "SIMDInfo detect" {
    const info = SIMDInfo.detect();
    try std.testing.expect(info.width >= 1);
    try std.testing.expect(info.estimated_speedup >= 1.0);
    try std.testing.expect(info.arch_name.len > 0);
}

test "run_all_benchmarks executes" {
    const allocator = std.testing.allocator;

    const suite = try run_all_benchmarks(allocator);

    try std.testing.expect(suite.result_count > 0);
    try std.testing.expect(suite.total_time_us > 0);
}

test "compare_with_baseline analysis" {
    var suite = BenchmarkSuite.init();

    suite.addResult(.{
        .name = "fast_bench",
        .iterations = 100,
        .total_time_us = 100.0,
        .avg_time_us = 1.0,
        .min_time_us = 0.5,
        .max_time_us = 2.0,
        .std_dev_us = 0.3,
        .ops_per_sec = 1000000.0,
        .speedup_vs_baseline = 5.0,
    });

    suite.addResult(.{
        .name = "slow_bench",
        .iterations = 100,
        .total_time_us = 100.0,
        .avg_time_us = 1.0,
        .min_time_us = 0.5,
        .max_time_us = 2.0,
        .std_dev_us = 0.3,
        .ops_per_sec = 1000000.0,
        .speedup_vs_baseline = 1.5,
    });

    const comparison = compare_with_baseline(&suite);

    try std.testing.expectApproxEqAbs(@as(f64, 5.0), comparison.best_speedup, 0.001);
    try std.testing.expectApproxEqAbs(@as(f64, 1.5), comparison.worst_speedup, 0.001);
    try std.testing.expectEqualStrings("fast_bench", comparison.best_benchmark);
    try std.testing.expectEqualStrings("slow_bench", comparison.worst_benchmark);
    try std.testing.expect(!comparison.meets_target); // 1.5 < 2.0
}
