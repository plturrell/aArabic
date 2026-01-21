//! Memory Profiler for nExtract
//!
//! This module provides:
//! - Allocation profiling and tracking
//! - Performance metrics collection
//! - Hot path identification
//! - Memory usage visualization
//! - Integration with benchmarking tools
//!
//! Author: nExtract Team
//! Date: January 17, 2026

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

// ============================================================================
// Memory Profiler
// ============================================================================

/// Memory profiler for tracking allocations and performance
pub const MemoryProfiler = struct {
    backing_allocator: Allocator,
    allocations: if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) 
        std.AutoHashMap(usize, AllocationProfile) 
        else void,
    call_sites: if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe)
        std.AutoHashMap(usize, CallSiteStats)
        else void,
    enabled: bool,
    start_time: i64,
    
    const AllocationProfile = struct {
        size: usize,
        alignment: u8,
        return_address: usize,
        timestamp: i64,
        call_site_id: usize,
    };
    
    const CallSiteStats = struct {
        return_address: usize,
        allocation_count: usize,
        total_bytes: usize,
        peak_bytes: usize,
        current_bytes: usize,
        avg_size: f64,
    };
    
    pub fn init(backing: Allocator) MemoryProfiler {
        if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
            return MemoryProfiler{
                .backing_allocator = backing,
                .allocations = std.AutoHashMap(usize, AllocationProfile).init(backing),
                .call_sites = std.AutoHashMap(usize, CallSiteStats).init(backing),
                .enabled = true,
                .start_time = std.time.milliTimestamp(),
            };
        } else {
            return MemoryProfiler{
                .backing_allocator = backing,
                .allocations = {},
                .call_sites = {},
                .enabled = false,
                .start_time = 0,
            };
        }
    }
    
    pub fn allocator(self: *MemoryProfiler) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr) orelse return null;
        
        if (self.enabled and (builtin.mode == .Debug or builtin.mode == .ReleaseSafe)) {
            const addr = @intFromPtr(result);
            const timestamp = std.time.milliTimestamp();
            
            // Track allocation
            self.allocations.put(addr, .{
                .size = len,
                .alignment = ptr_align,
                .return_address = ret_addr,
                .timestamp = timestamp,
                .call_site_id = ret_addr,
            }) catch {};
            
            // Update call site stats
            if (self.call_sites.getPtr(ret_addr)) |stats| {
                stats.allocation_count += 1;
                stats.total_bytes += len;
                stats.current_bytes += len;
                if (stats.current_bytes > stats.peak_bytes) {
                    stats.peak_bytes = stats.current_bytes;
                }
                stats.avg_size = @as(f64, @floatFromInt(stats.total_bytes)) / 
                                 @as(f64, @floatFromInt(stats.allocation_count));
            } else {
                self.call_sites.put(ret_addr, .{
                    .return_address = ret_addr,
                    .allocation_count = 1,
                    .total_bytes = len,
                    .peak_bytes = len,
                    .current_bytes = len,
                    .avg_size = @floatFromInt(len),
                }) catch {};
            }
        }
        
        return result;
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        
        if (self.enabled and (builtin.mode == .Debug or builtin.mode == .ReleaseSafe)) {
            const addr = @intFromPtr(buf.ptr);
            
            if (self.allocations.get(addr)) |profile| {
                const old_size = profile.size;
                
                if (self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr)) {
                    // Update allocation profile
                    self.allocations.put(addr, .{
                        .size = new_len,
                        .alignment = buf_align,
                        .return_address = profile.return_address,
                        .timestamp = profile.timestamp,
                        .call_site_id = profile.call_site_id,
                    }) catch {};
                    
                    // Update call site stats
                    if (self.call_sites.getPtr(profile.call_site_id)) |stats| {
                        stats.current_bytes = stats.current_bytes - old_size + new_len;
                        if (stats.current_bytes > stats.peak_bytes) {
                            stats.peak_bytes = stats.current_bytes;
                        }
                    }
                    
                    return true;
                }
                return false;
            }
        }
        
        return self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);
    }
    
    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *MemoryProfiler = @ptrCast(@alignCast(ctx));
        
        if (self.enabled and (builtin.mode == .Debug or builtin.mode == .ReleaseSafe)) {
            const addr = @intFromPtr(buf.ptr);
            
            if (self.allocations.fetchRemove(addr)) |entry| {
                const profile = entry.value;
                
                // Update call site stats
                if (self.call_sites.getPtr(profile.call_site_id)) |stats| {
                    stats.current_bytes -= profile.size;
                }
            }
        }
        
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }
    
    pub fn deinit(self: *MemoryProfiler) void {
        if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
            self.allocations.deinit();
            self.call_sites.deinit();
        }
    }
    
    /// Get profiling report
    pub fn getReport(self: *const MemoryProfiler, allocator: Allocator) !ProfileReport {
        if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
            var hot_spots = std.ArrayList(HotSpot).init(allocator);
            
            var iter = self.call_sites.iterator();
            while (iter.next()) |entry| {
                try hot_spots.append(.{
                    .return_address = entry.value_ptr.return_address,
                    .allocation_count = entry.value_ptr.allocation_count,
                    .total_bytes = entry.value_ptr.total_bytes,
                    .peak_bytes = entry.value_ptr.peak_bytes,
                    .avg_size = entry.value_ptr.avg_size,
                });
            }
            
            // Sort by total bytes (descending)
            std.mem.sort(HotSpot, hot_spots.items, {}, hotSpotCompare);
            
            return ProfileReport{
                .hot_spots = try hot_spots.toOwnedSlice(),
                .current_allocations = self.allocations.count(),
                .call_sites = self.call_sites.count(),
                .elapsed_time_ms = std.time.milliTimestamp() - self.start_time,
            };
        } else {
            return ProfileReport{
                .hot_spots = &[_]HotSpot{},
                .current_allocations = 0,
                .call_sites = 0,
                .elapsed_time_ms = 0,
            };
        }
    }
    
    fn hotSpotCompare(_: void, a: HotSpot, b: HotSpot) bool {
        return a.total_bytes > b.total_bytes;
    }
    
    /// Print profiling report
    pub fn printReport(self: *const MemoryProfiler, allocator: Allocator) !void {
        const report = try self.getReport(allocator);
        defer allocator.free(report.hot_spots);
        
        std.debug.print("\n=== MEMORY PROFILING REPORT ===\n", .{});
        std.debug.print("Elapsed time: {}ms\n", .{report.elapsed_time_ms});
        std.debug.print("Active allocations: {}\n", .{report.current_allocations});
        std.debug.print("Call sites tracked: {}\n\n", .{report.call_sites});
        
        std.debug.print("Top allocation hot spots:\n", .{});
        const max_spots = @min(report.hot_spots.len, 20);
        for (report.hot_spots[0..max_spots], 0..) |spot, i| {
            std.debug.print("  #{}: addr=0x{x}, count={}, total={} bytes, peak={} bytes, avg={d:.1} bytes\n", .{
                i + 1,
                spot.return_address,
                spot.allocation_count,
                spot.total_bytes,
                spot.peak_bytes,
                spot.avg_size,
            });
        }
        
        if (report.hot_spots.len > max_spots) {
            std.debug.print("  ... and {} more\n", .{report.hot_spots.len - max_spots});
        }
    }
    
    /// Enable/disable profiling
    pub fn setEnabled(self: *MemoryProfiler, enabled: bool) void {
        self.enabled = enabled;
    }
    
    /// Reset profiling data
    pub fn reset(self: *MemoryProfiler) void {
        if (builtin.mode == .Debug or builtin.mode == .ReleaseSafe) {
            self.allocations.clearRetainingCapacity();
            self.call_sites.clearRetainingCapacity();
            self.start_time = std.time.milliTimestamp();
        }
    }
};

pub const HotSpot = struct {
    return_address: usize,
    allocation_count: usize,
    total_bytes: usize,
    peak_bytes: usize,
    avg_size: f64,
};

pub const ProfileReport = struct {
    hot_spots: []const HotSpot,
    current_allocations: usize,
    call_sites: usize,
    elapsed_time_ms: i64,
};

// ============================================================================
// Performance Metrics
// ============================================================================

/// Performance metrics collector
pub const PerformanceMetrics = struct {
    operations: std.StringHashMap(OperationStats),
    allocator: Allocator,
    
    const OperationStats = struct {
        count: usize,
        total_time_ns: u64,
        min_time_ns: u64,
        max_time_ns: u64,
        avg_time_ns: f64,
    };
    
    pub fn init(allocator: Allocator) PerformanceMetrics {
        return PerformanceMetrics{
            .operations = std.StringHashMap(OperationStats).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *PerformanceMetrics) void {
        self.operations.deinit();
    }
    
    /// Record operation timing
    pub fn recordOperation(self: *PerformanceMetrics, name: []const u8, duration_ns: u64) !void {
        if (self.operations.getPtr(name)) |stats| {
            stats.count += 1;
            stats.total_time_ns += duration_ns;
            if (duration_ns < stats.min_time_ns) {
                stats.min_time_ns = duration_ns;
            }
            if (duration_ns > stats.max_time_ns) {
                stats.max_time_ns = duration_ns;
            }
            stats.avg_time_ns = @as(f64, @floatFromInt(stats.total_time_ns)) / 
                                @as(f64, @floatFromInt(stats.count));
        } else {
            const name_copy = try self.allocator.dupe(u8, name);
            try self.operations.put(name_copy, .{
                .count = 1,
                .total_time_ns = duration_ns,
                .min_time_ns = duration_ns,
                .max_time_ns = duration_ns,
                .avg_time_ns = @floatFromInt(duration_ns),
            });
        }
    }
    
    /// Get statistics for an operation
    pub fn getStats(self: *const PerformanceMetrics, name: []const u8) ?OperationStats {
        return self.operations.get(name);
    }
    
    /// Print performance report
    pub fn printReport(self: *const PerformanceMetrics) void {
        std.debug.print("\n=== PERFORMANCE METRICS ===\n\n", .{});
        
        var iter = self.operations.iterator();
        while (iter.next()) |entry| {
            const stats = entry.value_ptr.*;
            std.debug.print("Operation: {s}\n", .{entry.key_ptr.*});
            std.debug.print("  Count: {}\n", .{stats.count});
            std.debug.print("  Total: {d:.2}ms\n", .{@as(f64, @floatFromInt(stats.total_time_ns)) / 1_000_000.0});
            std.debug.print("  Min: {d:.2}µs\n", .{@as(f64, @floatFromInt(stats.min_time_ns)) / 1_000.0});
            std.debug.print("  Max: {d:.2}µs\n", .{@as(f64, @floatFromInt(stats.max_time_ns)) / 1_000.0});
            std.debug.print("  Avg: {d:.2}µs\n\n", .{stats.avg_time_ns / 1_000.0});
        }
    }
    
    /// Reset all metrics
    pub fn reset(self: *PerformanceMetrics) void {
        var iter = self.operations.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.operations.clearRetainingCapacity();
    }
};

/// Timer for measuring operation duration
pub const Timer = struct {
    start_time: i128,
    
    pub fn start() Timer {
        return Timer{
            .start_time = std.time.nanoTimestamp(),
        };
    }
    
    pub fn elapsed(self: *const Timer) u64 {
        const now = std.time.nanoTimestamp();
        const duration = now - self.start_time;
        return @intCast(duration);
    }
    
    pub fn elapsedMs(self: *const Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1_000_000.0;
    }
    
    pub fn elapsedUs(self: *const Timer) f64 {
        return @as(f64, @floatFromInt(self.elapsed())) / 1_000.0;
    }
};

// ============================================================================
// Benchmarking Utilities
// ============================================================================

/// Run a benchmark
pub fn benchmark(
    allocator: Allocator,
    name: []const u8,
    comptime func: anytype,
    iterations: usize,
) !BenchmarkResult {
    var metrics = PerformanceMetrics.init(allocator);
    defer metrics.deinit();
    
    var min_time: u64 = std.math.maxInt(u64);
    var max_time: u64 = 0;
    var total_time: u64 = 0;
    
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        const timer = Timer.start();
        try func();
        const elapsed = timer.elapsed();
        
        if (elapsed < min_time) min_time = elapsed;
        if (elapsed > max_time) max_time = elapsed;
        total_time += elapsed;
    }
    
    const avg_time = @as(f64, @floatFromInt(total_time)) / @as(f64, @floatFromInt(iterations));
    
    return BenchmarkResult{
        .name = name,
        .iterations = iterations,
        .min_ns = min_time,
        .max_ns = max_time,
        .avg_ns = avg_time,
        .total_ns = total_time,
    };
}

pub const BenchmarkResult = struct {
    name: []const u8,
    iterations: usize,
    min_ns: u64,
    max_ns: u64,
    avg_ns: f64,
    total_ns: u64,
    
    pub fn print(self: *const BenchmarkResult) void {
        std.debug.print("\n=== BENCHMARK: {s} ===\n", .{self.name});
        std.debug.print("Iterations: {}\n", .{self.iterations});
        std.debug.print("Total time: {d:.2}ms\n", .{@as(f64, @floatFromInt(self.total_ns)) / 1_000_000.0});
        std.debug.print("Min: {d:.2}µs\n", .{@as(f64, @floatFromInt(self.min_ns)) / 1_000.0});
        std.debug.print("Max: {d:.2}µs\n", .{@as(f64, @floatFromInt(self.max_ns)) / 1_000.0});
        std.debug.print("Avg: {d:.2}µs\n", .{self.avg_ns / 1_000.0});
        std.debug.print("Ops/sec: {d:.0}\n", .{1_000_000_000.0 / self.avg_ns});
    }
};

// ============================================================================
// Tests
// ============================================================================

test "MemoryProfiler: basic tracking" {
    var profiler = MemoryProfiler.init(std.testing.allocator);
    defer profiler.deinit();
    
    const alloc = profiler.allocator();
    
    const bytes = try alloc.alloc(u8, 100);
    defer alloc.free(bytes);
    
    const report = try profiler.getReport(std.testing.allocator);
    defer std.testing.allocator.free(report.hot_spots);
    
    try std.testing.expectEqual(@as(usize, 1), report.current_allocations);
}

test "PerformanceMetrics: operation tracking" {
    var metrics = PerformanceMetrics.init(std.testing.allocator);
    defer metrics.deinit();
    
    try metrics.recordOperation("test_op", 1000);
    try metrics.recordOperation("test_op", 2000);
    try metrics.recordOperation("test_op", 3000);
    
    const stats = metrics.getStats("test_op").?;
    try std.testing.expectEqual(@as(usize, 3), stats.count);
    try std.testing.expectEqual(@as(u64, 6000), stats.total_time_ns);
    try std.testing.expectEqual(@as(u64, 1000), stats.min_time_ns);
    try std.testing.expectEqual(@as(u64, 3000), stats.max_time_ns);
}

test "Timer: elapsed time" {
    const timer = Timer.start();
    std.time.sleep(1 * std.time.ns_per_ms); // Sleep 1ms
    const elapsed = timer.elapsed();
    
    // Should be at least 1ms (in nanoseconds)
    try std.testing.expect(elapsed >= 1_000_000);
}

test "benchmark: basic usage" {
    const testFunc = struct {
        fn run() !void {
            var sum: u64 = 0;
            var i: usize = 0;
            while (i < 1000) : (i += 1) {
                sum += i;
            }
        }
    }.run;
    
    const result = try benchmark(std.testing.allocator, "sum_1000", testFunc, 100);
    try std.testing.expectEqual(@as(usize, 100), result.iterations);
    try std.testing.expect(result.avg_ns > 0);
}
