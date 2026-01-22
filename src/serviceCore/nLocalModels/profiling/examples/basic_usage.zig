// Basic Profiling Usage Example
// Demonstrates how to use the profiling tools in your application

const std = @import("std");
const profiling = @import("../profiling_api.zig");
const cpu_profiler = @import("../cpu_profiler.zig");
const memory_profiler = @import("../memory_profiler.zig");
const gpu_monitor = @import("../gpu_monitor.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== Performance Profiling Example ===\n\n", .{});

    // Example 1: Profile a single function
    try example1_profileFunction(allocator);

    // Example 2: Profile memory usage
    try example2_profileMemory(allocator);

    // Example 3: Monitor GPU
    try example3_monitorGpu(allocator);

    // Example 4: Full profiling session
    try example4_fullSession(allocator);
}

// Example 1: Profile CPU usage of a function
fn example1_profileFunction(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 1: CPU Profiling\n", .{});
    std.debug.print("-------------------------\n", .{});

    const config = cpu_profiler.CpuProfileConfig{
        .sample_rate_hz = 1000,
        .max_stack_depth = 32,
    };

    var profiler = try cpu_profiler.CpuProfiler.init(allocator, config);
    defer profiler.deinit();

    // Start profiling
    try profiler.start();

    // Run workload
    expensiveComputation();

    // Stop profiling
    profiler.stop();

    // Get results
    const profile = profiler.getProfile();
    std.debug.print("Total samples: {d}\n", .{profile.total_samples});
    std.debug.print("Duration: {d:.2}s\n\n", .{@as(f64, @floatFromInt(profile.duration_ns)) / 1_000_000_000.0});

    // Get top functions
    const top = try profile.getTopFunctions(5);
    defer allocator.free(top);

    std.debug.print("Top 5 functions:\n", .{});
    for (top, 0..) |func, i| {
        std.debug.print("  {d}. {s} - {d:.1}% ({d} samples)\n", .{
            i + 1,
            func.name,
            func.percent,
            func.samples,
        });
    }
    std.debug.print("\n", .{});
}

// Example 2: Profile memory allocations
fn example2_profileMemory(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 2: Memory Profiling\n", .{});
    std.debug.print("---------------------------\n", .{});

    const config = memory_profiler.MemoryProfileConfig{
        .track_allocations = true,
        .sample_rate = 1, // Track all allocations for demo
        .leak_detection = true,
        .capture_stack_traces = false, // Disable for simpler output
    };

    var profiler = try memory_profiler.MemoryProfiler.init(allocator, config);
    defer profiler.deinit();

    profiler.start();

    // Simulate some allocations
    try profiler.trackAllocation(0x1000, 1024 * 1024); // 1 MB
    try profiler.trackAllocation(0x2000, 2048 * 1024); // 2 MB
    try profiler.trackAllocation(0x3000, 512 * 1024); // 512 KB
    profiler.trackFree(0x1000); // Free first allocation

    profiler.stop();

    // Get statistics
    const profile = profiler.getProfile();
    std.debug.print("Total allocated: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.stats.total_allocated)) / 1_048_576.0,
    });
    std.debug.print("Total freed: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.stats.total_freed)) / 1_048_576.0,
    });
    std.debug.print("Peak usage: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.stats.peak_usage)) / 1_048_576.0,
    });
    std.debug.print("Current usage: {d:.2} MB\n", .{
        @as(f64, @floatFromInt(profile.stats.current_usage)) / 1_048_576.0,
    });
    std.debug.print("Allocation count: {d}\n\n", .{profile.stats.allocation_count});
}

// Example 3: Monitor GPU metrics
fn example3_monitorGpu(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 3: GPU Monitoring\n", .{});
    std.debug.print("-------------------------\n", .{});

    const config = gpu_monitor.GpuMonitorConfig{
        .enabled = true,
        .poll_interval_ms = 100,
    };

    var monitor = gpu_monitor.GpuMonitor.init(allocator, config) catch |err| {
        std.debug.print("GPU monitoring not available: {}\n\n", .{err});
        return;
    };
    defer monitor.deinit();

    try monitor.start();

    // Let it collect some samples
    std.time.sleep(500 * std.time.ns_per_ms);

    monitor.stop();

    const profile = monitor.getProfile();
    std.debug.print("GPU device count: {d}\n", .{profile.device_count});
    std.debug.print("GPU type: {s}\n", .{@tagName(profile.gpu_type)});

    if (profile.device_count > 0) {
        const avg_util = profile.getAverageUtilization(0);
        const peak_mem = profile.getPeakMemoryUsage(0);
        std.debug.print("Device 0 average utilization: {d:.1}%\n", .{avg_util});
        std.debug.print("Device 0 peak memory: {d} MB\n", .{peak_mem});
    }
    std.debug.print("\n", .{});
}

// Example 4: Full profiling session with all components
fn example4_fullSession(allocator: std.mem.Allocator) !void {
    std.debug.print("Example 4: Full Profiling Session\n", .{});
    std.debug.print("---------------------------------\n", .{});

    var manager = profiling.ProfilingManager.init(allocator);
    defer manager.deinit();

    // Start profiling session
    const session_id = try manager.startSession("example_workload", true, true, false);
    defer allocator.free(session_id);

    std.debug.print("Started profiling session: {s}\n", .{session_id});

    // Run workload
    expensiveComputation();

    // Stop profiling
    try manager.stopSession(session_id);

    // Get bottleneck report
    var report = try manager.getBottleneckReport(session_id);
    defer report.deinit();

    std.debug.print("Bottlenecks detected: {d}\n", .{report.bottlenecks.items.len});
    for (report.bottlenecks.items, 0..) |bottleneck, i| {
        std.debug.print("  {d}. [{s}] {s}\n", .{
            i + 1,
            @tagName(bottleneck.severity),
            bottleneck.description,
        });
        std.debug.print("     Recommendation: {s}\n", .{bottleneck.recommendation});
    }
    std.debug.print("\n", .{});

    // Export to JSON
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try manager.toJson(session_id, buffer.writer());
    std.debug.print("Profile JSON ({d} bytes):\n{s}\n\n", .{ buffer.items.len, buffer.items });
}

// Simulated expensive computation for demo
fn expensiveComputation() void {
    var sum: u64 = 0;
    var i: usize = 0;
    while (i < 10_000_000) : (i += 1) {
        sum +%= i * i;
    }
    std.debug.print("Workload completed (sum: {d})\n", .{sum});
}
