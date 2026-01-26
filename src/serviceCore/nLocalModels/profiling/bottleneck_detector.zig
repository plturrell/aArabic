// Bottleneck Detector - Automatic Performance Hotspot Identification
// Analyzes profiles to identify bottlenecks and generate recommendations

const std = @import("std");
const Allocator = std.mem.Allocator;
const cpu_profiler = @import("cpu_profiler.zig");
const memory_profiler = @import("memory_profiler.zig");
const gpu_monitor = @import("gpu_monitor.zig");

pub const BottleneckConfig = struct {
    threshold_ms: f32 = 10.0,
    auto_detect: bool = true,
    alert_on_regression: bool = true,
    min_samples: u32 = 100,
};

pub const BottleneckSeverity = enum {
    low,
    medium,
    high,
    critical,

    pub fn fromPercent(percent: f32) BottleneckSeverity {
        if (percent > 20.0) return .critical;
        if (percent > 10.0) return .high;
        if (percent > 5.0) return .medium;
        return .low;
    }
};

pub const BottleneckType = enum {
    cpu_hotspot,
    memory_leak,
    memory_hotspot,
    gpu_underutilization,
    gpu_memory_pressure,
    synchronization_overhead,
    io_wait,
};

pub const Bottleneck = struct {
    type: BottleneckType,
    severity: BottleneckSeverity,
    location: []const u8,
    description: []const u8,
    metric_value: f32,
    recommendation: []const u8,
    allocator: Allocator,

    pub fn deinit(self: *Bottleneck) void {
        self.allocator.free(self.location);
        self.allocator.free(self.description);
        self.allocator.free(self.recommendation);
    }
};

pub const BottleneckReport = struct {
    bottlenecks: std.ArrayList(Bottleneck),
    total_cpu_time_ms: f32,
    total_memory_mb: f32,
    avg_gpu_utilization: f32,
    timestamp_ns: i64,
    allocator: Allocator,

    pub fn init(allocator: Allocator) BottleneckReport {
        return .{
            .bottlenecks = std.ArrayList(Bottleneck){},
            .total_cpu_time_ms = 0.0,
            .total_memory_mb = 0.0,
            .avg_gpu_utilization = 0.0,
            .timestamp_ns = std.time.nanoTimestamp(),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BottleneckReport) void {
        for (self.bottlenecks.items) |*bottleneck| {
            bottleneck.deinit();
        }
        self.bottlenecks.deinit();
    }

    pub fn addBottleneck(self: *BottleneckReport, bottleneck: Bottleneck) !void {
        try self.bottlenecks.append(bottleneck);
    }

    pub fn getCriticalBottlenecks(self: *const BottleneckReport) []const Bottleneck {
        var count: usize = 0;
        for (self.bottlenecks.items) |bottleneck| {
            if (bottleneck.severity == .critical) count += 1;
        }

        var result = self.allocator.alloc(Bottleneck, count) catch return &[_]Bottleneck{};
        var i: usize = 0;
        for (self.bottlenecks.items) |bottleneck| {
            if (bottleneck.severity == .critical) {
                result[i] = bottleneck;
                i += 1;
            }
        }

        return result;
    }

    pub fn toJson(self: *const BottleneckReport, writer: anytype) !void {
        try writer.writeAll("{");
        try writer.print("\"timestamp\":{d},", .{self.timestamp_ns});
        try writer.print("\"total_cpu_time_ms\":{d:.2},", .{self.total_cpu_time_ms});
        try writer.print("\"total_memory_mb\":{d:.2},", .{self.total_memory_mb});
        try writer.print("\"avg_gpu_utilization\":{d:.2},", .{self.avg_gpu_utilization});
        try writer.print("\"bottleneck_count\":{d},", .{self.bottlenecks.items.len});

        try writer.writeAll("\"bottlenecks\":[");
        for (self.bottlenecks.items, 0..) |bottleneck, i| {
            if (i > 0) try writer.writeAll(",");
            try writer.writeAll("{");
            try writer.print("\"type\":\"{s}\",", .{@tagName(bottleneck.type)});
            try writer.print("\"severity\":\"{s}\",", .{@tagName(bottleneck.severity)});
            try writer.print("\"location\":\"{s}\",", .{bottleneck.location});
            try writer.print("\"description\":\"{s}\",", .{bottleneck.description});
            try writer.print("\"metric_value\":{d:.2},", .{bottleneck.metric_value});
            try writer.print("\"recommendation\":\"{s}\"", .{bottleneck.recommendation});
            try writer.writeAll("}");
        }
        try writer.writeAll("]");
        try writer.writeAll("}");
    }
};

pub const BottleneckDetector = struct {
    config: BottleneckConfig,
    allocator: Allocator,

    pub fn init(allocator: Allocator, config: BottleneckConfig) !*BottleneckDetector {
        const detector = try allocator.create(BottleneckDetector);
        detector.* = .{
            .config = config,
            .allocator = allocator,
        };
        return detector;
    }

    pub fn deinit(self: *BottleneckDetector) void {
        self.allocator.destroy(self);
    }

    pub fn analyze(
        self: *BottleneckDetector,
        cpu_profile: ?*const cpu_profiler.CpuProfile,
        mem_profile: ?*const memory_profiler.MemoryProfile,
        gpu_profile: ?*const gpu_monitor.GpuProfile,
    ) !BottleneckReport {
        var report = BottleneckReport.init(self.allocator);
        errdefer report.deinit();

        // Analyze CPU bottlenecks
        if (cpu_profile) |cpu| {
            try self.analyzeCpuProfile(&report, cpu);
            report.total_cpu_time_ms = @as(f32, @floatFromInt(cpu.duration_ns)) / 1_000_000.0;
        }

        // Analyze memory bottlenecks
        if (mem_profile) |mem| {
            try self.analyzeMemoryProfile(&report, mem);
            report.total_memory_mb = @as(f32, @floatFromInt(mem.stats.peak_usage)) / 1_048_576.0;
        }

        // Analyze GPU bottlenecks
        if (gpu_profile) |gpu| {
            try self.analyzeGpuProfile(&report, gpu);
            if (gpu.device_count > 0) {
                report.avg_gpu_utilization = gpu.getAverageUtilization(0);
            }
        }

        // Sort by severity
        std.sort.pdq(Bottleneck, report.bottlenecks.items, {}, struct {
            fn lessThan(_: void, a: Bottleneck, b: Bottleneck) bool {
                return @intFromEnum(a.severity) > @intFromEnum(b.severity);
            }
        }.lessThan);

        return report;
    }

    fn analyzeCpuProfile(self: *BottleneckDetector, report: *BottleneckReport, profile: *const cpu_profiler.CpuProfile) !void {
        if (profile.total_samples < self.config.min_samples) return;

        const top_functions = try profile.getTopFunctions(20);
        defer self.allocator.free(top_functions);

        for (top_functions) |func| {
            // Check if function is a hotspot
            if (func.percent > 5.0) {
                const severity = BottleneckSeverity.fromPercent(func.percent);

                const location = try std.fmt.allocPrint(
                    self.allocator,
                    "{s}:{d}",
                    .{ func.file, func.line },
                );

                const description = try std.fmt.allocPrint(
                    self.allocator,
                    "Function '{s}' consumes {d:.1}% of CPU time ({d} samples)",
                    .{ func.name, func.percent, func.samples },
                );

                const recommendation = try self.generateCpuRecommendation(func.name, func.percent);

                try report.addBottleneck(.{
                    .type = .cpu_hotspot,
                    .severity = severity,
                    .location = location,
                    .description = description,
                    .metric_value = func.percent,
                    .recommendation = recommendation,
                    .allocator = self.allocator,
                });
            }
        }
    }

    fn analyzeMemoryProfile(self: *BottleneckDetector, report: *BottleneckReport, profile: *const memory_profiler.MemoryProfile) !void {
        // Check for memory leaks
        if (profile.stats.leaked_bytes > 1_048_576) { // > 1 MB
            const leaked_mb = @as(f32, @floatFromInt(profile.stats.leaked_bytes)) / 1_048_576.0;

            try report.addBottleneck(.{
                .type = .memory_leak,
                .severity = if (leaked_mb > 100) .critical else if (leaked_mb > 10) .high else .medium,
                .location = try self.allocator.dupe(u8, "multiple"),
                .description = try std.fmt.allocPrint(
                    self.allocator,
                    "Detected {d:.1} MB of leaked memory across {d} allocations",
                    .{ leaked_mb, profile.stats.leaked_count },
                ),
                .metric_value = leaked_mb,
                .recommendation = try self.allocator.dupe(u8, "Review allocation patterns and ensure proper cleanup. Use valgrind or similar tools for detailed leak analysis."),
                .allocator = self.allocator,
            });
        }

        // Check for memory hotspots
        const hotspots = try profile.getHotspots(10);
        defer self.allocator.free(hotspots);

        for (hotspots) |hotspot| {
            const total_mb = @as(f32, @floatFromInt(hotspot.total_bytes)) / 1_048_576.0;
            if (total_mb > 10.0) {
                try report.addBottleneck(.{
                    .type = .memory_hotspot,
                    .severity = if (total_mb > 1000) .critical else if (total_mb > 100) .high else .medium,
                    .location = try self.allocator.dupe(u8, "see_stack_trace"),
                    .description = try std.fmt.allocPrint(
                        self.allocator,
                        "Memory hotspot: {d:.1} MB allocated across {d} calls",
                        .{ total_mb, hotspot.count },
                    ),
                    .metric_value = total_mb,
                    .recommendation = try self.allocator.dupe(u8, "Consider object pooling, arena allocators, or reducing allocation frequency. Review memory lifetime and reuse opportunities."),
                    .allocator = self.allocator,
                });
            }
        }
    }

    fn analyzeGpuProfile(self: *BottleneckDetector, report: *BottleneckReport, profile: *const gpu_monitor.GpuProfile) !void {
        for (0..profile.device_count) |device_id| {
            const avg_util = profile.getAverageUtilization(@intCast(device_id));
            const peak_mem = profile.getPeakMemoryUsage(@intCast(device_id));

            // Check for GPU underutilization
            if (avg_util < 50.0 and profile.metrics_history.items.len > 10) {
                try report.addBottleneck(.{
                    .type = .gpu_underutilization,
                    .severity = if (avg_util < 20.0) .high else .medium,
                    .location = try std.fmt.allocPrint(self.allocator, "GPU {d}", .{device_id}),
                    .description = try std.fmt.allocPrint(
                        self.allocator,
                        "GPU {d} average utilization is only {d:.1}%",
                        .{ device_id, avg_util },
                    ),
                    .metric_value = avg_util,
                    .recommendation = try self.allocator.dupe(u8, "Increase batch size, optimize kernel launch parameters, reduce CPU-GPU synchronization, or consider using CPU for this workload."),
                    .allocator = self.allocator,
                });
            }

            // Check GPU memory pressure
            if (profile.metrics_history.items.len > 0) {
                const latest = profile.metrics_history.items[profile.metrics_history.items.len - 1];
                if (latest.device_id == device_id) {
                    const mem_percent = (@as(f32, @floatFromInt(latest.memory_used_mb)) / @as(f32, @floatFromInt(latest.memory_total_mb))) * 100.0;

                    if (mem_percent > 85.0) {
                        try report.addBottleneck(.{
                            .type = .gpu_memory_pressure,
                            .severity = if (mem_percent > 95.0) .critical else .high,
                            .location = try std.fmt.allocPrint(self.allocator, "GPU {d}", .{device_id}),
                            .description = try std.fmt.allocPrint(
                                self.allocator,
                                "GPU {d} memory usage is {d:.1}% ({d} MB / {d} MB)",
                                .{ device_id, mem_percent, latest.memory_used_mb, latest.memory_total_mb },
                            ),
                            .metric_value = mem_percent,
                            .recommendation = try self.allocator.dupe(u8, "Reduce batch size, enable gradient checkpointing, use mixed precision (FP16), or consider model quantization to reduce memory footprint."),
                            .allocator = self.allocator,
                        });
                    }
                }
            }
        }
    }

    fn generateCpuRecommendation(self: *BottleneckDetector, function_name: []const u8, percent: f32) ![]const u8 {
        _ = percent;

        // Pattern matching for common bottlenecks
        if (std.mem.indexOf(u8, function_name, "matmul") != null or
            std.mem.indexOf(u8, function_name, "gemm") != null)
        {
            return try self.allocator.dupe(u8, "Consider using optimized BLAS libraries (cuBLAS, MKL), SIMD instructions, or GPU acceleration for matrix operations.");
        }

        if (std.mem.indexOf(u8, function_name, "softmax") != null or
            std.mem.indexOf(u8, function_name, "norm") != null)
        {
            return try self.allocator.dupe(u8, "Optimize numerical operations with SIMD, fused kernels, or GPU acceleration. Consider approximations if precision allows.");
        }

        if (std.mem.indexOf(u8, function_name, "alloc") != null or
            std.mem.indexOf(u8, function_name, "malloc") != null)
        {
            return try self.allocator.dupe(u8, "High allocation overhead detected. Use arena allocators, object pooling, or preallocated buffers to reduce allocation frequency.");
        }

        if (std.mem.indexOf(u8, function_name, "lock") != null or
            std.mem.indexOf(u8, function_name, "mutex") != null)
        {
            return try self.allocator.dupe(u8, "Synchronization overhead detected. Consider lock-free data structures, finer-grained locking, or thread-local storage.");
        }

        if (std.mem.indexOf(u8, function_name, "memcpy") != null or
            std.mem.indexOf(u8, function_name, "copy") != null)
        {
            return try self.allocator.dupe(u8, "High memory copy overhead. Reduce unnecessary copies, use move semantics, or optimize data layout for better cache utilization.");
        }

        // Generic recommendation
        return try self.allocator.dupe(u8, "Profile this function in detail to identify optimization opportunities. Consider algorithmic improvements, caching, or parallelization.");
    }
};

// Testing
test "BottleneckDetector basic" {
    const allocator = std.testing.allocator;

    const config = BottleneckConfig{
        .threshold_ms = 10.0,
        .auto_detect = true,
    };

    var detector = try BottleneckDetector.init(allocator, config);
    defer detector.deinit();

    // Test with empty profiles
    var report = try detector.analyze(null, null, null);
    defer report.deinit();

    try std.testing.expect(report.bottlenecks.items.len == 0);
}
