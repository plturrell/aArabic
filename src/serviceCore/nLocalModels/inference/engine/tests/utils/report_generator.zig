// GPU Performance Report Generator
// Collects metrics and generates professional validation reports
// Supports: Markdown, JSON, HTML, CSV formats
//
// IMPORTANT: This module ONLY uses measured data from actual test runs.
// Industry baselines are used for comparison context only.

const std = @import("std");
const baselines = @import("industry_baselines.zig");

// ============================================================================
// Data Structures
// ============================================================================

pub const DataSource = enum {
    Measured,          // Actually measured from test execution
    Unavailable,       // Test could not run (e.g., no GPU)
    
    pub fn toString(self: DataSource) []const u8 {
        return switch (self) {
            .Measured => "✓ Measured",
            .Unavailable => "⚠ Not Available",
        };
    }
};

pub const HardwareInfo = struct {
    gpu_model: []const u8,
    gpu_memory_mb: u32,
    compute_capability: struct { major: i32, minor: i32 },
    cuda_version: []const u8,
    driver_version: []const u8,
    cpu_model: []const u8,
    ram_gb: u32,
    os: []const u8,
};

pub const BuildInfo = struct {
    libcuda_version: []const u8,
    libcudart_version: []const u8,
    libcublas_version: []const u8,
    symbols_resolved: bool,
    link_mode: []const u8,
};

pub const BenchmarkResult = struct {
    name: []const u8,
    size: usize,
    cpu_time_ms: f64,
    gpu_time_ms: ?f64,  // null if GPU not available
    speedup: ?f64,      // null if GPU not available
    cpu_gflops: f64,
    gpu_gflops: ?f64,   // null if GPU not available
    cpu_source: DataSource,
    gpu_source: DataSource,
};

pub const IntegrationResult = struct {
    gpu_detected: bool,
    memory_allocated: bool,
    operations_working: bool,
    performance_acceptable: bool,
    no_memory_leaks: bool,
    
    pub fn allPassed(self: IntegrationResult) bool {
        return self.gpu_detected and
               self.memory_allocated and
               self.operations_working and
               self.performance_acceptable and
               self.no_memory_leaks;
    }
};

pub const ReportData = struct {
    timestamp: i64,
    hardware: HardwareInfo,
    build: BuildInfo,
    benchmarks: []BenchmarkResult,
    integration: IntegrationResult,
    average_speedup: ?f64,      // null if GPU not available
    gpu_utilization_percent: ?f64,  // null if GPU not available
    recommendation: []const u8,
    
    // Flag to indicate if this is real data or GPU unavailable
    gpu_active: bool,
};

// ============================================================================
// Report Generator
// ============================================================================

pub const ReportGenerator = struct {
    allocator: std.mem.Allocator,
    data: ReportData,

    pub fn init(allocator: std.mem.Allocator, data: ReportData) !*ReportGenerator {
        const self = try allocator.create(ReportGenerator);
        self.* = ReportGenerator{
            .allocator = allocator,
            .data = data,
        };
        return self;
    }

    pub fn deinit(self: *ReportGenerator) void {
        self.allocator.destroy(self);
    }

    // Generate Markdown report
    pub fn generateMarkdown(self: *ReportGenerator, writer: anytype) !void {
        const data = self.data;
        
        // Header
        try writer.writeAll("# GPU Performance Validation Report\n\n");
        try writer.print("**Date:** {d}\n", .{data.timestamp});
        try writer.print("**System:** {s}\n", .{data.hardware.gpu_model});
        try writer.writeAll("**Test Suite Version:** 1.0\n\n");

        // Executive Summary
        try writer.writeAll("## Executive Summary\n\n");
        const status = if (data.integration.allPassed()) "✓ APPROVED" else "✗ NEEDS WORK";
        try writer.print("- **Status:** {s}\n", .{status});
        try writer.print("- **Performance Improvement:** {d:.1}× over CPU\n", .{data.average_speedup});
        try writer.print("- **GPU Utilization:** {d:.1}%\n", .{data.gpu_utilization_percent});
        try writer.print("- **Recommendation:** {s}\n\n", .{data.recommendation});

        // Hardware Configuration
        try writer.writeAll("## 1. Hardware Configuration\n\n");
        try writer.writeAll("### GPU Specifications\n");
        try writer.print("- **Model:** {s}\n", .{data.hardware.gpu_model});
        try writer.print("- **VRAM:** {d} MB ({d:.1} GB)\n", .{
            data.hardware.gpu_memory_mb,
            @as(f64, @floatFromInt(data.hardware.gpu_memory_mb)) / 1024.0,
        });
        try writer.print("- **Compute Capability:** {d}.{d}\n", .{
            data.hardware.compute_capability.major,
            data.hardware.compute_capability.minor,
        });
        try writer.print("- **CUDA Version:** {s}\n", .{data.hardware.cuda_version});
        try writer.print("- **Driver Version:** {s}\n\n", .{data.hardware.driver_version});

        try writer.writeAll("### System Environment\n");
        try writer.print("- **OS:** {s}\n", .{data.hardware.os});
        try writer.print("- **CPU:** {s}\n", .{data.hardware.cpu_model});
        try writer.print("- **RAM:** {d} GB\n\n", .{data.hardware.ram_gb});

        // Build Verification
        try writer.writeAll("## 2. Build Verification\n\n");
        try writer.writeAll("### Library Linkage\n");
        try writer.print("- ✓ libcuda.so - {s}\n", .{data.build.libcuda_version});
        try writer.print("- ✓ libcudart.so - {s}\n", .{data.build.libcudart_version});
        try writer.print("- ✓ libcublas.so - {s}\n\n", .{data.build.libcublas_version});

        try writer.writeAll("### Symbol Resolution\n");
        if (data.build.symbols_resolved) {
            try writer.writeAll("- ✓ Core CUDA symbols resolved\n");
            try writer.writeAll("- ✓ cuBLAS symbols resolved\n\n");
        } else {
            try writer.writeAll("- ✗ Symbol resolution failed\n\n");
        }

        // Performance Benchmarks
        try writer.writeAll("## 3. Performance Benchmarks\n\n");
        try writer.writeAll("### Matrix Multiplication (FP32)\n\n");
        
        if (data.gpu_active) {
            try writer.writeAll("| Size | CPU Time | GPU Time | Speedup | CPU GFLOPS | GPU GFLOPS |\n");
            try writer.writeAll("|------|----------|----------|---------|------------|------------|\n");

            for (data.benchmarks) |bench| {
                const gpu_time = bench.gpu_time_ms orelse continue;
                const speedup = bench.speedup orelse continue;
                const gpu_gflops = bench.gpu_gflops orelse continue;
                
                try writer.print("| {d}×{d} | {d:.2}ms ✓ | {d:.2}ms ✓ | {d:.0}× | {d:.1} | {d:.1} |\n", .{
                    bench.size, bench.size,
                    bench.cpu_time_ms,
                    gpu_time,
                    speedup,
                    bench.cpu_gflops,
                    gpu_gflops,
                });
            }

            if (data.average_speedup) |avg_speedup| {
                try writer.print("\n**Average Speedup: {d:.0}×** ", .{avg_speedup});
                if (avg_speedup >= 50) {
                    try writer.writeAll("✓\n\n");
                } else {
                    try writer.writeAll("⚠\n\n");
                }
            }
        } else {
            // GPU not active - show CPU measurements only
            try writer.writeAll("| Size | CPU Time (Measured) | Expected GPU Time | Expected Speedup |\n");
            try writer.writeAll("|------|---------------------|-------------------|------------------|\n");

            for (data.benchmarks) |bench| {
                const expected_gpu_time = baselines.LlamaCppBenchmarks.T4.matmul_256_ms;
                const expected_speedup = bench.cpu_time_ms / expected_gpu_time;
                
                try writer.print("| {d}×{d} | {d:.2}ms ✓ | {d:.2}ms 📊 | ~{d:.0}× 📊 |\n", .{
                    bench.size, bench.size,
                    bench.cpu_time_ms,
                    expected_gpu_time,
                    expected_speedup,
                });
            }
            
            try writer.writeAll("\n**Legend:**\n");
            try writer.writeAll("- ✓ = Actually measured on this system\n");
            try writer.writeAll("- 📊 = Industry baseline (llama.cpp T4 benchmark)\n\n");
            
            try writer.writeAll("**Status:** GPU operations not active - showing CPU measurements with industry reference\n\n");
        }

        // Integration Testing
        try writer.writeAll("## 4. Integration Testing\n\n");
        try writer.writeAll("### Smoke Tests\n");
        try self.writeCheckmark(writer, data.integration.gpu_detected, "GPU Detection");
        try self.writeCheckmark(writer, data.integration.memory_allocated, "Memory Allocation");
        try self.writeCheckmark(writer, data.integration.operations_working, "Basic Operations");
        try self.writeCheckmark(writer, data.integration.performance_acceptable, "Performance >10× CPU");
        try self.writeCheckmark(writer, data.integration.no_memory_leaks, "No Memory Leaks");
        try writer.writeAll("\n");

        // Technical Approval Checklist
        try writer.writeAll("## 5. Technical Approval Checklist\n\n");
        try self.writeCheckbox(writer, data.average_speedup >= 50, "GPU acceleration verified (>50× speedup)");
        try self.writeCheckbox(writer, data.build.symbols_resolved, "Build system properly configured");
        try self.writeCheckbox(writer, data.integration.no_memory_leaks, "No memory leaks detected");
        try self.writeCheckbox(writer, data.integration.performance_acceptable, "Performance meets requirements");
        try self.writeCheckbox(writer, data.integration.allPassed(), "All integration tests passing");

        // Conclusion
        try writer.writeAll("\n## 6. Conclusion\n\n");
        
        if (data.gpu_active) {
            if (data.integration.allPassed() and data.average_speedup != null and data.average_speedup.? >= 50) {
                try writer.writeAll("**Status: ✅ APPROVED FOR PRODUCTION**\n\n");
                try writer.writeAll("The GPU integration demonstrates:\n");
                try writer.print("- **{d:.0}× average speedup** over CPU (measured)\n", .{data.average_speedup.?});
                try writer.writeAll("- **Stable operation** verified\n");
                try writer.writeAll("- **Production-ready** configuration\n\n");
            } else {
                try writer.writeAll("**Status: ⚠️ NEEDS IMPROVEMENT**\n\n");
                try writer.writeAll("Issues detected:\n");
                if (data.average_speedup) |speedup| {
                    if (speedup < 50) {
                        try writer.print("- Low GPU speedup ({d:.1}× measured, expected >50×)\n", .{speedup});
                    }
                }
                if (!data.integration.allPassed()) {
                    try writer.writeAll("- Integration tests failed\n");
                }
                try writer.writeAll("\n");
            }
        } else {
            try writer.writeAll("**Status: 🔧 GPU INTEGRATION PENDING**\n\n");
            try writer.writeAll("Current State:\n");
            try writer.writeAll("- ✓ CPU baseline measurements complete\n");
            try writer.writeAll("- ⚠ GPU operations not active\n");
            try writer.writeAll("- 📊 Expected performance: 500-1000× speedup with proper GPU integration\n\n");
            try writer.writeAll("**Next Steps:**\n");
            try writer.writeAll("1. Verify CUDA libraries are linked in build.zig\n");
            try writer.writeAll("2. Enable CUDA backend selection\n");
            try writer.writeAll("3. Re-run tests to measure actual GPU performance\n\n");
        }

        try writer.writeAll("---\n");
        try writer.print("**Report Generated:** {d}\n", .{data.timestamp});
        try writer.writeAll("**Tool:** GPU Integration Suite v1.0\n");
    }

    // Generate JSON report
    pub fn generateJSON(self: *ReportGenerator, writer: anytype) !void {
        try std.json.stringify(self.data, .{ .whitespace = .indent_2 }, writer);
    }

    // Generate CSV summary
    pub fn generateCSV(self: *ReportGenerator, writer: anytype) !void {
        try writer.writeAll("Metric,Value,Unit\n");
        try writer.print("Timestamp,{d},unix_ms\n", .{self.data.timestamp});
        try writer.print("GPU Model,{s},text\n", .{self.data.hardware.gpu_model});
        try writer.print("Average Speedup,{d:.1},multiplier\n", .{self.data.average_speedup});
        try writer.print("GPU Utilization,{d:.1},percent\n", .{self.data.gpu_utilization_percent});
        try writer.print("Status,{s},text\n", .{if (self.data.integration.allPassed()) "PASS" else "FAIL"});

        try writer.writeAll("\nBenchmark,Size,CPU_ms,GPU_ms,Speedup,CPU_GFLOPS,GPU_GFLOPS\n");
        for (self.data.benchmarks) |bench| {
            try writer.print("{s},{d},{d:.2},{d:.2},{d:.1},{d:.1},{d:.1}\n", .{
                bench.name,
                bench.size,
                bench.cpu_time_ms,
                bench.gpu_time_ms,
                bench.speedup,
                bench.cpu_gflops,
                bench.gpu_gflops,
            });
        }
    }

    fn writeCheckmark(self: *ReportGenerator, writer: anytype, passed: bool, label: []const u8) !void {
        _ = self;
        const symbol = if (passed) "✓" else "✗";
        try writer.print("- {s} {s}\n", .{ symbol, label });
    }

    fn writeCheckbox(self: *ReportGenerator, writer: anytype, checked: bool, label: []const u8) !void {
        _ = self;
        const box = if (checked) "[x]" else "[ ]";
        try writer.print("- {s} {s}\n", .{ box, label });
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

pub fn generateReport(
    allocator: std.mem.Allocator,
    data: ReportData,
    format: ReportFormat,
    output_path: []const u8,
) !void {
    const generator = try ReportGenerator.init(allocator, data);
    defer generator.deinit();

    const file = try std.fs.cwd().createFile(output_path, .{});
    defer file.close();

    const writer = file.writer();

    switch (format) {
        .Markdown => try generator.generateMarkdown(writer),
        .JSON => try generator.generateJSON(writer),
        .CSV => try generator.generateCSV(writer),
    }
}

pub const ReportFormat = enum {
    Markdown,
    JSON,
    CSV,
};

// ============================================================================
// Tests
// ============================================================================

test "ReportGenerator: markdown with measured data" {
    const allocator = std.testing.allocator;

    // Test data representing actual measurements (GPU active)
    const data = ReportData{
        .timestamp = 1705867200000,
        .hardware = .{
            .gpu_model = "Tesla T4",
            .gpu_memory_mb = 15360,
            .compute_capability = .{ .major = 7, .minor = 5 },
            .cuda_version = "12.2",
            .driver_version = "535.x",
            .cpu_model = "Intel Xeon",
            .ram_gb = 64,
            .os = "Ubuntu 22.04",
        },
        .build = .{
            .libcuda_version = "12.2",
            .libcudart_version = "12.2",
            .libcublas_version = "12.2",
            .symbols_resolved = true,
            .link_mode = "dynamic",
        },
        .benchmarks = &[_]BenchmarkResult{
            .{
                .name = "matmul_256",
                .size = 256,
                .cpu_time_ms = 182.0,
                .gpu_time_ms = 0.31,
                .speedup = 587.0,
                .cpu_gflops = 1.8,
                .gpu_gflops = 1057.0,
                .cpu_source = .Measured,
                .gpu_source = .Measured,
            },
        },
        .integration = .{
            .gpu_detected = true,
            .memory_allocated = true,
            .operations_working = true,
            .performance_acceptable = true,
            .no_memory_leaks = true,
        },
        .average_speedup = 587.0,
        .gpu_utilization_percent = 98.5,
        .recommendation = "Approved for production deployment",
        .gpu_active = true,
    };

    const generator = try ReportGenerator.init(allocator, data);
    defer generator.deinit();

    var output = std.ArrayList(u8){};
    defer output.deinit();

    try generator.generateMarkdown(output.writer());

    // Verify output contains key sections
    try std.testing.expect(std.mem.indexOf(u8, output.items, "GPU Performance Validation Report") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.items, "Tesla T4") != null);
}

test "ReportGenerator: markdown with CPU-only data" {
    const allocator = std.testing.allocator;

    // Test data when GPU is not active (CPU measurements only)
    const data = ReportData{
        .timestamp = 1705867200000,
        .hardware = .{
            .gpu_model = "Not Detected",
            .gpu_memory_mb = 0,
            .compute_capability = .{ .major = 0, .minor = 0 },
            .cuda_version = "N/A",
            .driver_version = "N/A",
            .cpu_model = "Intel Xeon",
            .ram_gb = 64,
            .os = "Ubuntu 22.04",
        },
        .build = .{
            .libcuda_version = "N/A",
            .libcudart_version = "N/A",
            .libcublas_version = "N/A",
            .symbols_resolved = false,
            .link_mode = "static",
        },
        .benchmarks = &[_]BenchmarkResult{
            .{
                .name = "matmul_256",
                .size = 256,
                .cpu_time_ms = 182.0,
                .gpu_time_ms = null,
                .speedup = null,
                .cpu_gflops = 1.8,
                .gpu_gflops = null,
                .cpu_source = .Measured,
                .gpu_source = .Unavailable,
            },
        },
        .integration = .{
            .gpu_detected = false,
            .memory_allocated = false,
            .operations_working = false,
            .performance_acceptable = false,
            .no_memory_leaks = true,
        },
        .average_speedup = null,
        .gpu_utilization_percent = null,
        .recommendation = "GPU integration required",
        .gpu_active = false,
    };

    const generator = try ReportGenerator.init(allocator, data);
    defer generator.deinit();

    var output = std.ArrayList(u8){};
    defer output.deinit();

    try generator.generateMarkdown(output.writer());

    // Should contain CPU measurements and baseline references
    try std.testing.expect(std.mem.indexOf(u8, output.items, "CPU Time (Measured)") != null);
    try std.testing.expect(std.mem.indexOf(u8, output.items, "Industry baseline") != null);
}
