const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Analytics CLI
    const analytics_exe = b.addExecutable(.{
        .name = "analytics",
        .root_source_file = b.path("analytics.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(analytics_exe);

    // GPU Monitor CLI
    const gpu_monitor_exe = b.addExecutable(.{
        .name = "gpu_monitor",
        .root_source_file = b.path("gpu_monitor_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(gpu_monitor_exe);

    // Benchmark CLI
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_source_file = b.path("benchmark_cli.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(benchmark_exe);

    // Benchmark Validator CLI
    const validator_exe = b.addExecutable(.{
        .name = "benchmark_validator",
        .root_source_file = b.path("benchmark_validator.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(validator_exe);

    // HF Model Card Extractor CLI
    const extractor_exe = b.addExecutable(.{
        .name = "hf_extractor",
        .root_source_file = b.path("hf_model_card_extractor.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(extractor_exe);

    // Run commands for each tool
    const analytics_run = b.addRunArtifact(analytics_exe);
    analytics_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        analytics_run.addArgs(args);
    }
    const analytics_step = b.step("run-analytics", "Run the analytics tool");
    analytics_step.dependOn(&analytics_run.step);

    const gpu_monitor_run = b.addRunArtifact(gpu_monitor_exe);
    gpu_monitor_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        gpu_monitor_run.addArgs(args);
    }
    const gpu_monitor_step = b.step("run-gpu-monitor", "Run the GPU monitor");
    gpu_monitor_step.dependOn(&gpu_monitor_run.step);

    const benchmark_run = b.addRunArtifact(benchmark_exe);
    benchmark_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        benchmark_run.addArgs(args);
    }
    const benchmark_step = b.step("run-benchmark", "Run the benchmark tool");
    benchmark_step.dependOn(&benchmark_run.step);

    const validator_run = b.addRunArtifact(validator_exe);
    validator_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        validator_run.addArgs(args);
    }
    const validator_step = b.step("run-validator", "Run the benchmark validator");
    validator_step.dependOn(&validator_run.step);

    const extractor_run = b.addRunArtifact(extractor_exe);
    extractor_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        extractor_run.addArgs(args);
    }
    const extractor_step = b.step("run-extractor", "Run the HF model card extractor");
    extractor_step.dependOn(&extractor_run.step);
}
