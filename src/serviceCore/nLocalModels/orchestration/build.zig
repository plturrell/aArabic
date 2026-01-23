const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    
    // ✅ SCB OPTIMIZATION 1: Use ReleaseSafe by default for banking
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseSafe,
    });

    // Analytics CLI
    const analytics_exe = b.addExecutable(.{
        .name = "analytics",
        .root_module = b.createModule(.{
            .root_source_file = b.path("analytics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // ✅ SCB OPTIMIZATION 2: Enable LTO (only in release modes)
    if (optimize != .Debug) {
        analytics_exe.want_lto = true;
        analytics_exe.use_lld = true;
    }
    // ✅ SCB OPTIMIZATION 3: Banking flags
    analytics_exe.root_module.addOptions("build_options", b.addOptions());
    b.installArtifact(analytics_exe);

    // GPU Monitor CLI
    const gpu_monitor_exe = b.addExecutable(.{
        .name = "gpu_monitor",
        .root_module = b.createModule(.{
            .root_source_file = b.path("gpu_monitor_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (optimize != .Debug) {
        gpu_monitor_exe.want_lto = true;
        gpu_monitor_exe.use_lld = true;
    }
    b.installArtifact(gpu_monitor_exe);

    // Benchmark CLI
    const benchmark_exe = b.addExecutable(.{
        .name = "benchmark",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmark_cli.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (optimize != .Debug) {
        benchmark_exe.want_lto = true;
        benchmark_exe.use_lld = true;
    }
    b.installArtifact(benchmark_exe);

    // Benchmark Validator CLI
    const validator_exe = b.addExecutable(.{
        .name = "benchmark_validator",
        .root_module = b.createModule(.{
            .root_source_file = b.path("benchmark_validator.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (optimize != .Debug) {
        validator_exe.want_lto = true;
        validator_exe.use_lld = true;
    }
    b.installArtifact(validator_exe);

    // HF Model Card Extractor CLI
    const extractor_exe = b.addExecutable(.{
        .name = "hf_extractor",
        .root_module = b.createModule(.{
            .root_source_file = b.path("hf_model_card_extractor.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (optimize != .Debug) {
        extractor_exe.want_lto = true;
        extractor_exe.use_lld = true;
    }
    b.installArtifact(extractor_exe);

    // Dataset Loader CLI
    const dataset_loader_exe = b.addExecutable(.{
        .name = "dataset_loader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("dataset_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    if (optimize != .Debug) {
        dataset_loader_exe.want_lto = true;
        dataset_loader_exe.use_lld = true;
    }
    b.installArtifact(dataset_loader_exe);

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

    const dataset_loader_run = b.addRunArtifact(dataset_loader_exe);
    dataset_loader_run.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        dataset_loader_run.addArgs(args);
    }
    const dataset_loader_step = b.step("run-dataset-loader", "Run the dataset loader tool");
    dataset_loader_step.dependOn(&dataset_loader_run.step);

    // Tests for mojo integration (library only, not executable)
    const mojo_tests = b.addTest(.{
        .name = "test_mojo_integration",
        .root_module = b.createModule(.{
            .root_source_file = b.path("mojo_compiler_integration.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    
    const run_mojo_tests = b.addRunArtifact(mojo_tests);
    const mojo_test_step = b.step("test-mojo", "Run Mojo integration tests");
    mojo_test_step.dependOn(&run_mojo_tests.step);
    
    // Dataset loader tests
    const dataset_tests = b.addTest(.{
        .name = "test_dataset_loader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("dataset_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    
    const run_dataset_tests = b.addRunArtifact(dataset_tests);
    const dataset_test_step = b.step("test-dataset", "Run dataset loader tests");
    dataset_test_step.dependOn(&run_dataset_tests.step);
    
    // Main test step runs all tests
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_mojo_tests.step);
    test_step.dependOn(&run_dataset_tests.step);
}
