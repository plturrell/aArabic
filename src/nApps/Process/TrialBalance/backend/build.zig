const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add n-c-sdk dependency
    const n_c_sdk_path = b.pathFromRoot("../../../nLang/n-c-sdk");
    const n_c_sdk_module = b.addModule("n-c-sdk", .{
        .root_source_file = b.path(b.pathJoin(&.{ n_c_sdk_path, "src/lib.zig" })),
    });

    // Trial Balance Backend Executable
    const exe = b.addExecutable(.{
        .name = "trial-balance-backend",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.root_module.addImport("n-c-sdk", n_c_sdk_module);
    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the trial balance backend server");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    unit_tests.root_module.addImport("n-c-sdk", n_c_sdk_module);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Benchmarks
    const benchmark = b.addExecutable(.{
        .name = "trial-balance-benchmark",
        .root_source_file = b.path("benchmarks/benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });

    benchmark.root_module.addImport("n-c-sdk", n_c_sdk_module);

    const run_benchmark = b.addRunArtifact(benchmark);
    const benchmark_step = b.step("benchmark", "Run benchmarks");
    benchmark_step.dependOn(&run_benchmark.step);
}