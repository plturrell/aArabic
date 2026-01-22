// Build configuration for tiering module

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main library
    const lib = b.addStaticLibrary(.{
        .name = "shimmy_tiering",
        .root_source_file = b.path("mod.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(lib);

    // Shared library for FFI
    const shared = b.addSharedLibrary(.{
        .name = "shimmy_tiering",
        .root_source_file = b.path("unified_tier.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(shared);

    // Benchmark executable
    const bench = b.addExecutable(.{
        .name = "tiering_benchmark",
        .root_source_file = b.path("benchmark.zig"),
        .target = target,
        .optimize = optimize,
    });
    b.installArtifact(bench);

    const run_bench = b.addRunArtifact(bench);
    run_bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_bench.addArgs(args);
    }

    const bench_step = b.step("bench", "Run tiering benchmarks");
    bench_step.dependOn(&run_bench.step);

    // Tests
    const tests = b.addTest(.{
        .root_source_file = b.path("mod.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run tiering tests");
    test_step.dependOn(&run_tests.step);
}

