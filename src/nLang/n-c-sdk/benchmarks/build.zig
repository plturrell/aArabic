const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Benchmark framework module
    const framework_mod = b.createModule(.{
        .root_source_file = b.path("framework.zig"),
    });

    // Array operations benchmark
    const array_bench = b.addExecutable(.{
        .name = "array_operations",
        .root_module = b.createModule(.{
            .root_source_file = b.path("array_operations.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    array_bench.root_module.addImport("framework", framework_mod);
    b.installArtifact(array_bench);

    // String processing benchmark
    const string_bench = b.addExecutable(.{
        .name = "string_processing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("string_processing.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    string_bench.root_module.addImport("framework", framework_mod);
    b.installArtifact(string_bench);

    // Computation benchmark
    const compute_bench = b.addExecutable(.{
        .name = "computation",
        .root_module = b.createModule(.{
            .root_source_file = b.path("computation.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    compute_bench.root_module.addImport("framework", framework_mod);
    b.installArtifact(compute_bench);

    // Performance profiler
    const profiler = b.addExecutable(.{
        .name = "performance_profiler",
        .root_module = b.createModule(.{
            .root_source_file = b.path("performance_profiler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    profiler.root_module.addImport("framework", framework_mod);
    b.installArtifact(profiler);

    // Fuzz test suite
    const fuzz_test = b.addExecutable(.{
        .name = "fuzz_test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("fuzz_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    fuzz_test.root_module.addImport("framework", framework_mod);
    b.installArtifact(fuzz_test);

    // Run all benchmarks step
    const run_all = b.step("bench", "Run all benchmarks");
    
    const run_array = b.addRunArtifact(array_bench);
    const run_string = b.addRunArtifact(string_bench);
    const run_compute = b.addRunArtifact(compute_bench);
    const run_profiler = b.addRunArtifact(profiler);
    
    run_all.dependOn(&run_array.step);
    run_all.dependOn(&run_string.step);
    run_all.dependOn(&run_compute.step);
    run_all.dependOn(&run_profiler.step);

    // Fuzz testing step
    const fuzz_step = b.step("fuzz", "Run fuzz tests");
    const run_fuzz = b.addRunArtifact(fuzz_test);
    fuzz_step.dependOn(&run_fuzz.step);

    // Test step runs both benchmarks and fuzz tests
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(run_all);
    test_step.dependOn(fuzz_step);
}
