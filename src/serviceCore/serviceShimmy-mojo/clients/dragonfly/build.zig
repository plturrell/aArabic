const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Build shared library for Mojo integration
    const lib = b.addSharedLibrary(.{
        .name = "dragonfly_client",
        .root_source_file = b.path("dragonfly_client.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Install to lib directory
    b.installArtifact(lib);

    // Build standalone executable for testing
    const exe = b.addExecutable(.{
        .name = "dragonfly-test",
        .root_source_file = b.path("test.zig"),
        .target = target,
        .optimize = optimize,
    });

    b.installArtifact(exe);

    // Add run step for testing
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the DragonflyDB client test");
    run_step.dependOn(&run_cmd.step);

    // Add unit tests
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("dragonfly_client.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
}
