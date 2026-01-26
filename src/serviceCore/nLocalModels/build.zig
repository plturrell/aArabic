const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Main nLocalModels server executable
    const server_exe = b.addExecutable(.{
        .name = "nlocalmodels",
        .root_module = b.createModule(.{
            .root_source_file = b.path("inference/engine/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // For macOS Metal support
    if (target.result.os.tag == .macos) {
        server_exe.linkFramework("Metal");
        server_exe.linkFramework("Foundation");
    }

    // For Linux CUDA support
    if (target.result.os.tag == .linux) {
        server_exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        server_exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
        server_exe.root_module.addRPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        server_exe.root_module.addRPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
        server_exe.linkSystemLibrary("cuda");
        server_exe.linkSystemLibrary("cublas");
        server_exe.linkSystemLibrary("cudart");
    }

    b.installArtifact(server_exe);

    // Run command
    const run_cmd = b.addRunArtifact(server_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the nLocalModels server");
    run_step.dependOn(&run_cmd.step);

    // Build orchestration tools as well
    const orchestration_build = b.addExecutable(.{
        .name = "orchestration-tools",
        .root_module = b.createModule(.{
            .root_source_file = b.path("orchestration/analytics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(orchestration_build);

    // Test step - runs inference engine tests
    const test_step = b.step("test", "Run all tests");
    // Tests are in inference/engine subdirectory
    // Users should run: cd inference/engine && zig build test
}