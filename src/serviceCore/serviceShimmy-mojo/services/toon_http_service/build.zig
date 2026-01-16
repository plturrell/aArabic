const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // HTTP Server Executable
    const exe = b.addExecutable(.{
        .name = "toon_http",
        .root_module = b.createModule(.{
            .root_source_file = b.path("toon_http.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Link to existing TOON encoder
    const toon_encoder_path = "../../recursive_llm/toon/zig_toon.zig";
    const toon_module = b.createModule(.{
        .root_source_file = b.path(toon_encoder_path),
    });
    exe.root_module.addImport("toon_encoder", toon_module);

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the TOON HTTP server");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("toon_http.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    unit_tests.root_module.addImport("toon_encoder", toon_module);

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
