const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // HTTP server executable
    const server = b.addExecutable(.{
        .name = "nlocalmodels-server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("server.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    
    server.linkLibC();

    b.installArtifact(server);

    // Run command
    const run_cmd = b.addRunArtifact(server);
    run_cmd.step.dependOn(b.getInstallStep());

    const run_step = b.step("run", "Run the nLocalModels server");
    run_step.dependOn(&run_cmd.step);
}