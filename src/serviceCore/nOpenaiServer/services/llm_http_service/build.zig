const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create the executable
    const exe = b.addExecutable(.{
        .name = "llm_http",
        .root_source_file = b.path("llm_http.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Link to Mojo library (will be compiled separately)
    // Note: In production, link to compiled Mojo .o file
    // For now, we'll note the dependency
    
    // Add this when Mojo is compiled:
    // exe.addObjectFile(b.path("llm_server.o"));
    
    b.installArtifact(exe);

    // Create run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the LLM HTTP server");
    run_step.dependOn(&run_cmd.step);
}
