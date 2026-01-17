const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const lib = b.addLibrary(.{
        .name = "inference_fixture",
        .root_module = b.createModule(.{
            .root_source_file = .{ .src_path = "server/inference_fixture.zig" },
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
    });
    b.installArtifact(lib);
}
