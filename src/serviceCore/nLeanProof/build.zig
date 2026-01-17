const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const server_exe = b.addExecutable(.{
        .name = "leanshimmy",
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    b.installArtifact(server_exe);

    const io_lib = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "leanshimmy_io",
        .root_module = b.createModule(.{
            .root_source_file = b.path("io/leanshimmy_io.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(io_lib);

    const run_cmd = b.addRunArtifact(server_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run leanShimmy server");
    run_step.dependOn(&run_cmd.step);

    const conformance_exe = b.addExecutable(.{
        .name = "lean4-discover",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/conformance/discover.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(conformance_exe);

    const run_conformance = b.addRunArtifact(conformance_exe);
    run_conformance.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_conformance.addArgs(args);
    }

    const conformance_step = b.step("conformance-discover", "Discover Lean4 tests");
    conformance_step.dependOn(&run_conformance.step);

    const baseline_exe = b.addExecutable(.{
        .name = "lean4-baseline",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/conformance/baseline.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(baseline_exe);

    const run_baseline = b.addRunArtifact(baseline_exe);
    run_baseline.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_baseline.addArgs(args);
    }

    const baseline_step = b.step("conformance-baseline", "Generate Lean4 test baseline report");
    baseline_step.dependOn(&run_baseline.step);

    const manifest_exe = b.addExecutable(.{
        .name = "lean4-manifest",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/conformance/manifest.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(manifest_exe);

    const run_manifest = b.addRunArtifact(manifest_exe);
    run_manifest.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_manifest.addArgs(args);
    }

    const manifest_step = b.step("conformance-manifest", "Generate Lean4 test manifest");
    manifest_step.dependOn(&run_manifest.step);

    const summary_exe = b.addExecutable(.{
        .name = "lean4-summary",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/conformance/summary.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(summary_exe);

    const run_summary = b.addRunArtifact(summary_exe);
    run_summary.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_summary.addArgs(args);
    }

    const summary_step = b.step("conformance-summary", "Generate Lean4 conformance summary");
    summary_step.dependOn(&run_summary.step);

    const server_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_server_tests = b.addRunArtifact(server_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_server_tests.step);
}
