const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable - nMetaData HTTP server
    const exe = b.addExecutable(.{
        .name = "nmetadata_server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add C library for SQLite (testing only)
    exe.linkLibC();
    exe.linkSystemLibrary("sqlite3");

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the nMetaData server");
    run_step.dependOn(&run_cmd.step);

    // Unit tests
    const unit_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    unit_tests.linkLibC();
    unit_tests.linkSystemLibrary("sqlite3");

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Database abstraction layer tests (placeholder for now)
    // const db_tests = b.addTest(.{
    //     .root_source_file = .{ .cwd_relative = "zig/db/client.zig" },
    //     .target = target,
    //     .optimize = optimize,
    // });
    // db_tests.linkLibC();
    // db_tests.linkSystemLibrary("sqlite3");

    // const run_db_tests = b.addRunArtifact(db_tests);
    // const db_test_step = b.step("test-db", "Run database tests");
    // db_test_step.dependOn(&run_db_tests.step);
    // HTTP server tests (placeholder for now)
    // const http_tests = b.addTest(.{
    //     .root_source_file = .{ .cwd_relative = "zig/http/server.zig" },
    //     .target = target,
    //     .optimize = optimize,
    // });
    // const run_http_tests = b.addRunArtifact(http_tests);
    // const http_test_step = b.step("test-http", "Run HTTP server tests");
    // http_test_step.dependOn(&run_http_tests.step);

    // Benchmarks (placeholder for now)
    // const bench = b.addExecutable(.{
    //     .name = "benchmark",
    //     .root_source_file = .{ .cwd_relative = "zig/bench.zig" },
    //     .target = target,
    //     .optimize = .ReleaseFast,
    // });
    // bench.linkLibC();
    // bench.linkSystemLibrary("sqlite3");
    // const run_bench = b.addRunArtifact(bench);
    // const bench_step = b.step("bench", "Run performance benchmarks");
    // bench_step.dependOn(&run_bench.step);

    // Format check
    const fmt_step = b.step("fmt", "Format all Zig source files");
    const fmt = b.addFmt(.{
        .paths = &.{"zig"},
        .check = false,
    });
    fmt_step.dependOn(&fmt.step);

    // Lint check
    const lint_step = b.step("lint", "Check code formatting");
    const lint = b.addFmt(.{
        .paths = &.{"zig"},
        .check = true,
    });
    lint_step.dependOn(&lint.step);
}
