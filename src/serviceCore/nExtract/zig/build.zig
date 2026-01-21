const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ========== Main Library Module ==========
    const lib_module = b.createModule(.{
        .root_source_file = b.path("nExtract.zig"),
    });

    // ========== Static Library ==========
    const lib = b.addLibrary(.{
        .name = "nExtract",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nExtract.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .static,
    });
    lib.linkLibC();
    b.installArtifact(lib);

    // ========== Shared Library (for FFI) ==========
    const shared_lib = b.addLibrary(.{
        .name = "nExtract",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nExtract.zig"),
            .target = target,
            .optimize = optimize,
        }),
        .linkage = .dynamic,
    });
    shared_lib.linkLibC();
    b.installArtifact(shared_lib);

    // ========== Tests ==========
    const lib_unit_tests = b.addTest(.{
        .name = "test_nExtract",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nExtract.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);

    // ========== Benchmarks ==========
    // Benchmarks will be added as the project progresses
    const bench_step = b.step("bench", "Run benchmarks");
    _ = bench_step; // Placeholder

    // ========== Fuzzing ==========
    // Fuzz targets will be added as parsers are implemented
    const fuzz_step = b.step("fuzz", "Build fuzz targets");
    _ = fuzz_step; // Placeholder

    // ========== Documentation ==========
    const docs_step = b.step("docs", "Generate documentation");
    
    const lib_docs = lib.getEmittedDocs();
    const install_docs = b.addInstallDirectory(.{
        .source_dir = lib_docs,
        .install_dir = .prefix,
        .install_subdir = "docs/zig",
    });
    docs_step.dependOn(&install_docs.step);

    // ========== Install Headers ==========
    // Header installation will be added when nExtract.h is created
    _ = b.step("install-headers", "Install C headers for FFI");

    // ========== Format Check ==========
    const fmt_step = b.step("fmt", "Format source code");
    const fmt = b.addFmt(.{
        .paths = &.{
            "core",
            "parsers",
            "ocr",
            "ml",
            "pdf",
            "tests",
            "nExtract.zig",
            "build.zig",
        },
    });
    fmt_step.dependOn(&fmt.step);

    // ========== Lint ==========
    const lint_step = b.step("lint", "Run linter");
    // Zig doesn't have a separate linter; we use the compiler in check mode
    const lint = b.addTest(.{
        .name = "test_lint",
        .root_module = b.createModule(.{
            .root_source_file = b.path("nExtract.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    lint_step.dependOn(&lint.step);

    // ========== Default run ==========
    const default_run_step = b.step("run", "Run tests");
    default_run_step.dependOn(&run_lib_unit_tests.step);

    // Keep lib_module for future use
    _ = lib_module;
}
