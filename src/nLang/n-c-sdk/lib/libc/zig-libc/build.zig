const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Create a module for the library (for Zig usage)
    const lib_mod = b.addModule("zig-libc", .{
        .root_source_file = b.path("src/lib.zig"),
        .target = target,
        .optimize = optimize,
    });

    // Create static library (for C usage)
    const lib = b.addLibrary(.{
        .linkage = .static,
        .name = "zig-libc",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/lib.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // This library is a libc replacement, so it shouldn't link against libc
    // However, it uses Zig's std lib which might depend on system libraries (like libSystem on macOS)
    // We export C symbols, so this library can be linked by C programs
    b.installArtifact(lib);

    // Memory tests
    const memory_tests = b.addTest(.{
        .name = "test_memory",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_memory.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    memory_tests.root_module.addImport("zig-libc", lib_mod);

    // String conversion tests
    const conversion_tests = b.addTest(.{
        .name = "test_string_conversion",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_string_conversion.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    conversion_tests.root_module.addImport("zig-libc", lib_mod);

    // String literal tests
    const literal_tests = b.addTest(.{
        .name = "test_string_literals",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_string_literals.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    literal_tests.root_module.addImport("zig-libc", lib_mod);

    // Run step for all tests
    const run_memory_tests = b.addRunArtifact(memory_tests);
    const run_conversion_tests = b.addRunArtifact(conversion_tests);
    const run_literal_tests = b.addRunArtifact(literal_tests);
    const random_tests = b.addTest(.{
        .name = "test_random",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_random.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    random_tests.root_module.addImport("zig-libc", lib_mod);
    const run_random_tests = b.addRunArtifact(random_tests);

    const test_step = b.step("test", "Run all library tests");
    test_step.dependOn(&run_memory_tests.step);
    test_step.dependOn(&run_conversion_tests.step);
    test_step.dependOn(&run_literal_tests.step);
    test_step.dependOn(&run_random_tests.step);

    // Individual test steps
    const test_memory_step = b.step("test-memory", "Run memory allocation tests");
    test_memory_step.dependOn(&run_memory_tests.step);

    const test_conversion_step = b.step("test-conversion", "Run string conversion tests");
    test_conversion_step.dependOn(&run_conversion_tests.step);

    const test_literals_step = b.step("test-literals", "Run string literal tests");
    test_literals_step.dependOn(&run_literal_tests.step);

    const test_random_step = b.step("test-random", "Run random/distribution tests");
    test_random_step.dependOn(&run_random_tests.step);
}
