const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard target options - allows cross-compilation
    const target = b.standardTargetOptions(.{});

    // Standard optimization options
    const optimize = b.standardOptimizeOption(.{});

    // ========================================================================
    // Shared Library: libscip (libscip.dylib on macOS, libscip.so on Linux)
    // Exports C ABI functions for Mojo integration
    // ========================================================================
    const scip_reader_mod = b.createModule(.{
        .root_source_file = b.path("scip_reader.zig"),
        .target = target,
        .optimize = optimize,
    });

    const scip_writer_mod = b.createModule(.{
        .root_source_file = b.path("zig_scip_writer.zig"),
        .target = target,
        .optimize = optimize,
    });
    scip_writer_mod.addImport("scip_reader", scip_reader_mod);

    const libscip = b.addLibrary(.{
        .linkage = .dynamic,
        .name = "scip",
        .root_module = scip_writer_mod,
    });

    // Install the shared library
    b.installArtifact(libscip);

    // "lib" step - builds and installs shared library only
    const lib_step = b.step("lib", "Build the SCIP shared library only");
    lib_step.dependOn(&b.addInstallArtifact(libscip, .{}).step);

    // ========================================================================
    // Server Executable: ncode-server
    // ========================================================================
    const server_mod = b.createModule(.{
        .root_source_file = b.path("server/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_mod.addImport("scip_reader", scip_reader_mod);

    const server_exe = b.addExecutable(.{
        .name = "ncode-server",
        .root_module = server_mod,
    });

    // Install the server executable
    b.installArtifact(server_exe);

    // "server" step - builds and installs server only
    const server_step = b.step("server", "Build the nCode server only");
    server_step.dependOn(&b.addInstallArtifact(server_exe, .{}).step);

    // "run" step - runs the server
    const run_cmd = b.addRunArtifact(server_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    // Allow passing arguments to the server
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the nCode server");
    run_step.dependOn(&run_cmd.step);

    // ========================================================================
    // Test Targets
    // ========================================================================

    // test-scip-writer: tests for SCIP writing functionality
    const test_scip_writer = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("zig_scip_writer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_writer = b.addRunArtifact(test_scip_writer);
    const test_writer_step = b.step("test-scip-writer", "Run SCIP writer tests");
    test_writer_step.dependOn(&run_test_writer.step);

    // test-scip-reader: tests for SCIP parsing functionality
    const test_scip_reader = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("scip_reader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_reader = b.addRunArtifact(test_scip_reader);
    const test_reader_step = b.step("test-scip-reader", "Run SCIP reader tests");
    test_reader_step.dependOn(&run_test_reader.step);

    // test-server: integration tests for the server
    const server_test_mod = b.createModule(.{
        .root_source_file = b.path("server/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_test_mod.addImport("scip_reader", scip_reader_mod);

    const test_server = b.addTest(.{
        .root_module = server_test_mod,
    });

    const run_test_server = b.addRunArtifact(test_server);
    const test_server_step = b.step("test-server", "Run server integration tests");
    test_server_step.dependOn(&run_test_server.step);

    // "test" step - runs all tests
    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_test_writer.step);
    test_step.dependOn(&run_test_reader.step);
    test_step.dependOn(&run_test_server.step);

    // ========================================================================
    // API Integration Test Executable
    // ========================================================================
    const api_test_exe = b.addExecutable(.{
        .name = "api-test",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/api_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(api_test_exe);

    const api_test_step = b.step("api-test", "Build the API integration test executable");
    api_test_step.dependOn(&b.addInstallArtifact(api_test_exe, .{}).step);

    // ========================================================================
    // Tree-sitter Indexer Executable: ncode-treesitter
    // For indexing data languages (JSON, XML, YAML, SQL, etc.)
    // ========================================================================
    const treesitter_mod = b.createModule(.{
        .root_source_file = b.path("treesitter_indexer.zig"),
        .target = target,
        .optimize = optimize,
    });
    treesitter_mod.addImport("zig_scip_writer", scip_writer_mod);

    const treesitter_exe = b.addExecutable(.{
        .name = "ncode-treesitter",
        .root_module = treesitter_mod,
    });
    b.installArtifact(treesitter_exe);

    const treesitter_step = b.step("treesitter", "Build the tree-sitter indexer for data languages");
    treesitter_step.dependOn(&b.addInstallArtifact(treesitter_exe, .{}).step);

    // ========================================================================
    // Default Build (zig build) - builds everything
    // ========================================================================
    b.default_step.dependOn(&libscip.step);
    b.default_step.dependOn(&server_exe.step);
    b.default_step.dependOn(&treesitter_exe.step);
}

