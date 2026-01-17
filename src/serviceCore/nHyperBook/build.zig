const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ====================================================================
    // Server Executable (Main Entry Point)
    // ====================================================================
    
    // Create I/O modules
    const pdf_parser_mod = b.addModule("pdf_parser", .{
        .root_source_file = b.path("io/pdf_parser.zig"),
    });
    
    const html_parser_mod = b.addModule("html_parser", .{
        .root_source_file = b.path("io/html_parser.zig"),
    });
    
    const http_client_mod = b.addModule("http_client", .{
        .root_source_file = b.path("io/http_client.zig"),
    });
    
    // Create server module with dependencies
    const server_mod = b.createModule(.{
        .root_source_file = b.path("server/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    server_mod.addImport("pdf_parser", pdf_parser_mod);
    server_mod.addImport("html_parser", html_parser_mod);
    server_mod.addImport("http_client", http_client_mod);
    
    const server_exe = b.addExecutable(.{
        .name = "hypershimmy",
        .root_module = server_mod,
    });
    
    b.installArtifact(server_exe);

    // Create a run step
    const run_cmd = b.addRunArtifact(server_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run HyperShimmy server");
    run_step.dependOn(&run_cmd.step);

    // ====================================================================
    // Tests
    // ====================================================================
    
    // Server tests
    const server_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("server/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_server_tests = b.addRunArtifact(server_tests);
    
    // I/O tests
    const io_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("io/http_client.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_io_tests = b.addRunArtifact(io_tests);
    
    // HTML parser tests
    const html_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("io/html_parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_html_tests = b.addRunArtifact(html_tests);
    
    // Web scraper tests
    const scraper_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("io/web_scraper.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_scraper_tests = b.addRunArtifact(scraper_tests);
    
    // PDF parser tests
    const pdf_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("io/pdf_parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_pdf_tests = b.addRunArtifact(pdf_tests);

    // Unit tests for new modules
    const sources_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/test_sources.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_sources_tests = b.addRunArtifact(sources_tests);
    
    const security_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/test_security.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_security_tests = b.addRunArtifact(security_tests);
    
    const json_utils_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/unit/test_json_utils.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_json_utils_tests = b.addRunArtifact(json_utils_tests);

    // Integration tests
    const odata_integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/test_odata_endpoints.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_odata_integration_tests = b.addRunArtifact(odata_integration_tests);
    
    const upload_integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/test_file_upload_workflow.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_upload_integration_tests = b.addRunArtifact(upload_integration_tests);
    
    const ai_pipeline_integration_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/integration/test_ai_pipeline.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_ai_pipeline_integration_tests = b.addRunArtifact(ai_pipeline_integration_tests);

    // Test step (unit tests)
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_server_tests.step);
    test_step.dependOn(&run_io_tests.step);
    test_step.dependOn(&run_html_tests.step);
    test_step.dependOn(&run_scraper_tests.step);
    test_step.dependOn(&run_pdf_tests.step);
    test_step.dependOn(&run_sources_tests.step);
    test_step.dependOn(&run_security_tests.step);
    test_step.dependOn(&run_json_utils_tests.step);
    
    // Integration test step
    const integration_test_step = b.step("test-integration", "Run integration tests");
    integration_test_step.dependOn(&run_odata_integration_tests.step);
    integration_test_step.dependOn(&run_upload_integration_tests.step);
    integration_test_step.dependOn(&run_ai_pipeline_integration_tests.step);
    
    // All tests step
    const all_tests_step = b.step("test-all", "Run all tests (unit + integration)");
    all_tests_step.dependOn(test_step);
    all_tests_step.dependOn(integration_test_step);
}
