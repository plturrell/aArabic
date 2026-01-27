const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Add calculation engine modules (now in backend/models/calculation)
    const balance_engine_mod = b.addModule("balance_engine", .{
        .root_source_file = b.path("models/calculation/balance_engine.zig"),
    });

    const fx_converter_mod = b.addModule("fx_converter", .{
        .root_source_file = b.path("models/calculation/fx_converter.zig"),
    });

    const sqlite_adapter_mod = b.addModule("sqlite_adapter", .{
        .root_source_file = b.path("models/calculation/sqlite_adapter.zig"),
    });
    
    const websocket_mod = b.addModule("websocket", .{
        .root_source_file = b.path("websocket.zig"),
    });

    // Add data model modules (Day 1 implementation)
    const trial_balance_models_mod = b.addModule("trial_balance_models", .{
        .root_source_file = b.path("src/models/trial_balance_models.zig"),
    });

    const data_quality_mod = b.addModule("data_quality", .{
        .root_source_file = b.path("src/models/data_quality.zig"),
    });

    const csv_loader_mod = b.addModule("csv_loader", .{
        .root_source_file = b.path("src/data/csv_loader.zig"),
    });
    csv_loader_mod.addImport("trial_balance_models", trial_balance_models_mod);
    csv_loader_mod.addImport("data_quality", data_quality_mod);

    // Add ACDOCA table module (Day 1 - Phase 5)
    const acdoca_table_mod = b.addModule("acdoca_table", .{
        .root_source_file = b.path("src/data/acdoca_table.zig"),
    });
    acdoca_table_mod.addImport("trial_balance_models", trial_balance_models_mod);
    acdoca_table_mod.addImport("data_quality", data_quality_mod);

    // Add ODPS mapper module (Day 1 - Phase 4)
    const yaml_parser_mod = b.addModule("yaml_parser", .{
        .root_source_file = b.path("src/metadata/yaml_parser.zig"),
    });

    const odps_mapper_mod = b.addModule("odps_mapper", .{
        .root_source_file = b.path("src/metadata/odps_mapper.zig"),
    });
    odps_mapper_mod.addImport("yaml_parser", yaml_parser_mod);

    // ODPS-Petri Net Bridge
    const odps_petrinet_bridge_mod = b.addModule("odps_petrinet_bridge", .{
        .root_source_file = b.path("src/workflow/odps_petrinet_bridge.zig"),
    });
    odps_petrinet_bridge_mod.addImport("odps_mapper", odps_mapper_mod);

    // ODPS Quality Service
    const odps_quality_service_mod = b.addModule("odps_quality_service", .{
        .root_source_file = b.path("src/services/odps_quality_service.zig"),
    });
    odps_quality_service_mod.addImport("odps_mapper", odps_mapper_mod);
    odps_quality_service_mod.addImport("data_quality", data_quality_mod);
    odps_quality_service_mod.addImport("acdoca_table", acdoca_table_mod);

    // ODPS REST API
    const odps_api_mod = b.addModule("odps_api", .{
        .root_source_file = b.path("src/api/odps_api.zig"),
    });
    odps_api_mod.addImport("odps_mapper", odps_mapper_mod);
    odps_api_mod.addImport("odps_quality_service", odps_quality_service_mod);

    // Main server executable
    const exe = b.addExecutable(.{
        .name = "trial-balance-server",
        .root_module = b.createModule(.{
            .root_source_file = b.path("server.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    // Add module imports
    exe.root_module.addImport("balance_engine", balance_engine_mod);
    exe.root_module.addImport("fx_converter", fx_converter_mod);
    exe.root_module.addImport("sqlite_adapter", sqlite_adapter_mod);
    exe.root_module.addImport("websocket", websocket_mod);

    // Link SQLite
    exe.linkSystemLibrary("sqlite3");
    exe.linkLibC();

    b.installArtifact(exe);

    // Run command
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the Trial Balance server");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("server.zig"),
        .target = target,
        .optimize = optimize,
    });
    test_mod.addImport("balance_engine", balance_engine_mod);
    test_mod.addImport("fx_converter", fx_converter_mod);
    test_mod.addImport("sqlite_adapter", sqlite_adapter_mod);

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });

    unit_tests.linkSystemLibrary("sqlite3");
    unit_tests.linkLibC();

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Day 1 CSV Loader Tests
    const csv_test_mod = b.createModule(.{
        .root_source_file = b.path("src/data/test_csv_loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    csv_test_mod.addImport("csv_loader", csv_loader_mod);
    csv_test_mod.addImport("trial_balance_models", trial_balance_models_mod);
    csv_test_mod.addImport("data_quality", data_quality_mod);

    const csv_tests = b.addTest(.{
        .root_module = csv_test_mod,
    });

    const run_csv_tests = b.addRunArtifact(csv_tests);
    const csv_test_step = b.step("test-csv", "Run CSV loader tests");
    csv_test_step.dependOn(&run_csv_tests.step);

    // Data quality enum tests
    const quality_test_mod = b.createModule(.{
        .root_source_file = b.path("src/models/data_quality.zig"),
        .target = target,
        .optimize = optimize,
    });

    const quality_tests = b.addTest(.{
        .root_module = quality_test_mod,
    });

    const run_quality_tests = b.addRunArtifact(quality_tests);
    const quality_test_step = b.step("test-quality", "Run data quality enum tests");
    quality_test_step.dependOn(&run_quality_tests.step);

    // CSV loader unit tests
    const loader_test_mod = b.createModule(.{
        .root_source_file = b.path("src/data/csv_loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    loader_test_mod.addImport("trial_balance_models", trial_balance_models_mod);
    loader_test_mod.addImport("data_quality", data_quality_mod);

    const loader_tests = b.addTest(.{
        .root_module = loader_test_mod,
    });

    const run_loader_tests = b.addRunArtifact(loader_tests);
    const loader_test_step = b.step("test-loader", "Run CSV loader unit tests");
    loader_test_step.dependOn(&run_loader_tests.step);

    // ODPS Integration Tests
    const odps_test_mod = b.createModule(.{
        .root_source_file = b.path("src/tests/test_odps_integration.zig"),
        .target = target,
        .optimize = optimize,
    });
    // Only import what the test needs - not the sub-dependencies
    odps_test_mod.addImport("yaml_parser", yaml_parser_mod);
    odps_test_mod.addImport("odps_mapper", odps_mapper_mod);
    odps_test_mod.addImport("odps_petrinet_bridge", odps_petrinet_bridge_mod);
    odps_test_mod.addImport("odps_quality_service", odps_quality_service_mod);
    odps_test_mod.addImport("odps_api", odps_api_mod);
    // Don't add sub-dependencies that are already in the modules

    const odps_tests = b.addTest(.{
        .root_module = odps_test_mod,
    });

    const run_odps_tests = b.addRunArtifact(odps_tests);
    const odps_test_step = b.step("test-odps", "Run ODPS integration tests");
    odps_test_step.dependOn(&run_odps_tests.step);

    // Day 1 test suite - runs all Day 1 tests
    const day1_test_step = b.step("test-day1", "Run all Day 1 tests");
    day1_test_step.dependOn(&run_quality_tests.step);
    day1_test_step.dependOn(&run_loader_tests.step);
    day1_test_step.dependOn(&run_csv_tests.step);
    day1_test_step.dependOn(&run_odps_tests.step);
    
    // Complete test suite - runs all tests
    const all_tests_step = b.step("test-all", "Run complete test suite");
    all_tests_step.dependOn(&run_unit_tests.step);
    all_tests_step.dependOn(&run_csv_tests.step);
    all_tests_step.dependOn(&run_quality_tests.step);
    all_tests_step.dependOn(&run_loader_tests.step);
    all_tests_step.dependOn(&run_odps_tests.step);
}
