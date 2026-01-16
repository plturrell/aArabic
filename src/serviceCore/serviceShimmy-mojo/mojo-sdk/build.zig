// Mojo SDK - Build System
// Day 1: Lexer build configuration

const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ========================================================================
    // Modules
    // ========================================================================

    // Lexer module
    const lexer_module = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/lexer.zig"),
    });

    // AST module
    const ast_module = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/ast.zig"),
    });
    ast_module.addImport("lexer", lexer_module);

    // Parser module
    const parser_module = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/parser.zig"),
    });
    parser_module.addImport("lexer", lexer_module);
    parser_module.addImport("ast", ast_module);

    // Symbol Table module
    const symbol_table_module = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/symbol_table.zig"),
    });
    symbol_table_module.addImport("ast", ast_module);

    // Semantic Analyzer module
    const semantic_analyzer_module = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/semantic_analyzer.zig"),
    });
    semantic_analyzer_module.addImport("ast", ast_module);
    semantic_analyzer_module.addImport("symbol_table", symbol_table_module);

    // IR module
    const ir_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/ir.zig"),
    });

    // IR Builder module
    const ir_builder_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/ir_builder.zig"),
    });
    ir_builder_module.addImport("ast", ast_module);
    ir_builder_module.addImport("ir", ir_module);
    ir_builder_module.addImport("symbol_table", symbol_table_module);

    // Optimizer module
    const optimizer_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/optimizer.zig"),
    });
    optimizer_module.addImport("ir", ir_module);

    // SIMD module
    const simd_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/simd.zig"),
    });
    simd_module.addImport("ir", ir_module);

    // MLIR Setup module (Day 11)
    const mlir_setup_module = b.createModule(.{
        .root_source_file = b.path("compiler/middle/mlir_setup.zig"),
    });
    const default_mlir_include = if (target.result.os.tag == .macos)
        "/opt/homebrew/opt/llvm/include"
    else
        "";
    const default_mlir_lib = if (target.result.os.tag == .macos)
        "/opt/homebrew/opt/llvm/lib"
    else
        "";

    const mlir_include = b.option(
        []const u8,
        "mlir-include",
        "Path to MLIR headers",
    ) orelse default_mlir_include;
    const mlir_lib = b.option(
        []const u8,
        "mlir-lib",
        "Path to MLIR libraries",
    ) orelse default_mlir_lib;

    if (mlir_include.len > 0) {
        mlir_setup_module.addIncludePath(.{ .cwd_relative = mlir_include });
    }
    if (mlir_lib.len > 0) {
        mlir_setup_module.addLibraryPath(.{ .cwd_relative = mlir_lib });
    }

    // Mojo Dialect module (Day 12)
    const mojo_dialect_module = b.createModule(.{
        .root_source_file = b.path("compiler/middle/mojo_dialect.zig"),
    });
    mojo_dialect_module.addImport("mlir_setup", mlir_setup_module);

    // IR to MLIR module (Day 13)
    const ir_to_mlir_module = b.createModule(.{
        .root_source_file = b.path("compiler/middle/ir_to_mlir.zig"),
    });
    ir_to_mlir_module.addImport("ir", ir_module);
    ir_to_mlir_module.addImport("mlir_setup", mlir_setup_module);
    ir_to_mlir_module.addImport("mojo_dialect", mojo_dialect_module);

    // MLIR Optimizer module (Day 14)
    const mlir_optimizer_module = b.createModule(.{
        .root_source_file = b.path("compiler/middle/mlir_optimizer.zig"),
    });
    mlir_optimizer_module.addImport("mlir_setup", mlir_setup_module);
    mlir_optimizer_module.addImport("mojo_dialect", mojo_dialect_module);
    mlir_optimizer_module.addImport("ir_to_mlir", ir_to_mlir_module);

    // LLVM Lowering module (Day 15)
    const llvm_lowering_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/llvm_lowering.zig"),
    });
    llvm_lowering_module.addImport("mlir_setup", mlir_setup_module);
    llvm_lowering_module.addImport("mojo_dialect", mojo_dialect_module);
    llvm_lowering_module.addImport("ir_to_mlir", ir_to_mlir_module);

    // Code Generation module (Day 16)
    const codegen_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/codegen.zig"),
    });
    codegen_module.addImport("llvm_lowering", llvm_lowering_module);

    // Native Compiler module (Day 17)
    const native_compiler_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/native_compiler.zig"),
    });
    native_compiler_module.addImport("llvm_lowering", llvm_lowering_module);
    native_compiler_module.addImport("codegen", codegen_module);

    // Tool Executor module (Day 18)
    const tool_executor_module = b.createModule(.{
        .root_source_file = b.path("compiler/backend/tool_executor.zig"),
    });
    tool_executor_module.addImport("native_compiler", native_compiler_module);

    // Compiler Driver module (Day 19)
    const driver_module = b.createModule(.{
        .root_source_file = b.path("compiler/driver.zig"),
    });
    driver_module.addImport("lexer", lexer_module);
    driver_module.addImport("parser", parser_module);
    driver_module.addImport("ast", ast_module);
    driver_module.addImport("semantic_analyzer", semantic_analyzer_module);
    driver_module.addImport("symbol_table", symbol_table_module);
    driver_module.addImport("ir", ir_module);
    driver_module.addImport("ir_builder", ir_builder_module);
    driver_module.addImport("optimizer", optimizer_module);
    driver_module.addImport("mlir_setup", mlir_setup_module);
    driver_module.addImport("ir_to_mlir", ir_to_mlir_module);
    driver_module.addImport("mlir_optimizer", mlir_optimizer_module);
    driver_module.addImport("llvm_lowering", llvm_lowering_module);
    driver_module.addImport("codegen", codegen_module);
    driver_module.addImport("native_compiler", native_compiler_module);
    driver_module.addImport("tool_executor", tool_executor_module);

    // Advanced Compilation module (Day 20)
    const advanced_module = b.createModule(.{
        .root_source_file = b.path("compiler/advanced.zig"),
    });
    advanced_module.addImport("driver", driver_module);

    // Testing & QA module (Day 21)
    const testing_module = b.createModule(.{
        .root_source_file = b.path("compiler/testing.zig"),
    });
    testing_module.addImport("driver", driver_module);
    testing_module.addImport("advanced", advanced_module);

    // Enhanced Type System module (Day 22)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/types.zig"),
    });

    // Pattern Matching module (Day 23)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/pattern.zig"),
    });

    // Trait System module (Day 24)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/traits.zig"),
    });

    // Advanced Generics module (Day 25)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/generics.zig"),
    });

    // Memory Management module (Day 26)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/memory.zig"),
    });

    // Error Handling module (Day 27)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/errors.zig"),
    });

    // Metaprogramming module (Day 28)
    _ = b.createModule(.{
        .root_source_file = b.path("compiler/frontend/metaprogramming.zig"),
    });

    // ========================================================================
    // Runtime Library (Day 30 - Priority Fix)
    // ========================================================================

    // Runtime Core module - Memory allocator, reference counting
    const runtime_core_module = b.createModule(.{
        .root_source_file = b.path("runtime/core.zig"),
    });

    // Runtime Memory module - String, List, Dict, Set
    const runtime_memory_module = b.createModule(.{
        .root_source_file = b.path("runtime/memory.zig"),
    });
    runtime_memory_module.addImport("core", runtime_core_module);

    // Runtime FFI module - C interop bridge
    const runtime_ffi_module = b.createModule(.{
        .root_source_file = b.path("runtime/ffi.zig"),
    });
    runtime_ffi_module.addImport("core", runtime_core_module);
    runtime_ffi_module.addImport("memory", runtime_memory_module);

    // Runtime Startup module - Entry point
    const runtime_startup_module = b.createModule(.{
        .root_source_file = b.path("runtime/startup.zig"),
    });
    runtime_startup_module.addImport("core", runtime_core_module);
    runtime_startup_module.addImport("memory", runtime_memory_module);
    runtime_startup_module.addImport("ffi", runtime_ffi_module);

    // ========================================================================
    // CLI Tool Executable (Days 22-24 Catchup)
    // ========================================================================

    // const cli_exe = b.addExecutable(.{
    //     .name = "mojo",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("tools/cli/main.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //     }),
    // });

    // b.installArtifact(cli_exe);

    // const run_cli = b.addRunArtifact(cli_exe);
    // if (b.args) |args| {
    //     run_cli.addArgs(args);
    // }

    // const run_cli_step = b.step("cli", "Run the Mojo CLI tool");
    // run_cli_step.dependOn(&run_cli.step);

    // ========================================================================
    // Tests
    // ========================================================================

    // Lexer tests
    const test_lexer = b.addTest(.{
        .name = "test_lexer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/tests/test_lexer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_lexer.root_module.addImport("lexer", lexer_module);

    const run_test_lexer = b.addRunArtifact(test_lexer);
    run_test_lexer.step.dependOn(b.getInstallStep());

    const test_lexer_step = b.step("test-lexer", "Run Day 1 tests (Lexer)");
    test_lexer_step.dependOn(&run_test_lexer.step);

    // ========================================================================
    // Combined Test
    // ========================================================================

    // Parser tests
    const test_parser = b.addTest(.{
        .name = "test_parser",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/tests/test_parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_parser.root_module.addImport("parser", parser_module);
    test_parser.root_module.addImport("lexer", lexer_module);
    test_parser.root_module.addImport("ast", ast_module);

    const run_test_parser = b.addRunArtifact(test_parser);
    run_test_parser.step.dependOn(b.getInstallStep());

    const test_parser_step = b.step("test-parser", "Run Day 2 tests (Parser)");
    test_parser_step.dependOn(&run_test_parser.step);

    // Symbol Table tests
    const test_symbol_table = b.addTest(.{
        .name = "test_symbol_table",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/symbol_table.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_symbol_table.root_module.addImport("ast", ast_module);

    const run_test_symbol_table = b.addRunArtifact(test_symbol_table);
    run_test_symbol_table.step.dependOn(b.getInstallStep());

    const test_symbol_table_step = b.step("test-symbol-table", "Run Day 5 tests (Symbol Table)");
    test_symbol_table_step.dependOn(&run_test_symbol_table.step);

    // Semantic Analyzer tests
    const test_semantic = b.addTest(.{
        .name = "test_semantic",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/semantic_analyzer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_semantic.root_module.addImport("ast", ast_module);
    test_semantic.root_module.addImport("symbol_table", symbol_table_module);

    const run_test_semantic = b.addRunArtifact(test_semantic);
    run_test_semantic.step.dependOn(b.getInstallStep());

    const test_semantic_step = b.step("test-semantic", "Run Day 6 tests (Semantic Analyzer)");
    test_semantic_step.dependOn(&run_test_semantic.step);

    // IR tests
    const test_ir = b.addTest(.{
        .name = "test_ir",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/ir.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_ir = b.addRunArtifact(test_ir);
    run_test_ir.step.dependOn(b.getInstallStep());

    const test_ir_step = b.step("test-ir", "Run Day 7 tests (IR)");
    test_ir_step.dependOn(&run_test_ir.step);

    // IR Builder tests
    const test_ir_builder = b.addTest(.{
        .name = "test_ir_builder",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/ir_builder.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_ir_builder.root_module.addImport("ast", ast_module);
    test_ir_builder.root_module.addImport("ir", ir_module);
    test_ir_builder.root_module.addImport("symbol_table", symbol_table_module);
    test_ir_builder.root_module.addImport("lexer", lexer_module);

    const run_test_ir_builder = b.addRunArtifact(test_ir_builder);
    run_test_ir_builder.step.dependOn(b.getInstallStep());

    const test_ir_builder_step = b.step("test-ir-builder", "Run Day 8 tests (IR Builder)");
    test_ir_builder_step.dependOn(&run_test_ir_builder.step);

    // Optimizer tests
    const test_optimizer = b.addTest(.{
        .name = "test_optimizer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/optimizer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_optimizer.root_module.addImport("ir", ir_module);

    const run_test_optimizer = b.addRunArtifact(test_optimizer);
    run_test_optimizer.step.dependOn(b.getInstallStep());

    const test_optimizer_step = b.step("test-optimizer", "Run Day 9 tests (Optimizer)");
    test_optimizer_step.dependOn(&run_test_optimizer.step);

    // SIMD tests
    const test_simd = b.addTest(.{
        .name = "test_simd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/simd.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_simd.root_module.addImport("ir", ir_module);

    const run_test_simd = b.addRunArtifact(test_simd);
    run_test_simd.step.dependOn(b.getInstallStep());

    const test_simd_step = b.step("test-simd", "Run Day 10 tests (SIMD)");
    test_simd_step.dependOn(&run_test_simd.step);

    // MLIR Setup tests
    const test_mlir_setup = b.addTest(.{
        .name = "test_mlir_setup",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/middle/mlir_setup.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    // Link MLIR libraries (using cwd_relative for absolute paths)
    test_mlir_setup.root_module.addIncludePath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/include" });
    test_mlir_setup.root_module.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/opt/llvm/lib" });
    test_mlir_setup.linkSystemLibrary("MLIR");
    test_mlir_setup.linkSystemLibrary("MLIRCAPIIR");
    test_mlir_setup.linkSystemLibrary("LLVMSupport");
    test_mlir_setup.linkSystemLibrary("LLVMDemangle");
    test_mlir_setup.linkSystemLibrary("c++");

    const run_test_mlir_setup = b.addRunArtifact(test_mlir_setup);
    run_test_mlir_setup.step.dependOn(b.getInstallStep());

    const test_mlir_setup_step = b.step("test-mlir-setup", "Run Day 11 tests (MLIR Setup)");
    test_mlir_setup_step.dependOn(&run_test_mlir_setup.step);

    // Mojo Dialect tests
    const test_mojo_dialect = b.addTest(.{
        .name = "test_mojo_dialect",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/middle/mojo_dialect.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_mojo_dialect.root_module.addImport("mlir_setup", mlir_setup_module);

    const run_test_mojo_dialect = b.addRunArtifact(test_mojo_dialect);
    run_test_mojo_dialect.step.dependOn(b.getInstallStep());

    const test_mojo_dialect_step = b.step("test-mojo-dialect", "Run Day 12 tests (Mojo Dialect)");
    test_mojo_dialect_step.dependOn(&run_test_mojo_dialect.step);

    // IR to MLIR tests
    const test_ir_to_mlir = b.addTest(.{
        .name = "test_ir_to_mlir",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/middle/ir_to_mlir.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_ir_to_mlir.root_module.addImport("ir", ir_module);
    test_ir_to_mlir.root_module.addImport("mlir_setup", mlir_setup_module);
    test_ir_to_mlir.root_module.addImport("mojo_dialect", mojo_dialect_module);

    const run_test_ir_to_mlir = b.addRunArtifact(test_ir_to_mlir);
    run_test_ir_to_mlir.step.dependOn(b.getInstallStep());

    const test_ir_to_mlir_step = b.step("test-ir-to-mlir", "Run Day 13 tests (IR to MLIR)");
    test_ir_to_mlir_step.dependOn(&run_test_ir_to_mlir.step);

    // MLIR Optimizer tests
    const test_mlir_optimizer = b.addTest(.{
        .name = "test_mlir_optimizer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/middle/mlir_optimizer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_mlir_optimizer.root_module.addImport("mlir_setup", mlir_setup_module);
    test_mlir_optimizer.root_module.addImport("mojo_dialect", mojo_dialect_module);
    test_mlir_optimizer.root_module.addImport("ir_to_mlir", ir_to_mlir_module);

    const run_test_mlir_optimizer = b.addRunArtifact(test_mlir_optimizer);
    run_test_mlir_optimizer.step.dependOn(b.getInstallStep());

    const test_mlir_optimizer_step = b.step("test-mlir-optimizer", "Run Day 14 tests (MLIR Optimizer)");
    test_mlir_optimizer_step.dependOn(&run_test_mlir_optimizer.step);

    // LLVM Lowering tests
    const test_llvm_lowering = b.addTest(.{
        .name = "test_llvm_lowering",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/llvm_lowering.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_llvm_lowering.root_module.addImport("mlir_setup", mlir_setup_module);
    test_llvm_lowering.root_module.addImport("mojo_dialect", mojo_dialect_module);
    test_llvm_lowering.root_module.addImport("ir_to_mlir", ir_to_mlir_module);

    const run_test_llvm_lowering = b.addRunArtifact(test_llvm_lowering);
    run_test_llvm_lowering.step.dependOn(b.getInstallStep());

    const test_llvm_lowering_step = b.step("test-llvm-lowering", "Run Day 15 tests (LLVM Lowering)");
    test_llvm_lowering_step.dependOn(&run_test_llvm_lowering.step);

    // Code Generation tests
    const test_codegen = b.addTest(.{
        .name = "test_codegen",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/codegen.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_codegen.root_module.addImport("llvm_lowering", llvm_lowering_module);

    const run_test_codegen = b.addRunArtifact(test_codegen);
    run_test_codegen.step.dependOn(b.getInstallStep());

    const test_codegen_step = b.step("test-codegen", "Run Day 16 tests (Code Generation)");
    test_codegen_step.dependOn(&run_test_codegen.step);

    // Native Compiler tests
    const test_native_compiler = b.addTest(.{
        .name = "test_native_compiler",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/native_compiler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_native_compiler.root_module.addImport("llvm_lowering", llvm_lowering_module);
    test_native_compiler.root_module.addImport("codegen", codegen_module);

    const run_test_native_compiler = b.addRunArtifact(test_native_compiler);
    run_test_native_compiler.step.dependOn(b.getInstallStep());

    const test_native_compiler_step = b.step("test-native-compiler", "Run Day 17 tests (Native Compiler)");
    test_native_compiler_step.dependOn(&run_test_native_compiler.step);

    // Tool Executor tests
    const test_tool_executor = b.addTest(.{
        .name = "test_tool_executor",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/backend/tool_executor.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_tool_executor.root_module.addImport("native_compiler", native_compiler_module);

    const run_test_tool_executor = b.addRunArtifact(test_tool_executor);
    run_test_tool_executor.step.dependOn(b.getInstallStep());

    const test_tool_executor_step = b.step("test-tool-executor", "Run Day 18 tests (Tool Executor)");
    test_tool_executor_step.dependOn(&run_test_tool_executor.step);

    // Driver tests
    const test_driver = b.addTest(.{
        .name = "test_driver",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/driver.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_driver.root_module.addImport("lexer", lexer_module);
    test_driver.root_module.addImport("parser", parser_module);
    test_driver.root_module.addImport("ast", ast_module);
    test_driver.root_module.addImport("semantic_analyzer", semantic_analyzer_module);
    test_driver.root_module.addImport("symbol_table", symbol_table_module);
    test_driver.root_module.addImport("ir", ir_module);
    test_driver.root_module.addImport("ir_builder", ir_builder_module);
    test_driver.root_module.addImport("optimizer", optimizer_module);
    test_driver.root_module.addImport("mlir_setup", mlir_setup_module);
    test_driver.root_module.addImport("ir_to_mlir", ir_to_mlir_module);
    test_driver.root_module.addImport("mlir_optimizer", mlir_optimizer_module);
    test_driver.root_module.addImport("llvm_lowering", llvm_lowering_module);
    test_driver.root_module.addImport("codegen", codegen_module);
    test_driver.root_module.addImport("native_compiler", native_compiler_module);
    test_driver.root_module.addImport("tool_executor", tool_executor_module);

    const run_test_driver = b.addRunArtifact(test_driver);
    run_test_driver.step.dependOn(b.getInstallStep());

    const test_driver_step = b.step("test-driver", "Run Day 19 tests (Compiler Driver)");
    test_driver_step.dependOn(&run_test_driver.step);

    // Advanced Compilation tests
    const test_advanced = b.addTest(.{
        .name = "test_advanced",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/advanced.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_advanced.root_module.addImport("driver", driver_module);

    const run_test_advanced = b.addRunArtifact(test_advanced);
    run_test_advanced.step.dependOn(b.getInstallStep());

    const test_advanced_step = b.step("test-advanced", "Run Day 20 tests (Advanced Compilation)");
    test_advanced_step.dependOn(&run_test_advanced.step);

    // Testing & QA tests
    const test_testing = b.addTest(.{
        .name = "test_testing",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/testing.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_testing.root_module.addImport("driver", driver_module);
    test_testing.root_module.addImport("advanced", advanced_module);

    const run_test_testing = b.addRunArtifact(test_testing);
    run_test_testing.step.dependOn(b.getInstallStep());

    const test_testing_step = b.step("test-testing", "Run Day 21 tests (Testing & QA)");
    test_testing_step.dependOn(&run_test_testing.step);

    // Type System tests
    const test_types = b.addTest(.{
        .name = "test_types",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/types.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_types = b.addRunArtifact(test_types);
    run_test_types.step.dependOn(b.getInstallStep());

    const test_types_step = b.step("test-types", "Run Day 22 tests (Type System)");
    test_types_step.dependOn(&run_test_types.step);

    // Pattern Matching tests
    const test_pattern = b.addTest(.{
        .name = "test_pattern",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/pattern.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_pattern = b.addRunArtifact(test_pattern);
    run_test_pattern.step.dependOn(b.getInstallStep());

    const test_pattern_step = b.step("test-pattern", "Run Day 23 tests (Pattern Matching)");
    test_pattern_step.dependOn(&run_test_pattern.step);

    // Trait System tests
    const test_traits = b.addTest(.{
        .name = "test_traits",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/traits.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_traits = b.addRunArtifact(test_traits);
    run_test_traits.step.dependOn(b.getInstallStep());

    const test_traits_step = b.step("test-traits", "Run Day 24 tests (Trait System)");
    test_traits_step.dependOn(&run_test_traits.step);

    // Advanced Generics tests
    const test_generics = b.addTest(.{
        .name = "test_generics",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/generics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_generics = b.addRunArtifact(test_generics);
    run_test_generics.step.dependOn(b.getInstallStep());

    const test_generics_step = b.step("test-generics", "Run Day 25 tests (Advanced Generics)");
    test_generics_step.dependOn(&run_test_generics.step);

    // Memory Management tests
    const test_memory = b.addTest(.{
        .name = "test_memory",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/memory.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_memory = b.addRunArtifact(test_memory);
    run_test_memory.step.dependOn(b.getInstallStep());

    const test_memory_step = b.step("test-memory", "Run Day 26 tests (Memory Management)");
    test_memory_step.dependOn(&run_test_memory.step);

    // Error Handling tests
    const test_errors = b.addTest(.{
        .name = "test_errors",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/errors.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_errors = b.addRunArtifact(test_errors);
    run_test_errors.step.dependOn(b.getInstallStep());

    const test_errors_step = b.step("test-errors", "Run Day 27 tests (Error Handling)");
    test_errors_step.dependOn(&run_test_errors.step);

    // Metaprogramming tests
    const test_metaprogramming = b.addTest(.{
        .name = "test_metaprogramming",
        .root_module = b.createModule(.{
            .root_source_file = b.path("compiler/frontend/metaprogramming.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_metaprogramming = b.addRunArtifact(test_metaprogramming);
    run_test_metaprogramming.step.dependOn(b.getInstallStep());

    const test_metaprogramming_step = b.step("test-metaprogramming", "Run Day 28 tests (Metaprogramming)");
    test_metaprogramming_step.dependOn(&run_test_metaprogramming.step);

    // ========================================================================
    // Runtime Tests (Day 30)
    // ========================================================================

    // Runtime Core tests
    const test_runtime_core = b.addTest(.{
        .name = "test_runtime_core",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runtime/core.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_runtime_core = b.addRunArtifact(test_runtime_core);

    const test_runtime_core_step = b.step("test-runtime-core", "Run Day 30 tests (Runtime Core)");
    test_runtime_core_step.dependOn(&run_test_runtime_core.step);

    // Runtime Memory tests
    const test_runtime_memory = b.addTest(.{
        .name = "test_runtime_memory",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runtime/memory.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_runtime_memory.root_module.addImport("core", runtime_core_module);

    const run_test_runtime_memory = b.addRunArtifact(test_runtime_memory);

    const test_runtime_memory_step = b.step("test-runtime-memory", "Run Day 30 tests (Runtime Memory)");
    test_runtime_memory_step.dependOn(&run_test_runtime_memory.step);

    // Runtime FFI tests
    const test_runtime_ffi = b.addTest(.{
        .name = "test_runtime_ffi",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runtime/ffi.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_runtime_ffi.root_module.addImport("core", runtime_core_module);
    test_runtime_ffi.root_module.addImport("memory", runtime_memory_module);

    const run_test_runtime_ffi = b.addRunArtifact(test_runtime_ffi);

    const test_runtime_ffi_step = b.step("test-runtime-ffi", "Run Day 30 tests (Runtime FFI)");
    test_runtime_ffi_step.dependOn(&run_test_runtime_ffi.step);

    // Runtime Startup tests
    const test_runtime_startup = b.addTest(.{
        .name = "test_runtime_startup",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runtime/startup.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_runtime_startup.root_module.addImport("core", runtime_core_module);
    test_runtime_startup.root_module.addImport("memory", runtime_memory_module);
    test_runtime_startup.root_module.addImport("ffi", runtime_ffi_module);

    const run_test_runtime_startup = b.addRunArtifact(test_runtime_startup);

    const test_runtime_startup_step = b.step("test-runtime-startup", "Run Day 30 tests (Runtime Startup)");
    test_runtime_startup_step.dependOn(&run_test_runtime_startup.step);

    // Combined runtime test step (without integration - that runs separately)
    const test_runtime_step = b.step("test-runtime", "Run all Day 30 runtime tests");
    test_runtime_step.dependOn(&run_test_runtime_core.step);
    test_runtime_step.dependOn(&run_test_runtime_memory.step);
    test_runtime_step.dependOn(&run_test_runtime_ffi.step);
    test_runtime_step.dependOn(&run_test_runtime_startup.step);

    // Runtime unified module (for integration tests and external use)
    const runtime_module = b.createModule(.{
        .root_source_file = b.path("runtime/mod.zig"),
    });

    // Runtime Integration test
    const test_runtime_integration = b.addTest(.{
        .name = "test_runtime_integration",
        .root_module = b.createModule(.{
            .root_source_file = b.path("runtime/tests/integration_test.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_runtime_integration.root_module.addImport("runtime", runtime_module);

    const run_test_runtime_integration = b.addRunArtifact(test_runtime_integration);

    const test_runtime_integration_step = b.step("test-runtime-integration", "Run Day 30 integration tests");
    test_runtime_integration_step.dependOn(&run_test_runtime_integration.step);

    // ========================================================================
    // Stdlib Tests (Days 29-37)
    // ========================================================================

    const test_stdlib = b.addTest(.{
        .name = "test_stdlib",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/stdlib_tests.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });

    const run_test_stdlib = b.addRunArtifact(test_stdlib);

    const test_stdlib_step = b.step("test-stdlib", "Run stdlib tests (Days 29-37)");
    test_stdlib_step.dependOn(&run_test_stdlib.step);

    // ========================================================================
    // Combined Test
    // ========================================================================

    // ========================================================================
    // Fuzzing Tools (Day 110)
    // ========================================================================

    const fuzz_parser_exe = b.addExecutable(.{
        .name = "fuzz-parser",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fuzz/fuzz_parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    fuzz_parser_exe.root_module.addImport("parser", parser_module);
    fuzz_parser_exe.root_module.addImport("lexer", lexer_module);

    b.installArtifact(fuzz_parser_exe);

    const run_fuzz_parser = b.addRunArtifact(fuzz_parser_exe);
    if (b.args) |args| {
        run_fuzz_parser.addArgs(args);
    }

    const fuzz_parser_step = b.step("fuzz-parser", "Run the parser fuzzing driver (dummy mode)");
    fuzz_parser_step.dependOn(&run_fuzz_parser.step);

    // ========================================================================
    // LSP Server (Day 113)
    // ========================================================================

    const lsp_server_exe = b.addExecutable(.{
        .name = "mojo-lsp",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/lsp/lsp_server.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(lsp_server_exe);

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_test_lexer.step);
    test_step.dependOn(&run_test_parser.step);
    test_step.dependOn(&run_test_symbol_table.step);
    test_step.dependOn(&run_test_semantic.step);
    test_step.dependOn(&run_test_ir.step);
    test_step.dependOn(&run_test_ir_builder.step);
    test_step.dependOn(&run_test_optimizer.step);
    test_step.dependOn(&run_test_simd.step);
    test_step.dependOn(&run_test_mlir_setup.step);
    test_step.dependOn(&run_test_mojo_dialect.step);
    test_step.dependOn(&run_test_ir_to_mlir.step);
    test_step.dependOn(&run_test_mlir_optimizer.step);
    test_step.dependOn(&run_test_llvm_lowering.step);
    test_step.dependOn(&run_test_codegen.step);
    test_step.dependOn(&run_test_native_compiler.step);
    test_step.dependOn(&run_test_tool_executor.step);
    test_step.dependOn(&run_test_driver.step);
    test_step.dependOn(&run_test_advanced.step);
    test_step.dependOn(&run_test_testing.step);
    test_step.dependOn(&run_test_types.step);
    test_step.dependOn(&run_test_pattern.step);
    test_step.dependOn(&run_test_traits.step);
    test_step.dependOn(&run_test_generics.step);
    test_step.dependOn(&run_test_memory.step);
    test_step.dependOn(&run_test_errors.step);
    test_step.dependOn(&run_test_metaprogramming.step);
    // Day 30 - Runtime tests
    test_step.dependOn(&run_test_runtime_core.step);
    test_step.dependOn(&run_test_runtime_memory.step);
    test_step.dependOn(&run_test_runtime_ffi.step);
    test_step.dependOn(&run_test_runtime_startup.step);
    test_step.dependOn(&run_test_runtime_integration.step);
    // Stdlib tests (Days 29-37)
    test_step.dependOn(&run_test_stdlib.step);

    // ========================================================================
    // Default build
    // ========================================================================

    const default_run_step = b.step("run", "Run lexer tests");
    default_run_step.dependOn(&run_test_lexer.step);
}
