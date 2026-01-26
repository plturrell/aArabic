const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Create base modules first (no dependencies)
    const performance_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/performance.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const matrix_ops_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/matrix_ops.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    const gguf_loader_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/gguf_loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // KV cache module
    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Transformer module - depends on matrix_ops
    const transformer_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/transformer.zig"),
        .target = target,
        .optimize = optimize,
    });
    transformer_mod.addImport("matrix_ops", matrix_ops_mod);
    
    // Llama model - depends on gguf_loader, transformer, kv_cache, etc.
    const llama_model_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/llama_model.zig"),
        .target = target,
        .optimize = optimize,
    });
    llama_model_mod.addImport("gguf_loader", gguf_loader_mod);
    llama_model_mod.addImport("matrix_ops", matrix_ops_mod);
    llama_model_mod.addImport("performance", performance_mod);
    llama_model_mod.addImport("transformer", transformer_mod);
    llama_model_mod.addImport("kv_cache", kv_cache_mod);
    
    // GGUF model loader - depends on llama_model and gguf_loader
    const gguf_model_loader_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/loader/gguf_model_loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_model_loader_mod.addImport("llama_model", llama_model_mod);
    gguf_model_loader_mod.addImport("gguf_loader", gguf_loader_mod);
    
    // Batch processor - depends on llama_model
    const batch_processor_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/batch_processor.zig"),
        .target = target,
        .optimize = optimize,
    });
    batch_processor_mod.addImport("llama_model", llama_model_mod);
    batch_processor_mod.addImport("performance", performance_mod);
    
    // Sampler module
    const sampler_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/sampling/sampler.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Main server module - depends on all of the above
    const server_module = b.createModule(.{
        .root_source_file = b.path("inference/engine/cli/main.zig"),
        .target = target,
        .optimize = optimize,
    });
    server_module.addImport("gguf_loader", gguf_loader_mod);
    server_module.addImport("llama_model", llama_model_mod);
    server_module.addImport("gguf_model_loader", gguf_model_loader_mod);
    server_module.addImport("batch_processor", batch_processor_mod);
    server_module.addImport("performance", performance_mod);
    server_module.addImport("sampler", sampler_mod);
    server_module.addImport("matrix_ops", matrix_ops_mod);
    
    const server_exe = b.addExecutable(.{
        .name = "nlocalmodels",
        .root_module = server_module,
    });

    // For macOS Metal support
    if (target.result.os.tag == .macos) {
        server_exe.linkFramework("Metal");
        server_exe.linkFramework("Foundation");
    }

    // For Linux CUDA support
    if (target.result.os.tag == .linux) {
        server_exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        server_exe.root_module.addLibraryPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
        server_exe.root_module.addRPath(.{ .cwd_relative = "/usr/local/cuda/lib64" });
        server_exe.root_module.addRPath(.{ .cwd_relative = "/usr/local/cuda/targets/x86_64-linux/lib" });
        server_exe.linkSystemLibrary("cuda");
        server_exe.linkSystemLibrary("cublas");
        server_exe.linkSystemLibrary("cudart");
    }

    b.installArtifact(server_exe);

    // Run command
    const run_cmd = b.addRunArtifact(server_exe);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the nLocalModels server");
    run_step.dependOn(&run_cmd.step);

    // Build orchestration tools as well
    const orchestration_build = b.addExecutable(.{
        .name = "orchestration-tools",
        .root_module = b.createModule(.{
            .root_source_file = b.path("orchestration/analytics.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    b.installArtifact(orchestration_build);

    // Test step - runs inference engine tests
    const test_step = b.step("test", "Run all tests");
    
    const tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("inference/engine/cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    
    const run_tests = b.addRunArtifact(tests);
    test_step.dependOn(&run_tests.step);
}