const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{
        .preferred_optimize_mode = .ReleaseFast,
    });

    // Create build options
    const options = b.addOptions();
    options.addOption(bool, "enable_cuda", target.result.os.tag == .linux);
    const build_options_mod = options.createModule();

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
    
    // Note: matrix_ops needs gguf_loader, but we'll add it after both are created
    
    // MHC configuration module
    const mhc_configuration_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/mhc_configuration.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // MHC constraints module
    const mhc_constraints_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/mhc_constraints.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // GGUF MHC parser module - depends on gguf_loader
    const gguf_mhc_parser_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/gguf_mhc_parser.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_mhc_parser_mod.addImport("gguf_loader", gguf_loader_mod);
    
    gguf_loader_mod.addImport("mhc_constraints", mhc_constraints_mod);
    gguf_loader_mod.addImport("mhc_configuration", mhc_configuration_mod);
    gguf_loader_mod.addImport("gguf_mhc_parser", gguf_mhc_parser_mod);
    
    // Now add gguf_loader and thread_pool to matrix_ops (thread_pool defined later)
    matrix_ops_mod.addImport("gguf_loader", gguf_loader_mod);
    
    // Common module (quantization common)
    const common_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/quantization/common.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Quantization format modules - depend on common
    const q4_0_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/quantization/q4_0.zig"),
        .target = target,
        .optimize = optimize,
    });
    q4_0_mod.addImport("common", common_mod);
    
    const q4_k_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/quantization/q4_k.zig"),
        .target = target,
        .optimize = optimize,
    });
    q4_k_mod.addImport("common", common_mod);
    
    const q6_k_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/quantization/q6_k.zig"),
        .target = target,
        .optimize = optimize,
    });
    q6_k_mod.addImport("common", common_mod);
    
    const q8_0_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/quantization/q8_0.zig"),
        .target = target,
        .optimize = optimize,
    });
    q8_0_mod.addImport("common", common_mod);
    
    // KV cache module
    const kv_cache_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/kv_cache.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Tokenizer module - depends on gguf_loader
    const tokenizer_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/tokenization/tokenizer.zig"),
        .target = target,
        .optimize = optimize,
    });
    tokenizer_mod.addImport("gguf_loader", gguf_loader_mod);
    
    // Thread pool module
    const thread_pool_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/thread_pool.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Now that thread_pool is defined, add it to matrix_ops
    matrix_ops_mod.addImport("thread_pool", thread_pool_mod);
    
    // Compute module - depends on gguf_loader
    const compute_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/compute.zig"),
        .target = target,
        .optimize = optimize,
    });
    compute_mod.addImport("gguf_loader", gguf_loader_mod);
    
    // Backend modules - depend on compute and gguf_loader
    const backend_cpu_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/backend_cpu.zig"),
        .target = target,
        .optimize = optimize,
    });
    backend_cpu_mod.addImport("compute", compute_mod);
    backend_cpu_mod.addImport("thread_pool", thread_pool_mod);
    backend_cpu_mod.addImport("gguf_loader", gguf_loader_mod);
    backend_cpu_mod.addImport("matrix_ops", matrix_ops_mod);
    
    const backend_metal_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/backend_metal.zig"),
        .target = target,
        .optimize = optimize,
    });
    backend_metal_mod.addImport("compute", compute_mod);
    backend_metal_mod.addImport("gguf_loader", gguf_loader_mod);
    backend_metal_mod.addImport("matrix_ops", matrix_ops_mod);
    
    const backend_cuda_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/backend_cuda.zig"),
        .target = target,
        .optimize = optimize,
    });
    backend_cuda_mod.addImport("compute", compute_mod);
    backend_cuda_mod.addImport("gguf_loader", gguf_loader_mod);
    
    // Config parser module
    const config_parser_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/loader/config_parser.zig"),
        .target = target,
        .optimize = optimize,
    });
    
    // Attention module - depends on config_parser
    const attention_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/attention.zig"),
        .target = target,
        .optimize = optimize,
    });
    attention_mod.addImport("config_parser", config_parser_mod);
    
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
    llama_model_mod.addImport("tokenizer", tokenizer_mod);
    llama_model_mod.addImport("thread_pool", thread_pool_mod);
    llama_model_mod.addImport("compute", compute_mod);
    llama_model_mod.addImport("build_options", build_options_mod);
    llama_model_mod.addImport("backend_cpu", backend_cpu_mod);
    llama_model_mod.addImport("backend_metal", backend_metal_mod);
    llama_model_mod.addImport("backend_cuda", backend_cuda_mod);
    llama_model_mod.addImport("attention", attention_mod);
    
    // GGUF model loader - depends on llama_model, gguf_loader, tokenizer, and common
    const gguf_model_loader_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/loader/gguf_model_loader.zig"),
        .target = target,
        .optimize = optimize,
    });
    gguf_model_loader_mod.addImport("llama_model", llama_model_mod);
    gguf_model_loader_mod.addImport("gguf_loader", gguf_loader_mod);
    gguf_model_loader_mod.addImport("tokenizer", tokenizer_mod);
    gguf_model_loader_mod.addImport("transformer", transformer_mod);
    gguf_model_loader_mod.addImport("matrix_ops", matrix_ops_mod);
    gguf_model_loader_mod.addImport("common", common_mod);
    gguf_model_loader_mod.addImport("q4_0", q4_0_mod);
    gguf_model_loader_mod.addImport("q4_k", q4_k_mod);
    gguf_model_loader_mod.addImport("q6_k", q6_k_mod);
    gguf_model_loader_mod.addImport("q8_0", q8_0_mod);
    
    // Batch processor - depends on llama_model, kv_cache, and performance
    const batch_processor_mod = b.createModule(.{
        .root_source_file = b.path("inference/engine/core/batch_processor.zig"),
        .target = target,
        .optimize = optimize,
    });
    batch_processor_mod.addImport("llama_model", llama_model_mod);
    batch_processor_mod.addImport("kv_cache", kv_cache_mod);
    batch_processor_mod.addImport("matrix_ops", matrix_ops_mod);
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