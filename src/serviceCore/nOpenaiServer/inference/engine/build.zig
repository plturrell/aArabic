const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // ========================================================================
    // Core Modules
    // ========================================================================

    // Thread Pool module (Day 14) - Moved up for core dependencies
    const thread_pool_module = b.createModule(.{
        .root_source_file = b.path("threading/thread_pool.zig"),
    });

    // mHC Configuration module (must come before modules that use it)
    const mhc_configuration_module = b.createModule(.{
        .root_source_file = b.path("core/mhc_configuration.zig"),
    });

    // mHC Constraints module (must come before modules that use it)
    const mhc_constraints_module = b.createModule(.{
        .root_source_file = b.path("core/mhc_constraints.zig"),
    });

    // GGUF loader module (without mhc_parser dependency initially)
    const gguf_module = b.createModule(.{
        .root_source_file = b.path("core/gguf_loader.zig"),
    });
    gguf_module.addImport("mhc_configuration", mhc_configuration_module);
    gguf_module.addImport("mhc_constraints", mhc_constraints_module);

    // GGUF mHC Parser module (internal helper for gguf_loader)
    const gguf_mhc_parser_module = b.createModule(.{
        .root_source_file = b.path("core/gguf_mhc_parser.zig"),
    });
    gguf_mhc_parser_module.addImport("mhc_constraints", mhc_constraints_module);
    gguf_mhc_parser_module.addImport("gguf_loader", gguf_module);

    // Add mhc_parser to gguf_loader now that it's defined
    gguf_module.addImport("gguf_mhc_parser", gguf_mhc_parser_module);

    // Quantization common module (must come first!)
    const quant_common_module = b.createModule(.{
        .root_source_file = b.path("quantization/common.zig"),
    });

    // Q4_0 quantization module
    const q4_0_module = b.createModule(.{
        .root_source_file = b.path("quantization/q4_0.zig"),
    });
    q4_0_module.addImport("common", quant_common_module);

    // Q4_K quantization module (Advanced)
    const q4_k_module = b.createModule(.{
        .root_source_file = b.path("quantization/q4_k.zig"),
    });
    q4_k_module.addImport("common", quant_common_module);

    // Q6_K quantization module
    const q6_k_module = b.createModule(.{
        .root_source_file = b.path("quantization/q6_k.zig"),
    });
    q6_k_module.addImport("common", quant_common_module);

    // Matrix operations module (now q4_k is defined)
    const matrix_ops_module = b.createModule(.{
        .root_source_file = b.path("core/matrix_ops.zig"),
    });
    matrix_ops_module.addImport("gguf_loader", gguf_module);
    matrix_ops_module.addImport("thread_pool", thread_pool_module);
    matrix_ops_module.addImport("q4_k", q4_k_module);
    matrix_ops_module.addImport("q6_k", q6_k_module);
    matrix_ops_module.addImport("mhc_configuration", mhc_configuration_module);
    matrix_ops_module.addImport("mhc_constraints", mhc_constraints_module);

    // Compute interface module (Step 4)
    const compute_module = b.createModule(.{
        .root_source_file = b.path("core/compute.zig"),
    });
    compute_module.addImport("gguf_loader", gguf_module);
    compute_module.addImport("thread_pool", thread_pool_module);

    // CPU Backend module (Step 4)
    const backend_cpu_module = b.createModule(.{
        .root_source_file = b.path("core/backend_cpu.zig"),
    });
    backend_cpu_module.addImport("compute", compute_module);
    backend_cpu_module.addImport("matrix_ops", matrix_ops_module);
    backend_cpu_module.addImport("thread_pool", thread_pool_module);
    backend_cpu_module.addImport("gguf_loader", gguf_module);

    // Metal Backend module (Step 5)
    const backend_metal_module = b.createModule(.{
        .root_source_file = b.path("core/backend_metal.zig"),
    });
    backend_metal_module.addImport("compute", compute_module);
    backend_metal_module.addImport("gguf_loader", gguf_module);
    backend_metal_module.addImport("matrix_ops", matrix_ops_module);

    // CUDA Bindings module (GPU Foundation)
    const cuda_bindings_module = b.createModule(.{
        .root_source_file = b.path("cuda/cuda_bindings.zig"),
    });

    // CUDA Memory module
    const cuda_memory_module = b.createModule(.{
        .root_source_file = b.path("cuda/cuda_memory.zig"),
    });
    cuda_memory_module.addImport("cuda_bindings", cuda_bindings_module);

    // CUDA Streams module
    const cuda_streams_module = b.createModule(.{
        .root_source_file = b.path("cuda/cuda_streams.zig"),
    });
    cuda_streams_module.addImport("cuda_bindings", cuda_bindings_module);

    // CUDA Context module
    const cuda_context_module = b.createModule(.{
        .root_source_file = b.path("cuda/cuda_context.zig"),
    });
    cuda_context_module.addImport("cuda_bindings", cuda_bindings_module);

    // cuBLAS Bindings module (Tensor Core acceleration)
    const cublas_bindings_module = b.createModule(.{
        .root_source_file = b.path("cuda/cublas_bindings.zig"),
    });

    // Dequantization Bindings module (GPU dequant for Tensor Core input)
    const dequant_bindings_module = b.createModule(.{
        .root_source_file = b.path("cuda/dequant_bindings.zig"),
    });
    dequant_bindings_module.addImport("cuda_bindings", cuda_bindings_module);
    dequant_bindings_module.addImport("gguf_loader", gguf_module);

    // CUDA Backend module (Step 6 - T4 GPU support)
    const backend_cuda_module = b.createModule(.{
        .root_source_file = b.path("core/backend_cuda.zig"),
    });
    backend_cuda_module.addImport("compute", compute_module);
    backend_cuda_module.addImport("gguf_loader", gguf_module);
    backend_cuda_module.addImport("matrix_ops", matrix_ops_module);
    backend_cuda_module.addImport("cuda_bindings", cuda_bindings_module);
    backend_cuda_module.addImport("cuda_memory", cuda_memory_module);
    backend_cuda_module.addImport("cuda_streams", cuda_streams_module);
    backend_cuda_module.addImport("cuda_context", cuda_context_module);
    backend_cuda_module.addImport("cublas_bindings", cublas_bindings_module);
    backend_cuda_module.addImport("dequant_bindings", dequant_bindings_module);

    // Q8_0 quantization module (Day 13)
    const q8_0_module = b.createModule(.{
        .root_source_file = b.path("quantization/q8_0.zig"),
    });
    q8_0_module.addImport("common", quant_common_module);

    // Tokenizer module
    const tokenizer_module = b.createModule(.{
        .root_source_file = b.path("tokenization/tokenizer.zig"),
    });
    tokenizer_module.addImport("gguf_loader", gguf_module);
    tokenizer_module.addImport("matrix_ops", matrix_ops_module);

    // KV Cache module
    const kv_cache_module = b.createModule(.{
        .root_source_file = b.path("core/kv_cache.zig"),
    });

    // Attention module
    const attention_module = b.createModule(.{
        .root_source_file = b.path("core/attention.zig"),
    });
    attention_module.addImport("matrix_ops", matrix_ops_module);
    attention_module.addImport("kv_cache", kv_cache_module);
    attention_module.addImport("thread_pool", thread_pool_module);

    // Feed-Forward module
    const feed_forward_module = b.createModule(.{
        .root_source_file = b.path("core/feed_forward.zig"),
    });
    feed_forward_module.addImport("matrix_ops", matrix_ops_module);
    feed_forward_module.addImport("thread_pool", thread_pool_module);

    // Transformer module
    const transformer_module = b.createModule(.{
        .root_source_file = b.path("core/transformer.zig"),
    });
    transformer_module.addImport("matrix_ops", matrix_ops_module);
    transformer_module.addImport("attention", attention_module);
    transformer_module.addImport("feed_forward", feed_forward_module);
    transformer_module.addImport("kv_cache", kv_cache_module);
    transformer_module.addImport("thread_pool", thread_pool_module);
    transformer_module.addImport("mhc_configuration", mhc_configuration_module);
    transformer_module.addImport("mhc_constraints", mhc_constraints_module);

    // Llama Model module
    const llama_model_module = b.createModule(.{
        .root_source_file = b.path("core/llama_model.zig"),
    });
    llama_model_module.addImport("gguf_loader", gguf_module);
    llama_model_module.addImport("transformer", transformer_module);
    llama_model_module.addImport("attention", attention_module);
    llama_model_module.addImport("tokenizer", tokenizer_module);
    llama_model_module.addImport("kv_cache", kv_cache_module);
    llama_model_module.addImport("matrix_ops", matrix_ops_module);
    llama_model_module.addImport("thread_pool", thread_pool_module);
    llama_model_module.addImport("compute", compute_module);
    llama_model_module.addImport("backend_cpu", backend_cpu_module);
    llama_model_module.addImport("backend_metal", backend_metal_module);
    llama_model_module.addImport("backend_cuda", backend_cuda_module);

    // LFM2 Model module
    const lfm2_model_module = b.createModule(.{
        .root_source_file = b.path("core/lfm2_model.zig"),
    });
    lfm2_model_module.addImport("gguf_loader", gguf_module);
    lfm2_model_module.addImport("tokenizer", tokenizer_module);
    lfm2_model_module.addImport("matrix_ops", matrix_ops_module);
    lfm2_model_module.addImport("attention", attention_module);
    lfm2_model_module.addImport("kv_cache", kv_cache_module);
    lfm2_model_module.addImport("thread_pool", thread_pool_module);

    // GGUF Model Loader module (Day 6)
    const gguf_model_loader_module = b.createModule(.{
        .root_source_file = b.path("loader/gguf_model_loader.zig"),
    });
    gguf_model_loader_module.addImport("gguf_loader", gguf_module);
    gguf_model_loader_module.addImport("llama_model", llama_model_module);
    gguf_model_loader_module.addImport("tokenizer", tokenizer_module);
    gguf_model_loader_module.addImport("transformer", transformer_module);
    gguf_model_loader_module.addImport("matrix_ops", matrix_ops_module);
    gguf_model_loader_module.addImport("q4_0", q4_0_module);
    gguf_model_loader_module.addImport("q4_k", q4_k_module);
    gguf_model_loader_module.addImport("q6_k", q6_k_module);
    gguf_model_loader_module.addImport("common", quant_common_module);
    gguf_model_loader_module.addImport("lfm2_model", lfm2_model_module);

    // Batch Processor module (Day 7)
    const batch_processor_module = b.createModule(.{
        .root_source_file = b.path("core/batch_processor.zig"),
    });
    batch_processor_module.addImport("llama_model", llama_model_module);
    batch_processor_module.addImport("transformer", transformer_module);
    batch_processor_module.addImport("matrix_ops", matrix_ops_module);
    batch_processor_module.addImport("kv_cache", kv_cache_module);

    // Performance module (Day 8)
    const performance_module = b.createModule(.{
        .root_source_file = b.path("core/performance.zig"),
    });

    // Sampler module (Day 11)
    const sampler_module = b.createModule(.{
        .root_source_file = b.path("sampling/sampler.zig"),
    });

    // KV Cache v2 module (Day 16)
    const kv_cache_v2_module = b.createModule(.{
        .root_source_file = b.path("cache/kv_cache.zig"),
    });

    // Cache Manager module (Day 17)
    const cache_manager_module = b.createModule(.{
        .root_source_file = b.path("cache/cache_manager.zig"),
    });

    // Flash Attention module (Day 18)
    const flash_attention_module = b.createModule(.{
        .root_source_file = b.path("attention/flash_attention.zig"),
    });

    // Advanced Attention module (Day 19)
    const advanced_attention_module = b.createModule(.{
        .root_source_file = b.path("attention/advanced_attention.zig"),
    });

    // Batch Inference module (Day 20)
    const batch_inference_module = b.createModule(.{
        .root_source_file = b.path("batch/batch_inference.zig"),
    });

    // Optimized Inference module (Day 21)
    const optimized_inference_module = b.createModule(.{
        .root_source_file = b.path("integration/optimized_inference.zig"),
    });
    optimized_inference_module.addImport("kv_cache", kv_cache_v2_module);
    optimized_inference_module.addImport("cache_manager", cache_manager_module);
    optimized_inference_module.addImport("flash_attention", flash_attention_module);
    optimized_inference_module.addImport("advanced_attention", advanced_attention_module);
    optimized_inference_module.addImport("batch_inference", batch_inference_module);

    // Memory Pool module (Improvement 1)
    const memory_pool_module = b.createModule(.{
        .root_source_file = b.path("core/memory_pool.zig"),
    });

    // Advanced Sampler module (Improvement 2)
    const advanced_sampler_module = b.createModule(.{
        .root_source_file = b.path("sampling/advanced_sampler.zig"),
    });

    // SafeTensors Loader module (Production Feature)
    const safetensors_loader_module = b.createModule(.{
        .root_source_file = b.path("loader/safetensors_loader.zig"),
    });

    // SafeTensors Sharded Loader module (Production Feature)
    const safetensors_sharded_module = b.createModule(.{
        .root_source_file = b.path("loader/safetensors_sharded.zig"),
    });
    safetensors_sharded_module.addImport("safetensors_loader", safetensors_loader_module);

    // Config Parser module (Production Feature)
    const config_parser_module = b.createModule(.{
        .root_source_file = b.path("loader/config_parser.zig"),
    });

    // BPE Tokenizer module (Production Feature)
    const bpe_tokenizer_module = b.createModule(.{
        .root_source_file = b.path("tokenization/bpe_tokenizer.zig"),
    });

    // HuggingFace Loader module (Production Feature - Integration)
    const huggingface_loader_module = b.createModule(.{
        .root_source_file = b.path("loader/huggingface_loader.zig"),
    });
    huggingface_loader_module.addImport("config_parser", config_parser_module);
    huggingface_loader_module.addImport("safetensors_sharded", safetensors_sharded_module);
    huggingface_loader_module.addImport("safetensors_loader", safetensors_loader_module);
    huggingface_loader_module.addImport("bpe_tokenizer", bpe_tokenizer_module);

    // HF → LLaMA Bridge module (Connects HF loader to inference engine)
    const hf_to_llama_bridge_module = b.createModule(.{
        .root_source_file = b.path("loader/hf_to_llama_bridge.zig"),
    });
    hf_to_llama_bridge_module.addImport("huggingface_loader", huggingface_loader_module);
    hf_to_llama_bridge_module.addImport("llama_model", llama_model_module);
    hf_to_llama_bridge_module.addImport("transformer", transformer_module);
    hf_to_llama_bridge_module.addImport("tokenizer", tokenizer_module);
    hf_to_llama_bridge_module.addImport("matrix_ops", matrix_ops_module);
    hf_to_llama_bridge_module.addImport("gguf_loader", gguf_module);

    // ========================================================================
    // HF → LLaMA Bridge Test
    // ========================================================================

    const test_hf_to_llama = b.addExecutable(.{
        .name = "test_hf_to_llama",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_hf_to_llama.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_hf_to_llama.root_module.addImport("hf_to_llama_bridge", hf_to_llama_bridge_module);
    b.installArtifact(test_hf_to_llama);

    const run_test_hf_to_llama = b.addRunArtifact(test_hf_to_llama);
    run_test_hf_to_llama.step.dependOn(b.getInstallStep());

    const test_hf_to_llama_step = b.step("test-hf-to-llama", "Test HF → LLaMA bridge");
    test_hf_to_llama_step.dependOn(&run_test_hf_to_llama.step);

    // LFM2 smoke test
    const test_lfm2_smoke = b.addExecutable(.{
        .name = "test_lfm2_smoke",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_lfm2_smoke.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_lfm2_smoke.root_module.addImport("lfm2_model", lfm2_model_module);
    test_lfm2_smoke.root_module.addImport("tokenizer", tokenizer_module);
    test_lfm2_smoke.root_module.addImport("gguf_loader", gguf_module);
    test_lfm2_smoke.root_module.addImport("matrix_ops", matrix_ops_module);
    b.installArtifact(test_lfm2_smoke);

    const run_test_lfm2_smoke = b.addRunArtifact(test_lfm2_smoke);
    run_test_lfm2_smoke.step.dependOn(b.getInstallStep());

    const test_lfm2_smoke_step = b.step("test-lfm2-smoke", "Run LFM2 smoke test");
    test_lfm2_smoke_step.dependOn(&run_test_lfm2_smoke.step);

    // ========================================================================
    // Mojo Bridge Test (C API for Mojo integration)
    // ========================================================================

    const inference_lib = b.addLibrary(.{
        .name = "inference",
        .linkage = .dynamic,
        .root_module = b.createModule(.{
            .root_source_file = b.path("mojo_bridge.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    inference_lib.root_module.addImport("huggingface_loader", huggingface_loader_module);
    inference_lib.root_module.addImport("hf_to_llama_bridge", hf_to_llama_bridge_module);
    inference_lib.root_module.addImport("llama_model", llama_model_module);
    inference_lib.root_module.addImport("gguf_model_loader", gguf_model_loader_module);
    inference_lib.root_module.addImport("lfm2_model", lfm2_model_module);
    b.installArtifact(inference_lib);

    // Note: mojo_bridge.zig is a library module without main(), used only via inference lib
    // const test_mojo_bridge = b.addExecutable(.{
    //     .name = "test_mojo_bridge",
    //     .root_module = b.createModule(.{
    //         .root_source_file = b.path("mojo_bridge.zig"),
    //         .target = target,
    //         .optimize = optimize,
    //     }),
    // });
    // test_mojo_bridge.root_module.addImport("huggingface_loader", huggingface_loader_module);
    // test_mojo_bridge.root_module.addImport("hf_to_llama_bridge", hf_to_llama_bridge_module);
    // test_mojo_bridge.root_module.addImport("llama_model", llama_model_module);
    // test_mojo_bridge.root_module.addImport("gguf_model_loader", gguf_model_loader_module);
    // test_mojo_bridge.root_module.addImport("lfm2_model", lfm2_model_module);
    // b.installArtifact(test_mojo_bridge);
    //
    // const run_test_mojo_bridge = b.addRunArtifact(test_mojo_bridge);
    // run_test_mojo_bridge.step.dependOn(b.getInstallStep());
    //
    // const test_mojo_bridge_step = b.step("test-mojo-bridge", "Test Mojo bridge C API");
    // test_mojo_bridge_step.dependOn(&run_test_mojo_bridge.step);

    // ========================================================================
    // CLI Executable (Day 9)
    // ========================================================================

    const cli = b.addExecutable(.{
        .name = "zig-inference",
        .root_module = b.createModule(.{
            .root_source_file = b.path("cli/main.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    cli.root_module.addImport("gguf_loader", gguf_module);
    cli.root_module.addImport("llama_model", llama_model_module);
    cli.root_module.addImport("gguf_model_loader", gguf_model_loader_module);
    cli.root_module.addImport("batch_processor", batch_processor_module);
    cli.root_module.addImport("performance", performance_module);
    cli.root_module.addImport("sampler", sampler_module);
    cli.root_module.addImport("lfm2_model", lfm2_model_module);

    if (target.result.os.tag == .macos) {
        cli.linkFramework("Metal");
        cli.linkFramework("Foundation");
    }

    b.installArtifact(cli);

    const run_cli = b.addRunArtifact(cli);
    run_cli.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cli.addArgs(args);
    }

    const run_step = b.step("run-cli", "Run the CLI inference tool");
    run_step.dependOn(&run_cli.step);

    // ========================================================================
    // Day 1 Test: GGUF Loader
    // ========================================================================

    const test_gguf = b.addExecutable(.{
        .name = "test_gguf_loader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_gguf_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_gguf.root_module.addImport("gguf_loader", gguf_module);
    b.installArtifact(test_gguf);

    const run_test_gguf = b.addRunArtifact(test_gguf);
    run_test_gguf.step.dependOn(b.getInstallStep());

    const test_day1_step = b.step("test-day1", "Run Day 1 tests (GGUF loader)");
    test_day1_step.dependOn(&run_test_gguf.step);

    // ========================================================================
    // Day 2 Test: Matrix Operations & Quantization
    // ========================================================================

    const test_day2 = b.addExecutable(.{
        .name = "test_day2",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day2.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day2.root_module.addImport("matrix_ops", matrix_ops_module);
    test_day2.root_module.addImport("quantization_common", quant_common_module);
    test_day2.root_module.addImport("q4_0", q4_0_module);
    b.installArtifact(test_day2);

    const run_test_day2 = b.addRunArtifact(test_day2);
    run_test_day2.step.dependOn(b.getInstallStep());

    const test_day2_step = b.step("test-day2", "Run Day 2 tests (Matrix ops & Quantization)");
    test_day2_step.dependOn(&run_test_day2.step);

    // ========================================================================
    // Day 3 Test: Tokenizer & KV Cache
    // ========================================================================

    const test_day3 = b.addExecutable(.{
        .name = "test_day3",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day3.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day3.root_module.addImport("tokenizer", tokenizer_module);
    test_day3.root_module.addImport("kv_cache", kv_cache_module);
    b.installArtifact(test_day3);

    const run_test_day3 = b.addRunArtifact(test_day3);
    run_test_day3.step.dependOn(b.getInstallStep());

    const test_day3_step = b.step("test-day3", "Run Day 3 tests (Tokenizer & KV Cache)");
    test_day3_step.dependOn(&run_test_day3.step);

    // ========================================================================
    // Day 4 Test: Transformer Layer
    // ========================================================================

    const test_day4 = b.addExecutable(.{
        .name = "test_day4",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day4.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day4.root_module.addImport("attention", attention_module);
    test_day4.root_module.addImport("feed_forward", feed_forward_module);
    test_day4.root_module.addImport("transformer", transformer_module);
    b.installArtifact(test_day4);

    const run_test_day4 = b.addRunArtifact(test_day4);
    run_test_day4.step.dependOn(b.getInstallStep());

    const test_day4_step = b.step("test-day4", "Run Day 4 tests (Transformer Layer)");
    test_day4_step.dependOn(&run_test_day4.step);

    // ========================================================================
    // Day 5 Test: Full Model Integration
    // ========================================================================

    const test_day5 = b.addExecutable(.{
        .name = "test_day5",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day5.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day5.root_module.addImport("llama_model", llama_model_module);
    b.installArtifact(test_day5);

    const run_test_day5 = b.addRunArtifact(test_day5);
    run_test_day5.step.dependOn(b.getInstallStep());

    const test_day5_step = b.step("test-day5", "Run Day 5 tests (Full Model Integration)");
    test_day5_step.dependOn(&run_test_day5.step);

    // ========================================================================
    // Day 6 Test: Quantized Inference Integration
    // ========================================================================

    const test_day6 = b.addExecutable(.{
        .name = "test_day6",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day6.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day6.root_module.addImport("gguf_model_loader", gguf_model_loader_module);
    test_day6.root_module.addImport("llama_model", llama_model_module);
    b.installArtifact(test_day6);

    const run_test_day6 = b.addRunArtifact(test_day6);
    run_test_day6.step.dependOn(b.getInstallStep());

    const test_day6_step = b.step("test-day6", "Run Day 6 tests (Quantized Inference)");
    test_day6_step.dependOn(&run_test_day6.step);

    // ========================================================================
    // Day 7 Test: Batch Processing
    // ========================================================================

    const test_day7 = b.addExecutable(.{
        .name = "test_day7",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day7.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day7.root_module.addImport("batch_processor", batch_processor_module);
    test_day7.root_module.addImport("llama_model", llama_model_module);
    test_day7.root_module.addImport("gguf_loader", gguf_module);
    test_day7.root_module.addImport("transformer", transformer_module);
    test_day7.root_module.addImport("tokenizer", tokenizer_module);
    test_day7.root_module.addImport("matrix_ops", matrix_ops_module);
    b.installArtifact(test_day7);

    const run_test_day7 = b.addRunArtifact(test_day7);
    run_test_day7.step.dependOn(b.getInstallStep());

    const test_day7_step = b.step("test-day7", "Run Day 7 tests (Batch Processing)");
    test_day7_step.dependOn(&run_test_day7.step);

    // ========================================================================
    // Day 8 Test: Performance Optimization
    // ========================================================================

    const test_day8 = b.addExecutable(.{
        .name = "test_day8",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day8.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day8.root_module.addImport("performance", performance_module);
    b.installArtifact(test_day8);

    const run_test_day8 = b.addRunArtifact(test_day8);
    run_test_day8.step.dependOn(b.getInstallStep());

    const test_day8_step = b.step("test-day8", "Run Day 8 tests (Performance Optimization)");
    test_day8_step.dependOn(&run_test_day8.step);

    // ========================================================================
    // Day 11 Test: Advanced Sampling
    // ========================================================================

    const test_day11 = b.addExecutable(.{
        .name = "test_day11",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day11.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day11.root_module.addImport("sampler", sampler_module);
    b.installArtifact(test_day11);

    const run_test_day11 = b.addRunArtifact(test_day11);
    run_test_day11.step.dependOn(b.getInstallStep());

    const test_day11_step = b.step("test-day11", "Run Day 11 tests (Advanced Sampling)");
    test_day11_step.dependOn(&run_test_day11.step);

    // ========================================================================
    // Day 13 Test: Q8_0 Quantization
    // ========================================================================

    const test_day13 = b.addExecutable(.{
        .name = "test_day13",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day13.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day13.root_module.addImport("q8_0", q8_0_module);
    b.installArtifact(test_day13);

    const run_test_day13 = b.addRunArtifact(test_day13);
    run_test_day13.step.dependOn(b.getInstallStep());

    const test_day13_step = b.step("test-day13", "Run Day 13 tests (Q8_0 Quantization)");
    test_day13_step.dependOn(&run_test_day13.step);

    // ========================================================================
    // Day 14 Test: Multi-threading
    // ========================================================================

    const test_day14 = b.addExecutable(.{
        .name = "test_day14",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day14.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day14.root_module.addImport("thread_pool", thread_pool_module);
    b.installArtifact(test_day14);

    const run_test_day14 = b.addRunArtifact(test_day14);
    run_test_day14.step.dependOn(b.getInstallStep());

    const test_day14_step = b.step("test-day14", "Run Day 14 tests (Multi-threading)");
    test_day14_step.dependOn(&run_test_day14.step);

    // ========================================================================
    // Day 16 Test: KV Cache v2
    // ========================================================================

    const test_day16 = b.addExecutable(.{
        .name = "test_day16",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day16.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day16.root_module.addImport("kv_cache", kv_cache_v2_module);
    b.installArtifact(test_day16);

    const run_test_day16 = b.addRunArtifact(test_day16);
    run_test_day16.step.dependOn(b.getInstallStep());

    const test_day16_step = b.step("test-day16", "Run Day 16 tests (KV Cache v2)");
    test_day16_step.dependOn(&run_test_day16.step);

    // ========================================================================
    // Day 17 Test: Cache Management
    // ========================================================================

    const test_day17 = b.addExecutable(.{
        .name = "test_day17",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day17.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day17.root_module.addImport("cache_manager", cache_manager_module);
    b.installArtifact(test_day17);

    const run_test_day17 = b.addRunArtifact(test_day17);
    run_test_day17.step.dependOn(b.getInstallStep());

    const test_day17_step = b.step("test-day17", "Run Day 17 tests (Cache Management)");
    test_day17_step.dependOn(&run_test_day17.step);

    // ========================================================================
    // Day 18 Test: Flash Attention
    // ========================================================================

    const test_day18 = b.addExecutable(.{
        .name = "test_day18",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day18.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day18.root_module.addImport("flash_attention", flash_attention_module);
    b.installArtifact(test_day18);

    const run_test_day18 = b.addRunArtifact(test_day18);
    run_test_day18.step.dependOn(b.getInstallStep());

    const test_day18_step = b.step("test-day18", "Run Day 18 tests (Flash Attention)");
    test_day18_step.dependOn(&run_test_day18.step);

    // ========================================================================
    // Day 19 Test: Advanced Attention
    // ========================================================================

    const test_day19 = b.addExecutable(.{
        .name = "test_day19",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day19.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day19.root_module.addImport("advanced_attention", advanced_attention_module);
    b.installArtifact(test_day19);

    const run_test_day19 = b.addRunArtifact(test_day19);
    run_test_day19.step.dependOn(b.getInstallStep());

    const test_day19_step = b.step("test-day19", "Run Day 19 tests (Advanced Attention)");
    test_day19_step.dependOn(&run_test_day19.step);

    // ========================================================================
    // Day 20 Test: Batch Inference
    // ========================================================================

    const test_day20 = b.addExecutable(.{
        .name = "test_day20",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day20.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day20.root_module.addImport("batch_inference", batch_inference_module);
    b.installArtifact(test_day20);

    const run_test_day20 = b.addRunArtifact(test_day20);
    run_test_day20.step.dependOn(b.getInstallStep());

    const test_day20_step = b.step("test-day20", "Run Day 20 tests (Batch Inference)");
    test_day20_step.dependOn(&run_test_day20.step);

    // ========================================================================
    // Day 21 Test: Week 4 Integration
    // ========================================================================

    const test_day21 = b.addExecutable(.{
        .name = "test_day21",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_day21.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_day21.root_module.addImport("optimized_inference", optimized_inference_module);
    b.installArtifact(test_day21);

    const run_test_day21 = b.addRunArtifact(test_day21);
    run_test_day21.step.dependOn(b.getInstallStep());

    const test_day21_step = b.step("test-day21", "Run Day 21 tests (Week 4 Integration)");
    test_day21_step.dependOn(&run_test_day21.step);

    // ========================================================================
    // Memory Pool Test (Improvement 1)
    // ========================================================================

    const test_memory_pool = b.addExecutable(.{
        .name = "test_memory_pool",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_memory_pool.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_memory_pool.root_module.addImport("memory_pool", memory_pool_module);
    b.installArtifact(test_memory_pool);

    const run_test_memory_pool = b.addRunArtifact(test_memory_pool);
    run_test_memory_pool.step.dependOn(b.getInstallStep());

    const test_memory_pool_step = b.step("test-memory-pool", "Run Memory Pool tests");
    test_memory_pool_step.dependOn(&run_test_memory_pool.step);

    // ========================================================================
    // Advanced Sampler Test (Improvement 2)
    // ========================================================================

    const test_advanced_sampler = b.addExecutable(.{
        .name = "test_advanced_sampler",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_advanced_sampler.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_advanced_sampler.root_module.addImport("advanced_sampler", advanced_sampler_module);
    b.installArtifact(test_advanced_sampler);

    const run_test_advanced_sampler = b.addRunArtifact(test_advanced_sampler);
    run_test_advanced_sampler.step.dependOn(b.getInstallStep());

    const test_advanced_sampler_step = b.step("test-advanced-sampler", "Run Advanced Sampler tests");
    test_advanced_sampler_step.dependOn(&run_test_advanced_sampler.step);

    // ========================================================================
    // SafeTensors Loader Test (Production Feature)
    // ========================================================================

    const test_safetensors = b.addExecutable(.{
        .name = "test_safetensors",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_safetensors.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_safetensors.root_module.addImport("safetensors_loader", safetensors_loader_module);
    b.installArtifact(test_safetensors);

    const run_test_safetensors = b.addRunArtifact(test_safetensors);
    run_test_safetensors.step.dependOn(b.getInstallStep());

    const test_safetensors_step = b.step("test-safetensors", "Run SafeTensors Loader tests");
    test_safetensors_step.dependOn(&run_test_safetensors.step);

    // ========================================================================
    // SafeTensors Sharded Loader Test (Production Feature)
    // ========================================================================

    const test_safetensors_sharded = b.addExecutable(.{
        .name = "test_safetensors_sharded",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_safetensors_sharded.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_safetensors_sharded.root_module.addImport("safetensors_sharded", safetensors_sharded_module);
    b.installArtifact(test_safetensors_sharded);

    const run_test_safetensors_sharded = b.addRunArtifact(test_safetensors_sharded);
    run_test_safetensors_sharded.step.dependOn(b.getInstallStep());

    const test_safetensors_sharded_step = b.step("test-safetensors-sharded", "Run Sharded SafeTensors Loader tests");
    test_safetensors_sharded_step.dependOn(&run_test_safetensors_sharded.step);

    // ========================================================================
    // Config Parser Test (Production Feature)
    // ========================================================================

    const test_config_parser = b.addExecutable(.{
        .name = "test_config_parser",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_config_parser.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_config_parser.root_module.addImport("config_parser", config_parser_module);
    b.installArtifact(test_config_parser);

    const run_test_config_parser = b.addRunArtifact(test_config_parser);
    run_test_config_parser.step.dependOn(b.getInstallStep());

    const test_config_parser_step = b.step("test-config-parser", "Run Config Parser tests");
    test_config_parser_step.dependOn(&run_test_config_parser.step);

    // ========================================================================
    // BPE Tokenizer Test (Production Feature)
    // ========================================================================

    const test_bpe_tokenizer = b.addExecutable(.{
        .name = "test_bpe_tokenizer",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_bpe_tokenizer.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_bpe_tokenizer.root_module.addImport("bpe_tokenizer", bpe_tokenizer_module);
    b.installArtifact(test_bpe_tokenizer);

    const run_test_bpe_tokenizer = b.addRunArtifact(test_bpe_tokenizer);
    run_test_bpe_tokenizer.step.dependOn(b.getInstallStep());

    const test_bpe_tokenizer_step = b.step("test-bpe-tokenizer", "Run BPE Tokenizer tests");
    test_bpe_tokenizer_step.dependOn(&run_test_bpe_tokenizer.step);

    // ========================================================================
    // HuggingFace Loader Test (Production Feature - Integration)
    // ========================================================================

    const test_huggingface_loader = b.addExecutable(.{
        .name = "test_huggingface_loader",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_huggingface_loader.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_huggingface_loader.root_module.addImport("huggingface_loader", huggingface_loader_module);
    b.installArtifact(test_huggingface_loader);

    const run_test_huggingface_loader = b.addRunArtifact(test_huggingface_loader);
    run_test_huggingface_loader.step.dependOn(b.getInstallStep());

    const test_huggingface_loader_step = b.step("test-huggingface-loader", "Run HuggingFace Loader tests");
    test_huggingface_loader_step.dependOn(&run_test_huggingface_loader.step);

    // ========================================================================
    // All Models Test (Dynamic Discovery)
    // ========================================================================

    const test_all_models = b.addExecutable(.{
        .name = "test_all_models",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tests/test_all_models.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    test_all_models.root_module.addImport("huggingface_loader", huggingface_loader_module);
    b.installArtifact(test_all_models);

    const run_test_all_models = b.addRunArtifact(test_all_models);
    run_test_all_models.step.dependOn(b.getInstallStep());

    const test_all_models_step = b.step("test-all-models", "Dynamically discover and test all models");
    test_all_models_step.dependOn(&run_test_all_models.step);

    // ========================================================================
    // Combined Test
    // ========================================================================

    const test_step = b.step("test", "Run all tests");
    test_step.dependOn(&run_test_gguf.step);
    test_step.dependOn(&run_test_day2.step);
    test_step.dependOn(&run_test_day3.step);
    test_step.dependOn(&run_test_day4.step);
    test_step.dependOn(&run_test_day5.step);
    test_step.dependOn(&run_test_day6.step);
    test_step.dependOn(&run_test_day7.step);
    test_step.dependOn(&run_test_day8.step);
    test_step.dependOn(&run_test_day11.step);
    test_step.dependOn(&run_test_day13.step);
    test_step.dependOn(&run_test_day14.step);
    test_step.dependOn(&run_test_day16.step);
    test_step.dependOn(&run_test_day17.step);
    test_step.dependOn(&run_test_day18.step);
    test_step.dependOn(&run_test_day19.step);
    test_step.dependOn(&run_test_day20.step);
    test_step.dependOn(&run_test_day21.step);
    test_step.dependOn(&run_test_memory_pool.step);
    test_step.dependOn(&run_test_advanced_sampler.step);
    test_step.dependOn(&run_test_safetensors.step);
    test_step.dependOn(&run_test_safetensors_sharded.step);
    test_step.dependOn(&run_test_config_parser.step);
    test_step.dependOn(&run_test_bpe_tokenizer.step);
    test_step.dependOn(&run_test_huggingface_loader.step);

    // ========================================================================
    // Default build
    // ========================================================================

    const default_run_step = b.step("run", "Run GGUF loader tests");
    default_run_step.dependOn(&run_test_gguf.step);
}
