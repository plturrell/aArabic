const std = @import("std");
const hf = @import("huggingface_loader");
const bridge = @import("hf_to_llama_bridge");
const llama = @import("llama_model");
const gguf_loader = @import("gguf_model_loader");
const lfm2 = @import("lfm2_model");
// Note: memory_check module not available - commented out for now
// const memory_check = @import("../../shared/system/memory_check.zig");

const ModelContext = struct {
    id: []const u8,
    hf_model: ?hf.HuggingFaceModel = null,
    llama_model: ?llama.LlamaModel = null,
    lfm2_model: ?lfm2.Lfm2Model = null,
    loaded: bool = false,
    mutex: std.Thread.Mutex = .{},
};

const ContextMap = std.StringHashMap(*ModelContext);

var log_enabled: ?bool = null;
var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();
var contexts = ContextMap.init(allocator);
const default_model_id_c = [_:0]u8{ 'd', 'e', 'f', 'a', 'u', 'l', 't', 0 };

fn logEnabled() bool {
    if (log_enabled) |enabled| {
        return enabled;
    }
    log_enabled = std.posix.getenv("SHIMMY_DEBUG") != null;
    return log_enabled.?;
}

fn log(comptime fmt: []const u8, args: anytype) void {
    if (logEnabled()) {
        std.debug.print(fmt, args);
    }
}

fn spanFromPtr(ptr: [*:0]const u8) []const u8 {
    return std.mem.span(ptr);
}

fn getContext(model_id: []const u8) !*ModelContext {
    if (contexts.get(model_id)) |ctx| {
        return ctx;
    }

    const id_copy = try allocator.dupe(u8, model_id);
    const ctx = try allocator.create(ModelContext);
    ctx.* = ModelContext{
        .id = id_copy,
        .hf_model = null,
        .llama_model = null,
        .lfm2_model = null,
        .loaded = false,
    };
    try contexts.put(ctx.id, ctx);
    return ctx;
}

fn findContext(model_id: []const u8) ?*ModelContext {
    return contexts.get(model_id);
}

fn unloadContext(ctx: *ModelContext) void {
    if (ctx.llama_model) |*lm| {
        lm.deinit();
    }
    if (ctx.lfm2_model) |*m| {
        m.deinit();
    }
    if (ctx.hf_model) |*hm| {
        hm.deinit();
    }
    ctx.llama_model = null;
    ctx.lfm2_model = null;
    ctx.hf_model = null;
    ctx.loaded = false;
}

fn loadGGUFModel(ctx: *ModelContext, path: []const u8) !void {
    const file = try std.fs.cwd().openFile(path, .{});
    const file_size = try file.getEndPos();
    _ = file_size; // Suppress unused warning
    file.close();

    // Note: memory_check module not available - skipping memory validation
    // const mem_result = memory_check.checkMemoryForGGUF(file_size, .Q4_0);
    // memory_check.printMemoryCheck(.{
    //     .param_count = file_size,
    //     .bytes_per_param = 4,
    // });
    //
    // if (!mem_result.ok) {
    //     std.debug.print("❌ Insufficient memory to load model\n", .{});
    //     std.debug.print("   Required: {d} MB, Available: {d} MB\n", .{
    //         mem_result.required.total_mb,
    //         mem_result.system.available_mb,
    //     });
    //     return error.InsufficientMemory;
    // }

    // Auto-select optimal loading strategy based on model size
    // For now, use OnTheFly to save memory on all models
    // TODO: Pass tier_config from config.json for smarter selection
    var loader = gguf_loader.GGUFModelLoader.init(allocator, .OnTheFly);
    std.debug.print("   📋 Using OnTheFly strategy (quantized weights, low memory)\n", .{});
    const llama_result = loader.loadModel(path) catch |err| switch (err) {
        error.UnsupportedArchitecture => null,
        else => return err,
    };
    if (llama_result) |model| {
        ctx.llama_model = model;
        ctx.lfm2_model = null;
        ctx.hf_model = null;
        ctx.loaded = true;
        return;
    }

    const lfm2_result = loader.loadLfm2Model(path) catch |err| switch (err) {
        else => return err,
    };
    ctx.lfm2_model = lfm2_result;
    ctx.llama_model = null;
    ctx.hf_model = null;
    ctx.loaded = true;
}

fn loadModelFromPath(ctx: *ModelContext, path: []const u8) !void {
    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".gguf")) {
        try loadGGUFModel(ctx, path);
        log("GGUF model loaded successfully for {s}\n", .{ctx.id});
        return;
    }

    var hf_m = hf.loadModel(allocator, path) catch |err| {
        std.debug.print("❌ Failed to load model: {}\n", .{err});
        return err;
    };

    const llama_m = bridge.convertHFToLLaMA(allocator, &hf_m) catch |err| {
        std.debug.print("❌ Failed to convert model: {}\n", .{err});
        hf_m.deinit();
        return err;
    };

    ctx.hf_model = hf_m;
    ctx.llama_model = llama_m;
    ctx.lfm2_model = null;
    ctx.loaded = true;
    log("Model loaded and converted successfully for {s}\n", .{ctx.id});
}

fn ensureContext(model_id: []const u8) !*ModelContext {
    return try getContext(model_id);
}

export fn inference_load_model_v2(
    model_id_ptr: [*:0]const u8,
    model_path_ptr: [*:0]const u8,
) i32 {
    std.debug.print("\n📂 inference_load_model_v2 called!\n", .{});
    const model_id = spanFromPtr(model_id_ptr);
    const model_path = spanFromPtr(model_path_ptr);
    std.debug.print("   Model ID: \"{s}\"\n", .{model_id});
    std.debug.print("   Model path: \"{s}\"\n", .{model_path});

    const ctx = ensureContext(model_id) catch return -1;
    ctx.mutex.lock();
    defer ctx.mutex.unlock();
    std.debug.print("   Context created with id: \"{s}\"\n", .{ctx.id});

    unloadContext(ctx);

    loadModelFromPath(ctx, model_path) catch |err| {
        std.debug.print("❌ Failed to load {s}: {any}\n", .{ model_id, err });
        return -1;
    };

    return 0;
}

export fn inference_generate_v2(
    model_id_ptr: [*:0]const u8,
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    std.debug.print("\n🔥🔥🔥 inference_generate_v2 CALLED! 🔥🔥🔥\n", .{});
    const model_id = spanFromPtr(model_id_ptr);
    std.debug.print("   Model ID: {s}\n", .{model_id});
    const ctx = findContext(model_id) orelse {
        std.debug.print("❌ Model not found: {s}\n", .{model_id});
        return -1;
    };
    std.debug.print("   Context found, llama_model={}, lfm2_model={}\n", .{ ctx.llama_model != null, ctx.lfm2_model != null });

    ctx.mutex.lock();
    defer ctx.mutex.unlock();

    if (!ctx.loaded) {
        std.debug.print("❌ Model not loaded: {s}\n", .{model_id});
        return -1;
    }

    const prompt = prompt_ptr[0..prompt_len];

    log("\nGenerating response ({s})...\n", .{model_id});
    log("   Prompt: \"{s}\"\n", .{prompt});
    log("   Max tokens: {d}\n", .{max_tokens});
    log("   Temperature: {d:.2}\n", .{temperature});

    if (ctx.llama_model) |*lm| {
        const output = lm.generate(
            prompt,
            max_tokens,
            temperature,
            40,
            0.9,
        ) catch |err| {
            std.debug.print("❌ Generation failed: {}\n", .{err});
            return -1;
        };
        defer allocator.free(output);

        if (output.len >= buffer_size) {
            std.debug.print("❌ Buffer too small for response\n", .{});
            return -1;
        }

        @memcpy(result_buffer[0..output.len], output);
        log("Generated {d} bytes for {s}\n", .{ output.len, model_id });
        return @intCast(output.len);
    }

    if (ctx.lfm2_model) |*m| {
        return generateWithLfm2(ctx, m, prompt, max_tokens, temperature, result_buffer, buffer_size);
    }

    std.debug.print("❌ No model backend available for {s}\n", .{model_id});
    return -1;
}

fn generateWithLfm2(
    ctx: *ModelContext,
    m: *lfm2.Lfm2Model,
    prompt: []const u8,
    max_tokens: u32,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    _ = ctx;
    _ = temperature;

    // Use debug.print for output
    std.debug.print("\n🔥🔥🔥 generateWithLfm2 ENTRY 🔥🔥🔥\n", .{});
    std.debug.print("   Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("   Max tokens: {d}\n", .{max_tokens});

    m.resetCaches();
    const tokens = m.tok.encode(prompt, allocator) catch |err| {
        std.debug.print("❌ Tokenize failed: {}\n", .{err});
        return -1;
    };
    defer allocator.free(tokens);

    for (tokens, 0..) |tok_id, pos| {
        _ = m.forwardToken(tok_id, @intCast(pos)) catch |err| {
            std.debug.print("❌ Forward failed: {}\n", .{err});
            return -1;
        };
        if (pos < tokens.len - 1) m.advanceCaches();
    }

    var generated = std.ArrayList(u8).empty;
    errdefer generated.deinit(allocator);
    generated.ensureTotalCapacity(allocator, max_tokens * 4) catch |err| {
        std.debug.print("❌ Pre-alloc failed: {}\n", .{err});
        return -1;
    };
    defer generated.deinit(allocator);

    var last_token = tokens[tokens.len - 1];
    var position: u32 = @intCast(tokens.len);

    var count: u32 = 0;
    var prev_token: u32 = last_token;
    var repeat_count: u32 = 0;
    while (count < max_tokens) : (count += 1) {
        const logits = m.forwardToken(last_token, position) catch |err| {
            std.debug.print("❌ Forward failed at pos {d}: {}\n", .{ position, err });
            return -1;
        };
        defer allocator.free(logits);

        // Debug: Check logits statistics
        var max_idx: usize = 0;
        var max_val: f32 = logits[0];
        var min_val: f32 = logits[0];
        var sum: f64 = 0;
        var nan_count: usize = 0;
        var inf_count: usize = 0;
        for (logits, 0..) |v, i| {
            if (std.math.isNan(v)) {
                nan_count += 1;
                continue;
            }
            if (std.math.isInf(v)) {
                inf_count += 1;
                continue;
            }
            sum += v;
            if (v > max_val) {
                max_val = v;
                max_idx = i;
            }
            if (v < min_val) {
                min_val = v;
            }
        }
        const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(logits.len)));

        // Print debug info for first few tokens
        if (count < 3) {
            std.debug.print("🔍 Logits[{d}]: min={d:.6}, max={d:.6}, mean={d:.6}, argmax={d}, nan={d}, inf={d}\n", .{ count, min_val, max_val, mean, max_idx, nan_count, inf_count });
            // Print top 5 logits
            std.debug.print("   Top logits: [{d}]={d:.6}", .{ max_idx, max_val });
            // Find 2nd, 3rd highest
            var second_idx: usize = 0;
            var second_val: f32 = -std.math.inf(f32);
            var third_idx: usize = 0;
            var third_val: f32 = -std.math.inf(f32);
            for (logits, 0..) |v, i| {
                if (i == max_idx or std.math.isNan(v) or std.math.isInf(v)) continue;
                if (v > second_val) {
                    third_val = second_val;
                    third_idx = second_idx;
                    second_val = v;
                    second_idx = i;
                } else if (v > third_val) {
                    third_val = v;
                    third_idx = i;
                }
            }
            std.debug.print(", [{d}]={d:.6}, [{d}]={d:.6}\n", .{ second_idx, second_val, third_idx, third_val });
        }

        last_token = @intCast(max_idx);

        if (last_token == m.tok.eos_token or last_token == m.tok.pad_token) {
            break;
        }

        const piece = m.tok.decode(&[_]u32{last_token}, allocator) catch |err| {
            std.debug.print("❌ Decode failed: {}\n", .{err});
            return -1;
        };
        defer allocator.free(piece);

        if (std.mem.indexOf(u8, piece, "<|img") != null or
            std.mem.indexOf(u8, piece, "<|image") != null or
            std.mem.startsWith(u8, piece, "::"))
        {
            continue;
        }

        if (last_token == prev_token) {
            repeat_count += 1;
            if (repeat_count >= 3) break;
        } else {
            repeat_count = 0;
            prev_token = last_token;
        }

        generated.appendSlice(allocator, piece) catch |err| {
            std.debug.print("❌ Append failed: {}\n", .{err});
            return -1;
        };

        m.advanceCaches();
        position += 1;
    }

    const out_slice = generated.toOwnedSlice(allocator) catch |err| {
        std.debug.print("❌ Slice failed: {}\n", .{err});
        return -1;
    };
    defer allocator.free(out_slice);

    if (out_slice.len >= buffer_size) {
        std.debug.print("❌ Buffer too small for response\n", .{});
        return -1;
    }

    @memcpy(result_buffer[0..out_slice.len], out_slice);
    return @intCast(out_slice.len);
}

export fn inference_is_loaded_v2(
    model_id_ptr: [*:0]const u8,
) i32 {
    const model_id = spanFromPtr(model_id_ptr);
    if (findContext(model_id)) |ctx| {
        return if (ctx.loaded) 1 else 0;
    }
    return 0;
}

export fn inference_get_info_v2(
    model_id_ptr: [*:0]const u8,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    const model_id = spanFromPtr(model_id_ptr);
    const ctx = findContext(model_id) orelse {
        const msg = "Model not loaded";
        if (msg.len >= buffer_size) return -1;
        @memcpy(result_buffer[0..msg.len], msg);
        return @intCast(msg.len);
    };

    if (!ctx.loaded) {
        const msg = "Model not loaded";
        if (msg.len >= buffer_size) return -1;
        @memcpy(result_buffer[0..msg.len], msg);
        return @intCast(msg.len);
    }

    const info = std.fmt.allocPrint(
        allocator,
        "Model ID: {s} | Backend: {s}",
        .{
            ctx.id,
            if (ctx.llama_model != null) "llama" else if (ctx.lfm2_model != null) "lfm2" else "unknown",
        },
    ) catch return -1;
    defer allocator.free(info);

    if (info.len >= buffer_size) return -1;
    @memcpy(result_buffer[0..info.len], info);
    return @intCast(info.len);
}

export fn inference_unload_v2(
    model_id_ptr: [*:0]const u8,
) void {
    const model_id = spanFromPtr(model_id_ptr);
    if (findContext(model_id)) |ctx| {
        ctx.mutex.lock();
        defer ctx.mutex.unlock();
        unloadContext(ctx);
        std.debug.print("✅ Model unloaded: {s}\n", .{model_id});
    }
}

export fn inference_unload_all() void {
    var it = contexts.iterator();
    while (it.next()) |entry| {
        const ctx = entry.value_ptr.*;
        ctx.mutex.lock();
        unloadContext(ctx);
        ctx.mutex.unlock();
    }
    std.debug.print("✅ All models unloaded\n", .{});
}

// Legacy wrappers (default model ID) ----------------------------------------

export fn inference_load_model(model_path: [*:0]const u8) i32 {
    return inference_load_model_v2(&default_model_id_c, model_path);
}

export fn inference_generate(
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    return inference_generate_v2(
        &default_model_id_c,
        prompt_ptr,
        prompt_len,
        max_tokens,
        temperature,
        result_buffer,
        buffer_size,
    );
}

export fn inference_is_loaded() i32 {
    return inference_is_loaded_v2(&default_model_id_c);
}

export fn inference_get_info(
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    return inference_get_info_v2(&default_model_id_c, result_buffer, buffer_size);
}

export fn inference_unload() void {
    inference_unload_v2(&default_model_id_c);
}

// ============================================================================
// GPU Batched Inference Server API (High-throughput: 15,000+ tok/s)
// ============================================================================

var gpu_server: ?*GpuInferenceServer = null;
var gpu_weight_cache: ?*GpuWeightCache = null;

const GpuInferenceServer = @import("gpu_inference_server").GpuInferenceServer;
const GpuWeightCache = @import("gpu_weight_cache").GpuWeightCache;
const ServerConfig = @import("gpu_inference_server").ServerConfig;

/// Initialize GPU inference server with batched processing
/// Returns 0 on success, -1 on error
export fn inference_init_gpu_server(
    model_path: [*:0]const u8,
    batch_size: u32,
) i32 {
    const path = spanFromPtr(model_path);
    std.debug.print("\n🚀 Initializing GPU Inference Server...\n", .{});
    std.debug.print("   Model: {s}\n", .{path});
    std.debug.print("   Batch size: {}\n", .{batch_size});

    // Load model to get config
    var loader = gguf_loader.GGUFModelLoader.init(allocator, .OnTheFly);
    const model = loader.loadGGUF(path) catch |err| {
        std.debug.print("❌ Failed to load model: {}\n", .{err});
        return -1;
    };

    // Initialize weight cache
    gpu_weight_cache = allocator.create(GpuWeightCache) catch |err| {
        std.debug.print("❌ Failed to allocate weight cache: {}\n", .{err});
        return -1;
    };
    gpu_weight_cache.?.* = GpuWeightCache.init(
        allocator,
        model.config.embed_dim,
        model.config.hidden_dim,
        model.config.n_heads,
        model.config.n_kv_heads,
        model.config.vocab_size,
        model.config.n_layers,
    ) catch |err| {
        std.debug.print("❌ Failed to init weight cache: {}\n", .{err});
        return -1;
    };

    // Load weights to GPU
    gpu_weight_cache.?.loadFromGGUF(&model) catch |err| {
        std.debug.print("❌ Failed to load weights: {}\n", .{err});
        return -1;
    };

    // Initialize server
    const config = ServerConfig{
        .batch_size = batch_size,
        .embed_dim = model.config.embed_dim,
        .hidden_dim = model.config.hidden_dim,
        .n_heads = model.config.n_heads,
        .n_kv_heads = model.config.n_kv_heads,
        .vocab_size = model.config.vocab_size,
        .n_layers = model.config.n_layers,
    };

    gpu_server = allocator.create(GpuInferenceServer) catch |err| {
        std.debug.print("❌ Failed to allocate server: {}\n", .{err});
        return -1;
    };
    gpu_server.?.* = GpuInferenceServer.init(
        allocator,
        gpu_weight_cache.?,
        config,
    ) catch |err| {
        std.debug.print("❌ Failed to init server: {}\n", .{err});
        return -1;
    };

    std.debug.print("✅ GPU Inference Server ready!\n", .{});
    return 0;
}

/// Submit a request to the GPU inference queue
/// Returns request ID on success, -1 on error
export fn inference_submit_gpu_request(
    token_ids: [*]const u32,
    token_count: u32,
    max_new_tokens: u32,
    temperature: f32,
) i64 {
    const server = gpu_server orelse return -1;

    const request = GpuInferenceServer.InferenceRequest{
        .id = @intCast(std.time.nanoTimestamp()),
        .token_ids = token_ids[0..token_count],
        .max_new_tokens = max_new_tokens,
        .temperature = temperature,
        .callback = null,
        .generated_tokens = undefined,
        .current_position = 0,
        .is_complete = false,
    };

    const id = server.submitRequest(request) catch return -1;
    return @intCast(id);
}

/// Process one batch step, returns number of tokens generated
export fn inference_process_gpu_batch() i32 {
    const server = gpu_server orelse return -1;
    const tokens = server.processBatchStep() catch return -1;
    return @intCast(tokens);
}

/// Get GPU server statistics
export fn inference_get_gpu_stats(
    queue_depth: *u32,
    active_batch: *u32,
    total_tokens: *u64,
) void {
    const server = gpu_server orelse {
        queue_depth.* = 0;
        active_batch.* = 0;
        total_tokens.* = 0;
        return;
    };

    const stats = server.getStats();
    queue_depth.* = @intCast(stats.queue_depth);
    active_batch.* = @intCast(stats.active_batch_size);
    total_tokens.* = stats.total_tokens;
}

/// Shutdown GPU inference server
export fn inference_shutdown_gpu_server() void {
    if (gpu_server) |server| {
        server.deinit();
        allocator.destroy(server);
        gpu_server = null;
    }
    if (gpu_weight_cache) |cache| {
        cache.deinit();
        allocator.destroy(cache);
        gpu_weight_cache = null;
    }
    std.debug.print("✅ GPU server shutdown\n", .{});
}
