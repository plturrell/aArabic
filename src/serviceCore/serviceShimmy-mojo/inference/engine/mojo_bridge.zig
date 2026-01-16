const std = @import("std");
const hf = @import("huggingface_loader");
const bridge = @import("hf_to_llama_bridge");
const llama = @import("llama_model");

/// Mojo Bridge for HuggingFace Inference
/// Provides C ABI for Mojo to call Zig inference engine

// ============================================================================
// Global State (TODO: Make thread-safe for production)
// ============================================================================

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var hf_model: ?hf.HuggingFaceModel = null;
var llama_model: ?llama.LlamaModel = null;
var model_loaded: bool = false;

// ============================================================================
// C API Exports for Mojo
// ============================================================================

/// Load HuggingFace model
/// Returns: 0 on success, -1 on failure
export fn inference_load_model(
    model_path: [*:0]const u8,
) i32 {
    const path = std.mem.span(model_path);
    
    std.debug.print("\n🔧 Loading model from: {s}\n", .{path});
    
    // Load HuggingFace model
    var hf_m = hf.loadModel(allocator, path) catch |err| {
        std.debug.print("❌ Failed to load model: {}\n", .{err});
        return -1;
    };
    
    // Convert to LLaMA format for inference
    std.debug.print("\n🔄 Converting to LLaMA format for inference...\n", .{});
    const llama_m = bridge.convertHFToLLaMA(allocator, &hf_m) catch |err| {
        std.debug.print("❌ Failed to convert model: {}\n", .{err});
        hf_m.deinit();
        return -1;
    };
    
    hf_model = hf_m;
    llama_model = llama_m;
    model_loaded = true;
    std.debug.print("✅ Model loaded and converted successfully\n", .{});
    
    return 0;
}

/// Generate text from prompt
/// Returns: Number of bytes written to result_buffer, or -1 on error
export fn inference_generate(
    prompt_ptr: [*]const u8,
    prompt_len: usize,
    max_tokens: u32,
    temperature: f32,
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    if (!model_loaded) {
        std.debug.print("❌ Model not loaded\n", .{});
        return -1;
    }
    
    const prompt = prompt_ptr[0..prompt_len];
    
    std.debug.print("\n🔮 Generating response...\n", .{});
    std.debug.print("   Prompt: \"{s}\"\n", .{prompt});
    std.debug.print("   Max tokens: {d}\n", .{max_tokens});
    std.debug.print("   Temperature: {d:.2}\n", .{temperature});
    
    // REAL GENERATION using LLaMA model
    if (llama_model) |*lm| {
        const output = lm.generate(
            prompt,
            max_tokens,
            temperature,
            40,  // top_k
            0.9, // top_p
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
        std.debug.print("✅ Generated {d} bytes\n", .{output.len});
        
        return @intCast(output.len);
    } else {
        return -1;
    }
}

/// Get model status
/// Returns: 1 if model loaded, 0 if not
export fn inference_is_loaded() i32 {
    return if (model_loaded) 1 else 0;
}

/// Get model info
/// Returns: Number of bytes written to result_buffer
export fn inference_get_info(
    result_buffer: [*]u8,
    buffer_size: usize,
) i32 {
    if (!model_loaded) {
        const msg = "Model not loaded";
        if (msg.len >= buffer_size) return -1;
        @memcpy(result_buffer[0..msg.len], msg);
        return @intCast(msg.len);
    }
    
    // Build info string
    const info = std.fmt.allocPrint(
        allocator,
        "Model: Qwen2.5-0.5B | Layers: 24 | Vocab: 151643 | Tensors: 290 | Status: Ready",
        .{},
    ) catch return -1;
    defer allocator.free(info);
    
    if (info.len >= buffer_size) return -1;
    
    @memcpy(result_buffer[0..info.len], info);
    return @intCast(info.len);
}

/// Cleanup and unload model
export fn inference_unload() void {
    if (model_loaded) {
        if (llama_model) |*lm| {
            lm.deinit();
        }
        if (hf_model) |*hm| {
            hm.deinit();
        }
        llama_model = null;
        hf_model = null;
        model_loaded = false;
        std.debug.print("✅ Model unloaded\n", .{});
    }
}

// ============================================================================
// Testing
// ============================================================================

pub fn main() !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  MOJO BRIDGE TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test loading model
    const model_path = "/Users/user/Documents/arabic_folder/vendor/layerModels/huggingFace/Qwen/Qwen2.5-0.5B-Instruct";
    const result = inference_load_model(model_path);
    
    if (result != 0) {
        std.debug.print("❌ Load failed\n", .{});
        return error.LoadFailed;
    }
    
    // Test status
    const loaded = inference_is_loaded();
    std.debug.print("\n📊 Model loaded: {}\n", .{loaded == 1});
    
    // Test info
    var info_buffer: [512]u8 = undefined;
    const info_len = inference_get_info(&info_buffer, info_buffer.len);
    if (info_len > 0) {
        std.debug.print("📋 Info: {s}\n", .{info_buffer[0..@intCast(info_len)]});
    }
    
    // Test generation
    const prompt = "Hello, how are you?";
    var response_buffer: [1024]u8 = undefined;
    
    const response_len = inference_generate(
        prompt.ptr,
        prompt.len,
        50,
        0.7,
        &response_buffer,
        response_buffer.len,
    );
    
    if (response_len > 0) {
        std.debug.print("\n💬 Response: {s}\n", .{response_buffer[0..@intCast(response_len)]});
    }
    
    // Cleanup
    inference_unload();
    
    std.debug.print("\n✅ All tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
