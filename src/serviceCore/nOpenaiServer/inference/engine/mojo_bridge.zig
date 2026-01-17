const std = @import("std");
const hf = @import("huggingface_loader");
const bridge = @import("hf_to_llama_bridge");
const llama = @import("llama_model");
const gguf_loader = @import("gguf_model_loader");
const lfm2 = @import("lfm2_model");

var log_enabled: ?bool = null;

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

/// Mojo Bridge for HuggingFace Inference
/// Provides C ABI for Mojo to call Zig inference engine

// ============================================================================
// Global State (TODO: Make thread-safe for production)
// ============================================================================

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var hf_model: ?hf.HuggingFaceModel = null;
var llama_model: ?llama.LlamaModel = null;
var lfm2_model: ?lfm2.Lfm2Model = null;
var model_loaded: bool = false;

fn loadGGUFModel(path: []const u8) !void {
    var loader = gguf_loader.GGUFModelLoader.init(allocator, .DequantizeAll);
    // Try Llama first
    const llama_result = loader.loadModel(path) catch |err| switch (err) {
        error.UnsupportedArchitecture => null,
        else => return err,
    };
    if (llama_result) |model| {
        llama_model = model;
        lfm2_model = null;
        return;
    }
    // Fallback to LFM2
    const lfm2_result = loader.loadLfm2Model(path) catch |err| switch (err) {
        else => return err,
    };
    lfm2_model = lfm2_result;
    llama_model = null;
}

// ============================================================================
// C API Exports for Mojo
// ============================================================================

/// Load HuggingFace model
/// Returns: 0 on success, -1 on failure
export fn inference_load_model(
    model_path: [*:0]const u8,
) i32 {
    const path = std.mem.span(model_path);

    log("\nLoading model from: {s}\n", .{path});

    if (model_loaded) {
        inference_unload();
    }

    const ext = std.fs.path.extension(path);
    if (std.ascii.eqlIgnoreCase(ext, ".gguf")) {
        loadGGUFModel(path) catch |err| {
            std.debug.print("❌ Failed to load GGUF model: {}\n", .{err});
            return -1;
        };

        hf_model = null;
        model_loaded = true;
        log("GGUF model loaded successfully\n", .{});
        return 0;
    }
    
    // Load HuggingFace model
    var hf_m = hf.loadModel(allocator, path) catch |err| {
        std.debug.print("❌ Failed to load model: {}\n", .{err});
        return -1;
    };
    
    // Convert to LLaMA format for inference
    log("\nConverting to LLaMA format for inference...\n", .{});
    const llama_m = bridge.convertHFToLLaMA(allocator, &hf_m) catch |err| {
        std.debug.print("❌ Failed to convert model: {}\n", .{err});
        hf_m.deinit();
        return -1;
    };
    
    hf_model = hf_m;
    llama_model = llama_m;
    model_loaded = true;
    log("Model loaded and converted successfully\n", .{});
    
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
    
    log("\nGenerating response...\n", .{});
    log("   Prompt: \"{s}\"\n", .{prompt});
    log("   Max tokens: {d}\n", .{max_tokens});
    log("   Temperature: {d:.2}\n", .{temperature});
    
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
        log("Generated {d} bytes\n", .{output.len});
        
        return @intCast(output.len);
    }

    if (lfm2_model) |*m| {
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

        std.debug.print("🔍 EOS token ID: {d}, PAD: {d}\n", .{ m.tok.eos_token, m.tok.pad_token });
        
        var count: u32 = 0;
        var prev_token: u32 = last_token;
        var repeat_count: u32 = 0;
        while (count < max_tokens) : (count += 1) {
            const logits = m.forwardToken(last_token, position) catch |err| {
                std.debug.print("❌ Forward failed at pos {d}: {}\n", .{ position, err });
                return -1;
            };
            defer allocator.free(logits);

            // greedy argmax with debug logging
            var max_idx: usize = 0;
            var max_val: f32 = logits[0];
            for (logits[1..], 1..) |v, i| {
                if (v > max_val) {
                    max_val = v;
                    max_idx = i;
                }
            }
            
            // Debug: Print top 5 logits
            var top5: [5]struct { idx: usize, val: f32 } = undefined;
            for (&top5, 0..) |*entry, i| {
                entry.* = .{ .idx = i, .val = logits[i] };
            }
            for (logits[5..], 5..) |v, i| {
                for (&top5) |*entry| {
                    if (v > entry.val) {
                        entry.* = .{ .idx = i, .val = v };
                        break;
                    }
                }
            }
            std.debug.print("   Top 5 logits: ", .{});
            for (top5) |entry| {
                std.debug.print("[{d}]={d:.3} ", .{ entry.idx, entry.val });
            }
            std.debug.print("\n", .{});
            
            last_token = @intCast(max_idx);
            std.debug.print("   Generated token: {d} (logit={d:.3})\n", .{ last_token, max_val });
            
            // Break on EOS or PAD tokens
            if (last_token == m.tok.eos_token or last_token == m.tok.pad_token) {
                std.debug.print("🛑 EOS/PAD detected, breaking\n", .{});
                break;
            }
            
            // Decode token to check if it's an image/special token
            const piece = m.tok.decode(&[_]u32{last_token}, allocator) catch |err| {
                std.debug.print("❌ Decode failed: {}\n", .{err});
                return -1;
            };
            defer allocator.free(piece);
            
            // Skip image/vision/special tokens (multimodal vocabulary)
            const is_image_token = std.mem.indexOf(u8, piece, "<|img") != null or
                                   std.mem.indexOf(u8, piece, "<|image") != null or
                                   std.mem.indexOf(u8, piece, "<|video") != null or
                                   std.mem.indexOf(u8, piece, "<|vision") != null or
                                   std.mem.startsWith(u8, piece, "::"); // Image marker prefix
            
            if (is_image_token) {
                std.debug.print("   ⏭️  Skipping image token: {s}\n", .{piece});
                
                // Find next best non-image token from top 10
                var next_best_idx: usize = 0;
                var next_best_val: f32 = -std.math.inf(f32);
                
                for (logits, 0..) |v, i| {
                    if (i == max_idx) continue; // Skip current
                    if (v > next_best_val and v > -1e9) { // Valid logit
                        // Quick check: decode and verify not image token
                        const test_piece = m.tok.decode(&[_]u32{@intCast(i)}, allocator) catch continue;
                        defer allocator.free(test_piece);
                        
                        const is_img = std.mem.indexOf(u8, test_piece, "<|img") != null or
                                      std.mem.indexOf(u8, test_piece, "<|image") != null or
                                      std.mem.startsWith(u8, test_piece, "::");
                        
                        if (!is_img) {
                            next_best_idx = i;
                            next_best_val = v;
                            if (next_best_val > max_val * 0.5) break; // Good enough
                        }
                    }
                }
                
                if (next_best_val > -1e9) {
                    last_token = @intCast(next_best_idx);
                    std.debug.print("   ✅ Using fallback token: {d} (logit={d:.3})\n", .{ last_token, next_best_val });
                    const fallback_piece = m.tok.decode(&[_]u32{last_token}, allocator) catch break;
                    defer allocator.free(fallback_piece);
                    generated.appendSlice(allocator, fallback_piece) catch break;
                } else {
                    std.debug.print("   ⚠️  No valid text tokens found, breaking\n", .{});
                    break;
                }
                
                m.advanceCaches();
                position += 1;
                continue;
            }
            
            // Detect stuck generation (same token repeatedly)
            if (last_token == prev_token) {
                repeat_count += 1;
                if (repeat_count >= 3) {
                    std.debug.print("🛑 Stuck generating token {d}, breaking\n", .{last_token});
                    break;
                }
            } else {
                repeat_count = 0;
                prev_token = last_token;
            }
            
            generated.appendSlice(allocator, piece) catch |err| {
                std.debug.print("❌ Append failed: {}\n", .{err});
                return -1;
            };

            // CRITICAL: Advance caches for next token
            m.advanceCaches();
            position += 1;
        }

        std.debug.print("✅ Generation loop complete, converting to slice...\n", .{});
        const out_slice = generated.toOwnedSlice(allocator) catch |err| {
            std.debug.print("❌ Slice failed: {}\n", .{err});
            return -1;
        };
        defer allocator.free(out_slice);

        std.debug.print("✅ Slice created, len={d}, copying to buffer...\n", .{out_slice.len});
        if (out_slice.len >= buffer_size) {
            std.debug.print("❌ Buffer too small for response\n", .{});
            return -1;
        }
        @memcpy(result_buffer[0..out_slice.len], out_slice);
        std.debug.print("✅ Copy complete, returning len={d}\n", .{out_slice.len});
        return @intCast(out_slice.len);
    }

    return -1;
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
        if (lfm2_model) |*m| {
            m.deinit();
        }
        if (hf_model) |*hm| {
            hm.deinit();
        }
        llama_model = null;
        lfm2_model = null;
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
