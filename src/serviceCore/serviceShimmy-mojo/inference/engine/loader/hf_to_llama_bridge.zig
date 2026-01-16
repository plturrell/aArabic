const std = @import("std");
const hf = @import("huggingface_loader");
const llama = @import("llama_model");
const transformer = @import("transformer");
const tokenizer = @import("tokenizer");
const matrix_ops = @import("matrix_ops");
const gguf = @import("gguf_loader");

/// Bridge between HuggingFace models and LLaMA inference engine
/// Converts HF SafeTensors format → LLaMA weight format

// ============================================================================
// Tensor Name Mapping
// ============================================================================

/// HuggingFace → LLaMA tensor name mapping
const TensorNameMap = struct {
    // Token embeddings
    pub const TOKEN_EMBEDDING = "model.embed_tokens.weight";
    
    // Output head
    pub const OUTPUT_NORM = "model.norm.weight";
    pub const OUTPUT_WEIGHT = "lm_head.weight";
    
    // Per-layer tensors (format: "model.layers.{layer}.{component}")
    pub fn layerAttnNorm(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.input_layernorm.weight", .{layer});
    }
    
    pub fn layerQProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.q_proj.weight", .{layer});
    }
    
    pub fn layerKProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.k_proj.weight", .{layer});
    }
    
    pub fn layerVProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.v_proj.weight", .{layer});
    }
    
    pub fn layerOProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.self_attn.o_proj.weight", .{layer});
    }
    
    pub fn layerFFNNorm(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.post_attention_layernorm.weight", .{layer});
    }
    
    pub fn layerGateProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.gate_proj.weight", .{layer});
    }
    
    pub fn layerUpProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.up_proj.weight", .{layer});
    }
    
    pub fn layerDownProj(layer: usize, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "model.layers.{d}.mlp.down_proj.weight", .{layer});
    }
};

// ============================================================================
// HF → LLaMA Conversion
// ============================================================================

pub fn convertHFToLLaMA(
    allocator: std.mem.Allocator,
    hf_model: *hf.HuggingFaceModel,
) !llama.LlamaModel {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  CONVERTING HUGGINGFACE → LLAMA\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Step 1: Create LLaMA config from HF config
    const config = try createLLaMAConfig(hf_model);
    
    std.debug.print("\n📋 Configuration:\n", .{});
    std.debug.print("   Vocab size: {d}\n", .{config.vocab_size});
    std.debug.print("   Layers: {d}\n", .{config.n_layers});
    std.debug.print("   Embedding: {d}\n", .{config.embed_dim});
    std.debug.print("   FFN dim: {d}\n", .{config.ffn_dim});
    std.debug.print("   Heads: {d}\n", .{config.n_heads});
    std.debug.print("   KV heads: {d}\n", .{config.n_kv_heads});
    
    // Step 2: Convert weights
    const weights = try convertWeights(allocator, hf_model, config);
    
    // Step 3: Create tokenizer wrapper
    const tok = try createTokenizerWrapper(allocator, hf_model);
    
    // Step 4: Initialize LLaMA model
    const model = try llama.LlamaModel.init(allocator, config, weights, tok);

    std.debug.print("\n✅ Conversion complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
    
    return model;
}

fn createLLaMAConfig(hf_model: *hf.HuggingFaceModel) !llama.LlamaConfig {
    const hf_config = hf_model.config;
    
    return llama.LlamaConfig{
        .vocab_size = @as(u32, @intCast(hf_config.vocab_size)),
        .n_layers = @as(u32, @intCast(hf_config.num_hidden_layers)),
        .embed_dim = @as(u32, @intCast(hf_config.hidden_size)),
        .ffn_dim = @as(u32, @intCast(hf_config.intermediate_size)),
        .n_heads = @as(u32, @intCast(hf_config.num_attention_heads)),
        .n_kv_heads = @as(u32, @intCast(hf_config.num_key_value_heads)),
        .head_dim = @as(u32, @intCast(hf_config.hidden_size / hf_config.num_attention_heads)),
        .max_seq_len = @as(u32, @intCast(hf_config.max_position_embeddings)),
        .rope_theta = hf_config.rope_theta,
        .rms_norm_eps = 1e-5,
    };
}

fn convertWeights(
    allocator: std.mem.Allocator,
    hf_model: *hf.HuggingFaceModel,
    config: llama.LlamaConfig,
) !llama.LlamaWeights {
    std.debug.print("\n⚖️  Converting weights...\n", .{});
    
    // Load token embedding
    std.debug.print("   Loading token embedding...\n", .{});
    const token_embedding_data = try hf_model.getTensor(TensorNameMap.TOKEN_EMBEDDING);
    const token_embedding = matrix_ops.Weight{ .f32 = token_embedding_data };
    
    // Load output norm
    std.debug.print("   Loading output norm...\n", .{});
    const output_norm = try hf_model.getTensor(TensorNameMap.OUTPUT_NORM);
    
    // Load output weight (lm_head)
    std.debug.print("   Loading output weight...\n", .{});
    const output_weight_data = try hf_model.getTensor(TensorNameMap.OUTPUT_WEIGHT);
    const output_weight = matrix_ops.Weight{ .f32 = output_weight_data };
    
    // Load per-layer weights
    std.debug.print("   Loading {d} layers...\n", .{config.n_layers});
    const layer_weights = try allocator.alloc(transformer.TransformerWeights, config.n_layers);
    errdefer allocator.free(layer_weights);
    
    for (0..config.n_layers) |layer_idx| {
        if (layer_idx % 5 == 0) {
            std.debug.print("      Layer {d}/{d}...\n", .{ layer_idx + 1, config.n_layers });
        }
        
        // Attention norm
        const attn_norm_name = try TensorNameMap.layerAttnNorm(layer_idx, allocator);
        defer allocator.free(attn_norm_name);
        const attn_norm = try hf_model.getTensor(attn_norm_name);
        
        // Q, K, V, O projections
        const q_proj_name = try TensorNameMap.layerQProj(layer_idx, allocator);
        defer allocator.free(q_proj_name);
        const wq = try hf_model.getTensor(q_proj_name);
        
        const k_proj_name = try TensorNameMap.layerKProj(layer_idx, allocator);
        defer allocator.free(k_proj_name);
        const wk = try hf_model.getTensor(k_proj_name);
        
        const v_proj_name = try TensorNameMap.layerVProj(layer_idx, allocator);
        defer allocator.free(v_proj_name);
        const wv = try hf_model.getTensor(v_proj_name);
        
        const o_proj_name = try TensorNameMap.layerOProj(layer_idx, allocator);
        defer allocator.free(o_proj_name);
        const wo = try hf_model.getTensor(o_proj_name);
        
        // FFN norm
        const ffn_norm_name = try TensorNameMap.layerFFNNorm(layer_idx, allocator);
        defer allocator.free(ffn_norm_name);
        const ffn_norm = try hf_model.getTensor(ffn_norm_name);
        
        // FFN projections
        const gate_proj_name = try TensorNameMap.layerGateProj(layer_idx, allocator);
        defer allocator.free(gate_proj_name);
        const w_gate = try hf_model.getTensor(gate_proj_name);
        
        const up_proj_name = try TensorNameMap.layerUpProj(layer_idx, allocator);
        defer allocator.free(up_proj_name);
        const w_up = try hf_model.getTensor(up_proj_name);
        
        const down_proj_name = try TensorNameMap.layerDownProj(layer_idx, allocator);
        defer allocator.free(down_proj_name);
        const w_down = try hf_model.getTensor(down_proj_name);
        
        layer_weights[layer_idx] = transformer.TransformerWeights{
            .allocator = allocator,
            .attn_norm = attn_norm,
            .wq = .{ .f32 = wq },
            .wk = .{ .f32 = wk },
            .wv = .{ .f32 = wv },
            .wo = .{ .f32 = wo },
            .ffn_norm = ffn_norm,
            .w_gate = .{ .f32 = w_gate },
            .w_up = .{ .f32 = w_up },
            .w_down = .{ .f32 = w_down },
        };
    }
    
    std.debug.print("   ✅ All weights loaded\n", .{});
    
    return llama.LlamaWeights{
        .allocator = allocator,
        .token_embedding = token_embedding,
        .output_norm = output_norm,
        .output_weight = output_weight,
        .layer_weights = layer_weights,
    };
}

fn createTokenizerWrapper(
    allocator: std.mem.Allocator,
    hf_model: *hf.HuggingFaceModel,
) !tokenizer.Tokenizer {
    std.debug.print("\n📝 Creating tokenizer wrapper...\n", .{});
    
    // Create dummy GGUF model for tokenizer interface
    // (In production, we'd create a proper HF tokenizer wrapper)
    var dummy_model = gguf.GGUFModel{
        .allocator = allocator,
        .file = undefined,
        .header = undefined,
        .metadata = .{
            .architecture = .Llama,
            .vocab_size = @as(u32, @intCast(hf_model.config.vocab_size)),
            .n_layers = @as(u32, @intCast(hf_model.config.num_hidden_layers)),
            .n_heads = @as(u32, @intCast(hf_model.config.num_attention_heads)),
            .n_kv_heads = @as(u32, @intCast(hf_model.config.num_key_value_heads)),
            .hidden_size = @as(u32, @intCast(hf_model.config.hidden_size)),
            .intermediate_size = @as(u32, @intCast(hf_model.config.intermediate_size)),
            .max_seq_len = @as(u32, @intCast(hf_model.config.max_position_embeddings)),
            .rope_theta = hf_model.config.rope_theta,
        },
        .tensors = &[_]gguf.TensorInfo{},
        .vocab_tokens = &[_][]u8{},
        .vocab_scores = &[_]f32{},
    };
    
    var tok = try tokenizer.Tokenizer.loadFromModel(allocator, &dummy_model);
    
    // Override with HF tokenizer EOS/BOS
    tok.bos_token = hf_model.tokenizer.bos_token_id;
    tok.eos_token = hf_model.tokenizer.eos_token_id;
    
    std.debug.print("   BOS token: {d}\n", .{tok.bos_token});
    std.debug.print("   EOS token: {d}\n", .{tok.eos_token});
    std.debug.print("   ✅ Tokenizer wrapper created\n", .{});
    
    return tok;
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_hf_to_llama_bridge(
    allocator: std.mem.Allocator,
    model_path: []const u8,
) !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  HF → LLAMA BRIDGE TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Load HF model
    std.debug.print("\n1️⃣  Loading HuggingFace model...\n", .{});
    var hf_model = try hf.loadModel(allocator, model_path);
    defer hf_model.deinit();
    
    // Convert to LLaMA format
    std.debug.print("\n2️⃣  Converting to LLaMA format...\n", .{});
    var llama_model = try convertHFToLLaMA(allocator, &hf_model);
    defer llama_model.deinit();
    
    // Test forward pass
    std.debug.print("\n3️⃣  Testing forward pass...\n", .{});
    const logits = try llama_model.forward(1, 0);
    defer allocator.free(logits);
    
    std.debug.print("   Logits shape: {d}\n", .{logits.len});
    std.debug.print("   First few logits: {d:.4}, {d:.4}, {d:.4}\n", .{
        logits[0], logits[1], logits[2]
    });
    
    std.debug.print("\n✅ Bridge test complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
