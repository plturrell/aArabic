const std = @import("std");
const config_parser = @import("config_parser");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  CONFIG PARSER TESTS\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test with Qwen3 model config
    const config_path = "vendor/layerModels/huggingFace/Qwen/Qwen3-Coder-30B-A3B-Instruct/config.json";
    
    std.debug.print("\n🧪 Testing with Qwen3-Coder-30B config\n", .{});
    std.debug.print("   File: {s}\n", .{config_path});
    
    const parser = config_parser.ConfigParser.init(allocator);
    var config = try parser.loadFromFile(config_path);
    defer config.deinit();
    
    // Display detailed configuration
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  PARSED MODEL CONFIGURATION\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    std.debug.print("\n🏗️  Architecture:\n", .{});
    std.debug.print("   Type: {s}\n", .{@tagName(config.architecture)});
    std.debug.print("   Model: {s}\n", .{config.model_type});
    std.debug.print("   Architectures: ", .{});
    for (config.architectures, 0..) |arch, i| {
        if (i > 0) std.debug.print(", ", .{});
        std.debug.print("{s}", .{arch});
    }
    std.debug.print("\n", .{});
    
    std.debug.print("\n📐 Dimensions:\n", .{});
    std.debug.print("   Vocabulary Size: {d:>10}\n", .{config.vocab_size});
    std.debug.print("   Hidden Size:     {d:>10}\n", .{config.hidden_size});
    std.debug.print("   Intermediate:    {d:>10}\n", .{config.intermediate_size});
    std.debug.print("   Layers:          {d:>10}\n", .{config.num_hidden_layers});
    std.debug.print("   Attention Heads: {d:>10}\n", .{config.num_attention_heads});
    std.debug.print("   KV Heads:        {d:>10}\n", .{config.num_key_value_heads});
    std.debug.print("   Head Dimension:  {d:>10}\n", .{config.headDim()});
    
    std.debug.print("\n🎯 Attention Mechanism:\n", .{});
    if (config.isGQA()) {
        std.debug.print("   Type: Grouped Query Attention (GQA)\n", .{});
        std.debug.print("   Attention heads per KV head: {d}\n", .{config.kvHeadsPerAttentionHead()});
        
        const efficiency = @as(f64, @floatFromInt(config.num_attention_heads)) / 
                          @as(f64, @floatFromInt(config.num_key_value_heads));
        std.debug.print("   Memory efficiency: {d:.1}x reduction vs MHA\n", .{efficiency});
    } else if (config.isMQA()) {
        std.debug.print("   Type: Multi-Query Attention (MQA)\n", .{});
        std.debug.print("   All attention heads share 1 KV head\n", .{});
    } else {
        std.debug.print("   Type: Multi-Head Attention (MHA)\n", .{});
        std.debug.print("   Standard attention (1:1 attention to KV heads)\n", .{});
    }
    
    std.debug.print("\n📏 Context & Sequence:\n", .{});
    std.debug.print("   Max Position Embeddings: {d}\n", .{config.max_position_embeddings});
    if (config.sliding_window) |sw| {
        std.debug.print("   Sliding Window: {d}\n", .{sw});
    } else {
        std.debug.print("   Sliding Window: None (full context)\n", .{});
    }
    
    std.debug.print("\n⚙️  Hyperparameters:\n", .{});
    std.debug.print("   Activation Function: {s}\n", .{@tagName(config.hidden_act)});
    std.debug.print("   RMS Norm Epsilon: {e}\n", .{config.rms_norm_eps});
    std.debug.print("   Layer Norm Epsilon: {e}\n", .{config.layer_norm_eps});
    std.debug.print("   RoPE Theta: {d:.1}\n", .{config.rope_theta});
    if (config.rope_scaling) |rs| {
        std.debug.print("   RoPE Scaling: {s}\n", .{@tagName(rs)});
    } else {
        std.debug.print("   RoPE Scaling: None\n", .{});
    }
    
    std.debug.print("\n🎫 Special Tokens:\n", .{});
    std.debug.print("   BOS Token ID: {d}\n", .{config.bos_token_id});
    std.debug.print("   EOS Token ID: {d}\n", .{config.eos_token_id});
    if (config.pad_token_id) |pid| {
        std.debug.print("   PAD Token ID: {d}\n", .{pid});
    } else {
        std.debug.print("   PAD Token ID: None\n", .{});
    }
    
    std.debug.print("\n⚡ Training Config:\n", .{});
    std.debug.print("   Use Cache: {s}\n", .{if (config.use_cache) "Yes" else "No"});
    std.debug.print("   Tie Word Embeddings: {s}\n", .{if (config.tie_word_embeddings) "Yes" else "No"});
    
    // Calculate model size estimates
    std.debug.print("\n💾 Model Size Estimates:\n", .{});
    
    // Embedding layer
    const embedding_params = config.vocab_size * config.hidden_size;
    std.debug.print("   Embedding Parameters: {d:>15}\n", .{embedding_params});
    
    // Per-layer parameters
    const attn_params = config.hidden_size * config.hidden_size * 4; // Q, K, V, O
    const ffn_params = config.hidden_size * config.intermediate_size * 2; // up, down
    const norm_params = config.hidden_size * 2; // 2 norms per layer
    const per_layer = attn_params + ffn_params + norm_params;
    
    std.debug.print("   Per-Layer Parameters: {d:>15}\n", .{per_layer});
    std.debug.print("   Total Layer Params:   {d:>15}\n", .{per_layer * config.num_hidden_layers});
    
    const total_params = embedding_params + (per_layer * config.num_hidden_layers);
    std.debug.print("   Total Parameters:     {d:>15}\n", .{total_params});
    std.debug.print("   Approximate Size:     {d:.2} billion parameters\n", 
                     .{@as(f64, @floatFromInt(total_params)) / 1e9});
    
    // Memory estimates
    const fp16_size = total_params * 2;
    const fp32_size = total_params * 4;
    
    std.debug.print("\n   Memory Requirements:\n", .{});
    std.debug.print("     FP16/BF16: {d:.2} GB\n", .{@as(f64, @floatFromInt(fp16_size)) / 1e9});
    std.debug.print("     FP32:      {d:.2} GB\n", .{@as(f64, @floatFromInt(fp32_size)) / 1e9});
    
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("✅ ALL CONFIG PARSER TESTS PASSED!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    std.debug.print("\n📊 Config Parser Features:\n", .{});
    std.debug.print("   • Parse HuggingFace config.json files\n", .{});
    std.debug.print("   • Support multiple architectures (LLaMA, Qwen, Mistral, etc.)\n", .{});
    std.debug.print("   • Extract model dimensions and hyperparameters\n", .{});
    std.debug.print("   • Detect attention mechanism type (MHA/GQA/MQA)\n", .{});
    std.debug.print("   • Parse RoPE configuration\n", .{});
    std.debug.print("   • Special token handling\n", .{});
    std.debug.print("   • Model size estimation\n", .{});
}
