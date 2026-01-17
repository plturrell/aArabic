const std = @import("std");

/// HuggingFace Model Configuration Parser
/// Parses config.json files from HuggingFace models
/// Supports various architectures: LLaMA, Qwen, Mistral, etc.

// ============================================================================
// Model Architecture Types
// ============================================================================

pub const ModelArchitecture = enum {
    llama,
    qwen2,
    mistral,
    phi,
    gemma,
    unknown,
    
    pub fn fromString(s: []const u8) ModelArchitecture {
        if (std.mem.eql(u8, s, "LlamaForCausalLM")) return .llama;
        if (std.mem.eql(u8, s, "Qwen2ForCausalLM")) return .qwen2;
        if (std.mem.eql(u8, s, "MistralForCausalLM")) return .mistral;
        if (std.mem.eql(u8, s, "PhiForCausalLM")) return .phi;
        if (std.mem.eql(u8, s, "GemmaForCausalLM")) return .gemma;
        return .unknown;
    }
};

pub const ActivationType = enum {
    silu,
    gelu,
    gelu_new,
    relu,
    
    pub fn fromString(s: []const u8) ActivationType {
        if (std.mem.eql(u8, s, "silu")) return .silu;
        if (std.mem.eql(u8, s, "gelu")) return .gelu;
        if (std.mem.eql(u8, s, "gelu_new")) return .gelu_new;
        if (std.mem.eql(u8, s, "relu")) return .relu;
        return .silu; // Default
    }
};

pub const RopeScalingType = enum {
    none,
    linear,
    dynamic,
    yarn,
    
    pub fn fromString(s: []const u8) RopeScalingType {
        if (std.mem.eql(u8, s, "linear")) return .linear;
        if (std.mem.eql(u8, s, "dynamic")) return .dynamic;
        if (std.mem.eql(u8, s, "yarn")) return .yarn;
        return .none;
    }
};

// ============================================================================
// Model Configuration
// ============================================================================

pub const ModelConfig = struct {
    // Architecture
    architectures: [][]const u8,
    model_type: []const u8,
    architecture: ModelArchitecture,
    
    // Model dimensions
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    
    // Context and sequence
    max_position_embeddings: usize,
    sliding_window: ?usize,
    
    // Normalization
    rms_norm_eps: f32,
    layer_norm_eps: f32,
    
    // Activation
    hidden_act: ActivationType,
    
    // RoPE (Rotary Position Embedding)
    rope_theta: f32,
    rope_scaling: ?RopeScalingType,
    
    // Tokenizer
    bos_token_id: i32,
    eos_token_id: i32,
    pad_token_id: ?i32,
    
    // Training
    use_cache: bool,
    tie_word_embeddings: bool,
    
    // Allocator for cleanup
    allocator: std.mem.Allocator,
    
    pub fn deinit(self: *ModelConfig) void {
        for (self.architectures) |arch| {
            self.allocator.free(arch);
        }
        self.allocator.free(self.architectures);
        self.allocator.free(self.model_type);
    }
    
    /// Get head dimension
    pub fn headDim(self: ModelConfig) usize {
        return self.hidden_size / self.num_attention_heads;
    }
    
    /// Check if using Grouped Query Attention (GQA)
    pub fn isGQA(self: ModelConfig) bool {
        return self.num_key_value_heads < self.num_attention_heads;
    }
    
    /// Check if using Multi-Query Attention (MQA)
    pub fn isMQA(self: ModelConfig) bool {
        return self.num_key_value_heads == 1;
    }
    
    /// Get number of KV heads per attention head
    pub fn kvHeadsPerAttentionHead(self: ModelConfig) usize {
        return self.num_attention_heads / self.num_key_value_heads;
    }
};

// ============================================================================
// Configuration Parser
// ============================================================================

pub const ConfigParser = struct {
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator) ConfigParser {
        return .{ .allocator = allocator };
    }
    
    /// Load and parse config.json file
    pub fn loadFromFile(self: ConfigParser, config_path: []const u8) !ModelConfig {
        std.debug.print("\n📋 Loading model config: {s}\n", .{config_path});
        
        // Read file
        const file = try std.fs.cwd().openFile(config_path, .{});
        defer file.close();
        
        const file_size = (try file.stat()).size;
        const config_json = try self.allocator.alloc(u8, file_size);
        defer self.allocator.free(config_json);
        
        _ = try file.read(config_json);
        
        // Parse JSON
        return try self.parseConfig(config_json);
    }
    
    /// Parse config JSON
    fn parseConfig(self: ConfigParser, json_data: []const u8) !ModelConfig {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            self.allocator,
            json_data,
            .{},
        );
        defer parsed.deinit();
        
        const root = parsed.value.object;
        
        // Parse architectures array
        const arch_array = root.get("architectures") orelse return error.MissingArchitectures;
        var architectures = std.ArrayList([]const u8){};
        
        if (arch_array == .array) {
            for (arch_array.array.items) |item| {
                if (item == .string) {
                    try architectures.append(self.allocator, try self.allocator.dupe(u8, item.string));
                }
            }
        }
        
        // Determine architecture type
        const model_type_str = if (root.get("model_type")) |mt|
            if (mt == .string) mt.string else "unknown"
        else
            "unknown";
        
        const architecture = if (architectures.items.len > 0)
            ModelArchitecture.fromString(architectures.items[0])
        else
            .unknown;
        
        // Parse dimensions
        const vocab_size = try self.getInt(root, "vocab_size");
        const hidden_size = try self.getInt(root, "hidden_size");
        const intermediate_size = try self.getInt(root, "intermediate_size");
        const num_hidden_layers = try self.getInt(root, "num_hidden_layers");
        const num_attention_heads = try self.getInt(root, "num_attention_heads");
        
        // KV heads (may not exist in all configs)
        const num_key_value_heads = self.getIntOptional(root, "num_key_value_heads") orelse num_attention_heads;
        
        // Context length
        const max_position_embeddings = try self.getInt(root, "max_position_embeddings");
        const sliding_window = self.getIntOptional(root, "sliding_window");
        
        // Normalization
        const rms_norm_eps = self.getFloat(root, "rms_norm_eps") catch 1e-5;
        const layer_norm_eps = self.getFloat(root, "layer_norm_eps") catch 1e-5;
        
        // Activation
        const hidden_act_str = if (root.get("hidden_act")) |ha|
            if (ha == .string) ha.string else "silu"
        else
            "silu";
        const hidden_act = ActivationType.fromString(hidden_act_str);
        
        // RoPE
        const rope_theta = self.getFloat(root, "rope_theta") catch 10000.0;
        const rope_scaling = self.parseRopeScaling(root);
        
        // Tokenizer IDs
        const bos_token_id = self.getIntOptional(root, "bos_token_id") orelse 1;
        const eos_token_id = self.getIntOptional(root, "eos_token_id") orelse 2;
        const pad_token_id = self.getIntOptional(root, "pad_token_id");
        
        // Training config
        const use_cache = self.getBool(root, "use_cache") catch true;
        const tie_word_embeddings = self.getBool(root, "tie_word_embeddings") catch false;
        
        std.debug.print("   Model type: {s}\n", .{model_type_str});
        std.debug.print("   Architecture: {s}\n", .{@tagName(architecture)});
        std.debug.print("   Vocab size: {d}\n", .{vocab_size});
        std.debug.print("   Hidden size: {d}\n", .{hidden_size});
        std.debug.print("   Layers: {d}\n", .{num_hidden_layers});
        std.debug.print("   Attention heads: {d}\n", .{num_attention_heads});
        std.debug.print("   KV heads: {d}\n", .{num_key_value_heads});
        std.debug.print("   Max position: {d}\n", .{max_position_embeddings});
        std.debug.print("✅ Config loaded successfully\n", .{});
        
        return ModelConfig{
            .architectures = try architectures.toOwnedSlice(self.allocator),
            .model_type = try self.allocator.dupe(u8, model_type_str),
            .architecture = architecture,
            .vocab_size = vocab_size,
            .hidden_size = hidden_size,
            .intermediate_size = intermediate_size,
            .num_hidden_layers = num_hidden_layers,
            .num_attention_heads = num_attention_heads,
            .num_key_value_heads = num_key_value_heads,
            .max_position_embeddings = max_position_embeddings,
            .sliding_window = sliding_window,
            .rms_norm_eps = rms_norm_eps,
            .layer_norm_eps = layer_norm_eps,
            .hidden_act = hidden_act,
            .rope_theta = rope_theta,
            .rope_scaling = rope_scaling,
            .bos_token_id = @intCast(bos_token_id),
            .eos_token_id = @intCast(eos_token_id),
            .pad_token_id = if (pad_token_id) |pid| @as(i32, @intCast(pid)) else null,
            .use_cache = use_cache,
            .tie_word_embeddings = tie_word_embeddings,
            .allocator = self.allocator,
        };
    }
    
    /// Parse RoPE scaling configuration
    fn parseRopeScaling(self: ConfigParser, root: std.json.ObjectMap) ?RopeScalingType {
        _ = self;
        const rope_scaling = root.get("rope_scaling") orelse return null;
        
        if (rope_scaling == .object) {
            const scaling_type = rope_scaling.object.get("type") orelse return null;
            if (scaling_type == .string) {
                return RopeScalingType.fromString(scaling_type.string);
            }
        }
        
        return null;
    }
    
    /// Helper: Get integer value
    fn getInt(self: ConfigParser, obj: std.json.ObjectMap, key: []const u8) !usize {
        _ = self;
        const value = obj.get(key) orelse return error.MissingKey;
        if (value != .integer) return error.WrongType;
        return @intCast(value.integer);
    }
    
    /// Helper: Get optional integer value
    fn getIntOptional(self: ConfigParser, obj: std.json.ObjectMap, key: []const u8) ?usize {
        _ = self;
        const value = obj.get(key) orelse return null;
        if (value != .integer) return null;
        return @intCast(value.integer);
    }
    
    /// Helper: Get float value
    fn getFloat(self: ConfigParser, obj: std.json.ObjectMap, key: []const u8) !f32 {
        _ = self;
        const value = obj.get(key) orelse return error.MissingKey;
        return switch (value) {
            .float => @floatCast(value.float),
            .integer => @floatFromInt(value.integer),
            else => error.WrongType,
        };
    }
    
    /// Helper: Get boolean value
    fn getBool(self: ConfigParser, obj: std.json.ObjectMap, key: []const u8) !bool {
        _ = self;
        const value = obj.get(key) orelse return error.MissingKey;
        if (value != .bool) return error.WrongType;
        return value.bool;
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_config_parser(allocator: std.mem.Allocator, config_path: []const u8) !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  CONFIG PARSER TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    const parser = ConfigParser.init(allocator);
    var config = try parser.loadFromFile(config_path);
    defer config.deinit();
    
    // Display config details
    std.debug.print("\n📊 Model Configuration:\n", .{});
    std.debug.print("   Architecture: {s}\n", .{@tagName(config.architecture)});
    std.debug.print("   Model Type: {s}\n", .{config.model_type});
    std.debug.print("\n   Dimensions:\n", .{});
    std.debug.print("     Vocab Size: {d}\n", .{config.vocab_size});
    std.debug.print("     Hidden Size: {d}\n", .{config.hidden_size});
    std.debug.print("     Intermediate Size: {d}\n", .{config.intermediate_size});
    std.debug.print("     Layers: {d}\n", .{config.num_hidden_layers});
    std.debug.print("     Attention Heads: {d}\n", .{config.num_attention_heads});
    std.debug.print("     KV Heads: {d}\n", .{config.num_key_value_heads});
    std.debug.print("     Head Dim: {d}\n", .{config.headDim()});
    std.debug.print("\n   Context:\n", .{});
    std.debug.print("     Max Position: {d}\n", .{config.max_position_embeddings});
    if (config.sliding_window) |sw| {
        std.debug.print("     Sliding Window: {d}\n", .{sw});
    }
    std.debug.print("\n   Attention Type:\n", .{});
    if (config.isGQA()) {
        std.debug.print("     Grouped Query Attention (GQA)\n", .{});
        std.debug.print("     Heads per KV head: {d}\n", .{config.kvHeadsPerAttentionHead()});
    } else if (config.isMQA()) {
        std.debug.print("     Multi-Query Attention (MQA)\n", .{});
    } else {
        std.debug.print("     Multi-Head Attention (MHA)\n", .{});
    }
    std.debug.print("\n   Activation: {s}\n", .{@tagName(config.hidden_act)});
    std.debug.print("   RoPE Theta: {d:.1}\n", .{config.rope_theta});
    if (config.rope_scaling) |rs| {
        std.debug.print("   RoPE Scaling: {s}\n", .{@tagName(rs)});
    }
    std.debug.print("\n   Token IDs:\n", .{});
    std.debug.print("     BOS: {d}\n", .{config.bos_token_id});
    std.debug.print("     EOS: {d}\n", .{config.eos_token_id});
    if (config.pad_token_id) |pid| {
        std.debug.print("     PAD: {d}\n", .{pid});
    }
    
    std.debug.print("\n✅ Config parser test complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
