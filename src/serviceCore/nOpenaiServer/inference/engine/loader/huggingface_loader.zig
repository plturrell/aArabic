const std = @import("std");
const config_parser = @import("config_parser");
const safetensors = @import("safetensors_sharded");
const safetensors_loader = @import("safetensors_loader");
const bpe = @import("bpe_tokenizer");

/// HuggingFace Model Loader
/// Integrates SafeTensors, Config Parser, and BPE Tokenizer
/// Provides unified interface for loading HuggingFace models

// ============================================================================
// Model Structure
// ============================================================================

pub const HuggingFaceModel = struct {
    config: config_parser.ModelConfig,
    weights: safetensors.SafeTensorsSharded,
    tokenizer: bpe.BPETokenizer,
    allocator: std.mem.Allocator,
    base_path: []const u8,
    
    pub fn init(allocator: std.mem.Allocator, base_path: []const u8) HuggingFaceModel {
        return .{
            .config = undefined,
            .weights = safetensors.SafeTensorsSharded.init(allocator, base_path),
            .tokenizer = bpe.BPETokenizer.init(allocator),
            .allocator = allocator,
            .base_path = base_path,
        };
    }
    
    pub fn deinit(self: *HuggingFaceModel) void {
        self.config.deinit();
        self.weights.deinit();
        self.tokenizer.deinit();
    }
    
    /// Load complete model (config + weights + tokenizer)
    pub fn load(self: *HuggingFaceModel) !void {
        std.debug.print("\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("  LOADING HUGGINGFACE MODEL\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("\n📂 Base path: {s}\n", .{self.base_path});
        
        // Load configuration
        try self.loadConfig();
        
        // Load weights
        try self.loadWeights();
        
        // Load tokenizer
        try self.loadTokenizer();
        
        std.debug.print("\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        std.debug.print("✅ HUGGINGFACE MODEL LOADED SUCCESSFULLY\n", .{});
        std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
        
        self.printSummary();
    }
    
    /// Load model configuration
    fn loadConfig(self: *HuggingFaceModel) !void {
        std.debug.print("\n🔧 Loading configuration...\n", .{});
        
        const config_path = try std.fs.path.join(
            self.allocator,
            &[_][]const u8{ self.base_path, "config.json" },
        );
        defer self.allocator.free(config_path);
        
        const parser = config_parser.ConfigParser.init(self.allocator);
        self.config = try parser.loadFromFile(config_path);
    }
    
    /// Load model weights
    fn loadWeights(self: *HuggingFaceModel) !void {
        std.debug.print("\n⚖️  Loading weights...\n", .{});
        
        // Check if this is a sharded model (has index.json)
        const index_path = try std.fs.path.join(
            self.allocator,
            &[_][]const u8{ self.base_path, "model.safetensors.index.json" },
        );
        defer self.allocator.free(index_path);
        
        // Try to open index file
        const index_file = std.fs.cwd().openFile(index_path, .{}) catch |err| {
            if (err == error.FileNotFound) {
                // Single file model - load model.safetensors directly
                std.debug.print("   Single-file model detected\n", .{});
                
                const model_path = try std.fs.path.join(
                    self.allocator,
                    &[_][]const u8{ self.base_path, "model.safetensors" },
                );
                // Don't free model_path - it's stored by SafeTensorsFile

                // Load the single safetensors file as a "shard"
                const shard_name = try self.allocator.dupe(u8, "model.safetensors");
                try self.weights.shard_files.append(shard_name);

                var shard = safetensors_loader.SafeTensorsFile.init(self.allocator, model_path);
                try shard.load();
                
                // Add all tensors to weight map
                var tensor_it = shard.header.tensors.iterator();
                while (tensor_it.next()) |entry| {
                    try self.weights.index.weight_map.put(
                        try self.allocator.dupe(u8, entry.key_ptr.*),
                        try self.allocator.dupe(u8, "model.safetensors"),
                    );
                }
                
                try self.weights.shards.put(shard_name, shard);
                
                std.debug.print("   Loaded {d} tensors from single file\n", .{self.weights.index.weight_map.count()});
                return;
            }
            return err;
        };
        index_file.close();
        
        // Sharded model
        try self.weights.loadFromIndex(index_path);
    }
    
    /// Load tokenizer
    fn loadTokenizer(self: *HuggingFaceModel) !void {
        std.debug.print("\n📝 Loading tokenizer...\n", .{});

        const vocab_path = try std.fs.path.join(
            self.allocator,
            &[_][]const u8{ self.base_path, "vocab.json" },
        );
        const merges_path = try std.fs.path.join(
            self.allocator,
            &[_][]const u8{ self.base_path, "merges.txt" },
        );
        const tokenizer_json_path = try std.fs.path.join(
            self.allocator,
            &[_][]const u8{ self.base_path, "tokenizer.json" },
        );
        defer self.allocator.free(vocab_path);
        defer self.allocator.free(merges_path);
        defer self.allocator.free(tokenizer_json_path);

        const has_vocab = blk: {
            const res = std.fs.cwd().access(vocab_path, .{}) catch |err| {
                if (err == error.FileNotFound) break :blk false;
                return err;
            };
            _ = res;
            break :blk true;
        };
        const has_merges = blk: {
            const res = std.fs.cwd().access(merges_path, .{}) catch |err| {
                if (err == error.FileNotFound) break :blk false;
                return err;
            };
            _ = res;
            break :blk true;
        };
        if (has_vocab and has_merges) {
            try self.tokenizer.loadVocab(vocab_path);
            try self.tokenizer.loadMerges(merges_path);
            return;
        }

        const has_tokenizer_json = blk: {
            const res = std.fs.cwd().access(tokenizer_json_path, .{}) catch |err| {
                if (err == error.FileNotFound) break :blk false;
                return err;
            };
            _ = res;
            break :blk true;
        };
        if (has_tokenizer_json) {
            try self.loadTokenizerFromJson(tokenizer_json_path);
            return;
        }

        return error.FileNotFound;
    }

    fn loadTokenizerFromJson(self: *HuggingFaceModel, path: []const u8) !void {
        std.debug.print("   Loading tokenizer.json: {s}\n", .{path});
        var file = try std.fs.cwd().openFile(path, .{});
        defer file.close();
        const size = (try file.stat()).size;
        const data = try self.allocator.alloc(u8, size);
        defer self.allocator.free(data);
        _ = try file.read(data);

        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, data, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const root = parsed.value;
        if (root != .object) return error.InvalidTokenizerFormat;
        const model_val_opt = root.object.get("model") orelse return error.InvalidTokenizerFormat;
        if (model_val_opt != .object) return error.InvalidTokenizerFormat;
        const vocab_val_opt = model_val_opt.object.get("vocab") orelse return error.InvalidTokenizerFormat;
        if (vocab_val_opt != .object) return error.InvalidTokenizerFormat;
        var vocab_it = vocab_val_opt.object.iterator();
        while (vocab_it.next()) |entry| {
            if (entry.value_ptr.* == .integer) {
                const id: u32 = @intCast(entry.value_ptr.*.integer);
                try self.tokenizer.vocab.add(entry.key_ptr.*, id);
            }
        }
        const merges_val_opt = model_val_opt.object.get("merges") orelse return error.InvalidTokenizerFormat;
        if (merges_val_opt != .array) return error.InvalidTokenizerFormat;
        var rank: usize = 0;
        for (merges_val_opt.array.items) |merge_item| {
            if (merge_item != .string) continue;
            const merge_str = merge_item.string;
            const merge_key = try std.fmt.allocPrint(self.allocator, "{s}", .{merge_str});
            try self.tokenizer.merges.put(merge_key, rank);
            rank += 1;
        }
        std.debug.print("   Loaded {d} tokens from tokenizer.json\n", .{self.tokenizer.vocab.size()});
        std.debug.print("   Loaded {d} merges from tokenizer.json\n", .{rank});
    }
    
    /// Print model summary
    fn printSummary(self: *HuggingFaceModel) void {
        std.debug.print("\n📊 Model Summary:\n", .{});
        std.debug.print("\n   Architecture:\n", .{});
        std.debug.print("     Type: {s}\n", .{@tagName(self.config.architecture)});
        std.debug.print("     Model: {s}\n", .{self.config.model_type});
        std.debug.print("     Layers: {d}\n", .{self.config.num_hidden_layers});
        
        std.debug.print("\n   Dimensions:\n", .{});
        std.debug.print("     Hidden size: {d}\n", .{self.config.hidden_size});
        std.debug.print("     Attention heads: {d}\n", .{self.config.num_attention_heads});
        std.debug.print("     KV heads: {d}\n", .{self.config.num_key_value_heads});
        
        if (self.config.isGQA()) {
            std.debug.print("     Attention: GQA ({d}:1 ratio)\n", .{self.config.kvHeadsPerAttentionHead()});
        } else if (self.config.isMQA()) {
            std.debug.print("     Attention: MQA\n", .{});
        } else {
            std.debug.print("     Attention: MHA\n", .{});
        }
        
        std.debug.print("\n   Weights:\n", .{});
        std.debug.print("     Tensor count: {d}\n", .{self.weights.index.weight_map.count()});
        std.debug.print("     Shard count: {d}\n", .{self.weights.shard_files.items.len});
        
        std.debug.print("\n   Tokenizer:\n", .{});
        std.debug.print("     Vocab size: {d}\n", .{self.tokenizer.vocabSize()});
        std.debug.print("     BPE merges: {d}\n", .{self.tokenizer.merges.count()});
        std.debug.print("     BOS token: {d}\n", .{self.tokenizer.bos_token_id});
        std.debug.print("     EOS token: {d}\n", .{self.tokenizer.eos_token_id});
    }
    
    /// Get tensor from model
    pub fn getTensor(self: *HuggingFaceModel, tensor_name: []const u8) ![]f32 {
        return try self.weights.getTensor(tensor_name);
    }
    
    /// Check if tensor exists
    pub fn hasTensor(self: *HuggingFaceModel, tensor_name: []const u8) bool {
        return self.weights.hasTensor(tensor_name);
    }
    
    /// Encode text to token IDs
    pub fn encode(self: *HuggingFaceModel, text: []const u8) ![]u32 {
        return try self.tokenizer.encode(text);
    }
    
    /// Decode token IDs to text
    pub fn decode(self: *HuggingFaceModel, token_ids: []const u32) ![]const u8 {
        return try self.tokenizer.decode(token_ids);
    }
};

// ============================================================================
// Convenience Functions
// ============================================================================

/// Quick load a HuggingFace model
pub fn loadModel(allocator: std.mem.Allocator, model_path: []const u8) !HuggingFaceModel {
    var model = HuggingFaceModel.init(allocator, model_path);
    try model.load();
    return model;
}

/// Check if directory contains a HuggingFace model
pub fn isHuggingFaceModel(allocator: std.mem.Allocator, path: []const u8) bool {
    // Check for required files
    const config_path = std.fs.path.join(
        allocator,
        &[_][]const u8{ path, "config.json" },
    ) catch return false;
    defer allocator.free(config_path);
    
    const vocab_path = std.fs.path.join(
        allocator,
        &[_][]const u8{ path, "vocab.json" },
    ) catch return false;
    defer allocator.free(vocab_path);
    
    // Try to open files
    const config_file = std.fs.cwd().openFile(config_path, .{}) catch return false;
    config_file.close();
    
    const vocab_file = std.fs.cwd().openFile(vocab_path, .{}) catch return false;
    vocab_file.close();
    
    return true;
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_huggingface_loader(allocator: std.mem.Allocator, model_path: []const u8) !void {
    std.debug.print("\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    std.debug.print("  HUGGINGFACE LOADER TEST\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    std.debug.print("\n🧪 Testing model loading: {s}\n", .{model_path});
    
    var model = HuggingFaceModel.init(allocator, model_path);
    defer model.deinit();
    
    try model.load();
    
    // Test tensor access
    std.debug.print("\n🔍 Testing tensor access...\n", .{});
    
    const test_tensors = [_][]const u8{
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "lm_head.weight",
    };
    
    for (test_tensors) |tensor_name| {
        if (model.hasTensor(tensor_name)) {
            std.debug.print("   ✅ Found: {s}\n", .{tensor_name});
            
            if (model.weights.getTensorInfo(tensor_name)) |info| {
                std.debug.print("      Shape: [", .{});
                for (info.shape, 0..) |dim, i| {
                    if (i > 0) std.debug.print(", ", .{});
                    std.debug.print("{d}", .{dim});
                }
                std.debug.print("]\n", .{});
                std.debug.print("      Dtype: {s}\n", .{@tagName(info.dtype)});
            } else |_| {}
        } else {
            std.debug.print("   ❌ Not found: {s}\n", .{tensor_name});
        }
    }
    
    std.debug.print("\n✅ HuggingFace loader test complete!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
