// Tiered Model Loader
// Load GGUF models using memory-mapped tensors with SSD tiering
//
// This enables loading 70B+ models on limited RAM by keeping only
// hot tensors in memory and streaming others from SSD on demand.

const std = @import("std");
const mmap_gguf = @import("../tiering/mmap_gguf.zig");
const tiered_tensors = @import("../tiering/tiered_tensors.zig");

pub const TieredModelConfig = struct {
    model_path: []const u8,
    max_ram_mb: u32 = 4096,       // 4GB default RAM budget
    hot_layers: ?[]const []const u8 = null,  // Force specific layers hot
    warm_layers: u32 = 4,          // Number of warm layers to keep
};

pub const TieredModelLoader = struct {
    allocator: std.mem.Allocator,
    config: TieredModelConfig,
    
    // Memory-mapped GGUF file
    gguf: ?*mmap_gguf.MmapGGUF,
    
    // Tiered tensor manager
    tensors: ?*tiered_tensors.TieredTensorManager,
    
    // Model metadata extracted from GGUF
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    hidden_size: u32,
    vocab_size: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: TieredModelConfig) !*TieredModelLoader {
        const self = try allocator.create(TieredModelLoader);
        errdefer allocator.destroy(self);
        
        std.debug.print("\n📂 Tiered Model Loader\n", .{});
        std.debug.print("   Path: {s}\n", .{config.model_path});
        std.debug.print("   RAM budget: {d} MB\n", .{config.max_ram_mb});
        
        // Open GGUF with mmap
        const gguf = try mmap_gguf.MmapGGUF.open(allocator, config.model_path);
        errdefer gguf.close();
        
        // Extract model config from GGUF metadata
        var n_layers: u32 = 32;
        var n_heads: u32 = 32;
        var n_kv_heads: u32 = 8;
        var head_dim: u32 = 128;
        var hidden_size: u32 = 4096;
        var vocab_size: u32 = 32000;
        
        // Try to read from GGUF metadata
        if (gguf.getMetaInt("llama.block_count")) |v| n_layers = @intCast(v);
        if (gguf.getMetaInt("llama.attention.head_count")) |v| n_heads = @intCast(v);
        if (gguf.getMetaInt("llama.attention.head_count_kv")) |v| n_kv_heads = @intCast(v);
        if (gguf.getMetaInt("llama.embedding_length")) |v| hidden_size = @intCast(v);
        if (gguf.getMetaInt("llama.vocab_size")) |v| vocab_size = @intCast(v);
        head_dim = hidden_size / n_heads;
        
        std.debug.print("   Layers: {d}, Heads: {d}/{d}, Dim: {d}\n", .{
            n_layers, n_heads, n_kv_heads, hidden_size,
        });
        
        // Initialize tiered tensor manager
        const tensor_config = tiered_tensors.TieredTensorConfig{
            .max_hot_bytes = @as(u64, config.max_ram_mb) * 1024 * 1024 / 2,  // Half for hot
            .max_warm_bytes = @as(u64, config.max_ram_mb) * 1024 * 1024 / 2, // Half for warm
            .n_layers = n_layers,
        };
        
        const tensors = try tiered_tensors.TieredTensorManager.init(allocator, gguf, tensor_config);
        
        // Classify tensors
        tensors.classifyLayers();
        
        std.debug.print("   ✅ Model loaded with tiering\n", .{});
        
        self.* = TieredModelLoader{
            .allocator = allocator,
            .config = config,
            .gguf = gguf,
            .tensors = tensors,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .head_dim = head_dim,
            .hidden_size = hidden_size,
            .vocab_size = vocab_size,
        };
        
        return self;
    }
    
    pub fn deinit(self: *TieredModelLoader) void {
        if (self.tensors) |t| t.deinit();
        if (self.gguf) |g| g.close();
        self.allocator.destroy(self);
    }
    
    /// Get tensor data (zero-copy from mmap or RAM copy)
    pub fn getTensor(self: *TieredModelLoader, name: []const u8) ![]const f32 {
        if (self.tensors) |t| {
            return try t.getTensorF32(name);
        }
        return error.TensorNotFound;
    }
    
    /// Prefetch tensors for a layer (call before processing layer)
    pub fn prefetchLayer(self: *TieredModelLoader, layer: u32) void {
        if (self.tensors) |t| {
            t.prefetchLayer(layer);
        }
    }
    
    /// Print memory usage
    pub fn printStatus(self: *TieredModelLoader) void {
        if (self.tensors) |t| {
            t.printStatus();
        }
        if (self.gguf) |g| {
            g.printInfo();
        }
    }
    
    /// Get model info
    pub fn getInfo(self: *TieredModelLoader) struct {
        n_layers: u32,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        hidden_size: u32,
        vocab_size: u32,
    } {
        return .{
            .n_layers = self.n_layers,
            .n_heads = self.n_heads,
            .n_kv_heads = self.n_kv_heads,
            .head_dim = self.head_dim,
            .hidden_size = self.hidden_size,
            .vocab_size = self.vocab_size,
        };
    }
};

