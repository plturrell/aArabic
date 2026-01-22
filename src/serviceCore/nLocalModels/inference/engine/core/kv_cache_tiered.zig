// Tiered KV Cache Integration
// Drop-in replacement for kv_cache.zig with SSD tiering support
//
// This module wraps the tiering/tiered_kv_cache.zig to provide the same API
// as the existing KVCache, enabling seamless integration with transformer.zig
// and llama_model.zig.

const std = @import("std");
const tiered_kv = @import("../tiering/tiered_kv_cache.zig");
const ssd_tier = @import("../tiering/ssd_tier.zig");
const kv_cache = @import("kv_cache.zig");

pub const TieredKVCacheConfig = struct {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    
    // Tiering options
    hot_tokens: u32 = 2048,        // Tokens kept in RAM
    ssd_path: []const u8 = "/tmp/shimmy_kv.tier",
    max_ssd_mb: u32 = 16384,       // 16GB SSD tier by default
    enable_tiering: bool = true,   // Disable to use RAM-only mode
};

/// Tiered KV Cache - API compatible with KVCache
/// Automatically tiers to SSD when RAM fills up
pub const TieredKVCache = struct {
    allocator: std.mem.Allocator,
    config: TieredKVCacheConfig,
    
    // Either RAM-only or tiered storage
    ram_cache: ?*kv_cache.KVCache,
    tiered_cache: ?*tiered_kv.TieredKVCache,
    
    // API compatibility fields
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    seq_pos: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: TieredKVCacheConfig) !*TieredKVCache {
        const self = try allocator.create(TieredKVCache);
        errdefer allocator.destroy(self);
        
        std.debug.print("\n🗄️  Initializing Tiered KV Cache...\n", .{});
        std.debug.print("   Layers: {d}, Heads: {d}, Head dim: {d}\n", .{
            config.n_layers, config.n_heads, config.head_dim,
        });
        std.debug.print("   Max sequence: {d} tokens\n", .{config.max_seq_len});
        std.debug.print("   Tiering: {s}\n", .{if (config.enable_tiering) "enabled" else "disabled"});
        
        if (config.enable_tiering and config.max_seq_len > config.hot_tokens) {
            // Use tiered storage
            std.debug.print("   Hot tokens: {d} (in RAM)\n", .{config.hot_tokens});
            std.debug.print("   Cold tier: up to {d} MB on SSD\n", .{config.max_ssd_mb});
            
            const tiered_config = tiered_kv.TieredKVConfig{
                .n_layers = config.n_layers,
                .n_heads = config.n_heads,
                .head_dim = config.head_dim,
                .max_seq_len = config.max_seq_len,
                .hot_tokens = config.hot_tokens,
                .ssd_path = config.ssd_path,
                .max_ssd_mb = config.max_ssd_mb,
            };
            
            const tiered_cache = try tiered_kv.TieredKVCache.init(allocator, tiered_config);
            
            self.* = TieredKVCache{
                .allocator = allocator,
                .config = config,
                .ram_cache = null,
                .tiered_cache = tiered_cache,
                .n_layers = config.n_layers,
                .n_heads = config.n_heads,
                .head_dim = config.head_dim,
                .max_seq_len = config.max_seq_len,
                .seq_pos = 0,
            };
        } else {
            // Use RAM-only storage (fall back to original KVCache)
            std.debug.print("   Using RAM-only mode\n", .{});
            
            var ram_cache = try allocator.create(kv_cache.KVCache);
            ram_cache.* = try kv_cache.KVCache.init(
                allocator,
                config.n_layers,
                config.n_heads,
                config.head_dim,
                config.max_seq_len,
            );
            
            self.* = TieredKVCache{
                .allocator = allocator,
                .config = config,
                .ram_cache = ram_cache,
                .tiered_cache = null,
                .n_layers = config.n_layers,
                .n_heads = config.n_heads,
                .head_dim = config.head_dim,
                .max_seq_len = config.max_seq_len,
                .seq_pos = 0,
            };
        }
        
        std.debug.print("   ✅ Tiered KV cache ready\n", .{});
        return self;
    }
    
    pub fn deinit(self: *TieredKVCache) void {
        if (self.tiered_cache) |tc| tc.deinit();
        if (self.ram_cache) |rc| {
            rc.deinit();
            self.allocator.destroy(rc);
        }
        self.allocator.destroy(self);
    }
    
    /// Store keys and values for a layer (API compatible)
    pub fn store(self: *TieredKVCache, layer: u32, keys: []const f32, values: []const f32) void {
        if (self.tiered_cache) |tc| {
            tc.store(layer, keys, values) catch {};
        } else if (self.ram_cache) |rc| {
            rc.store(layer, keys, values);
        }
    }
    
    /// Advance position after storing all layers
    pub fn advance(self: *TieredKVCache) void {
        if (self.tiered_cache) |tc| {
            tc.advance();
            self.seq_pos = tc.seq_pos;
        } else if (self.ram_cache) |rc| {
            rc.advance();
            self.seq_pos = rc.seq_pos;
        }
    }
    
    /// Get keys for a layer up to current position
    pub fn getKeys(self: *TieredKVCache, layer: u32) []f32 {
        if (self.ram_cache) |rc| {
            return rc.getKeys(layer);
        }
        // For tiered, return hot cache slice (attention only uses recent anyway)
        if (self.tiered_cache) |tc| {
            return tc.getHotKeys(layer);
        }
        return &[_]f32{};
    }
    
    /// Get values for a layer up to current position
    pub fn getValues(self: *TieredKVCache, layer: u32) []f32 {
        if (self.ram_cache) |rc| {
            return rc.getValues(layer);
        }
        if (self.tiered_cache) |tc| {
            return tc.getHotValues(layer);
        }
        return &[_]f32{};
    }
    
    /// Reset cache
    pub fn reset(self: *TieredKVCache) void {
        if (self.tiered_cache) |tc| tc.reset();
        if (self.ram_cache) |rc| rc.reset();
        self.seq_pos = 0;
    }
    
    /// Print status
    pub fn printStatus(self: *TieredKVCache) void {
        if (self.tiered_cache) |tc| {
            tc.printStatus();
        } else {
            std.debug.print("\n📊 KV Cache Status (RAM-only)\n", .{});
            std.debug.print("   Sequence position: {d}/{d}\n", .{self.seq_pos, self.max_seq_len});
        }
    }
};

/// Create a tiered KV cache (convenience function)
pub fn createTieredCache(
    allocator: std.mem.Allocator,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
) !*TieredKVCache {
    return try TieredKVCache.init(allocator, .{
        .n_layers = n_layers,
        .n_heads = n_heads,
        .head_dim = head_dim,
        .max_seq_len = max_seq_len,
    });
}

