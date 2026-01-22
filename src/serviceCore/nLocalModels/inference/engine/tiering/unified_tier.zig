// Unified Tiering API
// Single interface for all tiering operations
//
// Combines:
// - Local SSD tier (ssd_tier.zig)
// - Tiered KV cache (tiered_kv_cache.zig)
// - Memory-mapped GGUF (mmap_gguf.zig)
// - Tiered tensors (tiered_tensors.zig)
// - Distributed tier (distributed_tier.zig)
//
// This is the main entry point for the tiering subsystem

const std = @import("std");
const ssd = @import("ssd_tier.zig");
const tiered_kv = @import("tiered_kv_cache.zig");
const mmap_gguf = @import("mmap_gguf.zig");
const tiered_tensors = @import("tiered_tensors.zig");
const distributed = @import("distributed_tier.zig");

// ============================================================================
// Unified Configuration
// ============================================================================

pub const UnifiedTierConfig = struct {
    // Model path
    model_path: []const u8,
    
    // Model dimensions (auto-detected if possible)
    n_layers: ?u32 = null,
    n_heads: ?u32 = null,
    head_dim: ?u32 = null,
    max_seq_len: u32 = 8192,
    
    // Memory budget
    max_ram_mb: u64 = 4096,        // 4GB total RAM budget
    kv_cache_ram_mb: u64 = 1024,   // 1GB for KV cache
    tensor_hot_mb: u64 = 512,      // 512MB for hot tensors
    tensor_warm_mb: u64 = 1024,    // 1GB for warm tensors
    
    // SSD settings
    ssd_path: []const u8 = "/tmp/shimmy_tier",
    max_ssd_mb: u64 = 32768,       // 32GB SSD budget
    
    // Distributed settings
    enable_distributed: bool = false,
    dragonfly_host: []const u8 = "127.0.0.1",
    dragonfly_port: u16 = 6379,
    
    // Feature flags
    enable_kv_tiering: bool = true,
    enable_tensor_tiering: bool = true,
    enable_prompt_caching: bool = true,
};

// ============================================================================
// Unified Tier Manager
// ============================================================================

pub const UnifiedTierManager = struct {
    allocator: std.mem.Allocator,
    config: UnifiedTierConfig,
    
    // Components
    kv_cache: ?*tiered_kv.TieredKVCache,
    tensor_manager: ?*tiered_tensors.TieredTensorManager,
    distributed_tier: ?*distributed.DistributedKVTier,
    
    // Model info (from GGUF)
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    vocab_size: u32,
    
    // Session tracking
    current_session: ?[]const u8,
    
    pub fn init(allocator: std.mem.Allocator, config: UnifiedTierConfig) !*UnifiedTierManager {
        std.debug.print("\n" ++ "═" ** 70 ++ "\n", .{});
        std.debug.print("🚀 Initializing Unified Tiering System\n", .{});
        std.debug.print("═" ** 70 ++ "\n", .{});
        std.debug.print("   Model: {s}\n", .{config.model_path});
        std.debug.print("   RAM budget: {d} MB\n", .{config.max_ram_mb});
        std.debug.print("   SSD budget: {d} MB\n", .{config.max_ssd_mb});
        
        const self = try allocator.create(UnifiedTierManager);
        errdefer allocator.destroy(self);
        
        // Initialize tensor manager (loads GGUF)
        var tensor_manager: ?*tiered_tensors.TieredTensorManager = null;
        if (config.enable_tensor_tiering) {
            const tensor_config = tiered_tensors.TieredTensorConfig{
                .hot_tier_mb = config.tensor_hot_mb,
                .warm_tier_mb = config.tensor_warm_mb,
            };
            tensor_manager = try tiered_tensors.TieredTensorManager.init(
                allocator,
                config.model_path,
                tensor_config,
            );
        }
        
        // Extract model dimensions from GGUF or use provided
        const n_layers = config.n_layers orelse (if (tensor_manager) |tm| tm.n_layers else 32);
        const n_heads = config.n_heads orelse 32;
        const head_dim = config.head_dim orelse 128;
        
        // Initialize KV cache
        var kv_cache: ?*tiered_kv.TieredKVCache = null;
        if (config.enable_kv_tiering) {
            const kv_config = tiered_kv.TieredKVConfig{
                .n_layers = n_layers,
                .n_heads = n_heads,
                .head_dim = head_dim,
                .max_seq_len = config.max_seq_len,
                .max_ram_mb = config.kv_cache_ram_mb,
                .max_ssd_mb = config.max_ssd_mb / 2, // Half for KV cache
                .ssd_path = config.ssd_path,
            };
            kv_cache = try tiered_kv.TieredKVCache.init(allocator, kv_config);
        }
        
        // Initialize distributed tier
        var distributed_tier: ?*distributed.DistributedKVTier = null;
        if (config.enable_distributed) {
            const dist_config = distributed.DistributedConfig{
                .dragonfly_host = config.dragonfly_host,
                .dragonfly_port = config.dragonfly_port,
            };
            distributed_tier = distributed.DistributedKVTier.init(allocator, dist_config) catch |err| {
                std.debug.print("   ⚠️  Distributed tier unavailable: {}\n", .{err});
                null;
            };
        }
        
        self.* = UnifiedTierManager{
            .allocator = allocator,
            .config = config,
            .kv_cache = kv_cache,
            .tensor_manager = tensor_manager,
            .distributed_tier = distributed_tier,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .vocab_size = 32000, // Default, should be from GGUF
            .current_session = null,
        };
        
        std.debug.print("\n✅ Unified Tiering System Ready\n", .{});
        std.debug.print("   Layers: {d}, Heads: {d}, Head dim: {d}\n", .{
            n_layers, n_heads, head_dim,
        });
        std.debug.print("   KV tiering: {s}\n", .{if (kv_cache != null) "enabled" else "disabled"});
        std.debug.print("   Tensor tiering: {s}\n", .{if (tensor_manager != null) "enabled" else "disabled"});
        std.debug.print("   Distributed: {s}\n", .{if (distributed_tier != null) "enabled" else "disabled"});
        std.debug.print("═" ** 70 ++ "\n\n", .{});

        return self;
    }

    // ========================================================================
    // Session Management
    // ========================================================================

    pub fn startSession(self: *UnifiedTierManager, session_id: []const u8) !void {
        self.current_session = try self.allocator.dupe(u8, session_id);

        // ✅ P2-16 FIXED: Try to restore from distributed cache
        if (self.distributed_tier) |dt| {
            if (try dt.loadSession(session_id)) |serialized_state| {
                try self.deserializeKVCache(serialized_state);
                std.debug.print("📥 Restored session: {s}\n", .{session_id});
            }
        }
    }

    pub fn endSession(self: *UnifiedTierManager) !void {
        if (self.current_session) |session_id| {
            // ✅ P2-16 FIXED: Save to distributed cache
            if (self.distributed_tier) |dt| {
                const serialized_state = try self.serializeKVCache();
                defer self.allocator.free(serialized_state);
                try dt.storeSession(session_id, serialized_state);
                std.debug.print("💾 Saved session: {s} ({d} bytes)\n", .{session_id, serialized_state.len});
            }

            self.allocator.free(session_id);
            self.current_session = null;
        }

        // Reset KV cache
        if (self.kv_cache) |kv| {
            kv.reset();
        }
    }
    
    // ========================================================================
    // ✅ P2-16: KV Cache Serialization
    // ========================================================================
    
    /// Serialize KV cache state to bytes
    /// Format: [version:u32][n_layers:u32][seq_len:u32][layer_data...]
    /// Layer data: [layer_id:u32][keys_len:u32][keys...][values_len:u32][values...]
    fn serializeKVCache(self: *UnifiedTierManager) ![]u8 {
        if (self.kv_cache == null) {
            // No KV cache, return empty state
            return try self.allocator.dupe(u8, "");
        }
        
        const kv = self.kv_cache.?;
        const stats = kv.getStats();
        
        // Calculate total size needed
        // Header: version(4) + n_layers(4) + seq_len(4) = 12 bytes
        var total_size: usize = 12;
        
        // For each layer: layer_id(4) + keys_len(4) + keys + values_len(4) + values
        const kv_dim = self.n_heads * self.head_dim;
        const layer_size = 4 + 4 + (stats.hot_tokens * kv_dim * @sizeOf(f32)) + 
                          4 + (stats.hot_tokens * kv_dim * @sizeOf(f32));
        total_size += layer_size * self.n_layers;
        
        var buffer = try self.allocator.alloc(u8, total_size);
        errdefer self.allocator.free(buffer);
        
        var offset: usize = 0;
        
        // Write header
        std.mem.writeInt(u32, buffer[offset..][0..4], 1, .little); // version
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], self.n_layers, .little);
        offset += 4;
        std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(stats.hot_tokens), .little);
        offset += 4;
        
        // Write each layer's KV data
        var keys_buf = try self.allocator.alloc(f32, stats.hot_tokens * kv_dim);
        defer self.allocator.free(keys_buf);
        var values_buf = try self.allocator.alloc(f32, stats.hot_tokens * kv_dim);
        defer self.allocator.free(values_buf);
        
        for (0..self.n_layers) |layer| {
            // Write layer ID
            std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(layer), .little);
            offset += 4;
            
            // Get keys for this layer
            kv.getKeys(@intCast(layer), 0, @intCast(stats.hot_tokens), keys_buf) catch {
                // If layer not in cache, write zeros
                @memset(keys_buf, 0.0);
            };
            
            // Write keys length and data
            const keys_bytes = stats.hot_tokens * kv_dim * @sizeOf(f32);
            std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(keys_bytes), .little);
            offset += 4;
            @memcpy(buffer[offset..][0..keys_bytes], std.mem.sliceAsBytes(keys_buf)[0..keys_bytes]);
            offset += keys_bytes;
            
            // Get values for this layer
            kv.getValues(@intCast(layer), 0, @intCast(stats.hot_tokens), values_buf) catch {
                @memset(values_buf, 0.0);
            };
            
            // Write values length and data
            const values_bytes = stats.hot_tokens * kv_dim * @sizeOf(f32);
            std.mem.writeInt(u32, buffer[offset..][0..4], @intCast(values_bytes), .little);
            offset += 4;
            @memcpy(buffer[offset..][0..values_bytes], std.mem.sliceAsBytes(values_buf)[0..values_bytes]);
            offset += values_bytes;
        }
        
        return buffer;
    }
    
    /// Deserialize KV cache state from bytes
    fn deserializeKVCache(self: *UnifiedTierManager, data: []const u8) !void {
        if (data.len == 0) {
            // Empty state, nothing to restore
            return;
        }
        
        if (self.kv_cache == null) {
            return error.NoKVCache;
        }
        
        if (data.len < 12) {
            return error.InvalidSerializedState;
        }
        
        var offset: usize = 0;
        
        // Read header
        const version = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        if (version != 1) {
            return error.UnsupportedVersion;
        }
        
        const n_layers = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        if (n_layers != self.n_layers) {
            std.debug.print("⚠️  Layer count mismatch: expected {d}, got {d}\n", .{self.n_layers, n_layers});
            return error.LayerCountMismatch;
        }
        
        const seq_len = std.mem.readInt(u32, data[offset..][0..4], .little);
        offset += 4;
        
        const kv_dim = self.n_heads * self.head_dim;
        const kv = self.kv_cache.?;
        
        // Reset cache before restoring
        kv.reset();
        
        // Read each layer's KV data
        for (0..n_layers) |layer| {
            if (offset + 4 > data.len) break;
            
            // Read layer ID
            const layer_id = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset += 4;
            
            if (layer_id != layer) {
                std.debug.print("⚠️  Layer ID mismatch at layer {d}\n", .{layer});
                return error.LayerIDMismatch;
            }
            
            // Read keys
            if (offset + 4 > data.len) break;
            const keys_bytes = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset += 4;
            
            if (offset + keys_bytes > data.len) {
                return error.TruncatedData;
            }
            
            const keys_data = data[offset..][0..keys_bytes];
            const keys_slice = std.mem.bytesAsSlice(f32, keys_data);
            offset += keys_bytes;
            
            // Read values
            if (offset + 4 > data.len) break;
            const values_bytes = std.mem.readInt(u32, data[offset..][0..4], .little);
            offset += 4;
            
            if (offset + values_bytes > data.len) {
                return error.TruncatedData;
            }
            
            const values_data = data[offset..][0..values_bytes];
            const values_slice = std.mem.bytesAsSlice(f32, values_data);
            offset += values_bytes;
            
            // Restore KV data for this layer
            // Store each token's KV
            const tokens_in_layer = keys_bytes / (kv_dim * @sizeOf(f32));
            for (0..tokens_in_layer) |token_idx| {
                const token_keys = keys_slice[token_idx * kv_dim .. (token_idx + 1) * kv_dim];
                const token_values = values_slice[token_idx * kv_dim .. (token_idx + 1) * kv_dim];
                
                try kv.store(@intCast(layer), token_keys, token_values);
                kv.advance();
            }
        }
        
        std.debug.print("✅ Restored {d} tokens across {d} layers\n", .{seq_len, n_layers});
    }

    // ========================================================================
    // KV Cache Operations
    // ========================================================================

    /// Store KV for current token
    pub fn storeKV(
        self: *UnifiedTierManager,
        layer: u32,
        keys: []const f32,
        values: []const f32,
    ) !void {
        if (self.kv_cache) |kv| {
            try kv.store(layer, keys, values);
        }
    }

    /// Get keys for attention
    pub fn getKeys(
        self: *UnifiedTierManager,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
        dest: []f32,
    ) !void {
        if (self.kv_cache) |kv| {
            try kv.getKeys(layer, start_pos, end_pos, dest);
        }
    }

    /// Get values for attention
    pub fn getValues(
        self: *UnifiedTierManager,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
        dest: []f32,
    ) !void {
        if (self.kv_cache) |kv| {
            try kv.getValues(layer, start_pos, end_pos, dest);
        }
    }

    /// Advance KV cache position
    pub fn advanceKV(self: *UnifiedTierManager) void {
        if (self.kv_cache) |kv| {
            kv.advance();
        }
    }

    // ========================================================================
    // Tensor Operations
    // ========================================================================

    /// Get tensor data (handles tiering transparently)
    pub fn getTensor(self: *UnifiedTierManager, name: []const u8) ![]const u8 {
        if (self.tensor_manager) |tm| {
            return tm.getTensor(name);
        }
        return error.TensorTieringDisabled;
    }

    /// Get tensor as f32
    pub fn getTensorF32(self: *UnifiedTierManager, name: []const u8) ![]const f32 {
        if (self.tensor_manager) |tm| {
            return tm.getTensorF32(name);
        }
        return error.TensorTieringDisabled;
    }

    /// Notify layer start (for prefetching)
    pub fn startLayer(self: *UnifiedTierManager, layer: u32) void {
        if (self.tensor_manager) |tm| {
            tm.startLayer(layer);
        }
    }

    // ========================================================================
    // Prompt Caching
    // ========================================================================

    /// Check if prompt is cached
    pub fn checkPromptCache(self: *UnifiedTierManager, prompt: []const u8) !?[]const u8 {
        if (!self.config.enable_prompt_caching) return null;

        if (self.distributed_tier) |dt| {
            const hash = distributed.hashPrompt(prompt);
            const hex = distributed.hashToHex(hash);
            return dt.loadPromptCache(&hex);
        }
        return null;
    }

    /// Cache prompt state
    pub fn cachePrompt(self: *UnifiedTierManager, prompt: []const u8, state: []const u8) !void {
        if (!self.config.enable_prompt_caching) return;

        if (self.distributed_tier) |dt| {
            const hash = distributed.hashPrompt(prompt);
            const hex = distributed.hashToHex(hash);
            try dt.storePromptCache(&hex, state);
        }
    }

    // ========================================================================
    // Statistics & Monitoring
    // ========================================================================

    pub fn printStatus(self: *UnifiedTierManager) void {
        std.debug.print("\n" ++ "═" ** 70 ++ "\n", .{});
        std.debug.print("📊 Unified Tiering Status\n", .{});
        std.debug.print("═" ** 70 ++ "\n", .{});

        if (self.kv_cache) |kv| {
            kv.printStatus();
        }

        if (self.tensor_manager) |tm| {
            tm.printStatus();
        }

        if (self.distributed_tier) |dt| {
            dt.printStatus();
        }

        std.debug.print("═" ** 70 ++ "\n\n", .{});
    }

    pub fn getMemoryUsage(self: *UnifiedTierManager) struct {
        kv_hot_mb: u64,
        kv_cold_mb: u64,
        tensor_hot_mb: u64,
        tensor_warm_mb: u64,
        total_ram_mb: u64,
        total_ssd_mb: u64,
    } {
        var kv_hot: u64 = 0;
        var kv_cold: u64 = 0;
        var tensor_hot: u64 = 0;
        var tensor_warm: u64 = 0;

        if (self.kv_cache) |kv| {
            const stats = kv.getStats();
            kv_hot = @as(u64, stats.hot_tokens) * self.config.n_layers.? * 2 * self.head_dim * self.n_heads * @sizeOf(f32) / (1024 * 1024);
            kv_cold = stats.ssd_usage_mb;
        }

        if (self.tensor_manager) |tm| {
            tensor_hot = tm.hot_bytes / (1024 * 1024);
            tensor_warm = tm.warm_bytes / (1024 * 1024);
        }

        return .{
            .kv_hot_mb = kv_hot,
            .kv_cold_mb = kv_cold,
            .tensor_hot_mb = tensor_hot,
            .tensor_warm_mb = tensor_warm,
            .total_ram_mb = kv_hot + tensor_hot + tensor_warm,
            .total_ssd_mb = kv_cold,
        };
    }

    pub fn deinit(self: *UnifiedTierManager) void {
        if (self.current_session) |session| {
            self.allocator.free(session);
        }
        if (self.kv_cache) |kv| {
            kv.deinit();
        }
        if (self.tensor_manager) |tm| {
            tm.deinit();
        }
        if (self.distributed_tier) |dt| {
            dt.deinit();
        }
        self.allocator.destroy(self);
    }
};

// ============================================================================
// C ABI Exports for Mojo Integration
// ============================================================================

export fn shimmy_tier_init(
    model_path_ptr: [*]const u8,
    model_path_len: usize,
    max_ram_mb: u64,
    max_ssd_mb: u64,
) ?*UnifiedTierManager {
    const allocator = std.heap.c_allocator;
    const model_path = model_path_ptr[0..model_path_len];

    const config = UnifiedTierConfig{
        .model_path = model_path,
        .max_ram_mb = max_ram_mb,
        .max_ssd_mb = max_ssd_mb,
    };

    return UnifiedTierManager.init(allocator, config) catch null;
}

export fn shimmy_tier_deinit(manager: *UnifiedTierManager) void {
    manager.deinit();
}

export fn shimmy_tier_store_kv(
    manager: *UnifiedTierManager,
    layer: u32,
    keys_ptr: [*]const f32,
    values_ptr: [*]const f32,
    kv_dim: usize,
) bool {
    const keys = keys_ptr[0..kv_dim];
    const values = values_ptr[0..kv_dim];
    manager.storeKV(layer, keys, values) catch return false;
    return true;
}

export fn shimmy_tier_advance_kv(manager: *UnifiedTierManager) void {
    manager.advanceKV();
}

export fn shimmy_tier_get_tensor(
    manager: *UnifiedTierManager,
    name_ptr: [*]const u8,
    name_len: usize,
    out_ptr: *[*]const u8,
    out_len: *usize,
) bool {
    const name = name_ptr[0..name_len];
    const data = manager.getTensor(name) catch return false;
    out_ptr.* = data.ptr;
    out_len.* = data.len;
    return true;
}

export fn shimmy_tier_print_status(manager: *UnifiedTierManager) void {
    manager.printStatus();
}
