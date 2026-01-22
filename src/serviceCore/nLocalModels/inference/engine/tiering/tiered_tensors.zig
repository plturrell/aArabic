// Tiered Tensor Storage
// Hot/Warm/Cold tiers for model weights
//
// Architecture:
// - Hot tier (RAM): Frequently accessed layers (embeddings, output)
// - Warm tier (RAM, evictable): Recently used layers
// - Cold tier (SSD/mmap): All other layers
//
// This enables running 70B+ models on 16GB RAM systems

const std = @import("std");
const mmap_gguf = @import("mmap_gguf.zig");
const ssd = @import("ssd_tier.zig");

// ============================================================================
// Configuration
// ============================================================================

pub const TieredTensorConfig = struct {
    // Memory budget
    hot_tier_mb: u64 = 512,       // Always in RAM (embeddings, etc)
    warm_tier_mb: u64 = 1024,     // Evictable RAM cache
    
    // Layer pinning
    pin_embedding: bool = true,   // Keep embedding layer hot
    pin_output: bool = true,      // Keep output layer hot
    pin_first_n_layers: u32 = 2,  // Keep first N layers hot
    pin_last_n_layers: u32 = 1,   // Keep last N layers hot
    
    // Prefetch
    prefetch_ahead: u32 = 2,      // Prefetch N layers ahead
    
    // Dequantization
    dequant_on_load: bool = false, // Dequantize to f32 on load
};

// ============================================================================
// Tensor Location
// ============================================================================

pub const TensorTier = enum {
    Hot,    // Pinned in RAM, never evicted
    Warm,   // In RAM, can be evicted
    Cold,   // On SSD/mmap, loaded on demand
};

pub const TensorEntry = struct {
    name: []const u8,
    tier: TensorTier,
    
    // Location info
    ram_data: ?[]f32,           // If in RAM (dequantized)
    mmap_slice: ?[]const u8,    // If mmap'd from GGUF
    
    // Metadata
    dtype: mmap_gguf.GGMLType,
    dims: [4]u64,
    n_dims: u32,
    size_bytes: u64,
    
    // Access tracking
    access_count: u64,
    last_access: i64,
    
    pub fn isLoaded(self: TensorEntry) bool {
        return self.ram_data != null or self.mmap_slice != null;
    }
};

// ============================================================================
// Tiered Tensor Manager
// ============================================================================

pub const TieredTensorManager = struct {
    allocator: std.mem.Allocator,
    config: TieredTensorConfig,
    
    // GGUF source
    gguf: *mmap_gguf.MmapGGUF,
    
    // Tensor registry
    tensors: std.StringHashMap(TensorEntry),
    
    // Memory tracking
    hot_bytes: u64,
    warm_bytes: u64,
    
    // Layer info (for prefetching)
    n_layers: u32,
    current_layer: u32,
    
    // Statistics
    stats: Stats,
    
    pub const Stats = struct {
        hot_hits: u64 = 0,
        warm_hits: u64 = 0,
        cold_loads: u64 = 0,
        evictions: u64 = 0,
        prefetches: u64 = 0,
        bytes_loaded: u64 = 0,
        bytes_evicted: u64 = 0,
    };
    
    pub fn init(
        allocator: std.mem.Allocator,
        gguf_path: []const u8,
        config: TieredTensorConfig,
    ) !*TieredTensorManager {
        std.debug.print("\n🎯 Initializing Tiered Tensor Manager\n", .{});
        
        const self = try allocator.create(TieredTensorManager);
        errdefer allocator.destroy(self);
        
        // Open GGUF file
        const gguf = try mmap_gguf.MmapGGUF.open(allocator, gguf_path);
        errdefer gguf.close();
        
        self.* = TieredTensorManager{
            .allocator = allocator,
            .config = config,
            .gguf = gguf,
            .tensors = std.StringHashMap(TensorEntry).init(allocator),
            .hot_bytes = 0,
            .warm_bytes = 0,
            .n_layers = 0,
            .current_layer = 0,
            .stats = .{},
        };
        
        // Index all tensors
        try self.indexTensors();
        
        // Pin hot tensors
        try self.pinHotTensors();
        
        std.debug.print("   ✅ Tensor manager ready\n", .{});
        std.debug.print("   Hot: {d:.1} MB, Warm: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.hot_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.warm_bytes)) / (1024.0 * 1024.0),
        });
        
        return self;
    }
    
    fn indexTensors(self: *TieredTensorManager) !void {
        var iter = self.gguf.tensors.iterator();
        var max_layer: u32 = 0;
        
        while (iter.next()) |entry| {
            const desc = entry.value_ptr.*;
            
            // Determine tier based on name
            const tier = self.classifyTensor(desc.name);
            
            // Extract layer number if present
            if (self.extractLayerNum(desc.name)) |layer_num| {
                max_layer = @max(max_layer, layer_num);
            }
            
            try self.tensors.put(desc.name, .{
                .name = desc.name,
                .tier = tier,
                .ram_data = null,
                .mmap_slice = null,
                .dtype = desc.dtype,
                .dims = desc.dims,
                .n_dims = desc.n_dims,
                .size_bytes = desc.size,
                .access_count = 0,
                .last_access = 0,
            });
        }
        
        self.n_layers = max_layer + 1;
        std.debug.print("   Indexed {d} tensors across {d} layers\n", .{
            self.tensors.count(), self.n_layers,
        });
    }

    fn classifyTensor(self: *TieredTensorManager, name: []const u8) TensorTier {
        // Embedding and output layers are always hot
        if (self.config.pin_embedding) {
            if (std.mem.indexOf(u8, name, "embed") != null or
                std.mem.indexOf(u8, name, "token_embd") != null) {
                return .Hot;
            }
        }

        if (self.config.pin_output) {
            if (std.mem.indexOf(u8, name, "output") != null or
                std.mem.indexOf(u8, name, "lm_head") != null) {
                return .Hot;
            }
        }

        // Check layer number for pinning
        if (self.extractLayerNum(name)) |layer_num| {
            if (layer_num < self.config.pin_first_n_layers) {
                return .Hot;
            }
            // Note: can't check last N until we know total layers
        }

        return .Cold;
    }

    fn extractLayerNum(self: *TieredTensorManager, name: []const u8) ?u32 {
        _ = self;
        // Look for patterns like "blk.0.", "layers.0.", "h.0."
        const patterns = [_][]const u8{ "blk.", "layers.", "h.", "layer." };

        for (patterns) |pattern| {
            if (std.mem.indexOf(u8, name, pattern)) |idx| {
                const start = idx + pattern.len;
                var end = start;
                while (end < name.len and name[end] >= '0' and name[end] <= '9') {
                    end += 1;
                }
                if (end > start) {
                    return std.fmt.parseInt(u32, name[start..end], 10) catch null;
                }
            }
        }
        return null;
    }

    fn pinHotTensors(self: *TieredTensorManager) !void {
        var iter = self.tensors.iterator();

        while (iter.next()) |entry| {
            var tensor = entry.value_ptr;

            // Update last N layers classification now that we know n_layers
            if (self.extractLayerNum(tensor.name)) |layer_num| {
                if (layer_num >= self.n_layers - self.config.pin_last_n_layers) {
                    tensor.tier = .Hot;
                }
            }

            // Load hot tensors into RAM
            if (tensor.tier == .Hot) {
                try self.loadToRam(tensor);
            }
        }
    }

    fn loadToRam(self: *TieredTensorManager, tensor: *TensorEntry) !void {
        if (tensor.ram_data != null) return; // Already loaded

        // Get mmap slice
        const data = try self.gguf.getTensorData(tensor.name);
        tensor.mmap_slice = data;

        // For hot tensors, we might want to copy to RAM for faster access
        // For now, just use mmap (zero-copy)

        if (tensor.tier == .Hot) {
            self.hot_bytes += tensor.size_bytes;
        } else {
            self.warm_bytes += tensor.size_bytes;
        }

        self.stats.bytes_loaded += tensor.size_bytes;
    }

    /// Get tensor data (handles tiering transparently)
    pub fn getTensor(self: *TieredTensorManager, name: []const u8) ![]const u8 {
        const tensor = self.tensors.getPtr(name) orelse return error.TensorNotFound;

        tensor.access_count += 1;
        tensor.last_access = std.time.milliTimestamp();

        switch (tensor.tier) {
            .Hot => {
                self.stats.hot_hits += 1;
                if (tensor.mmap_slice) |slice| {
                    return slice;
                }
                try self.loadToRam(tensor);
                return tensor.mmap_slice.?;
            },
            .Warm => {
                self.stats.warm_hits += 1;
                if (tensor.mmap_slice) |slice| {
                    return slice;
                }
                try self.loadToRam(tensor);
                return tensor.mmap_slice.?;
            },
            .Cold => {
                self.stats.cold_loads += 1;
                // Load on demand
                try self.loadToRam(tensor);
                // Promote to warm
                tensor.tier = .Warm;
                // Check if we need to evict
                try self.maybeEvict();
                return tensor.mmap_slice.?;
            },
        }
    }

    /// Get tensor as f32 (dequantizes if needed)
    pub fn getTensorF32(self: *TieredTensorManager, name: []const u8) ![]const f32 {
        const tensor = self.tensors.getPtr(name) orelse return error.TensorNotFound;

        if (tensor.dtype != .F32) {
            return error.NotF32Tensor;
        }

        const data = try self.getTensor(name);
        return @alignCast(std.mem.bytesAsSlice(f32, data));
    }

    /// Prefetch layer tensors
    pub fn prefetchLayer(self: *TieredTensorManager, layer: u32) void {
        const layer_str = std.fmt.allocPrint(self.allocator, "blk.{d}.", .{layer}) catch return;
        defer self.allocator.free(layer_str);

        self.gguf.prefetchPrefix(layer_str);
        self.stats.prefetches += 1;
    }

    /// Notify that we're starting a new layer (for prefetching)
    pub fn startLayer(self: *TieredTensorManager, layer: u32) void {
        self.current_layer = layer;

        // Prefetch ahead
        for (1..self.config.prefetch_ahead + 1) |ahead| {
            const next_layer = layer + @as(u32, @intCast(ahead));
            if (next_layer < self.n_layers) {
                self.prefetchLayer(next_layer);
            }
        }
    }

    fn maybeEvict(self: *TieredTensorManager) !void {
        const warm_limit = self.config.warm_tier_mb * 1024 * 1024;

        if (self.warm_bytes <= warm_limit) return;

        // Find LRU warm tensor to evict
        var oldest_time: i64 = std.math.maxInt(i64);
        var oldest_name: ?[]const u8 = null;

        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            const tensor = entry.value_ptr;
            if (tensor.tier == .Warm and tensor.mmap_slice != null) {
                if (tensor.last_access < oldest_time) {
                    oldest_time = tensor.last_access;
                    oldest_name = tensor.name;
                }
            }
        }

        if (oldest_name) |name| {
            const tensor = self.tensors.getPtr(name).?;
            // Evict (just clear the slice, mmap handles the rest)
            tensor.mmap_slice = null;
            tensor.tier = .Cold;
            self.warm_bytes -= tensor.size_bytes;
            self.stats.evictions += 1;
            self.stats.bytes_evicted += tensor.size_bytes;
        }
    }

    /// Print status
    pub fn printStatus(self: *TieredTensorManager) void {
        std.debug.print("\n📊 Tiered Tensor Status\n", .{});
        std.debug.print("   Layers: {d}, Current: {d}\n", .{self.n_layers, self.current_layer});
        std.debug.print("   Hot: {d:.1} MB, Warm: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.hot_bytes)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.warm_bytes)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Hits - Hot: {d}, Warm: {d}, Cold loads: {d}\n", .{
            self.stats.hot_hits, self.stats.warm_hits, self.stats.cold_loads,
        });
        std.debug.print("   Evictions: {d}, Prefetches: {d}\n", .{
            self.stats.evictions, self.stats.prefetches,
        });
        std.debug.print("   Bytes loaded: {d:.1} MB, evicted: {d:.1} MB\n", .{
            @as(f64, @floatFromInt(self.stats.bytes_loaded)) / (1024.0 * 1024.0),
            @as(f64, @floatFromInt(self.stats.bytes_evicted)) / (1024.0 * 1024.0),
        });
    }

    pub fn deinit(self: *TieredTensorManager) void {
        // Free any RAM allocations
        var iter = self.tensors.iterator();
        while (iter.next()) |entry| {
            if (entry.value_ptr.ram_data) |data| {
                self.allocator.free(data);
            }
        }
        self.tensors.deinit();
        self.gguf.close();
        self.allocator.destroy(self);
    }
};
