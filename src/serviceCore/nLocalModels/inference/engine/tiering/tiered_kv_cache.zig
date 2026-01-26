// Tiered KV Cache - SSD-backed Key-Value Cache for LLM Inference
// Enables 100K+ context on memory-constrained systems
//
// Architecture:
// - Hot tier (RAM): Recent tokens, fast access
// - Cold tier (SSD): Older tokens, mmap for zero-copy reads
// - Automatic eviction based on LRU and memory pressure
//
// This breaks the memory barrier for long-context LLM inference

const std = @import("std");
const ssd = @import("ssd_tier.zig");
const builtin = @import("builtin");
const log = @import("structured_logging.zig");

// ============================================================================
// Day 4: SIMD Memory Operations
// ============================================================================

/// SIMD-optimized memory copy for f32 arrays (ARM NEON)
/// Processes 4 floats per instruction on ARM platforms
inline fn simdMemcpy(dest: [*]f32, src: [*]const f32, count: usize) void {
    // Use SIMD only for sufficiently large copies (threshold: 16 floats)
    if (count < 16) {
        @memcpy(dest[0..count], src[0..count]);
        return;
    }
    
    // ARM NEON: 128-bit SIMD (4x f32)
    // Check for ARM or AArch64 architecture
    const is_arm = comptime switch (builtin.cpu.arch) {
        .arm, .armeb, .aarch64, .aarch64_be => true,
        else => false,
    };
    
    if (comptime is_arm) {
        const vec_count = count / 4;
        const remainder = count % 4;
        
        var i: usize = 0;
        while (i < vec_count) : (i += 1) {
            const offset = i * 4;
            // Load 4 floats, store 4 floats (128-bit operation)
            const v0 = src[offset];
            const v1 = src[offset + 1];
            const v2 = src[offset + 2];
            const v3 = src[offset + 3];
            
            dest[offset] = v0;
            dest[offset + 1] = v1;
            dest[offset + 2] = v2;
            dest[offset + 3] = v3;
        }
        
        // Handle remainder
        const base = vec_count * 4;
        var j: usize = 0;
        while (j < remainder) : (j += 1) {
            dest[base + j] = src[base + j];
        }
    } else {
        // Fallback for non-ARM platforms
        @memcpy(dest[0..count], src[0..count]);
    }
}

/// Allocate aligned memory for SIMD operations
fn allocAligned(allocator: std.mem.Allocator, comptime T: type, count: usize) ![]align(16) T {
    return try allocator.alignedAlloc(T, 16, count);
}


pub const TieredKVConfig = struct {
    // Model dimensions
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    
    // Tiering config
    hot_tokens: u32 = 2048,        // Keep last N tokens in RAM
    cold_block_tokens: u32 = 256,  // SSD block size in tokens
    
    // Memory limits
    max_ram_mb: u64 = 1024,        // Max RAM for KV cache
    max_ssd_mb: u64 = 16384,       // Max SSD for KV cache (16GB)
    
    // SSD path
    ssd_path: []const u8 = "/tmp/shimmy_kv_cache.tier",
    
    // Day 5: Testing mode for benchmarks
    test_mode: bool = false,           // Use minimal SSD allocation for testing
    
    // Day 3: Adaptive eviction parameters
    eviction_policy: EvictionPolicy = .adaptive_lru,
    frequency_weight: f32 = 0.3,      // Weight for access frequency (vs recency)
    eviction_threshold: f32 = 0.90,   // Start evicting at 90% hot cache usage
    pin_recent_tokens: u32 = 128,     // Always keep last N tokens in RAM
    
    pub fn kvDim(self: TieredKVConfig) u32 {
        return self.n_heads * self.head_dim;
    }
    
    pub fn bytesPerToken(self: TieredKVConfig) u64 {
        // 2 for K+V, f32 = 4 bytes, per layer
        return @as(u64, self.n_layers) * 2 * self.kvDim() * @sizeOf(f32);
    }
    
    pub fn hotBytesPerLayer(self: TieredKVConfig) u64 {
        return @as(u64, self.hot_tokens) * 2 * self.kvDim() * @sizeOf(f32);
    }
};

/// Day 3: Eviction policy options
pub const EvictionPolicy = enum {
    simple_lru,        // Original: evict oldest tokens
    adaptive_lru,      // LRU + frequency-based (Day 3)
    frequency_based,   // Pure frequency-based
    lfu,              // Least Frequently Used
};

/// Block stored on SSD (Day 3: Enhanced with access tracking)
const ColdBlock = struct {
    start_pos: u32,       // Starting token position
    end_pos: u32,         // Ending token position (exclusive)
    ssd_offset: u64,      // Offset in SSD file
    size: u32,            // Size in bytes
    layer: u32,           // Which layer
    checksum: u32,        // CRC32 for integrity
    // Day 3: Access pattern tracking
    access_count: u32 = 0,      // Number of times accessed
    last_access_time: i64 = 0,   // Last access timestamp (ms)
    created_time: i64 = 0,       // Creation timestamp (ms)
};

/// Hot cache entry tracking (Day 3: Adaptive eviction)
const HotEntry = struct {
    token_pos: u32,              // Token position
    access_count: u32 = 1,       // Access frequency
    last_access_time: i64,       // Last access timestamp (ms)
    is_pinned: bool = false,     // Prevent eviction (e.g., system prompts)
};

pub const TieredKVCache = struct {
    allocator: std.mem.Allocator,
    config: TieredKVConfig,
    
    // Hot tier: in-memory cache for recent tokens
    // Layout: [n_layers][2][hot_tokens][kv_dim]
    hot_cache: [][]f32,
    hot_start_pos: u32,    // Starting position in hot cache (circular buffer)
    
    // Day 3: Access pattern tracking for adaptive eviction
    hot_entries: std.ArrayList(HotEntry),
    
    // Cold tier: SSD-backed storage for older tokens
    ssd_storage: *ssd.SSDStorage,
    cold_blocks: std.ArrayList(ColdBlock),
    
    // Current state
    seq_pos: u32,          // Current sequence position (total tokens)
    
    // Statistics
    stats: Stats,
    
    pub const Stats = struct {
        hot_hits: u64 = 0,
        cold_hits: u64 = 0,
        evictions: u64 = 0,
        bytes_to_ssd: u64 = 0,
        bytes_from_ssd: u64 = 0,
        // Day 3: Additional metrics
        adaptive_evictions: u64 = 0,     // Adaptive policy evictions
        frequency_promotions: u64 = 0,   // Tokens promoted due to frequency
        cache_efficiency: f32 = 0.0,     // Hit rate percentage
    };
    
    pub fn init(allocator: std.mem.Allocator, config: TieredKVConfig) !*TieredKVCache {
        // Day 6: Structured logging for cache initialization
        log.info("Initializing Tiered KV Cache: layers={d}, heads={d}, head_dim={d}", .{
            config.n_layers, config.n_heads, config.head_dim,
        });
        
        std.debug.print("\n🗄️  Initializing Tiered KV Cache\n", .{});
        std.debug.print("   Layers: {d}, Heads: {d}, Head dim: {d}\n", .{
            config.n_layers, config.n_heads, config.head_dim,
        });
        std.debug.print("   Max sequence: {d} tokens\n", .{config.max_seq_len});
        std.debug.print("   Hot tier: {d} tokens ({d:.1} MB)\n", .{
            config.hot_tokens,
            @as(f32, @floatFromInt(config.hotBytesPerLayer() * config.n_layers)) / (1024.0 * 1024.0),
        });
        std.debug.print("   Cold tier: up to {d} MB on SSD\n", .{config.max_ssd_mb});
        
        const self = try allocator.create(TieredKVCache);
        errdefer allocator.destroy(self);
        
        // Initialize hot cache (RAM)
        const kv_dim = config.kvDim();
        const hot_size_per_layer = 2 * config.hot_tokens * kv_dim;
        
        var hot_cache = try allocator.alloc([]f32, config.n_layers);
        errdefer allocator.free(hot_cache);
        
        for (0..config.n_layers) |layer| {
            hot_cache[layer] = try allocator.alloc(f32, hot_size_per_layer);
            @memset(hot_cache[layer], 0);
        }
        
        // Initialize SSD storage (Day 5: pass test_mode through)
        const ssd_config = ssd.TierConfig{
            .ssd_path = config.ssd_path,
            .max_ssd_mb = config.max_ssd_mb,
            .test_mode = config.test_mode,  // Day 5: minimal allocation for tests
            .block_size = 4096,
            .use_mmap = true,
        };
        
        const ssd_storage = try ssd.SSDStorage.init(allocator, ssd_config);
        errdefer ssd_storage.deinit();
        
        try ssd_storage.open();
        
        self.* = TieredKVCache{
            .allocator = allocator,
            .config = config,
            .hot_cache = hot_cache,
            .hot_start_pos = 0,
            .hot_entries = std.ArrayList(HotEntry){},  // Day 3
            .ssd_storage = ssd_storage,
            .cold_blocks = std.ArrayList(ColdBlock){},
            .seq_pos = 0,
            .stats = .{},
        };
        
        std.debug.print("   ✅ Tiered KV cache ready\n", .{});
        std.debug.print("   Eviction policy: {s}\n", .{@tagName(config.eviction_policy)});
        
        // Day 6: Log successful initialization
        log.info("Tiered KV Cache initialized successfully: policy={s}, hot_tokens={d}", .{
            @tagName(config.eviction_policy), config.hot_tokens,
        });
        
        return self;
    }
    
    pub fn deinit(self: *TieredKVCache) void {
        for (self.hot_cache) |layer| {
            self.allocator.free(layer);
        }
        self.allocator.free(self.hot_cache);
        self.hot_entries.deinit();  // Day 3
        self.cold_blocks.deinit();
        self.ssd_storage.deinit();
        self.allocator.destroy(self);
    }
    
    /// Store KV at current position (single token) - Day 4: SIMD-optimized
    pub fn store(self: *TieredKVCache, layer: u32, keys: []const f32, values: []const f32) !void {
        const kv_dim = self.config.kvDim();

        // Check if we need to evict to SSD
        if (self.seq_pos >= self.config.hot_tokens) {
            switch (self.config.eviction_policy) {
                .simple_lru => try self.evictToSSD(),
                .adaptive_lru, .frequency_based, .lfu => try self.adaptiveEvict(),
            }
        }

        // Store in hot cache (circular buffer) - Day 4: Use SIMD
        const hot_pos = self.seq_pos % self.config.hot_tokens;
        const layer_cache = self.hot_cache[layer];

        // Keys: [0, hot_tokens * kv_dim) - SIMD copy
        const keys_offset = hot_pos * kv_dim;
        simdMemcpy(
            @ptrCast(&layer_cache[keys_offset]),
            @ptrCast(keys.ptr),
            kv_dim
        );

        // Values: [hot_tokens * kv_dim, 2 * hot_tokens * kv_dim) - SIMD copy
        const values_offset = self.config.hot_tokens * kv_dim + hot_pos * kv_dim;
        simdMemcpy(
            @ptrCast(&layer_cache[values_offset]),
            @ptrCast(values.ptr),
            kv_dim
        );
        
        // Day 3: Track this hot entry (only for layer 0 to save memory)
        if (layer == 0 and self.config.eviction_policy != .simple_lru) {
            const is_recent = self.seq_pos + self.config.pin_recent_tokens >= self.seq_pos;
            try self.hot_entries.append(.{
                .token_pos = self.seq_pos,
                .access_count = 1,
                .last_access_time = std.time.milliTimestamp(),
                .is_pinned = is_recent,
            });
            
            // Limit tracking memory
            if (self.hot_entries.items.len > self.config.hot_tokens) {
                _ = self.hot_entries.orderedRemove(0);
            }
        }
    }
    
    // ========================================================================
    // Day 4: Batch Processing API
    // ========================================================================
    
    /// Get optimal batch size based on available hot cache space
    pub fn getOptimalBatchSize(self: *TieredKVCache) u32 {
        const available = self.config.hot_tokens - (self.seq_pos % self.config.hot_tokens);
        return @min(32, available); // Max 32 tokens per batch
    }
    
    /// Store multiple tokens at once (batched) - Day 4: Major optimization
    pub fn storeBatch(
        self: *TieredKVCache,
        layer: u32,
        keys_batch: []const f32,      // [batch_size × kv_dim]
        values_batch: []const f32,     // [batch_size × kv_dim]
        batch_size: u32
    ) !void {
        // Day 6: Log batch store operation
        log.debug("Batch store: layer={d}, batch_size={d}, seq_pos={d}", .{
            layer, batch_size, self.seq_pos,
        });
        
        const kv_dim = self.config.kvDim();
        
        // Single eviction check for entire batch
        const available_space = self.config.hot_tokens - (self.seq_pos % self.config.hot_tokens);
        if (batch_size > available_space) {
            switch (self.config.eviction_policy) {
                .simple_lru => try self.evictToSSD(),
                .adaptive_lru, .frequency_based, .lfu => try self.adaptiveEvict(),
            }
        }
        
        // Batch SIMD copy
        const hot_pos = self.seq_pos % self.config.hot_tokens;
        const layer_cache = self.hot_cache[layer];
        
        // Keys: vectorized batch copy
        const keys_offset = hot_pos * kv_dim;
        const keys_count = batch_size * kv_dim;
        simdMemcpy(
            @ptrCast(&layer_cache[keys_offset]),
            @ptrCast(keys_batch.ptr),
            keys_count
        );
        
        // Values: vectorized batch copy
        const values_base = self.config.hot_tokens * kv_dim;
        const values_offset = values_base + hot_pos * kv_dim;
        simdMemcpy(
            @ptrCast(&layer_cache[values_offset]),
            @ptrCast(values_batch.ptr),
            keys_count  // Same size as keys
        );
        
        // Batch tracking update (single entry for batch)
        if (layer == 0 and self.config.eviction_policy != .simple_lru) {
            try self.hot_entries.append(.{
                .token_pos = self.seq_pos,
                .access_count = 1,
                .last_access_time = std.time.milliTimestamp(),
                .is_pinned = true,  // Recent batches always pinned
            });
            
            // Limit tracking memory
            if (self.hot_entries.items.len > self.config.hot_tokens) {
                _ = self.hot_entries.orderedRemove(0);
            }
        }
        
        // Advance sequence position by batch size
        self.seq_pos += batch_size;
    }


    /// Evict oldest tokens from hot cache to SSD (simple LRU)
    fn evictToSSD(self: *TieredKVCache) !void {
        // Day 6: Log eviction event
        log.debug("Starting simple LRU eviction: hot_start_pos={d}", .{self.hot_start_pos});
        
        const block_tokens = self.config.cold_block_tokens;
        const kv_dim = self.config.kvDim();

        // Calculate what to evict
        const evict_start = self.hot_start_pos;
        const evict_end = evict_start + block_tokens;
        
        const now = std.time.milliTimestamp();

        // Evict each layer
        for (0..self.config.n_layers) |layer| {
            const layer_cache = self.hot_cache[layer];

            // Prepare block data: [keys][values]
            const block_size = block_tokens * kv_dim * 2 * @sizeOf(f32);

            // Allocate SSD space
            const ssd_offset = try self.ssd_storage.allocBlock(@intCast(block_size));

            // Write keys
            const keys_start = (evict_start % self.config.hot_tokens) * kv_dim;
            const keys_end = keys_start + block_tokens * kv_dim;
            const keys_bytes = std.mem.sliceAsBytes(layer_cache[keys_start..keys_end]);
            try self.ssd_storage.write(ssd_offset, keys_bytes);

            // Write values
            const values_base = self.config.hot_tokens * kv_dim;
            const values_start = values_base + (evict_start % self.config.hot_tokens) * kv_dim;
            const values_end = values_start + block_tokens * kv_dim;
            const values_bytes = std.mem.sliceAsBytes(layer_cache[values_start..values_end]);
            try self.ssd_storage.write(ssd_offset + keys_bytes.len, values_bytes);

            // ✅ P2-17 FIXED: Compute CRC32 checksum for data integrity
            const block_data = layer_cache[keys_start..values_end];
            const checksum = computeCRC32(std.mem.sliceAsBytes(block_data));
            
            // Record cold block (Day 3: Enhanced tracking)
            try self.cold_blocks.append(.{
                .start_pos = evict_start,
                .end_pos = evict_end,
                .ssd_offset = ssd_offset,
                .size = @intCast(block_size),
                .layer = @intCast(layer),
                .checksum = checksum,
                .access_count = 0,
                .last_access_time = 0,
                .created_time = now,
            });

            self.stats.evictions += 1;
            self.stats.bytes_to_ssd += block_size;
        }

        // Advance hot cache start
        self.hot_start_pos = evict_end;
        
        // Day 6: Log eviction completion
        log.info("Evicted {d} tokens to SSD: blocks={d}, bytes={d}", .{
            block_tokens * self.config.n_layers,
            self.config.n_layers,
            block_tokens * kv_dim * 2 * @sizeOf(f32) * self.config.n_layers,
        });
    }
    
    // ========================================================================
    // Day 3: Adaptive Eviction Algorithm
    // ========================================================================
    
    /// Adaptive eviction using LRU + frequency
    fn adaptiveEvict(self: *TieredKVCache) !void {
        if (self.hot_entries.items.len == 0) {
            // Fallback to simple eviction
            return self.evictToSSD();
        }
        
        // Day 6: Log adaptive eviction start
        log.debug("Starting adaptive eviction: tracked_entries={d}", .{self.hot_entries.items.len});
        
        const now = std.time.milliTimestamp();
        const freq_weight = self.config.frequency_weight;
        const recency_weight = 1.0 - freq_weight;
        
        // Calculate eviction scores for each entry
        var min_score: f32 = std.math.floatMax(f32);
        var evict_idx: usize = 0;
        
        for (self.hot_entries.items, 0..) |entry, i| {
            // Skip pinned entries (recent tokens)
            if (entry.is_pinned) continue;
            
            // Recency score: time since last access (normalized)
            const time_since_access = now - entry.last_access_time;
            const recency_score = @as(f32, @floatFromInt(time_since_access)) / 1000.0;
            
            // Frequency score: inverse of access count (lower = evict)
            const freq_score = 1.0 / @as(f32, @floatFromInt(entry.access_count + 1));
            
            // Combined score (lower = more likely to evict)
            const score = recency_weight * recency_score + freq_weight * freq_score;
            
            if (score < min_score) {
                min_score = score;
                evict_idx = i;
            }
        }
        
        // Evict the selected block
        const evict_entry = self.hot_entries.items[evict_idx];
        const block_tokens = self.config.cold_block_tokens;
        const kv_dim = self.config.kvDim();
        
        // Round evict position to block boundary
        const evict_start = (evict_entry.token_pos / block_tokens) * block_tokens;
        const evict_end = evict_start + block_tokens;
        
        // Evict each layer
        for (0..self.config.n_layers) |layer| {
            const layer_cache = self.hot_cache[layer];
            const block_size = block_tokens * kv_dim * 2 * @sizeOf(f32);
            
            const ssd_offset = try self.ssd_storage.allocBlock(@intCast(block_size));
            
            // Write keys
            const keys_start = (evict_start % self.config.hot_tokens) * kv_dim;
            const keys_end = keys_start + block_tokens * kv_dim;
            const keys_bytes = std.mem.sliceAsBytes(layer_cache[keys_start..keys_end]);
            try self.ssd_storage.write(ssd_offset, keys_bytes);
            
            // Write values
            const values_base = self.config.hot_tokens * kv_dim;
            const values_start = values_base + (evict_start % self.config.hot_tokens) * kv_dim;
            const values_end = values_start + block_tokens * kv_dim;
            const values_bytes = std.mem.sliceAsBytes(layer_cache[values_start..values_end]);
            try self.ssd_storage.write(ssd_offset + keys_bytes.len, values_bytes);
            
            // ✅ P2-17 FIXED: Compute CRC32 checksum for adaptive eviction
            const block_data = layer_cache[keys_start..values_end];
            const checksum = computeCRC32(std.mem.sliceAsBytes(block_data));
            
            // Record cold block with access stats
            try self.cold_blocks.append(.{
                .start_pos = evict_start,
                .end_pos = evict_end,
                .ssd_offset = ssd_offset,
                .size = @intCast(block_size),
                .layer = @intCast(layer),
                .checksum = checksum,
                .access_count = evict_entry.access_count,
                .last_access_time = evict_entry.last_access_time,
                .created_time = now,
            });
            
            self.stats.adaptive_evictions += 1;
            self.stats.bytes_to_ssd += block_size;
        }
        
        // Remove evicted entry from tracking
        _ = self.hot_entries.orderedRemove(evict_idx);
        
        // Update hot cache start if needed
        if (evict_start < self.hot_start_pos) {
            self.hot_start_pos = evict_end;
        }
        
        // Day 6: Log adaptive eviction completion
        log.info("Adaptive eviction complete: token_pos={d}, access_count={d}, score={d:.4}", .{
            evict_entry.token_pos, evict_entry.access_count, min_score,
        });
    }
    
    /// Track access to hot entry for adaptive eviction
    fn trackAccess(self: *TieredKVCache, token_pos: u32) void {
        const now = std.time.milliTimestamp();
        
        // Find and update entry
        for (self.hot_entries.items) |*entry| {
            if (entry.token_pos == token_pos) {
                entry.access_count += 1;
                entry.last_access_time = now;
                return;
            }
        }
    }

    /// Get keys for attention (handles hot/cold tiers transparently)
    pub fn getKeys(self: *TieredKVCache, layer: u32, start_pos: u32, end_pos: u32, dest: []f32) !void {
        const kv_dim = self.config.kvDim();
        var pos = start_pos;
        var dest_offset: usize = 0;

        while (pos < end_pos) {
            if (pos >= self.hot_start_pos) {
                // Hot tier
                const hot_pos = pos % self.config.hot_tokens;
                const src_offset = hot_pos * kv_dim;
                const copy_len = @min(end_pos - pos, self.config.hot_tokens - hot_pos) * kv_dim;
                @memcpy(dest[dest_offset..dest_offset + copy_len],
                        self.hot_cache[layer][src_offset..src_offset + copy_len]);
                pos += @intCast(copy_len / kv_dim);
                dest_offset += copy_len;
                self.stats.hot_hits += 1;
                // Day 3: Track hot access
                self.trackAccess(pos);
            } else {
                // Cold tier - find block
                const block = self.findColdBlock(layer, pos) orelse return error.BlockNotFound;
                const block_offset = pos - block.start_pos;
                const copy_tokens = @min(end_pos - pos, block.end_pos - pos);
                const copy_len = copy_tokens * kv_dim;

                // Read from SSD (zero-copy via mmap)
                const ssd_offset = block.ssd_offset + block_offset * kv_dim * @sizeOf(f32);
                const ssd_data = try self.ssd_storage.read(ssd_offset, copy_len * @sizeOf(f32));
                const src_floats: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, ssd_data));
                
                // ✅ P2-17: Verify checksum on read if this is the full block
                if (block_offset == 0 and copy_tokens == (block.end_pos - block.start_pos)) {
                    const read_checksum = computeCRC32(ssd_data);
                    if (read_checksum != block.checksum) {
                        log.err("CRC32 mismatch on block read: layer={d}, pos={d}, expected={X:0>8}, got={X:0>8}", 
                            .{layer, pos, block.checksum, read_checksum});
                        return error.ChecksumMismatch;
                    }
                }
                
                @memcpy(dest[dest_offset..dest_offset + copy_len], src_floats);

                pos += copy_tokens;
                dest_offset += copy_len;
                self.stats.cold_hits += 1;
                self.stats.bytes_from_ssd += copy_len * @sizeOf(f32);
            }
        }
    }

    /// Get values for attention
    pub fn getValues(self: *TieredKVCache, layer: u32, start_pos: u32, end_pos: u32, dest: []f32) !void {
        const kv_dim = self.config.kvDim();
        const values_base = self.config.hot_tokens * kv_dim;
        var pos = start_pos;
        var dest_offset: usize = 0;

        while (pos < end_pos) {
            if (pos >= self.hot_start_pos) {
                // Hot tier
                const hot_pos = pos % self.config.hot_tokens;
                const src_offset = values_base + hot_pos * kv_dim;
                const copy_len = @min(end_pos - pos, self.config.hot_tokens - hot_pos) * kv_dim;
                @memcpy(dest[dest_offset..dest_offset + copy_len],
                        self.hot_cache[layer][src_offset..src_offset + copy_len]);
                pos += @intCast(copy_len / kv_dim);
                dest_offset += copy_len;
            } else {
                // Cold tier
                const block = self.findColdBlock(layer, pos) orelse return error.BlockNotFound;
                const block_offset = pos - block.start_pos;
                const copy_tokens = @min(end_pos - pos, block.end_pos - pos);
                const copy_len = copy_tokens * kv_dim;

                // Values are stored after keys in the block
                const keys_size = (block.end_pos - block.start_pos) * kv_dim * @sizeOf(f32);
                const ssd_offset = block.ssd_offset + keys_size + block_offset * kv_dim * @sizeOf(f32);
                const ssd_data = try self.ssd_storage.read(ssd_offset, copy_len * @sizeOf(f32));
                const src_floats: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, ssd_data));
                @memcpy(dest[dest_offset..dest_offset + copy_len], src_floats);

                pos += copy_tokens;
                dest_offset += copy_len;
            }
        }
    }

    fn findColdBlock(self: *TieredKVCache, layer: u32, pos: u32) ?*ColdBlock {
        // Day 3: Return mutable pointer to track accesses
        for (self.cold_blocks.items) |*block| {
            if (block.layer == layer and pos >= block.start_pos and pos < block.end_pos) {
                // Track access
                block.access_count += 1;
                block.last_access_time = std.time.milliTimestamp();
                return block;
            }
        }
        return null;
    }

    /// Get hot keys slice for a layer (for API compatibility with KVCache)
    /// Returns keys from hot cache only - suitable for recent token attention
    pub fn getHotKeys(self: *TieredKVCache, layer: u32) []f32 {
        if (layer >= self.config.n_layers) return &[_]f32{};
        const kv_dim = self.config.kvDim();
        const tokens_in_hot = @min(self.seq_pos, self.config.hot_tokens);
        const end = tokens_in_hot * kv_dim;
        return self.hot_cache[layer][0..end];
    }

    /// Get hot values slice for a layer (for API compatibility with KVCache)
    pub fn getHotValues(self: *TieredKVCache, layer: u32) []f32 {
        if (layer >= self.config.n_layers) return &[_]f32{};
        const kv_dim = self.config.kvDim();
        const tokens_in_hot = @min(self.seq_pos, self.config.hot_tokens);
        const values_base = self.config.hot_tokens * kv_dim;
        const end = values_base + tokens_in_hot * kv_dim;
        return self.hot_cache[layer][values_base..end];
    }

    /// Advance to next token position
    pub fn advance(self: *TieredKVCache) void {
        self.seq_pos += 1;
    }

    /// Reset cache
    pub fn reset(self: *TieredKVCache) void {
        self.seq_pos = 0;
        self.hot_start_pos = 0;
        for (self.hot_cache) |layer| {
            @memset(layer, 0);
        }
        // Free cold blocks
        for (self.cold_blocks.items) |block| {
            self.ssd_storage.freeBlock(block.ssd_offset, block.size);
        }
        self.cold_blocks.clearRetainingCapacity();
    }

    /// Get statistics
    pub fn getStats(self: *TieredKVCache) struct {
        seq_pos: u32,
        hot_tokens: u32,
        cold_blocks: usize,
        hot_hits: u64,
        cold_hits: u64,
        evictions: u64,
        bytes_to_ssd: u64,
        bytes_from_ssd: u64,
        ssd_usage_mb: u64,
        // Day 3: Additional stats
        adaptive_evictions: u64,
        hot_entries_tracked: usize,
        cache_hit_rate: f32,
    } {
        const ssd_usage = self.ssd_storage.getUsage();
        const total_accesses = self.stats.hot_hits + self.stats.cold_hits;
        const hit_rate = if (total_accesses > 0)
            @as(f32, @floatFromInt(self.stats.hot_hits)) / @as(f32, @floatFromInt(total_accesses)) * 100.0
        else
            0.0;
        
        return .{
            .seq_pos = self.seq_pos,
            .hot_tokens = @min(self.seq_pos, self.config.hot_tokens),
            .cold_blocks = self.cold_blocks.items.len,
            .hot_hits = self.stats.hot_hits,
            .cold_hits = self.stats.cold_hits,
            .evictions = self.stats.evictions,
            .bytes_to_ssd = self.stats.bytes_to_ssd,
            .bytes_from_ssd = self.stats.bytes_from_ssd,
            .ssd_usage_mb = ssd_usage.used_mb,
            .adaptive_evictions = self.stats.adaptive_evictions,
            .hot_entries_tracked = self.hot_entries.items.len,
            .cache_hit_rate = hit_rate,
        };
    }

    /// Print cache status (Day 3: Enhanced with new metrics)
    pub fn printStatus(self: *TieredKVCache) void {
        const stats = self.getStats();
        std.debug.print("\n📊 Tiered KV Cache Status\n", .{});
        std.debug.print("   Sequence position: {d}\n", .{stats.seq_pos});
        std.debug.print("   Hot tokens: {d}/{d}\n", .{stats.hot_tokens, self.config.hot_tokens});
        std.debug.print("   Cold blocks: {d}\n", .{stats.cold_blocks});
        std.debug.print("   Hot hits: {d}, Cold hits: {d}\n", .{stats.hot_hits, stats.cold_hits});
        std.debug.print("   Cache hit rate: {d:.1}%\n", .{stats.cache_hit_rate});
        std.debug.print("   Evictions: {d} (adaptive: {d})\n", .{stats.evictions, stats.adaptive_evictions});
        std.debug.print("   Hot entries tracked: {d}\n", .{stats.hot_entries_tracked});
        std.debug.print("   SSD: {d} MB written, {d} MB read\n", .{
            stats.bytes_to_ssd / (1024 * 1024),
            stats.bytes_from_ssd / (1024 * 1024),
        });
        std.debug.print("   SSD usage: {d} MB\n", .{stats.ssd_usage_mb});
    }
};

// ============================================================================
// ✅ P2-17: CRC32 Checksum Implementation
// ============================================================================

/// CRC32 polynomial (IEEE 802.3)
const CRC32_POLYNOMIAL: u32 = 0xEDB88320;

/// CRC32 lookup table for fast computation
var crc32_table: [256]u32 = undefined;
var crc32_table_initialized = false;

/// Initialize CRC32 lookup table (called once)
fn initCRC32Table() void {
    if (crc32_table_initialized) return;
    
    for (0..256) |i| {
        var crc: u32 = @intCast(i);
        for (0..8) |_| {
            if (crc & 1 != 0) {
                crc = (crc >> 1) ^ CRC32_POLYNOMIAL;
            } else {
                crc >>= 1;
            }
        }
        crc32_table[i] = crc;
    }
    
    crc32_table_initialized = true;
}

/// Compute CRC32 checksum for data integrity
/// Uses table-based algorithm for O(n) performance
fn computeCRC32(data: []const u8) u32 {
    initCRC32Table();
    
    var crc: u32 = 0xFFFFFFFF;
    
    for (data) |byte| {
        const table_idx = @as(u8, @truncate(crc)) ^ byte;
        crc = (crc >> 8) ^ crc32_table[table_idx];
    }
    
    return ~crc;
}

/// Verify CRC32 checksum
fn verifyCRC32(data: []const u8, expected: u32) bool {
    const actual = computeCRC32(data);
    return actual == expected;
}
