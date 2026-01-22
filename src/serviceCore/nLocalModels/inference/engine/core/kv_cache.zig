const std = @import("std");

/// KV (Key-Value) Cache for Transformer attention
/// Stores keys and values from previous tokens to enable efficient autoregressive generation

// ============================================================================
// Structures
// ============================================================================

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    
    // Cache dimensions
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    
    // Current position
    seq_pos: u32,
    
    // Storage: [n_layers][2][max_seq_len][n_heads * head_dim]
    // [2] = [keys, values]
    cache: [][]f32,
    
    pub fn init(
        allocator: std.mem.Allocator,
        n_layers: u32,
        n_heads: u32,
        head_dim: u32,
        max_seq_len: u32,
    ) !KVCache {
        std.debug.print("\n🗄️  Initializing KV cache...\n", .{});
        std.debug.print("   Layers: {d}, Heads: {d}, Head dim: {d}\n", .{ n_layers, n_heads, head_dim });
        std.debug.print("   Max sequence length: {d}\n", .{max_seq_len});

        // Clamp max_seq_len to prevent integer overflow and excessive memory usage
        // Most inference doesn't need >32K context, and 128K would use too much RAM
        const max_reasonable_seq: u32 = 32768;
        const effective_max_seq = @min(max_seq_len, max_reasonable_seq);
        if (effective_max_seq != max_seq_len) {
            std.debug.print("   ⚠️  Clamping max_seq_len from {d} to {d} to prevent overflow\n", .{ max_seq_len, effective_max_seq });
        }

        // Use u64 for intermediate calculations to prevent overflow
        const kv_dim: u64 = @as(u64, n_heads) * @as(u64, head_dim);
        const cache_size_per_layer: u64 = 2 * @as(u64, effective_max_seq) * kv_dim; // 2 for keys + values
        const total_size: u64 = @as(u64, n_layers) * cache_size_per_layer;
        
        std.debug.print("   Total cache size: {d} floats ({d:.2} MB)\n", .{
            total_size,
            @as(f64, @floatFromInt(total_size * @sizeOf(f32))) / (1024.0 * 1024.0),
        });

        // Check if allocation size is reasonable (max ~4GB per layer)
        const max_alloc_size: u64 = 1024 * 1024 * 1024; // 1GB max per layer
        if (cache_size_per_layer > max_alloc_size) {
            std.debug.print("   ❌ Cache size per layer ({d}) exceeds maximum ({d})\n", .{ cache_size_per_layer, max_alloc_size });
            return error.CacheSizeTooLarge;
        }

        // Allocate cache storage
        var cache = try allocator.alloc([]f32, n_layers);
        errdefer allocator.free(cache);

        const cache_size_usize: usize = @intCast(cache_size_per_layer);
        for (0..n_layers) |layer| {
            cache[layer] = try allocator.alloc(f32, cache_size_usize);
            errdefer {
                for (0..layer) |l| allocator.free(cache[l]);
                allocator.free(cache);
            }

            // Initialize to zeros
            @memset(cache[layer], 0.0);
        }

        std.debug.print("   ✅ KV cache initialized\n", .{});

        return KVCache{
            .allocator = allocator,
            .n_layers = n_layers,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_seq_len = effective_max_seq,
            .seq_pos = 0,
            .cache = cache,
        };
    }
    
    pub fn deinit(self: *KVCache) void {
        for (self.cache) |layer_cache| {
            self.allocator.free(layer_cache);
        }
        self.allocator.free(self.cache);
    }
    
    /// Store keys and values for a layer at current position
    pub fn store(
        self: *KVCache,
        layer: u32,
        keys: []const f32,
        values: []const f32,
    ) void {
        if (layer >= self.n_layers) return;
        if (self.seq_pos >= self.max_seq_len) return;
        
        const kv_dim = self.n_heads * self.head_dim;
        const layer_cache = self.cache[layer];
        
        // Keys offset: pos * kv_dim
        const keys_offset = self.seq_pos * kv_dim;
        @memcpy(layer_cache[keys_offset .. keys_offset + kv_dim], keys[0..kv_dim]);
        
        // Values offset: max_seq_len * kv_dim + pos * kv_dim
        const values_offset = self.max_seq_len * kv_dim + self.seq_pos * kv_dim;
        @memcpy(layer_cache[values_offset .. values_offset + kv_dim], values[0..kv_dim]);
    }
    
    /// Retrieve all keys up to current position for a layer
    pub fn getKeys(self: *KVCache, layer: u32) []const f32 {
        if (layer >= self.n_layers) return &[_]f32{};
        
        const kv_dim = self.n_heads * self.head_dim;
        const layer_cache = self.cache[layer];
        const len = (self.seq_pos + 1) * kv_dim;
        
        return layer_cache[0..len];
    }
    
    /// Retrieve all values up to current position for a layer
    pub fn getValues(self: *KVCache, layer: u32) []const f32 {
        if (layer >= self.n_layers) return &[_]f32{};
        
        const kv_dim = self.n_heads * self.head_dim;
        const layer_cache = self.cache[layer];
        const values_start = self.max_seq_len * kv_dim;
        const len = (self.seq_pos + 1) * kv_dim;
        
        return layer_cache[values_start .. values_start + len];
    }
    
    /// Get keys for a specific position range
    pub fn getKeysRange(
        self: *KVCache,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
    ) []const f32 {
        if (layer >= self.n_layers) return &[_]f32{};
        if (start_pos > end_pos or end_pos > self.seq_pos) return &[_]f32{};
        
        const kv_dim = self.n_heads * self.head_dim;
        const layer_cache = self.cache[layer];
        const start_offset = start_pos * kv_dim;
        const end_offset = (end_pos + 1) * kv_dim;
        
        return layer_cache[start_offset..end_offset];
    }
    
    /// Get values for a specific position range
    pub fn getValuesRange(
        self: *KVCache,
        layer: u32,
        start_pos: u32,
        end_pos: u32,
    ) []const f32 {
        if (layer >= self.n_layers) return &[_]f32{};
        if (start_pos > end_pos or end_pos > self.seq_pos) return &[_]f32{};
        
        const kv_dim = self.n_heads * self.head_dim;
        const layer_cache = self.cache[layer];
        const values_start = self.max_seq_len * kv_dim;
        const start_offset = values_start + start_pos * kv_dim;
        const end_offset = values_start + (end_pos + 1) * kv_dim;
        
        return layer_cache[start_offset..end_offset];
    }
    
    /// Advance position for next token
    pub fn advance(self: *KVCache) void {
        if (self.seq_pos < self.max_seq_len - 1) {
            self.seq_pos += 1;
        }
    }
    
    /// Reset cache to start
    pub fn reset(self: *KVCache) void {
        self.seq_pos = 0;
        
        // Clear all cache data
        for (self.cache) |layer_cache| {
            @memset(layer_cache, 0.0);
        }
    }
    
    /// Get current sequence position
    pub fn getPosition(self: *KVCache) u32 {
        return self.seq_pos;
    }
    
    /// Get current sequence length (number of cached tokens)
    pub fn getSequenceLength(self: *KVCache) u32 {
        return self.seq_pos + 1;
    }
    
    /// Check if cache is full
    pub fn isFull(self: *KVCache) bool {
        return self.seq_pos >= self.max_seq_len - 1;
    }
    
    /// Get cache statistics
    pub fn getStats(self: *KVCache) CacheStats {
        const kv_dim = self.n_heads * self.head_dim;
        const used_per_layer = (self.seq_pos + 1) * kv_dim * 2; // 2 for keys + values
        const total_per_layer = self.max_seq_len * kv_dim * 2;
        const usage_pct = @as(f32, @floatFromInt(used_per_layer)) / @as(f32, @floatFromInt(total_per_layer)) * 100.0;
        
        return CacheStats{
            .position = self.seq_pos,
            .sequence_length = self.seq_pos + 1,
            .max_length = self.max_seq_len,
            .n_layers = self.n_layers,
            .n_heads = self.n_heads,
            .head_dim = self.head_dim,
            .used_floats = self.n_layers * used_per_layer,
            .total_floats = self.n_layers * total_per_layer,
            .usage_percent = usage_pct,
        };
    }

    /// Gather keys for a specific head into a contiguous buffer
    pub fn gatherHeadKeys(
        self: *KVCache,
        layer: u32,
        head: u32,
        start_pos: u32,
        end_pos: u32,
        dest: []f32,
    ) void {
        const kv_dim = self.n_heads * self.head_dim;
        const head_offset = head * self.head_dim;
        const layer_cache = self.cache[layer];
        const dim = self.head_dim;
        
        var src_idx = start_pos * kv_dim + head_offset;
        var dst_idx: usize = 0;
        
        for (start_pos..end_pos) |_| {
            @memcpy(dest[dst_idx .. dst_idx + dim], layer_cache[src_idx .. src_idx + dim]);
            src_idx += kv_dim;
            dst_idx += dim;
        }
    }

    /// Gather values for a specific head into a contiguous buffer
    pub fn gatherHeadValues(
        self: *KVCache,
        layer: u32,
        head: u32,
        start_pos: u32,
        end_pos: u32,
        dest: []f32,
    ) void {
        const kv_dim = self.n_heads * self.head_dim;
        const head_offset = head * self.head_dim;
        const layer_cache = self.cache[layer];
        const dim = self.head_dim;
        const values_start = self.max_seq_len * kv_dim;
        
        var src_idx = values_start + start_pos * kv_dim + head_offset;
        var dst_idx: usize = 0;
        
        for (start_pos..end_pos) |_| {
            @memcpy(dest[dst_idx .. dst_idx + dim], layer_cache[src_idx .. src_idx + dim]);
            src_idx += kv_dim;
            dst_idx += dim;
        }
    }
};

pub const CacheStats = struct {
    position: u32,
    sequence_length: u32,
    max_length: u32,
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    used_floats: u32,
    total_floats: u32,
    usage_percent: f32,
};

// ============================================================================
// Multi-Head Attention Helpers
// ============================================================================

/// Split QKV tensor into separate heads
pub fn splitHeads(
    output: []f32,
    input: []const f32,
    n_heads: u32,
    head_dim: u32,
) void {
    const seq_len = input.len / (n_heads * head_dim);
    
    // Reshape from [seq_len, n_heads * head_dim] to [n_heads, seq_len, head_dim]
    for (0..n_heads) |head| {
        for (0..seq_len) |pos| {
            for (0..head_dim) |dim| {
                const src_idx = pos * (n_heads * head_dim) + head * head_dim + dim;
                const dst_idx = head * (seq_len * head_dim) + pos * head_dim + dim;
                output[dst_idx] = input[src_idx];
            }
        }
    }
}

/// Merge heads back to single tensor
pub fn mergeHeads(
    output: []f32,
    input: []const f32,
    n_heads: u32,
    head_dim: u32,
) void {
    const seq_len = output.len / (n_heads * head_dim);
    
    // Reshape from [n_heads, seq_len, head_dim] to [seq_len, n_heads * head_dim]
    for (0..seq_len) |pos| {
        for (0..n_heads) |head| {
            for (0..head_dim) |dim| {
                const src_idx = head * (seq_len * head_dim) + pos * head_dim + dim;
                const dst_idx = pos * (n_heads * head_dim) + head * head_dim + dim;
                output[dst_idx] = input[src_idx];
            }
        }
    }
}

// ============================================================================
// Testing
// ============================================================================

pub fn test_kv_cache(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing KV Cache\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test configuration
    const n_layers: u32 = 4;
    const n_heads: u32 = 8;
    const head_dim: u32 = 64;
    const max_seq_len: u32 = 128;
    const kv_dim = n_heads * head_dim;
    
    var cache = try KVCache.init(allocator, n_layers, n_heads, head_dim, max_seq_len);
    defer cache.deinit();
    
    // Test 1: Store and retrieve
    {
        std.debug.print("\n1️⃣  Testing store and retrieve...\n", .{});
        
        // Create test data
        const keys = try allocator.alloc(f32, kv_dim);
        defer allocator.free(keys);
        const values = try allocator.alloc(f32, kv_dim);
        defer allocator.free(values);
        
        // Fill with test pattern
        for (0..kv_dim) |i| {
            keys[i] = @as(f32, @floatFromInt(i));
            values[i] = @as(f32, @floatFromInt(i)) * 2.0;
        }
        
        // Store at position 0, layer 0
        cache.store(0, keys, values);
        
        // Retrieve
        const retrieved_keys = cache.getKeys(0);
        const retrieved_values = cache.getValues(0);
        
        // Verify
        var mismatch = false;
        for (0..kv_dim) |i| {
            if (retrieved_keys[i] != keys[i] or retrieved_values[i] != values[i]) {
                mismatch = true;
                break;
            }
        }
        
        if (mismatch) {
            std.debug.print("   ❌ Retrieved data doesn't match stored data\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Store/retrieve correct\n", .{});
    }
    
    // Test 2: Multiple positions
    {
        std.debug.print("\n2️⃣  Testing multiple positions...\n", .{});
        
        cache.reset();
        
        const keys = try allocator.alloc(f32, kv_dim);
        defer allocator.free(keys);
        const values = try allocator.alloc(f32, kv_dim);
        defer allocator.free(values);
        
        // Store 5 tokens (positions 0-4)
        for (0..5) |pos| {
            // Unique pattern for each position
            for (0..kv_dim) |i| {
                keys[i] = @as(f32, @floatFromInt(pos * 1000 + i));
                values[i] = @as(f32, @floatFromInt(pos * 2000 + i));
            }
            
            cache.store(0, keys, values);
            if (pos < 4) cache.advance(); // Don't advance after last store
        }
        
        // Verify position (should be 4 after storing at positions 0,1,2,3,4)
        if (cache.getPosition() != 4) {
            std.debug.print("   ❌ Position tracking incorrect: {d} vs 4\n", .{cache.getPosition()});
            return error.TestFailed;
        }
        
        // Verify sequence length
        if (cache.getSequenceLength() != 5) {
            std.debug.print("   ❌ Sequence length incorrect: {d} vs 5\n", .{cache.getSequenceLength()});
            return error.TestFailed;
        }
        
        std.debug.print("   Position: {d}, Length: {d}\n", .{ cache.getPosition(), cache.getSequenceLength() });
        std.debug.print("   ✅ Multiple positions working\n", .{});
    }
    
    // Test 3: Range retrieval
    {
        std.debug.print("\n3️⃣  Testing range retrieval...\n", .{});
        
        const keys_range = cache.getKeysRange(0, 1, 3);
        const expected_len = 3 * kv_dim; // Positions 1, 2, 3
        
        if (keys_range.len != expected_len) {
            std.debug.print("   ❌ Range length incorrect: {d} vs {d}\n", .{ keys_range.len, expected_len });
            return error.TestFailed;
        }
        
        std.debug.print("   Retrieved range [1-3]: {d} floats\n", .{keys_range.len});
        std.debug.print("   ✅ Range retrieval correct\n", .{});
    }
    
    // Test 4: Cache statistics
    {
        std.debug.print("\n4️⃣  Testing cache statistics...\n", .{});
        
        const stats = cache.getStats();
        
        std.debug.print("   Position: {d}/{d}\n", .{ stats.position, stats.max_length });
        std.debug.print("   Sequence length: {d}\n", .{stats.sequence_length});
        std.debug.print("   Layers: {d}, Heads: {d}, Head dim: {d}\n", .{
            stats.n_layers,
            stats.n_heads,
            stats.head_dim,
        });
        std.debug.print("   Used: {d}/{d} floats ({d:.1}%)\n", .{
            stats.used_floats,
            stats.total_floats,
            stats.usage_percent,
        });
        
        if (stats.sequence_length != 5) {
            std.debug.print("   ❌ Stats incorrect\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Statistics correct\n", .{});
    }
    
    // Test 5: Reset
    {
        std.debug.print("\n5️⃣  Testing reset...\n", .{});
        
        cache.reset();
        
        if (cache.getPosition() != 0 or cache.getSequenceLength() != 1) {
            std.debug.print("   ❌ Reset failed\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   Position after reset: {d}\n", .{cache.getPosition()});
        std.debug.print("   ✅ Reset working\n", .{});
    }
    
    // Test 6: Head splitting/merging
    {
        std.debug.print("\n6️⃣  Testing head operations...\n", .{});
        
        const seq_len = 4;
        const input_size = seq_len * n_heads * head_dim;
        
        const input = try allocator.alloc(f32, input_size);
        defer allocator.free(input);
        const split = try allocator.alloc(f32, input_size);
        defer allocator.free(split);
        const merged = try allocator.alloc(f32, input_size);
        defer allocator.free(merged);
        
        // Fill with test pattern
        for (0..input_size) |i| {
            input[i] = @as(f32, @floatFromInt(i));
        }
        
        // Split and merge
        splitHeads(split, input, n_heads, head_dim);
        mergeHeads(merged, split, n_heads, head_dim);
        
        // Verify round-trip
        var mismatch = false;
        for (0..input_size) |i| {
            if (merged[i] != input[i]) {
                mismatch = true;
                break;
            }
        }
        
        if (mismatch) {
            std.debug.print("   ❌ Split/merge round-trip failed\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Head operations correct\n", .{});
    }
    
    std.debug.print("\n✅ All KV cache tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
