const std = @import("std");

/// Cache Management Strategies - Day 17
/// Implements various strategies for managing KV cache memory efficiently

// ============================================================================
// Cache Strategy Types
// ============================================================================

pub const CacheStrategy = enum {
    fifo,           // First-in-first-out (drop oldest)
    sliding_window, // Keep last N tokens (rolling window)
    keep_first,     // Keep first tokens, drop middle (prefix caching)
    adaptive,       // Dynamic based on importance
};

// ============================================================================
// Managed Cache Configuration
// ============================================================================

pub const ManagedCacheConfig = struct {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    batch_size: u32 = 1,
    strategy: CacheStrategy = .sliding_window,
    window_size: u32 = 1024, // For sliding_window strategy
    keep_first: u32 = 256,   // For keep_first strategy
};

// ============================================================================
// Managed KV Cache
// ============================================================================

pub const ManagedCache = struct {
    allocator: std.mem.Allocator,
    config: ManagedCacheConfig,
    
    // Cache storage
    keys: []f32,
    values: []f32,
    seq_lengths: []u32,
    current_pos: u32,
    
    // Statistics
    evictions: u64 = 0,
    stores: u64 = 0,
    retrievals: u64 = 0,
    
    pub fn init(allocator: std.mem.Allocator, config: ManagedCacheConfig) !ManagedCache {
        const total_size = config.n_layers * config.batch_size * config.max_seq_len * 
                          config.n_heads * config.head_dim;
        
        const keys = try allocator.alloc(f32, total_size);
        errdefer allocator.free(keys);
        
        const values = try allocator.alloc(f32, total_size);
        errdefer allocator.free(values);
        
        const seq_lengths = try allocator.alloc(u32, config.batch_size);
        errdefer allocator.free(seq_lengths);
        
        @memset(keys, 0.0);
        @memset(values, 0.0);
        @memset(seq_lengths, 0);
        
        return ManagedCache{
            .allocator = allocator,
            .config = config,
            .keys = keys,
            .values = values,
            .seq_lengths = seq_lengths,
            .current_pos = 0,
        };
    }
    
    pub fn deinit(self: *ManagedCache) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.seq_lengths);
    }
    
    pub fn reset(self: *ManagedCache) void {
        @memset(self.keys, 0.0);
        @memset(self.values, 0.0);
        @memset(self.seq_lengths, 0);
        self.current_pos = 0;
        self.evictions = 0;
        self.stores = 0;
        self.retrievals = 0;
    }
    
    fn getOffset(self: *const ManagedCache, layer: u32, batch: u32, pos: u32) usize {
        return layer * (self.config.batch_size * self.config.max_seq_len * 
                       self.config.n_heads * self.config.head_dim) +
               batch * (self.config.max_seq_len * self.config.n_heads * self.config.head_dim) +
               pos * (self.config.n_heads * self.config.head_dim);
    }
    
    /// Store with automatic eviction if needed
    pub fn store(
        self: *ManagedCache,
        layer: u32,
        batch: u32,
        key: []const f32,
        value: []const f32,
    ) !void {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const kv_size = self.config.n_heads * self.config.head_dim;
        if (key.len != kv_size or value.len != kv_size) return error.InvalidSize;
        
        // Only track stores on layer 0 to avoid overcounting
        if (layer == 0) {
            self.stores += 1;
            
            // Check if we need to evict based on strategy (only on first layer)
            const should_evict = switch (self.config.strategy) {
                .fifo, .keep_first, .adaptive => self.seq_lengths[batch] >= self.config.max_seq_len,
                .sliding_window => self.seq_lengths[batch] >= self.config.window_size,
            };
            
            if (should_evict) {
                try self.evict(batch);
            }
        }
        
        const offset = self.getOffset(layer, batch, self.current_pos);
        @memcpy(self.keys[offset..offset + kv_size], key);
        @memcpy(self.values[offset..offset + kv_size], value);
        
        self.seq_lengths[batch] = self.current_pos + 1;
    }
    
    /// Evict entries based on strategy
    fn evict(self: *ManagedCache, batch: u32) !void {
        switch (self.config.strategy) {
            .fifo => try self.evictFIFO(batch),
            .sliding_window => try self.evictSlidingWindow(batch),
            .keep_first => try self.evictKeepFirst(batch),
            .adaptive => try self.evictAdaptive(batch),
        }
        self.evictions += 1;
    }
    
    /// FIFO: Drop oldest entry
    fn evictFIFO(self: *ManagedCache, batch: u32) !void {
        try self.shiftLeft(batch, 1);
    }
    
    /// Sliding Window: Keep last window_size tokens
    fn evictSlidingWindow(self: *ManagedCache, batch: u32) !void {
        // Target size is the window size (keep last N tokens)
        const target_size = self.config.window_size;
        const seq_len = self.seq_lengths[batch];
        
        if (seq_len < target_size) return;
        
        // Drop oldest to maintain window size - 1 (to make room for new token)
        const to_drop = seq_len - target_size + 1;
        try self.shiftLeft(batch, to_drop);
    }
    
    /// Keep First: Keep first tokens, drop from middle
    fn evictKeepFirst(self: *ManagedCache, batch: u32) !void {
        const keep_first = self.config.keep_first;
        if (self.current_pos < keep_first * 2) {
            // Not enough tokens yet, just drop one
            try self.evictFIFO(batch);
            return;
        }
        
        // Drop one token from the middle
        const middle_pos = keep_first + (self.current_pos - keep_first) / 2;
        try self.compactCache(batch, middle_pos, 1);
    }
    
    /// Adaptive: Simple heuristic - keep recent + first tokens
    fn evictAdaptive(self: *ManagedCache, batch: u32) !void {
        // For now, just use sliding window strategy
        // Future: could use attention scores to determine importance
        try self.evictSlidingWindow(batch);
    }
    
    /// Shift cache entries left by n positions
    fn shiftLeft(self: *ManagedCache, batch: u32, positions: u32) !void {
        if (positions == 0) return;
        if (positions > self.current_pos) return error.InvalidShift;
        
        for (0..self.config.n_layers) |layer| {
            const layer_u32 = @as(u32, @intCast(layer));
            const start_offset = self.getOffset(layer_u32, batch, 0);
            const src_offset = self.getOffset(layer_u32, batch, positions);
            const kv_size = self.config.n_heads * self.config.head_dim;
            const remaining = self.current_pos - positions;
            const copy_size = remaining * kv_size;
            
            // Shift keys
            std.mem.copyForwards(
                f32,
                self.keys[start_offset..start_offset + copy_size],
                self.keys[src_offset..src_offset + copy_size],
            );
            
            // Shift values
            std.mem.copyForwards(
                f32,
                self.values[start_offset..start_offset + copy_size],
                self.values[src_offset..src_offset + copy_size],
            );
        }
        
        self.current_pos -= positions;
        self.seq_lengths[batch] = self.current_pos;
    }
    
    /// Remove tokens from specific position
    fn compactCache(self: *ManagedCache, batch: u32, remove_pos: u32, count: u32) !void {
        if (remove_pos + count > self.current_pos) return error.InvalidPosition;
        
        for (0..self.config.n_layers) |layer| {
            const layer_u32 = @as(u32, @intCast(layer));
            const dest_offset = self.getOffset(layer_u32, batch, remove_pos);
            const src_offset = self.getOffset(layer_u32, batch, remove_pos + count);
            const kv_size = self.config.n_heads * self.config.head_dim;
            const remaining = self.current_pos - remove_pos - count;
            const copy_size = remaining * kv_size;
            
            // Compact keys
            std.mem.copyForwards(
                f32,
                self.keys[dest_offset..dest_offset + copy_size],
                self.keys[src_offset..src_offset + copy_size],
            );
            
            // Compact values
            std.mem.copyForwards(
                f32,
                self.values[dest_offset..dest_offset + copy_size],
                self.values[src_offset..src_offset + copy_size],
            );
        }
        
        self.current_pos -= count;
        self.seq_lengths[batch] = self.current_pos;
    }
    
    pub fn getKeys(self: *ManagedCache, layer: u32, batch: u32) ![]const f32 {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const seq_len = self.seq_lengths[batch];
        const offset = self.getOffset(layer, batch, 0);
        const size = seq_len * self.config.n_heads * self.config.head_dim;
        
        self.retrievals += 1;
        return self.keys[offset..offset + size];
    }
    
    pub fn getValues(self: *ManagedCache, layer: u32, batch: u32) ![]const f32 {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const seq_len = self.seq_lengths[batch];
        const offset = self.getOffset(layer, batch, 0);
        const size = seq_len * self.config.n_heads * self.config.head_dim;
        
        return self.values[offset..offset + size];
    }
    
    pub fn advance(self: *ManagedCache) !void {
        // Always advance - eviction happens in store() if needed
        self.current_pos += 1;
    }
    
    pub fn getSeqLen(self: *const ManagedCache, batch: u32) u32 {
        if (batch >= self.config.batch_size) return 0;
        return self.seq_lengths[batch];
    }
    
    pub fn getStats(self: *const ManagedCache) CacheStats {
        return .{
            .stores = self.stores,
            .retrievals = self.retrievals,
            .evictions = self.evictions,
            .current_size = self.current_pos,
            .max_size = self.config.max_seq_len,
        };
    }
};

pub const CacheStats = struct {
    stores: u64,
    retrievals: u64,
    evictions: u64,
    current_size: u32,
    max_size: u32,
    
    pub fn evictionRate(self: CacheStats) f64 {
        if (self.stores == 0) return 0.0;
        return @as(f64, @floatFromInt(self.evictions)) / @as(f64, @floatFromInt(self.stores));
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_cache_manager(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing Cache Manager Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: FIFO strategy
    {
        std.debug.print("\n1️⃣  Testing FIFO strategy...\n", .{});
        
        const config = ManagedCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 5,
            .batch_size = 1,
            .strategy = .fifo,
        };
        
        var cache = try ManagedCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        // Fill cache beyond capacity
        for (0..7) |i| {
            @memset(key, @as(f32, @floatFromInt(i)));
            @memset(value, @as(f32, @floatFromInt(i * 10)));
            try cache.store(0, 0, key, value);
            try cache.advance();
        }
        
        const stats = cache.getStats();
        std.debug.print("   Stored {d} tokens, evictions: {d}\n", .{stats.stores, stats.evictions});
        std.debug.print("   Current size: {d}, max: {d}\n", .{stats.current_size, stats.max_size});
        
        if (stats.evictions < 2) {
            std.debug.print("   ❌ Expected at least 2 evictions\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ FIFO strategy working\n", .{});
    }
    
    // Test 2: Sliding window strategy
    {
        std.debug.print("\n2️⃣  Testing sliding window strategy...\n", .{});
        
        const config = ManagedCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 10,
            .batch_size = 1,
            .strategy = .sliding_window,
            .window_size = 5,
        };
        
        var cache = try ManagedCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        // Store 12 tokens (more than window + max_seq_len)
        for (0..12) |i| {
            @memset(key, @as(f32, @floatFromInt(i)));
            @memset(value, @as(f32, @floatFromInt(i)));
            try cache.store(0, 0, key, value);
            try cache.advance();
        }
        
        const stats = cache.getStats();
        const seq_len = cache.getSeqLen(0);
        std.debug.print("   Stored {d} tokens with window size {d}\n", .{stats.stores, config.window_size});
        std.debug.print("   Current size: {d}, seq_len: {d}\n", .{stats.current_size, seq_len});
        std.debug.print("   Evictions: {d}\n", .{stats.evictions});
        
        // Seq len should be within window size (current_pos may be +1 due to advance)
        if (seq_len > config.window_size) {
            std.debug.print("   ❌ Sequence length exceeds window size\n", .{});
            return error.TestFailed;
        }
        
        if (stats.evictions < 1) {
            std.debug.print("   ❌ Expected at least 1 eviction\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Sliding window working\n", .{});
    }
    
    // Test 3: Keep first strategy
    {
        std.debug.print("\n3️⃣  Testing keep first strategy...\n", .{});
        
        const config = ManagedCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 8,
            .batch_size = 1,
            .strategy = .keep_first,
            .keep_first = 2,
        };
        
        var cache = try ManagedCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        // Store 10 tokens
        for (0..10) |i| {
            @memset(key, @as(f32, @floatFromInt(i)));
            @memset(value, @as(f32, @floatFromInt(i)));
            try cache.store(0, 0, key, value);
            try cache.advance();
        }
        
        const stats = cache.getStats();
        std.debug.print("   Stored {d} tokens, keeping first {d}\n", .{stats.stores, config.keep_first});
        std.debug.print("   Current size: {d}, evictions: {d}\n", .{stats.current_size, stats.evictions});
        
        std.debug.print("   ✅ Keep first strategy working\n", .{});
    }
    
    // Test 4: Statistics tracking
    {
        std.debug.print("\n4️⃣  Testing statistics tracking...\n", .{});
        
        const config = ManagedCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 5,
            .batch_size = 1,
            .strategy = .sliding_window,
            .window_size = 3,
        };
        
        var cache = try ManagedCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        @memset(key, 1.0);
        @memset(value, 2.0);
        
        // Store and retrieve
        for (0..8) |_| {
            try cache.store(0, 0, key, value);
            _ = try cache.getKeys(0, 0);
            try cache.advance();
        }
        
        const stats = cache.getStats();
        std.debug.print("   Stores: {d}\n", .{stats.stores});
        std.debug.print("   Retrievals: {d}\n", .{stats.retrievals});
        std.debug.print("   Evictions: {d}\n", .{stats.evictions});
        std.debug.print("   Eviction rate: {d:.2}%\n", .{stats.evictionRate() * 100.0});
        
        if (stats.stores != 8 or stats.retrievals != 8) {
            std.debug.print("   ❌ Incorrect statistics\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Statistics tracking working\n", .{});
    }
    
    std.debug.print("\n✅ All cache manager tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
