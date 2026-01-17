const std = @import("std");

/// KV Cache Implementation - Day 16
/// Stores key-value pairs for transformer attention to avoid recomputation

// ============================================================================
// Configuration
// ============================================================================

pub const KVCacheConfig = struct {
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    batch_size: u32 = 1,
    
    pub fn memorySize(self: KVCacheConfig) usize {
        // Keys + Values: 2 * n_layers * batch_size * max_seq_len * n_heads * head_dim * sizeof(f32)
        const per_kv = self.n_layers * self.batch_size * self.max_seq_len * 
                       self.n_heads * self.head_dim;
        return per_kv * 2 * @sizeOf(f32);
    }
    
    pub fn default() KVCacheConfig {
        return .{
            .n_layers = 12,
            .n_heads = 12,
            .head_dim = 64,
            .max_seq_len = 2048,
            .batch_size = 1,
        };
    }
};

// ============================================================================
// KV Cache Structure
// ============================================================================

pub const KVCache = struct {
    allocator: std.mem.Allocator,
    config: KVCacheConfig,
    
    // Cache storage: [n_layers][batch_size][max_seq_len][n_heads][head_dim]
    keys: []f32,
    values: []f32,
    
    // Sequence lengths for each batch item
    seq_lengths: []u32,
    
    // Current position in cache
    current_pos: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: KVCacheConfig) !KVCache {
        const total_size = config.n_layers * config.batch_size * config.max_seq_len * 
                          config.n_heads * config.head_dim;
        
        const keys = try allocator.alloc(f32, total_size);
        errdefer allocator.free(keys);
        
        const values = try allocator.alloc(f32, total_size);
        errdefer allocator.free(values);
        
        const seq_lengths = try allocator.alloc(u32, config.batch_size);
        errdefer allocator.free(seq_lengths);
        
        // Initialize to zero
        @memset(keys, 0.0);
        @memset(values, 0.0);
        @memset(seq_lengths, 0);
        
        return KVCache{
            .allocator = allocator,
            .config = config,
            .keys = keys,
            .values = values,
            .seq_lengths = seq_lengths,
            .current_pos = 0,
        };
    }
    
    pub fn deinit(self: *KVCache) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.seq_lengths);
    }
    
    /// Reset cache to empty state
    pub fn reset(self: *KVCache) void {
        @memset(self.keys, 0.0);
        @memset(self.values, 0.0);
        @memset(self.seq_lengths, 0);
        self.current_pos = 0;
    }
    
    /// Get offset for accessing cache at specific position
    fn getOffset(self: *const KVCache, layer: u32, batch: u32, pos: u32) usize {
        return layer * (self.config.batch_size * self.config.max_seq_len * 
                       self.config.n_heads * self.config.head_dim) +
               batch * (self.config.max_seq_len * self.config.n_heads * self.config.head_dim) +
               pos * (self.config.n_heads * self.config.head_dim);
    }
    
    /// Store key-value pair in cache
    pub fn store(
        self: *KVCache,
        layer: u32,
        batch: u32,
        key: []const f32,
        value: []const f32,
    ) !void {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        if (self.current_pos >= self.config.max_seq_len) return error.CacheFull;
        
        const kv_size = self.config.n_heads * self.config.head_dim;
        if (key.len != kv_size or value.len != kv_size) return error.InvalidSize;
        
        const offset = self.getOffset(layer, batch, self.current_pos);
        
        // Copy key and value
        @memcpy(self.keys[offset..offset + kv_size], key);
        @memcpy(self.values[offset..offset + kv_size], value);
        
        // Update sequence length
        self.seq_lengths[batch] = self.current_pos + 1;
    }
    
    /// Retrieve keys from cache for a specific layer and batch
    pub fn getKeys(
        self: *const KVCache,
        layer: u32,
        batch: u32,
    ) ![]const f32 {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const seq_len = self.seq_lengths[batch];
        const offset = self.getOffset(layer, batch, 0);
        const size = seq_len * self.config.n_heads * self.config.head_dim;
        
        return self.keys[offset..offset + size];
    }
    
    /// Retrieve values from cache for a specific layer and batch
    pub fn getValues(
        self: *const KVCache,
        layer: u32,
        batch: u32,
    ) ![]const f32 {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const seq_len = self.seq_lengths[batch];
        const offset = self.getOffset(layer, batch, 0);
        const size = seq_len * self.config.n_heads * self.config.head_dim;
        
        return self.values[offset..offset + size];
    }
    
    /// Advance position (call after storing all layers for current token)
    pub fn advance(self: *KVCache) !void {
        if (self.current_pos >= self.config.max_seq_len - 1) {
            return error.CacheFull;
        }
        self.current_pos += 1;
    }
    
    /// Get current sequence length for a batch
    pub fn getSeqLen(self: *const KVCache, batch: u32) u32 {
        if (batch >= self.config.batch_size) return 0;
        return self.seq_lengths[batch];
    }
    
    /// Get memory usage in bytes
    pub fn getMemoryUsage(self: *const KVCache) usize {
        return self.keys.len * @sizeOf(f32) + 
               self.values.len * @sizeOf(f32) +
               self.seq_lengths.len * @sizeOf(u32);
    }
};

// ============================================================================
// Testing
// ============================================================================

pub fn test_kv_cache(allocator: std.mem.Allocator) !void {
    std.debug.print("\n🧪 Testing KV Cache Module\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n", .{});
    
    // Test 1: Basic cache operations
    {
        std.debug.print("\n1️⃣  Testing basic cache operations...\n", .{});
        
        const config = KVCacheConfig{
            .n_layers = 2,
            .n_heads = 4,
            .head_dim = 8,
            .max_seq_len = 10,
            .batch_size = 1,
        };
        
        var cache = try KVCache.init(allocator, config);
        defer cache.deinit();
        
        std.debug.print("   Cache created: {d} layers, {d} heads, {d} head_dim\n", 
            .{config.n_layers, config.n_heads, config.head_dim});
        std.debug.print("   Memory usage: {d} bytes\n", .{cache.getMemoryUsage()});
        
        // Store some K/V pairs
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        for (0..kv_size) |i| {
            key[i] = @as(f32, @floatFromInt(i));
            value[i] = @as(f32, @floatFromInt(i)) * 2.0;
        }
        
        try cache.store(0, 0, key, value);
        try cache.store(1, 0, key, value);
        try cache.advance();
        
        std.debug.print("   Stored K/V for 2 layers\n", .{});
        std.debug.print("   Current position: {d}\n", .{cache.current_pos});
        std.debug.print("   Sequence length: {d}\n", .{cache.getSeqLen(0)});
        std.debug.print("   ✅ Basic operations working\n", .{});
    }
    
    // Test 2: Retrieve cached values
    {
        std.debug.print("\n2️⃣  Testing cache retrieval...\n", .{});
        
        const config = KVCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 5,
            .batch_size = 1,
        };
        
        var cache = try KVCache.init(allocator, config);
        defer cache.deinit();
        
        // Store 3 tokens
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        for (0..3) |token| {
            for (0..kv_size) |i| {
                key[i] = @as(f32, @floatFromInt(token * 10 + i));
                value[i] = @as(f32, @floatFromInt(token * 100 + i));
            }
            try cache.store(0, 0, key, value);
            try cache.advance();
        }
        
        // Retrieve and verify
        const retrieved_keys = try cache.getKeys(0, 0);
        const retrieved_values = try cache.getValues(0, 0);
        
        std.debug.print("   Stored 3 tokens\n", .{});
        std.debug.print("   Retrieved keys length: {d}\n", .{retrieved_keys.len});
        std.debug.print("   Retrieved values length: {d}\n", .{retrieved_values.len});
        std.debug.print("   Expected length: {d}\n", .{3 * kv_size});
        
        if (retrieved_keys.len != 3 * kv_size or retrieved_values.len != 3 * kv_size) {
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Retrieval working\n", .{});
    }
    
    // Test 3: Cache full handling
    {
        std.debug.print("\n3️⃣  Testing cache full handling...\n", .{});
        
        const config = KVCacheConfig{
            .n_layers = 1,
            .n_heads = 1,
            .head_dim = 4,
            .max_seq_len = 3,
            .batch_size = 1,
        };
        
        var cache = try KVCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        @memset(key, 1.0);
        @memset(value, 2.0);
        
        // Fill cache
        try cache.store(0, 0, key, value);
        try cache.advance();
        try cache.store(0, 0, key, value);
        try cache.advance();
        try cache.store(0, 0, key, value);
        
        std.debug.print("   Filled cache to capacity\n", .{});
        
        // Try to advance beyond capacity
        const result = cache.advance();
        if (result) {
            std.debug.print("   ❌ Should have returned error\n", .{});
            return error.TestFailed;
        } else |err| {
            if (err == error.CacheFull) {
                std.debug.print("   ✅ Cache full error handled correctly\n", .{});
            } else {
                std.debug.print("   ❌ Wrong error type\n", .{});
                return error.TestFailed;
            }
        }
    }
    
    // Test 4: Reset functionality
    {
        std.debug.print("\n4️⃣  Testing cache reset...\n", .{});
        
        const config = KVCacheConfig{
            .n_layers = 1,
            .n_heads = 2,
            .head_dim = 4,
            .max_seq_len = 10,
            .batch_size = 1,
        };
        
        var cache = try KVCache.init(allocator, config);
        defer cache.deinit();
        
        const kv_size = config.n_heads * config.head_dim;
        const key = try allocator.alloc(f32, kv_size);
        defer allocator.free(key);
        const value = try allocator.alloc(f32, kv_size);
        defer allocator.free(value);
        
        @memset(key, 5.0);
        @memset(value, 10.0);
        
        // Add some data
        try cache.store(0, 0, key, value);
        try cache.advance();
        try cache.store(0, 0, key, value);
        try cache.advance();
        
        std.debug.print("   Added 2 tokens\n", .{});
        std.debug.print("   Position before reset: {d}\n", .{cache.current_pos});
        
        // Reset
        cache.reset();
        
        std.debug.print("   Position after reset: {d}\n", .{cache.current_pos});
        std.debug.print("   Sequence length after reset: {d}\n", .{cache.getSeqLen(0)});
        
        if (cache.current_pos != 0 or cache.getSeqLen(0) != 0) {
            std.debug.print("   ❌ Reset failed\n", .{});
            return error.TestFailed;
        }
        
        std.debug.print("   ✅ Reset working\n", .{});
    }
    
    // Test 5: Memory size calculation
    {
        std.debug.print("\n5️⃣  Testing memory size calculation...\n", .{});
        
        const config = KVCacheConfig{
            .n_layers = 12,
            .n_heads = 12,
            .head_dim = 64,
            .max_seq_len = 2048,
            .batch_size = 1,
        };
        
        const mem_size = config.memorySize();
        const mem_mb = @as(f64, @floatFromInt(mem_size)) / (1024.0 * 1024.0);
        
        std.debug.print("   Config: {d} layers, {d} heads, {d} dim\n", 
            .{config.n_layers, config.n_heads, config.head_dim});
        std.debug.print("   Max sequence: {d}\n", .{config.max_seq_len});
        std.debug.print("   Memory required: {d:.2} MB\n", .{mem_mb});
        std.debug.print("   ✅ Memory calculation working\n", .{});
    }
    
    std.debug.print("\n✅ All KV cache tests passed!\n", .{});
    std.debug.print("═══════════════════════════════════════════════════════════════════════\n\n", .{});
}
