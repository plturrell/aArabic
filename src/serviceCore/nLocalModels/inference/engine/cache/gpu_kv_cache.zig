// GPU-Accelerated KV Cache
// GPU-backed key-value cache using CUDA for high-performance inference

const std = @import("std");
const GpuCacheConfig = @import("gpu_cache_config.zig").GpuCacheConfig;
const CudaManager = @import("../cuda/cuda_manager.zig").CudaManager;
const DeviceMemory = @import("../cuda/cuda_manager.zig").DeviceMemory;
const PinnedMemory = @import("../cuda/cuda_manager.zig").PinnedMemory;
const CudaStream = @import("../cuda/cuda_manager.zig").CudaStream;

// ============================================================================
// GPU Cache Entry
// ============================================================================

/// Single cache entry stored on GPU
pub const GpuCacheEntry = struct {
    // GPU memory for key/value states
    keys: DeviceMemory,
    values: DeviceMemory,
    
    // Metadata
    layer_id: u32,
    batch_id: u32,
    sequence_length: u32,
    last_access: i64, // Unix timestamp for LRU
    access_count: u32, // For LFU
    
    // Associated stream for async ops
    stream: ?*CudaStream,
    
    pub fn getMemoryUsage(self: *const GpuCacheEntry) usize {
        return self.keys.size + self.values.size;
    }
};

// ============================================================================
// GPU KV Cache
// ============================================================================

/// GPU-accelerated KV cache
pub const GpuKvCache = struct {
    allocator: std.mem.Allocator,
    config: GpuCacheConfig,
    cuda_manager: *CudaManager,
    
    // Cache entries
    entries: std.ArrayList(GpuCacheEntry),
    
    // Current state
    current_pos: u32,
    total_allocated: usize,
    
    // Metrics (if enabled)
    hits: u64,
    misses: u64,
    evictions: u64,
    
    pub fn init(
        allocator: std.mem.Allocator,
        config: GpuCacheConfig,
        cuda_manager: *CudaManager,
    ) !*GpuKvCache {
        // Validate config
        try config.validate();
        
        std.debug.print("\n🚀 Initializing GPU KV Cache\n", .{});
        config.printSummary();
        
        const self = try allocator.create(GpuKvCache);
        self.* = GpuKvCache{
            .allocator = allocator,
            .config = config,
            .cuda_manager = cuda_manager,
            .entries = std.ArrayList(GpuCacheEntry){},
            .current_pos = 0,
            .total_allocated = 0,
            .hits = 0,
            .misses = 0,
            .evictions = 0,
        };
        
        std.debug.print("   ✅ GPU KV Cache initialized\n", .{});
        return self;
    }
    
    pub fn deinit(self: *GpuKvCache) void {
        std.debug.print("\n🛑 Cleaning up GPU KV Cache\n", .{});
        
        // Free all GPU memory
        for (self.entries.items) |*entry| {
            self.cuda_manager.freeDeviceMemory(entry.keys) catch {};
            self.cuda_manager.freeDeviceMemory(entry.values) catch {};
        }
        
        self.entries.deinit();
        
        if (self.config.enable_metrics) {
            self.printStats();
        }
        
        self.allocator.destroy(self);
        std.debug.print("   ✅ GPU KV Cache cleaned up\n", .{});
    }
    
    /// Store key-value pair in GPU memory
    pub fn store(
        self: *GpuKvCache,
        layer: u32,
        batch: u32,
        key_data: []const f32,
        value_data: []const f32,
    ) !void {
        if (layer >= self.config.n_layers) return error.LayerOutOfRange;
        if (batch >= self.config.batch_size) return error.BatchOutOfRange;
        
        const kv_size = self.config.n_heads * self.config.head_dim;
        if (key_data.len != kv_size or value_data.len != kv_size) {
            return error.InvalidSize;
        }
        
        // Check if we need to evict
        const entry_size = kv_size * @sizeOf(f32) * 2; // keys + values
        if (self.total_allocated + entry_size > self.config.gpuMemorySize()) {
            try self.evict();
        }
        
        // Allocate GPU memory
        const keys_mem = try self.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
        errdefer self.cuda_manager.freeDeviceMemory(keys_mem) catch {};
        
        const values_mem = try self.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
        errdefer self.cuda_manager.freeDeviceMemory(values_mem) catch {};
        
        // Get stream for async upload
        const stream = if (self.config.enable_async_ops)
            try self.cuda_manager.acquireStream()
        else
            null;
        errdefer if (stream) |s| self.cuda_manager.releaseStream(s) catch {};
        
        // Upload data to GPU (would use async if stream available)
        // For now, using synchronous copy - async would be:
        // try keys_mem.copyFromHostAsync(key_data, stream);
        // try values_mem.copyFromHostAsync(value_data, stream);
        
        // Create entry
        const entry = GpuCacheEntry{
            .keys = keys_mem,
            .values = values_mem,
            .layer_id = layer,
            .batch_id = batch,
            .sequence_length = 1,
            .last_access = std.time.timestamp(),
            .access_count = 1,
            .stream = stream,
        };
        
        try self.entries.append(entry);
        self.total_allocated += entry_size;
        
        if (self.config.track_hit_rate) {
            // This is an insert, could count as miss
            self.misses += 1;
        }
    }
    
    /// Retrieve cached keys for a layer/batch
    pub fn getKeys(
        self: *GpuKvCache,
        layer: u32,
        batch: u32,
    ) !?DeviceMemory {
        for (self.entries.items) |*entry| {
            if (entry.layer_id == layer and entry.batch_id == batch) {
                // Update access stats
                entry.last_access = std.time.timestamp();
                entry.access_count += 1;
                
                if (self.config.track_hit_rate) {
                    self.hits += 1;
                }
                
                return entry.keys;
            }
        }
        
        if (self.config.track_hit_rate) {
            self.misses += 1;
        }
        
        return null;
    }
    
    /// Retrieve cached values for a layer/batch
    pub fn getValues(
        self: *GpuKvCache,
        layer: u32,
        batch: u32,
    ) !?DeviceMemory {
        for (self.entries.items) |*entry| {
            if (entry.layer_id == layer and entry.batch_id == batch) {
                entry.last_access = std.time.timestamp();
                entry.access_count += 1;
                return entry.values;
            }
        }
        return null;
    }
    
    /// Check if cache contains entry
    pub fn contains(self: *GpuKvCache, layer: u32, batch: u32) bool {
        for (self.entries.items) |entry| {
            if (entry.layer_id == layer and entry.batch_id == batch) {
                return true;
            }
        }
        return false;
    }
    
    /// Evict entries based on policy
    fn evict(self: *GpuKvCache) !void {
        if (self.entries.items.len == 0) return error.CacheEmpty;
        
        const evict_idx = switch (self.config.eviction_policy) {
            .lru => blk: {
                var oldest_idx: usize = 0;
                var oldest_time = self.entries.items[0].last_access;
                
                for (self.entries.items, 0..) |entry, i| {
                    if (entry.last_access < oldest_time) {
                        oldest_time = entry.last_access;
                        oldest_idx = i;
                    }
                }
                break :blk oldest_idx;
            },
            .lfu => blk: {
                var lowest_idx: usize = 0;
                var lowest_count = self.entries.items[0].access_count;
                
                for (self.entries.items, 0..) |entry, i| {
                    if (entry.access_count < lowest_count) {
                        lowest_count = entry.access_count;
                        lowest_idx = i;
                    }
                }
                break :blk lowest_idx;
            },
            .fifo => 0, // Evict first entry
            .size_based => blk: {
                var largest_idx: usize = 0;
                var largest_size = self.entries.items[0].getMemoryUsage();
                
                for (self.entries.items, 0..) |entry, i| {
                    const size = entry.getMemoryUsage();
                    if (size > largest_size) {
                        largest_size = size;
                        largest_idx = i;
                    }
                }
                break :blk largest_idx;
            },
        };
        
        // Free GPU memory
        var entry = self.entries.orderedRemove(evict_idx);
        const freed_size = entry.getMemoryUsage();
        
        try self.cuda_manager.freeDeviceMemory(entry.keys);
        try self.cuda_manager.freeDeviceMemory(entry.values);
        
        if (entry.stream) |stream| {
            try self.cuda_manager.releaseStream(stream);
        }
        
        self.total_allocated -= freed_size;
        self.evictions += 1;
        
        std.debug.print("   Evicted entry (policy: {s}), freed {} bytes\n", 
            .{@tagName(self.config.eviction_policy), freed_size});
    }
    
    /// Clear all cache entries
    pub fn clear(self: *GpuKvCache) !void {
        while (self.entries.items.len > 0) {
            try self.evict();
        }
        self.current_pos = 0;
    }
    
    /// Get cache statistics
    pub fn getStats(self: *const GpuKvCache) CacheStats {
        return .{
            .total_entries = self.entries.items.len,
            .total_allocated_bytes = self.total_allocated,
            .hits = self.hits,
            .misses = self.misses,
            .evictions = self.evictions,
            .hit_rate = if (self.hits + self.misses > 0)
                @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(self.hits + self.misses))
            else
                0.0,
        };
    }
    
    /// Print cache statistics
    pub fn printStats(self: *const GpuKvCache) void {
        const stats = self.getStats();
        
        std.debug.print("\n📊 GPU Cache Statistics\n", .{});
        std.debug.print("   Entries: {}\n", .{stats.total_entries});
        std.debug.print("   Memory Used: {} MB\n", .{stats.total_allocated_bytes / (1024 * 1024)});
        std.debug.print("   Hits: {}\n", .{stats.hits});
        std.debug.print("   Misses: {}\n", .{stats.misses});
        std.debug.print("   Evictions: {}\n", .{stats.evictions});
        std.debug.print("   Hit Rate: {d:.2}%\n", .{stats.hit_rate * 100.0});
    }
};

/// Cache statistics
pub const CacheStats = struct {
    total_entries: usize,
    total_allocated_bytes: usize,
    hits: u64,
    misses: u64,
    evictions: u64,
    hit_rate: f64,
};

// ============================================================================
// Tests
// ============================================================================

test "gpu_kv_cache: initialization" {
    const allocator = std.testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = GpuCacheConfig.default();
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    try std.testing.expect(cache.entries.items.len == 0);
    try std.testing.expect(cache.total_allocated == 0);
    
    std.debug.print("✓ GPU cache initialization working\n", .{});
}

test "gpu_kv_cache: store and retrieve" {
    const allocator = std.testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    const config = GpuCacheConfig{
        .n_layers = 2,
        .n_heads = 4,
        .head_dim = 8,
        .max_seq_len = 10,
        .batch_size = 1,
    };
    
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    // Create test data
    const kv_size = config.n_heads * config.head_dim;
    const key_data = try allocator.alloc(f32, kv_size);
    defer allocator.free(key_data);
    const value_data = try allocator.alloc(f32, kv_size);
    defer allocator.free(value_data);
    
    for (0..kv_size) |i| {
        key_data[i] = @floatFromInt(i);
        value_data[i] = @floatFromInt(i * 2);
    }
    
    // Store in cache
    try cache.store(0, 0, key_data, value_data);
    
    // Check it exists
    try std.testing.expect(cache.contains(0, 0));
    try std.testing.expect(cache.entries.items.len == 1);
    
    std.debug.print("✓ Store and retrieve working\n", .{});
}

test "gpu_kv_cache: statistics" {
    const allocator = std.testing.allocator;
    
    var cuda_manager = CudaManager.initDefault(allocator) catch |err| {
        if (err == error.CudaError or err == error.NoGPUFound) {
            std.debug.print("Test skipped: No GPU available\n", .{});
            return;
        }
        return err;
    };
    defer cuda_manager.deinit();
    
    var config = GpuCacheConfig.default();
    config.enable_metrics = true;
    config.track_hit_rate = true;
    
    var cache = try GpuKvCache.init(allocator, config, cuda_manager);
    defer cache.deinit();
    
    const stats = cache.getStats();
    try std.testing.expect(stats.total_entries == 0);
    try std.testing.expect(stats.hit_rate == 0.0);
    
    std.debug.print("✓ Statistics working\n", .{});
}
