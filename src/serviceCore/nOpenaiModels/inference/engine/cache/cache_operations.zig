// Cache Operations Implementation
// Core CRUD operations for GPU-accelerated KV cache with async support

const std = @import("std");
const GpuKvCache = @import("gpu_kv_cache.zig").GpuKvCache;
const GpuCacheEntry = @import("gpu_kv_cache.zig").GpuCacheEntry;
const GpuCacheConfig = @import("gpu_cache_config.zig").GpuCacheConfig;
const CudaManager = @import("../cuda/cuda_manager.zig").CudaManager;
const DeviceMemory = @import("../cuda/cuda_manager.zig").DeviceMemory;
const CudaStream = @import("../cuda/cuda_manager.zig").CudaStream;

// ============================================================================
// Cache Key
// ============================================================================

/// Unique identifier for cache entries
pub const CacheKey = struct {
    layer_id: u32,
    batch_id: u32,
    sequence_id: u64, // For multi-sequence batches
    
    pub fn init(layer: u32, batch: u32, sequence: u64) CacheKey {
        return .{
            .layer_id = layer,
            .batch_id = batch,
            .sequence_id = sequence,
        };
    }
    
    pub fn hash(self: CacheKey) u64 {
        var h: u64 = 0;
        h ^= @as(u64, self.layer_id);
        h ^= @as(u64, self.batch_id) << 16;
        h ^= self.sequence_id << 32;
        return h;
    }
    
    pub fn eql(self: CacheKey, other: CacheKey) bool {
        return self.layer_id == other.layer_id and
               self.batch_id == other.batch_id and
               self.sequence_id == other.sequence_id;
    }
};

// ============================================================================
// Cache Operations
// ============================================================================

/// High-level cache operations with async support
pub const CacheOperations = struct {
    cache: *GpuKvCache,
    allocator: std.mem.Allocator,
    
    // Operation statistics
    insert_count: u64,
    lookup_count: u64,
    update_count: u64,
    delete_count: u64,
    
    pub fn init(allocator: std.mem.Allocator, cache: *GpuKvCache) !*CacheOperations {
        const self = try allocator.create(CacheOperations);
        self.* = CacheOperations{
            .cache = cache,
            .allocator = allocator,
            .insert_count = 0,
            .lookup_count = 0,
            .update_count = 0,
            .delete_count = 0,
        };
        return self;
    }
    
    pub fn deinit(self: *CacheOperations) void {
        self.allocator.destroy(self);
    }
    
    // ========================================================================
    // Insert Operation
    // ========================================================================
    
    /// Insert new cache entry (synchronous)
    pub fn insert(
        self: *CacheOperations,
        key: CacheKey,
        key_data: []const f32,
        value_data: []const f32,
    ) !void {
        try self.cache.store(key.layer_id, key.batch_id, key_data, value_data);
        self.insert_count += 1;
    }
    
    /// Insert new cache entry (asynchronous with stream)
    pub fn insertAsync(
        self: *CacheOperations,
        key: CacheKey,
        key_data: []const f32,
        value_data: []const f32,
        stream: *CudaStream,
    ) !void {
        // Allocate GPU memory
        const kv_size = self.cache.config.n_heads * self.cache.config.head_dim;
        if (key_data.len != kv_size or value_data.len != kv_size) {
            return error.InvalidSize;
        }
        
        const keys_mem = try self.cache.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
        errdefer self.cache.cuda_manager.freeDeviceMemory(keys_mem) catch {};
        
        const values_mem = try self.cache.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
        errdefer self.cache.cuda_manager.freeDeviceMemory(values_mem) catch {};
        
        // Async copy to GPU (would use cudaMemcpyAsync in real implementation)
        // For now, this is a placeholder for the async interface
        
        // Create entry with stream
        const entry = GpuCacheEntry{
            .keys = keys_mem,
            .values = values_mem,
            .layer_id = key.layer_id,
            .batch_id = key.batch_id,
            .sequence_length = 1,
            .last_access = std.time.timestamp(),
            .access_count = 1,
            .stream = stream,
        };
        
        try self.cache.entries.append(entry);
        self.cache.total_allocated += kv_size * @sizeOf(f32) * 2;
        self.insert_count += 1;
    }
    
    /// Batch insert multiple entries
    pub fn insertBatch(
        self: *CacheOperations,
        keys: []const CacheKey,
        key_data: []const []const f32,
        value_data: []const []const f32,
    ) !void {
        if (keys.len != key_data.len or keys.len != value_data.len) {
            return error.MismatchedBatchSizes;
        }
        
        for (keys, 0..) |key, i| {
            try self.insert(key, key_data[i], value_data[i]);
        }
    }
    
    // ========================================================================
    // Lookup Operation
    // ========================================================================
    
    /// Lookup cache entry by key
    pub fn lookup(
        self: *CacheOperations,
        key: CacheKey,
    ) !?CacheLookupResult {
        self.lookup_count += 1;
        
        for (self.cache.entries.items) |*entry| {
            if (entry.layer_id == key.layer_id and entry.batch_id == key.batch_id) {
                // Update access statistics
                entry.last_access = std.time.timestamp();
                entry.access_count += 1;
                
                return CacheLookupResult{
                    .keys = entry.keys,
                    .values = entry.values,
                    .sequence_length = entry.sequence_length,
                    .found = true,
                };
            }
        }
        
        return CacheLookupResult{
            .keys = undefined,
            .values = undefined,
            .sequence_length = 0,
            .found = false,
        };
    }
    
    /// Lookup multiple entries in batch
    pub fn lookupBatch(
        self: *CacheOperations,
        keys: []const CacheKey,
        allocator: std.mem.Allocator,
    ) ![]CacheLookupResult {
        const results = try allocator.alloc(CacheLookupResult, keys.len);
        
        for (keys, 0..) |key, i| {
            results[i] = (try self.lookup(key)) orelse CacheLookupResult{
                .keys = undefined,
                .values = undefined,
                .sequence_length = 0,
                .found = false,
            };
        }
        
        return results;
    }
    
    // ========================================================================
    // Update Operation
    // ========================================================================
    
    /// Update existing cache entry
    pub fn update(
        self: *CacheOperations,
        key: CacheKey,
        key_data: []const f32,
        value_data: []const f32,
    ) !void {
        self.update_count += 1;
        
        // Find existing entry
        for (self.cache.entries.items) |*entry| {
            if (entry.layer_id == key.layer_id and entry.batch_id == key.batch_id) {
                // Free old memory
                try self.cache.cuda_manager.freeDeviceMemory(entry.keys);
                try self.cache.cuda_manager.freeDeviceMemory(entry.values);
                
                // Allocate new memory
                const kv_size = self.cache.config.n_heads * self.cache.config.head_dim;
                entry.keys = try self.cache.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
                entry.values = try self.cache.cuda_manager.allocDeviceMemory(kv_size * @sizeOf(f32));
                
                // Update metadata
                entry.last_access = std.time.timestamp();
                entry.access_count += 1;
                
                return;
            }
        }
        
        // If not found, insert as new
        try self.insert(key, key_data, value_data);
    }
    
    /// Update or insert (upsert) operation
    pub fn upsert(
        self: *CacheOperations,
        key: CacheKey,
        key_data: []const f32,
        value_data: []const f32,
    ) !void {
        if (self.cache.contains(key.layer_id, key.batch_id)) {
            try self.update(key, key_data, value_data);
        } else {
            try self.insert(key, key_data, value_data);
        }
    }
    
    // ========================================================================
    // Delete Operation
    // ========================================================================
    
    /// Delete cache entry by key
    pub fn delete(
        self: *CacheOperations,
        key: CacheKey,
    ) !bool {
        self.delete_count += 1;
        
        var i: usize = 0;
        while (i < self.cache.entries.items.len) : (i += 1) {
            const entry = &self.cache.entries.items[i];
            if (entry.layer_id == key.layer_id and entry.batch_id == key.batch_id) {
                // Free GPU memory
                const freed_size = entry.getMemoryUsage();
                try self.cache.cuda_manager.freeDeviceMemory(entry.keys);
                try self.cache.cuda_manager.freeDeviceMemory(entry.values);
                
                if (entry.stream) |stream| {
                    try self.cache.cuda_manager.releaseStream(stream);
                }
                
                // Remove from list
                _ = self.cache.entries.orderedRemove(i);
                self.cache.total_allocated -= freed_size;
                
                return true;
            }
        }
        
        return false;
    }
    
    /// Delete multiple entries in batch
    pub fn deleteBatch(
        self: *CacheOperations,
        keys: []const CacheKey,
    ) !usize {
        var deleted: usize = 0;
        for (keys) |key| {
            if (try self.delete(key)) {
                deleted += 1;
            }
        }
        return deleted;
    }
    
    // ========================================================================
    // Prefetch Operation
    // ========================================================================
    
    /// Prefetch data for upcoming operations
    pub fn prefetch(
        self: *CacheOperations,
        keys: []const CacheKey,
    ) !void {
        // Mark entries as likely to be accessed soon
        for (keys) |key| {
            for (self.cache.entries.items) |*entry| {
                if (entry.layer_id == key.layer_id and entry.batch_id == key.batch_id) {
                    // Boost priority by updating access time
                    entry.last_access = std.time.timestamp();
                    break;
                }
            }
        }
    }
    
    // ========================================================================
    // Statistics
    // ========================================================================
    
    /// Get operation statistics
    pub fn getOperationStats(self: *const CacheOperations) OperationStats {
        return .{
            .inserts = self.insert_count,
            .lookups = self.lookup_count,
            .updates = self.update_count,
            .deletes = self.delete_count,
            .total_ops = self.insert_count + self.lookup_count + 
                        self.update_count + self.delete_count,
        };
    }
    
    /// Print operation statistics
    pub fn printOperationStats(self: *const CacheOperations) void {
        const stats = self.getOperationStats();
        
        std.debug.print("\n📊 Cache Operation Statistics\n", .{});
        std.debug.print("   Inserts: {}\n", .{stats.inserts});
        std.debug.print("   Lookups: {}\n", .{stats.lookups});
        std.debug.print("   Updates: {}\n", .{stats.updates});
        std.debug.print("   Deletes: {}\n", .{stats.deletes});
        std.debug.print("   Total Operations: {}\n", .{stats.total_ops});
    }
};

// ============================================================================
// Supporting Types
// ============================================================================

/// Result of a cache lookup operation
pub const CacheLookupResult = struct {
    keys: DeviceMemory,
    values: DeviceMemory,
    sequence_length: u32,
    found: bool,
};

/// Operation statistics
pub const OperationStats = struct {
    inserts: u64,
    lookups: u64,
    updates: u64,
    deletes: u64,
    total_ops: u64,
};

// ============================================================================
// Tests
// ============================================================================

test "cache_operations: initialization" {
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
    
    var ops = try CacheOperations.init(allocator, cache);
    defer ops.deinit();
    
    try std.testing.expect(ops.insert_count == 0);
    try std.testing.expect(ops.lookup_count == 0);
    
    std.debug.print("✓ Cache operations initialization working\n", .{});
}

test "cache_operations: cache key" {
    const key1 = CacheKey.init(0, 0, 0);
    const key2 = CacheKey.init(0, 0, 0);
    const key3 = CacheKey.init(1, 0, 0);
    
    try std.testing.expect(key1.eql(key2));
    try std.testing.expect(!key1.eql(key3));
    try std.testing.expect(key1.hash() == key2.hash());
    
    std.debug.print("✓ Cache key working\n", .{});
}

test "cache_operations: operation stats" {
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
    
    var ops = try CacheOperations.init(allocator, cache);
    defer ops.deinit();
    
    const stats = ops.getOperationStats();
    try std.testing.expect(stats.total_ops == 0);
    
    std.debug.print("✓ Operation statistics working\n", .{});
}
