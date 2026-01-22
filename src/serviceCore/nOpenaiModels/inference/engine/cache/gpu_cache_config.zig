// GPU Cache Configuration
// Configuration options for GPU-accelerated KV cache

const std = @import("std");

/// GPU cache configuration for optimal T4 performance
pub const GpuCacheConfig = struct {
    // Model dimensions
    n_layers: u32,
    n_heads: u32,
    head_dim: u32,
    max_seq_len: u32,
    batch_size: u32 = 1,
    
    // GPU-specific settings
    use_pinned_memory: bool = true,       // Use pinned host memory for faster transfers
    enable_async_ops: bool = true,        // Enable async GPU operations
    stream_pool_size: usize = 8,          // Number of CUDA streams (T4 optimal)
    memory_pool_block_size: usize = 4 * 1024 * 1024, // 4MB blocks
    
    // Cache management
    max_gpu_memory_mb: usize = 14 * 1024, // Reserve 14GB for cache (leave 2GB for inference)
    eviction_policy: EvictionPolicy = .lru,
    enable_compression: bool = false,      // Future: compress older cache entries
    
    // Performance tuning
    memory_alignment: usize = 128,         // 128-byte alignment for coalescing
    prefetch_distance: u32 = 2,            // Prefetch N tokens ahead
    
    // Monitoring
    enable_metrics: bool = true,
    track_hit_rate: bool = true,
    
    /// Calculate total GPU memory required
    pub fn gpuMemorySize(self: GpuCacheConfig) usize {
        // Keys + Values: 2 * n_layers * batch_size * max_seq_len * n_heads * head_dim * sizeof(f32)
        // Use u64 to avoid overflow for large models
        const per_kv: u64 = @as(u64, self.n_layers) * @as(u64, self.batch_size) * 
                            @as(u64, self.max_seq_len) * @as(u64, self.n_heads) * 
                            @as(u64, self.head_dim);
        return @intCast(per_kv * 2 * @sizeOf(f32));
    }
    
    /// Calculate host memory required (for pinned staging)
    pub fn hostMemorySize(self: GpuCacheConfig) usize {
        if (!self.use_pinned_memory) return 0;
        // Allocate pinned memory for double-buffering
        return self.gpuMemorySize() / 10; // 10% for staging
    }
    
    /// Get cache capacity in tokens
    pub fn getCacheCapacity(self: GpuCacheConfig) usize {
        return self.max_seq_len * self.batch_size;
    }
    
    /// Check if config fits in available GPU memory
    pub fn fitsInGpuMemory(self: GpuCacheConfig) bool {
        const required_mb = self.gpuMemorySize() / (1024 * 1024);
        return required_mb <= self.max_gpu_memory_mb;
    }
    
    /// Get recommended config for T4 GPU
    pub fn forT4(n_layers: u32, n_heads: u32, head_dim: u32) GpuCacheConfig {
        return .{
            .n_layers = n_layers,
            .n_heads = n_heads,
            .head_dim = head_dim,
            .max_seq_len = 2048,
            .batch_size = 1,
            .use_pinned_memory = true,
            .enable_async_ops = true,
            .stream_pool_size = 8,
            .memory_pool_block_size = 4 * 1024 * 1024,
            .max_gpu_memory_mb = 14 * 1024,
            .eviction_policy = .lru,
            .memory_alignment = 128,
            .prefetch_distance = 2,
            .enable_metrics = true,
            .track_hit_rate = true,
        };
    }
    
    /// Get config for 7B model on T4
    pub fn for7B() GpuCacheConfig {
        return forT4(32, 32, 128); // LLaMA-like 7B
    }
    
    /// Get config for 13B model on T4
    pub fn for13B() GpuCacheConfig {
        return forT4(40, 40, 128); // LLaMA-like 13B
    }
    
    /// Get config for 70B model on T4 (aggressive tiering)
    pub fn for70B() GpuCacheConfig {
        var config = forT4(80, 64, 128); // LLaMA-like 70B
        config.max_seq_len = 1024; // Reduced for 70B
        config.max_gpu_memory_mb = 12 * 1024; // More conservative
        return config;
    }
    
    /// Default config for testing
    pub fn default() GpuCacheConfig {
        return .{
            .n_layers = 12,
            .n_heads = 12,
            .head_dim = 64,
            .max_seq_len = 2048,
            .batch_size = 1,
        };
    }
    
    /// Validate configuration
    pub fn validate(self: GpuCacheConfig) !void {
        if (self.n_layers == 0) return error.InvalidLayers;
        if (self.n_heads == 0) return error.InvalidHeads;
        if (self.head_dim == 0) return error.InvalidHeadDim;
        if (self.max_seq_len == 0) return error.InvalidSeqLen;
        if (self.batch_size == 0) return error.InvalidBatchSize;
        if (self.stream_pool_size == 0) return error.InvalidStreamPoolSize;
        
        if (!self.fitsInGpuMemory()) return error.InsufficientGpuMemory;
    }
    
    /// Print config summary
    pub fn printSummary(self: GpuCacheConfig) void {
        std.debug.print("\n📋 GPU Cache Configuration\n", .{});
        std.debug.print("   Model Dimensions:\n", .{});
        std.debug.print("     Layers: {d}\n", .{self.n_layers});
        std.debug.print("     Heads: {d}\n", .{self.n_heads});
        std.debug.print("     Head Dim: {d}\n", .{self.head_dim});
        std.debug.print("     Max Seq Length: {d}\n", .{self.max_seq_len});
        std.debug.print("     Batch Size: {d}\n", .{self.batch_size});
        
        std.debug.print("   GPU Settings:\n", .{});
        std.debug.print("     Pinned Memory: {s}\n", .{if (self.use_pinned_memory) "enabled" else "disabled"});
        std.debug.print("     Async Ops: {s}\n", .{if (self.enable_async_ops) "enabled" else "disabled"});
        std.debug.print("     Stream Pool Size: {d}\n", .{self.stream_pool_size});
        std.debug.print("     Memory Alignment: {d} bytes\n", .{self.memory_alignment});
        
        const gpu_mem_mb = self.gpuMemorySize() / (1024 * 1024);
        const host_mem_mb = self.hostMemorySize() / (1024 * 1024);
        std.debug.print("   Memory:\n", .{});
        std.debug.print("     GPU Required: {d} MB\n", .{gpu_mem_mb});
        std.debug.print("     GPU Budget: {d} MB\n", .{self.max_gpu_memory_mb});
        std.debug.print("     Host Staging: {d} MB\n", .{host_mem_mb});
        std.debug.print("     Fits in GPU: {s}\n", .{if (self.fitsInGpuMemory()) "✓" else "✗"});
        
        std.debug.print("   Cache:\n", .{});
        std.debug.print("     Capacity: {d} tokens\n", .{self.getCacheCapacity()});
        std.debug.print("     Eviction: {s}\n", .{@tagName(self.eviction_policy)});
        std.debug.print("     Prefetch Distance: {d}\n", .{self.prefetch_distance});
        
        std.debug.print("   Monitoring:\n", .{});
        std.debug.print("     Metrics: {s}\n", .{if (self.enable_metrics) "enabled" else "disabled"});
        std.debug.print("     Hit Rate Tracking: {s}\n", .{if (self.track_hit_rate) "enabled" else "disabled"});
    }
};

/// Cache eviction policies
pub const EvictionPolicy = enum {
    lru,          // Least Recently Used
    lfu,          // Least Frequently Used
    fifo,         // First In First Out
    size_based,   // Evict largest entries first
    
    pub fn describe(self: EvictionPolicy) []const u8 {
        return switch (self) {
            .lru => "Least Recently Used - evict oldest accessed",
            .lfu => "Least Frequently Used - evict least accessed",
            .fifo => "First In First Out - evict oldest created",
            .size_based => "Size Based - evict largest entries first",
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "gpu_cache_config: memory calculations" {
    const config = GpuCacheConfig.default();
    
    const gpu_mem = config.gpuMemorySize();
    const host_mem = config.hostMemorySize();
    
    try std.testing.expect(gpu_mem > 0);
    try std.testing.expect(host_mem >= 0);
    
    std.debug.print("✓ Memory calculations working\n", .{});
}

test "gpu_cache_config: T4 presets" {
    const config_7b = GpuCacheConfig.for7B();
    const config_13b = GpuCacheConfig.for13B();
    const config_70b = GpuCacheConfig.for70B();
    
    try config_7b.validate();
    try config_13b.validate();
    try config_70b.validate();
    
    try std.testing.expect(config_7b.fitsInGpuMemory());
    try std.testing.expect(config_13b.fitsInGpuMemory());
    try std.testing.expect(config_70b.fitsInGpuMemory());
    
    std.debug.print("✓ T4 presets validated\n", .{});
}

test "gpu_cache_config: validation" {
    var config = GpuCacheConfig.default();
    
    // Valid config should pass
    try config.validate();
    
    // Invalid configs should fail
    config.n_layers = 0;
    try std.testing.expectError(error.InvalidLayers, config.validate());
    
    config = GpuCacheConfig.default();
    config.n_heads = 0;
    try std.testing.expectError(error.InvalidHeads, config.validate());
    
    std.debug.print("✓ Validation working\n", .{});
}

test "gpu_cache_config: capacity calculation" {
    const config = GpuCacheConfig{
        .n_layers = 12,
        .n_heads = 12,
        .head_dim = 64,
        .max_seq_len = 2048,
        .batch_size = 4,
    };
    
    const capacity = config.getCacheCapacity();
    try std.testing.expect(capacity == 2048 * 4);
    
    std.debug.print("✓ Capacity calculation working\n", .{});
}

test "gpu_cache_config: eviction policies" {
    const policies = [_]EvictionPolicy{ .lru, .lfu, .fifo, .size_based };
    
    for (policies) |policy| {
        const desc = policy.describe();
        try std.testing.expect(desc.len > 0);
    }
    
    std.debug.print("✓ Eviction policies defined\n", .{});
}
