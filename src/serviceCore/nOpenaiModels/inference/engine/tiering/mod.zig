// Tiering Module
// SSD-backed storage for breaking memory/GPU constraints
//
// This module provides:
// - SSD-backed KV cache for 100K+ context on limited RAM
// - Memory-mapped GGUF loading for zero-copy model weights
// - Tiered tensor storage (hot/warm/cold)
// - Distributed caching via DragonflyDB
// - Unified API for all tiering operations
//
// Architecture inspired by DragonflyDB's tiering system

pub const ssd_tier = @import("ssd_tier.zig");
pub const tiered_kv_cache = @import("tiered_kv_cache.zig");
pub const mmap_gguf = @import("mmap_gguf.zig");
pub const tiered_tensors = @import("tiered_tensors.zig");
pub const distributed_tier = @import("distributed_tier.zig");
pub const unified_tier = @import("unified_tier.zig");

// Advanced features
pub const compression = @import("compression.zig");
pub const encryption = @import("encryption.zig");
pub const async_io = @import("async_io.zig");

// Re-export main types
pub const SSDStorage = ssd_tier.SSDStorage;
pub const TierConfig = ssd_tier.TierConfig;
pub const TieredKVCache = tiered_kv_cache.TieredKVCache;
pub const TieredKVConfig = tiered_kv_cache.TieredKVConfig;
pub const MmapGGUF = mmap_gguf.MmapGGUF;
pub const TieredTensorManager = tiered_tensors.TieredTensorManager;
pub const TieredTensorConfig = tiered_tensors.TieredTensorConfig;
pub const DistributedKVTier = distributed_tier.DistributedKVTier;
pub const DistributedConfig = distributed_tier.DistributedConfig;
pub const UnifiedTierManager = unified_tier.UnifiedTierManager;
pub const UnifiedTierConfig = unified_tier.UnifiedTierConfig;

// Advanced feature types
pub const KVCompressor = compression.KVCompressor;
pub const CompressionConfig = compression.CompressionConfig;
pub const TierEncryptor = encryption.TierEncryptor;
pub const EncryptionConfig = encryption.EncryptionConfig;
pub const AsyncIOEngine = async_io.AsyncIOEngine;
pub const AsyncIOConfig = async_io.AsyncIOConfig;

// ============================================================================
// Quick Start Examples
// ============================================================================
//
// 1. Basic SSD-backed KV cache:
//
//    const kv = try TieredKVCache.init(allocator, .{
//        .n_layers = 32,
//        .n_heads = 32,
//        .head_dim = 128,
//        .max_seq_len = 100000,  // 100K context!
//        .hot_tokens = 2048,     // Keep last 2K in RAM
//        .max_ssd_mb = 16384,    // 16GB on SSD
//    });
//    defer kv.deinit();
//
//    // Store KV (automatically tiers to SSD when RAM full)
//    try kv.store(layer, keys, values);
//    kv.advance();
//
//    // Get KV (transparently loads from SSD if needed)
//    try kv.getKeys(layer, 0, seq_pos, dest);
//
// 2. Memory-mapped GGUF (zero-copy model loading):
//
//    const gguf = try MmapGGUF.open(allocator, "model.gguf");
//    defer gguf.close();
//
//    // Get tensor data (zero-copy slice into mmap)
//    const weights = try gguf.getTensorData("blk.0.attn_q.weight");
//
//    // Prefetch next layer
//    gguf.prefetchPrefix("blk.1.");
//
// 3. Unified tiering (recommended):
//
//    const tier = try UnifiedTierManager.init(allocator, .{
//        .model_path = "model.gguf",
//        .max_ram_mb = 4096,
//        .max_ssd_mb = 32768,
//        .enable_distributed = true,
//        .dragonfly_host = "localhost",
//    });
//    defer tier.deinit();
//
//    // Everything is handled automatically
//    try tier.storeKV(layer, keys, values);
//    tier.advanceKV();
//    const weights = try tier.getTensor("blk.0.attn_q.weight");
//
// ============================================================================
// Performance Characteristics
// ============================================================================
//
// | Operation          | Hot (RAM)  | Cold (SSD) | Distributed |
// |--------------------|------------|------------|-------------|
// | KV store           | ~100ns     | ~10µs      | ~100µs      |
// | KV load            | ~50ns      | ~5µs       | ~50µs       |
// | Tensor load        | ~10ns      | ~1µs       | N/A         |
// | Throughput         | 10M ops/s  | 500K ops/s | 100K ops/s  |
//
// SSD performance assumes NVMe with ~500K IOPS, 3GB/s sequential.
// Distributed assumes local DragonflyDB with <1ms RTT.
//
// ============================================================================
// Memory Savings
// ============================================================================
//
// Example: LLaMA 70B with 100K context
//
// Without tiering:
//   - Model weights: 140GB (FP16)
//   - KV cache: 100K * 80 layers * 2 * 8192 * 4 bytes = 524GB
//   - Total: 664GB RAM required
//
// With tiering:
//   - Hot tensors: 2GB (embeddings, output, first/last layers)
//   - Warm tensors: 4GB (recently used layers)
//   - Hot KV cache: 2GB (last 2K tokens)
//   - Total RAM: 8GB
//   - SSD: 656GB (model + cold KV cache)
//
// This enables running 70B models on consumer hardware!
//

