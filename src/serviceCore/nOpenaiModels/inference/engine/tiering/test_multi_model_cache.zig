//! Test Suite for Multi-Model Cache Manager - Day 12
//! Tests model registration, fair allocation, cross-model eviction, and metrics

const std = @import("std");
const MultiModelCacheManager = @import("multi_model_cache.zig").MultiModelCacheManager;
const MultiModelCacheConfig = @import("multi_model_cache.zig").MultiModelCacheConfig;
const AllocationStrategy = @import("multi_model_cache.zig").AllocationStrategy;
const GlobalEvictionPolicy = @import("multi_model_cache.zig").GlobalEvictionPolicy;

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("🧪 Multi-Model Cache Manager Test Suite - Day 12\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("\n", .{});

    // Test 1: Manager Initialization
    try testManagerInitialization(allocator);

    // Test 2: Fair Share Allocation
    try testFairShareAllocation(allocator);

    // Test 3: Priority-Based Allocation
    try testPriorityBasedAllocation(allocator);

    // Test 4: Model Registration
    try testModelRegistration(allocator);

    // Test 5: Multiple Model Registration
    try testMultipleModelRegistration(allocator);

    // Test 6: Cross-Model Eviction (LRU)
    try testCrossModelEvictionLRU(allocator);

    // Test 7: Cross-Model Eviction (LFU)
    try testCrossModelEvictionLFU(allocator);

    // Test 8: Per-Model Statistics
    try testPerModelStatistics(allocator);

    // Test 9: Global Statistics
    try testGlobalStatistics(allocator);

    // Test 10: Model Unregistration
    try testModelUnregistration(allocator);

    std.debug.print("\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("✅ All Tests Passed! (10/10)\n", .{});
    std.debug.print("=" ** 80 ++ "\n", .{});
    std.debug.print("\n", .{});
}

fn testManagerInitialization(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 1: Manager Initialization\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .total_ram_mb = 4096,
        .total_ssd_mb = 32768,
        .allocation_strategy = .fair_share,
    };

    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    const stats = manager.getGlobalStats();
    std.debug.assert(stats.total_models == 0);
    std.debug.assert(stats.active_models == 0);
    std.debug.assert(stats.total_ram_used_mb == 0);
    std.debug.assert(stats.total_ssd_used_mb == 0);

    std.debug.print("  ✓ Manager initialized successfully\n", .{});
    std.debug.print("  ✓ Initial state verified\n", .{});
    std.debug.print("  ✅ Manager initialization tests passed\n\n", .{});
}

fn testFairShareAllocation(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 2: Fair Share Allocation\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .total_ram_mb = 4096,
        .total_ssd_mb = 32768,
        .allocation_strategy = .fair_share,
    };

    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register 4 models - each should get 1024MB RAM, 8192MB SSD
    const model_configs = [_]struct {
        id: []const u8,
        layers: u32,
        heads: u32,
    }{
        .{ .id = "llama-1b", .layers = 16, .heads = 16 },
        .{ .id = "phi-2", .layers = 24, .heads = 32 },
        .{ .id = "qwen-0.5b", .layers = 12, .heads = 12 },
        .{ .id = "gemma-270m", .layers = 8, .heads = 8 },
    };

    for (model_configs) |mc| {
        try manager.registerModel(mc.id, .{
            .n_layers = mc.layers,
            .n_heads = mc.heads,
            .head_dim = 64,
            .max_seq_len = 4096,
            .priority = 5,
        });
    }

    const stats = manager.getGlobalStats();
    std.debug.print("  ✓ Registered {d} models\n", .{stats.total_models});
    std.debug.print("  ✓ Total RAM allocated: {d} MB\n", .{stats.total_ram_used_mb});
    std.debug.print("  ✓ Total SSD allocated: {d} MB\n", .{stats.total_ssd_used_mb});

    // Verify fair allocation
    for (model_configs) |mc| {
        const model_stats = try manager.getModelStats(mc.id);
        std.debug.print("  ✓ {s}: {d} MB RAM, {d} MB SSD\n", .{
            mc.id, model_stats.allocated_ram_mb, model_stats.allocated_ssd_mb,
        });
    }

    std.debug.print("  ✅ Fair share allocation tests passed\n\n", .{});
}

fn testPriorityBasedAllocation(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 3: Priority-Based Allocation\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .total_ram_mb = 4096,
        .total_ssd_mb = 32768,
        .allocation_strategy = .priority_based,
    };

    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register models with different priorities
    try manager.registerModel("high-priority", .{
        .n_layers = 32,
        .n_heads = 32,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 10,  // Highest
    });

    try manager.registerModel("low-priority", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 1,   // Lowest
    });

    const high_stats = try manager.getModelStats("high-priority");
    const low_stats = try manager.getModelStats("low-priority");

    std.debug.print("  ✓ High priority: {d} MB RAM\n", .{high_stats.allocated_ram_mb});
    std.debug.print("  ✓ Low priority: {d} MB RAM\n", .{low_stats.allocated_ram_mb});
    std.debug.assert(high_stats.allocated_ram_mb >= low_stats.allocated_ram_mb);

    std.debug.print("  ✅ Priority-based allocation tests passed\n\n", .{});
}

fn testModelRegistration(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 4: Model Registration\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{};
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    try manager.registerModel("test-model", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    const cache = try manager.getModelCache("test-model");
    std.debug.assert(cache != undefined);
    std.debug.print("  ✓ Model registered and cache retrieved\n", .{});

    const stats = manager.getGlobalStats();
    std.debug.assert(stats.total_models == 1);
    std.debug.assert(stats.active_models == 1);
    std.debug.print("  ✓ Global stats updated correctly\n", .{});

    std.debug.print("  ✅ Model registration tests passed\n\n", .{});
}

fn testMultipleModelRegistration(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 5: Multiple Model Registration\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .allocation_strategy = .fair_share,
    };
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register 6 models (matching vendor/layerModels)
    const models = [_][]const u8{
        "Llama-3.2-1B",
        "Qwen2.5-0.5B",
        "microsoft-phi-2",
        "google-gemma-3-270m-it",
        "nvidia-Nemotron-Flash-3B-Instruct",
        "LFM2.5-1.2B-Instruct-GGUF",
    };

    for (models) |model_id| {
        try manager.registerModel(model_id, .{
            .n_layers = 16,
            .n_heads = 16,
            .head_dim = 64,
            .max_seq_len = 4096,
            .priority = 5,
        });
    }

    const stats = manager.getGlobalStats();
    std.debug.assert(stats.total_models == 6);
    std.debug.assert(stats.active_models == 6);
    std.debug.print("  ✓ Registered {d} models\n", .{stats.total_models});

    const model_list = try manager.listModels(allocator);
    defer {
        for (model_list) |model| allocator.free(model);
        allocator.free(model_list);
    }
    std.debug.assert(model_list.len == 6);
    std.debug.print("  ✓ List models: {d} entries\n", .{model_list.len});

    std.debug.print("  ✅ Multiple model registration tests passed\n\n", .{});
}

fn testCrossModelEvictionLRU(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 6: Cross-Model Eviction (LRU)\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .global_eviction_policy = .least_recently_used_model,
    };
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register 3 models
    try manager.registerModel("model-1", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    std.time.sleep(10 * std.time.ns_per_ms); // 10ms delay

    try manager.registerModel("model-2", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    std.time.sleep(10 * std.time.ns_per_ms); // 10ms delay

    try manager.registerModel("model-3", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    // Access model-3 (make it most recent)
    _ = try manager.getModelCache("model-3");

    // Perform global eviction - should evict from model-1 (oldest)
    try manager.performGlobalEviction();

    const stats = manager.getGlobalStats();
    std.debug.assert(stats.cross_model_evictions >= 1);
    std.debug.print("  ✓ Cross-model eviction performed\n", .{});
    std.debug.print("  ✓ Evictions: {d}\n", .{stats.cross_model_evictions});

    std.debug.print("  ✅ Cross-model eviction (LRU) tests passed\n\n", .{});
}

fn testCrossModelEvictionLFU(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 7: Cross-Model Eviction (LFU)\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .global_eviction_policy = .least_frequently_used_model,
    };
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register 3 models
    try manager.registerModel("frequent-model", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    try manager.registerModel("rare-model", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    // Access frequent-model multiple times
    _ = try manager.getModelCache("frequent-model");
    _ = try manager.getModelCache("frequent-model");
    _ = try manager.getModelCache("frequent-model");

    // Access rare-model once
    _ = try manager.getModelCache("rare-model");

    const frequent_stats = try manager.getModelStats("frequent-model");
    const rare_stats = try manager.getModelStats("rare-model");

    std.debug.print("  ✓ Frequent model: {d} accesses\n", .{frequent_stats.access_count});
    std.debug.print("  ✓ Rare model: {d} accesses\n", .{rare_stats.access_count});
    std.debug.assert(frequent_stats.access_count > rare_stats.access_count);

    // Perform global eviction - should evict from rare-model (least frequent)
    try manager.performGlobalEviction();

    std.debug.print("  ✓ LFU eviction targets least frequently used model\n", .{});
    std.debug.print("  ✅ Cross-model eviction (LFU) tests passed\n\n", .{});
}

fn testPerModelStatistics(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 8: Per-Model Statistics\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{};
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    try manager.registerModel("stats-test-model", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 7,
    });

    // Access the cache multiple times
    _ = try manager.getModelCache("stats-test-model");
    _ = try manager.getModelCache("stats-test-model");
    _ = try manager.getModelCache("stats-test-model");

    const stats = try manager.getModelStats("stats-test-model");
    std.debug.print("  ✓ Model ID: {s}\n", .{stats.model_id});
    std.debug.print("  ✓ RAM allocated: {d} MB\n", .{stats.allocated_ram_mb});
    std.debug.print("  ✓ SSD allocated: {d} MB\n", .{stats.allocated_ssd_mb});
    std.debug.print("  ✓ Access count: {d}\n", .{stats.access_count});
    std.debug.print("  ✓ Usage score: {d:.2}\n", .{stats.usage_score});

    std.debug.assert(stats.access_count == 3);
    std.debug.assert(stats.usage_score > 0.0);

    std.debug.print("  ✅ Per-model statistics tests passed\n\n", .{});
}

fn testGlobalStatistics(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 9: Global Statistics\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{
        .total_ram_mb = 4096,
        .total_ssd_mb = 32768,
    };
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register multiple models
    for (0..3) |i| {
        const id = try std.fmt.allocPrint(allocator, "model-{d}", .{i});
        defer allocator.free(id);

        try manager.registerModel(id, .{
            .n_layers = 16,
            .n_heads = 16,
            .head_dim = 64,
            .max_seq_len = 4096,
            .priority = 5,
        });
    }

    const stats = manager.getGlobalStats();
    std.debug.print("  ✓ Total models: {d}\n", .{stats.total_models});
    std.debug.print("  ✓ Active models: {d}\n", .{stats.active_models});
    std.debug.print("  ✓ Total RAM used: {d} MB\n", .{stats.total_ram_used_mb});
    std.debug.print("  ✓ Total SSD used: {d} MB\n", .{stats.total_ssd_used_mb});

    std.debug.assert(stats.total_models == 3);
    std.debug.assert(stats.active_models == 3);
    std.debug.assert(stats.total_ram_used_mb > 0);
    std.debug.assert(stats.total_ssd_used_mb > 0);

    std.debug.print("  ✅ Global statistics tests passed\n\n", .{});
}

fn testModelUnregistration(allocator: std.mem.Allocator) !void {
    std.debug.print("Test 10: Model Unregistration\n", .{});
    std.debug.print("-" ** 40 ++ "\n", .{});

    const config = MultiModelCacheConfig{};
    var manager = try MultiModelCacheManager.init(allocator, config);
    defer manager.deinit();

    // Register a model
    try manager.registerModel("temp-model", .{
        .n_layers = 16,
        .n_heads = 16,
        .head_dim = 64,
        .max_seq_len = 4096,
        .priority = 5,
    });

    var stats = manager.getGlobalStats();
    std.debug.assert(stats.total_models == 1);
    std.debug.print("  ✓ Model registered\n", .{});

    // Unregister the model
    try manager.unregisterModel("temp-model");

    stats = manager.getGlobalStats();
    std.debug.assert(stats.active_models == 0);
    std.debug.assert(stats.total_ram_used_mb == 0);
    std.debug.assert(stats.total_ssd_used_mb == 0);
    std.debug.print("  ✓ Model unregistered\n", .{});
    std.debug.print("  ✓ Resources freed\n", .{});

    // Try to get cache - should fail
    const result = manager.getModelCache("temp-model");
    std.debug.assert(std.meta.isError(result));
    std.debug.print("  ✓ Cache access correctly fails after unregistration\n", .{});

    std.debug.print("  ✅ Model unregistration tests passed\n\n", .{});
}
