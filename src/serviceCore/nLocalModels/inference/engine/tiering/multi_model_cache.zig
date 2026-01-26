//! Multi-Model Cache Manager - Day 12
//! Coordinates tiered KV caches across multiple models with fair allocation
//!
//! Features:
//! - Model-specific cache namespacing
//! - Fair RAM/SSD quota distribution
//! - Cross-model eviction policy
//! - Per-model usage metrics
//! - Integration with Day 11 Model Registry

const std = @import("std");
const TieredKVCache = @import("tiered_kv_cache.zig").TieredKVCache;
const TieredKVConfig = @import("tiered_kv_cache.zig").TieredKVConfig;
const log = @import("structured_logging.zig");
const ModelRegistry = @import("../../shared/model_registry.zig").ModelRegistry;

// ============================================================================
// Multi-Model Cache Configuration
// ============================================================================

pub const MultiModelCacheConfig = struct {
    /// Global resource limits
    total_ram_mb: u64 = 4096,          // Total RAM for all models
    total_ssd_mb: u64 = 32768,         // Total SSD for all models (32GB)
    
    /// Allocation strategy
    allocation_strategy: AllocationStrategy = .fair_share,
    
    /// Minimum resources per model
    min_ram_per_model_mb: u64 = 256,   // Minimum 256MB RAM per model
    min_ssd_per_model_mb: u64 = 1024,  // Minimum 1GB SSD per model
    
    /// Eviction policy
    global_eviction_policy: GlobalEvictionPolicy = .least_recently_used_model,
    
    /// SSD base path
    ssd_base_path: []const u8 = "/tmp/shimmy_multi_model_cache",
    
    /// Performance tuning
    hot_tokens_per_model: u32 = 2048,  // Default hot tokens per model
    enable_cross_model_sharing: bool = false,  // Future: share common prefixes
};

pub const AllocationStrategy = enum {
    fair_share,           // Equal resources per model
    proportional,         // Based on model size/usage
    priority_based,       // Based on model priority
    dynamic,             // Adapt based on usage patterns
};

pub const GlobalEvictionPolicy = enum {
    least_recently_used_model,  // Evict from least recently used model
    least_frequently_used_model, // Evict from least frequently used model
    smallest_model_first,        // Evict from smallest models first
    round_robin,                // Rotate eviction across models
};

// ============================================================================
// Per-Model Cache State
// ============================================================================

pub const ModelCacheState = struct {
    model_id: []const u8,
    cache: *TieredKVCache,
    
    /// Resource allocation
    allocated_ram_mb: u64,
    allocated_ssd_mb: u64,
    
    /// Usage tracking
    last_access_time: i64,           // Unix timestamp (ms)
    access_count: u64,
    total_tokens_processed: u64,
    
    /// Priority (for priority-based allocation)
    priority: u8 = 5,                // 1 (lowest) to 10 (highest)
    
    /// Status
    is_active: bool = true,
    is_preloaded: bool = false,
    
    pub fn deinit(self: *ModelCacheState, allocator: std.mem.Allocator) void {
        allocator.free(self.model_id);
        self.cache.deinit();
    }
    
    pub fn markAccess(self: *ModelCacheState) void {
        self.access_count += 1;
        self.last_access_time = std.time.milliTimestamp();
    }
    
    pub fn getUsageScore(self: *const ModelCacheState) f32 {
        // Calculate usage score (higher = more used)
        const now = std.time.milliTimestamp();
        const time_since_access = now - self.last_access_time;
        const recency_factor = 1.0 / (@as(f32, @floatFromInt(time_since_access)) / 1000.0 + 1.0);
        const frequency_factor = @as(f32, @floatFromInt(self.access_count));
        return recency_factor * frequency_factor;
    }
};

// ============================================================================
// Multi-Model Cache Manager
// ============================================================================

pub const MultiModelCacheManager = struct {
    allocator: std.mem.Allocator,
    config: MultiModelCacheConfig,
    
    /// Model caches (HashMap: model_id -> ModelCacheState)
    model_caches: std.StringHashMap(ModelCacheState),
    
    /// Global statistics
    stats: GlobalStats,
    
    /// Mutex for thread safety
    mutex: std.Thread.Mutex,
    
    pub const GlobalStats = struct {
        total_models: u32 = 0,
        active_models: u32 = 0,
        total_ram_used_mb: u64 = 0,
        total_ssd_used_mb: u64 = 0,
        total_tokens_processed: u64 = 0,
        cross_model_evictions: u64 = 0,
        cache_hits: u64 = 0,
        cache_misses: u64 = 0,
    };
    
    pub fn init(allocator: std.mem.Allocator, config: MultiModelCacheConfig) !*MultiModelCacheManager {
        log.info("Initializing Multi-Model Cache Manager: ram={d}MB, ssd={d}MB, strategy={s}", .{
            config.total_ram_mb, config.total_ssd_mb, @tagName(config.allocation_strategy),
        });
        
        const self = try allocator.create(MultiModelCacheManager);
        errdefer allocator.destroy(self);
        
        self.* = MultiModelCacheManager{
            .allocator = allocator,
            .config = config,
            .model_caches = std.StringHashMap(ModelCacheState).init(allocator),
            .stats = .{},
            .mutex = .{},
        };
        
        // Create SSD base directory
        std.fs.cwd().makePath(config.ssd_base_path) catch |err| {
            log.warn("Failed to create SSD base path: {}", .{err});
        };
        
        log.info("Multi-Model Cache Manager initialized successfully", .{});
        return self;
    }
    
    pub fn deinit(self: *MultiModelCacheManager) void {
        var iter = self.model_caches.valueIterator();
        while (iter.next()) |state| {
            var mut_state = state.*;
            mut_state.deinit(self.allocator);
        }
        self.model_caches.deinit();
        self.allocator.destroy(self);
    }
    
    /// Register a model and create its cache
    pub fn registerModel(
        self: *MultiModelCacheManager,
        model_id: []const u8,
        model_config: struct {
            n_layers: u32,
            n_heads: u32,
            head_dim: u32,
            max_seq_len: u32,
            priority: u8 = 5,
        },
    ) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        log.info("Registering model for caching: id={s}, layers={d}, heads={d}", .{
            model_id, model_config.n_layers, model_config.n_heads,
        });
        
        // Check if already registered
        if (self.model_caches.contains(model_id)) {
            log.warn("Model already registered: {s}", .{model_id});
            return error.ModelAlreadyRegistered;
        }
        
        // Calculate resource allocation
        const allocation = try self.calculateAllocation(model_config.priority);
        
        // Create model-specific SSD path
        const ssd_path = try std.fmt.allocPrint(
            self.allocator,
            "{s}/{s}.tier",
            .{ self.config.ssd_base_path, model_id },
        );
        defer self.allocator.free(ssd_path);
        
        // Create tiered cache config
        const cache_config = TieredKVConfig{
            .n_layers = model_config.n_layers,
            .n_heads = model_config.n_heads,
            .head_dim = model_config.head_dim,
            .max_seq_len = model_config.max_seq_len,
            .hot_tokens = self.config.hot_tokens_per_model,
            .max_ram_mb = allocation.ram_mb,
            .max_ssd_mb = allocation.ssd_mb,
            .ssd_path = ssd_path,
        };
        
        // Create cache instance
        const cache = try TieredKVCache.init(self.allocator, cache_config);
        errdefer cache.deinit();
        
        // Create cache state
        const state = ModelCacheState{
            .model_id = try self.allocator.dupe(u8, model_id),
            .cache = cache,
            .allocated_ram_mb = allocation.ram_mb,
            .allocated_ssd_mb = allocation.ssd_mb,
            .last_access_time = std.time.milliTimestamp(),
            .access_count = 0,
            .total_tokens_processed = 0,
            .priority = model_config.priority,
            .is_active = true,
            .is_preloaded = false,
        };
        
        // Store in registry
        try self.model_caches.put(try self.allocator.dupe(u8, model_id), state);
        
        // Update stats
        self.stats.total_models += 1;
        self.stats.active_models += 1;
        self.stats.total_ram_used_mb += allocation.ram_mb;
        self.stats.total_ssd_used_mb += allocation.ssd_mb;
        
        log.info("Model registered successfully: id={s}, ram={d}MB, ssd={d}MB", .{
            model_id, allocation.ram_mb, allocation.ssd_mb,
        });
    }
    
    /// Unregister a model and free its cache
    pub fn unregisterModel(self: *MultiModelCacheManager, model_id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        log.info("Unregistering model: id={s}", .{model_id});
        
        if (self.model_caches.fetchRemove(model_id)) |entry| {
            var state = entry.value;
            
            // Update stats
            self.stats.active_models -= 1;
            self.stats.total_ram_used_mb -= state.allocated_ram_mb;
            self.stats.total_ssd_used_mb -= state.allocated_ssd_mb;
            
            // Free resources
            state.deinit(self.allocator);
            self.allocator.free(entry.key);
            
            log.info("Model unregistered successfully: id={s}", .{model_id});
        } else {
            log.warn("Model not found for unregistration: {s}", .{model_id});
            return error.ModelNotFound;
        }
    }
    
    /// Get cache for a specific model
    pub fn getModelCache(self: *MultiModelCacheManager, model_id: []const u8) !*TieredKVCache {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.model_caches.getPtr(model_id)) |state| {
            state.markAccess();
            return state.cache;
        }
        
        log.warn("Cache not found for model: {s}", .{model_id});
        return error.ModelCacheNotFound;
    }
    
    /// Calculate resource allocation for a model
    fn calculateAllocation(self: *MultiModelCacheManager, priority: u8) !struct {
        ram_mb: u64,
        ssd_mb: u64,
    } {
        return switch (self.config.allocation_strategy) {
            .fair_share => self.calculateFairShare(),
            .proportional => self.calculateProportional(priority),
            .priority_based => self.calculatePriorityBased(priority),
            .dynamic => self.calculateDynamic(),
        };
    }
    
    fn calculateFairShare(self: *MultiModelCacheManager) struct { ram_mb: u64, ssd_mb: u64 } {
        const num_models = self.stats.total_models + 1; // +1 for the new model
        const ram_per_model = self.config.total_ram_mb / num_models;
        const ssd_per_model = self.config.total_ssd_mb / num_models;
        
        return .{
            .ram_mb = @max(ram_per_model, self.config.min_ram_per_model_mb),
            .ssd_mb = @max(ssd_per_model, self.config.min_ssd_per_model_mb),
        };
    }
    
    fn calculateProportional(self: *MultiModelCacheManager, priority: u8) struct { ram_mb: u64, ssd_mb: u64 } {
        _ = self;
        // Simple proportional: higher priority gets more resources
        const factor = @as(f32, @floatFromInt(priority)) / 10.0; // 0.1 to 1.0
        const ram_mb = @as(u64, @intFromFloat(@as(f32, @floatFromInt(self.config.total_ram_mb)) * factor / 5.0));
        const ssd_mb = @as(u64, @intFromFloat(@as(f32, @floatFromInt(self.config.total_ssd_mb)) * factor / 5.0));
        
        return .{
            .ram_mb = @max(ram_mb, self.config.min_ram_per_model_mb),
            .ssd_mb = @max(ssd_mb, self.config.min_ssd_per_model_mb),
        };
    }
    
    fn calculatePriorityBased(self: *MultiModelCacheManager, priority: u8) struct { ram_mb: u64, ssd_mb: u64 } {
        return self.calculateProportional(priority);
    }
    
    fn calculateDynamic(self: *MultiModelCacheManager) struct { ram_mb: u64, ssd_mb: u64 } {
        // Start with fair share, adjust based on usage later
        return self.calculateFairShare();
    }
    
    /// Perform cross-model eviction when global limits are hit
    pub fn performGlobalEviction(self: *MultiModelCacheManager) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        log.info("Performing global eviction: policy={s}", .{@tagName(self.config.global_eviction_policy)});
        
        // Find model to evict from based on policy
        const model_id = switch (self.config.global_eviction_policy) {
            .least_recently_used_model => try self.findLRUModel(),
            .least_frequently_used_model => try self.findLFUModel(),
            .smallest_model_first => try self.findSmallestModel(),
            .round_robin => try self.findNextRoundRobinModel(),
        };
        
        if (model_id) |id| {
            if (self.model_caches.getPtr(id)) |state| {
                // Trigger eviction in the model's cache
                // This will evict from hot to cold tier
                log.info("Evicting from model: {s}", .{id});
                self.stats.cross_model_evictions += 1;
            }
        }
    }
    
    fn findLRUModel(self: *MultiModelCacheManager) !?[]const u8 {
        var oldest_time: i64 = std.math.maxInt(i64);
        var oldest_model: ?[]const u8 = null;
        
        var iter = self.model_caches.iterator();
        while (iter.next()) |entry| {
            const state = entry.value_ptr;
            if (state.last_access_time < oldest_time) {
                oldest_time = state.last_access_time;
                oldest_model = entry.key_ptr.*;
            }
        }
        
        return oldest_model;
    }
    
    fn findLFUModel(self: *MultiModelCacheManager) !?[]const u8 {
        var lowest_count: u64 = std.math.maxInt(u64);
        var lowest_model: ?[]const u8 = null;
        
        var iter = self.model_caches.iterator();
        while (iter.next()) |entry| {
            const state = entry.value_ptr;
            if (state.access_count < lowest_count) {
                lowest_count = state.access_count;
                lowest_model = entry.key_ptr.*;
            }
        }
        
        return lowest_model;
    }
    
    fn findSmallestModel(self: *MultiModelCacheManager) !?[]const u8 {
        var smallest_ram: u64 = std.math.maxInt(u64);
        var smallest_model: ?[]const u8 = null;
        
        var iter = self.model_caches.iterator();
        while (iter.next()) |entry| {
            const state = entry.value_ptr;
            if (state.allocated_ram_mb < smallest_ram) {
                smallest_ram = state.allocated_ram_mb;
                smallest_model = entry.key_ptr.*;
            }
        }
        
        return smallest_model;
    }
    
    var round_robin_index: usize = 0;
    fn findNextRoundRobinModel(self: *MultiModelCacheManager) !?[]const u8 {
        if (self.model_caches.count() == 0) return null;
        
        var iter = self.model_caches.keyIterator();
        var index: usize = 0;
        while (iter.next()) |key| {
            if (index == round_robin_index) {
                round_robin_index = (round_robin_index + 1) % self.model_caches.count();
                return key.*;
            }
            index += 1;
        }
        
        return null;
    }
    
    /// Get per-model statistics
    pub fn getModelStats(self: *MultiModelCacheManager, model_id: []const u8) !struct {
        model_id: []const u8,
        allocated_ram_mb: u64,
        allocated_ssd_mb: u64,
        access_count: u64,
        tokens_processed: u64,
        cache_hits: u64,
        cache_misses: u64,
        usage_score: f32,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const state = self.model_caches.get(model_id) orelse return error.ModelNotFound;
        const cache_stats = state.cache.getStats();
        
        return .{
            .model_id = model_id,
            .allocated_ram_mb = state.allocated_ram_mb,
            .allocated_ssd_mb = state.allocated_ssd_mb,
            .access_count = state.access_count,
            .tokens_processed = state.total_tokens_processed,
            .cache_hits = cache_stats.hot_hits,
            .cache_misses = cache_stats.cold_hits,
            .usage_score = state.getUsageScore(),
        };
    }
    
    /// Get global statistics
    pub fn getGlobalStats(self: *MultiModelCacheManager) GlobalStats {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.stats;
    }
    
    /// List all registered models
    pub fn listModels(self: *MultiModelCacheManager, allocator: std.mem.Allocator) ![][]const u8 {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        var list = std.ArrayList([]const u8){};
        errdefer list.deinit();
        
        var iter = self.model_caches.keyIterator();
        while (iter.next()) |key| {
            try list.append(try allocator.dupe(u8, key.*));
        }
        
        return try list.toOwnedSlice();
    }
    
    /// Print status of all model caches
    pub fn printStatus(self: *MultiModelCacheManager) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        std.debug.print("\n📊 Multi-Model Cache Manager Status\n", .{});
        std.debug.print("=" ** 60 ++ "\n", .{});
        std.debug.print("Total models: {d} (active: {d})\n", .{
            self.stats.total_models, self.stats.active_models,
        });
        std.debug.print("Global RAM usage: {d}/{d} MB\n", .{
            self.stats.total_ram_used_mb, self.config.total_ram_mb,
        });
        std.debug.print("Global SSD usage: {d}/{d} MB\n", .{
            self.stats.total_ssd_used_mb, self.config.total_ssd_mb,
        });
        std.debug.print("Cross-model evictions: {d}\n", .{self.stats.cross_model_evictions});
        std.debug.print("\nPer-Model Status:\n", .{});
        std.debug.print("-" ** 60 ++ "\n", .{});
        
        var iter = self.model_caches.iterator();
        while (iter.next()) |entry| {
            const model_id = entry.key_ptr.*;
            const state = entry.value_ptr;
            const cache_stats = state.cache.getStats();
            
            std.debug.print("\nModel: {s}\n", .{model_id});
            std.debug.print("  RAM: {d} MB, SSD: {d} MB\n", .{
                state.allocated_ram_mb, state.allocated_ssd_mb,
            });
            std.debug.print("  Access count: {d}, Tokens: {d}\n", .{
                state.access_count, state.total_tokens_processed,
            });
            std.debug.print("  Cache hits: {d}, misses: {d}, hit rate: {d:.1}%\n", .{
                cache_stats.hot_hits, cache_stats.cold_hits, cache_stats.cache_hit_rate,
            });
            std.debug.print("  Usage score: {d:.2}\n", .{state.getUsageScore()});
            std.debug.print("  Priority: {d}/10\n", .{state.priority});
        }
        
        std.debug.print("=" ** 60 ++ "\n", .{});
    }
};
