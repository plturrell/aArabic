// Model Cache Layer for LLM Inference Engine
// Provides LRU caching with configurable eviction for model files
//
// Features:
// - LRU cache with configurable max size
// - Cache entries with metadata (last_access, size, checksum)
// - Thread-safe operations with mutex protection
// - Local cache with remote backend fallback
// - Background cache warming support

const std = @import("std");
const storage_backend = @import("storage_backend.zig");
const StorageBackend = storage_backend.StorageBackend;
const StorageError = storage_backend.StorageError;
const LocalStorage = @import("local_storage.zig").LocalStorage;

// ========== Configuration ==========

/// Eviction policy for cache management
pub const EvictionPolicy = enum {
    lru, // Least Recently Used
    lfu, // Least Frequently Used (future)
    fifo, // First In First Out (future)

    pub fn toString(self: EvictionPolicy) []const u8 {
        return switch (self) {
            .lru => "lru",
            .lfu => "lfu",
            .fifo => "fifo",
        };
    }
};

/// Configuration for ModelCache
pub const ModelCacheConfig = struct {
    /// Maximum cache size in gigabytes
    max_cache_size_gb: f64 = 10.0,
    /// Local cache directory path
    cache_path: []const u8,
    /// Cache eviction policy
    eviction_policy: EvictionPolicy = .lru,
    /// Enable background warming
    enable_warming: bool = true,
    /// Number of warming threads
    warming_threads: u8 = 2,

    /// Calculate max cache size in bytes
    pub fn maxCacheSizeBytes(self: ModelCacheConfig) u64 {
        return @intFromFloat(self.max_cache_size_gb * 1024.0 * 1024.0 * 1024.0);
    }
};

// ========== Cache Entry ==========

/// Metadata for a cached model file
pub const ModelCacheEntry = struct {
    /// Relative path in cache
    path: []const u8,
    /// Size in bytes
    size: u64,
    /// MD5 or SHA256 checksum
    checksum: [32]u8,
    /// Last access timestamp (nanoseconds since epoch)
    last_access: i128,
    /// Access count for LFU
    access_count: u64,

    /// Update last access time to now
    pub fn touch(self: *ModelCacheEntry) void {
        self.last_access = std.time.nanoTimestamp();
        self.access_count += 1;
    }

    /// Check if entry is stale (not accessed in given duration)
    pub fn isStale(self: *const ModelCacheEntry, max_age_ns: i128) bool {
        const now = std.time.nanoTimestamp();
        return (now - self.last_access) > max_age_ns;
    }
};

// ========== Warming Task ==========

/// Task for background cache warming
pub const WarmingTask = struct {
    model_path: []const u8,
    priority: u8,
    callback: ?*const fn (path: []const u8, success: bool) void,
};

// ========== Model Cache ==========

/// Thread-safe model cache with LRU eviction
pub const ModelCache = struct {
    /// Allocator for dynamic memory
    allocator: std.mem.Allocator,
    /// Cache configuration
    config: ModelCacheConfig,
    /// Remote storage backend for fallback
    remote_backend: StorageBackend,
    /// Local cache storage
    local_storage: *LocalStorage,
    /// Cache entries indexed by path
    entries: std.StringHashMap(ModelCacheEntry),
    /// Current total cache size in bytes
    current_size: u64,
    /// Mutex for thread safety
    mutex: std.Thread.Mutex,
    /// Warming task queue
    warming_queue: std.ArrayList(WarmingTask),
    /// Warming thread handles
    warming_threads: []std.Thread,
    /// Flag to stop warming threads
    stop_warming: std.atomic.Value(bool),

    const Self = @This();

    /// Initialize model cache
    pub fn init(
        allocator: std.mem.Allocator,
        config: ModelCacheConfig,
        remote_backend: StorageBackend,
    ) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        const local_storage = try LocalStorage.init(allocator, config.cache_path);
        errdefer local_storage.deinit();

        self.* = .{
            .allocator = allocator,
            .config = .{
                .max_cache_size_gb = config.max_cache_size_gb,
                .cache_path = try allocator.dupe(u8, config.cache_path),
                .eviction_policy = config.eviction_policy,
                .enable_warming = config.enable_warming,
                .warming_threads = config.warming_threads,
            },
            .remote_backend = remote_backend,
            .local_storage = local_storage,
            .entries = std.StringHashMap(ModelCacheEntry).init(allocator),
            .current_size = 0,
            .mutex = .{},
            .warming_queue = std.ArrayList(WarmingTask){},
            .warming_threads = &.{},
            .stop_warming = std.atomic.Value(bool).init(false),
        };

        // Start warming threads if enabled
        if (config.enable_warming and config.warming_threads > 0) {
            try self.startWarmingThreads();
        }

        return self;
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.stopWarmingThreads();

        // Free entry paths
        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
        }
        self.entries.deinit();

        self.warming_queue.deinit();
        self.local_storage.deinit();
        self.allocator.free(self.config.cache_path);
        self.allocator.destroy(self);
    }

    // ========== Core Operations ==========

    /// Get model from cache or fetch from remote backend
    pub fn get(self: *Self, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check local cache first
        if (self.entries.getPtr(path)) |entry| {
            entry.touch();
            std.log.debug("[model_cache] cache hit: {s}", .{path});
            return self.local_storage.backend().read(path, allocator);
        }

        // Cache miss - fetch from remote
        std.log.debug("[model_cache] cache miss, fetching: {s}", .{path});
        const data = try self.remote_backend.read(path, allocator);
        errdefer allocator.free(data);

        // Store in cache (may trigger eviction)
        try self.putInternal(path, data);

        return data;
    }

    /// Store model in cache with automatic eviction if needed
    pub fn put(self: *Self, path: []const u8, data: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        try self.putInternal(path, data);
    }

    /// Internal put without locking (caller must hold mutex)
    fn putInternal(self: *Self, path: []const u8, data: []const u8) !void {
        const size: u64 = data.len;

        // Evict entries if needed to make room
        try self.evictIfNeeded(size);

        // Write to local storage
        try self.local_storage.backend().write(path, data);

        // Create cache entry
        const path_copy = try self.allocator.dupe(u8, path);
        errdefer self.allocator.free(path_copy);

        var checksum: [32]u8 = undefined;
        computeChecksum(data, &checksum);

        const entry = ModelCacheEntry{
            .path = path_copy,
            .size = size,
            .checksum = checksum,
            .last_access = std.time.nanoTimestamp(),
            .access_count = 1,
        };

        // Remove old entry if exists
        if (self.entries.fetchRemove(path)) |old| {
            self.current_size -= old.value.size;
            self.allocator.free(old.key);
        }

        try self.entries.put(path_copy, entry);
        self.current_size += size;

        std.log.info("[model_cache] cached: {s} ({d} bytes)", .{ path, size });
    }

    /// Remove model from cache
    pub fn invalidate(self: *Self, path: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.entries.fetchRemove(path)) |removed| {
            self.current_size -= removed.value.size;
            self.allocator.free(removed.key);

            // Delete from local storage
            self.local_storage.backend().delete(path) catch |err| {
                std.log.warn("[model_cache] failed to delete {s}: {}", .{ path, err });
            };

            std.log.info("[model_cache] invalidated: {s}", .{path});
        }
    }

    /// Pre-fetch models from remote backend for cache warming
    pub fn warm(self: *Self, paths: []const []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        for (paths) |path| {
            // Skip if already cached
            if (self.entries.contains(path)) continue;

            const task = WarmingTask{
                .model_path = try self.allocator.dupe(u8, path),
                .priority = 0,
                .callback = null,
            };
            try self.warming_queue.append(task);
        }

        std.log.info("[model_cache] queued {d} models for warming", .{paths.len});
    }

    // ========== Eviction ==========

    /// Evict entries to make room for new data
    fn evictIfNeeded(self: *Self, required_size: u64) !void {
        const max_size = self.config.maxCacheSizeBytes();

        while (self.current_size + required_size > max_size and self.entries.count() > 0) {
            const victim = self.findEvictionVictim() orelse break;
            const victim_path = victim.path;

            // Remove from storage
            self.local_storage.backend().delete(victim_path) catch {};

            // Remove from entries
            if (self.entries.fetchRemove(victim_path)) |removed| {
                self.current_size -= removed.value.size;
                std.log.info("[model_cache] evicted: {s} ({d} bytes)", .{ victim_path, removed.value.size });
                self.allocator.free(removed.key);
            }
        }
    }

    /// Find the best entry to evict based on policy
    fn findEvictionVictim(self: *Self) ?*ModelCacheEntry {
        var victim: ?*ModelCacheEntry = null;
        var iter = self.entries.valueIterator();

        while (iter.next()) |entry| {
            if (victim == null) {
                victim = entry;
                continue;
            }

            switch (self.config.eviction_policy) {
                .lru => {
                    if (entry.last_access < victim.?.last_access) {
                        victim = entry;
                    }
                },
                .lfu => {
                    if (entry.access_count < victim.?.access_count) {
                        victim = entry;
                    }
                },
                .fifo => {
                    // Use last_access as insertion time for FIFO
                    if (entry.last_access < victim.?.last_access) {
                        victim = entry;
                    }
                },
            }
        }

        return victim;
    }


    // ========== Warming Threads ==========

    /// Start background warming threads
    fn startWarmingThreads(self: *Self) !void {
        const thread_count = self.config.warming_threads;
        self.warming_threads = try self.allocator.alloc(std.Thread, thread_count);

        for (0..thread_count) |i| {
            self.warming_threads[i] = try std.Thread.spawn(.{}, warmingWorker, .{self});
        }
    }

    /// Stop all warming threads
    fn stopWarmingThreads(self: *Self) void {
        self.stop_warming.store(true, .release);

        for (self.warming_threads) |thread| {
            thread.join();
        }

        if (self.warming_threads.len > 0) {
            self.allocator.free(self.warming_threads);
        }

        // Clean up remaining queue items
        for (self.warming_queue.items) |task| {
            self.allocator.free(task.model_path);
        }
    }

    /// Worker function for warming threads
    fn warmingWorker(self: *Self) void {
        while (!self.stop_warming.load(.acquire)) {
            const task = blk: {
                self.mutex.lock();
                defer self.mutex.unlock();

                if (self.warming_queue.items.len > 0) {
                    break :blk self.warming_queue.orderedRemove(0);
                }
                break :blk null;
            };

            if (task) |t| {
                defer self.allocator.free(t.model_path);

                const success = self.warmSingle(t.model_path);
                if (t.callback) |cb| {
                    cb(t.model_path, success);
                }
            } else {
                // No work, sleep briefly
                std.time.sleep(100 * std.time.ns_per_ms);
            }
        }
    }

    /// Warm a single model path
    fn warmSingle(self: *Self, path: []const u8) bool {
        // Check if already cached
        {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.entries.contains(path)) return true;
        }

        // Fetch from remote
        const data = self.remote_backend.read(path, self.allocator) catch |err| {
            std.log.warn("[model_cache] warming failed for {s}: {}", .{ path, err });
            return false;
        };
        defer self.allocator.free(data);

        // Store in cache
        self.put(path, data) catch |err| {
            std.log.warn("[model_cache] warming cache write failed: {}", .{err});
            return false;
        };

        std.log.info("[model_cache] warmed: {s}", .{path});
        return true;
    }

    // ========== Statistics ==========

    /// Cache statistics
    pub const CacheStats = struct {
        entry_count: usize,
        current_size_bytes: u64,
        max_size_bytes: u64,
        utilization_percent: f64,
        warming_queue_size: usize,
    };

    /// Get current cache statistics
    pub fn getStats(self: *Self) CacheStats {
        self.mutex.lock();
        defer self.mutex.unlock();

        const max_size = self.config.maxCacheSizeBytes();
        const utilization = if (max_size > 0)
            @as(f64, @floatFromInt(self.current_size)) / @as(f64, @floatFromInt(max_size)) * 100.0
        else
            0.0;

        return .{
            .entry_count = self.entries.count(),
            .current_size_bytes = self.current_size,
            .max_size_bytes = max_size,
            .utilization_percent = utilization,
            .warming_queue_size = self.warming_queue.items.len,
        };
    }

    /// Check if path exists in cache
    pub fn contains(self: *Self, path: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.entries.contains(path);
    }

    /// Clear entire cache
    pub fn clear(self: *Self) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var iter = self.entries.iterator();
        while (iter.next()) |entry| {
            self.local_storage.backend().delete(entry.key_ptr.*) catch {};
            self.allocator.free(entry.key_ptr.*);
        }
        self.entries.clearRetainingCapacity();
        self.current_size = 0;

        std.log.info("[model_cache] cache cleared", .{});
    }
};

// ========== Utility Functions ==========

/// Compute checksum for data validation
fn computeChecksum(data: []const u8, out: *[32]u8) void {
    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    hasher.update(data);
    hasher.final(out);
}

// ========== Tests ==========

test "ModelCacheConfig maxCacheSizeBytes" {
    const config = ModelCacheConfig{
        .cache_path = "/tmp/cache",
        .max_cache_size_gb = 1.0,
    };
    try std.testing.expectEqual(@as(u64, 1073741824), config.maxCacheSizeBytes());
}

test "ModelCacheEntry touch updates access time" {
    var entry = ModelCacheEntry{
        .path = "test",
        .size = 100,
        .checksum = [_]u8{0} ** 32,
        .last_access = 0,
        .access_count = 0,
    };

    entry.touch();
    try std.testing.expect(entry.last_access > 0);
    try std.testing.expectEqual(@as(u64, 1), entry.access_count);
}

test "EvictionPolicy toString" {
    try std.testing.expectEqualStrings("lru", EvictionPolicy.lru.toString());
    try std.testing.expectEqualStrings("lfu", EvictionPolicy.lfu.toString());
    try std.testing.expectEqualStrings("fifo", EvictionPolicy.fifo.toString());
}

test "computeChecksum produces consistent results" {
    const data = "test data for checksum";
    var checksum1: [32]u8 = undefined;
    var checksum2: [32]u8 = undefined;

    computeChecksum(data, &checksum1);
    computeChecksum(data, &checksum2);

    try std.testing.expectEqualSlices(u8, &checksum1, &checksum2);
}

