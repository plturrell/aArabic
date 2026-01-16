// Type System Performance Optimization
// Day 69: Caching, incremental checking, and benchmarks

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const AutoHashMap = std.AutoHashMap;

// ============================================================================
// Type Cache
// ============================================================================

/// Cache for type checking results
pub const TypeCache = struct {
    allocator: Allocator,
    protocol_cache: StringHashMap(CachedProtocol),
    impl_cache: ImplCache,
    constraint_cache: ConstraintCache,
    hits: usize,
    misses: usize,
    
    pub const CachedProtocol = struct {
        valid: bool,
        timestamp: i64,
        error_count: usize,
    };
    
    pub const ImplCache = struct {
        entries: StringHashMap(bool),
        
        pub fn init(allocator: Allocator) ImplCache {
            return ImplCache{
                .entries = StringHashMap(bool).init(allocator),
            };
        }
        
        pub fn deinit(self: *ImplCache) void {
            var it = self.entries.keyIterator();
            while (it.next()) |key| {
                self.entries.allocator.free(key.*);
            }
            self.entries.deinit();
        }
        
        pub fn makeKey(allocator: Allocator, protocol: []const u8, type_name: []const u8) ![]const u8 {
            return try std.fmt.allocPrint(allocator, "{s}::{s}", .{ protocol, type_name });
        }
    };
    
    pub const ConstraintCache = struct {
        entries: StringHashMap(bool),
        
        pub fn init(allocator: Allocator) ConstraintCache {
            return ConstraintCache{
                .entries = StringHashMap(bool).init(allocator),
            };
        }
        
        pub fn deinit(self: *ConstraintCache) void {
            var it = self.entries.keyIterator();
            while (it.next()) |key| {
                self.entries.allocator.free(key.*);
            }
            self.entries.deinit();
        }
    };
    
    pub fn init(allocator: Allocator) TypeCache {
        return TypeCache{
            .allocator = allocator,
            .protocol_cache = StringHashMap(CachedProtocol).init(allocator),
            .impl_cache = ImplCache.init(allocator),
            .constraint_cache = ConstraintCache.init(allocator),
            .hits = 0,
            .misses = 0,
        };
    }
    
    pub fn deinit(self: *TypeCache) void {
        var it = self.protocol_cache.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.protocol_cache.deinit();
        self.impl_cache.deinit();
        self.constraint_cache.deinit();
    }
    
    pub fn cacheProtocol(self: *TypeCache, name: []const u8, valid: bool, error_count: usize) !void {
        const key = try self.allocator.dupe(u8, name);
        try self.protocol_cache.put(key, .{
            .valid = valid,
            .timestamp = std.time.timestamp(),
            .error_count = error_count,
        });
    }
    
    pub fn getProtocol(self: *TypeCache, name: []const u8) ?CachedProtocol {
        if (self.protocol_cache.get(name)) |cached| {
            self.hits += 1;
            return cached;
        }
        self.misses += 1;
        return null;
    }
    
    pub fn cacheImpl(self: *TypeCache, protocol: []const u8, type_name: []const u8, valid: bool) !void {
        const key = try ImplCache.makeKey(self.allocator, protocol, type_name);
        try self.impl_cache.entries.put(key, valid);
    }
    
    pub fn getImpl(self: *TypeCache, protocol: []const u8, type_name: []const u8) !?bool {
        const key = try ImplCache.makeKey(self.allocator, protocol, type_name);
        defer self.allocator.free(key);
        
        if (self.impl_cache.entries.get(key)) |valid| {
            self.hits += 1;
            return valid;
        }
        self.misses += 1;
        return null;
    }
    
    pub fn getHitRate(self: *TypeCache) f64 {
        const total = self.hits + self.misses;
        if (total == 0) return 0.0;
        return @as(f64, @floatFromInt(self.hits)) / @as(f64, @floatFromInt(total));
    }
    
    pub fn clear(self: *TypeCache) void {
        var it = self.protocol_cache.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.protocol_cache.clearRetainingCapacity();
        
        var impl_it = self.impl_cache.entries.keyIterator();
        while (impl_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.impl_cache.entries.clearRetainingCapacity();
        
        var constraint_it = self.constraint_cache.entries.keyIterator();
        while (constraint_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.constraint_cache.entries.clearRetainingCapacity();
        
        self.hits = 0;
        self.misses = 0;
    }
};

// ============================================================================
// Memory Pool
// ============================================================================

/// Memory pool for frequently allocated types
pub const MemoryPool = struct {
    allocator: Allocator,
    small_blocks: ArrayList([]u8),
    medium_blocks: ArrayList([]u8),
    large_blocks: ArrayList([]u8),
    
    const SMALL_SIZE = 64;
    const MEDIUM_SIZE = 256;
    const LARGE_SIZE = 1024;
    
    pub fn init(allocator: Allocator) MemoryPool {
        return MemoryPool{
            .allocator = allocator,
            .small_blocks = ArrayList([]u8){},
            .medium_blocks = ArrayList([]u8){},
            .large_blocks = ArrayList([]u8){},
        };
    }
    
    pub fn deinit(self: *MemoryPool) void {
        for (self.small_blocks.items) |block| {
            self.allocator.free(block);
        }
        self.small_blocks.deinit(self.allocator);
        
        for (self.medium_blocks.items) |block| {
            self.allocator.free(block);
        }
        self.medium_blocks.deinit(self.allocator);
        
        for (self.large_blocks.items) |block| {
            self.allocator.free(block);
        }
        self.large_blocks.deinit(self.allocator);
    }
    
    pub fn allocate(self: *MemoryPool, size: usize) ![]u8 {
        if (size <= SMALL_SIZE) {
            if (self.small_blocks.items.len > 0) {
                return self.small_blocks.pop() orelse return try self.allocator.alloc(u8, SMALL_SIZE);
            }
            return try self.allocator.alloc(u8, SMALL_SIZE);
        } else if (size <= MEDIUM_SIZE) {
            if (self.medium_blocks.items.len > 0) {
                return self.medium_blocks.pop() orelse return try self.allocator.alloc(u8, MEDIUM_SIZE);
            }
            return try self.allocator.alloc(u8, MEDIUM_SIZE);
        } else if (size <= LARGE_SIZE) {
            if (self.large_blocks.items.len > 0) {
                return self.large_blocks.pop() orelse return try self.allocator.alloc(u8, LARGE_SIZE);
            }
            return try self.allocator.alloc(u8, LARGE_SIZE);
        }
        return try self.allocator.alloc(u8, size);
    }
    
    pub fn free(self: *MemoryPool, block: []u8) !void {
        if (block.len == SMALL_SIZE) {
            try self.small_blocks.append(self.allocator, block);
        } else if (block.len == MEDIUM_SIZE) {
            try self.medium_blocks.append(self.allocator, block);
        } else if (block.len == LARGE_SIZE) {
            try self.large_blocks.append(self.allocator, block);
        } else {
            self.allocator.free(block);
        }
    }
};

// ============================================================================
// Incremental Checker
// ============================================================================

/// Incremental type checking state
pub const IncrementalChecker = struct {
    allocator: Allocator,
    cache: TypeCache,
    dirty: StringHashMap(void),
    dependencies: StringHashMap(ArrayList([]const u8)),
    
    pub fn init(allocator: Allocator) IncrementalChecker {
        return IncrementalChecker{
            .allocator = allocator,
            .cache = TypeCache.init(allocator),
            .dirty = StringHashMap(void).init(allocator),
            .dependencies = StringHashMap(ArrayList([]const u8)).init(allocator),
        };
    }
    
    pub fn deinit(self: *IncrementalChecker) void {
        self.cache.deinit();
        
        var dirty_it = self.dirty.keyIterator();
        while (dirty_it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.dirty.deinit();
        
        var deps_it = self.dependencies.iterator();
        while (deps_it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            for (entry.value_ptr.items) |dep| {
                self.allocator.free(dep);
            }
            entry.value_ptr.deinit(self.allocator);
        }
        self.dependencies.deinit();
    }
    
    pub fn markDirty(self: *IncrementalChecker, name: []const u8) !void {
        const key = try self.allocator.dupe(u8, name);
        try self.dirty.put(key, {});
        
        // Propagate to dependencies
        if (self.dependencies.get(name)) |deps| {
            for (deps.items) |dep| {
                try self.markDirty(dep);
            }
        }
    }
    
    pub fn isDirty(self: *IncrementalChecker, name: []const u8) bool {
        return self.dirty.contains(name);
    }
    
    pub fn clearDirty(self: *IncrementalChecker) void {
        var it = self.dirty.keyIterator();
        while (it.next()) |key| {
            self.allocator.free(key.*);
        }
        self.dirty.clearRetainingCapacity();
    }
    
    pub fn addDependency(self: *IncrementalChecker, from: []const u8, to: []const u8) !void {
        const key = try self.allocator.dupe(u8, from);
        const value = try self.allocator.dupe(u8, to);
        
        if (self.dependencies.getPtr(key)) |deps| {
            try deps.append(self.allocator, value);
        } else {
            var deps = ArrayList([]const u8){};
            try deps.append(self.allocator, value);
            try self.dependencies.put(key, deps);
        }
    }
};

// ============================================================================
// Performance Metrics
// ============================================================================

pub const PerformanceMetrics = struct {
    protocol_checks: usize,
    impl_validations: usize,
    constraint_resolutions: usize,
    cache_hits: usize,
    cache_misses: usize,
    total_time_ns: u64,
    
    pub fn init() PerformanceMetrics {
        return PerformanceMetrics{
            .protocol_checks = 0,
            .impl_validations = 0,
            .constraint_resolutions = 0,
            .cache_hits = 0,
            .cache_misses = 0,
            .total_time_ns = 0,
        };
    }
    
    pub fn report(self: *PerformanceMetrics, allocator: Allocator) ![]const u8 {
        const total_ops = self.protocol_checks + self.impl_validations + self.constraint_resolutions;
        const cache_total = self.cache_hits + self.cache_misses;
        const hit_rate = if (cache_total > 0)
            @as(f64, @floatFromInt(self.cache_hits)) / @as(f64, @floatFromInt(cache_total)) * 100.0
        else
            0.0;
        
        return try std.fmt.allocPrint(allocator,
            \\Performance Metrics:
            \\  Operations: {}
            \\    Protocol checks: {}
            \\    Impl validations: {}
            \\    Constraint resolutions: {}
            \\  Cache:
            \\    Hits: {}
            \\    Misses: {}
            \\    Hit rate: {d:.1}%
            \\  Time: {d:.2}ms
        , .{
            total_ops,
            self.protocol_checks,
            self.impl_validations,
            self.constraint_resolutions,
            self.cache_hits,
            self.cache_misses,
            hit_rate,
            @as(f64, @floatFromInt(self.total_time_ns)) / 1_000_000.0,
        });
    }
};

// ============================================================================
// Benchmark Suite
// ============================================================================

pub const BenchmarkSuite = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) BenchmarkSuite {
        return BenchmarkSuite{ .allocator = allocator };
    }
    
    pub fn benchmarkCachePerformance(self: *BenchmarkSuite) !u64 {
        var cache = TypeCache.init(self.allocator);
        defer cache.deinit();
        
        const start = std.time.nanoTimestamp();
        
        // Warm up cache
        var i: usize = 0;
        while (i < 1000) : (i += 1) {
            const name = try std.fmt.allocPrint(self.allocator, "Protocol{}", .{i});
            defer self.allocator.free(name);
            try cache.cacheProtocol(name, true, 0);
        }
        
        // Benchmark lookups
        i = 0;
        while (i < 10000) : (i += 1) {
            const name = try std.fmt.allocPrint(self.allocator, "Protocol{}", .{i % 1000});
            defer self.allocator.free(name);
            _ = cache.getProtocol(name);
        }
        
        return @as(u64, @intCast(std.time.nanoTimestamp() - start));
    }
    
    pub fn benchmarkMemoryPool(self: *BenchmarkSuite) !u64 {
        var pool = MemoryPool.init(self.allocator);
        defer pool.deinit();
        
        const start = std.time.nanoTimestamp();
        
        var blocks = ArrayList([]u8){};
        defer blocks.deinit(self.allocator);
        
        // Allocate
        var i: usize = 0;
        while (i < 1000) : (i += 1) {
            const block = try pool.allocate(64);
            try blocks.append(self.allocator, block);
        }
        
        // Free
        for (blocks.items) |block| {
            try pool.free(block);
        }
        
        return @as(u64, @intCast(std.time.nanoTimestamp() - start));
    }
    
    pub fn benchmarkIncrementalChecking(self: *BenchmarkSuite) !u64 {
        var checker = IncrementalChecker.init(self.allocator);
        defer checker.deinit();
        
        const start = std.time.nanoTimestamp();
        
        // Build dependency graph
        var i: usize = 0;
        while (i < 100) : (i += 1) {
            const from = try std.fmt.allocPrint(self.allocator, "Type{}", .{i});
            defer self.allocator.free(from);
            const to = try std.fmt.allocPrint(self.allocator, "Type{}", .{i + 1});
            defer self.allocator.free(to);
            try checker.addDependency(from, to);
        }
        
        // Mark dirty and propagate
        try checker.markDirty("Type0");
        
        return @as(u64, @intCast(std.time.nanoTimestamp() - start));
    }
    
    pub fn runAll(self: *BenchmarkSuite, allocator: Allocator) ![]const u8 {
        const cache_time = try self.benchmarkCachePerformance();
        const pool_time = try self.benchmarkMemoryPool();
        const incremental_time = try self.benchmarkIncrementalChecking();
        
        return try std.fmt.allocPrint(allocator,
            \\Benchmark Results:
            \\  Cache Performance: {d:.2}ms
            \\  Memory Pool: {d:.2}ms
            \\  Incremental Checking: {d:.2}ms
        , .{
            @as(f64, @floatFromInt(cache_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(pool_time)) / 1_000_000.0,
            @as(f64, @floatFromInt(incremental_time)) / 1_000_000.0,
        });
    }
};

// ============================================================================
// Tests
// ============================================================================

test "type cache basic operations" {
    const allocator = std.testing.allocator;
    
    var cache = TypeCache.init(allocator);
    defer cache.deinit();
    
    try cache.cacheProtocol("Drawable", true, 0);
    
    const cached = cache.getProtocol("Drawable");
    try std.testing.expect(cached != null);
    try std.testing.expect(cached.?.valid);
}

test "type cache hit rate" {
    const allocator = std.testing.allocator;
    
    var cache = TypeCache.init(allocator);
    defer cache.deinit();
    
    try cache.cacheProtocol("Test", true, 0);
    
    _ = cache.getProtocol("Test"); // hit
    _ = cache.getProtocol("Missing"); // miss
    
    const hit_rate = cache.getHitRate();
    try std.testing.expect(hit_rate > 0.0);
    try std.testing.expect(hit_rate < 1.0);
}

test "impl cache" {
    const allocator = std.testing.allocator;
    
    var cache = TypeCache.init(allocator);
    defer cache.deinit();
    
    try cache.cacheImpl("Drawable", "Circle", true);
    
    const cached = try cache.getImpl("Drawable", "Circle");
    try std.testing.expect(cached != null);
    try std.testing.expect(cached.?);
}

test "memory pool allocation" {
    const allocator = std.testing.allocator;
    
    var pool = MemoryPool.init(allocator);
    defer pool.deinit();
    
    const block = try pool.allocate(64);
    try std.testing.expectEqual(@as(usize, 64), block.len);
    
    try pool.free(block);
}

test "incremental checker dirty tracking" {
    const allocator = std.testing.allocator;
    
    var checker = IncrementalChecker.init(allocator);
    defer checker.deinit();
    
    try checker.markDirty("Protocol1");
    
    try std.testing.expect(checker.isDirty("Protocol1"));
    try std.testing.expect(!checker.isDirty("Protocol2"));
}

test "incremental checker dependencies" {
    const allocator = std.testing.allocator;
    
    var checker = IncrementalChecker.init(allocator);
    defer checker.deinit();
    
    try checker.addDependency("TypeA", "TypeB");
    
    const deps = checker.dependencies.get("TypeA");
    try std.testing.expect(deps != null);
}

test "performance metrics" {
    var metrics = PerformanceMetrics.init();
    
    metrics.protocol_checks = 100;
    metrics.impl_validations = 50;
    metrics.cache_hits = 75;
    metrics.cache_misses = 25;
    
    try std.testing.expectEqual(@as(usize, 150), metrics.protocol_checks + metrics.impl_validations);
}

test "benchmark cache performance" {
    const allocator = std.testing.allocator;
    
    var suite = BenchmarkSuite.init(allocator);
    const time = try suite.benchmarkCachePerformance();
    
    try std.testing.expect(time > 0);
}

test "benchmark memory pool" {
    const allocator = std.testing.allocator;
    
    var suite = BenchmarkSuite.init(allocator);
    const time = try suite.benchmarkMemoryPool();
    
    try std.testing.expect(time > 0);
}

test "cache clear" {
    const allocator = std.testing.allocator;
    
    var cache = TypeCache.init(allocator);
    defer cache.deinit();
    
    try cache.cacheProtocol("Test", true, 0);
    cache.clear();
    
    const cached = cache.getProtocol("Test");
    try std.testing.expect(cached == null);
}
