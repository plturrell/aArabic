// ============================================================================
// HyperShimmy Performance Optimization Module
// ============================================================================
// Performance monitoring, profiling, and optimization utilities
// Day 52: Performance Optimization
// ============================================================================

const std = @import("std");
const mem = std.mem;
const time = std.time;

// ============================================================================
// Performance Metrics
// ============================================================================

/// Performance metric for tracking operation timing
pub const Metric = struct {
    name: []const u8,
    start_time: i128,
    end_time: i128,
    duration_ns: u64,
    memory_used: usize,
    
    pub fn duration_ms(self: Metric) f64 {
        return @as(f64, @floatFromInt(self.duration_ns)) / 1_000_000.0;
    }
    
    pub fn duration_us(self: Metric) f64 {
        return @as(f64, @floatFromInt(self.duration_ns)) / 1_000.0;
    }
};

/// Performance tracker for monitoring operations
pub const PerformanceTracker = struct {
    allocator: mem.Allocator,
    metrics: std.ArrayList(Metric),
    enable_tracking: bool = true,
    
    pub fn init(allocator: mem.Allocator) PerformanceTracker {
        return .{
            .allocator = allocator,
            .metrics = std.ArrayList(Metric){},
            .enable_tracking = true,
        };
    }
    
    pub fn deinit(self: *PerformanceTracker) void {
        for (self.metrics.items) |metric| {
            self.allocator.free(metric.name);
        }
        self.metrics.deinit(self.allocator);
    }
    
    /// Start tracking an operation
    pub fn startOperation(self: *PerformanceTracker, name: []const u8) !usize {
        if (!self.enable_tracking) return 0;
        
        const metric = Metric{
            .name = try self.allocator.dupe(u8, name),
            .start_time = time.nanoTimestamp(),
            .end_time = 0,
            .duration_ns = 0,
            .memory_used = 0,
        };
        
        try self.metrics.append(self.allocator, metric);
        return self.metrics.items.len - 1;
    }
    
    /// End tracking an operation
    pub fn endOperation(self: *PerformanceTracker, index: usize) void {
        if (!self.enable_tracking or index >= self.metrics.items.len) return;
        
        self.metrics.items[index].end_time = time.nanoTimestamp();
        self.metrics.items[index].duration_ns = @intCast(
            self.metrics.items[index].end_time - self.metrics.items[index].start_time
        );
    }
    
    /// Get average duration for operations with the same name
    pub fn getAverageDuration(self: *PerformanceTracker, name: []const u8) ?f64 {
        var total_ns: u64 = 0;
        var count: usize = 0;
        
        for (self.metrics.items) |metric| {
            if (mem.eql(u8, metric.name, name)) {
                total_ns += metric.duration_ns;
                count += 1;
            }
        }
        
        if (count == 0) return null;
        return @as(f64, @floatFromInt(total_ns)) / @as(f64, @floatFromInt(count)) / 1_000_000.0;
    }
    
    /// Get metrics summary as JSON
    pub fn toJson(self: *PerformanceTracker) ![]const u8 {
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        
        try buffer.appendSlice("{\"metrics\":[");
        
        for (self.metrics.items, 0..) |metric, i| {
            if (i > 0) try buffer.appendSlice(",");
            
            try std.fmt.format(buffer.writer(),
                \\{{"name":"{s}","duration_ms":{d:.2},"memory_bytes":{d}}}
            , .{
                metric.name,
                metric.duration_ms(),
                metric.memory_used,
            });
        }
        
        try buffer.appendSlice("]}");
        return try buffer.toOwnedSlice();
    }
    
    /// Clear all metrics
    pub fn clear(self: *PerformanceTracker) void {
        for (self.metrics.items) |metric| {
            self.allocator.free(metric.name);
        }
        self.metrics.clearRetainingCapacity();
    }
};

// ============================================================================
// Memory Pool for Reduced Allocations
// ============================================================================

/// Simple memory pool for frequent small allocations
pub const MemoryPool = struct {
    allocator: mem.Allocator,
    block_size: usize,
    blocks: std.ArrayList([]u8),
    current_block: usize = 0,
    current_offset: usize = 0,
    
    pub fn init(allocator: mem.Allocator, block_size: usize) MemoryPool {
        return .{
            .allocator = allocator,
            .block_size = block_size,
            .blocks = std.ArrayList([]u8){},
        };
    }
    
    pub fn deinit(self: *MemoryPool) void {
        for (self.blocks.items) |block| {
            self.allocator.free(block);
        }
        self.blocks.deinit(self.allocator);
    }
    
    /// Allocate memory from the pool
    pub fn alloc(self: *MemoryPool, size: usize) ![]u8 {
        if (size > self.block_size) {
            // Large allocation, bypass pool
            return try self.allocator.alloc(u8, size);
        }
        
        // Check if current block has space
        if (self.blocks.items.len == 0 or 
            self.current_offset + size > self.block_size) {
            // Allocate new block
            const new_block = try self.allocator.alloc(u8, self.block_size);
            try self.blocks.append(self.allocator, new_block);
            self.current_block = self.blocks.items.len - 1;
            self.current_offset = 0;
        }
        
        const block = self.blocks.items[self.current_block];
        const result = block[self.current_offset..self.current_offset + size];
        self.current_offset += size;
        
        return result;
    }
    
    /// Reset the pool (doesn't free memory, just resets pointers)
    pub fn reset(self: *MemoryPool) void {
        self.current_block = 0;
        self.current_offset = 0;
    }
};

// ============================================================================
// String Interning for Memory Efficiency
// ============================================================================

/// String interner to reduce memory usage for repeated strings
pub const StringInterner = struct {
    allocator: mem.Allocator,
    strings: std.StringHashMap([]const u8),
    
    pub fn init(allocator: mem.Allocator) StringInterner {
        return .{
            .allocator = allocator,
            .strings = std.StringHashMap([]const u8).init(allocator),
        };
    }
    
    pub fn deinit(self: *StringInterner) void {
        var it = self.strings.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.value_ptr.*);
        }
        self.strings.deinit();
    }
    
    /// Intern a string (returns existing copy if already interned)
    pub fn intern(self: *StringInterner, s: []const u8) ![]const u8 {
        if (self.strings.get(s)) |interned| {
            return interned;
        }
        
        const copy = try self.allocator.dupe(u8, s);
        try self.strings.put(copy, copy);
        return copy;
    }
    
    /// Get count of unique strings
    pub fn count(self: *StringInterner) usize {
        return self.strings.count();
    }
};

// ============================================================================
// Cache for Frequently Accessed Data
// ============================================================================

/// Simple LRU cache implementation
pub fn Cache(comptime K: type, comptime V: type, comptime max_size: usize) type {
    return struct {
        const Self = @This();
        const Entry = struct {
            key: K,
            value: V,
            access_count: usize,
        };
        
        allocator: mem.Allocator,
        entries: std.ArrayList(Entry),
        
        pub fn init(allocator: mem.Allocator) Self {
            return .{
                .allocator = allocator,
                .entries = std.ArrayList(Entry){},
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.entries.deinit(self.allocator);
        }
        
        /// Get value from cache
        pub fn get(self: *Self, key: K) ?V {
            for (self.entries.items, 0..) |*entry, i| {
                if (std.meta.eql(entry.key, key)) {
                    entry.access_count += 1;
                    // Move to front (MRU)
                    if (i > 0) {
                        const temp = entry.*;
                        _ = self.entries.orderedRemove(i);
                        self.entries.insert(self.allocator, 0, temp) catch {};
                    }
                    return entry.value;
                }
            }
            return null;
        }
        
        /// Put value into cache
        pub fn put(self: *Self, key: K, value: V) !void {
            // Check if key exists
            for (self.entries.items, 0..) |*entry, i| {
                if (std.meta.eql(entry.key, key)) {
                    entry.value = value;
                    entry.access_count += 1;
                    if (i > 0) {
                        const temp = entry.*;
                        _ = self.entries.orderedRemove(i);
                        try self.entries.insert(self.allocator, 0, temp);
                    }
                    return;
                }
            }
            
            // Add new entry
            if (self.entries.items.len >= max_size) {
                // Remove least recently used (last item)
                _ = self.entries.pop();
            }
            
            try self.entries.insert(self.allocator, 0, Entry{
                .key = key,
                .value = value,
                .access_count = 1,
            });
        }
        
        /// Clear cache
        pub fn clear(self: *Self) void {
            self.entries.clearRetainingCapacity();
        }
        
        /// Get cache hit rate
        pub fn hitRate(self: *Self) f64 {
            if (self.entries.items.len == 0) return 0.0;
            
            var total_accesses: usize = 0;
            for (self.entries.items) |entry| {
                total_accesses += entry.access_count;
            }
            
            if (total_accesses == 0) return 0.0;
            return @as(f64, @floatFromInt(total_accesses - self.entries.items.len)) / 
                   @as(f64, @floatFromInt(total_accesses));
        }
    };
}

// ============================================================================
// Batch Processing for Efficiency
// ============================================================================

/// Batch processor for efficient bulk operations
pub fn BatchProcessor(comptime T: type) type {
    return struct {
        const Self = @This();
        
        allocator: mem.Allocator,
        items: std.ArrayList(T),
        batch_size: usize,
        process_fn: *const fn([]const T) anyerror!void,
        
        pub fn init(
            allocator: mem.Allocator,
            batch_size: usize,
            process_fn: *const fn([]const T) anyerror!void,
        ) Self {
            return .{
                .allocator = allocator,
                .items = std.ArrayList(T){},
                .batch_size = batch_size,
                .process_fn = process_fn,
            };
        }
        
        pub fn deinit(self: *Self) void {
            self.items.deinit(self.allocator);
        }
        
        /// Add item to batch
        pub fn add(self: *Self, item: T) !void {
            try self.items.append(self.allocator, item);
            
            if (self.items.items.len >= self.batch_size) {
                try self.flush();
            }
        }
        
        /// Flush remaining items
        pub fn flush(self: *Self) !void {
            if (self.items.items.len == 0) return;
            
            try self.process_fn(self.items.items);
            self.items.clearRetainingCapacity();
        }
    };
}

// ============================================================================
// Performance Utilities
// ============================================================================

/// Measure execution time of a function
pub fn measureTime(comptime func: anytype, args: anytype) !struct { result: @TypeOf(@call(.auto, func, args)), duration_ns: u64 } {
    const start = time.nanoTimestamp();
    const result = try @call(.auto, func, args);
    const end = time.nanoTimestamp();
    
    return .{
        .result = result,
        .duration_ns = @intCast(end - start),
    };
}

/// Format bytes to human-readable string
pub fn formatBytes(allocator: mem.Allocator, bytes: usize) ![]const u8 {
    const units = [_][]const u8{ "B", "KB", "MB", "GB", "TB" };
    var size = @as(f64, @floatFromInt(bytes));
    var unit_index: usize = 0;
    
    while (size >= 1024.0 and unit_index < units.len - 1) {
        size /= 1024.0;
        unit_index += 1;
    }
    
    return try std.fmt.allocPrint(allocator, "{d:.2} {s}", .{ size, units[unit_index] });
}

// ============================================================================
// Tests
// ============================================================================

test "performance tracker" {
    const allocator = std.testing.allocator;
    var tracker = PerformanceTracker.init(allocator);
    defer tracker.deinit();
    
    const idx = try tracker.startOperation("test_op");
    // Small delay to ensure measurable duration
    var i: usize = 0;
    while (i < 1000000) : (i += 1) {}
    tracker.endOperation(idx);
    
    try std.testing.expect(tracker.metrics.items.len == 1);
    try std.testing.expect(tracker.metrics.items[0].duration_ns > 0);
}

test "memory pool" {
    const allocator = std.testing.allocator;
    var pool = MemoryPool.init(allocator, 1024);
    defer pool.deinit();
    
    const mem1 = try pool.alloc(100);
    const mem2 = try pool.alloc(100);
    
    try std.testing.expect(mem1.len == 100);
    try std.testing.expect(mem2.len == 100);
    try std.testing.expect(pool.blocks.items.len == 1);
}

test "string interner" {
    const allocator = std.testing.allocator;
    var interner = StringInterner.init(allocator);
    defer interner.deinit();
    
    const s1 = try interner.intern("test");
    const s2 = try interner.intern("test");
    
    try std.testing.expect(s1.ptr == s2.ptr); // Same pointer
    try std.testing.expectEqual(@as(usize, 1), interner.count());
}

test "cache" {
    const allocator = std.testing.allocator;
    var cache = Cache(i32, i32, 10).init(allocator);
    defer cache.deinit();
    
    try cache.put(1, 42);
    
    const val1 = cache.get(1);
    const val2 = cache.get(2);
    
    try std.testing.expectEqual(@as(?i32, 42), val1);
    try std.testing.expectEqual(@as(?i32, null), val2);
    
    // Test cache has item
    try std.testing.expect(cache.entries.items.len == 1);
}

test "format bytes" {
    const allocator = std.testing.allocator;
    
    const str1 = try formatBytes(allocator, 1024);
    defer allocator.free(str1);
    try std.testing.expect(mem.indexOf(u8, str1, "KB") != null);
    
    const str2 = try formatBytes(allocator, 1024 * 1024);
    defer allocator.free(str2);
    try std.testing.expect(mem.indexOf(u8, str2, "MB") != null);
}
