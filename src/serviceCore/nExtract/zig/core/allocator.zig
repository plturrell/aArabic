//! Memory Management Infrastructure for nExtract
//!
//! This module provides:
//! - Arena allocator with configurable block size
//! - Object pool with type-safe reuse
//! - Memory usage tracking
//! - Leak detection (debug builds)
//! - Integration with Zig's standard allocator interface
//!
//! Author: nExtract Team
//! Date: January 17, 2026

const std = @import("std");
const builtin = @import("builtin");
const Allocator = std.mem.Allocator;

// ============================================================================
// Arena Allocator
// ============================================================================

/// Arena allocator for document processing.
/// Allocates memory in large blocks and frees all at once.
/// Perfect for per-document allocations that are freed together.
pub const ArenaAllocator = struct {
    backing_allocator: Allocator,
    current_block: ?*Block,
    first_block: ?*Block,
    block_size: usize,
    total_allocated: usize,
    total_freed: usize,
    
    /// Memory block in the arena
    const Block = struct {
        data: []u8,
        used: usize,
        next: ?*Block,
    };
    
    /// Default block size (1MB)
    pub const default_block_size = 1024 * 1024;
    
    /// Create a new arena allocator
    pub fn init(backing: Allocator) ArenaAllocator {
        return initWithSize(backing, default_block_size);
    }
    
    /// Create arena with custom block size
    pub fn initWithSize(backing: Allocator, block_size: usize) ArenaAllocator {
        return ArenaAllocator{
            .backing_allocator = backing,
            .current_block = null,
            .first_block = null,
            .block_size = block_size,
            .total_allocated = 0,
            .total_freed = 0,
        };
    }
    
    /// Get an allocator interface
    pub fn allocator(self: *ArenaAllocator) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    /// Allocate memory from the arena
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        _ = ret_addr;
        const self: *ArenaAllocator = @ptrCast(@alignCast(ctx));
        
        const alignment = @as(usize, 1) << @as(u6, @intCast(ptr_align));
        
        // Find or create a block with enough space
        var block = self.current_block;
        while (block) |b| {
            const available = b.data.len - b.used;
            const aligned_offset = std.mem.alignForward(usize, b.used, alignment);
            const padding = aligned_offset - b.used;
            
            if (available >= len + padding) {
                b.used = aligned_offset;
                const result = b.data.ptr + b.used;
                b.used += len;
                self.total_allocated += len;
                return result;
            }
            
            block = b.next;
        }
        
        // Need a new block
        const new_block_size = @max(self.block_size, len + alignment);
        const new_block = self.createBlock(new_block_size) catch return null;
        
        // Add to chain
        if (self.current_block) |current| {
            current.next = new_block;
        } else {
            self.first_block = new_block;
        }
        self.current_block = new_block;
        
        // Allocate from new block
        const aligned_offset = std.mem.alignForward(usize, 0, alignment);
        new_block.used = aligned_offset + len;
        self.total_allocated += len;
        return new_block.data.ptr + aligned_offset;
    }
    
    /// Resize allocation (not supported for arena)
    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = new_len;
        _ = ret_addr;
        return false; // Arena doesn't support resize
    }
    
    /// Free memory (no-op for arena, use deinit to free all)
    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        _ = ctx;
        _ = buf;
        _ = buf_align;
        _ = ret_addr;
        // No-op: arena frees all memory at once
    }
    
    /// Create a new memory block
    fn createBlock(self: *ArenaAllocator, size: usize) !*Block {
        const block_mem = try self.backing_allocator.create(Block);
        errdefer self.backing_allocator.destroy(block_mem);
        
        const data = try self.backing_allocator.alloc(u8, size);
        
        block_mem.* = Block{
            .data = data,
            .used = 0,
            .next = null,
        };
        
        return block_mem;
    }
    
    /// Free all memory in the arena
    pub fn deinit(self: *ArenaAllocator) void {
        var block = self.first_block;
        while (block) |b| {
            const next = b.next;
            self.backing_allocator.free(b.data);
            self.backing_allocator.destroy(b);
            block = next;
        }
        
        self.current_block = null;
        self.first_block = null;
        self.total_freed = self.total_allocated;
    }
    
    /// Reset arena for reuse (keeps first block)
    pub fn reset(self: *ArenaAllocator) void {
        // Free all blocks except the first
        if (self.first_block) |first| {
            var block = first.next;
            while (block) |b| {
                const next = b.next;
                self.backing_allocator.free(b.data);
                self.backing_allocator.destroy(b);
                block = next;
            }
            
            // Reset first block
            first.used = 0;
            first.next = null;
            self.current_block = first;
        }
        
        self.total_freed = self.total_allocated;
        self.total_allocated = 0;
    }
    
    /// Get memory usage statistics
    pub fn getStats(self: *const ArenaAllocator) MemoryStats {
        var total_capacity: usize = 0;
        var total_used: usize = 0;
        var block_count: usize = 0;
        
        var block = self.first_block;
        while (block) |b| {
            total_capacity += b.data.len;
            total_used += b.used;
            block_count += 1;
            block = b.next;
        }
        
        return MemoryStats{
            .allocated = self.total_allocated,
            .freed = self.total_freed,
            .capacity = total_capacity,
            .used = total_used,
            .block_count = block_count,
        };
    }
};

// ============================================================================
// Object Pool
// ============================================================================

/// Object pool for frequently allocated objects.
/// Reuses memory to reduce allocation overhead.
pub fn ObjectPool(comptime T: type) type {
    return struct {
        const Self = @This();
        
        allocator: Allocator,
        pool: std.ArrayList(?*T),
        active_count: usize,
        total_created: usize,
        total_reused: usize,
        
        /// Initialize object pool
        pub fn init(allocator: Allocator) Self {
            return Self{
                .allocator = allocator,
                .pool = std.ArrayList(?*T).init(allocator),
                .active_count = 0,
                .total_created = 0,
                .total_reused = 0,
            };
        }
        
        /// Acquire an object from the pool
        pub fn acquire(self: *Self) !*T {
            // Try to reuse from pool
            while (self.pool.items.len > 0) {
                if (self.pool.pop()) |obj| {
                    if (obj) |o| {
                        self.active_count += 1;
                        self.total_reused += 1;
                        return o;
                    }
                }
            }
            
            // Create new object
            const obj = try self.allocator.create(T);
            self.active_count += 1;
            self.total_created += 1;
            return obj;
        }
        
        /// Release object back to pool
        pub fn release(self: *Self, obj: *T) !void {
            // Reset object to default state if it has a reset method
            if (@hasDecl(T, "reset")) {
                obj.reset();
            }
            
            self.active_count -= 1;
            try self.pool.append(obj);
        }
        
        /// Shrink pool to size (free excess objects)
        pub fn shrink(self: *Self, target_size: usize) void {
            while (self.pool.items.len > target_size) {
                if (self.pool.pop()) |obj| {
                    if (obj) |o| {
                        self.allocator.destroy(o);
                    }
                }
            }
        }
        
        /// Free all pooled objects
        pub fn deinit(self: *Self) void {
            for (self.pool.items) |obj| {
                if (obj) |o| {
                    self.allocator.destroy(o);
                }
            }
            self.pool.deinit();
        }
        
        /// Get pool statistics
        pub fn getStats(self: *const Self) PoolStats {
            return PoolStats{
                .pooled_count = self.pool.items.len,
                .active_count = self.active_count,
                .total_created = self.total_created,
                .total_reused = self.total_reused,
            };
        }
    };
}

// ============================================================================
// Memory Statistics
// ============================================================================

pub const MemoryStats = struct {
    allocated: usize,
    freed: usize,
    capacity: usize,
    used: usize,
    block_count: usize,
    
    pub fn format(
        self: MemoryStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        try writer.print(
            "MemoryStats{{ allocated={}, freed={}, capacity={}, used={}, blocks={} }}",
            .{ self.allocated, self.freed, self.capacity, self.used, self.block_count },
        );
    }
};

pub const PoolStats = struct {
    pooled_count: usize,
    active_count: usize,
    total_created: usize,
    total_reused: usize,
    
    pub fn format(
        self: PoolStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        try writer.print(
            "PoolStats{{ pooled={}, active={}, created={}, reused={} }}",
            .{ self.pooled_count, self.active_count, self.total_created, self.total_reused },
        );
    }
};

// ============================================================================
// Leak Detection (Debug Builds)
// ============================================================================

/// Leak-detecting allocator wrapper (debug only)
pub const LeakDetector = struct {
    backing_allocator: Allocator,
    allocations: if (builtin.mode == .Debug) std.AutoHashMap(usize, AllocationInfo) else void,
    total_allocated: usize,
    total_freed: usize,
    peak_memory: usize,
    current_memory: usize,
    
    const AllocationInfo = struct {
        size: usize,
        alignment: u8,
        return_address: usize,
    };
    
    pub fn init(backing: Allocator) LeakDetector {
        if (builtin.mode == .Debug) {
            return LeakDetector{
                .backing_allocator = backing,
                .allocations = std.AutoHashMap(usize, AllocationInfo).init(backing),
                .total_allocated = 0,
                .total_freed = 0,
                .peak_memory = 0,
                .current_memory = 0,
            };
        } else {
            return LeakDetector{
                .backing_allocator = backing,
                .allocations = {},
                .total_allocated = 0,
                .total_freed = 0,
                .peak_memory = 0,
                .current_memory = 0,
            };
        }
    }
    
    pub fn allocator(self: *LeakDetector) Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .free = free,
            },
        };
    }
    
    fn alloc(ctx: *anyopaque, len: usize, ptr_align: u8, ret_addr: usize) ?[*]u8 {
        const self: *LeakDetector = @ptrCast(@alignCast(ctx));
        
        const result = self.backing_allocator.rawAlloc(len, ptr_align, ret_addr) orelse return null;
        
        self.total_allocated += len;
        self.current_memory += len;
        if (self.current_memory > self.peak_memory) {
            self.peak_memory = self.current_memory;
        }
        
        if (builtin.mode == .Debug) {
            const addr = @intFromPtr(result);
            self.allocations.put(addr, .{
                .size = len,
                .alignment = ptr_align,
                .return_address = ret_addr,
            }) catch {};
        }
        
        return result;
    }
    
    fn resize(ctx: *anyopaque, buf: []u8, buf_align: u8, new_len: usize, ret_addr: usize) bool {
        const self: *LeakDetector = @ptrCast(@alignCast(ctx));
        
        if (builtin.mode == .Debug) {
            const addr = @intFromPtr(buf.ptr);
            if (self.allocations.get(addr)) |info| {
                const old_size = info.size;
                
                if (self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr)) {
                    self.current_memory = self.current_memory - old_size + new_len;
                    if (self.current_memory > self.peak_memory) {
                        self.peak_memory = self.current_memory;
                    }
                    
                    self.allocations.put(addr, .{
                        .size = new_len,
                        .alignment = buf_align,
                        .return_address = ret_addr,
                    }) catch {};
                    
                    return true;
                }
                return false;
            }
        }
        
        return self.backing_allocator.rawResize(buf, buf_align, new_len, ret_addr);
    }
    
    fn free(ctx: *anyopaque, buf: []u8, buf_align: u8, ret_addr: usize) void {
        const self: *LeakDetector = @ptrCast(@alignCast(ctx));
        
        if (builtin.mode == .Debug) {
            const addr = @intFromPtr(buf.ptr);
            if (self.allocations.fetchRemove(addr)) |entry| {
                self.total_freed += entry.value.size;
                self.current_memory -= entry.value.size;
            }
        } else {
            self.total_freed += buf.len;
            self.current_memory -= buf.len;
        }
        
        self.backing_allocator.rawFree(buf, buf_align, ret_addr);
    }
    
    pub fn deinit(self: *LeakDetector) void {
        if (builtin.mode == .Debug) {
            if (self.allocations.count() > 0) {
                std.debug.print("\n=== MEMORY LEAKS DETECTED ===\n", .{});
                std.debug.print("Total leaks: {}\n", .{self.allocations.count()});
                
                var iter = self.allocations.iterator();
                var i: usize = 0;
                while (iter.next()) |entry| : (i += 1) {
                    if (i < 10) { // Print first 10 leaks
                        std.debug.print("  Leak #{}: addr=0x{x}, size={}, alignment={}\n", .{
                            i + 1,
                            entry.key_ptr.*,
                            entry.value_ptr.size,
                            entry.value_ptr.alignment,
                        });
                    }
                }
                
                if (self.allocations.count() > 10) {
                    std.debug.print("  ... and {} more\n", .{self.allocations.count() - 10});
                }
            }
            
            self.allocations.deinit();
        }
    }
    
    pub fn getStats(self: *const LeakDetector) LeakStats {
        return LeakStats{
            .allocated = self.total_allocated,
            .freed = self.total_freed,
            .current = self.current_memory,
            .peak = self.peak_memory,
            .leak_count = if (builtin.mode == .Debug) self.allocations.count() else 0,
        };
    }
};

pub const LeakStats = struct {
    allocated: usize,
    freed: usize,
    current: usize,
    peak: usize,
    leak_count: usize,
    
    pub fn format(
        self: LeakStats,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = fmt;
        _ = options;
        
        try writer.print(
            "LeakStats{{ allocated={}, freed={}, current={}, peak={}, leaks={} }}",
            .{ self.allocated, self.freed, self.current, self.peak, self.leak_count },
        );
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ArenaAllocator: basic allocation" {
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    
    const alloc = arena.allocator();
    
    const bytes = try alloc.alloc(u8, 100);
    try std.testing.expectEqual(@as(usize, 100), bytes.len);
    
    const stats = arena.getStats();
    try std.testing.expect(stats.allocated >= 100);
    try std.testing.expectEqual(@as(usize, 1), stats.block_count);
}

test "ArenaAllocator: multiple allocations" {
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    
    const alloc = arena.allocator();
    
    var allocations: [10][]u8 = undefined;
    for (&allocations, 0..) |*a, i| {
        a.* = try alloc.alloc(u8, (i + 1) * 10);
    }
    
    const stats = arena.getStats();
    try std.testing.expect(stats.allocated >= 550); // Sum of 10+20+...+100
}

test "ArenaAllocator: reset" {
    var arena = ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    
    const alloc = arena.allocator();
    
    _ = try alloc.alloc(u8, 1000);
    const stats1 = arena.getStats();
    
    arena.reset();
    const stats2 = arena.getStats();
    
    try std.testing.expectEqual(@as(usize, 0), stats2.allocated);
    try std.testing.expectEqual(stats1.allocated, stats2.freed);
}

test "ObjectPool: basic usage" {
    const TestObj = struct {
        value: i32,
        
        pub fn reset(self: *@This()) void {
            self.value = 0;
        }
    };
    
    var pool = ObjectPool(TestObj).init(std.testing.allocator);
    defer pool.deinit();
    
    const obj1 = try pool.acquire();
    obj1.value = 42;
    
    try pool.release(obj1);
    
    const obj2 = try pool.acquire();
    try std.testing.expectEqual(@as(i32, 0), obj2.value); // Should be reset
    
    const stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 1), stats.total_created);
    try std.testing.expectEqual(@as(usize, 1), stats.total_reused);
}

test "ObjectPool: shrink" {
    const TestObj = struct {
        value: i32,
    };
    
    var pool = ObjectPool(TestObj).init(std.testing.allocator);
    defer pool.deinit();
    
    // Acquire and release 10 objects
    var objs: [10]*TestObj = undefined;
    for (&objs) |*obj| {
        obj.* = try pool.acquire();
    }
    for (objs) |obj| {
        try pool.release(obj);
    }
    
    var stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 10), stats.pooled_count);
    
    // Shrink to 5
    pool.shrink(5);
    stats = pool.getStats();
    try std.testing.expectEqual(@as(usize, 5), stats.pooled_count);
}

test "LeakDetector: no leaks" {
    var detector = LeakDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    const alloc = detector.allocator();
    
    const bytes = try alloc.alloc(u8, 100);
    alloc.free(bytes);
    
    const stats = detector.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.leak_count);
    try std.testing.expectEqual(stats.allocated, stats.freed);
}

test "LeakDetector: memory tracking" {
    var detector = LeakDetector.init(std.testing.allocator);
    defer detector.deinit();
    
    const alloc = detector.allocator();
    
    const bytes1 = try alloc.alloc(u8, 100);
    const bytes2 = try alloc.alloc(u8, 200);
    
    var stats = detector.getStats();
    try std.testing.expectEqual(@as(usize, 300), stats.current);
    try std.testing.expectEqual(@as(usize, 300), stats.peak);
    
    alloc.free(bytes1);
    stats = detector.getStats();
    try std.testing.expectEqual(@as(usize, 200), stats.current);
    
    alloc.free(bytes2);
    stats = detector.getStats();
    try std.testing.expectEqual(@as(usize, 0), stats.current);
}
