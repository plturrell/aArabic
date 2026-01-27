// Memory Allocation Tests - Phase 2 Testing
// Tests for malloc, free, calloc, realloc, reallocarray
// Verifies fixes for critical security bugs

const std = @import("std");
const testing = std.testing;
const libc = @import("zig-libc");
const memory = libc.stdlib.memory;

test "malloc basic allocation" {
    const ptr = memory.malloc(100);
    try testing.expect(ptr != null);
    
    // Write to memory to ensure it's valid
    const slice = @as([*]u8, @ptrCast(ptr.?))[0..100];
    @memset(slice, 0x42);
    
    // Verify write worked
    try testing.expectEqual(@as(u8, 0x42), slice[0]);
    try testing.expectEqual(@as(u8, 0x42), slice[99]);
    
    memory.free(ptr);
}

test "malloc zero size returns null" {
    const ptr = memory.malloc(0);
    try testing.expect(ptr == null);
}

test "free null pointer is safe" {
    memory.free(null);
    // Should not crash
}

test "calloc zeroes memory" {
    const ptr = memory.calloc(10, 10);
    try testing.expect(ptr != null);
    
    const slice = @as([*]u8, @ptrCast(ptr.?))[0..100];
    
    // Verify all bytes are zero
    for (slice) |byte| {
        try testing.expectEqual(@as(u8, 0), byte);
    }
    
    memory.free(ptr);
}

test "calloc overflow detection" {
    // This should fail due to overflow
    const max = std.math.maxInt(usize);
    const ptr = memory.calloc(max, 2);
    try testing.expect(ptr == null);
}

test "realloc preserves data - grow" {
    // Allocate small block
    const ptr1 = memory.malloc(10);
    try testing.expect(ptr1 != null);
    
    // Write pattern
    const slice1 = @as([*]u8, @ptrCast(ptr1.?))[0..10];
    for (slice1, 0..) |*byte, i| {
        byte.* = @intCast(i);
    }
    
    // Grow the allocation
    const ptr2 = memory.realloc(ptr1, 100);
    try testing.expect(ptr2 != null);
    
    // Verify data preserved
    const slice2 = @as([*]u8, @ptrCast(ptr2.?))[0..10];
    for (slice2, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i)), byte);
    }
    
    memory.free(ptr2);
}

test "realloc preserves data - shrink" {
    // Allocate large block
    const ptr1 = memory.malloc(100);
    try testing.expect(ptr1 != null);
    
    // Write pattern
    const slice1 = @as([*]u8, @ptrCast(ptr1.?))[0..100];
    for (slice1, 0..) |*byte, i| {
        byte.* = @intCast(i % 256);
    }
    
    // Shrink the allocation
    const ptr2 = memory.realloc(ptr1, 10);
    try testing.expect(ptr2 != null);
    
    // Verify first 10 bytes preserved
    const slice2 = @as([*]u8, @ptrCast(ptr2.?))[0..10];
    for (slice2, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i)), byte);
    }
    
    memory.free(ptr2);
}

test "realloc null pointer acts like malloc" {
    const ptr = memory.realloc(null, 50);
    try testing.expect(ptr != null);
    
    // Should work like malloc
    const slice = @as([*]u8, @ptrCast(ptr.?))[0..50];
    @memset(slice, 0xAB);
    try testing.expectEqual(@as(u8, 0xAB), slice[0]);
    
    memory.free(ptr);
}

test "realloc zero size acts like free" {
    const ptr1 = memory.malloc(50);
    try testing.expect(ptr1 != null);
    
    const ptr2 = memory.realloc(ptr1, 0);
    try testing.expect(ptr2 == null);
    // Memory should be freed
}

test "multiple alloc/free cycles" {
    // Test for memory leaks over multiple cycles
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        const ptr = memory.malloc(1000);
        try testing.expect(ptr != null);
        
        // Write to ensure valid
        const slice = @as([*]u8, @ptrCast(ptr.?))[0..1000];
        @memset(slice, 0xFF);
        
        memory.free(ptr);
    }
    // Should not leak memory
}

test "reallocarray basic" {
    const utilities = libc.stdlib.utilities;
    
    const ptr = utilities.reallocarray(null, 10, 10);
    try testing.expect(ptr != null);
    
    // Should allocate 100 bytes
    const slice = @as([*]u8, @ptrCast(ptr.?))[0..100];
    @memset(slice, 0xCC);
    try testing.expectEqual(@as(u8, 0xCC), slice[99]);
    
    memory.free(ptr);
}

test "reallocarray preserves array data" {
    const utilities = libc.stdlib.utilities;
    
    // Allocate array of 10 ints (40 bytes on 32-bit, 80 on 64-bit)
    const ptr1 = utilities.reallocarray(null, 10, @sizeOf(usize));
    try testing.expect(ptr1 != null);
    
    // Write array data
    const arr1 = @as([*]usize, @ptrCast(@alignCast(ptr1.?)))[0..10];
    for (arr1, 0..) |*elem, i| {
        elem.* = i * 100;
    }
    
    // Resize to 20 elements
    const ptr2 = utilities.reallocarray(ptr1, 20, @sizeOf(usize));
    try testing.expect(ptr2 != null);
    
    // Verify first 10 elements preserved
    const arr2 = @as([*]usize, @ptrCast(@alignCast(ptr2.?)))[0..10];
    for (arr2, 0..) |elem, i| {
        try testing.expectEqual(i * 100, elem);
    }
    
    memory.free(ptr2);
}

test "reallocarray overflow detection" {
    const utilities = libc.stdlib.utilities;
    
    const max = std.math.maxInt(usize);
    const ptr = utilities.reallocarray(null, max, 2);
    try testing.expect(ptr == null);
}

test "stress test - multiple allocations" {
    var ptrs: [50]*anyopaque = undefined;
    
    // Allocate many blocks
    for (&ptrs, 0..) |*ptr, i| {
        const size = (i + 1) * 100;
        ptr.* = memory.malloc(size) orelse unreachable;
        
        // Write pattern
        const slice = @as([*]u8, @ptrCast(ptr.*))[0..size];
        @memset(slice, @intCast(i));
    }
    
    // Verify all blocks
    for (ptrs, 0..) |ptr, i| {
        const size = (i + 1) * 100;
        const slice = @as([*]u8, @ptrCast(ptr))[0..size];
        try testing.expectEqual(@as(u8, @intCast(i)), slice[0]);
    }
    
    // Free all blocks
    for (ptrs) |ptr| {
        memory.free(ptr);
    }
}

test "realloc preserves exact data" {
    const ptr1 = memory.malloc(256);
    try testing.expect(ptr1 != null);
    
    // Write specific pattern
    const slice1 = @as([*]u8, @ptrCast(ptr1.?))[0..256];
    for (slice1, 0..) |*byte, i| {
        byte.* = @intCast(i);
    }
    
    // Realloc to larger size
    const ptr2 = memory.realloc(ptr1, 512);
    try testing.expect(ptr2 != null);
    
    // Verify ALL 256 bytes preserved
    const slice2 = @as([*]u8, @ptrCast(ptr2.?))[0..256];
    for (slice2, 0..) |byte, i| {
        try testing.expectEqual(@as(u8, @intCast(i)), byte);
    }
    
    memory.free(ptr2);
}

test "malloc_usable_size returns tracked size" {
    const ptr = memory.malloc(64);
    try testing.expect(ptr != null);
    const usable = memory.malloc_usable_size(ptr);
    try testing.expectEqual(@as(usize, 64), usable);
    memory.free(ptr);
}

test "memalign and valloc return aligned blocks" {
    const p1 = memory.memalign(32, 64);
    try testing.expect(p1 != null);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(p1.?) % 32);
    memory.free(p1);

    const p2 = memory.valloc(128);
    try testing.expect(p2 != null);
    try testing.expectEqual(@as(usize, 0), @intFromPtr(p2.?) % std.heap.page_size_min);
    memory.free(p2);
}

test "reallocf frees on failure" {
    // Force failure by requesting huge size (may still succeed, so not guaranteed)
    const ptr = memory.malloc(16);
    try testing.expect(ptr != null);
    const new_ptr = memory.reallocf(ptr, std.math.maxInt(usize));
    try testing.expect(new_ptr == null);
}

test "random/srandom deterministic" {
    libc.stdlib.srandom(1);
    const a1 = libc.stdlib.random();
    const a2 = libc.stdlib.random();

    libc.stdlib.srandom(1);
    const b1 = libc.stdlib.random();
    const b2 = libc.stdlib.random();

    try testing.expectEqual(a1, b1);
    try testing.expectEqual(a2, b2);
}

test "lsearch finds and appends" {
    var buf: [4]usize = [_]usize{ 10, 20, 0, 0 };
    var nel: usize = 2;
    const cmp = struct {
        pub fn cmp_fn(a: ?*const anyopaque, b: ?*const anyopaque) callconv(.c) c_int {
            const aa: *const usize = @ptrCast(@alignCast(a.?));
            const bb: *const usize = @ptrCast(@alignCast(b.?));
            return if (aa.* == bb.*) 0 else if (aa.* < bb.*) -1 else 1;
        }
    }.cmp_fn;

    // Existing element
    const key1: usize = 20;
    const found = libc.stdlib.lsearch(&key1, &buf, &nel, @sizeOf(usize), cmp);
    try testing.expect(found != null);
    try testing.expectEqual(@as(usize, 2), nel);
    try testing.expectEqual(@as(usize, 20), @as(*usize, @ptrCast(@alignCast(found.?))).*);

    // Missing element should append
    const key2: usize = 30;
    const appended = libc.stdlib.lsearch(&key2, &buf, &nel, @sizeOf(usize), cmp);
    try testing.expect(appended != null);
    try testing.expectEqual(@as(usize, 3), nel);
    try testing.expectEqual(@as(usize, 30), @as(*usize, @ptrCast(@alignCast(appended.?))).*);
}
