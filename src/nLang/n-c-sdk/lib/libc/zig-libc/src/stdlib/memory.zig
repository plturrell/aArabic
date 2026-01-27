// Memory allocation functions for stdlib
// Phase 1.2 - Week 25
// Implements malloc, free, calloc, realloc with explicit tracking

const std = @import("std");

const allocator = std.heap.page_allocator;

const AllocationInfo = struct { size: usize };
var allocation_tracker = std.AutoHashMap(usize, AllocationInfo).init(allocator);
var tracker_mutex = std.Thread.Mutex{};

fn remember(ptr: *anyopaque, size: usize) bool {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();
    allocation_tracker.put(@intFromPtr(ptr), .{ .size = size }) catch return false;
    return true;
}

/// Public wrapper for remember - used by aligned_alloc and posix_memalign
pub fn rememberAllocation(ptr: [*]u8, size: usize) bool {
    return remember(@ptrCast(ptr), size);
}

fn forget(ptr: *anyopaque) ?usize {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();
    const removed = allocation_tracker.fetchRemove(@intFromPtr(ptr)) orelse return null;
    return removed.value.size;
}

fn lookupSize(ptr: *anyopaque) ?usize {
    tracker_mutex.lock();
    defer tracker_mutex.unlock();
    if (allocation_tracker.get(@intFromPtr(ptr))) |info| {
        return info.size;
    }
    return null;
}

/// Allocate memory block
pub export fn malloc(size: usize) ?*anyopaque {
    if (size == 0) return null;
    const slice = allocator.alloc(u8, size) catch return null;
    if (!remember(slice.ptr, size)) {
        allocator.free(slice);
        return null;
    }
    return @ptrCast(slice.ptr);
}

/// Free memory block
pub export fn free(ptr: ?*anyopaque) void {
    if (ptr == null) return;
    if (forget(ptr.?)) |len| {
        const slice = @as([*]u8, @ptrCast(ptr.?))[0..len];
        allocator.free(slice);
    }
}

/// Allocate and zero-initialize memory
pub export fn calloc(nmemb: usize, size: usize) ?*anyopaque {
    const total = std.math.mul(usize, nmemb, size) catch return null;
    if (total == 0) return null;
    const ptr = malloc(total) orelse return null;
    const slice = @as([*]u8, @ptrCast(ptr))[0..total];
    @memset(slice, 0);
    return ptr;
}

/// Reallocate memory block
pub export fn realloc(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    if (ptr == null) return malloc(size);
    if (size == 0) {
        free(ptr);
        return null;
    }
    const old_size = forget(ptr.?) orelse return null;
    const new_ptr = malloc(size) orelse {
        // Restore tracking so old pointer remains valid to caller.
        _ = remember(ptr.?, old_size);
        return null;
    };
    const copy_len = @min(old_size, size);
    const src = @as([*]u8, @ptrCast(ptr.?))[0..copy_len];
    const dst = @as([*]u8, @ptrCast(new_ptr))[0..copy_len];
    @memcpy(dst, src);
    const old_slice = @as([*]u8, @ptrCast(ptr.?))[0..old_size];
    allocator.free(old_slice);
    return new_ptr;
}

/// Alias for free (GNU extension)
pub export fn cfree(ptr: ?*anyopaque) void {
    free(ptr);
}

/// Allocate aligned block (GNU/BSD memalign)
pub export fn memalign(alignment: usize, size: usize) ?*anyopaque {
    if (!std.math.isPowerOfTwo(alignment) or alignment < @sizeOf(usize)) return null;
    if (size == 0) return null;
    // Use rawAlloc which accepts runtime alignment
    const align_enum = std.mem.Alignment.fromByteUnits(alignment);
    const ptr = allocator.rawAlloc(size, align_enum, @returnAddress()) orelse return null;
    if (!remember(ptr, size)) {
        allocator.rawFree(ptr[0..size], align_enum, @returnAddress());
        return null;
    }
    return @ptrCast(ptr);
}

/// Page-aligned allocation (legacy)
pub export fn valloc(size: usize) ?*anyopaque {
    const page = std.heap.page_size_min;
    return memalign(page, size);
}

/// realloc that frees input on failure (BSD extension)
pub export fn reallocf(ptr: ?*anyopaque, size: usize) ?*anyopaque {
    const new_ptr = realloc(ptr, size);
    if (new_ptr == null and ptr != null) {
        free(ptr);
    }
    return new_ptr;
}

/// Get usable size of allocation (non-standard but provided by many libcs)
pub export fn malloc_usable_size(ptr: ?*anyopaque) usize {
    if (ptr == null) return 0;
    if (allocation_tracker.get(@intFromPtr(ptr))) |entry| {
        return entry.size;
    }
    return 0;
}
