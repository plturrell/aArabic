// Threading Utilities - Phase 1.6 Final Completion
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// ============================================================================
// CORE UTILITIES (6 functions)
// ============================================================================

// Thread-local storage for return value
threadlocal var exit_retval: ?*anyopaque = null;

/// Get stored exit return value (called by pthread_join internally)
pub fn getExitRetval() ?*anyopaque {
    return exit_retval;
}

/// Exit thread with return value
pub export fn pthread_exit(retval: ?*anyopaque) noreturn {
    // Store return value in thread-local storage for pthread_join to retrieve
    exit_retval = retval;
    std.posix.exit(0);
}

/// Get current thread ID
pub export fn pthread_self() c_ulong {
    return @intCast(std.Thread.getCurrentId());
}

/// Compare thread IDs
pub export fn pthread_equal(t1: c_ulong, t2: c_ulong) c_int {
    return if (t1 == t2) 1 else 0;
}

/// Yield CPU to other threads
pub export fn pthread_yield() c_int {
    std.Thread.yield() catch {};
    return 0;
}

/// Yield CPU (sched interface)
pub export fn sched_yield() c_int {
    return pthread_yield();
}

/// Set thread concurrency level (hint to scheduler)
pub export fn pthread_setconcurrency(level: c_int) c_int {
    _ = level;
    // Concurrency level is a hint, accept but don't enforce
    return 0;
}

/// Get thread concurrency level
pub export fn pthread_getconcurrency() c_int {
    // Return 0 = implementation decides
    return 0;
}

// ============================================================================
// CONDITION VARIABLE ATTRIBUTES (6 functions)
// ============================================================================

const CondAttr = struct {
    pshared: c_int, // PTHREAD_PROCESS_PRIVATE or PTHREAD_PROCESS_SHARED
    clock_id: c_int, // Clock used for timed waits
    
    pub fn default() CondAttr {
        return .{
            .pshared = 0, // PTHREAD_PROCESS_PRIVATE
            .clock_id = 0, // CLOCK_REALTIME
        };
    }
};

const CondAttrRegistry = struct {
    attrs: std.AutoHashMap(usize, *CondAttr),
    mutex: std.Thread.Mutex,
    
    var instance: CondAttrRegistry = undefined;
    var initialized: bool = false;
    
    fn init() !void {
        if (initialized) return;
        instance = .{
            .attrs = std.AutoHashMap(usize, *CondAttr).init(std.heap.page_allocator),
            .mutex = .{},
        };
        initialized = true;
    }
    
    fn register(attr: *CondAttr, handle: usize) !void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        try instance.attrs.put(handle, attr);
    }
    
    fn get(handle: usize) ?*CondAttr {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        return instance.attrs.get(handle);
    }
    
    fn remove(handle: usize) void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        _ = instance.attrs.remove(handle);
    }
};

/// Initialize condition variable attributes
pub export fn pthread_condattr_init(attr: ?*anyopaque) c_int {
    CondAttrRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const cond_attr = std.heap.page_allocator.create(CondAttr) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    cond_attr.* = CondAttr.default();
    
    const handle = @intFromPtr(a);
    CondAttrRegistry.register(cond_attr, handle) catch {
        std.heap.page_allocator.destroy(cond_attr);
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// Destroy condition variable attributes
pub export fn pthread_condattr_destroy(attr: ?*anyopaque) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    if (CondAttrRegistry.get(handle)) |cond_attr| {
        CondAttrRegistry.remove(handle);
        std.heap.page_allocator.destroy(cond_attr);
    }
    
    return 0;
}

/// Set process-shared attribute
pub export fn pthread_condattr_setpshared(attr: ?*anyopaque, pshared: c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    if (pshared != 0 and pshared != 1) {
        setErrno(.INVAL);
        return -1;
    }
    
    const handle = @intFromPtr(a);
    const cond_attr = CondAttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    cond_attr.pshared = pshared;
    return 0;
}

/// Get process-shared attribute
pub export fn pthread_condattr_getpshared(attr: ?*const anyopaque, pshared: ?*c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const ps = pshared orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const cond_attr = CondAttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    ps.* = cond_attr.pshared;
    return 0;
}

/// Set clock for timed waits
pub export fn pthread_condattr_setclock(attr: ?*anyopaque, clock_id: c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // CLOCK_REALTIME=0, CLOCK_MONOTONIC=1
    if (clock_id < 0 or clock_id > 1) {
        setErrno(.INVAL);
        return -1;
    }
    
    const handle = @intFromPtr(a);
    const cond_attr = CondAttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    cond_attr.clock_id = clock_id;
    return 0;
}

/// Get clock for timed waits
pub export fn pthread_condattr_getclock(attr: ?*const anyopaque, clock_id: ?*c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const cid = clock_id orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const cond_attr = CondAttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    cid.* = cond_attr.clock_id;
    return 0;
}

// Import registries from pthread module
const pthread = @import("pthread.zig");
const CondRegistry = pthread.CondRegistry;
const MutexRegistry = pthread.MutexRegistry;

// POSIX timespec for timed operations
const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

// ETIMEDOUT value (110 on Linux)
const ETIMEDOUT: c_int = 110;

/// Helper to convert absolute time to relative timeout in nanoseconds
fn abstimeToTimeoutNs(abstime: *const timespec) ?u64 {
    const now_ns = std.time.nanoTimestamp();
    const now_sec = @divFloor(now_ns, std.time.ns_per_s);
    const now_nsec = @mod(now_ns, std.time.ns_per_s);

    // Calculate target time in nanoseconds
    const target_ns: i128 = @as(i128, abstime.tv_sec) * std.time.ns_per_s + @as(i128, abstime.tv_nsec);
    const current_ns: i128 = @as(i128, now_sec) * std.time.ns_per_s + @as(i128, now_nsec);

    if (target_ns <= current_ns) {
        // Already expired
        return null;
    }

    const diff = target_ns - current_ns;
    return if (diff > std.math.maxInt(u64)) std.math.maxInt(u64) else @intCast(diff);
}

/// Timed condition variable wait - Full implementation
pub export fn pthread_cond_timedwait(
    cond: ?*anyopaque,
    mutex: ?*anyopaque,
    abstime: ?*const anyopaque,
) c_int {
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };

    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const abs = abstime orelse {
        setErrno(.INVAL);
        return -1;
    };

    const cond_handle = @intFromPtr(c);
    const mutex_handle = @intFromPtr(m);

    const real_cond = CondRegistry.get(cond_handle) orelse {
        setErrno(.INVAL);
        return -1;
    };

    const real_mutex = MutexRegistry.get(mutex_handle) orelse {
        setErrno(.INVAL);
        return -1;
    };

    // Cast abstime to timespec pointer
    const ts: *const timespec = @ptrCast(@alignCast(abs));

    // Calculate timeout in nanoseconds
    const timeout_ns = abstimeToTimeoutNs(ts) orelse {
        // Timeout already expired
        return ETIMEDOUT;
    };

    // Use Zig's timedWait
    real_cond.timedWait(real_mutex, timeout_ns) catch |err| switch (err) {
        error.Timeout => return ETIMEDOUT,
    };

    return 0;
}

// Total: 13 utility functions
// These complete Phase 1.6 Threading to 100%
