// Thread Attributes Implementation - Week 2 Stub Removal Session 2
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Thread attribute structure
pub const ThreadAttr = struct {
    detachstate: c_int, // PTHREAD_CREATE_DETACHED or PTHREAD_CREATE_JOINABLE
    stacksize: usize,
    stackaddr: ?*anyopaque,
    guardsize: usize,
    schedpolicy: c_int,
    schedparam: SchedParam,
    inheritsched: c_int,
    scope: c_int,
    
    pub const SchedParam = struct {
        sched_priority: c_int,
    };
    
    pub fn default() ThreadAttr {
        return .{
            .detachstate = 0, // PTHREAD_CREATE_JOINABLE
            .stacksize = 8 * 1024 * 1024, // 8MB default
            .stackaddr = null,
            .guardsize = 4096, // 4KB guard page
            .schedpolicy = 0, // SCHED_OTHER
            .schedparam = .{ .sched_priority = 0 },
            .inheritsched = 0, // PTHREAD_INHERIT_SCHED
            .scope = 0, // PTHREAD_SCOPE_SYSTEM
        };
    }
};

// Attribute registry for pthread_attr_t handles
const AttrRegistry = struct {
    attrs: std.AutoHashMap(usize, *ThreadAttr),
    mutex: std.Thread.Mutex,
    
    var instance: AttrRegistry = undefined;
    var initialized: bool = false;
    
    fn init() !void {
        if (initialized) return;
        instance = .{
            .attrs = std.AutoHashMap(usize, *ThreadAttr).init(std.heap.page_allocator),
            .mutex = .{},
        };
        initialized = true;
    }
    
    fn register(attr: *ThreadAttr, handle: usize) !void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        try instance.attrs.put(handle, attr);
    }
    
    fn get(handle: usize) ?*ThreadAttr {
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

/// FULL IMPLEMENTATION: Initialize thread attributes
pub export fn pthread_attr_init(attr: ?*anyopaque) c_int {
    AttrRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const thread_attr = std.heap.page_allocator.create(ThreadAttr) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    thread_attr.* = ThreadAttr.default();
    
    const handle = @intFromPtr(a);
    AttrRegistry.register(thread_attr, handle) catch {
        std.heap.page_allocator.destroy(thread_attr);
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Destroy thread attributes
pub export fn pthread_attr_destroy(attr: ?*anyopaque) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    if (AttrRegistry.get(handle)) |thread_attr| {
        AttrRegistry.remove(handle);
        std.heap.page_allocator.destroy(thread_attr);
    }
    
    return 0;
}

/// FULL IMPLEMENTATION: Set detach state
pub export fn pthread_attr_setdetachstate(attr: ?*anyopaque, detachstate: c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // PTHREAD_CREATE_DETACHED = 1, PTHREAD_CREATE_JOINABLE = 0
    if (detachstate != 0 and detachstate != 1) {
        setErrno(.INVAL);
        return -1;
    }
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    thread_attr.detachstate = detachstate;
    return 0;
}

/// FULL IMPLEMENTATION: Get detach state
pub export fn pthread_attr_getdetachstate(attr: ?*const anyopaque, detachstate: ?*c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const d = detachstate orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    d.* = thread_attr.detachstate;
    return 0;
}

/// FULL IMPLEMENTATION: Set stack size
pub export fn pthread_attr_setstacksize(attr: ?*anyopaque, stacksize: usize) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // Minimum stack size check (16KB)
    if (stacksize < 16384) {
        setErrno(.INVAL);
        return -1;
    }
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    thread_attr.stacksize = stacksize;
    return 0;
}

/// FULL IMPLEMENTATION: Get stack size
pub export fn pthread_attr_getstacksize(attr: ?*const anyopaque, stacksize: ?*usize) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const s = stacksize orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    s.* = thread_attr.stacksize;
    return 0;
}

/// FULL IMPLEMENTATION: Set stack address and size
pub export fn pthread_attr_setstack(attr: ?*anyopaque, stackaddr: ?*anyopaque, stacksize: usize) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // Minimum stack size check
    if (stacksize < 16384) {
        setErrno(.INVAL);
        return -1;
    }
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    thread_attr.stackaddr = stackaddr;
    thread_attr.stacksize = stacksize;
    return 0;
}

/// FULL IMPLEMENTATION: Get stack address and size
pub export fn pthread_attr_getstack(
    attr: ?*const anyopaque,
    stackaddr: ?*?*anyopaque,
    stacksize: ?*usize,
) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    if (stackaddr) |sa| sa.* = thread_attr.stackaddr;
    if (stacksize) |ss| ss.* = thread_attr.stacksize;
    
    return 0;
}

/// FULL IMPLEMENTATION: Set guard size
pub export fn pthread_attr_setguardsize(attr: ?*anyopaque, guardsize: usize) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    thread_attr.guardsize = guardsize;
    return 0;
}

/// FULL IMPLEMENTATION: Get guard size
pub export fn pthread_attr_getguardsize(attr: ?*const anyopaque, guardsize: ?*usize) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const g = guardsize orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    g.* = thread_attr.guardsize;
    return 0;
}

/// FULL IMPLEMENTATION: Set scheduling policy
pub export fn pthread_attr_setschedpolicy(attr: ?*anyopaque, policy: c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // SCHED_OTHER=0, SCHED_FIFO=1, SCHED_RR=2
    if (policy < 0 or policy > 2) {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    thread_attr.schedpolicy = policy;
    return 0;
}

/// FULL IMPLEMENTATION: Get scheduling policy
pub export fn pthread_attr_getschedpolicy(attr: ?*const anyopaque, policy: ?*c_int) c_int {
    const a = attr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const p = policy orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(a);
    const thread_attr = AttrRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    p.* = thread_attr.schedpolicy;
    return 0;
}

// Helper function to get attributes (for pthread_create to use)
pub fn getAttr(attr: ?*const anyopaque) ?*ThreadAttr {
    const a = attr orelse return null;
    const handle = @intFromPtr(a);
    return AttrRegistry.get(handle);
}

// Total: 12 thread attribute functions fully implemented
// All core pthread_attr_* operations with proper storage and validation
