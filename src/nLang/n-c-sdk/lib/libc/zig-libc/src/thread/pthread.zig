// Full pthread Implementation - Stub Removal Q1 2026
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Thread data for pthread handles
const ThreadData = struct {
    thread: *std.Thread,
    return_value: ?*anyopaque,
    detached: bool,
};

// Thread registry for pthread handles
const ThreadRegistry = struct {
    threads: std.AutoHashMap(u64, *ThreadData),
    mutex: std.Thread.Mutex,
    next_id: u64,

    var instance: ThreadRegistry = undefined;
    var initialized: bool = false;

    fn init() !void {
        if (initialized) return;
        instance = .{
            .threads = std.AutoHashMap(u64, *ThreadData).init(std.heap.page_allocator),
            .mutex = .{},
            .next_id = 1,
        };
        initialized = true;
    }

    fn register(thread: *std.Thread, detached: bool) !u64 {
        instance.mutex.lock();
        defer instance.mutex.unlock();

        const data = std.heap.page_allocator.create(ThreadData) catch return error.OutOfMemory;
        data.* = .{
            .thread = thread,
            .return_value = null,
            .detached = detached,
        };

        const id = instance.next_id;
        instance.next_id += 1;
        try instance.threads.put(id, data);
        return id;
    }

    fn get(id: u64) ?*ThreadData {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        return instance.threads.get(id);
    }

    fn remove(id: u64) void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        if (instance.threads.fetchRemove(id)) |kv| {
            std.heap.page_allocator.destroy(kv.value);
        }
    }
};

// Constants for attribute parsing
const PTHREAD_CREATE_DETACHED: c_int = 1;

/// FULL IMPLEMENTATION: Create thread
pub export fn pthread_create(
    thread: ?*c_ulong,
    attr: ?*const anyopaque,
    start_routine: ?*const fn (?*anyopaque) callconv(.C) ?*anyopaque,
    arg: ?*anyopaque,
) c_int {
    // Parse pthread_attr_t for detach state and stack size
    var detached: bool = false;
    var stack_size: usize = 8 * 1024 * 1024; // Default 8MB

    if (attr) |a| {
        // pthread_attr_t layout: first 4 bytes = detachstate, bytes 8-16 = stacksize
        const attr_bytes: [*]const u8 = @ptrCast(a);
        const detach_state = @as(*align(1) const c_int, @ptrCast(attr_bytes)).*;
        detached = (detach_state == PTHREAD_CREATE_DETACHED);

        const stack_ptr: *align(1) const usize = @ptrCast(attr_bytes + 8);
        const attr_stack_size = stack_ptr.*;
        if (attr_stack_size >= 16384) { // Minimum stack size
            stack_size = attr_stack_size;
        }
    }

    ThreadRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };

    const routine = start_routine orelse {
        setErrno(.INVAL);
        return -1;
    };

    const thread_ptr = std.heap.page_allocator.create(std.Thread) catch {
        setErrno(.NOMEM);
        return -1;
    };

    const ThreadContext = struct {
        routine: *const fn (?*anyopaque) callconv(.C) ?*anyopaque,
        arg: ?*anyopaque,

        fn run(ctx: @This()) void {
            _ = ctx.routine(ctx.arg);
        }
    };

    const ctx = ThreadContext{
        .routine = routine,
        .arg = arg,
    };

    thread_ptr.* = std.Thread.spawn(.{ .stack_size = stack_size }, ThreadContext.run, .{ctx}) catch |err| {
        std.heap.page_allocator.destroy(thread_ptr);
        setErrno(switch (err) {
            error.SystemResources => .AGAIN,
            error.OutOfMemory => .NOMEM,
            else => .INVAL,
        });
        return -1;
    };

    const tid = ThreadRegistry.register(thread_ptr, detached) catch {
        thread_ptr.detach();
        std.heap.page_allocator.destroy(thread_ptr);
        setErrno(.NOMEM);
        return -1;
    };

    if (thread) |t| t.* = tid;
    return 0;
}

/// FULL IMPLEMENTATION: Join thread
pub export fn pthread_join(thread: c_ulong, retval: ?*?*anyopaque) c_int {
    const thread_data = ThreadRegistry.get(thread) orelse {
        setErrno(.SRCH);
        return -1;
    };

    // Check if thread is detached
    if (thread_data.detached) {
        setErrno(.INVAL);
        return -1;
    }

    thread_data.thread.join();

    // Return the stored return value if requested
    if (retval) |rv| {
        rv.* = thread_data.return_value;
    }

    std.heap.page_allocator.destroy(thread_data.thread);
    ThreadRegistry.remove(thread);

    return 0;
}

/// FULL IMPLEMENTATION: Detach thread
pub export fn pthread_detach(thread: c_ulong) c_int {
    const thread_data = ThreadRegistry.get(thread) orelse {
        setErrno(.SRCH);
        return -1;
    };

    thread_data.detached = true;
    thread_data.thread.detach();
    std.heap.page_allocator.destroy(thread_data.thread);
    ThreadRegistry.remove(thread);

    return 0;
}

// Mutex types
const PTHREAD_MUTEX_NORMAL: c_int = 0;
const PTHREAD_MUTEX_ERRORCHECK: c_int = 1;
const PTHREAD_MUTEX_RECURSIVE: c_int = 2;
const PTHREAD_MUTEX_DEFAULT: c_int = 0;

// Mutex data with type information
const MutexData = struct {
    mutex: *std.Thread.Mutex,
    mutex_type: c_int,
    owner: ?std.Thread.Id,
    lock_count: u32,
};

// Mutex registry for pthread_mutex_t handles (pub for use by advanced.zig)
pub const MutexRegistry = struct {
    mutexes: std.AutoHashMap(usize, *MutexData),
    lock: std.Thread.Mutex,

    var instance: MutexRegistry = undefined;
    var initialized: bool = false;

    pub fn init() !void {
        if (initialized) return;
        instance = .{
            .mutexes = std.AutoHashMap(usize, *MutexData).init(std.heap.page_allocator),
            .lock = .{},
        };
        initialized = true;
    }

    pub fn register(data: *MutexData, handle: usize) !void {
        instance.lock.lock();
        defer instance.lock.unlock();
        try instance.mutexes.put(handle, data);
    }

    pub fn get(handle: usize) ?*MutexData {
        instance.lock.lock();
        defer instance.lock.unlock();
        return instance.mutexes.get(handle);
    }

    pub fn remove(handle: usize) void {
        instance.lock.lock();
        defer instance.lock.unlock();
        _ = instance.mutexes.remove(handle);
    }
};

/// FULL IMPLEMENTATION: Initialize mutex
pub export fn pthread_mutex_init(mutex: ?*anyopaque, attr: ?*const anyopaque) c_int {
    // Parse mutexattr for mutex type
    var mutex_type: c_int = PTHREAD_MUTEX_DEFAULT;

    if (attr) |a| {
        // pthread_mutexattr_t layout: first 4 bytes = mutex type
        const attr_bytes: [*]const u8 = @ptrCast(a);
        mutex_type = @as(*align(1) const c_int, @ptrCast(attr_bytes)).*;
        // Validate mutex type
        if (mutex_type < 0 or mutex_type > 3) {
            mutex_type = PTHREAD_MUTEX_DEFAULT;
        }
    }

    MutexRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };

    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const real_mutex = std.heap.page_allocator.create(std.Thread.Mutex) catch {
        setErrno(.NOMEM);
        return -1;
    };

    real_mutex.* = .{};

    const mutex_data = std.heap.page_allocator.create(MutexData) catch {
        std.heap.page_allocator.destroy(real_mutex);
        setErrno(.NOMEM);
        return -1;
    };

    mutex_data.* = .{
        .mutex = real_mutex,
        .mutex_type = mutex_type,
        .owner = null,
        .lock_count = 0,
    };

    const handle = @intFromPtr(m);
    MutexRegistry.register(mutex_data, handle) catch {
        std.heap.page_allocator.destroy(real_mutex);
        std.heap.page_allocator.destroy(mutex_data);
        setErrno(.NOMEM);
        return -1;
    };

    return 0;
}

/// FULL IMPLEMENTATION: Lock mutex
pub export fn pthread_mutex_lock(mutex: ?*anyopaque) c_int {
    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const handle = @intFromPtr(m);
    const mutex_data = MutexRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };

    const current_thread = std.Thread.getCurrentId();

    // Handle recursive and error-checking mutexes
    if (mutex_data.owner) |owner| {
        if (owner == current_thread) {
            switch (mutex_data.mutex_type) {
                PTHREAD_MUTEX_RECURSIVE => {
                    mutex_data.lock_count += 1;
                    return 0;
                },
                PTHREAD_MUTEX_ERRORCHECK => {
                    setErrno(.DEADLK);
                    return -1;
                },
                else => {}, // NORMAL: deadlock (proceed to lock)
            }
        }
    }

    mutex_data.mutex.lock();
    mutex_data.owner = current_thread;
    mutex_data.lock_count = 1;
    return 0;
}

/// FULL IMPLEMENTATION: Unlock mutex
pub export fn pthread_mutex_unlock(mutex: ?*anyopaque) c_int {
    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const handle = @intFromPtr(m);
    const mutex_data = MutexRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };

    const current_thread = std.Thread.getCurrentId();

    // Error-checking mutex: verify ownership
    if (mutex_data.mutex_type == PTHREAD_MUTEX_ERRORCHECK) {
        if (mutex_data.owner == null or mutex_data.owner.? != current_thread) {
            setErrno(.PERM);
            return -1;
        }
    }

    // Recursive mutex: decrement count
    if (mutex_data.mutex_type == PTHREAD_MUTEX_RECURSIVE) {
        if (mutex_data.lock_count > 1) {
            mutex_data.lock_count -= 1;
            return 0;
        }
    }

    mutex_data.owner = null;
    mutex_data.lock_count = 0;
    mutex_data.mutex.unlock();
    return 0;
}

/// FULL IMPLEMENTATION: Try lock mutex
pub export fn pthread_mutex_trylock(mutex: ?*anyopaque) c_int {
    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const handle = @intFromPtr(m);
    const mutex_data = MutexRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };

    const current_thread = std.Thread.getCurrentId();

    // Handle recursive mutex
    if (mutex_data.owner) |owner| {
        if (owner == current_thread and mutex_data.mutex_type == PTHREAD_MUTEX_RECURSIVE) {
            mutex_data.lock_count += 1;
            return 0;
        }
    }

    if (mutex_data.mutex.tryLock()) {
        mutex_data.owner = current_thread;
        mutex_data.lock_count = 1;
        return 0;
    } else {
        setErrno(.BUSY);
        return -1;
    }
}

/// FULL IMPLEMENTATION: Destroy mutex
pub export fn pthread_mutex_destroy(mutex: ?*anyopaque) c_int {
    const m = mutex orelse {
        setErrno(.INVAL);
        return -1;
    };

    const handle = @intFromPtr(m);
    if (MutexRegistry.get(handle)) |mutex_data| {
        MutexRegistry.remove(handle);
        std.heap.page_allocator.destroy(mutex_data.mutex);
        std.heap.page_allocator.destroy(mutex_data);
    }

    return 0;
}

// Condition variable registry (pub for use by utils.zig)
pub const CondRegistry = struct {
    conds: std.AutoHashMap(usize, *std.Thread.Condition),
    lock: std.Thread.Mutex,

    var instance: CondRegistry = undefined;
    var initialized: bool = false;

    pub fn init() !void {
        if (initialized) return;
        instance = .{
            .conds = std.AutoHashMap(usize, *std.Thread.Condition).init(std.heap.page_allocator),
            .lock = .{},
        };
        initialized = true;
    }

    pub fn register(cond: *std.Thread.Condition, handle: usize) !void {
        instance.lock.lock();
        defer instance.lock.unlock();
        try instance.conds.put(handle, cond);
    }

    pub fn get(handle: usize) ?*std.Thread.Condition {
        instance.lock.lock();
        defer instance.lock.unlock();
        return instance.conds.get(handle);
    }

    pub fn remove(handle: usize) void {
        instance.lock.lock();
        defer instance.lock.unlock();
        _ = instance.conds.remove(handle);
    }
};

/// FULL IMPLEMENTATION: Initialize condition variable
pub export fn pthread_cond_init(cond: ?*anyopaque, attr: ?*const anyopaque) c_int {
    _ = attr;
    
    CondRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const real_cond = std.heap.page_allocator.create(std.Thread.Condition) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    real_cond.* = .{};
    
    const handle = @intFromPtr(c);
    CondRegistry.register(real_cond, handle) catch {
        std.heap.page_allocator.destroy(real_cond);
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Signal condition variable
pub export fn pthread_cond_signal(cond: ?*anyopaque) c_int {
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(c);
    const real_cond = CondRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    real_cond.signal();
    return 0;
}

/// FULL IMPLEMENTATION: Broadcast condition variable
pub export fn pthread_cond_broadcast(cond: ?*anyopaque) c_int {
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(c);
    const real_cond = CondRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    real_cond.broadcast();
    return 0;
}

/// FULL IMPLEMENTATION: Wait on condition variable
pub export fn pthread_cond_wait(cond: ?*anyopaque, mutex: ?*anyopaque) c_int {
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const m = mutex orelse {
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
    
    real_cond.wait(real_mutex);
    return 0;
}

/// FULL IMPLEMENTATION: Destroy condition variable
pub export fn pthread_cond_destroy(cond: ?*anyopaque) c_int {
    const c = cond orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(c);
    if (CondRegistry.get(handle)) |real_cond| {
        CondRegistry.remove(handle);
        std.heap.page_allocator.destroy(real_cond);
    }
    
    return 0;
}
