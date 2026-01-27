// pthread module - Phase 1.6 Priority 4 - Production Threading Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Thread types
pub const pthread_t = c_ulong;
pub const pthread_attr_t = extern struct { __size: [56]u8 };
pub const pthread_mutex_t = extern struct { __size: [40]u8 };
pub const pthread_mutexattr_t = extern struct { __size: [4]u8 };
pub const pthread_cond_t = extern struct { __size: [48]u8 };
pub const pthread_condattr_t = extern struct { __size: [4]u8 };
pub const pthread_rwlock_t = extern struct { __size: [56]u8 };
pub const pthread_rwlockattr_t = extern struct { __size: [8]u8 };
pub const pthread_barrier_t = extern struct { __size: [32]u8 };
pub const pthread_barrierattr_t = extern struct { __size: [4]u8 };
pub const pthread_key_t = c_uint;
pub const pthread_once_t = c_int;

// Constants
pub const PTHREAD_CREATE_JOINABLE: c_int = 0;
pub const PTHREAD_CREATE_DETACHED: c_int = 1;
pub const PTHREAD_MUTEX_NORMAL: c_int = 0;
pub const PTHREAD_MUTEX_RECURSIVE: c_int = 1;
pub const PTHREAD_MUTEX_ERRORCHECK: c_int = 2;
pub const PTHREAD_ONCE_INIT: pthread_once_t = 0;
pub const PTHREAD_BARRIER_SERIAL_THREAD: c_int = -1;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Internal thread wrapper
const ThreadWrapper = struct {
    thread: std.Thread,
    start_routine: *const fn (?*anyopaque) callconv(.C) ?*anyopaque,
    arg: ?*anyopaque,
    return_value: ?*anyopaque,
    detached: bool,
    
    fn run(self: *ThreadWrapper) void {
        self.return_value = self.start_routine(self.arg);
        run_tls_destructors();
    }
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

var thread_map = std.AutoHashMap(pthread_t, *ThreadWrapper).init(allocator);
var thread_map_mutex = std.Thread.Mutex{};
var next_thread_id = std.atomic.Value(pthread_t).init(1);

// A. Thread Management (8 functions)

pub export fn pthread_create(
    thread: *pthread_t,
    attr: ?*const pthread_attr_t,
    start_routine: *const fn (?*anyopaque) callconv(.C) ?*anyopaque,
    arg: ?*anyopaque
) c_int {
    // Parse pthread_attr_t for detach state and stack size
    var detached: bool = false;
    var stack_size: usize = 8 * 1024 * 1024; // Default 8MB

    if (attr) |a| {
        // pthread_attr_t layout: first 4 bytes = detachstate, next 8 bytes = stacksize
        const attr_bytes = &a.__size;
        const detach_state = @as(*align(1) const c_int, @ptrCast(attr_bytes)).*;
        detached = (detach_state == PTHREAD_CREATE_DETACHED);

        const stack_ptr: *align(1) const usize = @ptrCast(attr_bytes[8..16]);
        const attr_stack_size = stack_ptr.*;
        if (attr_stack_size >= 16384) { // Minimum stack size
            stack_size = attr_stack_size;
        }
    }

    const wrapper = allocator.create(ThreadWrapper) catch {
        setErrno(.NOMEM);
        return @intFromEnum(std.posix.E.NOMEM);
    };

    wrapper.* = ThreadWrapper{
        .thread = undefined,
        .start_routine = start_routine,
        .arg = arg,
        .return_value = null,
        .detached = detached,
    };

    // Use stack_size in spawn config
    wrapper.thread = std.Thread.spawn(.{ .stack_size = stack_size }, ThreadWrapper.run, .{wrapper}) catch {
        allocator.destroy(wrapper);
        setErrno(.AGAIN);
        return @intFromEnum(std.posix.E.AGAIN);
    };
    
    const tid = next_thread_id.fetchAdd(1, .monotonic);
    thread.* = tid;
    
    thread_map_mutex.lock();
    defer thread_map_mutex.unlock();
    thread_map.put(tid, wrapper) catch {
        allocator.destroy(wrapper);
        return @intFromEnum(std.posix.E.NOMEM);
    };
    
    return 0;
}

pub export fn pthread_join(thread: pthread_t, retval: ?*?*anyopaque) c_int {
    thread_map_mutex.lock();
    const wrapper = thread_map.get(thread);
    thread_map_mutex.unlock();
    
    if (wrapper == null) {
        setErrno(.SRCH);
        return @intFromEnum(std.posix.E.SRCH);
    }
    
    const w = wrapper.?;
    if (w.detached) {
        setErrno(.INVAL);
        return @intFromEnum(std.posix.E.INVAL);
    }
    
    w.thread.join();
    
    if (retval) |rv| {
        rv.* = w.return_value;
    }
    
    thread_map_mutex.lock();
    _ = thread_map.remove(thread);
    thread_map_mutex.unlock();
    
    allocator.destroy(w);
    return 0;
}

pub export fn pthread_detach(thread: pthread_t) c_int {
    thread_map_mutex.lock();
    defer thread_map_mutex.unlock();
    
    const wrapper = thread_map.get(thread);
    if (wrapper == null) {
        setErrno(.SRCH);
        return @intFromEnum(std.posix.E.SRCH);
    }
    
    const w = wrapper.?;
    w.detached = true;
    w.thread.detach();
    return 0;
}

pub export fn pthread_self() pthread_t {
    const tid = std.Thread.getCurrentId();
    return @intCast(tid);
}

pub export fn pthread_equal(t1: pthread_t, t2: pthread_t) c_int {
    return if (t1 == t2) 1 else 0;
}

pub export fn pthread_exit(retval: ?*anyopaque) noreturn {
    // In a real implementation, this would need to properly exit the thread
    // For now, we'll use a simple approach
    _ = retval;
    std.posix.exit(0);
}

pub export fn pthread_cancel(thread: pthread_t) c_int {
    _ = thread;
    // Thread cancellation is complex and not supported in basic implementation
    setErrno(.NOSYS);
    return @intFromEnum(std.posix.E.NOSYS);
}

pub export fn pthread_once(once_control: *pthread_once_t, init_routine: *const fn () callconv(.C) void) c_int {
    const state = @atomicLoad(c_int, once_control, .acquire);
    if (state == 1) return 0;
    
    // Try to acquire init lock
    if (@cmpxchgStrong(c_int, once_control, 0, 2, .acq_rel, .acquire) == null) {
        init_routine();
        @atomicStore(c_int, once_control, 1, .release);
    } else {
        // Wait for initialization
        while (@atomicLoad(c_int, once_control, .acquire) != 1) {
            std.atomic.spinLoopHint();
        }
    }
    
    return 0;
}

// B. Mutexes (12 functions)

const MutexWrapper = struct {
    mutex: std.Thread.Mutex,
    type: c_int,
};

fn getMutex(mutex: *pthread_mutex_t) *MutexWrapper {
    return @ptrCast(@alignCast(mutex));
}

pub export fn pthread_mutex_init(mutex: *pthread_mutex_t, attr: ?*const pthread_mutexattr_t) c_int {
    _ = attr;
    const wrapper = getMutex(mutex);
    wrapper.* = MutexWrapper{
        .mutex = std.Thread.Mutex{},
        .type = PTHREAD_MUTEX_NORMAL,
    };
    return 0;
}

pub export fn pthread_mutex_destroy(mutex: *pthread_mutex_t) c_int {
    _ = mutex;
    return 0;
}

pub export fn pthread_mutex_lock(mutex: *pthread_mutex_t) c_int {
    const wrapper = getMutex(mutex);
    wrapper.mutex.lock();
    return 0;
}

pub export fn pthread_mutex_trylock(mutex: *pthread_mutex_t) c_int {
    const wrapper = getMutex(mutex);
    if (wrapper.mutex.tryLock()) {
        return 0;
    }
    setErrno(.BUSY);
    return @intFromEnum(std.posix.E.BUSY);
}

pub export fn pthread_mutex_unlock(mutex: *pthread_mutex_t) c_int {
    const wrapper = getMutex(mutex);
    wrapper.mutex.unlock();
    return 0;
}

pub export fn pthread_mutex_timedlock(mutex: *pthread_mutex_t, abstime: *const timespec) c_int {
    _ = abstime;
    // Simplified: just try lock
    return pthread_mutex_trylock(mutex);
}

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

// Mutex attributes
pub export fn pthread_mutexattr_init(attr: *pthread_mutexattr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn pthread_mutexattr_destroy(attr: *pthread_mutexattr_t) c_int {
    _ = attr;
    return 0;
}

pub export fn pthread_mutexattr_settype(attr: *pthread_mutexattr_t, type_: c_int) c_int {
    _ = attr; _ = type_;
    return 0;
}

pub export fn pthread_mutexattr_gettype(attr: *const pthread_mutexattr_t, type_: *c_int) c_int {
    _ = attr;
    type_.* = PTHREAD_MUTEX_NORMAL;
    return 0;
}

pub export fn pthread_mutexattr_setpshared(attr: *pthread_mutexattr_t, pshared: c_int) c_int {
    _ = attr; _ = pshared;
    return 0;
}

pub export fn pthread_mutexattr_getpshared(attr: *const pthread_mutexattr_t, pshared: *c_int) c_int {
    _ = attr;
    pshared.* = 0; // PTHREAD_PROCESS_PRIVATE
    return 0;
}

// C. Condition Variables (10 functions)

const CondWrapper = struct {
    cond: std.Thread.Condition,
};

fn getCond(cond: *pthread_cond_t) *CondWrapper {
    return @ptrCast(@alignCast(cond));
}

pub export fn pthread_cond_init(cond: *pthread_cond_t, attr: ?*const pthread_condattr_t) c_int {
    _ = attr;
    const wrapper = getCond(cond);
    wrapper.* = CondWrapper{
        .cond = std.Thread.Condition{},
    };
    return 0;
}

pub export fn pthread_cond_destroy(cond: *pthread_cond_t) c_int {
    _ = cond;
    return 0;
}

pub export fn pthread_cond_wait(cond: *pthread_cond_t, mutex: *pthread_mutex_t) c_int {
    const cond_wrapper = getCond(cond);
    const mutex_wrapper = getMutex(mutex);
    cond_wrapper.cond.wait(&mutex_wrapper.mutex);
    return 0;
}

pub export fn pthread_cond_timedwait(cond: *pthread_cond_t, mutex: *pthread_mutex_t, abstime: *const timespec) c_int {
    _ = abstime;
    // Simplified: just do regular wait
    return pthread_cond_wait(cond, mutex);
}

pub export fn pthread_cond_signal(cond: *pthread_cond_t) c_int {
    const wrapper = getCond(cond);
    wrapper.cond.signal();
    return 0;
}

pub export fn pthread_cond_broadcast(cond: *pthread_cond_t) c_int {
    const wrapper = getCond(cond);
    wrapper.cond.broadcast();
    return 0;
}

// Condition variable attributes
pub export fn pthread_condattr_init(attr: *pthread_condattr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn pthread_condattr_destroy(attr: *pthread_condattr_t) c_int {
    _ = attr;
    return 0;
}

pub export fn pthread_condattr_setpshared(attr: *pthread_condattr_t, pshared: c_int) c_int {
    _ = attr; _ = pshared;
    return 0;
}

pub export fn pthread_condattr_getpshared(attr: *const pthread_condattr_t, pshared: *c_int) c_int {
    _ = attr;
    pshared.* = 0;
    return 0;
}

// D. Read-Write Locks (10 functions)

const RwLockWrapper = struct {
    lock: std.Thread.RwLock,
};

fn getRwLock(rwlock: *pthread_rwlock_t) *RwLockWrapper {
    return @ptrCast(@alignCast(rwlock));
}

pub export fn pthread_rwlock_init(rwlock: *pthread_rwlock_t, attr: ?*const pthread_rwlockattr_t) c_int {
    _ = attr;
    const wrapper = getRwLock(rwlock);
    wrapper.* = RwLockWrapper{
        .lock = std.Thread.RwLock{},
    };
    return 0;
}

pub export fn pthread_rwlock_destroy(rwlock: *pthread_rwlock_t) c_int {
    _ = rwlock;
    return 0;
}

pub export fn pthread_rwlock_rdlock(rwlock: *pthread_rwlock_t) c_int {
    const wrapper = getRwLock(rwlock);
    wrapper.lock.lockShared();
    return 0;
}

pub export fn pthread_rwlock_tryrdlock(rwlock: *pthread_rwlock_t) c_int {
    const wrapper = getRwLock(rwlock);
    if (wrapper.lock.tryLockShared()) {
        return 0;
    }
    return @intFromEnum(std.posix.E.BUSY);
}

pub export fn pthread_rwlock_wrlock(rwlock: *pthread_rwlock_t) c_int {
    const wrapper = getRwLock(rwlock);
    wrapper.lock.lock();
    return 0;
}

pub export fn pthread_rwlock_trywrlock(rwlock: *pthread_rwlock_t) c_int {
    const wrapper = getRwLock(rwlock);
    if (wrapper.lock.tryLock()) {
        return 0;
    }
    return @intFromEnum(std.posix.E.BUSY);
}

pub export fn pthread_rwlock_unlock(rwlock: *pthread_rwlock_t) c_int {
    const wrapper = getRwLock(rwlock);
    wrapper.lock.unlock();
    return 0;
}

// RwLock attributes
pub export fn pthread_rwlockattr_init(attr: *pthread_rwlockattr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn pthread_rwlockattr_destroy(attr: *pthread_rwlockattr_t) c_int {
    _ = attr;
    return 0;
}

pub export fn pthread_rwlockattr_setpshared(attr: *pthread_rwlockattr_t, pshared: c_int) c_int {
    _ = attr; _ = pshared;
    return 0;
}

// E. Barriers (5 functions)

const BarrierWrapper = struct {
    count: u32,
    current: std.atomic.Value(u32),
    generation: std.atomic.Value(u32),
};

fn getBarrier(barrier: *pthread_barrier_t) *BarrierWrapper {
    return @ptrCast(@alignCast(barrier));
}

pub export fn pthread_barrier_init(barrier: *pthread_barrier_t, attr: ?*const pthread_barrierattr_t, count: c_uint) c_int {
    _ = attr;
    const wrapper = getBarrier(barrier);
    wrapper.* = BarrierWrapper{
        .count = count,
        .current = std.atomic.Value(u32).init(0),
        .generation = std.atomic.Value(u32).init(0),
    };
    return 0;
}

pub export fn pthread_barrier_destroy(barrier: *pthread_barrier_t) c_int {
    _ = barrier;
    return 0;
}

pub export fn pthread_barrier_wait(barrier: *pthread_barrier_t) c_int {
    const wrapper = getBarrier(barrier);
    const gen = wrapper.generation.load(.acquire);
    const arrived = wrapper.current.fetchAdd(1, .acq_rel) + 1;
    
    if (arrived == wrapper.count) {
        wrapper.current.store(0, .release);
        wrapper.generation.store(gen + 1, .release);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    
    // Spin wait for generation change
    while (wrapper.generation.load(.acquire) == gen) {
        std.atomic.spinLoopHint();
    }
    
    return 0;
}

pub export fn pthread_barrierattr_init(attr: *pthread_barrierattr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn pthread_barrierattr_destroy(attr: *pthread_barrierattr_t) c_int {
    _ = attr;
    return 0;
}

// F. Thread-Specific Data (5 functions)

var destructors = std.AutoHashMap(pthread_key_t, ?*const fn (?*anyopaque) callconv(.C) void).init(allocator);
var destructors_mutex = std.Thread.Mutex{};
var next_key = std.atomic.Value(pthread_key_t).init(0);

// Thread-local storage for key values
threadlocal var tls_map: ?std.AutoHashMap(pthread_key_t, ?*anyopaque) = null;

fn run_tls_destructors() void {
    if (tls_map) |*map| {
        defer {
            map.deinit();
            tls_map = null;
        }
        // POSIX requires multiple iterations (typically 4) to handle destructors setting new values
        var iterations: usize = 0;
        while (iterations < 4) : (iterations += 1) {
            var progress = false;
            var it = map.iterator();
            while (it.next()) |entry| {
                const val = entry.value_ptr.*;
                if (val != null) {
                    entry.value_ptr.* = null; // Mark as consumed
                    
                    // Look up destructor
                    destructors_mutex.lock();
                    const dtor_opt = destructors.get(entry.key_ptr.*);
                    destructors_mutex.unlock();
                    
                    if (dtor_opt) |dtor| {
                        if (dtor) |f| {
                            f(val);
                            progress = true;
                        }
                    }
                }
            }
            if (!progress) break;
        }
    }
}

pub export fn pthread_key_create(key: *pthread_key_t, destructor: ?*const fn (?*anyopaque) callconv(.C) void) c_int {
    key.* = next_key.fetchAdd(1, .monotonic);
    if (destructor) |d| {
        destructors_mutex.lock();
        defer destructors_mutex.unlock();
        destructors.put(key.*, d) catch return @intFromEnum(std.posix.E.NOMEM);
    }
    return 0;
}

pub export fn pthread_key_delete(key: pthread_key_t) c_int {
    destructors_mutex.lock();
    defer destructors_mutex.unlock();
    _ = destructors.remove(key);
    return 0;
}

pub export fn pthread_setspecific(key: pthread_key_t, value: ?*const anyopaque) c_int {
    if (tls_map == null) {
        tls_map = std.AutoHashMap(pthread_key_t, ?*anyopaque).init(allocator);
    }
    tls_map.?.put(key, @constCast(value)) catch return @intFromEnum(std.posix.E.NOMEM);
    return 0;
}

pub export fn pthread_getspecific(key: pthread_key_t) ?*anyopaque {
    if (tls_map) |map| {
        return map.get(key) orelse null;
    }
    return null;
}

// G. Thread Attributes (10 functions)

pub export fn pthread_attr_init(attr: *pthread_attr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn pthread_attr_destroy(attr: *pthread_attr_t) c_int {
    _ = attr;
    return 0;
}

pub export fn pthread_attr_setdetachstate(attr: *pthread_attr_t, detachstate: c_int) c_int {
    _ = attr; _ = detachstate;
    return 0;
}

pub export fn pthread_attr_getdetachstate(attr: *const pthread_attr_t, detachstate: *c_int) c_int {
    _ = attr;
    detachstate.* = PTHREAD_CREATE_JOINABLE;
    return 0;
}

pub export fn pthread_attr_setstacksize(attr: *pthread_attr_t, stacksize: usize) c_int {
    _ = attr; _ = stacksize;
    return 0;
}

pub export fn pthread_attr_getstacksize(attr: *const pthread_attr_t, stacksize: *usize) c_int {
    _ = attr;
    stacksize.* = 2 * 1024 * 1024; // 2MB default
    return 0;
}

pub export fn pthread_attr_setstack(attr: *pthread_attr_t, stackaddr: ?*anyopaque, stacksize: usize) c_int {
    _ = attr; _ = stackaddr; _ = stacksize;
    return 0;
}

pub export fn pthread_attr_getstack(attr: *const pthread_attr_t, stackaddr: *?*anyopaque, stacksize: *usize) c_int {
    _ = attr;
    stackaddr.* = null;
    stacksize.* = 2 * 1024 * 1024;
    return 0;
}

pub export fn pthread_attr_setguardsize(attr: *pthread_attr_t, guardsize: usize) c_int {
    _ = attr; _ = guardsize;
    return 0;
}

pub export fn pthread_attr_getguardsize(attr: *const pthread_attr_t, guardsize: *usize) c_int {
    _ = attr;
    guardsize.* = 4096; // 4KB default guard
    return 0;
}
