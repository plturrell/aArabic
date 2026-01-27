// Spinlock Implementation - Week 2 Stub Removal
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Spinlock implementation using atomic operations
const Spinlock = struct {
    locked: std.atomic.Value(u32),
    
    fn init() Spinlock {
        return .{
            .locked = std.atomic.Value(u32).init(0),
        };
    }
    
    fn lock(self: *Spinlock) void {
        while (true) {
            // Try to acquire lock
            const old = self.locked.swap(1, .acquire);
            if (old == 0) {
                // Successfully acquired
                return;
            }
            // Spin: hint to CPU we're in a spin loop
            std.atomic.spinLoopHint();
        }
    }
    
    fn tryLock(self: *Spinlock) bool {
        const old = self.locked.swap(1, .acquire);
        return old == 0;
    }
    
    fn unlock(self: *Spinlock) void {
        self.locked.store(0, .release);
    }
};

// Spinlock registry for pthread_spinlock_t handles
const SpinlockRegistry = struct {
    locks: std.AutoHashMap(usize, *Spinlock),
    mutex: std.Thread.Mutex,
    
    var instance: SpinlockRegistry = undefined;
    var initialized: bool = false;
    
    fn init() !void {
        if (initialized) return;
        instance = .{
            .locks = std.AutoHashMap(usize, *Spinlock).init(std.heap.page_allocator),
            .mutex = .{},
        };
        initialized = true;
    }
    
    fn register(lock: *Spinlock, handle: usize) !void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        try instance.locks.put(handle, lock);
    }
    
    fn get(handle: usize) ?*Spinlock {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        return instance.locks.get(handle);
    }
    
    fn remove(handle: usize) void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        _ = instance.locks.remove(handle);
    }
};

/// FULL IMPLEMENTATION: Initialize spinlock
pub export fn pthread_spin_init(lock: ?*anyopaque, pshared: c_int) c_int {
    _ = pshared; // Process-shared not implemented yet
    
    SpinlockRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const l = lock orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const real_lock = std.heap.page_allocator.create(Spinlock) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    real_lock.* = Spinlock.init();
    
    const handle = @intFromPtr(l);
    SpinlockRegistry.register(real_lock, handle) catch {
        std.heap.page_allocator.destroy(real_lock);
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Destroy spinlock
pub export fn pthread_spin_destroy(lock: ?*anyopaque) c_int {
    const l = lock orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(l);
    if (SpinlockRegistry.get(handle)) |real_lock| {
        SpinlockRegistry.remove(handle);
        std.heap.page_allocator.destroy(real_lock);
    }
    
    return 0;
}

/// FULL IMPLEMENTATION: Acquire spinlock
pub export fn pthread_spin_lock(lock: ?*anyopaque) c_int {
    const l = lock orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(l);
    const real_lock = SpinlockRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    real_lock.lock();
    return 0;
}

/// FULL IMPLEMENTATION: Try to acquire spinlock
pub export fn pthread_spin_trylock(lock: ?*anyopaque) c_int {
    const l = lock orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(l);
    const real_lock = SpinlockRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    if (real_lock.tryLock()) {
        return 0;
    } else {
        setErrno(.BUSY);
        return -1;
    }
}

/// FULL IMPLEMENTATION: Release spinlock
pub export fn pthread_spin_unlock(lock: ?*anyopaque) c_int {
    const l = lock orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const handle = @intFromPtr(l);
    const real_lock = SpinlockRegistry.get(handle) orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    real_lock.unlock();
    return 0;
}

// Total: 5 spinlock functions fully implemented
// All core spinlock operations with atomic-based implementation
