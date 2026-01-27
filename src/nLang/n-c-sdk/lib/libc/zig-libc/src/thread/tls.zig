// Thread-Local Storage Implementation - Week 2 Stub Removal
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Maximum number of TLS keys (POSIX minimum is 128)
const PTHREAD_KEYS_MAX = 1024;

// TLS key data
const TLSKey = struct {
    in_use: bool,
    destructor: ?*const fn (?*anyopaque) callconv(.C) void,
    sequence: u32, // For ABA problem prevention
};

// Per-thread TLS data
const ThreadTLS = struct {
    values: [PTHREAD_KEYS_MAX]?*anyopaque,
};

// Global TLS management
const TLSManager = struct {
    keys: [PTHREAD_KEYS_MAX]TLSKey,
    mutex: std.Thread.Mutex,
    next_sequence: u32,
    
    var instance: TLSManager = undefined;
    var initialized: bool = false;
    
    fn init() void {
        if (initialized) return;
        instance = .{
            .keys = [_]TLSKey{.{ .in_use = false, .destructor = null, .sequence = 0 }} ** PTHREAD_KEYS_MAX,
            .mutex = .{},
            .next_sequence = 1,
        };
        initialized = true;
    }
    
    fn allocateKey(destructor: ?*const fn (?*anyopaque) callconv(.C) void) ?u32 {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        
        for (&instance.keys, 0..) |*key, i| {
            if (!key.in_use) {
                key.in_use = true;
                key.destructor = destructor;
                key.sequence = instance.next_sequence;
                instance.next_sequence +%= 1;
                return @intCast(i);
            }
        }
        return null;
    }
    
    fn freeKey(key_id: u32) void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        
        if (key_id < PTHREAD_KEYS_MAX) {
            instance.keys[key_id].in_use = false;
            instance.keys[key_id].destructor = null;
        }
    }
    
    fn getDestructor(key_id: u32) ?*const fn (?*anyopaque) callconv(.C) void {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        
        if (key_id < PTHREAD_KEYS_MAX and instance.keys[key_id].in_use) {
            return instance.keys[key_id].destructor;
        }
        return null;
    }
};

// Thread-local storage using a hashmap indexed by thread ID
const ThreadTLSRegistry = struct {
    storage: std.AutoHashMap(u64, *ThreadTLS),
    mutex: std.Thread.Mutex,
    
    var instance: ThreadTLSRegistry = undefined;
    var initialized: bool = false;
    
    fn init() !void {
        if (initialized) return;
        instance = .{
            .storage = std.AutoHashMap(u64, *ThreadTLS).init(std.heap.page_allocator),
            .mutex = .{},
        };
        initialized = true;
    }
    
    fn getOrCreate(thread_id: u64) !*ThreadTLS {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        
        if (instance.storage.get(thread_id)) |tls| {
            return tls;
        }
        
        const tls = try std.heap.page_allocator.create(ThreadTLS);
        tls.* = .{
            .values = [_]?*anyopaque{null} ** PTHREAD_KEYS_MAX,
        };
        try instance.storage.put(thread_id, tls);
        return tls;
    }
    
    fn get(thread_id: u64) ?*ThreadTLS {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        return instance.storage.get(thread_id);
    }
};

/// FULL IMPLEMENTATION: Create TLS key
pub export fn pthread_key_create(
    key: ?*c_uint,
    destructor: ?*const fn (?*anyopaque) callconv(.C) void,
) c_int {
    TLSManager.init();
    ThreadTLSRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const k = key orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const key_id = TLSManager.allocateKey(destructor) orelse {
        setErrno(.AGAIN); // PTHREAD_KEYS_MAX exceeded
        return -1;
    };
    
    k.* = key_id;
    return 0;
}

/// FULL IMPLEMENTATION: Delete TLS key
pub export fn pthread_key_delete(key_id: c_uint) c_int {
    if (key_id >= PTHREAD_KEYS_MAX) {
        setErrno(.INVAL);
        return -1;
    }
    
    TLSManager.freeKey(key_id);
    return 0;
}

/// FULL IMPLEMENTATION: Set thread-specific data
pub export fn pthread_setspecific(key_id: c_uint, value: ?*const anyopaque) c_int {
    if (key_id >= PTHREAD_KEYS_MAX) {
        setErrno(.INVAL);
        return -1;
    }
    
    const thread_id = std.Thread.getCurrentId();
    const tls = ThreadTLSRegistry.getOrCreate(thread_id) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    tls.values[key_id] = @constCast(value);
    return 0;
}

/// FULL IMPLEMENTATION: Get thread-specific data
pub export fn pthread_getspecific(key_id: c_uint) ?*anyopaque {
    if (key_id >= PTHREAD_KEYS_MAX) {
        return null;
    }
    
    const thread_id = std.Thread.getCurrentId();
    const tls = ThreadTLSRegistry.get(thread_id) orelse return null;
    
    return tls.values[key_id];
}

// One-time initialization control
const OnceControl = struct {
    state: std.atomic.Value(u32), // 0 = not called, 1 = in progress, 2 = done
    
    fn init() OnceControl {
        return .{
            .state = std.atomic.Value(u32).init(0),
        };
    }
};

const OnceRegistry = struct {
    controls: std.AutoHashMap(usize, *OnceControl),
    mutex: std.Thread.Mutex,
    
    var instance: OnceRegistry = undefined;
    var initialized: bool = false;
    
    fn init() !void {
        if (initialized) return;
        instance = .{
            .controls = std.AutoHashMap(usize, *OnceControl).init(std.heap.page_allocator),
            .mutex = .{},
        };
        initialized = true;
    }
    
    fn getOrCreate(handle: usize) !*OnceControl {
        instance.mutex.lock();
        defer instance.mutex.unlock();
        
        if (instance.controls.get(handle)) |ctrl| {
            return ctrl;
        }
        
        const ctrl = try std.heap.page_allocator.create(OnceControl);
        ctrl.* = OnceControl.init();
        try instance.controls.put(handle, ctrl);
        return ctrl;
    }
};

/// FULL IMPLEMENTATION: One-time initialization
pub export fn pthread_once(
    once_control: ?*anyopaque,
    init_routine: ?*const fn () callconv(.C) void,
) c_int {
    const oc = once_control orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const routine = init_routine orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    OnceRegistry.init() catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    const handle = @intFromPtr(oc);
    const ctrl = OnceRegistry.getOrCreate(handle) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    // Fast path: already initialized
    if (ctrl.state.load(.acquire) == 2) {
        return 0;
    }
    
    // Try to be the initializer
    const old = ctrl.state.cmpxchgStrong(0, 1, .acquire, .acquire);
    if (old == null) {
        // We're the initializer
        routine();
        ctrl.state.store(2, .release);
        return 0;
    }
    
    // Wait for initialization to complete
    while (ctrl.state.load(.acquire) != 2) {
        std.atomic.spinLoopHint();
    }
    
    return 0;
}

// Maximum number of atfork handlers
const MAX_ATFORK_HANDLERS = 32;

// Fork handler entry
const AtforkHandler = struct {
    prepare: ?*const fn () callconv(.C) void,
    parent: ?*const fn () callconv(.C) void,
    child: ?*const fn () callconv(.C) void,
};

// Global fork handler registry
const AtforkRegistry = struct {
    handlers: [MAX_ATFORK_HANDLERS]?AtforkHandler,
    count: usize,
    mutex: std.Thread.Mutex,

    var instance: AtforkRegistry = .{
        .handlers = [_]?AtforkHandler{null} ** MAX_ATFORK_HANDLERS,
        .count = 0,
        .mutex = .{},
    };

    fn register(prepare: ?*const fn () callconv(.C) void, parent: ?*const fn () callconv(.C) void, child: ?*const fn () callconv(.C) void) bool {
        instance.mutex.lock();
        defer instance.mutex.unlock();

        if (instance.count >= MAX_ATFORK_HANDLERS) {
            return false;
        }

        instance.handlers[instance.count] = .{
            .prepare = prepare,
            .parent = parent,
            .child = child,
        };
        instance.count += 1;
        return true;
    }

    // Called before fork() in parent
    pub fn runPrepare() void {
        instance.mutex.lock();
        defer instance.mutex.unlock();

        // Run prepare handlers in reverse order (LIFO)
        var i = instance.count;
        while (i > 0) {
            i -= 1;
            if (instance.handlers[i]) |h| {
                if (h.prepare) |p| p();
            }
        }
    }

    // Called after fork() in parent
    pub fn runParent() void {
        instance.mutex.lock();
        defer instance.mutex.unlock();

        // Run parent handlers in order (FIFO)
        for (instance.handlers[0..instance.count]) |maybe_h| {
            if (maybe_h) |h| {
                if (h.parent) |p| p();
            }
        }
    }

    // Called after fork() in child
    pub fn runChild() void {
        // Note: In child process, mutex state is undefined after fork
        // Run child handlers in order (FIFO)
        for (instance.handlers[0..instance.count]) |maybe_h| {
            if (maybe_h) |h| {
                if (h.child) |c| c();
            }
        }
    }
};

/// Register fork handlers - Full implementation
pub export fn pthread_atfork(
    prepare: ?*const fn () callconv(.C) void,
    parent: ?*const fn () callconv(.C) void,
    child: ?*const fn () callconv(.C) void,
) c_int {
    if (AtforkRegistry.register(prepare, parent, child)) {
        return 0;
    }
    setErrno(.NOMEM);
    return -1;
}

// Export helper functions for fork() implementation to call
pub export fn __pthread_atfork_prepare() void {
    AtforkRegistry.runPrepare();
}

pub export fn __pthread_atfork_parent() void {
    AtforkRegistry.runParent();
}

pub export fn __pthread_atfork_child() void {
    AtforkRegistry.runChild();
}

// Total: 6 TLS functions fully implemented
// pthread_key_create, pthread_key_delete, pthread_setspecific,
// pthread_getspecific, pthread_once, pthread_atfork
