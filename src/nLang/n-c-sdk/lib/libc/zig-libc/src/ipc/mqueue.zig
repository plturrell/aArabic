// POSIX Message Queues - Phase 1.3 Extended IPC
// High-performance, type-safe message queue implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const builtin = @import("builtin");

// Message queue descriptor
pub const mqd_t = c_int;

// Message queue attributes
pub const mq_attr = extern struct {
    mq_flags: c_long,      // Message queue flags (O_NONBLOCK)
    mq_maxmsg: c_long,     // Maximum number of messages
    mq_msgsize: c_long,    // Maximum message size
    mq_curmsgs: c_long,    // Current number of messages
};

// Open flags
pub const O_RDONLY: c_int = 0x0000;
pub const O_WRONLY: c_int = 0x0001;
pub const O_RDWR: c_int = 0x0002;
pub const O_CREAT: c_int = 0x0040;
pub const O_EXCL: c_int = 0x0080;
pub const O_NONBLOCK: c_int = 0x0800;
pub const O_CLOEXEC: c_int = 0x80000;

// Priority levels
pub const MQ_PRIO_MAX: c_int = 32768;

// Notification types (SIGEV_* constants)
pub const SIGEV_NONE: c_int = 0;
pub const SIGEV_SIGNAL: c_int = 1;
pub const SIGEV_THREAD: c_int = 2;

// sigval union for notification value
pub const sigval = extern union {
    sival_int: c_int,
    sival_ptr: ?*anyopaque,
};

// sigevent structure for mq_notify
pub const sigevent = extern struct {
    sigev_value: sigval,
    sigev_signo: c_int,
    sigev_notify: c_int,
    sigev_notify_function: ?*const fn (sigval) callconv(.C) void,
    sigev_notify_attributes: ?*anyopaque,
    _pad: [32]u8 = [_]u8{0} ** 32, // Padding for compatibility
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

inline fn failIfErrno(rc: anytype) bool {
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return true;
    }
    return false;
}

// Notification registration for a message queue
const NotificationRegistration = struct {
    notification: sigevent,
    registered: bool,
};

// Message queue internal structure (not exposed to C)
const MessageQueue = struct {
    fd: c_int,
    name: []const u8,
    flags: c_int,
    attr: mq_attr,
    notification: ?NotificationRegistration,
};

// Global message queue registry (simplified)
var mq_registry = std.AutoHashMap(c_int, MessageQueue).init(std.heap.page_allocator);
var mq_registry_mutex = std.Thread.Mutex{};
var next_mqd: c_int = 100; // Start at 100 to avoid conflicts

/// Open a message queue
pub export fn mq_open(name: [*:0]const u8, oflag: c_int, ...) mqd_t {
    mq_registry_mutex.lock();
    defer mq_registry_mutex.unlock();
    
    // Validate name
    const name_slice = std.mem.span(name);
    if (name_slice.len == 0 or name_slice[0] != '/') {
        setErrno(.INVAL);
        return -1;
    }
    
    // On Linux, message queues are in /dev/mqueue/
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/dev/mqueue{s}", .{name_slice}) catch {
        setErrno(.NAMETOOLONG);
        return -1;
    };
    
    // Open with flags
    const fd = std.posix.open(path, @intCast(oflag), 0o666) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            error.PathAlreadyExists => .EXIST,
            else => .INVAL,
        });
        return -1;
    };
    
    // Create descriptor
    const mqd = next_mqd;
    next_mqd += 1;
    
    // Default attributes
    var attr = mq_attr{
        .mq_flags = if (oflag & O_NONBLOCK != 0) O_NONBLOCK else 0,
        .mq_maxmsg = 10,
        .mq_msgsize = 8192,
        .mq_curmsgs = 0,
    };
    
    // Store in registry
    const queue = MessageQueue{
        .fd = @intCast(fd),
        .name = std.heap.page_allocator.dupe(u8, name_slice) catch {
            std.posix.close(@intCast(fd));
            setErrno(.NOMEM);
            return -1;
        },
        .flags = oflag,
        .attr = attr,
        .notification = null,
    };
    
    mq_registry.put(mqd, queue) catch {
        std.posix.close(@intCast(fd));
        setErrno(.NOMEM);
        return -1;
    };
    
    return mqd;
}

/// Close a message queue
pub export fn mq_close(mqdes: mqd_t) c_int {
    mq_registry_mutex.lock();
    defer mq_registry_mutex.unlock();
    
    const queue = mq_registry.get(mqdes) orelse {
        setErrno(.EBADF);
        return -1;
    };
    
    std.posix.close(@intCast(queue.fd));
    _ = mq_registry.remove(mqdes);
    std.heap.page_allocator.free(queue.name);
    
    return 0;
}

/// Unlink (remove) a message queue
pub export fn mq_unlink(name: [*:0]const u8) c_int {
    const name_slice = std.mem.span(name);
    if (name_slice.len == 0 or name_slice[0] != '/') {
        setErrno(.INVAL);
        return -1;
    }
    
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "/dev/mqueue{s}", .{name_slice}) catch {
        setErrno(.NAMETOOLONG);
        return -1;
    };
    
    std.posix.unlink(path) catch |err| {
        setErrno(switch (err) {
            error.FileNotFound => .NOENT,
            error.AccessDenied => .ACCES,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}

/// Send a message to the queue
pub export fn mq_send(mqdes: mqd_t, msg_ptr: [*]const u8, msg_len: usize, msg_prio: c_uint) c_int {
    mq_registry_mutex.lock();
    const queue = mq_registry.get(mqdes) orelse {
        mq_registry_mutex.unlock();
        setErrno(.EBADF);
        return -1;
    };
    mq_registry_mutex.unlock();
    
    // Validate priority
    if (msg_prio >= MQ_PRIO_MAX) {
        setErrno(.EINVAL);
        return -1;
    }
    
    // Validate message size
    if (msg_len > queue.attr.mq_msgsize) {
        setErrno(.EMSGSIZE);
        return -1;
    }
    
    // Write message (simplified: direct write)
    const written = std.posix.write(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            error.AccessDenied => .ACCES,
            else => .INVAL,
        });
        return -1;
    };
    
    if (written != msg_len) {
        setErrno(.EINVAL);
        return -1;
    }
    
    return 0;
}

/// Receive a message from the queue
pub export fn mq_receive(mqdes: mqd_t, msg_ptr: [*]u8, msg_len: usize, msg_prio: ?*c_uint) isize {
    mq_registry_mutex.lock();
    const queue = mq_registry.get(mqdes) orelse {
        mq_registry_mutex.unlock();
        setErrno(.EBADF);
        return -1;
    };
    mq_registry_mutex.unlock();
    
    // Validate buffer size
    if (msg_len < queue.attr.mq_msgsize) {
        setErrno(.EMSGSIZE);
        return -1;
    }
    
    // Read message
    const bytes_read = std.posix.read(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            error.AccessDenied => .ACCES,
            else => .INVAL,
        });
        return -1;
    };
    
    // Set priority if requested (simplified: always 0)
    if (msg_prio) |prio| {
        prio.* = 0;
    }
    
    return @intCast(bytes_read);
}

/// Timed send
pub export fn mq_timedsend(mqdes: mqd_t, msg_ptr: [*]const u8, msg_len: usize, msg_prio: c_uint, abs_timeout: *const timespec) c_int {
    mq_registry_mutex.lock();
    const queue = mq_registry.get(mqdes) orelse {
        mq_registry_mutex.unlock();
        setErrno(.EBADF);
        return -1;
    };
    mq_registry_mutex.unlock();

    // Validate priority
    if (msg_prio >= MQ_PRIO_MAX) {
        setErrno(.EINVAL);
        return -1;
    }

    // Validate message size
    if (msg_len > queue.attr.mq_msgsize) {
        setErrno(.EMSGSIZE);
        return -1;
    }

    // Calculate relative timeout from absolute timeout
    const timeout_ns = calculateRelativeTimeout(abs_timeout) orelse {
        // Timeout already expired - check if non-blocking would succeed
        if (queue.attr.mq_flags & O_NONBLOCK != 0) {
            // Non-blocking queue, try once
            const written = std.posix.write(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
                setErrno(switch (err) {
                    error.WouldBlock => .AGAIN,
                    error.AccessDenied => .ACCES,
                    else => .INVAL,
                });
                return -1;
            };
            if (written != msg_len) {
                setErrno(.EINVAL);
                return -1;
            }
            return 0;
        }
        setErrno(.TIMEDOUT);
        return -1;
    };

    // For a proper implementation, we would use poll/select with timeout
    // For now, use a simplified approach: try the operation with the timeout
    // If non-blocking and would block, sleep and retry until timeout

    const start_time = std.time.nanoTimestamp();
    const deadline_ns: i128 = start_time + @as(i128, timeout_ns);

    while (true) {
        const written = std.posix.write(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
            switch (err) {
                error.WouldBlock => {
                    // Check if we've exceeded the timeout
                    if (std.time.nanoTimestamp() >= deadline_ns) {
                        setErrno(.TIMEDOUT);
                        return -1;
                    }
                    // Sleep a bit and retry (1ms intervals)
                    std.time.sleep(1_000_000);
                    continue;
                },
                error.AccessDenied => {
                    setErrno(.ACCES);
                    return -1;
                },
                else => {
                    setErrno(.INVAL);
                    return -1;
                },
            }
        };

        if (written != msg_len) {
            setErrno(.EINVAL);
            return -1;
        }
        return 0;
    }
}

/// Timed receive
pub export fn mq_timedreceive(mqdes: mqd_t, msg_ptr: [*]u8, msg_len: usize, msg_prio: ?*c_uint, abs_timeout: *const timespec) isize {
    mq_registry_mutex.lock();
    const queue = mq_registry.get(mqdes) orelse {
        mq_registry_mutex.unlock();
        setErrno(.EBADF);
        return -1;
    };
    mq_registry_mutex.unlock();

    // Validate buffer size
    if (msg_len < queue.attr.mq_msgsize) {
        setErrno(.EMSGSIZE);
        return -1;
    }

    // Calculate relative timeout from absolute timeout
    const timeout_ns = calculateRelativeTimeout(abs_timeout) orelse {
        // Timeout already expired - check if non-blocking would succeed
        if (queue.attr.mq_flags & O_NONBLOCK != 0) {
            // Non-blocking queue, try once
            const bytes_read = std.posix.read(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
                setErrno(switch (err) {
                    error.WouldBlock => .AGAIN,
                    error.AccessDenied => .ACCES,
                    else => .INVAL,
                });
                return -1;
            };
            if (msg_prio) |prio| {
                prio.* = 0;
            }
            return @intCast(bytes_read);
        }
        setErrno(.TIMEDOUT);
        return -1;
    };

    // Use timeout-based polling approach
    const start_time = std.time.nanoTimestamp();
    const deadline_ns: i128 = start_time + @as(i128, timeout_ns);

    while (true) {
        const bytes_read = std.posix.read(@intCast(queue.fd), msg_ptr[0..msg_len]) catch |err| {
            switch (err) {
                error.WouldBlock => {
                    // Check if we've exceeded the timeout
                    if (std.time.nanoTimestamp() >= deadline_ns) {
                        setErrno(.TIMEDOUT);
                        return -1;
                    }
                    // Sleep a bit and retry (1ms intervals)
                    std.time.sleep(1_000_000);
                    continue;
                },
                error.AccessDenied => {
                    setErrno(.ACCES);
                    return -1;
                },
                else => {
                    setErrno(.INVAL);
                    return -1;
                },
            }
        };

        // Set priority if requested (simplified: always 0)
        if (msg_prio) |prio| {
            prio.* = 0;
        }

        return @intCast(bytes_read);
    }
}

/// Calculate relative timeout in nanoseconds from absolute timespec
/// Returns null if the timeout has already expired
fn calculateRelativeTimeout(abs_timeout: *const timespec) ?u64 {
    // Get current time as nanoseconds since epoch
    const now_ns: i128 = std.time.nanoTimestamp();
    const now_sec: i64 = @intCast(@divFloor(now_ns, std.time.ns_per_s));
    const now_nsec: i64 = @intCast(@mod(now_ns, std.time.ns_per_s));

    // Calculate difference
    const diff_sec: i64 = abs_timeout.tv_sec - now_sec;
    const diff_nsec: i64 = @as(i64, abs_timeout.tv_nsec) - now_nsec;

    // Convert to total nanoseconds
    const total_ns: i128 = @as(i128, diff_sec) * std.time.ns_per_s + diff_nsec;

    // If timeout is in the past, return null
    if (total_ns <= 0) {
        return null;
    }

    // Return as u64 (cap at max value if somehow huge)
    return @intCast(@min(total_ns, std.math.maxInt(u64)));
}

/// Set message queue attributes
pub export fn mq_setattr(mqdes: mqd_t, newattr: *const mq_attr, oldattr: ?*mq_attr) c_int {
    mq_registry_mutex.lock();
    defer mq_registry_mutex.unlock();
    
    var queue = mq_registry.get(mqdes) orelse {
        setErrno(.EBADF);
        return -1;
    };
    
    // Save old attributes if requested
    if (oldattr) |old| {
        old.* = queue.attr;
    }
    
    // Only mq_flags can be changed
    queue.attr.mq_flags = newattr.mq_flags;
    mq_registry.put(mqdes, queue) catch {
        setErrno(.NOMEM);
        return -1;
    };
    
    return 0;
}

/// Get message queue attributes
pub export fn mq_getattr(mqdes: mqd_t, attr: *mq_attr) c_int {
    mq_registry_mutex.lock();
    defer mq_registry_mutex.unlock();
    
    const queue = mq_registry.get(mqdes) orelse {
        setErrno(.EBADF);
        return -1;
    };
    
    attr.* = queue.attr;
    return 0;
}

/// Register notification
/// If notification is NULL, any existing notification registration is removed.
/// Otherwise, registers the process for notification when a message arrives on an empty queue.
pub export fn mq_notify(mqdes: mqd_t, notification: ?*const sigevent) c_int {
    mq_registry_mutex.lock();
    defer mq_registry_mutex.unlock();

    var queue = mq_registry.get(mqdes) orelse {
        setErrno(.EBADF);
        return -1;
    };

    if (notification) |notif| {
        // Check if already registered (only one process can register at a time)
        if (queue.notification) |existing| {
            if (existing.registered) {
                setErrno(.EBUSY);
                return -1;
            }
        }

        // Validate notification type
        const notify_type = notif.sigev_notify;
        if (notify_type != SIGEV_NONE and notify_type != SIGEV_SIGNAL and notify_type != SIGEV_THREAD) {
            setErrno(.EINVAL);
            return -1;
        }

        // Store the notification request
        queue.notification = NotificationRegistration{
            .notification = notif.*,
            .registered = true,
        };
    } else {
        // NULL notification - remove existing registration
        queue.notification = null;
    }

    // Update the registry with the modified queue
    mq_registry.put(mqdes, queue) catch {
        setErrno(.NOMEM);
        return -1;
    };

    return 0;
}

const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};
