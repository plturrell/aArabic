//! I/O Multiplexing Functions
//! POSIX select(), poll(), and Linux epoll() families
//! 
//! These functions enable monitoring multiple file descriptors for I/O readiness,
//! critical for high-performance network servers and event-driven applications.

const std = @import("std");
const posix = std.posix;
const linux = std.os.linux;
const builtin = @import("builtin");

// ============================================================================
// SELECT Family (POSIX)
// ============================================================================

/// File descriptor set for select()
pub const fd_set = extern struct {
    fds_bits: [32]u32, // 1024 bits on most systems

    pub fn init() fd_set {
        return .{ .fds_bits = [_]u32{0} ** 32 };
    }
};

/// Clear all file descriptors from set
pub export fn FD_ZERO(set: *fd_set) void {
    @memset(&set.fds_bits, 0);
}

/// Add file descriptor to set
pub export fn FD_SET(fd: c_int, set: *fd_set) void {
    if (fd < 0 or fd >= 1024) return;
    const idx = @as(usize, @intCast(fd)) / 32;
    const bit = @as(u5, @intCast(@as(usize, @intCast(fd)) % 32));
    set.fds_bits[idx] |= @as(u32, 1) << bit;
}

/// Remove file descriptor from set
pub export fn FD_CLR(fd: c_int, set: *fd_set) void {
    if (fd < 0 or fd >= 1024) return;
    const idx = @as(usize, @intCast(fd)) / 32;
    const bit = @as(u5, @intCast(@as(usize, @intCast(fd)) % 32));
    set.fds_bits[idx] &= ~(@as(u32, 1) << bit);
}

/// Test if file descriptor is in set
pub export fn FD_ISSET(fd: c_int, set: *const fd_set) c_int {
    if (fd < 0 or fd >= 1024) return 0;
    const idx = @as(usize, @intCast(fd)) / 32;
    const bit = @as(u5, @intCast(@as(usize, @intCast(fd)) % 32));
    return if ((set.fds_bits[idx] & (@as(u32, 1) << bit)) != 0) 1 else 0;
}

/// POSIX timeval structure
pub const timeval = extern struct {
    tv_sec: i64,  // seconds
    tv_usec: i64, // microseconds
};

/// POSIX timespec structure
pub const timespec = extern struct {
    tv_sec: i64,  // seconds
    tv_nsec: i64, // nanoseconds
};

/// Monitor multiple file descriptors for I/O readiness
/// 
/// Blocks until one or more file descriptors are ready for I/O,
/// or until timeout expires.
/// 
/// @param nfds: Highest numbered fd + 1
/// @param readfds: Set of fds to check for read readiness
/// @param writefds: Set of fds to check for write readiness
/// @param exceptfds: Set of fds to check for exceptional conditions
/// @param timeout: Maximum time to wait (NULL = block indefinitely)
/// @return: Number of ready fds, 0 on timeout, -1 on error (errno set)
pub export fn @"select"(
    nfds: c_int,
    readfds: ?*fd_set,
    writefds: ?*fd_set,
    exceptfds: ?*fd_set,
    timeout: ?*timeval,
) c_int {
    if (nfds < 0 or nfds > 1024) {
        posix.errno(posix.E.INVAL);
        return -1;
    }

    // Convert timeval to timespec for pselect
    var ts: ?timespec = null;
    var ts_storage: timespec = undefined;
    if (timeout) |tv| {
        ts_storage = .{
            .tv_sec = tv.tv_sec,
            .tv_nsec = tv.tv_usec * 1000, // microseconds to nanoseconds
        };
        ts = ts_storage;
    }

    // Use pselect with no signal mask
    const result = pselect(nfds, readfds, writefds, exceptfds, if (ts != null) &ts_storage else null, null);
    
    // Update timeout with remaining time if provided
    if (timeout != null and ts != null) {
        timeout.?.tv_sec = ts_storage.tv_sec;
        timeout.?.tv_usec = @divTrunc(ts_storage.tv_nsec, 1000);
    }

    return result;
}

/// Signal mask type (simplified)
pub const sigset_t = extern struct {
    __bits: [16]u64,
};

/// pselect - select with signal mask atomicity
/// 
/// Like select(), but uses timespec (nanosecond precision) and
/// atomically sets signal mask during the call.
/// 
/// @param nfds: Highest numbered fd + 1
/// @param readfds: Set of fds to check for read readiness
/// @param writefds: Set of fds to check for write readiness
/// @param exceptfds: Set of fds to check for exceptional conditions
/// @param timeout: Maximum time to wait (NULL = block indefinitely)
/// @param sigmask: Signal mask to set during call (NULL = no change)
/// @return: Number of ready fds, 0 on timeout, -1 on error (errno set)
pub export fn pselect(
    nfds: c_int,
    readfds: ?*fd_set,
    writefds: ?*fd_set,
    exceptfds: ?*fd_set,
    timeout: ?*const timespec,
    sigmask: ?*const sigset_t,
) c_int {
    if (nfds < 0 or nfds > 1024) {
        posix.errno(posix.E.INVAL);
        return -1;
    }

    // Platform-specific implementation
    if (builtin.os.tag == .linux) {
        // Linux pselect6 syscall
        const result = linux.pselect6(
            @intCast(nfds),
            readfds,
            writefds,
            exceptfds,
            timeout,
            sigmask,
        );
        
        if (linux.getErrno(result) != .SUCCESS) {
            posix.errno(linux.getErrno(result));
            return -1;
        }
        return @intCast(result);
    } else if (builtin.os.tag == .macos) {
        // macOS has native pselect
        const result = std.c.pselect(
            nfds,
            readfds,
            writefds,
            exceptfds,
            timeout,
            sigmask,
        );
        return result;
    } else {
        // Fallback: use select() (loses signal mask atomicity)
        _ = sigmask; // Ignored in fallback
        
        var tv: ?timeval = null;
        var tv_storage: timeval = undefined;
        if (timeout) |ts| {
            tv_storage = .{
                .tv_sec = ts.tv_sec,
                .tv_usec = @divTrunc(ts.tv_nsec, 1000),
            };
            tv = tv_storage;
        }
        
        return @"select"(nfds, readfds, writefds, exceptfds, if (tv != null) &tv_storage else null);
    }
}

// ============================================================================
// POLL Family (POSIX)
// ============================================================================

/// Poll events flags
pub const POLLIN: i16 = 0x0001;     // Data available to read
pub const POLLPRI: i16 = 0x0002;    // High priority data available
pub const POLLOUT: i16 = 0x0004;    // File descriptor writable
pub const POLLERR: i16 = 0x0008;    // Error condition
pub const POLLHUP: i16 = 0x0010;    // Hang up
pub const POLLNVAL: i16 = 0x0020;   // Invalid request

/// Poll file descriptor structure
pub const pollfd = extern struct {
    fd: c_int,        // File descriptor
    events: i16,      // Requested events (bitmask)
    revents: i16,     // Returned events (bitmask)
};

/// Wait for events on multiple file descriptors
/// 
/// More efficient than select() for large numbers of file descriptors.
/// 
/// @param fds: Array of pollfd structures
/// @param nfds: Number of elements in fds array
/// @param timeout: Timeout in milliseconds (-1 = infinite, 0 = return immediately)
/// @return: Number of ready fds, 0 on timeout, -1 on error (errno set)
pub export fn poll(fds: [*]pollfd, nfds: c_ulong, timeout: c_int) c_int {
    if (builtin.os.tag == .linux) {
        const result = linux.poll(fds, nfds, timeout);
        
        if (linux.getErrno(result) != .SUCCESS) {
            posix.errno(linux.getErrno(result));
            return -1;
        }
        return @intCast(result);
    } else {
        // Use libc poll for other platforms
        return std.c.poll(fds, nfds, timeout);
    }
}

/// ppoll - poll with signal mask atomicity
/// 
/// Like poll(), but uses timespec (nanosecond precision) and
/// atomically sets signal mask during the call.
/// 
/// @param fds: Array of pollfd structures
/// @param nfds: Number of elements in fds array
/// @param timeout: Maximum time to wait (NULL = block indefinitely)
/// @param sigmask: Signal mask to set during call (NULL = no change)
/// @return: Number of ready fds, 0 on timeout, -1 on error (errno set)
pub export fn ppoll(
    fds: [*]pollfd,
    nfds: c_ulong,
    timeout: ?*const timespec,
    sigmask: ?*const sigset_t,
) c_int {
    if (builtin.os.tag == .linux) {
        const result = linux.ppoll(fds, nfds, timeout, sigmask, 8); // 8 = sizeof(sigset_t)
        
        if (linux.getErrno(result) != .SUCCESS) {
            posix.errno(linux.getErrno(result));
            return -1;
        }
        return @intCast(result);
    } else {
        // Fallback to poll (loses signal mask atomicity and nanosecond precision)
        _ = sigmask; // Ignored
        
        var timeout_ms: c_int = -1;
        if (timeout) |ts| {
            const ms = @divTrunc(ts.tv_sec * 1000 + @divTrunc(ts.tv_nsec, 1_000_000), 1);
            timeout_ms = @intCast(@min(ms, std.math.maxInt(c_int)));
        }
        
        return poll(fds, nfds, timeout_ms);
    }
}

// ============================================================================
// EPOLL Family (Linux-specific, high-performance)
// ============================================================================

/// epoll events
pub const EPOLLIN: u32 = 0x001;      // Available for read
pub const EPOLLOUT: u32 = 0x004;     // Available for write
pub const EPOLLPRI: u32 = 0x002;     // Urgent data available
pub const EPOLLERR: u32 = 0x008;     // Error condition
pub const EPOLLHUP: u32 = 0x010;     // Hang up
pub const EPOLLET: u32 = 0x80000000; // Edge-triggered mode
pub const EPOLLONESHOT: u32 = 0x40000000; // One-shot mode

/// epoll operation codes
pub const EPOLL_CTL_ADD: u32 = 1;    // Add fd to epoll
pub const EPOLL_CTL_DEL: u32 = 2;    // Remove fd from epoll
pub const EPOLL_CTL_MOD: u32 = 3;    // Modify fd in epoll

/// epoll_data union
pub const epoll_data = extern union {
    ptr: ?*anyopaque,
    fd: c_int,
    u32_value: u32,
    u64_value: u64,
};

/// epoll_event structure
pub const epoll_event = extern struct {
    events: u32,           // Event mask
    data: epoll_data,      // User data
};

/// Create an epoll file descriptor
/// 
/// Creates a new epoll instance with specified flags.
/// 
/// @param flags: Creation flags (EPOLL_CLOEXEC)
/// @return: epoll file descriptor on success, -1 on error (errno set)
pub export fn epoll_create1(flags: c_int) c_int {
    if (builtin.os.tag != .linux) {
        // epoll is Linux-specific
        posix.errno(posix.E.NOSYS);
        return -1;
    }

    const result = linux.epoll_create1(@intCast(flags));
    
    if (linux.getErrno(result) != .SUCCESS) {
        posix.errno(linux.getErrno(result));
        return -1;
    }
    return @intCast(result);
}

/// Control interface for an epoll file descriptor
/// 
/// Add, modify, or remove file descriptors from epoll interest list.
/// 
/// @param epfd: epoll file descriptor
/// @param op: Operation (EPOLL_CTL_ADD/MOD/DEL)
/// @param fd: File descriptor to operate on
/// @param event: Event structure (NULL for EPOLL_CTL_DEL)
/// @return: 0 on success, -1 on error (errno set)
pub export fn epoll_ctl(
    epfd: c_int,
    op: c_int,
    fd: c_int,
    event: ?*epoll_event,
) c_int {
    if (builtin.os.tag != .linux) {
        posix.errno(posix.E.NOSYS);
        return -1;
    }

    const result = linux.epoll_ctl(@intCast(epfd), @intCast(op), @intCast(fd), event);
    
    if (linux.getErrno(result) != .SUCCESS) {
        posix.errno(linux.getErrno(result));
        return -1;
    }
    return 0;
}

/// Wait for events on an epoll file descriptor
/// 
/// Blocks until events are available or timeout expires.
/// 
/// @param epfd: epoll file descriptor
/// @param events: Buffer to store triggered events
/// @param maxevents: Maximum number of events to return
/// @param timeout: Timeout in milliseconds (-1 = infinite)
/// @return: Number of triggered events, 0 on timeout, -1 on error (errno set)
pub export fn epoll_wait(
    epfd: c_int,
    events: [*]epoll_event,
    maxevents: c_int,
    timeout: c_int,
) c_int {
    if (builtin.os.tag != .linux) {
        posix.errno(posix.E.NOSYS);
        return -1;
    }

    if (maxevents <= 0) {
        posix.errno(posix.E.INVAL);
        return -1;
    }

    const result = linux.epoll_wait(@intCast(epfd), events, @intCast(maxevents), timeout);
    
    if (linux.getErrno(result) != .SUCCESS) {
        posix.errno(linux.getErrno(result));
        return -1;
    }
    return @intCast(result);
}

/// Wait for events on an epoll file descriptor with signal mask
/// 
/// Like epoll_wait(), but atomically sets signal mask during the call.
/// 
/// @param epfd: epoll file descriptor
/// @param events: Buffer to store triggered events
/// @param maxevents: Maximum number of events to return
/// @param timeout: Timeout in milliseconds (-1 = infinite)
/// @param sigmask: Signal mask to set during call (NULL = no change)
/// @return: Number of triggered events, 0 on timeout, -1 on error (errno set)
pub export fn epoll_pwait(
    epfd: c_int,
    events: [*]epoll_event,
    maxevents: c_int,
    timeout: c_int,
    sigmask: ?*const sigset_t,
) c_int {
    if (builtin.os.tag != .linux) {
        posix.errno(posix.E.NOSYS);
        return -1;
    }

    if (maxevents <= 0) {
        posix.errno(posix.E.INVAL);
        return -1;
    }

    const result = linux.epoll_pwait(
        @intCast(epfd),
        events,
        @intCast(maxevents),
        timeout,
        sigmask,
        8, // sizeof(sigset_t)
    );
    
    if (linux.getErrno(result) != .SUCCESS) {
        posix.errno(linux.getErrno(result));
        return -1;
    }
    return @intCast(result);
}

// Legacy epoll_create (size parameter ignored in modern kernels)
pub export fn epoll_create(size: c_int) c_int {
    _ = size; // Ignored in modern Linux
    return epoll_create1(0);
}
