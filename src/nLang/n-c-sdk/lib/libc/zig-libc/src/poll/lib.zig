// poll module - Phase 1.14 - Real Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const POLLIN: c_short = 0x0001;
pub const POLLPRI: c_short = 0x0002;
pub const POLLOUT: c_short = 0x0004;
pub const POLLERR: c_short = 0x0008;
pub const POLLHUP: c_short = 0x0010;
pub const POLLNVAL: c_short = 0x0020;

pub const pollfd = extern struct {
    fd: c_int,
    events: c_short,
    revents: c_short,
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

pub export fn poll(fds: [*]pollfd, nfds: c_ulong, timeout: c_int) c_int {
    // Cast to std.posix.pollfd (should be compatible layout)
    const posix_fds = @as([*]std.posix.pollfd, @ptrCast(fds))[0..nfds];
    
    const count = std.posix.poll(posix_fds, timeout) catch |err| {
        setErrno(err);
        return -1;
    };
    
    return @intCast(count);
}

/// FULL IMPLEMENTATION: Poll with signal mask and nanosecond timeout
pub export fn ppoll(fds: [*]pollfd, nfds: c_ulong, timeout_ts: ?*const anyopaque, sigmask: ?*const anyopaque) c_int {
    if (@hasDecl(std.posix.system, "ppoll")) {
        const rc = std.posix.system.ppoll(@ptrCast(fds), nfds, @ptrCast(timeout_ts), @ptrCast(sigmask), @sizeOf(std.posix.sigset_t));
        if (failIfErrno(rc)) return -1;
        return @intCast(rc);
    }
    
    // Fallback: Convert timespec to milliseconds and use regular poll
    var timeout_ms: c_int = -1;
    
    if (timeout_ts) |ts_ptr| {
        const timespec = @as(*const extern struct {
            tv_sec: i64,
            tv_nsec: c_long,
        }, @ptrCast(@alignCast(ts_ptr)));
        
        const ms_from_sec = timespec.tv_sec * 1000;
        const ms_from_nsec = @divFloor(timespec.tv_nsec, 1000000);
        timeout_ms = @intCast(ms_from_sec + ms_from_nsec);
    }
    
    // Note: In fallback mode, sigmask parameter is not used
    // (signal handling is less precise without ppoll system call)
    if (sigmask == null) {
        // Explicitly handle null case
    }
    
    return poll(fds, nfds, timeout_ms);
}
