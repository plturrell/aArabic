// sys/select module - Phase 1.10
// Real select/pselect implementation using syscalls
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const errno_mod = @import("../errno/lib.zig");

// fd_set type - supports up to 1024 fds (FD_SETSIZE)
pub const FD_SETSIZE: usize = 1024;
pub const fd_set = extern struct {
    fds_bits: [FD_SETSIZE / @bitSizeOf(c_ulong)]c_ulong,
};

pub const timeval = extern struct {
    tv_sec: c_long,
    tv_usec: c_long,
};

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

inline fn setErrno(err: posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

/// Clear all bits in fd_set
pub export fn FD_ZERO(set: *fd_set) void {
    @memset(std.mem.asBytes(set), 0);
}

/// Set bit for fd in fd_set
pub export fn FD_SET(fd: c_int, set: *fd_set) void {
    if (fd < 0 or fd >= FD_SETSIZE) return;
    const ufd: usize = @intCast(fd);
    const idx = ufd / @bitSizeOf(c_ulong);
    const bit: u6 = @intCast(ufd % @bitSizeOf(c_ulong));
    set.fds_bits[idx] |= @as(c_ulong, 1) << bit;
}

/// Clear bit for fd in fd_set
pub export fn FD_CLR(fd: c_int, set: *fd_set) void {
    if (fd < 0 or fd >= FD_SETSIZE) return;
    const ufd: usize = @intCast(fd);
    const idx = ufd / @bitSizeOf(c_ulong);
    const bit: u6 = @intCast(ufd % @bitSizeOf(c_ulong));
    set.fds_bits[idx] &= ~(@as(c_ulong, 1) << bit);
}

/// Test if bit for fd is set in fd_set
pub export fn FD_ISSET(fd: c_int, set: *const fd_set) c_int {
    if (fd < 0 or fd >= FD_SETSIZE) return 0;
    const ufd: usize = @intCast(fd);
    const idx = ufd / @bitSizeOf(c_ulong);
    const bit: u6 = @intCast(ufd % @bitSizeOf(c_ulong));
    return if ((set.fds_bits[idx] & (@as(c_ulong, 1) << bit)) != 0) 1 else 0;
}

/// Synchronous I/O multiplexing
pub export fn select(nfds: c_int, readfds: ?*fd_set, writefds: ?*fd_set, exceptfds: ?*fd_set, timeout: ?*timeval) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.select, .{
            @as(usize, @intCast(nfds)),
            @intFromPtr(readfds),
            @intFromPtr(writefds),
            @intFromPtr(exceptfds),
            @intFromPtr(timeout),
        });
        if (@as(isize, @bitCast(rc)) < 0) {
            setErrno(posix.errno(@as(isize, @bitCast(rc))));
            return -1;
        }
        return @intCast(rc);
    } else if (builtin.os.tag == .macos) {
        // macOS: use the select syscall
        const rc = std.os.darwin.syscall(.select, .{
            @as(usize, @intCast(nfds)),
            @intFromPtr(readfds),
            @intFromPtr(writefds),
            @intFromPtr(exceptfds),
            @intFromPtr(timeout),
        });
        if (@as(isize, @bitCast(rc)) < 0) {
            setErrno(posix.errno(@as(isize, @bitCast(rc))));
            return -1;
        }
        return @intCast(rc);
    } else {
        // Fallback: convert to poll
        return selectViaPoll(nfds, readfds, writefds, exceptfds, timeout);
    }
}

/// Select with signal mask and nanosecond timeout
pub export fn pselect(nfds: c_int, readfds: ?*fd_set, writefds: ?*fd_set, exceptfds: ?*fd_set, timeout: ?*const timespec, sigmask: ?*const anyopaque) c_int {
    if (builtin.os.tag == .linux) {
        // Linux pselect6 syscall
        const sigmask_ptr = @intFromPtr(sigmask);
        const data = .{
            sigmask_ptr,
            @as(usize, 8), // sigset size
        };
        const rc = std.os.linux.syscall(.pselect6, .{
            @as(usize, @intCast(nfds)),
            @intFromPtr(readfds),
            @intFromPtr(writefds),
            @intFromPtr(exceptfds),
            @intFromPtr(timeout),
            @intFromPtr(&data),
        });
        if (@as(isize, @bitCast(rc)) < 0) {
            setErrno(posix.errno(@as(isize, @bitCast(rc))));
            return -1;
        }
        return @intCast(rc);
    } else {
        // Fallback: convert timespec to timeval and use select
        // Note: sigmask parameter not supported in fallback mode
        var tv: ?timeval = null;
        var tv_storage: timeval = undefined;

        if (timeout) |ts| {
            tv_storage = .{
                .tv_sec = @intCast(ts.tv_sec),
                .tv_usec = @intCast(@divTrunc(ts.tv_nsec, 1000)),
            };
            tv = tv_storage;
        }

        return select(nfds, readfds, writefds, exceptfds, if (tv != null) &tv_storage else null);
    }
}

/// Fallback: implement select using poll
fn selectViaPoll(nfds: c_int, readfds: ?*fd_set, writefds: ?*fd_set, exceptfds: ?*fd_set, timeout: ?*timeval) c_int {
    const poll_mod = @import("../poll/lib.zig");

    // Count how many fds we need to poll
    var poll_count: usize = 0;
    const max_fd: usize = @intCast(nfds);

    for (0..max_fd) |fd| {
        var needed = false;
        if (readfds) |r| {
            if (FD_ISSET(@intCast(fd), r) != 0) needed = true;
        }
        if (writefds) |w| {
            if (FD_ISSET(@intCast(fd), w) != 0) needed = true;
        }
        if (exceptfds) |e| {
            if (FD_ISSET(@intCast(fd), e) != 0) needed = true;
        }
        if (needed) poll_count += 1;
    }

    if (poll_count == 0) {
        // No fds to poll, just sleep if timeout
        if (timeout) |tv| {
            const ms = tv.tv_sec * 1000 + @divTrunc(tv.tv_usec, 1000);
            std.time.sleep(@intCast(ms * 1_000_000));
        }
        return 0;
    }

    // Build poll array
    var pollfds: [64]poll_mod.pollfd = undefined;
    var fd_map: [64]c_int = undefined;
    var idx: usize = 0;

    for (0..max_fd) |fd| {
        var events: c_short = 0;
        if (readfds) |r| {
            if (FD_ISSET(@intCast(fd), r) != 0) events |= poll_mod.POLLIN;
        }
        if (writefds) |w| {
            if (FD_ISSET(@intCast(fd), w) != 0) events |= poll_mod.POLLOUT;
        }
        if (exceptfds) |e| {
            if (FD_ISSET(@intCast(fd), e) != 0) events |= poll_mod.POLLPRI;
        }
        if (events != 0 and idx < 64) {
            pollfds[idx] = .{ .fd = @intCast(fd), .events = events, .revents = 0 };
            fd_map[idx] = @intCast(fd);
            idx += 1;
        }
    }

    // Convert timeout
    var timeout_ms: c_int = -1;
    if (timeout) |tv| {
        timeout_ms = @intCast(tv.tv_sec * 1000 + @divTrunc(tv.tv_usec, 1000));
    }

    // Do the poll
    const result = poll_mod.poll(&pollfds, @intCast(idx), timeout_ms);
    if (result < 0) return result;

    // Clear fd_sets and set only ready fds
    if (readfds) |r| FD_ZERO(r);
    if (writefds) |w| FD_ZERO(w);
    if (exceptfds) |e| FD_ZERO(e);

    var ready_count: c_int = 0;
    for (0..idx) |i| {
        const fd = fd_map[i];
        const revents = pollfds[i].revents;

        if (readfds) |r| {
            if ((revents & (poll_mod.POLLIN | poll_mod.POLLHUP | poll_mod.POLLERR)) != 0) {
                FD_SET(fd, r);
                ready_count += 1;
            }
        }
        if (writefds) |w| {
            if ((revents & poll_mod.POLLOUT) != 0) {
                FD_SET(fd, w);
                ready_count += 1;
            }
        }
        if (exceptfds) |e| {
            if ((revents & poll_mod.POLLPRI) != 0) {
                FD_SET(fd, e);
                ready_count += 1;
            }
        }
    }

    return ready_count;
}
