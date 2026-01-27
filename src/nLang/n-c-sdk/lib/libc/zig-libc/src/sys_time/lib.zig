// sys/time module - Phase 1.10 - Priority 7 - Time Operations
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const fcntl = @import("../fcntl/lib.zig");

// Time structures
pub const timeval = extern struct {
    tv_sec: c_long,
    tv_usec: c_long,
};

pub const timezone = extern struct {
    tz_minuteswest: c_int,
    tz_dsttime: c_int,
};

pub const itimerval = extern struct {
    it_interval: timeval,
    it_value: timeval,
};

// Timer types
pub const ITIMER_REAL: c_int = 0;
pub const ITIMER_VIRTUAL: c_int = 1;
pub const ITIMER_PROF: c_int = 2;

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

// Functions
pub export fn gettimeofday(tv: ?*timeval, tz: ?*timezone) c_int {
    if (tv) |t| {
        const now_us = std.time.microTimestamp();
        t.tv_sec = @intCast(@divFloor(now_us, 1_000_000));
        t.tv_usec = @intCast(@mod(now_us, 1_000_000));
    }
    if (tz) |z| {
        z.tz_minuteswest = 0;
        z.tz_dsttime = 0;
    }
    return 0;
}

pub export fn settimeofday(tv: ?*const timeval, tz: ?*const timezone) c_int {
    // Requires CAP_SYS_TIME, simplified to EPERM for now as we don't want to mess with system time
    _ = tv;
    _ = tz;
    setErrno(.PERM);
    return -1;
}

pub export fn getitimer(which: c_int, curr_value: *itimerval) c_int {
    if (@hasDecl(std.posix.system, "getitimer")) {
        const rc = std.posix.system.getitimer(which, @ptrCast(curr_value));
        if (failIfErrno(rc)) return -1;
        return 0;
    } else {
        @memset(std.mem.asBytes(curr_value), 0);
        return 0;
    }
}

pub export fn setitimer(which: c_int, new_value: *const itimerval, old_value: ?*itimerval) c_int {
    if (@hasDecl(std.posix.system, "setitimer")) {
        const rc = std.posix.system.setitimer(which, @ptrCast(new_value), @ptrCast(old_value));
        if (failIfErrno(rc)) return -1;
        return 0;
    } else {
        if (old_value) |ov| {
            @memset(std.mem.asBytes(ov), 0);
        }
        return 0;
    }
}

fn timevalToTimespec(tv: *const timeval) std.posix.timespec {
    return .{
        .tv_sec = @intCast(tv.tv_sec),
        .tv_nsec = @intCast(tv.tv_usec * 1000),
    };
}

pub export fn utimes(filename: [*:0]const u8, times: ?*const [2]timeval) c_int {
    var ts: [2]std.posix.timespec = undefined;
    if (times) |t| {
        ts[0] = timevalToTimespec(&t[0]);
        ts[1] = timevalToTimespec(&t[1]);
    } else {
        // NULL means set to current time
        ts[0].tv_nsec = std.posix.UTIME.NOW;
        ts[1].tv_nsec = std.posix.UTIME.NOW;
    }
    
    const rc = std.posix.system.utimensat(std.posix.AT.FDCWD, filename, &ts, 0);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn lutimes(filename: [*:0]const u8, times: ?*const [2]timeval) c_int {
    var ts: [2]std.posix.timespec = undefined;
    if (times) |t| {
        ts[0] = timevalToTimespec(&t[0]);
        ts[1] = timevalToTimespec(&t[1]);
    } else {
        ts[0].tv_nsec = std.posix.UTIME.NOW;
        ts[1].tv_nsec = std.posix.UTIME.NOW;
    }
    
    const rc = std.posix.system.utimensat(std.posix.AT.FDCWD, filename, &ts, std.posix.AT.SYMLINK_NOFOLLOW);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn futimes(fd: c_int, times: ?*const [2]timeval) c_int {
    var ts: [2]std.posix.timespec = undefined;
    if (times) |t| {
        ts[0] = timevalToTimespec(&t[0]);
        ts[1] = timevalToTimespec(&t[1]);
    } else {
        ts[0].tv_nsec = std.posix.UTIME.NOW;
        ts[1].tv_nsec = std.posix.UTIME.NOW;
    }
    
    const rc = std.posix.system.futimens(fd, &ts);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn futimesat(dirfd: c_int, pathname: [*:0]const u8, times: ?*const [2]timeval) c_int {
    var ts: [2]std.posix.timespec = undefined;
    if (times) |t| {
        ts[0] = timevalToTimespec(&t[0]);
        ts[1] = timevalToTimespec(&t[1]);
    } else {
        ts[0].tv_nsec = std.posix.UTIME.NOW;
        ts[1].tv_nsec = std.posix.UTIME.NOW;
    }
    
    const rc = std.posix.system.utimensat(dirfd, pathname, &ts, 0);
    if (failIfErrno(rc)) return -1;
    return 0;
}
