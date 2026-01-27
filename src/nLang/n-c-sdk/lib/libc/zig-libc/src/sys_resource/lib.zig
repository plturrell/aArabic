// sys/resource module - Phase 1.11 Priority 9 - Resource Limits
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const rlim_t = c_ulong;
pub const RLIM_INFINITY: rlim_t = std.math.maxInt(c_ulong);

pub const rlimit = extern struct {
    rlim_cur: rlim_t,
    rlim_max: rlim_t,
};

pub const rusage = extern struct {
    ru_utime: timeval,
    ru_stime: timeval,
    ru_maxrss: c_long,
    ru_ixrss: c_long,
    ru_idrss: c_long,
    ru_isrss: c_long,
    ru_minflt: c_long,
    ru_majflt: c_long,
    ru_nswap: c_long,
    ru_inblock: c_long,
    ru_oublock: c_long,
    ru_msgsnd: c_long,
    ru_msgrcv: c_long,
    ru_nsignals: c_long,
    ru_nvcsw: c_long,
    ru_nivcsw: c_long,
};

pub const timeval = extern struct {
    tv_sec: i64,
    tv_usec: c_long,
};

// Resource types
pub const RLIMIT_CPU: c_int = 0;
pub const RLIMIT_FSIZE: c_int = 1;
pub const RLIMIT_DATA: c_int = 2;
pub const RLIMIT_STACK: c_int = 3;
pub const RLIMIT_CORE: c_int = 4;
pub const RLIMIT_RSS: c_int = 5;
pub const RLIMIT_NPROC: c_int = 6;
pub const RLIMIT_NOFILE: c_int = 7;
pub const RLIMIT_MEMLOCK: c_int = 8;
pub const RLIMIT_AS: c_int = 9;

// rusage who
pub const RUSAGE_SELF: c_int = 0;
pub const RUSAGE_CHILDREN: c_int = -1;
pub const RUSAGE_THREAD: c_int = 1;

// Priority
pub const PRIO_PROCESS: c_int = 0;
pub const PRIO_PGRP: c_int = 1;
pub const PRIO_USER: c_int = 2;

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

pub export fn getrlimit(resource: c_int, rlim: *rlimit) c_int {
    const rc = std.posix.system.getrlimit(resource, @ptrCast(rlim));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn setrlimit(resource: c_int, rlim: *const rlimit) c_int {
    const rc = std.posix.system.setrlimit(resource, @ptrCast(rlim));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn getrusage(who: c_int, usage: *rusage) c_int {
    const rc = std.posix.system.getrusage(who, @ptrCast(usage));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn getpriority(which: c_int, who: c_int) c_int {
    const rc = std.posix.system.getpriority(which, @intCast(who));
    if (rc < 0 and std.posix.errno(rc) != .SUCCESS) {
        failIfErrno(rc);
        return -1;
    }
    return rc;
}

pub export fn setpriority(which: c_int, who: c_int, prio: c_int) c_int {
    const rc = std.posix.system.setpriority(which, @intCast(who), prio);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn nice(inc: c_int) c_int {
    const old = getpriority(PRIO_PROCESS, 0);
    if (old == -1) return -1;
    return if (setpriority(PRIO_PROCESS, 0, old + inc) == 0) old + inc else -1;
}
