// sched module - Phase 1.18 - Scheduler Interface
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Scheduling policies
pub const SCHED_OTHER: c_int = 0;
pub const SCHED_FIFO: c_int = 1;
pub const SCHED_RR: c_int = 2;
pub const SCHED_BATCH: c_int = 3;
pub const SCHED_IDLE: c_int = 5;
pub const SCHED_DEADLINE: c_int = 6;

pub const sched_param = extern struct {
    sched_priority: c_int,
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

/// FULL IMPLEMENTATION: Yield CPU to other threads/processes
pub export fn sched_yield() c_int {
    const rc = std.posix.system.sched_yield();
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get maximum priority for scheduling policy
pub export fn sched_get_priority_max(policy: c_int) c_int {
    const rc = std.posix.system.sched_get_priority_max(policy);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Get minimum priority for scheduling policy
pub export fn sched_get_priority_min(policy: c_int) c_int {
    const rc = std.posix.system.sched_get_priority_min(policy);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Set scheduling policy and parameters
pub export fn sched_setscheduler(pid: c_int, policy: c_int, param: *const sched_param) c_int {
    const rc = std.posix.system.sched_setscheduler(pid, policy, @ptrCast(param));
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get scheduling policy
pub export fn sched_getscheduler(pid: c_int) c_int {
    const rc = std.posix.system.sched_getscheduler(pid);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Set scheduling parameters
pub export fn sched_setparam(pid: c_int, param: *const sched_param) c_int {
    const rc = std.posix.system.sched_setparam(pid, @ptrCast(param));
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get scheduling parameters
pub export fn sched_getparam(pid: c_int, param: *sched_param) c_int {
    const rc = std.posix.system.sched_getparam(pid, @ptrCast(param));
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get round-robin scheduling interval
pub export fn sched_rr_get_interval(pid: c_int, tp: ?*anyopaque) c_int {
    const timespec_ptr = tp orelse {
        setErrno(.INVAL);
        return -1;
    };

    const rc = std.posix.system.sched_rr_get_interval(pid, @ptrCast(@alignCast(timespec_ptr)));
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Set CPU affinity mask (Linux-specific)
pub export fn sched_setaffinity(pid: c_int, cpusetsize: usize, mask: ?*const anyopaque) c_int {
    const cpu_mask = mask orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const rc = std.posix.system.sched_setaffinity(pid, cpusetsize, cpu_mask);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get CPU affinity mask (Linux-specific)
pub export fn sched_getaffinity(pid: c_int, cpusetsize: usize, mask: ?*anyopaque) c_int {
    const cpu_mask = mask orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const rc = std.posix.system.sched_getaffinity(pid, cpusetsize, cpu_mask);
    if (failIfErrno(rc)) return -1;
    return 0;
}
