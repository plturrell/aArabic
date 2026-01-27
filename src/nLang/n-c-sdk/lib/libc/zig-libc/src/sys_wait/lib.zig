// sys/wait module - Phase 1.11 Priority 9 - Process Wait Operations
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const pid_t = c_int;
pub const idtype_t = c_uint;

// Wait options
pub const WNOHANG: c_int = 1;
pub const WUNTRACED: c_int = 2;
pub const WCONTINUED: c_int = 8;

// idtype values
pub const P_ALL: idtype_t = 0;
pub const P_PID: idtype_t = 1;
pub const P_PGID: idtype_t = 2;

// Status macros as inline functions
pub inline fn WIFEXITED(status: c_int) bool {
    return (status & 0x7f) == 0;
}

pub inline fn WEXITSTATUS(status: c_int) c_int {
    return (status >> 8) & 0xff;
}

pub inline fn WIFSIGNALED(status: c_int) bool {
    return (status & 0x7f) != 0 and (status & 0x7f) != 0x7f;
}

pub inline fn WTERMSIG(status: c_int) c_int {
    return status & 0x7f;
}

pub inline fn WIFSTOPPED(status: c_int) bool {
    return (status & 0xff) == 0x7f;
}

pub inline fn WSTOPSIG(status: c_int) c_int {
    return (status >> 8) & 0xff;
}

pub inline fn WIFCONTINUED(status: c_int) bool {
    return status == 0xffff;
}

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

pub export fn wait(stat_loc: ?*c_int) pid_t {
    return waitpid(-1, stat_loc, 0);
}

pub export fn waitpid(pid: pid_t, stat_loc: ?*c_int, options: c_int) pid_t {
    const rc = std.posix.system.wait4(pid, @ptrCast(stat_loc), options, null);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn waitid(idtype: idtype_t, id: pid_t, infop: ?*anyopaque, options: c_int) c_int {
    _ = idtype; _ = id; _ = infop; _ = options;
    setErrno(.NOSYS);
    return -1;
}

pub export fn wait3(stat_loc: ?*c_int, options: c_int, rusage: ?*anyopaque) pid_t {
    const rc = std.posix.system.wait4(-1, @ptrCast(stat_loc), options, rusage);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn wait4(pid: pid_t, stat_loc: ?*c_int, options: c_int, rusage: ?*anyopaque) pid_t {
    const rc = std.posix.system.wait4(pid, @ptrCast(stat_loc), options, rusage);
    if (failIfErrno(rc)) return -1;
    return rc;
}
