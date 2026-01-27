// Process Control - Phase 1.4 Process Management
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const pid_t = c_int;
pub const id_t = c_uint;

pub const P_ALL: c_int = 0;
pub const P_PID: c_int = 1;
pub const P_PGID: c_int = 2;

pub const WNOHANG: c_int = 1;
pub const WUNTRACED: c_int = 2;
pub const WCONTINUED: c_int = 8;
pub const WEXITED: c_int = 4;
pub const WSTOPPED: c_int = 2;
pub const WNOWAIT: c_int = 0x1000000;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

/// Fork process
pub export fn fork() pid_t {
    const pid = std.posix.fork() catch |err| {
        setErrno(switch (err) {
            error.SystemResources => .AGAIN,
            else => .NOMEM,
        });
        return -1;
    };
    return @intCast(pid);
}

/// Fork with copy-on-write optimization
pub export fn vfork() pid_t {
    // On modern systems, vfork() behaves like fork()
    return fork();
}

/// Wait for any child
pub export fn wait(wstatus: ?*c_int) pid_t {
    return waitpid(-1, wstatus, 0);
}

/// Wait for specific process
pub export fn waitpid(pid: pid_t, wstatus: ?*c_int, options: c_int) pid_t {
    const result = std.posix.waitpid(pid, @intCast(options)) catch |err| {
        setErrno(switch (err) {
            error.ChildExecFailed => .CHILD,
            else => .INVAL,
        });
        return -1;
    };
    
    if (wstatus) |status| {
        status.* = @intCast(result.status);
    }
    
    return @intCast(result.pid);
}

/// FULL IMPLEMENTATION: Wait with detailed info
pub export fn waitid(idtype: c_int, id: id_t, infop: ?*anyopaque, options: c_int) c_int {
    const info = infop orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // siginfo_t structure (simplified for waitid)
    const siginfo_t = extern struct {
        si_signo: c_int,
        si_errno: c_int,
        si_code: c_int,
        si_pid: pid_t,
        si_uid: c_uint,
        si_status: c_int,
        si_utime: c_long,
        si_stime: c_long,
    };
    
    const info_ptr = @as(*siginfo_t, @ptrCast(@alignCast(info)));
    
    // Convert idtype and call appropriate wait function
    const wait_pid: pid_t = switch (idtype) {
        P_ALL => -1,
        P_PID => @intCast(id),
        P_PGID => -@as(pid_t, @intCast(id)),
        else => {
            setErrno(.INVAL);
            return -1;
        },
    };
    
    const result = std.posix.waitpid(wait_pid, @intCast(options)) catch |err| {
        setErrno(switch (err) {
            error.ChildExecFailed => .CHILD,
            else => .INVAL,
        });
        return -1;
    };
    
    // Fill in siginfo_t
    info_ptr.si_signo = 17; // SIGCHLD
    info_ptr.si_errno = 0;
    info_ptr.si_pid = @intCast(result.pid);
    info_ptr.si_uid = 0;
    info_ptr.si_status = @intCast(result.status);
    info_ptr.si_utime = 0;
    info_ptr.si_stime = 0;
    
    // Set si_code based on status
    if (result.status & 0x7f == 0) {
        info_ptr.si_code = 1; // CLD_EXITED
    } else if ((result.status & 0x7f) + 1 >> 1 > 0) {
        info_ptr.si_code = 2; // CLD_KILLED
    } else {
        info_ptr.si_code = 5; // CLD_STOPPED
    }
    
    return 0;
}

/// Wait for any child (BSD)
pub export fn wait3(wstatus: ?*c_int, options: c_int, rusage: ?*anyopaque) pid_t {
    _ = rusage;
    return waitpid(-1, wstatus, options);
}

/// Wait for specific process (BSD)
pub export fn wait4(pid: pid_t, wstatus: ?*c_int, options: c_int, rusage: ?*anyopaque) pid_t {
    _ = rusage;
    return waitpid(pid, wstatus, options);
}

// Process groups

/// Set process group ID
pub export fn setpgid(pid: pid_t, pgid: pid_t) c_int {
    std.posix.setpgid(pid, pgid) catch |err| {
        setErrno(switch (err) {
            error.AccessDenied => .ACCES,
            error.PermissionDenied => .PERM,
            else => .INVAL,
        });
        return -1;
    };
    return 0;
}

/// Get process group ID
pub export fn getpgid(pid: pid_t) pid_t {
    const pgid = std.posix.getpgid(pid) catch {
        setErrno(.SRCH);
        return -1;
    };
    return @intCast(pgid);
}

/// Set process group (legacy)
pub export fn setpgrp() pid_t {
    return setpgid(0, 0);
}

/// Get process group (legacy)
pub export fn getpgrp() pid_t {
    return getpgid(0);
}

/// Get process group of session leader
pub export fn getpgrp_old() pid_t {
    return getpgrp();
}

// Sessions

/// Create new session
pub export fn setsid() pid_t {
    const sid = std.posix.setsid() catch |err| {
        setErrno(switch (err) {
            error.PermissionDenied => .PERM,
            else => .INVAL,
        });
        return -1;
    };
    return @intCast(sid);
}

/// Get session ID
pub export fn getsid(pid: pid_t) pid_t {
    const sid = std.posix.getsid(pid) catch {
        setErrno(.SRCH);
        return -1;
    };
    return @intCast(sid);
}

// Priority constants
pub const PRIO_PROCESS: c_int = 0;
pub const PRIO_PGRP: c_int = 1;
pub const PRIO_USER: c_int = 2;

/// FULL IMPLEMENTATION: Get process priority
pub export fn getpriority(which: c_int, who: id_t) c_int {
    // Linux syscall returns 20 - priority (so higher values = lower priority)
    // We need to convert back to priority range -20 to 19
    const rc = std.posix.system.syscall2(.getpriority, @as(usize, @bitCast(@as(isize, which))), @as(usize, who));
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return -1;
    }
    // Convert from kernel's 1-40 range to -20 to 19
    return 20 - @as(c_int, @intCast(rc));
}

/// FULL IMPLEMENTATION: Set process priority
pub export fn setpriority(which: c_int, who: id_t, prio: c_int) c_int {
    const rc = std.posix.system.syscall3(
        .setpriority,
        @as(usize, @bitCast(@as(isize, which))),
        @as(usize, who),
        @as(usize, @bitCast(@as(isize, prio))),
    );
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return -1;
    }
    return 0;
}

/// FULL IMPLEMENTATION: Adjust process priority (nice value)
pub export fn nice(inc: c_int) c_int {
    // Get current priority, add increment, set new priority
    // The syscall needs PRIO_PROCESS=0, current process=0
    const current = getpriority(PRIO_PROCESS, 0);
    if (current == -1) {
        // Check if it was an actual error by checking errno
        const err = errno_mod.__errno_location().*;
        if (err != 0) {
            return -1;
        }
    }

    var new_nice = current + inc;
    // Clamp to valid range
    if (new_nice < -20) new_nice = -20;
    if (new_nice > 19) new_nice = 19;

    if (setpriority(PRIO_PROCESS, 0, new_nice) < 0) {
        return -1;
    }
    return new_nice;
}

// Process IDs

/// Get process ID
pub export fn getpid() pid_t {
    return @intCast(std.posix.getpid());
}

/// Get parent process ID
pub export fn getppid() pid_t {
    return @intCast(std.posix.getppid());
}

/// Get user ID
pub export fn getuid() c_uint {
    return std.posix.getuid();
}

/// Get effective user ID
pub export fn geteuid() c_uint {
    return std.posix.geteuid();
}

/// Get group ID
pub export fn getgid() c_uint {
    return std.posix.getgid();
}

/// Get effective group ID
pub export fn getegid() c_uint {
    return std.posix.getegid();
}

/// Set user ID
pub export fn setuid(uid: c_uint) c_int {
    std.posix.setuid(uid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Set effective user ID
pub export fn seteuid(uid: c_uint) c_int {
    std.posix.seteuid(uid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Set group ID
pub export fn setgid(gid: c_uint) c_int {
    std.posix.setgid(gid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Set effective group ID
pub export fn setegid(gid: c_uint) c_int {
    std.posix.setegid(gid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Set real and effective user IDs
pub export fn setreuid(ruid: c_uint, euid: c_uint) c_int {
    std.posix.setreuid(ruid, euid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Set real and effective group IDs
pub export fn setregid(rgid: c_uint, egid: c_uint) c_int {
    std.posix.setregid(rgid, egid) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}

/// Get supplementary groups
pub export fn getgroups(size: c_int, list: [*]c_uint) c_int {
    if (size < 0) {
        setErrno(.INVAL);
        return -1;
    }
    
    const groups = std.posix.getgroups(list[0..@intCast(size)]) catch {
        setErrno(.INVAL);
        return -1;
    };
    
    return @intCast(groups.len);
}

/// Set supplementary groups
pub export fn setgroups(size: usize, list: [*]const c_uint) c_int {
    std.posix.setgroups(list[0..size]) catch {
        setErrno(.PERM);
        return -1;
    };
    return 0;
}
