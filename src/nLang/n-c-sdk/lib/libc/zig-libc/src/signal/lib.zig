// signal module - Phase 1.8 Priority 6 - Production Signal Handling
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Signal handler types
pub const sighandler_t = *const fn (c_int) callconv(.C) void;
pub const SA_SIGINFO: c_int = 0x00000004;

// Special signal handlers
pub const SIG_DFL: sighandler_t = @ptrFromInt(0);
pub const SIG_IGN: sighandler_t = @ptrFromInt(1);
pub const SIG_ERR: sighandler_t = @ptrFromInt(std.math.maxInt(usize));

// Signal numbers (POSIX standard)
pub const SIGHUP: c_int = 1;    // Hangup
pub const SIGINT: c_int = 2;    // Interrupt
pub const SIGQUIT: c_int = 3;   // Quit
pub const SIGILL: c_int = 4;    // Illegal instruction
pub const SIGTRAP: c_int = 5;   // Trace trap
pub const SIGABRT: c_int = 6;   // Abort
pub const SIGBUS: c_int = 7;    // Bus error
pub const SIGFPE: c_int = 8;    // Floating point exception
pub const SIGKILL: c_int = 9;   // Kill (unblockable)
pub const SIGUSR1: c_int = 10;  // User signal 1
pub const SIGSEGV: c_int = 11;  // Segmentation fault
pub const SIGUSR2: c_int = 12;  // User signal 2
pub const SIGPIPE: c_int = 13;  // Broken pipe
pub const SIGALRM: c_int = 14;  // Alarm clock
pub const SIGTERM: c_int = 15;  // Termination
pub const SIGCHLD: c_int = 17;  // Child status changed
pub const SIGCONT: c_int = 18;  // Continue
pub const SIGSTOP: c_int = 19;  // Stop (unblockable)
pub const SIGTSTP: c_int = 20;  // Keyboard stop
pub const SIGTTIN: c_int = 21;  // Background read
pub const SIGTTOU: c_int = 22;  // Background write
pub const SIGURG: c_int = 23;   // Urgent condition
pub const SIGXCPU: c_int = 24;  // CPU time limit
pub const SIGXFSZ: c_int = 25;  // File size limit
pub const SIGVTALRM: c_int = 26; // Virtual timer
pub const SIGPROF: c_int = 27;  // Profiling timer
pub const SIGWINCH: c_int = 28; // Window size change
pub const SIGIO: c_int = 29;    // I/O possible
pub const SIGPWR: c_int = 30;   // Power failure
pub const SIGSYS: c_int = 31;   // Bad system call

// Signal set operations
pub const sigset_t = extern struct {
    __val: [16]c_ulong,
};

// sigprocmask how parameter
pub const SIG_BLOCK: c_int = 0;
pub const SIG_UNBLOCK: c_int = 1;
pub const SIG_SETMASK: c_int = 2;

// sigaction flags
pub const SA_NOCLDSTOP: c_int = 0x00000001;
pub const SA_NOCLDWAIT: c_int = 0x00000002;
pub const SA_NODEFER: c_int = 0x40000000;
pub const SA_ONSTACK: c_int = 0x08000000;
pub const SA_RESETHAND: c_int = 0x80000000;
pub const SA_RESTART: c_int = 0x10000000;

// Signal info structure
pub const siginfo_t = extern struct {
    si_signo: c_int,
    si_errno: c_int,
    si_code: c_int,
    _pad: [29]c_int,
};

// Signal action structure
pub const sigaction_t = extern struct {
    __sigaction_handler: extern union {
        sa_handler: sighandler_t,
        sa_sigaction: *const fn (c_int, *siginfo_t, ?*anyopaque) callconv(.C) void,
    },
    sa_mask: sigset_t,
    sa_flags: c_int,
    sa_restorer: ?*const fn () callconv(.C) void,
};

// Stack structure for sigaltstack
pub const stack_t = extern struct {
    ss_sp: ?*anyopaque,
    ss_flags: c_int,
    ss_size: usize,
};

pub const SIGSTKSZ: usize = 8192;
pub const MINSIGSTKSZ: usize = 2048;

// sigaltstack flags
pub const SS_ONSTACK: c_int = 1;
pub const SS_DISABLE: c_int = 2;

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

// A. Signal Management (8 functions)

pub export fn signal(sig: c_int, handler: sighandler_t) sighandler_t {
    var act: sigaction_t = undefined;
    var oldact: sigaction_t = undefined;
    
    act.__sigaction_handler.sa_handler = handler;
    _ = sigemptyset(&act.sa_mask);
    act.sa_flags = SA_RESTART;
    act.sa_restorer = null;
    
    if (sigaction(sig, &act, &oldact) < 0) {
        return SIG_ERR;
    }
    
    return oldact.__sigaction_handler.sa_handler;
}

pub export fn sigaction(sig: c_int, act: ?*const sigaction_t, oact: ?*sigaction_t) c_int {
    const rc = std.posix.system.sigaction(sig, @ptrCast(act), @ptrCast(oact));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn kill(pid: c_int, sig: c_int) c_int {
    const rc = std.posix.system.kill(pid, sig);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn killpg(pgrp: c_int, sig: c_int) c_int {
    return kill(-pgrp, sig);
}

pub export fn raise(sig: c_int) c_int {
    const pid = std.posix.system.getpid();
    return kill(pid, sig);
}

pub export fn sigqueue(pid: c_int, sig: c_int, value: siginfo_t) c_int {
    _ = value;
    // Simplified: just send signal
    return kill(pid, sig);
}

pub export fn alarm(seconds: c_uint) c_uint {
    const rc = std.posix.system.alarm(seconds);
    if (rc < 0) return 0;
    return @intCast(rc);
}

pub export fn pause() c_int {
    const rc = std.posix.system.pause();
    if (failIfErrno(rc)) return -1;
    return 0;
}

// B. Signal Mask Operations (6 functions)

pub export fn sigprocmask(how: c_int, set: ?*const sigset_t, oldset: ?*sigset_t) c_int {
    const rc = std.posix.system.sigprocmask(how, @ptrCast(set), @ptrCast(oldset));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn sigsuspend(mask: *const sigset_t) c_int {
    const rc = std.posix.system.sigsuspend(@ptrCast(mask));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn sigpending(set: *sigset_t) c_int {
    const rc = std.posix.system.sigpending(@ptrCast(set));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn sigwait(set: *const sigset_t, sig: *c_int) c_int {
    const rc = std.posix.system.sigwait(@ptrCast(set), sig);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn sigwaitinfo(set: *const sigset_t, info: ?*siginfo_t) c_int {
    const rc = std.posix.system.sigwaitinfo(@ptrCast(set), @ptrCast(info));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn sigtimedwait(set: *const sigset_t, info: ?*siginfo_t, timeout: ?*const timespec) c_int {
    const rc = std.posix.system.sigtimedwait(@ptrCast(set), @ptrCast(info), @ptrCast(timeout));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

// C. Signal Set Operations (6 functions)

pub export fn sigemptyset(set: *sigset_t) c_int {
    @memset(std.mem.asBytes(set), 0);
    return 0;
}

pub export fn sigfillset(set: *sigset_t) c_int {
    @memset(std.mem.asBytes(set), 0xff);
    return 0;
}

pub export fn sigaddset(set: *sigset_t, signum: c_int) c_int {
    if (signum < 1 or signum > 64) {
        setErrno(.INVAL);
        return -1;
    }
    
    const word = @as(usize, @intCast(signum - 1)) / @bitSizeOf(c_ulong);
    const bit = @as(usize, @intCast(signum - 1)) % @bitSizeOf(c_ulong);
    
    set.__val[word] |= (@as(c_ulong, 1) << @intCast(bit));
    return 0;
}

pub export fn sigdelset(set: *sigset_t, signum: c_int) c_int {
    if (signum < 1 or signum > 64) {
        setErrno(.INVAL);
        return -1;
    }
    
    const word = @as(usize, @intCast(signum - 1)) / @bitSizeOf(c_ulong);
    const bit = @as(usize, @intCast(signum - 1)) % @bitSizeOf(c_ulong);
    
    set.__val[word] &= ~(@as(c_ulong, 1) << @intCast(bit));
    return 0;
}

pub export fn sigismember(set: *const sigset_t, signum: c_int) c_int {
    if (signum < 1 or signum > 64) {
        setErrno(.INVAL);
        return -1;
    }
    
    const word = @as(usize, @intCast(signum - 1)) / @bitSizeOf(c_ulong);
    const bit = @as(usize, @intCast(signum - 1)) % @bitSizeOf(c_ulong);
    
    return if ((set.__val[word] & (@as(c_ulong, 1) << @intCast(bit))) != 0) 1 else 0;
}

pub export fn sigandset(dest: *sigset_t, left: *const sigset_t, right: *const sigset_t) c_int {
    for (0..dest.__val.len) |i| {
        dest.__val[i] = left.__val[i] & right.__val[i];
    }
    return 0;
}

pub export fn sigorset(dest: *sigset_t, left: *const sigset_t, right: *const sigset_t) c_int {
    for (0..dest.__val.len) |i| {
        dest.__val[i] = left.__val[i] | right.__val[i];
    }
    return 0;
}

// D. Signal Utilities (5 functions)

pub export fn sigaltstack(ss: ?*const stack_t, old_ss: ?*stack_t) c_int {
    const rc = std.posix.system.sigaltstack(@ptrCast(ss), @ptrCast(old_ss));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn siginterrupt(sig: c_int, flag: c_int) c_int {
    var act: sigaction_t = undefined;
    
    if (sigaction(sig, null, &act) < 0) {
        return -1;
    }
    
    if (flag != 0) {
        act.sa_flags &= ~SA_RESTART;
    } else {
        act.sa_flags |= SA_RESTART;
    }
    
    return sigaction(sig, &act, null);
}

// String functions for signals
var signal_names = [_][:0]const u8{
    "Signal 0",
    "Hangup",                    // SIGHUP
    "Interrupt",                 // SIGINT
    "Quit",                      // SIGQUIT
    "Illegal instruction",       // SIGILL
    "Trace/breakpoint trap",     // SIGTRAP
    "Aborted",                   // SIGABRT
    "Bus error",                 // SIGBUS
    "Floating point exception",  // SIGFPE
    "Killed",                    // SIGKILL
    "User defined signal 1",     // SIGUSR1
    "Segmentation fault",        // SIGSEGV
    "User defined signal 2",     // SIGUSR2
    "Broken pipe",               // SIGPIPE
    "Alarm clock",               // SIGALRM
    "Terminated",                // SIGTERM
};

pub export fn strsignal(sig: c_int) [*:0]const u8 {
    if (sig >= 0 and sig < signal_names.len) {
        return signal_names[@intCast(sig)].ptr;
    }
    return "Unknown signal";
}

pub export fn psignal(sig: c_int, s: ?[*:0]const u8) void {
    const stderr = @cImport(@cInclude("stdio.h"));
    
    if (s) |str| {
        const len = std.mem.len(str);
        if (len > 0) {
            _ = stderr.fprintf(stderr.stderr, "%s: ", str);
        }
    }
    
    _ = stderr.fprintf(stderr.stderr, "%s\n", strsignal(sig));
}

pub export fn psiginfo(info: *const siginfo_t, s: ?[*:0]const u8) void {
    psignal(info.si_signo, s);
}

// Legacy BSD functions
pub export fn sighold(sig: c_int) c_int {
    var set: sigset_t = undefined;
    _ = sigemptyset(&set);
    _ = sigaddset(&set, sig);
    return sigprocmask(SIG_BLOCK, &set, null);
}

pub export fn sigrelse(sig: c_int) c_int {
    var set: sigset_t = undefined;
    _ = sigemptyset(&set);
    _ = sigaddset(&set, sig);
    return sigprocmask(SIG_UNBLOCK, &set, null);
}

pub export fn sigignore(sig: c_int) c_int {
    _ = signal(sig, SIG_IGN);
    return 0;
}

pub export fn sigset(sig: c_int, disp: sighandler_t) sighandler_t {
    return signal(sig, disp);
}
