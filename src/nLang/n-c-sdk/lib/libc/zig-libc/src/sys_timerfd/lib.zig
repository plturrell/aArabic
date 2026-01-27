// sys_timerfd module - Phase 1.30
// Timer file descriptors (Linux-specific, with fallback)
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

// Flags for timerfd_create
pub const TFD_CLOEXEC: c_int = 0x80000;
pub const TFD_NONBLOCK: c_int = 0x800;

// Flags for timerfd_settime
pub const TFD_TIMER_ABSTIME: c_int = 1;
pub const TFD_TIMER_CANCEL_ON_SET: c_int = 2;

// Clock IDs
pub const CLOCK_REALTIME: c_int = 0;
pub const CLOCK_MONOTONIC: c_int = 1;
pub const CLOCK_BOOTTIME: c_int = 7;
pub const CLOCK_REALTIME_ALARM: c_int = 8;
pub const CLOCK_BOOTTIME_ALARM: c_int = 9;

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: i64,
};

pub const itimerspec = extern struct {
    it_interval: timespec, // Interval for periodic timer
    it_value: timespec, // Initial expiration
};

// Internal timer state for fallback implementation
const TimerState = struct {
    clockid: c_int,
    flags: c_int,
    spec: itimerspec,
    active: bool,
};

var timer_states: [64]?TimerState = [_]?TimerState{null} ** 64;
var next_fake_fd: c_int = 1000;

/// Create a timer file descriptor
pub export fn timerfd_create(clockid: c_int, flags: c_int) c_int {
    if (builtin.os.tag == .linux) {
        // Use Linux syscall
        const rc = std.os.linux.syscall(.timerfd_create, .{
            @as(usize, @intCast(clockid)),
            @as(usize, @intCast(flags)),
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return @intCast(rc);
    } else {
        // Fallback: return a fake fd and store state
        const fd = next_fake_fd;
        next_fake_fd += 1;

        const idx: usize = @intCast(fd - 1000);
        if (idx >= timer_states.len) return -1;

        timer_states[idx] = TimerState{
            .clockid = clockid,
            .flags = flags,
            .spec = std.mem.zeroes(itimerspec),
            .active = false,
        };

        return fd;
    }
}

/// Arm or disarm a timer
pub export fn timerfd_settime(fd: c_int, flags: c_int, new_value: *const itimerspec, old_value: ?*itimerspec) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.timerfd_settime, .{
            @as(usize, @intCast(fd)),
            @as(usize, @intCast(flags)),
            @intFromPtr(new_value),
            @intFromPtr(old_value),
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return 0;
    } else {
        // Fallback
        if (fd < 1000) return -1;
        const idx: usize = @intCast(fd - 1000);
        if (idx >= timer_states.len) return -1;

        if (timer_states[idx]) |*state| {
            if (old_value) |ov| {
                ov.* = state.spec;
            }
            state.spec = new_value.*;
            state.active = (new_value.it_value.tv_sec != 0 or new_value.it_value.tv_nsec != 0);
            return 0;
        }
        return -1;
    }
}

/// Get current timer settings
pub export fn timerfd_gettime(fd: c_int, curr_value: *itimerspec) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.timerfd_gettime, .{
            @as(usize, @intCast(fd)),
            @intFromPtr(curr_value),
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return 0;
    } else {
        // Fallback
        if (fd < 1000) return -1;
        const idx: usize = @intCast(fd - 1000);
        if (idx >= timer_states.len) return -1;

        if (timer_states[idx]) |state| {
            curr_value.* = state.spec;
            return 0;
        }

        @memset(std.mem.asBytes(curr_value), 0);
        return -1;
    }
}

/// Close a timer fd (for fallback implementation)
pub fn timerfd_close(fd: c_int) c_int {
    if (fd >= 1000) {
        const idx: usize = @intCast(fd - 1000);
        if (idx < timer_states.len) {
            timer_states[idx] = null;
            return 0;
        }
    }
    return -1;
}
