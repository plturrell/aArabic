// sys_epoll module - Phase 1.29
// Real epoll implementation using Linux syscalls (with kqueue fallback for macOS)
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

pub const EPOLL_CTL_ADD: c_int = 1;
pub const EPOLL_CTL_DEL: c_int = 2;
pub const EPOLL_CTL_MOD: c_int = 3;

// Event types
pub const EPOLLIN: u32 = 0x001;
pub const EPOLLPRI: u32 = 0x002;
pub const EPOLLOUT: u32 = 0x004;
pub const EPOLLERR: u32 = 0x008;
pub const EPOLLHUP: u32 = 0x010;
pub const EPOLLRDNORM: u32 = 0x040;
pub const EPOLLRDBAND: u32 = 0x080;
pub const EPOLLWRNORM: u32 = 0x100;
pub const EPOLLWRBAND: u32 = 0x200;
pub const EPOLLMSG: u32 = 0x400;
pub const EPOLLRDHUP: u32 = 0x2000;
pub const EPOLLEXCLUSIVE: u32 = 1 << 28;
pub const EPOLLWAKEUP: u32 = 1 << 29;
pub const EPOLLONESHOT: u32 = 1 << 30;
pub const EPOLLET: u32 = 1 << 31;

// Flags for epoll_create1
pub const EPOLL_CLOEXEC: c_int = 0x80000;

pub const epoll_data = extern union {
    ptr: ?*anyopaque,
    fd: c_int,
    u32_val: u32,
    u64_val: u64,
};

pub const epoll_event = extern struct {
    events: u32,
    data: epoll_data,
};

/// Create epoll instance
pub export fn epoll_create(size: c_int) c_int {
    if (size <= 0) return -1;
    return epoll_create1(0);
}

/// Create epoll instance with flags
pub export fn epoll_create1(flags: c_int) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.epoll_create1, .{@as(usize, @intCast(flags))});
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return @intCast(rc);
    } else {
        // macOS/BSD: use kqueue as backend
        const kq = posix.kqueue() catch return -1;
        return @intCast(kq);
    }
}

/// Control epoll instance
pub export fn epoll_ctl(epfd: c_int, op: c_int, fd: c_int, event: ?*epoll_event) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.epoll_ctl, .{
            @as(usize, @intCast(epfd)),
            @as(usize, @intCast(op)),
            @as(usize, @intCast(fd)),
            @intFromPtr(event),
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return 0;
    } else {
        // macOS: translate to kevent
        return epollCtlKqueue(epfd, op, fd, event);
    }
}

fn epollCtlKqueue(kq: c_int, op: c_int, fd: c_int, event: ?*epoll_event) c_int {
    var changelist: [2]posix.Kevent = undefined;
    var nchanges: usize = 0;

    const ev = event orelse return -1;

    // Determine filter flags
    const flags: u16 = switch (op) {
        EPOLL_CTL_ADD => posix.system.EV_ADD,
        EPOLL_CTL_DEL => posix.system.EV_DELETE,
        EPOLL_CTL_MOD => posix.system.EV_ADD,
        else => return -1,
    };

    // Convert epoll events to kqueue filters
    if ((ev.events & EPOLLIN) != 0) {
        changelist[nchanges] = .{
            .ident = @intCast(fd),
            .filter = posix.system.EVFILT_READ,
            .flags = flags,
            .fflags = 0,
            .data = 0,
            .udata = @ptrFromInt(ev.data.u64_val),
        };
        nchanges += 1;
    }

    if ((ev.events & EPOLLOUT) != 0) {
        changelist[nchanges] = .{
            .ident = @intCast(fd),
            .filter = posix.system.EVFILT_WRITE,
            .flags = flags,
            .fflags = 0,
            .data = 0,
            .udata = @ptrFromInt(ev.data.u64_val),
        };
        nchanges += 1;
    }

    if (nchanges == 0) return 0;

    const kq_fd: posix.fd_t = @intCast(kq);
    _ = posix.kevent(kq_fd, changelist[0..nchanges], &[_]posix.Kevent{}, null) catch return -1;

    return 0;
}

/// Wait for events
pub export fn epoll_wait(epfd: c_int, events: [*]epoll_event, maxevents: c_int, timeout: c_int) c_int {
    return epoll_pwait(epfd, events, maxevents, timeout, null);
}

/// Wait for events with signal mask
pub export fn epoll_pwait(epfd: c_int, events: [*]epoll_event, maxevents: c_int, timeout: c_int, sigmask: ?*const anyopaque) c_int {
    if (maxevents <= 0) return -1;

    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.epoll_pwait, .{
            @as(usize, @intCast(epfd)),
            @intFromPtr(events),
            @as(usize, @intCast(maxevents)),
            @as(usize, @bitCast(@as(isize, timeout))),
            @intFromPtr(sigmask),
            @as(usize, 8), // sigset size
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return @intCast(rc);
    } else {
        // macOS: use kevent
        return epollWaitKqueue(epfd, events, maxevents, timeout);
    }
}

fn epollWaitKqueue(kq: c_int, events: [*]epoll_event, maxevents: c_int, timeout: c_int) c_int {
    const max: usize = @intCast(maxevents);

    // Create kevent buffer on stack (limited size)
    var kevents: [64]posix.Kevent = undefined;
    const actual_max = @min(max, 64);

    // Convert timeout
    const ts: ?posix.timespec = if (timeout < 0)
        null
    else
        .{
            .sec = @divTrunc(timeout, 1000),
            .nsec = @rem(timeout, 1000) * 1000000,
        };

    const kq_fd: posix.fd_t = @intCast(kq);
    const nready = posix.kevent(kq_fd, &[_]posix.Kevent{}, kevents[0..actual_max], ts) catch return -1;

    // Convert kevent results to epoll_event
    for (kevents[0..nready], 0..) |kev, i| {
        var ev_events: u32 = 0;

        if (kev.filter == posix.system.EVFILT_READ) {
            ev_events |= EPOLLIN;
        }
        if (kev.filter == posix.system.EVFILT_WRITE) {
            ev_events |= EPOLLOUT;
        }
        if ((kev.flags & posix.system.EV_EOF) != 0) {
            ev_events |= EPOLLHUP;
        }
        if ((kev.flags & posix.system.EV_ERROR) != 0) {
            ev_events |= EPOLLERR;
        }

        events[i].events = ev_events;
        events[i].data.u64_val = @intFromPtr(kev.udata);
    }

    return @intCast(nready);
}

/// Close epoll instance
pub export fn epoll_close(epfd: c_int) c_int {
    posix.close(@intCast(epfd));
    return 0;
}
