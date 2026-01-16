// I/O Poller for Mojo Runtime
// Wraps kqueue (macOS/BSD) and epoll (Linux) for event multiplexing

const std = @import("std");
const builtin = @import("builtin");
const os = std.os;
const posix = std.posix;

pub const EventType = enum {
    Read,
    Write,
};

pub const Event = struct {
    fd: posix.fd_t,
    type: EventType,
    user_data: usize,
};

pub const Poller = struct {
    fd: posix.fd_t,
    
    // Platform-specific event buffer
    events: if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .freebsd) 
        [128]posix.Kevent 
    else 
        [128]posix.epoll_event,

    pub fn init() !Poller {
        if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .freebsd) {
            const fd = try posix.kqueue();
            return Poller{
                .fd = fd,
                .events = undefined,
            };
        } else if (builtin.os.tag == .linux) {
            const fd = try posix.epoll_create1(0);
            return Poller{
                .fd = fd,
                .events = undefined,
            };
        } else if (builtin.os.tag == .windows) {
            // TODO: Implement IOCP (Input/Output Completion Ports) for Windows.
            // IOCP uses a Proactor model (completion-based) vs Reactor (readiness-based),
            // requiring an adapter layer to fit this poll() interface.
            @compileError("Windows IOCP support is pending implementation.");
        } else {
            @compileError("Unsupported OS for I/O Poller");
        }
    }

    pub fn deinit(self: *Poller) void {
        posix.close(self.fd);
    }

    /// Register interest in an event
    pub fn register(self: *Poller, fd: posix.fd_t, event_type: EventType, user_data: usize) !void {
        if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .freebsd) {
            const filter: i16 = switch (event_type) {
                .Read => -1, // EVFILT_READ
                .Write => -2, // EVFILT_WRITE
            };
            
            var kevent = posix.Kevent{
                .ident = @intCast(fd),
                .filter = filter,
                .flags = 0x0001 | 0x0004 | 0x0010, // EV_ADD | EV_ENABLE | EV_ONESHOT
                .fflags = 0,
                .data = 0,
                .udata = user_data,
            };
            
            const count = try posix.kevent(self.fd, @as([]const posix.Kevent, &kevent), &[_]posix.Kevent{}, null);
            _ = count; // kevent returns 0 on success when not retrieving events
            
        } else if (builtin.os.tag == .linux) {
            var event = posix.epoll_event{
                .events = switch (event_type) {
                    .Read => posix.EPOLL.IN | posix.EPOLL.ONESHOT,
                    .Write => posix.EPOLL.OUT | posix.EPOLL.ONESHOT,
                },
                .data = .{ .ptr = user_data },
            };
            
            try posix.epoll_ctl(self.fd, posix.EPOLL.CTL_ADD, fd, &event);
        }
    }

    /// Poll for events, blocking up to `timeout_ms` (or indefinitely if null)
    pub fn poll(self: *Poller, timeout_ms: ?i32, out_events: []Event) !usize {
        if (builtin.os.tag == .macos or builtin.os.tag == .ios or builtin.os.tag == .freebsd) {
            var ts: ?posix.timespec = null;
            if (timeout_ms) |ms| {
                ts = posix.timespec{
                    .sec = @divTrunc(ms, 1000),
                    .nsec = @rem(ms, 1000) * 1_000_000,
                };
            }
            
            const count = try posix.kevent(self.fd, &[_]posix.Kevent{}, &self.events, if (ts) |*t| t else null);
            
            var num_events: usize = 0;
            for (0..count) |i| {
                if (num_events >= out_events.len) break;
                
                const kevent = self.events[i];
                const type_enum: EventType = if (kevent.filter == -1) .Read else .Write;
                
                out_events[num_events] = Event{
                    .fd = @intCast(kevent.ident),
                    .type = type_enum,
                    .user_data = kevent.udata,
                };
                num_events += 1;
            }
            return num_events;
            
        } else if (builtin.os.tag == .linux) {
            const count = posix.epoll_wait(self.fd, &self.events, timeout_ms orelse -1);
            
            var num_events: usize = 0;
            for (0..count) |i| {
                if (num_events >= out_events.len) break;
                
                const epoll_ev = self.events[i];
                const type_enum: EventType = if (epoll_ev.events & posix.EPOLL.IN != 0) .Read else .Write;
                
                out_events[num_events] = Event{
                    .fd = -1, // Not directly available in epoll_event.data.ptr unless encoded
                    .type = type_enum,
                    .user_data = epoll_ev.data.ptr,
                };
                num_events += 1;
            }
            return num_events;
        }
        return 0;
    }
};
