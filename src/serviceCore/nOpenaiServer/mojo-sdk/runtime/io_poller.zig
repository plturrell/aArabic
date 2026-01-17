// I/O Poller for Mojo Runtime
// Wraps kqueue (macOS/BSD) and epoll (Linux) for event multiplexing

const std = @import("std");
const builtin = @import("builtin");
const os = std.os;
const posix = std.posix;
const IoUring = @import("io_uring.zig").Ring;

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
    ring: if (builtin.os.tag == .linux) IoUring else void,
    
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
                .ring = {},
                .events = undefined,
            };
        } else if (builtin.os.tag == .linux) {
            // Try to initialize io_uring, fall back to epoll if needed or just use it.
            // For this implementation, we use our IoUring wrapper.
            // fd is used for epoll in this branch usually, but we'll use 0 or ring's fd if mixed.
            // To fit the existing struct, we'll initialize the ring.
            
            // Note: In a real hybrid, we'd check kernel version.
            const ring = try IoUring.init(128, 0); // 128 entries
            
            // We still create epoll for compatibility if we want to support both or mixed,
            // but let's assume we replace epoll completely for this "World Class" step.
            // However, Poller struct has 'fd' which usually stores the epoll fd.
            // We can store ring.fd there.
            
            return Poller{
                .fd = ring.fd, // Use ring fd as the main handle
                .ring = ring,
                .events = undefined, // Unused in pure uring mode or used differently
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
        if (builtin.os.tag == .linux) {
            self.ring.deinit();
        } else {
            posix.close(self.fd);
        }
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
            // For io_uring, we submit a POLL_ADD opcode.
            // In a full implementation we'd get an SQE, set it up, and submit.
            // self.ring.submit_poll_add(fd, event_type, user_data);
            
            // Placeholder: The io_uring.zig wrapper needs methods to get SQE.
            // For now we assume the wrapper handles it or we call a helper.
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
            // For io_uring:
            // 1. Submit any pending SQEs
            // 2. Wait for CQEs
            // self.ring.submit_and_wait(timeout_ms)
            
            return 0; // Placeholder until io_uring.zig has full implementation
        }
        return 0;
    }
};
