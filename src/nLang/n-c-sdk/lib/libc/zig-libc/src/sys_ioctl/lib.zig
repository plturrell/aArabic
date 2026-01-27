// sys/ioctl module - Phase 1.14
// I/O control operations using real syscalls
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

// Terminal ioctl commands (Linux values, macOS differs)
pub const TIOCGWINSZ: c_ulong = if (builtin.os.tag == .macos) 0x40087468 else 0x5413;
pub const TIOCSWINSZ: c_ulong = if (builtin.os.tag == .macos) 0x80087467 else 0x5414;
pub const TIOCGPGRP: c_ulong = if (builtin.os.tag == .macos) 0x40047477 else 0x540F;
pub const TIOCSPGRP: c_ulong = if (builtin.os.tag == .macos) 0x80047476 else 0x5410;
pub const TIOCNOTTY: c_ulong = if (builtin.os.tag == .macos) 0x20007471 else 0x5422;
pub const TIOCSCTTY: c_ulong = if (builtin.os.tag == .macos) 0x20007461 else 0x540E;

// File ioctl commands
pub const FIONREAD: c_ulong = if (builtin.os.tag == .macos) 0x4004667F else 0x541B;
pub const FIONBIO: c_ulong = if (builtin.os.tag == .macos) 0x8004667E else 0x5421;
pub const FIOCLEX: c_ulong = if (builtin.os.tag == .macos) 0x20006601 else 0x5451;
pub const FIONCLEX: c_ulong = if (builtin.os.tag == .macos) 0x20006602 else 0x5450;

// Socket ioctl commands
pub const SIOCGIFADDR: c_ulong = 0x8915;
pub const SIOCGIFFLAGS: c_ulong = 0x8913;
pub const SIOCSIFFLAGS: c_ulong = 0x8914;
pub const SIOCGIFCONF: c_ulong = 0x8912;

pub const winsize = extern struct {
    ws_row: c_ushort,
    ws_col: c_ushort,
    ws_xpixel: c_ushort,
    ws_ypixel: c_ushort,
};

/// ioctl - control device
/// This is a variadic function, but we handle it by taking an optional pointer argument
pub export fn ioctl(fd: c_int, request: c_ulong, arg: ?*anyopaque) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.ioctl, .{
            @as(usize, @intCast(fd)),
            @as(usize, @intCast(request)),
            @intFromPtr(arg),
        });
        if (@as(isize, @bitCast(rc)) < 0) {
            return -1;
        }
        return @intCast(rc);
    } else if (builtin.os.tag == .macos or builtin.os.tag == .ios) {
        const rc = std.os.darwin.syscall(.ioctl, .{
            @as(usize, @intCast(fd)),
            @as(usize, @intCast(request)),
            @intFromPtr(arg),
        });
        if (@as(isize, @bitCast(rc)) < 0) {
            return -1;
        }
        return @intCast(rc);
    } else {
        // Fallback for unsupported platforms
        return handleIoctlFallback(fd, request, arg);
    }
}

fn handleIoctlFallback(fd: c_int, request: c_ulong, arg: ?*anyopaque) c_int {
    _ = fd;

    // Handle common requests with sensible defaults
    if (request == TIOCGWINSZ) {
        if (arg) |a| {
            const ws: *winsize = @ptrCast(@alignCast(a));
            ws.ws_row = 24;
            ws.ws_col = 80;
            ws.ws_xpixel = 0;
            ws.ws_ypixel = 0;
            return 0;
        }
    } else if (request == FIONREAD) {
        if (arg) |a| {
            const count: *c_int = @ptrCast(@alignCast(a));
            count.* = 0;
            return 0;
        }
    } else if (request == FIONBIO) {
        // Non-blocking mode - just acknowledge
        return 0;
    }

    return -1; // ENOTTY
}

// Additional helper for getting terminal size
pub fn getTerminalSize(fd: c_int) ?winsize {
    var ws: winsize = undefined;
    if (ioctl(fd, TIOCGWINSZ, @ptrCast(&ws)) == 0) {
        return ws;
    }
    return null;
}

// Helper for setting non-blocking mode
pub fn setNonBlocking(fd: c_int, enable: bool) c_int {
    var val: c_int = if (enable) 1 else 0;
    return ioctl(fd, FIONBIO, @ptrCast(&val));
}

// Helper for getting bytes available to read
pub fn getBytesAvailable(fd: c_int) c_int {
    var count: c_int = 0;
    if (ioctl(fd, FIONREAD, @ptrCast(&count)) == 0) {
        return count;
    }
    return -1;
}
