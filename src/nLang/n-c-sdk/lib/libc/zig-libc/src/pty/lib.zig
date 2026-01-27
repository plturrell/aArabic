// pty module - Pseudo-terminal - Phase 1.28
// Implements openpty, forkpty, login_tty using POSIX pty functions
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const fs = std.fs;

const ioctl_mod = @import("../sys_ioctl/lib.zig");

/// Open a pseudo-terminal pair
pub export fn openpty(amaster: *c_int, aslave: *c_int, name: ?[*:0]u8, termp: ?*const anyopaque, winp: ?*const anyopaque) c_int {
    // Open the master side using posix_openpt
    const master_fd = posix_openpt(O_RDWR | O_NOCTTY);
    if (master_fd < 0) return -1;

    // Grant access to slave
    if (grantpt(master_fd) != 0) {
        _ = posix.close(@intCast(master_fd));
        return -1;
    }

    // Unlock slave
    if (unlockpt(master_fd) != 0) {
        _ = posix.close(@intCast(master_fd));
        return -1;
    }

    // Get slave name
    var pts_name_buf: [128]u8 = undefined;
    const pts_name = ptsname_r(master_fd, &pts_name_buf, pts_name_buf.len);
    if (pts_name == null) {
        _ = posix.close(@intCast(master_fd));
        return -1;
    }

    // Open slave side
    const slave_fd = openSlave(pts_name.?);
    if (slave_fd < 0) {
        _ = posix.close(@intCast(master_fd));
        return -1;
    }

    // Set terminal attributes if provided
    if (termp) |t| {
        _ = tcsetattr(slave_fd, 0, t);
    }

    // Set window size if provided
    if (winp) |w| {
        _ = ioctl_mod.ioctl(slave_fd, ioctl_mod.TIOCSWINSZ, @constCast(w));
    }

    // Copy name if buffer provided
    if (name) |n| {
        const pn = pts_name.?;
        var i: usize = 0;
        while (pn[i] != 0 and i < 127) : (i += 1) {
            n[i] = pn[i];
        }
        n[i] = 0;
    }

    amaster.* = master_fd;
    aslave.* = slave_fd;

    return 0;
}

/// Fork with a new pseudo-terminal
pub export fn forkpty(amaster: *c_int, name: ?[*:0]u8, termp: ?*const anyopaque, winp: ?*const anyopaque) c_int {
    var master_fd: c_int = undefined;
    var slave_fd: c_int = undefined;

    if (openpty(&master_fd, &slave_fd, name, termp, winp) != 0) {
        return -1;
    }

    const pid = posix.fork() catch {
        _ = posix.close(@intCast(master_fd));
        _ = posix.close(@intCast(slave_fd));
        return -1;
    };

    if (pid == 0) {
        // Child process
        _ = posix.close(@intCast(master_fd));

        // Create new session and set controlling terminal
        if (login_tty(slave_fd) != 0) {
            std.process.exit(1);
        }

        return 0;
    } else {
        // Parent process
        _ = posix.close(@intCast(slave_fd));
        amaster.* = master_fd;
        return @intCast(pid);
    }
}

/// Make fd the controlling terminal
pub export fn login_tty(fd: c_int) c_int {
    // Create new session
    _ = posix.setsid() catch return -1;

    // Set controlling terminal
    if (ioctl_mod.ioctl(fd, ioctl_mod.TIOCSCTTY, null) < 0) {
        // Some systems require 0 as arg
        var zero: c_int = 0;
        if (ioctl_mod.ioctl(fd, ioctl_mod.TIOCSCTTY, @ptrCast(&zero)) < 0) {
            return -1;
        }
    }

    // Duplicate fd to stdin, stdout, stderr
    const fd_u: posix.fd_t = @intCast(fd);
    _ = posix.dup2(fd_u, 0) catch return -1;
    _ = posix.dup2(fd_u, 1) catch return -1;
    _ = posix.dup2(fd_u, 2) catch return -1;

    // Close original fd if not one of the standard fds
    if (fd > 2) {
        _ = posix.close(fd_u);
    }

    return 0;
}

// Constants
const O_RDWR: c_int = 0x02;
const O_NOCTTY: c_int = if (builtin.os.tag == .macos) 0x20000 else 0x100;

// POSIX pty functions
fn posix_openpt(flags: c_int) c_int {
    if (builtin.os.tag == .linux) {
        // Linux: open /dev/ptmx
        const file = fs.openFileAbsolute("/dev/ptmx", .{ .mode = .read_write }) catch return -1;
        return file.handle;
    } else if (builtin.os.tag == .macos) {
        // macOS: use posix_openpt syscall
        const rc = std.os.darwin.syscall(.posix_openpt, .{@as(usize, @intCast(flags))});
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return @intCast(rc);
    } else {
        return -1;
    }
}

fn grantpt(fd: c_int) c_int {
    // On modern Linux/macOS, this is typically a no-op or automatic
    _ = fd;
    return 0;
}

fn unlockpt(fd: c_int) c_int {
    if (builtin.os.tag == .linux) {
        // Linux: ioctl TIOCSPTLCK with 0 to unlock
        var unlock: c_int = 0;
        const TIOCSPTLCK: c_ulong = 0x40045431;
        return ioctl_mod.ioctl(fd, TIOCSPTLCK, @ptrCast(&unlock));
    }
    // macOS: automatic
    return 0;
}

fn ptsname_r(fd: c_int, buf: [*]u8, buflen: usize) ?[*:0]u8 {
    if (builtin.os.tag == .linux) {
        // Linux: ioctl TIOCGPTN to get pty number
        var pty_num: c_uint = 0;
        const TIOCGPTN: c_ulong = 0x80045430;
        if (ioctl_mod.ioctl(fd, TIOCGPTN, @ptrCast(&pty_num)) < 0) return null;

        const written = std.fmt.bufPrint(buf[0..buflen], "/dev/pts/{d}", .{pty_num}) catch return null;
        buf[written.len] = 0;
        return @ptrCast(buf);
    } else if (builtin.os.tag == .macos) {
        // macOS: use ioctl to get device name
        if (buflen < 128) return null;
        var name_buf: [128]u8 = undefined;
        const TIOCPTYGNAME: c_ulong = 0x40807453;
        if (ioctl_mod.ioctl(fd, TIOCPTYGNAME, @ptrCast(&name_buf)) < 0) return null;
        @memcpy(buf[0..128], &name_buf);
        return @ptrCast(buf);
    }
    return null;
}

fn openSlave(name: [*:0]const u8) c_int {
    const path = std.mem.span(name);
    const file = fs.openFileAbsolute(path, .{ .mode = .read_write }) catch return -1;
    return file.handle;
}

fn tcsetattr(fd: c_int, action: c_int, termios_p: *const anyopaque) c_int {
    _ = fd;
    _ = action;
    _ = termios_p;
    // Would need full termios implementation
    return 0;
}
