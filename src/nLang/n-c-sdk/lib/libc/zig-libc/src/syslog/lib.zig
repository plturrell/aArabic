// syslog module - Phase 1.15
// System logging using /dev/log socket or stderr fallback
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;
const fs = std.fs;
const net = std.net;

// Priority levels
pub const LOG_EMERG: c_int = 0; // System is unusable
pub const LOG_ALERT: c_int = 1; // Action must be taken immediately
pub const LOG_CRIT: c_int = 2; // Critical conditions
pub const LOG_ERR: c_int = 3; // Error conditions
pub const LOG_WARNING: c_int = 4; // Warning conditions
pub const LOG_NOTICE: c_int = 5; // Normal but significant condition
pub const LOG_INFO: c_int = 6; // Informational
pub const LOG_DEBUG: c_int = 7; // Debug-level messages

// Facility codes
pub const LOG_KERN: c_int = 0 << 3; // Kernel messages
pub const LOG_USER: c_int = 1 << 3; // User-level messages
pub const LOG_MAIL: c_int = 2 << 3; // Mail system
pub const LOG_DAEMON: c_int = 3 << 3; // System daemons
pub const LOG_AUTH: c_int = 4 << 3; // Security/auth messages
pub const LOG_SYSLOG: c_int = 5 << 3; // Syslogd internal
pub const LOG_LPR: c_int = 6 << 3; // Line printer
pub const LOG_NEWS: c_int = 7 << 3; // Network news
pub const LOG_UUCP: c_int = 8 << 3; // UUCP
pub const LOG_CRON: c_int = 9 << 3; // Cron daemon
pub const LOG_LOCAL0: c_int = 16 << 3;
pub const LOG_LOCAL1: c_int = 17 << 3;
pub const LOG_LOCAL2: c_int = 18 << 3;
pub const LOG_LOCAL3: c_int = 19 << 3;
pub const LOG_LOCAL4: c_int = 20 << 3;
pub const LOG_LOCAL5: c_int = 21 << 3;
pub const LOG_LOCAL6: c_int = 22 << 3;
pub const LOG_LOCAL7: c_int = 23 << 3;

// Option flags
pub const LOG_PID: c_int = 0x01; // Log the pid with each message
pub const LOG_CONS: c_int = 0x02; // Log to console if /dev/log unavailable
pub const LOG_ODELAY: c_int = 0x04; // Delay open until first syslog()
pub const LOG_NDELAY: c_int = 0x08; // Open connection immediately
pub const LOG_NOWAIT: c_int = 0x10; // Don't wait for child processes
pub const LOG_PERROR: c_int = 0x20; // Also log to stderr

// Macros
pub fn LOG_MASK(pri: c_int) c_int {
    return @as(c_int, 1) << @intCast(pri);
}

pub fn LOG_UPTO(pri: c_int) c_int {
    return (@as(c_int, 1) << @intCast(pri + 1)) - 1;
}

// Internal state
var log_ident: ?[*:0]const u8 = null;
var log_option: c_int = 0;
var log_facility: c_int = LOG_USER;
var log_mask: c_int = 0xFF; // All priorities enabled
var log_opened: bool = false;
var log_fd: ?posix.socket_t = null;

// Syslog socket path
const SYSLOG_PATH = if (builtin.os.tag == .macos) "/var/run/syslog" else "/dev/log";

/// Open connection to system logger
pub export fn openlog(ident: ?[*:0]const u8, option: c_int, facility: c_int) void {
    log_ident = ident;
    log_option = option;
    log_facility = facility;

    if ((option & LOG_NDELAY) != 0) {
        connectLog();
    }

    log_opened = true;
}

/// Generate a log message
pub export fn syslog(priority: c_int, format: [*:0]const u8, args: ?*anyopaque) void {
    _ = args; // Can't use va_list in Zig easily

    // Check mask
    if ((LOG_MASK(priority & 0x7) & log_mask) == 0) return;

    // Use facility from priority if specified, else default
    const fac = if ((priority & ~0x7) != 0) priority & ~0x7 else log_facility;
    const pri = priority & 0x7;

    // Build syslog message
    var buf: [1024]u8 = undefined;
    var msg_len: usize = 0;

    // Format: <priority>message
    const pri_val = fac | pri;
    const header = std.fmt.bufPrint(buf[0..64], "<{d}>", .{pri_val}) catch return;
    msg_len = header.len;

    // Add ident
    if (log_ident) |ident| {
        const ident_str = std.mem.span(ident);
        const ident_len = @min(ident_str.len, 64);
        @memcpy(buf[msg_len .. msg_len + ident_len], ident_str[0..ident_len]);
        msg_len += ident_len;

        // Add PID if requested
        if ((log_option & LOG_PID) != 0) {
            const pid_str = std.fmt.bufPrint(buf[msg_len .. msg_len + 20], "[{d}]", .{std.os.linux.getpid()}) catch "";
            msg_len += pid_str.len;
        }

        buf[msg_len] = ':';
        msg_len += 1;
        buf[msg_len] = ' ';
        msg_len += 1;
    }

    // Add message (just copy format string since we can't use varargs)
    const fmt_str = std.mem.span(format);
    const copy_len = @min(fmt_str.len, buf.len - msg_len - 1);
    @memcpy(buf[msg_len .. msg_len + copy_len], fmt_str[0..copy_len]);
    msg_len += copy_len;

    // Send to syslog
    sendLog(buf[0..msg_len]);

    // Also print to stderr if LOG_PERROR
    if ((log_option & LOG_PERROR) != 0) {
        std.io.getStdErr().writeAll(buf[0..msg_len]) catch {};
        std.io.getStdErr().writeAll("\n") catch {};
    }
}

/// Close connection to system logger
pub export fn closelog() void {
    if (log_fd) |fd| {
        posix.close(fd);
        log_fd = null;
    }
    log_opened = false;
}

/// Set the log priority mask
pub export fn setlogmask(mask: c_int) c_int {
    const old = log_mask;
    if (mask != 0) {
        log_mask = mask;
    }
    return old;
}

/// syslog with va_list (limited implementation)
pub export fn vsyslog(priority: c_int, format: [*:0]const u8, args: ?*anyopaque) void {
    syslog(priority, format, args);
}

fn connectLog() void {
    if (log_fd != null) return;

    // Try to connect to syslog socket
    const sock = posix.socket(posix.AF.UNIX, posix.SOCK.DGRAM, 0) catch return;

    var addr: posix.sockaddr.un = undefined;
    addr.family = posix.AF.UNIX;
    @memset(&addr.path, 0);

    const path_bytes = SYSLOG_PATH;
    @memcpy(addr.path[0..path_bytes.len], path_bytes);

    posix.connect(sock, @ptrCast(&addr), @sizeOf(posix.sockaddr.un)) catch {
        posix.close(sock);
        return;
    };

    log_fd = sock;
}

fn sendLog(msg: []const u8) void {
    // Connect if not connected
    if (log_fd == null) {
        connectLog();
    }

    if (log_fd) |fd| {
        _ = posix.send(fd, msg, 0) catch {
            // If send fails, try reconnect once
            posix.close(fd);
            log_fd = null;
            connectLog();
            if (log_fd) |fd2| {
                _ = posix.send(fd2, msg, 0) catch {};
            }
        };
    } else if ((log_option & LOG_CONS) != 0) {
        // Fall back to console
        const console = fs.openFileAbsolute("/dev/console", .{ .mode = .write_only }) catch return;
        defer console.close();
        _ = console.write(msg) catch {};
        _ = console.write("\n") catch {};
    }
}

// Priority name lookup
const priority_names = [_][]const u8{
    "emerg", "alert", "crit", "err", "warning", "notice", "info", "debug",
};

pub fn getPriorityName(pri: c_int) []const u8 {
    if (pri >= 0 and pri < 8) {
        return priority_names[@intCast(pri)];
    }
    return "unknown";
}
