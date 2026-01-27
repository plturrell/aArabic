// system module - Phase 1.11 Priority 9 - System Utilities (poll, termios, syslog, etc.)
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

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

// ============ POLL.H (3 functions) ============

pub const pollfd = extern struct {
    fd: c_int,
    events: c_short,
    revents: c_short,
};

pub const nfds_t = c_ulong;

pub const POLLIN: c_short = 0x001;
pub const POLLPRI: c_short = 0x002;
pub const POLLOUT: c_short = 0x004;
pub const POLLERR: c_short = 0x008;
pub const POLLHUP: c_short = 0x010;
pub const POLLNVAL: c_short = 0x020;

pub export fn poll(fds: [*]pollfd, nfds: nfds_t, timeout: c_int) c_int {
    const rc = std.posix.system.poll(@ptrCast(fds), nfds, timeout);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn ppoll(fds: [*]pollfd, nfds: nfds_t, timeout: ?*const timespec, sigmask: ?*const anyopaque) c_int {
    const rc = std.posix.system.ppoll(@ptrCast(fds), nfds, @ptrCast(timeout), @ptrCast(sigmask), 8);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

// ============ TERMIOS.H (15 functions) ============

pub const termios = extern struct {
    c_iflag: c_uint,
    c_oflag: c_uint,
    c_cflag: c_uint,
    c_lflag: c_uint,
    c_line: u8,
    c_cc: [32]u8,
    c_ispeed: c_uint,
    c_ospeed: c_uint,
};

pub const tcflag_t = c_uint;
pub const cc_t = u8;
pub const speed_t = c_uint;

// c_iflag
pub const IGNBRK: tcflag_t = 0x001;
pub const BRKINT: tcflag_t = 0x002;
pub const IGNPAR: tcflag_t = 0x004;
pub const PARMRK: tcflag_t = 0x008;
pub const INPCK: tcflag_t = 0x010;
pub const ISTRIP: tcflag_t = 0x020;
pub const INLCR: tcflag_t = 0x040;
pub const IGNCR: tcflag_t = 0x080;
pub const ICRNL: tcflag_t = 0x100;
pub const IXON: tcflag_t = 0x400;

// c_oflag
pub const OPOST: tcflag_t = 0x001;
pub const ONLCR: tcflag_t = 0x004;

// c_cflag
pub const CSIZE: tcflag_t = 0x030;
pub const CS5: tcflag_t = 0x000;
pub const CS6: tcflag_t = 0x010;
pub const CS7: tcflag_t = 0x020;
pub const CS8: tcflag_t = 0x030;
pub const CSTOPB: tcflag_t = 0x040;
pub const CREAD: tcflag_t = 0x080;
pub const PARENB: tcflag_t = 0x100;
pub const PARODD: tcflag_t = 0x200;
pub const HUPCL: tcflag_t = 0x400;
pub const CLOCAL: tcflag_t = 0x800;

// c_lflag
pub const ISIG: tcflag_t = 0x001;
pub const ICANON: tcflag_t = 0x002;
pub const ECHO: tcflag_t = 0x008;
pub const ECHOE: tcflag_t = 0x010;
pub const ECHOK: tcflag_t = 0x020;
pub const ECHONL: tcflag_t = 0x040;
pub const NOFLSH: tcflag_t = 0x080;
pub const TOSTOP: tcflag_t = 0x100;
pub const IEXTEN: tcflag_t = 0x8000;

// c_cc indices
pub const VINTR: usize = 0;
pub const VQUIT: usize = 1;
pub const VERASE: usize = 2;
pub const VKILL: usize = 3;
pub const VEOF: usize = 4;
pub const VTIME: usize = 5;
pub const VMIN: usize = 6;
pub const VSTART: usize = 8;
pub const VSTOP: usize = 9;
pub const VSUSP: usize = 10;

// tcsetattr actions
pub const TCSANOW: c_int = 0;
pub const TCSADRAIN: c_int = 1;
pub const TCSAFLUSH: c_int = 2;

// tcflush queue_selector
pub const TCIFLUSH: c_int = 0;
pub const TCOFLUSH: c_int = 1;
pub const TCIOFLUSH: c_int = 2;

// tcflow action
pub const TCOOFF: c_int = 0;
pub const TCOON: c_int = 1;
pub const TCIOFF: c_int = 2;
pub const TCION: c_int = 3;

// Speeds
pub const B0: speed_t = 0;
pub const B50: speed_t = 50;
pub const B75: speed_t = 75;
pub const B110: speed_t = 110;
pub const B134: speed_t = 134;
pub const B150: speed_t = 150;
pub const B200: speed_t = 200;
pub const B300: speed_t = 300;
pub const B600: speed_t = 600;
pub const B1200: speed_t = 1200;
pub const B1800: speed_t = 1800;
pub const B2400: speed_t = 2400;
pub const B4800: speed_t = 4800;
pub const B9600: speed_t = 9600;
pub const B19200: speed_t = 19200;
pub const B38400: speed_t = 38400;

pub export fn tcgetattr(fd: c_int, termios_p: *termios) c_int {
    const rc = std.posix.system.tcgetattr(fd, @ptrCast(termios_p));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn tcsetattr(fd: c_int, optional_actions: c_int, termios_p: *const termios) c_int {
    const rc = std.posix.system.tcsetattr(fd, optional_actions, @ptrCast(termios_p));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn tcsendbreak(fd: c_int, duration: c_int) c_int {
    _ = duration;
    const rc = std.posix.system.tcsendbreak(fd, 0);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn tcdrain(fd: c_int) c_int {
    const rc = std.posix.system.tcdrain(fd);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn tcflush(fd: c_int, queue_selector: c_int) c_int {
    const rc = std.posix.system.tcflush(fd, queue_selector);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn tcflow(fd: c_int, action: c_int) c_int {
    const rc = std.posix.system.tcflow(fd, action);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn cfgetispeed(termios_p: *const termios) speed_t {
    return termios_p.c_ispeed;
}

pub export fn cfgetospeed(termios_p: *const termios) speed_t {
    return termios_p.c_ospeed;
}

pub export fn cfsetispeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ispeed = speed;
    return 0;
}

pub export fn cfsetospeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ospeed = speed;
    return 0;
}

pub export fn cfsetspeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ispeed = speed;
    termios_p.c_ospeed = speed;
    return 0;
}

pub export fn cfmakeraw(termios_p: *termios) void {
    termios_p.c_iflag &= ~(IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    termios_p.c_oflag &= ~OPOST;
    termios_p.c_lflag &= ~(ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    termios_p.c_cflag &= ~(CSIZE | PARENB);
    termios_p.c_cflag |= CS8;
    termios_p.c_cc[VMIN] = 1;
    termios_p.c_cc[VTIME] = 0;
}

// ============ SYSLOG.H (8 functions) ============

pub const LOG_EMERG: c_int = 0;
pub const LOG_ALERT: c_int = 1;
pub const LOG_CRIT: c_int = 2;
pub const LOG_ERR: c_int = 3;
pub const LOG_WARNING: c_int = 4;
pub const LOG_NOTICE: c_int = 5;
pub const LOG_INFO: c_int = 6;
pub const LOG_DEBUG: c_int = 7;

pub const LOG_KERN: c_int = 0 << 3;
pub const LOG_USER: c_int = 1 << 3;
pub const LOG_MAIL: c_int = 2 << 3;
pub const LOG_DAEMON: c_int = 3 << 3;
pub const LOG_AUTH: c_int = 4 << 3;
pub const LOG_SYSLOG: c_int = 5 << 3;
pub const LOG_LPR: c_int = 6 << 3;
pub const LOG_NEWS: c_int = 7 << 3;
pub const LOG_UUCP: c_int = 8 << 3;
pub const LOG_CRON: c_int = 9 << 3;
pub const LOG_AUTHPRIV: c_int = 10 << 3;
pub const LOG_LOCAL0: c_int = 16 << 3;

pub const LOG_PID: c_int = 0x01;
pub const LOG_CONS: c_int = 0x02;
pub const LOG_ODELAY: c_int = 0x04;
pub const LOG_NDELAY: c_int = 0x08;
pub const LOG_NOWAIT: c_int = 0x10;
pub const LOG_PERROR: c_int = 0x20;

var syslog_ident: ?[*:0]const u8 = null;
var syslog_option: c_int = 0;
var syslog_facility: c_int = LOG_USER;
var syslog_mask: c_int = 0xff;

pub export fn openlog(ident: ?[*:0]const u8, option: c_int, facility: c_int) void {
    syslog_ident = ident;
    syslog_option = option;
    syslog_facility = facility;
}

pub export fn syslog(priority: c_int, format: [*:0]const u8, ...) void {
    _ = priority; _ = format;
    // Simplified: would write to system log
}

pub export fn closelog() void {
    syslog_ident = null;
}

pub export fn setlogmask(mask: c_int) c_int {
    const old = syslog_mask;
    if (mask != 0) {
        syslog_mask = mask;
    }
    return old;
}

pub export fn vsyslog(priority: c_int, format: [*:0]const u8, ap: *anyopaque) void {
    _ = priority; _ = format; _ = ap;
}

// ============ MISC SYSTEM UTILITIES (33 functions) ============

// uname
pub const utsname = extern struct {
    sysname: [65]u8,
    nodename: [65]u8,
    release: [65]u8,
    version: [65]u8,
    machine: [65]u8,
};

pub export fn uname(buf: *utsname) c_int {
    const rc = std.posix.system.uname(@ptrCast(buf));
    if (failIfErrno(rc)) return -1;
    return 0;
}

// ioctl
pub export fn ioctl(fd: c_int, request: c_ulong, ...) c_int {
    _ = fd; _ = request;
    setErrno(.NOSYS);
    return -1;
}

// sysconf
pub const _SC_ARG_MAX: c_int = 0;
pub const _SC_CHILD_MAX: c_int = 1;
pub const _SC_CLK_TCK: c_int = 2;
pub const _SC_NGROUPS_MAX: c_int = 3;
pub const _SC_OPEN_MAX: c_int = 4;
pub const _SC_PAGESIZE: c_int = 30;
pub const _SC_PAGE_SIZE: c_int = 30;
pub const _SC_NPROCESSORS_ONLN: c_int = 84;

pub export fn sysconf(name: c_int) c_long {
    return switch (name) {
        _SC_PAGESIZE, _SC_PAGE_SIZE => 4096,
        _SC_CLK_TCK => 100,
        _SC_OPEN_MAX => 1024,
        _SC_NPROCESSORS_ONLN => 8,
        else => -1,
    };
}

// pathconf
pub export fn pathconf(path: [*:0]const u8, name: c_int) c_long {
    _ = path; _ = name;
    return -1;
}

pub export fn fpathconf(fd: c_int, name: c_int) c_long {
    _ = fd; _ = name;
    return -1;
}

// confstr
pub export fn confstr(name: c_int, buf: ?[*]u8, len: usize) usize {
    _ = name; _ = buf; _ = len;
    return 0;
}

// getloadavg
pub export fn getloadavg(loadavg: [*]f64, nelem: c_int) c_int {
    for (0..@intCast(nelem)) |i| {
        loadavg[i] = 0.0;
    }
    return nelem;
}

// daemon
pub export fn daemon(nochdir: c_int, noclose: c_int) c_int {
    _ = nochdir; _ = noclose;
    setErrno(.NOSYS);
    return -1;
}

// sync
pub export fn sync() void {
    _ = std.posix.system.sync();
}

pub export fn syncfs(fd: c_int) c_int {
    _ = fd;
    sync();
    return 0;
}

pub export fn fsync(fd: c_int) c_int {
    const rc = std.posix.system.fsync(fd);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn fdatasync(fd: c_int) c_int {
    const rc = std.posix.system.fdatasync(fd);
    if (failIfErrno(rc)) return -1;
    return 0;
}

// reboot
pub export fn reboot(cmd: c_int) c_int {
    _ = cmd;
    setErrno(.PERM);
    return -1;
}

// getpagesize
pub export fn getpagesize() c_int {
    return 4096;
}

// getdtablesize  
pub export fn getdtablesize() c_int {
    return 1024;
}

// gethostname
pub export fn gethostname(name: [*]u8, len: usize) c_int {
    const hostname = "localhost";
    const copy_len = @min(len - 1, hostname.len);
    @memcpy(name[0..copy_len], hostname[0..copy_len]);
    name[copy_len] = 0;
    return 0;
}

pub export fn sethostname(name: [*:0]const u8, len: usize) c_int {
    _ = name; _ = len;
    setErrno(.PERM);
    return -1;
}

// getdomainname
pub export fn getdomainname(name: [*]u8, len: usize) c_int {
    if (len > 0) {
        name[0] = 0;
    }
    return 0;
}

pub export fn setdomainname(name: [*:0]const u8, len: usize) c_int {
    _ = name; _ = len;
    setErrno(.PERM);
    return -1;
}

// getgroups/setgroups
pub export fn getgroups(size: c_int, list: [*]c_uint) c_int {
    _ = size; _ = list;
    return 0;
}

pub export fn setgroups(size: usize, list: [*]const c_uint) c_int {
    _ = size; _ = list;
    setErrno(.PERM);
    return -1;
}

// initgroups
pub export fn initgroups(user: [*:0]const u8, group: c_uint) c_int {
    _ = user; _ = group;
    setErrno(.PERM);
    return -1;
}

// setregid/setreuid
pub export fn setregid(rgid: c_uint, egid: c_uint) c_int {
    _ = rgid; _ = egid;
    setErrno(.PERM);
    return -1;
}

pub export fn setreuid(ruid: c_uint, euid: c_uint) c_int {
    _ = ruid; _ = euid;
    setErrno(.PERM);
    return -1;
}

// seteuid/setegid
pub export fn seteuid(euid: c_uint) c_int {
    _ = euid;
    setErrno(.PERM);
    return -1;
}

pub export fn setegid(egid: c_uint) c_int {
    _ = egid;
    setErrno(.PERM);
    return -1;
}

// getlogin
pub export fn getlogin() ?[*:0]u8 {
    return @constCast("user");
}

pub export fn getlogin_r(name: [*]u8, namesize: usize) c_int {
    const username = "user";
    const copy_len = @min(namesize - 1, username.len);
    @memcpy(name[0..copy_len], username[0..copy_len]);
    name[copy_len] = 0;
    return 0;
}

// truncate/ftruncate
pub export fn truncate(path: [*:0]const u8, length: i64) c_int {
    const rc = std.posix.system.truncate(path, length);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn ftruncate(fd: c_int, length: i64) c_int {
    const rc = std.posix.system.ftruncate(fd, length);
    if (failIfErrno(rc)) return -1;
    return 0;
}

// lockf
pub export fn lockf(fd: c_int, cmd: c_int, len: i64) c_int {
    _ = fd; _ = cmd; _ = len;
    return 0; // Simplified
}

// gethostid/sethostid
pub export fn gethostid() c_long {
    return 0;
}

pub export fn sethostid(hostid: c_long) c_int {
    _ = hostid;
    setErrno(.PERM);
    return -1;
}
