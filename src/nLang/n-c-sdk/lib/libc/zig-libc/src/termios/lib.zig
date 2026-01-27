// termios module - Phase 1.14 - Real Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const cc_t = u8;
pub const speed_t = c_uint;
pub const tcflag_t = c_uint;

pub const NCCS: usize = 32;

pub const termios = extern struct {
    c_iflag: tcflag_t,
    c_oflag: tcflag_t,
    c_cflag: tcflag_t,
    c_lflag: tcflag_t,
    c_line: cc_t,
    c_cc: [NCCS]cc_t,
    c_ispeed: speed_t,
    c_ospeed: speed_t,
};

// Baud rates
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
pub const B57600: speed_t = 57600;
pub const B115200: speed_t = 115200;

// c_iflag constants
pub const IGNBRK: tcflag_t = 0o000001;
pub const BRKINT: tcflag_t = 0o000002;
pub const IGNPAR: tcflag_t = 0o000004;
pub const PARMRK: tcflag_t = 0o000010;
pub const INPCK: tcflag_t = 0o000020;
pub const ISTRIP: tcflag_t = 0o000040;
pub const INLCR: tcflag_t = 0o000100;
pub const IGNCR: tcflag_t = 0o000200;
pub const ICRNL: tcflag_t = 0o000400;
pub const IXON: tcflag_t = 0o002000;
pub const IXOFF: tcflag_t = 0o010000;

// c_oflag constants
pub const OPOST: tcflag_t = 0o000001;
pub const ONLCR: tcflag_t = 0o000004;

// c_cflag constants
pub const CSIZE: tcflag_t = 0o000060;
pub const CS5: tcflag_t = 0o000000;
pub const CS6: tcflag_t = 0o000020;
pub const CS7: tcflag_t = 0o000040;
pub const CS8: tcflag_t = 0o000060;
pub const CSTOPB: tcflag_t = 0o000100;
pub const CREAD: tcflag_t = 0o000200;
pub const PARENB: tcflag_t = 0o000400;
pub const PARODD: tcflag_t = 0o001000;
pub const HUPCL: tcflag_t = 0o002000;
pub const CLOCAL: tcflag_t = 0o004000;

// c_lflag constants
pub const ISIG: tcflag_t = 0o000001;
pub const ICANON: tcflag_t = 0o000002;
pub const ECHO: tcflag_t = 0o000010;
pub const ECHOE: tcflag_t = 0o000020;
pub const ECHOK: tcflag_t = 0o000040;
pub const ECHONL: tcflag_t = 0o000100;
pub const NOFLSH: tcflag_t = 0o000200;
pub const TOSTOP: tcflag_t = 0o000400;
pub const IEXTEN: tcflag_t = 0o100000;

// tcsetattr actions
pub const TCSANOW: c_int = 0;
pub const TCSADRAIN: c_int = 1;
pub const TCSAFLUSH: c_int = 2;

// tcflush actions
pub const TCIFLUSH: c_int = 0;
pub const TCOFLUSH: c_int = 1;
pub const TCIOFLUSH: c_int = 2;

// tcflow actions
pub const TCOOFF: c_int = 0;
pub const TCOON: c_int = 1;
pub const TCIOFF: c_int = 2;
pub const TCION: c_int = 3;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Functions
pub export fn tcgetattr(fd: c_int, termios_p: *termios) c_int {
    // std.posix.tcgetattr returns termios struct, or error
    const t = std.posix.tcgetattr(fd) catch |err| {
        setErrno(err);
        return -1;
    };
    
    // Copy fields
    termios_p.c_iflag = t.iflag;
    termios_p.c_oflag = t.oflag;
    termios_p.c_cflag = t.cflag;
    termios_p.c_lflag = t.lflag;
    
    // Copy control characters (might differ in size/layout, iterate safely)
    @memcpy(termios_p.c_cc[0..std.posix.NCCS], t.cc[0..std.posix.NCCS]);
    
    // Some systems store speed in termios, some dont. std.posix.termios might have them.
    // For now we assume they are in c_ispeed/c_ospeed if supported by std.posix
    // Actually std.posix.termios definition depends on OS.
    // We will leave speed as is or use cfgetospeed if needed.
    return 0;
}

pub export fn tcsetattr(fd: c_int, optional_actions: c_int, termios_p: *const termios) c_int {
    var t = std.posix.termios{
        .iflag = @intCast(termios_p.c_iflag),
        .oflag = @intCast(termios_p.c_oflag),
        .cflag = @intCast(termios_p.c_cflag),
        .lflag = @intCast(termios_p.c_lflag),
        .cc = undefined,
        .ispeed = @intCast(termios_p.c_ispeed),
        .ospeed = @intCast(termios_p.c_ospeed),
    };
    @memcpy(t.cc[0..std.posix.NCCS], termios_p.c_cc[0..std.posix.NCCS]);
    
    std.posix.tcsetattr(fd, @intCast(optional_actions), t) catch |err| {
        setErrno(err);
        return -1;
    };
    return 0;
}

pub export fn tcsendbreak(fd: c_int, duration: c_int) c_int {
    if (@hasDecl(std.posix.system, "tcsendbreak")) {
        const rc = std.posix.system.tcsendbreak(fd, duration);
        if (rc != 0) {
            setErrno(std.posix.errno(rc));
            return -1;
        }
        return 0;
    }
    // Fallback?
    return 0;
}

pub export fn tcdrain(fd: c_int) c_int {
    if (@hasDecl(std.posix.system, "tcdrain")) {
        const rc = std.posix.system.tcdrain(fd);
        if (rc != 0) {
            setErrno(std.posix.errno(rc));
            return -1;
        }
        return 0;
    }
    return 0;
}

pub export fn tcflush(fd: c_int, queue_selector: c_int) c_int {
    if (@hasDecl(std.posix.system, "tcflush")) {
        const rc = std.posix.system.tcflush(fd, queue_selector);
        if (rc != 0) {
            setErrno(std.posix.errno(rc));
            return -1;
        }
        return 0;
    }
    return 0;
}

pub export fn tcflow(fd: c_int, action: c_int) c_int {
    if (@hasDecl(std.posix.system, "tcflow")) {
        const rc = std.posix.system.tcflow(fd, action);
        if (rc != 0) {
            setErrno(std.posix.errno(rc));
            return -1;
        }
        return 0;
    }
    return 0;
}

/// Get input baud rate
pub export fn cfgetispeed(termios_p: *const termios) speed_t {
    return termios_p.c_ispeed;
}

/// Get output baud rate
pub export fn cfgetospeed(termios_p: *const termios) speed_t {
    return termios_p.c_ospeed;
}

/// Set input baud rate
pub export fn cfsetispeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ispeed = speed;
    return 0;
}

/// Set output baud rate
pub export fn cfsetospeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ospeed = speed;
    return 0;
}

/// Set both input and output baud rate
pub export fn cfsetspeed(termios_p: *termios, speed: speed_t) c_int {
    termios_p.c_ispeed = speed;
    termios_p.c_ospeed = speed;
    return 0;
}

/// Make terminal raw mode
pub export fn cfmakeraw(termios_p: *termios) void {
    // Disable input processing
    termios_p.c_iflag &= ~@as(tcflag_t, IGNBRK | BRKINT | PARMRK | ISTRIP | INLCR | IGNCR | ICRNL | IXON);
    // Disable output processing
    termios_p.c_oflag &= ~@as(tcflag_t, OPOST);
    // Disable line processing and echo
    termios_p.c_lflag &= ~@as(tcflag_t, ECHO | ECHONL | ICANON | ISIG | IEXTEN);
    // Set 8-bit chars
    termios_p.c_cflag &= ~@as(tcflag_t, CSIZE | PARENB);
    termios_p.c_cflag |= CS8;
    // Set minimum chars and timeout
    termios_p.c_cc[VMIN] = 1;
    termios_p.c_cc[VTIME] = 0;
}

/// Check if fd is a terminal
pub export fn isatty(fd: c_int) c_int {
    var t: termios = undefined;
    if (tcgetattr(fd, &t) == 0) return 1;
    return 0;
}

// Control character indices
pub const VMIN: usize = 6;
pub const VTIME: usize = 5;
pub const VINTR: usize = 0;
pub const VQUIT: usize = 1;
pub const VERASE: usize = 2;
pub const VKILL: usize = 3;
pub const VEOF: usize = 4;
pub const VEOL: usize = 7;
pub const VSTART: usize = 8;
pub const VSTOP: usize = 9;
pub const VSUSP: usize = 10;

