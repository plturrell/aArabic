// stdlib2 module - Extended stdlib - Phase 1.26
const std = @import("std");

pub export fn mkstemp(template: [*:0]u8) c_int {
    _ = template;
    return 3;
}

pub export fn mkostemp(template: [*:0]u8, flags: c_int) c_int {
    _ = template; _ = flags;
    return 3;
}

pub export fn mkdtemp(template: [*:0]u8) ?[*:0]u8 {
    return template;
}

pub export fn realpath(path: [*:0]const u8, resolved_path: ?[*:0]u8) ?[*:0]u8 {
    _ = path;
    return resolved_path;
}

pub export fn getloadavg(loadavg: [*]f64, nelem: c_int) c_int {
    _ = loadavg; _ = nelem;
    return 0;
}

pub export fn grantpt(fd: c_int) c_int {
    _ = fd;
    return 0;
}

pub export fn unlockpt(fd: c_int) c_int {
    _ = fd;
    return 0;
}

pub export fn ptsname(fd: c_int) ?[*:0]u8 {
    _ = fd;
    return @constCast("/dev/pts/0");
}

pub export fn posix_openpt(flags: c_int) c_int {
    _ = flags;
    return 3;
}

pub export fn getsubopt(optionp: *[*:0]u8, tokens: [*:null]const ?[*:0]const u8, valuep: *[*:0]u8) c_int {
    _ = optionp; _ = tokens; _ = valuep;
    return -1;
}

pub export fn l64a(n: c_long) [*:0]u8 {
    _ = n;
    return @constCast("0");
}

pub export fn a64l(s: [*:0]const u8) c_long {
    _ = s;
    return 0;
}

pub export fn ecvt(value: f64, ndigit: c_int, decpt: *c_int, sign: *c_int) [*:0]u8 {
    _ = value; _ = ndigit;
    decpt.* = 0;
    sign.* = 0;
    return @constCast("0");
}

pub export fn fcvt(value: f64, ndigit: c_int, decpt: *c_int, sign: *c_int) [*:0]u8 {
    _ = value; _ = ndigit;
    decpt.* = 0;
    sign.* = 0;
    return @constCast("0");
}

pub export fn gcvt(value: f64, ndigit: c_int, buf: [*:0]u8) [*:0]u8 {
    _ = value; _ = ndigit;
    buf[0] = '0';
    buf[1] = 0;
    return buf;
}
