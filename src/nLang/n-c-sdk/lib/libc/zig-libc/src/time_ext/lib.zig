// time extensions module - Phase 1.15
const std = @import("std");

pub const timespec = extern struct {
    tv_sec: c_long,
    tv_nsec: c_long,
};

pub const CLOCK_REALTIME: c_int = 0;
pub const CLOCK_MONOTONIC: c_int = 1;
pub const CLOCK_PROCESS_CPUTIME_ID: c_int = 2;
pub const CLOCK_THREAD_CPUTIME_ID: c_int = 3;

pub export fn clock_gettime(clk_id: c_int, tp: *timespec) c_int {
    _ = clk_id;
    const now = std.time.nanoTimestamp();
    tp.tv_sec = @divFloor(now, std.time.ns_per_s);
    tp.tv_nsec = @mod(now, std.time.ns_per_s);
    return 0;
}

pub export fn clock_settime(clk_id: c_int, tp: *const timespec) c_int {
    _ = clk_id; _ = tp;
    return 0;
}

pub export fn clock_getres(clk_id: c_int, res: *timespec) c_int {
    _ = clk_id;
    res.tv_sec = 0;
    res.tv_nsec = 1;
    return 0;
}

pub export fn clock_nanosleep(clk_id: c_int, flags: c_int, request: *const timespec, remain: ?*timespec) c_int {
    _ = clk_id; _ = flags; _ = request; _ = remain;
    return 0;
}

pub export fn nanosleep(req: *const timespec, rem: ?*timespec) c_int {
    _ = req; _ = rem;
    return 0;
}

pub export fn strftime(s: [*:0]u8, maxsize: usize, format: [*:0]const u8, timeptr: ?*const anyopaque) usize {
    _ = format; _ = timeptr;
    if (maxsize > 0) s[0] = 0;
    return 0;
}

pub export fn strptime(s: [*:0]const u8, format: [*:0]const u8, tm: ?*anyopaque) ?[*:0]u8 {
    _ = s; _ = format; _ = tm;
    return null;
}

pub export fn tzset() void {}

pub export fn timegm(tm: ?*anyopaque) c_long {
    _ = tm;
    return 0;
}

pub export fn timelocal(tm: ?*anyopaque) c_long {
    _ = tm;
    return 0;
}
