// sys_capability module - Linux capabilities - Phase 1.34
const std = @import("std");

pub const cap_t = ?*anyopaque;
pub const cap_flag_t = c_int;
pub const cap_flag_value_t = c_int;
pub const cap_value_t = c_int;

pub const CAP_EFFECTIVE: cap_flag_t = 0;
pub const CAP_PERMITTED: cap_flag_t = 1;
pub const CAP_INHERITABLE: cap_flag_t = 2;

pub const CAP_CLEAR: cap_flag_value_t = 0;
pub const CAP_SET: cap_flag_value_t = 1;

pub export fn cap_init() cap_t {
    return null;
}

pub export fn cap_free(cap_p: ?*anyopaque) c_int {
    _ = cap_p;
    return 0;
}

pub export fn cap_dup(cap_p: cap_t) cap_t {
    _ = cap_p;
    return null;
}

pub export fn cap_get_proc() cap_t {
    return null;
}

pub export fn cap_set_proc(cap_p: cap_t) c_int {
    _ = cap_p;
    return 0;
}

pub export fn cap_get_pid(pid: c_int) cap_t {
    _ = pid;
    return null;
}

pub export fn cap_get_file(path: [*:0]const u8) cap_t {
    _ = path;
    return null;
}

pub export fn cap_set_file(path: [*:0]const u8, cap_p: cap_t) c_int {
    _ = path; _ = cap_p;
    return 0;
}

pub export fn cap_get_fd(fd: c_int) cap_t {
    _ = fd;
    return null;
}

pub export fn cap_set_fd(fd: c_int, cap_p: cap_t) c_int {
    _ = fd; _ = cap_p;
    return 0;
}

pub export fn cap_clear(cap_p: cap_t) c_int {
    _ = cap_p;
    return 0;
}

pub export fn cap_clear_flag(cap_p: cap_t, flag: cap_flag_t) c_int {
    _ = cap_p; _ = flag;
    return 0;
}

pub export fn cap_get_flag(cap_p: cap_t, cap: cap_value_t, flag: cap_flag_t, value_p: *cap_flag_value_t) c_int {
    _ = cap_p; _ = cap; _ = flag;
    value_p.* = CAP_CLEAR;
    return 0;
}

pub export fn cap_set_flag(cap_p: cap_t, flag: cap_flag_t, ncap: c_int, caps: [*]const cap_value_t, value: cap_flag_value_t) c_int {
    _ = cap_p; _ = flag; _ = ncap; _ = caps; _ = value;
    return 0;
}

pub export fn cap_to_text(cap_p: cap_t, len_p: ?*isize) ?[*:0]u8 {
    _ = cap_p;
    if (len_p) |l| l.* = 0;
    return null;
}

pub export fn cap_from_text(buf_p: [*:0]const u8) cap_t {
    _ = buf_p;
    return null;
}
