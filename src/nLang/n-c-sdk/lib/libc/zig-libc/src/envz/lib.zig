// envz module - Environment vector strings - Phase 1.35
const std = @import("std");

pub export fn envz_add(envz: *[*]u8, envz_len: *usize, name: [*:0]const u8, value: ?[*:0]const u8) c_int {
    _ = envz; _ = name; _ = value;
    envz_len.* = 0;
    return 0;
}

pub export fn envz_entry(envz: [*]const u8, envz_len: usize, name: [*:0]const u8) ?[*]u8 {
    _ = envz; _ = envz_len; _ = name;
    return null;
}

pub export fn envz_get(envz: [*]const u8, envz_len: usize, name: [*:0]const u8) ?[*]u8 {
    _ = envz; _ = envz_len; _ = name;
    return null;
}

pub export fn envz_merge(envz: *[*]u8, envz_len: *usize, envz2: [*]const u8, envz2_len: usize, override_: c_int) c_int {
    _ = envz; _ = envz2; _ = envz2_len; _ = override_;
    envz_len.* = 0;
    return 0;
}

pub export fn envz_remove(envz: *[*]u8, envz_len: *usize, name: [*:0]const u8) void {
    _ = envz; _ = name;
    envz_len.* = 0;
}

pub export fn envz_strip(envz: *[*]u8, envz_len: *usize) void {
    _ = envz;
    envz_len.* = 0;
}
