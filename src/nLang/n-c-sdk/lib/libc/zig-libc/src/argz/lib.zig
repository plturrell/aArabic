// argz module - Argument vector strings - Phase 1.35
const std = @import("std");
const stdlib = @import("../stdlib/lib.zig");

pub export fn argz_create(argv: [*]const [*:0]const u8, argz: *[*]u8, argz_len: *usize) c_int {
    _ = argv;
    argz.* = @ptrCast(stdlib.malloc(1));
    argz_len.* = 0;
    return 0;
}

pub export fn argz_create_sep(string: [*:0]const u8, sep: c_int, argz: *[*]u8, argz_len: *usize) c_int {
    _ = string; _ = sep;
    argz.* = @ptrCast(stdlib.malloc(1));
    argz_len.* = 0;
    return 0;
}

pub export fn argz_count(argz: [*]const u8, argz_len: usize) usize {
    _ = argz;
    return if (argz_len == 0) 0 else 1;
}

pub export fn argz_extract(argz: [*]const u8, argz_len: usize, argv: [*][*:0]u8) void {
    _ = argz_len;
    argv[0] = @constCast(@ptrCast(argz));
}

pub export fn argz_stringify(argz: [*]u8, argz_len: usize, sep: c_int) void {
    _ = argz; _ = argz_len; _ = sep;
}

pub export fn argz_add(argz: *[*]u8, argz_len: *usize, str: [*:0]const u8) c_int {
    _ = argz; _ = str;
    argz_len.* = 0;
    return 0;
}

pub export fn argz_add_sep(argz: *[*]u8, argz_len: *usize, string: [*:0]const u8, delim: c_int) c_int {
    _ = argz; _ = string; _ = delim;
    argz_len.* = 0;
    return 0;
}

pub export fn argz_append(argz: *[*]u8, argz_len: *usize, buf: [*]const u8, buf_len: usize) c_int {
    _ = argz; _ = buf; _ = buf_len;
    argz_len.* = 0;
    return 0;
}

pub export fn argz_delete(argz: *[*]u8, argz_len: *usize, entry: [*]u8) void {
    _ = argz; _ = entry;
    argz_len.* = 0;
}

pub export fn argz_insert(argz: *[*]u8, argz_len: *usize, before: [*]u8, entry: [*:0]const u8) c_int {
    _ = argz; _ = before; _ = entry;
    argz_len.* = 0;
    return 0;
}

pub export fn argz_next(argz: [*]const u8, argz_len: usize, entry: ?[*]const u8) ?[*]u8 {
    _ = argz; _ = argz_len; _ = entry;
    return null;
}

pub export fn argz_replace(argz: *[*]u8, argz_len: *usize, str: [*:0]const u8, with: [*:0]const u8, replace_count: ?*usize) c_int {
    _ = argz; _ = str; _ = with;
    argz_len.* = 0;
    if (replace_count) |rc| rc.* = 0;
    return 0;
}
