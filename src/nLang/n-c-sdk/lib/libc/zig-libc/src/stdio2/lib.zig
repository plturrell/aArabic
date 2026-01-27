// stdio2 module - Extended stdio - Phase 1.25
const std = @import("std");

pub export fn getdelim(lineptr: *?[*]u8, n: *usize, delim: c_int, stream: *anyopaque) isize {
    _ = lineptr; _ = n; _ = delim; _ = stream;
    return -1;
}

pub export fn getline(lineptr: *?[*]u8, n: *usize, stream: *anyopaque) isize {
    return getdelim(lineptr, n, '\n', stream);
}

pub export fn dprintf(fd: c_int, format: [*:0]const u8, ...) c_int {
    _ = fd; _ = format;
    return 0;
}

pub export fn vdprintf(fd: c_int, format: [*:0]const u8, ap: *anyopaque) c_int {
    _ = fd; _ = format; _ = ap;
    return 0;
}

pub export fn asprintf(strp: *[*:0]u8, format: [*:0]const u8, ...) c_int {
    _ = strp; _ = format;
    return 0;
}

pub export fn vasprintf(strp: *[*:0]u8, format: [*:0]const u8, ap: *anyopaque) c_int {
    _ = strp; _ = format; _ = ap;
    return 0;
}

pub export fn fmemopen(buf: ?*anyopaque, size: usize, mode: [*:0]const u8) ?*anyopaque {
    _ = buf; _ = size; _ = mode;
    return null;
}

pub export fn open_memstream(ptr: *[*]u8, sizeloc: *usize) ?*anyopaque {
    _ = ptr; _ = sizeloc;
    return null;
}

pub export fn fopencookie(cookie: ?*anyopaque, mode: [*:0]const u8, funcs: anyopaque) ?*anyopaque {
    _ = cookie; _ = mode; _ = funcs;
    return null;
}

pub export fn popen(command: [*:0]const u8, mode: [*:0]const u8) ?*anyopaque {
    _ = command; _ = mode;
    return null;
}

pub export fn pclose(stream: ?*anyopaque) c_int {
    _ = stream;
    return 0;
}

pub export fn flockfile(file: *anyopaque) void {
    _ = file;
}

pub export fn ftrylockfile(file: *anyopaque) c_int {
    _ = file;
    return 0;
}

pub export fn funlockfile(file: *anyopaque) void {
    _ = file;
}

pub export fn getc_unlocked(stream: *anyopaque) c_int {
    _ = stream;
    return -1;
}

pub export fn getchar_unlocked() c_int {
    return -1;
}

pub export fn putc_unlocked(c: c_int, stream: *anyopaque) c_int {
    _ = stream;
    return c;
}

pub export fn putchar_unlocked(c: c_int) c_int {
    return c;
}
