// string2 module - Extended string - Phase 1.26
const std = @import("std");

pub export fn stpcpy(dest: [*:0]u8, src: [*:0]const u8) [*:0]u8 {
    var i: usize = 0;
    while (src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    dest[i] = 0;
    return dest + i;
}

pub export fn stpncpy(dest: [*]u8, src: [*:0]const u8, n: usize) [*]u8 {
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    while (i < n) : (i += 1) {
        dest[i] = 0;
    }
    return dest + i;
}

pub export fn strdup(s: [*:0]const u8) ?[*:0]u8 {
    _ = s;
    return null;
}

pub export fn strndup(s: [*:0]const u8, n: usize) ?[*:0]u8 {
    _ = s; _ = n;
    return null;
}

pub export fn strsep(stringp: *?[*:0]u8, delim: [*:0]const u8) ?[*:0]u8 {
    _ = stringp; _ = delim;
    return null;
}

pub export fn strsignal(sig: c_int) [*:0]u8 {
    _ = sig;
    return @constCast("Unknown signal");
}

pub export fn strerror_r(errnum: c_int, buf: [*]u8, buflen: usize) c_int {
    _ = errnum;
    if (buflen > 0) buf[0] = 0;
    return 0;
}

pub export fn memccpy(dest: ?*anyopaque, src: ?*const anyopaque, c: c_int, n: usize) ?*anyopaque {
    if (dest == null or src == null) return null;
    const d = @as([*]u8, @ptrCast(dest.?));
    const s = @as([*]const u8, @ptrCast(src.?));
    const ch = @as(u8, @intCast(c));
    for (0..n) |i| {
        d[i] = s[i];
        if (s[i] == ch) return @ptrFromInt(@intFromPtr(d) + i + 1);
    }
    return null;
}

pub export fn mempcpy(dest: ?*anyopaque, src: ?*const anyopaque, n: usize) ?*anyopaque {
    if (dest == null or src == null) return null;
    @memcpy(@as([*]u8, @ptrCast(dest.?))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
    return @ptrFromInt(@intFromPtr(dest.?) + n);
}

pub export fn memmem(haystack: ?*const anyopaque, haystacklen: usize, needle: ?*const anyopaque, needlelen: usize) ?*anyopaque {
    _ = haystack; _ = haystacklen; _ = needle; _ = needlelen;
    return null;
}

pub export fn basename(path: [*:0]u8) [*:0]u8 {
    var i: usize = 0;
    var last_slash: usize = 0;
    while (path[i] != 0) : (i += 1) {
        if (path[i] == '/') last_slash = i + 1;
    }
    return path + last_slash;
}

pub export fn strchrnul(s: [*:0]const u8, c: c_int) [*:0]u8 {
    var i: usize = 0;
    const ch = @as(u8, @intCast(c));
    while (s[i] != 0 and s[i] != ch) : (i += 1) {}
    return @constCast(s + i);
}

pub export fn strcasestr(haystack: [*:0]const u8, needle: [*:0]const u8) ?[*:0]u8 {
    _ = haystack; _ = needle;
    return null;
}

pub export fn strnlen(s: [*:0]const u8, maxlen: usize) usize {
    var i: usize = 0;
    while (i < maxlen and s[i] != 0) : (i += 1) {}
    return i;
}
