// bsd_string module - BSD string extensions - Phase 1.34
const std = @import("std");
const stdlib = @import("../stdlib/lib.zig");
const string = @import("../string/lib.zig");

pub export fn strlcpy(dst: [*]u8, src: [*:0]const u8, size: usize) usize {
    if (size == 0) return std.mem.len(src);
    var i: usize = 0;
    while (i < size - 1 and src[i] != 0) : (i += 1) {
        dst[i] = src[i];
    }
    if (size > 0) dst[i] = 0;
    while (src[i] != 0) : (i += 1) {}
    return i;
}

// ... (other functions remain the same until strndup)

pub export fn strndup(s: [*:0]const u8, n: usize) ?[*:0]u8 {
    const len = string.strnlen(s, n);
    const result = @as(?[*]u8, @ptrCast(stdlib.malloc(len + 1)));
    if (result) |r| {
        var i: usize = 0;
        while (i < len) : (i += 1) r[i] = s[i];
        r[len] = 0;
        return @ptrCast(r);
    }
    return null;
}

pub export fn strsignal(sig: c_int) [*:0]u8 {
    _ = sig;
    return @constCast("Unknown signal");
}

pub export fn strverscmp(s1: [*:0]const u8, s2: [*:0]const u8) c_int {
    var i: usize = 0;
    while (s1[i] != 0 and s2[i] != 0 and s1[i] == s2[i]) : (i += 1) {}
    return @as(c_int, s1[i]) - @as(c_int, s2[i]);
}

pub export fn mempcpy(dest: ?*anyopaque, src: ?*const anyopaque, n: usize) ?*anyopaque {
    if (dest == null or src == null) return null;
    const d = @as([*]u8, @ptrCast(dest));
    const s = @as([*]const u8, @ptrCast(src));
    var i: usize = 0;
    while (i < n) : (i += 1) d[i] = s[i];
    return @ptrCast(d + n);
}

pub export fn memrchr(s: ?*const anyopaque, c: c_int, n: usize) ?*anyopaque {
    if (s == null or n == 0) return null;
    const ptr = @as([*]const u8, @ptrCast(s));
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        if (ptr[i] == @as(u8, @intCast(c))) return @constCast(@ptrCast(ptr + i));
    }
    return null;
}

pub export fn memmem(haystack: ?*const anyopaque, haystacklen: usize, needle: ?*const anyopaque, needlelen: usize) ?*anyopaque {
    if (haystack == null or needle == null or needlelen == 0 or haystacklen < needlelen) return null;
    const h = @as([*]const u8, @ptrCast(haystack));
    const n = @as([*]const u8, @ptrCast(needle));
    var i: usize = 0;
    while (i <= haystacklen - needlelen) : (i += 1) {
        var j: usize = 0;
        while (j < needlelen and h[i + j] == n[j]) : (j += 1) {}
        if (j == needlelen) return @constCast(@ptrCast(h + i));
    }
    return null;
}
