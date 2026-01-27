// strings module - Phase 1.23
const std = @import("std");

pub export fn bcopy(src: ?*const anyopaque, dest: ?*anyopaque, n: usize) void {
    if (src == null or dest == null) return;
    @memcpy(@as([*]u8, @ptrCast(dest.?))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
}

pub export fn bzero(s: ?*anyopaque, n: usize) void {
    if (s == null) return;
    @memset(@as([*]u8, @ptrCast(s.?))[0..n], 0);
}

pub export fn bcmp(s1: ?*const anyopaque, s2: ?*const anyopaque, n: usize) c_int {
    if (s1 == null or s2 == null) return 0;
    const b1 = @as([*]const u8, @ptrCast(s1.?))[0..n];
    const b2 = @as([*]const u8, @ptrCast(s2.?))[0..n];
    return if (std.mem.eql(u8, b1, b2)) 0 else 1;
}

pub export fn index(s: [*:0]const u8, c: c_int) ?[*:0]u8 {
    var i: usize = 0;
    const ch = @as(u8, @intCast(c));
    while (s[i] != 0) : (i += 1) {
        if (s[i] == ch) return @constCast(@as([*:0]u8, @ptrCast(s + i)));
    }
    return null;
}

pub export fn rindex(s: [*:0]const u8, c: c_int) ?[*:0]u8 {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {}
    const ch = @as(u8, @intCast(c));
    while (i > 0) {
        i -= 1;
        if (s[i] == ch) return @constCast(@as([*:0]u8, @ptrCast(s + i)));
    }
    return null;
}

pub export fn strcasecmp(s1: [*:0]const u8, s2: [*:0]const u8) c_int {
    var i: usize = 0;
    while (s1[i] != 0 and s2[i] != 0) : (i += 1) {
        const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
        const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
        if (c1 != c2) return @as(c_int, c1) - @as(c_int, c2);
    }
    return @as(c_int, s1[i]) - @as(c_int, s2[i]);
}

pub export fn strncasecmp(s1: [*:0]const u8, s2: [*:0]const u8, n: usize) c_int {
    var i: usize = 0;
    while (i < n and s1[i] != 0 and s2[i] != 0) : (i += 1) {
        const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
        const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
        if (c1 != c2) return @as(c_int, c1) - @as(c_int, c2);
    }
    if (i >= n) return 0;
    return @as(c_int, s1[i]) - @as(c_int, s2[i]);
}
