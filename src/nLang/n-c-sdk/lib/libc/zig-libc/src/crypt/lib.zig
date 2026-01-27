// crypt module - Phase 1.27
const std = @import("std");

pub export fn crypt(key: [*:0]const u8, salt: [*:0]const u8) [*:0]u8 {
    _ = key; _ = salt;
    return @constCast("*");
}

pub export fn encrypt(block: [*]u8, edflag: c_int) void {
    _ = block; _ = edflag;
}

pub export fn setkey(key: [*:0]const u8) void {
    _ = key;
}

pub export fn crypt_r(key: [*:0]const u8, salt: [*:0]const u8, data: ?*anyopaque) [*:0]u8 {
    _ = key; _ = salt; _ = data;
    return @constCast("*");
}
