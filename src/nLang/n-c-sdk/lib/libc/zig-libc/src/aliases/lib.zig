// aliases module - Network aliases - Phase 1.33
const std = @import("std");

pub const aliasent = extern struct {
    alias_name: [*:0]u8,
    alias_members_len: usize,
    alias_members: [*][*:0]u8,
    alias_local: c_int,
};

pub export fn setaliasent() void {}
pub export fn endaliasent() void {}

pub export fn getaliasent() ?*aliasent {
    return null;
}

pub export fn getaliasent_r(result_buf: *aliasent, buffer: [*]u8, buflen: usize, result: **aliasent) c_int {
    _ = result_buf; _ = buffer; _ = buflen; _ = result;
    return -1;
}

pub export fn getaliasbyname(name: [*:0]const u8) ?*aliasent {
    _ = name;
    return null;
}

pub export fn getaliasbyname_r(name: [*:0]const u8, result_buf: *aliasent, buffer: [*]u8, buflen: usize, result: **aliasent) c_int {
    _ = name; _ = result_buf; _ = buffer; _ = buflen; _ = result;
    return -1;
}
