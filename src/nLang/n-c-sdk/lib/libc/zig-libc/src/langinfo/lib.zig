// langinfo module - Phase 1.21
const std = @import("std");

pub const nl_item = c_int;

pub const CODESET: nl_item = 0;
pub const D_T_FMT: nl_item = 1;
pub const D_FMT: nl_item = 2;
pub const T_FMT: nl_item = 3;
pub const T_FMT_AMPM: nl_item = 4;
pub const AM_STR: nl_item = 5;
pub const PM_STR: nl_item = 6;

pub export fn nl_langinfo(item: nl_item) [*:0]u8 {
    _ = item;
    return @constCast("UTF-8");
}
