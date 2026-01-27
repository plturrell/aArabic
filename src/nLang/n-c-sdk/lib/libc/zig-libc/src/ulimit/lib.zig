// ulimit module - Phase 1.23
const std = @import("std");

pub const UL_GETFSIZE: c_int = 1;
pub const UL_SETFSIZE: c_int = 2;

pub export fn ulimit(cmd: c_int, ...) c_long {
    _ = cmd;
    return 1024 * 1024;
}
