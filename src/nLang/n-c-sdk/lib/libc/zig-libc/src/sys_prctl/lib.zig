// sys_prctl module - Process control - Phase 1.31
const std = @import("std");

pub const PR_SET_NAME: c_int = 15;
pub const PR_GET_NAME: c_int = 16;
pub const PR_SET_DUMPABLE: c_int = 4;
pub const PR_GET_DUMPABLE: c_int = 3;

pub export fn prctl(option: c_int, arg2: c_ulong, arg3: c_ulong, arg4: c_ulong, arg5: c_ulong) c_int {
    _ = option; _ = arg2; _ = arg3; _ = arg4; _ = arg5;
    return 0;
}
