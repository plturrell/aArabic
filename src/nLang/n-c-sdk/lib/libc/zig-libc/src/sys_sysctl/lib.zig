// sys_sysctl module - Phase 1.29
const std = @import("std");

pub export fn sysctl(name: [*]const c_int, namelen: c_uint, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int {
    _ = name; _ = namelen; _ = oldp; _ = oldlenp; _ = newp; _ = newlen;
    return 0;
}

pub export fn sysctlbyname(name: [*:0]const u8, oldp: ?*anyopaque, oldlenp: ?*usize, newp: ?*const anyopaque, newlen: usize) c_int {
    _ = name; _ = oldp; _ = oldlenp; _ = newp; _ = newlen;
    return 0;
}

pub export fn sysctlnametomib(name: [*:0]const u8, mibp: [*]c_int, sizep: *usize) c_int {
    _ = name; _ = mibp; _ = sizep;
    return 0;
}
