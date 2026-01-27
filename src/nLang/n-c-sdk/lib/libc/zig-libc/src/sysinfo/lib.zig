// sysinfo module - System info - Phase 1.28
const std = @import("std");

pub const sysinfo_t = extern struct {
    uptime: c_long,
    loads: [3]c_ulong,
    totalram: c_ulong,
    freeram: c_ulong,
    sharedram: c_ulong,
    bufferram: c_ulong,
    totalswap: c_ulong,
    freeswap: c_ulong,
    procs: c_ushort,
    totalhigh: c_ulong,
    freehigh: c_ulong,
    mem_unit: c_uint,
};

pub export fn sysinfo(info: *sysinfo_t) c_int {
    @memset(std.mem.asBytes(info), 0);
    info.totalram = 8 * 1024 * 1024 * 1024;
    info.freeram = 4 * 1024 * 1024 * 1024;
    info.procs = 4;
    info.mem_unit = 1;
    return 0;
}

pub export fn get_nprocs() c_int {
    return 4;
}

pub export fn get_nprocs_conf() c_int {
    return 4;
}

pub export fn get_phys_pages() c_long {
    return 2 * 1024 * 1024;
}

pub export fn get_avphys_pages() c_long {
    return 1 * 1024 * 1024;
}
