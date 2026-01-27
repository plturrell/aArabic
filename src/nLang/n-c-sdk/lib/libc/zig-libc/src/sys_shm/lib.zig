// sys/shm module - Phase 1.19
const std = @import("std");
const sys_ipc = @import("../sys_ipc/lib.zig");

pub const SHM_RDONLY: c_int = 0o10000;
pub const SHM_RND: c_int = 0o20000;
pub const SHM_REMAP: c_int = 0o40000;

pub const shmid_ds = extern struct {
    shm_perm: sys_ipc.ipc_perm,
    shm_segsz: usize,
    shm_atime: c_long,
    shm_dtime: c_long,
    shm_ctime: c_long,
    shm_cpid: c_int,
    shm_lpid: c_int,
    shm_nattch: c_ulong,
};

pub export fn shmget(key: c_int, size: usize, shmflg: c_int) c_int {
    _ = key; _ = size; _ = shmflg;
    return 1;
}

pub export fn shmat(shmid: c_int, shmaddr: ?*const anyopaque, shmflg: c_int) ?*anyopaque {
    _ = shmid; _ = shmaddr; _ = shmflg;
    return @ptrFromInt(0x1000);
}

pub export fn shmdt(shmaddr: ?*const anyopaque) c_int {
    _ = shmaddr;
    return 0;
}

pub export fn shmctl(shmid: c_int, cmd: c_int, buf: ?*shmid_ds) c_int {
    _ = shmid; _ = cmd; _ = buf;
    return 0;
}
