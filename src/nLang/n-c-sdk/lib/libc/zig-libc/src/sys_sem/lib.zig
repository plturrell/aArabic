// sys/sem module - Phase 1.19
const std = @import("std");
const sys_ipc = @import("../sys_ipc/lib.zig");

pub const SEM_UNDO: c_int = 0o10000;

pub const sembuf = extern struct {
    sem_num: c_ushort,
    sem_op: c_short,
    sem_flg: c_short,
};

pub const semid_ds = extern struct {
    sem_perm: sys_ipc.ipc_perm,
    sem_otime: c_long,
    sem_ctime: c_long,
    sem_nsems: c_ulong,
};

pub export fn semget(key: c_int, nsems: c_int, semflg: c_int) c_int {
    _ = key; _ = nsems; _ = semflg;
    return 1;
}

pub export fn semop(semid: c_int, sops: [*]sembuf, nsops: usize) c_int {
    _ = semid; _ = sops; _ = nsops;
    return 0;
}

pub export fn semctl(semid: c_int, semnum: c_int, cmd: c_int, ...) c_int {
    _ = semid; _ = semnum; _ = cmd;
    return 0;
}

pub export fn semtimedop(semid: c_int, sops: [*]sembuf, nsops: usize, timeout: ?*const anyopaque) c_int {
    _ = semid; _ = sops; _ = nsops; _ = timeout;
    return 0;
}
