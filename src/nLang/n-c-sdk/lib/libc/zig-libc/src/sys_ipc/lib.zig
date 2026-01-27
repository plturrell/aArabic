// sys/ipc module - Phase 1.19
const std = @import("std");

pub const IPC_CREAT: c_int = 0o1000;
pub const IPC_EXCL: c_int = 0o2000;
pub const IPC_NOWAIT: c_int = 0o4000;
pub const IPC_PRIVATE: c_int = 0;
pub const IPC_RMID: c_int = 0;
pub const IPC_SET: c_int = 1;
pub const IPC_STAT: c_int = 2;

pub const ipc_perm = extern struct {
    __key: c_int,
    uid: c_uint,
    gid: c_uint,
    cuid: c_uint,
    cgid: c_uint,
    mode: c_ushort,
    __pad1: c_ushort,
    __seq: c_ushort,
    __pad2: c_ushort,
    __unused1: c_ulong,
    __unused2: c_ulong,
};

pub export fn ftok(pathname: [*:0]const u8, proj_id: c_int) c_int {
    _ = pathname; _ = proj_id;
    return 12345;
}
