// sys/msg module - Phase 1.19
const std = @import("std");
const sys_ipc = @import("../sys_ipc/lib.zig");

pub const msqid_ds = extern struct {
    msg_perm: sys_ipc.ipc_perm,
    msg_stime: c_long,
    msg_rtime: c_long,
    msg_ctime: c_long,
    __msg_cbytes: c_ulong,
    msg_qnum: c_ulong,
    msg_qbytes: c_ulong,
    msg_lspid: c_int,
    msg_lrpid: c_int,
};

pub export fn msgget(key: c_int, msgflg: c_int) c_int {
    _ = key; _ = msgflg;
    return 1;
}

pub export fn msgsnd(msqid: c_int, msgp: ?*const anyopaque, msgsz: usize, msgflg: c_int) c_int {
    _ = msqid; _ = msgp; _ = msgsz; _ = msgflg;
    return 0;
}

pub export fn msgrcv(msqid: c_int, msgp: ?*anyopaque, msgsz: usize, msgtyp: c_long, msgflg: c_int) isize {
    _ = msqid; _ = msgp; _ = msgsz; _ = msgtyp; _ = msgflg;
    return 0;
}

pub export fn msgctl(msqid: c_int, cmd: c_int, buf: ?*msqid_ds) c_int {
    _ = msqid; _ = cmd; _ = buf;
    return 0;
}
