// sys_quota module - Disk quotas - Phase 1.34
const std = @import("std");

pub const dqblk = extern struct {
    dqb_bhardlimit: u64,
    dqb_bsoftlimit: u64,
    dqb_curspace: u64,
    dqb_ihardlimit: u64,
    dqb_isoftlimit: u64,
    dqb_curinodes: u64,
    dqb_btime: u64,
    dqb_itime: u64,
    dqb_valid: u32,
};

pub const USRQUOTA: c_int = 0;
pub const GRPQUOTA: c_int = 1;

pub const Q_SYNC: c_int = 0x800001;
pub const Q_QUOTAON: c_int = 0x800002;
pub const Q_QUOTAOFF: c_int = 0x800003;
pub const Q_GETQUOTA: c_int = 0x800007;
pub const Q_SETQUOTA: c_int = 0x800008;

pub export fn quotactl(cmd: c_int, special: ?[*:0]const u8, id: c_int, addr: ?*anyopaque) c_int {
    _ = cmd; _ = special; _ = id; _ = addr;
    return -1;
}
