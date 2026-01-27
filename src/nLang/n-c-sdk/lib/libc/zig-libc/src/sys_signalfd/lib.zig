// sys_signalfd module - Phase 1.30
const std = @import("std");

pub const SFD_CLOEXEC: c_int = 0x80000;
pub const SFD_NONBLOCK: c_int = 0x800;

pub const signalfd_siginfo = extern struct {
    ssi_signo: u32,
    ssi_errno: i32,
    ssi_code: i32,
    ssi_pid: u32,
    ssi_uid: u32,
    ssi_fd: i32,
    ssi_tid: u32,
    ssi_band: u32,
    ssi_overrun: u32,
    ssi_trapno: u32,
    ssi_status: i32,
    ssi_int: i32,
    ssi_ptr: u64,
    ssi_utime: u64,
    ssi_stime: u64,
    ssi_addr: u64,
    __pad: [48]u8,
};

pub export fn signalfd(fd: c_int, mask: ?*const anyopaque, flags: c_int) c_int {
    _ = fd; _ = mask; _ = flags;
    return 3;
}
