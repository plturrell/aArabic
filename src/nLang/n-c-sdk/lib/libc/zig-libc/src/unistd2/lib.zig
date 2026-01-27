// unistd2 module - Extended unistd - Phase 1.27
const std = @import("std");

pub export fn nice(inc: c_int) c_int {
    _ = inc;
    return 0;
}

pub export fn sync() void {}

pub export fn syncfs(fd: c_int) c_int {
    _ = fd;
    return 0;
}

pub export fn fsync(fd: c_int) c_int {
    _ = fd;
    return 0;
}

pub export fn fdatasync(fd: c_int) c_int {
    _ = fd;
    return 0;
}

pub export fn getpagesize() c_int {
    return 4096;
}

pub export fn getdtablesize() c_int {
    return 1024;
}

pub export fn gethostid() c_long {
    return 0;
}

pub export fn sethostid(hostid: c_long) c_int {
    _ = hostid;
    return 0;
}

pub export fn daemon(nochdir: c_int, noclose: c_int) c_int {
    _ = nochdir; _ = noclose;
    return 0;
}

pub export fn acct(filename: ?[*:0]const u8) c_int {
    _ = filename;
    return 0;
}

pub export fn brk(addr: ?*anyopaque) c_int {
    _ = addr;
    return 0;
}

pub export fn sbrk(increment: isize) ?*anyopaque {
    _ = increment;
    return null;
}
