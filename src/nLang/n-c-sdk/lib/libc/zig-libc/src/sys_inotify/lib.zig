// sys_inotify module - Phase 1.30 - File System Notifications
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

inline fn failIfErrno(rc: anytype) bool {
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return true;
    }
    return false;
}

pub const IN_CLOEXEC: c_int = 0x80000;
pub const IN_NONBLOCK: c_int = 0x800;
pub const IN_ACCESS: u32 = 0x00000001;
pub const IN_MODIFY: u32 = 0x00000002;
pub const IN_ATTRIB: u32 = 0x00000004;
pub const IN_CLOSE_WRITE: u32 = 0x00000008;
pub const IN_CLOSE_NOWRITE: u32 = 0x00000010;
pub const IN_OPEN: u32 = 0x00000020;
pub const IN_MOVED_FROM: u32 = 0x00000040;
pub const IN_MOVED_TO: u32 = 0x00000080;
pub const IN_CREATE: u32 = 0x00000100;
pub const IN_DELETE: u32 = 0x00000200;

pub const inotify_event = extern struct {
    wd: c_int,
    mask: u32,
    cookie: u32,
    len: u32,
};

/// FULL IMPLEMENTATION: Initialize inotify instance
pub export fn inotify_init() c_int {
    const rc = std.posix.system.inotify_init();
    if (failIfErrno(rc)) return -1;
    return rc;
}

/// FULL IMPLEMENTATION: Initialize inotify instance with flags
pub export fn inotify_init1(flags: c_int) c_int {
    const rc = std.posix.system.inotify_init1(@intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

/// FULL IMPLEMENTATION: Add watch to inotify instance
pub export fn inotify_add_watch(fd: c_int, pathname: [*:0]const u8, mask: u32) c_int {
    const rc = std.posix.system.inotify_add_watch(fd, pathname, mask);
    if (failIfErrno(rc)) return -1;
    return rc;
}

/// FULL IMPLEMENTATION: Remove watch from inotify instance
pub export fn inotify_rm_watch(fd: c_int, wd: c_int) c_int {
    const rc = std.posix.system.inotify_rm_watch(fd, wd);
    if (failIfErrno(rc)) return -1;
    return 0;
}

// Total: 4 inotify functions - ALL FULLY IMPLEMENTED
// File system change notifications for real-time monitoring
