// sys_xattr module - Extended attributes - Phase 1.31
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Extended attribute flags
pub const XATTR_CREATE: c_int = 1;  // Set value, fail if exists
pub const XATTR_REPLACE: c_int = 2; // Set value, fail if not exists

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

/// FULL IMPLEMENTATION: Set extended attribute
pub export fn setxattr(path: [*:0]const u8, name: [*:0]const u8, value: ?*const anyopaque, size: usize, flags: c_int) c_int {
    const val = value orelse {
        if (size > 0) {
            setErrno(.INVAL);
            return -1;
        }
        return 0;
    };
    
    const rc = std.posix.system.setxattr(path, name, val, size, flags);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Set extended attribute (no follow symlinks)
pub export fn lsetxattr(path: [*:0]const u8, name: [*:0]const u8, value: ?*const anyopaque, size: usize, flags: c_int) c_int {
    const val = value orelse {
        if (size > 0) {
            setErrno(.INVAL);
            return -1;
        }
        return 0;
    };
    
    const rc = std.posix.system.lsetxattr(path, name, val, size, flags);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Set extended attribute (by file descriptor)
pub export fn fsetxattr(fd: c_int, name: [*:0]const u8, value: ?*const anyopaque, size: usize, flags: c_int) c_int {
    const val = value orelse {
        if (size > 0) {
            setErrno(.INVAL);
            return -1;
        }
        return 0;
    };
    
    const rc = std.posix.system.fsetxattr(fd, name, val, size, flags);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Get extended attribute
pub export fn getxattr(path: [*:0]const u8, name: [*:0]const u8, value: ?*anyopaque, size: usize) isize {
    const rc = std.posix.system.getxattr(path, name, value, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Get extended attribute (no follow symlinks)
pub export fn lgetxattr(path: [*:0]const u8, name: [*:0]const u8, value: ?*anyopaque, size: usize) isize {
    const rc = std.posix.system.lgetxattr(path, name, value, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Get extended attribute (by file descriptor)
pub export fn fgetxattr(fd: c_int, name: [*:0]const u8, value: ?*anyopaque, size: usize) isize {
    const rc = std.posix.system.fgetxattr(fd, name, value, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: List extended attributes
pub export fn listxattr(path: [*:0]const u8, list: ?[*]u8, size: usize) isize {
    const rc = std.posix.system.listxattr(path, if (list) |l| @ptrCast(l) else null, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: List extended attributes (no follow symlinks)
pub export fn llistxattr(path: [*:0]const u8, list: ?[*]u8, size: usize) isize {
    const rc = std.posix.system.llistxattr(path, if (list) |l| @ptrCast(l) else null, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: List extended attributes (by file descriptor)
pub export fn flistxattr(fd: c_int, list: ?[*]u8, size: usize) isize {
    const rc = std.posix.system.flistxattr(fd, if (list) |l| @ptrCast(l) else null, size);
    if (rc < 0) {
        if (failIfErrno(rc)) return -1;
    }
    return rc;
}

/// FULL IMPLEMENTATION: Remove extended attribute
pub export fn removexattr(path: [*:0]const u8, name: [*:0]const u8) c_int {
    const rc = std.posix.system.removexattr(path, name);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Remove extended attribute (no follow symlinks)
pub export fn lremovexattr(path: [*:0]const u8, name: [*:0]const u8) c_int {
    const rc = std.posix.system.lremovexattr(path, name);
    if (failIfErrno(rc)) return -1;
    return 0;
}

/// FULL IMPLEMENTATION: Remove extended attribute (by file descriptor)
pub export fn fremovexattr(fd: c_int, name: [*:0]const u8) c_int {
    const rc = std.posix.system.fremovexattr(fd, name);
    if (failIfErrno(rc)) return -1;
    return 0;
}

// Total: 12 xattr functions - ALL FULLY IMPLEMENTED
// Supports: set/get/list/remove extended attributes
// Path-based, symlink-aware (l*), and fd-based (f*) variants
