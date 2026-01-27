// sys/stat module - Phase 1.3 - File status with real implementations
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// File type constants
pub const S_IFMT: c_uint = 0o170000;
pub const S_IFSOCK: c_uint = 0o140000;
pub const S_IFLNK: c_uint = 0o120000;
pub const S_IFREG: c_uint = 0o100000;
pub const S_IFBLK: c_uint = 0o060000;
pub const S_IFDIR: c_uint = 0o040000;
pub const S_IFCHR: c_uint = 0o020000;
pub const S_IFIFO: c_uint = 0o010000;

// Permission bits
pub const S_ISUID: c_uint = 0o4000;
pub const S_ISGID: c_uint = 0o2000;
pub const S_ISVTX: c_uint = 0o1000;
pub const S_IRWXU: c_uint = 0o0700;
pub const S_IRUSR: c_uint = 0o0400;
pub const S_IWUSR: c_uint = 0o0200;
pub const S_IXUSR: c_uint = 0o0100;
pub const S_IRWXG: c_uint = 0o0070;
pub const S_IRGRP: c_uint = 0o0040;
pub const S_IWGRP: c_uint = 0o0020;
pub const S_IXGRP: c_uint = 0o0010;
pub const S_IRWXO: c_uint = 0o0007;
pub const S_IROTH: c_uint = 0o0004;
pub const S_IWOTH: c_uint = 0o0002;
pub const S_IXOTH: c_uint = 0o0001;

// Stat structure (simplified for cross-platform compatibility)
pub const stat = extern struct {
    st_dev: c_ulong,
    st_ino: c_ulong,
    st_mode: c_uint,
    st_nlink: c_ulong,
    st_uid: c_uint,
    st_gid: c_uint,
    st_rdev: c_ulong,
    st_size: i64,
    st_blksize: c_long,
    st_blocks: c_long,
    st_atime: c_long,
    st_mtime: c_long,
    st_ctime: c_long,
};

// timespec for utimensat
pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};

// Special time values
pub const UTIME_NOW: c_long = (1 << 30) - 1;
pub const UTIME_OMIT: c_long = (1 << 30) - 2;

// AT_* flags for *at functions
pub const AT_FDCWD: c_int = -100;
pub const AT_SYMLINK_NOFOLLOW: c_int = 0x100;
pub const AT_REMOVEDIR: c_int = 0x200;
pub const AT_SYMLINK_FOLLOW: c_int = 0x400;
pub const AT_EMPTY_PATH: c_int = 0x1000;

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

// Convert Zig stat to C stat
fn convertStat(zig_stat: std.posix.Stat, c_stat: *stat) void {
    c_stat.st_dev = zig_stat.dev;
    c_stat.st_ino = zig_stat.ino;
    c_stat.st_mode = zig_stat.mode;
    c_stat.st_nlink = zig_stat.nlink;
    c_stat.st_uid = zig_stat.uid;
    c_stat.st_gid = zig_stat.gid;
    c_stat.st_rdev = zig_stat.rdev;
    c_stat.st_size = zig_stat.size;
    c_stat.st_blksize = zig_stat.blksize;
    c_stat.st_blocks = zig_stat.blocks;
    c_stat.st_atime = @divTrunc(zig_stat.atime.tv_sec, 1);
    c_stat.st_mtime = @divTrunc(zig_stat.mtime.tv_sec, 1);
    c_stat.st_ctime = @divTrunc(zig_stat.ctime.tv_sec, 1);
}

// Type test macros as functions
pub export fn S_ISREG(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFREG) 1 else 0;
}

pub export fn S_ISDIR(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFDIR) 1 else 0;
}

pub export fn S_ISCHR(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFCHR) 1 else 0;
}

pub export fn S_ISBLK(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFBLK) 1 else 0;
}

pub export fn S_ISFIFO(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFIFO) 1 else 0;
}

pub export fn S_ISLNK(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFLNK) 1 else 0;
}

pub export fn S_ISSOCK(m: c_uint) c_int {
    return if ((m & S_IFMT) == S_IFSOCK) 1 else 0;
}

// Stat functions with real implementations

pub export fn stat_fn(pathname: [*:0]const u8, statbuf: *stat) c_int {
    return fstatat(AT_FDCWD, pathname, statbuf, 0);
}

pub export fn fstat(fd: c_int, statbuf: *stat) c_int {
    var zig_stat: std.posix.Stat = undefined;
    std.posix.fstat(fd, &zig_stat) catch |err| {
        setErrno(err);
        return -1;
    };
    convertStat(zig_stat, statbuf);
    return 0;
}

pub export fn lstat(pathname: [*:0]const u8, statbuf: *stat) c_int {
    return fstatat(AT_FDCWD, pathname, statbuf, AT_SYMLINK_NOFOLLOW);
}

pub export fn fstatat(dirfd: c_int, pathname: [*:0]const u8, statbuf: *stat, flags: c_int) c_int {
    var zig_stat: std.posix.Stat = undefined;
    const path_slice = std.mem.span(pathname);
    
    std.posix.fstatat(dirfd, path_slice, &zig_stat, @intCast(flags)) catch |err| {
        setErrno(err);
        return -1;
    };
    
    convertStat(zig_stat, statbuf);
    return 0;
}

pub export fn mkdir_fn(pathname: [*:0]const u8, mode: c_uint) c_int {
    const rc = std.posix.system.mkdir(pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mkdirat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint) c_int {
    const rc = std.posix.system.mkdirat(dirfd, pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mknod(pathname: [*:0]const u8, mode: c_uint, dev: c_ulong) c_int {
    const rc = std.posix.system.mknod(pathname, mode, dev);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mknodat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint, dev: c_ulong) c_int {
    const rc = std.posix.system.mknodat(dirfd, pathname, mode, dev);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn mkfifo(pathname: [*:0]const u8, mode: c_uint) c_int {
    return mknod(pathname, mode | S_IFIFO, 0);
}

pub export fn mkfifoat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint) c_int {
    return mknodat(dirfd, pathname, mode | S_IFIFO, 0);
}

pub export fn chmod_fn(pathname: [*:0]const u8, mode: c_uint) c_int {
    const rc = std.posix.system.chmod(pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fchmod_fn(fd: c_int, mode: c_uint) c_int {
    const rc = std.posix.system.fchmod(fd, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fchmodat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint, flags: c_int) c_int {
    const rc = std.posix.system.fchmodat(dirfd, pathname, mode, flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

// Global umask storage
var current_umask: c_uint = 0o022;
var umask_mutex = std.Thread.Mutex{};

pub export fn umask(mask: c_uint) c_uint {
    umask_mutex.lock();
    defer umask_mutex.unlock();
    
    const old = current_umask;
    current_umask = mask & 0o777;
    
    // Also set in kernel
    _ = std.posix.system.umask(mask);
    
    return old;
}

pub export fn futimens(fd: c_int, times: ?*const [2]timespec) c_int {
    if (times) |ts| {
        const rc = std.posix.system.utimensat(fd, null, @ptrCast(ts), 0);
        if (failIfErrno(rc)) return -1;
        return rc;
    } else {
        // NULL times means set to current time
        const rc = std.posix.system.utimensat(fd, null, null, 0);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
}

pub export fn utimensat(dirfd: c_int, pathname: ?[*:0]const u8, times: ?*const [2]timespec, flags: c_int) c_int {
    const rc = std.posix.system.utimensat(dirfd, pathname, @ptrCast(times), flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

// Additional stat functions

pub export fn fchownat(dirfd: c_int, pathname: [*:0]const u8, owner: c_uint, group: c_uint, flags: c_int) c_int {
    const rc = std.posix.system.fchownat(dirfd, pathname, owner, group, flags);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn lchown(pathname: [*:0]const u8, owner: c_uint, group: c_uint) c_int {
    const rc = std.posix.system.lchown(pathname, owner, group);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn stat64(pathname: [*:0]const u8, statbuf: *stat) c_int {
    return stat_fn(pathname, statbuf);
}

pub export fn fstat64(fd: c_int, statbuf: *stat) c_int {
    return fstat(fd, statbuf);
}

pub export fn lstat64(pathname: [*:0]const u8, statbuf: *stat) c_int {
    return lstat(pathname, statbuf);
}

pub export fn fstatat64(dirfd: c_int, pathname: [*:0]const u8, statbuf: *stat, flags: c_int) c_int {
    return fstatat(dirfd, pathname, statbuf, flags);
}
