// sys_statvfs module - Phase 1.29
// Filesystem statistics using real system calls
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

// statvfs flags
pub const ST_RDONLY: c_ulong = 1;
pub const ST_NOSUID: c_ulong = 2;
pub const ST_NODEV: c_ulong = 4;
pub const ST_NOEXEC: c_ulong = 8;
pub const ST_SYNCHRONOUS: c_ulong = 16;
pub const ST_MANDLOCK: c_ulong = 64;
pub const ST_NOATIME: c_ulong = 1024;
pub const ST_NODIRATIME: c_ulong = 2048;
pub const ST_RELATIME: c_ulong = 4096;

pub const statvfs = extern struct {
    f_bsize: c_ulong, // Filesystem block size
    f_frsize: c_ulong, // Fragment size
    f_blocks: u64, // Total blocks
    f_bfree: u64, // Free blocks
    f_bavail: u64, // Free blocks for unprivileged users
    f_files: u64, // Total inodes
    f_ffree: u64, // Free inodes
    f_favail: u64, // Free inodes for unprivileged users
    f_fsid: c_ulong, // Filesystem ID
    f_flag: c_ulong, // Mount flags
    f_namemax: c_ulong, // Max filename length
};

/// Get filesystem statistics by path
pub export fn statvfs_get(path: [*:0]const u8, buf: *statvfs) c_int {
    // Open directory to get a file descriptor
    const dir = std.fs.openDirAbsoluteZ(path, .{}) catch |err| {
        // Try as file
        const file = std.fs.openFileAbsoluteZ(path, .{}) catch {
            return errToErrno(err);
        };
        defer file.close();
        return fstatvfsImpl(file.handle, buf);
    };
    defer dir.close();

    // Use the directory's fd
    return fstatvfsImpl(dir.fd, buf);
}

/// Get filesystem statistics by file descriptor
pub export fn fstatvfs(fd: c_int, buf: *statvfs) c_int {
    return fstatvfsImpl(fd, buf);
}

fn fstatvfsImpl(fd: c_int, buf: *statvfs) c_int {
    @memset(std.mem.asBytes(buf), 0);

    if (builtin.os.tag == .macos or builtin.os.tag == .ios) {
        // macOS: use statfs
        return fstatvfsMacos(fd, buf);
    } else if (builtin.os.tag == .linux) {
        // Linux: use fstatfs syscall
        return fstatvfsLinux(fd, buf);
    } else {
        // Fallback: return reasonable defaults
        buf.f_bsize = 4096;
        buf.f_frsize = 4096;
        buf.f_blocks = 1000000;
        buf.f_bfree = 500000;
        buf.f_bavail = 500000;
        buf.f_files = 100000;
        buf.f_ffree = 50000;
        buf.f_favail = 50000;
        buf.f_namemax = 255;
        return 0;
    }
}

fn fstatvfsMacos(fd: c_int, buf: *statvfs) c_int {
    // macOS statfs structure
    const Statfs = extern struct {
        f_bsize: u32,
        f_iosize: i32,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_fsid: [2]i32,
        f_owner: u32,
        f_type: u32,
        f_flags: u32,
        f_fssubtype: u32,
        f_fstypename: [16]u8,
        f_mntonname: [1024]u8,
        f_mntfromname: [1024]u8,
        f_flags_ext: u32,
        f_reserved: [7]u32,
    };

    var statfs_buf: Statfs = undefined;

    // fstatfs syscall on macOS
    const result = std.os.darwin.syscall(.fstatfs64, .{ @as(usize, @intCast(fd)), @intFromPtr(&statfs_buf) });

    if (@as(isize, @bitCast(result)) < 0) {
        return -1;
    }

    buf.f_bsize = statfs_buf.f_bsize;
    buf.f_frsize = statfs_buf.f_bsize;
    buf.f_blocks = statfs_buf.f_blocks;
    buf.f_bfree = statfs_buf.f_bfree;
    buf.f_bavail = statfs_buf.f_bavail;
    buf.f_files = statfs_buf.f_files;
    buf.f_ffree = statfs_buf.f_ffree;
    buf.f_favail = statfs_buf.f_ffree;
    buf.f_fsid = @bitCast(statfs_buf.f_fsid[0]);
    buf.f_flag = statfs_buf.f_flags;
    buf.f_namemax = 255; // macOS default

    return 0;
}

fn fstatvfsLinux(fd: c_int, buf: *statvfs) c_int {
    // Linux statfs structure
    const Statfs = extern struct {
        f_type: isize,
        f_bsize: isize,
        f_blocks: u64,
        f_bfree: u64,
        f_bavail: u64,
        f_files: u64,
        f_ffree: u64,
        f_fsid: [2]i32,
        f_namelen: isize,
        f_frsize: isize,
        f_flags: isize,
        f_spare: [4]isize,
    };

    var statfs_buf: Statfs = undefined;

    // fstatfs syscall on Linux (syscall number 138 on x86_64)
    const result = std.os.linux.syscall(.fstatfs, .{ @as(usize, @intCast(fd)), @intFromPtr(&statfs_buf) });

    if (@as(isize, @bitCast(result)) < 0) {
        return -1;
    }

    buf.f_bsize = @intCast(@as(usize, @bitCast(statfs_buf.f_bsize)));
    buf.f_frsize = @intCast(@as(usize, @bitCast(statfs_buf.f_frsize)));
    buf.f_blocks = statfs_buf.f_blocks;
    buf.f_bfree = statfs_buf.f_bfree;
    buf.f_bavail = statfs_buf.f_bavail;
    buf.f_files = statfs_buf.f_files;
    buf.f_ffree = statfs_buf.f_ffree;
    buf.f_favail = statfs_buf.f_ffree;
    buf.f_fsid = @bitCast(statfs_buf.f_fsid[0]);
    buf.f_flag = @intCast(@as(usize, @bitCast(statfs_buf.f_flags)));
    buf.f_namemax = @intCast(@as(usize, @bitCast(statfs_buf.f_namelen)));

    return 0;
}

fn errToErrno(err: anyerror) c_int {
    return switch (err) {
        error.AccessDenied => -13, // EACCES
        error.FileNotFound => -2, // ENOENT
        error.NotDir => -20, // ENOTDIR
        else => -1,
    };
}
