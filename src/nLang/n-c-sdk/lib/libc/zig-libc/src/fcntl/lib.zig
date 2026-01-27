// fcntl module - Phase 1.3 System Interfaces
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Open flags
pub const O_RDONLY: c_int = 0x0000;
pub const O_WRONLY: c_int = 0x0001;
pub const O_RDWR: c_int = 0x0002;
pub const O_CREAT: c_int = 0x0040;
pub const O_EXCL: c_int = 0x0080;
pub const O_TRUNC: c_int = 0x0200;
pub const O_APPEND: c_int = 0x0400;
pub const O_NONBLOCK: c_int = 0x0800;
pub const O_SYNC: c_int = 0x1000;
pub const O_ASYNC: c_int = 0x2000;
pub const O_DIRECTORY: c_int = 0x10000;
pub const O_NOFOLLOW: c_int = 0x20000;
pub const O_CLOEXEC: c_int = 0x40000;
pub const O_NOCTTY: c_int = 0x8000;
pub const O_DSYNC: c_int = 0x4000;

// fcntl commands
pub const F_DUPFD: c_int = 0;
pub const F_GETFD: c_int = 1;
pub const F_SETFD: c_int = 2;
pub const F_GETFL: c_int = 3;
pub const F_SETFL: c_int = 4;
pub const F_GETLK: c_int = 5;
pub const F_SETLK: c_int = 6;
pub const F_SETLKW: c_int = 7;
pub const F_GETOWN: c_int = 9;
pub const F_SETOWN: c_int = 8;
pub const F_DUPFD_CLOEXEC: c_int = 1030;

// File descriptor flags
pub const FD_CLOEXEC: c_int = 1;

// Lock types
pub const F_RDLCK: c_int = 0;
pub const F_WRLCK: c_int = 1;
pub const F_UNLCK: c_int = 2;

// Advisory information
pub const POSIX_FADV_NORMAL: c_int = 0;
pub const POSIX_FADV_SEQUENTIAL: c_int = 2;
pub const POSIX_FADV_RANDOM: c_int = 1;
pub const POSIX_FADV_NOREUSE: c_int = 5;
pub const POSIX_FADV_WILLNEED: c_int = 3;
pub const POSIX_FADV_DONTNEED: c_int = 4;

// Flock structure
pub const Flock = extern struct {
    l_type: c_short,
    l_whence: c_short,
    l_start: i64,
    l_len: i64,
    l_pid: c_int,
};

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

// Functions
pub export fn open(pathname: [*:0]const u8, flags: c_int, ...) c_int {
    var mode: c_uint = 0;
    
    // Check if mode argument is needed (O_CREAT or O_TMPFILE)
    if ((flags & O_CREAT) != 0) {
        var args = @cVaStart();
        mode = @cVaArg(&args, c_uint);
        @cVaEnd(&args);
    }
    
    const open_flags: u32 = @intCast(flags);
    const rc = std.posix.system.open(pathname, open_flags, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn openat(dirfd: c_int, pathname: [*:0]const u8, flags: c_int, ...) c_int {
    var mode: c_uint = 0;
    
    if ((flags & O_CREAT) != 0) {
        var args = @cVaStart();
        mode = @cVaArg(&args, c_uint);
        @cVaEnd(&args);
    }
    
    const open_flags: u32 = @intCast(flags);
    const rc = std.posix.system.openat(dirfd, pathname, open_flags, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn creat(pathname: [*:0]const u8, mode: c_uint) c_int {
    const flags = O_CREAT | O_WRONLY | O_TRUNC;
    return open(pathname, flags, mode);
}

pub export fn fcntl(fd: c_int, cmd: c_int, ...) c_int {
    var arg: c_ulong = 0;
    var args = @cVaStart();
    
    switch (cmd) {
        F_DUPFD, F_DUPFD_CLOEXEC, F_SETFD, F_SETFL, F_SETOWN => {
            arg = @cVaArg(&args, c_ulong);
        },
        F_GETLK, F_SETLK, F_SETLKW => {
            // flock pointer argument - not fully implemented
            _ = @cVaArg(&args, *Flock);
        },
        else => {},
    }
    @cVaEnd(&args);
    
    const rc = std.posix.system.fcntl(fd, cmd, arg);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn posix_fadvise(fd: c_int, offset: i64, len: i64, advice: c_int) c_int {
    if (@hasDecl(std.posix.system, "fadvise")) {
        const rc = std.posix.system.fadvise(fd, offset, len, advice);
        if (failIfErrno(rc)) return -1;
        return 0;
    }
    // Not supported: indicate success
    return 0;
}

pub export fn posix_fallocate(fd: c_int, offset: i64, len: i64) c_int {
    if (@hasDecl(std.posix.system, "fallocate")) {
        const mode: c_int = 0; // Default mode
        const rc = std.posix.system.fallocate(fd, mode, offset, len);
        if (failIfErrno(rc)) return -1;
        return 0;
    }
    return 0;
}

// Additional file control functions

pub export fn flock(fd: c_int, operation: c_int) c_int {
    if (@hasDecl(std.posix.system, "flock")) {
        const rc = std.posix.system.flock(fd, operation);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

// lockf commands
const F_ULOCK: c_int = 0;  // Unlock
const F_LOCK: c_int = 1;   // Lock (blocking)
const F_TLOCK: c_int = 2;  // Try lock (non-blocking)
const F_TEST: c_int = 3;   // Test lock

pub export fn lockf(fd: c_int, cmd: c_int, len: i64) c_int {
    // Implement lockf using fcntl with file locks
    // lockf uses the current file offset, locks len bytes from there
    var fl = Flock{
        .l_type = F_WRLCK,  // Write lock (exclusive)
        .l_whence = 1,      // SEEK_CUR
        .l_start = 0,       // From current position
        .l_len = len,       // Length to lock (0 = until EOF)
        .l_pid = 0,
    };

    switch (cmd) {
        F_ULOCK => {
            // Unlock the region
            fl.l_type = F_UNLCK;
            return fcntl(fd, F_SETLK, @intFromPtr(&fl));
        },
        F_LOCK => {
            // Lock with blocking
            fl.l_type = F_WRLCK;
            return fcntl(fd, F_SETLKW, @intFromPtr(&fl));
        },
        F_TLOCK => {
            // Try lock without blocking
            fl.l_type = F_WRLCK;
            return fcntl(fd, F_SETLK, @intFromPtr(&fl));
        },
        F_TEST => {
            // Test if lock would succeed
            fl.l_type = F_WRLCK;
            const result = fcntl(fd, F_GETLK, @intFromPtr(&fl));
            if (result == -1) return -1;
            // If l_type is F_UNLCK, no conflicting lock exists
            if (fl.l_type == F_UNLCK) {
                return 0;
            } else {
                setErrno(.ACCES);
                return -1;
            }
        },
        else => {
            setErrno(.INVAL);
            return -1;
        },
    }
}

pub export fn splice(fd_in: c_int, off_in: ?*i64, fd_out: c_int, off_out: ?*i64, len: usize, flags: c_uint) isize {
    if (@hasDecl(std.posix.system, "splice")) {
        const rc = std.posix.system.splice(fd_in, off_in, fd_out, off_out, len, flags);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn sync_file_range(fd: c_int, offset: i64, nbytes: i64, flags: c_uint) c_int {
    if (@hasDecl(std.posix.system, "sync_file_range")) {
        const rc = std.posix.system.sync_file_range(fd, offset, nbytes, flags);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    return 0;
}

pub export fn readahead(fd: c_int, offset: i64, count: usize) isize {
    if (@hasDecl(std.posix.system, "readahead")) {
        const rc = std.posix.system.readahead(fd, offset, count);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    return 0;
}
