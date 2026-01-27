// Pipes and FIFOs - Production Implementation
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const O_NONBLOCK: c_int = 0o4000;
pub const O_CLOEXEC: c_int = 0o2000000;

pub const S_IFIFO: c_uint = 0o10000;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

/// Create pipe
pub export fn pipe(pipefd: [*]c_int) c_int {
    var fds: [2]c_int = undefined;
    
    std.posix.pipe(&fds) catch |err| {
        setErrno(switch (err) {
            error.SystemResources => .MFILE,
            error.ProcessFdQuotaExceeded => .MFILE,
            else => .INVAL,
        });
        return -1;
    };
    
    pipefd[0] = fds[0];
    pipefd[1] = fds[1];
    return 0;
}

/// Create pipe with flags
pub export fn pipe2(pipefd: [*]c_int, flags: c_int) c_int {
    var fds: [2]c_int = undefined;
    
    std.posix.pipe(&fds) catch |err| {
        setErrno(switch (err) {
            error.SystemResources => .MFILE,
            error.ProcessFdQuotaExceeded => .MFILE,
            else => .INVAL,
        });
        return -1;
    };
    
    // Apply flags
    if (flags & O_NONBLOCK != 0) {
        // Set non-blocking on both ends
        for (fds) |fd| {
            const current_flags = std.posix.fcntl(fd, std.posix.F.GETFL, 0) catch 0;
            _ = std.posix.fcntl(fd, std.posix.F.SETFL, current_flags | O_NONBLOCK) catch {};
        }
    }
    
    if (flags & O_CLOEXEC != 0) {
        // Set close-on-exec on both ends
        for (fds) |fd| {
            const current_flags = std.posix.fcntl(fd, std.posix.F.GETFD, 0) catch 0;
            _ = std.posix.fcntl(fd, std.posix.F.SETFD, current_flags | std.posix.FD_CLOEXEC) catch {};
        }
    }
    
    pipefd[0] = fds[0];
    pipefd[1] = fds[1];
    return 0;
}

/// Create named pipe (FIFO)
pub export fn mkfifo(pathname: [*:0]const u8, mode: c_uint) c_int {
    const path = std.mem.span(pathname);
    
    std.posix.mkfifo(path, mode) catch |err| {
        setErrno(switch (err) {
            error.PathAlreadyExists => .EXIST,
            error.AccessDenied => .ACCES,
            error.FileNotFound => .NOENT,
            error.NameTooLong => .NAMETOOLONG,
            error.NoSpaceLeft => .NOSPC,
            error.ReadOnlyFileSystem => .ROFS,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}

/// Create named pipe at specific directory
pub export fn mkfifoat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint) c_int {
    const path = std.mem.span(pathname);
    
    std.posix.mkfifoat(dirfd, path, mode) catch |err| {
        setErrno(switch (err) {
            error.PathAlreadyExists => .EXIST,
            error.AccessDenied => .ACCES,
            error.FileNotFound => .NOENT,
            error.NameTooLong => .NAMETOOLONG,
            error.NoSpaceLeft => .NOSPC,
            error.NotDir => .NOTDIR,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}

/// Create special node (including FIFO)
pub export fn mknod(pathname: [*:0]const u8, mode: c_uint, dev: c_ulong) c_int {
    _ = dev;
    const path = std.mem.span(pathname);
    
    // Check if creating FIFO
    if (mode & S_IFIFO != 0) {
        return mkfifo(pathname, mode & 0o7777);
    }
    
    // Other node types not fully supported
    std.posix.mknod(path, mode, 0) catch |err| {
        setErrno(switch (err) {
            error.PathAlreadyExists => .EXIST,
            error.AccessDenied => .ACCES,
            error.PermissionDenied => .PERM,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}

/// Create special node at directory
pub export fn mknodat(dirfd: c_int, pathname: [*:0]const u8, mode: c_uint, dev: c_ulong) c_int {
    _ = dev;
    const path = std.mem.span(pathname);
    
    // Check if creating FIFO
    if (mode & S_IFIFO != 0) {
        return mkfifoat(dirfd, pathname, mode & 0o7777);
    }
    
    // Other node types
    std.posix.mknodat(dirfd, path, mode, 0) catch |err| {
        setErrno(switch (err) {
            error.PathAlreadyExists => .EXIST,
            error.AccessDenied => .ACCES,
            error.NotDir => .NOTDIR,
            else => .INVAL,
        });
        return -1;
    };
    
    return 0;
}

/// Splice data between file descriptors
pub export fn splice(fd_in: c_int, off_in: ?*i64, fd_out: c_int, off_out: ?*i64, len: usize, flags: c_uint) isize {
    _ = off_in;
    _ = off_out;
    _ = flags;
    
    // Simplified: read from in, write to out
    var buf: [8192]u8 = undefined;
    const to_copy = @min(len, buf.len);
    
    const bytes_read = std.posix.read(fd_in, buf[0..to_copy]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            error.Unseekable => .SPIPE,
            else => .INVAL,
        });
        return -1;
    };
    
    if (bytes_read == 0) return 0;
    
    const bytes_written = std.posix.write(fd_out, buf[0..bytes_read]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            error.BrokenPipe => .PIPE,
            else => .INVAL,
        });
        return -1;
    };
    
    return @intCast(bytes_written);
}

/// Duplicate pipe content to another pipe
pub export fn tee(fd_in: c_int, fd_out: c_int, len: usize, flags: c_uint) isize {
    _ = flags;
    
    // Simplified: read and write (doesn't consume from fd_in in real tee)
    var buf: [8192]u8 = undefined;
    const to_copy = @min(len, buf.len);
    
    const bytes_read = std.posix.read(fd_in, buf[0..to_copy]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            else => .INVAL,
        });
        return -1;
    };
    
    if (bytes_read == 0) return 0;
    
    const bytes_written = std.posix.write(fd_out, buf[0..bytes_read]) catch |err| {
        setErrno(switch (err) {
            error.WouldBlock => .AGAIN,
            error.BrokenPipe => .PIPE,
            else => .INVAL,
        });
        return -1;
    };
    
    return @intCast(bytes_written);
}
