// unistd module - Phase 1.8 - POSIX functions
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// POSIX constants
pub const STDIN_FILENO: c_int = 0;
pub const STDOUT_FILENO: c_int = 1;
pub const STDERR_FILENO: c_int = 2;

pub const SEEK_SET: c_int = 0;
pub const SEEK_CUR: c_int = 1;
pub const SEEK_END: c_int = 2;

pub const F_OK: c_int = 0;
pub const X_OK: c_int = 1;
pub const W_OK: c_int = 2;
pub const R_OK: c_int = 4;

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

// File operations
pub export fn read(fd: c_int, buf: ?*anyopaque, count: usize) isize {
    if (count == 0) return 0;
    const ptr = buf orelse {
        setErrno(.FAULT);
        return -1;
    };

    const rc = std.posix.system.read(fd, ptr, count);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn write(fd: c_int, buf: ?*const anyopaque, count: usize) isize {
    if (count == 0) return 0;
    const ptr = buf orelse {
        setErrno(.FAULT);
        return -1;
    };

    const rc = std.posix.system.write(fd, ptr, count);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn close(fd: c_int) c_int {
    const rc = std.posix.system.close(fd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn lseek(fd: c_int, offset: i64, whence: c_int) i64 {
    const off: std.posix.off_t = @intCast(offset);
    const rc = std.posix.system.lseek(fd, off, whence);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn dup(fd: c_int) c_int {
    const rc = std.posix.system.dup(fd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn dup2(oldfd: c_int, newfd: c_int) c_int {
    const rc = std.posix.system.dup2(oldfd, newfd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn pipe(pipefd: *[2]c_int) c_int {
    const rc = std.posix.system.pipe(pipefd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn access(pathname: [*:0]const u8, mode: c_int) c_int {
    const rc = std.posix.system.access(pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn unlink(pathname: [*:0]const u8) c_int {
    const rc = std.posix.system.unlink(pathname);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn rmdir(pathname: [*:0]const u8) c_int {
    const rc = std.posix.system.rmdir(pathname);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn link(oldpath: [*:0]const u8, newpath: [*:0]const u8) c_int {
    const rc = std.posix.system.link(oldpath, newpath);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn symlink(target: [*:0]const u8, linkpath: [*:0]const u8) c_int {
    const rc = std.posix.system.symlink(target, linkpath);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn readlink(pathname: [*:0]const u8, buf: [*:0]u8, bufsiz: usize) isize {
    const raw_buf: [*]u8 = @ptrCast(buf);
    const rc = std.posix.system.readlink(pathname, raw_buf, bufsiz);
    if (failIfErrno(rc)) return -1;

    if (rc >= 0 and @as(usize, @intCast(rc)) < bufsiz) {
        buf[@intCast(rc)] = 0;
    }
    return rc;
}

pub export fn chdir(path: [*:0]const u8) c_int {
    const rc = std.posix.system.chdir(path);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fchdir(fd: c_int) c_int {
    const rc = std.posix.system.fchdir(fd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getcwd(buf: [*:0]u8, size: usize) ?[*:0]u8 {
    if (size == 0) {
        setErrno(.INVAL);
        return null;
    }

    const rc = std.posix.system.getcwd(@ptrCast(buf), size);
    if (rc == null) {
        setErrno(std.posix.errno(@as(c_int, -1)));
        return null;
    }
    return rc;
}

pub export fn mkdir(pathname: [*:0]const u8, mode: c_uint) c_int {
    const rc = std.posix.system.mkdir(pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn chmod(pathname: [*:0]const u8, mode: c_uint) c_int {
    const rc = std.posix.system.chmod(pathname, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fchmod(fd: c_int, mode: c_uint) c_int {
    const rc = std.posix.system.fchmod(fd, mode);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn chown(pathname: [*:0]const u8, owner: c_uint, group: c_uint) c_int {
    const rc = std.posix.system.chown(pathname, owner, group);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fchown(fd: c_int, owner: c_uint, group: c_uint) c_int {
    const rc = std.posix.system.fchown(fd, owner, group);
    if (failIfErrno(rc)) return -1;
    return rc;
}

// Process operations
pub export fn fork() c_int {
    const rc = std.posix.system.fork();
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn execve(pathname: [*:0]const u8, argv: [*:null]?[*:0]const u8, envp: [*:null]?[*:0]const u8) c_int {
    const rc = std.posix.system.execve(pathname, argv, envp);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn execv(pathname: [*:0]const u8, argv: [*:null]?[*:0]const u8) c_int {
    const rc = std.posix.system.execv(pathname, argv);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn execvp(file: [*:0]const u8, argv: [*:null]?[*:0]const u8) c_int {
    const rc = std.posix.system.execvp(file, argv);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getpid() c_int {
    return std.posix.system.getpid();
}

pub export fn getppid() c_int {
    return std.posix.system.getppid();
}

pub export fn getuid() c_uint {
    return std.posix.system.getuid();
}

pub export fn geteuid() c_uint {
    return std.posix.system.geteuid();
}

pub export fn getgid() c_uint {
    return std.posix.system.getgid();
}

pub export fn getegid() c_uint {
    return std.posix.system.getegid();
}

pub export fn setuid(uid: c_uint) c_int {
    const rc = std.posix.system.setuid(uid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setgid(gid: c_uint) c_int {
    const rc = std.posix.system.setgid(gid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn seteuid(euid: c_uint) c_int {
    const rc = std.posix.system.seteuid(euid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setegid(egid: c_uint) c_int {
    const rc = std.posix.system.setegid(egid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn sleep(seconds: c_uint) c_uint {
    return std.posix.system.sleep(seconds);
}

pub export fn usleep(usec: c_uint) c_int {
    const rc = std.posix.system.usleep(usec);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn isatty(fd: c_int) c_int {
    const rc = std.posix.system.isatty(fd);
    if (rc == 0) {
        setErrno(std.posix.errno(@as(c_int, -1)));
    }
    return rc;
}

pub export fn ttyname(fd: c_int) ?[*:0]u8 {
    const rc = std.posix.system.ttyname(fd);
    if (rc == null) {
        setErrno(std.posix.errno(@as(c_int, -1)));
    }
    return rc;
}

pub export fn getopt(argc: c_int, argv: [*:null]?[*:0]u8, optstring: [*:0]const u8) c_int {
    _ = argc; _ = argv; _ = optstring;
    return -1;
}

pub export fn getlogin() ?[*:0]u8 {
    const rc = std.posix.system.getlogin();
    if (rc == null) {
        setErrno(std.posix.errno(@as(c_int, -1)));
    }
    return @constCast(rc);
}

pub export fn gethostname(name: [*:0]u8, len: usize) c_int {
    const rc = std.posix.system.gethostname(@ptrCast(name), len);
    if (failIfErrno(rc)) return -1;
    if (len > 0) {
        name[len - 1] = 0;
    }
    return rc;
}

pub export fn sethostname(name: [*:0]const u8, len: usize) c_int {
    if (@hasDecl(std.posix.system, "sethostname")) {
        const rc = std.posix.system.sethostname(name, len);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn getpagesize() c_int {
    return @intCast(std.mem.page_size);
}

pub export fn truncate(path: [*:0]const u8, length: i64) c_int {
    const off: std.posix.off_t = @intCast(length);
    const rc = std.posix.system.truncate(path, off);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn ftruncate(fd: c_int, length: i64) c_int {
    const off: std.posix.off_t = @intCast(length);
    const rc = std.posix.system.ftruncate(fd, off);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fsync(fd: c_int) c_int {
    const rc = std.posix.system.fsync(fd);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn fdatasync(fd: c_int) c_int {
    if (@hasDecl(std.posix.system, "fdatasync")) {
        const rc = std.posix.system.fdatasync(fd);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    return fsync(fd);
}

pub export fn sync() void {
    std.posix.system.sync();
}

pub export fn pathconf(path: [*:0]const u8, name: c_int) c_long {
    _ = path; _ = name;
    return -1;
}

pub export fn fpathconf(fd: c_int, name: c_int) c_long {
    _ = fd; _ = name;
    return -1;
}

pub export fn sysconf(name: c_int) c_long {
    _ = name;
    return -1;
}

// === Additional POSIX Functions ===

pub export fn pread(fd: c_int, buf: ?*anyopaque, count: usize, offset: i64) isize {
    const ptr = buf orelse {
        setErrno(.FAULT);
        return -1;
    };
    const off: std.posix.off_t = @intCast(offset);
    const rc = std.posix.system.pread(fd, ptr, count, off);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn pwrite(fd: c_int, buf: ?*const anyopaque, count: usize, offset: i64) isize {
    const ptr = buf orelse {
        setErrno(.FAULT);
        return -1;
    };
    const off: std.posix.off_t = @intCast(offset);
    const rc = std.posix.system.pwrite(fd, ptr, count, off);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn alarm(seconds: c_uint) c_uint {
    return std.posix.system.alarm(seconds);
}

pub export fn pause() c_int {
    const rc = std.posix.system.pause();
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn chroot(path: [*:0]const u8) c_int {
    const rc = std.posix.system.chroot(path);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getdtablesize() c_int {
    return 1024; // Common default
}

pub export fn getgroups(size: c_int, list: [*]c_uint) c_int {
    if (size < 0) {
        setErrno(.INVAL);
        return -1;
    }
    const rc = std.posix.system.getgroups(@intCast(size), list);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setgroups(size: usize, list: [*]const c_uint) c_int {
    const rc = std.posix.system.setgroups(size, list);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setpgid(pid: c_int, pgid: c_int) c_int {
    const rc = std.posix.system.setpgid(pid, pgid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getpgid(pid: c_int) c_int {
    const rc = std.posix.system.getpgid(pid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getpgrp() c_int {
    return std.posix.system.getpgrp();
}

pub export fn setpgrp() c_int {
    const rc = std.posix.system.setpgid(0, 0);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setsid() c_int {
    const rc = std.posix.system.setsid();
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getsid(pid: c_int) c_int {
    const rc = std.posix.system.getsid(pid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn nice(inc: c_int) c_int {
    const rc = std.posix.system.nice(inc);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn _exit(status: c_int) noreturn {
    std.posix.system.exit_group(status);
}

pub export fn confstr(name: c_int, buf: ?[*:0]u8, len: usize) usize {
    _ = name;
    if (buf) |b| {
        if (len > 0) b[0] = 0;
    }
    return 0;
}

pub export fn setreuid(ruid: c_uint, euid: c_uint) c_int {
    const rc = std.posix.system.setreuid(ruid, euid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn setregid(rgid: c_uint, egid: c_uint) c_int {
    const rc = std.posix.system.setregid(rgid, egid);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn getresuid(ruid: *c_uint, euid: *c_uint, suid: *c_uint) c_int {
    if (@hasDecl(std.posix.system, "getresuid")) {
        const rc = std.posix.system.getresuid(ruid, euid, suid);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn getresgid(rgid: *c_uint, egid: *c_uint, sgid: *c_uint) c_int {
    if (@hasDecl(std.posix.system, "getresgid")) {
        const rc = std.posix.system.getresgid(rgid, egid, sgid);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn setresuid(ruid: c_uint, euid: c_uint, suid: c_uint) c_int {
    if (@hasDecl(std.posix.system, "setresuid")) {
        const rc = std.posix.system.setresuid(ruid, euid, suid);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}

pub export fn setresgid(rgid: c_uint, egid: c_uint, sgid: c_uint) c_int {
    if (@hasDecl(std.posix.system, "setresgid")) {
        const rc = std.posix.system.setresgid(rgid, egid, sgid);
        if (failIfErrno(rc)) return -1;
        return rc;
    }
    setErrno(.NOSYS);
    return -1;
}
