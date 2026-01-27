// utime module - Phase 1.25
// Set file access and modification times
const std = @import("std");
const posix = std.posix;
const fs = std.fs;

pub const utimbuf = extern struct {
    actime: c_long, // Access time
    modtime: c_long, // Modification time
};

pub const timeval = extern struct {
    tv_sec: c_long,
    tv_usec: c_long,
};

/// Set file access and modification times (seconds precision)
pub export fn utime(filename: [*:0]const u8, times: ?*const utimbuf) c_int {
    const path = std.mem.span(filename);

    // Open file to get handle
    const file = fs.openFileAbsolute(path, .{ .mode = .read_write }) catch |err| {
        return errToErrno(err);
    };
    defer file.close();

    if (times) |t| {
        // Convert seconds to nanoseconds
        const atime_ns: i128 = @as(i128, t.actime) * std.time.ns_per_s;
        const mtime_ns: i128 = @as(i128, t.modtime) * std.time.ns_per_s;

        file.updateTimes(atime_ns, mtime_ns) catch |err| {
            return errToErrno(err);
        };
    } else {
        // NULL means set to current time
        const now = std.time.nanoTimestamp();
        file.updateTimes(now, now) catch |err| {
            return errToErrno(err);
        };
    }

    return 0;
}

/// Set file access and modification times (microsecond precision)
pub export fn utimes(filename: [*:0]const u8, times: ?*const [2]timeval) c_int {
    const path = std.mem.span(filename);

    const file = fs.openFileAbsolute(path, .{ .mode = .read_write }) catch |err| {
        return errToErrno(err);
    };
    defer file.close();

    if (times) |t| {
        // Convert to nanoseconds: seconds * 1e9 + microseconds * 1e3
        const atime_ns: i128 = @as(i128, t[0].tv_sec) * std.time.ns_per_s +
            @as(i128, t[0].tv_usec) * std.time.ns_per_us;
        const mtime_ns: i128 = @as(i128, t[1].tv_sec) * std.time.ns_per_s +
            @as(i128, t[1].tv_usec) * std.time.ns_per_us;

        file.updateTimes(atime_ns, mtime_ns) catch |err| {
            return errToErrno(err);
        };
    } else {
        const now = std.time.nanoTimestamp();
        file.updateTimes(now, now) catch |err| {
            return errToErrno(err);
        };
    }

    return 0;
}

/// futimes - set times on file descriptor
pub export fn futimes(fd: c_int, times: ?*const [2]timeval) c_int {
    const file = fs.File{ .handle = fd };

    if (times) |t| {
        const atime_ns: i128 = @as(i128, t[0].tv_sec) * std.time.ns_per_s +
            @as(i128, t[0].tv_usec) * std.time.ns_per_us;
        const mtime_ns: i128 = @as(i128, t[1].tv_sec) * std.time.ns_per_s +
            @as(i128, t[1].tv_usec) * std.time.ns_per_us;

        file.updateTimes(atime_ns, mtime_ns) catch {
            return -1;
        };
    } else {
        const now = std.time.nanoTimestamp();
        file.updateTimes(now, now) catch {
            return -1;
        };
    }

    return 0;
}

/// lutimes - set times on symlink (not following)
pub export fn lutimes(filename: [*:0]const u8, times: ?*const [2]timeval) c_int {
    // For symlinks, we need special handling
    // Most systems don't support changing symlink times directly
    // Fall back to utimes for now
    return utimes(filename, times);
}

fn errToErrno(err: anyerror) c_int {
    return switch (err) {
        error.AccessDenied => -13, // EACCES
        error.FileNotFound => -2, // ENOENT
        error.NotDir => -20, // ENOTDIR
        error.IsDir => -21, // EISDIR
        error.InvalidArgument => -22, // EINVAL
        else => -1,
    };
}
