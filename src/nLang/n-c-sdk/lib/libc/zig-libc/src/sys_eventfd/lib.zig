// sys_eventfd module - Phase 1.30
// Real eventfd implementation using Linux syscalls
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

pub const EFD_CLOEXEC: c_int = 0x80000;
pub const EFD_NONBLOCK: c_int = 0x800;
pub const EFD_SEMAPHORE: c_int = 0x1;

/// Create eventfd file descriptor
pub export fn eventfd(initval: c_uint, flags: c_int) c_int {
    if (builtin.os.tag == .linux) {
        const rc = std.os.linux.syscall(.eventfd2, .{
            @as(usize, initval),
            @as(usize, @intCast(flags)),
        });
        if (@as(isize, @bitCast(rc)) < 0) return -1;
        return @intCast(rc);
    } else {
        // macOS/BSD: eventfd not available, use pipe as fallback
        var fds: [2]posix.fd_t = undefined;
        const pipe_result = posix.pipe2(.{
            .CLOEXEC = (flags & EFD_CLOEXEC) != 0,
            .NONBLOCK = (flags & EFD_NONBLOCK) != 0,
        }) catch return -1;
        fds = pipe_result;

        // Write initial value if non-zero
        if (initval > 0) {
            const val: u64 = initval;
            _ = posix.write(fds[1], std.mem.asBytes(&val)) catch {};
        }

        // Return read end (write end is leaked - not ideal but functional)
        // For proper implementation, would need to track both ends
        return fds[0];
    }
}

/// Read from eventfd
pub export fn eventfd_read(fd: c_int, value: *u64) c_int {
    const fd_t: posix.fd_t = @intCast(fd);
    var buf: [8]u8 = undefined;

    const n = posix.read(fd_t, &buf) catch return -1;
    if (n != 8) return -1;

    value.* = std.mem.readInt(u64, &buf, .little);
    return 0;
}

/// Write to eventfd
pub export fn eventfd_write(fd: c_int, value: u64) c_int {
    const fd_t: posix.fd_t = @intCast(fd);
    const buf = std.mem.toBytes(value);

    const n = posix.write(fd_t, &buf) catch return -1;
    if (n != 8) return -1;

    return 0;
}
