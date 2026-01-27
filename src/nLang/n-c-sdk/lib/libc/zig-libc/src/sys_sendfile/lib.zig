// sys_sendfile module - Phase 1.31
// Implements sendfile() to copy data between file descriptors
const std = @import("std");
const builtin = @import("builtin");
const posix = std.posix;

/// sendfile - copy data between file descriptors
/// Returns number of bytes written, or -1 on error (sets errno)
pub export fn sendfile(out_fd: c_int, in_fd: c_int, offset: ?*isize, count: usize) isize {
    return sendfileImpl(out_fd, in_fd, offset, count);
}

pub export fn sendfile64(out_fd: c_int, in_fd: c_int, offset: ?*i64, count: usize) isize {
    // On 64-bit systems, sendfile and sendfile64 are the same
    if (offset) |off_ptr| {
        var off_isize: isize = @intCast(off_ptr.*);
        const result = sendfileImpl(out_fd, in_fd, &off_isize, count);
        off_ptr.* = @intCast(off_isize);
        return result;
    }
    return sendfileImpl(out_fd, in_fd, null, count);
}

fn sendfileImpl(out_fd: c_int, in_fd: c_int, offset: ?*isize, count: usize) isize {
    // Use read/write fallback (works on all platforms)
    // Buffer size for copying
    const BUF_SIZE = 65536;
    var buf: [BUF_SIZE]u8 = undefined;

    var total_written: usize = 0;
    var remaining = count;

    // If offset is provided, seek to it first
    if (offset) |off_ptr| {
        const seek_result = posix.lseek(@intCast(in_fd), off_ptr.*, .set);
        if (seek_result == -1) return -1;
    }

    while (remaining > 0) {
        const to_read = @min(remaining, BUF_SIZE);

        // Read from input fd
        const read_result = posix.read(@intCast(in_fd), buf[0..to_read]);
        const bytes_read = read_result catch |err| {
            _ = err;
            if (total_written > 0) break; // Return partial write
            return -1;
        };

        if (bytes_read == 0) break; // EOF

        // Write to output fd
        var written: usize = 0;
        while (written < bytes_read) {
            const write_result = posix.write(@intCast(out_fd), buf[written..bytes_read]);
            const bytes_written = write_result catch |err| {
                _ = err;
                if (total_written > 0) break; // Return partial write
                return -1;
            };
            written += bytes_written;
        }

        total_written += written;
        remaining -= written;

        // Update offset if provided
        if (offset) |off_ptr| {
            off_ptr.* += @intCast(written);
        }
    }

    return @intCast(total_written);
}
