// Compatibility I/O layer for Zig 0.15
// Provides stdout using libc primitives

const std = @import("std");
const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("unistd.h");
});

pub const StdoutWriter = struct {
    pub const Error = error{
        WriteFailed,
    };

    pub const Writer = std.io.Writer(
        StdoutWriter,
        Error,
        write,
    );

    pub fn writer() Writer {
        return .{ .context = .{} };
    }

    pub fn write(_: StdoutWriter, bytes: []const u8) Error!usize {
        const result = c.write(c.STDOUT_FILENO, bytes.ptr, bytes.len);
        if (result < 0) {
            return Error.WriteFailed;
        }
        return @intCast(result);
    }
};

pub fn getStdOut() StdoutWriter {
    return .{};
}