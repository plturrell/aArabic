// Minimal stdio compatibility for Zig 0.15 using POSIX syscalls directly
const std = @import("std");

// Use POSIX write syscall directly
const STDOUT_FILENO = 1;

pub const stdout = struct {
    pub fn write(bytes: []const u8) void {
        _ = std.posix.write(STDOUT_FILENO, bytes) catch {};
    }
    
    pub fn print(comptime fmt: []const u8, args: anytype) void {
        var buf: [4096]u8 = undefined;
        const result = std.fmt.bufPrint(&buf, fmt, args) catch return;
        write(result);
    }
};

pub fn fprintf(stream: anytype, comptime fmt: []const u8, args: anytype) void {
    _ = stream;
    stdout.print(fmt, args);
}

pub fn fflush(stream: anytype) void {
    _ = stream;
    // No-op, writes are unbuffered
}