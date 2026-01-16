// Vendored from Zig stdlib, adapted for Zig 0.15.x compatibility.
// In Zig 0.12+, io.Reader was renamed to io.GenericReader.
// License: MIT (Zig project).
//
// NOTE: The LSP (main.zig) now uses low-level posix read/write directly,
// as Zig 0.15 completely restructured the Reader/Writer API to use vtables.
// This file is kept for potential use by other components.

const std = @import("std");
const io = std.io;

// Zig 0.15 uses GenericReader instead of Reader
const GenericReader = if (@hasDecl(io, "GenericReader")) io.GenericReader else io.Reader;

pub fn BufferedReader(comptime buffer_size: usize, comptime ReaderType: type) type {
    return struct {
        unbuffered_reader: ReaderType,
        buf: [buffer_size]u8 = undefined,
        start: usize = 0,
        end: usize = 0,

        pub const Error = ReaderType.Error;
        pub const Reader = GenericReader(*Self, Error, read);

        const Self = @This();

        pub fn read(self: *Self, dest: []u8) Error!usize {
            var dest_index: usize = 0;

            while (dest_index < dest.len) {
                const written = @min(dest.len - dest_index, self.end - self.start);
                @memcpy(dest[dest_index..][0..written], self.buf[self.start..][0..written]);
                if (written == 0) {
                    const n = try self.unbuffered_reader.read(self.buf[0..]);
                    if (n == 0) {
                        return dest_index;
                    }
                    self.start = 0;
                    self.end = n;
                }
                self.start += written;
                dest_index += written;
            }
            return dest.len;
        }

        pub fn reader(self: *Self) Reader {
            return .{ .context = self };
        }
    };
}

pub fn bufferedReader(reader: anytype) BufferedReader(4096, @TypeOf(reader)) {
    return .{ .unbuffered_reader = reader };
}

pub fn bufferedReaderSize(comptime size: usize, reader: anytype) BufferedReader(size, @TypeOf(reader)) {
    return .{ .unbuffered_reader = reader };
}
