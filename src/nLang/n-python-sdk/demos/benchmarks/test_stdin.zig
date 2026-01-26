const std = @import("std");
pub fn main() !void {
    const stdin = std.io.getStdIn();
    _ = try stdin.reader().readByte();
}
