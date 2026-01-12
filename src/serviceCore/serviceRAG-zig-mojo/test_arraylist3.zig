const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Zig 0.15 uses initCapacity or just {}
    var buffer: std.ArrayList(u8) = .{};
    try buffer.ensureTotalCapacity(allocator, 10);
    defer buffer.deinit(allocator);
    
    try buffer.appendSlice(allocator, "test");
    std.debug.print("Success: {s}\n", .{buffer.items});
}
