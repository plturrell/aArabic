const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Try different ArrayList syntax for 0.15
    var buffer = std.ArrayList(u8){};
    buffer.allocator = allocator;
    defer buffer.deinit();
    
    try buffer.appendSlice("test");
    std.debug.print("Success: {s}\n", .{buffer.items});
}
