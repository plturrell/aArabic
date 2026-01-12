const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Test json.stringify in 0.15
    const value = std.json.Value{ .string = "test" };
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    
    try std.json.stringify(&value, .{}, buffer.writer(allocator));
    std.debug.print("Success: {s}\n", .{buffer.items});
}
