const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    
    var list: std.ArrayList([]const u8) = .empty;
    defer list.deinit(allocator);
    
    try list.append(allocator, "hello");
    const slice = try list.toOwnedSlice(allocator);
    defer allocator.free(slice);
    
    std.debug.print("Success!\n", .{});
}
