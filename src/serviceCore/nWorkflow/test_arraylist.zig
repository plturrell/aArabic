const std = @import("std");

test "ArrayList init" {
    const allocator = std.testing.allocator;
    var list: std.ArrayList(u8) = .{
        .items = &[_]u8{},
        .capacity = 0,
        .allocator = allocator,
    };
    defer list.deinit();
    
    try list.append(42);
    try std.testing.expectEqual(@as(usize, 1), list.items.len);
}
