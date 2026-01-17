const std = @import("std");

test "arraylist init test - correct syntax" {
    const ArrayListType = std.ArrayList(u8);
    var buffer = ArrayListType.init(std.testing.allocator);
    defer buffer.deinit();
    
    try buffer.append('H');
    try std.testing.expectEqual(@as(usize, 1), buffer.items.len);
}
