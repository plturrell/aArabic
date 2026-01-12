const std = @import("std");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();
    
    // Try stringifyArbitraryDepth or Value.jsonStringify
    const value = std.json.Value{ .string = "test" };
    var buffer: std.ArrayList(u8) = .{};
    defer buffer.deinit(allocator);
    
    try value.jsonStringify(.{}, buffer.writer(allocator));
    std.debug.print("Success: {s}\n", .{buffer.items});
}
