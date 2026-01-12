// DragonflyDB Client Test
// Tests basic operations against a running DragonflyDB instance

const std = @import("std");
const dragonfly = @import("dragonfly_client.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("🧪 DragonflyDB Client Test\n", .{});
    std.debug.print("=" ** 50 ++ "\n\n", .{});

    // Connect to DragonflyDB
    std.debug.print("Connecting to localhost:6379...\n", .{});
    const client = try dragonfly.DragonflyClient.init(allocator, "127.0.0.1", 6379);
    defer client.deinit();
    
    std.debug.print("✅ Connected successfully!\n\n", .{});

    // Test 1: SET and GET
    std.debug.print("Test 1: SET and GET\n", .{});
    try client.set("test:key1", "Hello, Dragonfly!", null);
    std.debug.print("  SET test:key1 = 'Hello, Dragonfly!'\n", .{});
    
    const value1 = try client.get("test:key1");
    if (value1) |val| {
        defer allocator.free(val);
        std.debug.print("  GET test:key1 = '{s}'\n", .{val});
        std.debug.print("  ✅ PASS\n\n", .{});
    } else {
        std.debug.print("  ❌ FAIL: Key not found\n\n", .{});
        return error.TestFailed;
    }

    // Test 2: SET with expiration
    std.debug.print("Test 2: SET with expiration\n", .{});
    try client.set("test:key2", "Expires in 60s", 60);
    std.debug.print("  SET test:key2 = 'Expires in 60s' EX 60\n", .{});
    
    const value2 = try client.get("test:key2");
    if (value2) |val| {
        defer allocator.free(val);
        std.debug.print("  GET test:key2 = '{s}'\n", .{val});
        std.debug.print("  ✅ PASS\n\n", .{});
    } else {
        std.debug.print("  ❌ FAIL: Key not found\n\n", .{});
        return error.TestFailed;
    }

    // Test 3: EXISTS
    std.debug.print("Test 3: EXISTS\n", .{});
    const keys_to_check = [_][]const u8{ "test:key1", "test:key2", "test:nonexistent" };
    const exists_count = try client.exists(&keys_to_check);
    std.debug.print("  EXISTS test:key1 test:key2 test:nonexistent = {d}\n", .{exists_count});
    if (exists_count == 2) {
        std.debug.print("  ✅ PASS (2 keys exist)\n\n", .{});
    } else {
        std.debug.print("  ❌ FAIL: Expected 2, got {d}\n\n", .{exists_count});
        return error.TestFailed;
    }

    // Test 4: DEL
    std.debug.print("Test 4: DEL\n", .{});
    const keys_to_delete = [_][]const u8{ "test:key1", "test:key2" };
    const deleted_count = try client.del(&keys_to_delete);
    std.debug.print("  DEL test:key1 test:key2 = {d} keys deleted\n", .{deleted_count});
    if (deleted_count == 2) {
        std.debug.print("  ✅ PASS\n\n", .{});
    } else {
        std.debug.print("  ❌ FAIL: Expected 2, got {d}\n\n", .{deleted_count});
        return error.TestFailed;
    }

    // Test 5: GET non-existent key
    std.debug.print("Test 5: GET non-existent key\n", .{});
    const value3 = try client.get("test:key1");
    if (value3) |val| {
        defer allocator.free(val);
        std.debug.print("  ❌ FAIL: Found value for deleted key: '{s}'\n\n", .{val});
        return error.TestFailed;
    } else {
        std.debug.print("  GET test:key1 = (nil)\n", .{});
        std.debug.print("  ✅ PASS\n\n", .{});
    }

    // Test 6: MGET
    std.debug.print("Test 6: MGET (multiple keys)\n", .{});
    try client.set("test:a", "value_a", null);
    try client.set("test:b", "value_b", null);
    try client.set("test:c", "value_c", null);
    std.debug.print("  SET test:a, test:b, test:c\n", .{});
    
    const mget_keys = [_][]const u8{ "test:a", "test:b", "test:c", "test:d" };
    const mget_results = try client.mget(&mget_keys);
    defer {
        for (mget_results) |result| {
            if (result) |val| allocator.free(val);
        }
        allocator.free(mget_results);
    }
    
    std.debug.print("  MGET results:\n", .{});
    for (mget_results, 0..) |result, i| {
        if (result) |val| {
            std.debug.print("    [{d}] = '{s}'\n", .{ i, val });
        } else {
            std.debug.print("    [{d}] = (nil)\n", .{i});
        }
    }
    
    if (mget_results.len == 4 and mget_results[3] == null) {
        std.debug.print("  ✅ PASS\n\n", .{});
    } else {
        std.debug.print("  ❌ FAIL\n\n", .{});
        return error.TestFailed;
    }

    // Cleanup
    const cleanup_keys = [_][]const u8{ "test:a", "test:b", "test:c" };
    _ = try client.del(&cleanup_keys);

    std.debug.print("=" ** 50 ++ "\n", .{});
    std.debug.print("🎉 All tests PASSED!\n", .{});
    std.debug.print("\n✨ DragonflyDB Zig client is working!\n", .{});
    std.debug.print("   Target: 10-20x faster than Python\n", .{});
    std.debug.print("   Features: RESP protocol, connection pooling, C ABI\n", .{});
}
