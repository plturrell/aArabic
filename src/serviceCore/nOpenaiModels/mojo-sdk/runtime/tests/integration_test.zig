// Mojo Runtime Integration Test
// Tests that all runtime components work together correctly
//
// This test simulates the lifecycle of a Mojo program:
// 1. Runtime initialization
// 2. Memory allocation and data structures
// 3. String/List operations
// 4. Cleanup and shutdown

const std = @import("std");
const runtime = @import("runtime");
const core = runtime.core;
const memory = runtime.memory;
const ffi = runtime.ffi;
const startup = runtime.startup;

// ============================================================================
// Integration Test: Full Lifecycle
// ============================================================================

test "full runtime lifecycle" {
    // 1. Initialize runtime (simulating program start)
    const test_args = [_][:0]const u8{ "test_program", "--mode", "integration" };
    try startup.startup(&test_args);
    defer startup.cleanup();

    // 2. Verify arguments are accessible
    const args = startup.getArgs();
    try std.testing.expect(args.count() == 3);
    try std.testing.expectEqualStrings("test_program", args.get(0).?);
    try std.testing.expectEqualStrings("--mode", args.get(1).?);
    try std.testing.expectEqualStrings("integration", args.get(2).?);

    // 3. Create and manipulate Mojo strings
    var greeting = try memory.MojoString.fromSlice("Hello, ");
    defer greeting.deinit();

    var name = try memory.MojoString.fromSlice("Mojo Runtime!");
    defer name.deinit();

    // Append strings
    try greeting.appendSlice(name.asSlice());
    try std.testing.expectEqualStrings("Hello, Mojo Runtime!", greeting.asSlice());

    // 4. Create and use a Mojo list
    var numbers = memory.MojoList(i64).empty();
    defer numbers.deinit();

    // Add some numbers
    try numbers.append(10);
    try numbers.append(20);
    try numbers.append(30);
    try std.testing.expect(numbers.length() == 3);

    // Verify values
    try std.testing.expect(numbers.get(0).? == 10);
    try std.testing.expect(numbers.get(1).? == 20);
    try std.testing.expect(numbers.get(2).? == 30);

    // 5. Create a reference-counted object
    const allocator = core.getAllocator();
    const data = try allocator.alloc(u8, 64);
    defer allocator.free(u8, data);

    // Fill with pattern
    @memset(data, 0xAB);
    try std.testing.expect(data[0] == 0xAB);
    try std.testing.expect(data[63] == 0xAB);

    // 6. Register and retrieve a callback
    var registry = ffi.CallbackRegistry.init(std.heap.page_allocator);
    defer registry.deinit();

    const test_callback = struct {
        fn call() void {}
    }.call;

    try registry.register("test_func", @ptrCast(&test_callback), "void()");
    const entry = registry.get("test_func");
    try std.testing.expect(entry != null);
    try std.testing.expectEqualStrings("void()", entry.?.signature);

    // 7. Verify runtime stats are tracking
    try std.testing.expect(core.stats.total_allocations > 0);

    // 8. Print final stats (only in debug)
    if (@import("builtin").mode == .Debug) {
        core.stats.print();
    }
}

// ============================================================================
// Integration Test: String Operations
// ============================================================================

test "string integration" {
    try core.initDefault();
    defer core.deinit();

    // Test SSO (Small String Optimization)
    var small = try memory.MojoString.fromSlice("Hi"); // Should use SSO
    defer small.deinit();
    try std.testing.expect(small.isSSO());
    try std.testing.expect(small.len == 2);

    // Test heap allocation (string > 23 bytes)
    var large = try memory.MojoString.fromSlice("This is a longer string that exceeds SSO limit");
    defer large.deinit();
    try std.testing.expect(!large.isSSO());
    try std.testing.expect(large.len == 46);

    // Test concatenation
    var result = try memory.MojoString.fromSlice("Part1");
    defer result.deinit();
    try result.appendSlice(" + Part2");
    try std.testing.expectEqualStrings("Part1 + Part2", result.asSlice());

    // Test substring via slice
    const sub = result.asSlice()[0..5];
    try std.testing.expectEqualStrings("Part1", sub);

    // Test indexOf via std.mem
    const idx = std.mem.indexOf(u8, result.asSlice(), "+");
    try std.testing.expect(idx == 6);

    // Test C string conversion
    const cstr = ffi.TypeConverter.stringToC(&result);
    try std.testing.expectEqualStrings("Part1 + Part2", std.mem.span(cstr));
}

// ============================================================================
// Integration Test: Collection Operations
// ============================================================================

test "collection integration" {
    try core.initDefault();
    defer core.deinit();

    // Create a list of strings
    var messages = memory.MojoList(memory.MojoString).empty();
    defer {
        // Clean up each string
        for (messages.asMutSlice()) |*s| {
            s.deinit();
        }
        messages.deinit();
    }

    // Add messages
    try messages.append(try memory.MojoString.fromSlice("First message"));
    try messages.append(try memory.MojoString.fromSlice("Second message"));
    try messages.append(try memory.MojoString.fromSlice("Third message"));

    try std.testing.expect(messages.length() == 3);
    try std.testing.expectEqualStrings("First message", messages.get(0).?.asSlice());
    try std.testing.expectEqualStrings("Second message", messages.get(1).?.asSlice());
    try std.testing.expectEqualStrings("Third message", messages.get(2).?.asSlice());

    // Test iteration
    var count: usize = 0;
    for (messages.asSlice()) |_| {
        count += 1;
    }
    try std.testing.expect(count == 3);
}

// ============================================================================
// Integration Test: Memory Safety
// ============================================================================

test "memory safety integration" {
    try core.initDefault();
    defer core.deinit();

    const allocator = core.getAllocator();

    // Allocate multiple blocks (store slices)
    var blocks: [10][]u8 = undefined;
    for (0..10) |i| {
        blocks[i] = try allocator.alloc(u8, 100);
        @memset(blocks[i], @intCast(i));
    }

    // Verify each block has correct content
    for (0..10) |i| {
        const byte: u8 = @intCast(i);
        try std.testing.expect(blocks[i][0] == byte);
        try std.testing.expect(blocks[i][99] == byte);
    }

    // Free in reverse order
    for (0..10) |i| {
        const idx = 9 - i;
        allocator.free(u8, blocks[idx]);
    }

    // Verify stats
    try std.testing.expect(core.stats.live_allocations == 0);
}

// ============================================================================
// Integration Test: Exit Handler
// ============================================================================

test "exit handler integration" {
    const test_args = [_][:0]const u8{"test"};
    try startup.startup(&test_args);
    defer startup.cleanup();

    // Track handler execution
    const TestState = struct {
        var executed: bool = false;
        var order: u32 = 0;

        fn handler1() void {
            executed = true;
            order = 1;
        }

        fn handler2() void {
            order = 2;
        }
    };

    // Register handlers
    try startup.atExit(&TestState.handler1);
    try startup.atExit(&TestState.handler2);

    // Handlers will be called in LIFO order during cleanup
    // We can't test the actual exit, but we verified registration works
}

// ============================================================================
// Integration Test: C ABI Compatibility
// ============================================================================

test "c abi integration" {
    try core.initDefault();
    defer core.deinit();

    // Test integer conversion
    const mojo_int: i64 = 12345;
    const converted_int = ffi.TypeConverter.intToC(mojo_int);
    const back_int = ffi.TypeConverter.intFromC(converted_int);
    try std.testing.expect(back_int == mojo_int);

    // Test float conversion
    const mojo_float: f64 = 3.14159;
    const c_float = ffi.TypeConverter.floatToC(mojo_float);
    const back_float = ffi.TypeConverter.floatFromC(c_float);
    try std.testing.expect(back_float == mojo_float);

    // Test bool conversion
    try std.testing.expect(ffi.TypeConverter.boolToC(true) == 1);
    try std.testing.expect(ffi.TypeConverter.boolToC(false) == 0);
    try std.testing.expect(ffi.TypeConverter.boolFromC(1) == true);
    try std.testing.expect(ffi.TypeConverter.boolFromC(0) == false);
    try std.testing.expect(ffi.TypeConverter.boolFromC(42) == true);
}

// ============================================================================
// Integration Test: Environment Access
// ============================================================================

test "environment integration" {
    try core.initDefault();
    defer core.deinit();

    // PATH should exist on most systems
    const path = startup.Env.get("PATH");
    try std.testing.expect(path != null);
    try std.testing.expect(path.?.len > 0);

    // Test non-existent variable
    const nonexistent = startup.Env.get("MOJO_TEST_VAR_DOES_NOT_EXIST");
    try std.testing.expect(nonexistent == null);

    // Test has() function
    try std.testing.expect(startup.Env.has("PATH"));
    try std.testing.expect(!startup.Env.has("MOJO_TEST_VAR_DOES_NOT_EXIST"));
}
