///! Test program for Mojo FFI bridge
///! Verifies that Zig can communicate with Mojo through the C ABI

const std = @import("std");
const mojo = @import("mojo_bridge.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    std.debug.print("=== HyperShimmy FFI Bridge Test ===\n\n", .{});

    // Test 1: Initialize context
    std.debug.print("Test 1: Initializing Mojo context...\n", .{});
    const ctx = mojo.Context.init() catch |err| {
        std.debug.print("  ❌ Failed to initialize: {}\n", .{err});
        return err;
    };
    defer ctx.deinit();
    std.debug.print("  ✅ Context initialized successfully\n\n", .{});

    // Test 2: Check if initialized
    std.debug.print("Test 2: Checking if context is initialized...\n", .{});
    const is_init = ctx.isInitialized();
    if (is_init) {
        std.debug.print("  ✅ Context is initialized\n\n", .{});
    } else {
        std.debug.print("  ❌ Context is not initialized\n\n", .{});
        return error.NotInitialized;
    }

    // Test 3: Get version
    std.debug.print("Test 3: Getting Mojo runtime version...\n", .{});
    const version = ctx.getVersion(allocator) catch |err| {
        std.debug.print("  ❌ Failed to get version: {}\n", .{err});
        return err;
    };
    defer allocator.free(version);
    std.debug.print("  ✅ Mojo runtime version: {s}\n\n", .{version});

    // Test 4: Test FFI string conversion
    std.debug.print("Test 4: Testing FFI string conversion...\n", .{});
    const test_str = "Hello from Zig!";
    const ffi_str = mojo.FFIString.init(test_str);
    const converted = ffi_str.toSlice();
    if (std.mem.eql(u8, test_str, converted)) {
        std.debug.print("  ✅ String conversion successful: \"{s}\"\n\n", .{converted});
    } else {
        std.debug.print("  ❌ String conversion failed\n\n", .{});
        return error.StringConversionFailed;
    }

    // Test 5: Test FFI buffer conversion
    std.debug.print("Test 5: Testing FFI buffer conversion...\n", .{});
    const test_buf = "Binary data test";
    const ffi_buf = mojo.FFIBuffer.init(test_buf);
    const buf_converted = ffi_buf.toSlice();
    if (std.mem.eql(u8, test_buf, buf_converted)) {
        std.debug.print("  ✅ Buffer conversion successful: \"{s}\"\n\n", .{buf_converted});
    } else {
        std.debug.print("  ❌ Buffer conversion failed\n\n", .{});
        return error.BufferConversionFailed;
    }

    // Test 6: Test error handling
    std.debug.print("Test 6: Testing error handling...\n", .{});
    ctx.clearError() catch |err| {
        std.debug.print("  ❌ Failed to clear error: {}\n", .{err});
        return err;
    };
    std.debug.print("  ✅ Error handling working correctly\n\n", .{});

    // Test 7: Test not-yet-implemented functions (should return error)
    std.debug.print("Test 7: Testing source creation (should return NotImplemented)...\n", .{});
    const source_id = ctx.createSource(
        allocator,
        "Test Source",
        .url,
        "https://example.com",
        "Test content",
    ) catch |err| {
        if (err == error.NotImplemented) {
            std.debug.print("  ✅ Correctly returned NotImplemented (will be implemented in Day 8)\n\n", .{});
        } else {
            std.debug.print("  ❌ Unexpected error: {}\n\n", .{err});
            return err;
        }
        // Get the error message from Mojo
        const error_msg = ctx.getLastError(allocator) catch |err2| {
            std.debug.print("  ⚠️  Could not get error message: {}\n", .{err2});
            return;
        };
        defer allocator.free(error_msg);
        std.debug.print("  📝 Mojo error message: \"{s}\"\n\n", .{error_msg});
        return;
    };
    defer allocator.free(source_id);

    std.debug.print("=== All tests passed! ===\n", .{});
}
