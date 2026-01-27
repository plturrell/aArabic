// String Literal Tests - Phase 2 Testing
// Tests for basename, dirname, ctime NUL termination
// Verifies fixes for buffer overrun vulnerabilities

const std = @import("std");
const testing = std.testing;
const libc = @import("zig-libc");
const utilities = libc.stdlib.utilities;
const time_funcs = libc.stdlib.time;

test "basename with simple path" {
    var path = [_:0]u8{'/', 'u', 's', 'r', '/', 'b', 'i', 'n', '/', 'l', 's', 0};
    const result = utilities.basename(&path);
    
    // Should return "ls"
    try testing.expectEqual(@as(u8, 'l'), result[0]);
    try testing.expectEqual(@as(u8, 's'), result[1]);
    try testing.expectEqual(@as(u8, 0), result[2]); // NUL terminated!
}

test "basename empty path returns dot" {
    var path = [_:0]u8{0};
    const result = utilities.basename(&path);
    
    // Should return "."
    try testing.expectEqual(@as(u8, '.'), result[0]);
    try testing.expectEqual(@as(u8, 0), result[1]); // NUL terminated!
    
    // Verify we can use strlen safely (no buffer overrun)
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 1), len);
}

test "basename root path returns slash" {
    var path = [_:0]u8{'/', 0};
    const result = utilities.basename(&path);
    
    // Should return "/"
    try testing.expectEqual(@as(u8, '/'), result[0]);
    try testing.expectEqual(@as(u8, 0), result[1]); // NUL terminated!
    
    // Verify strlen works
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 1), len);
}

test "basename with trailing slashes" {
    var path = [_:0]u8{'/', 't', 'm', 'p', '/', '/', '/', 0};
    const result = utilities.basename(&path);
    
    // Should return "tmp"
    try testing.expectEqual(@as(u8, 't'), result[0]);
    try testing.expectEqual(@as(u8, 'm'), result[1]);
    try testing.expectEqual(@as(u8, 'p'), result[2]);
}

test "dirname with simple path" {
    var path = [_:0]u8{'/', 'u', 's', 'r', '/', 'b', 'i', 'n', 0};
    const result = utilities.dirname(&path);
    
    // Should return "/usr"
    try testing.expectEqual(@as(u8, '/'), result[0]);
    try testing.expectEqual(@as(u8, 'u'), result[1]);
    try testing.expectEqual(@as(u8, 's'), result[2]);
    try testing.expectEqual(@as(u8, 'r'), result[3]);
}

test "dirname empty path returns dot" {
    var path = [_:0]u8{0};
    const result = utilities.dirname(&path);
    
    // Should return "."
    try testing.expectEqual(@as(u8, '.'), result[0]);
    try testing.expectEqual(@as(u8, 0), result[1]); // NUL terminated!
    
    // Verify strlen works (no buffer overrun)
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 1), len);
}

test "dirname root path returns slash" {
    var path = [_:0]u8{'/', 0};
    const result = utilities.dirname(&path);
    
    // Should return "/"
    try testing.expectEqual(@as(u8, '/'), result[0]);
    try testing.expectEqual(@as(u8, 0), result[1]); // NUL terminated!
    
    // Verify strlen works
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 1), len);
}

test "ctime returns NUL-terminated string" {
    const timestamp: i64 = 0;
    const result = time_funcs.ctime(&timestamp);
    
    // Should return "Thu Jan  1 00:00:00 1970\n\0" (26 chars including NUL)
    try testing.expectEqual(@as(u8, 'T'), result[0]);
    try testing.expectEqual(@as(u8, '\n'), result[24]);
    try testing.expectEqual(@as(u8, 0), result[25]); // NUL terminated!
    
    // Verify strlen works (no buffer overrun)
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 25), len); // 25 chars + NUL
}

test "ctime can be used with string functions" {
    const timestamp: i64 = 0;
    const result = time_funcs.ctime(&timestamp);
    
    // Should be safe to use with string.h functions
    const len = std.mem.len(result);
    try testing.expect(len > 0);
    try testing.expect(len < 100); // Reasonable length
    
    // Should be safe to copy
    var buffer: [50:0]u8 = undefined;
    @memcpy(buffer[0..len], result[0..len]);
    buffer[len] = 0;
    
    try testing.expectEqual(@as(u8, 'T'), buffer[0]);
}

test "basename result can be used with strlen" {
    var path = [_:0]u8{'/', 'h', 'o', 'm', 'e', '/', 'u', 's', 'e', 'r', 0};
    const result = utilities.basename(&path);
    
    // Should be safe to use strlen
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 4), len); // "user"
    
    // Verify contents
    try testing.expectEqual(@as(u8, 'u'), result[0]);
    try testing.expectEqual(@as(u8, 's'), result[1]);
    try testing.expectEqual(@as(u8, 'e'), result[2]);
    try testing.expectEqual(@as(u8, 'r'), result[3]);
    try testing.expectEqual(@as(u8, 0), result[4]);
}

test "dirname result can be used with strlen" {
    var path = [_:0]u8{'/', 'h', 'o', 'm', 'e', '/', 'u', 's', 'e', 'r', 0};
    const result = utilities.dirname(&path);
    
    // Should be safe to use strlen
    const len = std.mem.len(result);
    try testing.expectEqual(@as(usize, 5), len); // "/home"
}

test "basename special cases are NUL terminated" {
    // Test "." case
    var empty = [_:0]u8{0};
    const dot_result = utilities.basename(&empty);
    try testing.expectEqual(@as(u8, '.'), dot_result[0]);
    try testing.expectEqual(@as(u8, 0), dot_result[1]);
    
    // Test "/" case
    var root = [_:0]u8{'/', 0};
    const slash_result = utilities.basename(&root);
    try testing.expectEqual(@as(u8, '/'), slash_result[0]);
    try testing.expectEqual(@as(u8, 0), slash_result[1]);
}

test "dirname special cases are NUL terminated" {
    // Test "." case
    var empty = [_:0]u8{0};
    const dot_result = utilities.dirname(&empty);
    try testing.expectEqual(@as(u8, '.'), dot_result[0]);
    try testing.expectEqual(@as(u8, 0), dot_result[1]);
    
    // Test "/" case
    var root = [_:0]u8{'/', 0};
    const slash_result = utilities.dirname(&root);
    try testing.expectEqual(@as(u8, '/'), slash_result[0]);
    try testing.expectEqual(@as(u8, 0), slash_result[1]);
}

test "string literals are safe for string operations" {
    // All special return values should be safe for C string operations
    var empty = [_:0]u8{0};
    
    const base_dot = utilities.basename(&empty);
    const dir_dot = utilities.dirname(&empty);
    
    // Should be safe to compare, copy, etc.
    try testing.expect(std.mem.len(base_dot) < 100);
    try testing.expect(std.mem.len(dir_dot) < 100);
    
    // Should be safe to use in comparisons
    const is_dot_base = std.mem.eql(u8, std.mem.span(base_dot), ".");
    const is_dot_dir = std.mem.eql(u8, std.mem.span(dir_dot), ".");
    
    try testing.expect(is_dot_base);
    try testing.expect(is_dot_dir);
}

test "ctime always returns 26 character string" {
    const timestamps = [_]i64{ 0, 1000000, -1000000, 1234567890 };
    
    for (timestamps) |ts| {
        const result = time_funcs.ctime(&ts);
        const len = std.mem.len(result);
        
        // ctime format is always 25 chars + newline + NUL = 26 total
        try testing.expectEqual(@as(usize, 25), len);
        try testing.expectEqual(@as(u8, 0), result[25]);
    }
}
