// Integration test runner for zig-libc
// Phase 1.1: Foundation

const std = @import("std");
const zig_libc = @import("zig-libc");

// Import comprehensive integration tests
test {
    _ = @import("test_integration.zig");
}

test "integration - string and ctype together" {
    // Test string operations work with character operations
    const test_str: [*:0]const u8 = "Hello123";
    const len = zig_libc.string.strlen(test_str);
    
    try std.testing.expectEqual(@as(usize, 8), len);
    
    // Check first character is uppercase
    try std.testing.expect(zig_libc.ctype.isalpha(test_str[0]));
    try std.testing.expect(zig_libc.ctype.toupper(@as(i32, test_str[0])) == 'H');
    
    // Check last characters are digits
    try std.testing.expect(zig_libc.ctype.isdigit(@as(i32, test_str[len-1])));
}
