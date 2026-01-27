// Unit test runner for zig-libc
// Phase 1.1: Foundation

const std = @import("std");
const zig_libc = @import("zig-libc");

// Import all modules to run their tests
test {
    @import("std").testing.refAllDecls(@This());
    _ = zig_libc;
    _ = zig_libc.string;
    _ = zig_libc.ctype;
    _ = zig_libc.memory;
    _ = zig_libc.stdlib;
    _ = @import("test_stdlib.zig");  // Phase 1.2 - Week 25
}
