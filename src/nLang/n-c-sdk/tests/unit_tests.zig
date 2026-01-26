const std = @import("std");

/// Entry point for all unit tests
/// Imports and re-exports all unit test modules

pub const hana_tests = @import("unit/hana_test.zig");

test {
    // Automatically include all unit tests
    std.testing.refAllDecls(@This());
}