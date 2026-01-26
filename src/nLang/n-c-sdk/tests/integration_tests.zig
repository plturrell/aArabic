const std = @import("std");

/// Entry point for all integration tests
/// Imports and re-exports all integration test modules

pub const http_integration = @import("integration/http_integration_test.zig");
pub const hana_integration = @import("integration/hana_integration_test.zig");

test {
    // Automatically include all integration tests
    std.testing.refAllDecls(@This());
}