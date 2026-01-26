//! SAP HANA Client - Uses centralized HANA SDK
//!
//! This wrapper provides backward compatibility while using the new centralized HANA SDK
//! located at: /Users/user/Documents/arabic_folder/src/nLang/n-c-sdk/lib/hana/

const std = @import("std");

// Import centralized HANA SDK wrapper
const hana = @import("../../data/hana_client.zig");

// Re-export all types from the SDK
pub const HanaError = hana.HanaError;
pub const HanaConfig = hana.HanaConfig;
pub const HanaClient = hana.HanaClient;
pub const QueryResult = hana.QueryResult;
pub const Row = hana.Row;
pub const Value = hana.Value;

// Legacy type aliases for backward compatibility
pub const HanaValue = Value;
pub const HanaRow = Row;
pub const HanaResult = QueryResult;

// ============================================================================
// Convenience Functions
// ============================================================================

/// Connect using the SDK
pub fn connect(config: HanaConfig) !*HanaClient {
    return hana.connect(config);
}

/// Connect with custom allocator
pub fn connectWithAllocator(allocator: std.mem.Allocator, config: HanaConfig) !*HanaClient {
    return hana.connectWithAllocator(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "SDK integration" {
    _ = hana_sdk;
    std.testing.refAllDecls(@This());
}
