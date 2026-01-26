//! SAP HANA Client - Uses centralized HANA SDK
//!
//! This wrapper provides backward compatibility while using the new centralized HANA SDK
//! located at: /Users/user/Documents/arabic_folder/src/nLang/n-c-sdk/lib/hana/

const std = @import("std");

// Import centralized HANA SDK
const hana_sdk = @import("hana_sdk");
const hana_client = hana_sdk.client;

// Re-export all types from the SDK
pub const HanaError = anyerror;
pub const HanaConfig = hana_sdk.Config;
pub const HanaClient = hana_sdk.Client;
pub const QueryResult = hana_sdk.QueryResult;
pub const Row = hana_client.Row;
pub const Value = hana_client.Value;

// Legacy type aliases for backward compatibility
pub const HanaValue = Value;
pub const HanaRow = Row;
pub const HanaResult = QueryResult;

// ============================================================================
// Convenience Functions
// ============================================================================

/// Connect using the SDK
pub fn connect(config: HanaConfig) !*HanaClient {
    return hana_sdk.connect(config);
}

/// Connect with custom allocator
pub fn connectWithAllocator(allocator: std.mem.Allocator, config: HanaConfig) !*HanaClient {
    return hana_sdk.connectWithAllocator(allocator, config);
}

// ============================================================================
// Tests
// ============================================================================

test "SDK integration" {
    _ = hana_sdk;
    std.testing.refAllDecls(@This());
}
