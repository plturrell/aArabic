//! SAP HANA Client - Uses centralized HANA SDK
//! 
//! This wrapper provides backward compatibility while using the new centralized HANA SDK
//! located at: /Users/user/Documents/arabic_folder/src/nLang/n-c-sdk/zig-out/lib/zig/hana/

const std = @import("std");

// Import centralized HANA SDK
const hana_sdk = @import("../../../nLang/n-c-sdk/zig-out/lib/zig/hana/hana.zig");

// Re-export all types from the SDK
pub const HanaError = @import("../../data/hana_client.zig").HanaError;
pub const HanaConfig = hana_sdk.Config;
pub const HanaClient = hana_sdk.Client;
pub const QueryResult = hana_sdk.QueryResult;
pub const HanaConnection = hana_sdk.HanaConnection;
pub const HanaConnectionConfig = hana_sdk.HanaConnectionConfig;

// Re-export protocol types
pub const Protocol = hana_sdk.Protocol;
pub const Auth = hana_sdk.Auth;

// Re-export query types
pub const Query = hana_sdk.Query;
pub const ResultSet = hana_sdk.Query.ResultSet;
pub const Row = hana_sdk.Query.Row;
pub const Value = hana_sdk.Query.Value;

// Re-export Graph Engine
pub const Graph = hana_sdk.Graph;

// Re-export OData
pub const ODataClient = hana_sdk.ODataClient;
pub const ODataConfig = hana_sdk.ODataConfig;
pub const ODataConfigWithToken = hana_sdk.ODataConfigWithToken;

// Legacy type aliases for backward compatibility
pub const HanaValue = Value;
pub const HanaRow = Row;
pub const HanaResult = ResultSet;

// Connection pool re-export
pub const HanaConnectionPool = hana_sdk.Client; // The SDK client has built-in pooling

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