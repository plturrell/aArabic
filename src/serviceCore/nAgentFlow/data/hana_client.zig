//! SAP HANA Client for nAgentFlow - Uses the centralized HANA SDK
//!
//! This module provides a lightweight wrapper around the HANA SDK located at:
//! /Users/user/Documents/arabic_folder/src/nLang/n-c-sdk/zig-out/lib/zig/hana/
//!
//! The SDK provides:
//! - Native SAP HANA SQL Command Network Protocol v2.0
//! - Connection pooling with ODBC
//! - OData v4 REST API persistence
//! - HANA Graph Engine (10x faster lineage queries)
//! - Enterprise authentication (SCRAM-SHA-256, JWT, SAML)
//!
//! Usage:
//!   const hana_sdk = @import("../../../nLang/n-c-sdk/zig-out/lib/zig/hana/hana.zig");
//!   var client = try hana_sdk.connect(config);
//!   defer client.deinit();

const std = @import("std");

// Import the centralized HANA SDK
const hana_sdk_path = "../../../nLang/n-c-sdk/zig-out/lib/zig/hana/hana.zig";
const hana = @import(hana_sdk_path);

// Re-export types from HANA SDK for convenience
pub const HanaError = error{
    ConnectionFailed,
    AuthenticationFailed,
    TlsInitFailed,
    TlsHandshakeFailed,
    QueryFailed,
    ExecuteFailed,
    TransactionError,
    PrepareStatementFailed,
    InvalidParameter,
    Timeout,
    PoolExhausted,
    AlreadyConnected,
    NotConnected,
    AlreadyInTransaction,
    NotInTransaction,
    InvalidConfig,
    ProtocolError,
    EndOfStream,
    OutOfMemory,
    UnsupportedAuthType,
    PasswordRequired,
    SchemaNotSet,
    StatementNotPrepared,
    InvalidResponse,
};

// Re-export SDK types
pub const HanaConfig = hana.Config;
pub const HanaClient = hana.Client;
pub const QueryResult = hana.QueryResult;

// Re-export native protocol types
pub const HanaConnection = hana.HanaConnection;
pub const HanaConnectionConfig = hana.HanaConnectionConfig;
pub const Protocol = hana.Protocol;
pub const Auth = hana.Auth;

// Re-export query execution types
pub const Query = hana.Query;
pub const ResultSet = hana.Query.ResultSet;
pub const Row = hana.Query.Row;
pub const Value = hana.Query.Value;

// Re-export Graph Engine
pub const Graph = hana.Graph;

// Re-export OData
pub const ODataClient = hana.ODataClient;
pub const ODataConfig = hana.ODataConfig;
pub const ODataConfigWithToken = hana.ODataConfigWithToken;

// Re-export caching
pub const LRUCache = hana.LRUCache;
pub const QueryCache = hana.QueryCache;
pub const CacheStats = hana.CacheStats;

// ============================================================================
// Convenience Functions (forward to SDK)
// ============================================================================

/// Connect using ODBC (legacy)
pub fn connect(config: HanaConfig) !*HanaClient {
    return hana.connect(config);
}

/// Connect with custom allocator
pub fn connectWithAllocator(allocator: std.mem.Allocator, config: HanaConfig) !*HanaClient {
    return hana.connectWithAllocator(allocator, config);
}

/// Connect using OData with JWT token
pub fn odataConnectWithToken(allocator: std.mem.Allocator, config: ODataConfigWithToken) !*ODataClient {
    return hana.odataConnectWithToken(allocator, config);
}

/// Load configuration from environment variables
pub fn loadConfigFromEnv(allocator: std.mem.Allocator) !HanaConfig {
    return hana.client.loadConfigFromEnv(allocator);
}

// ============================================================================
// nAgentFlow-specific Helpers
// ============================================================================

/// Create a connection pool for workflow execution
pub fn createPool(allocator: std.mem.Allocator, config: HanaConfig) !*HanaClient {
    // For now, return a single client
    // In future, could wrap hana.Client with pooling logic
    return hana.connectWithAllocator(allocator, config);
}

/// Execute workflow-specific query with retry logic
pub fn executeWorkflowQuery(client: *HanaClient, sql: []const u8, max_retries: u32) !QueryResult {
    var retries: u32 = 0;
    while (retries < max_retries) : (retries += 1) {
        return client.query(sql) catch |err| {
            if (retries == max_retries - 1) return err;
            // Wait before retry
            std.time.sleep(100 * std.time.ns_per_ms * (retries + 1));
            continue;
        };
    }
    return error.QueryFailed;
}

// ============================================================================
// Tests
// ============================================================================

test "HANA SDK integration" {
    // Verify we can import the SDK
    _ = hana;
    std.testing.refAllDecls(@This());
}

test "Type re-exports" {
    // Verify types are properly re-exported
    _ = HanaConfig;
    _ = HanaClient;
    _ = HanaConnection;
    _ = Protocol;
    _ = Auth;
    _ = Query;
    _ = Graph;
}