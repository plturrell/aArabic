//! SAP HANA Client for nAgentFlow - Uses the centralized HANA SDK
//!
//! This module provides a lightweight wrapper around the centralized HANA SDK
//!
//! The SDK provides:
//! - Native SAP HANA SQL Command Network Protocol v2.0
//! - Connection pooling with ODBC
//! - OData v4 REST API persistence
//! - HANA Graph Engine (10x faster lineage queries)
//! - Enterprise authentication (SCRAM-SHA-256, JWT, SAML)
//!
//! Usage:
//!   const hana = @import("hana_sdk");
//!   var client = try hana.connect(config);
//!   defer client.deinit();

const std = @import("std");

// Import the centralized HANA SDK that build.zig wires in for us
const hana = @import("hana_sdk");
const hana_client = hana.client;

// Re-export SDK types (only those that exist in the SDK)
pub const HanaError = anyerror;
pub const HanaConfig = hana.Config;
pub const HanaClient = hana.Client;
pub const QueryResult = hana.QueryResult;
pub const Row = hana_client.Row;
pub const Value = hana_client.Value;
pub const Parameter = hana_client.Parameter;

// Re-export caching helpers
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

/// Load configuration from environment variables
pub fn loadConfigFromEnv(allocator: std.mem.Allocator) !HanaConfig {
    return hana.client.loadConfigFromEnv(allocator);
}

/// Execute a query using the provided allocator (compat helper)
pub fn queryWithAllocator(client: *HanaClient, allocator: std.mem.Allocator, sql: []const u8) !QueryResult {
    return client.query(sql, allocator);
}

/// Execute a statement (compat helper)
pub fn execute(client: *HanaClient, sql: []const u8) !void {
    return client.execute(sql);
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
    _ = QueryResult;
    _ = Value;
}
