//! Keycloak Configuration - Day 34
//!
//! Configuration structures and validation for Keycloak client.

const std = @import("std");

/// Keycloak client configuration
pub const KeycloakConfig = struct {
    server_url: []const u8,
    realm: []const u8,
    client_id: []const u8,
    client_secret: []const u8,
    timeout_ms: u32 = 5000,
    max_retries: u8 = 3,
    verify_ssl: bool = true,
    
    /// Validate configuration
    pub fn validate(self: *const KeycloakConfig) !void {
        if (self.server_url.len == 0) {
            return error.InvalidServerUrl;
        }
        if (self.realm.len == 0) {
            return error.InvalidRealm;
        }
        if (self.client_id.len == 0) {
            return error.InvalidClientId;
        }
        if (self.client_secret.len == 0) {
            return error.InvalidClientSecret;
        }
        if (self.timeout_ms == 0) {
            return error.InvalidTimeout;
        }
    }
    
    /// Get token endpoint URL
    pub fn getTokenEndpoint(self: *const KeycloakConfig, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{s}/realms/{s}/protocol/openid-connect/token",
            .{ self.server_url, self.realm },
        );
    }
    
    /// Get userinfo endpoint URL
    pub fn getUserinfoEndpoint(self: *const KeycloakConfig, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{s}/realms/{s}/protocol/openid-connect/userinfo",
            .{ self.server_url, self.realm },
        );
    }
    
    /// Get logout endpoint URL
    pub fn getLogoutEndpoint(self: *const KeycloakConfig, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{s}/realms/{s}/protocol/openid-connect/logout",
            .{ self.server_url, self.realm },
        );
    }
    
    /// Get admin API base URL
    pub fn getAdminApiBase(self: *const KeycloakConfig, allocator: std.mem.Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{s}/admin/realms/{s}",
            .{ self.server_url, self.realm },
        );
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "KeycloakConfig validation - valid config" {
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    try config.validate();
}

test "KeycloakConfig validation - empty server URL" {
    const config = KeycloakConfig{
        .server_url = "",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    try std.testing.expectError(error.InvalidServerUrl, config.validate());
}

test "KeycloakConfig validation - empty realm" {
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    try std.testing.expectError(error.InvalidRealm, config.validate());
}

test "KeycloakConfig validation - empty client ID" {
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "",
        .client_secret = "test-secret",
    };
    
    try std.testing.expectError(error.InvalidClientId, config.validate());
}

test "KeycloakConfig validation - empty client secret" {
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "",
    };
    
    try std.testing.expectError(error.InvalidClientSecret, config.validate());
}

test "KeycloakConfig get token endpoint" {
    const allocator = std.testing.allocator;
    
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    const endpoint = try config.getTokenEndpoint(allocator);
    defer allocator.free(endpoint);
    
    try std.testing.expectEqualStrings(
        "http://localhost:8180/realms/nucleus-realm/protocol/openid-connect/token",
        endpoint,
    );
}

test "KeycloakConfig get userinfo endpoint" {
    const allocator = std.testing.allocator;
    
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    const endpoint = try config.getUserinfoEndpoint(allocator);
    defer allocator.free(endpoint);
    
    try std.testing.expectEqualStrings(
        "http://localhost:8180/realms/nucleus-realm/protocol/openid-connect/userinfo",
        endpoint,
    );
}

test "KeycloakConfig default values" {
    const config = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "nucleus-realm",
        .client_id = "nworkflow-service",
        .client_secret = "test-secret",
    };
    
    try std.testing.expectEqual(@as(u32, 5000), config.timeout_ms);
    try std.testing.expectEqual(@as(u8, 3), config.max_retries);
    try std.testing.expect(config.verify_ssl);
}
