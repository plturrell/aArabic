//! Keycloak Client - Day 34
//!
//! Main client for Keycloak authentication and authorization operations.
//! Supports OAuth2 flows, token management, and user operations.

const std = @import("std");
const Allocator = std.mem.Allocator;
const http_client = @import("http_client.zig");
const types = @import("keycloak_types.zig");
const config_mod = @import("keycloak_config.zig");

const HttpClient = http_client.HttpClient;
const HttpResponse = http_client.HttpResponse;
const TokenResponse = types.TokenResponse;
const TokenInfo = types.TokenInfo;
const UserInfo = types.UserInfo;
const KeycloakConfig = config_mod.KeycloakConfig;

/// Main Keycloak client
pub const KeycloakClient = struct {
    allocator: Allocator,
    config: KeycloakConfig,
    http: HttpClient,
    
    pub fn init(allocator: Allocator, cfg: KeycloakConfig) !KeycloakClient {
        try cfg.validate();
        
        return KeycloakClient{
            .allocator = allocator,
            .config = cfg,
            .http = HttpClient.init(allocator),
        };
    }
    
    pub fn deinit(self: *KeycloakClient) void {
        self.http.deinit();
    }
    
    /// Get service token using client credentials flow
    pub fn getServiceToken(self: *KeycloakClient) !TokenResponse {
        const url = try self.config.getTokenEndpoint(self.allocator);
        defer self.allocator.free(url);
        
        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=client_credentials&client_id={s}&client_secret={s}",
            .{ self.config.client_id, self.config.client_secret },
        );
        defer self.allocator.free(body);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer headers.deinit();
        try headers.put("Content-Type", "application/x-www-form-urlencoded");
        
        var response = try self.http.post(url, headers, body);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 200) {
            std.log.err("Keycloak service token failed: {d} - {s}", .{ response.status_code, response.body });
            return error.AuthenticationFailed;
        }
        
        return try TokenResponse.fromJson(self.allocator, response.body);
    }
    
    /// Login with username and password
    pub fn login(
        self: *KeycloakClient,
        username: []const u8,
        password: []const u8,
    ) !TokenResponse {
        const url = try self.config.getTokenEndpoint(self.allocator);
        defer self.allocator.free(url);
        
        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=password&client_id={s}&client_secret={s}&username={s}&password={s}",
            .{ self.config.client_id, self.config.client_secret, username, password },
        );
        defer self.allocator.free(body);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer headers.deinit();
        try headers.put("Content-Type", "application/x-www-form-urlencoded");
        
        var response = try self.http.post(url, headers, body);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 200) {
            std.log.err("Keycloak login failed: {d} - {s}", .{ response.status_code, response.body });
            return error.LoginFailed;
        }
        
        return try TokenResponse.fromJson(self.allocator, response.body);
    }
    
    /// Validate and get info from token
    pub fn validateToken(
        self: *KeycloakClient,
        token: []const u8,
    ) !TokenInfo {
        const url = try self.config.getUserinfoEndpoint(self.allocator);
        defer self.allocator.free(url);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer {
            var iter = headers.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            headers.deinit();
        }
        
        const auth_header = try std.fmt.allocPrint(
            self.allocator,
            "Bearer {s}",
            .{token},
        );
        defer self.allocator.free(auth_header);
        
        const auth_key = try self.allocator.dupe(u8, "Authorization");
        const auth_value = try self.allocator.dupe(u8, auth_header);
        try headers.put(auth_key, auth_value);
        
        var response = try self.http.get(url, headers);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 200) {
            std.log.err("Token validation failed: {d}", .{response.status_code});
            return error.InvalidToken;
        }
        
        return try TokenInfo.fromJson(self.allocator, response.body);
    }
    
    /// Refresh access token
    pub fn refreshToken(
        self: *KeycloakClient,
        refresh_token: []const u8,
    ) !TokenResponse {
        const url = try self.config.getTokenEndpoint(self.allocator);
        defer self.allocator.free(url);
        
        const body = try std.fmt.allocPrint(
            self.allocator,
            "grant_type=refresh_token&client_id={s}&client_secret={s}&refresh_token={s}",
            .{ self.config.client_id, self.config.client_secret, refresh_token },
        );
        defer self.allocator.free(body);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer headers.deinit();
        try headers.put("Content-Type", "application/x-www-form-urlencoded");
        
        var response = try self.http.post(url, headers, body);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 200) {
            std.log.err("Token refresh failed: {d}", .{response.status_code});
            return error.RefreshFailed;
        }
        
        return try TokenResponse.fromJson(self.allocator, response.body);
    }
    
    /// Logout and invalidate tokens
    pub fn logout(
        self: *KeycloakClient,
        refresh_token: []const u8,
    ) !void {
        const url = try self.config.getLogoutEndpoint(self.allocator);
        defer self.allocator.free(url);
        
        const body = try std.fmt.allocPrint(
            self.allocator,
            "client_id={s}&client_secret={s}&refresh_token={s}",
            .{ self.config.client_id, self.config.client_secret, refresh_token },
        );
        defer self.allocator.free(body);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer headers.deinit();
        try headers.put("Content-Type", "application/x-www-form-urlencoded");
        
        var response = try self.http.post(url, headers, body);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 204 and response.status_code != 200) {
            std.log.err("Logout failed: {d}", .{response.status_code});
            return error.LogoutFailed;
        }
    }
    
    /// Get user information by ID (requires admin token)
    pub fn getUser(
        self: *KeycloakClient,
        admin_token: []const u8,
        user_id: []const u8,
    ) !UserInfo {
        const base_url = try self.config.getAdminApiBase(self.allocator);
        defer self.allocator.free(base_url);
        
        const url = try std.fmt.allocPrint(
            self.allocator,
            "{s}/users/{s}",
            .{ base_url, user_id },
        );
        defer self.allocator.free(url);
        
        var headers = std.StringHashMap([]const u8).init(self.allocator);
        defer {
            var iter = headers.iterator();
            while (iter.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                self.allocator.free(entry.value_ptr.*);
            }
            headers.deinit();
        }
        
        const auth_header = try std.fmt.allocPrint(
            self.allocator,
            "Bearer {s}",
            .{admin_token},
        );
        defer self.allocator.free(auth_header);
        
        const auth_key = try self.allocator.dupe(u8, "Authorization");
        const auth_value = try self.allocator.dupe(u8, auth_header);
        try headers.put(auth_key, auth_value);
        
        var response = try self.http.get(url, headers);
        defer response.deinit(self.allocator);
        
        if (response.status_code != 200) {
            return error.UserNotFound;
        }
        
        return try UserInfo.fromJson(self.allocator, response.body);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "KeycloakClient initialization" {
    const allocator = std.testing.allocator;
    
    const cfg = KeycloakConfig{
        .server_url = "http://localhost:8180",
        .realm = "test-realm",
        .client_id = "test-client",
        .client_secret = "test-secret",
    };
    
    var client = try KeycloakClient.init(allocator, cfg);
    defer client.deinit();
    
    try std.testing.expectEqualStrings("http://localhost:8180", client.config.server_url);
    try std.testing.expectEqualStrings("test-realm", client.config.realm);
}

test "KeycloakClient initialization with invalid config" {
    const allocator = std.testing.allocator;
    
    const cfg = KeycloakConfig{
        .server_url = "",
        .realm = "test-realm",
        .client_id = "test-client",
        .client_secret = "test-secret",
    };
    
    try std.testing.expectError(error.InvalidServerUrl, KeycloakClient.init(allocator, cfg));
}
