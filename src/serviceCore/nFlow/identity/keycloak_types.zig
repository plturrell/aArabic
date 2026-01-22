//! Keycloak Type Definitions - Day 34
//!
//! Defines data structures for Keycloak API responses and requests.
//! Includes proper memory management for all owned data.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// Token response from Keycloak
pub const TokenResponse = struct {
    access_token: []const u8,
    refresh_token: ?[]const u8,
    expires_in: u32,
    refresh_expires_in: u32,
    token_type: []const u8,
    session_state: ?[]const u8,
    scope: ?[]const u8,
    
    pub fn deinit(self: *TokenResponse, allocator: Allocator) void {
        allocator.free(self.access_token);
        if (self.refresh_token) |token| {
            allocator.free(token);
        }
        allocator.free(self.token_type);
        if (self.session_state) |state| {
            allocator.free(state);
        }
        if (self.scope) |s| {
            allocator.free(s);
        }
    }
    
    /// Parse from JSON
    pub fn fromJson(allocator: Allocator, json_str: []const u8) !TokenResponse {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_str,
            .{},
        );
        defer parsed.deinit();
        
        const obj = parsed.value.object;
        
        const access_token = try allocator.dupe(u8, obj.get("access_token").?.string);
        errdefer allocator.free(access_token);
        
        const token_type = try allocator.dupe(u8, obj.get("token_type").?.string);
        errdefer allocator.free(token_type);
        
        const refresh_token = if (obj.get("refresh_token")) |rt|
            try allocator.dupe(u8, rt.string)
        else
            null;
        errdefer if (refresh_token) |rt| allocator.free(rt);
        
        const session_state = if (obj.get("session_state")) |ss|
            try allocator.dupe(u8, ss.string)
        else
            null;
        errdefer if (session_state) |ss| allocator.free(ss);
        
        const scope = if (obj.get("scope")) |s|
            try allocator.dupe(u8, s.string)
        else
            null;
        errdefer if (scope) |s| allocator.free(s);
        
        return TokenResponse{
            .access_token = access_token,
            .refresh_token = refresh_token,
            .expires_in = @intCast(obj.get("expires_in").?.integer),
            .refresh_expires_in = if (obj.get("refresh_expires_in")) |exp|
                @intCast(exp.integer)
            else
                0,
            .token_type = token_type,
            .session_state = session_state,
            .scope = scope,
        };
    }
};

/// Token introspection info
pub const TokenInfo = struct {
    sub: []const u8,
    email: ?[]const u8,
    preferred_username: []const u8,
    realm_roles: [][]const u8,
    exp: i64,
    iat: i64,
    active: bool,
    
    pub fn deinit(self: *TokenInfo, allocator: Allocator) void {
        allocator.free(self.sub);
        if (self.email) |e| {
            allocator.free(e);
        }
        allocator.free(self.preferred_username);
        for (self.realm_roles) |role| {
            allocator.free(role);
        }
        allocator.free(self.realm_roles);
    }
    
    /// Parse from JSON
    pub fn fromJson(allocator: Allocator, json_str: []const u8) !TokenInfo {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_str,
            .{},
        );
        defer parsed.deinit();
        
        const obj = parsed.value.object;
        
        const sub = try allocator.dupe(u8, obj.get("sub").?.string);
        errdefer allocator.free(sub);
        
        const preferred_username = try allocator.dupe(u8, obj.get("preferred_username").?.string);
        errdefer allocator.free(preferred_username);
        
        const email = if (obj.get("email")) |e|
            try allocator.dupe(u8, e.string)
        else
            null;
        errdefer if (email) |e| allocator.free(e);
        
        // Parse realm roles
        var roles = try std.ArrayList([]const u8).initCapacity(allocator, 4);
        defer roles.deinit(allocator);
        errdefer {
            for (roles.items) |role| allocator.free(role);
        }
        
        if (obj.get("realm_access")) |realm_access| {
            if (realm_access.object.get("roles")) |roles_array| {
                for (roles_array.array.items) |role_value| {
                    const role = try allocator.dupe(u8, role_value.string);
                    try roles.append(allocator, role);
                }
            }
        }
        
        const realm_roles = try roles.toOwnedSlice(allocator);
        
        return TokenInfo{
            .sub = sub,
            .email = email,
            .preferred_username = preferred_username,
            .realm_roles = realm_roles,
            .exp = obj.get("exp").?.integer,
            .iat = obj.get("iat").?.integer,
            .active = if (obj.get("active")) |a| a.bool else true,
        };
    }
    
    /// Check if token is expired
    pub fn isExpired(self: *const TokenInfo) bool {
        const now = std.time.timestamp();
        return now >= self.exp;
    }
    
    /// Check if user has role
    pub fn hasRole(self: *const TokenInfo, role: []const u8) bool {
        for (self.realm_roles) |r| {
            if (std.mem.eql(u8, r, role)) {
                return true;
            }
        }
        return false;
    }
};

/// User information from Keycloak
pub const UserInfo = struct {
    id: []const u8,
    username: []const u8,
    email: ?[]const u8,
    first_name: ?[]const u8,
    last_name: ?[]const u8,
    enabled: bool,
    email_verified: bool,
    
    pub fn deinit(self: *UserInfo, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.username);
        if (self.email) |e| allocator.free(e);
        if (self.first_name) |f| allocator.free(f);
        if (self.last_name) |l| allocator.free(l);
    }
    
    /// Parse from JSON
    pub fn fromJson(allocator: Allocator, json_str: []const u8) !UserInfo {
        const parsed = try std.json.parseFromSlice(
            std.json.Value,
            allocator,
            json_str,
            .{},
        );
        defer parsed.deinit();
        
        const obj = parsed.value.object;
        
        const id = try allocator.dupe(u8, obj.get("id").?.string);
        errdefer allocator.free(id);
        
        const username = try allocator.dupe(u8, obj.get("username").?.string);
        errdefer allocator.free(username);
        
        const email = if (obj.get("email")) |e|
            try allocator.dupe(u8, e.string)
        else
            null;
        errdefer if (email) |e| allocator.free(e);
        
        const first_name = if (obj.get("firstName")) |f|
            try allocator.dupe(u8, f.string)
        else
            null;
        errdefer if (first_name) |f| allocator.free(f);
        
        const last_name = if (obj.get("lastName")) |l|
            try allocator.dupe(u8, l.string)
        else
            null;
        errdefer if (last_name) |l| allocator.free(l);
        
        return UserInfo{
            .id = id,
            .username = username,
            .email = email,
            .first_name = first_name,
            .last_name = last_name,
            .enabled = if (obj.get("enabled")) |e| e.bool else false,
            .email_verified = if (obj.get("emailVerified")) |ev| ev.bool else false,
        };
    }
};

/// Role information
pub const RoleInfo = struct {
    id: []const u8,
    name: []const u8,
    description: ?[]const u8,
    composite: bool,
    
    pub fn deinit(self: *RoleInfo, allocator: Allocator) void {
        allocator.free(self.id);
        allocator.free(self.name);
        if (self.description) |d| {
            allocator.free(d);
        }
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "TokenResponse parsing from JSON" {
    const allocator = std.testing.allocator;
    
    const json =
        \\{
        \\  "access_token": "test_access_token",
        \\  "refresh_token": "test_refresh_token",
        \\  "expires_in": 300,
        \\  "refresh_expires_in": 1800,
        \\  "token_type": "Bearer"
        \\}
    ;
    
    var token = try TokenResponse.fromJson(allocator, json);
    defer token.deinit(allocator);
    
    try std.testing.expectEqualStrings("test_access_token", token.access_token);
    try std.testing.expectEqualStrings("Bearer", token.token_type);
    try std.testing.expectEqual(@as(u32, 300), token.expires_in);
}

test "TokenInfo parsing from JSON" {
    const allocator = std.testing.allocator;
    
    const json =
        \\{
        \\  "sub": "user-123",
        \\  "preferred_username": "testuser",
        \\  "email": "test@example.com",
        \\  "realm_access": {
        \\    "roles": ["user", "admin"]
        \\  },
        \\  "exp": 1234567890,
        \\  "iat": 1234567800
        \\}
    ;
    
    var info = try TokenInfo.fromJson(allocator, json);
    defer info.deinit(allocator);
    
    try std.testing.expectEqualStrings("user-123", info.sub);
    try std.testing.expectEqualStrings("testuser", info.preferred_username);
    try std.testing.expectEqual(@as(usize, 2), info.realm_roles.len);
}

test "TokenInfo has role check" {
    const allocator = std.testing.allocator;
    
    const json =
        \\{
        \\  "sub": "user-123",
        \\  "preferred_username": "testuser",
        \\  "realm_access": {
        \\    "roles": ["user", "admin"]
        \\  },
        \\  "exp": 9999999999,
        \\  "iat": 1234567800
        \\}
    ;
    
    var info = try TokenInfo.fromJson(allocator, json);
    defer info.deinit(allocator);
    
    try std.testing.expect(info.hasRole("admin"));
    try std.testing.expect(info.hasRole("user"));
    try std.testing.expect(!info.hasRole("superuser"));
}

test "TokenInfo expiry check" {
    const allocator = std.testing.allocator;
    
    const json_expired =
        \\{
        \\  "sub": "user-123",
        \\  "preferred_username": "testuser",
        \\  "realm_access": {"roles": []},
        \\  "exp": 1000000000,
        \\  "iat": 999999900
        \\}
    ;
    
    var info = try TokenInfo.fromJson(allocator, json_expired);
    defer info.deinit(allocator);
    
    try std.testing.expect(info.isExpired());
}

test "UserInfo parsing from JSON" {
    const allocator = std.testing.allocator;
    
    const json =
        \\{
        \\  "id": "user-123",
        \\  "username": "testuser",
        \\  "email": "test@example.com",
        \\  "firstName": "Test",
        \\  "lastName": "User",
        \\  "enabled": true,
        \\  "emailVerified": true
        \\}
    ;
    
    var user = try UserInfo.fromJson(allocator, json);
    defer user.deinit(allocator);
    
    try std.testing.expectEqualStrings("user-123", user.id);
    try std.testing.expectEqualStrings("testuser", user.username);
    try std.testing.expect(user.enabled);
    try std.testing.expect(user.email_verified);
}
