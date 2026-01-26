//! Keycloak Authentication Middleware for nWorkflow HTTP Server
//!
//! Provides JWT token validation and authentication context for
//! protecting API endpoints with Keycloak-issued tokens.

const std = @import("std");
const mem = std.mem;
const Allocator = std.mem.Allocator;
const base64 = std.base64;

// ============================================================================
// Constants
// ============================================================================

/// Default Keycloak server URL
pub const DEFAULT_KEYCLOAK_URL = "http://localhost:8080";

/// Default realm for nWorkflow
pub const DEFAULT_REALM = "nworkflow";

/// Bearer token prefix
const BEARER_PREFIX = "Bearer ";

/// Allowed clock skew in seconds for token expiration checks
const CLOCK_SKEW_SECONDS: i64 = 30;

// ============================================================================
// Error Types
// ============================================================================

/// Authentication errors
pub const AuthError = error{
    /// No Authorization header present
    MissingAuthHeader,
    /// Authorization header format is invalid
    InvalidAuthHeader,
    /// Token format is invalid (not a valid JWT)
    InvalidTokenFormat,
    /// Base64 decoding failed
    Base64DecodeFailed,
    /// JSON parsing failed
    JsonParseFailed,
    /// Token has expired
    TokenExpired,
    /// Token issuer doesn't match expected realm
    InvalidIssuer,
    /// Token is missing required claims
    MissingClaims,
    /// Memory allocation failed
    OutOfMemory,
    /// Token signature validation failed (placeholder for future JWKS support)
    InvalidSignature,
};

// ============================================================================
// Configuration
// ============================================================================

/// Keycloak configuration for authentication middleware
pub const KeycloakConfig = struct {
    /// Keycloak realm name
    realm: []const u8 = DEFAULT_REALM,
    /// Keycloak server URL (e.g., "http://localhost:8080")
    auth_server_url: []const u8 = DEFAULT_KEYCLOAK_URL,
    /// OAuth2 client ID
    client_id: []const u8,
    /// OAuth2 client secret (optional for public clients)
    client_secret: ?[]const u8 = null,

    /// Get expected issuer URL for token validation
    pub fn getExpectedIssuer(self: *const KeycloakConfig, allocator: Allocator) ![]const u8 {
        return try std.fmt.allocPrint(
            allocator,
            "{s}/realms/{s}",
            .{ self.auth_server_url, self.realm },
        );
    }
};

// ============================================================================
// JWT Payload Structure
// ============================================================================

/// Decoded JWT payload from Keycloak token
pub const JwtPayload = struct {
    /// Subject (user ID)
    sub: []const u8,
    /// Preferred username
    preferred_username: ?[]const u8,
    /// Email address
    email: ?[]const u8,
    /// Token expiration timestamp (Unix epoch)
    exp: i64,
    /// Token issued at timestamp (Unix epoch)
    iat: i64,
    /// Token issuer (Keycloak realm URL)
    iss: ?[]const u8,
    /// Audience
    aud: ?[]const u8,
    /// Realm access roles
    realm_roles: [][]const u8,
    /// Tenant ID (custom claim)
    tenant_id: ?[]const u8,

    pub fn deinit(self: *JwtPayload, allocator: Allocator) void {
        allocator.free(self.sub);
        if (self.preferred_username) |u| allocator.free(u);
        if (self.email) |e| allocator.free(e);
        if (self.iss) |i| allocator.free(i);
        if (self.aud) |a| allocator.free(a);
        if (self.tenant_id) |t| allocator.free(t);
        for (self.realm_roles) |role| {
            allocator.free(role);
        }
        allocator.free(self.realm_roles);
    }
};

// ============================================================================
// Authentication Context
// ============================================================================

/// Authenticated user context extracted from validated JWT
pub const AuthContext = struct {
    /// Keycloak user ID (subject claim)
    user_id: []const u8,
    /// Username from token
    username: []const u8,
    /// User email (optional)
    email: ?[]const u8,
    /// Assigned roles from realm_access
    roles: [][]const u8,
    /// Tenant ID for multi-tenancy (custom claim)
    tenant_id: ?[]const u8,
    /// Token expiration timestamp
    token_exp: i64,

    /// Free all owned memory
    pub fn deinit(self: *AuthContext, allocator: Allocator) void {
        allocator.free(self.user_id);
        allocator.free(self.username);
        if (self.email) |e| allocator.free(e);
        if (self.tenant_id) |t| allocator.free(t);
        for (self.roles) |role| {
            allocator.free(role);
        }
        allocator.free(self.roles);
    }

    /// Check if user is expired
    pub fn isExpired(self: *const AuthContext) bool {
        const now = std.time.timestamp();
        return now >= self.token_exp;
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse Authorization header and extract Bearer token
/// Returns null if header is missing or malformed
pub fn parseAuthHeader(header: ?[]const u8) ?[]const u8 {
    const auth_header = header orelse return null;

    // Check for "Bearer " prefix (case-insensitive for "Bearer")
    if (auth_header.len < BEARER_PREFIX.len) {
        return null;
    }

    // Check prefix
    const prefix = auth_header[0..BEARER_PREFIX.len];
    if (!mem.eql(u8, prefix, BEARER_PREFIX) and
        !mem.eql(u8, prefix, "bearer "))
    {
        return null;
    }

    const token = auth_header[BEARER_PREFIX.len..];
    if (token.len == 0) {
        return null;
    }

    return token;
}

/// Decode a base64url-encoded string (JWT uses URL-safe base64 without padding)
fn decodeBase64Url(allocator: Allocator, encoded: []const u8) ![]const u8 {
    // Convert base64url to standard base64
    var buffer = try allocator.alloc(u8, encoded.len);
    defer allocator.free(buffer);

    for (encoded, 0..) |c, i| {
        buffer[i] = switch (c) {
            '-' => '+',
            '_' => '/',
            else => c,
        };
    }

    // Add padding if necessary
    const padding_needed = (4 - (buffer.len % 4)) % 4;
    var padded = try allocator.alloc(u8, buffer.len + padding_needed);
    defer allocator.free(padded);

    @memcpy(padded[0..buffer.len], buffer);
    for (buffer.len..padded.len) |i| {
        padded[i] = '=';
    }

    // Decode
    const decoder = base64.standard.Decoder;
    const decoded_len = decoder.calcSizeForSlice(padded) catch return AuthError.Base64DecodeFailed;
    const decoded = try allocator.alloc(u8, decoded_len);
    errdefer allocator.free(decoded);

    decoder.decode(decoded, padded) catch return AuthError.Base64DecodeFailed;

    return decoded;
}

/// Decode JWT token and extract payload
/// JWT format: header.payload.signature (base64url encoded)
pub fn decodeJwt(allocator: Allocator, token: []const u8) !JwtPayload {
    // Split token into parts
    var parts = mem.splitScalar(u8, token, '.');

    // Skip header (first part)
    _ = parts.next() orelse return AuthError.InvalidTokenFormat;

    // Get payload (second part)
    const payload_b64 = parts.next() orelse return AuthError.InvalidTokenFormat;

    // Verify signature part exists (third part)
    _ = parts.next() orelse return AuthError.InvalidTokenFormat;

    // Ensure no extra parts
    if (parts.next() != null) {
        return AuthError.InvalidTokenFormat;
    }

    // Decode payload
    const payload_json = try decodeBase64Url(allocator, payload_b64);
    defer allocator.free(payload_json);

    // Parse JSON
    const parsed = std.json.parseFromSlice(
        std.json.Value,
        allocator,
        payload_json,
        .{},
    ) catch return AuthError.JsonParseFailed;
    defer parsed.deinit();

    const obj = parsed.value.object;

    // Extract required claims
    const sub_val = obj.get("sub") orelse return AuthError.MissingClaims;
    const sub = try allocator.dupe(u8, sub_val.string);
    errdefer allocator.free(sub);

    const exp = if (obj.get("exp")) |e| e.integer else return AuthError.MissingClaims;
    const iat = if (obj.get("iat")) |i| i.integer else 0;

    // Extract optional claims
    const preferred_username = if (obj.get("preferred_username")) |u|
        try allocator.dupe(u8, u.string)
    else
        null;
    errdefer if (preferred_username) |u| allocator.free(u);

    const email = if (obj.get("email")) |e|
        try allocator.dupe(u8, e.string)
    else
        null;
    errdefer if (email) |e| allocator.free(e);

    const iss = if (obj.get("iss")) |i|
        try allocator.dupe(u8, i.string)
    else
        null;
    errdefer if (iss) |i| allocator.free(i);

    const aud = if (obj.get("aud")) |a| blk: {
        // aud can be string or array
        if (a == .string) {
            break :blk try allocator.dupe(u8, a.string);
        } else if (a == .array and a.array.items.len > 0) {
            break :blk try allocator.dupe(u8, a.array.items[0].string);
        }
        break :blk null;
    } else null;
    errdefer if (aud) |a| allocator.free(a);

    // Extract tenant_id custom claim
    const tenant_id = if (obj.get("tenant_id")) |t|
        try allocator.dupe(u8, t.string)
    else
        null;
    errdefer if (tenant_id) |t| allocator.free(t);

    // Extract realm roles
    var roles = std.ArrayList([]const u8){};
    errdefer {
        for (roles.items) |role| allocator.free(role);
        roles.deinit();
    }

    if (obj.get("realm_access")) |realm_access| {
        if (realm_access.object.get("roles")) |roles_array| {
            for (roles_array.array.items) |role_value| {
                const role = try allocator.dupe(u8, role_value.string);
                try roles.append(role);
            }
        }
    }

    const realm_roles = try roles.toOwnedSlice();

    return JwtPayload{
        .sub = sub,
        .preferred_username = preferred_username,
        .email = email,
        .exp = exp,
        .iat = iat,
        .iss = iss,
        .aud = aud,
        .realm_roles = realm_roles,
        .tenant_id = tenant_id,
    };
}


/// Validate JWT token against Keycloak configuration and return AuthContext
/// Note: This performs offline validation only (no signature verification via JWKS).
/// For production, consider adding JWKS-based signature verification.
pub fn validateToken(
    allocator: Allocator,
    token: []const u8,
    config: KeycloakConfig,
) !AuthContext {
    // Decode JWT payload
    var payload = try decodeJwt(allocator, token);
    defer payload.deinit(allocator);

    // Check token expiration (with clock skew allowance)
    const now = std.time.timestamp();
    if (now > payload.exp + CLOCK_SKEW_SECONDS) {
        return AuthError.TokenExpired;
    }

    // Verify issuer matches expected Keycloak realm
    if (payload.iss) |iss| {
        const expected_issuer = try config.getExpectedIssuer(allocator);
        defer allocator.free(expected_issuer);

        if (!mem.eql(u8, iss, expected_issuer)) {
            return AuthError.InvalidIssuer;
        }
    }

    // Build AuthContext from validated payload
    const user_id = try allocator.dupe(u8, payload.sub);
    errdefer allocator.free(user_id);

    const username = if (payload.preferred_username) |u|
        try allocator.dupe(u8, u)
    else
        try allocator.dupe(u8, payload.sub);
    errdefer allocator.free(username);

    const email = if (payload.email) |e|
        try allocator.dupe(u8, e)
    else
        null;
    errdefer if (email) |e| allocator.free(e);

    const tenant_id = if (payload.tenant_id) |t|
        try allocator.dupe(u8, t)
    else
        null;
    errdefer if (tenant_id) |t| allocator.free(t);

    // Copy roles
    var roles = try allocator.alloc([]const u8, payload.realm_roles.len);
    errdefer allocator.free(roles);
    var copied_count: usize = 0;
    errdefer {
        for (roles[0..copied_count]) |role| allocator.free(role);
    }

    for (payload.realm_roles, 0..) |role, i| {
        roles[i] = try allocator.dupe(u8, role);
        copied_count += 1;
    }

    return AuthContext{
        .user_id = user_id,
        .username = username,
        .email = email,
        .roles = roles,
        .tenant_id = tenant_id,
        .token_exp = payload.exp,
    };
}

/// Check if authenticated user has a specific role
pub fn hasRole(ctx: *const AuthContext, role: []const u8) bool {
    for (ctx.roles) |r| {
        if (mem.eql(u8, r, role)) {
            return true;
        }
    }
    return false;
}

/// Check if authenticated user belongs to a specific tenant
pub fn hasTenant(ctx: *const AuthContext, tenant: []const u8) bool {
    if (ctx.tenant_id) |t| {
        return mem.eql(u8, t, tenant);
    }
    return false;
}

/// Check if user has any of the specified roles
pub fn hasAnyRole(ctx: *const AuthContext, required_roles: []const []const u8) bool {
    for (required_roles) |required| {
        if (hasRole(ctx, required)) {
            return true;
        }
    }
    return false;
}

/// Check if user has all of the specified roles
pub fn hasAllRoles(ctx: *const AuthContext, required_roles: []const []const u8) bool {
    for (required_roles) |required| {
        if (!hasRole(ctx, required)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Middleware Integration Helper
// ============================================================================

/// Extract auth header from raw HTTP request
pub fn extractAuthHeaderFromRequest(request: []const u8) ?[]const u8 {
    var lines = mem.splitSequence(u8, request, "\r\n");
    while (lines.next()) |line| {
        // Skip empty lines
        if (line.len == 0) continue;

        // Look for Authorization header (case-insensitive match)
        if (line.len > 14) {
            const header_name = line[0..14];
            if (mem.eql(u8, header_name, "Authorization:") or
                mem.eql(u8, header_name, "authorization:"))
            {
                // Skip "Authorization: " and trim whitespace
                var value = line[14..];
                while (value.len > 0 and value[0] == ' ') {
                    value = value[1..];
                }
                return value;
            }
        }
    }
    return null;
}

/// Authenticate request and return AuthContext
/// Combines extractAuthHeaderFromRequest, parseAuthHeader, and validateToken
pub fn authenticateRequest(
    allocator: Allocator,
    request: []const u8,
    config: KeycloakConfig,
) !AuthContext {
    const auth_header = extractAuthHeaderFromRequest(request) orelse
        return AuthError.MissingAuthHeader;

    const token = parseAuthHeader(auth_header) orelse
        return AuthError.InvalidAuthHeader;

    return validateToken(allocator, token, config);
}

// ============================================================================
// Tests
// ============================================================================

test "parseAuthHeader - valid bearer token" {
    const header = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.sig";
    const token = parseAuthHeader(header);
    try std.testing.expect(token != null);
    try std.testing.expectEqualStrings("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.sig", token.?);
}

test "parseAuthHeader - lowercase bearer" {
    const header = "bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test.sig";
    const token = parseAuthHeader(header);
    try std.testing.expect(token != null);
}

test "parseAuthHeader - null header" {
    const token = parseAuthHeader(null);
    try std.testing.expect(token == null);
}

test "parseAuthHeader - empty token" {
    const header = "Bearer ";
    const token = parseAuthHeader(header);
    try std.testing.expect(token == null);
}

test "parseAuthHeader - invalid prefix" {
    const header = "Basic dXNlcjpwYXNz";
    const token = parseAuthHeader(header);
    try std.testing.expect(token == null);
}

test "KeycloakConfig - get expected issuer" {
    const allocator = std.testing.allocator;

    const config = KeycloakConfig{
        .realm = "nworkflow",
        .auth_server_url = "http://localhost:8080",
        .client_id = "nworkflow-client",
    };

    const issuer = try config.getExpectedIssuer(allocator);
    defer allocator.free(issuer);

    try std.testing.expectEqualStrings("http://localhost:8080/realms/nworkflow", issuer);
}

test "hasRole - role exists" {
    const allocator = std.testing.allocator;

    var roles = [_][]const u8{ "admin", "user", "viewer" };
    var ctx = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 9999999999,
    };

    try std.testing.expect(hasRole(&ctx, "admin"));
    try std.testing.expect(hasRole(&ctx, "user"));
    try std.testing.expect(!hasRole(&ctx, "superadmin"));

    // No deinit needed as we used stack-allocated test data
    _ = allocator;
}

test "hasTenant - tenant matches" {
    const allocator = std.testing.allocator;

    var roles = [_][]const u8{};
    var ctx = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = "tenant-123",
        .token_exp = 9999999999,
    };

    try std.testing.expect(hasTenant(&ctx, "tenant-123"));
    try std.testing.expect(!hasTenant(&ctx, "tenant-456"));

    _ = allocator;
}

test "hasTenant - no tenant" {
    const allocator = std.testing.allocator;

    var roles = [_][]const u8{};
    var ctx = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 9999999999,
    };

    try std.testing.expect(!hasTenant(&ctx, "any-tenant"));

    _ = allocator;
}

test "hasAnyRole - at least one role matches" {
    var roles = [_][]const u8{ "admin", "user" };
    var ctx = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 9999999999,
    };

    const required = [_][]const u8{ "admin", "superuser" };
    try std.testing.expect(hasAnyRole(&ctx, &required));

    const none_match = [_][]const u8{ "superuser", "owner" };
    try std.testing.expect(!hasAnyRole(&ctx, &none_match));
}

test "hasAllRoles - all roles required" {
    var roles = [_][]const u8{ "admin", "user", "viewer" };
    var ctx = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 9999999999,
    };

    const all_present = [_][]const u8{ "admin", "user" };
    try std.testing.expect(hasAllRoles(&ctx, &all_present));

    const some_missing = [_][]const u8{ "admin", "superuser" };
    try std.testing.expect(!hasAllRoles(&ctx, &some_missing));
}

test "extractAuthHeaderFromRequest - finds header" {
    const request = "GET /api/v1/workflows HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer test-token\r\nContent-Type: application/json\r\n\r\n";
    const auth = extractAuthHeaderFromRequest(request);
    try std.testing.expect(auth != null);
    try std.testing.expectEqualStrings("Bearer test-token", auth.?);
}

test "extractAuthHeaderFromRequest - no auth header" {
    const request = "GET /api/v1/workflows HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n\r\n";
    const auth = extractAuthHeaderFromRequest(request);
    try std.testing.expect(auth == null);
}

test "AuthContext - isExpired check" {
    var roles = [_][]const u8{};

    // Not expired (far future)
    var ctx_valid = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 9999999999,
    };
    try std.testing.expect(!ctx_valid.isExpired());

    // Expired (past timestamp)
    var ctx_expired = AuthContext{
        .user_id = "test-user",
        .username = "testuser",
        .email = null,
        .roles = &roles,
        .tenant_id = null,
        .token_exp = 1000000000,
    };
    try std.testing.expect(ctx_expired.isExpired());
}

