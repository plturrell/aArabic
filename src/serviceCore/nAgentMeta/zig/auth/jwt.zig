//! JWT (JSON Web Token) - Day 32
//!
//! JWT implementation for nMetaData authentication.
//! Provides token generation, validation, and claims management.
//!
//! Features:
//! - HS256 signing (HMAC SHA-256)
//! - Token generation
//! - Token validation
//! - Claims extraction
//! - Expiration checking
//!
//! Token Structure:
//! ```
//! Header.Payload.Signature
//! eyJhbGc...  .  eyJzdWI...  .  SflKxwRJ...
//! ```

const std = @import("std");
const base64 = std.base64;
const crypto = std.crypto;
const json = std.json;
const Allocator = std.mem.Allocator;

/// JWT Claims structure
pub const Claims = struct {
    /// Subject (user ID)
    sub: []const u8,
    
    /// Issued at (Unix timestamp)
    iat: i64,
    
    /// Expiration (Unix timestamp)
    exp: i64,
    
    /// Issuer
    iss: []const u8 = "nMetaData",
    
    /// Roles/permissions
    roles: [][]const u8 = &[_][]const u8{},
};

/// JWT configuration
pub const JwtConfig = struct {
    /// Secret key for signing
    secret: []const u8,
    
    /// Token expiration time (seconds)
    expiration: i64 = 3600, // 1 hour default
    
    /// Issuer name
    issuer: []const u8 = "nMetaData",
};

/// JWT Manager
pub const Jwt = struct {
    allocator: Allocator,
    config: JwtConfig,
    
    pub fn init(allocator: Allocator, config: JwtConfig) Jwt {
        return Jwt{
            .allocator = allocator,
            .config = config,
        };
    }
    
    /// Generate JWT token
    pub fn generate(self: *Jwt, user_id: []const u8, roles: [][]const u8) ![]const u8 {
        const now = std.time.timestamp();
        
        // Create claims
        const claims = Claims{
            .sub = user_id,
            .iat = now,
            .exp = now + self.config.expiration,
            .iss = self.config.issuer,
            .roles = roles,
        };
        
        // Create header
        const header = .{
            .alg = "HS256",
            .typ = "JWT",
        };
        
        // Encode header
        var header_buf = std.ArrayList(u8){};
        defer header_buf.deinit();
        try json.stringify(header, .{}, header_buf.writer());
        const header_b64 = try self.base64UrlEncode(header_buf.items);
        defer self.allocator.free(header_b64);
        
        // Encode payload
        var payload_buf = std.ArrayList(u8){};
        defer payload_buf.deinit();
        try json.stringify(claims, .{}, payload_buf.writer());
        const payload_b64 = try self.base64UrlEncode(payload_buf.items);
        defer self.allocator.free(payload_b64);
        
        // Create signature
        const message = try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}",
            .{ header_b64, payload_b64 },
        );
        defer self.allocator.free(message);
        
        const signature = try self.sign(message);
        defer self.allocator.free(signature);
        
        // Combine into JWT
        return try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}.{s}",
            .{ header_b64, payload_b64, signature },
        );
    }
    
    /// Validate JWT token
    pub fn validate(self: *Jwt, token: []const u8) !Claims {
        // Split token
        var parts = std.mem.splitScalar(u8, token, '.');
        const header_b64 = parts.next() orelse return error.InvalidToken;
        const payload_b64 = parts.next() orelse return error.InvalidToken;
        const signature_b64 = parts.next() orelse return error.InvalidToken;
        
        // Verify signature
        const message = try std.fmt.allocPrint(
            self.allocator,
            "{s}.{s}",
            .{ header_b64, payload_b64 },
        );
        defer self.allocator.free(message);
        
        const expected_signature = try self.sign(message);
        defer self.allocator.free(expected_signature);
        
        if (!std.mem.eql(u8, signature_b64, expected_signature)) {
            return error.InvalidSignature;
        }
        
        // Decode payload
        const payload = try self.base64UrlDecode(payload_b64);
        defer self.allocator.free(payload);
        
        // Parse claims
        const parsed = try json.parseFromSlice(
            Claims,
            self.allocator,
            payload,
            .{},
        );
        defer parsed.deinit();
        
        const claims = parsed.value;
        
        // Check expiration
        const now = std.time.timestamp();
        if (claims.exp < now) {
            return error.TokenExpired;
        }
        
        return claims;
    }
    
    /// Sign message with HMAC-SHA256
    fn sign(self: *Jwt, message: []const u8) ![]const u8 {
        var hmac: [32]u8 = undefined;
        crypto.auth.hmac.sha2.HmacSha256.create(&hmac, message, self.config.secret);
        return try self.base64UrlEncode(&hmac);
    }
    
    /// Base64 URL encode
    fn base64UrlEncode(self: *Jwt, data: []const u8) ![]const u8 {
        const encoder = base64.url_safe_no_pad.Encoder;
        const encoded_len = encoder.calcSize(data.len);
        const encoded = try self.allocator.alloc(u8, encoded_len);
        _ = encoder.encode(encoded, data);
        return encoded;
    }
    
    /// Base64 URL decode
    fn base64UrlDecode(self: *Jwt, data: []const u8) ![]const u8 {
        const decoder = base64.url_safe_no_pad.Decoder;
        const decoded_len = try decoder.calcSizeForSlice(data);
        const decoded = try self.allocator.alloc(u8, decoded_len);
        try decoder.decode(decoded, data);
        return decoded;
    }
    
    /// Extract user ID from token
    pub fn extractUserId(self: *Jwt, token: []const u8) ![]const u8 {
        const claims = try self.validate(token);
        return try self.allocator.dupe(u8, claims.sub);
    }
    
    /// Check if user has role
    pub fn hasRole(self: *Jwt, token: []const u8, role: []const u8) !bool {
        const claims = try self.validate(token);
        for (claims.roles) |user_role| {
            if (std.mem.eql(u8, user_role, role)) {
                return true;
            }
        }
        return false;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "JWT: generate and validate" {
    const allocator = std.testing.allocator;
    
    const config = JwtConfig{
        .secret = "test-secret-key",
        .expiration = 3600,
    };
    
    var jwt = Jwt.init(allocator, config);
    
    const roles = [_][]const u8{ "admin", "user" };
    const token = try jwt.generate("user123", &roles);
    defer allocator.free(token);
    
    try std.testing.expect(token.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, token, ".") != null);
}

test "JWT: validate token" {
    const allocator = std.testing.allocator;
    
    const config = JwtConfig{
        .secret = "test-secret-key",
        .expiration = 3600,
    };
    
    var jwt = Jwt.init(allocator, config);
    
    const roles = [_][]const u8{"user"};
    const token = try jwt.generate("user123", &roles);
    defer allocator.free(token);
    
    const claims = try jwt.validate(token);
    try std.testing.expectEqualStrings("user123", claims.sub);
}

test "JWT: extract user ID" {
    const allocator = std.testing.allocator;
    
    const config = JwtConfig{
        .secret = "test-secret-key",
    };
    
    var jwt = Jwt.init(allocator, config);
    
    const roles = [_][]const u8{};
    const token = try jwt.generate("user456", &roles);
    defer allocator.free(token);
    
    const user_id = try jwt.extractUserId(token);
    defer allocator.free(user_id);
    
    try std.testing.expectEqualStrings("user456", user_id);
}

test "JWT: check role" {
    const allocator = std.testing.allocator;
    
    const config = JwtConfig{
        .secret = "test-secret-key",
    };
    
    var jwt = Jwt.init(allocator, config);
    
    const roles = [_][]const u8{ "admin", "user" };
    const token = try jwt.generate("user789", &roles);
    defer allocator.free(token);
    
    try std.testing.expect(try jwt.hasRole(token, "admin"));
    try std.testing.expect(try jwt.hasRole(token, "user"));
    try std.testing.expect(!try jwt.hasRole(token, "superadmin"));
}
