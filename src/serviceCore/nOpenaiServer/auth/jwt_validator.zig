//! JWT Token Validation Module
//! Validates JWT tokens from SAP IAS or Keycloak
//! Extracts user context for authorization

const std = @import("std");
const mem = std.mem;
const json = std.json;

// ============================================================================
// Data Structures
// ============================================================================

pub const UserContext = struct {
    user_id: []const u8,
    email: []const u8,
    name: []const u8,
    roles: []const []const u8,
    allocator: mem.Allocator,

    pub fn deinit(self: *UserContext) void {
        self.allocator.free(self.user_id);
        self.allocator.free(self.email);
        self.allocator.free(self.name);
        for (self.roles) |role| {
            self.allocator.free(role);
        }
        self.allocator.free(self.roles);
    }
};

pub const JWTClaims = struct {
    sub: []const u8,          // Subject (user_id)
    email: ?[]const u8,       // Email address
    name: ?[]const u8,        // Full name
    exp: i64,                 // Expiration timestamp
    iat: i64,                 // Issued at timestamp
    roles: ?[]const []const u8, // User roles
};

// ============================================================================
// JWT Validation
// ============================================================================

/// Validate JWT token and extract user context
/// For now, this is a simplified version that decodes the payload
/// In production, you would verify the signature with a public key
pub fn validateToken(
    allocator: mem.Allocator,
    token: []const u8,
) !UserContext {
    // Split token into parts (header.payload.signature)
    var parts = mem.splitSequence(u8, token, ".");
    
    const header = parts.next() orelse return error.InvalidToken;
    const payload = parts.next() orelse return error.InvalidToken;
    const signature = parts.next() orelse return error.InvalidToken;
    
    _ = header;
    _ = signature;
    
    // Decode base64url payload
    const decoded_payload = try decodeBase64Url(allocator, payload);
    defer allocator.free(decoded_payload);
    
    // Parse JSON payload
    const parsed = try json.parseFromSlice(
        json.Value,
        allocator,
        decoded_payload,
        .{},
    );
    defer parsed.deinit();
    
    const claims = parsed.value.object;
    
    // Extract required claims
    const sub = claims.get("sub") orelse return error.MissingSubject;
    const user_id = try allocator.dupe(u8, sub.string);
    
    // Extract optional claims
    const email_value = claims.get("email");
    const email = if (email_value) |e|
        try allocator.dupe(u8, e.string)
    else
        try allocator.dupe(u8, "unknown@example.com");
    
    const name_value = claims.get("name");
    const name = if (name_value) |n|
        try allocator.dupe(u8, n.string)
    else
        try allocator.dupe(u8, "Unknown User");
    
    // Extract roles (if present)
    const roles_value = claims.get("roles");
    var roles: std.ArrayList([]const u8) = .{};
    defer roles.deinit(allocator);
    
    if (roles_value) |r| {
        if (r == .array) {
            for (r.array.items) |role_item| {
                if (role_item == .string) {
                    try roles.append(allocator, try allocator.dupe(u8, role_item.string));
                }
            }
        }
    }
    
    // Check expiration
    const exp_value = claims.get("exp");
    if (exp_value) |exp| {
        const exp_timestamp = exp.integer;
        const now = std.time.timestamp();
        
        if (now > exp_timestamp) {
            // Token expired, cleanup and return error
            allocator.free(user_id);
            allocator.free(email);
            allocator.free(name);
            for (roles.items) |role| allocator.free(role);
            return error.TokenExpired;
        }
    }
    
    return UserContext{
        .user_id = user_id,
        .email = email,
        .name = name,
        .roles = try roles.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

/// Decode base64url encoded string
fn decodeBase64Url(allocator: mem.Allocator, input: []const u8) ![]u8 {
    // Convert base64url to standard base64
    var base64 = try allocator.alloc(u8, input.len);
    defer allocator.free(base64);
    
    for (input, 0..) |c, i| {
        base64[i] = switch (c) {
            '-' => '+',
            '_' => '/',
            else => c,
        };
    }
    
    // Add padding if needed
    const padding_needed = (4 - (base64.len % 4)) % 4;
    var padded = try allocator.alloc(u8, base64.len + padding_needed);
    defer allocator.free(padded);
    
    @memcpy(padded[0..base64.len], base64);
    for (0..padding_needed) |i| {
        padded[base64.len + i] = '=';
    }
    
    // Decode base64
    const decoder = std.base64.standard.Decoder;
    const decoded_size = try decoder.calcSizeForSlice(padded);
    const decoded = try allocator.alloc(u8, decoded_size);
    
    try decoder.decode(decoded, padded);
    
    return decoded;
}

/// Extract user_id from Authorization header
pub fn extractUserId(
    allocator: mem.Allocator,
    auth_header: []const u8,
) ![]const u8 {
    // Check if header starts with "Bearer "
    if (!mem.startsWith(u8, auth_header, "Bearer ")) {
        return error.InvalidAuthHeader;
    }
    
    const token = auth_header[7..]; // Skip "Bearer "
    
    // Validate token and extract user context
    var user_ctx = try validateToken(allocator, token);
    defer user_ctx.deinit();
    
    // Return a copy of user_id that caller owns
    return try allocator.dupe(u8, user_ctx.user_id);
}

/// Check if user has required role
pub fn hasRole(user_ctx: *const UserContext, required_role: []const u8) bool {
    for (user_ctx.roles) |role| {
        if (mem.eql(u8, role, required_role)) {
            return true;
        }
    }
    return false;
}

// ============================================================================
// Mock Token Generation (for testing)
// ============================================================================

/// Generate a mock JWT token for testing
/// DO NOT use in production - this is only for development/testing
pub fn generateMockToken(
    allocator: mem.Allocator,
    user_id: []const u8,
    email: []const u8,
    name: []const u8,
) ![]const u8 {
    const header = "{\"alg\":\"none\",\"typ\":\"JWT\"}";
    
    // Create payload with 1 hour expiration
    const exp = std.time.timestamp() + 3600;
    const payload = try std.fmt.allocPrint(
        allocator,
        "{{\"sub\":\"{s}\",\"email\":\"{s}\",\"name\":\"{s}\",\"exp\":{d},\"iat\":{d}}}",
        .{ user_id, email, name, exp, std.time.timestamp() },
    );
    defer allocator.free(payload);
    
    // Encode parts
    const header_encoded = try encodeBase64Url(allocator, header);
    defer allocator.free(header_encoded);
    
    const payload_encoded = try encodeBase64Url(allocator, payload);
    defer allocator.free(payload_encoded);
    
    // Create token (no signature for mock)
    return try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.mock_signature",
        .{ header_encoded, payload_encoded },
    );
}

/// Encode string to base64url
fn encodeBase64Url(allocator: mem.Allocator, input: []const u8) ![]const u8 {
    const encoder = std.base64.standard.Encoder;
    const encoded_size = encoder.calcSize(input.len);
    const encoded = try allocator.alloc(u8, encoded_size);
    
    const result = encoder.encode(encoded, input);
    
    // Convert to base64url (replace + with -, / with _, remove =)
    var url_encoded: std.ArrayList(u8) = .{};
    defer url_encoded.deinit(allocator);
    
    for (result) |c| {
        switch (c) {
            '+' => try url_encoded.append(allocator, '-'),
            '/' => try url_encoded.append(allocator, '_'),
            '=' => {}, // Skip padding
            else => try url_encoded.append(allocator, c),
        }
    }
    
    allocator.free(encoded);
    return try url_encoded.toOwnedSlice(allocator);
}

// ============================================================================
// Tests
// ============================================================================

test "generate and validate mock token" {
    const allocator = std.testing.allocator;
    
    const token = try generateMockToken(
        allocator,
        "user123",
        "test@example.com",
        "Test User",
    );
    defer allocator.free(token);
    
    var user_ctx = try validateToken(allocator, token);
    defer user_ctx.deinit();
    
    try std.testing.expectEqualStrings("user123", user_ctx.user_id);
    try std.testing.expectEqualStrings("test@example.com", user_ctx.email);
    try std.testing.expectEqualStrings("Test User", user_ctx.name);
}

test "extract user_id from auth header" {
    const allocator = std.testing.allocator;
    
    const token = try generateMockToken(
        allocator,
        "user456",
        "another@example.com",
        "Another User",
    );
    defer allocator.free(token);
    
    const auth_header = try std.fmt.allocPrint(allocator, "Bearer {s}", .{token});
    defer allocator.free(auth_header);
    
    const user_id = try extractUserId(allocator, auth_header);
    defer allocator.free(user_id);
    
    try std.testing.expectEqualStrings("user456", user_id);
}

test "reject expired token" {
    const allocator = std.testing.allocator;
    
    // Create token with past expiration
    const header = "{\"alg\":\"none\",\"typ\":\"JWT\"}";
    const payload = "{\"sub\":\"user789\",\"exp\":1000000000}"; // Expired
    
    const header_encoded = try encodeBase64Url(allocator, header);
    defer allocator.free(header_encoded);
    
    const payload_encoded = try encodeBase64Url(allocator, payload);
    defer allocator.free(payload_encoded);
    
    const token = try std.fmt.allocPrint(
        allocator,
        "{s}.{s}.signature",
        .{ header_encoded, payload_encoded },
    );
    defer allocator.free(token);
    
    const result = validateToken(allocator, token);
    try std.testing.expectError(error.TokenExpired, result);
}
