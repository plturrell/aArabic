//! Auth Handlers - Day 32
//!
//! Authentication endpoint handlers for nMetaData API.
//! Provides login, logout, token refresh, and user management.

const std = @import("std");
const Request = @import("../http/types.zig").Request;
const Response = @import("../http/types.zig").Response;
const Jwt = @import("../auth/jwt.zig").Jwt;
const JwtConfig = @import("../auth/jwt.zig").JwtConfig;

/// Hardcoded users for demo (in production, use database)
const User = struct {
    id: []const u8,
    username: []const u8,
    password_hash: []const u8, // In production: bcrypt hash
    roles: []const []const u8,
};

const DEMO_USERS = [_]User{
    .{
        .id = "user-001",
        .username = "admin",
        .password_hash = "admin123", // In production: hash this!
        .roles = &[_][]const u8{ "admin", "user" },
    },
    .{
        .id = "user-002",
        .username = "user",
        .password_hash = "user123",
        .roles = &[_][]const u8{"user"},
    },
};

/// JWT secret (in production: load from environment/config)
const JWT_SECRET = "nmetadata-secret-key-change-in-production";

/// Login handler
pub fn loginHandler(req: *Request, resp: *Response) !void {
    // Parse login request
    const LoginRequest = struct {
        username: []const u8,
        password: []const u8,
    };
    
    const body = req.jsonBody(LoginRequest) catch {
        try resp.error_(400, "Invalid login request");
        return;
    };
    
    // Find user
    var found_user: ?User = null;
    for (DEMO_USERS) |user| {
        if (std.mem.eql(u8, user.username, body.username)) {
            found_user = user;
            break;
        }
    }
    
    const user = found_user orelse {
        try resp.error_(401, "Invalid username or password");
        return;
    };
    
    // Verify password (in production: use bcrypt)
    if (!std.mem.eql(u8, user.password_hash, body.password)) {
        try resp.error_(401, "Invalid username or password");
        return;
    }
    
    // Generate JWT token
    const jwt_config = JwtConfig{
        .secret = JWT_SECRET,
        .expiration = 3600, // 1 hour
    };
    
    var jwt = Jwt.init(req.allocator, jwt_config);
    const token = try jwt.generate(user.id, user.roles);
    
    // Generate refresh token (longer expiration)
    const refresh_config = JwtConfig{
        .secret = JWT_SECRET,
        .expiration = 604800, // 7 days
    };
    
    var refresh_jwt = Jwt.init(req.allocator, refresh_config);
    const refresh_token = try refresh_jwt.generate(user.id, user.roles);
    
    // Return tokens
    const LoginResponse = struct {
        token: []const u8,
        refresh_token: []const u8,
        expires_in: i64,
        user: struct {
            id: []const u8,
            username: []const u8,
            roles: []const []const u8,
        },
    };
    
    resp.status = 200;
    try resp.json(LoginResponse{
        .token = token,
        .refresh_token = refresh_token,
        .expires_in = 3600,
        .user = .{
            .id = user.id,
            .username = user.username,
            .roles = user.roles,
        },
    });
}

/// Logout handler
pub fn logoutHandler(req: *Request, resp: *Response) !void {
    _ = req;
    
    // In production: invalidate token (add to blacklist, etc.)
    
    const LogoutResponse = struct {
        message: []const u8,
    };
    
    resp.status = 200;
    try resp.json(LogoutResponse{
        .message = "Logged out successfully",
    });
}

/// Refresh token handler
pub fn refreshTokenHandler(req: *Request, resp: *Response) !void {
    // Parse refresh request
    const RefreshRequest = struct {
        refresh_token: []const u8,
    };
    
    const body = req.jsonBody(RefreshRequest) catch {
        try resp.error_(400, "Invalid refresh request");
        return;
    };
    
    // Validate refresh token
    const jwt_config = JwtConfig{
        .secret = JWT_SECRET,
    };
    
    var jwt = Jwt.init(req.allocator, jwt_config);
    const claims = jwt.validate(body.refresh_token) catch {
        try resp.error_(401, "Invalid or expired refresh token");
        return;
    };
    
    // Generate new access token
    const new_token = try jwt.generate(claims.sub, claims.roles);
    
    const RefreshResponse = struct {
        token: []const u8,
        expires_in: i64,
    };
    
    resp.status = 200;
    try resp.json(RefreshResponse{
        .token = new_token,
        .expires_in = 3600,
    });
}

/// Get current user handler (requires authentication)
pub fn getCurrentUserHandler(req: *Request, resp: *Response) !void {
    // Get user ID from auth middleware
    const user_id = req.param("auth_user_id") orelse {
        try resp.error_(401, "Not authenticated");
        return;
    };
    
    // Find user
    var found_user: ?User = null;
    for (DEMO_USERS) |user| {
        if (std.mem.eql(u8, user.id, user_id)) {
            found_user = user;
            break;
        }
    }
    
    const user = found_user orelse {
        try resp.error_(404, "User not found");
        return;
    };
    
    const UserResponse = struct {
        id: []const u8,
        username: []const u8,
        roles: []const []const u8,
    };
    
    resp.status = 200;
    try resp.json(UserResponse{
        .id = user.id,
        .username = user.username,
        .roles = user.roles,
    });
}

/// Verify token handler
pub fn verifyTokenHandler(req: *Request, resp: *Response) !void {
    // Get token from Authorization header
    const auth_header = req.header("Authorization") orelse {
        try resp.error_(401, "Missing Authorization header");
        return;
    };
    
    if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
        try resp.error_(401, "Invalid Authorization header");
        return;
    };
    
    const token = auth_header[7..];
    
    // Validate token
    const jwt_config = JwtConfig{
        .secret = JWT_SECRET,
    };
    
    var jwt = Jwt.init(req.allocator, jwt_config);
    const claims = jwt.validate(token) catch |err| {
        const msg = switch (err) {
            error.TokenExpired => "Token expired",
            error.InvalidSignature => "Invalid token signature",
            error.InvalidToken => "Invalid token format",
            else => "Token validation failed",
        };
        try resp.error_(401, msg);
        return;
    };
    
    const VerifyResponse = struct {
        valid: bool,
        user_id: []const u8,
        roles: []const []const u8,
        expires_at: i64,
    };
    
    resp.status = 200;
    try resp.json(VerifyResponse{
        .valid = true,
        .user_id = claims.sub,
        .roles = claims.roles,
        .expires_at = claims.exp,
    });
}
