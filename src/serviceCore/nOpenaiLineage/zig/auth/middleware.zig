//! Auth Middleware - Day 32
//!
//! Authentication and authorization middleware for nMetaData API.
//! Protects endpoints with JWT validation and role-based access control.

const std = @import("std");
const Request = @import("../http/types.zig").Request;
const Response = @import("../http/types.zig").Response;
const Middleware = @import("../http/middleware.zig").Middleware;
const Jwt = @import("jwt.zig").Jwt;
const JwtConfig = @import("jwt.zig").JwtConfig;

/// Auth middleware configuration
pub const AuthConfig = struct {
    jwt_secret: []const u8,
    jwt_expiration: i64 = 3600,
    required_role: ?[]const u8 = null,
};

/// Create JWT authentication middleware
pub fn jwtAuthMiddleware(config: AuthConfig) Middleware {
    return Middleware{
        .name = "jwt_auth",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Get Authorization header
                const auth_header = req.header("Authorization") orelse {
                    try resp.error_(401, "Missing Authorization header");
                    return false;
                };
                
                // Extract token (Bearer <token>)
                if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
                    try resp.error_(401, "Invalid Authorization header format");
                    return false;
                }
                
                const token = auth_header[7..]; // Skip "Bearer "
                
                // Validate token
                const jwt_config = JwtConfig{
                    .secret = config.jwt_secret,
                    .expiration = config.jwt_expiration,
                };
                
                var jwt = Jwt.init(req.allocator, jwt_config);
                const claims = jwt.validate(token) catch |err| {
                    const msg = switch (err) {
                        error.TokenExpired => "Token expired",
                        error.InvalidSignature => "Invalid token signature",
                        error.InvalidToken => "Invalid token format",
                        else => "Authentication failed",
                    };
                    try resp.error_(401, msg);
                    return false;
                };
                
                // Check required role if specified
                if (config.required_role) |required| {
                    var has_role = false;
                    for (claims.roles) |role| {
                        if (std.mem.eql(u8, role, required)) {
                            has_role = true;
                            break;
                        }
                    }
                    
                    if (!has_role) {
                        try resp.error_(403, "Insufficient permissions");
                        return false;
                    }
                }
                
                // Store user ID in request for handlers
                try req.params.put(
                    try req.allocator.dupe(u8, "auth_user_id"),
                    try req.allocator.dupe(u8, claims.sub),
                );
                
                return true; // Continue to next middleware/handler
            }
        }.handle,
    };
}

/// Create API key authentication middleware
pub fn apiKeyAuthMiddleware(valid_keys: []const []const u8) Middleware {
    return Middleware{
        .name = "api_key_auth",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Get API key from header
                const api_key = req.header("X-API-Key") orelse {
                    try resp.error_(401, "Missing X-API-Key header");
                    return false;
                };
                
                // Validate API key
                for (valid_keys) |valid_key| {
                    if (std.mem.eql(u8, api_key, valid_key)) {
                        return true; // Valid key, continue
                    }
                }
                
                try resp.error_(401, "Invalid API key");
                return false;
            }
        }.handle,
    };
}

/// Create role-based access control middleware
pub fn rbacMiddleware(jwt_secret: []const u8, required_roles: []const []const u8) Middleware {
    return Middleware{
        .name = "rbac",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                // Get token
                const auth_header = req.header("Authorization") orelse {
                    try resp.error_(401, "Missing Authorization header");
                    return false;
                };
                
                if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
                    try resp.error_(401, "Invalid Authorization header");
                    return false;
                }
                
                const token = auth_header[7..];
                
                // Validate and extract claims
                const jwt_config = JwtConfig{
                    .secret = jwt_secret,
                };
                
                var jwt = Jwt.init(req.allocator, jwt_config);
                const claims = jwt.validate(token) catch {
                    try resp.error_(401, "Invalid or expired token");
                    return false;
                };
                
                // Check if user has any of the required roles
                for (required_roles) |required_role| {
                    for (claims.roles) |user_role| {
                        if (std.mem.eql(u8, user_role, required_role)) {
                            return true; // Has required role
                        }
                    }
                }
                
                try resp.error_(403, "Insufficient permissions");
                return false;
            }
        }.handle,
    };
}

/// Create optional authentication middleware (doesn't block if no token)
pub fn optionalAuthMiddleware(jwt_secret: []const u8) Middleware {
    return Middleware{
        .name = "optional_auth",
        .handler = struct {
            fn handle(req: *Request, resp: *Response) !bool {
                _ = resp;
                
                // Try to get Authorization header
                const auth_header = req.header("Authorization") orelse {
                    return true; // No auth header, continue without auth
                };
                
                if (!std.mem.startsWith(u8, auth_header, "Bearer ")) {
                    return true; // Invalid format, continue without auth
                }
                
                const token = auth_header[7..];
                
                // Try to validate token
                const jwt_config = JwtConfig{
                    .secret = jwt_secret,
                };
                
                var jwt = Jwt.init(req.allocator, jwt_config);
                const claims = jwt.validate(token) catch {
                    return true; // Invalid token, continue without auth
                };
                
                // Store user ID if valid
                try req.params.put(
                    try req.allocator.dupe(u8, "auth_user_id"),
                    try req.allocator.dupe(u8, claims.sub),
                );
                
                return true;
            }
        }.handle,
    };
}
