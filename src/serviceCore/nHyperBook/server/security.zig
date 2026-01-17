const std = @import("std");
const mem = std.mem;
const crypto = std.crypto;

/// Security module for HyperShimmy
/// Provides input validation, sanitization, rate limiting, and security headers
pub const Security = struct {
    allocator: mem.Allocator,
    rate_limiter: RateLimiter,
    csp_policy: []const u8,
    
    pub fn init(allocator: mem.Allocator) !Security {
        return Security{
            .allocator = allocator,
            .rate_limiter = RateLimiter.init(allocator),
            .csp_policy = "default-src 'self'; script-src 'self' 'unsafe-inline' https://sapui5.hana.ondemand.com; style-src 'self' 'unsafe-inline' https://sapui5.hana.ondemand.com; img-src 'self' data: https:; font-src 'self' https://sapui5.hana.ondemand.com; connect-src 'self'; frame-ancestors 'none';",
        };
    }
    
    pub fn deinit(self: *Security) void {
        self.rate_limiter.deinit();
    }
    
    /// Get security headers for HTTP responses
    pub fn getSecurityHeaders(self: *Security, allocator: mem.Allocator) !std.StringHashMap([]const u8) {
        var headers = std.StringHashMap([]const u8).init(allocator);
        
        // Content Security Policy
        try headers.put("Content-Security-Policy", self.csp_policy);
        
        // X-Frame-Options: Prevent clickjacking
        try headers.put("X-Frame-Options", "DENY");
        
        // X-Content-Type-Options: Prevent MIME type sniffing
        try headers.put("X-Content-Type-Options", "nosniff");
        
        // X-XSS-Protection: Enable XSS filter
        try headers.put("X-XSS-Protection", "1; mode=block");
        
        // Strict-Transport-Security: Force HTTPS
        try headers.put("Strict-Transport-Security", "max-age=31536000; includeSubDomains");
        
        // Referrer-Policy: Control referrer information
        try headers.put("Referrer-Policy", "strict-origin-when-cross-origin");
        
        // Permissions-Policy: Disable unnecessary features
        try headers.put("Permissions-Policy", "geolocation=(), microphone=(), camera=()");
        
        return headers;
    }
    
    /// Check if request should be rate limited
    pub fn checkRateLimit(self: *Security, client_id: []const u8) !bool {
        return self.rate_limiter.checkLimit(client_id);
    }
};

/// Input validator for user-provided data
pub const InputValidator = struct {
    /// Validate URL format
    pub fn validateURL(url: []const u8) bool {
        if (url.len == 0 or url.len > 2048) return false;
        
        // Must start with http:// or https://
        if (!mem.startsWith(u8, url, "http://") and !mem.startsWith(u8, url, "https://")) {
            return false;
        }
        
        // Check for obvious injection attempts
        if (mem.indexOf(u8, url, "<script") != null or
            mem.indexOf(u8, url, "javascript:") != null or
            mem.indexOf(u8, url, "data:") != null) {
            return false;
        }
        
        return true;
    }
    
    /// Validate file path to prevent directory traversal
    pub fn validatePath(path: []const u8) bool {
        if (path.len == 0 or path.len > 1024) return false;
        
        // Prevent directory traversal
        if (mem.indexOf(u8, path, "..") != null) return false;
        if (mem.indexOf(u8, path, "~") != null) return false;
        if (mem.startsWith(u8, path, "/")) return false;
        
        // Check for null bytes
        if (mem.indexOf(u8, path, "\x00") != null) return false;
        
        return true;
    }
    
    /// Validate file extension is allowed
    pub fn validateFileExtension(filename: []const u8, allowed: []const []const u8) bool {
        for (allowed) |ext| {
            if (mem.endsWith(u8, filename, ext)) {
                return true;
            }
        }
        return false;
    }
    
    /// Validate string length
    pub fn validateLength(input: []const u8, min: usize, max: usize) bool {
        return input.len >= min and input.len <= max;
    }
    
    /// Validate that string contains only safe characters
    pub fn validateSafeString(input: []const u8) bool {
        for (input) |c| {
            // Allow alphanumeric, space, and common punctuation
            if (!std.ascii.isAlphanumeric(c) and 
                c != ' ' and c != '-' and c != '_' and 
                c != '.' and c != ',' and c != '!' and c != '?') {
                return false;
            }
        }
        return true;
    }
    
    /// Validate email format (basic check)
    pub fn validateEmail(email: []const u8) bool {
        if (email.len < 3 or email.len > 254) return false;
        
        const at_pos = mem.indexOf(u8, email, "@") orelse return false;
        const dot_pos = mem.lastIndexOf(u8, email, ".") orelse return false;
        
        return at_pos > 0 and dot_pos > at_pos + 1 and dot_pos < email.len - 1;
    }
};

/// String sanitizer to prevent XSS attacks
pub const Sanitizer = struct {
    /// HTML escape special characters
    pub fn escapeHTML(allocator: mem.Allocator, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();
        
        for (input) |c| {
            switch (c) {
                '<' => try result.appendSlice("&lt;"),
                '>' => try result.appendSlice("&gt;"),
                '&' => try result.appendSlice("&amp;"),
                '"' => try result.appendSlice("&quot;"),
                '\'' => try result.appendSlice("&#x27;"),
                '/' => try result.appendSlice("&#x2F;"),
                else => try result.append(c),
            }
        }
        
        return result.toOwnedSlice();
    }
    
    /// Remove potentially dangerous characters
    pub fn sanitizeInput(allocator: mem.Allocator, input: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();
        
        for (input) |c| {
            // Skip control characters except newline and tab
            if (c < 32 and c != '\n' and c != '\t') continue;
            
            // Skip null bytes
            if (c == 0) continue;
            
            try result.append(c);
        }
        
        return result.toOwnedSlice();
    }
    
    /// Sanitize filename to prevent path traversal
    pub fn sanitizeFilename(allocator: mem.Allocator, filename: []const u8) ![]const u8 {
        var result = std.ArrayList(u8).init(allocator);
        defer result.deinit();
        
        for (filename) |c| {
            switch (c) {
                // Replace path separators and dangerous characters
                '/', '\\', ':', '*', '?', '"', '<', '>', '|' => try result.append('_'),
                // Allow safe characters
                'a'...'z', 'A'...'Z', '0'...'9', '.', '-', '_' => try result.append(c),
                else => {},
            }
        }
        
        return result.toOwnedSlice();
    }
};

/// Rate limiter to prevent abuse
pub const RateLimiter = struct {
    allocator: mem.Allocator,
    limits: std.StringHashMap(RateLimit),
    max_requests: u32 = 100,
    window_seconds: i64 = 60,
    
    const RateLimit = struct {
        count: u32,
        window_start: i64,
    };
    
    pub fn init(allocator: mem.Allocator) RateLimiter {
        return RateLimiter{
            .allocator = allocator,
            .limits = std.StringHashMap(RateLimit).init(allocator),
        };
    }
    
    pub fn deinit(self: *RateLimiter) void {
        self.limits.deinit();
    }
    
    /// Check if client has exceeded rate limit
    pub fn checkLimit(self: *RateLimiter, client_id: []const u8) !bool {
        const now = std.time.timestamp();
        
        if (self.limits.get(client_id)) |limit| {
            // Check if we're in a new window
            if (now - limit.window_start >= self.window_seconds) {
                // Reset counter for new window
                try self.limits.put(client_id, .{
                    .count = 1,
                    .window_start = now,
                });
                return true;
            }
            
            // Check if limit exceeded
            if (limit.count >= self.max_requests) {
                return false;
            }
            
            // Increment counter
            try self.limits.put(client_id, .{
                .count = limit.count + 1,
                .window_start = limit.window_start,
            });
            return true;
        }
        
        // First request from this client
        try self.limits.put(client_id, .{
            .count = 1,
            .window_start = now,
        });
        return true;
    }
    
    /// Clean up old entries
    pub fn cleanup(self: *RateLimiter) !void {
        const now = std.time.timestamp();
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();
        
        var iter = self.limits.iterator();
        while (iter.next()) |entry| {
            if (now - entry.value_ptr.window_start >= self.window_seconds * 2) {
                try to_remove.append(entry.key_ptr.*);
            }
        }
        
        for (to_remove.items) |key| {
            _ = self.limits.remove(key);
        }
    }
};

/// CSRF token generator and validator
pub const CSRFProtection = struct {
    allocator: mem.Allocator,
    tokens: std.StringHashMap(i64),
    token_lifetime: i64 = 3600, // 1 hour
    
    pub fn init(allocator: mem.Allocator) CSRFProtection {
        return CSRFProtection{
            .allocator = allocator,
            .tokens = std.StringHashMap(i64).init(allocator),
        };
    }
    
    pub fn deinit(self: *CSRFProtection) void {
        self.tokens.deinit();
    }
    
    /// Generate a new CSRF token
    pub fn generateToken(self: *CSRFProtection, allocator: mem.Allocator) ![]const u8 {
        var random_bytes: [32]u8 = undefined;
        crypto.random.bytes(&random_bytes);
        
        // Convert to hex string
        var token = try allocator.alloc(u8, 64);
        _ = std.fmt.bufPrint(token, "{x}", .{std.fmt.fmtSliceHexLower(&random_bytes)}) catch unreachable;
        
        // Store token with timestamp
        const now = std.time.timestamp();
        try self.tokens.put(token, now);
        
        return token;
    }
    
    /// Validate a CSRF token
    pub fn validateToken(self: *CSRFProtection, token: []const u8) bool {
        const timestamp = self.tokens.get(token) orelse return false;
        const now = std.time.timestamp();
        
        // Check if token has expired
        if (now - timestamp > self.token_lifetime) {
            _ = self.tokens.remove(token);
            return false;
        }
        
        return true;
    }
    
    /// Clean up expired tokens
    pub fn cleanup(self: *CSRFProtection) !void {
        const now = std.time.timestamp();
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();
        
        var iter = self.tokens.iterator();
        while (iter.next()) |entry| {
            if (now - entry.value_ptr.* > self.token_lifetime) {
                try to_remove.append(entry.key_ptr.*);
            }
        }
        
        for (to_remove.items) |key| {
            _ = self.tokens.remove(key);
        }
    }
};

/// Password strength validator
pub const PasswordValidator = struct {
    min_length: usize = 12,
    require_uppercase: bool = true,
    require_lowercase: bool = true,
    require_digit: bool = true,
    require_special: bool = true,
    
    pub const ValidationResult = struct {
        valid: bool,
        errors: []const []const u8,
    };
    
    /// Validate password strength
    pub fn validate(self: *const PasswordValidator, allocator: mem.Allocator, password: []const u8) !ValidationResult {
        var errors = std.ArrayList([]const u8).init(allocator);
        
        // Check minimum length
        if (password.len < self.min_length) {
            try errors.append(try std.fmt.allocPrint(allocator, "Password must be at least {} characters", .{self.min_length}));
        }
        
        // Check for required character types
        var has_uppercase = false;
        var has_lowercase = false;
        var has_digit = false;
        var has_special = false;
        
        for (password) |c| {
            if (std.ascii.isUpper(c)) has_uppercase = true;
            if (std.ascii.isLower(c)) has_lowercase = true;
            if (std.ascii.isDigit(c)) has_digit = true;
            if (!std.ascii.isAlphanumeric(c)) has_special = true;
        }
        
        if (self.require_uppercase and !has_uppercase) {
            try errors.append("Password must contain at least one uppercase letter");
        }
        if (self.require_lowercase and !has_lowercase) {
            try errors.append("Password must contain at least one lowercase letter");
        }
        if (self.require_digit and !has_digit) {
            try errors.append("Password must contain at least one digit");
        }
        if (self.require_special and !has_special) {
            try errors.append("Password must contain at least one special character");
        }
        
        return ValidationResult{
            .valid = errors.items.len == 0,
            .errors = errors.toOwnedSlice(),
        };
    }
};

/// Secure session manager
pub const SessionManager = struct {
    allocator: mem.Allocator,
    sessions: std.StringHashMap(Session),
    session_lifetime: i64 = 86400, // 24 hours
    
    pub const Session = struct {
        id: []const u8,
        user_id: []const u8,
        created_at: i64,
        last_active: i64,
        data: std.StringHashMap([]const u8),
    };
    
    pub fn init(allocator: mem.Allocator) SessionManager {
        return SessionManager{
            .allocator = allocator,
            .sessions = std.StringHashMap(Session).init(allocator),
        };
    }
    
    pub fn deinit(self: *SessionManager) void {
        var iter = self.sessions.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.data.deinit();
        }
        self.sessions.deinit();
    }
    
    /// Create a new session
    pub fn createSession(self: *SessionManager, user_id: []const u8) ![]const u8 {
        var random_bytes: [32]u8 = undefined;
        crypto.random.bytes(&random_bytes);
        
        var session_id = try self.allocator.alloc(u8, 64);
        _ = std.fmt.bufPrint(session_id, "{x}", .{std.fmt.fmtSliceHexLower(&random_bytes)}) catch unreachable;
        
        const now = std.time.timestamp();
        try self.sessions.put(session_id, .{
            .id = session_id,
            .user_id = user_id,
            .created_at = now,
            .last_active = now,
            .data = std.StringHashMap([]const u8).init(self.allocator),
        });
        
        return session_id;
    }
    
    /// Validate and update session
    pub fn validateSession(self: *SessionManager, session_id: []const u8) bool {
        var session = self.sessions.getPtr(session_id) orelse return false;
        
        const now = std.time.timestamp();
        
        // Check if session has expired
        if (now - session.last_active > self.session_lifetime) {
            _ = self.destroySession(session_id);
            return false;
        }
        
        // Update last active time
        session.last_active = now;
        return true;
    }
    
    /// Destroy a session
    pub fn destroySession(self: *SessionManager, session_id: []const u8) bool {
        if (self.sessions.getPtr(session_id)) |session| {
            session.data.deinit();
            return self.sessions.remove(session_id);
        }
        return false;
    }
    
    /// Clean up expired sessions
    pub fn cleanup(self: *SessionManager) !void {
        const now = std.time.timestamp();
        var to_remove = std.ArrayList([]const u8).init(self.allocator);
        defer to_remove.deinit();
        
        var iter = self.sessions.iterator();
        while (iter.next()) |entry| {
            if (now - entry.value_ptr.last_active > self.session_lifetime) {
                try to_remove.append(entry.key_ptr.*);
            }
        }
        
        for (to_remove.items) |key| {
            _ = self.destroySession(key);
        }
    }
};

// Tests
test "input validation" {
    const testing = std.testing;
    
    // URL validation
    try testing.expect(InputValidator.validateURL("https://example.com"));
    try testing.expect(InputValidator.validateURL("http://localhost:8080"));
    try testing.expect(!InputValidator.validateURL("javascript:alert(1)"));
    try testing.expect(!InputValidator.validateURL("https://example.com/<script>"));
    
    // Path validation
    try testing.expect(InputValidator.validatePath("file.txt"));
    try testing.expect(InputValidator.validatePath("folder/file.txt"));
    try testing.expect(!InputValidator.validatePath("../../../etc/passwd"));
    try testing.expect(!InputValidator.validatePath("/etc/passwd"));
    
    // Email validation
    try testing.expect(InputValidator.validateEmail("user@example.com"));
    try testing.expect(!InputValidator.validateEmail("invalid"));
    try testing.expect(!InputValidator.validateEmail("@example.com"));
}

test "sanitization" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    // HTML escaping
    const escaped = try Sanitizer.escapeHTML(allocator, "<script>alert('xss')</script>");
    defer allocator.free(escaped);
    try testing.expectEqualStrings("&lt;script&gt;alert(&#x27;xss&#x27;)&lt;&#x2F;script&gt;", escaped);
    
    // Filename sanitization
    const sanitized = try Sanitizer.sanitizeFilename(allocator, "../../../etc/passwd");
    defer allocator.free(sanitized);
    try testing.expect(std.mem.indexOf(u8, sanitized, "/") == null);
    try testing.expect(std.mem.indexOf(u8, sanitized, ".") == null);
}

test "rate limiting" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var limiter = RateLimiter.init(allocator);
    defer limiter.deinit();
    
    limiter.max_requests = 3;
    
    // Should allow first 3 requests
    try testing.expect(try limiter.checkLimit("client1"));
    try testing.expect(try limiter.checkLimit("client1"));
    try testing.expect(try limiter.checkLimit("client1"));
    
    // Should block 4th request
    try testing.expect(!try limiter.checkLimit("client1"));
    
    // Different client should be allowed
    try testing.expect(try limiter.checkLimit("client2"));
}

test "CSRF protection" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    var csrf = CSRFProtection.init(allocator);
    defer csrf.deinit();
    
    const token = try csrf.generateToken(allocator);
    defer allocator.free(token);
    
    // Token should be valid
    try testing.expect(csrf.validateToken(token));
    
    // Invalid token should fail
    try testing.expect(!csrf.validateToken("invalid_token"));
}

test "password validation" {
    const testing = std.testing;
    const allocator = testing.allocator;
    
    const validator = PasswordValidator{};
    
    // Strong password should pass
    const result1 = try validator.validate(allocator, "StrongP@ssw0rd123");
    defer allocator.free(result1.errors);
    try testing.expect(result1.valid);
    
    // Weak password should fail
    const result2 = try validator.validate(allocator, "weak");
    defer allocator.free(result2.errors);
    try testing.expect(!result2.valid);
    try testing.expect(result2.errors.len > 0);
}
