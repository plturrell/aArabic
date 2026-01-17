// ============================================================================
// HyperShimmy Unit Tests - Security Module
// ============================================================================
// Day 56: Comprehensive unit tests for security operations
// ============================================================================

const std = @import("std");
const testing = std.testing;
const security = @import("../../server/security.zig");

// ============================================================================
// Input Validation Tests
// ============================================================================

test "InputValidator - valid alphanumeric" {
    var validator = security.InputValidator.init();
    
    try testing.expect(validator.isValidAlphanumeric("abc123"));
    try testing.expect(validator.isValidAlphanumeric("Test123"));
    try testing.expect(validator.isValidAlphanumeric("UPPERCASE"));
    try testing.expect(validator.isValidAlphanumeric("lowercase"));
}

test "InputValidator - invalid alphanumeric" {
    var validator = security.InputValidator.init();
    
    try testing.expect(!validator.isValidAlphanumeric("test@example"));
    try testing.expect(!validator.isValidAlphanumeric("hello world"));
    try testing.expect(!validator.isValidAlphanumeric("test-123"));
    try testing.expect(!validator.isValidAlphanumeric(""));
}

test "InputValidator - valid URL" {
    var validator = security.InputValidator.init();
    
    try testing.expect(validator.isValidUrl("https://example.com"));
    try testing.expect(validator.isValidUrl("http://test.org/path"));
    try testing.expect(validator.isValidUrl("https://sub.domain.com:8080/path?query=value"));
}

test "InputValidator - invalid URL" {
    var validator = security.InputValidator.init();
    
    try testing.expect(!validator.isValidUrl("ftp://example.com"));
    try testing.expect(!validator.isValidUrl("javascript:alert(1)"));
    try testing.expect(!validator.isValidUrl("//example.com"));
    try testing.expect(!validator.isValidUrl(""));
    try testing.expect(!validator.isValidUrl("not a url"));
}

test "InputValidator - valid email" {
    var validator = security.InputValidator.init();
    
    try testing.expect(validator.isValidEmail("test@example.com"));
    try testing.expect(validator.isValidEmail("user.name@domain.co.uk"));
    try testing.expect(validator.isValidEmail("firstname+lastname@example.com"));
}

test "InputValidator - invalid email" {
    var validator = security.InputValidator.init();
    
    try testing.expect(!validator.isValidEmail("invalid"));
    try testing.expect(!validator.isValidEmail("@example.com"));
    try testing.expect(!validator.isValidEmail("user@"));
    try testing.expect(!validator.isValidEmail(""));
    try testing.expect(!validator.isValidEmail("user name@example.com"));
}

test "InputValidator - length check" {
    var validator = security.InputValidator.init();
    
    try testing.expect(validator.checkLength("test", 1, 10));
    try testing.expect(validator.checkLength("exact", 5, 5));
    try testing.expect(!validator.checkLength("", 1, 10));
    try testing.expect(!validator.checkLength("toolong", 1, 5));
}

// ============================================================================
// Sanitization Tests
// ============================================================================

test "Sanitizer - HTML escape" {
    const allocator = testing.allocator;
    var sanitizer = security.Sanitizer.init(allocator);
    
    const input = "<script>alert('xss')</script>";
    const output = try sanitizer.escapeHtml(input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, "&lt;") != null);
    try testing.expect(std.mem.indexOf(u8, output, "&gt;") != null);
    try testing.expect(std.mem.indexOf(u8, output, "<script>") == null);
}

test "Sanitizer - SQL escape" {
    const allocator = testing.allocator;
    var sanitizer = security.Sanitizer.init(allocator);
    
    const input = "test' OR '1'='1";
    const output = try sanitizer.escapeSql(input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, "''") != null);
}

test "Sanitizer - path traversal prevention" {
    const allocator = testing.allocator;
    var sanitizer = security.Sanitizer.init(allocator);
    
    const safe = try sanitizer.sanitizePath("safe/path/file.txt");
    defer allocator.free(safe);
    try testing.expect(std.mem.indexOf(u8, safe, "..") == null);
    
    const unsafe = try sanitizer.sanitizePath("../../../etc/passwd");
    defer allocator.free(unsafe);
    try testing.expect(std.mem.indexOf(u8, unsafe, "..") == null);
}

test "Sanitizer - filename sanitization" {
    const allocator = testing.allocator;
    var sanitizer = security.Sanitizer.init(allocator);
    
    const input = "my file (1).pdf";
    const output = try sanitizer.sanitizeFilename(input);
    defer allocator.free(output);
    
    try testing.expect(std.mem.indexOf(u8, output, " ") == null);
    try testing.expect(std.mem.indexOf(u8, output, "(") == null);
}

// ============================================================================
// Rate Limiting Tests
// ============================================================================

test "RateLimiter - basic operation" {
    const allocator = testing.allocator;
    var limiter = security.RateLimiter.init(allocator, 5, 60);
    defer limiter.deinit();
    
    const client_id = "test-client";
    
    // First 5 requests should succeed
    var i: usize = 0;
    while (i < 5) : (i += 1) {
        try testing.expect(try limiter.checkLimit(client_id));
    }
    
    // 6th request should fail
    try testing.expect(!try limiter.checkLimit(client_id));
}

test "RateLimiter - multiple clients" {
    const allocator = testing.allocator;
    var limiter = security.RateLimiter.init(allocator, 3, 60);
    defer limiter.deinit();
    
    try testing.expect(try limiter.checkLimit("client1"));
    try testing.expect(try limiter.checkLimit("client2"));
    try testing.expect(try limiter.checkLimit("client1"));
    try testing.expect(try limiter.checkLimit("client2"));
}

test "RateLimiter - window expiry" {
    const allocator = testing.allocator;
    var limiter = security.RateLimiter.init(allocator, 2, 1);
    defer limiter.deinit();
    
    const client_id = "test-client";
    
    // Use up limit
    try testing.expect(try limiter.checkLimit(client_id));
    try testing.expect(try limiter.checkLimit(client_id));
    try testing.expect(!try limiter.checkLimit(client_id));
    
    // Wait for window to expire
    std.time.sleep(1 * std.time.ns_per_s + 100 * std.time.ns_per_ms);
    
    // Should work again
    try testing.expect(try limiter.checkLimit(client_id));
}

// ============================================================================
// CORS Tests
// ============================================================================

test "CorsConfig - default configuration" {
    const allocator = testing.allocator;
    const config = security.CorsConfig.initDefault(allocator);
    defer config.deinit();
    
    try testing.expectEqualStrings("*", config.allowed_origins);
    try testing.expectEqualStrings("GET,POST,PUT,DELETE,OPTIONS", config.allowed_methods);
    try testing.expectEqual(@as(u32, 86400), config.max_age);
}

test "CorsConfig - custom configuration" {
    const allocator = testing.allocator;
    var config = try security.CorsConfig.init(
        allocator,
        "https://example.com",
        "GET,POST",
        "Content-Type",
        3600,
        true,
    );
    defer config.deinit();
    
    try testing.expectEqualStrings("https://example.com", config.allowed_origins);
    try testing.expectEqualStrings("GET,POST", config.allowed_methods);
    try testing.expectEqual(@as(u32, 3600), config.max_age);
    try testing.expect(config.allow_credentials);
}

test "CorsConfig - origin check" {
    const allocator = testing.allocator;
    var config = try security.CorsConfig.init(
        allocator,
        "https://example.com,https://test.com",
        "GET,POST",
        "Content-Type",
        3600,
        false,
    );
    defer config.deinit();
    
    try testing.expect(config.isOriginAllowed("https://example.com"));
    try testing.expect(config.isOriginAllowed("https://test.com"));
    try testing.expect(!config.isOriginAllowed("https://evil.com"));
}

test "CorsConfig - wildcard origin" {
    const allocator = testing.allocator;
    const config = security.CorsConfig.initDefault(allocator);
    defer config.deinit();
    
    try testing.expect(config.isOriginAllowed("https://any-domain.com"));
    try testing.expect(config.isOriginAllowed("http://localhost:3000"));
}

// ============================================================================
// Token Generation Tests
// ============================================================================

test "TokenGenerator - generate token" {
    const allocator = testing.allocator;
    var generator = security.TokenGenerator.init(allocator);
    
    const token1 = try generator.generate(32);
    defer allocator.free(token1);
    
    const token2 = try generator.generate(32);
    defer allocator.free(token2);
    
    try testing.expectEqual(@as(usize, 64), token1.len); // hex encoded
    try testing.expectEqual(@as(usize, 64), token2.len);
    
    // Tokens should be different
    try testing.expect(!std.mem.eql(u8, token1, token2));
}

test "TokenGenerator - token length" {
    const allocator = testing.allocator;
    var generator = security.TokenGenerator.init(allocator);
    
    const token8 = try generator.generate(8);
    defer allocator.free(token8);
    try testing.expectEqual(@as(usize, 16), token8.len);
    
    const token16 = try generator.generate(16);
    defer allocator.free(token16);
    try testing.expectEqual(@as(usize, 32), token16.len);
}

test "TokenGenerator - token characters" {
    const allocator = testing.allocator;
    var generator = security.TokenGenerator.init(allocator);
    
    const token = try generator.generate(32);
    defer allocator.free(token);
    
    // Check all characters are valid hex
    for (token) |c| {
        try testing.expect((c >= '0' and c <= '9') or (c >= 'a' and c <= 'f'));
    }
}

// ============================================================================
// Content Security Policy Tests
// ============================================================================

test "CSPBuilder - default policy" {
    const allocator = testing.allocator;
    const builder = security.CSPBuilder.init(allocator);
    
    const policy = try builder.buildDefault();
    defer allocator.free(policy);
    
    try testing.expect(std.mem.indexOf(u8, policy, "default-src") != null);
    try testing.expect(std.mem.indexOf(u8, policy, "'self'") != null);
}

test "CSPBuilder - custom directives" {
    const allocator = testing.allocator;
    var builder = security.CSPBuilder.init(allocator);
    
    try builder.addDirective("script-src", "'self' 'unsafe-inline'");
    try builder.addDirective("style-src", "'self' https://fonts.googleapis.com");
    
    const policy = try builder.build();
    defer allocator.free(policy);
    
    try testing.expect(std.mem.indexOf(u8, policy, "script-src") != null);
    try testing.expect(std.mem.indexOf(u8, policy, "style-src") != null);
    try testing.expect(std.mem.indexOf(u8, policy, "unsafe-inline") != null);
}

// ============================================================================
// Security Headers Tests
// ============================================================================

test "SecurityHeaders - default headers" {
    const allocator = testing.allocator;
    const headers = security.SecurityHeaders.initDefault(allocator);
    defer headers.deinit();
    
    try testing.expectEqualStrings("nosniff", headers.x_content_type_options);
    try testing.expectEqualStrings("DENY", headers.x_frame_options);
    try testing.expectEqualStrings("1; mode=block", headers.x_xss_protection);
}

test "SecurityHeaders - HSTS configuration" {
    const allocator = testing.allocator;
    var headers = security.SecurityHeaders.initDefault(allocator);
    defer headers.deinit();
    
    try testing.expect(std.mem.indexOf(u8, headers.strict_transport_security, "max-age=") != null);
    try testing.expect(std.mem.indexOf(u8, headers.strict_transport_security, "includeSubDomains") != null);
}

// ============================================================================
// File Upload Security Tests
// ============================================================================

test "FileUploadValidator - valid file types" {
    var validator = security.FileUploadValidator.init();
    
    try testing.expect(validator.isAllowedFileType("document.pdf"));
    try testing.expect(validator.isAllowedFileType("image.jpg"));
    try testing.expect(validator.isAllowedFileType("text.txt"));
    try testing.expect(validator.isAllowedFileType("data.json"));
}

test "FileUploadValidator - invalid file types" {
    var validator = security.FileUploadValidator.init();
    
    try testing.expect(!validator.isAllowedFileType("script.exe"));
    try testing.expect(!validator.isAllowedFileType("malware.dll"));
    try testing.expect(!validator.isAllowedFileType("virus.bat"));
    try testing.expect(!validator.isAllowedFileType("hack.sh"));
}

test "FileUploadValidator - file size limit" {
    var validator = security.FileUploadValidator.init();
    validator.max_file_size = 1024 * 1024; // 1MB
    
    try testing.expect(validator.isValidFileSize(500 * 1024));
    try testing.expect(validator.isValidFileSize(1024 * 1024));
    try testing.expect(!validator.isValidFileSize(2 * 1024 * 1024));
}

test "FileUploadValidator - filename sanitization" {
    const allocator = testing.allocator;
    var validator = security.FileUploadValidator.init();
    
    const safe1 = try validator.sanitizeFilename(allocator, "my document.pdf");
    defer allocator.free(safe1);
    try testing.expectEqualStrings("my_document.pdf", safe1);
    
    const safe2 = try validator.sanitizeFilename(allocator, "../../../etc/passwd");
    defer allocator.free(safe2);
    try testing.expect(std.mem.indexOf(u8, safe2, "..") == null);
}

// ============================================================================
// Request Validation Tests
// ============================================================================

test "RequestValidator - valid content type" {
    var validator = security.RequestValidator.init();
    
    try testing.expect(validator.isValidContentType("application/json"));
    try testing.expect(validator.isValidContentType("multipart/form-data"));
    try testing.expect(validator.isValidContentType("text/plain"));
}

test "RequestValidator - invalid content type" {
    var validator = security.RequestValidator.init();
    
    try testing.expect(!validator.isValidContentType("application/x-malicious"));
    try testing.expect(!validator.isValidContentType(""));
}

test "RequestValidator - header validation" {
    var validator = security.RequestValidator.init();
    
    try testing.expect(validator.isValidHeader("Content-Type", "application/json"));
    try testing.expect(validator.isValidHeader("Authorization", "Bearer token123"));
    try testing.expect(!validator.isValidHeader("X-Script", "<script>alert(1)</script>"));
}

// ============================================================================
// Security Metrics Tests
// ============================================================================

test "SecurityMetrics - record events" {
    var metrics = security.SecurityMetrics{};
    
    metrics.recordBlocked(.xss_attempt);
    metrics.recordBlocked(.sql_injection);
    metrics.recordBlocked(.rate_limit);
    
    try testing.expectEqual(@as(u64, 3), metrics.total_blocked);
    try testing.expectEqual(@as(u64, 1), metrics.xss_blocked);
    try testing.expectEqual(@as(u64, 1), metrics.sql_injection_blocked);
    try testing.expectEqual(@as(u64, 1), metrics.rate_limit_blocked);
}

test "SecurityMetrics - reset" {
    var metrics = security.SecurityMetrics{};
    
    metrics.recordBlocked(.xss_attempt);
    metrics.recordBlocked(.sql_injection);
    
    try testing.expectEqual(@as(u64, 2), metrics.total_blocked);
    
    metrics.reset();
    
    try testing.expectEqual(@as(u64, 0), metrics.total_blocked);
    try testing.expectEqual(@as(u64, 0), metrics.xss_blocked);
}

test "SecurityMetrics - JSON export" {
    const allocator = testing.allocator;
    var metrics = security.SecurityMetrics{};
    
    metrics.recordBlocked(.xss_attempt);
    metrics.recordBlocked(.sql_injection);
    
    const json = try metrics.toJson(allocator);
    defer allocator.free(json);
    
    try testing.expect(std.mem.indexOf(u8, json, "\"total_blocked\":2") != null);
    try testing.expect(std.mem.indexOf(u8, json, "\"xss_blocked\":1") != null);
}
