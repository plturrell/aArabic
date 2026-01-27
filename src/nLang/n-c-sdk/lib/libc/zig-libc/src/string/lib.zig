// String operations module
// Phase 1.1: Foundation - Basic string functions

const std = @import("std");

// ============================================================================
// Phase 1.1 Functions (Month 2)
// ============================================================================

/// Get the length of a null-terminated string
/// POSIX: size_t strlen(const char *s);
pub fn strlen(s: [*:0]const u8) usize {
    var len: usize = 0;
    while (s[len] != 0) : (len += 1) {}
    return len;
}

/// Copy string src to dest
/// POSIX: char *strcpy(char *dest, const char *src);
/// WARNING: No bounds checking - use strncpy for safety
pub fn strcpy(dest: [*:0]u8, src: [*:0]const u8) [*:0]u8 {
    var i: usize = 0;
    while (src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    dest[i] = 0;
    return dest;
}

/// Compare two null-terminated strings
/// POSIX: int strcmp(const char *s1, const char *s2);
/// Returns: <0 if s1 < s2, 0 if s1 == s2, >0 if s1 > s2
pub fn strcmp(s1: [*:0]const u8, s2: [*:0]const u8) i32 {
    var i: usize = 0;
    while (s1[i] != 0 and s2[i] != 0) : (i += 1) {
        if (s1[i] != s2[i]) {
            return @as(i32, s1[i]) - @as(i32, s2[i]);
        }
    }
    return @as(i32, s1[i]) - @as(i32, s2[i]);
}

/// Concatenate src to dest
/// POSIX: char *strcat(char *dest, const char *src);
/// WARNING: No bounds checking - ensure dest has enough space
pub fn strcat(dest: [*:0]u8, src: [*:0]const u8) [*:0]u8 {
    const dest_len = strlen(dest);
    var i: usize = 0;
    while (src[i] != 0) : (i += 1) {
        dest[dest_len + i] = src[i];
    }
    dest[dest_len + i] = 0;
    return dest;
}

/// Copy at most n bytes from src to dest
/// POSIX: char *strncpy(char *dest, const char *src, size_t n);
pub fn strncpy(dest: [*]u8, src: [*:0]const u8, n: usize) [*]u8 {
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[i] = src[i];
    }
    // Pad with zeros if needed
    while (i < n) : (i += 1) {
        dest[i] = 0;
    }
    return dest;
}

/// Compare at most n bytes of two strings
/// POSIX: int strncmp(const char *s1, const char *s2, size_t n);
pub fn strncmp(s1: [*:0]const u8, s2: [*:0]const u8, n: usize) i32 {
    if (n == 0) return 0;
    var i: usize = 0;
    while (i < n and s1[i] != 0 and s2[i] != 0) : (i += 1) {
        if (s1[i] != s2[i]) {
            return @as(i32, s1[i]) - @as(i32, s2[i]);
        }
    }
    if (i < n) {
        return @as(i32, s1[i]) - @as(i32, s2[i]);
    }
    return 0;
}

/// Concatenate at most n bytes from src to dest
/// POSIX: char *strncat(char *dest, const char *src, size_t n);
pub fn strncat(dest: [*:0]u8, src: [*:0]const u8, n: usize) [*:0]u8 {
    const dest_len = strlen(dest);
    var i: usize = 0;
    while (i < n and src[i] != 0) : (i += 1) {
        dest[dest_len + i] = src[i];
    }
    dest[dest_len + i] = 0;
    return dest;
}

/// Find first occurrence of character in string
/// POSIX: char *strchr(const char *s, int c);
pub fn strchr(s: [*:0]const u8, c: i32) ?[*:0]const u8 {
    const target = @as(u8, @intCast(c & 0xFF));
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (s[i] == target) {
            return s + i;
        }
    }
    // Check for null terminator match
    if (target == 0) return s + i;
    return null;
}

/// Find last occurrence of character in string
/// POSIX: char *strrchr(const char *s, int c);
pub fn strrchr(s: [*:0]const u8, c: i32) ?[*:0]const u8 {
    const target = @as(u8, @intCast(c & 0xFF));
    var last: ?[*:0]const u8 = null;
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (s[i] == target) {
            last = s + i;
        }
    }
    // Check for null terminator match
    if (target == 0) return s + i;
    return last;
}

/// Find first occurrence of substring
/// POSIX: char *strstr(const char *haystack, const char *needle);
pub fn strstr(haystack: [*:0]const u8, needle: [*:0]const u8) ?[*:0]const u8 {
    const needle_len = strlen(needle);
    if (needle_len == 0) return haystack;
    
    var i: usize = 0;
    while (haystack[i] != 0) : (i += 1) {
        var j: usize = 0;
        while (j < needle_len and haystack[i + j] != 0 and haystack[i + j] == needle[j]) : (j += 1) {}
        if (j == needle_len) {
            return haystack + i;
        }
    }
    return null;
}

/// Tokenize string (NOT thread-safe)
/// POSIX: char *strtok(char *str, const char *delim);
var strtok_state: ?[*:0]u8 = null;
pub fn strtok(str: ?[*:0]u8, delim: [*:0]const u8) ?[*:0]u8 {
    return strtok_r(str, delim, &strtok_state);
}

/// Tokenize string (thread-safe/reentrant)
/// POSIX: char *strtok_r(char *str, const char *delim, char **saveptr);
pub fn strtok_r(str: ?[*:0]u8, delim: [*:0]const u8, saveptr: *?[*:0]u8) ?[*:0]u8 {
    var s = str orelse saveptr.* orelse return null;
    
    // Skip leading delimiters
    while (s[0] != 0 and strpbrk(s, delim) == s) {
        s += 1;
    }
    
    if (s[0] == 0) {
        saveptr.* = null;
        return null;
    }
    
    // Find end of token
    const token_start = s;
    if (strpbrk(s, delim)) |delim_pos| {
        const offset = @intFromPtr(delim_pos) - @intFromPtr(s);
        s[offset] = 0;
        saveptr.* = s + offset + 1;
    } else {
        saveptr.* = null;
    }
    
    return token_start;
}

/// Get span of characters in set
/// POSIX: size_t strspn(const char *s, const char *accept);
pub fn strspn(s: [*:0]const u8, accept: [*:0]const u8) usize {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (strchr(accept, s[i]) == null) {
            break;
        }
    }
    return i;
}

/// Get span of characters NOT in set
/// POSIX: size_t strcspn(const char *s, const char *reject);
pub fn strcspn(s: [*:0]const u8, reject: [*:0]const u8) usize {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (strchr(reject, s[i]) != null) {
            break;
        }
    }
    return i;
}

/// Find first occurrence of any character from set
/// POSIX: char *strpbrk(const char *s, const char *accept);
pub fn strpbrk(s: [*:0]const u8, accept: [*:0]const u8) ?[*:0]const u8 {
    var i: usize = 0;
    while (s[i] != 0) : (i += 1) {
        if (strchr(accept, s[i]) != null) {
            return s + i;
        }
    }
    return null;
}

/// Get length of prefix matching reject set
/// POSIX: size_t strverscmp(const char *s1, const char *s2);  
/// Note: Simplified version without locale support
pub fn strnlen(s: [*:0]const u8, maxlen: usize) usize {
    var len: usize = 0;
    while (len < maxlen and s[len] != 0) : (len += 1) {}
    return len;
}

/// Case-insensitive string compare
/// POSIX: int strcasecmp(const char *s1, const char *s2);
pub fn strcasecmp(s1: [*:0]const u8, s2: [*:0]const u8) i32 {
    var i: usize = 0;
    while (s1[i] != 0 and s2[i] != 0) : (i += 1) {
        const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
        const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
        if (c1 != c2) {
            return @as(i32, c1) - @as(i32, c2);
        }
    }
    const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
    const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
    return @as(i32, c1) - @as(i32, c2);
}

/// Case-insensitive string compare (n bytes)
/// POSIX: int strncasecmp(const char *s1, const char *s2, size_t n);
pub fn strncasecmp(s1: [*:0]const u8, s2: [*:0]const u8, n: usize) i32 {
    if (n == 0) return 0;
    var i: usize = 0;
    while (i < n and s1[i] != 0 and s2[i] != 0) : (i += 1) {
        const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
        const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
        if (c1 != c2) {
            return @as(i32, c1) - @as(i32, c2);
        }
    }
    if (i < n) {
        const c1 = if (s1[i] >= 'A' and s1[i] <= 'Z') s1[i] + 32 else s1[i];
        const c2 = if (s2[i] >= 'A' and s2[i] <= 'Z') s2[i] + 32 else s2[i];
        return @as(i32, c1) - @as(i32, c2);
    }
    return 0;
}

/// Find first occurrence of substring (case-insensitive)
/// Note: Extension function, not POSIX standard
pub fn strcasestr(haystack: [*:0]const u8, needle: [*:0]const u8) ?[*:0]const u8 {
    const needle_len = strlen(needle);
    if (needle_len == 0) return haystack;
    
    var i: usize = 0;
    while (haystack[i] != 0) : (i += 1) {
        var j: usize = 0;
        while (j < needle_len and haystack[i + j] != 0) : (j += 1) {
            const c1 = if (haystack[i + j] >= 'A' and haystack[i + j] <= 'Z') 
                haystack[i + j] + 32 else haystack[i + j];
            const c2 = if (needle[j] >= 'A' and needle[j] <= 'Z') 
                needle[j] + 32 else needle[j];
            if (c1 != c2) break;
        }
        if (j == needle_len) {
            return haystack + i;
        }
    }
    return null;
}

// ============================================================================
// C-Compatible Exports (TODO: Add in Phase 1.2 with proper FFI)
// ============================================================================

// Note: C exports commented out for Phase 1.1
// Will be re-added with proper ABI compatibility testing in Phase 1.2

// ============================================================================
// Unit Tests
// ============================================================================

test "strlen - empty string" {
    const s: [*:0]const u8 = "";
    try std.testing.expectEqual(@as(usize, 0), strlen(s));
}

test "strlen - basic string" {
    const s: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(usize, 5), strlen(s));
}

test "strcpy - basic copy" {
    var dest: [10:0]u8 = undefined;
    const src: [*:0]const u8 = "test";
    _ = strcpy(&dest, src);
    try std.testing.expectEqualStrings("test", std.mem.span(&dest));
}

test "strcmp - equal strings" {
    const s1: [*:0]const u8 = "hello";
    const s2: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(i32, 0), strcmp(s1, s2));
}

test "strcmp - different strings" {
    const s1: [*:0]const u8 = "apple";
    const s2: [*:0]const u8 = "banana";
    try std.testing.expect(strcmp(s1, s2) < 0);
}

test "strcat - basic concatenation" {
    var dest: [20:0]u8 = undefined;
    @memcpy(dest[0..5], "hello");
    dest[5] = 0;
    const src: [*:0]const u8 = " world";
    _ = strcat(&dest, src);
    try std.testing.expectEqualStrings("hello world", std.mem.span(&dest));
}

test "strncpy - full copy" {
    var dest: [10]u8 = undefined;
    const src: [*:0]const u8 = "test";
    _ = strncpy(&dest, src, 10);
    try std.testing.expectEqualStrings("test", dest[0..4]);
    // Check zero padding
    try std.testing.expectEqual(@as(u8, 0), dest[4]);
}

test "strncpy - partial copy" {
    var dest: [10]u8 = undefined;
    const src: [*:0]const u8 = "testing";
    _ = strncpy(&dest, src, 4);
    try std.testing.expectEqualStrings("test", dest[0..4]);
}

test "strncmp - equal within n" {
    const s1: [*:0]const u8 = "hello";
    const s2: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(i32, 0), strncmp(s1, s2, 5));
}

test "strncmp - different within n" {
    const s1: [*:0]const u8 = "hello";
    const s2: [*:0]const u8 = "world";
    try std.testing.expect(strncmp(s1, s2, 5) != 0);
}

test "strncmp - equal before n" {
    const s1: [*:0]const u8 = "hello";
    const s2: [*:0]const u8 = "hello world";
    try std.testing.expectEqual(@as(i32, 0), strncmp(s1, s2, 5));
}

test "strncat - basic concatenation" {
    var dest: [20:0]u8 = undefined;
    @memcpy(dest[0..5], "hello");
    dest[5] = 0;
    const src: [*:0]const u8 = " world!!!";
    _ = strncat(&dest, src, 6);
    try std.testing.expectEqualStrings("hello world", std.mem.span(&dest));
}

test "strchr - find character" {
    const s: [*:0]const u8 = "hello world";
    const result = strchr(s, 'o');
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 'o'), result.?[0]);
}

test "strchr - not found" {
    const s: [*:0]const u8 = "hello";
    try std.testing.expect(strchr(s, 'z') == null);
}

test "strchr - find null terminator" {
    const s: [*:0]const u8 = "test";
    const result = strchr(s, 0);
    try std.testing.expect(result != null);
}

test "strrchr - find last occurrence" {
    const s: [*:0]const u8 = "hello world";
    const result = strrchr(s, 'o');
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 'o'), result.?[0]);
    try std.testing.expectEqual(@as(u8, 'r'), result.?[1]);
}

test "strstr - find substring" {
    const haystack: [*:0]const u8 = "hello world";
    const needle: [*:0]const u8 = "world";
    const result = strstr(haystack, needle);
    try std.testing.expect(result != null);
    try std.testing.expectEqualStrings("world", std.mem.span(result.?));
}

test "strstr - not found" {
    const haystack: [*:0]const u8 = "hello";
    const needle: [*:0]const u8 = "xyz";
    try std.testing.expect(strstr(haystack, needle) == null);
}

test "strstr - empty needle" {
    const haystack: [*:0]const u8 = "hello";
    const needle: [*:0]const u8 = "";
    try std.testing.expect(strstr(haystack, needle) == haystack);
}

test "strtok - basic tokenization" {
    var s: [20:0]u8 = undefined;
    @memcpy(s[0..11], "a,b,c,d,e");
    s[11] = 0;
    const delim: [*:0]const u8 = ",";
    
    const t1 = strtok(&s, delim);
    try std.testing.expect(t1 != null);
    try std.testing.expectEqualStrings("a", std.mem.span(t1.?));
    
    const t2 = strtok(null, delim);
    try std.testing.expect(t2 != null);
    try std.testing.expectEqualStrings("b", std.mem.span(t2.?));
}

test "strtok_r - reentrant tokenization" {
    var s: [20:0]u8 = undefined;
    @memcpy(s[0..11], "a,b,c,d,e");
    s[11] = 0;
    const delim: [*:0]const u8 = ",";
    var saveptr: ?[*:0]u8 = null;
    
    const t1 = strtok_r(&s, delim, &saveptr);
    try std.testing.expect(t1 != null);
    try std.testing.expectEqualStrings("a", std.mem.span(t1.?));
    
    const t2 = strtok_r(null, delim, &saveptr);
    try std.testing.expect(t2 != null);
    try std.testing.expectEqualStrings("b", std.mem.span(t2.?));
}

test "strspn - span of accept chars" {
    const s: [*:0]const u8 = "abcXYZ";
    const accept: [*:0]const u8 = "abc";
    try std.testing.expectEqual(@as(usize, 3), strspn(s, accept));
}

test "strcspn - span until reject char" {
    const s: [*:0]const u8 = "abcXYZ";
    const reject: [*:0]const u8 = "XYZ";
    try std.testing.expectEqual(@as(usize, 3), strcspn(s, reject));
}

test "strpbrk - find any of set" {
    const s: [*:0]const u8 = "hello world";
    const accept: [*:0]const u8 = "aeiou";
    const result = strpbrk(s, accept);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 'e'), result.?[0]);
}

test "strpbrk - not found" {
    const s: [*:0]const u8 = "xyz";
    const accept: [*:0]const u8 = "abc";
    try std.testing.expect(strpbrk(s, accept) == null);
}

test "strnlen - within limit" {
    const s: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(usize, 5), strnlen(s, 10));
}

test "strnlen - at limit" {
    const s: [*:0]const u8 = "hello world";
    try std.testing.expectEqual(@as(usize, 5), strnlen(s, 5));
}

test "strcasecmp - equal ignoring case" {
    const s1: [*:0]const u8 = "Hello";
    const s2: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(i32, 0), strcasecmp(s1, s2));
}

test "strcasecmp - different" {
    const s1: [*:0]const u8 = "Hello";
    const s2: [*:0]const u8 = "World";
    try std.testing.expect(strcasecmp(s1, s2) < 0);
}

test "strncasecmp - equal within n" {
    const s1: [*:0]const u8 = "Hello";
    const s2: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(i32, 0), strncasecmp(s1, s2, 5));
}

test "strncasecmp - different case" {
    const s1: [*:0]const u8 = "HELLO";
    const s2: [*:0]const u8 = "hello";
    try std.testing.expectEqual(@as(i32, 0), strncasecmp(s1, s2, 5));
}

test "strcasestr - find case insensitive" {
    const haystack: [*:0]const u8 = "Hello World";
    const needle: [*:0]const u8 = "WORLD";
    const result = strcasestr(haystack, needle);
    try std.testing.expect(result != null);
}

test "strcasestr - not found" {
    const haystack: [*:0]const u8 = "hello";
    const needle: [*:0]const u8 = "xyz";
    try std.testing.expect(strcasestr(haystack, needle) == null);
}

// ============================================================================
// TODO: Phase 1.1 Remaining Functions
// ============================================================================

// Week 7-8: Implemented ✅
// - [x] strcat
// - [x] strncpy
// - [x] strncmp
// - [x] strncat

// Week 13-14: Implemented ✅
// - [x] strchr
// - [x] strrchr
// - [x] strstr
// - [x] strtok
// - [x] strtok_r
// - [x] strspn
// - [x] strcspn
// - [x] strpbrk

// Week 17-18: Implemented ✅
// - [x] strnlen
// - [x] strcasecmp
// - [x] strncasecmp
// - [x] strcasestr

// Phase 1.2+: To be implemented
// - [ ] strdup (needs allocator)
// - [ ] strndup (needs allocator)
// - [ ] strerror (needs error table)
// - [ ] strsignal (needs signal table)
