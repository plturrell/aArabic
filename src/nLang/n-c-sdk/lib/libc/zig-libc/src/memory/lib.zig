// Memory operations module
// Phase 1.1: Foundation - Memory functions

const std = @import("std");

// ============================================================================
// Phase 1.1 Functions (Month 3, Week 9-10)
// ============================================================================

/// Copy memory from src to dest
/// POSIX: void *memcpy(void *dest, const void *src, size_t n);
/// WARNING: Undefined behavior if regions overlap - use memmove for that
pub fn memcpy(dest: [*]u8, src: [*]const u8, n: usize) [*]u8 {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        dest[i] = src[i];
    }
    return dest;
}

/// Fill memory with a constant byte
/// POSIX: void *memset(void *s, int c, size_t n);
pub fn memset(s: [*]u8, c: i32, n: usize) [*]u8 {
    const byte = @as(u8, @intCast(c & 0xFF));
    var i: usize = 0;
    while (i < n) : (i += 1) {
        s[i] = byte;
    }
    return s;
}

/// Compare two memory blocks
/// POSIX: int memcmp(const void *s1, const void *s2, size_t n);
/// Returns: <0 if s1 < s2, 0 if s1 == s2, >0 if s1 > s2
pub fn memcmp(s1: [*]const u8, s2: [*]const u8, n: usize) i32 {
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (s1[i] != s2[i]) {
            return @as(i32, s1[i]) - @as(i32, s2[i]);
        }
    }
    return 0;
}

/// Copy memory, handling overlapping regions correctly
/// POSIX: void *memmove(void *dest, const void *src, size_t n);
pub fn memmove(dest: [*]u8, src: [*]const u8, n: usize) [*]u8 {
    if (n == 0) return dest;
    
    // Check if regions overlap
    const dest_addr = @intFromPtr(dest);
    const src_addr = @intFromPtr(src);
    
    if (dest_addr == src_addr) {
        return dest;
    } else if (dest_addr < src_addr) {
        // Copy forward
        var i: usize = 0;
        while (i < n) : (i += 1) {
            dest[i] = src[i];
        }
    } else {
        // Copy backward to handle overlap
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            dest[i] = src[i];
        }
    }
    return dest;
}

/// Find first occurrence of byte in memory
/// POSIX: void *memchr(const void *s, int c, size_t n);
pub fn memchr(s: [*]const u8, c: i32, n: usize) ?[*]const u8 {
    const target = @as(u8, @intCast(c & 0xFF));
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (s[i] == target) {
            return s + i;
        }
    }
    return null;
}

/// Find last occurrence of byte in memory (GNU extension)
/// Returns pointer to last occurrence of c in first n bytes of s
pub fn memrchr(s: [*]const u8, c: i32, n: usize) ?[*]const u8 {
    const target = @as(u8, @intCast(c & 0xFF));
    if (n == 0) return null;
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        if (s[i] == target) {
            return s + i;
        }
    }
    return null;
}

/// Find substring in memory (GNU extension)
/// Returns pointer to first occurrence of needle in haystack
pub fn memmem(haystack: [*]const u8, haystack_len: usize, needle: [*]const u8, needle_len: usize) ?[*]const u8 {
    if (needle_len == 0) return haystack;
    if (needle_len > haystack_len) return null;
    
    const search_len = haystack_len - needle_len + 1;
    var i: usize = 0;
    while (i < search_len) : (i += 1) {
        var j: usize = 0;
        while (j < needle_len and haystack[i + j] == needle[j]) : (j += 1) {}
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

test "memcpy - basic copy" {
    var dest: [10]u8 = undefined;
    const src = [_]u8{ 1, 2, 3, 4, 5 };
    _ = memcpy(&dest, &src, 5);
    try std.testing.expectEqualSlices(u8, &src, dest[0..5]);
}

test "memcpy - zero length" {
    var dest: [5]u8 = undefined;
    const src = [_]u8{ 1, 2, 3, 4, 5 };
    _ = memcpy(&dest, &src, 0);
    // No changes expected, just shouldn't crash
}

test "memset - fill with byte" {
    var buf: [10]u8 = undefined;
    _ = memset(&buf, 0x42, 10);
    for (buf) |byte| {
        try std.testing.expectEqual(@as(u8, 0x42), byte);
    }
}

test "memset - fill with zero" {
    var buf: [10]u8 = undefined;
    _ = memset(&buf, 0, 10);
    for (buf) |byte| {
        try std.testing.expectEqual(@as(u8, 0), byte);
    }
}

test "memcmp - equal blocks" {
    const s1 = [_]u8{ 1, 2, 3, 4, 5 };
    const s2 = [_]u8{ 1, 2, 3, 4, 5 };
    try std.testing.expectEqual(@as(i32, 0), memcmp(&s1, &s2, 5));
}

test "memcmp - different blocks" {
    const s1 = [_]u8{ 1, 2, 3, 4, 5 };
    const s2 = [_]u8{ 1, 2, 4, 4, 5 };
    try std.testing.expect(memcmp(&s1, &s2, 5) < 0);
}

test "memcmp - zero length" {
    const s1 = [_]u8{ 1, 2, 3 };
    const s2 = [_]u8{ 9, 9, 9 };
    try std.testing.expectEqual(@as(i32, 0), memcmp(&s1, &s2, 0));
}

test "memmove - non-overlapping" {
    var buf: [10]u8 = undefined;
    const src = [_]u8{ 1, 2, 3, 4, 5 };
    _ = memmove(&buf, &src, 5);
    try std.testing.expectEqualSlices(u8, &src, buf[0..5]);
}

test "memmove - overlapping forward" {
    var buf = [_]u8{ 1, 2, 3, 4, 5, 0, 0, 0 };
    _ = memmove(buf[2..].ptr, &buf, 5);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 1, 2, 3, 4, 5, 0 }, &buf);
}

test "memmove - overlapping backward" {
    var buf = [_]u8{ 0, 0, 0, 1, 2, 3, 4, 5 };
    _ = memmove(&buf, buf[3..].ptr, 5);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5, 3, 4, 5 }, &buf);
}

test "memmove - same location" {
    var buf = [_]u8{ 1, 2, 3, 4, 5 };
    _ = memmove(&buf, &buf, 5);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 1, 2, 3, 4, 5 }, &buf);
}

test "memchr - find byte" {
    const buf = [_]u8{ 1, 2, 3, 4, 5 };
    const result = memchr(&buf, 3, 5);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 3), result.?[0]);
}

test "memchr - not found" {
    const buf = [_]u8{ 1, 2, 3, 4, 5 };
    try std.testing.expect(memchr(&buf, 9, 5) == null);
}

test "memchr - zero length" {
    const buf = [_]u8{ 1, 2, 3 };
    try std.testing.expect(memchr(&buf, 1, 0) == null);
}

test "memrchr - find last occurrence" {
    const buf = [_]u8{ 1, 2, 3, 2, 5 };
    const result = memrchr(&buf, 2, 5);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 2), result.?[0]);
    try std.testing.expectEqual(@as(u8, 5), result.?[1]);
}

test "memrchr - not found" {
    const buf = [_]u8{ 1, 2, 3, 4, 5 };
    try std.testing.expect(memrchr(&buf, 9, 5) == null);
}

test "memrchr - zero length" {
    const buf = [_]u8{ 1, 2, 3 };
    try std.testing.expect(memrchr(&buf, 1, 0) == null);
}

test "memmem - find sequence" {
    const haystack = [_]u8{ 1, 2, 3, 4, 5, 6 };
    const needle = [_]u8{ 3, 4, 5 };
    const result = memmem(&haystack, 6, &needle, 3);
    try std.testing.expect(result != null);
    try std.testing.expectEqual(@as(u8, 3), result.?[0]);
}

test "memmem - not found" {
    const haystack = [_]u8{ 1, 2, 3, 4, 5 };
    const needle = [_]u8{ 6, 7, 8 };
    try std.testing.expect(memmem(&haystack, 5, &needle, 3) == null);
}

test "memmem - empty needle" {
    const haystack = [_]u8{ 1, 2, 3 };
    const needle = [_]u8{};
    try std.testing.expect(memmem(&haystack, 3, &needle, 0) == &haystack);
}

test "memmem - needle larger than haystack" {
    const haystack = [_]u8{ 1, 2 };
    const needle = [_]u8{ 1, 2, 3 };
    try std.testing.expect(memmem(&haystack, 2, &needle, 3) == null);
}

// ============================================================================
// TODO: Additional Memory Functions (Future phases)
// ============================================================================

// Phase 1.1: Implemented ✅
// - [x] memcpy  - Copy memory block
// - [x] memset  - Fill memory with byte  
// - [x] memcmp  - Compare memory blocks
// - [x] memmove - Copy overlapping memory
// - [x] memchr  - Find byte in memory
// - [x] memrchr - Find byte (reverse)
// - [x] memmem  - Find sequence in memory

// Phase 1.2+:
// - [ ] mempcpy - Copy memory, return end pointer
// - [ ] memccpy - Copy until character found
// - [ ] rawmemchr - Find byte (no length limit)
