// String utility functions for stdlib
const std = @import("std");

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

/// Duplicate string
pub export fn strdup(s: [*:0]const u8) ?[*:0]u8 {
    const len = std.mem.len(s);
    const buf = allocator.alloc(u8, len + 1) catch return null;
    @memcpy(buf[0..len], s[0..len]);
    buf[len] = 0;
    return @ptrCast(buf.ptr);
}

/// Duplicate n characters
pub export fn strndup(s: [*:0]const u8, n: usize) ?[*:0]u8 {
    const len = @min(std.mem.len(s), n);
    const buf = allocator.alloc(u8, len + 1) catch return null;
    @memcpy(buf[0..len], s[0..len]);
    buf[len] = 0;
    return @ptrCast(buf.ptr);
}

/// Get error string
pub export fn strerror(errnum: c_int) [*:0]const u8 {
    return switch (errnum) {
        1 => "Operation not permitted",
        2 => "No such file or directory",
        3 => "No such process",
        4 => "Interrupted system call",
        5 => "Input/output error",
        12 => "Cannot allocate memory",
        13 => "Permission denied",
        22 => "Invalid argument",
        else => "Unknown error",
    };
}

/// Safe string copy
pub export fn strlcpy(dst: [*:0]u8, src: [*:0]const u8, size: usize) usize {
    if (size == 0) return std.mem.len(src);
    const src_len = std.mem.len(src);
    const copy_len = @min(src_len, size - 1);
    @memcpy(dst[0..copy_len], src[0..copy_len]);
    dst[copy_len] = 0;
    return src_len;
}

/// Safe string concatenate
pub export fn strlcat(dst: [*:0]u8, src: [*:0]const u8, size: usize) usize {
    const dst_len = std.mem.len(dst);
    const src_len = std.mem.len(src);
    if (dst_len >= size) return size + src_len;
    const copy_len = @min(src_len, size - dst_len - 1);
    @memcpy(dst[dst_len..][0..copy_len], src[0..copy_len]);
    dst[dst_len + copy_len] = 0;
    return dst_len + src_len;
}

// === Advanced String Search Functions ===

/// Find substring in string
pub export fn strstr(haystack: [*:0]const u8, needle: [*:0]const u8) ?[*:0]const u8 {
    const haystack_len = std.mem.len(haystack);
    const needle_len = std.mem.len(needle);
    
    if (needle_len == 0) return haystack;
    if (needle_len > haystack_len) return null;
    
    var i: usize = 0;
    while (i <= haystack_len - needle_len) : (i += 1) {
        var match = true;
        var j: usize = 0;
        while (j < needle_len) : (j += 1) {
            if (haystack[i + j] != needle[j]) {
                match = false;
                break;
            }
        }
        if (match) return @ptrCast(&haystack[i]);
    }
    return null;
}

/// Get length of prefix not containing reject chars
pub export fn strcspn(s: [*:0]const u8, reject: [*:0]const u8) usize {
    const s_len = std.mem.len(s);
    const reject_len = std.mem.len(reject);
    
    var i: usize = 0;
    while (i < s_len) : (i += 1) {
        var j: usize = 0;
        while (j < reject_len) : (j += 1) {
            if (s[i] == reject[j]) return i;
        }
    }
    return i;
}

/// Get length of prefix containing only accept chars
pub export fn strspn(s: [*:0]const u8, accept: [*:0]const u8) usize {
    const s_len = std.mem.len(s);
    const accept_len = std.mem.len(accept);
    
    var i: usize = 0;
    while (i < s_len) : (i += 1) {
        var found = false;
        var j: usize = 0;
        while (j < accept_len) : (j += 1) {
            if (s[i] == accept[j]) {
                found = true;
                break;
            }
        }
        if (!found) return i;
    }
    return i;
}

/// Find first occurrence of any char from accept in s
pub export fn strpbrk(s: [*:0]const u8, accept: [*:0]const u8) ?[*:0]const u8 {
    const s_len = std.mem.len(s);
    const accept_len = std.mem.len(accept);
    
    var i: usize = 0;
    while (i < s_len) : (i += 1) {
        var j: usize = 0;
        while (j < accept_len) : (j += 1) {
            if (s[i] == accept[j]) {
                return @ptrCast(&s[i]);
            }
        }
    }
    return null;
}

// === String Tokenization ===

// Thread-local state for strtok
threadlocal var strtok_state: ?[*:0]u8 = null;

/// Tokenize string (not thread-safe)
pub export fn strtok(s: ?[*:0]u8, delim: [*:0]const u8) ?[*:0]u8 {
    var str = s orelse strtok_state orelse return null;
    
    // Skip leading delimiters
    const skip = strspn(str, delim);
    str += skip;
    
    if (str[0] == 0) {
        strtok_state = null;
        return null;
    }
    
    // Find end of token
    const len = strcspn(str, delim);
    const token = str;
    
    if (str[len] == 0) {
        strtok_state = null;
    } else {
        str[len] = 0;
        strtok_state = str + len + 1;
    }
    
    return token;
}

/// Tokenize string (thread-safe, reentrant)
pub export fn strtok_r(s: ?[*:0]u8, delim: [*:0]const u8, saveptr: *?[*:0]u8) ?[*:0]u8 {
    var str = s orelse saveptr.* orelse return null;
    
    // Skip leading delimiters
    const skip = strspn(str, delim);
    str += skip;
    
    if (str[0] == 0) {
        saveptr.* = null;
        return null;
    }
    
    // Find end of token
    const len = strcspn(str, delim);
    const token = str;
    
    if (str[len] == 0) {
        saveptr.* = null;
    } else {
        str[len] = 0;
        saveptr.* = str + len + 1;
    }
    
    return token;
}

// === Memory Functions ===

/// Move memory region (handles overlapping regions)
pub export fn memmove(dest: ?*anyopaque, src: ?*const anyopaque, n: usize) ?*anyopaque {
    if (dest == null or src == null or n == 0) return dest;
    
    const d = @as([*]u8, @ptrCast(dest));
    const s = @as([*]const u8, @ptrCast(src));
    
    // Check for overlap and copy in appropriate direction
    if (@intFromPtr(d) < @intFromPtr(s)) {
        // Copy forward
        var i: usize = 0;
        while (i < n) : (i += 1) {
            d[i] = s[i];
        }
    } else if (@intFromPtr(d) > @intFromPtr(s)) {
        // Copy backward
        var i: usize = n;
        while (i > 0) {
            i -= 1;
            d[i] = s[i];
        }
    }
    
    return dest;
}

/// Find byte in memory
pub export fn memchr(s: ?*const anyopaque, c: c_int, n: usize) ?*anyopaque {
    if (s == null) return null;
    
    const bytes = @as([*]const u8, @ptrCast(s));
    const byte: u8 = @intCast(c & 0xFF);
    
    var i: usize = 0;
    while (i < n) : (i += 1) {
        if (bytes[i] == byte) {
            return @constCast(@as(*const anyopaque, @ptrCast(&bytes[i])));
        }
    }
    
    return null;
}

/// Find last occurrence of byte in memory
pub export fn memrchr(s: ?*const anyopaque, c: c_int, n: usize) ?*anyopaque {
    if (s == null or n == 0) return null;
    
    const bytes = @as([*]const u8, @ptrCast(s));
    const byte: u8 = @intCast(c & 0xFF);
    
    var i: usize = n;
    while (i > 0) {
        i -= 1;
        if (bytes[i] == byte) {
            return @constCast(@as(*const anyopaque, @ptrCast(&bytes[i])));
        }
    }
    
    return null;
}

/// Set memory to value
pub export fn memset(s: ?*anyopaque, c: c_int, n: usize) ?*anyopaque {
    if (s == null or n == 0) return s;
    
    const bytes = @as([*]u8, @ptrCast(s));
    const byte: u8 = @intCast(c & 0xFF);
    
    var i: usize = 0;
    while (i < n) : (i += 1) {
        bytes[i] = byte;
    }
    
    return s;
}

/// Find character in string
pub export fn strchr(s: [*:0]const u8, c: c_int) ?[*:0]const u8 {
    const char: u8 = @intCast(c & 0xFF);
    var i: usize = 0;
    
    while (true) {
        if (s[i] == char) return @ptrCast(&s[i]);
        if (s[i] == 0) break;
        i += 1;
    }
    
    // Check if searching for null terminator
    if (char == 0) return @ptrCast(&s[i]);
    return null;
}

/// Find last occurrence of character in string  
pub export fn strrchr(s: [*:0]const u8, c: c_int) ?[*:0]const u8 {
    const char: u8 = @intCast(c & 0xFF);
    const len = std.mem.len(s);
    var i: usize = len + 1; // Include null terminator
    
    while (i > 0) {
        i -= 1;
        if (s[i] == char) return @ptrCast(&s[i]);
    }
    
    return null;
}
