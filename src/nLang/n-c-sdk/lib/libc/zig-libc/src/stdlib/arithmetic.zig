// Integer arithmetic functions
const std = @import("std");
const sort_impl = @import("sort.zig");

// === Absolute Value Functions ===

/// Absolute value of int
pub export fn abs(n: c_int) c_int {
    return if (n < 0) -n else n;
}

/// Absolute value of long
pub export fn labs(n: c_long) c_long {
    return if (n < 0) -n else n;
}

/// Absolute value of long long
pub export fn llabs(n: c_longlong) c_longlong {
    return if (n < 0) -n else n;
}

// === Division Functions ===

pub const div_t = extern struct {
    quot: c_int,
    rem: c_int,
};

pub const ldiv_t = extern struct {
    quot: c_long,
    rem: c_long,
};

pub const lldiv_t = extern struct {
    quot: c_longlong,
    rem: c_longlong,
};

/// Integer division (returns quotient and remainder)
pub export fn div(numer: c_int, denom: c_int) div_t {
    return div_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}

/// Long integer division
pub export fn ldiv(numer: c_long, denom: c_long) ldiv_t {
    return ldiv_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}

/// Long long integer division
pub export fn lldiv(numer: c_longlong, denom: c_longlong) lldiv_t {
    return lldiv_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}

// === Comparison Functions ===
// Re-export from sort.zig to avoid duplication (proper quicksort implementation)

pub const CompareFn = sort_impl.CompareFn;
pub const qsort = sort_impl.qsort;
pub const bsearch = sort_impl.bsearch;

// === Multi-byte/Wide Character Functions ===

/// Get length of multibyte character
pub export fn mblen(s: ?[*:0]const u8, n: usize) c_int {
    _ = n;
    if (s == null) return 0; // No state-dependent encoding
    const ptr = s.?;
    if (ptr[0] == 0) return 0;
    // Simplified: assume UTF-8
    if (ptr[0] & 0x80 == 0) return 1;
    if (ptr[0] & 0xE0 == 0xC0) return 2;
    if (ptr[0] & 0xF0 == 0xE0) return 3;
    if (ptr[0] & 0xF8 == 0xF0) return 4;
    return -1;
}

/// Convert multibyte to wide character
pub export fn mbtowc(pwc: ?*u32, s: ?[*:0]const u8, n: usize) c_int {
    _ = pwc;
    if (s == null) return 0;
    return mblen(s, n);
}

/// Convert wide character to multibyte
pub export fn wctomb(s: ?[*:0]u8, wchar: u32) c_int {
    if (s == null) return 0;
    const ptr = s.?;
    
    // Simplified UTF-8 encoding
    if (wchar < 0x80) {
        ptr[0] = @intCast(wchar);
        return 1;
    } else if (wchar < 0x800) {
        ptr[0] = @intCast(0xC0 | (wchar >> 6));
        ptr[1] = @intCast(0x80 | (wchar & 0x3F));
        return 2;
    } else if (wchar < 0x10000) {
        ptr[0] = @intCast(0xE0 | (wchar >> 12));
        ptr[1] = @intCast(0x80 | ((wchar >> 6) & 0x3F));
        ptr[2] = @intCast(0x80 | (wchar & 0x3F));
        return 3;
    } else if (wchar < 0x110000) {
        ptr[0] = @intCast(0xF0 | (wchar >> 18));
        ptr[1] = @intCast(0x80 | ((wchar >> 12) & 0x3F));
        ptr[2] = @intCast(0x80 | ((wchar >> 6) & 0x3F));
        ptr[3] = @intCast(0x80 | (wchar & 0x3F));
        return 4;
    }
    
    return -1;
}

/// Convert multibyte string to wide character string
pub export fn mbstowcs(pwcs: ?[*]u32, s: ?[*:0]const u8, n: usize) usize {
    if (pwcs == null or s == null) return 0;
    
    const src = s.?;
    const dst = pwcs.?;
    var i: usize = 0;
    var j: usize = 0;
    
    while (i < n and src[j] != 0) {
        const len = mblen(@ptrCast(&src[j]), 6);
        if (len <= 0) break;
        
        // Simplified: just copy byte value
        dst[i] = src[j];
        i += 1;
        j += @intCast(len);
    }
    
    return i;
}

/// Convert wide character string to multibyte string
pub export fn wcstombs(s: ?[*:0]u8, pwcs: ?[*]const u32, n: usize) usize {
    if (s == null or pwcs == null) return 0;
    
    const src = pwcs.?;
    const dst = s.?;
    var i: usize = 0;
    var j: usize = 0;
    
    while (j < n and src[i] != 0) {
        // Simplified: just copy byte value
        if (j + 1 >= n) break;
        dst[j] = @intCast(src[i] & 0xFF);
        i += 1;
        j += 1;
    }
    
    if (j < n) dst[j] = 0;
    return j;
}
