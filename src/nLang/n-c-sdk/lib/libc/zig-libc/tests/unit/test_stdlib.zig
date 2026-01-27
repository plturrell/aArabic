// Unit tests for stdlib module
// Phase 1.2 - Week 25
// Tests for malloc, free, calloc, realloc, atoi, atol, atof

const std = @import("std");
const testing = std.testing;
const zig_libc = @import("zig-libc");

// Memory allocation tests
test "malloc: basic allocation" {
    const ptr = zig_libc.stdlib.malloc(100);
    try testing.expect(ptr != null);
    // Note: free is not yet fully functional (documented limitation)
    zig_libc.stdlib.free(ptr);
}

test "malloc: zero size returns null" {
    const ptr = zig_libc.stdlib.malloc(0);
    try testing.expect(ptr == null);
}

test "calloc: basic allocation and initialization" {
    const ptr = zig_libc.stdlib.calloc(10, 10);
    try testing.expect(ptr != null);
    
    // Verify zeroed memory
    const bytes: [*]u8 = @ptrCast(ptr);
    for (0..100) |i| {
        try testing.expectEqual(@as(u8, 0), bytes[i]);
    }
    
    zig_libc.stdlib.free(ptr);
}

test "calloc: zero size returns null" {
    const ptr1 = zig_libc.stdlib.calloc(0, 10);
    try testing.expect(ptr1 == null);
    
    const ptr2 = zig_libc.stdlib.calloc(10, 0);
    try testing.expect(ptr2 == null);
}

test "free: null pointer is safe" {
    zig_libc.stdlib.free(null);
    // Should not crash
}

// String conversion tests
test "atoi: positive integer" {
    const result = zig_libc.stdlib.atoi("12345");
    try testing.expectEqual(@as(c_int, 12345), result);
}

test "atoi: negative integer" {
    const result = zig_libc.stdlib.atoi("-9876");
    try testing.expectEqual(@as(c_int, -9876), result);
}

test "atoi: with leading whitespace" {
    const result = zig_libc.stdlib.atoi("  42");
    try testing.expectEqual(@as(c_int, 42), result);
}

test "atoi: with plus sign" {
    const result = zig_libc.stdlib.atoi("+100");
    try testing.expectEqual(@as(c_int, 100), result);
}

test "atoi: stops at non-digit" {
    const result = zig_libc.stdlib.atoi("123abc");
    try testing.expectEqual(@as(c_int, 123), result);
}

test "atoi: empty string" {
    const result = zig_libc.stdlib.atoi("");
    try testing.expectEqual(@as(c_int, 0), result);
}

test "atol: positive long" {
    const result = zig_libc.stdlib.atol("1234567890");
    try testing.expectEqual(@as(c_long, 1234567890), result);
}

test "atol: negative long" {
    const result = zig_libc.stdlib.atol("-987654321");
    try testing.expectEqual(@as(c_long, -987654321), result);
}

test "atof: positive float" {
    const result = zig_libc.stdlib.atof("123.456");
    try testing.expectApproxEqRel(@as(f64, 123.456), result, 0.0001);
}

test "atof: negative float" {
    const result = zig_libc.stdlib.atof("-78.9");
    try testing.expectApproxEqRel(@as(f64, -78.9), result, 0.0001);
}

test "atof: integer part only" {
    const result = zig_libc.stdlib.atof("42");
    try testing.expectApproxEqRel(@as(f64, 42.0), result, 0.0001);
}

test "atof: decimal part only" {
    const result = zig_libc.stdlib.atof(".5");
    try testing.expectApproxEqRel(@as(f64, 0.5), result, 0.0001);
}

test "atof: with leading whitespace" {
    const result = zig_libc.stdlib.atof("  3.14");
    try testing.expectApproxEqRel(@as(f64, 3.14), result, 0.0001);
}

test "atof: empty string" {
    const result = zig_libc.stdlib.atof("");
    try testing.expectApproxEqRel(@as(f64, 0.0), result, 0.0001);
}

// Math function tests - Week 26
test "abs: positive number" {
    const result = zig_libc.stdlib.abs(42);
    try testing.expectEqual(@as(c_int, 42), result);
}

test "abs: negative number" {
    const result = zig_libc.stdlib.abs(-42);
    try testing.expectEqual(@as(c_int, 42), result);
}

test "abs: zero" {
    const result = zig_libc.stdlib.abs(0);
    try testing.expectEqual(@as(c_int, 0), result);
}

test "labs: positive long" {
    const result = zig_libc.stdlib.labs(123456);
    try testing.expectEqual(@as(c_long, 123456), result);
}

test "labs: negative long" {
    const result = zig_libc.stdlib.labs(-123456);
    try testing.expectEqual(@as(c_long, 123456), result);
}

test "llabs: long long" {
    const result = zig_libc.stdlib.llabs(-9876543210);
    try testing.expectEqual(@as(c_longlong, 9876543210), result);
}

test "div: positive numbers" {
    const result = zig_libc.stdlib.div(17, 5);
    try testing.expectEqual(@as(c_int, 3), result.quot);
    try testing.expectEqual(@as(c_int, 2), result.rem);
}

test "div: negative dividend" {
    const result = zig_libc.stdlib.div(-17, 5);
    try testing.expectEqual(@as(c_int, -3), result.quot);
    try testing.expectEqual(@as(c_int, -2), result.rem);
}

test "ldiv: long division" {
    const result = zig_libc.stdlib.ldiv(100, 7);
    try testing.expectEqual(@as(c_long, 14), result.quot);
    try testing.expectEqual(@as(c_long, 2), result.rem);
}

test "lldiv: long long division" {
    const result = zig_libc.stdlib.lldiv(1000000, 3);
    try testing.expectEqual(@as(c_longlong, 333333), result.quot);
    try testing.expectEqual(@as(c_longlong, 1), result.rem);
}

// Random number tests - Week 26
test "rand: generates numbers in range" {
    zig_libc.stdlib.srand(42);
    const r1 = zig_libc.stdlib.rand();
    try testing.expect(r1 >= 0);
    try testing.expect(r1 <= zig_libc.stdlib.RAND_MAX);
}

test "rand: deterministic with same seed" {
    zig_libc.stdlib.srand(12345);
    const r1 = zig_libc.stdlib.rand();
    const r2 = zig_libc.stdlib.rand();
    
    zig_libc.stdlib.srand(12345);
    const r3 = zig_libc.stdlib.rand();
    const r4 = zig_libc.stdlib.rand();
    
    try testing.expectEqual(r1, r3);
    try testing.expectEqual(r2, r4);
}

test "rand: different seeds produce different sequences" {
    zig_libc.stdlib.srand(1);
    const r1 = zig_libc.stdlib.rand();
    
    zig_libc.stdlib.srand(2);
    const r2 = zig_libc.stdlib.rand();
    
    try testing.expect(r1 != r2);
}

test "RAND_MAX: is 32767" {
    try testing.expectEqual(@as(c_int, 32767), zig_libc.stdlib.RAND_MAX);
}
