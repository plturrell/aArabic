// String Conversion Tests - Phase 2 Testing
// Tests for strtod, strtol, strtoll, strtoul
// Verifies bounds checking fixes for buffer overrun vulnerabilities

const std = @import("std");
const testing = std.testing;
const libc = @import("zig-libc");
const conversion = libc.stdlib.conversion;

test "strtod valid number" {
    const str: [*:0]const u8 = "123.456";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectApproxEqAbs(123.456, result, 0.001);
    
    // endptr should point to end
    if (endptr) |ptr| {
        try testing.expectEqual(@as(u8, 0), ptr[0]);
    }
}

test "strtod edge case - lone 'e'" {
    // BUG FIX TEST: "1e" should not crash
    const str: [*:0]const u8 = "1e";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    
    // Should parse "1" and stop at "e"
    try testing.expectEqual(@as(f64, 1.0), result);
}

test "strtod edge case - lone sign" {
    // BUG FIX TEST: "+" or "-" should not crash
    const plus: [*:0]const u8 = "+";
    const minus: [*:0]const u8 = "-";
    var endptr: ?[*:0]u8 = null;
    
    const r1 = conversion.strtod(plus, &endptr);
    try testing.expectEqual(@as(f64, 0.0), r1);
    
    const r2 = conversion.strtod(minus, &endptr);
    try testing.expectEqual(@as(f64, 0.0), r2);
}

test "strtod with exponent" {
    const str: [*:0]const u8 = "1.5e2";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectApproxEqAbs(150.0, result, 0.001);
}

test "strtod negative exponent" {
    const str: [*:0]const u8 = "5e-2";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectApproxEqAbs(0.05, result, 0.001);
}

test "strtol base 10" {
    const str: [*:0]const u8 = "12345";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 10);
    try testing.expectEqual(@as(c_long, 12345), result);
}

test "strtol base 16 with 0x prefix" {
    const str: [*:0]const u8 = "0x1A";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 16);
    try testing.expectEqual(@as(c_long, 26), result);
}

test "strtol edge case - lone 0x" {
    // BUG FIX TEST: "0x" should not crash
    const str: [*:0]const u8 = "0x";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 16);
    
    // Should parse "0" and stop
    try testing.expectEqual(@as(c_long, 0), result);
}

test "strtol auto-detect hex" {
    const str: [*:0]const u8 = "0xFF";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 0);
    try testing.expectEqual(@as(c_long, 255), result);
}

test "strtol auto-detect octal" {
    const str: [*:0]const u8 = "077";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 0);
    try testing.expectEqual(@as(c_long, 63), result); // 7*8 + 7
}

test "strtol negative number" {
    const str: [*:0]const u8 = "-42";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 10);
    try testing.expectEqual(@as(c_long, -42), result);
}

test "strtoll large number" {
    const str: [*:0]const u8 = "9223372036854775807"; // max i64
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtoll(str, &endptr, 10);
    try testing.expect(result > 0);
}

test "strtoll edge case - 0x without digits" {
    // BUG FIX TEST: "0x" should not crash
    const str: [*:0]const u8 = "0x";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtoll(str, &endptr, 0);
    try testing.expectEqual(@as(c_longlong, 0), result);
}

test "strtoul unsigned max" {
    const str: [*:0]const u8 = "18446744073709551615"; // max u64
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtoul(str, &endptr, 10);
    try testing.expect(result > 0);
}

test "strtoul hex" {
    const str: [*:0]const u8 = "0xDEADBEEF";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtoul(str, &endptr, 0);
    try testing.expectEqual(@as(c_ulong, 0xDEADBEEF), result);
}

test "atoi simple" {
    const str: [*:0]const u8 = "42";
    const result = conversion.atoi(str);
    try testing.expectEqual(@as(c_int, 42), result);
}

test "atoi with whitespace" {
    const str: [*:0]const u8 = "  123";
    const result = conversion.atoi(str);
    try testing.expectEqual(@as(c_int, 123), result);
}

test "atoi negative" {
    const str: [*:0]const u8 = "-99";
    const result = conversion.atoi(str);
    try testing.expectEqual(@as(c_int, -99), result);
}

test "atof simple" {
    const str: [*:0]const u8 = "3.14";
    const result = conversion.atof(str);
    try testing.expectApproxEqAbs(3.14, result, 0.01);
}

test "atof with sign" {
    const str: [*:0]const u8 = "-2.5";
    const result = conversion.atof(str);
    try testing.expectApproxEqAbs(-2.5, result, 0.01);
}

test "strtod empty string" {
    const str: [*:0]const u8 = "";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectEqual(@as(f64, 0.0), result);
}

test "strtol invalid base" {
    const str: [*:0]const u8 = "123";
    var endptr: ?[*:0]u8 = null;
    
    // Base 37 is invalid, should handle gracefully
    const result = conversion.strtol(str, &endptr, 37);
    _ = result; // Implementation dependent
}

test "strtod with decimal only" {
    const str: [*:0]const u8 = ".5";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectApproxEqAbs(0.5, result, 0.01);
}

test "strtod scientific notation positive" {
    const str: [*:0]const u8 = "1.5e+3";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtod(str, &endptr);
    try testing.expectApproxEqAbs(1500.0, result, 0.1);
}

test "strtol with trailing garbage" {
    const str: [*:0]const u8 = "123abc";
    var endptr: ?[*:0]u8 = null;
    
    const result = conversion.strtol(str, &endptr, 10);
    try testing.expectEqual(@as(c_long, 123), result);
    
    // endptr should point to 'a'
    if (endptr) |ptr| {
        try testing.expectEqual(@as(u8, 'a'), ptr[0]);
    }
}

test "bounds safety - edge case 0x" {
    const str: [*:0]const u8 = "0x";
    var endptr: ?[*:0]u8 = null;
    
    _ = conversion.strtod(str, &endptr);
    _ = conversion.strtol(str, &endptr, 0);
    _ = conversion.strtoll(str, &endptr, 0);
    _ = conversion.strtoul(str, &endptr, 0);
}

test "bounds safety - edge case 1e" {
    const str: [*:0]const u8 = "1e";
    var endptr: ?[*:0]u8 = null;
    
    _ = conversion.strtod(str, &endptr);
    _ = conversion.strtol(str, &endptr, 0);
    _ = conversion.strtoll(str, &endptr, 0);
    _ = conversion.strtoul(str, &endptr, 0);
}

test "bounds safety - edge case plus" {
    const str: [*:0]const u8 = "+";
    var endptr: ?[*:0]u8 = null;
    
    _ = conversion.strtod(str, &endptr);
    _ = conversion.strtol(str, &endptr, 0);
    _ = conversion.strtoll(str, &endptr, 0);
    _ = conversion.strtoul(str, &endptr, 0);
}

test "bounds safety - edge case minus" {
    const str: [*:0]const u8 = "-";
    var endptr: ?[*:0]u8 = null;
    
    _ = conversion.strtod(str, &endptr);
    _ = conversion.strtol(str, &endptr, 0);
    _ = conversion.strtoll(str, &endptr, 0);
}

test "bounds safety - edge case dot" {
    const str: [*:0]const u8 = ".";
    var endptr: ?[*:0]u8 = null;
    
    _ = conversion.strtod(str, &endptr);
}
