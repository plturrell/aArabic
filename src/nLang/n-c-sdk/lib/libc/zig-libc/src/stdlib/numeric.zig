// Numeric conversion functions
const std = @import("std");

fn isspace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == '\x0b' or c == '\x0c';
}

/// String to unsigned long long
pub export fn strtoull(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) c_ulonglong {
    var i: usize = 0;
    var result: c_ulonglong = 0;
    var actual_base: c_int = base;
    
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    if (nptr[i] == '+') i += 1;
    
    if (actual_base == 0) {
        if (nptr[i] == '0') {
            if (nptr[i + 1] != 0 and (nptr[i + 1] == 'x' or nptr[i + 1] == 'X')) {
                actual_base = 16;
                i += 2;
            } else {
                actual_base = 8;
                i += 1;
            }
        } else actual_base = 10;
    }
    
    const start_i = i;
    while (nptr[i] != 0) : (i += 1) {
        var digit: c_ulonglong = @as(c_ulonglong, @bitCast(@as(c_longlong, -1)));
        if (nptr[i] >= '0' and nptr[i] <= '9') {
            digit = nptr[i] - '0';
        } else if (nptr[i] >= 'a' and nptr[i] <= 'z') {
            digit = nptr[i] - 'a' + 10;
        } else if (nptr[i] >= 'A' and nptr[i] <= 'Z') {
            digit = nptr[i] - 'A' + 10;
        }
        if (digit >= @as(c_ulonglong, @intCast(actual_base))) break;
        result = result * @as(c_ulonglong, @intCast(actual_base)) + digit;
    }
    
    if (endptr) |ptr| {
        ptr.* = if (i == start_i) @ptrCast(@constCast(nptr)) else @ptrCast(@constCast(&nptr[i]));
    }
    return result;
}

/// String to long double
pub export fn strtold(nptr: [*:0]const u8, endptr: ?*?[*:0]u8) c_longdouble {
    var i: usize = 0;
    var result: c_longdouble = 0.0;
    var negative = false;
    var scale: c_longdouble = 1.0;
    var in_decimal = false;
    var saw_digit = false;

    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') i += 1;

    while (nptr[i] != 0) : (i += 1) {
        if (nptr[i] >= '0' and nptr[i] <= '9') {
            saw_digit = true;
            const digit = @as(c_longdouble, @floatFromInt(nptr[i] - '0'));
            if (in_decimal) {
                scale /= 10.0;
                result += digit * scale;
            } else {
                result = result * 10.0 + digit;
            }
        } else if (nptr[i] == '.' and !in_decimal) {
            in_decimal = true;
        } else break;
    }

    if (endptr) |ptr| {
        ptr.* = if (!saw_digit) @ptrCast(@constCast(nptr)) else @ptrCast(@constCast(&nptr[i]));
    }
    return if (negative) -result else result;
}
