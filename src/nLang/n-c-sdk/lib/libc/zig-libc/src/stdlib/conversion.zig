// String conversion functions for stdlib
// Phase 1.2 - Week 25
// Implements atoi, atol, atoll, atof

const std = @import("std");

/// Convert string to integer
/// C signature: int atoi(const char *nptr);
pub export fn atoi(nptr: [*:0]const u8) c_int {
    if (nptr[0] == 0) {
        return 0;
    }
    
    var result: c_int = 0;
    var i: usize = 0;
    var negative = false;
    
    // Skip leading whitespace
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    // Handle sign
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    // Convert digits
    while (nptr[i] != 0 and isdigit(nptr[i])) : (i += 1) {
        const digit = nptr[i] - '0';
        result = result * 10 + @as(c_int, digit);
    }
    
    return if (negative) -result else result;
}

/// Convert string to double with error detection
/// C signature: double strtod(const char *nptr, char **endptr);
pub export fn strtod(nptr: [*:0]const u8, endptr: ?*?[*:0]u8) f64 {
    var i: usize = 0;
    var result: f64 = 0.0;
    var negative = false;
    var decimal_places: f64 = 0;
    var in_decimal = false;
    var exp_value: i32 = 0;
    var exp_negative = false;
    var saw_digit = false;
    
    // Skip whitespace
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    // Handle sign
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    // Parse integral and fractional parts
    while (nptr[i] != 0) : (i += 1) {
        if (isdigit(nptr[i])) {
            saw_digit = true;
            const digit = @as(f64, @floatFromInt(nptr[i] - '0'));
            if (in_decimal) {
                decimal_places += 1;
                result = result + digit / std.math.pow(f64, 10.0, decimal_places);
            } else {
                result = result * 10.0 + digit;
            }
        } else if (nptr[i] == '.' and !in_decimal) {
            in_decimal = true;
        } else {
            break;
        }
    }
    
    // Parse exponent only if a digit was seen
    if (saw_digit and (nptr[i] == 'e' or nptr[i] == 'E')) {
        var j = i + 1;
        if (nptr[j] == '-') {
            exp_negative = true;
            j += 1;
        } else if (nptr[j] == '+') {
            j += 1;
        }

        var exp_digits = false;
        while (nptr[j] != 0 and isdigit(nptr[j])) : (j += 1) {
            exp_digits = true;
            exp_value = exp_value * 10 + @as(i32, nptr[j] - '0');
        }

        if (exp_digits) {
            i = j;
        } else {
            exp_value = 0;
            exp_negative = false;
        }
    }
    
    if (endptr) |ptr| {
        ptr.* = if (!saw_digit)
            @ptrCast(@constCast(nptr))
        else
            @ptrCast(@constCast(&nptr[i]));
    }
    
    // Apply exponent
    if (exp_value != 0) {
        const exp_mult = std.math.pow(f64, 10.0, @as(f64, @floatFromInt(exp_value)));
        result = if (exp_negative) result / exp_mult else result * exp_mult;
    }
    
    return if (negative) -result else result;
}

/// Convert string to float with error detection
/// C signature: float strtof(const char *nptr, char **endptr);
pub export fn strtof(nptr: [*:0]const u8, endptr: ?*?[*:0]u8) f32 {
    return @floatCast(strtod(nptr, endptr));
}

/// Convert string to long
/// C signature: long atol(const char *nptr);
pub export fn atol(nptr: [*:0]const u8) c_long {
    if (nptr[0] == 0) {
        return 0;
    }
    
    var result: c_long = 0;
    var i: usize = 0;
    var negative = false;
    
    // Skip leading whitespace
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    // Handle sign
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    // Convert digits
    while (nptr[i] != 0 and isdigit(nptr[i])) : (i += 1) {
        const digit = nptr[i] - '0';
        result = result * 10 + @as(c_long, digit);
    }
    
    return if (negative) -result else result;
}

/// Convert string to double
/// C signature: double atof(const char *nptr);
pub export fn atof(nptr: [*:0]const u8) f64 {
    if (nptr[0] == 0) {
        return 0.0;
    }
    
    var result: f64 = 0.0;
    var i: usize = 0;
    var negative = false;
    var decimal_places: f64 = 0;
    var in_decimal = false;
    
    // Skip leading whitespace
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    // Handle sign
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    // Convert digits
    while (nptr[i] != 0) : (i += 1) {
        if (isdigit(nptr[i])) {
            const digit = @as(f64, @floatFromInt(nptr[i] - '0'));
            if (in_decimal) {
                decimal_places += 1;
                result = result + digit / std.math.pow(f64, 10.0, decimal_places);
            } else {
                result = result * 10.0 + digit;
            }
        } else if (nptr[i] == '.' and !in_decimal) {
            in_decimal = true;
        } else {
            break;
        }
    }
    
    return if (negative) -result else result;
}

// Helper functions
fn isspace(c: u8) bool {
    return c == ' ' or c == '\t' or c == '\n' or c == '\r' or c == '\x0b' or c == '\x0c';
}

fn isdigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

/// Convert string to long with error detection
/// C signature: long strtol(const char *nptr, char **endptr, int base);
pub export fn strtol(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) c_long {
    var i: usize = 0;
    var result: c_long = 0;
    var negative = false;
    var actual_base: c_int = base;
    
    // Skip whitespace
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    // Handle sign
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    // Auto-detect base if 0 - FIXED: Bounds checking
    if (actual_base == 0) {
        if (nptr[i] == '0' and nptr[i + 1] != 0) {
            if (nptr[i + 1] == 'x' or nptr[i + 1] == 'X') {
                actual_base = 16;
                i += 2;
            } else {
                actual_base = 8;
                i += 1;
            }
        } else {
            actual_base = 10;
        }
    } else if (actual_base == 16 and nptr[i] == '0' and nptr[i + 1] != 0 and (nptr[i + 1] == 'x' or nptr[i + 1] == 'X')) {
        i += 2;
    }
    
    const start_i = i;
    
    // Convert digits
    while (nptr[i] != 0) : (i += 1) {
        var digit: c_long = -1;
        
        if (nptr[i] >= '0' and nptr[i] <= '9') {
            digit = nptr[i] - '0';
        } else if (nptr[i] >= 'a' and nptr[i] <= 'z') {
            digit = nptr[i] - 'a' + 10;
        } else if (nptr[i] >= 'A' and nptr[i] <= 'Z') {
            digit = nptr[i] - 'A' + 10;
        }
        
        if (digit < 0 or digit >= actual_base) break;
        result = result * actual_base + digit;
    }
    
    if (endptr) |ptr| {
        if (i == start_i) {
            ptr.* = @ptrCast(@constCast(nptr));
        } else {
            ptr.* = @ptrCast(@constCast(&nptr[i]));
        }
    }
    
    return if (negative) -result else result;
}

/// Convert string to long long with error detection
/// C signature: long long strtoll(const char *nptr, char **endptr, int base);
pub export fn strtoll(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) c_longlong {
    var i: usize = 0;
    var result: c_longlong = 0;
    var negative = false;
    var actual_base: c_int = base;
    
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    if (actual_base == 0) {
        if (nptr[i] == '0' and nptr[i + 1] != 0) {
            if (nptr[i + 1] == 'x' or nptr[i + 1] == 'X') {
                actual_base = 16;
                i += 2;
            } else {
                actual_base = 8;
                i += 1;
            }
        } else {
            actual_base = 10;
        }
    } else if (actual_base == 16 and nptr[i] == '0' and nptr[i + 1] != 0 and (nptr[i + 1] == 'x' or nptr[i + 1] == 'X')) {
        i += 2;
    }
    
    const start_i = i;
    
    while (nptr[i] != 0) : (i += 1) {
        var digit: c_longlong = -1;
        
        if (nptr[i] >= '0' and nptr[i] <= '9') {
            digit = nptr[i] - '0';
        } else if (nptr[i] >= 'a' and nptr[i] <= 'z') {
            digit = nptr[i] - 'a' + 10;
        } else if (nptr[i] >= 'A' and nptr[i] <= 'Z') {
            digit = nptr[i] - 'A' + 10;
        }
        
        if (digit < 0 or digit >= actual_base) break;
        result = result * actual_base + digit;
    }
    
    if (endptr) |ptr| {
        if (i == start_i) {
            ptr.* = @ptrCast(@constCast(nptr));
        } else {
            ptr.* = @ptrCast(@constCast(&nptr[i]));
        }
    }
    
    return if (negative) -result else result;
}

/// Convert string to unsigned long
/// C signature: unsigned long strtoul(const char *nptr, char **endptr, int base);
pub export fn strtoul(nptr: [*:0]const u8, endptr: ?*?[*:0]u8, base: c_int) c_ulong {
    var i: usize = 0;
    var result: c_ulong = 0;
    var actual_base: c_int = base;
    
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    if (nptr[i] == '+') {
        i += 1;
    }
    
    if (actual_base == 0) {
        if (nptr[i] == '0' and nptr[i + 1] != 0) {
            if (nptr[i + 1] == 'x' or nptr[i + 1] == 'X') {
                actual_base = 16;
                i += 2;
            } else {
                actual_base = 8;
                i += 1;
            }
        } else {
            actual_base = 10;
        }
    } else if (actual_base == 16 and nptr[i] == '0' and nptr[i + 1] != 0 and (nptr[i + 1] == 'x' or nptr[i + 1] == 'X')) {
        i += 2;
    }
    
    const start_i = i;
    
    while (nptr[i] != 0) : (i += 1) {
        var digit: c_ulong = @as(c_ulong, @bitCast(@as(c_long, -1)));
        
        if (nptr[i] >= '0' and nptr[i] <= '9') {
            digit = nptr[i] - '0';
        } else if (nptr[i] >= 'a' and nptr[i] <= 'z') {
            digit = nptr[i] - 'a' + 10;
        } else if (nptr[i] >= 'A' and nptr[i] <= 'Z') {
            digit = nptr[i] - 'A' + 10;
        }
        
        if (digit >= @as(c_ulong, @intCast(actual_base))) break;
        result = result * @as(c_ulong, @intCast(actual_base)) + digit;
    }
    
    if (endptr) |ptr| {
        if (i == start_i) {
            ptr.* = @ptrCast(@constCast(nptr));
        } else {
            ptr.* = @ptrCast(@constCast(&nptr[i]));
        }
    }
    
    return result;
}

/// Convert string to long long (simple version)
/// C signature: long long atoll(const char *nptr);
pub export fn atoll(nptr: [*:0]const u8) c_longlong {
    if (nptr[0] == 0) return 0;
    
    var result: c_longlong = 0;
    var i: usize = 0;
    var negative = false;
    
    while (nptr[i] != 0 and isspace(nptr[i])) : (i += 1) {}
    
    if (nptr[i] == '-') {
        negative = true;
        i += 1;
    } else if (nptr[i] == '+') {
        i += 1;
    }
    
    while (nptr[i] != 0 and isdigit(nptr[i])) : (i += 1) {
        const digit = nptr[i] - '0';
        result = result * 10 + @as(c_longlong, digit);
    }
    
    return if (negative) -result else result;
}
