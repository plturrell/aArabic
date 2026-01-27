// Math functions for stdlib
// Phase 1.2 - Week 26
// Implements abs, labs, llabs, div, ldiv, lldiv

const std = @import("std");

/// Compute absolute value of an integer
/// C signature: int abs(int j);
pub export fn abs(j: c_int) c_int {
    if (j < 0) {
        return -j;
    }
    return j;
}

/// Compute absolute value of a long integer
/// C signature: long labs(long j);
pub export fn labs(j: c_long) c_long {
    if (j < 0) {
        return -j;
    }
    return j;
}

/// Compute absolute value of a long long integer
/// C signature: long long llabs(long long j);
pub export fn llabs(j: c_longlong) c_longlong {
    if (j < 0) {
        return -j;
    }
    return j;
}

/// div_t structure for integer division result
pub const div_t = extern struct {
    quot: c_int,  // Quotient
    rem: c_int,   // Remainder
};

/// Compute quotient and remainder of integer division
/// C signature: div_t div(int numer, int denom);
pub export fn div(numer: c_int, denom: c_int) div_t {
    return div_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}

/// ldiv_t structure for long integer division result
pub const ldiv_t = extern struct {
    quot: c_long,
    rem: c_long,
};

/// Compute quotient and remainder of long integer division
/// C signature: ldiv_t ldiv(long numer, long denom);
pub export fn ldiv(numer: c_long, denom: c_long) ldiv_t {
    return ldiv_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}

/// lldiv_t structure for long long integer division result
pub const lldiv_t = extern struct {
    quot: c_longlong,
    rem: c_longlong,
};

/// Compute quotient and remainder of long long integer division
/// C signature: lldiv_t lldiv(long long numer, long long denom);
pub export fn lldiv(numer: c_longlong, denom: c_longlong) lldiv_t {
    return lldiv_t{
        .quot = @divTrunc(numer, denom),
        .rem = @rem(numer, denom),
    };
}
