// math2 module - Extended math functions - Phase 1.24
const std = @import("std");
const m = std.math;

pub export fn cbrt(x: f64) f64 {
    return m.cbrt(x);
}

pub export fn cbrtf(x: f32) f32 {
    return m.cbrt(x);
}

pub export fn hypot(x: f64, y: f64) f64 {
    return m.hypot(x, y);
}

pub export fn hypotf(x: f32, y: f32) f32 {
    return m.hypot(x, y);
}

pub export fn erf(x: f64) f64 {
    _ = x;
    return 0.0;
}

pub export fn erff(x: f32) f32 {
    _ = x;
    return 0.0;
}

pub export fn erfc(x: f64) f64 {
    _ = x;
    return 1.0;
}

pub export fn erfcf(x: f32) f32 {
    _ = x;
    return 1.0;
}

pub export fn lgamma(x: f64) f64 {
    _ = x;
    return 0.0;
}

pub export fn lgammaf(x: f32) f32 {
    _ = x;
    return 0.0;
}

pub export fn tgamma(x: f64) f64 {
    _ = x;
    return 1.0;
}

pub export fn tgammaf(x: f32) f32 {
    _ = x;
    return 1.0;
}

pub export fn remainder(x: f64, y: f64) f64 {
    return @rem(x, y);
}

pub export fn remainderf(x: f32, y: f32) f32 {
    return @rem(x, y);
}

pub export fn remquo(x: f64, y: f64, quo: *c_int) f64 {
    quo.* = @intFromFloat(@divTrunc(x, y));
    return @rem(x, y);
}

pub export fn remquof(x: f32, y: f32, quo: *c_int) f32 {
    quo.* = @intFromFloat(@divTrunc(x, y));
    return @rem(x, y);
}

pub export fn fdim(x: f64, y: f64) f64 {
    return if (x > y) x - y else 0.0;
}

pub export fn fdimf(x: f32, y: f32) f32 {
    return if (x > y) x - y else 0.0;
}

pub export fn fmax(x: f64, y: f64) f64 {
    return @max(x, y);
}

pub export fn fmaxf(x: f32, y: f32) f32 {
    return @max(x, y);
}

pub export fn fmin(x: f64, y: f64) f64 {
    return @min(x, y);
}

pub export fn fminf(x: f32, y: f32) f32 {
    return @min(x, y);
}

pub export fn fma(x: f64, y: f64, z: f64) f64 {
    return @mulAdd(f64, x, y, z);
}

pub export fn fmaf(x: f32, y: f32, z: f32) f32 {
    return @mulAdd(f32, x, y, z);
}

pub export fn nan(tag: [*:0]const u8) f64 {
    _ = tag;
    return m.nan(f64);
}

pub export fn nanf(tag: [*:0]const u8) f32 {
    _ = tag;
    return m.nan(f32);
}
