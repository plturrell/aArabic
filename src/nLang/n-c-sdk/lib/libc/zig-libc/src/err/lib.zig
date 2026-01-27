// err module - Phase 1.23
const std = @import("std");

pub export fn err(eval: c_int, fmt: [*:0]const u8, ...) noreturn {
    _ = fmt;
    std.process.exit(@intCast(eval));
}

pub export fn verr(eval: c_int, fmt: [*:0]const u8, ap: *anyopaque) noreturn {
    _ = fmt; _ = ap;
    std.process.exit(@intCast(eval));
}

pub export fn errx(eval: c_int, fmt: [*:0]const u8, ...) noreturn {
    _ = fmt;
    std.process.exit(@intCast(eval));
}

pub export fn verrx(eval: c_int, fmt: [*:0]const u8, ap: *anyopaque) noreturn {
    _ = fmt; _ = ap;
    std.process.exit(@intCast(eval));
}

pub export fn warn(fmt: [*:0]const u8, ...) void {
    _ = fmt;
}

pub export fn vwarn(fmt: [*:0]const u8, ap: *anyopaque) void {
    _ = fmt; _ = ap;
}

pub export fn warnx(fmt: [*:0]const u8, ...) void {
    _ = fmt;
}

pub export fn vwarnx(fmt: [*:0]const u8, ap: *anyopaque) void {
    _ = fmt; _ = ap;
}
