// assert module - Phase 1.5
const std = @import("std");

pub export fn __assert_fail(
    assertion: [*:0]const u8,
    file: [*:0]const u8,
    line: c_uint,
    function: [*:0]const u8,
) noreturn {
    _ = assertion;
    _ = file;
    _ = line;
    _ = function;
    @panic("assertion failed");
}

pub export fn __assert_perror_fail(
    errnum: c_int,
    file: [*:0]const u8,
    line: c_uint,
    function: [*:0]const u8,
) noreturn {
    _ = errnum;
    _ = file;
    _ = line;
    _ = function;
    @panic("assertion failed");
}

pub export fn __assert(
    assertion: [*:0]const u8,
    file: [*:0]const u8,
    line: c_uint,
) noreturn {
    _ = assertion;
    _ = file;
    _ = line;
    @panic("assertion failed");
}
