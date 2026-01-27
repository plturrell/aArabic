// monetary module - Phase 1.21
const std = @import("std");

pub export fn strfmon(s: [*:0]u8, maxsize: usize, format: [*:0]const u8, ...) isize {
    _ = format;
    if (maxsize > 0) s[0] = 0;
    return 0;
}
