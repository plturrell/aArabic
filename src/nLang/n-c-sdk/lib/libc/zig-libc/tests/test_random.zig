const std = @import("std");
const testing = std.testing;
const libc = @import("zig-libc");

test "rand/rand_u32 deterministic with seed" {
    libc.stdlib.srandom(1234);
    const a1 = libc.stdlib.rand();
    const a2 = libc.stdlib.rand();

    libc.stdlib.srandom(1234);
    const b1 = libc.stdlib.rand();
    const b2 = libc.stdlib.rand();

    try testing.expectEqual(a1, b1);
    try testing.expectEqual(a2, b2);
}

test "uniform_u64 bounds" {
    const max: u64 = 10;
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const v = libc.stdlib.uniform_u64(max);
        try testing.expect(v <= max);
    }
}

test "normal mean near zero" {
    libc.stdlib.srandom(42);
    var sum: f64 = 0;
    const n: usize = 10_000;
    var i: usize = 0;
    while (i < n) : (i += 1) {
        sum += libc.stdlib.normal();
    }
    const mean = sum / @as(f64, @floatFromInt(n));
    const abs_mean = if (mean < 0) -mean else mean;
    try testing.expect(abs_mean < 0.1);
}

test "exponential positive" {
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const v = libc.stdlib.exponential(2.0);
        try testing.expect(v >= 0);
    }
}

test "gamma shape>1 produces positive" {
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const v = libc.stdlib.gamma(2.5, 1.0);
        try testing.expect(v > 0);
    }
}

test "poisson non-negative" {
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        const v = libc.stdlib.poisson(4.0);
        try testing.expect(v >= 0);
    }
}
