// ucontext module - Phase 1.22
const std = @import("std");

pub const ucontext_t = extern struct {
    uc_flags: c_ulong,
    uc_link: ?*ucontext_t,
    uc_stack: extern struct {
        ss_sp: ?*anyopaque,
        ss_flags: c_int,
        ss_size: usize,
    },
    uc_mcontext: [256]u8,
    uc_sigmask: [128]u8,
};

pub export fn getcontext(ucp: *ucontext_t) c_int {
    @memset(std.mem.asBytes(ucp), 0);
    return 0;
}

pub export fn setcontext(ucp: *const ucontext_t) c_int {
    _ = ucp;
    return 0;
}

pub export fn makecontext(ucp: *ucontext_t, func: *const fn () callconv(.C) void, argc: c_int, ...) void {
    _ = ucp; _ = func; _ = argc;
}

pub export fn swapcontext(oucp: *ucontext_t, ucp: *const ucontext_t) c_int {
    _ = oucp; _ = ucp;
    return 0;
}
