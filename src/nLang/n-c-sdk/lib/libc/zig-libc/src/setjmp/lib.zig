// setjmp module - Phase 1.6
const std = @import("std");

// Jump buffer type (architecture-specific)
pub const jmp_buf = extern struct {
    __jmpbuf: [8]c_long,
    __mask_was_saved: c_int,
    __saved_mask: [16]c_ulong,
};

pub const sigjmp_buf = jmp_buf;

// setjmp/longjmp functions
pub export fn setjmp(env: *jmp_buf) c_int {
    _ = env;
    return 0;
}

pub export fn longjmp(env: *jmp_buf, val: c_int) noreturn {
    _ = env;
    _ = val;
    @panic("longjmp called");
}

pub export fn _setjmp(env: *jmp_buf) c_int {
    _ = env;
    return 0;
}

pub export fn _longjmp(env: *jmp_buf, val: c_int) noreturn {
    _ = env;
    _ = val;
    @panic("_longjmp called");
}

pub export fn sigsetjmp(env: *sigjmp_buf, savemask: c_int) c_int {
    _ = env;
    _ = savemask;
    return 0;
}

pub export fn siglongjmp(env: *sigjmp_buf, val: c_int) noreturn {
    _ = env;
    _ = val;
    @panic("siglongjmp called");
}
