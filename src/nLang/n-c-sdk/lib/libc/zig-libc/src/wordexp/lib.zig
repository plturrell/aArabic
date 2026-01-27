// wordexp module - Phase 1.17
const std = @import("std");

pub const WRDE_APPEND: c_int = 1 << 0;
pub const WRDE_DOOFFS: c_int = 1 << 1;
pub const WRDE_NOCMD: c_int = 1 << 2;
pub const WRDE_REUSE: c_int = 1 << 3;
pub const WRDE_SHOWERR: c_int = 1 << 4;
pub const WRDE_UNDEF: c_int = 1 << 5;

pub const WRDE_BADCHAR: c_int = 1;
pub const WRDE_BADVAL: c_int = 2;
pub const WRDE_CMDSUB: c_int = 3;
pub const WRDE_NOSPACE: c_int = 4;
pub const WRDE_SYNTAX: c_int = 5;

pub const wordexp_t = extern struct {
    we_wordc: usize,
    we_wordv: [*:null]?[*:0]u8,
    we_offs: usize,
};

pub export fn wordexp(words: [*:0]const u8, pwordexp: *wordexp_t, flags: c_int) c_int {
    _ = words; _ = flags;
    pwordexp.we_wordc = 0;
    pwordexp.we_wordv = @ptrCast(&[_:null]?[*:0]u8{null});
    pwordexp.we_offs = 0;
    return 0;
}

pub export fn wordfree(pwordexp: *wordexp_t) void {
    pwordexp.we_wordc = 0;
}
