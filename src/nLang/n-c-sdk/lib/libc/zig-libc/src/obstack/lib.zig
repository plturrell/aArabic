// obstack module - Object stack allocation - Phase 1.35
const std = @import("std");

pub const obstack = extern struct {
    chunk_size: usize,
    chunk: ?*anyopaque,
    object_base: [*]u8,
    next_free: [*]u8,
    chunk_limit: [*]u8,
    temp: extern union {
        i: c_int,
        p: ?*anyopaque,
    },
    alignment_mask: c_int,
    use_extra_arg: c_int,
    maybe_empty_object: c_int,
    alloc_failed: c_int,
};

pub export fn _obstack_begin(h: *obstack, size: c_int, alignment: c_int, chunkfun: ?*const fn(usize) callconv(.C) ?*anyopaque, freefun: ?*const fn(?*anyopaque) callconv(.C) void) c_int {
    _ = h; _ = size; _ = alignment; _ = chunkfun; _ = freefun;
    return 1;
}

pub export fn _obstack_newchunk(h: *obstack, length: c_int) void {
    _ = h; _ = length;
}

pub export fn _obstack_free(h: *obstack, obj: ?*anyopaque) void {
    _ = h; _ = obj;
}

pub export fn _obstack_allocated_p(h: *obstack, obj: ?*anyopaque) c_int {
    _ = h; _ = obj;
    return 0;
}

pub export fn _obstack_memory_used(h: *obstack) usize {
    _ = h;
    return 0;
}

pub export fn obstack_printf(obstack_: *obstack, format: [*:0]const u8, ...) c_int {
    _ = obstack_; _ = format;
    return 0;
}

pub export fn obstack_vprintf(obstack_: *obstack, format: [*:0]const u8, args: *anyopaque) c_int {
    _ = obstack_; _ = format; _ = args;
    return 0;
}
