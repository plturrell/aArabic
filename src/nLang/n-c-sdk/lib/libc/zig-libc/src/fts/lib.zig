// fts module - Phase 1.20
const std = @import("std");

pub const FTS = opaque {};
pub const FTSENT = extern struct {
    fts_cycle: ?*FTSENT,
    fts_parent: ?*FTSENT,
    fts_link: ?*FTSENT,
    fts_number: c_long,
    fts_pointer: ?*anyopaque,
    fts_accpath: [*:0]u8,
    fts_path: [*:0]u8,
    fts_errno: c_int,
    fts_symfd: c_int,
    fts_pathlen: c_ushort,
    fts_namelen: c_ushort,
    fts_ino: u64,
    fts_dev: u64,
    fts_nlink: u64,
    fts_level: c_short,
    fts_info: c_ushort,
    fts_flags: c_ushort,
    fts_instr: c_ushort,
    fts_statp: ?*anyopaque,
    fts_name: [1]u8,
};

pub export fn fts_open(path_argv: [*:null]?[*:0]const u8, options: c_int, compar: ?*const fn (?*const ?*FTSENT, ?*const ?*FTSENT) callconv(.C) c_int) ?*FTS {
    _ = path_argv; _ = options; _ = compar;
    return @ptrFromInt(1);
}

pub export fn fts_read(ftsp: ?*FTS) ?*FTSENT {
    _ = ftsp;
    return null;
}

pub export fn fts_children(ftsp: ?*FTS, options: c_int) ?*FTSENT {
    _ = ftsp; _ = options;
    return null;
}

pub export fn fts_set(ftsp: ?*FTS, f: ?*FTSENT, options: c_int) c_int {
    _ = ftsp; _ = f; _ = options;
    return 0;
}

pub export fn fts_close(ftsp: ?*FTS) c_int {
    _ = ftsp;
    return 0;
}
