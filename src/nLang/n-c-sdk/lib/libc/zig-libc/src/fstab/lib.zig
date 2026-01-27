// fstab module - Filesystem table - Phase 1.32
const std = @import("std");

pub const fstab = extern struct {
    fs_spec: [*:0]u8,
    fs_file: [*:0]u8,
    fs_vfstype: [*:0]u8,
    fs_mntops: [*:0]u8,
    fs_type: [*:0]u8,
    fs_freq: c_int,
    fs_passno: c_int,
};

pub export fn getfsent() ?*fstab {
    return null;
}

pub export fn getfsspec(spec: [*:0]const u8) ?*fstab {
    _ = spec;
    return null;
}

pub export fn getfsfile(file: [*:0]const u8) ?*fstab {
    _ = file;
    return null;
}

pub export fn setfsent() c_int {
    return 0;
}

pub export fn endfsent() void {}
