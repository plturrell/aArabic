// grp and pwd modules - Phase 1.13
const std = @import("std");

// Group structure
pub const group = extern struct {
    gr_name: [*:0]u8,
    gr_passwd: [*:0]u8,
    gr_gid: c_uint,
    gr_mem: [*:null]?[*:0]u8,
};

var static_group: group = .{
    .gr_name = @constCast("root"),
    .gr_passwd = @constCast("x"),
    .gr_gid = 0,
    .gr_mem = @constCast(&[_:null]?[*:0]u8{null}),
};

// Password structure
pub const passwd = extern struct {
    pw_name: [*:0]u8,
    pw_passwd: [*:0]u8,
    pw_uid: c_uint,
    pw_gid: c_uint,
    pw_gecos: [*:0]u8,
    pw_dir: [*:0]u8,
    pw_shell: [*:0]u8,
};

var static_passwd: passwd = .{
    .pw_name = @constCast("root"),
    .pw_passwd = @constCast("x"),
    .pw_uid = 0,
    .pw_gid = 0,
    .pw_gecos = @constCast("root"),
    .pw_dir = @constCast("/root"),
    .pw_shell = @constCast("/bin/sh"),
};

// Group functions
pub export fn getgrgid(gid: c_uint) ?*group {
    _ = gid;
    return &static_group;
}

pub export fn getgrnam(name: [*:0]const u8) ?*group {
    _ = name;
    return &static_group;
}

pub export fn getgrent() ?*group {
    return null;
}

pub export fn setgrent() void {}

pub export fn endgrent() void {}

// Password functions
pub export fn getpwuid(uid: c_uint) ?*passwd {
    _ = uid;
    return &static_passwd;
}

pub export fn getpwnam(name: [*:0]const u8) ?*passwd {
    _ = name;
    return &static_passwd;
}

pub export fn getpwent() ?*passwd {
    return null;
}

pub export fn setpwent() void {}

pub export fn endpwent() void {}
