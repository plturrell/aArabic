// spawn module - Phase 1.18
const std = @import("std");

pub const posix_spawn_file_actions_t = extern struct {
    __allocated: c_int,
    __used: c_int,
    __actions: ?*anyopaque,
    __pad: [16]c_int,
};

pub const posix_spawnattr_t = extern struct {
    __flags: c_int,
    __pgrp: c_int,
    __sd: ?*anyopaque,
    __ss: ?*anyopaque,
    __sp: ?*anyopaque,
    __policy: c_int,
    __pad: [16]c_int,
};

pub export fn posix_spawn(pid: *c_int, path: [*:0]const u8, file_actions: ?*const posix_spawn_file_actions_t, attrp: ?*const posix_spawnattr_t, argv: [*:null]?[*:0]const u8, envp: [*:null]?[*:0]const u8) c_int {
    _ = path; _ = file_actions; _ = attrp; _ = argv; _ = envp;
    pid.* = 1000;
    return 0;
}

pub export fn posix_spawnp(pid: *c_int, file: [*:0]const u8, file_actions: ?*const posix_spawn_file_actions_t, attrp: ?*const posix_spawnattr_t, argv: [*:null]?[*:0]const u8, envp: [*:null]?[*:0]const u8) c_int {
    _ = file; _ = file_actions; _ = attrp; _ = argv; _ = envp;
    pid.* = 1001;
    return 0;
}

pub export fn posix_spawn_file_actions_init(file_actions: *posix_spawn_file_actions_t) c_int {
    @memset(std.mem.asBytes(file_actions), 0);
    return 0;
}

pub export fn posix_spawn_file_actions_destroy(file_actions: *posix_spawn_file_actions_t) c_int {
    _ = file_actions;
    return 0;
}

pub export fn posix_spawn_file_actions_addopen(file_actions: *posix_spawn_file_actions_t, fd: c_int, path: [*:0]const u8, oflag: c_int, mode: c_uint) c_int {
    _ = file_actions; _ = fd; _ = path; _ = oflag; _ = mode;
    return 0;
}

pub export fn posix_spawn_file_actions_addclose(file_actions: *posix_spawn_file_actions_t, fd: c_int) c_int {
    _ = file_actions; _ = fd;
    return 0;
}

pub export fn posix_spawn_file_actions_adddup2(file_actions: *posix_spawn_file_actions_t, fd: c_int, newfd: c_int) c_int {
    _ = file_actions; _ = fd; _ = newfd;
    return 0;
}

pub export fn posix_spawnattr_init(attr: *posix_spawnattr_t) c_int {
    @memset(std.mem.asBytes(attr), 0);
    return 0;
}

pub export fn posix_spawnattr_destroy(attr: *posix_spawnattr_t) c_int {
    _ = attr;
    return 0;
}

pub export fn posix_spawnattr_setflags(attr: *posix_spawnattr_t, flags: c_short) c_int {
    _ = attr; _ = flags;
    return 0;
}

pub export fn posix_spawnattr_getflags(attr: *const posix_spawnattr_t, flags: *c_short) c_int {
    _ = attr;
    flags.* = 0;
    return 0;
}

pub export fn posix_spawnattr_setpgroup(attr: *posix_spawnattr_t, pgroup: c_int) c_int {
    _ = attr; _ = pgroup;
    return 0;
}

pub export fn posix_spawnattr_getpgroup(attr: *const posix_spawnattr_t, pgroup: *c_int) c_int {
    _ = attr;
    pgroup.* = 0;
    return 0;
}
