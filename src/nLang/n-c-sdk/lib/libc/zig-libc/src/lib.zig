// zig-libc - Pure Zig C Standard Library Implementation
// Phase 1.1: Foundation (Experimental)

const std = @import("std");
const config = @import("config");

// Module exports
pub const string = @import("string/lib.zig");
pub const stdio = @import("stdio/lib.zig");
pub const ctype = @import("ctype/lib.zig");
pub const math = @import("math/lib.zig");
pub const memory = @import("memory/lib.zig");
pub const stdlib = @import("stdlib/lib.zig");  // Phase 1.2 - Week 25
pub const errno = @import("errno/lib.zig");    // Phase 1.5
pub const assert_mod = @import("assert/lib.zig"); // Phase 1.5
pub const limits = @import("limits/lib.zig");  // Phase 1.5
pub const signal = @import("signal/lib.zig");  // Phase 1.6
pub const setjmp = @import("setjmp/lib.zig");  // Phase 1.6
pub const locale = @import("locale/lib.zig");  // Phase 1.7
pub const wchar = @import("wchar/lib.zig");    // Phase 1.7
pub const unistd = @import("unistd/lib.zig");  // Phase 1.8
pub const fcntl = @import("fcntl/lib.zig");    // Phase 1.9
pub const sys_stat = @import("sys_stat/lib.zig"); // Phase 1.9
pub const dirent = @import("dirent/lib.zig");  // Phase 1.10
pub const sys_time = @import("sys_time/lib.zig"); // Phase 1.10
pub const sys_select = @import("sys_select/lib.zig"); // Phase 1.10
pub const sys_mman = @import("sys_mman/lib.zig"); // Phase 1.11
pub const pthread = @import("pthread/lib.zig"); // Phase 1.11
pub const netinet_in = @import("netinet_in/lib.zig"); // Phase 1.12
pub const sys_socket = @import("sys_socket/lib.zig"); // Phase 1.12
pub const regex = @import("regex/lib.zig"); // Phase 1.12
pub const sys_utsname = @import("sys_utsname/lib.zig"); // Phase 1.13
pub const sys_wait = @import("sys_wait/lib.zig"); // Phase 1.13
pub const sys_resource = @import("sys_resource/lib.zig"); // Phase 1.13
pub const grp_pwd = @import("grp_pwd/lib.zig"); // Phase 1.13
pub const poll = @import("poll/lib.zig"); // Phase 1.14
pub const sys_ioctl = @import("sys_ioctl/lib.zig"); // Phase 1.14
pub const termios = @import("termios/lib.zig"); // Phase 1.14
pub const syslog = @import("syslog/lib.zig"); // Phase 1.15
pub const time_ext = @import("time_ext/lib.zig"); // Phase 1.15
pub const semaphore = @import("semaphore/lib.zig"); // Phase 1.15
pub const netdb = @import("netdb/lib.zig"); // Phase 1.16
pub const arpa_inet = @import("arpa_inet/lib.zig"); // Phase 1.16
pub const getopt = @import("getopt/lib.zig"); // Phase 1.16
pub const glob = @import("glob/lib.zig"); // Phase 1.17
pub const fnmatch = @import("fnmatch/lib.zig"); // Phase 1.17
pub const wordexp = @import("wordexp/lib.zig"); // Phase 1.17
pub const libgen = @import("libgen/lib.zig"); // Phase 1.17
pub const spawn = @import("spawn/lib.zig"); // Phase 1.18
pub const env = @import("env/lib.zig"); // Phase 1.18
pub const sched = @import("sched/lib.zig"); // Phase 1.18
pub const sys_ipc = @import("sys_ipc/lib.zig"); // Phase 1.19
pub const sys_msg = @import("sys_msg/lib.zig"); // Phase 1.19
pub const sys_shm = @import("sys_shm/lib.zig"); // Phase 1.19
pub const sys_sem = @import("sys_sem/lib.zig"); // Phase 1.19
pub const aio = @import("aio/lib.zig"); // Phase 1.20
pub const fts = @import("fts/lib.zig"); // Phase 1.20
pub const ftw = @import("ftw/lib.zig"); // Phase 1.20
pub const net_if = @import("net_if/lib.zig"); // Phase 1.21
pub const mqueue = @import("mqueue/lib.zig"); // Phase 1.21
pub const langinfo = @import("langinfo/lib.zig"); // Phase 1.21
pub const monetary = @import("monetary/lib.zig"); // Phase 1.21
pub const ucontext = @import("ucontext/lib.zig"); // Phase 1.22
pub const dlfcn = @import("dlfcn/lib.zig"); // Phase 1.22
pub const search = @import("search/lib.zig"); // Phase 1.22
pub const ulimit = @import("ulimit/lib.zig"); // Phase 1.23
pub const strings = @import("strings/lib.zig"); // Phase 1.23
pub const err = @import("err/lib.zig"); // Phase 1.23
pub const math2 = @import("math2/lib.zig"); // Phase 1.24
pub const iconv = @import("iconv/lib.zig"); // Phase 1.24
pub const stdio2 = @import("stdio2/lib.zig"); // Phase 1.25
pub const utime = @import("utime/lib.zig"); // Phase 1.25
pub const string2 = @import("string2/lib.zig"); // Phase 1.26
pub const stdlib2 = @import("stdlib2/lib.zig"); // Phase 1.26
pub const crypt = @import("crypt/lib.zig"); // Phase 1.27
pub const unistd2 = @import("unistd2/lib.zig"); // Phase 1.27
pub const pty = @import("pty/lib.zig"); // Phase 1.28
pub const sysinfo = @import("sysinfo/lib.zig"); // Phase 1.28
pub const shadow = @import("shadow/lib.zig"); // Phase 1.28
pub const sys_sysctl = @import("sys_sysctl/lib.zig"); // Phase 1.29
pub const sys_statvfs = @import("sys_statvfs/lib.zig"); // Phase 1.29
pub const sys_epoll = @import("sys_epoll/lib.zig"); // Phase 1.29
pub const sys_eventfd = @import("sys_eventfd/lib.zig"); // Phase 1.30
pub const sys_signalfd = @import("sys_signalfd/lib.zig"); // Phase 1.30
pub const sys_timerfd = @import("sys_timerfd/lib.zig"); // Phase 1.30
pub const sys_inotify = @import("sys_inotify/lib.zig"); // Phase 1.30
pub const sys_xattr = @import("sys_xattr/lib.zig"); // Phase 1.31
pub const sys_prctl = @import("sys_prctl/lib.zig"); // Phase 1.31
pub const sys_sendfile = @import("sys_sendfile/lib.zig"); // Phase 1.31
pub const resolv = @import("resolv/lib.zig"); // Phase 1.32
pub const mntent = @import("mntent/lib.zig"); // Phase 1.32
pub const fstab = @import("fstab/lib.zig"); // Phase 1.32
pub const utmp = @import("utmp/lib.zig"); // Phase 1.32
// pub const sys_time = @import("sys_time/lib.zig"); // Phase 1.10 (Disabled: Compilation Error)
// pub const regex = @import("regex/lib.zig"); // Phase 1.12 (Disabled: Compilation Error)
pub const bsd_string = @import("bsd_string/lib.zig"); // Phase 1.34
pub const sys_capability = @import("sys_capability/lib.zig"); // Phase 1.34
pub const sys_quota = @import("sys_quota/lib.zig"); // Phase 1.34
pub const obstack = @import("obstack/lib.zig"); // Phase 1.35
pub const argz = @import("argz/lib.zig"); // Phase 1.35
pub const envz = @import("envz/lib.zig"); // Phase 1.35
pub const petri = @import("petri/lib.zig"); // Phase 1.8 (Petri Net)
// pub const stdio = @import("stdio/lib.zig");    // Phase 1.2

// Version info
pub const version = .{
    .major = 0,
    .minor = 1,
    .patch = 0,
    .phase = "1.1",
};

// Feature flag check
pub const use_zig_libc = config.use_zig_libc;

// Test entry point
test "zig-libc basic" {
    try std.testing.expect(version.major == 0);
    try std.testing.expect(version.minor == 1);
}
