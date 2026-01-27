// sys/socket module - Phase 1.7 Priority 5 - Production Networking
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

// Socket address families
pub const AF_UNSPEC: c_int = 0;
pub const AF_UNIX: c_int = 1;
pub const AF_LOCAL: c_int = AF_UNIX;
pub const AF_INET: c_int = 2;
pub const AF_INET6: c_int = 10;

// Socket types
pub const SOCK_STREAM: c_int = 1;
pub const SOCK_DGRAM: c_int = 2;
pub const SOCK_RAW: c_int = 3;
pub const SOCK_SEQPACKET: c_int = 5;
pub const SOCK_NONBLOCK: c_int = 0o4000;
pub const SOCK_CLOEXEC: c_int = 0o2000000;

// Socket protocols
pub const IPPROTO_IP: c_int = 0;
pub const IPPROTO_ICMP: c_int = 1;
pub const IPPROTO_TCP: c_int = 6;
pub const IPPROTO_UDP: c_int = 17;
pub const IPPROTO_IPV6: c_int = 41;
pub const IPPROTO_RAW: c_int = 255;

// Socket options
pub const SOL_SOCKET: c_int = 1;
pub const SO_DEBUG: c_int = 1;
pub const SO_REUSEADDR: c_int = 2;
pub const SO_TYPE: c_int = 3;
pub const SO_ERROR: c_int = 4;
pub const SO_DONTROUTE: c_int = 5;
pub const SO_BROADCAST: c_int = 6;
pub const SO_SNDBUF: c_int = 7;
pub const SO_RCVBUF: c_int = 8;
pub const SO_KEEPALIVE: c_int = 9;
pub const SO_OOBINLINE: c_int = 10;
pub const SO_LINGER: c_int = 13;
pub const SO_REUSEPORT: c_int = 15;
pub const SO_RCVTIMEO: c_int = 20;
pub const SO_SNDTIMEO: c_int = 21;

// Shutdown modes
pub const SHUT_RD: c_int = 0;
pub const SHUT_WR: c_int = 1;
pub const SHUT_RDWR: c_int = 2;

// Message flags
pub const MSG_OOB: c_int = 0x01;
pub const MSG_PEEK: c_int = 0x02;
pub const MSG_DONTROUTE: c_int = 0x04;
pub const MSG_CTRUNC: c_int = 0x08;
pub const MSG_TRUNC: c_int = 0x20;
pub const MSG_DONTWAIT: c_int = 0x40;
pub const MSG_EOR: c_int = 0x80;
pub const MSG_WAITALL: c_int = 0x100;
pub const MSG_NOSIGNAL: c_int = 0x4000;

// Socket address structures
pub const sockaddr = extern struct {
    sa_family: c_ushort,
    sa_data: [14]u8,
};

pub const socklen_t = u32;

pub const sockaddr_storage = extern struct {
    ss_family: c_ushort,
    __ss_padding: [118]u8,
    __ss_align: c_ulong,
};

pub const sockaddr_in = extern struct {
    sin_family: c_ushort,
    sin_port: c_ushort,
    sin_addr: extern struct { s_addr: u32 },
    sin_zero: [8]u8,
};

pub const sockaddr_in6 = extern struct {
    sin6_family: c_ushort,
    sin6_port: c_ushort,
    sin6_flowinfo: u32,
    sin6_addr: extern struct { s6_addr: [16]u8 },
    sin6_scope_id: u32,
};

pub const iovec = extern struct {
    iov_base: ?*anyopaque,
    iov_len: usize,
};

pub const msghdr = extern struct {
    msg_name: ?*anyopaque,
    msg_namelen: socklen_t,
    msg_iov: ?[*]iovec,
    msg_iovlen: usize,
    msg_control: ?*anyopaque,
    msg_controllen: usize,
    msg_flags: c_int,
};

pub const cmsghdr = extern struct {
    cmsg_len: usize,
    cmsg_level: c_int,
    cmsg_type: c_int,
};

pub const linger = extern struct {
    l_onoff: c_int,
    l_linger: c_int,
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

inline fn failIfErrno(rc: anytype) bool {
    const err = std.posix.errno(rc);
    if (err != .SUCCESS) {
        setErrno(err);
        return true;
    }
    return false;
}

// A. Core Socket Operations (15 functions)

pub export fn socket(domain: c_int, type_: c_int, protocol: c_int) c_int {
    const rc = std.posix.system.socket(domain, type_, protocol);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn socketpair(domain: c_int, type_: c_int, protocol: c_int, sv: *[2]c_int) c_int {
    const rc = std.posix.system.socketpair(domain, type_, protocol, @ptrCast(sv));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn bind(sockfd: c_int, addr: *const sockaddr, addrlen: socklen_t) c_int {
    const rc = std.posix.system.bind(sockfd, @ptrCast(addr), addrlen);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn listen(sockfd: c_int, backlog: c_int) c_int {
    const rc = std.posix.system.listen(sockfd, @intCast(backlog));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn accept(sockfd: c_int, addr: ?*sockaddr, addrlen: ?*socklen_t) c_int {
    const rc = std.posix.system.accept(sockfd, @ptrCast(addr), @ptrCast(addrlen));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn accept4(sockfd: c_int, addr: ?*sockaddr, addrlen: ?*socklen_t, flags: c_int) c_int {
    const rc = std.posix.system.accept4(sockfd, @ptrCast(addr), @ptrCast(addrlen), @intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn connect(sockfd: c_int, addr: *const sockaddr, addrlen: socklen_t) c_int {
    const rc = std.posix.system.connect(sockfd, @ptrCast(addr), addrlen);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn shutdown(sockfd: c_int, how: c_int) c_int {
    const rc = std.posix.system.shutdown(sockfd, how);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn getsockname(sockfd: c_int, addr: *sockaddr, addrlen: *socklen_t) c_int {
    const rc = std.posix.system.getsockname(sockfd, @ptrCast(addr), @ptrCast(addrlen));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn getpeername(sockfd: c_int, addr: *sockaddr, addrlen: *socklen_t) c_int {
    const rc = std.posix.system.getpeername(sockfd, @ptrCast(addr), @ptrCast(addrlen));
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn setsockopt(sockfd: c_int, level: c_int, optname: c_int, optval: ?*const anyopaque, optlen: socklen_t) c_int {
    const rc = std.posix.system.setsockopt(sockfd, level, optname, optval, optlen);
    if (failIfErrno(rc)) return -1;
    return 0;
}

pub export fn getsockopt(sockfd: c_int, level: c_int, optname: c_int, optval: ?*anyopaque, optlen: *socklen_t) c_int {
    const rc = std.posix.system.getsockopt(sockfd, level, optname, optval, @ptrCast(optlen));
    if (failIfErrno(rc)) return -1;
    return 0;
}

// B. Data Transfer Operations (15 functions)

pub export fn send(sockfd: c_int, buf: ?*const anyopaque, len: usize, flags: c_int) isize {
    const rc = std.posix.system.send(sockfd, buf, len, @intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn sendto(sockfd: c_int, buf: ?*const anyopaque, len: usize, flags: c_int, dest_addr: ?*const sockaddr, addrlen: socklen_t) isize {
    const rc = std.posix.system.sendto(sockfd, buf, len, @intCast(flags), @ptrCast(dest_addr), addrlen);
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn sendmsg(sockfd: c_int, msg: *const msghdr, flags: c_int) isize {
    const rc = std.posix.system.sendmsg(sockfd, @ptrCast(msg), @intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn recv(sockfd: c_int, buf: ?*anyopaque, len: usize, flags: c_int) isize {
    const rc = std.posix.system.recv(sockfd, buf, len, @intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn recvfrom(sockfd: c_int, buf: ?*anyopaque, len: usize, flags: c_int, src_addr: ?*sockaddr, addrlen: ?*socklen_t) isize {
    const rc = std.posix.system.recvfrom(sockfd, buf, len, @intCast(flags), @ptrCast(src_addr), @ptrCast(addrlen));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn recvmsg(sockfd: c_int, msg: *msghdr, flags: c_int) isize {
    const rc = std.posix.system.recvmsg(sockfd, @ptrCast(msg), @intCast(flags));
    if (failIfErrno(rc)) return -1;
    return rc;
}

pub export fn sendmmsg(sockfd: c_int, msgvec: *msghdr, vlen: c_uint, flags: c_int) c_int {
    _ = sockfd; _ = msgvec; _ = vlen; _ = flags;
    setErrno(.NOSYS);
    return -1;
}

pub export fn recvmmsg(sockfd: c_int, msgvec: *msghdr, vlen: c_uint, flags: c_int, timeout: ?*const timespec) c_int {
    _ = sockfd; _ = msgvec; _ = vlen; _ = flags; _ = timeout;
    setErrno(.NOSYS);
    return -1;
}

pub const timespec = extern struct {
    tv_sec: i64,
    tv_nsec: c_long,
};
