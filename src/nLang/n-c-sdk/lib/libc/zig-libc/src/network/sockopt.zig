// Socket Options & Info - Week 3 Networking Session 1
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const posix = std.posix;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Import types
const socklen_t = @import("lib.zig").socklen_t;
const sockaddr = @import("lib.zig").sockaddr;

/// FULL IMPLEMENTATION: Get socket options
pub export fn getsockopt(
    sockfd: c_int,
    level: c_int,
    optname: c_int,
    optval: ?*anyopaque,
    optlen: ?*socklen_t,
) c_int {
    const val = optval orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const len_ptr = optlen orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    var opt_len: u32 = len_ptr.*;
    
    posix.getsockopt(
        @intCast(sockfd),
        @intCast(level),
        @intCast(optname),
        @as([*]u8, @ptrCast(val))[0..opt_len],
    ) catch |err| {
        switch (err) {
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.InvalidSockOptLevel => setErrno(.ENOPROTOOPT),
            error.PermissionDenied => setErrno(.ACCES),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    len_ptr.* = opt_len;
    return 0;
}

/// FULL IMPLEMENTATION: Set socket options
pub export fn setsockopt(
    sockfd: c_int,
    level: c_int,
    optname: c_int,
    optval: ?*const anyopaque,
    optlen: socklen_t,
) c_int {
    const val = optval orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const opt_bytes = @as([*]const u8, @ptrCast(val))[0..optlen];
    
    posix.setsockopt(
        @intCast(sockfd),
        @intCast(level),
        @intCast(optname),
        opt_bytes,
    ) catch |err| {
        switch (err) {
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.InvalidSockOptValue => setErrno(.INVAL),
            error.TimeoutTooBig => setErrno(.INVAL),
            error.PermissionDenied => setErrno(.ACCES),
            error.NetworkSubsystemFailed => setErrno(.NOBUFS),
            error.SocketNotBound => setErrno(.INVAL),
            error.AlreadyConnected => setErrno(.ISCONN),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Get socket name (local address)
pub export fn getsockname(
    sockfd: c_int,
    addr: ?*sockaddr,
    addrlen: ?*socklen_t,
) c_int {
    const address = addr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const len_ptr = addrlen orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // Prepare address buffer
    var addr_buf: [256]u8 = undefined;
    var addr_len: posix.socklen_t = addr_buf.len;
    
    posix.getsockname(@intCast(sockfd), &addr_buf, &addr_len) catch |err| {
        switch (err) {
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.SystemResources => setErrno(.NOBUFS),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    // Copy address
    const copy_len = @min(addr_len, len_ptr.*);
    @memcpy(
        @as([*]u8, @ptrCast(address))[0..copy_len],
        addr_buf[0..copy_len],
    );
    len_ptr.* = addr_len;
    
    return 0;
}

/// FULL IMPLEMENTATION: Get peer name (remote address)
pub export fn getpeername(
    sockfd: c_int,
    addr: ?*sockaddr,
    addrlen: ?*socklen_t,
) c_int {
    const address = addr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const len_ptr = addrlen orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // Prepare address buffer
    var addr_buf: [256]u8 = undefined;
    var addr_len: posix.socklen_t = addr_buf.len;
    
    posix.getpeername(@intCast(sockfd), &addr_buf, &addr_len) catch |err| {
        switch (err) {
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.SocketNotConnected => setErrno(.NOTCONN),
            error.SystemResources => setErrno(.NOBUFS),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    // Copy address
    const copy_len = @min(addr_len, len_ptr.*);
    @memcpy(
        @as([*]u8, @ptrCast(address))[0..copy_len],
        addr_buf[0..copy_len],
    );
    len_ptr.* = addr_len;
    
    return 0;
}

// Total: 4 socket option/info functions fully implemented
// Critical for socket configuration and connection management
