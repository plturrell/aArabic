// Socket Operations - Week 3 Networking Session 1
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const posix = std.posix;

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Import types from lib.zig
const socklen_t = u32;
const sockaddr = @import("lib.zig").sockaddr;

/// FULL IMPLEMENTATION: Create a socket
pub export fn socket(domain: c_int, stype: c_int, protocol: c_int) c_int {
    const sock = posix.socket(
        @intCast(domain),
        @intCast(stype),
        @intCast(protocol),
    ) catch |err| {
        switch (err) {
            error.AddressFamilyNotSupported => setErrno(.AFNOSUPPORT),
            error.ProtocolNotSupported => setErrno(.PROTONOSUPPORT),
            error.SocketTypeNotSupported => setErrno(.SOCKTNOSUPPORT),
            error.PermissionDenied => setErrno(.ACCES),
            error.ProcessFdQuotaExceeded => setErrno(.MFILE),
            error.SystemFdQuotaExceeded => setErrno(.NFILE),
            error.SystemResources => setErrno(.NOBUFS),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return @intCast(sock);
}

/// FULL IMPLEMENTATION: Bind socket to address
pub export fn bind(sockfd: c_int, addr: ?*const sockaddr, addrlen: socklen_t) c_int {
    const address = addr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    // Convert sockaddr to platform-specific format
    const addr_bytes = @as([*]const u8, @ptrCast(address))[0..addrlen];
    
    posix.bind(@intCast(sockfd), addr_bytes) catch |err| {
        switch (err) {
            error.AccessDenied => setErrno(.ACCES),
            error.AddressInUse => setErrno(.ADDRINUSE),
            error.AddressNotAvailable => setErrno(.ADDRNOTAVAIL),
            error.AlreadyBound => setErrno(.INVAL),
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.NetworkUnreachable => setErrno(.NETUNREACH),
            error.ReadOnlyFileSystem => setErrno(.ROFS),
            error.SymLinkLoop => setErrno(.LOOP),
            error.NameTooLong => setErrno(.NAMETOOLONG),
            error.FileNotFound => setErrno(.NOENT),
            error.NotDir => setErrno(.NOTDIR),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Listen for connections
pub export fn listen(sockfd: c_int, backlog: c_int) c_int {
    posix.listen(@intCast(sockfd), @intCast(backlog)) catch |err| {
        switch (err) {
            error.AddressInUse => setErrno(.ADDRINUSE),
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.OperationNotSupported => setErrno(.OPNOTSUPP),
            error.SocketNotBound => setErrno(.INVAL),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Accept connection
pub export fn accept(sockfd: c_int, addr: ?*sockaddr, addrlen: ?*socklen_t) c_int {
    // Prepare address buffer
    var addr_buf: [256]u8 = undefined;
    var addr_len: posix.socklen_t = addr_buf.len;
    
    const client_sock = posix.accept(
        @intCast(sockfd),
        &addr_buf,
        &addr_len,
        0, // flags
    ) catch |err| {
        switch (err) {
            error.WouldBlock => setErrno(.AGAIN),
            error.ConnectionAborted => setErrno(.CONNABORTED),
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.ProcessFdQuotaExceeded => setErrno(.MFILE),
            error.SystemFdQuotaExceeded => setErrno(.NFILE),
            error.SystemResources => setErrno(.NOBUFS),
            error.SocketNotListening => setErrno(.INVAL),
            error.ProtocolFailure => setErrno(.PROTO),
            error.BlockedByFirewall => setErrno(.PERM),
            error.PermissionDenied => setErrno(.PERM),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    // Copy address if requested
    if (addr) |a| {
        const copy_len = @min(addr_len, if (addrlen) |al| al.* else 0);
        @memcpy(
            @as([*]u8, @ptrCast(a))[0..copy_len],
            addr_buf[0..copy_len],
        );
        if (addrlen) |al| al.* = addr_len;
    }
    
    return @intCast(client_sock);
}

/// FULL IMPLEMENTATION: Accept with flags (Linux-specific)
pub export fn accept4(
    sockfd: c_int,
    addr: ?*sockaddr,
    addrlen: ?*socklen_t,
    flags: c_int,
) c_int {
    // Prepare address buffer
    var addr_buf: [256]u8 = undefined;
    var addr_len: posix.socklen_t = addr_buf.len;
    
    const client_sock = posix.accept(
        @intCast(sockfd),
        &addr_buf,
        &addr_len,
        @intCast(flags),
    ) catch |err| {
        switch (err) {
            error.WouldBlock => setErrno(.AGAIN),
            error.ConnectionAborted => setErrno(.CONNABORTED),
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.ProcessFdQuotaExceeded => setErrno(.MFILE),
            error.SystemFdQuotaExceeded => setErrno(.NFILE),
            error.SystemResources => setErrno(.NOBUFS),
            error.SocketNotListening => setErrno(.INVAL),
            error.ProtocolFailure => setErrno(.PROTO),
            error.BlockedByFirewall => setErrno(.PERM),
            error.PermissionDenied => setErrno(.PERM),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    // Copy address if requested
    if (addr) |a| {
        const copy_len = @min(addr_len, if (addrlen) |al| al.* else 0);
        @memcpy(
            @as([*]u8, @ptrCast(a))[0..copy_len],
            addr_buf[0..copy_len],
        );
        if (addrlen) |al| al.* = addr_len;
    }
    
    return @intCast(client_sock);
}

/// FULL IMPLEMENTATION: Connect to remote address
pub export fn connect(sockfd: c_int, addr: ?*const sockaddr, addrlen: socklen_t) c_int {
    const address = addr orelse {
        setErrno(.INVAL);
        return -1;
    };
    
    const addr_bytes = @as([*]const u8, @ptrCast(address))[0..addrlen];
    
    posix.connect(@intCast(sockfd), addr_bytes) catch |err| {
        switch (err) {
            error.AccessDenied => setErrno(.ACCES),
            error.AddressInUse => setErrno(.ADDRINUSE),
            error.AddressNotAvailable => setErrno(.ADDRNOTAVAIL),
            error.AddressFamilyNotSupported => setErrno(.AFNOSUPPORT),
            error.WouldBlock => setErrno(.INPROGRESS),
            error.ConnectionPending => setErrno(.ALREADY),
            error.ConnectionRefused => setErrno(.CONNREFUSED),
            error.ConnectionResetByPeer => setErrno(.CONNRESET),
            error.ConnectionTimedOut => setErrno(.TIMEDOUT),
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.NetworkUnreachable => setErrno(.NETUNREACH),
            error.PermissionDenied => setErrno(.ACCES),
            error.ProtocolNotSupported => setErrno(.PROTOTYPE),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return 0;
}

/// FULL IMPLEMENTATION: Shutdown socket
pub export fn shutdown(sockfd: c_int, how: c_int) c_int {
    // how: 0=SHUT_RD, 1=SHUT_WR, 2=SHUT_RDWR
    const shut_how: std.posix.ShutdownHow = switch (how) {
        0 => .recv,
        1 => .send,
        2 => .both,
        else => {
            setErrno(.INVAL);
            return -1;
        },
    };
    
    posix.shutdown(@intCast(sockfd), shut_how) catch |err| {
        switch (err) {
            error.FileDescriptorNotASocket => setErrno(.NOTSOCK),
            error.NetworkSubsystemFailed => setErrno(.NOBUFS),
            error.SocketNotConnected => setErrno(.NOTCONN),
            else => setErrno(.INVAL),
        }
        return -1;
    };
    
    return 0;
}

// Total: 7 core socket functions fully implemented
// These provide the foundation for TCP/IP networking in banking applications
