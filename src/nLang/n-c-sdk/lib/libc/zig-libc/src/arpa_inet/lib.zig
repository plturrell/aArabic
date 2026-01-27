// arpa/inet module - Phase 1.7 Priority 5 - Network Address Conversion
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");

pub const in_addr_t = u32;
pub const in_port_t = u16;

pub const in_addr = extern struct {
    s_addr: in_addr_t,
};

pub const in6_addr = extern struct {
    s6_addr: [16]u8,
};

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// A. Address Conversion Functions (15 functions)

pub export fn htons(hostshort: u16) u16 {
    return std.mem.nativeToBig(u16, hostshort);
}

pub export fn htonl(hostlong: u32) u32 {
    return std.mem.nativeToBig(u32, hostlong);
}

pub export fn ntohs(netshort: u16) u16 {
    return std.mem.bigToNative(u16, netshort);
}

pub export fn ntohl(netlong: u32) u32 {
    return std.mem.bigToNative(u32, netlong);
}

pub export fn inet_addr(cp: [*:0]const u8) in_addr_t {
    var addr: in_addr = undefined;
    if (inet_aton(cp, &addr) == 1) {
        return addr.s_addr;
    }
    return 0xffffffff; // INADDR_NONE
}

pub export fn inet_aton(cp: [*:0]const u8, inp: *in_addr) c_int {
    return inet_pton(2, cp, @ptrCast(inp)); // AF_INET = 2
}

pub export fn inet_ntoa(in: in_addr) [*:0]const u8 {
    var buf: [16]u8 = undefined;
    _ = inet_ntop(2, @ptrCast(&in), &buf, 16);
    return @ptrCast(&buf);
}

pub export fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int {
    const src_slice = std.mem.span(src);
    
    if (af == 2) { // AF_INET
        var octets: [4]u8 = undefined;
        var octet_idx: usize = 0;
        var current: u32 = 0;
        var has_digit = false;
        
        for (src_slice) |ch| {
            if (ch >= '0' and ch <= '9') {
                current = current * 10 + (ch - '0');
                has_digit = true;
                if (current > 255) return 0;
            } else if (ch == '.') {
                if (!has_digit or octet_idx >= 3) return 0;
                octets[octet_idx] = @intCast(current);
                octet_idx += 1;
                current = 0;
                has_digit = false;
            } else {
                return 0;
            }
        }
        
        if (!has_digit or octet_idx != 3) return 0;
        octets[3] = @intCast(current);
        
        const addr: *in_addr = @ptrCast(@alignCast(dst));
        addr.s_addr = @bitCast(octets);
        return 1;
    }
    
    setErrno(.AFNOSUPPORT);
    return -1;
}

pub export fn inet_ntop(af: c_int, src: *const anyopaque, dst: [*]u8, size: u32) ?[*:0]const u8 {
    if (af == 2) { // AF_INET
        const addr: *const in_addr = @ptrCast(@alignCast(src));
        const octets: [4]u8 = @bitCast(addr.s_addr);
        
        const len = std.fmt.bufPrint(dst[0..size], "{d}.{d}.{d}.{d}\x00", .{
            octets[0], octets[1], octets[2], octets[3]
        }) catch {
            setErrno(.NOSPC);
            return null;
        };
        
        dst[len] = 0;
        return dst;
    }
    
    setErrno(.AFNOSUPPORT);
    return null;
}

pub export fn inet_network(cp: [*:0]const u8) in_addr_t {
    var addr: in_addr = undefined;
    if (inet_aton(cp, &addr) == 1) {
        return ntohl(addr.s_addr);
    }
    return 0xffffffff;
}

pub export fn inet_makeaddr(net: in_addr_t, host: in_addr_t) in_addr {
    return in_addr{ .s_addr = htonl((net << 8) | host) };
}

pub export fn inet_lnaof(in: in_addr) in_addr_t {
    const addr = ntohl(in.s_addr);
    return addr & 0xff;
}

pub export fn inet_netof(in: in_addr) in_addr_t {
    const addr = ntohl(in.s_addr);
    return addr >> 8;
}
