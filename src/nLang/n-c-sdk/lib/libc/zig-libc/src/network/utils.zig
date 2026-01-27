//! Network Utility Functions
//! Byte order conversion, address manipulation, and helper functions
//! 
//! These functions provide essential utilities for network programming,
//! including endianness conversion and legacy address functions.

const std = @import("std");
const posix = std.posix;
const mem = std.mem;
const builtin = @import("builtin");

// ============================================================================
// Byte Order Conversion (Host <-> Network)
// ============================================================================

/// Convert 16-bit value from host byte order to network byte order (big-endian)
pub export fn htons(hostshort: u16) u16 {
    if (builtin.cpu.arch.endian() == .big) {
        return hostshort;
    } else {
        return @byteSwap(hostshort);
    }
}

/// Convert 16-bit value from network byte order to host byte order
pub export fn ntohs(netshort: u16) u16 {
    if (builtin.cpu.arch.endian() == .big) {
        return netshort;
    } else {
        return @byteSwap(netshort);
    }
}

/// Convert 32-bit value from host byte order to network byte order (big-endian)
pub export fn htonl(hostlong: u32) u32 {
    if (builtin.cpu.arch.endian() == .big) {
        return hostlong;
    } else {
        return @byteSwap(hostlong);
    }
}

/// Convert 32-bit value from network byte order to host byte order
pub export fn ntohl(netlong: u32) u32 {
    if (builtin.cpu.arch.endian() == .big) {
        return netlong;
    } else {
        return @byteSwap(netlong);
    }
}

/// Convert 64-bit value from host byte order to network byte order (big-endian)
pub export fn htonll(hostlonglong: u64) u64 {
    if (builtin.cpu.arch.endian() == .big) {
        return hostlonglong;
    } else {
        return @byteSwap(hostlonglong);
    }
}

/// Convert 64-bit value from network byte order to host byte order
pub export fn ntohll(netlonglong: u64) u64 {
    if (builtin.cpu.arch.endian() == .big) {
        return netlonglong;
    } else {
        return @byteSwap(netlonglong);
    }
}

// ============================================================================
// Legacy Address Conversion Functions (IPv4 only)
// ============================================================================

/// IPv4 address structure for inet_addr/inet_ntoa
pub const in_addr = extern struct {
    s_addr: u32, // Network byte order
};

/// Convert IPv4 dotted-decimal string to network byte order binary
/// 
/// Legacy function - prefer inet_pton() for new code.
/// 
/// @param cp: String in dotted-decimal notation (e.g., "192.168.1.1")
/// @return: IPv4 address in network byte order, INADDR_NONE on error
pub export fn inet_addr(cp: [*:0]const u8) u32 {
    const INADDR_NONE: u32 = 0xFFFFFFFF;
    
    var octets: [4]u8 = undefined;
    var idx: usize = 0;
    var current: u32 = 0;
    var has_digit = false;
    var i: usize = 0;
    
    while (cp[i] != 0 and idx < 4) : (i += 1) {
        const c = cp[i];
        if (c >= '0' and c <= '9') {
            const digit = c - '0';
            current = current * 10 + digit;
            if (current > 255) return INADDR_NONE;
            has_digit = true;
        } else if (c == '.') {
            if (!has_digit) return INADDR_NONE;
            octets[idx] = @intCast(current);
            idx += 1;
            current = 0;
            has_digit = false;
        } else {
            return INADDR_NONE;
        }
    }
    
    if (!has_digit or idx != 3) return INADDR_NONE;
    octets[idx] = @intCast(current);
    
    // Return in network byte order (big-endian)
    return @byteSwap(@as(u32, octets[0]) << 24 |
                     @as(u32, octets[1]) << 16 |
                     @as(u32, octets[2]) << 8 |
                     @as(u32, octets[3]));
}

/// Convert IPv4 address to dotted-decimal string
/// 
/// Legacy function - prefer inet_ntop() for new code.
/// Returns pointer to static buffer (not thread-safe).
/// 
/// @param in: IPv4 address structure
/// @return: Pointer to static string buffer
pub export fn inet_ntoa(in: in_addr) [*:0]u8 {
    // Thread-local static buffer
    const S = struct {
        threadlocal var buffer: [16]u8 = undefined;
    };
    
    const addr = @byteSwap(in.s_addr);
    const octet1 = (addr >> 24) & 0xFF;
    const octet2 = (addr >> 16) & 0xFF;
    const octet3 = (addr >> 8) & 0xFF;
    const octet4 = addr & 0xFF;
    
    const len = std.fmt.bufPrint(&S.buffer, "{d}.{d}.{d}.{d}", .{
        octet1, octet2, octet3, octet4
    }) catch {
        S.buffer[0] = '0';
        S.buffer[1] = '.';
        S.buffer[2] = '0';
        S.buffer[3] = '.';
        S.buffer[4] = '0';
        S.buffer[5] = '.';
        S.buffer[6] = '0';
        S.buffer[7] = 0;
        return &S.buffer;
    };
    
    S.buffer[len.len] = 0;
    return &S.buffer;
}

/// Convert IPv4 dotted-decimal string to network byte order binary (reentrant)
/// 
/// Reentrant version of inet_addr() that stores result in provided buffer.
/// 
/// @param cp: String in dotted-decimal notation
/// @param inp: Pointer to in_addr structure to store result
/// @return: 1 on success, 0 on error
pub export fn inet_aton(cp: [*:0]const u8, inp: *in_addr) c_int {
    const addr = inet_addr(cp);
    const INADDR_NONE: u32 = 0xFFFFFFFF;
    
    if (addr == INADDR_NONE) {
        // Check if it's actually 255.255.255.255 (valid but equals INADDR_NONE)
        var is_all_255 = true;
        var i: usize = 0;
        while (cp[i] != 0) : (i += 1) {
            if (cp[i] != '2' and cp[i] != '5' and cp[i] != '.') {
                is_all_255 = false;
                break;
            }
        }
        
        if (!is_all_255) return 0;
    }
    
    inp.s_addr = addr;
    return 1;
}

// ============================================================================
// Network Address Manipulation
// ============================================================================

/// Convert IPv4 network address to dotted-decimal string
/// 
/// Similar to inet_ntoa but takes network byte ordered u32.
/// 
/// @param net: Network address in network byte order
/// @return: Pointer to static string buffer
pub export fn inet_network(cp: [*:0]const u8) u32 {
    // Parse like inet_addr but return in host byte order
    const addr = inet_addr(cp);
    const INADDR_NONE: u32 = 0xFFFFFFFF;
    if (addr == INADDR_NONE) return INADDR_NONE;
    return @byteSwap(addr); // Convert to host byte order
}

/// Make IPv4 address from network and host parts
/// 
/// Combines network and host portions into complete address.
/// 
/// @param net: Network portion (host byte order)
/// @param host: Host portion (host byte order)
/// @return: Complete IPv4 address (network byte order)
pub export fn inet_makeaddr(net: u32, host: u32) in_addr {
    // Simplified version - assumes Class C
    const addr = (net << 8) | (host & 0xFF);
    return .{ .s_addr = @byteSwap(addr) };
}

/// Extract network portion from IPv4 address
/// 
/// @param in: IPv4 address structure
/// @return: Network portion (host byte order)
pub export fn inet_netof(in: in_addr) u32 {
    const addr = @byteSwap(in.s_addr);
    
    // Determine class and extract network portion
    if ((addr & 0x80000000) == 0) {
        // Class A
        return (addr >> 24) & 0xFF;
    } else if ((addr & 0xC0000000) == 0x80000000) {
        // Class B
        return (addr >> 16) & 0xFFFF;
    } else if ((addr & 0xE0000000) == 0xC0000000) {
        // Class C
        return (addr >> 8) & 0xFFFFFF;
    } else {
        // Class D/E - no network portion
        return addr;
    }
}

/// Extract host portion from IPv4 address
/// 
/// @param in: IPv4 address structure
/// @return: Host portion (host byte order)
pub export fn inet_lnaof(in: in_addr) u32 {
    const addr = @byteSwap(in.s_addr);
    
    // Determine class and extract host portion
    if ((addr & 0x80000000) == 0) {
        // Class A
        return addr & 0x00FFFFFF;
    } else if ((addr & 0xC0000000) == 0x80000000) {
        // Class B
        return addr & 0x0000FFFF;
    } else if ((addr & 0xE0000000) == 0xC0000000) {
        // Class C
        return addr & 0x000000FF;
    } else {
        // Class D/E - no host portion
        return 0;
    }
}

// ============================================================================
// Socket Type and Protocol Helpers
// ============================================================================

/// Protocol family codes (same as AF_* in most systems)
pub const PF_UNSPEC: c_int = 0;
pub const PF_INET: c_int = 2;
pub const PF_INET6: c_int = 10;
pub const PF_UNIX: c_int = 1;

/// Special IPv4 addresses
pub const INADDR_ANY: u32 = 0x00000000;
pub const INADDR_BROADCAST: u32 = 0xFFFFFFFF;
pub const INADDR_LOOPBACK: u32 = 0x7F000001; // 127.0.0.1 in network byte order
pub const INADDR_NONE: u32 = 0xFFFFFFFF;

/// Socket levels for setsockopt/getsockopt
pub const SOL_SOCKET: c_int = 1;
pub const IPPROTO_IP: c_int = 0;
pub const IPPROTO_IPV6: c_int = 41;
pub const IPPROTO_ICMP: c_int = 1;
pub const IPPROTO_TCP: c_int = 6;
pub const IPPROTO_UDP: c_int = 17;
pub const IPPROTO_RAW: c_int = 255;

/// Socket option names for SOL_SOCKET level
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

/// Shutdown modes
pub const SHUT_RD: c_int = 0;    // Further receives disallowed
pub const SHUT_WR: c_int = 1;    // Further sends disallowed
pub const SHUT_RDWR: c_int = 2;  // Further sends and receives disallowed

/// Message flags for send/recv
pub const MSG_OOB: c_int = 0x01;        // Out-of-band data
pub const MSG_PEEK: c_int = 0x02;       // Peek at incoming message
pub const MSG_DONTROUTE: c_int = 0x04;  // Don't use routing
pub const MSG_WAITALL: c_int = 0x100;   // Wait for full request
pub const MSG_DONTWAIT: c_int = 0x40;   // Non-blocking operation
pub const MSG_NOSIGNAL: c_int = 0x4000; // Don't generate SIGPIPE

// ============================================================================
// Helper Functions
// ============================================================================

/// Check if address is multicast
pub export fn is_multicast_ipv4(addr: u32) c_int {
    const host_addr = @byteSwap(addr);
    return if ((host_addr & 0xF0000000) == 0xE0000000) 1 else 0;
}

/// Check if address is loopback
pub export fn is_loopback_ipv4(addr: u32) c_int {
    const host_addr = @byteSwap(addr);
    return if ((host_addr & 0xFF000000) == 0x7F000000) 1 else 0;
}

/// Check if address is link-local
pub export fn is_linklocal_ipv4(addr: u32) c_int {
    const host_addr = @byteSwap(addr);
    return if ((host_addr & 0xFFFF0000) == 0xA9FE0000) 1 else 0; // 169.254.0.0/16
}

/// Check if address is private
pub export fn is_private_ipv4(addr: u32) c_int {
    const host_addr = @byteSwap(addr);
    
    // 10.0.0.0/8
    if ((host_addr & 0xFF000000) == 0x0A000000) return 1;
    
    // 172.16.0.0/12
    if ((host_addr & 0xFFF00000) == 0xAC100000) return 1;
    
    // 192.168.0.0/16
    if ((host_addr & 0xFFFF0000) == 0xC0A80000) return 1;
    
    return 0;
}

/// Get protocol name from number
pub export fn getprotobynumber(proto: c_int) ?*anyopaque {
    // Stub - would need protocol database
    _ = proto;
    return null;
}

/// Get protocol number from name
pub export fn getprotobyname(name: [*:0]const u8) ?*anyopaque {
    // Stub - would need protocol database
    _ = name;
    return null;
}

/// Get service by port number
pub export fn getservbyport(port: c_int, proto: ?[*:0]const u8) ?*anyopaque {
    // Stub - would need services database
    _ = port;
    _ = proto;
    return null;
}

/// Get service by name
pub export fn getservbyname(name: [*:0]const u8, proto: ?[*:0]const u8) ?*anyopaque {
    // Stub - would need services database
    _ = name;
    _ = proto;
    return null;
}
