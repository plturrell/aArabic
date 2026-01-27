//! Address Resolution Functions
//! POSIX getaddrinfo() family for DNS resolution and address conversion
//! 
//! These functions provide protocol-independent address resolution,
//! essential for modern network programming.

const std = @import("std");
const posix = std.posix;
const builtin = @import("builtin");
const mem = std.mem;

// ============================================================================
// Address Info Structures
// ============================================================================

/// Socket address family codes
pub const AF_UNSPEC: c_int = 0;     // Unspecified
pub const AF_INET: c_int = 2;       // IPv4
pub const AF_INET6: c_int = 10;     // IPv6

/// Socket types
pub const SOCK_STREAM: c_int = 1;   // TCP
pub const SOCK_DGRAM: c_int = 2;    // UDP
pub const SOCK_RAW: c_int = 3;      // Raw socket

/// Protocol codes
pub const IPPROTO_IP: c_int = 0;    // IP
pub const IPPROTO_TCP: c_int = 6;   // TCP
pub const IPPROTO_UDP: c_int = 17;  // UDP

/// getaddrinfo() flags
pub const AI_PASSIVE: c_int = 0x0001;       // Use for bind()
pub const AI_CANONNAME: c_int = 0x0002;     // Fill in ai_canonname
pub const AI_NUMERICHOST: c_int = 0x0004;   // Prevent name resolution
pub const AI_NUMERICSERV: c_int = 0x0400;   // Prevent service resolution
pub const AI_V4MAPPED: c_int = 0x0008;      // Map IPv6 to IPv4
pub const AI_ALL: c_int = 0x0010;           // Return both IPv4 and IPv6
pub const AI_ADDRCONFIG: c_int = 0x0020;    // Only return addresses for configured types

/// getnameinfo() flags
pub const NI_NUMERICHOST: c_int = 0x01;     // Return numeric address
pub const NI_NUMERICSERV: c_int = 0x02;     // Return numeric port
pub const NI_NOFQDN: c_int = 0x04;          // Don't return FQDN
pub const NI_NAMEREQD: c_int = 0x08;        // Error if name can't be resolved
pub const NI_DGRAM: c_int = 0x10;           // Service is datagram

/// Maximum hostname and service name lengths
pub const NI_MAXHOST: usize = 1025;
pub const NI_MAXSERV: usize = 32;

/// Generic socket address structure
pub const sockaddr = extern struct {
    sa_family: u16,         // Address family
    sa_data: [14]u8,        // Protocol-specific address data
};

/// IPv4 socket address
pub const sockaddr_in = extern struct {
    sin_family: u16,        // AF_INET
    sin_port: u16,          // Port number (network byte order)
    sin_addr: extern struct {
        s_addr: u32,        // IPv4 address (network byte order)
    },
    sin_zero: [8]u8,        // Padding
};

/// IPv6 socket address
pub const sockaddr_in6 = extern struct {
    sin6_family: u16,       // AF_INET6
    sin6_port: u16,         // Port number (network byte order)
    sin6_flowinfo: u32,     // IPv6 flow information
    sin6_addr: extern struct {
        s6_addr: [16]u8,    // IPv6 address
    },
    sin6_scope_id: u32,     // Scope ID
};

/// Address information structure
pub const addrinfo = extern struct {
    ai_flags: c_int,            // Input flags
    ai_family: c_int,           // Address family
    ai_socktype: c_int,         // Socket type
    ai_protocol: c_int,         // Protocol
    ai_addrlen: u32,            // Length of socket address
    ai_addr: ?*sockaddr,        // Socket address
    ai_canonname: ?[*:0]u8,     // Canonical name
    ai_next: ?*addrinfo,        // Next structure in linked list
};

// ============================================================================
// Error Codes
// ============================================================================

/// getaddrinfo() error codes
pub const EAI_AGAIN: c_int = -3;        // Temporary failure
pub const EAI_BADFLAGS: c_int = -1;     // Invalid flags
pub const EAI_FAIL: c_int = -4;         // Non-recoverable failure
pub const EAI_FAMILY: c_int = -6;       // Unsupported address family
pub const EAI_MEMORY: c_int = -10;      // Out of memory
pub const EAI_NONAME: c_int = -2;       // Name or service not known
pub const EAI_SERVICE: c_int = -8;      // Service not available
pub const EAI_SOCKTYPE: c_int = -7;     // Unsupported socket type
pub const EAI_SYSTEM: c_int = -11;      // System error (check errno)
pub const EAI_OVERFLOW: c_int = -12;    // Buffer overflow

// ============================================================================
// Address Resolution Functions
// ============================================================================

/// Translate host and service names to socket address
/// 
/// Provides protocol-independent translation from host name and service name
/// to socket address structure. Supports both IPv4 and IPv6.
/// 
/// @param node: Host name or numeric address string (NULL = localhost)
/// @param service: Service name or port number string (NULL = any port)
/// @param hints: Hints to filter results (NULL = any)
/// @param res: Result list (must be freed with freeaddrinfo)
/// @return: 0 on success, error code on failure (use gai_strerror)
pub export fn getaddrinfo(
    node: ?[*:0]const u8,
    service: ?[*:0]const u8,
    hints: ?*const addrinfo,
    res: *?*addrinfo,
) c_int {
    // Validate parameters
    if (res == null) {
        return EAI_SYSTEM;
    }
    
    // Use system getaddrinfo - available on all POSIX platforms
    const result = std.c.getaddrinfo(node, service, hints, res);
    return result;
}

/// Free address information structure
/// 
/// Frees the linked list of addrinfo structures returned by getaddrinfo().
/// 
/// @param res: Address information list to free
pub export fn freeaddrinfo(res: ?*addrinfo) void {
    if (res == null) return;
    std.c.freeaddrinfo(res);
}

/// Get error message for getaddrinfo() error code
/// 
/// Translates getaddrinfo() error codes to human-readable strings.
/// 
/// @param errcode: Error code from getaddrinfo()
/// @return: Error message string (do not free)
pub export fn gai_strerror(errcode: c_int) [*:0]const u8 {
    return std.c.gai_strerror(errcode);
}

/// Translate socket address to host and service names
/// 
/// Reverse of getaddrinfo() - converts socket address structure
/// to host name and service name strings.
/// 
/// @param sa: Socket address to translate
/// @param salen: Length of socket address structure
/// @param host: Buffer for host name (NULL = don't return)
/// @param hostlen: Size of host buffer
/// @param serv: Buffer for service name (NULL = don't return)
/// @param servlen: Size of service buffer
/// @param flags: Flags to control translation (NI_NUMERICHOST, etc.)
/// @return: 0 on success, error code on failure
pub export fn getnameinfo(
    sa: *const sockaddr,
    salen: u32,
    host: ?[*]u8,
    hostlen: u32,
    serv: ?[*]u8,
    servlen: u32,
    flags: c_int,
) c_int {
    // Validate parameters
    if (sa == null or salen == 0) {
        return EAI_FAIL;
    }
    
    if (host != null and hostlen == 0) {
        return EAI_OVERFLOW;
    }
    
    if (serv != null and servlen == 0) {
        return EAI_OVERFLOW;
    }
    
    // Use system getnameinfo
    const result = std.c.getnameinfo(sa, salen, host, hostlen, serv, servlen, flags);
    return result;
}

// ============================================================================
// Address Conversion Functions
// ============================================================================

/// Convert presentation format address to network format
/// 
/// Converts string representation of IP address to binary format.
/// 
/// @param af: Address family (AF_INET or AF_INET6)
/// @param src: String representation of address
/// @param dst: Buffer for binary address
/// @return: 1 on success, 0 if src is not valid, -1 on error
pub export fn inet_pton(af: c_int, src: [*:0]const u8, dst: *anyopaque) c_int {
    if (af == AF_INET) {
        // IPv4
        var octets: [4]u8 = undefined;
        var idx: usize = 0;
        var current: u32 = 0;
        var has_digit = false;
        var i: usize = 0;
        
        while (src[i] != 0 and idx < 4) : (i += 1) {
            const c = src[i];
            if (c >= '0' and c <= '9') {
                const digit = c - '0';
                current = current * 10 + digit;
                if (current > 255) return 0; // Invalid octet
                has_digit = true;
            } else if (c == '.') {
                if (!has_digit) return 0; // Empty octet
                octets[idx] = @intCast(current);
                idx += 1;
                current = 0;
                has_digit = false;
            } else {
                return 0; // Invalid character
            }
        }
        
        // Handle last octet
        if (!has_digit or idx != 3) return 0;
        octets[idx] = @intCast(current);
        
        // Convert to network byte order (big-endian)
        const result = @as(u32, octets[0]) << 24 |
                      @as(u32, octets[1]) << 16 |
                      @as(u32, octets[2]) << 8 |
                      @as(u32, octets[3]);
        
        const dst_u32: *u32 = @ptrCast(@alignCast(dst));
        dst_u32.* = @byteSwap(result); // Store in network byte order
        
        return 1;
    } else if (af == AF_INET6) {
        // IPv6 - use system implementation for complexity
        return std.c.inet_pton(af, src, dst);
    } else {
        posix.errno(posix.E.AFNOSUPPORT);
        return -1;
    }
}

/// Convert network format address to presentation format
/// 
/// Converts binary IP address to string representation.
/// 
/// @param af: Address family (AF_INET or AF_INET6)
/// @param src: Binary address
/// @param dst: Buffer for string representation
/// @param size: Size of dst buffer
/// @return: dst on success, NULL on error
pub export fn inet_ntop(af: c_int, src: *const anyopaque, dst: [*]u8, size: u32) ?[*]u8 {
    if (af == AF_INET) {
        // IPv4 - need at least 16 bytes ("255.255.255.255\0")
        if (size < 16) {
            posix.errno(posix.E.NOSPC);
            return null;
        }
        
        const src_u32: *const u32 = @ptrCast(@alignCast(src));
        const addr = @byteSwap(src_u32.*); // Convert from network byte order
        
        const octet1 = (addr >> 24) & 0xFF;
        const octet2 = (addr >> 16) & 0xFF;
        const octet3 = (addr >> 8) & 0xFF;
        const octet4 = addr & 0xFF;
        
        // Format as string
        var buf: [16]u8 = undefined;
        const len = std.fmt.bufPrint(&buf, "{d}.{d}.{d}.{d}", .{
            octet1, octet2, octet3, octet4
        }) catch {
            posix.errno(posix.E.INVAL);
            return null;
        };
        
        @memcpy(dst[0..len.len], len);
        dst[len.len] = 0; // Null terminate
        
        return dst;
    } else if (af == AF_INET6) {
        // IPv6 - use system implementation for complexity
        return std.c.inet_ntop(af, src, dst, size);
    } else {
        posix.errno(posix.E.AFNOSUPPORT);
        return null;
    }
}
