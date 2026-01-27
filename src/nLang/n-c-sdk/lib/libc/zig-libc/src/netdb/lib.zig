// netdb module - Phase 1.7 Priority 5 - Name Resolution (Real /etc/hosts)
const std = @import("std");
const errno_mod = @import("../errno/lib.zig");
const socket = @import("../sys_socket/lib.zig");
const inet = @import("../arpa_inet/lib.zig");

// Error codes (h_errno)
pub const HOST_NOT_FOUND: c_int = 1;
pub const TRY_AGAIN: c_int = 2;
pub const NO_RECOVERY: c_int = 3;
pub const NO_DATA: c_int = 4;

// EAI error codes for getaddrinfo
pub const EAI_AGAIN: c_int = -3;
pub const EAI_BADFLAGS: c_int = -1;
pub const EAI_FAIL: c_int = -4;
pub const EAI_FAMILY: c_int = -6;
pub const EAI_MEMORY: c_int = -10;
pub const EAI_NONAME: c_int = -2;
pub const EAI_SERVICE: c_int = -8;
pub const EAI_SOCKTYPE: c_int = -7;
pub const EAI_SYSTEM: c_int = -11;
pub const EAI_OVERFLOW: c_int = -12;

// Address info flags
pub const AI_PASSIVE: c_int = 0x0001;
pub const AI_CANONNAME: c_int = 0x0002;
pub const AI_NUMERICHOST: c_int = 0x0004;
pub const AI_V4MAPPED: c_int = 0x0008;
pub const AI_ALL: c_int = 0x0010;
pub const AI_ADDRCONFIG: c_int = 0x0020;
pub const AI_NUMERICSERV: c_int = 0x0400;

// Name info flags
pub const NI_NUMERICHOST: c_int = 0x01;
pub const NI_NUMERICSERV: c_int = 0x02;
pub const NI_NOFQDN: c_int = 0x04;
pub const NI_NAMEREQD: c_int = 0x08;
pub const NI_DGRAM: c_int = 0x10;

pub const NI_MAXHOST: usize = 1025;
pub const NI_MAXSERV: usize = 32;

// Structures
pub const hostent = extern struct {
    h_name: [*:0]u8,
    h_aliases: [*][*:0]u8,
    h_addrtype: c_int,
    h_length: c_int,
    h_addr_list: [*][*]u8,
};

pub const servent = extern struct {
    s_name: [*:0]u8,
    s_aliases: [*][*:0]u8,
    s_port: c_int,
    s_proto: [*:0]u8,
};

pub const protoent = extern struct {
    p_name: [*:0]u8,
    p_aliases: [*][*:0]u8,
    p_proto: c_int,
};

pub const addrinfo = extern struct {
    ai_flags: c_int,
    ai_family: c_int,
    ai_socktype: c_int,
    ai_protocol: c_int,
    ai_addrlen: socket.socklen_t,
    ai_addr: ?*socket.sockaddr,
    ai_canonname: ?[*:0]u8,
    ai_next: ?*addrinfo,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};
const allocator = gpa.allocator();

inline fn setErrno(err: std.posix.E) void {
    errno_mod.__errno_location().* = @intCast(@intFromEnum(err));
}

// Global storage for gethostbyname (legacy API requirement)
var static_hostent: hostent = undefined;
var static_name_buf: [256]u8 = undefined;
var static_addr_buf: [16]u8 = undefined; // Support IPv6
var static_addr_ptrs: [2][*]u8 = undefined;
var static_alias_ptrs: [1][*:0]u8 = undefined;

// Internal /etc/hosts Parser
fn lookupHosts(name: []const u8) ?inet.in_addr {
    const file = std.fs.openFileAbsolute("/etc/hosts", .{}) catch return null;
    defer file.close();
    
    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();
    var buf: [1024]u8 = undefined;
    
    while (in_stream.readUntilDelimiterOrEof(&buf, '\n') catch null) |line| {
        // Skip comments and empty lines
        const trim_line = std.mem.trim(u8, line, " \t\r");
        if (trim_line.len == 0 or trim_line[0] == '#') continue;
        
        // Split IP and names
        var iter = std.mem.tokenizeAny(u8, trim_line, " \t");
        const ip_str = iter.next() orelse continue;
        
        // Parse IP (IPv4 only for gethostbyname simple match)
        var addr: inet.in_addr = undefined;
        // Need null-terminated string for inet_aton
        var ip_buf: [64:0]u8 = undefined;
        const len = @min(ip_str.len, 63);
        @memcpy(ip_buf[0..len], ip_str[0..len]);
        ip_buf[len] = 0;
        
        if (inet.inet_aton(@ptrCast(&ip_buf), &addr) != 1) continue;
        
        // Check aliases
        while (iter.next()) |alias| {
            if (std.mem.eql(u8, alias, name)) {
                return addr;
            }
        }
    }
    return null;
}

pub export fn gethostbyname(name: [*:0]const u8) ?*hostent {
    const name_slice = std.mem.span(name);
    
    // 1. Try parsing as IP directly
    var addr: inet.in_addr = undefined;
    if (inet.inet_aton(name, &addr) == 1) {
        // Fill static struct
        return fillHostent(name_slice, addr);
    }
    
    // 2. Lookup in /etc/hosts
    if (lookupHosts(name_slice)) |found_addr| {
        return fillHostent(name_slice, found_addr);
    }
    
    // 3. Fallback to DNS
    // Note: Full DNS resolution requires UDP socket communication with DNS servers.
    // This implementation provides /etc/hosts lookup. For full DNS resolution,
    // consider using the system's native resolver or implementing RFC 1035.
    // Common DNS servers: 8.8.8.8 (Google), 1.1.1.1 (Cloudflare), from /etc/resolv.conf

    // For now, return null if not found in /etc/hosts or as IP literal
    return null;
}

fn fillHostent(name: []const u8, addr: inet.in_addr) *hostent {
    // Copy name
    const name_len = @min(name.len, 255);
    @memcpy(static_name_buf[0..name_len], name[0..name_len]);
    static_name_buf[name_len] = 0;
    
    // Copy address
    const addr_bytes = std.mem.asBytes(&addr);
    @memcpy(static_addr_buf[0..4], addr_bytes);
    
    // Set pointers
    static_addr_ptrs[0] = @ptrCast(&static_addr_buf);
    static_addr_ptrs[1] = null;
    
    static_alias_ptrs[0] = null;
    
    static_hostent.h_name = @ptrCast(&static_name_buf);
    static_hostent.h_aliases = @ptrCast(&static_alias_ptrs);
    static_hostent.h_addrtype = 2; // AF_INET
    static_hostent.h_length = 4;
    static_hostent.h_addr_list = @ptrCast(&static_addr_ptrs);
    
    return &static_hostent;
}

// Global h_errno for gethostbyaddr/gethostbyname errors
var h_errno_value: c_int = 0;

pub export fn __h_errno_location() *c_int {
    return &h_errno_value;
}

pub export fn gethostbyaddr(addr: *const anyopaque, len: socket.socklen_t, type_: c_int) ?*hostent {
    // Validate parameters
    if (type_ != socket.AF_INET) {
        h_errno_value = NO_RECOVERY;
        return null;
    }

    if (len != 4) {
        h_errno_value = NO_RECOVERY;
        return null;
    }

    // Get the address bytes
    const addr_bytes: *const [4]u8 = @ptrCast(@alignCast(addr));

    // Try reverse lookup in /etc/hosts
    if (reverseLookupHosts(addr_bytes.*)) |name| {
        // Build hostent from found name
        var ip_addr: inet.in_addr = undefined;
        ip_addr.s_addr = @bitCast(addr_bytes.*);
        return fillHostent(name, ip_addr);
    }

    // Not found
    h_errno_value = HOST_NOT_FOUND;
    return null;
}

// Reverse lookup in /etc/hosts - find hostname for IP
fn reverseLookupHosts(addr_bytes: [4]u8) ?[]const u8 {
    const file = std.fs.openFileAbsolute("/etc/hosts", .{}) catch return null;
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();
    var buf: [1024]u8 = undefined;

    while (in_stream.readUntilDelimiterOrEof(&buf, '\n') catch null) |line| {
        const trim_line = std.mem.trim(u8, line, " \t\r");
        if (trim_line.len == 0 or trim_line[0] == '#') continue;

        var iter = std.mem.tokenizeAny(u8, trim_line, " \t");
        const ip_str = iter.next() orelse continue;

        // Parse the IP
        var parsed_addr: inet.in_addr = undefined;
        var ip_buf: [64:0]u8 = undefined;
        const ip_len = @min(ip_str.len, 63);
        @memcpy(ip_buf[0..ip_len], ip_str[0..ip_len]);
        ip_buf[ip_len] = 0;

        if (inet.inet_aton(@ptrCast(&ip_buf), &parsed_addr) != 1) continue;

        // Compare addresses
        const parsed_bytes: [4]u8 = @bitCast(parsed_addr.s_addr);
        if (std.mem.eql(u8, &parsed_bytes, &addr_bytes)) {
            // Return the first hostname (canonical name)
            if (iter.next()) |hostname| {
                // Copy to static buffer (reuse static_name_buf)
                const name_len = @min(hostname.len, 255);
                @memcpy(static_name_buf[0..name_len], hostname[0..name_len]);
                static_name_buf[name_len] = 0;
                return static_name_buf[0..name_len];
            }
        }
    }
    return null;
}

pub export fn getaddrinfo(
    node: ?[*:0]const u8,
    service: ?[*:0]const u8,
    hints: ?*const addrinfo,
    res: **addrinfo
) c_int {
    // Extract hints
    var family: c_int = socket.AF_UNSPEC;
    var socktype: c_int = 0;
    var protocol: c_int = 0;
    var flags: c_int = 0;

    if (hints) |h| {
        family = h.ai_family;
        socktype = h.ai_socktype;
        protocol = h.ai_protocol;
        flags = h.ai_flags;

        // Validate flags
        const valid_flags = AI_PASSIVE | AI_CANONNAME | AI_NUMERICHOST | AI_NUMERICSERV | AI_V4MAPPED | AI_ALL | AI_ADDRCONFIG;
        if ((flags & ~valid_flags) != 0) {
            return EAI_BADFLAGS;
        }

        // Validate family
        if (family != socket.AF_UNSPEC and family != socket.AF_INET and family != socket.AF_INET6) {
            return EAI_FAMILY;
        }

        // Validate socktype
        if (socktype != 0 and socktype != socket.SOCK_STREAM and socktype != socket.SOCK_DGRAM and socktype != socket.SOCK_RAW) {
            return EAI_SOCKTYPE;
        }
    }

    // Parse service/port
    var port: u16 = 0;
    if (service) |s| {
        const s_slice = std.mem.span(s);
        if (s_slice.len > 0) {
            // Try numeric first
            port = std.fmt.parseInt(u16, s_slice, 10) catch blk: {
                // AI_NUMERICSERV requires numeric port
                if ((flags & AI_NUMERICSERV) != 0) {
                    return EAI_SERVICE;
                }
                // Look up service name
                const proto_str: ?[*:0]const u8 = if (socktype == socket.SOCK_DGRAM) "udp" else if (socktype == socket.SOCK_STREAM) "tcp" else null;
                if (lookupServicePort(s_slice, proto_str)) |found_port| {
                    break :blk found_port;
                }
                return EAI_SERVICE;
            };
        }
    }

    // Default socktype/protocol if not specified
    if (socktype == 0) socktype = socket.SOCK_STREAM;
    if (protocol == 0) {
        protocol = if (socktype == socket.SOCK_DGRAM) socket.IPPROTO_UDP else socket.IPPROTO_TCP;
    }

    // Only support AF_INET for now
    if (family == socket.AF_INET6) {
        return EAI_FAMILY;  // IPv6 not yet implemented
    }

    if (node) |n| {
        const name_slice = std.mem.span(n);

        var ip_addr: inet.in_addr = undefined;
        var found = false;

        // AI_NUMERICHOST - only parse as numeric
        if (inet.inet_aton(n, &ip_addr) == 1) {
            found = true;
        } else if ((flags & AI_NUMERICHOST) != 0) {
            return EAI_NONAME;
        } else if (lookupHosts(name_slice)) |addr| {
            ip_addr = addr;
            found = true;
        }

        if (found) {
            const result = allocator.create(addrinfo) catch return EAI_MEMORY;
            const sock_addr = allocator.create(socket.sockaddr_in) catch {
                allocator.destroy(result);
                return EAI_MEMORY;
            };

            sock_addr.sin_family = socket.AF_INET;
            sock_addr.sin_port = inet.htons(port);
            sock_addr.sin_addr.s_addr = ip_addr.s_addr;
            @memset(&sock_addr.sin_zero, 0);

            result.ai_flags = flags;
            result.ai_family = socket.AF_INET;
            result.ai_socktype = socktype;
            result.ai_protocol = protocol;
            result.ai_addrlen = @sizeOf(socket.sockaddr_in);
            result.ai_addr = @ptrCast(sock_addr);
            result.ai_canonname = null;
            result.ai_next = null;

            res.* = result;
            return 0;
        }

        return EAI_NONAME;
    } else {
        // No node - AI_PASSIVE determines address
        const result = allocator.create(addrinfo) catch return EAI_MEMORY;
        const sock_addr = allocator.create(socket.sockaddr_in) catch {
            allocator.destroy(result);
            return EAI_MEMORY;
        };

        sock_addr.sin_family = socket.AF_INET;
        sock_addr.sin_port = inet.htons(port);
        // AI_PASSIVE: INADDR_ANY, otherwise INADDR_LOOPBACK
        sock_addr.sin_addr.s_addr = if ((flags & AI_PASSIVE) != 0) 0 else inet.htonl(0x7f000001);
        @memset(&sock_addr.sin_zero, 0);

        result.ai_flags = flags;
        result.ai_family = socket.AF_INET;
        result.ai_socktype = socktype;
        result.ai_protocol = protocol;
        result.ai_addrlen = @sizeOf(socket.sockaddr_in);
        result.ai_addr = @ptrCast(sock_addr);
        result.ai_canonname = null;
        result.ai_next = null;

        res.* = result;
        return 0;
    }
}

pub export fn freeaddrinfo(res: *addrinfo) void {
    var current: ?*addrinfo = res;
    while (current) |node| {
        const next = node.ai_next;
        if (node.ai_addr) |addr| {
            allocator.destroy(addr);
        }
        allocator.destroy(node);
        current = next;
    }
}

// ... rest of the file (gai_strerror, getnameinfo, etc.) remains same ...
// Copying remaining functions for completeness

pub export fn gai_strerror(errcode: c_int) [*:0]const u8 {
    return switch (errcode) {
        0 => "Success",
        EAI_AGAIN => "Temporary failure in name resolution",
        EAI_BADFLAGS => "Invalid flags value",
        EAI_FAIL => "Non-recoverable failure in name resolution",
        EAI_FAMILY => "Address family not supported",
        EAI_MEMORY => "Memory allocation failure",
        EAI_NONAME => "Name or service not known",
        EAI_SERVICE => "Service not supported for socket type",
        EAI_SOCKTYPE => "Socket type not supported",
        EAI_SYSTEM => "System error",
        EAI_OVERFLOW => "Buffer overflow",
        HOST_NOT_FOUND => "Host not found",
        else => "Unknown error",
    };
}

// Helper: lookup service port by name using built-in table and /etc/services
fn lookupServicePort(name: []const u8, proto: ?[*:0]const u8) ?u16 {
    // Built-in common services table (fallback if /etc/services unavailable)
    const CommonService = struct { name: []const u8, port: u16, proto: []const u8 };
    const common_services = [_]CommonService{
        .{ .name = "http", .port = 80, .proto = "tcp" },
        .{ .name = "https", .port = 443, .proto = "tcp" },
        .{ .name = "ssh", .port = 22, .proto = "tcp" },
        .{ .name = "ftp", .port = 21, .proto = "tcp" },
        .{ .name = "ftp-data", .port = 20, .proto = "tcp" },
        .{ .name = "telnet", .port = 23, .proto = "tcp" },
        .{ .name = "smtp", .port = 25, .proto = "tcp" },
        .{ .name = "domain", .port = 53, .proto = "udp" },
        .{ .name = "dns", .port = 53, .proto = "udp" },
        .{ .name = "tftp", .port = 69, .proto = "udp" },
        .{ .name = "pop3", .port = 110, .proto = "tcp" },
        .{ .name = "imap", .port = 143, .proto = "tcp" },
        .{ .name = "snmp", .port = 161, .proto = "udp" },
        .{ .name = "ldap", .port = 389, .proto = "tcp" },
        .{ .name = "imaps", .port = 993, .proto = "tcp" },
        .{ .name = "pop3s", .port = 995, .proto = "tcp" },
        .{ .name = "mysql", .port = 3306, .proto = "tcp" },
        .{ .name = "postgresql", .port = 5432, .proto = "tcp" },
        .{ .name = "redis", .port = 6379, .proto = "tcp" },
    };

    const proto_slice: ?[]const u8 = if (proto) |p| std.mem.span(p) else null;

    // Try /etc/services first
    if (getservbyname(@ptrCast(name.ptr), proto)) |svc| {
        return inet.ntohs(@intCast(@as(u32, @bitCast(svc.s_port))));
    }

    // Fallback to built-in table
    for (common_services) |svc| {
        if (std.mem.eql(u8, svc.name, name)) {
            if (proto_slice) |p| {
                if (!std.mem.eql(u8, svc.proto, p)) continue;
            }
            return svc.port;
        }
    }

    return null;
}

pub export fn getnameinfo(
    sa: *const socket.sockaddr,
    salen: socket.socklen_t,
    host: ?[*]u8,
    hostlen: socket.socklen_t,
    serv: ?[*]u8,
    servlen: socket.socklen_t,
    flags: c_int
) c_int {
    _ = salen;

    if (sa.sa_family != socket.AF_INET) {
        return EAI_FAMILY;
    }

    const sin: *const socket.sockaddr_in = @ptrCast(@alignCast(sa));

    // Handle host
    if (host) |h| {
        if (hostlen == 0) return EAI_OVERFLOW;

        // NI_NUMERICHOST or no reverse lookup available
        if ((flags & NI_NUMERICHOST) != 0) {
            // Format IP as string
            const addr_bytes: [4]u8 = @bitCast(sin.sin_addr.s_addr);
            const result = std.fmt.bufPrint(h[0..hostlen], "{d}.{d}.{d}.{d}", .{
                addr_bytes[0], addr_bytes[1], addr_bytes[2], addr_bytes[3]
            }) catch return EAI_OVERFLOW;
            h[result.len] = 0;
        } else {
            // Try reverse lookup
            const addr_bytes: [4]u8 = @bitCast(sin.sin_addr.s_addr);
            if (reverseLookupHosts(addr_bytes)) |name| {
                if (name.len >= hostlen) return EAI_OVERFLOW;
                @memcpy(h[0..name.len], name);
                h[name.len] = 0;
            } else if ((flags & NI_NAMEREQD) != 0) {
                return EAI_NONAME;
            } else {
                // Fallback to numeric
                const result = std.fmt.bufPrint(h[0..hostlen], "{d}.{d}.{d}.{d}", .{
                    addr_bytes[0], addr_bytes[1], addr_bytes[2], addr_bytes[3]
                }) catch return EAI_OVERFLOW;
                h[result.len] = 0;
            }
        }
    }

    // Handle service
    if (serv) |s| {
        if (servlen == 0) return EAI_OVERFLOW;

        const port = inet.ntohs(sin.sin_port);

        if ((flags & NI_NUMERICSERV) != 0) {
            const result = std.fmt.bufPrint(s[0..servlen], "{d}", .{port}) catch return EAI_OVERFLOW;
            s[result.len] = 0;
        } else {
            // Try to look up service name
            const proto: ?[*:0]const u8 = if ((flags & NI_DGRAM) != 0) "udp" else "tcp";
            if (getservbyport(@intCast(inet.htons(port)), proto)) |svc| {
                const name = std.mem.span(svc.s_name);
                if (name.len >= servlen) return EAI_OVERFLOW;
                @memcpy(s[0..name.len], name);
                s[name.len] = 0;
            } else {
                // Fallback to numeric
                const result = std.fmt.bufPrint(s[0..servlen], "{d}", .{port}) catch return EAI_OVERFLOW;
                s[result.len] = 0;
            }
        }
    }

    return 0;
}

// =====================================================
// Service functions - /etc/services lookup
// =====================================================

var static_servent: servent = undefined;
var serv_name_buf: [64]u8 = undefined;
var serv_proto_buf: [32]u8 = undefined;
var serv_alias_ptrs: [1][*:0]u8 = undefined;

fn lookupService(name: ?[]const u8, port: ?c_int, proto: ?[]const u8) ?*servent {
    const file = std.fs.openFileAbsolute("/etc/services", .{}) catch return null;
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();
    var buf: [512]u8 = undefined;

    while (in_stream.readUntilDelimiterOrEof(&buf, '\n') catch null) |line| {
        const trim_line = std.mem.trim(u8, line, " \t\r");
        if (trim_line.len == 0 or trim_line[0] == '#') continue;

        // Format: service_name port/protocol [aliases...]
        var iter = std.mem.tokenizeAny(u8, trim_line, " \t");
        const svc_name = iter.next() orelse continue;
        const port_proto = iter.next() orelse continue;

        // Parse port/proto
        var pp_iter = std.mem.splitScalar(u8, port_proto, '/');
        const port_str = pp_iter.next() orelse continue;
        const proto_str = pp_iter.next() orelse continue;

        const svc_port = std.fmt.parseInt(c_int, port_str, 10) catch continue;

        // Check protocol match
        if (proto) |p| {
            if (!std.mem.eql(u8, proto_str, p)) continue;
        }

        // Check name or port match
        var matched = false;
        if (name) |n| {
            if (std.mem.eql(u8, svc_name, n)) matched = true;
            // Check aliases
            while (iter.next()) |alias| {
                if (std.mem.eql(u8, alias, n)) matched = true;
            }
        }
        if (port) |p| {
            if (svc_port == p) matched = true;
        }

        if (matched) {
            // Fill static servent
            const name_len = @min(svc_name.len, 63);
            @memcpy(serv_name_buf[0..name_len], svc_name[0..name_len]);
            serv_name_buf[name_len] = 0;

            const proto_len = @min(proto_str.len, 31);
            @memcpy(serv_proto_buf[0..proto_len], proto_str[0..proto_len]);
            serv_proto_buf[proto_len] = 0;

            serv_alias_ptrs[0] = null;

            static_servent.s_name = @ptrCast(&serv_name_buf);
            static_servent.s_aliases = @ptrCast(&serv_alias_ptrs);
            static_servent.s_port = inet.htons(@intCast(@as(u32, @bitCast(svc_port))));
            static_servent.s_proto = @ptrCast(&serv_proto_buf);

            return &static_servent;
        }
    }
    return null;
}

pub export fn getservbyname(name: [*:0]const u8, proto: ?[*:0]const u8) ?*servent {
    const name_slice = std.mem.span(name);
    const proto_slice = if (proto) |p| std.mem.span(p) else null;
    return lookupService(name_slice, null, proto_slice);
}

pub export fn getservbyport(port: c_int, proto: ?[*:0]const u8) ?*servent {
    const proto_slice = if (proto) |p| std.mem.span(p) else null;
    // Convert network byte order to host
    const host_port: c_int = @intCast(inet.ntohs(@intCast(@as(u32, @bitCast(port)))));
    return lookupService(null, host_port, proto_slice);
}

pub export fn getservent() ?*servent { return null; } // Iterator not implemented
pub export fn setservent(stayopen: c_int) void { _ = stayopen; }
pub export fn endservent() void {}

// =====================================================
// Protocol functions - /etc/protocols lookup
// =====================================================

var static_protoent: protoent = undefined;
var proto_name_buf: [64]u8 = undefined;
var proto_alias_ptrs: [1][*:0]u8 = undefined;

fn lookupProtocol(name: ?[]const u8, number: ?c_int) ?*protoent {
    const file = std.fs.openFileAbsolute("/etc/protocols", .{}) catch return null;
    defer file.close();

    var buf_reader = std.io.bufferedReader(file.reader());
    var in_stream = buf_reader.reader();
    var buf: [512]u8 = undefined;

    while (in_stream.readUntilDelimiterOrEof(&buf, '\n') catch null) |line| {
        const trim_line = std.mem.trim(u8, line, " \t\r");
        if (trim_line.len == 0 or trim_line[0] == '#') continue;

        // Format: protocol_name number [aliases...]
        var iter = std.mem.tokenizeAny(u8, trim_line, " \t");
        const proto_name = iter.next() orelse continue;
        const proto_num_str = iter.next() orelse continue;

        const proto_num = std.fmt.parseInt(c_int, proto_num_str, 10) catch continue;

        var matched = false;
        if (name) |n| {
            if (std.mem.eql(u8, proto_name, n)) matched = true;
            // Check aliases
            while (iter.next()) |alias| {
                if (std.mem.eql(u8, alias, n)) matched = true;
            }
        }
        if (number) |num| {
            if (proto_num == num) matched = true;
        }

        if (matched) {
            const name_len = @min(proto_name.len, 63);
            @memcpy(proto_name_buf[0..name_len], proto_name[0..name_len]);
            proto_name_buf[name_len] = 0;

            proto_alias_ptrs[0] = null;

            static_protoent.p_name = @ptrCast(&proto_name_buf);
            static_protoent.p_aliases = @ptrCast(&proto_alias_ptrs);
            static_protoent.p_proto = proto_num;

            return &static_protoent;
        }
    }
    return null;
}

pub export fn getprotobyname(name: [*:0]const u8) ?*protoent {
    const name_slice = std.mem.span(name);
    return lookupProtocol(name_slice, null);
}

pub export fn getprotobynumber(proto: c_int) ?*protoent {
    return lookupProtocol(null, proto);
}

pub export fn getprotoent() ?*protoent { return null; } // Iterator not implemented
pub export fn setprotoent(stayopen: c_int) void { _ = stayopen; }
pub export fn endprotoent() void {}

// Host functions
pub export fn sethostent(stayopen: c_int) void { _ = stayopen; }
pub export fn endhostent() void {}
pub export fn gethostent() ?*hostent { return null; }
pub export fn herror(s: [*:0]const u8) void { _ = s; }
pub export fn hstrerror(err: c_int) [*:0]const u8 {
    return switch (err) {
        0 => "Resolver Error 0 (no error)",
        HOST_NOT_FOUND => "Host not found",
        TRY_AGAIN => "Try again",
        NO_RECOVERY => "Non-recoverable error",
        NO_DATA => "No data",
        else => "Unknown resolver error",
    };
}