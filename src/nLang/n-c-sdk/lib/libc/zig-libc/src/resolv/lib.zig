// resolv module - DNS resolver - Phase 1.32
const std = @import("std");

pub const RES_INIT: c_int = 0x00000001;
pub const RES_DEBUG: c_int = 0x00000002;
pub const RES_AAONLY: c_int = 0x00000004;
pub const RES_USEVC: c_int = 0x00000008;

// DNS record types
pub const NS_T_A: c_int = 1;
pub const NS_T_AAAA: c_int = 28;
pub const NS_T_CNAME: c_int = 5;

// DNS classes
pub const NS_C_IN: c_int = 1;

pub export fn res_init() c_int {
    return 0;
}

/// Query DNS for a domain name. Builds a minimal DNS response in the answer buffer.
/// Returns the length of the response, or -1 on error.
pub export fn res_query(dname: [*:0]const u8, class: c_int, type_: c_int, answer: [*]u8, anslen: c_int) c_int {
    if (anslen < 12) return -1;

    const name = std.mem.span(dname);
    if (name.len == 0 or name.len > 253) return -1;

    // Use Zig's std.net to resolve the hostname
    const addresses = std.net.getAddressList(std.heap.page_allocator, name, 0) catch {
        return -1;
    };
    defer addresses.deinit();

    if (addresses.addrs.len == 0) return -1;

    // Build a minimal DNS response
    const answer_slice = answer[0..@intCast(anslen)];
    return buildDnsResponse(answer_slice, name, class, type_, addresses.addrs) catch -1;
}

/// Build a DNS response packet
fn buildDnsResponse(buf: []u8, name: []const u8, class: c_int, type_: c_int, addrs: []const std.net.Address) !c_int {
    if (buf.len < 12) return error.BufferTooSmall;

    var pos: usize = 0;

    // DNS Header (12 bytes)
    // ID (2 bytes) - use 0
    buf[pos] = 0;
    buf[pos + 1] = 0;
    pos += 2;

    // Flags (2 bytes): QR=1 (response), OPCODE=0, AA=0, TC=0, RD=1, RA=1, Z=0, RCODE=0
    buf[pos] = 0x81; // QR=1, RD=1
    buf[pos + 1] = 0x80; // RA=1
    pos += 2;

    // QDCOUNT (2 bytes) - 1 question
    buf[pos] = 0;
    buf[pos + 1] = 1;
    pos += 2;

    // Count matching answers
    var answer_count: u16 = 0;
    for (addrs) |addr| {
        if (type_ == NS_T_A and addr.any.family == std.posix.AF.INET) {
            answer_count += 1;
        } else if (type_ == NS_T_AAAA and addr.any.family == std.posix.AF.INET6) {
            answer_count += 1;
        }
    }

    // ANCOUNT (2 bytes)
    buf[pos] = @intCast(answer_count >> 8);
    buf[pos + 1] = @intCast(answer_count & 0xFF);
    pos += 2;

    // NSCOUNT, ARCOUNT (4 bytes) - 0
    buf[pos] = 0;
    buf[pos + 1] = 0;
    buf[pos + 2] = 0;
    buf[pos + 3] = 0;
    pos += 4;

    // Question section
    const qname_start = pos;
    pos = try encodeDomainName(buf, pos, name);

    // QTYPE (2 bytes)
    if (pos + 4 > buf.len) return error.BufferTooSmall;
    buf[pos] = @intCast(@as(u32, @bitCast(type_)) >> 8);
    buf[pos + 1] = @intCast(@as(u32, @bitCast(type_)) & 0xFF);
    pos += 2;

    // QCLASS (2 bytes)
    buf[pos] = @intCast(@as(u32, @bitCast(class)) >> 8);
    buf[pos + 1] = @intCast(@as(u32, @bitCast(class)) & 0xFF);
    pos += 2;

    // Answer section
    for (addrs) |addr| {
        if (type_ == NS_T_A and addr.any.family == std.posix.AF.INET) {
            // Use pointer compression to reference the question name
            if (pos + 16 > buf.len) return error.BufferTooSmall;

            // NAME - pointer to qname (0xC0 | offset)
            buf[pos] = 0xC0;
            buf[pos + 1] = @intCast(qname_start);
            pos += 2;

            // TYPE (2 bytes)
            buf[pos] = 0;
            buf[pos + 1] = NS_T_A;
            pos += 2;

            // CLASS (2 bytes)
            buf[pos] = 0;
            buf[pos + 1] = NS_C_IN;
            pos += 2;

            // TTL (4 bytes) - 300 seconds
            buf[pos] = 0;
            buf[pos + 1] = 0;
            buf[pos + 2] = 0x01;
            buf[pos + 3] = 0x2C;
            pos += 4;

            // RDLENGTH (2 bytes) - 4 for IPv4
            buf[pos] = 0;
            buf[pos + 1] = 4;
            pos += 2;

            // RDATA - IPv4 address
            const ip4 = @as(*const [4]u8, @ptrCast(&addr.in.sa.addr));
            buf[pos] = ip4[0];
            buf[pos + 1] = ip4[1];
            buf[pos + 2] = ip4[2];
            buf[pos + 3] = ip4[3];
            pos += 4;
        } else if (type_ == NS_T_AAAA and addr.any.family == std.posix.AF.INET6) {
            if (pos + 28 > buf.len) return error.BufferTooSmall;

            // NAME - pointer to qname
            buf[pos] = 0xC0;
            buf[pos + 1] = @intCast(qname_start);
            pos += 2;

            // TYPE (2 bytes)
            buf[pos] = 0;
            buf[pos + 1] = NS_T_AAAA;
            pos += 2;

            // CLASS (2 bytes)
            buf[pos] = 0;
            buf[pos + 1] = NS_C_IN;
            pos += 2;

            // TTL (4 bytes)
            buf[pos] = 0;
            buf[pos + 1] = 0;
            buf[pos + 2] = 0x01;
            buf[pos + 3] = 0x2C;
            pos += 4;

            // RDLENGTH (2 bytes) - 16 for IPv6
            buf[pos] = 0;
            buf[pos + 1] = 16;
            pos += 2;

            // RDATA - IPv6 address
            const ip6 = &addr.in6.sa.addr;
            @memcpy(buf[pos..][0..16], ip6);
            pos += 16;
        }
    }

    return @intCast(pos);
}

/// Encode a domain name in DNS wire format (length-prefixed labels)
fn encodeDomainName(buf: []u8, start_pos: usize, name: []const u8) !usize {
    var pos = start_pos;
    var label_start: usize = 0;

    for (name, 0..) |c, i| {
        if (c == '.') {
            const label_len = i - label_start;
            if (label_len > 63 or label_len == 0) return error.InvalidName;
            if (pos + 1 + label_len > buf.len) return error.BufferTooSmall;

            buf[pos] = @intCast(label_len);
            pos += 1;
            @memcpy(buf[pos..][0..label_len], name[label_start..i]);
            pos += label_len;
            label_start = i + 1;
        }
    }

    // Handle last label (or only label if no dots)
    const label_len = name.len - label_start;
    if (label_len > 0) {
        if (label_len > 63) return error.InvalidName;
        if (pos + 1 + label_len > buf.len) return error.BufferTooSmall;

        buf[pos] = @intCast(label_len);
        pos += 1;
        @memcpy(buf[pos..][0..label_len], name[label_start..]);
        pos += label_len;
    }

    // Null terminator
    if (pos >= buf.len) return error.BufferTooSmall;
    buf[pos] = 0;
    pos += 1;

    return pos;
}

/// Like res_query but searches domain list. For basic implementation, delegates to res_query.
pub export fn res_search(dname: [*:0]const u8, class: c_int, type_: c_int, answer: [*]u8, anslen: c_int) c_int {
    return res_query(dname, class, type_, answer, anslen);
}

/// Query with domain suffix appended
pub export fn res_querydomain(name: [*:0]const u8, domain: [*:0]const u8, class: c_int, type_: c_int, answer: [*]u8, anslen: c_int) c_int {
    const name_str = std.mem.span(name);
    const domain_str = std.mem.span(domain);

    // Build full name: name.domain
    var full_name: [512]u8 = undefined;
    var pos: usize = 0;

    if (name_str.len > 0) {
        const copy_len = @min(name_str.len, 250);
        @memcpy(full_name[0..copy_len], name_str[0..copy_len]);
        pos = copy_len;

        if (domain_str.len > 0) {
            full_name[pos] = '.';
            pos += 1;
        }
    }

    if (domain_str.len > 0) {
        const copy_len = @min(domain_str.len, 250 - pos);
        @memcpy(full_name[pos..][0..copy_len], domain_str[0..copy_len]);
        pos += copy_len;
    }

    full_name[pos] = 0;

    return res_query(@ptrCast(&full_name), class, type_, answer, anslen);
}

// DNS opcodes
pub const NS_O_QUERY: c_int = 0;
pub const NS_O_IQUERY: c_int = 1;
pub const NS_O_STATUS: c_int = 2;

/// Build a DNS query message
pub export fn res_mkquery(op: c_int, dname: [*:0]const u8, class: c_int, type_: c_int, data: ?[*]const u8, datalen: c_int, newrr: ?*const anyopaque, buf: [*]u8, buflen: c_int) c_int {
    _ = data;
    _ = datalen;
    _ = newrr;

    if (buflen < 12) return -1;
    if (op != NS_O_QUERY) return -1; // Only standard queries supported

    const name = std.mem.span(dname);
    if (name.len > 253) return -1;

    var pos: usize = 0;
    const buf_len: usize = @intCast(buflen);

    // DNS Header (12 bytes)
    // ID - random identifier
    const id = std.crypto.random.int(u16);
    buf[pos] = @intCast(id >> 8);
    buf[pos + 1] = @intCast(id & 0xFF);
    pos += 2;

    // Flags: QR=0 (query), OPCODE=0, RD=1 (recursion desired)
    buf[pos] = 0x01; // RD=1
    buf[pos + 1] = 0x00;
    pos += 2;

    // QDCOUNT = 1
    buf[pos] = 0;
    buf[pos + 1] = 1;
    pos += 2;

    // ANCOUNT, NSCOUNT, ARCOUNT = 0
    buf[pos] = 0;
    buf[pos + 1] = 0;
    buf[pos + 2] = 0;
    buf[pos + 3] = 0;
    buf[pos + 4] = 0;
    buf[pos + 5] = 0;
    pos += 6;

    // Question section - encode domain name
    pos = encodeDomainName(buf[0..buf_len], pos, name) catch return -1;

    // QTYPE
    if (pos + 4 > buf_len) return -1;
    buf[pos] = @intCast(@as(u32, @bitCast(type_)) >> 8);
    buf[pos + 1] = @intCast(@as(u32, @bitCast(type_)) & 0xFF);
    pos += 2;

    // QCLASS
    buf[pos] = @intCast(@as(u32, @bitCast(class)) >> 8);
    buf[pos + 1] = @intCast(@as(u32, @bitCast(class)) & 0xFF);
    pos += 2;

    return @intCast(pos);
}

/// Send a pre-built DNS query and receive response
pub export fn res_send(msg: [*]const u8, msglen: c_int, answer: [*]u8, anslen: c_int) c_int {
    if (msglen < 12 or anslen < 12) return -1;

    const msg_slice = msg[0..@intCast(msglen)];
    const ans_slice = answer[0..@intCast(anslen)];

    // Get DNS server from /etc/resolv.conf or use default
    const dns_server = getDnsServer() orelse "8.8.8.8";

    // Create UDP socket and send query
    const sock = std.posix.socket(std.posix.AF.INET, std.posix.SOCK.DGRAM, 0) catch return -1;
    defer std.posix.close(sock);

    // Set timeout
    const timeout = std.posix.timeval{ .sec = 5, .usec = 0 };
    std.posix.setsockopt(sock, std.posix.SOL.SOCKET, std.posix.SO.RCVTIMEO, std.mem.asBytes(&timeout)) catch {};

    // Parse DNS server address
    var addr: std.net.Address = undefined;
    addr = std.net.Address.parseIp4(dns_server, 53) catch return -1;

    // Send query
    _ = std.posix.sendto(sock, msg_slice, 0, &addr.any, addr.getOsSockLen()) catch return -1;

    // Receive response
    var from_addr: std.posix.sockaddr = undefined;
    var from_len: std.posix.socklen_t = @sizeOf(std.posix.sockaddr);
    const recv_len = std.posix.recvfrom(sock, ans_slice, 0, &from_addr, &from_len) catch return -1;

    return @intCast(recv_len);
}

fn getDnsServer() ?[]const u8 {
    const file = std.fs.openFileAbsolute("/etc/resolv.conf", .{}) catch return null;
    defer file.close();

    var buf: [1024]u8 = undefined;
    var line_buf: [256]u8 = undefined;

    const reader = file.reader();
    while (reader.readUntilDelimiterOrEof(&line_buf, '\n') catch return null) |line| {
        if (std.mem.startsWith(u8, line, "nameserver ")) {
            const server = std.mem.trim(u8, line[11..], " \t\r\n");
            if (server.len > 0 and server.len < buf.len) {
                @memcpy(buf[0..server.len], server);
                return buf[0..server.len];
            }
        }
    }
    return null;
}

/// Compress a domain name from dotted format to DNS wire format (length-prefixed labels).
/// Returns the length of the compressed name, or -1 on error.
/// Note: This basic implementation does not use pointer compression (dnptrs/lastdnptr are ignored).
pub export fn dn_comp(exp_dn: [*:0]const u8, comp_dn: [*]u8, length: c_int, dnptrs: ?[*][*]u8, lastdnptr: ?[*][*]u8) c_int {
    _ = dnptrs;
    _ = lastdnptr;

    if (length <= 0) return -1;

    const name = std.mem.span(exp_dn);
    const buf_len: usize = @intCast(length);

    // Handle empty name (root domain)
    if (name.len == 0) {
        comp_dn[0] = 0;
        return 1;
    }

    // Strip trailing dot if present
    var src_len = name.len;
    if (name[src_len - 1] == '.') {
        src_len -= 1;
    }

    if (src_len > 253) return -1;

    var pos: usize = 0;
    var label_start: usize = 0;

    var i: usize = 0;
    while (i < src_len) : (i += 1) {
        if (name[i] == '.') {
            const label_len = i - label_start;
            if (label_len > 63 or label_len == 0) return -1;
            if (pos + 1 + label_len >= buf_len) return -1;

            comp_dn[pos] = @intCast(label_len);
            pos += 1;
            @memcpy(comp_dn[pos..][0..label_len], name[label_start..i]);
            pos += label_len;
            label_start = i + 1;
        }
    }

    // Handle last label
    const label_len = src_len - label_start;
    if (label_len > 0) {
        if (label_len > 63) return -1;
        if (pos + 1 + label_len >= buf_len) return -1;

        comp_dn[pos] = @intCast(label_len);
        pos += 1;
        @memcpy(comp_dn[pos..][0..label_len], name[label_start..src_len]);
        pos += label_len;
    }

    // Null terminator
    if (pos >= buf_len) return -1;
    comp_dn[pos] = 0;
    pos += 1;

    return @intCast(pos);
}

/// Expand a compressed domain name from DNS wire format to dotted format.
/// Returns the length of the compressed name consumed, or -1 on error.
pub export fn dn_expand(msg: [*]const u8, eomorig: [*]const u8, comp_dn: [*]const u8, exp_dn: [*]u8, length: c_int) c_int {
    if (length <= 0) return -1;

    const base = msg;
    const end = eomorig;
    var p = comp_dn;

    // Calculate message size
    const msg_size = @intFromPtr(end) - @intFromPtr(base);
    if (msg_size <= 0) return -1;

    var dest = exp_dn;
    const dest_start = exp_dn;
    const space: usize = @intCast(length);
    const dest_end = exp_dn + if (space > 254) 254 else space;

    var len: c_int = -1;

    // Detect reference loop using an iteration counter
    var iterations: usize = 0;
    const max_iterations: usize = @intCast(msg_size);

    while (iterations < max_iterations) : (iterations += 2) {
        // Check if p is within bounds
        if (@intFromPtr(p) >= @intFromPtr(end)) return -1;

        if ((p[0] & 0xC0) != 0) {
            // Pointer compression
            if ((p[0] & 0xC0) != 0xC0) return -1; // Invalid compression

            if (@intFromPtr(p) + 1 >= @intFromPtr(end)) return -1;

            const offset = (@as(usize, p[0] & 0x3F) << 8) | @as(usize, p[1]);
            if (len < 0) {
                len = @intCast(@intFromPtr(p) + 2 - @intFromPtr(comp_dn));
            }
            if (offset >= msg_size) return -1;
            p = base + offset;
        } else if (p[0] != 0) {
            // Regular label
            if (dest != dest_start) {
                if (@intFromPtr(dest) >= @intFromPtr(dest_end)) return -1;
                dest[0] = '.';
                dest += 1;
            }

            const label_len: usize = p[0];
            p += 1;

            if (@intFromPtr(p) + label_len > @intFromPtr(end)) return -1;
            if (@intFromPtr(dest) + label_len > @intFromPtr(dest_end)) return -1;

            // Copy label
            var j: usize = 0;
            while (j < label_len) : (j += 1) {
                dest[j] = p[j];
            }
            dest += label_len;
            p += label_len;
        } else {
            // End of name (null label)
            dest[0] = 0;
            if (len < 0) {
                len = @intCast(@intFromPtr(p) + 1 - @intFromPtr(comp_dn));
            }
            return len;
        }
    }

    return -1;
}
