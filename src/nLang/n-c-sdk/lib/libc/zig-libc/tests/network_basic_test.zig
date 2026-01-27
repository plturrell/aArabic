//! Basic Network Functions Tests
//! Quick validation of core networking implementations

const std = @import("std");
const testing = std.testing;
const network = @import("../src/network/lib.zig");

// ============================================================================
// Byte Order Conversion Tests
// ============================================================================

test "htons/ntohs round-trip" {
    const host_value: u16 = 0x1234;
    const net_value = network.htons(host_value);
    const back_value = network.ntohs(net_value);
    try testing.expectEqual(host_value, back_value);
}

test "htonl/ntohl round-trip" {
    const host_value: u32 = 0x12345678;
    const net_value = network.htonl(host_value);
    const back_value = network.ntohl(net_value);
    try testing.expectEqual(host_value, back_value);
}

test "htonll/ntohll round-trip" {
    const host_value: u64 = 0x123456789ABCDEF0;
    const net_value = network.htonll(host_value);
    const back_value = network.ntohll(net_value);
    try testing.expectEqual(host_value, back_value);
}

test "byte order conversion on big-endian returns unchanged" {
    // This is more of a documentation test
    // On big-endian systems, htonX should be no-op
    const value: u32 = 0x12345678;
    const converted = network.htonl(value);
    
    if (std.builtin.cpu.arch.endian() == .big) {
        try testing.expectEqual(value, converted);
    } else {
        try testing.expect(value != converted);
    }
}

// ============================================================================
// IPv4 Address Conversion Tests
// ============================================================================

test "inet_addr parses valid IPv4" {
    const addr = network.inet_addr("192.168.1.1");
    try testing.expect(addr != network.INADDR_NONE);
}

test "inet_addr rejects invalid IPv4" {
    const addr = network.inet_addr("999.999.999.999");
    try testing.expectEqual(network.INADDR_NONE, addr);
}

test "inet_addr handles localhost" {
    const addr = network.inet_addr("127.0.0.1");
    try testing.expect(addr != network.INADDR_NONE);
}

test "inet_aton round-trip" {
    const test_addr = "10.0.0.1";
    var in: network.in_addr = undefined;
    const result = network.inet_aton(test_addr, &in);
    
    try testing.expectEqual(@as(c_int, 1), result);
    
    // Convert back and compare
    const str = network.inet_ntoa(in);
    const str_slice = std.mem.span(str);
    try testing.expectEqualStrings(test_addr, str_slice);
}

test "inet_pton IPv4 basic" {
    var addr: u32 = undefined;
    const result = network.inet_pton(network.AF_INET, "192.168.1.1", &addr);
    try testing.expectEqual(@as(c_int, 1), result);
}

test "inet_pton IPv4 invalid" {
    var addr: u32 = undefined;
    const result = network.inet_pton(network.AF_INET, "invalid", &addr);
    try testing.expectEqual(@as(c_int, 0), result);
}

test "inet_ntop IPv4 basic" {
    const addr: u32 = network.htonl(0xC0A80101); // 192.168.1.1
    var buffer: [16]u8 = undefined;
    
    const result = network.inet_ntop(network.AF_INET, &addr, &buffer, 16);
    try testing.expect(result != null);
    
    const str = std.mem.span(@as([*:0]u8, @ptrCast(result.?)));
    try testing.expectEqualStrings("192.168.1.1", str);
}

// ============================================================================
// IPv4 Address Helper Tests
// ============================================================================

test "is_loopback_ipv4 detects loopback" {
    const loopback = network.htonl(0x7F000001); // 127.0.0.1
    try testing.expectEqual(@as(c_int, 1), network.is_loopback_ipv4(loopback));
    
    const not_loopback = network.htonl(0xC0A80101); // 192.168.1.1
    try testing.expectEqual(@as(c_int, 0), network.is_loopback_ipv4(not_loopback));
}

test "is_private_ipv4 detects private addresses" {
    // 10.0.0.0/8
    const private1 = network.htonl(0x0A000001);
    try testing.expectEqual(@as(c_int, 1), network.is_private_ipv4(private1));
    
    // 172.16.0.0/12
    const private2 = network.htonl(0xAC100001);
    try testing.expectEqual(@as(c_int, 1), network.is_private_ipv4(private2));
    
    // 192.168.0.0/16
    const private3 = network.htonl(0xC0A80001);
    try testing.expectEqual(@as(c_int, 1), network.is_private_ipv4(private3));
    
    // Public address
    const public = network.htonl(0x08080808); // 8.8.8.8
    try testing.expectEqual(@as(c_int, 0), network.is_private_ipv4(public));
}

test "is_multicast_ipv4 detects multicast" {
    const multicast = network.htonl(0xE0000001); // 224.0.0.1
    try testing.expectEqual(@as(c_int, 1), network.is_multicast_ipv4(multicast));
    
    const unicast = network.htonl(0xC0A80101);
    try testing.expectEqual(@as(c_int, 0), network.is_multicast_ipv4(unicast));
}

// ============================================================================
// FD_SET Operations Tests
// ============================================================================

test "FD_ZERO clears all bits" {
    var fds: network.fd_set = undefined;
    network.FD_ZERO(&fds);
    
    // Verify all bits are zero
    for (fds.fds_bits) |word| {
        try testing.expectEqual(@as(u32, 0), word);
    }
}

test "FD_SET and FD_ISSET work correctly" {
    var fds: network.fd_set = undefined;
    network.FD_ZERO(&fds);
    
    const test_fd: c_int = 42;
    network.FD_SET(test_fd, &fds);
    
    const is_set = network.FD_ISSET(test_fd, &fds);
    try testing.expectEqual(@as(c_int, 1), is_set);
    
    const not_set = network.FD_ISSET(100, &fds);
    try testing.expectEqual(@as(c_int, 0), not_set);
}

test "FD_CLR removes file descriptor" {
    var fds: network.fd_set = undefined;
    network.FD_ZERO(&fds);
    
    const test_fd: c_int = 42;
    network.FD_SET(test_fd, &fds);
    try testing.expectEqual(@as(c_int, 1), network.FD_ISSET(test_fd, &fds));
    
    network.FD_CLR(test_fd, &fds);
    try testing.expectEqual(@as(c_int, 0), network.FD_ISSET(test_fd, &fds));
}

// ============================================================================
// Socket Constants Tests
// ============================================================================

test "address family constants are defined" {
    try testing.expect(network.AF_INET > 0);
    try testing.expect(network.AF_INET6 > 0);
    try testing.expect(network.AF_UNSPEC == 0);
}

test "socket type constants are defined" {
    try testing.expect(network.SOCK_STREAM > 0);
    try testing.expect(network.SOCK_DGRAM > 0);
    try testing.expect(network.SOCK_RAW > 0);
}

test "protocol constants are defined" {
    try testing.expect(network.IPPROTO_TCP > 0);
    try testing.expect(network.IPPROTO_UDP > 0);
    try testing.expect(network.IPPROTO_IP == 0);
}

test "special addresses are correct" {
    try testing.expectEqual(@as(u32, 0x00000000), network.INADDR_ANY);
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), network.INADDR_BROADCAST);
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), network.INADDR_NONE);
}

// ============================================================================
// Poll Constants Tests
// ============================================================================

test "poll event constants are defined" {
    try testing.expect(network.POLLIN != 0);
    try testing.expect(network.POLLOUT != 0);
    try testing.expect(network.POLLERR != 0);
    try testing.expect(network.POLLHUP != 0);
}

// ============================================================================
// Epoll Constants Tests (Linux-specific)
// ============================================================================

test "epoll event constants are defined" {
    try testing.expect(network.EPOLLIN != 0);
    try testing.expect(network.EPOLLOUT != 0);
    try testing.expect(network.EPOLLET != 0);
}

test "epoll operation constants are defined" {
    try testing.expectEqual(@as(u32, 1), network.EPOLL_CTL_ADD);
    try testing.expectEqual(@as(u32, 2), network.EPOLL_CTL_DEL);
    try testing.expectEqual(@as(u32, 3), network.EPOLL_CTL_MOD);
}

// ============================================================================
// Message Flag Tests
// ============================================================================

test "message flag constants are defined" {
    try testing.expect(network.MSG_OOB != 0);
    try testing.expect(network.MSG_PEEK != 0);
    try testing.expect(network.MSG_WAITALL != 0);
    try testing.expect(network.MSG_DONTWAIT != 0);
}

test "shutdown mode constants are defined" {
    try testing.expectEqual(@as(c_int, 0), network.SHUT_RD);
    try testing.expectEqual(@as(c_int, 1), network.SHUT_WR);
    try testing.expectEqual(@as(c_int, 2), network.SHUT_RDWR);
}

// ============================================================================
// Summary
// ============================================================================

// This test file validates:
// ✅ Byte order conversion (hton*/ntoh*)
// ✅ IPv4 address parsing (inet_addr, inet_aton)
// ✅ IPv4 address formatting (inet_ntoa, inet_ntop)
// ✅ IPv4 address helpers (is_loopback, is_private, etc.)
// ✅ FD_SET operations (FD_ZERO, FD_SET, FD_CLR, FD_ISSET)
// ✅ All networking constants
// ✅ Poll/epoll constants
// ✅ Message and shutdown flags

// Integration tests (require actual sockets) should be in separate file
// These tests validate the utility functions and constants
