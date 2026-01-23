const std = @import("std");
const protocol = @import("protocol.zig");

/// HANA connection configuration
pub const HanaConnectionConfig = struct {
    host: []const u8,
    port: u16 = 443,  // Default for HANA Cloud
    user: []const u8,
    password: []const u8,
    schema: ?[]const u8 = null,
    database: ?[]const u8 = null,
    
    // TLS settings (required for HANA Cloud)
    use_tls: bool = true,
    validate_certificate: bool = true,
    
    // Connection timeouts
    connect_timeout_ms: u32 = 30000,
    read_timeout_ms: u32 = 60000,
    write_timeout_ms: u32 = 60000,
    
    // Connection options
    client_version: []const u8 = "nMetaData-1.0",
    client_type: []const u8 = "nMetaData",
    locale: []const u8 = "en_US",
    
    pub fn validate(self: HanaConnectionConfig) !void {
        if (self.host.len == 0) return error.InvalidHost;
        if (self.user.len == 0) return error.InvalidUser;
        if (self.password.len == 0) return error.InvalidPassword;
        if (self.port == 0) return error.InvalidPort;
        
        // HANA Cloud requires TLS
        if (std.mem.indexOf(u8, self.host, "hanacloud.ondemand.com") != null) {
            if (!self.use_tls) return error.TLSRequired;
        }
    }
};

/// HANA connection state
pub const HanaConnectionState = enum {
    disconnected,
    connecting,
    authenticating,
    connected,
    error_state,
    closing,
    
    pub fn isActive(self: HanaConnectionState) bool {
        return self == .connected;
    }
    
    pub fn canExecute(self: HanaConnectionState) bool {
        return self == .connected;
    }
};

/// HANA connection
pub const HanaConnection = struct {
    allocator: std.mem.Allocator,
    config: HanaConnectionConfig,
    state: HanaConnectionState,
    socket: ?std.net.Stream,
    session_id: i64,
    packet_count: u32,
    
    pub fn init(allocator: std.mem.Allocator, config: HanaConnectionConfig) !HanaConnection {
        try config.validate();
        
        return HanaConnection{
            .allocator = allocator,
            .config = config,
            .state = .disconnected,
            .socket = null,
            .session_id = 0,
            .packet_count = 0,
        };
    }
    
    pub fn deinit(self: *HanaConnection) void {
        if (self.state.isActive()) {
            self.disconnect();
        }
        if (self.socket) |sock| {
            sock.close();
            self.socket = null;
        }
    }
    
    /// Connect to HANA database
    pub fn connect(self: *HanaConnection) !void {
        if (self.state.isActive()) {
            return error.AlreadyConnected;
        }
        
        self.state = .connecting;
        errdefer self.state = .error_state;
        
        // Parse host and port
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        
        // Establish TCP connection
        const stream = try std.net.tcpConnectToAddress(address);
        errdefer stream.close();
        
        self.socket = stream;
        
        // For HANA Cloud, TLS would be initiated here
        // In real implementation: try self.initiateTLS();
        
        self.state = .authenticating;
        
        // Send CONNECT message
        try self.sendConnect();
        
        // Receive CONNECT response
        try self.receiveConnectResponse();
        
        self.state = .connected;
    }
    
    /// Disconnect from HANA database
    pub fn disconnect(self: *HanaConnection) void {
        if (self.state == .disconnected) {
            return;
        }
        
        self.state = .closing;
        
        // Send DISCONNECT message (best effort)
        self.sendDisconnect() catch {};
        
        if (self.socket) |sock| {
            sock.close();
            self.socket = null;
        }
        
        self.state = .disconnected;
    }
    
    /// Check if connection is active
    pub fn isConnected(self: HanaConnection) bool {
        return self.state.isActive() and self.socket != null;
    }
    
    /// Send CONNECT message
    fn sendConnect(self: *HanaConnection) !void {
        const socket = self.socket orelse return error.NotConnected;
        
        // Build CONNECT segment
        var segment = protocol.SegmentHeader.init(.connect, 1);
        
        // In real implementation, would:
        // 1. Add ConnectOptions part
        // 2. Calculate segment length
        // 3. Encode and send
        
        var buffer: [1024]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buffer);
        try segment.encode(fbs.writer());
        
        // Send to socket (simplified)
        _ = try socket.write(fbs.getWritten());
        
        self.packet_count += 1;
    }
    
    /// Receive CONNECT response
    fn receiveConnectResponse(self: *HanaConnection) !void {
        const socket = self.socket orelse return error.NotConnected;
        
        // Read segment header
        var header_buf: [protocol.SEGMENT_HEADER_SIZE]u8 = undefined;
        _ = try socket.readAll(&header_buf);
        
        var fbs = std.io.fixedBufferStream(&header_buf);
        const segment = try protocol.SegmentHeader.decode(fbs.reader());
        
        // Verify it's a response to CONNECT
        if (segment.message_type != .connect) {
            return error.UnexpectedMessageType;
        }
        
        // In real implementation, would:
        // 1. Read and parse all parts
        // 2. Extract session_id
        // 3. Handle any errors
        
        self.session_id = 1; // Placeholder
    }
    
    /// Send DISCONNECT message
    fn sendDisconnect(self: *HanaConnection) !void {
        const socket = self.socket orelse return error.NotConnected;
        
        var segment = protocol.SegmentHeader.init(.disconnect, 0);
        
        var buffer: [protocol.SEGMENT_HEADER_SIZE]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buffer);
        try segment.encode(fbs.writer());
        
        _ = try socket.write(fbs.getWritten());
    }
    
    /// Send a raw segment
    pub fn sendSegment(self: *HanaConnection, segment: protocol.SegmentHeader) !void {
        if (!self.isConnected()) {
            return error.NotConnected;
        }
        
        const socket = self.socket.?;
        
        var buffer: [4096]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buffer);
        try segment.encode(fbs.writer());
        
        _ = try socket.write(fbs.getWritten());
        self.packet_count += 1;
    }
    
    /// Receive a raw segment
    pub fn receiveSegment(self: *HanaConnection) !protocol.SegmentHeader {
        if (!self.isConnected()) {
            return error.NotConnected;
        }
        
        const socket = self.socket.?;
        
        var header_buf: [protocol.SEGMENT_HEADER_SIZE]u8 = undefined;
        _ = try socket.readAll(&header_buf);
        
        var fbs = std.io.fixedBufferStream(&header_buf);
        return try protocol.SegmentHeader.decode(fbs.reader());
    }
    
    /// Get connection state
    pub fn getState(self: HanaConnection) HanaConnectionState {
        return self.state;
    }
    
    /// Get session ID
    pub fn getSessionId(self: HanaConnection) i64 {
        return self.session_id;
    }
    
    /// Get packet count
    pub fn getPacketCount(self: HanaConnection) u32 {
        return self.packet_count;
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "HanaConnectionConfig - validation" {
    // Valid config
    const valid = HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "secret",
    };
    try valid.validate();
    
    // Invalid: empty host
    const invalid1 = HanaConnectionConfig{
        .host = "",
        .user = "DBADMIN",
        .password = "secret",
    };
    try std.testing.expectError(error.InvalidHost, invalid1.validate());
    
    // Invalid: empty user
    const invalid2 = HanaConnectionConfig{
        .host = "localhost",
        .user = "",
        .password = "secret",
    };
    try std.testing.expectError(error.InvalidUser, invalid2.validate());
}

test "HanaConnectionConfig - HANA Cloud TLS requirement" {
    const cloud_config = HanaConnectionConfig{
        .host = "xxx.hanacloud.ondemand.com",
        .user = "DBADMIN",
        .password = "secret",
        .use_tls = false,
    };
    
    try std.testing.expectError(error.TLSRequired, cloud_config.validate());
}

test "HanaConnectionState - isActive" {
    try std.testing.expect(!HanaConnectionState.disconnected.isActive());
    try std.testing.expect(!HanaConnectionState.connecting.isActive());
    try std.testing.expect(!HanaConnectionState.authenticating.isActive());
    try std.testing.expect(HanaConnectionState.connected.isActive());
    try std.testing.expect(!HanaConnectionState.error_state.isActive());
}

test "HanaConnectionState - canExecute" {
    try std.testing.expect(!HanaConnectionState.disconnected.canExecute());
    try std.testing.expect(HanaConnectionState.connected.canExecute());
    try std.testing.expect(!HanaConnectionState.error_state.canExecute());
}

test "HanaConnection - init and deinit" {
    const allocator = std.testing.allocator;
    
    const config = HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "secret",
    };
    
    var conn = try HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    try std.testing.expectEqual(HanaConnectionState.disconnected, conn.state);
    try std.testing.expectEqual(@as(i64, 0), conn.session_id);
    try std.testing.expectEqual(@as(u32, 0), conn.packet_count);
}

test "HanaConnection - state tracking" {
    const allocator = std.testing.allocator;
    
    const config = HanaConnectionConfig{
        .host = "localhost",
        .user = "DBADMIN",
        .password = "secret",
    };
    
    var conn = try HanaConnection.init(allocator, config);
    defer conn.deinit();
    
    try std.testing.expectEqual(HanaConnectionState.disconnected, conn.getState());
    try std.testing.expect(!conn.isConnected());
    try std.testing.expectEqual(@as(i64, 0), conn.getSessionId());
    try std.testing.expectEqual(@as(u32, 0), conn.getPacketCount());
}
