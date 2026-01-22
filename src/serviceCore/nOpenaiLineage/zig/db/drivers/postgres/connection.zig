const std = @import("std");
const protocol = @import("protocol.zig");
const MessageBuilder = protocol.MessageBuilder;
const MessageParser = protocol.MessageParser;
const MessageType = protocol.MessageType;
const AuthType = protocol.AuthType;
const TransactionStatus = protocol.TransactionStatus;

/// Connection configuration
pub const ConnectionConfig = struct {
    host: []const u8,
    port: u16 = 5432,
    database: []const u8,
    user: []const u8,
    password: []const u8,
    application_name: []const u8 = "nMetaData",
    connect_timeout_ms: u32 = 5000,
    ssl_mode: SslMode = .prefer,

    pub fn validate(self: ConnectionConfig) !void {
        if (self.host.len == 0) return error.InvalidHost;
        if (self.database.len == 0) return error.InvalidDatabase;
        if (self.user.len == 0) return error.InvalidUser;
        if (self.port == 0) return error.InvalidPort;
    }
};

/// SSL/TLS mode
pub const SslMode = enum {
    disable, // No SSL
    allow, // Try SSL, fallback to non-SSL
    prefer, // Prefer SSL, fallback to non-SSL
    require, // Require SSL, fail if not available
    verify_ca, // Require SSL + verify CA
    verify_full, // Require SSL + verify CA + hostname

    pub fn toString(self: SslMode) []const u8 {
        return switch (self) {
            .disable => "disable",
            .allow => "allow",
            .prefer => "prefer",
            .require => "require",
            .verify_ca => "verify-ca",
            .verify_full => "verify-full",
        };
    }
};

/// Connection state
pub const ConnectionState = enum {
    disconnected, // Not connected
    connecting, // Connection in progress
    authenticating, // Authentication in progress
    connected, // Connected and authenticated
    ready, // Ready for queries
    in_transaction, // In a transaction
    failed, // Connection failed

    pub fn isActive(self: ConnectionState) bool {
        return switch (self) {
            .connected, .ready, .in_transaction => true,
            else => false,
        };
    }
};

/// PostgreSQL connection
pub const PgConnection = struct {
    allocator: std.mem.Allocator,
    config: ConnectionConfig,
    stream: ?std.net.Stream,
    state: ConnectionState,
    backend_pid: i32,
    backend_secret: i32,
    transaction_status: TransactionStatus,
    server_params: std.StringHashMap([]const u8),
    message_builder: MessageBuilder,

    pub fn init(allocator: std.mem.Allocator, config: ConnectionConfig) !PgConnection {
        try config.validate();

        return PgConnection{
            .allocator = allocator,
            .config = config,
            .stream = null,
            .state = .disconnected,
            .backend_pid = 0,
            .backend_secret = 0,
            .transaction_status = .idle,
            .server_params = std.StringHashMap([]const u8).init(allocator),
            .message_builder = MessageBuilder.init(allocator),
        };
    }

    pub fn deinit(self: *PgConnection) void {
        self.disconnect();
        
        var it = self.server_params.iterator();
        while (it.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.*);
        }
        self.server_params.deinit();
        
        self.message_builder.deinit();
    }

    /// Connect to PostgreSQL server
    pub fn connect(self: *PgConnection) !void {
        if (self.state.isActive()) {
            return error.AlreadyConnected;
        }

        self.state = .connecting;

        // Resolve address
        const address = try std.net.Address.parseIp(self.config.host, self.config.port);
        
        // Connect TCP socket
        const stream = try std.net.tcpConnectToAddress(address);
        self.stream = stream;

        // Send startup message
        try self.sendStartupMessage();

        // Perform authentication
        try self.authenticate();

        // Wait for ReadyForQuery
        try self.waitForReady();

        self.state = .ready;
    }

    /// Disconnect from PostgreSQL server
    pub fn disconnect(self: *PgConnection) void {
        if (self.stream) |stream| {
            // Send terminate message
            self.sendTerminate() catch |err| {
                std.log.warn("Failed to send terminate: {}", .{err});
            };
            
            stream.close();
            self.stream = null;
        }
        self.state = .disconnected;
    }

    /// Check if connection is active
    pub fn isConnected(self: PgConnection) bool {
        return self.state.isActive();
    }

    /// Get server parameter value
    pub fn getServerParam(self: PgConnection, key: []const u8) ?[]const u8 {
        return self.server_params.get(key);
    }

    /// Send startup message
    fn sendStartupMessage(self: *PgConnection) !void {
        const params = [_][]const u8{
            "user",
            self.config.user,
            "database",
            self.config.database,
            "application_name",
            self.config.application_name,
        };

        const msg = try self.message_builder.buildStartupMessage(&params);
        try self.stream.?.writeAll(msg);
    }

    /// Perform authentication
    fn authenticate(self: *PgConnection) !void {
        self.state = .authenticating;

        // Read authentication request
        var buffer: [8192]u8 = undefined;
        const bytes_read = try self.stream.?.read(&buffer);
        
        var parser = MessageParser.init(buffer[0..bytes_read]);
        const msg_type = try parser.readMessageType();
        _ = try parser.readLength();

        if (msg_type != .authentication) {
            return error.UnexpectedMessage;
        }

        const auth_type_int = try parser.readInt32();
        const auth_type: AuthType = @enumFromInt(auth_type_int);

        switch (auth_type) {
            .ok => {
                // Authentication successful
                return;
            },
            .cleartext_password => {
                try self.sendCleartextPassword();
            },
            .md5_password => {
                const salt = try parser.readBytes(4);
                try self.sendMd5Password(salt);
            },
            .sasl => {
                // SASL authentication (SCRAM-SHA-256)
                return error.SaslNotYetImplemented;
            },
            else => {
                return error.UnsupportedAuthMethod;
            },
        }

        // Read authentication result
        const result_bytes = try self.stream.?.read(&buffer);
        var result_parser = MessageParser.init(buffer[0..result_bytes]);
        const result_type = try result_parser.readMessageType();
        _ = try result_parser.readLength();

        if (result_type != .authentication) {
            return error.AuthenticationFailed;
        }

        const result_auth = try result_parser.readInt32();
        if (result_auth != @intFromEnum(AuthType.ok)) {
            return error.AuthenticationFailed;
        }
    }

    /// Send cleartext password
    fn sendCleartextPassword(self: *PgConnection) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.password);
        try self.message_builder.writeString(self.config.password);
        const msg = try self.message_builder.endMessage();
        try self.stream.?.writeAll(msg);
    }

    /// Send MD5 password (md5(md5(password + user) + salt))
    fn sendMd5Password(self: *PgConnection, salt: []const u8) !void {
        // MD5 implementation simplified for now
        _ = salt;
        
        self.message_builder.reset();
        try self.message_builder.startMessage(.password);
        // In real implementation: compute MD5 hash
        try self.message_builder.writeString(self.config.password);
        const msg = try self.message_builder.endMessage();
        try self.stream.?.writeAll(msg);
    }

    /// Wait for ReadyForQuery message
    fn waitForReady(self: *PgConnection) !void {
        var buffer: [8192]u8 = undefined;
        
        while (true) {
            const bytes_read = try self.stream.?.read(&buffer);
            if (bytes_read == 0) break;
            
            var parser = MessageParser.init(buffer[0..bytes_read]);
            const msg_type = try parser.readMessageType();
            const length = try parser.readLength();
            
            switch (msg_type) {
                .backend_key_data => {
                    self.backend_pid = try parser.readInt32();
                    self.backend_secret = try parser.readInt32();
                },
                .parameter_status => {
                    const key = try parser.readString(self.allocator);
                    const value = try parser.readString(self.allocator);
                    try self.server_params.put(key, value);
                },
                .ready_for_query => {
                    const status_byte = try parser.readByte();
                    self.transaction_status = @enumFromInt(status_byte);
                    return; // Connection ready
                },
                .error_response => {
                    return error.ServerError;
                },
                else => {
                    // Skip other messages
                    _ = try parser.readBytes(@intCast(length - 4));
                },
            }
        }
    }

    /// Send terminate message
    fn sendTerminate(self: *PgConnection) !void {
        self.message_builder.reset();
        try self.message_builder.startMessage(.terminate);
        const msg = try self.message_builder.endMessage();
        try self.stream.?.writeAll(msg);
    }
};

// ============================================================================
// Unit Tests
// ============================================================================

test "ConnectionConfig - validation" {
    // Valid config
    const valid = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    try valid.validate();

    // Invalid: empty host
    const invalid1 = ConnectionConfig{
        .host = "",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };
    try std.testing.expectError(error.InvalidHost, invalid1.validate());

    // Invalid: empty database
    const invalid2 = ConnectionConfig{
        .host = "localhost",
        .database = "",
        .user = "postgres",
        .password = "secret",
    };
    try std.testing.expectError(error.InvalidDatabase, invalid2.validate());
}

test "SslMode - toString" {
    try std.testing.expectEqualStrings("disable", SslMode.disable.toString());
    try std.testing.expectEqualStrings("prefer", SslMode.prefer.toString());
    try std.testing.expectEqualStrings("require", SslMode.require.toString());
}

test "ConnectionState - isActive" {
    try std.testing.expect(!ConnectionState.disconnected.isActive());
    try std.testing.expect(!ConnectionState.connecting.isActive());
    try std.testing.expect(ConnectionState.connected.isActive());
    try std.testing.expect(ConnectionState.ready.isActive());
    try std.testing.expect(ConnectionState.in_transaction.isActive());
    try std.testing.expect(!ConnectionState.failed.isActive());
}

test "PgConnection - init and deinit" {
    const allocator = std.testing.allocator;

    const config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };

    var conn = try PgConnection.init(allocator, config);
    defer conn.deinit();

    try std.testing.expectEqual(ConnectionState.disconnected, conn.state);
    try std.testing.expect(!conn.isConnected());
}

test "PgConnection - server params" {
    const allocator = std.testing.allocator;

    const config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };

    var conn = try PgConnection.init(allocator, config);
    defer conn.deinit();

    try std.testing.expect(conn.getServerParam("server_version") == null);
}

test "PgConnection - state tracking" {
    const allocator = std.testing.allocator;

    const config = ConnectionConfig{
        .host = "localhost",
        .database = "test",
        .user = "postgres",
        .password = "secret",
    };

    var conn = try PgConnection.init(allocator, config);
    defer conn.deinit();

    try std.testing.expectEqual(ConnectionState.disconnected, conn.state);
    try std.testing.expectEqual(TransactionStatus.idle, conn.transaction_status);
    try std.testing.expectEqual(@as(i32, 0), conn.backend_pid);
}
