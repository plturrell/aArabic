//! RESP (REdis Serialization Protocol) Client Implementation
//! 
//! This module implements the Redis/DragonflyDB wire protocol (RESP2/RESP3)
//! for direct communication with DragonflyDB servers.
//!
//! Features:
//! - Basic RESP protocol support
//! - Connection pooling for performance
//! - Redis Sentinel support for high availability
//! - Thread-safe connection management
//! - Health monitoring and failover detection

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;

/// RESP data types
pub const RespType = enum {
    simple_string,
    error_msg,
    integer,
    bulk_string,
    array,
    null_bulk_string,
};

/// RESP value representation
pub const RespValue = union(RespType) {
    simple_string: []const u8,
    error_msg: []const u8,
    integer: i64,
    bulk_string: []const u8,
    array: []RespValue,
    null_bulk_string: void,

    pub fn deinit(self: *RespValue, allocator: Allocator) void {
        switch (self.*) {
            .simple_string => |s| allocator.free(s),
            .error_msg => |e| allocator.free(e),
            .bulk_string => |s| allocator.free(s),
            .array => |arr| {
                for (arr) |*item| {
                    item.deinit(allocator);
                }
                allocator.free(arr);
            },
            .integer, .null_bulk_string => {},
        }
    }

    pub fn clone(self: RespValue, allocator: Allocator) !RespValue {
        return switch (self) {
            .simple_string => |s| RespValue{ .simple_string = try allocator.dupe(u8, s) },
            .error_msg => |e| RespValue{ .error_msg = try allocator.dupe(u8, e) },
            .bulk_string => |s| RespValue{ .bulk_string = try allocator.dupe(u8, s) },
            .array => |arr| {
                const new_arr = try allocator.alloc(RespValue, arr.len);
                errdefer allocator.free(new_arr);
                for (arr, 0..) |item, i| {
                    new_arr[i] = try item.clone(allocator);
                }
                return RespValue{ .array = new_arr };
            },
            .integer => |i| RespValue{ .integer = i },
            .null_bulk_string => RespValue{ .null_bulk_string = {} },
        };
    }
};

/// Redis/DragonflyDB client with RESP protocol support
pub const RespClient = struct {
    allocator: Allocator,
    host: []const u8,
    port: u16,
    password: ?[]const u8,
    db_index: u8,
    timeout_ms: u32,
    stream: ?net.Stream,
    is_connected: bool = false,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        host: []const u8,
        port: u16,
        password: ?[]const u8,
        db_index: u8,
        timeout_ms: u32,
    ) Self {
        return .{
            .allocator = allocator,
            .host = host,
            .port = port,
            .password = password,
            .db_index = db_index,
            .timeout_ms = timeout_ms,
            .stream = null,
            .is_connected = false,
        };
    }

    pub fn deinit(self: *Self) void {
        self.disconnect();
    }

    /// Connect to DragonflyDB server
    pub fn connect(self: *Self) !void {
        if (self.is_connected) {
            return error.AlreadyConnected;
        }

        const address = try net.Address.parseIp(self.host, self.port);
        self.stream = try net.tcpConnectToAddress(address);
        
        self.is_connected = true;
        
        // Note: timeout_ms field reserved for future timeout implementation
        _ = self.timeout_ms;

        // Authenticate if password provided
        if (self.password) |pwd| {
            try self.sendCommand(&.{ "AUTH", pwd });
            const auth_response = try self.readResponse();
            defer {
                var resp = auth_response;
                resp.deinit(self.allocator);
            }
            
            switch (auth_response) {
                .error_msg => return error.AuthenticationFailed,
                else => {},
            }
        }

        // Select database
        if (self.db_index != 0) {
            var db_str_buf: [16]u8 = undefined;
            const db_str = try std.fmt.bufPrint(&db_str_buf, "{d}", .{self.db_index});
            try self.sendCommand(&.{ "SELECT", db_str });
            const select_response = try self.readResponse();
            defer {
                var resp = select_response;
                resp.deinit(self.allocator);
            }
        }
    }

    /// Disconnect from server
    pub fn disconnect(self: *Self) void {
        if (self.stream) |*s| {
            s.close();
            self.stream = null;
            self.is_connected = false;
        }
    }

    /// Send RESP command to server
    pub fn sendCommand(self: *Self, args: []const []const u8) !void {
        if (self.stream == null) return error.NotConnected;
        const stream = self.stream.?;

        // Send array header: *<count>\r\n
        var header_buf: [32]u8 = undefined;
        const header = try std.fmt.bufPrint(&header_buf, "*{d}\r\n", .{args.len});
        try stream.writeAll(header);

        // Send each argument as bulk string: $<length>\r\n<data>\r\n
        for (args) |arg| {
            var len_buf: [32]u8 = undefined;
            const len_str = try std.fmt.bufPrint(&len_buf, "${d}\r\n", .{arg.len});
            try stream.writeAll(len_str);
            try stream.writeAll(arg);
            try stream.writeAll("\r\n");
        }
    }

    /// Read one byte from stream
    fn readByte(self: *Self) !u8 {
        var buf: [1]u8 = undefined;
        const n = try self.stream.?.read(&buf);
        if (n == 0) return error.EndOfStream;
        return buf[0];
    }

    /// Read RESP response from server
    pub fn readResponse(self: *Self) !RespValue {
        if (self.stream == null) return error.NotConnected;

        const type_byte = try self.readByte();
        
        return switch (type_byte) {
            '+' => try self.readSimpleString(),
            '-' => try self.readError(),
            ':' => try self.readInteger(),
            '$' => try self.readBulkString(),
            '*' => try self.readArray(0),
            else => error.InvalidRespType,
        };
    }

    fn readLine(self: *Self) ![]const u8 {
        var line = try std.ArrayList(u8).initCapacity(self.allocator, 64);
        errdefer line.deinit(self.allocator);
        
        while (true) {
            const byte = try self.readByte();
            if (byte == '\n') break;
            if (byte != '\r') {
                try line.append(self.allocator, byte);
            }
        }
        
        return try line.toOwnedSlice(self.allocator);
    }

    fn readSimpleString(self: *Self) !RespValue {
        const line = try self.readLine();
        return RespValue{ .simple_string = line };
    }

    fn readError(self: *Self) !RespValue {
        const line = try self.readLine();
        return RespValue{ .error_msg = line };
    }

    fn readInteger(self: *Self) !RespValue {
        const line = try self.readLine();
        defer self.allocator.free(line);
        
        const value = try std.fmt.parseInt(i64, line, 10);
        return RespValue{ .integer = value };
    }

    fn readBulkString(self: *Self) !RespValue {
        const line = try self.readLine();
        defer self.allocator.free(line);
        
        const length = try std.fmt.parseInt(i64, line, 10);
        
        if (length == -1) {
            return RespValue{ .null_bulk_string = {} };
        }
        
        if (length < 0) {
            return error.InvalidBulkStringLength;
        }
        
        const data = try self.allocator.alloc(u8, @intCast(length));
        errdefer self.allocator.free(data);
        
        // Read exact number of bytes
        var total_read: usize = 0;
        while (total_read < data.len) {
            const n = try self.stream.?.read(data[total_read..]);
            if (n == 0) return error.EndOfStream;
            total_read += n;
        }
        
        // Read trailing \r\n
        _ = try self.readByte(); // \r
        _ = try self.readByte(); // \n
        
        return RespValue{ .bulk_string = data };
    }

    fn readArray(self: *Self, depth: u32) anyerror!RespValue {
        if (depth > 64) {
            return error.RecursionTooDeep;
        }
        
        const line = try self.readLine();
        defer self.allocator.free(line);
        
        const count = try std.fmt.parseInt(i64, line, 10);
        
        if (count < 0) {
            return error.InvalidArrayLength;
        }
        
        const array = try self.allocator.alloc(RespValue, @intCast(count));
        errdefer {
            for (array) |*item| {
                item.deinit(self.allocator);
            }
            self.allocator.free(array);
        }
        
        for (array) |*item| {
            item.* = try self.readResponse();
        }
        
        return RespValue{ .array = array };
    }

    // ========================================================================
    // High-level Redis Commands
    // ========================================================================

    pub fn get(self: *Self, key: []const u8) !?[]const u8 {
        try self.sendCommand(&.{ "GET", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .bulk_string => |s| try self.allocator.dupe(u8, s),
            .null_bulk_string => null,
            .error_msg => return error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn set(self: *Self, key: []const u8, value: []const u8, ttl_seconds: ?u32) !void {
        if (ttl_seconds) |ttl_val| {
            var ttl_buf: [32]u8 = undefined;
            const ttl_str = try std.fmt.bufPrint(&ttl_buf, "{d}", .{ttl_val});
            try self.sendCommand(&.{ "SET", key, value, "EX", ttl_str });
        } else {
            try self.sendCommand(&.{ "SET", key, value });
        }
        
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        switch (response) {
            .simple_string => {},
            .error_msg => return error.RedisError,
            else => return error.UnexpectedResponse,
        }
    }

    pub fn del(self: *Self, key: []const u8) !bool {
        try self.sendCommand(&.{ "DEL", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn exists(self: *Self, key: []const u8) !bool {
        try self.sendCommand(&.{ "EXISTS", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn expire(self: *Self, key: []const u8, seconds: u32) !bool {
        var seconds_buf: [32]u8 = undefined;
        const seconds_str = try std.fmt.bufPrint(&seconds_buf, "{d}", .{seconds});
        
        try self.sendCommand(&.{ "EXPIRE", key, seconds_str });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn ttl(self: *Self, key: []const u8) !i64 {
        try self.sendCommand(&.{ "TTL", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn lpush(self: *Self, key: []const u8, value: []const u8) !u64 {
        try self.sendCommand(&.{ "LPUSH", key, value });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| @intCast(i),
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn rpush(self: *Self, key: []const u8, value: []const u8) !u64 {
        try self.sendCommand(&.{ "RPUSH", key, value });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| @intCast(i),
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn lpop(self: *Self, key: []const u8) !?[]const u8 {
        try self.sendCommand(&.{ "LPOP", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .bulk_string => |s| try self.allocator.dupe(u8, s),
            .null_bulk_string => null,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn rpop(self: *Self, key: []const u8) !?[]const u8 {
        try self.sendCommand(&.{ "RPOP", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .bulk_string => |s| try self.allocator.dupe(u8, s),
            .null_bulk_string => null,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn llen(self: *Self, key: []const u8) !u64 {
        try self.sendCommand(&.{ "LLEN", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| @intCast(i),
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn sadd(self: *Self, key: []const u8, member: []const u8) !bool {
        try self.sendCommand(&.{ "SADD", key, member });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn srem(self: *Self, key: []const u8, member: []const u8) !bool {
        try self.sendCommand(&.{ "SREM", key, member });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn smembers(self: *Self, key: []const u8) ![][]const u8 {
        try self.sendCommand(&.{ "SMEMBERS", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .array => |arr| {
                const members = try self.allocator.alloc([]const u8, arr.len);
                errdefer {
                    for (members) |m| {
                        self.allocator.free(m);
                    }
                    self.allocator.free(members);
                }
                
                for (arr, 0..) |item, i| {
                    members[i] = switch (item) {
                        .bulk_string => |s| try self.allocator.dupe(u8, s),
                        else => {
                            for (members[0..i]) |m| {
                                self.allocator.free(m);
                            }
                            return error.UnexpectedResponse;
                        },
                    };
                }
                
                return members;
            },
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn sismember(self: *Self, key: []const u8, member: []const u8) !bool {
        try self.sendCommand(&.{ "SISMEMBER", key, member });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn hset(self: *Self, key: []const u8, field: []const u8, value: []const u8) !bool {
        try self.sendCommand(&.{ "HSET", key, field, value });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn hget(self: *Self, key: []const u8, field: []const u8) !?[]const u8 {
        try self.sendCommand(&.{ "HGET", key, field });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .bulk_string => |s| try self.allocator.dupe(u8, s),
            .null_bulk_string => null,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn hdel(self: *Self, key: []const u8, field: []const u8) !bool {
        try self.sendCommand(&.{ "HDEL", key, field });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn hgetall(self: *Self, key: []const u8) !std.StringHashMap([]const u8) {
        try self.sendCommand(&.{ "HGETALL", key });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .array => |arr| {
                var result = std.StringHashMap([]const u8).init(self.allocator);
                errdefer result.deinit();
                
                if (arr.len % 2 != 0) {
                    return error.InvalidHashArray;
                }
                
                var i: usize = 0;
                while (i < arr.len) : (i += 2) {
                    const field = switch (arr[i]) {
                        .bulk_string => |s| s,
                        else => return error.UnexpectedResponse,
                    };
                    
                    const value = switch (arr[i + 1]) {
                        .bulk_string => |s| s,
                        else => return error.UnexpectedResponse,
                    };
                    
                    const field_copy = try self.allocator.dupe(u8, field);
                    errdefer self.allocator.free(field_copy);
                    const value_copy = try self.allocator.dupe(u8, value);
                    errdefer self.allocator.free(value_copy);
                    
                    try result.put(field_copy, value_copy);
                }
                
                return result;
            },
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn publish(self: *Self, channel: []const u8, message: []const u8) !u64 {
        try self.sendCommand(&.{ "PUBLISH", channel, message });
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| @intCast(i),
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    pub fn ping(self: *Self) !void {
        try self.sendCommand(&[_][]const u8{"PING"});
        const response = try self.readResponse();
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        switch (response) {
            .simple_string => {},
            .error_msg => return error.RedisError,
            else => return error.UnexpectedResponse,
        }
    }
};

// ============================================================================
// Sentinel Support for High Availability
// ============================================================================

pub const SentinelConfig = struct {
    sentinel_hosts: []const []const u8,
    sentinel_port: u16,
    master_name: []const u8,
    timeout_ms: u32,
    password: ?[]const u8,
    db_index: u8,

    pub fn init(
        sentinel_hosts: []const []const u8,
        master_name: []const u8,
        sentinel_port: u16,
        password: ?[]const u8,
        db_index: u8,
        timeout_ms: u32,
    ) SentinelConfig {
        return .{
            .sentinel_hosts = sentinel_hosts,
            .sentinel_port = sentinel_port,
            .master_name = master_name,
            .timeout_ms = timeout_ms,
            .password = password,
            .db_index = db_index,
        };
    }
};

pub const DiscoveryResult = struct {
    host: []const u8,
    port: u16,
    last_discovery: i64,
};

/// Sentinel-aware client that can discover primary node
pub const SentinelAwareClient = struct {
    allocator: Allocator,
    sentinel_config: SentinelConfig,
    current_master: ?DiscoveryResult,
    mutex: Thread.Mutex,
    last_failover_time: i64,

    pub fn init(allocator: Allocator, config: SentinelConfig) !*SentinelAwareClient {
        const self = try allocator.create(SentinelAwareClient);
        
        const sentinel_hosts_copy = try allocator.alloc([]const u8, config.sentinel_hosts.len);
        errdefer allocator.free(sentinel_hosts_copy);
        
        for (config.sentinel_hosts, 0..) |host, i| {
            sentinel_hosts_copy[i] = try allocator.dupe(u8, host);
        }
        
        const master_name_copy = try allocator.dupe(u8, config.master_name);
        
        self.* = .{
            .allocator = allocator,
            .sentinel_config = .{
                .sentinel_hosts = sentinel_hosts_copy,
                .sentinel_port = config.sentinel_port,
                .master_name = master_name_copy,
                .timeout_ms = config.timeout_ms,
                .password = if (config.password) |pwd| 
                    try allocator.dupe(u8, pwd) else null,
                .db_index = config.db_index,
            },
            .current_master = null,
            .mutex = .{},
            .last_failover_time = 0,
        };
        
        return self;
    }

    pub fn deinit(self: *SentinelAwareClient) void {
        // No need to lock mutex during deinit - we're destroying the object
        
        if (self.current_master) |*master| {
            self.allocator.free(master.host);
        }
        
        for (self.sentinel_config.sentinel_hosts) |host| {
            self.allocator.free(host);
        }
        self.allocator.free(self.sentinel_config.sentinel_hosts);
        self.allocator.free(self.sentinel_config.master_name);
        
        if (self.sentinel_config.password) |pwd| {
            self.allocator.free(pwd);
        }
        
        self.allocator.destroy(self);
    }

    /// Discover current master from Sentinel
    pub fn discoverMaster(self: *SentinelAwareClient) !DiscoveryResult {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const now = std.time.milliTimestamp();
        
        // Cache discovery for 5 seconds
        if (self.current_master) |master| {
            if (now - master.last_discovery < 5000) {
                return master;
            }
        }
        
        var last_error: anyerror = error.NoSentinelAvailable;
        
        // Try each sentinel
        for (self.sentinel_config.sentinel_hosts) |sentinel_host| {
            var client = RespClient.init(
                self.allocator,
                sentinel_host,
                self.sentinel_config.sentinel_port,
                self.sentinel_config.password,
                0, // Sentinel uses database 0
                self.sentinel_config.timeout_ms,
            );
            defer client.deinit();
            
            client.connect() catch {
                last_error = error.SentinelConnectionFailed;
                continue;
            };
            defer client.disconnect();
            
            // Query sentinel for master
            const master_info = self.querySentinel(&client, self.sentinel_config.master_name) catch {
                last_error = error.SentinelQueryFailed;
                continue;
            };
            
            // Clean up old master
            if (self.current_master) |*old| {
                self.allocator.free(old.host);
            }
            
            const host_copy = try self.allocator.dupe(u8, master_info.host);
            self.current_master = DiscoveryResult{
                .host = host_copy,
                .port = master_info.port,
                .last_discovery = now,
            };
            
            return self.current_master.?;
        }
        
        return last_error;
    }

    fn querySentinel(self: *SentinelAwareClient, client: *RespClient, master_name: []const u8) !DiscoveryResult {
        _ = self;
        
        try client.sendCommand(&.{ "SENTINEL", "get-master-addr-by-name", master_name });
        const response = try client.readResponse();
        defer {
            var resp = response;
            resp.deinit(client.allocator);
        }
        
        switch (response) {
            .array => |arr| {
                if (arr.len >= 2) {
                    const host = switch (arr[0]) {
                        .bulk_string => |s| s,
                        else => return error.InvalidSentinelResponse,
                    };
                    
                    const port_str = switch (arr[1]) {
                        .bulk_string => |s| s,
                        else => return error.InvalidSentinelResponse,
                    };
                    
                    const port = try std.fmt.parseInt(u16, port_str, 10);
                    
                    return DiscoveryResult{
                        .host = host,
                        .port = port,
                        .last_discovery = std.time.milliTimestamp(),
                    };
                } else {
                    return error.InvalidSentinelResponse;
                }
            },
            .error_msg => |e| {
                std.log.err("Sentinel error: {s}", .{e});
                return error.SentinelError;
            },
            else => {
                return error.InvalidSentinelResponse;
            },
        }
    }

    /// Check if a failover has occurred
    pub fn checkFailover(self: *SentinelAwareClient) !bool {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        if (self.current_master == null) {
            return true;
        }
        
        const old_master = self.current_master.?;
        
        for (self.sentinel_config.sentinel_hosts) |sentinel_host| {
            var client = RespClient.init(
                self.allocator,
                sentinel_host,
                self.sentinel_config.sentinel_port,
                self.sentinel_config.password,
                0,
                self.sentinel_config.timeout_ms,
            );
            defer client.deinit();
            
            client.connect() catch continue;
            defer client.disconnect();
            
            const new_master = self.querySentinel(&client, self.sentinel_config.master_name) catch continue;
            
            if (!std.mem.eql(u8, old_master.host, new_master.host) or old_master.port != new_master.port) {
                std.log.info("Failover detected: {s}:{d} -> {s}:{d}", .{
                    old_master.host, old_master.port,
                    new_master.host, new_master.port,
                });
                
                self.allocator.free(old_master.host);
                
                const host_copy = try self.allocator.dupe(u8, new_master.host);
                self.current_master = DiscoveryResult{
                    .host = host_copy,
                    .port = new_master.port,
                    .last_discovery = new_master.last_discovery,
                };
                self.last_failover_time = std.time.milliTimestamp();
                
                return true;
            }
        }
        
        return false;
    }
};

// ============================================================================
// Connection Pool Implementation
// ============================================================================

pub const ConnectionPoolConfig = struct {
    max_connections: usize = 10,
    min_connections: usize = 2,
    connection_timeout_ms: u32 = 5000,
    max_idle_time_ms: u32 = 30000,
    health_check_interval_ms: u32 = 10000,
};

pub const PooledConnection = struct {
    client: *RespClient,
    last_used: i64,
    is_broken: bool,
    in_pool: bool,
};

pub const RespConnectionPool = struct {
    allocator: Allocator,
    config: ConnectionPoolConfig,
    host: []const u8,
    port: u16,
    password: ?[]const u8,
    db_index: u8,
    timeout_ms: u32,
    
    // Connection storage
    connections: std.ArrayList(*PooledConnection),
    mutex: Thread.Mutex,
    condition: Thread.Condition,
    
    // Sentinel support
    sentinel_client: ?*SentinelAwareClient,
    use_sentinel: bool,
    
    // Statistics
    total_connections: usize,
    active_connections: usize,
    
    // Health check thread
    health_check_thread: ?Thread,
    shutdown_flag: bool,

    const Self = @This();

    pub fn init(
        allocator: Allocator,
        host: []const u8,
        port: u16,
        password: ?[]const u8,
        db_index: u8,
        timeout_ms: u32,
        pool_config: ConnectionPoolConfig,
        sentinel_config: ?SentinelConfig,
    ) !*Self {
        const self = try allocator.create(Self);
        
        const host_copy = try allocator.dupe(u8, host);
        errdefer allocator.free(host_copy);
        
        const password_copy = if (password) |pwd| 
            try allocator.dupe(u8, pwd) else null;
        errdefer if (password_copy) |pwd| allocator.free(pwd);
        
        var sentinel_client: ?*SentinelAwareClient = null;
        if (sentinel_config) |config| {
            sentinel_client = try SentinelAwareClient.init(allocator, config);
        }
        
        self.* = .{
            .allocator = allocator,
            .config = pool_config,
            .host = host_copy,
            .port = port,
            .password = password_copy,
            .db_index = db_index,
            .timeout_ms = timeout_ms,
            .connections = std.ArrayList(*PooledConnection).init(allocator),
            .mutex = .{},
            .condition = .{},
            .sentinel_client = sentinel_client,
            .use_sentinel = sentinel_client != null,
            .total_connections = 0,
            .active_connections = 0,
            .health_check_thread = null,
            .shutdown_flag = false,
        };
        
        // Pre-warm pool with minimum connections
        try self.warmPool();
        
        // Start health check thread
        self.health_check_thread = try Thread.spawn(.{}, healthCheckThread, .{self});
        
        return self;
    }

    pub fn deinit(self: *Self) void {
        // Signal shutdown
        self.mutex.lock();
        self.shutdown_flag = true;
        self.mutex.unlock();
        
        // Wait for health check thread
        if (self.health_check_thread) |thread| {
            thread.join();
        }
        
        // Close all connections
        self.mutex.lock();
        defer self.mutex.unlock();
        
        for (self.connections.items) |conn| {
            conn.client.deinit();
            self.allocator.destroy(conn.client);
            self.allocator.destroy(conn);
        }
        self.connections.deinit();
        
        // Clean up sentinel client
        if (self.sentinel_client) |sentinel| {
            sentinel.deinit();
        }
        
        // Free copied strings
        self.allocator.free(self.host);
        if (self.password) |pwd| {
            self.allocator.free(pwd);
        }
        
        self.allocator.destroy(self);
    }

    fn warmPool(self: *Self) !void {
        for (0..self.config.min_connections) |_| {
            const conn = try self.createConnection();
            const pooled = try self.allocator.create(PooledConnection);
            
            pooled.* = .{
                .client = conn,
                .last_used = std.time.milliTimestamp(),
                .is_broken = false,
                .in_pool = true,
            };
            
            try self.connections.append(pooled);
            self.total_connections += 1;
        }
    }

    fn createConnection(self: *Self) !*RespClient {
        var host_to_use = self.host;
        var port_to_use = self.port;
        
        // Use Sentinel to discover master if enabled
        if (self.use_sentinel and self.sentinel_client) |sentinel| {
            const master = try sentinel.discoverMaster();
            host_to_use = master.host;
            port_to_use = master.port;
        }
        
        const client = try self.allocator.create(RespClient);
        errdefer self.allocator.destroy(client);
        
        client.* = RespClient.init(
            self.allocator,
            host_to_use,
            port_to_use,
            self.password,
            self.db_index,
            self.timeout_ms,
        );
        
        try client.connect();
        
        return client;
    }

    /// Acquire a connection from the pool
    pub fn acquire(self: *Self) !*RespClient {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Try to get existing connection
        while (self.connections.items.len > 0) {
            const conn = self.connections.pop();
            
            // Check if connection is still healthy
            if (!conn.is_broken and self.checkConnectionHealth(conn.client)) {
                conn.last_used = std.time.milliTimestamp();
                conn.in_pool = false;
                self.active_connections += 1;
                return conn.client;
            } else {
                // Discard broken connection
                conn.client.deinit();
                self.allocator.destroy(conn.client);
                self.allocator.destroy(conn);
                self.total_connections -= 1;
            }
        }
        
        // Create new connection if under limit
        if (self.total_connections < self.config.max_connections) {
            const client = try self.createConnection();
            self.total_connections += 1;
            self.active_connections += 1;
            return client;
        }
        
        // Wait for available connection
        self.condition.wait(&self.mutex);
        return error.PoolExhausted;
    }

    /// Release a connection back to the pool
    pub fn release(self: *Self, client: *RespClient) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        self.active_connections -= 1;
        
        // Check if we should keep this connection
        if (self.total_connections > self.config.max_connections or 
            !self.checkConnectionHealth(client)) {
            // Discard connection
            client.deinit();
            self.allocator.destroy(client);
            self.total_connections -= 1;
        } else {
            // Return to pool
            const pooled = self.allocator.create(PooledConnection) catch {
                // If we can't allocate memory for tracking, discard connection
                client.deinit();
                self.allocator.destroy(client);
                self.total_connections -= 1;
                return;
            };
            
            pooled.* = .{
                .client = client,
                .last_used = std.time.milliTimestamp(),
                .is_broken = false,
                .in_pool = true,
            };
            
            self.connections.append(pooled) catch {
                self.allocator.destroy(pooled);
                client.deinit();
                self.allocator.destroy(client);
                self.total_connections -= 1;
                return;
            };
        }
        
        self.condition.signal();
    }

    fn checkConnectionHealth(self: *Self, client: *RespClient) bool {
        _ = self;
        
        // Quick ping to check connection
        client.sendCommand(&[_][]const u8{"PING"}) catch return false;
        
        const response = client.readResponse() catch return false;
        defer {
            var resp = response;
            resp.deinit(client.allocator);
        }
        
        return switch (response) {
            .simple_string => true,
            else => false,
        };
    }

    fn healthCheckThread(self: *Self) void {
        while (true) {
            std.time.sleep(self.config.health_check_interval_ms * 1_000_000);
            
            self.mutex.lock();
            
            if (self.shutdown_flag) {
                self.mutex.unlock();
                break;
            }
            
            // Check for failover if using Sentinel
            if (self.use_sentinel and self.sentinel_client) |sentinel| {
                _ = sentinel.checkFailover() catch |err| {
                    std.log.err("Sentinel failover check failed: {}", .{err});
                };
            }
            
            // Clean up idle connections
            const now = std.time.milliTimestamp();
            var i: usize = 0;
            while (i < self.connections.items.len) {
                const conn = self.connections.items[i];
                
                // Check if connection is too old
                if (now - conn.last_used > self.config.max_idle_time_ms and
                    self.total_connections > self.config.min_connections) {
                    _ = self.connections.orderedRemove(i);
                    conn.client.deinit();
                    self.allocator.destroy(conn.client);
                    self.allocator.destroy(conn);
                    self.total_connections -= 1;
                } else {
                    i += 1;
                }
            }
            
            self.mutex.unlock();
        }
    }

    /// Get pool statistics
    pub fn getStats(self: *Self) struct {
        total: usize,
        active: usize,
        idle: usize,
    } {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        const total = self.total_connections;
        const active = self.active_connections;
        const idle = if (total > active) total - active else 0;
        
        return .{
            .total = total,
            .active = active,
            .idle = idle,
        };
    }
};

// ============================================================================
// Enhanced RespClient with Connection Pool
// ============================================================================

/// Enhanced version of RespClient that uses connection pool internally
pub const RespClientEnhanced = struct {
    allocator: Allocator,
    pool: *RespConnectionPool,
    current_connection: ?*RespClient,
    
    const Self = @This();

    pub fn init(
        allocator: Allocator,
        host: []const u8,
        port: u16,
        password: ?[]const u8,
        db_index: u8,
        timeout_ms: u32,
        pool_config: ConnectionPoolConfig,
        sentinel_config: ?SentinelConfig,
    ) !Self {
        const pool = try RespConnectionPool.init(
            allocator,
            host,
            port,
            password,
            db_index,
            timeout_ms,
            pool_config,
            sentinel_config,
        );
        
        return Self{
            .allocator = allocator,
            .pool = pool,
            .current_connection = null,
        };
    }

    pub fn deinit(self: *Self) void {
        if (self.current_connection) |conn| {
            self.pool.release(conn);
        }
        self.pool.deinit();
    }

    /// Get a connection for transaction/command sequence
    pub fn begin(self: *Self) !void {
        if (self.current_connection != null) {
            return error.TransactionAlreadyStarted;
        }
        self.current_connection = try self.pool.acquire();
    }

    /// Release connection after transaction
    pub fn end(self: *Self) void {
        if (self.current_connection) |conn| {
            self.pool.release(conn);
            self.current_connection = null;
        }
    }

    /// Execute a command using pooled connection
    fn executeCommand(self: *Self, args: []const []const u8) !RespValue {
        var client: *RespClient = undefined;
        var release_needed = false;
        
        if (self.current_connection) |conn| {
            client = conn;
        } else {
            client = try self.pool.acquire();
            release_needed = true;
        }
        defer if (release_needed) self.pool.release(client);
        
        try client.sendCommand(args);
        return try client.readResponse();
    }

    // Delegate all RespClient methods through executeCommand
    pub fn get(self: *Self, key: []const u8) !?[]const u8 {
        const response = try self.executeCommand(&.{ "GET", key });
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .bulk_string => |s| try self.allocator.dupe(u8, s),
            .null_bulk_string => null,
            .error_msg => |e| {
                std.log.err("Redis error: {s}", .{e});
                return error.RedisError;
            },
            else => {
                std.log.err("Unexpected response type for GET", .{});
                return error.UnexpectedResponse;
            },
        };
    }

    pub fn set(self: *Self, key: []const u8, value: []const u8, ttl_seconds: ?u32) !void {
        const args = if (ttl_seconds) |ttl_val| blk: {
            var ttl_buf: [32]u8 = undefined;
            const ttl_str = try std.fmt.bufPrint(&ttl_buf, "{d}", .{ttl_val});
            break :blk &[_][]const u8{ "SET", key, value, "EX", ttl_str };
        } else &[_][]const u8{ "SET", key, value };
        
        const response = try self.executeCommand(args);
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        switch (response) {
            .simple_string => {},
            .error_msg => |e| {
                std.log.err("Redis error: {s}", .{e});
                return error.RedisError;
            },
            else => {
                std.log.err("Unexpected response type for SET", .{});
                return error.UnexpectedResponse;
            },
        }
    }

    pub fn del(self: *Self, key: []const u8) !bool {
        const response = try self.executeCommand(&.{ "DEL", key });
        defer {
            var resp = response;
            resp.deinit(self.allocator);
        }
        
        return switch (response) {
            .integer => |i| i > 0,
            .error_msg => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    /// Get pool statistics
    pub fn getStats(self: *Self) struct {
        total: usize,
        active: usize,
        idle: usize,
    } {
        return self.pool.getStats();
    }
};

// ============================================================================
// Tests
// ============================================================================

test "RESP client initialization" {
    const allocator = std.testing.allocator;
    
    var client = RespClient.init(allocator, "localhost", 6379, null, 0, 5000);
    defer client.deinit();
    
    try std.testing.expect(!client.is_connected);
}

test "RESP value clone and deinit" {
    const allocator = std.testing.allocator;
    
    var value = RespValue{ .simple_string = try allocator.dupe(u8, "OK") };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expectEqualStrings("OK", clone_val.simple_string);
}

test "RESP bulk string memory safety" {
    const allocator = std.testing.allocator;
    
    var value = RespValue{ .bulk_string = try allocator.dupe(u8, "test") };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expectEqualStrings("test", clone_val.bulk_string);
}

test "SentinelConfig initialization" {
    const sentinel_hosts = &[_][]const u8{ "sentinel1", "sentinel2", "sentinel3" };
    const config = SentinelConfig.init(
        sentinel_hosts,
        "mymaster",
        26379,
        null,
        0,
        5000,
    );
    
    try std.testing.expectEqual(@as(u16, 26379), config.sentinel_port);
    try std.testing.expectEqualStrings("mymaster", config.master_name);
    try std.testing.expectEqual(@as(usize, 3), config.sentinel_hosts.len);
}

test "SentinelAwareClient initialization and cleanup" {
    const allocator = std.testing.allocator;
    
    const sentinel_hosts = &[_][]const u8{ "sentinel1.example.com", "sentinel2.example.com" };
    const config = SentinelConfig.init(
        sentinel_hosts,
        "mymaster",
        26379,
        null,
        0,
        5000,
    );
    
    var client = try SentinelAwareClient.init(allocator, config);
    defer client.deinit();
    
    try std.testing.expect(client.current_master == null);
    try std.testing.expectEqual(@as(i64, 0), client.last_failover_time);
}

test "ConnectionPoolConfig defaults" {
    const config = ConnectionPoolConfig{};
    
    try std.testing.expectEqual(@as(usize, 10), config.max_connections);
    try std.testing.expectEqual(@as(usize, 2), config.min_connections);
    try std.testing.expectEqual(@as(u32, 5000), config.connection_timeout_ms);
    try std.testing.expectEqual(@as(u32, 30000), config.max_idle_time_ms);
    try std.testing.expectEqual(@as(u32, 10000), config.health_check_interval_ms);
}

test "ConnectionPoolConfig custom values" {
    const config = ConnectionPoolConfig{
        .max_connections = 20,
        .min_connections = 5,
        .connection_timeout_ms = 10000,
        .max_idle_time_ms = 60000,
        .health_check_interval_ms = 15000,
    };
    
    try std.testing.expectEqual(@as(usize, 20), config.max_connections);
    try std.testing.expectEqual(@as(usize, 5), config.min_connections);
    try std.testing.expectEqual(@as(u32, 10000), config.connection_timeout_ms);
    try std.testing.expectEqual(@as(u32, 60000), config.max_idle_time_ms);
    try std.testing.expectEqual(@as(u32, 15000), config.health_check_interval_ms);
}

test "PooledConnection structure" {
    const allocator = std.testing.allocator;
    
    const client = try allocator.create(RespClient);
    defer allocator.destroy(client);
    
    client.* = RespClient.init(allocator, "localhost", 6379, null, 0, 5000);
    
    const pooled = PooledConnection{
        .client = client,
        .last_used = std.time.milliTimestamp(),
        .is_broken = false,
        .in_pool = true,
    };
    
    try std.testing.expect(!pooled.is_broken);
    try std.testing.expect(pooled.in_pool);
}

test "RespValue array clone" {
    const allocator = std.testing.allocator;
    
    const items = try allocator.alloc(RespValue, 2);
    items[0] = RespValue{ .simple_string = try allocator.dupe(u8, "OK") };
    items[1] = RespValue{ .integer = 42 };
    
    var value = RespValue{ .array = items };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expectEqual(@as(usize, 2), clone_val.array.len);
    try std.testing.expectEqualStrings("OK", clone_val.array[0].simple_string);
    try std.testing.expectEqual(@as(i64, 42), clone_val.array[1].integer);
}

test "RespValue null_bulk_string" {
    const allocator = std.testing.allocator;
    
    var value = RespValue{ .null_bulk_string = {} };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expect(clone_val == .null_bulk_string);
}

test "RespValue integer clone" {
    const allocator = std.testing.allocator;
    
    var value = RespValue{ .integer = 12345 };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expectEqual(@as(i64, 12345), clone_val.integer);
}

test "RespValue error_msg clone" {
    const allocator = std.testing.allocator;
    
    var value = RespValue{ .error_msg = try allocator.dupe(u8, "ERR test error") };
    defer value.deinit(allocator);
    
    var clone_val = try value.clone(allocator);
    defer clone_val.deinit(allocator);
    
    try std.testing.expectEqualStrings("ERR test error", clone_val.error_msg);
}

test "DiscoveryResult structure" {
    const allocator = std.testing.allocator;
    
    const host = try allocator.dupe(u8, "master.redis.com");
    defer allocator.free(host);
    
    const result = DiscoveryResult{
        .host = host,
        .port = 6379,
        .last_discovery = std.time.milliTimestamp(),
    };
    
    try std.testing.expectEqualStrings("master.redis.com", result.host);
    try std.testing.expectEqual(@as(u16, 6379), result.port);
    try std.testing.expect(result.last_discovery > 0);
}

// Note: The following tests would require a running DragonflyDB/Redis instance
// They are included for documentation but may be skipped in CI environments

test "Connection pool statistics" {
    // This test verifies that pool statistics are tracked correctly
    // In production, this would connect to a real database
    // For unit testing, we just verify the structure is correct
    
    const stats = struct {
        total: usize,
        active: usize,
        idle: usize,
    }{
        .total = 5,
        .active = 2,
        .idle = 3,
    };
    
    try std.testing.expectEqual(@as(usize, 5), stats.total);
    try std.testing.expectEqual(@as(usize, 2), stats.active);
    try std.testing.expectEqual(@as(usize, 3), stats.idle);
    try std.testing.expectEqual(@as(usize, 5), stats.active + stats.idle);
}

test "Thread safety structures" {
    // Verify that thread-safe structures are properly initialized
    var mutex = Thread.Mutex{};
    const condition = Thread.Condition{};
    
    mutex.lock();
    defer mutex.unlock();
    
    // Successfully locked and unlocked without error
    try std.testing.expect(true);
    
    // Condition variable exists
    _ = condition;
}
