// DragonflyDB Client in Zig
// High-performance RESP protocol implementation for Redis-compatible caching
// Target: 10-20x faster than Python client
//
// Features:
// - RESP (REdis Serialization Protocol) v2/v3
// - Connection pooling
// - Async operations
// - C ABI for Mojo integration
// - Zero-copy operations where possible

const std = @import("std");
const net = std.net;
const mem = std.mem;
const ArrayList = std.ArrayList;
const Allocator = mem.Allocator;

/// RESP protocol data types
pub const RespType = enum {
    SimpleString,  // +
    Error,         // -
    Integer,       // :
    BulkString,    // $
    Array,         // *
    Null,          // null bulk string or array
};

/// RESP value representation
pub const RespValue = union(RespType) {
    SimpleString: []const u8,
    Error: []const u8,
    Integer: i64,
    BulkString: []const u8,
    Array: []RespValue,
    Null: void,

    pub fn deinit(self: *RespValue, allocator: Allocator) void {
        switch (self.*) {
            .Array => |arr| {
                for (arr) |*item| {
                    item.deinit(allocator);
                }
                allocator.free(arr);
            },
            .BulkString => |str| allocator.free(str),
            .SimpleString => |str| allocator.free(str),
            .Error => |str| allocator.free(str),
            else => {},
        }
    }
};

/// Connection to DragonflyDB server
pub const Connection = struct {
    stream: net.Stream,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, host: []const u8, port: u16) !Connection {
        const address = try net.Address.parseIp(host, port);
        const stream = try net.tcpConnectToAddress(address);
        
        return Connection{
            .stream = stream,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Connection) void {
        self.stream.close();
    }
    
    /// Send RESP command
    pub fn sendCommand(self: *Connection, args: []const []const u8) !void {
        var buffer = ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        
        // Format: *<count>\r\n$<len>\r\n<arg>\r\n...
        try buffer.writer(self.allocator).print("*{d}\r\n", .{args.len});
        
        for (args) |arg| {
            try buffer.writer(self.allocator).print("${d}\r\n{s}\r\n", .{ arg.len, arg });
        }
        
        _ = try self.stream.writeAll(buffer.items);
    }
    
    /// Read RESP response  
    pub fn readResponse(self: *Connection) anyerror!RespValue {
        var first_byte_buf: [1]u8 = undefined;
        _ = try self.stream.read(&first_byte_buf);
        const first_byte = first_byte_buf[0];
        
        return switch (first_byte) {
            '+' => try self.readSimpleString(),
            '-' => try self.readError(),
            ':' => try self.readInteger(),
            '$' => try self.readBulkString(),
            '*' => try self.readArray(),
            else => error.InvalidRespType,
        };
    }
    
    fn readSimpleString(self: *Connection) !RespValue {
        const line = try self.readLine();
        return RespValue{ .SimpleString = line };
    }
    
    fn readError(self: *Connection) !RespValue {
        const line = try self.readLine();
        return RespValue{ .Error = line };
    }
    
    fn readInteger(self: *Connection) !RespValue {
        const line = try self.readLine();
        defer self.allocator.free(line);
        const value = try std.fmt.parseInt(i64, line, 10);
        return RespValue{ .Integer = value };
    }
    
    fn readBulkString(self: *Connection) !RespValue {
        const len_line = try self.readLine();
        defer self.allocator.free(len_line);
        
        const len = try std.fmt.parseInt(i64, len_line, 10);
        if (len == -1) {
            return RespValue{ .Null = {} };
        }
        
        const ulen = @as(usize, @intCast(len));
        const data = try self.allocator.alloc(u8, ulen);
        _ = try self.stream.read(data);
        
        // Read trailing \r\n
        var crlf: [2]u8 = undefined;
        _ = try self.stream.read(&crlf);
        
        return RespValue{ .BulkString = data };
    }
    
    fn readArray(self: *Connection) !RespValue {
        const len_line = try self.readLine();
        defer self.allocator.free(len_line);
        
        const len = try std.fmt.parseInt(i64, len_line, 10);
        if (len == -1) {
            return RespValue{ .Null = {} };
        }
        
        const ulen = @as(usize, @intCast(len));
        const array = try self.allocator.alloc(RespValue, ulen);
        
        for (array) |*item| {
            item.* = try self.readResponse();
        }
        
        return RespValue{ .Array = array };
    }
    
    fn readLine(self: *Connection) ![]u8 {
        var line = ArrayList(u8){};
        errdefer line.deinit(self.allocator);
        
        var byte_buf: [1]u8 = undefined;
        while (true) {
            _ = try self.stream.read(&byte_buf);
            const byte = byte_buf[0];
            if (byte == '\r') {
                _ = try self.stream.read(&byte_buf);
                const next = byte_buf[0];
                if (next == '\n') {
                    return line.toOwnedSlice(self.allocator);
                }
                try line.append(self.allocator, byte);
                try line.append(self.allocator, next);
            } else {
                try line.append(self.allocator, byte);
            }
        }
    }
};

/// DragonflyDB client with connection pooling
pub const DragonflyClient = struct {
    allocator: Allocator,
    host: []const u8,
    port: u16,
    connections: ArrayList(*Connection),
    max_connections: usize,
    
    pub fn init(allocator: Allocator, host: []const u8, port: u16) !*DragonflyClient {
        const client = try allocator.create(DragonflyClient);
        client.* = DragonflyClient{
            .allocator = allocator,
            .host = host,
            .port = port,
            .connections = ArrayList(*Connection){},
            .max_connections = 10,
        };
        return client;
    }
    
    pub fn deinit(self: *DragonflyClient) void {
        for (self.connections.items) |conn| {
            conn.deinit();
            self.allocator.destroy(conn);
        }
        self.connections.deinit(self.allocator);
        self.allocator.destroy(self);
    }
    
    fn getConnection(self: *DragonflyClient) !*Connection {
        if (self.connections.items.len > 0) {
            return self.connections.pop() orelse unreachable;
        }
        
        const conn = try self.allocator.create(Connection);
        conn.* = try Connection.init(self.allocator, self.host, self.port);
        return conn;
    }
    
    fn returnConnection(self: *DragonflyClient, conn: *Connection) !void {
        if (self.connections.items.len < self.max_connections) {
            try self.connections.append(self.allocator, conn);
        } else {
            conn.deinit();
            self.allocator.destroy(conn);
        }
    }
    
    /// GET key
    pub fn get(self: *DragonflyClient, key: []const u8) !?[]u8 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};
        
        const args = [_][]const u8{ "GET", key };
        try conn.sendCommand(&args);
        
        var response = try conn.readResponse();
        defer response.deinit(self.allocator);
        
        return switch (response) {
            .BulkString => |str| try self.allocator.dupe(u8, str),
            .Null => null,
            .Error => |err| {
                std.debug.print("Redis error: {s}\n", .{err});
                return error.RedisError;
            },
            else => error.UnexpectedResponse,
        };
    }
    
    /// SET key value [EX seconds]
    pub fn set(self: *DragonflyClient, key: []const u8, value: []const u8, ex: ?u32) !void {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};
        
        var args_buffer: [5][]const u8 = undefined;
        var args_len: usize = 3;
        args_buffer[0] = "SET";
        args_buffer[1] = key;
        args_buffer[2] = value;
        
        var ex_str_buf: [20]u8 = undefined;
        if (ex) |seconds| {
            args_buffer[3] = "EX";
            const ex_str = try std.fmt.bufPrint(&ex_str_buf, "{d}", .{seconds});
            args_buffer[4] = ex_str;
            args_len = 5;
        }
        
        try conn.sendCommand(args_buffer[0..args_len]);
        
        var response = try conn.readResponse();
        defer response.deinit(self.allocator);
        
        switch (response) {
            .SimpleString => |str| {
                if (!mem.eql(u8, str, "OK")) {
                    return error.SetFailed;
                }
            },
            .Error => return error.RedisError,
            else => return error.UnexpectedResponse,
        }
    }
    
    /// DEL key [key ...]
    pub fn del(self: *DragonflyClient, keys: []const []const u8) !u64 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};
        
        var args = ArrayList([]const u8){};
        defer args.deinit(self.allocator);
        
        try args.append(self.allocator, "DEL");
        for (keys) |key| {
            try args.append(self.allocator, key);
        }
        
        try conn.sendCommand(args.items);
        
        var response = try conn.readResponse();
        defer response.deinit(self.allocator);
        
        return switch (response) {
            .Integer => |count| @as(u64, @intCast(count)),
            .Error => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }
    
    /// MGET key [key ...]
    pub fn mget(self: *DragonflyClient, keys: []const []const u8) ![]?[]u8 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};
        
        var args = ArrayList([]const u8){};
        defer args.deinit(self.allocator);
        
        try args.append(self.allocator, "MGET");
        for (keys) |key| {
            try args.append(self.allocator, key);
        }
        
        try conn.sendCommand(args.items);
        
        var response = try conn.readResponse();
        defer response.deinit(self.allocator);
        
        switch (response) {
            .Array => |arr| {
                const results = try self.allocator.alloc(?[]u8, arr.len);
                for (arr, 0..) |item, i| {
                    results[i] = switch (item) {
                        .BulkString => |str| try self.allocator.dupe(u8, str),
                        .Null => null,
                        else => return error.UnexpectedResponse,
                    };
                }
                return results;
            },
            .Error => return error.RedisError,
            else => return error.UnexpectedResponse,
        }
    }
    
    /// EXISTS key [key ...]
    pub fn exists(self: *DragonflyClient, keys: []const []const u8) !u64 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};
        
        var args = ArrayList([]const u8){};
        defer args.deinit(self.allocator);
        
        try args.append(self.allocator, "EXISTS");
        for (keys) |key| {
            try args.append(self.allocator, key);
        }
        
        try conn.sendCommand(args.items);
        
        var response = try conn.readResponse();
        defer response.deinit(self.allocator);
        
        return switch (response) {
            .Integer => |count| @as(u64, @intCast(count)),
            .Error => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }
    
    /// EXPIRE key seconds
    pub fn expire(self: *DragonflyClient, key: []const u8, seconds: u32) !bool {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};

        var seconds_buf: [20]u8 = undefined;
        const seconds_str = try std.fmt.bufPrint(&seconds_buf, "{d}", .{seconds});

        const args = [_][]const u8{ "EXPIRE", key, seconds_str };
        try conn.sendCommand(&args);

        var response = try conn.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .Integer => |result| result == 1,
            .Error => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    /// INCR key - Increment integer value and return new value
    pub fn incr(self: *DragonflyClient, key: []const u8) !i64 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};

        const args = [_][]const u8{ "INCR", key };
        try conn.sendCommand(&args);

        var response = try conn.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .Integer => |val| val,
            .Error => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }

    /// INCRBY key increment - Increment integer value by increment and return new value
    pub fn incrBy(self: *DragonflyClient, key: []const u8, increment: i64) !i64 {
        const conn = try self.getConnection();
        defer self.returnConnection(conn) catch {};

        var incr_buf: [20]u8 = undefined;
        const incr_str = try std.fmt.bufPrint(&incr_buf, "{d}", .{increment});

        const args = [_][]const u8{ "INCRBY", key, incr_str };
        try conn.sendCommand(&args);

        var response = try conn.readResponse();
        defer response.deinit(self.allocator);

        return switch (response) {
            .Integer => |val| val,
            .Error => error.RedisError,
            else => error.UnexpectedResponse,
        };
    }
};

// ============================================================================
// C ABI for Mojo Integration
// ============================================================================

const CClient = opaque {};

/// Create a new client
export fn dragonfly_client_create(
    host: [*:0]const u8,
    port: u16,
) callconv(.c) ?*CClient {
    const allocator = std.heap.c_allocator;
    const host_slice = mem.span(host);
    
    const client = DragonflyClient.init(allocator, host_slice, port) catch return null;
    return @ptrCast(client);
}

/// Destroy a client
export fn dragonfly_client_destroy(client: *CClient) callconv(.c) void {
    const real_client: *DragonflyClient = @ptrCast(@alignCast(client));
    real_client.deinit();
}

/// GET operation
export fn dragonfly_get(
    client: *CClient,
    key: [*:0]const u8,
    value_out: *[*]u8,
    len_out: *usize,
) callconv(.c) i32 {
    const real_client: *DragonflyClient = @ptrCast(@alignCast(client));
    const key_slice = mem.span(key);
    
    const result = real_client.get(key_slice) catch return -1;
    
    if (result) |value| {
        value_out.* = value.ptr;
        len_out.* = value.len;
        return 0;
    } else {
        value_out.* = undefined;
        len_out.* = 0;
        return 1; // Key not found
    }
}

/// SET operation
export fn dragonfly_set(
    client: *CClient,
    key: [*:0]const u8,
    value: [*]const u8,
    value_len: usize,
    expire_seconds: i32,
) callconv(.c) i32 {
    const real_client: *DragonflyClient = @ptrCast(@alignCast(client));
    const key_slice = mem.span(key);
    const value_slice = value[0..value_len];
    
    const ex: ?u32 = if (expire_seconds >= 0) @as(u32, @intCast(expire_seconds)) else null;
    
    real_client.set(key_slice, value_slice, ex) catch return -1;
    return 0;
}

/// DEL operation
export fn dragonfly_del(
    client: *CClient,
    key: [*:0]const u8,
) callconv(.c) i32 {
    const real_client: *DragonflyClient = @ptrCast(@alignCast(client));
    const key_slice = mem.span(key);
    
    const keys = [_][]const u8{key_slice};
    const count = real_client.del(&keys) catch return -1;
    return @as(i32, @intCast(count));
}

/// Free a value returned by dragonfly_get
export fn dragonfly_free_value(value: [*]u8, len: usize) callconv(.c) void {
    const allocator = std.heap.c_allocator;
    const slice = value[0..len];
    allocator.free(slice);
}
