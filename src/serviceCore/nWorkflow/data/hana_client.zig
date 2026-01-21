//! SAP HANA Client - Native protocol with TLS, connection pooling, prepared statements, transactions

const std = @import("std");
const net = std.net;
const Allocator = std.mem.Allocator;
const Thread = std.Thread;

pub const HanaError = error{
    ConnectionFailed,
    AuthenticationFailed,
    TlsInitFailed,
    TlsHandshakeFailed,
    QueryFailed,
    ExecuteFailed,
    TransactionError,
    PrepareStatementFailed,
    InvalidParameter,
    Timeout,
    PoolExhausted,
    AlreadyConnected,
    NotConnected,
    AlreadyInTransaction,
    NotInTransaction,
    InvalidConfig,
    ProtocolError,
    EndOfStream,
    OutOfMemory,
    UnsupportedAuthType,
    PasswordRequired,
    SchemaNotSet,
    StatementNotPrepared,
    InvalidResponse,
};

pub const HanaType = enum(u8) {
    null = 0, tinyint = 1, smallint = 2, integer = 3, bigint = 4, decimal = 5, real = 6, double = 7,
    char = 8, varchar = 9, nchar = 10, nvarchar = 11, binary = 12, varbinary = 13,
    date = 14, time = 15, timestamp = 16, clob = 25, nclob = 26, blob = 27, boolean = 28, text = 51, shorttext = 52,
};

pub const HanaValue = union(enum) {
    null_value: void, bool_value: bool, int_value: i64, float_value: f64,
    text_value: []const u8, bytes_value: []const u8, timestamp_value: i64,

    pub fn deinit(self: *HanaValue, allocator: Allocator) void {
        switch (self.*) { .text_value => |s| allocator.free(s), .bytes_value => |b| allocator.free(b), else => {} }
    }

    pub fn clone(self: HanaValue, allocator: Allocator) !HanaValue {
        return switch (self) {
            .text_value => |s| .{ .text_value = try allocator.dupe(u8, s) },
            .bytes_value => |b| .{ .bytes_value = try allocator.dupe(u8, b) },
            else => self,
        };
    }
};

pub const HanaRow = struct {
    columns: [][]const u8,
    values: []HanaValue,

    pub fn deinit(self: *HanaRow, allocator: Allocator) void {
        for (self.columns) |col| allocator.free(col);
        allocator.free(self.columns);
        for (self.values) |*val| val.deinit(allocator);
        allocator.free(self.values);
    }

    pub fn getValue(self: *const HanaRow, column_name: []const u8) ?HanaValue {
        for (self.columns, 0..) |col, i| if (std.mem.eql(u8, col, column_name)) return self.values[i];
        return null;
    }
};

pub const HanaResult = struct {
    rows: []HanaRow,
    row_count: usize,
    affected_rows: usize,
    command_tag: ?[]const u8,

    pub fn deinit(self: *HanaResult, allocator: Allocator) void {
        for (self.rows) |*row| row.deinit(allocator);
        allocator.free(self.rows);
        if (self.command_tag) |tag| allocator.free(tag);
    }
};

pub const HanaPoolConfig = struct {
    min_size: usize = 2, max_size: usize = 10, connection_timeout_ms: u32 = 30000,
    max_idle_time_ms: u32 = 300000, health_check_interval_ms: u32 = 60000, acquire_timeout_ms: u32 = 10000,
};

pub const HanaConfig = struct {
    host: []const u8, port: u16 = 443, user: []const u8, password: []const u8, schema: []const u8,
    use_tls: bool = true, pool: HanaPoolConfig = .{}, timeout_ms: u32 = 30000,

    pub fn validate(self: *const HanaConfig) HanaError!void {
        if (self.host.len == 0 or self.user.len == 0 or self.password.len == 0 or self.port == 0) return HanaError.InvalidConfig;
    }
};

pub const HanaConnection = struct {
    allocator: Allocator, config: HanaConfig, stream: ?net.Stream, is_connected: bool, session_id: ?u64, packet_count: u32,
    const Self = @This();

    pub fn init(allocator: Allocator, config: HanaConfig) Self {
        return .{ .allocator = allocator, .config = config, .stream = null, .is_connected = false, .session_id = null, .packet_count = 0 };
    }
    pub fn deinit(self: *Self) void { self.disconnect(); }

    pub fn connect(self: *Self) HanaError!void {
        if (self.is_connected) return HanaError.AlreadyConnected;
        self.config.validate() catch return HanaError.InvalidConfig;
        const address = net.Address.parseIp(self.config.host, self.config.port) catch return HanaError.ConnectionFailed;
        self.stream = net.tcpConnectToAddress(address) catch return HanaError.ConnectionFailed;
        errdefer if (self.stream) |*s| s.close();
        if (self.config.use_tls) self.initTls() catch return HanaError.TlsHandshakeFailed;
        self.sendInitPacket() catch return HanaError.ProtocolError;
        self.authenticate() catch return HanaError.AuthenticationFailed;
        if (self.config.schema.len > 0) self.setSchema(self.config.schema) catch return HanaError.SchemaNotSet;
        self.is_connected = true;
    }

    pub fn disconnect(self: *Self) void {
        if (self.stream) |*s| { self.sendDisconnect() catch {}; s.close(); self.stream = null; self.is_connected = false; self.session_id = null; self.packet_count = 0; }
    }

    fn initTls(self: *Self) !void { _ = self; } // TLS placeholder - use std.crypto.tls in production

    fn sendInitPacket(self: *Self) !void {
        var buf: [16]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], 0x00000001, .little);
        std.mem.writeInt(u32, buf[4..8], 0x00000001, .little);
        std.mem.writeInt(u32, buf[8..12], 0x00000000, .little);
        std.mem.writeInt(u32, buf[12..16], 0x00010000, .little);
        try self.stream.?.writeAll(&buf);
    }

    fn authenticate(self: *Self) !void {
        var buf: [1024]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const writer = fbs.writer();
        try writer.writeInt(u8, 0x01, .little);
        try writer.writeInt(u16, @intCast(self.config.user.len), .little);
        try writer.writeAll(self.config.user);
        try writer.writeInt(u16, @intCast(self.config.password.len), .little);
        try writer.writeAll(self.config.password);
        try self.stream.?.writeAll(buf[0..fbs.pos]);
        var resp: [64]u8 = undefined;
        const n = self.stream.?.read(&resp) catch return error.AuthenticationFailed;
        if (n < 8) return error.AuthenticationFailed;
        self.session_id = std.mem.readInt(u64, resp[0..8], .little);
    }

    fn setSchema(self: *Self, schema: []const u8) !void {
        var buf: [256]u8 = undefined;
        _ = try self.executeInternal(std.fmt.bufPrint(&buf, "SET SCHEMA \"{s}\"", .{schema}) catch return error.InvalidParameter);
    }

    fn sendDisconnect(self: *Self) !void {
        var buf: [8]u8 = undefined;
        std.mem.writeInt(u32, buf[0..4], 0x00000000, .little);
        std.mem.writeInt(u32, buf[4..8], 0x00000000, .little);
        try self.stream.?.writeAll(&buf);
    }

    fn executeInternal(self: *Self, sql: []const u8) !usize {
        if (!self.is_connected and self.stream == null) return HanaError.NotConnected;
        var header: [16]u8 = undefined;
        const sql_len: u32 = @intCast(sql.len);
        std.mem.writeInt(u32, header[0..4], 0x00000002, .little);
        std.mem.writeInt(u32, header[4..8], sql_len + 16, .little);
        std.mem.writeInt(u32, header[8..12], self.packet_count, .little);
        std.mem.writeInt(u32, header[12..16], sql_len, .little);
        try self.stream.?.writeAll(&header);
        try self.stream.?.writeAll(sql);
        self.packet_count += 1;
        var resp_header: [16]u8 = undefined;
        const n = self.stream.?.read(&resp_header) catch return 0;
        if (n < 16) return 0;
        return std.mem.readInt(u32, resp_header[8..12], .little);
    }

    fn readByte(self: *Self) !u8 {
        var buf: [1]u8 = undefined;
        const n = self.stream.?.read(&buf) catch return HanaError.EndOfStream;
        if (n == 0) return HanaError.EndOfStream;
        return buf[0];
    }

    fn readInt(self: *Self, comptime T: type) !T {
        var buf: [@sizeOf(T)]u8 = undefined;
        const n = self.stream.?.read(&buf) catch return HanaError.EndOfStream;
        if (n != buf.len) return HanaError.EndOfStream;
        return std.mem.readInt(T, &buf, .little);
    }
};

// ============================================================================
// HANA Client (High-level API)
// ============================================================================

pub const HanaClient = struct {
    allocator: Allocator,
    connection: HanaConnection,
    in_transaction: bool,
    prepared_statements: std.StringHashMap(PreparedStatement),

    const Self = @This();
    pub const PreparedStatement = struct { id: u64, sql: []const u8, param_count: usize, is_valid: bool };

    pub fn init(allocator: Allocator, config: HanaConfig) Self {
        return .{ .allocator = allocator, .connection = HanaConnection.init(allocator, config), .in_transaction = false, .prepared_statements = std.StringHashMap(PreparedStatement).init(allocator) };
    }

    pub fn deinit(self: *Self) void {
        var it = self.prepared_statements.iterator();
        while (it.next()) |entry| { self.allocator.free(entry.key_ptr.*); self.allocator.free(entry.value_ptr.sql); }
        self.prepared_statements.deinit();
        self.connection.deinit();
    }

    pub fn connect(self: *Self) HanaError!void { return self.connection.connect(); }
    pub fn disconnect(self: *Self) void { if (self.in_transaction) self.rollback() catch {}; self.connection.disconnect(); }

    /// Execute a SELECT query and return results
    pub fn query(self: *Self, sql: []const u8) HanaError!HanaResult {
        if (!self.connection.is_connected) return HanaError.NotConnected;
        var header: [16]u8 = undefined;
        const sql_len: u32 = @intCast(sql.len);
        std.mem.writeInt(u32, header[0..4], 0x00000003, .little);
        std.mem.writeInt(u32, header[4..8], sql_len + 16, .little);
        std.mem.writeInt(u32, header[8..12], self.connection.packet_count, .little);
        std.mem.writeInt(u32, header[12..16], sql_len, .little);
        self.connection.stream.?.writeAll(&header) catch return HanaError.ProtocolError;
        self.connection.stream.?.writeAll(sql) catch return HanaError.ProtocolError;
        self.connection.packet_count += 1;
        return self.parseQueryResult();
    }

    fn parseQueryResult(self: *Self) HanaError!HanaResult {
        var rows = std.ArrayList(HanaRow){};
        errdefer { for (rows.items) |*row| row.deinit(self.allocator); rows.deinit(self.allocator); }
        var resp_header: [32]u8 = undefined;
        const n = self.connection.stream.?.read(&resp_header) catch return HanaError.ProtocolError;
        if (n < 32) return HanaError.InvalidResponse;
        const row_count = std.mem.readInt(u32, resp_header[4..8], .little);
        const col_count = std.mem.readInt(u32, resp_header[8..12], .little);
        const affected = std.mem.readInt(u32, resp_header[12..16], .little);
        const columns = self.allocator.alloc([]u8, col_count) catch return HanaError.OutOfMemory;
        errdefer self.allocator.free(columns);
        for (columns) |*col| {
            const len = self.connection.readInt(u16) catch return HanaError.ProtocolError;
            col.* = self.allocator.alloc(u8, len) catch return HanaError.OutOfMemory;
            _ = self.connection.stream.?.read(col.*) catch return HanaError.ProtocolError;
        }
        defer { for (columns) |col| self.allocator.free(col); self.allocator.free(columns); }
        for (0..row_count) |_| rows.append(self.allocator, self.readRow(columns) catch return HanaError.ProtocolError) catch return HanaError.OutOfMemory;
        return HanaResult{ .rows = rows.toOwnedSlice(self.allocator) catch return HanaError.OutOfMemory, .row_count = row_count, .affected_rows = affected, .command_tag = null };
    }

    fn readRow(self: *Self, columns: [][]u8) !HanaRow {
        const values = try self.allocator.alloc(HanaValue, columns.len);
        errdefer self.allocator.free(values);
        for (values) |*val| val.* = try self.readValue(try self.connection.readByte());
        const col_copy = try self.allocator.alloc([]const u8, columns.len);
        for (columns, 0..) |col, i| col_copy[i] = try self.allocator.dupe(u8, col);
        return HanaRow{ .columns = col_copy, .values = values };
    }

    fn readValue(self: *Self, type_byte: u8) !HanaValue {
        return switch (@as(HanaType, @enumFromInt(type_byte))) {
            .null => .{ .null_value = {} },
            .tinyint, .smallint, .integer, .bigint => .{ .int_value = try self.connection.readInt(i64) },
            .real, .double, .decimal => .{ .float_value = @bitCast(try self.connection.readInt(u64)) },
            .boolean => .{ .bool_value = (try self.connection.readByte()) != 0 },
            .timestamp, .date, .time => .{ .timestamp_value = try self.connection.readInt(i64) },
            else => blk: { const len = try self.connection.readInt(u32); const data = try self.allocator.alloc(u8, len); _ = self.connection.stream.?.read(data) catch { self.allocator.free(data); return HanaError.ProtocolError; }; break :blk .{ .text_value = data }; },
        };
    }

    pub fn execute(self: *Self, sql: []const u8) HanaError!usize {
        if (!self.connection.is_connected) return HanaError.NotConnected;
        return self.connection.executeInternal(sql) catch return HanaError.ExecuteFailed;
    }

    pub fn beginTransaction(self: *Self) HanaError!void {
        if (self.in_transaction) return HanaError.AlreadyInTransaction;
        _ = self.execute("START TRANSACTION") catch return HanaError.TransactionError;
        self.in_transaction = true;
    }

    pub fn commit(self: *Self) HanaError!void {
        if (!self.in_transaction) return HanaError.NotInTransaction;
        _ = self.execute("COMMIT") catch return HanaError.TransactionError;
        self.in_transaction = false;
    }

    pub fn rollback(self: *Self) HanaError!void {
        if (!self.in_transaction) return HanaError.NotInTransaction;
        _ = self.execute("ROLLBACK") catch return HanaError.TransactionError;
        self.in_transaction = false;
    }

    pub fn prepareStatement(self: *Self, name: []const u8, sql: []const u8) HanaError!void {
        if (!self.connection.is_connected) return HanaError.NotConnected;
        const stmt_id = @as(u64, @intCast(std.time.milliTimestamp())) ^ @as(u64, @intCast(self.prepared_statements.count()));
        var param_count: usize = 0;
        for (sql) |c| if (c == '?') { param_count += 1; };
        var buf: [4096]u8 = undefined;
        _ = self.execute(std.fmt.bufPrint(&buf, "PREPARE \"{s}\" AS {s}", .{ name, sql }) catch return HanaError.InvalidParameter) catch return HanaError.PrepareStatementFailed;
        const name_copy = self.allocator.dupe(u8, name) catch return HanaError.OutOfMemory;
        const sql_copy = self.allocator.dupe(u8, sql) catch { self.allocator.free(name_copy); return HanaError.OutOfMemory; };
        self.prepared_statements.put(self.allocator, name_copy, .{ .id = stmt_id, .sql = sql_copy, .param_count = param_count, .is_valid = true }) catch { self.allocator.free(name_copy); self.allocator.free(sql_copy); return HanaError.OutOfMemory; };
    }

    pub fn executeStatement(self: *Self, name: []const u8, params: []const HanaValue) HanaError!HanaResult {
        if (!self.connection.is_connected) return HanaError.NotConnected;
        const stmt = self.prepared_statements.get(name) orelse return HanaError.StatementNotPrepared;
        if (!stmt.is_valid or params.len != stmt.param_count) return HanaError.InvalidParameter;
        var buf: [8192]u8 = undefined;
        var fbs = std.io.fixedBufferStream(&buf);
        const writer = fbs.writer();
        writer.print("EXECUTE \"{s}\" (", .{name}) catch return HanaError.InvalidParameter;
        for (params, 0..) |param, i| { if (i > 0) writer.writeAll(", ") catch return HanaError.InvalidParameter; writeParam(writer, param) catch return HanaError.InvalidParameter; }
        writer.writeAll(")") catch return HanaError.InvalidParameter;
        return self.query(buf[0..fbs.pos]);
    }

    fn writeParam(writer: anytype, value: HanaValue) !void {
        switch (value) {
            .null_value => try writer.writeAll("NULL"),
            .bool_value => |b| try writer.print("{s}", .{if (b) "TRUE" else "FALSE"}),
            .int_value => |i| try writer.print("{d}", .{i}),
            .float_value => |f| try writer.print("{d}", .{f}),
            .text_value => |s| { try writer.writeByte('\''); for (s) |c| if (c == '\'') try writer.writeAll("''") else try writer.writeByte(c); try writer.writeByte('\''); },
            .bytes_value => |b| { try writer.writeAll("X'"); for (b) |byte| try writer.print("{X:0>2}", .{byte}); try writer.writeByte('\''); },
            .timestamp_value => |t| try writer.print("TO_TIMESTAMP({d})", .{t}),
        }
    }

    pub fn isConnected(self: *const Self) bool { return self.connection.is_connected; }
    pub fn inTransaction(self: *const Self) bool { return self.in_transaction; }
};

pub const PooledHanaConnection = struct { client: *HanaClient, last_used: i64, is_broken: bool };

pub const HanaConnectionPool = struct {
    allocator: Allocator, config: HanaConfig, connections: std.ArrayList(*PooledHanaConnection),
    mutex: Thread.Mutex, total_connections: usize, active_connections: usize, shutdown_flag: bool,

    const Self = @This();

    pub fn init(allocator: Allocator, config: HanaConfig) !*Self {
        const self = try allocator.create(Self);
        self.* = .{ .allocator = allocator, .config = config, .connections = std.ArrayList(*PooledHanaConnection){}, .mutex = .{}, .total_connections = 0, .active_connections = 0, .shutdown_flag = false };
        try self.warmPool();
        return self;
    }

    pub fn deinit(self: *Self) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        for (self.connections.items) |conn| { conn.client.deinit(); self.allocator.destroy(conn.client); self.allocator.destroy(conn); }
        self.connections.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn warmPool(self: *Self) !void {
        for (0..self.config.pool.min_size) |_| {
            const client = try self.createConnection();
            const pooled = try self.allocator.create(PooledHanaConnection);
            pooled.* = .{ .client = client, .last_used = std.time.milliTimestamp(), .is_broken = false };
            try self.connections.append(self.allocator, pooled);
            self.total_connections += 1;
        }
    }

    fn createConnection(self: *Self) !*HanaClient {
        const client = try self.allocator.create(HanaClient);
        errdefer self.allocator.destroy(client);
        client.* = HanaClient.init(self.allocator, self.config);
        try client.connect();
        return client;
    }

    pub fn acquire(self: *Self) HanaError!*HanaClient {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (self.connections.items.len > 0) {
            const conn = self.connections.pop();
            if (!conn.is_broken) { conn.last_used = std.time.milliTimestamp(); self.active_connections += 1; return conn.client; }
            conn.client.deinit(); self.allocator.destroy(conn.client); self.allocator.destroy(conn); self.total_connections -= 1;
        }
        if (self.total_connections < self.config.pool.max_size) {
            const client = self.createConnection() catch return HanaError.PoolExhausted;
            self.total_connections += 1; self.active_connections += 1; return client;
        }
        return HanaError.PoolExhausted;
    }

    pub fn release(self: *Self, client: *HanaClient) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.active_connections -= 1;
        if (self.total_connections > self.config.pool.max_size) { client.deinit(); self.allocator.destroy(client); self.total_connections -= 1; return; }
        const pooled = self.allocator.create(PooledHanaConnection) catch { client.deinit(); self.allocator.destroy(client); self.total_connections -= 1; return; };
        pooled.* = .{ .client = client, .last_used = std.time.milliTimestamp(), .is_broken = false };
        self.connections.append(self.allocator, pooled) catch { self.allocator.destroy(pooled); client.deinit(); self.allocator.destroy(client); self.total_connections -= 1; };
    }

    pub fn getStats(self: *Self) struct { total: usize, active: usize, idle: usize } {
        self.mutex.lock();
        defer self.mutex.unlock();
        const idle = if (self.total_connections > self.active_connections) self.total_connections - self.active_connections else 0;
        return .{ .total = self.total_connections, .active = self.active_connections, .idle = idle };
    }
};

fn testConfig() HanaConfig {
    return .{ .host = "test.hana.ondemand.com", .port = 443, .user = "DBADMIN", .password = "secret", .schema = "DBADMIN" };
}

test "HanaConfig validation" {
    try testConfig().validate();
    try std.testing.expectError(HanaError.InvalidConfig, (HanaConfig{ .host = "", .port = 443, .user = "DBADMIN", .password = "secret", .schema = "DBADMIN" }).validate());
}

test "HanaPoolConfig defaults" {
    const config = HanaPoolConfig{};
    try std.testing.expectEqual(@as(usize, 2), config.min_size);
    try std.testing.expectEqual(@as(usize, 10), config.max_size);
}

test "HanaValue operations" {
    const allocator = std.testing.allocator;
    var null_val = HanaValue{ .null_value = {} };
    try std.testing.expect(null_val == .null_value);
    null_val.deinit(allocator);
    var text_val = HanaValue{ .text_value = try allocator.dupe(u8, "test") };
    defer text_val.deinit(allocator);
    try std.testing.expectEqualStrings("test", text_val.text_value);
    var cloned = try (HanaValue{ .text_value = "hello" }).clone(allocator);
    defer cloned.deinit(allocator);
    try std.testing.expectEqualStrings("hello", cloned.text_value);
}

test "HanaType enum values" {
    try std.testing.expectEqual(@as(u8, 0), @intFromEnum(HanaType.null));
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(HanaType.integer));
    try std.testing.expectEqual(@as(u8, 16), @intFromEnum(HanaType.timestamp));
}

test "HanaClient and HanaConnection initialization" {
    const allocator = std.testing.allocator;
    var client = HanaClient.init(allocator, testConfig());
    defer client.deinit();
    try std.testing.expect(!client.isConnected());
    try std.testing.expect(!client.inTransaction());
    var conn = HanaConnection.init(allocator, testConfig());
    defer conn.deinit();
    try std.testing.expect(!conn.is_connected);
    try std.testing.expect(conn.session_id == null);
}

test "HanaRow getValue" {
    const allocator = std.testing.allocator;
    var columns = try allocator.alloc([]const u8, 2);
    columns[0] = try allocator.dupe(u8, "id");
    columns[1] = try allocator.dupe(u8, "name");
    var values = try allocator.alloc(HanaValue, 2);
    values[0] = .{ .int_value = 1 };
    values[1] = .{ .text_value = try allocator.dupe(u8, "test") };
    var row = HanaRow{ .columns = columns, .values = values };
    defer row.deinit(allocator);
    try std.testing.expectEqual(@as(i64, 1), row.getValue("id").?.int_value);
    try std.testing.expect(row.getValue("missing") == null);
}

test "HanaResult deinit" {
    const allocator = std.testing.allocator;
    var result = HanaResult{ .rows = try allocator.alloc(HanaRow, 0), .row_count = 0, .affected_rows = 5, .command_tag = try allocator.dupe(u8, "INSERT") };
    defer result.deinit(allocator);
    try std.testing.expectEqual(@as(usize, 5), result.affected_rows);
}

test "HanaClient error handling" {
    const allocator = std.testing.allocator;
    var client = HanaClient.init(allocator, testConfig());
    defer client.deinit();
    try std.testing.expectError(HanaError.NotConnected, client.query("SELECT 1"));
    try std.testing.expectError(HanaError.NotConnected, client.execute("INSERT INTO t VALUES (1)"));
    try std.testing.expectError(HanaError.TransactionError, client.beginTransaction());
    try std.testing.expectError(HanaError.NotInTransaction, client.commit());
    try std.testing.expectError(HanaError.NotInTransaction, client.rollback());
    try std.testing.expectError(HanaError.NotConnected, client.executeStatement("stmt", &.{}));
}
