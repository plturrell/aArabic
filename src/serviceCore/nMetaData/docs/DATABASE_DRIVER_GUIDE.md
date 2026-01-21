# Database Driver Implementation Guide

**Version:** 1.0  
**Last Updated:** January 20, 2026  
**Audience:** Developers implementing new database drivers

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Implementing a New Driver](#implementing-a-new-driver)
4. [Wire Protocol Guidelines](#wire-protocol-guidelines)
5. [Type System Integration](#type-system-integration)
6. [Connection Management](#connection-management)
7. [Transaction Support](#transaction-support)
8. [Testing Requirements](#testing-requirements)
9. [Performance Optimization](#performance-optimization)
10. [Examples](#examples)

---

## Overview

This guide provides detailed instructions for implementing new database drivers for the nMetaData abstraction layer. The abstraction layer supports multiple databases through a common interface while allowing database-specific optimizations.

### Design Goals

- **Abstraction:** Hide database-specific details behind common interface
- **Performance:** Allow database-specific optimizations
- **Safety:** Type-safe operations, memory safety
- **Testability:** Easy to mock and test
- **Extensibility:** Simple to add new databases

### Current Drivers

| Driver | Protocol | Status | Performance |
|--------|----------|--------|-------------|
| PostgreSQL | Wire Protocol | ✅ Production | 1,000+ QPS |
| SAP HANA | SQL Protocol | ✅ Production | 2,000+ QPS |
| SQLite | C API | ✅ Production | 15,000+ QPS |

---

## Architecture

### Core Abstraction: DbClient

The `DbClient` interface defines the contract all drivers must implement:

```zig
pub const DbClient = struct {
    vtable: *const VTable,
    context: *anyopaque,
    allocator: Allocator,
    
    pub const VTable = struct {
        // Connection lifecycle
        connect: *const fn (*anyopaque, []const u8) anyerror!void,
        disconnect: *const fn (*anyopaque) void,
        ping: *const fn (*anyopaque) anyerror!bool,
        
        // Query execution
        execute: *const fn (*anyopaque, []const u8, []const Value) anyerror!ResultSet,
        prepare: *const fn (*anyopaque, []const u8) anyerror!*PreparedStatement,
        
        // Transaction management
        begin: *const fn (*anyopaque) anyerror!*Transaction,
        
        // Metadata
        get_dialect: *const fn (*anyopaque) Dialect,
        get_server_version: *const fn (*anyopaque) anyerror![]const u8,
    };
    
    // Public API wraps vtable calls
    pub fn connect(self: *DbClient, connection_string: []const u8) !void {
        return self.vtable.connect(self.context, connection_string);
    }
    
    pub fn execute(self: *DbClient, sql: []const u8, params: []const Value) !ResultSet {
        return self.vtable.execute(self.context, sql, params);
    }
    
    // ... more methods
};
```

### Value Type System

The `Value` enum provides cross-database type representation:

```zig
pub const Value = union(enum) {
    null,
    bool: bool,
    int32: i32,
    int64: i64,
    float32: f32,
    float64: f64,
    string: []const u8,
    bytes: []const u8,
    timestamp: i64,      // Unix timestamp (microseconds)
    uuid: [16]u8,
    json: []const u8,    // JSON string
    array: []const Value,
    
    pub fn format(
        self: Value,
        comptime fmt: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        // Format for debugging/logging
    }
    
    pub fn eql(self: Value, other: Value) bool {
        // Deep equality check
    }
};
```

### Result Set

Query results are returned as a `ResultSet`:

```zig
pub const ResultSet = struct {
    columns: []const Column,
    rows: std.ArrayList(Row),
    allocator: Allocator,
    
    pub const Column = struct {
        name: []const u8,
        type: ValueType,
        nullable: bool,
    };
    
    pub const Row = struct {
        values: []const Value,
        
        pub fn get(self: Row, index: usize) ?Value {
            if (index >= self.values.len) return null;
            return self.values[index];
        }
        
        pub fn getByName(self: Row, columns: []const Column, name: []const u8) ?Value {
            for (columns, 0..) |col, i| {
                if (std.mem.eql(u8, col.name, name)) {
                    return self.get(i);
                }
            }
            return null;
        }
    };
    
    pub fn deinit(self: *ResultSet) void {
        // Free all allocated memory
    }
};
```

---

## Implementing a New Driver

### Step 1: Create Driver Structure

Create a new file `src/serviceCore/nMetaData/zig/db/drivers/mydb.zig`:

```zig
const std = @import("std");
const DbClient = @import("../client.zig").DbClient;
const Value = @import("../client.zig").Value;
const ResultSet = @import("../client.zig").ResultSet;
const Dialect = @import("../client.zig").Dialect;

pub const MyDbClient = struct {
    allocator: std.mem.Allocator,
    connection: ?Connection = null,
    dialect: Dialect,
    
    // Driver-specific state
    const Connection = struct {
        socket: std.net.Stream,
        state: ConnectionState,
        // Add database-specific fields
    };
    
    const ConnectionState = enum {
        idle,
        in_transaction,
        error_state,
    };
    
    pub fn init(allocator: std.mem.Allocator) !*MyDbClient {
        const self = try allocator.create(MyDbClient);
        self.* = .{
            .allocator = allocator,
            .dialect = .MyDb,
        };
        return self;
    }
    
    pub fn deinit(self: *MyDbClient) void {
        if (self.connection) |*conn| {
            conn.socket.close();
        }
        self.allocator.destroy(self);
    }
    
    // Implement DbClient interface
    pub fn toDbClient(self: *MyDbClient) DbClient {
        return DbClient{
            .vtable = &vtable,
            .context = self,
            .allocator = self.allocator,
        };
    }
    
    const vtable = DbClient.VTable{
        .connect = connect,
        .disconnect = disconnect,
        .ping = ping,
        .execute = execute,
        .prepare = prepare,
        .begin = begin,
        .get_dialect = getDialect,
        .get_server_version = getServerVersion,
    };
    
    // VTable implementations
    fn connect(ctx: *anyopaque, connection_string: []const u8) !void {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Parse connection string
        // Establish connection
        // Authenticate
    }
    
    fn disconnect(ctx: *anyopaque) void {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        if (self.connection) |*conn| {
            conn.socket.close();
        }
        self.connection = null;
    }
    
    fn ping(ctx: *anyopaque) !bool {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Send ping packet
        // Return true if successful
    }
    
    fn execute(ctx: *anyopaque, sql: []const u8, params: []const Value) !ResultSet {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Send query
        // Parse results
        // Return ResultSet
    }
    
    fn prepare(ctx: *anyopaque, sql: []const u8) !*PreparedStatement {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Prepare statement
        // Return prepared statement handle
    }
    
    fn begin(ctx: *anyopaque) !*Transaction {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Start transaction
        // Return transaction handle
    }
    
    fn getDialect(ctx: *anyopaque) Dialect {
        _ = ctx;
        return .MyDb;
    }
    
    fn getServerVersion(ctx: *anyopaque) ![]const u8 {
        const self = @as(*MyDbClient, @ptrCast(@alignCast(ctx)));
        // Query server version
    }
};
```

### Step 2: Add Dialect Support

Update `src/serviceCore/nMetaData/zig/db/client.zig`:

```zig
pub const Dialect = enum {
    PostgreSQL,
    HANA,
    SQLite,
    MyDb,  // Add your database
    
    pub fn fromString(s: []const u8) ?Dialect {
        if (std.mem.eql(u8, s, "postgresql")) return .PostgreSQL;
        if (std.mem.eql(u8, s, "hana")) return .HANA;
        if (std.mem.eql(u8, s, "sqlite")) return .SQLite;
        if (std.mem.eql(u8, s, "mydb")) return .MyDb;
        return null;
    }
    
    pub fn toString(self: Dialect) []const u8 {
        return switch (self) {
            .PostgreSQL => "postgresql",
            .HANA => "hana",
            .SQLite => "sqlite",
            .MyDb => "mydb",
        };
    }
};
```

### Step 3: Implement Query Builder Support

Update `src/serviceCore/nMetaData/zig/db/query_builder.zig`:

```zig
fn generateSelectDialect(self: *QueryBuilder, dialect: Dialect) !void {
    switch (dialect) {
        .PostgreSQL => try self.generatePostgreSQL(),
        .HANA => try self.generateHANA(),
        .SQLite => try self.generateSQLite(),
        .MyDb => try self.generateMyDb(),
    }
}

fn generateMyDb(self: *QueryBuilder) !void {
    // Generate MyDB-specific SQL
    // Handle MyDB-specific features
    // Optimize for MyDB
}
```

### Step 4: Add Connection Factory

Update client initialization:

```zig
pub fn createClient(allocator: Allocator, config: DbConfig) !DbClient {
    return switch (config.dialect) {
        .PostgreSQL => {
            const pg = try PostgresClient.init(allocator);
            return pg.toDbClient();
        },
        .HANA => {
            const hana = try HanaClient.init(allocator);
            return hana.toDbClient();
        },
        .SQLite => {
            const sqlite = try SqliteClient.init(allocator);
            return sqlite.toDbClient();
        },
        .MyDb => {
            const mydb = try MyDbClient.init(allocator);
            return mydb.toDbClient();
        },
    };
}
```

---

## Wire Protocol Guidelines

### Connection Handshake

Most database wire protocols follow this pattern:

1. **Client connects** to server
2. **Server sends** startup message
3. **Client sends** authentication credentials
4. **Server responds** with success/failure
5. **Connection established**

Example pattern:

```zig
fn connect(self: *MyDbClient, connection_string: []const u8) !void {
    // 1. Parse connection string
    const config = try parseConnectionString(connection_string);
    
    // 2. Connect TCP socket
    const address = try std.net.Address.parseIp4(config.host, config.port);
    const socket = try std.net.tcpConnectToAddress(address);
    
    // 3. Read server greeting
    var greeting = try readPacket(socket);
    defer greeting.deinit();
    
    // 4. Send authentication
    try sendAuthPacket(socket, config.username, config.password);
    
    // 5. Read auth response
    var auth_response = try readPacket(socket);
    defer auth_response.deinit();
    
    if (!auth_response.isSuccess()) {
        return error.AuthenticationFailed;
    }
    
    // 6. Store connection
    self.connection = Connection{
        .socket = socket,
        .state = .idle,
    };
}
```

### Packet Structure

Define clear packet structures:

```zig
const Packet = struct {
    header: PacketHeader,
    payload: []const u8,
    
    const PacketHeader = packed struct {
        length: u32,
        type: PacketType,
        sequence: u16,
    };
    
    const PacketType = enum(u8) {
        ok = 0x00,
        error = 0xFF,
        query = 0x03,
        result = 0x04,
        // ... more types
    };
    
    fn read(stream: std.net.Stream, allocator: Allocator) !Packet {
        // Read header
        var header_bytes: [@sizeOf(PacketHeader)]u8 = undefined;
        _ = try stream.readAll(&header_bytes);
        const header = std.mem.bytesToValue(PacketHeader, &header_bytes);
        
        // Read payload
        const payload = try allocator.alloc(u8, header.length);
        _ = try stream.readAll(payload);
        
        return Packet{
            .header = header,
            .payload = payload,
        };
    }
    
    fn write(self: Packet, stream: std.net.Stream) !void {
        // Write header
        const header_bytes = std.mem.toBytes(self.header);
        try stream.writeAll(&header_bytes);
        
        // Write payload
        try stream.writeAll(self.payload);
    }
    
    fn deinit(self: *Packet, allocator: Allocator) void {
        allocator.free(self.payload);
    }
};
```

### Binary Protocol vs Text Protocol

**Binary Protocol (Recommended):**
- More efficient
- Type-safe
- Compact representation
- Examples: PostgreSQL extended protocol, MySQL binary protocol

**Text Protocol:**
- Easier to debug
- Human-readable
- Less efficient
- Examples: HTTP, Redis protocol

Choose binary protocol for performance, text protocol for simplicity.

---

## Type System Integration

### Mapping Database Types to Value

Create a type mapping table:

```zig
const TypeMap = struct {
    fn toValue(db_type: MyDbType, data: []const u8) !Value {
        return switch (db_type) {
            .INT => .{ .int32 = try std.fmt.parseInt(i32, data, 10) },
            .BIGINT => .{ .int64 = try std.fmt.parseInt(i64, data, 10) },
            .VARCHAR => .{ .string = data },
            .TIMESTAMP => .{ .timestamp = try parseTimestamp(data) },
            .UUID => .{ .uuid = try parseUuid(data) },
            // ... more types
        };
    }
    
    fn fromValue(value: Value) !struct { MyDbType, []const u8 } {
        return switch (value) {
            .int32 => |i| .{ .INT, try std.fmt.allocPrint(allocator, "{d}", .{i}) },
            .int64 => |i| .{ .BIGINT, try std.fmt.allocPrint(allocator, "{d}", .{i}) },
            .string => |s| .{ .VARCHAR, s },
            .timestamp => |ts| .{ .TIMESTAMP, try formatTimestamp(ts) },
            // ... more types
        };
    }
};
```

### Handling NULL Values

Always check for NULL:

```zig
fn readValue(reader: anytype, db_type: MyDbType, is_null: bool) !Value {
    if (is_null) {
        return .null;
    }
    
    // Read actual value
    const data = try reader.readValue(db_type);
    return try TypeMap.toValue(db_type, data);
}
```

### Type Conversion Best Practices

1. **Preserve precision:** Use appropriate numeric types
2. **Handle overflow:** Check bounds before conversion
3. **Validate strings:** Ensure valid UTF-8
4. **Normalize timestamps:** Use consistent timezone (UTC)
5. **Document limitations:** Be clear about unsupported types

---

## Connection Management

### Connection Pool Implementation

Implement pooling for production use:

```zig
pub const ConnectionPool = struct {
    allocator: Allocator,
    connections: std.ArrayList(*MyDbClient),
    available: std.ArrayList(*MyDbClient),
    mutex: std.Thread.Mutex,
    max_size: usize,
    
    pub fn init(allocator: Allocator, config: PoolConfig) !*ConnectionPool {
        const pool = try allocator.create(ConnectionPool);
        pool.* = .{
            .allocator = allocator,
            .connections = std.ArrayList(*MyDbClient).init(allocator),
            .available = std.ArrayList(*MyDbClient).init(allocator),
            .mutex = .{},
            .max_size = config.max_size,
        };
        
        // Pre-create minimum connections
        for (0..config.min_size) |_| {
            const conn = try MyDbClient.init(allocator);
            try conn.connect(config.connection_string);
            try pool.connections.append(conn);
            try pool.available.append(conn);
        }
        
        return pool;
    }
    
    pub fn acquire(self: *ConnectionPool) !*MyDbClient {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Return available connection
        if (self.available.items.len > 0) {
            return self.available.pop();
        }
        
        // Create new connection if under limit
        if (self.connections.items.len < self.max_size) {
            const conn = try MyDbClient.init(self.allocator);
            try self.connections.append(conn);
            return conn;
        }
        
        // Wait for available connection (or timeout)
        return error.PoolExhausted;
    }
    
    pub fn release(self: *ConnectionPool, conn: *MyDbClient) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        
        // Validate connection still healthy
        if (conn.ping() catch false) {
            self.available.append(conn) catch {};
        } else {
            // Connection broken, remove from pool
            conn.deinit();
            // Remove from connections list
        }
    }
    
    pub fn deinit(self: *ConnectionPool) void {
        for (self.connections.items) |conn| {
            conn.deinit();
        }
        self.connections.deinit();
        self.available.deinit();
        self.allocator.destroy(self);
    }
};
```

### Health Checks

Implement robust health checking:

```zig
fn ping(self: *MyDbClient) !bool {
    if (self.connection == null) return false;
    
    // Send lightweight query
    const result = self.execute("SELECT 1", &.{}) catch return false;
    defer result.deinit();
    
    return result.rows.items.len == 1;
}

fn validateConnection(self: *MyDbClient) !void {
    // Check connection state
    if (self.connection == null) return error.NotConnected;
    
    // Verify can execute query
    if (!try self.ping()) return error.ConnectionDead;
    
    // Check transaction state is clean
    if (self.connection.?.state != .idle) return error.InvalidState;
}
```

---

## Transaction Support

### Transaction Interface

```zig
pub const Transaction = struct {
    client: *MyDbClient,
    id: []const u8,
    state: TransactionState,
    savepoints: std.ArrayList([]const u8),
    
    const TransactionState = enum {
        active,
        committed,
        rolled_back,
        error,
    };
    
    pub fn init(client: *MyDbClient) !*Transaction {
        const tx = try client.allocator.create(Transaction);
        
        // Start transaction
        _ = try client.execute("BEGIN", &.{});
        
        tx.* = .{
            .client = client,
            .id = try generateTransactionId(),
            .state = .active,
            .savepoints = std.ArrayList([]const u8).init(client.allocator),
        };
        
        return tx;
    }
    
    pub fn commit(self: *Transaction) !void {
        if (self.state != .active) return error.TransactionNotActive;
        
        _ = try self.client.execute("COMMIT", &.{});
        self.state = .committed;
    }
    
    pub fn rollback(self: *Transaction) !void {
        if (self.state != .active) return;
        
        _ = try self.client.execute("ROLLBACK", &.{});
        self.state = .rolled_back;
    }
    
    pub fn savepoint(self: *Transaction, name: []const u8) !void {
        const sql = try std.fmt.allocPrint(
            self.client.allocator,
            "SAVEPOINT {s}",
            .{name}
        );
        defer self.client.allocator.free(sql);
        
        _ = try self.client.execute(sql, &.{});
        try self.savepoints.append(name);
    }
    
    pub fn rollbackTo(self: *Transaction, name: []const u8) !void {
        const sql = try std.fmt.allocPrint(
            self.client.allocator,
            "ROLLBACK TO SAVEPOINT {s}",
            .{name}
        );
        defer self.client.allocator.free(sql);
        
        _ = try self.client.execute(sql, &.{});
    }
    
    pub fn deinit(self: *Transaction) void {
        // Auto-rollback if not committed
        if (self.state == .active) {
            self.rollback() catch {};
        }
        
        self.savepoints.deinit();
        self.client.allocator.destroy(self);
    }
};
```

### Isolation Levels

Support standard isolation levels:

```zig
pub const IsolationLevel = enum {
    read_uncommitted,
    read_committed,
    repeatable_read,
    serializable,
    
    pub fn toSQL(self: IsolationLevel, dialect: Dialect) []const u8 {
        return switch (dialect) {
            .PostgreSQL => switch (self) {
                .read_uncommitted => "READ UNCOMMITTED",
                .read_committed => "READ COMMITTED",
                .repeatable_read => "REPEATABLE READ",
                .serializable => "SERIALIZABLE",
            },
            .MyDb => switch (self) {
                // MyDB-specific syntax
            },
        };
    }
};

pub fn beginWithIsolation(
    client: *MyDbClient,
    level: IsolationLevel,
) !*Transaction {
    const sql = try std.fmt.allocPrint(
        client.allocator,
        "BEGIN ISOLATION LEVEL {s}",
        .{level.toSQL(client.dialect)}
    );
    defer client.allocator.free(sql);
    
    _ = try client.execute(sql, &.{});
    return Transaction.init(client);
}
```

---

## Testing Requirements

### Unit Tests

Every driver must include comprehensive unit tests:

```zig
// src/serviceCore/nMetaData/zig/db/drivers/mydb_test.zig

const std = @import("std");
const testing = std.testing;
const MyDbClient = @import("mydb.zig").MyDbClient;

test "MyDb: connect and disconnect" {
    const allocator = testing.allocator;
    var client = try MyDbClient.init(allocator);
    defer client.deinit();
    
    try client.connect("mydb://localhost:1234/testdb");
    try testing.expect(client.connection != null);
    
    client.disconnect();
    try testing.expect(client.connection == null);
}

test "MyDb: execute simple query" {
    const allocator = testing.allocator;
    var client = try MyDbClient.init(allocator);
    defer client.deinit();
    
    try client.connect("mydb://localhost:1234/testdb");
    defer client.disconnect();
    
    var result = try client.execute("SELECT 1 as num", &.{});
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 1), result.rows.items.len);
    const value = result.rows.items[0].get(0).?;
    try testing.expectEqual(@as(i32, 1), value.int32);
}

test "MyDb: prepared statements" {
    // Test prepared statement functionality
}

test "MyDb: transactions" {
    // Test transaction commit/rollback
}

test "MyDb: type conversions" {
    // Test all type mappings
}

test "MyDb: error handling" {
    // Test error cases
}

test "MyDb: connection pool" {
    // Test pooling behavior
}
```

### Integration Tests

Test against real database:

```zig
test "MyDb integration: full CRUD cycle" {
    const allocator = testing.allocator;
    var client = try MyDbClient.init(allocator);
    defer client.deinit();
    
    // Use test database
    try client.connect(std.os.getenv("MYDB_TEST_URL") orelse return error.SkipZigTest);
    defer client.disconnect();
    
    // Create table
    _ = try client.execute(
        \\CREATE TABLE test_users (
        \\  id SERIAL PRIMARY KEY,
        \\  name VARCHAR(100),
        \\  email VARCHAR(100)
        \\)
    , &.{});
    
    // Insert
    _ = try client.execute(
        "INSERT INTO test_users (name, email) VALUES ($1, $2)",
        &.{
            .{ .string = "John Doe" },
            .{ .string = "john@example.com" },
        }
    );
    
    // Select
    var result = try client.execute(
        "SELECT name, email FROM test_users WHERE name = $1",
        &.{ .{ .string = "John Doe" } }
    );
    defer result.deinit();
    
    try testing.expectEqual(@as(usize, 1), result.rows.items.len);
    
    // Update
    _ = try client.execute(
        "UPDATE test_users SET email = $1 WHERE name = $2",
        &.{
            .{ .string = "newemail@example.com" },
            .{ .string = "John Doe" },
        }
    );
    
    // Delete
    _ = try client.execute(
        "DELETE FROM test_users WHERE name = $1",
        &.{ .{ .string = "John Doe" } }
    );
    
    // Cleanup
    _ = try client.execute("DROP TABLE test_users", &.{});
}
```

### Performance Benchmarks

Include performance tests:

```zig
test "MyDb benchmark: query throughput" {
    const allocator = testing.allocator;
    var client = try MyDbClient.init(allocator);
    defer client.deinit();
    
    try client.connect("mydb://localhost:1234/benchdb");
    defer client.disconnect();
    
    const iterations = 1000;
    const start = std.time.nanoTimestamp();
    
    for (0..iterations) |_| {
        var result = try client.execute("SELECT 1", &.{});
        result.deinit();
    }
    
    const end = std.time.nanoTimestamp();
    const duration_ns = @as(u64, @intCast(end - start));
    const qps = @as(f64, iterations) / (@as(f64, @floatFromInt(duration_ns)) / 1_000_000_000.0);
    
    std.debug.print("MyDb QPS: {d:.2}\n", .{qps});
    try testing.expect(qps > 100.0); // Minimum acceptable QPS
}
```

---

## Performance Optimization

### Connection Reuse

```zig
// Reuse connections instead of creating new ones
var pool = try ConnectionPool.init(allocator, .{
    .min_size = 5,
    .max_size = 20,
    .connection_string = conn_str,
});

var conn = try pool.acquire();
defer pool.release(conn);
// Use connection
```

### Prepared Statement Caching

```zig
const PreparedStatementCache = struct {
    cache: std.StringHashMap(*PreparedStatement),
    lru: std.ArrayList([]const u8),
    max_size: usize = 100,
    
    pub fn get(self: *PreparedStatementCache, sql: []const u8) ?*PreparedStatement {
        return self.cache.get(sql);
    }
    
    pub fn put(self: *PreparedStatementCache, sql: []const u8, stmt: *PreparedStatement) !void {
        if (self.cache.count() >= self.max_size) {
            // Evict least recently used
            const lru_key = self.lru.orderedRemove(0);
            _ = self.cache.remove(lru_key);
        }
        
        try self.cache.put(sql, stmt);
        try self.lru.append(sql);
    }
};
```

### Batch Operations

```zig
pub fn executeBatch(
    client: *MyDbClient,
    sql: []const u8,
    params_batch: []const []const Value,
) !void {
    // Use database-specific batch protocol if available
    // Otherwise, fall back to transaction with multiple statements
    
    var tx = try client.begin();
    defer tx.deinit();
    
    for (params_batch) |params| {
        _ = try client.execute(sql, params);
    }
    
    try tx.commit();
}
```

### Memory Optimization

```zig
// Reuse buffers
const BufferPool = struct {
    buffers: std.ArrayList([]u8),
    allocator: Allocator,
    
    pub fn acquire(self: *BufferPool, min_size: usize) ![]u8 {
        // Return existing buffer if available
        if (self.buffers.items.len > 0) {
            const buf = self.buffers.pop();
            if (buf.len >= min_size) return buf;
            self.allocator.free(buf);
        }
        
        // Allocate new buffer
        return try self.allocator.alloc(u8, min_size);
    }
    
    pub fn release(self: *BufferPool, buffer: []u8) !void {
        try self.buffers.append(buffer);
    }
};
```

---

## Examples

### Complete Driver Example

See `src/serviceCore/nMetaData/zig/db/drivers/postgres.zig` for a complete PostgreSQL driver implementation.

### Minimal Driver Template

```zig
const std = @import("std");
const DbClient = @import("../client.zig").DbClient;

pub const MinimalDriver = struct {
    allocator: std.mem.Allocator,
    connection: ?std.net.Stream = null,
    
    pub fn init(allocator: std.mem.Allocator) !*MinimalDriver {
        const self = try allocator.create(MinimalDriver);
        self.* = .{ .allocator = allocator };
        return self;
    }
    
    pub fn deinit(self: *MinimalDriver) void {
        if (self.connection) |conn| {
            conn.close();
        }
        self.allocator.destroy(self);
    }
    
    pub fn toDbClient(self: *MinimalDriver) DbClient {
        return DbClient{
            .vtable = &vtable,
            .context = self,
            .allocator = self.allocator,
        };
    }
    
    const vtable = DbClient.VTable{
        .connect = connect,
        .disconnect = disconnect,
        .ping = ping,
        .execute = execute,
        .prepare = prepare,
        .begin = begin,
        .get_dialect = getDialect,
        .get_server_version = getServerVersion,
    };
    
    fn connect(ctx: *anyopaque, connection_string: []const u8) !void {
        _ = ctx;
        _ = connection_string;
        // TODO: Implement
    }
    
    fn disconnect(ctx: *anyopaque) void {
        _ = ctx;
        // TODO: Implement
    }
    
    fn ping(ctx: *anyopaque) !bool {
        _ = ctx;
        return true;
    }
    
    fn execute(ctx: *anyopaque, sql: []const u8, params: []const Value) !ResultSet {
        _ = ctx;
        _ = sql;
        _ = params;
        // TODO: Implement
        return error.NotImplemented;
    }
    
    fn prepare(ctx: *anyopaque, sql: []const u8) !*PreparedStatement {
        _ = ctx;
        _ = sql;
        return error.NotImplemented;
    }
    
    fn begin(ctx: *anyopaque) !*Transaction {
        _ = ctx;
        return error.NotImplemented;
    }
    
    fn getDialect(ctx: *anyopaque) Dialect {
        _ = ctx;
        return .MyDb;
    }
    
    fn getServerVersion(ctx: *anyopaque) ![]const u8 {
        _ = ctx;
        return "1.0.0";
    }
};
```

---

## Checklist for New Drivers

- [ ] Driver structure created
- [ ] VTable implementation complete
- [ ] Connection management implemented
- [ ] Query execution working
- [ ] Prepared statements supported
- [ ] Transaction support added
- [ ] Type mapping complete
- [ ] Error handling robust
- [ ] Connection pooling implemented
- [ ] Unit tests written (80%+ coverage)
- [ ] Integration tests passing
- [ ] Performance benchmarks run
- [ ] Documentation complete
- [ ] Query builder integration
- [ ] Dialect support added
- [ ] Migration support tested
- [ ] Production deployment tested

---

## Resources

### Reference Implementations

- **PostgreSQL:** `src/serviceCore/nMetaData/zig/db/drivers/postgres.zig`
- **SAP HANA:** `src/serviceCore/nMetaData/zig/db/drivers/hana.zig`
- **SQLite:** `src/serviceCore/nMetaData/zig/db/drivers/sqlite.zig`

### External Documentation

- Zig Language Reference: https://ziglang.org/documentation/master/
- Database wire protocol specifications
- SQL standard documentation

### Getting Help

- Review existing driver implementations
- Check unit test examples
- Consult project maintainers
- Reference database vendor documentation

---

**Version History:**
- v1.0 (2026-01-20): Initial comprehensive driver implementation guide

**Last Updated:** January 20, 2026
