# SQLite FFI Integration Guide

**nMetaData Project**  
**Version:** 1.0  
**Last Updated:** January 20, 2026

---

## Overview

This guide provides comprehensive instructions for integrating the SQLite C library with the nMetaData SQLite driver using Zig's Foreign Function Interface (FFI). The current implementation uses stubbed C API calls that can be replaced with real SQLite bindings.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [SQLite C API Overview](#sqlite-c-api-overview)
3. [Zig FFI Basics](#zig-ffi-basics)
4. [Integration Approaches](#integration-approaches)
5. [Step-by-Step Implementation](#step-by-step-implementation)
6. [Testing with Real SQLite](#testing-with-real-sqlite)
7. [Performance Considerations](#performance-considerations)
8. [Troubleshooting](#troubleshooting)
9. [References](#references)

---

## Prerequisites

### Required Components

1. **SQLite Library**
   - SQLite 3.x development headers and libraries
   - Install via package manager or build from source

2. **Zig Compiler**
   - Zig 0.13.0 or later
   - C library linking support

3. **Build Tools**
   - C compiler (GCC, Clang, MSVC)
   - System linker

### Installation

**macOS:**
```bash
brew install sqlite3
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libsqlite3-dev
```

**Linux (RHEL/Fedora):**
```bash
sudo dnf install sqlite-devel
```

**Windows:**
```bash
# Download from https://www.sqlite.org/download.html
# Or use vcpkg:
vcpkg install sqlite3
```

---

## SQLite C API Overview

### Core Functions Used

The nMetaData SQLite driver uses the following C API functions:

#### Connection Management
```c
int sqlite3_open(const char *filename, sqlite3 **ppDb);
int sqlite3_close(sqlite3 *db);
```

#### Query Execution
```c
int sqlite3_prepare_v2(sqlite3 *db, const char *sql, int nByte, 
                       sqlite3_stmt **ppStmt, const char **pzTail);
int sqlite3_step(sqlite3_stmt *pStmt);
int sqlite3_finalize(sqlite3_stmt *pStmt);
```

#### Parameter Binding
```c
int sqlite3_bind_int(sqlite3_stmt *pStmt, int idx, int value);
int sqlite3_bind_int64(sqlite3_stmt *pStmt, int idx, sqlite3_int64 value);
int sqlite3_bind_double(sqlite3_stmt *pStmt, int idx, double value);
int sqlite3_bind_text(sqlite3_stmt *pStmt, int idx, const char *value, 
                      int n, void(*destructor)(void*));
int sqlite3_bind_blob(sqlite3_stmt *pStmt, int idx, const void *value, 
                      int n, void(*destructor)(void*));
int sqlite3_bind_null(sqlite3_stmt *pStmt, int idx);
```

#### Result Retrieval
```c
int sqlite3_column_type(sqlite3_stmt *pStmt, int iCol);
int sqlite3_column_int(sqlite3_stmt *pStmt, int iCol);
sqlite3_int64 sqlite3_column_int64(sqlite3_stmt *pStmt, int iCol);
double sqlite3_column_double(sqlite3_stmt *pStmt, int iCol);
const unsigned char *sqlite3_column_text(sqlite3_stmt *pStmt, int iCol);
const void *sqlite3_column_blob(sqlite3_stmt *pStmt, int iCol);
int sqlite3_column_bytes(sqlite3_stmt *pStmt, int iCol);
```

#### Configuration
```c
int sqlite3_exec(sqlite3 *db, const char *sql, 
                 int (*callback)(void*,int,char**,char**),
                 void *arg, char **errmsg);
const char *sqlite3_errmsg(sqlite3 *db);
int sqlite3_errcode(sqlite3 *db);
```

---

## Zig FFI Basics

### Linking C Libraries

Zig can link C libraries in two ways:

#### 1. System Library Linking
```zig
// In build.zig
const exe = b.addExecutable(.{
    .name = "nmetadata",
    .root_source_file = .{ .path = "src/main.zig" },
    .target = target,
    .optimize = optimize,
});

// Link SQLite
exe.linkSystemLibrary("sqlite3");
exe.linkLibC();
```

#### 2. Static Linking
```zig
// Build SQLite from source
const sqlite = b.addStaticLibrary(.{
    .name = "sqlite3",
    .target = target,
    .optimize = optimize,
});
sqlite.addCSourceFile(.{
    .file = .{ .path = "vendor/sqlite3.c" },
    .flags = &[_][]const u8{"-DSQLITE_THREADSAFE=1"},
});
sqlite.linkLibC();

exe.linkLibrary(sqlite);
```

### C Type Translation

Zig automatically translates C types:

```zig
// C type              -> Zig type
// int                 -> c_int
// long                -> c_long
// unsigned int        -> c_uint
// char*               -> [*c]u8 or [*:0]const u8
// void*               -> ?*anyopaque
// sqlite3*            -> ?*c.sqlite3
// sqlite3_stmt*       -> ?*c.sqlite3_stmt
```

---

## Integration Approaches

### Approach 1: @cImport (Recommended)

Use Zig's `@cImport` to automatically generate bindings:

```zig
const c = @cImport({
    @cInclude("sqlite3.h");
});

pub const sqlite3 = c.sqlite3;
pub const sqlite3_stmt = c.sqlite3_stmt;
pub const SQLITE_OK = c.SQLITE_OK;
pub const SQLITE_ROW = c.SQLITE_ROW;
pub const SQLITE_DONE = c.SQLITE_DONE;
```

**Advantages:**
- Automatic type translation
- No manual binding creation
- Compiler-verified API

**Disadvantages:**
- Requires C headers at compile time
- Less control over types

### Approach 2: Manual Extern Declarations

Manually declare C functions:

```zig
extern fn sqlite3_open(
    filename: [*:0]const u8,
    ppDb: *?*anyopaque,
) c_int;

extern fn sqlite3_close(db: ?*anyopaque) c_int;

extern fn sqlite3_prepare_v2(
    db: ?*anyopaque,
    sql: [*:0]const u8,
    nByte: c_int,
    ppStmt: *?*anyopaque,
    pzTail: ?*[*:0]const u8,
) c_int;
```

**Advantages:**
- No C headers required
- Full control over types
- Portable

**Disadvantages:**
- Manual maintenance
- Risk of API mismatch

### Approach 3: Hybrid

Use `@cImport` for complex types, manual declarations for simple functions:

```zig
const c = @cImport({
    @cInclude("sqlite3.h");
    @cDefine("SQLITE_THREADSAFE", "1");
});

// Use c.sqlite3 types, but wrap in Zig-friendly API
```

---

## Step-by-Step Implementation

### Step 1: Update build.zig

```zig
const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "nmetadata",
        .root_source_file = .{ .path = "zig/main.zig" },
        .target = target,
        .optimize = optimize,
    });

    // Link SQLite
    exe.linkSystemLibrary("sqlite3");
    exe.linkLibC();

    // Add include path if needed
    // exe.addIncludePath(.{ .path = "/usr/include" });

    b.installArtifact(exe);
}
```

### Step 2: Create FFI Bindings

Create `zig/db/drivers/sqlite/ffi.zig`:

```zig
const std = @import("std");

// Import C headers
pub const c = @cImport({
    @cInclude("sqlite3.h");
});

// Re-export types
pub const sqlite3 = c.sqlite3;
pub const sqlite3_stmt = c.sqlite3_stmt;

// Result codes
pub const SQLITE_OK = c.SQLITE_OK;
pub const SQLITE_ERROR = c.SQLITE_ERROR;
pub const SQLITE_ROW = c.SQLITE_ROW;
pub const SQLITE_DONE = c.SQLITE_DONE;

// Column types
pub const SQLITE_INTEGER = c.SQLITE_INTEGER;
pub const SQLITE_FLOAT = c.SQLITE_FLOAT;
pub const SQLITE_TEXT = c.SQLITE_TEXT;
pub const SQLITE_BLOB = c.SQLITE_BLOB;
pub const SQLITE_NULL = c.SQLITE_NULL;

// Open flags
pub const SQLITE_OPEN_READONLY = c.SQLITE_OPEN_READONLY;
pub const SQLITE_OPEN_READWRITE = c.SQLITE_OPEN_READWRITE;
pub const SQLITE_OPEN_CREATE = c.SQLITE_OPEN_CREATE;
pub const SQLITE_OPEN_MEMORY = c.SQLITE_OPEN_MEMORY;

// Wrapper functions for error handling
pub fn open(filename: [*:0]const u8, db: *?*c.sqlite3) !void {
    const rc = c.sqlite3_open(filename, db);
    if (rc != SQLITE_OK) {
        return error.SqliteOpenFailed;
    }
}

pub fn close(db: ?*c.sqlite3) !void {
    const rc = c.sqlite3_close(db);
    if (rc != SQLITE_OK) {
        return error.SqliteCloseFailed;
    }
}

pub fn prepare(
    db: ?*c.sqlite3,
    sql: [*:0]const u8,
    stmt: *?*c.sqlite3_stmt,
) !void {
    const rc = c.sqlite3_prepare_v2(db, sql, -1, stmt, null);
    if (rc != SQLITE_OK) {
        return error.SqlitePrepareFailed;
    }
}

pub fn step(stmt: ?*c.sqlite3_stmt) !c_int {
    return c.sqlite3_step(stmt);
}

pub fn finalize(stmt: ?*c.sqlite3_stmt) !void {
    const rc = c.sqlite3_finalize(stmt);
    if (rc != SQLITE_OK) {
        return error.SqliteFinalizeFailed;
    }
}

// Parameter binding
pub fn bindInt(stmt: ?*c.sqlite3_stmt, idx: c_int, value: i32) !void {
    const rc = c.sqlite3_bind_int(stmt, idx, value);
    if (rc != SQLITE_OK) {
        return error.SqliteBindFailed;
    }
}

pub fn bindInt64(stmt: ?*c.sqlite3_stmt, idx: c_int, value: i64) !void {
    const rc = c.sqlite3_bind_int64(stmt, idx, value);
    if (rc != SQLITE_OK) {
        return error.SqliteBindFailed;
    }
}

pub fn bindDouble(stmt: ?*c.sqlite3_stmt, idx: c_int, value: f64) !void {
    const rc = c.sqlite3_bind_double(stmt, idx, value);
    if (rc != SQLITE_OK) {
        return error.SqliteBindFailed;
    }
}

pub fn bindText(
    stmt: ?*c.sqlite3_stmt,
    idx: c_int,
    value: []const u8,
) !void {
    const rc = c.sqlite3_bind_text(
        stmt,
        idx,
        value.ptr,
        @intCast(value.len),
        c.SQLITE_TRANSIENT,
    );
    if (rc != SQLITE_OK) {
        return error.SqliteBindFailed;
    }
}

pub fn bindNull(stmt: ?*c.sqlite3_stmt, idx: c_int) !void {
    const rc = c.sqlite3_bind_null(stmt, idx);
    if (rc != SQLITE_OK) {
        return error.SqliteBindFailed;
    }
}

// Column retrieval
pub fn columnType(stmt: ?*c.sqlite3_stmt, col: c_int) c_int {
    return c.sqlite3_column_type(stmt, col);
}

pub fn columnInt(stmt: ?*c.sqlite3_stmt, col: c_int) i32 {
    return c.sqlite3_column_int(stmt, col);
}

pub fn columnInt64(stmt: ?*c.sqlite3_stmt, col: c_int) i64 {
    return c.sqlite3_column_int64(stmt, col);
}

pub fn columnDouble(stmt: ?*c.sqlite3_stmt, col: c_int) f64 {
    return c.sqlite3_column_double(stmt, col);
}

pub fn columnText(
    stmt: ?*c.sqlite3_stmt,
    col: c_int,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const text = c.sqlite3_column_text(stmt, col);
    if (text == null) return "";
    
    const len = c.sqlite3_column_bytes(stmt, col);
    const result = try allocator.alloc(u8, @intCast(len));
    @memcpy(result, text[0..@intCast(len)]);
    return result;
}

pub fn columnBlob(
    stmt: ?*c.sqlite3_stmt,
    col: c_int,
    allocator: std.mem.Allocator,
) ![]const u8 {
    const blob = c.sqlite3_column_blob(stmt, col);
    if (blob == null) return &[_]u8{};
    
    const len = c.sqlite3_column_bytes(stmt, col);
    const result = try allocator.alloc(u8, @intCast(len));
    const bytes: [*]const u8 = @ptrCast(blob);
    @memcpy(result, bytes[0..@intCast(len)]);
    return result;
}

// Error handling
pub fn errmsg(db: ?*c.sqlite3) []const u8 {
    const msg = c.sqlite3_errmsg(db);
    return std.mem.span(msg);
}

pub fn errcode(db: ?*c.sqlite3) c_int {
    return c.sqlite3_errcode(db);
}
```

### Step 3: Update Connection Module

Modify `connection.zig` to use real FFI:

```zig
const ffi = @import("ffi.zig");

pub const SqliteConnection = struct {
    allocator: std.mem.Allocator,
    config: ConnectionConfig,
    handle: ?*ffi.c.sqlite3,
    connected: bool,
    
    pub fn connect(self: *SqliteConnection) !void {
        const path_z = try self.allocator.dupeZ(u8, self.config.path);
        defer self.allocator.free(path_z);
        
        try ffi.open(path_z.ptr, &self.handle);
        self.connected = true;
        
        // Configure pragmas
        try self.configurePragmas();
    }
    
    pub fn disconnect(self: *SqliteConnection) void {
        if (self.handle) |handle| {
            ffi.close(handle) catch {};
            self.handle = null;
        }
        self.connected = false;
    }
    
    fn configurePragmas(self: *SqliteConnection) !void {
        // Set journal mode
        const journal_sql = try std.fmt.allocPrintZ(
            self.allocator,
            "PRAGMA journal_mode = {s}",
            .{@tagName(self.config.journal_mode)},
        );
        defer self.allocator.free(journal_sql);
        
        _ = ffi.c.sqlite3_exec(self.handle, journal_sql, null, null, null);
        
        // Set synchronous mode
        const sync_sql = try std.fmt.allocPrintZ(
            self.allocator,
            "PRAGMA synchronous = {s}",
            .{@tagName(self.config.synchronous)},
        );
        defer self.allocator.free(sync_sql);
        
        _ = ffi.c.sqlite3_exec(self.handle, sync_sql, null, null, null);
        
        // Set cache size
        const cache_sql = try std.fmt.allocPrintZ(
            self.allocator,
            "PRAGMA cache_size = {d}",
            .{self.config.cache_size},
        );
        defer self.allocator.free(cache_sql);
        
        _ = ffi.c.sqlite3_exec(self.handle, cache_sql, null, null, null);
        
        // Enable/disable foreign keys
        const fk_sql = if (self.config.foreign_keys)
            "PRAGMA foreign_keys = ON"
        else
            "PRAGMA foreign_keys = OFF";
            
        _ = ffi.c.sqlite3_exec(
            self.handle,
            fk_sql,
            null,
            null,
            null,
        );
    }
};
```

### Step 4: Update Query Module

Modify `query.zig` to use real prepared statements:

```zig
const ffi = @import("ffi.zig");

pub const QueryExecutor = struct {
    allocator: std.mem.Allocator,
    connection: *SqliteConnection,
    
    pub fn executeQuery(
        self: *QueryExecutor,
        sql: []const u8,
        params: []const Value,
    ) !QueryResult {
        const sql_z = try self.allocator.dupeZ(u8, sql);
        defer self.allocator.free(sql_z);
        
        var stmt: ?*ffi.c.sqlite3_stmt = null;
        try ffi.prepare(self.connection.handle, sql_z.ptr, &stmt);
        defer _ = ffi.finalize(stmt) catch {};
        
        // Bind parameters
        for (params, 1..) |param, i| {
            try self.bindParameter(stmt, @intCast(i), param);
        }
        
        // Execute and collect results
        var result = QueryResult.init(self.allocator);
        
        while (true) {
            const rc = try ffi.step(stmt);
            if (rc == ffi.SQLITE_DONE) break;
            if (rc != ffi.SQLITE_ROW) return error.SqliteStepFailed;
            
            const row = try self.fetchRow(stmt);
            try result.rows.append(row);
        }
        
        return result;
    }
    
    fn bindParameter(
        self: *QueryExecutor,
        stmt: ?*ffi.c.sqlite3_stmt,
        idx: c_int,
        value: Value,
    ) !void {
        _ = self;
        switch (value) {
            .null_value => try ffi.bindNull(stmt, idx),
            .integer => |v| try ffi.bindInt64(stmt, idx, v),
            .real => |v| try ffi.bindDouble(stmt, idx, v),
            .text => |v| try ffi.bindText(stmt, idx, v),
            .blob => |v| try ffi.bindText(stmt, idx, v),
        }
    }
    
    fn fetchRow(
        self: *QueryExecutor,
        stmt: ?*ffi.c.sqlite3_stmt,
    ) !Row {
        var row = Row.init(self.allocator);
        
        const col_count = ffi.c.sqlite3_column_count(stmt);
        var i: c_int = 0;
        while (i < col_count) : (i += 1) {
            const col_type = ffi.columnType(stmt, i);
            const value = switch (col_type) {
                ffi.SQLITE_INTEGER => Value{
                    .integer = ffi.columnInt64(stmt, i),
                },
                ffi.SQLITE_FLOAT => Value{
                    .real = ffi.columnDouble(stmt, i),
                },
                ffi.SQLITE_TEXT => Value{
                    .text = try ffi.columnText(stmt, i, self.allocator),
                },
                ffi.SQLITE_BLOB => Value{
                    .blob = try ffi.columnBlob(stmt, i, self.allocator),
                },
                ffi.SQLITE_NULL => Value.null_value,
                else => Value.null_value,
            };
            
            try row.values.append(value);
        }
        
        return row;
    }
};
```

### Step 5: Test the Integration

Create a test to verify real SQLite works:

```zig
test "Real SQLite integration" {
    const allocator = std.testing.allocator;
    
    const config = ConnectionConfig{
        .path = ":memory:",
        .mode = .memory,
        .journal_mode = .wal,
        .synchronous = .normal,
        .cache_size = 2000,
        .foreign_keys = true,
        .timeout_ms = 5000,
    };
    
    var conn = try SqliteConnection.init(allocator, config);
    defer conn.deinit();
    
    try conn.connect();
    defer conn.disconnect();
    
    var executor = QueryExecutor.init(allocator, &conn);
    defer executor.deinit();
    
    // Create table
    _ = try executor.executeQuery(
        "CREATE TABLE test (id INTEGER PRIMARY KEY, name TEXT)",
        &[_]Value{},
    );
    
    // Insert data
    const params = [_]Value{
        Value{ .text = "Alice" },
    };
    _ = try executor.executeQuery(
        "INSERT INTO test (name) VALUES (?1)",
        &params,
    );
    
    // Query data
    const result = try executor.executeQuery(
        "SELECT * FROM test",
        &[_]Value{},
    );
    defer result.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), result.rows.items.len);
}
```

---

## Testing with Real SQLite

### Build and Run Tests

```bash
# Build with SQLite linking
zig build

# Run unit tests
zig build test

# Run integration tests (requires SQLite)
zig build test-integration

# Run benchmarks
zig build bench
```

### Verify SQLite Version

```zig
const version = ffi.c.sqlite3_libversion();
std.debug.print("SQLite version: {s}\n", .{std.mem.span(version)});
```

---

## Performance Considerations

### 1. Prepared Statement Caching

Cache frequently-used prepared statements:

```zig
pub const StatementCache = struct {
    cache: std.StringHashMap(*ffi.c.sqlite3_stmt),
    
    pub fn get(self: *StatementCache, sql: []const u8) !*ffi.c.sqlite3_stmt {
        if (self.cache.get(sql)) |stmt| {
            _ = ffi.c.sqlite3_reset(stmt);
            return stmt;
        }
        
        // Prepare and cache
        var stmt: ?*ffi.c.sqlite3_stmt = null;
        try ffi.prepare(self.db, sql, &stmt);
        try self.cache.put(sql, stmt.?);
        return stmt.?;
    }
};
```

### 2. Batch Operations

Use transactions for bulk inserts:

```zig
try executor.executeQuery("BEGIN", &[_]Value{});
for (items) |item| {
    try executor.executeQuery(insert_sql, &[_]Value{item});
}
try executor.executeQuery("COMMIT", &[_]Value{});
```

### 3. Memory Management

Use arena allocators for temporary data:

```zig
var arena = std.heap.ArenaAllocator.init(allocator);
defer arena.deinit();

const result = try executor.executeQuery(sql, params);
// arena cleans up all allocations
```

---

## Troubleshooting

### Common Issues

**1. Library Not Found**
```
error: unable to find library 'sqlite3'
```
**Solution:** Install SQLite development package or add library path:
```zig
exe.addLibraryPath(.{ .path = "/usr/local/lib" });
```

**2. Header Not Found**
```
error: unable to find header 'sqlite3.h'
```
**Solution:** Add include path:
```zig
exe.addIncludePath(.{ .path = "/usr/local/include" });
```

**3. Symbol Undefined**
```
error: undefined symbol: sqlite3_open
```
**Solution:** Ensure `linkLibC()` is called:
```zig
exe.linkLibC();
```

**4. ABI Compatibility**
```
error: incompatible types
```
**Solution:** Use correct C type conversions:
```zig
const len: c_int = @intCast(value.len);
```

---

## References

### Official Documentation

- [SQLite C API Documentation](https://www.sqlite.org/c3ref/intro.html)
- [Zig FFI Documentation](https://ziglang.org/documentation/master/#C)
- [Zig Build System](https://ziglang.org/learn/build-system/)

### SQLite Resources

- [SQLite Tutorial](https://www.sqlitetutorial.net/)
- [SQLite Performance Tuning](https://www.sqlite.org/speed.html)
- [SQLite Best Practices](https://www.sqlite.org/bestpractice.html)

### Example Projects

- [zig-sqlite](https://github.com/vrischmann/zig-sqlite)
- [zqlite](https://github.com/cryptocode/zqlite)

---

## Conclusion

This guide provides everything needed to integrate real SQLite C library bindings with the nMetaData SQLite driver. The FFI layer abstracts the C API while maintaining type safety and idiomatic Zig code.

**Next Steps:**
1. Implement `ffi.zig` module
2. Update connection and query modules
3. Run integration tests
4. Benchmark performance
5. Document any platform-specific issues

---

**Last Updated:** January 20, 2026  
**Version:** 1.0  
**Status:** Complete
