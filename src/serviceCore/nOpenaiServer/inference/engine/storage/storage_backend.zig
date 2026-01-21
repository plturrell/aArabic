// Storage Backend Interface for LLM Inference Engine
// Provides polymorphic storage abstraction using vtable pattern
//
// Features:
// - Virtual table pattern for runtime polymorphism
// - Comprehensive error types for storage operations
// - Helper methods for common operations
// - Structured logging integration

const std = @import("std");
const builtin = @import("builtin");

// ========== Error Types ==========

/// Comprehensive error types for storage operations
pub const StorageError = error{
    /// Path not found in storage
    PathNotFound,
    /// Permission denied for operation
    PermissionDenied,
    /// Storage quota exceeded
    QuotaExceeded,
    /// Invalid path format
    InvalidPath,
    /// Storage backend unavailable
    BackendUnavailable,
    /// Read operation failed
    ReadFailed,
    /// Write operation failed
    WriteFailed,
    /// Delete operation failed
    DeleteFailed,
    /// List operation failed
    ListFailed,
    /// Data corruption detected
    DataCorruption,
    /// Connection to storage lost
    ConnectionLost,
    /// Operation timeout
    Timeout,
    /// Path already exists
    AlreadyExists,
    /// Invalid data format
    InvalidData,
    /// Out of memory
    OutOfMemory,
    /// Directory not found
    DirectoryNotFound,
    /// Disk full / no space left
    DiskFull,
    /// Generic I/O error
    IoError,
};

/// Storage operation result status for monitoring
pub const OperationStatus = enum {
    success,
    not_found,
    permission_denied,
    backend_error,
    timeout,

    pub fn toString(self: OperationStatus) []const u8 {
        return switch (self) {
            .success => "success",
            .not_found => "not_found",
            .permission_denied => "permission_denied",
            .backend_error => "backend_error",
            .timeout => "timeout",
        };
    }
};

// ========== Storage Backend Interface ==========

/// Storage backend interface with vtable pattern for polymorphism
pub const StorageBackend = struct {
    vtable: *const VTable,
    ctx: *anyopaque,

    /// Virtual table defining storage operations
    pub const VTable = struct {
        read: *const fn (ctx: *anyopaque, path: []const u8, allocator: std.mem.Allocator) anyerror![]u8,
        write: *const fn (ctx: *anyopaque, path: []const u8, data: []const u8) anyerror!void,
        exists: *const fn (ctx: *anyopaque, path: []const u8) bool,
        list: *const fn (ctx: *anyopaque, prefix: []const u8, allocator: std.mem.Allocator) anyerror![][]const u8,
        delete: *const fn (ctx: *anyopaque, path: []const u8) anyerror!void,
    };

    // ========== Core Operations ==========

    /// Read data from storage path
    pub fn read(self: StorageBackend, path: []const u8, allocator: std.mem.Allocator) ![]u8 {
        logOperation(.debug, "read", path, null);
        const result = self.vtable.read(self.ctx, path, allocator);
        if (result) |data| {
            logOperation(.debug, "read_complete", path, data.len);
            return data;
        } else |err| {
            logError("read", path, err);
            return err;
        }
    }

    /// Write data to storage path
    pub fn write(self: StorageBackend, path: []const u8, data: []const u8) !void {
        logOperation(.debug, "write", path, data.len);
        self.vtable.write(self.ctx, path, data) catch |err| {
            logError("write", path, err);
            return err;
        };
        logOperation(.info, "write_complete", path, data.len);
    }

    /// Check if path exists in storage
    pub fn exists(self: StorageBackend, path: []const u8) bool {
        return self.vtable.exists(self.ctx, path);
    }

    /// List all paths matching prefix
    pub fn list(self: StorageBackend, prefix: []const u8, allocator: std.mem.Allocator) ![][]const u8 {
        logOperation(.debug, "list", prefix, null);
        const result = self.vtable.list(self.ctx, prefix, allocator);
        if (result) |items| {
            logOperation(.debug, "list_complete", prefix, items.len);
            return items;
        } else |err| {
            logError("list", prefix, err);
            return err;
        }
    }

    /// Delete path from storage
    pub fn delete(self: StorageBackend, path: []const u8) !void {
        logOperation(.debug, "delete", path, null);
        self.vtable.delete(self.ctx, path) catch |err| {
            logError("delete", path, err);
            return err;
        };
        logOperation(.info, "delete_complete", path, null);
    }

    // ========== Helper Methods ==========

    /// Read data if exists, otherwise return null
    pub fn readIfExists(self: StorageBackend, path: []const u8, allocator: std.mem.Allocator) !?[]u8 {
        if (!self.exists(path)) {
            return null;
        }
        return self.read(path, allocator);
    }

    /// Write data only if path doesn't exist, returns true if written
    pub fn writeIfNotExists(self: StorageBackend, path: []const u8, data: []const u8) !bool {
        if (self.exists(path)) {
            return false;
        }
        try self.write(path, data);
        return true;
    }

    /// Delete path if it exists, returns true if deleted
    pub fn deleteIfExists(self: StorageBackend, path: []const u8) !bool {
        if (!self.exists(path)) {
            return false;
        }
        try self.delete(path);
        return true;
    }

    /// Count items matching prefix
    pub fn count(self: StorageBackend, prefix: []const u8, allocator: std.mem.Allocator) !usize {
        const items = try self.list(prefix, allocator);
        defer freeStringList(allocator, items);
        return items.len;
    }

    /// Copy data from one path to another
    pub fn copy(self: StorageBackend, src: []const u8, dst: []const u8, allocator: std.mem.Allocator) !void {
        const data = try self.read(src, allocator);
        defer allocator.free(data);
        try self.write(dst, data);
    }

    /// Move data from one path to another
    pub fn move(self: StorageBackend, src: []const u8, dst: []const u8, allocator: std.mem.Allocator) !void {
        try self.copy(src, dst, allocator);
        try self.delete(src);
    }
};

// ========== Utility Functions ==========

/// Helper to free a list of strings returned by list()
pub fn freeStringList(allocator: std.mem.Allocator, items: [][]const u8) void {
    for (items) |item| {
        allocator.free(item);
    }
    allocator.free(items);
}

/// Log level for storage operations
const LogLevel = enum { debug, info, warn, err };

/// Log storage operation for debugging and monitoring
fn logOperation(level: LogLevel, operation: []const u8, path: []const u8, size: ?usize) void {
    if (builtin.mode == .Debug or level != .debug) {
        if (size) |s| {
            switch (level) {
                .debug => std.log.debug("[storage] {s}: {s} ({d} bytes)", .{ operation, path, s }),
                .info => std.log.info("[storage] {s}: {s} ({d} bytes)", .{ operation, path, s }),
                .warn => std.log.warn("[storage] {s}: {s} ({d} bytes)", .{ operation, path, s }),
                .err => std.log.err("[storage] {s}: {s} ({d} bytes)", .{ operation, path, s }),
            }
        } else {
            switch (level) {
                .debug => std.log.debug("[storage] {s}: {s}", .{ operation, path }),
                .info => std.log.info("[storage] {s}: {s}", .{ operation, path }),
                .warn => std.log.warn("[storage] {s}: {s}", .{ operation, path }),
                .err => std.log.err("[storage] {s}: {s}", .{ operation, path }),
            }
        }
    }
}

/// Log storage error
fn logError(operation: []const u8, path: []const u8, err: anyerror) void {
    std.log.err("[storage] {s} failed for {s}: {}", .{ operation, path, err });
}

// ========== Tests ==========

test "StorageError enum values" {
    const err: StorageError = StorageError.PathNotFound;
    try std.testing.expect(err == StorageError.PathNotFound);
}

test "OperationStatus toString" {
    try std.testing.expectEqualStrings("success", OperationStatus.success.toString());
    try std.testing.expectEqualStrings("not_found", OperationStatus.not_found.toString());
    try std.testing.expectEqualStrings("permission_denied", OperationStatus.permission_denied.toString());
    try std.testing.expectEqualStrings("backend_error", OperationStatus.backend_error.toString());
    try std.testing.expectEqualStrings("timeout", OperationStatus.timeout.toString());
}
