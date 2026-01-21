///! Local Filesystem Storage Backend
///! Implements StorageBackend interface for local filesystem operations

const std = @import("std");
const storage_backend = @import("storage_backend.zig");
const StorageBackend = storage_backend.StorageBackend;
const StorageError = storage_backend.StorageError;

/// Local filesystem storage implementation
pub const LocalStorage = struct {
    base_path: []const u8,
    allocator: std.mem.Allocator,

    const Self = @This();

    /// VTable for StorageBackend interface
    pub const vtable = StorageBackend.VTable{
        .read = read,
        .write = write,
        .exists = exists,
        .list = list,
        .delete = deleteFile,
    };

    /// Initialize local storage with a base path
    pub fn init(allocator: std.mem.Allocator, base_path: []const u8) !*Self {
        const self = try allocator.create(Self);
        errdefer allocator.destroy(self);

        self.* = .{
            .base_path = try allocator.dupe(u8, base_path),
            .allocator = allocator,
        };
        return self;
    }

    /// Get the StorageBackend interface
    pub fn backend(self: *Self) StorageBackend {
        return .{
            .vtable = &vtable,
            .ctx = @ptrCast(self),
        };
    }

    /// Clean up resources
    pub fn deinit(self: *Self) void {
        self.allocator.free(self.base_path);
        self.allocator.destroy(self);
    }

    /// Build full path from relative path
    fn buildFullPath(self: *const Self, allocator: std.mem.Allocator, path: []const u8) ![]u8 {
        if (self.base_path.len == 0) {
            return allocator.dupe(u8, path);
        }
        return std.fmt.allocPrint(allocator, "{s}/{s}", .{ self.base_path, path });
    }

    // === VTable Implementation Functions ===

    fn read(ctx: *anyopaque, path: []const u8, allocator: std.mem.Allocator) anyerror![]u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const full_path = try self.buildFullPath(allocator, path);
        defer allocator.free(full_path);

        const file = std.fs.cwd().openFile(full_path, .{}) catch |err| {
            return mapOpenError(err);
        };
        defer file.close();

        const stat = try file.stat();
        const size = stat.size;

        const buffer = try allocator.alloc(u8, size);
        errdefer allocator.free(buffer);

        const bytes_read = try file.readAll(buffer);
        if (bytes_read != size) {
            return StorageError.IoError;
        }

        return buffer;
    }

    fn write(ctx: *anyopaque, path: []const u8, data: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const full_path = try self.buildFullPath(self.allocator, path);
        defer self.allocator.free(full_path);

        // Ensure parent directories exist
        try ensureParentDirs(full_path);

        const file = std.fs.cwd().createFile(full_path, .{}) catch |err| {
            return mapCreateError(err);
        };
        defer file.close();

        try file.writeAll(data);
    }

    fn exists(ctx: *anyopaque, path: []const u8) bool {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const full_path = self.buildFullPath(self.allocator, path) catch return false;
        defer self.allocator.free(full_path);

        const stat = std.fs.cwd().statFile(full_path) catch return false;
        _ = stat;
        return true;
    }

    fn list(ctx: *anyopaque, prefix: []const u8, allocator: std.mem.Allocator) anyerror![][]const u8 {
        const self: *Self = @ptrCast(@alignCast(ctx));

        // Parse prefix to get directory and file prefix
        const dir_path = if (std.mem.lastIndexOf(u8, prefix, "/")) |idx|
            prefix[0..idx]
        else
            "";

        const file_prefix = if (std.mem.lastIndexOf(u8, prefix, "/")) |idx|
            prefix[idx + 1 ..]
        else
            prefix;

        const search_path = try self.buildFullPath(allocator, dir_path);
        defer allocator.free(search_path);

        var result: std.ArrayListUnmanaged([]const u8) = .empty;
        errdefer {
            for (result.items) |item| allocator.free(item);
            result.deinit(allocator);
        }

        const search_dir = if (search_path.len == 0) "." else search_path;
        var dir = std.fs.cwd().openDir(search_dir, .{ .iterate = true }) catch |err| {
            if (err == error.FileNotFound) return result.toOwnedSlice(allocator);
            return mapOpenError(err);
        };
        defer dir.close();

        var iter = dir.iterate();
        while (try iter.next()) |entry| {
            if (entry.kind != .file) continue;
            if (file_prefix.len == 0 or std.mem.startsWith(u8, entry.name, file_prefix)) {
                const full_name = if (dir_path.len > 0)
                    try std.fmt.allocPrint(allocator, "{s}/{s}", .{ dir_path, entry.name })
                else
                    try allocator.dupe(u8, entry.name);
                try result.append(allocator, full_name);
            }
        }

        return result.toOwnedSlice(allocator);
    }

    fn deleteFile(ctx: *anyopaque, path: []const u8) anyerror!void {
        const self: *Self = @ptrCast(@alignCast(ctx));
        const full_path = try self.buildFullPath(self.allocator, path);
        defer self.allocator.free(full_path);

        std.fs.cwd().deleteFile(full_path) catch |err| {
            return mapDeleteError(err);
        };
    }
};

// === Helper Functions ===

/// Ensure parent directories exist for a given path
fn ensureParentDirs(path: []const u8) !void {
    if (std.mem.lastIndexOf(u8, path, "/")) |idx| {
        const parent = path[0..idx];
        if (parent.len > 0) {
            std.fs.cwd().makePath(parent) catch |err| {
                if (err != error.PathAlreadyExists) {
                    return StorageError.IoError;
                }
            };
        }
    }
}

/// Map std.fs open errors to StorageError
fn mapOpenError(err: anytype) StorageError {
    return switch (err) {
        error.FileNotFound => StorageError.PathNotFound,
        error.AccessDenied => StorageError.PermissionDenied,
        else => StorageError.IoError,
    };
}

/// Map std.fs create errors to StorageError
fn mapCreateError(err: anytype) StorageError {
    return switch (err) {
        error.AccessDenied => StorageError.PermissionDenied,
        error.NoSpaceLeft => StorageError.DiskFull,
        else => StorageError.IoError,
    };
}

/// Map std.fs delete errors to StorageError
fn mapDeleteError(err: anytype) StorageError {
    return switch (err) {
        error.FileNotFound => StorageError.PathNotFound,
        error.AccessDenied => StorageError.PermissionDenied,
        else => StorageError.IoError,
    };
}

// === Tests ===

test "LocalStorage init and deinit" {
    const allocator = std.testing.allocator;
    const storage = try LocalStorage.init(allocator, "/tmp/test_storage");
    storage.deinit();
}

test "LocalStorage write and read" {
    const allocator = std.testing.allocator;
    const storage = try LocalStorage.init(allocator, "/tmp/test_local_storage");
    defer storage.deinit();

    const backend = storage.backend();
    const test_data = "Hello, World!";

    // Write
    try backend.write("test_file.txt", test_data);

    // Read
    const data = try backend.read("test_file.txt", allocator);
    defer allocator.free(data);

    try std.testing.expectEqualStrings(test_data, data);

    // Cleanup
    try backend.delete("test_file.txt");
}

test "LocalStorage exists" {
    const allocator = std.testing.allocator;
    const storage = try LocalStorage.init(allocator, "/tmp/test_local_storage");
    defer storage.deinit();

    const backend = storage.backend();

    // Should not exist initially
    try std.testing.expect(!backend.exists("nonexistent.txt"));

    // Write a file
    try backend.write("exists_test.txt", "data");

    // Should exist now
    try std.testing.expect(backend.exists("exists_test.txt"));

    // Cleanup
    try backend.delete("exists_test.txt");
}

test "LocalStorage list with prefix" {
    const allocator = std.testing.allocator;
    const storage = try LocalStorage.init(allocator, "/tmp/test_local_storage_list");
    defer storage.deinit();

    const backend = storage.backend();

    // Create some files
    try backend.write("prefix_a.txt", "a");
    try backend.write("prefix_b.txt", "b");
    try backend.write("other.txt", "c");

    // List with prefix
    const files = try backend.list("prefix_", allocator);
    defer storage_backend.freeStringList(allocator, files);

    try std.testing.expectEqual(@as(usize, 2), files.len);

    // Cleanup
    try backend.delete("prefix_a.txt");
    try backend.delete("prefix_b.txt");
    try backend.delete("other.txt");
}

test "LocalStorage nested directories" {
    const allocator = std.testing.allocator;
    const storage = try LocalStorage.init(allocator, "/tmp/test_local_storage_nested");
    defer storage.deinit();

    const backend = storage.backend();

    // Write to nested path (should create directories)
    try backend.write("deep/nested/path/file.txt", "nested content");

    // Read back
    const data = try backend.read("deep/nested/path/file.txt", allocator);
    defer allocator.free(data);

    try std.testing.expectEqualStrings("nested content", data);

    // Cleanup
    try backend.delete("deep/nested/path/file.txt");
}
