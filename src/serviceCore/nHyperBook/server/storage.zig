///! In-memory storage for sources
///! Thread-safe storage using HashMap

const std = @import("std");
const sources = @import("sources.zig");

/// In-memory storage for Source entities
pub const SourceStorage = struct {
    allocator: std.mem.Allocator,
    map: std.StringHashMap(sources.Source),
    mutex: std.Thread.Mutex,

    pub fn init(allocator: std.mem.Allocator) SourceStorage {
        return .{
            .allocator = allocator,
            .map = std.StringHashMap(sources.Source).init(allocator),
            .mutex = .{},
        };
    }

    pub fn deinit(self: *SourceStorage) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Free all stored sources
        var it = self.map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }

        self.map.deinit();
    }

    /// Store a source (takes ownership)
    pub fn put(self: *SourceStorage, id: []const u8, source: sources.Source) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        // Check if source already exists
        if (self.map.get(id)) |existing| {
            // Free the old source
            existing.deinit(self.allocator);
        }

        // Store with duplicated key
        const key = try self.allocator.dupe(u8, id);
        errdefer self.allocator.free(key);

        try self.map.put(key, source);
    }

    /// Get a source by ID (returns a clone)
    pub fn get(self: *SourceStorage, id: []const u8) !?sources.Source {
        self.mutex.lock();
        defer self.mutex.unlock();

        const source = self.map.get(id);
        if (source) |s| {
            return try s.clone(self.allocator);
        }
        return null;
    }

    /// Get all sources (returns clones)
    pub fn getAll(self: *SourceStorage, allocator: std.mem.Allocator) ![]sources.Source {
        self.mutex.lock();
        defer self.mutex.unlock();

        const count = self.map.count();
        if (count == 0) {
            return &[_]sources.Source{};
        }

        var result = try allocator.alloc(sources.Source, count);
        errdefer allocator.free(result);

        var i: usize = 0;
        var it = self.map.valueIterator();
        while (it.next()) |source| : (i += 1) {
            result[i] = try source.clone(allocator);
        }

        return result;
    }

    /// Delete a source by ID
    pub fn delete(self: *SourceStorage, id: []const u8) !void {
        self.mutex.lock();
        defer self.mutex.unlock();

        if (self.map.fetchRemove(id)) |kv| {
            kv.value.deinit(self.allocator);
            self.allocator.free(kv.key);
        } else {
            return error.SourceNotFound;
        }
    }

    /// Check if a source exists
    pub fn exists(self: *SourceStorage, id: []const u8) bool {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.map.contains(id);
    }

    /// Get the count of stored sources
    pub fn count(self: *SourceStorage) usize {
        self.mutex.lock();
        defer self.mutex.unlock();

        return self.map.count();
    }

    /// Clear all sources
    pub fn clear(self: *SourceStorage) void {
        self.mutex.lock();
        defer self.mutex.unlock();

        var it = self.map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }

        self.map.clearRetainingCapacity();
    }
};

// Tests
test "Storage put and get" {
    const allocator = std.testing.allocator;

    var storage = SourceStorage.init(allocator);
    defer storage.deinit();

    const source = try sources.Source.init(
        allocator,
        "test_id",
        "Test Title",
        .url,
        "https://test.com",
        "Test content",
        .ready,
        "2026-01-16T00:00:00Z",
        "2026-01-16T00:00:00Z",
    );

    try storage.put("test_id", source);

    const retrieved = try storage.get("test_id");
    try std.testing.expect(retrieved != null);

    const s = retrieved.?;
    defer s.deinit(allocator);

    try std.testing.expectEqualStrings("Test Title", s.title);
}

test "Storage delete" {
    const allocator = std.testing.allocator;

    var storage = SourceStorage.init(allocator);
    defer storage.deinit();

    const source = try sources.Source.init(
        allocator,
        "test_id",
        "Test Title",
        .url,
        "https://test.com",
        "Test content",
        .ready,
        "2026-01-16T00:00:00Z",
        "2026-01-16T00:00:00Z",
    );

    try storage.put("test_id", source);
    try std.testing.expect(storage.exists("test_id"));

    try storage.delete("test_id");
    try std.testing.expect(!storage.exists("test_id"));
}

test "Storage getAll" {
    const allocator = std.testing.allocator;

    var storage = SourceStorage.init(allocator);
    defer storage.deinit();

    // Add multiple sources
    for (0..3) |i| {
        var buf: [32]u8 = undefined;
        const id = try std.fmt.bufPrint(&buf, "test_{d}", .{i});

        const source = try sources.Source.init(
            allocator,
            id,
            "Test Title",
            .url,
            "https://test.com",
            "Test content",
            .ready,
            "2026-01-16T00:00:00Z",
            "2026-01-16T00:00:00Z",
        );

        try storage.put(id, source);
    }

    const all = try storage.getAll(allocator);
    defer {
        for (all) |s| {
            s.deinit(allocator);
        }
        allocator.free(all);
    }

    try std.testing.expectEqual(@as(usize, 3), all.len);
}
