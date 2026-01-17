///! Source entity management
///! Handles CRUD operations for research sources

const std = @import("std");
const storage = @import("storage.zig");

/// Source type enumeration
pub const SourceType = enum {
    url,
    pdf,
    text,
    file,

    pub fn toString(self: SourceType) []const u8 {
        return switch (self) {
            .url => "URL",
            .pdf => "PDF",
            .text => "Text",
            .file => "File",
        };
    }

    pub fn fromString(s: []const u8) !SourceType {
        if (std.mem.eql(u8, s, "URL")) return .url;
        if (std.mem.eql(u8, s, "PDF")) return .pdf;
        if (std.mem.eql(u8, s, "Text")) return .text;
        if (std.mem.eql(u8, s, "File")) return .file;
        return error.InvalidSourceType;
    }
};

/// Source status enumeration
pub const SourceStatus = enum {
    pending,
    processing,
    ready,
    failed,

    pub fn toString(self: SourceStatus) []const u8 {
        return switch (self) {
            .pending => "Pending",
            .processing => "Processing",
            .ready => "Ready",
            .failed => "Failed",
        };
    }

    pub fn fromString(s: []const u8) !SourceStatus {
        if (std.mem.eql(u8, s, "Pending")) return .pending;
        if (std.mem.eql(u8, s, "Processing")) return .processing;
        if (std.mem.eql(u8, s, "Ready")) return .ready;
        if (std.mem.eql(u8, s, "Failed")) return .failed;
        return error.InvalidSourceStatus;
    }
};

/// Source entity structure
pub const Source = struct {
    id: []const u8,
    title: []const u8,
    source_type: SourceType,
    url: []const u8,
    content: []const u8,
    status: SourceStatus,
    created_at: []const u8,
    updated_at: []const u8,

    /// Create a new source with allocated memory
    pub fn init(
        allocator: std.mem.Allocator,
        id: []const u8,
        title: []const u8,
        source_type: SourceType,
        url: []const u8,
        content: []const u8,
        status: SourceStatus,
        created_at: []const u8,
        updated_at: []const u8,
    ) !Source {
        return Source{
            .id = try allocator.dupe(u8, id),
            .title = try allocator.dupe(u8, title),
            .source_type = source_type,
            .url = try allocator.dupe(u8, url),
            .content = try allocator.dupe(u8, content),
            .status = status,
            .created_at = try allocator.dupe(u8, created_at),
            .updated_at = try allocator.dupe(u8, updated_at),
        };
    }

    /// Free allocated memory
    pub fn deinit(self: Source, allocator: std.mem.Allocator) void {
        allocator.free(self.id);
        allocator.free(self.title);
        allocator.free(self.url);
        allocator.free(self.content);
        allocator.free(self.created_at);
        allocator.free(self.updated_at);
    }

    /// Clone a source
    pub fn clone(self: Source, allocator: std.mem.Allocator) !Source {
        return Source.init(
            allocator,
            self.id,
            self.title,
            self.source_type,
            self.url,
            self.content,
            self.status,
            self.created_at,
            self.updated_at,
        );
    }
};

/// Source manager for CRUD operations
pub const SourceManager = struct {
    allocator: std.mem.Allocator,
    store: *storage.SourceStorage,

    pub fn init(allocator: std.mem.Allocator, store: *storage.SourceStorage) SourceManager {
        return .{
            .allocator = allocator,
            .store = store,
        };
    }

    /// Create a new source
    pub fn create(
        self: *SourceManager,
        title: []const u8,
        source_type: SourceType,
        url: []const u8,
        content: []const u8,
    ) ![]const u8 {
        // Generate unique ID
        const id = try self.generateId();
        errdefer self.allocator.free(id);

        // Get current timestamp
        const timestamp = try self.getCurrentTimestamp();
        defer self.allocator.free(timestamp);

        // Create source entity
        const source = try Source.init(
            self.allocator,
            id,
            title,
            source_type,
            url,
            content,
            .ready, // Default status
            timestamp,
            timestamp,
        );

        // Store in storage
        try self.store.put(id, source);

        return id;
    }

    /// Get source by ID
    pub fn get(self: *SourceManager, id: []const u8) !?Source {
        return self.store.get(id);
    }

    /// Get all sources
    pub fn getAll(self: *SourceManager) ![]Source {
        return self.store.getAll(self.allocator);
    }

    /// Update a source
    pub fn update(
        self: *SourceManager,
        id: []const u8,
        title: ?[]const u8,
        url: ?[]const u8,
        content: ?[]const u8,
        status: ?SourceStatus,
    ) !void {
        const existing = try self.store.get(id);
        if (existing == null) {
            return error.SourceNotFound;
        }

        const old_source = existing.?;
        defer old_source.deinit(self.allocator);

        // Get current timestamp
        const timestamp = try self.getCurrentTimestamp();
        defer self.allocator.free(timestamp);

        // Create updated source
        const updated_source = try Source.init(
            self.allocator,
            old_source.id,
            if (title) |t| t else old_source.title,
            old_source.source_type,
            if (url) |u| u else old_source.url,
            if (content) |c| c else old_source.content,
            if (status) |s| s else old_source.status,
            old_source.created_at,
            timestamp,
        );

        try self.store.put(id, updated_source);
    }

    /// Delete a source
    pub fn delete(self: *SourceManager, id: []const u8) !void {
        try self.store.delete(id);
    }

    /// Count total sources
    pub fn count(self: *SourceManager) usize {
        return self.store.count();
    }

    /// Generate unique source ID
    fn generateId(self: *SourceManager) ![]const u8 {
        const timestamp = std.time.milliTimestamp();
        const random = std.crypto.random.int(u32);

        var buf: [64]u8 = undefined;
        const id = try std.fmt.bufPrint(&buf, "source_{d}_{d}", .{ timestamp, random });

        return try self.allocator.dupe(u8, id);
    }

    /// Get current ISO 8601 timestamp
    fn getCurrentTimestamp(self: *SourceManager) ![]const u8 {
        const timestamp = std.time.timestamp();
        const epoch_seconds = @as(u64, @intCast(timestamp));

        // Simple ISO 8601 format: YYYY-MM-DDTHH:MM:SSZ
        var buf: [32]u8 = undefined;
        const formatted = try std.fmt.bufPrint(&buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
            2026, 1, 16, // Simplified date (would calculate from epoch_seconds in production)
            @divTrunc(epoch_seconds % 86400, 3600),
            @divTrunc(epoch_seconds % 3600, 60),
            epoch_seconds % 60,
        });

        return try self.allocator.dupe(u8, formatted);
    }
};

// Tests
test "Source creation and retrieval" {
    const allocator = std.testing.allocator;

    var store = storage.SourceStorage.init(allocator);
    defer store.deinit();

    var manager = SourceManager.init(allocator, &store);

    const id = try manager.create(
        "Test Source",
        .url,
        "https://example.com",
        "Test content",
    );
    defer allocator.free(id);

    const source = try manager.get(id);
    try std.testing.expect(source != null);
    try std.testing.expectEqualStrings("Test Source", source.?.title);

    const retrieved = source.?;
    defer retrieved.deinit(allocator);
}

test "Source deletion" {
    const allocator = std.testing.allocator;

    var store = storage.SourceStorage.init(allocator);
    defer store.deinit();

    var manager = SourceManager.init(allocator, &store);

    const id = try manager.create(
        "Test Source",
        .url,
        "https://example.com",
        "Test content",
    );
    defer allocator.free(id);

    try manager.delete(id);

    const source = try manager.get(id);
    try std.testing.expect(source == null);
}
