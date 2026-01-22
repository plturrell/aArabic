// Day 28: Langflow Component Parity - File Utilities
// File reading, writing, and processing components

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

/// File Reader Node - Read file contents
pub const FileReaderNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    file_path: []const u8,
    encoding: []const u8, // "utf-8", "ascii", "binary"

    pub fn init(allocator: Allocator, node_id: []const u8, file_path: []const u8, encoding: []const u8) !*FileReaderNode {
        const node = try allocator.create(FileReaderNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .file_path = try allocator.dupe(u8, file_path),
            .encoding = try allocator.dupe(u8, encoding),
        };
        return node;
    }

    pub fn deinit(self: *FileReaderNode) void {
        self.allocator.free(self.node_id);
        self.allocator.free(self.file_path);
        self.allocator.free(self.encoding);
        self.allocator.destroy(self);
    }

    pub fn execute(self: *FileReaderNode) ![]const u8 {
        const file = try std.fs.cwd().openFile(self.file_path, .{});
        defer file.close();

        const stat = try file.stat();
        const content = try file.readToEndAlloc(self.allocator, stat.size);
        return content;
    }

    pub fn readLines(self: *FileReaderNode) !ArrayList([]const u8) {
        const content = try self.execute();
        defer self.allocator.free(content);

        var lines = ArrayList([]const u8){};
        errdefer {
            for (lines.items) |line| {
                self.allocator.free(line);
            }
            lines.deinit(self.allocator);
        }

        var iter = std.mem.splitSequence(u8, content, "\n");
        while (iter.next()) |line| {
            try lines.append(self.allocator, try self.allocator.dupe(u8, line));
        }

        return lines;
    }
};

/// File Writer Node - Write content to file
pub const FileWriterNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    file_path: []const u8,
    append: bool,
    create_dirs: bool,

    pub fn init(allocator: Allocator, node_id: []const u8, file_path: []const u8, append: bool, create_dirs: bool) !*FileWriterNode {
        const node = try allocator.create(FileWriterNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .file_path = try allocator.dupe(u8, file_path),
            .append = append,
            .create_dirs = create_dirs,
        };
        return node;
    }

    pub fn deinit(self: *FileWriterNode) void {
        self.allocator.free(self.node_id);
        self.allocator.free(self.file_path);
        self.allocator.destroy(self);
    }

    pub fn execute(self: *FileWriterNode, content: []const u8) !void {
        if (self.create_dirs) {
            if (std.fs.path.dirname(self.file_path)) |dir_path| {
                try std.fs.cwd().makePath(dir_path);
            }
        }

        const file = if (self.append)
            try std.fs.cwd().createFile(self.file_path, .{ .truncate = false })
        else
            try std.fs.cwd().createFile(self.file_path, .{ .truncate = true });
        defer file.close();

        if (self.append) {
            try file.seekFromEnd(0);
        }

        try file.writeAll(content);
    }

    pub fn writeLines(self: *FileWriterNode, lines: []const []const u8) !void {
        if (self.create_dirs) {
            if (std.fs.path.dirname(self.file_path)) |dir_path| {
                try std.fs.cwd().makePath(dir_path);
            }
        }

        const file = if (self.append)
            try std.fs.cwd().createFile(self.file_path, .{ .truncate = false })
        else
            try std.fs.cwd().createFile(self.file_path, .{ .truncate = true });
        defer file.close();

        if (self.append) {
            try file.seekFromEnd(0);
        }

        for (lines) |line| {
            try file.writeAll(line);
            try file.writeAll("\n");
        }
    }
};

/// CSV Parser Node - Parse CSV files
pub const CSVParserNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    delimiter: u8,
    has_header: bool,

    pub fn init(allocator: Allocator, node_id: []const u8, delimiter: u8, has_header: bool) !*CSVParserNode {
        const node = try allocator.create(CSVParserNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .delimiter = delimiter,
            .has_header = has_header,
        };
        return node;
    }

    pub fn deinit(self: *CSVParserNode) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn execute(self: *CSVParserNode, csv_content: []const u8) !ArrayList(ArrayList([]const u8)) {
        var rows = ArrayList(ArrayList([]const u8)){};
        errdefer {
            for (rows.items) |*row| {
                for (row.items) |cell| {
                    self.allocator.free(cell);
                }
                row.deinit(self.allocator);
            }
            rows.deinit(self.allocator);
        }

        var line_iter = std.mem.splitSequence(u8, csv_content, "\n");
        var skip_first = self.has_header;

        while (line_iter.next()) |line| {
            if (skip_first) {
                skip_first = false;
                continue;
            }

            if (line.len == 0) continue;

            var row = ArrayList([]const u8){};
            errdefer {
                for (row.items) |cell| {
                    self.allocator.free(cell);
                }
                row.deinit(self.allocator);
            }

            var field_iter = std.mem.splitSequence(u8, line, &[_]u8{self.delimiter});
            while (field_iter.next()) |field| {
                const trimmed = std.mem.trim(u8, field, " \t\r");
                try row.append(self.allocator, try self.allocator.dupe(u8, trimmed));
            }

            try rows.append(self.allocator, row);
        }

        return rows;
    }

    pub fn getHeader(self: *CSVParserNode, csv_content: []const u8) !?ArrayList([]const u8) {
        if (!self.has_header) return null;

        var line_iter = std.mem.splitSequence(u8, csv_content, "\n");
        const first_line = line_iter.next() orelse return null;

        var headers = ArrayList([]const u8){};
        errdefer {
            for (headers.items) |header| {
                self.allocator.free(header);
            }
            headers.deinit(self.allocator);
        }

        var field_iter = std.mem.splitSequence(u8, first_line, &[_]u8{self.delimiter});
        while (field_iter.next()) |field| {
            const trimmed = std.mem.trim(u8, field, " \t\r");
            try headers.append(self.allocator, try self.allocator.dupe(u8, trimmed));
        }

        return headers;
    }
};

/// JSON Parser Node - Parse and stringify JSON
pub const JSONParserNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    pretty: bool,

    pub fn init(allocator: Allocator, node_id: []const u8, pretty: bool) !*JSONParserNode {
        const node = try allocator.create(JSONParserNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .pretty = pretty,
        };
        return node;
    }

    pub fn deinit(self: *JSONParserNode) void {
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn parse(self: *JSONParserNode, json_str: []const u8) !std.json.Value {
        const parsed = try std.json.parseFromSlice(std.json.Value, self.allocator, json_str, .{});
        return parsed.value;
    }

    pub fn stringify(self: *JSONParserNode, value: std.json.Value) ![]const u8 {
        var buffer = ArrayList(u8){};
        errdefer buffer.deinit(self.allocator);

        if (self.pretty) {
            try std.json.stringify(value, .{ .whitespace = .indent_2 }, buffer.writer(self.allocator));
        } else {
            try std.json.stringify(value, .{}, buffer.writer(self.allocator));
        }

        return try buffer.toOwnedSlice(self.allocator);
    }
};

/// Cache Node - In-memory cache with TTL
pub const CacheNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    cache: StringHashMap(CacheEntry),
    default_ttl_ms: u64,

    const CacheEntry = struct {
        value: []const u8,
        expiry: i64, // millisecond timestamp
    };

    pub fn init(allocator: Allocator, node_id: []const u8, default_ttl_ms: u64) !*CacheNode {
        const node = try allocator.create(CacheNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .cache = StringHashMap(CacheEntry).init(allocator),
            .default_ttl_ms = default_ttl_ms,
        };
        return node;
    }

    pub fn deinit(self: *CacheNode) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.value);
        }
        self.cache.deinit();
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn set(self: *CacheNode, key: []const u8, value: []const u8, ttl_ms: ?u64) !void {
        const ttl = ttl_ms orelse self.default_ttl_ms;
        const expiry = std.time.milliTimestamp() + @as(i64, @intCast(ttl));

        // Remove existing entry if present
        if (self.cache.get(key)) |existing| {
            self.allocator.free(existing.value);
        }

        const key_copy = try self.allocator.dupe(u8, key);
        errdefer self.allocator.free(key_copy);
        const value_copy = try self.allocator.dupe(u8, value);
        errdefer self.allocator.free(value_copy);

        try self.cache.put(key_copy, CacheEntry{
            .value = value_copy,
            .expiry = expiry,
        });
    }

    pub fn get(self: *CacheNode, key: []const u8) ?[]const u8 {
        const entry = self.cache.get(key) orelse return null;

        // Check if expired
        const now = std.time.milliTimestamp();
        if (now > entry.expiry) {
            return null;
        }

        return entry.value;
    }

    pub fn delete(self: *CacheNode, key: []const u8) void {
        if (self.cache.fetchRemove(key)) |kv| {
            self.allocator.free(kv.key);
            self.allocator.free(kv.value.value);
        }
    }

    pub fn clear(self: *CacheNode) void {
        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            self.allocator.free(entry.key_ptr.*);
            self.allocator.free(entry.value_ptr.value);
        }
        self.cache.clearRetainingCapacity();
    }

    pub fn cleanExpired(self: *CacheNode) void {
        const now = std.time.milliTimestamp();
        var keys_to_remove = ArrayList([]const u8){};
        defer keys_to_remove.deinit(self.allocator);

        var iter = self.cache.iterator();
        while (iter.next()) |entry| {
            if (now > entry.value_ptr.expiry) {
                keys_to_remove.append(self.allocator, self.allocator, self.allocator, entry.key_ptr.*) catch continue;
            }
        }

        for (keys_to_remove.items) |key| {
            self.delete(key);
        }
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "FileWriterNode and FileReaderNode - roundtrip" {
    const allocator = std.testing.allocator;

    // Write
    var writer = try FileWriterNode.init(allocator, "writer-1", "/tmp/test_nworkflow.txt", false, true);
    defer writer.deinit();

    const content = "Hello, World!\nLine 2\nLine 3";
    try writer.execute(content);

    // Read
    var reader = try FileReaderNode.init(allocator, "reader-1", "/tmp/test_nworkflow.txt", "utf-8");
    defer reader.deinit();

    const read_content = try reader.execute();
    defer allocator.free(read_content);

    try std.testing.expectEqualStrings(content, read_content);

    // Cleanup
    try std.fs.cwd().deleteFile("/tmp/test_nworkflow.txt");
}

test "FileReaderNode - read lines" {
    const allocator = std.testing.allocator;

    // Create test file
    var writer = try FileWriterNode.init(allocator, "writer-2", "/tmp/test_lines.txt", false, true);
    defer writer.deinit();
    try writer.execute("Line 1\nLine 2\nLine 3");

    // Read lines
    var reader = try FileReaderNode.init(allocator, "reader-2", "/tmp/test_lines.txt", "utf-8");
    defer reader.deinit();

    var lines = try reader.readLines();
    defer {
        for (lines.items) |line| {
            allocator.free(line);
        }
        lines.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), lines.items.len);
    try std.testing.expectEqualStrings("Line 1", lines.items[0]);
    try std.testing.expectEqualStrings("Line 2", lines.items[1]);

    // Cleanup
    try std.fs.cwd().deleteFile("/tmp/test_lines.txt");
}

test "CSVParserNode - basic parsing" {
    const allocator = std.testing.allocator;

    var parser = try CSVParserNode.init(allocator, "csv-1", ',', true);
    defer parser.deinit();

    const csv = "Name,Age,City\nAlice,30,NYC\nBob,25,LA";
    var rows = try parser.execute(csv);
    defer {
        for (rows.items) |*row| {
            for (row.items) |cell| {
                allocator.free(cell);
            }
            row.deinit(allocator);
        }
        rows.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 2), rows.items.len);
    try std.testing.expectEqual(@as(usize, 3), rows.items[0].items.len);
    try std.testing.expectEqualStrings("Alice", rows.items[0].items[0]);
    try std.testing.expectEqualStrings("30", rows.items[0].items[1]);
}

test "CSVParserNode - get header" {
    const allocator = std.testing.allocator;

    var parser = try CSVParserNode.init(allocator, "csv-2", ',', true);
    defer parser.deinit();

    const csv = "Name,Age,City\nAlice,30,NYC";
    var headers_opt = try parser.getHeader(csv);
    if (headers_opt) |*headers| {
        defer {
            for (headers.items) |header| {
                allocator.free(header);
            }
            headers.deinit(allocator);
        }

        try std.testing.expectEqual(@as(usize, 3), headers.items.len);
        try std.testing.expectEqualStrings("Name", headers.items[0]);
        try std.testing.expectEqualStrings("Age", headers.items[1]);
        try std.testing.expectEqualStrings("City", headers.items[2]);
    } else {
        try std.testing.expect(false); // Should have header
    }
}

test "CacheNode - set and get" {
    const allocator = std.testing.allocator;

    var cache = try CacheNode.init(allocator, "cache-1", 5000);
    defer cache.deinit();

    try cache.set("key1", "value1", null);

    const value = cache.get("key1");
    try std.testing.expect(value != null);
    try std.testing.expectEqualStrings("value1", value.?);
}

test "CacheNode - TTL expiry" {
    const allocator = std.testing.allocator;

    var cache = try CacheNode.init(allocator, "cache-2", 50);
    defer cache.deinit();

    try cache.set("key1", "value1", 50); // 50ms TTL

    // Should exist immediately
    try std.testing.expect(cache.get("key1") != null);

    // Wait for expiry
    std.Thread.sleep(100 * std.time.ns_per_ms);

    // Should be expired
    try std.testing.expect(cache.get("key1") == null);
}

test "CacheNode - delete" {
    const allocator = std.testing.allocator;

    var cache = try CacheNode.init(allocator, "cache-3", 5000);
    defer cache.deinit();

    try cache.set("key1", "value1", null);
    try std.testing.expect(cache.get("key1") != null);

    cache.delete("key1");
    try std.testing.expect(cache.get("key1") == null);
}

test "CacheNode - clear" {
    const allocator = std.testing.allocator;

    var cache = try CacheNode.init(allocator, "cache-4", 5000);
    defer cache.deinit();

    try cache.set("key1", "value1", null);
    try cache.set("key2", "value2", null);

    cache.clear();

    try std.testing.expect(cache.get("key1") == null);
    try std.testing.expect(cache.get("key2") == null);
}
