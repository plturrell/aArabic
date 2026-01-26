// Text Document Synchronization
// Day 72: Document sync with incremental and full change support

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Text Document Change Types
// ============================================================================

/// Text Document Content Change Kind
pub const TextDocumentSyncKind = enum(u8) {
    None = 0,
    Full = 1,
    Incremental = 2,
};

/// Position in a text document
pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
    
    pub fn compare(self: Position, other: Position) i8 {
        if (self.line < other.line) return -1;
        if (self.line > other.line) return 1;
        if (self.character < other.character) return -1;
        if (self.character > other.character) return 1;
        return 0;
    }
};

/// Range in a text document
pub const Range = struct {
    start: Position,
    end: Position,
    
    pub fn init(start: Position, end: Position) Range {
        return Range{ .start = start, .end = end };
    }
    
    pub fn isEmpty(self: Range) bool {
        return self.start.compare(self.end) == 0;
    }
};

/// Text Document Content Change Event (Incremental)
pub const TextDocumentContentChangeEvent = struct {
    range: ?Range = null, // If null, full document
    range_length: ?u32 = null, // Deprecated but may be present
    text: []const u8,
    
    pub fn init(text: []const u8) TextDocumentContentChangeEvent {
        return TextDocumentContentChangeEvent{
            .text = text,
        };
    }
    
    pub fn initWithRange(range: Range, text: []const u8) TextDocumentContentChangeEvent {
        return TextDocumentContentChangeEvent{
            .range = range,
            .text = text,
        };
    }
};

// ============================================================================
// Versioned Text Document
// ============================================================================

pub const VersionedTextDocument = struct {
    uri: []const u8,
    language_id: []const u8,
    version: i32,
    content: std.ArrayList(u8),
    allocator: Allocator,
    
    // Line offsets for efficient position lookup
    line_offsets: std.ArrayList(u32),
    
    pub fn init(allocator: Allocator, uri: []const u8, language_id: []const u8, version: i32, initial_content: []const u8) !VersionedTextDocument {
        const uri_copy = try allocator.dupe(u8, uri);
        errdefer allocator.free(uri_copy);
        
        const lang_copy = try allocator.dupe(u8, language_id);
        errdefer allocator.free(lang_copy);
        
        var content = try std.ArrayList(u8).initCapacity(allocator, initial_content.len);
        try content.appendSlice(allocator, initial_content);
        
        var doc = VersionedTextDocument{
            .uri = uri_copy,
            .language_id = lang_copy,
            .version = version,
            .content = content,
            .allocator = allocator,
            .line_offsets = try std.ArrayList(u32).initCapacity(allocator, 16),
        };
        
        try doc.updateLineOffsets();
        
        return doc;
    }
    
    pub fn deinit(self: *VersionedTextDocument) void {
        self.allocator.free(self.uri);
        self.allocator.free(self.language_id);
        self.content.deinit(self.allocator);
        self.line_offsets.deinit(self.allocator);
    }
    
    /// Update line offsets for efficient position calculations
    fn updateLineOffsets(self: *VersionedTextDocument) !void {
        self.line_offsets.clearRetainingCapacity();
        try self.line_offsets.append(self.allocator, 0); // Line 0 starts at offset 0
        
        for (self.content.items, 0..) |c, i| {
            if (c == '\n') {
                try self.line_offsets.append(self.allocator, @intCast(i + 1));
            }
        }
    }
    
    /// Get the byte offset for a position
    pub fn positionToOffset(self: VersionedTextDocument, pos: Position) !u32 {
        if (pos.line >= self.line_offsets.items.len) {
            return error.PositionOutOfBounds;
        }
        
        const line_start = self.line_offsets.items[pos.line];
        return line_start + pos.character;
    }
    
    /// Get the position for a byte offset
    pub fn offsetToPosition(self: VersionedTextDocument, offset: u32) Position {
        var line: u32 = 0;
        while (line + 1 < self.line_offsets.items.len and self.line_offsets.items[line + 1] <= offset) {
            line += 1;
        }
        
        const line_start = self.line_offsets.items[line];
        return Position.init(line, offset - line_start);
    }
    
    /// Apply a full document change
    pub fn applyFullChange(self: *VersionedTextDocument, new_content: []const u8, new_version: i32) !void {
        self.content.clearRetainingCapacity();
        try self.content.appendSlice(self.allocator, new_content);
        self.version = new_version;
        try self.updateLineOffsets();
    }
    
    /// Apply an incremental change
    pub fn applyIncrementalChange(self: *VersionedTextDocument, change: TextDocumentContentChangeEvent, new_version: i32) !void {
        if (change.range) |range| {
            // Incremental change
            const start_offset = try self.positionToOffset(range.start);
            const end_offset = try self.positionToOffset(range.end);
            
            // Remove old text
            const remove_count = end_offset - start_offset;
            var i: u32 = 0;
            while (i < remove_count) : (i += 1) {
                _ = self.content.orderedRemove(start_offset);
            }
            
            // Insert new text
            try self.content.insertSlice(self.allocator, start_offset, change.text);
            
            self.version = new_version;
            try self.updateLineOffsets();
        } else {
            // Full document change
            try self.applyFullChange(change.text, new_version);
        }
    }
    
    /// Get line count
    pub fn getLineCount(self: VersionedTextDocument) u32 {
        return @intCast(self.line_offsets.items.len);
    }
    
    /// Get a specific line
    pub fn getLine(self: VersionedTextDocument, line_number: u32) ?[]const u8 {
        if (line_number >= self.line_offsets.items.len) {
            return null;
        }
        
        const start = self.line_offsets.items[line_number];
        const end = if (line_number + 1 < self.line_offsets.items.len)
            self.line_offsets.items[line_number + 1] - 1 // -1 to exclude \n
        else
            @as(u32, @intCast(self.content.items.len));
        
        return self.content.items[start..end];
    }
    
    /// Get the entire content as a slice
    pub fn getContent(self: VersionedTextDocument) []const u8 {
        return self.content.items;
    }
};

// ============================================================================
// Synchronization Manager
// ============================================================================

pub const TextSyncManager = struct {
    documents: std.StringHashMap(VersionedTextDocument),
    allocator: Allocator,
    sync_kind: TextDocumentSyncKind,
    
    pub fn init(allocator: Allocator, sync_kind: TextDocumentSyncKind) TextSyncManager {
        return TextSyncManager{
            .documents = std.StringHashMap(VersionedTextDocument).init(allocator),
            .allocator = allocator,
            .sync_kind = sync_kind,
        };
    }
    
    pub fn deinit(self: *TextSyncManager) void {
        var iter = self.documents.iterator();
        while (iter.next()) |entry| {
            var doc = entry.value_ptr;
            doc.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.documents.deinit();
    }
    
    /// Handle didOpen notification
    pub fn didOpen(self: *TextSyncManager, uri: []const u8, language_id: []const u8, version: i32, text: []const u8) !void {
        const doc = try VersionedTextDocument.init(self.allocator, uri, language_id, version, text);
        const uri_copy = try self.allocator.dupe(u8, uri);
        try self.documents.put(uri_copy, doc);
    }
    
    /// Handle didChange notification
    pub fn didChange(self: *TextSyncManager, uri: []const u8, version: i32, changes: []const TextDocumentContentChangeEvent) !void {
        var doc = self.documents.getPtr(uri) orelse return error.DocumentNotFound;
        
        for (changes) |change| {
            try doc.applyIncrementalChange(change, version);
        }
    }
    
    /// Handle didClose notification
    pub fn didClose(self: *TextSyncManager, uri: []const u8) !void {
        if (self.documents.fetchRemove(uri)) |kv| {
            var doc = kv.value;
            doc.deinit();
            self.allocator.free(kv.key);
        }
    }
    
    /// Get a document
    pub fn getDocument(self: *TextSyncManager, uri: []const u8) ?*VersionedTextDocument {
        return self.documents.getPtr(uri);
    }
    
    /// Check if document exists
    pub fn hasDocument(self: *TextSyncManager, uri: []const u8) bool {
        return self.documents.contains(uri);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Position: comparison" {
    const pos1 = Position.init(5, 10);
    const pos2 = Position.init(5, 10);
    const pos3 = Position.init(5, 20);
    const pos4 = Position.init(6, 0);
    
    try std.testing.expectEqual(@as(i8, 0), pos1.compare(pos2));
    try std.testing.expectEqual(@as(i8, -1), pos1.compare(pos3));
    try std.testing.expectEqual(@as(i8, -1), pos1.compare(pos4));
    try std.testing.expectEqual(@as(i8, 1), pos3.compare(pos1));
}

test "Range: isEmpty" {
    const range1 = Range.init(Position.init(5, 10), Position.init(5, 10));
    const range2 = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    try std.testing.expect(range1.isEmpty());
    try std.testing.expect(!range2.isEmpty());
}

test "VersionedTextDocument: initialization" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "line 1\nline 2\nline 3",
    );
    defer doc.deinit();
    
    try std.testing.expectEqual(@as(i32, 1), doc.version);
    try std.testing.expectEqual(@as(u32, 3), doc.getLineCount());
}

test "VersionedTextDocument: position to offset" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "line 1\nline 2\nline 3",
    );
    defer doc.deinit();
    
    const offset1 = try doc.positionToOffset(Position.init(0, 0));
    try std.testing.expectEqual(@as(u32, 0), offset1);
    
    const offset2 = try doc.positionToOffset(Position.init(1, 0));
    try std.testing.expectEqual(@as(u32, 7), offset2);
    
    const offset3 = try doc.positionToOffset(Position.init(1, 4));
    try std.testing.expectEqual(@as(u32, 11), offset3);
}

test "VersionedTextDocument: offset to position" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "line 1\nline 2\nline 3",
    );
    defer doc.deinit();
    
    const pos1 = doc.offsetToPosition(0);
    try std.testing.expectEqual(@as(u32, 0), pos1.line);
    try std.testing.expectEqual(@as(u32, 0), pos1.character);
    
    const pos2 = doc.offsetToPosition(7);
    try std.testing.expectEqual(@as(u32, 1), pos2.line);
    try std.testing.expectEqual(@as(u32, 0), pos2.character);
}

test "VersionedTextDocument: full change" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "old content",
    );
    defer doc.deinit();
    
    try doc.applyFullChange("new content", 2);
    
    try std.testing.expectEqual(@as(i32, 2), doc.version);
    try std.testing.expectEqualStrings("new content", doc.getContent());
}

test "VersionedTextDocument: incremental change - insert" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "Hello World",
    );
    defer doc.deinit();
    
    // Insert " Beautiful" between "Hello" and " World"
    const change = TextDocumentContentChangeEvent.initWithRange(
        Range.init(Position.init(0, 5), Position.init(0, 5)),
        " Beautiful",
    );
    
    try doc.applyIncrementalChange(change, 2);
    
    try std.testing.expectEqualStrings("Hello Beautiful World", doc.getContent());
    try std.testing.expectEqual(@as(i32, 2), doc.version);
}

test "VersionedTextDocument: incremental change - replace" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "Hello World",
    );
    defer doc.deinit();
    
    // Replace "World" with "Universe"
    const change = TextDocumentContentChangeEvent.initWithRange(
        Range.init(Position.init(0, 6), Position.init(0, 11)),
        "Universe",
    );
    
    try doc.applyIncrementalChange(change, 2);
    
    try std.testing.expectEqualStrings("Hello Universe", doc.getContent());
}

test "VersionedTextDocument: get line" {
    var doc = try VersionedTextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "line 1\nline 2\nline 3",
    );
    defer doc.deinit();
    
    const line1 = doc.getLine(0).?;
    try std.testing.expectEqualStrings("line 1", line1);
    
    const line2 = doc.getLine(1).?;
    try std.testing.expectEqualStrings("line 2", line2);
}

test "TextSyncManager: document lifecycle" {
    var manager = TextSyncManager.init(std.testing.allocator, .Incremental);
    defer manager.deinit();
    
    // Open document
    try manager.didOpen("file:///test.mojo", "mojo", 1, "initial content");
    try std.testing.expect(manager.hasDocument("file:///test.mojo"));
    
    // Change document
    const changes = [_]TextDocumentContentChangeEvent{
        TextDocumentContentChangeEvent.init("new content"),
    };
    try manager.didChange("file:///test.mojo", 2, &changes);
    
    const doc = manager.getDocument("file:///test.mojo").?;
    try std.testing.expectEqual(@as(i32, 2), doc.version);
    
    // Close document
    try manager.didClose("file:///test.mojo");
    try std.testing.expect(!manager.hasDocument("file:///test.mojo"));
}

test "TextSyncManager: multiple changes" {
    var manager = TextSyncManager.init(std.testing.allocator, .Incremental);
    defer manager.deinit();
    
    try manager.didOpen("file:///test.mojo", "mojo", 1, "Hello");
    
    const changes = [_]TextDocumentContentChangeEvent{
        TextDocumentContentChangeEvent.initWithRange(
            Range.init(Position.init(0, 5), Position.init(0, 5)),
            " World",
        ),
    };
    
    try manager.didChange("file:///test.mojo", 2, &changes);
    
    const doc = manager.getDocument("file:///test.mojo").?;
    try std.testing.expectEqualStrings("Hello World", doc.getContent());
}
