// Day 28: Langflow Component Parity - Text Splitter
// Text splitting and chunking component for processing large text

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

/// Split strategy for text processing
pub const SplitStrategy = enum {
    by_delimiter, // Split by delimiter (newline, comma, etc.)
    by_length, // Split into fixed-size chunks
    by_sentence, // Split by sentence boundaries
    by_paragraph, // Split by paragraphs
    by_word_count, // Split by number of words
    recursive, // Recursive splitting (try larger chunks first)

    pub fn toString(self: SplitStrategy) []const u8 {
        return switch (self) {
            .by_delimiter => "by_delimiter",
            .by_length => "by_length",
            .by_sentence => "by_sentence",
            .by_paragraph => "by_paragraph",
            .by_word_count => "by_word_count",
            .recursive => "recursive",
        };
    }

    pub fn fromString(s: []const u8) !SplitStrategy {
        if (std.mem.eql(u8, s, "by_delimiter")) return .by_delimiter;
        if (std.mem.eql(u8, s, "by_length")) return .by_length;
        if (std.mem.eql(u8, s, "by_sentence")) return .by_sentence;
        if (std.mem.eql(u8, s, "by_paragraph")) return .by_paragraph;
        if (std.mem.eql(u8, s, "by_word_count")) return .by_word_count;
        if (std.mem.eql(u8, s, "recursive")) return .recursive;
        return error.InvalidStrategy;
    }
};

/// Text Splitter Node - Splits text into chunks
pub const TextSplitterNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    strategy: SplitStrategy,
    delimiter: []const u8,
    chunk_size: usize,
    chunk_overlap: usize,
    trim_whitespace: bool,
    remove_empty: bool,

    pub fn init(
        allocator: Allocator,
        node_id: []const u8,
        strategy: SplitStrategy,
        delimiter: []const u8,
        chunk_size: usize,
        chunk_overlap: usize,
    ) !*TextSplitterNode {
        const node = try allocator.create(TextSplitterNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .strategy = strategy,
            .delimiter = try allocator.dupe(u8, delimiter),
            .chunk_size = chunk_size,
            .chunk_overlap = chunk_overlap,
            .trim_whitespace = true,
            .remove_empty = true,
        };
        return node;
    }

    pub fn deinit(self: *TextSplitterNode) void {
        self.allocator.free(self.node_id);
        self.allocator.free(self.delimiter);
        self.allocator.destroy(self);
    }

    pub fn execute(self: *TextSplitterNode, input_text: []const u8) !ArrayList([]const u8) {
        var chunks = ArrayList([]const u8){};
        errdefer {
            for (chunks.items) |chunk| {
                self.allocator.free(chunk);
            }
            chunks.deinit(self.allocator);
        }

        switch (self.strategy) {
            .by_delimiter => try self.splitByDelimiter(input_text, &chunks),
            .by_length => try self.splitByLength(input_text, &chunks),
            .by_sentence => try self.splitBySentence(input_text, &chunks),
            .by_paragraph => try self.splitByParagraph(input_text, &chunks),
            .by_word_count => try self.splitByWordCount(input_text, &chunks),
            .recursive => try self.splitRecursive(input_text, &chunks),
        }

        return chunks;
    }

    fn splitByDelimiter(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        var iter = std.mem.splitSequence(u8, text, self.delimiter);
        while (iter.next()) |chunk| {
            const processed = if (self.trim_whitespace) std.mem.trim(u8, chunk, " \t\r\n") else chunk;
            if (self.remove_empty and processed.len == 0) continue;
            try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
        }
    }

    fn splitByLength(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        if (self.chunk_size == 0) return error.InvalidChunkSize;

        var start: usize = 0;
        while (start < text.len) {
            const end = @min(start + self.chunk_size, text.len);
            const chunk = text[start..end];

            const processed = if (self.trim_whitespace) std.mem.trim(u8, chunk, " \t\r\n") else chunk;
            if (!self.remove_empty or processed.len > 0) {
                try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
            }

            // Move start position with overlap
            if (self.chunk_overlap > 0 and end < text.len) {
                start = end -| self.chunk_overlap; // Saturating subtraction
            } else {
                start = end;
            }
        }
    }

    fn splitBySentence(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        // Split by common sentence endings: . ! ?
        var current_chunk = ArrayList(u8){};
        defer current_chunk.deinit(self.allocator);

        var i: usize = 0;
        while (i < text.len) {
            const char = text[i];
            try current_chunk.append(self.allocator, char);

            // Check for sentence ending
            if (char == '.' or char == '!' or char == '?') {
                // Look ahead to see if this is end of sentence
                const next_idx = i + 1;
                if (next_idx >= text.len or text[next_idx] == ' ' or text[next_idx] == '\n') {
                    // End of sentence found
                    const sentence = try current_chunk.toOwnedSlice(self.allocator);
                    defer self.allocator.free(sentence);

                    const processed = if (self.trim_whitespace) std.mem.trim(u8, sentence, " \t\r\n") else sentence;
                    if (!self.remove_empty or processed.len > 0) {
                        try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
                    }

                    current_chunk.clearRetainingCapacity();
                }
            }

            i += 1;
        }

        // Add remaining text as final chunk
        if (current_chunk.items.len > 0) {
            const sentence = try current_chunk.toOwnedSlice(self.allocator);
            defer self.allocator.free(sentence);

            const processed = if (self.trim_whitespace) std.mem.trim(u8, sentence, " \t\r\n") else sentence;
            if (!self.remove_empty or processed.len > 0) {
                try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
            }
        }
    }

    fn splitByParagraph(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        // Split by double newlines (paragraphs)
        var iter = std.mem.splitSequence(u8, text, "\n\n");
        while (iter.next()) |paragraph| {
            const processed = if (self.trim_whitespace) std.mem.trim(u8, paragraph, " \t\r\n") else paragraph;
            if (self.remove_empty and processed.len == 0) continue;
            try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
        }
    }

    fn splitByWordCount(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        if (self.chunk_size == 0) return error.InvalidChunkSize;

        var words = std.mem.tokenizeAny(u8, text, " \t\r\n");
        var current_chunk = ArrayList(u8){};
        defer current_chunk.deinit(self.allocator);

        var word_count: usize = 0;

        while (words.next()) |word| {
            if (word_count > 0) {
                try current_chunk.append(self.allocator, ' ');
            }
            try current_chunk.appendSlice(self.allocator, word);
            word_count += 1;

            if (word_count >= self.chunk_size) {
                const chunk = try current_chunk.toOwnedSlice(self.allocator);
                defer self.allocator.free(chunk);

                try chunks.append(self.allocator, try self.allocator.dupe(u8, chunk));
                current_chunk.clearRetainingCapacity();
                word_count = 0;
            }
        }

        // Add remaining words as final chunk
        if (current_chunk.items.len > 0) {
            const chunk = try current_chunk.toOwnedSlice(self.allocator);
            defer self.allocator.free(chunk);

            if (!self.remove_empty or chunk.len > 0) {
                try chunks.append(self.allocator, try self.allocator.dupe(u8, chunk));
            }
        }
    }

    fn splitRecursive(self: *TextSplitterNode, text: []const u8, chunks: *ArrayList([]const u8)) !void {
        // Try multiple strategies in order: paragraph -> sentence -> length
        const strategies = [_][]const u8{ "\n\n", "\n", ". ", " " };

        var remaining = text;
        while (remaining.len > 0) {
            if (remaining.len <= self.chunk_size) {
                // Small enough, add as chunk
                const processed = if (self.trim_whitespace) std.mem.trim(u8, remaining, " \t\r\n") else remaining;
                if (!self.remove_empty or processed.len > 0) {
                    try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
                }
                break;
            }

            // Try to find a good split point
            var split_found = false;
            for (strategies) |sep| {
                // Look for separator within chunk_size
                const search_end = @min(self.chunk_size, remaining.len);
                if (std.mem.lastIndexOf(u8, remaining[0..search_end], sep)) |idx| {
                    const chunk = remaining[0 .. idx + sep.len];
                    const processed = if (self.trim_whitespace) std.mem.trim(u8, chunk, " \t\r\n") else chunk;
                    if (!self.remove_empty or processed.len > 0) {
                        try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
                    }
                    remaining = remaining[idx + sep.len ..];
                    split_found = true;
                    break;
                }
            }

            if (!split_found) {
                // No separator found, force split at chunk_size
                const end = @min(self.chunk_size, remaining.len);
                const chunk = remaining[0..end];
                const processed = if (self.trim_whitespace) std.mem.trim(u8, chunk, " \t\r\n") else chunk;
                if (!self.remove_empty or processed.len > 0) {
                    try chunks.append(self.allocator, try self.allocator.dupe(u8, processed));
                }
                remaining = remaining[end..];
            }
        }
    }

    pub fn getMetrics(_: *TextSplitterNode, input_text: []const u8, chunks: *const ArrayList([]const u8)) !ChunkMetrics {
        var total_chars: usize = 0;
        var min_size: usize = std.math.maxInt(usize);
        var max_size: usize = 0;

        for (chunks.items) |chunk| {
            total_chars += chunk.len;
            min_size = @min(min_size, chunk.len);
            max_size = @max(max_size, chunk.len);
        }

        const avg_size: f64 = if (chunks.items.len > 0) @as(f64, @floatFromInt(total_chars)) / @as(f64, @floatFromInt(chunks.items.len)) else 0.0;

        return ChunkMetrics{
            .original_size = input_text.len,
            .chunk_count = chunks.items.len,
            .total_chars = total_chars,
            .min_chunk_size = if (chunks.items.len > 0) min_size else 0,
            .max_chunk_size = max_size,
            .avg_chunk_size = avg_size,
        };
    }
};

/// Metrics for text splitting operations
pub const ChunkMetrics = struct {
    original_size: usize,
    chunk_count: usize,
    total_chars: usize,
    min_chunk_size: usize,
    max_chunk_size: usize,
    avg_chunk_size: f64,
};

// ============================================================================
// TESTS
// ============================================================================

test "TextSplitterNode - split by delimiter" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-1", .by_delimiter, ",", 0, 0);
    defer node.deinit();

    const input = "apple,banana,cherry,date";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 4), chunks.items.len);
    try std.testing.expectEqualStrings("apple", chunks.items[0]);
    try std.testing.expectEqualStrings("banana", chunks.items[1]);
    try std.testing.expectEqualStrings("cherry", chunks.items[2]);
    try std.testing.expectEqualStrings("date", chunks.items[3]);
}

test "TextSplitterNode - split by length" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-2", .by_length, "", 10, 0);
    defer node.deinit();

    const input = "This is a test string for chunking";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expect(chunks.items.len >= 3);
    try std.testing.expect(chunks.items[0].len <= 10);
}

test "TextSplitterNode - split by length with overlap" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-3", .by_length, "", 10, 3);
    defer node.deinit();

    const input = "This is a test string for chunking";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    // With overlap, we should have more chunks
    try std.testing.expect(chunks.items.len >= 3);
}

test "TextSplitterNode - split by sentence" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-4", .by_sentence, "", 0, 0);
    defer node.deinit();

    const input = "First sentence. Second sentence! Third sentence? Fourth.";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 4), chunks.items.len);
}

test "TextSplitterNode - split by paragraph" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-5", .by_paragraph, "", 0, 0);
    defer node.deinit();

    const input = "Paragraph one.\nLine two.\n\nParagraph two.\nLine two.\n\nParagraph three.";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), chunks.items.len);
}

test "TextSplitterNode - split by word count" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-6", .by_word_count, "", 3, 0);
    defer node.deinit();

    const input = "one two three four five six seven eight nine";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expectEqual(@as(usize, 3), chunks.items.len);
}

test "TextSplitterNode - recursive split" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-7", .recursive, "", 20, 0);
    defer node.deinit();

    const input = "Paragraph one.\n\nParagraph two with longer text. More text here.\n\nParagraph three.";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    try std.testing.expect(chunks.items.len >= 3);
}

test "TextSplitterNode - metrics" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-8", .by_delimiter, ",", 0, 0);
    defer node.deinit();

    const input = "apple,banana,cherry";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    const metrics = try node.getMetrics(input, &chunks);
    try std.testing.expectEqual(@as(usize, 19), metrics.original_size);
    try std.testing.expectEqual(@as(usize, 3), metrics.chunk_count);
    try std.testing.expectEqual(@as(usize, 5), metrics.min_chunk_size);
    try std.testing.expectEqual(@as(usize, 6), metrics.max_chunk_size);
}

test "TextSplitterNode - empty chunks removal" {
    const allocator = std.testing.allocator;

    var node = try TextSplitterNode.init(allocator, "splitter-9", .by_delimiter, ",", 0, 0);
    defer node.deinit();

    const input = "apple,,banana,,cherry";
    var chunks = try node.execute(input);
    defer {
        for (chunks.items) |chunk| {
            allocator.free(chunk);
        }
        chunks.deinit(allocator);
    }

    // Should have 3 chunks (empty ones removed)
    try std.testing.expectEqual(@as(usize, 3), chunks.items.len);
}

test "SplitStrategy - toString and fromString" {
    try std.testing.expectEqualStrings("by_delimiter", SplitStrategy.by_delimiter.toString());
    try std.testing.expectEqualStrings("by_length", SplitStrategy.by_length.toString());
    try std.testing.expectEqualStrings("by_sentence", SplitStrategy.by_sentence.toString());

    try std.testing.expectEqual(SplitStrategy.by_delimiter, try SplitStrategy.fromString("by_delimiter"));
    try std.testing.expectEqual(SplitStrategy.by_length, try SplitStrategy.fromString("by_length"));
    try std.testing.expectEqual(SplitStrategy.recursive, try SplitStrategy.fromString("recursive"));
}
