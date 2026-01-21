// Day 28: Langflow Component Parity - Text Cleaner
// Text cleaning and normalization component

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

/// Text cleaning operations
pub const CleanOperation = enum {
    lowercase,
    uppercase,
    trim_whitespace,
    remove_special_chars,
    remove_numbers,
    remove_extra_spaces,
    normalize_unicode,
    remove_urls,
    remove_emails,
    remove_html_tags,

    pub fn toString(self: CleanOperation) []const u8 {
        return switch (self) {
            .lowercase => "lowercase",
            .uppercase => "uppercase",
            .trim_whitespace => "trim_whitespace",
            .remove_special_chars => "remove_special_chars",
            .remove_numbers => "remove_numbers",
            .remove_extra_spaces => "remove_extra_spaces",
            .normalize_unicode => "normalize_unicode",
            .remove_urls => "remove_urls",
            .remove_emails => "remove_emails",
            .remove_html_tags => "remove_html_tags",
        };
    }
};

/// Text Cleaner Node - Cleans and normalizes text
pub const TextCleanerNode = struct {
    allocator: Allocator,
    node_id: []const u8,
    operations: ArrayList(CleanOperation),

    pub fn init(allocator: Allocator, node_id: []const u8) !*TextCleanerNode {
        const node = try allocator.create(TextCleanerNode);
        node.* = .{
            .allocator = allocator,
            .node_id = try allocator.dupe(u8, node_id),
            .operations = ArrayList(CleanOperation){},
        };
        return node;
    }

    pub fn deinit(self: *TextCleanerNode) void {
        self.operations.deinit(self.allocator);
        self.allocator.free(self.node_id);
        self.allocator.destroy(self);
    }

    pub fn addOperation(self: *TextCleanerNode, operation: CleanOperation) !void {
        try self.operations.append(self.allocator, operation);
    }

    pub fn execute(self: *TextCleanerNode, input_text: []const u8) ![]const u8 {
        var result: []const u8 = try self.allocator.dupe(u8, input_text);
        errdefer self.allocator.free(result);

        for (self.operations.items) |operation| {
            const temp = result;
            result = try self.applyOperation(temp, operation);
            self.allocator.free(temp);
        }

        return result;
    }

    fn applyOperation(self: *TextCleanerNode, text: []const u8, operation: CleanOperation) ![]u8 {
        return switch (operation) {
            .lowercase => try self.toLowercase(text),
            .uppercase => try self.toUppercase(text),
            .trim_whitespace => try self.trimWhitespace(text),
            .remove_special_chars => try self.removeSpecialChars(text),
            .remove_numbers => try self.removeNumbers(text),
            .remove_extra_spaces => try self.removeExtraSpaces(text),
            .normalize_unicode => try self.normalizeUnicode(text),
            .remove_urls => try self.removeUrls(text),
            .remove_emails => try self.removeEmails(text),
            .remove_html_tags => try self.removeHtmlTags(text),
        };
    }

    fn toLowercase(self: *TextCleanerNode, text: []const u8) ![]u8 {
        const result = try self.allocator.alloc(u8, text.len);
        errdefer self.allocator.free(result);

        for (text, 0..) |char, i| {
            result[i] = std.ascii.toLower(char);
        }

        return result;
    }

    fn toUppercase(self: *TextCleanerNode, text: []const u8) ![]u8 {
        const result = try self.allocator.alloc(u8, text.len);
        errdefer self.allocator.free(result);

        for (text, 0..) |char, i| {
            result[i] = std.ascii.toUpper(char);
        }

        return result;
    }

    fn trimWhitespace(self: *TextCleanerNode, text: []const u8) ![]u8 {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        return try self.allocator.dupe(u8, trimmed);
    }

    fn removeSpecialChars(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        for (text) |char| {
            if (std.ascii.isAlphanumeric(char) or char == ' ') {
                try result.append(self.allocator, char);
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn removeNumbers(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        for (text) |char| {
            if (!std.ascii.isDigit(char)) {
                try result.append(self.allocator, char);
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn removeExtraSpaces(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var prev_was_space = false;
        for (text) |char| {
            const is_space = char == ' ' or char == '\t' or char == '\r' or char == '\n';
            if (is_space) {
                if (!prev_was_space) {
                    try result.append(self.allocator, ' ');
                    prev_was_space = true;
                }
            } else {
                try result.append(self.allocator, char);
                prev_was_space = false;
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn normalizeUnicode(self: *TextCleanerNode, text: []const u8) ![]u8 {
        // Simplified Unicode normalization - replace common non-ASCII with ASCII equivalents
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        for (text) |char| {
            if (char < 128) {
                try result.append(self.allocator, char);
            } else {
                // Replace with space for now (full Unicode normalization is complex)
                try result.append(self.allocator, ' ');
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn removeUrls(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < text.len) {
            // Simple URL detection: http:// or https://
            if (i + 7 <= text.len and std.mem.eql(u8, text[i .. i + 7], "http://")) {
                // Skip until space or end
                while (i < text.len and text[i] != ' ' and text[i] != '\n') {
                    i += 1;
                }
            } else if (i + 8 <= text.len and std.mem.eql(u8, text[i .. i + 8], "https://")) {
                // Skip until space or end
                while (i < text.len and text[i] != ' ' and text[i] != '\n') {
                    i += 1;
                }
            } else {
                try result.append(self.allocator, text[i]);
                i += 1;
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn removeEmails(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var i: usize = 0;
        while (i < text.len) {
            // Simple email detection: find @ symbol
            if (text[i] == '@') {
                // Backtrack to find start of email
                var start = i;
                while (start > 0 and (std.ascii.isAlphanumeric(text[start - 1]) or text[start - 1] == '.' or text[start - 1] == '_')) {
                    start -= 1;
                }

                // Remove characters we added that were part of the email
                const chars_to_remove = i - start;
                if (result.items.len >= chars_to_remove) {
                    result.items.len -= chars_to_remove;
                }

                // Skip forward to find end of email (after @)
                i += 1;
                while (i < text.len and (std.ascii.isAlphanumeric(text[i]) or text[i] == '.' or text[i] == '_')) {
                    i += 1;
                }

                // Email completely skipped
            } else {
                try result.append(self.allocator, text[i]);
                i += 1;
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }

    fn removeHtmlTags(self: *TextCleanerNode, text: []const u8) ![]u8 {
        var result = ArrayList(u8){};
        errdefer result.deinit(self.allocator);

        var in_tag = false;
        for (text) |char| {
            if (char == '<') {
                in_tag = true;
            } else if (char == '>') {
                in_tag = false;
            } else if (!in_tag) {
                try result.append(self.allocator, char);
            }
        }

        return try result.toOwnedSlice(self.allocator);
    }
};

// ============================================================================
// TESTS
// ============================================================================

test "TextCleanerNode - lowercase" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-1");
    defer node.deinit();
    try node.addOperation(.lowercase);

    const result = try node.execute("Hello WORLD!");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello world!", result);
}

test "TextCleanerNode - uppercase" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-2");
    defer node.deinit();
    try node.addOperation(.uppercase);

    const result = try node.execute("Hello World!");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("HELLO WORLD!", result);
}

test "TextCleanerNode - trim whitespace" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-3");
    defer node.deinit();
    try node.addOperation(.trim_whitespace);

    const result = try node.execute("  Hello World!  \n");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World!", result);
}

test "TextCleanerNode - remove special chars" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-4");
    defer node.deinit();
    try node.addOperation(.remove_special_chars);

    const result = try node.execute("Hello, World! #test");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World test", result);
}

test "TextCleanerNode - remove numbers" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-5");
    defer node.deinit();
    try node.addOperation(.remove_numbers);

    const result = try node.execute("Test 123 hello 456");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Test  hello ", result);
}

test "TextCleanerNode - remove extra spaces" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-6");
    defer node.deinit();
    try node.addOperation(.remove_extra_spaces);

    const result = try node.execute("Hello    World!  \n\n  Test");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World! Test", result);
}

test "TextCleanerNode - remove URLs" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-7");
    defer node.deinit();
    try node.addOperation(.remove_urls);

    const result = try node.execute("Visit https://example.com for more info");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Visit  for more info", result);
}

test "TextCleanerNode - remove emails" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-8");
    defer node.deinit();
    try node.addOperation(.remove_emails);

    const result = try node.execute("Contact user@example.com for help");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Contact  for help", result);
}

test "TextCleanerNode - remove HTML tags" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-9");
    defer node.deinit();
    try node.addOperation(.remove_html_tags);

    const result = try node.execute("<p>Hello <b>World</b>!</p>");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("Hello World!", result);
}

test "TextCleanerNode - multiple operations" {
    const allocator = std.testing.allocator;

    var node = try TextCleanerNode.init(allocator, "cleaner-10");
    defer node.deinit();
    try node.addOperation(.remove_html_tags);
    try node.addOperation(.lowercase);
    try node.addOperation(.trim_whitespace);

    const result = try node.execute("  <p>Hello WORLD!</p>  ");
    defer allocator.free(result);

    try std.testing.expectEqualStrings("hello world!", result);
}
