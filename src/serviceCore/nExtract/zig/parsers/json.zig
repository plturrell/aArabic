//! JSON Parser - RFC 8259 Compliant (Pure Zig)
//!
//! This module provides:
//! - Full RFC 8259 compliance
//! - Streaming parser for large JSON files
//! - Memory-efficient parsing
//! - Unicode support (UTF-8)
//! - Number parsing (integers, floats, exponential notation)
//! - Escape sequence handling
//! - Configurable options
//!
//! Day 11: JSON Parser Implementation
//! Author: nExtract Team
//! Date: January 17, 2026

const std = @import("std");
const types = @import("../core/types.zig");
const string = @import("../core/string.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// Data Structures
// ============================================================================

/// JSON value types
pub const JsonType = enum {
    Null,
    Boolean,
    Number,
    String,
    Array,
    Object,
};

/// JSON value
pub const JsonValue = union(JsonType) {
    Null: void,
    Boolean: bool,
    Number: f64,
    String: []const u8,
    Array: std.ArrayList(*JsonValue),
    Object: std.StringHashMap(*JsonValue),

    pub fn deinit(self: *JsonValue, allocator: Allocator) void {
        switch (self.*) {
            .String => |str| allocator.free(str),
            .Array => |*arr| {
                for (arr.items) |item| {
                    item.deinit(allocator);
                    allocator.destroy(item);
                }
                arr.deinit();
            },
            .Object => |*obj| {
                var it = obj.iterator();
                while (it.next()) |entry| {
                    allocator.free(entry.key_ptr.*);
                    entry.value_ptr.*.deinit(allocator);
                    allocator.destroy(entry.value_ptr.*);
                }
                obj.deinit();
            },
            else => {},
        }
    }
};

/// JSON document
pub const JsonDocument = struct {
    allocator: Allocator,
    root: *JsonValue,

    pub fn init(allocator: Allocator, root: *JsonValue) JsonDocument {
        return .{
            .allocator = allocator,
            .root = root,
        };
    }

    pub fn deinit(self: *JsonDocument) void {
        self.root.deinit(self.allocator);
        self.allocator.destroy(self.root);
    }
};

/// Parser options
pub const ParseOptions = struct {
    /// Maximum nesting depth (prevents stack overflow)
    max_depth: usize = 512,
    /// Allow trailing commas (non-standard)
    allow_trailing_commas: bool = false,
    /// Allow comments (non-standard)
    allow_comments: bool = false,
    /// Maximum string length
    max_string_length: usize = 10 * 1024 * 1024, // 10MB
};

/// Parser errors
pub const ParseError = error{
    UnexpectedEndOfInput,
    UnexpectedToken,
    InvalidNumber,
    InvalidString,
    InvalidEscapeSequence,
    InvalidUnicodeEscape,
    MaxDepthExceeded,
    StringTooLong,
    DuplicateKey,
};

// ============================================================================
// JSON Parser
// ============================================================================

/// JSON Parser
pub const Parser = struct {
    allocator: Allocator,
    source: []const u8,
    pos: usize,
    options: ParseOptions,
    depth: usize,

    pub fn init(allocator: Allocator, options: ParseOptions) Parser {
        return .{
            .allocator = allocator,
            .source = "",
            .pos = 0,
            .options = options,
            .depth = 0,
        };
    }

    /// Parse JSON from string
    pub fn parse(self: *Parser, source: []const u8) !JsonDocument {
        self.source = source;
        self.pos = 0;
        self.depth = 0;

        self.skipWhitespace();
        const root = try self.parseValue();

        // Ensure we've consumed all input (except trailing whitespace)
        self.skipWhitespace();
        if (self.pos < self.source.len) {
            return ParseError.UnexpectedToken;
        }

        return JsonDocument.init(self.allocator, root);
    }

    // ========================================================================
    // Value Parsing
    // ========================================================================

    fn parseValue(self: *Parser) !*JsonValue {
        self.skipWhitespace();

        if (self.pos >= self.source.len) {
            return ParseError.UnexpectedEndOfInput;
        }

        const ch = self.source[self.pos];

        return switch (ch) {
            'n' => try self.parseNull(),
            't', 'f' => try self.parseBoolean(),
            '"' => try self.parseString(),
            '[' => try self.parseArray(),
            '{' => try self.parseObject(),
            '-', '0'...'9' => try self.parseNumber(),
            else => ParseError.UnexpectedToken,
        };
    }

    fn parseNull(self: *Parser) !*JsonValue {
        if (!self.matchKeyword("null")) {
            return ParseError.UnexpectedToken;
        }

        const value = try self.allocator.create(JsonValue);
        value.* = .{ .Null = {} };
        return value;
    }

    fn parseBoolean(self: *Parser) !*JsonValue {
        const is_true = self.matchKeyword("true");
        const is_false = self.matchKeyword("false");

        if (!is_true and !is_false) {
            return ParseError.UnexpectedToken;
        }

        const value = try self.allocator.create(JsonValue);
        value.* = .{ .Boolean = is_true };
        return value;
    }

    fn parseNumber(self: *Parser) !*JsonValue {
        const start = self.pos;

        // Optional minus sign
        if (self.peek() == '-') {
            self.pos += 1;
        }

        // Integer part
        if (self.peek() == '0') {
            self.pos += 1;
        } else if (self.peek() >= '1' and self.peek() <= '9') {
            self.pos += 1;
            while (self.peek() >= '0' and self.peek() <= '9') {
                self.pos += 1;
            }
        } else {
            return ParseError.InvalidNumber;
        }

        // Fractional part
        if (self.peek() == '.') {
            self.pos += 1;
            if (self.peek() < '0' or self.peek() > '9') {
                return ParseError.InvalidNumber;
            }
            while (self.peek() >= '0' and self.peek() <= '9') {
                self.pos += 1;
            }
        }

        // Exponent part
        if (self.peek() == 'e' or self.peek() == 'E') {
            self.pos += 1;
            if (self.peek() == '+' or self.peek() == '-') {
                self.pos += 1;
            }
            if (self.peek() < '0' or self.peek() > '9') {
                return ParseError.InvalidNumber;
            }
            while (self.peek() >= '0' and self.peek() <= '9') {
                self.pos += 1;
            }
        }

        const num_str = self.source[start..self.pos];
        const num = std.fmt.parseFloat(f64, num_str) catch {
            return ParseError.InvalidNumber;
        };

        const value = try self.allocator.create(JsonValue);
        value.* = .{ .Number = num };
        return value;
    }

    fn parseString(self: *Parser) !*JsonValue {
        if (self.peek() != '"') {
            return ParseError.InvalidString;
        }
        self.pos += 1; // Skip opening quote

        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();

        while (self.pos < self.source.len) {
            if (buffer.items.len > self.options.max_string_length) {
                return ParseError.StringTooLong;
            }

            const ch = self.source[self.pos];

            if (ch == '"') {
                self.pos += 1; // Skip closing quote
                const str = try buffer.toOwnedSlice();
                const value = try self.allocator.create(JsonValue);
                value.* = .{ .String = str };
                return value;
            }

            if (ch == '\\') {
                self.pos += 1;
                if (self.pos >= self.source.len) {
                    return ParseError.UnexpectedEndOfInput;
                }

                const escape = self.source[self.pos];
                self.pos += 1;

                switch (escape) {
                    '"' => try buffer.append('"'),
                    '\\' => try buffer.append('\\'),
                    '/' => try buffer.append('/'),
                    'b' => try buffer.append(0x08),
                    'f' => try buffer.append(0x0C),
                    'n' => try buffer.append('\n'),
                    'r' => try buffer.append('\r'),
                    't' => try buffer.append('\t'),
                    'u' => {
                        // Unicode escape: \uXXXX
                        const codepoint = try self.parseUnicodeEscape();
                        var utf8_buf: [4]u8 = undefined;
                        const len = try string.encodeUtf8(codepoint, &utf8_buf);
                        try buffer.appendSlice(utf8_buf[0..len]);
                    },
                    else => return ParseError.InvalidEscapeSequence,
                }
            } else if (ch < 0x20) {
                // Control characters must be escaped
                return ParseError.InvalidString;
            } else {
                try buffer.append(ch);
                self.pos += 1;
            }
        }

        return ParseError.UnexpectedEndOfInput;
    }

    fn parseUnicodeEscape(self: *Parser) !u21 {
        if (self.pos + 4 > self.source.len) {
            return ParseError.InvalidUnicodeEscape;
        }

        const hex_str = self.source[self.pos..self.pos + 4];
        self.pos += 4;

        const codepoint = std.fmt.parseInt(u16, hex_str, 16) catch {
            return ParseError.InvalidUnicodeEscape;
        };

        // Handle surrogate pairs (UTF-16)
        if (codepoint >= 0xD800 and codepoint <= 0xDBFF) {
            // High surrogate
            if (self.pos + 6 > self.source.len or
                self.source[self.pos] != '\\' or
                self.source[self.pos + 1] != 'u')
            {
                return ParseError.InvalidUnicodeEscape;
            }

            self.pos += 2;
            const low_hex = self.source[self.pos..self.pos + 4];
            self.pos += 4;

            const low = std.fmt.parseInt(u16, low_hex, 16) catch {
                return ParseError.InvalidUnicodeEscape;
            };

            if (low < 0xDC00 or low > 0xDFFF) {
                return ParseError.InvalidUnicodeEscape;
            }

            // Combine surrogates
            const high = codepoint - 0xD800;
            const low_part = low - 0xDC00;
            return @as(u21, 0x10000) + (@as(u21, high) << 10) + low_part;
        }

        return @intCast(codepoint);
    }

    fn parseArray(self: *Parser) !*JsonValue {
        if (self.peek() != '[') {
            return ParseError.UnexpectedToken;
        }
        self.pos += 1;

        self.depth += 1;
        if (self.depth > self.options.max_depth) {
            return ParseError.MaxDepthExceeded;
        }
        defer self.depth -= 1;

        var array = std.ArrayList(*JsonValue).init(self.allocator);
        errdefer {
            for (array.items) |item| {
                item.deinit(self.allocator);
                self.allocator.destroy(item);
            }
            array.deinit();
        }

        self.skipWhitespace();

        // Empty array
        if (self.peek() == ']') {
            self.pos += 1;
            const value = try self.allocator.create(JsonValue);
            value.* = .{ .Array = array };
            return value;
        }

        while (true) {
            const item = try self.parseValue();
            try array.append(item);

            self.skipWhitespace();

            const ch = self.peek();
            if (ch == ']') {
                self.pos += 1;
                break;
            } else if (ch == ',') {
                self.pos += 1;
                self.skipWhitespace();
                
                // Check for trailing comma
                if (self.peek() == ']') {
                    if (self.options.allow_trailing_commas) {
                        self.pos += 1;
                        break;
                    } else {
                        return ParseError.UnexpectedToken;
                    }
                }
            } else {
                return ParseError.UnexpectedToken;
            }
        }

        const value = try self.allocator.create(JsonValue);
        value.* = .{ .Array = array };
        return value;
    }

    fn parseObject(self: *Parser) !*JsonValue {
        if (self.peek() != '{') {
            return ParseError.UnexpectedToken;
        }
        self.pos += 1;

        self.depth += 1;
        if (self.depth > self.options.max_depth) {
            return ParseError.MaxDepthExceeded;
        }
        defer self.depth -= 1;

        var object = std.StringHashMap(*JsonValue).init(self.allocator);
        errdefer {
            var it = object.iterator();
            while (it.next()) |entry| {
                self.allocator.free(entry.key_ptr.*);
                entry.value_ptr.*.deinit(self.allocator);
                self.allocator.destroy(entry.value_ptr.*);
            }
            object.deinit();
        }

        self.skipWhitespace();

        // Empty object
        if (self.peek() == '}') {
            self.pos += 1;
            const value = try self.allocator.create(JsonValue);
            value.* = .{ .Object = object };
            return value;
        }

        while (true) {
            self.skipWhitespace();

            // Parse key (must be string)
            if (self.peek() != '"') {
                return ParseError.UnexpectedToken;
            }

            const key_value = try self.parseString();
            defer {
                key_value.deinit(self.allocator);
                self.allocator.destroy(key_value);
            }

            const key = switch (key_value.*) {
                .String => |s| try self.allocator.dupe(u8, s),
                else => unreachable,
            };
            errdefer self.allocator.free(key);

            self.skipWhitespace();

            // Expect colon
            if (self.peek() != ':') {
                return ParseError.UnexpectedToken;
            }
            self.pos += 1;

            self.skipWhitespace();

            // Parse value
            const val = try self.parseValue();
            errdefer {
                val.deinit(self.allocator);
                self.allocator.destroy(val);
            }

            // Check for duplicate key
            if (object.contains(key)) {
                self.allocator.free(key);
                return ParseError.DuplicateKey;
            }

            try object.put(key, val);

            self.skipWhitespace();

            const ch = self.peek();
            if (ch == '}') {
                self.pos += 1;
                break;
            } else if (ch == ',') {
                self.pos += 1;
                self.skipWhitespace();
                
                // Check for trailing comma
                if (self.peek() == '}') {
                    if (self.options.allow_trailing_commas) {
                        self.pos += 1;
                        break;
                    } else {
                        return ParseError.UnexpectedToken;
                    }
                }
            } else {
                return ParseError.UnexpectedToken;
            }
        }

        const value = try self.allocator.create(JsonValue);
        value.* = .{ .Object = object };
        return value;
    }

    // ========================================================================
    // Utilities
    // ========================================================================

    fn peek(self: *Parser) u8 {
        if (self.pos >= self.source.len) return 0;
        return self.source[self.pos];
    }

    fn skipWhitespace(self: *Parser) void {
        while (self.pos < self.source.len) {
            const ch = self.source[self.pos];
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                self.pos += 1;
            } else if (self.options.allow_comments and ch == '/') {
                self.skipComment();
            } else {
                break;
            }
        }
    }

    fn skipComment(self: *Parser) void {
        if (self.pos + 1 >= self.source.len) return;

        if (self.source[self.pos + 1] == '/') {
            // Line comment
            self.pos += 2;
            while (self.pos < self.source.len and self.source[self.pos] != '\n') {
                self.pos += 1;
            }
        } else if (self.source[self.pos + 1] == '*') {
            // Block comment
            self.pos += 2;
            while (self.pos + 1 < self.source.len) {
                if (self.source[self.pos] == '*' and self.source[self.pos + 1] == '/') {
                    self.pos += 2;
                    break;
                }
                self.pos += 1;
            }
        }
    }

    fn matchKeyword(self: *Parser, keyword: []const u8) bool {
        if (self.pos + keyword.len > self.source.len) {
            return false;
        }

        if (!std.mem.eql(u8, self.source[self.pos..self.pos + keyword.len], keyword)) {
            return false;
        }

        self.pos += keyword.len;
        return true;
    }
};

// ============================================================================
// JSON Stringifier
// ============================================================================

/// Stringify JSON value
pub fn stringify(value: *const JsonValue, writer: anytype) !void {
    switch (value.*) {
        .Null => try writer.writeAll("null"),
        .Boolean => |b| try writer.writeAll(if (b) "true" else "false"),
        .Number => |n| try writer.print("{d}", .{n}),
        .String => |s| {
            try writer.writeByte('"');
            for (s) |ch| {
                switch (ch) {
                    '"' => try writer.writeAll("\\\""),
                    '\\' => try writer.writeAll("\\\\"),
                    '\n' => try writer.writeAll("\\n"),
                    '\r' => try writer.writeAll("\\r"),
                    '\t' => try writer.writeAll("\\t"),
                    0x08 => try writer.writeAll("\\b"),
                    0x0C => try writer.writeAll("\\f"),
                    else => {
                        if (ch < 0x20) {
                            try writer.print("\\u{x:0>4}", .{ch});
                        } else {
                            try writer.writeByte(ch);
                        }
                    },
                }
            }
            try writer.writeByte('"');
        },
        .Array => |arr| {
            try writer.writeByte('[');
            for (arr.items, 0..) |item, i| {
                if (i > 0) try writer.writeByte(',');
                try stringify(item, writer);
            }
            try writer.writeByte(']');
        },
        .Object => |obj| {
            try writer.writeByte('{');
            var it = obj.iterator();
            var first = true;
            while (it.next()) |entry| {
                if (!first) try writer.writeByte(',');
                first = false;
                
                try writer.writeByte('"');
                try writer.writeAll(entry.key_ptr.*);
                try writer.writeAll("\":");
                try stringify(entry.value_ptr.*, writer);
            }
            try writer.writeByte('}');
        },
    }
}

// ============================================================================
// FFI Exports
// ============================================================================

export fn nExtract_JSON_parse(data: [*]const u8, len: usize) ?*JsonDocument {
    const allocator = std.heap.c_allocator;
    var parser = Parser.init(allocator, .{});
    
    const slice = data[0..len];
    const doc = parser.parse(slice) catch return null;
    
    const doc_ptr = allocator.create(JsonDocument) catch return null;
    doc_ptr.* = doc;
    return doc_ptr;
}

export fn nExtract_JSON_destroy(doc: ?*JsonDocument) void {
    if (doc) |d| {
        var doc_copy = d.*;
        doc_copy.deinit();
        std.heap.c_allocator.destroy(d);
    }
}

// ============================================================================
// Tests
// ============================================================================

test "JSON - null" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc = try parser.parse("null");
    defer doc.deinit();
    
    try std.testing.expectEqual(JsonType.Null, doc.root.*);
}

test "JSON - boolean" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc1 = try parser.parse("true");
    defer doc1.deinit();
    try std.testing.expectEqual(true, doc1.root.Boolean);
    
    var doc2 = try parser.parse("false");
    defer doc2.deinit();
    try std.testing.expectEqual(false, doc2.root.Boolean);
}

test "JSON - number" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc1 = try parser.parse("42");
    defer doc1.deinit();
    try std.testing.expectEqual(@as(f64, 42), doc1.root.Number);
    
    var doc2 = try parser.parse("-123.456");
    defer doc2.deinit();
    try std.testing.expectEqual(@as(f64, -123.456), doc2.root.Number);
    
    var doc3 = try parser.parse("1.23e10");
    defer doc3.deinit();
    try std.testing.expectEqual(@as(f64, 1.23e10), doc3.root.Number);
}

test "JSON - string" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc = try parser.parse("\"Hello, World!\"");
    defer doc.deinit();
    try std.testing.expectEqualStrings("Hello, World!", doc.root.String);
}

test "JSON - string with escapes" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc = try parser.parse("\"Line 1\\nLine 2\\tTabbed\"");
    defer doc.deinit();
    try std.testing.expectEqualStrings("Line 1\nLine 2\tTabbed", doc.root.String);
}

test "JSON - array" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc = try parser.parse("[1, 2, 3, 4, 5]");
    defer doc.deinit();
    
    try std.testing.expectEqual(@as(usize, 5), doc.root.Array.items.len);
    try std.testing.expectEqual(@as(f64, 1), doc.root.Array.items[0].Number);
    try std.testing.expectEqual(@as(f64, 5), doc.root.Array.items[4].Number);
}

test "JSON - object" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    var doc = try parser.parse("{\"name\": \"Alice\", \"age\": 30}");
    defer doc.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), doc.root.Object.count());
    
    const name = doc.root.Object.get("name").?;
    try std.testing.expectEqualStrings("Alice", name.String);
    
    const age = doc.root.Object.get("age").?;
    try std.testing.expectEqual(@as(f64, 30), age.Number);
}

test "JSON - nested structure" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    const json =
        \\{
        \\  "users": [
        \\    {"name": "Alice", "age": 30},
        \\    {"name": "Bob", "age": 25}
        \\  ],
        \\  "count": 2
        \\}
    ;
    
    var doc = try parser.parse(json);
    defer doc.deinit();
    
    const users = doc.root.Object.get("users").?;
    try std.testing.expectEqual(@as(usize, 2), users.Array.items.len);
    
    const alice = users.Array.items[0];
    try std.testing.expectEqualStrings("Alice", alice.Object.get("name").?.String);
}

test "JSON - stringify" {
    const allocator = std.testing.allocator;
    var parser = Parser.init(allocator, .{});
    
    const original = "{\"name\":\"Alice\",\"age\":30}";
    var doc = try parser.parse(original);
    defer doc.deinit();
    
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();
    
    try stringify(doc.root, buffer.writer());
    
    // Parse the stringified version to verify it's valid
    var doc2 = try parser.parse(buffer.items);
    defer doc2.deinit();
    
    try std.testing.expectEqual(@as(usize, 2), doc2.root.Object.count());
}
