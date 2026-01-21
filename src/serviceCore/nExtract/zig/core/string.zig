// String and text utilities for nExtract
// Provides UTF-8 validation, string building, Unicode normalization, and text processing

const std = @import("std");
const Allocator = std.mem.Allocator;

// ========== UTF-8 Validation and Iteration ==========

/// UTF-8 validation errors
pub const Utf8Error = error{
    InvalidUtf8Sequence,
    OverlongEncoding,
    InvalidCodepoint,
    UnexpectedContinuationByte,
    TruncatedSequence,
};

/// Validate UTF-8 encoded string
pub fn validateUtf8(bytes: []const u8) !void {
    var i: usize = 0;
    while (i < bytes.len) {
        const len = try utf8SequenceLength(bytes[i]);
        if (i + len > bytes.len) {
            return Utf8Error.TruncatedSequence;
        }
        
        // Validate continuation bytes
        var j: usize = 1;
        while (j < len) : (j += 1) {
            if ((bytes[i + j] & 0xC0) != 0x80) {
                return Utf8Error.InvalidUtf8Sequence;
            }
        }
        
        // Check for overlong encodings
        if (len > 1) {
            const codepoint = try decodeUtf8Codepoint(bytes[i .. i + len]);
            if (isOverlongEncoding(codepoint, len)) {
                return Utf8Error.OverlongEncoding;
            }
        }
        
        i += len;
    }
}

/// Get the length of a UTF-8 sequence from the first byte
fn utf8SequenceLength(first_byte: u8) !u8 {
    if ((first_byte & 0x80) == 0) return 1; // 0xxxxxxx
    if ((first_byte & 0xE0) == 0xC0) return 2; // 110xxxxx
    if ((first_byte & 0xF0) == 0xE0) return 3; // 1110xxxx
    if ((first_byte & 0xF8) == 0xF0) return 4; // 11110xxx
    return Utf8Error.InvalidUtf8Sequence;
}

/// Decode a UTF-8 codepoint
fn decodeUtf8Codepoint(bytes: []const u8) !u21 {
    const len = bytes.len;
    if (len == 0) return Utf8Error.TruncatedSequence;
    
    if (len == 1) {
        return @as(u21, bytes[0]);
    } else if (len == 2) {
        const cp = (@as(u21, bytes[0] & 0x1F) << 6) |
                   (@as(u21, bytes[1] & 0x3F));
        return cp;
    } else if (len == 3) {
        const cp = (@as(u21, bytes[0] & 0x0F) << 12) |
                   (@as(u21, bytes[1] & 0x3F) << 6) |
                   (@as(u21, bytes[2] & 0x3F));
        return cp;
    } else if (len == 4) {
        const cp = (@as(u21, bytes[0] & 0x07) << 18) |
                   (@as(u21, bytes[1] & 0x3F) << 12) |
                   (@as(u21, bytes[2] & 0x3F) << 6) |
                   (@as(u21, bytes[3] & 0x3F));
        return cp;
    }
    return Utf8Error.InvalidUtf8Sequence;
}

/// Check if encoding is overlong
fn isOverlongEncoding(codepoint: u21, encoded_len: u8) bool {
    if (encoded_len == 2 and codepoint < 0x80) return true;
    if (encoded_len == 3 and codepoint < 0x800) return true;
    if (encoded_len == 4 and codepoint < 0x10000) return true;
    return false;
}

/// UTF-8 iterator
pub const Utf8Iterator = struct {
    bytes: []const u8,
    index: usize = 0,
    
    pub fn init(bytes: []const u8) Utf8Iterator {
        return .{ .bytes = bytes };
    }
    
    pub fn next(self: *Utf8Iterator) ?u21 {
        if (self.index >= self.bytes.len) return null;
        
        const len = utf8SequenceLength(self.bytes[self.index]) catch return null;
        if (self.index + len > self.bytes.len) return null;
        
        const codepoint = decodeUtf8Codepoint(
            self.bytes[self.index .. self.index + len]
        ) catch return null;
        
        self.index += len;
        return codepoint;
    }
    
    pub fn peek(self: *const Utf8Iterator) ?u21 {
        if (self.index >= self.bytes.len) return null;
        
        const len = utf8SequenceLength(self.bytes[self.index]) catch return null;
        if (self.index + len > self.bytes.len) return null;
        
        return decodeUtf8Codepoint(
            self.bytes[self.index .. self.index + len]
        ) catch null;
    }
};

/// Encode a Unicode codepoint to UTF-8
pub fn encodeUtf8(codepoint: u21, buffer: []u8) !usize {
    if (codepoint < 0x80) {
        if (buffer.len < 1) return error.BufferTooSmall;
        buffer[0] = @intCast(codepoint);
        return 1;
    } else if (codepoint < 0x800) {
        if (buffer.len < 2) return error.BufferTooSmall;
        buffer[0] = @intCast(0xC0 | (codepoint >> 6));
        buffer[1] = @intCast(0x80 | (codepoint & 0x3F));
        return 2;
    } else if (codepoint < 0x10000) {
        if (buffer.len < 3) return error.BufferTooSmall;
        buffer[0] = @intCast(0xE0 | (codepoint >> 12));
        buffer[1] = @intCast(0x80 | ((codepoint >> 6) & 0x3F));
        buffer[2] = @intCast(0x80 | (codepoint & 0x3F));
        return 3;
    } else if (codepoint < 0x110000) {
        if (buffer.len < 4) return error.BufferTooSmall;
        buffer[0] = @intCast(0xF0 | (codepoint >> 18));
        buffer[1] = @intCast(0x80 | ((codepoint >> 12) & 0x3F));
        buffer[2] = @intCast(0x80 | ((codepoint >> 6) & 0x3F));
        buffer[3] = @intCast(0x80 | (codepoint & 0x3F));
        return 4;
    }
    return Utf8Error.InvalidCodepoint;
}

// ========== String Builder with SSO ==========

/// Small String Optimization size (bytes that fit inline)
const SSO_SIZE = 24;

/// Dynamic string builder with Small String Optimization
pub const StringBuilder = struct {
    allocator: Allocator,
    data: union(enum) {
        small: struct {
            buffer: [SSO_SIZE]u8,
            len: u8,
        },
        large: struct {
            buffer: []u8,
            len: usize,
            capacity: usize,
        },
    },
    
    pub fn init(allocator: Allocator) StringBuilder {
        return .{
            .allocator = allocator,
            .data = .{
                .small = .{
                    .buffer = undefined,
                    .len = 0,
                },
            },
        };
    }
    
    pub fn deinit(self: *StringBuilder) void {
        switch (self.data) {
            .large => |large| {
                self.allocator.free(large.buffer);
            },
            .small => {},
        }
    }
    
    pub fn append(self: *StringBuilder, str: []const u8) !void {
        switch (self.data) {
            .small => |*small| {
                const new_len = small.len + str.len;
                if (new_len <= SSO_SIZE) {
                    // Still fits in small buffer
                    @memcpy(small.buffer[small.len..][0..str.len], str);
                    small.len = @intCast(new_len);
                } else {
                    // Need to upgrade to large
                    const capacity = std.math.ceilPowerOfTwo(usize, new_len) catch 
                        return error.OutOfMemory;
                    const new_buffer = try self.allocator.alloc(u8, capacity);
                    
                    // Copy existing data
                    @memcpy(new_buffer[0..small.len], small.buffer[0..small.len]);
                    // Copy new data
                    @memcpy(new_buffer[small.len..][0..str.len], str);
                    
                    self.data = .{
                        .large = .{
                            .buffer = new_buffer,
                            .len = new_len,
                            .capacity = capacity,
                        },
                    };
                }
            },
            .large => |*large| {
                const new_len = large.len + str.len;
                if (new_len > large.capacity) {
                    // Need to grow
                    const new_capacity = std.math.ceilPowerOfTwo(usize, new_len) catch 
                        return error.OutOfMemory;
                    const new_buffer = try self.allocator.realloc(large.buffer, new_capacity);
                    large.buffer = new_buffer;
                    large.capacity = new_capacity;
                }
                
                @memcpy(large.buffer[large.len..][0..str.len], str);
                large.len = new_len;
            },
        }
    }
    
    pub fn appendCodepoint(self: *StringBuilder, codepoint: u21) !void {
        var buffer: [4]u8 = undefined;
        const byte_len = try encodeUtf8(codepoint, &buffer);
        try self.append(buffer[0..byte_len]);
    }
    
    pub fn toOwnedSlice(self: *StringBuilder) ![]u8 {
        switch (self.data) {
            .small => |small| {
                const result = try self.allocator.alloc(u8, small.len);
                @memcpy(result, small.buffer[0..small.len]);
                return result;
            },
            .large => |*large| {
                const result = try self.allocator.realloc(large.buffer, large.len);
                self.data = .{ .small = .{ .buffer = undefined, .len = 0 } };
                return result;
            },
        }
    }
    
    pub fn toSlice(self: *const StringBuilder) []const u8 {
        return switch (self.data) {
            .small => |small| small.buffer[0..small.len],
            .large => |large| large.buffer[0..large.len],
        };
    }
    
    pub fn len(self: *const StringBuilder) usize {
        return switch (self.data) {
            .small => |small| small.len,
            .large => |large| large.len,
        };
    }
    
    pub fn clear(self: *StringBuilder) void {
        switch (self.data) {
            .small => |*small| small.len = 0,
            .large => |*large| large.len = 0,
        }
    }
};

// ========== Unicode Normalization ==========

/// Unicode normalization forms
pub const NormalizationForm = enum {
    NFD,  // Canonical Decomposition
    NFC,  // Canonical Decomposition followed by Canonical Composition
    NFKD, // Compatibility Decomposition
    NFKC, // Compatibility Decomposition followed by Canonical Composition
};

/// Normalize UTF-8 string (simplified implementation)
pub fn normalizeUtf8(
    allocator: Allocator,
    input: []const u8,
    form: NormalizationForm,
) ![]u8 {
    // For now, implement basic NFC normalization
    // Full implementation would require Unicode data tables
    
    var builder = StringBuilder.init(allocator);
    defer builder.deinit();
    
    var iter = Utf8Iterator.init(input);
    while (iter.next()) |codepoint| {
        // Basic normalization (decompose common accented characters)
        const normalized = normalizeCodepoint(codepoint, form);
        if (normalized) |cp| {
            try builder.appendCodepoint(cp);
        } else {
            try builder.appendCodepoint(codepoint);
        }
    }
    
    return builder.toOwnedSlice();
}

/// Normalize a single codepoint (basic implementation)
fn normalizeCodepoint(codepoint: u21, form: NormalizationForm) ?u21 {
    _ = form;
    
    // Basic decomposition for common accented Latin characters
    // Full implementation would use Unicode normalization tables
    return switch (codepoint) {
        0xE0...0xE5 => 'a', // à, á, â, ã, ä, å → a
        0xE8...0xEB => 'e', // è, é, ê, ë → e
        0xEC...0xEF => 'i', // ì, í, î, ï → i
        0xF2...0xF6 => 'o', // ò, ó, ô, õ, ö → o
        0xF9...0xFC => 'u', // ù, ú, û, ü → u
        0xC0...0xC5 => 'A', // À, Á, Â, Ã, Ä, Å → A
        0xC8...0xCB => 'E', // È, É, Ê, Ë → E
        0xCC...0xCF => 'I', // Ì, Í, Î, Ï → I
        0xD2...0xD6 => 'O', // Ò, Ó, Ô, Õ, Ö → O
        0xD9...0xDC => 'U', // Ù, Ú, Û, Ü → U
        else => null,
    };
}

// ========== Grapheme Cluster Iteration ==========

/// Grapheme cluster iterator (simplified)
pub const GraphemeIterator = struct {
    bytes: []const u8,
    index: usize = 0,
    
    pub fn init(bytes: []const u8) GraphemeIterator {
        return .{ .bytes = bytes };
    }
    
    pub fn next(self: *GraphemeIterator) ?[]const u8 {
        if (self.index >= self.bytes.len) return null;
        
        const start = self.index;
        const len = utf8SequenceLength(self.bytes[self.index]) catch return null;
        
        // Basic implementation: one codepoint = one grapheme
        // Full implementation would handle combining marks
        self.index += len;
        
        if (self.index <= self.bytes.len) {
            return self.bytes[start..self.index];
        }
        return null;
    }
};

// ========== Case Folding ==========

/// Convert to uppercase (ASCII only for now)
pub fn toUpper(allocator: Allocator, input: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, input.len);
    for (input, 0..) |byte, i| {
        result[i] = if (byte >= 'a' and byte <= 'z')
            byte - 32
        else
            byte;
    }
    return result;
}

/// Convert to lowercase (ASCII only for now)
pub fn toLower(allocator: Allocator, input: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, input.len);
    for (input, 0..) |byte, i| {
        result[i] = if (byte >= 'A' and byte <= 'Z')
            byte + 32
        else
            byte;
    }
    return result;
}

/// Convert to title case (ASCII only for now)
pub fn toTitle(allocator: Allocator, input: []const u8) ![]u8 {
    const result = try allocator.alloc(u8, input.len);
    var capitalize_next = true;
    
    for (input, 0..) |byte, i| {
        if (capitalize_next and byte >= 'a' and byte <= 'z') {
            result[i] = byte - 32;
            capitalize_next = false;
        } else if (!capitalize_next and byte >= 'A' and byte <= 'Z') {
            result[i] = byte + 32;
        } else {
            result[i] = byte;
        }
        
        // Capitalize after space or punctuation
        if (byte == ' ' or byte == '\t' or byte == '\n' or 
            byte == '.' or byte == '!' or byte == '?') {
            capitalize_next = true;
        }
    }
    return result;
}

// ========== String Searching ==========

/// Boyer-Moore string search algorithm
pub fn boyerMooreSearch(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;
    
    // Build bad character table
    var bad_char: [256]isize = undefined;
    for (&bad_char) |*entry| {
        entry.* = @intCast(needle.len);
    }
    for (needle, 0..) |char, i| {
        bad_char[char] = @intCast(needle.len - 1 - i);
    }
    
    // Search
    var pos: usize = 0;
    while (pos + needle.len <= haystack.len) {
        var i: isize = @intCast(needle.len - 1);
        while (i >= 0 and needle[@intCast(i)] == haystack[pos + @as(usize, @intCast(i))]) {
            i -= 1;
        }
        
        if (i < 0) {
            return pos;
        }
        
        const bad_char_skip = bad_char[haystack[pos + @as(usize, @intCast(i))]];
        pos += @intCast(@max(1, bad_char_skip));
    }
    
    return null;
}

/// KMP (Knuth-Morris-Pratt) string search algorithm
pub fn kmpSearch(allocator: Allocator, haystack: []const u8, needle: []const u8) !?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;
    
    // Build failure function
    const failure = try buildKmpFailureFunction(allocator, needle);
    defer allocator.free(failure);
    
    // Search
    var i: usize = 0;
    var j: usize = 0;
    
    while (i < haystack.len) {
        if (haystack[i] == needle[j]) {
            i += 1;
            j += 1;
            if (j == needle.len) {
                return i - j;
            }
        } else if (j > 0) {
            j = failure[j - 1];
        } else {
            i += 1;
        }
    }
    
    return null;
}

fn buildKmpFailureFunction(allocator: Allocator, pattern: []const u8) ![]usize {
    const failure = try allocator.alloc(usize, pattern.len);
    failure[0] = 0;
    
    var j: usize = 0;
    var i: usize = 1;
    
    while (i < pattern.len) {
        if (pattern[i] == pattern[j]) {
            j += 1;
            failure[i] = j;
            i += 1;
        } else if (j > 0) {
            j = failure[j - 1];
        } else {
            failure[i] = 0;
            i += 1;
        }
    }
    
    return failure;
}

/// Simple substring search (for comparison)
pub fn simpleSearch(haystack: []const u8, needle: []const u8) ?usize {
    if (needle.len == 0) return 0;
    if (needle.len > haystack.len) return null;
    
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        if (std.mem.eql(u8, haystack[i .. i + needle.len], needle)) {
            return i;
        }
    }
    return null;
}

// ========== Whitespace Processing ==========

/// Check if character is whitespace
pub fn isWhitespace(codepoint: u21) bool {
    return switch (codepoint) {
        ' ', '\t', '\n', '\r', 0x0B, 0x0C => true,
        0xA0 => true, // Non-breaking space
        0x1680 => true, // Ogham space mark
        0x2000...0x200A => true, // Various spaces
        0x2028 => true, // Line separator
        0x2029 => true, // Paragraph separator
        0x202F => true, // Narrow no-break space
        0x205F => true, // Medium mathematical space
        0x3000 => true, // Ideographic space
        else => false,
    };
}

/// Trim whitespace from both ends
pub fn trim(allocator: Allocator, input: []const u8) ![]u8 {
    var start: usize = 0;
    var end: usize = input.len;
    
    // Trim from start
    var iter = Utf8Iterator.init(input);
    while (iter.next()) |cp| {
        if (!isWhitespace(cp)) break;
        start = iter.index;
    }
    
    // Trim from end (reverse iteration)
    while (end > start) {
        const len = utf8SequenceLength(input[end - 1]) catch break;
        if (end < len) break;
        
        const cp = decodeUtf8Codepoint(input[end - len .. end]) catch break;
        if (!isWhitespace(cp)) break;
        end -= len;
    }
    
    return allocator.dupe(u8, input[start..end]);
}

/// Trim whitespace from start only
pub fn trimLeft(allocator: Allocator, input: []const u8) ![]u8 {
    var start: usize = 0;
    
    var iter = Utf8Iterator.init(input);
    while (iter.next()) |cp| {
        if (!isWhitespace(cp)) break;
        start = iter.index;
    }
    
    return allocator.dupe(u8, input[start..]);
}

/// Trim whitespace from end only
pub fn trimRight(allocator: Allocator, input: []const u8) ![]u8 {
    var end: usize = input.len;
    
    while (end > 0) {
        const len = utf8SequenceLength(input[end - 1]) catch break;
        if (end < len) break;
        
        const cp = decodeUtf8Codepoint(input[end - len .. end]) catch break;
        if (!isWhitespace(cp)) break;
        end -= len;
    }
    
    return allocator.dupe(u8, input[0..end]);
}

/// Normalize whitespace (collapse multiple spaces to single space)
pub fn normalizeWhitespace(allocator: Allocator, input: []const u8) ![]u8 {
    var builder = StringBuilder.init(allocator);
    defer builder.deinit();
    
    var prev_was_space = false;
    var iter = Utf8Iterator.init(input);
    
    while (iter.next()) |cp| {
        const is_space = isWhitespace(cp);
        
        if (is_space) {
            if (!prev_was_space) {
                try builder.append(" ");
                prev_was_space = true;
            }
        } else {
            try builder.appendCodepoint(cp);
            prev_was_space = false;
        }
    }
    
    return builder.toOwnedSlice();
}

// ========== String Utilities ==========

/// Split string by delimiter
pub fn split(
    allocator: Allocator,
    input: []const u8,
    delimiter: []const u8,
) ![][]const u8 {
    var result = std.ArrayList([]const u8).init(allocator);
    errdefer result.deinit();
    
    var start: usize = 0;
    var i: usize = 0;
    
    while (i + delimiter.len <= input.len) {
        if (std.mem.eql(u8, input[i .. i + delimiter.len], delimiter)) {
            try result.append(input[start..i]);
            i += delimiter.len;
            start = i;
        } else {
            i += 1;
        }
    }
    
    // Add remaining part
    if (start <= input.len) {
        try result.append(input[start..]);
    }
    
    return result.toOwnedSlice();
}

/// Join strings with delimiter
pub fn join(
    allocator: Allocator,
    strings: []const []const u8,
    delimiter: []const u8,
) ![]u8 {
    if (strings.len == 0) return try allocator.dupe(u8, "");
    
    // Calculate total length
    var total_len: usize = 0;
    for (strings) |str| {
        total_len += str.len;
    }
    total_len += delimiter.len * (strings.len - 1);
    
    // Build result
    var result = try allocator.alloc(u8, total_len);
    var pos: usize = 0;
    
    for (strings, 0..) |str, i| {
        @memcpy(result[pos..][0..str.len], str);
        pos += str.len;
        
        if (i < strings.len - 1) {
            @memcpy(result[pos..][0..delimiter.len], delimiter);
            pos += delimiter.len;
        }
    }
    
    return result;
}

/// Replace all occurrences of pattern with replacement
pub fn replaceAll(
    allocator: Allocator,
    input: []const u8,
    pattern: []const u8,
    replacement: []const u8,
) ![]u8 {
    if (pattern.len == 0) return try allocator.dupe(u8, input);
    
    var builder = StringBuilder.init(allocator);
    defer builder.deinit();
    
    var i: usize = 0;
    while (i < input.len) {
        if (i + pattern.len <= input.len and 
            std.mem.eql(u8, input[i .. i + pattern.len], pattern)) {
            try builder.append(replacement);
            i += pattern.len;
        } else {
            try builder.append(input[i .. i + 1]);
            i += 1;
        }
    }
    
    return builder.toOwnedSlice();
}

/// Check if string starts with prefix
pub fn startsWith(input: []const u8, prefix: []const u8) bool {
    if (prefix.len > input.len) return false;
    return std.mem.eql(u8, input[0..prefix.len], prefix);
}

/// Check if string ends with suffix
pub fn endsWith(input: []const u8, suffix: []const u8) bool {
    if (suffix.len > input.len) return false;
    return std.mem.eql(u8, input[input.len - suffix.len ..], suffix);
}

/// Count occurrences of substring
pub fn count(input: []const u8, pattern: []const u8) usize {
    if (pattern.len == 0) return 0;
    
    var result: usize = 0;
    var i: usize = 0;
    
    while (i + pattern.len <= input.len) {
        if (std.mem.eql(u8, input[i .. i + pattern.len], pattern)) {
            result += 1;
            i += pattern.len;
        } else {
            i += 1;
        }
    }
    
    return result;
}

// ========== FFI Exports ==========

/// Create a string builder
export fn nExtract_StringBuilder_create() ?*StringBuilder {
    const allocator = std.heap.c_allocator;
    const builder = allocator.create(StringBuilder) catch return null;
    builder.* = StringBuilder.init(allocator);
    return builder;
}

/// Destroy a string builder
export fn nExtract_StringBuilder_destroy(builder: ?*StringBuilder) void {
    if (builder) |b| {
        b.deinit();
        b.allocator.destroy(b);
    }
}

/// Append to string builder
export fn nExtract_StringBuilder_append(
    builder: ?*StringBuilder,
    data: [*]const u8,
    len: usize,
) bool {
    if (builder) |b| {
        b.append(data[0..len]) catch return false;
        return true;
    }
    return false;
}

/// Get string from builder
export fn nExtract_StringBuilder_toSlice(
    builder: ?*StringBuilder,
    out_len: *usize,
) ?[*]const u8 {
    if (builder) |b| {
        const slice = b.toSlice();
        out_len.* = slice.len;
        return slice.ptr;
    }
    return null;
}

/// Validate UTF-8 string
export fn nExtract_validateUtf8(data: [*]const u8, len: usize) bool {
    validateUtf8(data[0..len]) catch return false;
    return true;
}

/// Convert to uppercase
export fn nExtract_toUpper(
    data: [*]const u8,
    len: usize,
    out_len: *usize,
) ?[*]u8 {
    const allocator = std.heap.c_allocator;
    const result = toUpper(allocator, data[0..len]) catch return null;
    out_len.* = result.len;
    return result.ptr;
}

/// Convert to lowercase
export fn nExtract_toLower(
    data: [*]const u8,
    len: usize,
    out_len: *usize,
) ?[*]u8 {
    const allocator = std.heap.c_allocator;
    const result = toLower(allocator, data[0..len]) catch return null;
    out_len.* = result.len;
    return result.ptr;
}

/// Free string allocated by nExtract
export fn nExtract_freeString(data: ?[*]u8, len: usize) void {
    if (data) |ptr| {
        const allocator = std.heap.c_allocator;
        allocator.free(ptr[0..len]);
    }
}

// ========== Tests ==========

test "UTF-8 validation" {
    // Valid UTF-8
    try validateUtf8("Hello, world!");
    try validateUtf8("Привет мир"); // Russian
    try validateUtf8("你好世界"); // Chinese
    try validateUtf8("مرحبا"); // Arabic
    
    // Invalid UTF-8
    try std.testing.expectError(Utf8Error.InvalidUtf8Sequence, validateUtf8(&[_]u8{0xFF}));
    try std.testing.expectError(Utf8Error.TruncatedSequence, validateUtf8(&[_]u8{0xC2}));
}

test "UTF-8 iteration" {
    const text = "Hello 世界";
    var iter = Utf8Iterator.init(text);
    
    try std.testing.expectEqual(@as(?u21, 'H'), iter.next());
    try std.testing.expectEqual(@as(?u21, 'e'), iter.next());
    try std.testing.expectEqual(@as(?u21, 'l'), iter.next());
    try std.testing.expectEqual(@as(?u21, 'l'), iter.next());
    try std.testing.expectEqual(@as(?u21, 'o'), iter.next());
    try std.testing.expectEqual(@as(?u21, ' '), iter.next());
    try std.testing.expectEqual(@as(?u21, 0x4E16), iter.next()); // 世
    try std.testing.expectEqual(@as(?u21, 0x754C), iter.next()); // 界
    try std.testing.expectEqual(@as(?u21, null), iter.next());
}

test "UTF-8 encoding" {
    var buffer: [4]u8 = undefined;
    
    // ASCII
    const len1 = try encodeUtf8('A', &buffer);
    try std.testing.expectEqual(@as(usize, 1), len1);
    try std.testing.expectEqual(@as(u8, 'A'), buffer[0]);
    
    // 2-byte
    const len2 = try encodeUtf8(0xE9, &buffer); // é
    try std.testing.expectEqual(@as(usize, 2), len2);
    
    // 3-byte
    const len3 = try encodeUtf8(0x4E16, &buffer); // 世
    try std.testing.expectEqual(@as(usize, 3), len3);
    
    // 4-byte
    const len4 = try encodeUtf8(0x1F600, &buffer); // 😀
    try std.testing.expectEqual(@as(usize, 4), len4);
}

test "StringBuilder SSO" {
    const allocator = std.testing.allocator;
    var builder = StringBuilder.init(allocator);
    defer builder.deinit();
    
    // Small strings (should use SSO)
    try builder.append("Hello");
    try std.testing.expectEqual(@as(usize, 5), builder.len());
    try std.testing.expectEqualStrings("Hello", builder.toSlice());
    
    try builder.append(" World");
    try std.testing.expectEqual(@as(usize, 11), builder.len());
    try std.testing.expectEqualStrings("Hello World", builder.toSlice());
}

test "StringBuilder large strings" {
    const allocator = std.testing.allocator;
    var builder = StringBuilder.init(allocator);
    defer builder.deinit();
    
    // Force upgrade to large storage
    var i: usize = 0;
    while (i < 100) : (i += 1) {
        try builder.append("test ");
    }
    
    try std.testing.expect(builder.len() > SSO_SIZE);
}

test "Case conversion" {
    const allocator = std.testing.allocator;
    
    // Upper
    const upper = try toUpper(allocator, "hello world");
    defer allocator.free(upper);
    try std.testing.expectEqualStrings("HELLO WORLD", upper);
    
    // Lower
    const lower = try toLower(allocator, "HELLO WORLD");
    defer allocator.free(lower);
    try std.testing.expectEqualStrings("hello world", lower);
    
    // Title
    const title = try toTitle(allocator, "hello world. how are you?");
    defer allocator.free(title);
    try std.testing.expectEqualStrings("Hello World. How Are You?", title);
}

test "Boyer-Moore search" {
    const haystack = "The quick brown fox jumps over the lazy dog";
    
    try std.testing.expectEqual(@as(?usize, 16), boyerMooreSearch(haystack, "fox"));
    try std.testing.expectEqual(@as(?usize, 0), boyerMooreSearch(haystack, "The"));
    try std.testing.expectEqual(@as(?usize, 40), boyerMooreSearch(haystack, "dog"));
    try std.testing.expectEqual(@as(?usize, null), boyerMooreSearch(haystack, "cat"));
}

test "KMP search" {
    const allocator = std.testing.allocator;
    const haystack = "ABABDABACDABABCABAB";
    
    const pos1 = try kmpSearch(allocator, haystack, "ABABCABAB");
    try std.testing.expectEqual(@as(?usize, 10), pos1);
    
    const pos2 = try kmpSearch(allocator, haystack, "ABAB");
    try std.testing.expectEqual(@as(?usize, 0), pos2);
    
    const pos3 = try kmpSearch(allocator, haystack, "XYZ");
    try std.testing.expectEqual(@as(?usize, null), pos3);
}

test "Whitespace trimming" {
    const allocator = std.testing.allocator;
    
    const trimmed = try trim(allocator, "  hello world  ");
    defer allocator.free(trimmed);
    try std.testing.expectEqualStrings("hello world", trimmed);
    
    const left = try trimLeft(allocator, "  hello");
    defer allocator.free(left);
    try std.testing.expectEqualStrings("hello", left);
    
    const right = try trimRight(allocator, "hello  ");
    defer allocator.free(right);
    try std.testing.expectEqualStrings("hello", right);
}

test "Whitespace normalization" {
    const allocator = std.testing.allocator;
    
    const normalized = try normalizeWhitespace(allocator, "hello   world  \t  foo");
    defer allocator.free(normalized);
    try std.testing.expectEqualStrings("hello world foo", normalized);
}

test "String split" {
    const allocator = std.testing.allocator;
    
    const parts = try split(allocator, "a,b,c,d", ",");
    defer {
        allocator.free(parts);
    }
    
    try std.testing.expectEqual(@as(usize, 4), parts.len);
    try std.testing.expectEqualStrings("a", parts[0]);
    try std.testing.expectEqualStrings("b", parts[1]);
    try std.testing.expectEqualStrings("c", parts[2]);
    try std.testing.expectEqualStrings("d", parts[3]);
}

test "String join" {
    const allocator = std.testing.allocator;
    
    const strings = [_][]const u8{ "a", "b", "c", "d" };
    const joined = try join(allocator, &strings, ", ");
    defer allocator.free(joined);
    
    try std.testing.expectEqualStrings("a, b, c, d", joined);
}

test "String replace" {
    const allocator = std.testing.allocator;
    
    const replaced = try replaceAll(allocator, "hello world, hello universe", "hello", "hi");
    defer allocator.free(replaced);
    
    try std.testing.expectEqualStrings("hi world, hi universe", replaced);
}

test "String starts/ends with" {
    try std.testing.expect(startsWith("hello world", "hello"));
    try std.testing.expect(!startsWith("hello world", "world"));
    
    try std.testing.expect(endsWith("hello world", "world"));
    try std.testing.expect(!endsWith("hello world", "hello"));
}

test "Count occurrences" {
    try std.testing.expectEqual(@as(usize, 2), count("hello hello world", "hello"));
    try std.testing.expectEqual(@as(usize, 3), count("aaa", "a"));
    try std.testing.expectEqual(@as(usize, 0), count("hello", "xyz"));
}
