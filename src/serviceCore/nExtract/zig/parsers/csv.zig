//! CSV Parser - RFC 4180 Compliant (Pure Zig)
//! 
//! This module provides:
//! - RFC 4180 full compliance
//! - Streaming parser for large files (O(1) memory)
//! - Delimiter auto-detection (comma, tab, semicolon, pipe)
//! - Encoding detection (UTF-8, UTF-16, Latin1, ASCII)
//! - Quoted fields with escape sequences
//! - Multi-line field support
//! - Configurable options
//!
//! Day 6: CSV Parser Implementation
//! Author: nExtract Team
//! Date: January 17, 2026

const std = @import("std");
const types = @import("../core/types.zig");
const string = @import("../core/string.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// Data Structures
// ============================================================================

/// CSV Document structure
pub const CsvDocument = struct {
    allocator: Allocator,
    rows: std.ArrayList(Row),
    headers: ?Row,
    delimiter: u8,
    encoding: Encoding,
    quote_char: u8,
    row_count: usize,
    column_count: usize,

    pub fn init(allocator: Allocator) CsvDocument {
        return .{
            .allocator = allocator,
            .rows = std.ArrayList(Row).init(allocator),
            .headers = null,
            .delimiter = ',',
            .encoding = .utf8,
            .quote_char = '"',
            .row_count = 0,
            .column_count = 0,
        };
    }

    pub fn deinit(self: *CsvDocument) void {
        if (self.headers) |*headers| {
            headers.deinit();
        }
        for (self.rows.items) |*row| {
            row.deinit();
        }
        self.rows.deinit();
    }

    pub fn rowCount(self: *const CsvDocument) usize {
        return self.row_count;
    }

    pub fn columnCount(self: *const CsvDocument) usize {
        return self.column_count;
    }
    
    pub fn getCell(self: *const CsvDocument, row: usize, col: usize) ?[]const u8 {
        if (row >= self.rows.items.len) return null;
        if (col >= self.rows.items[row].fields.items.len) return null;
        return self.rows.items[row].fields.items[col];
    }
    
    pub fn getHeader(self: *const CsvDocument, col: usize) ?[]const u8 {
        if (self.headers) |headers| {
            if (col >= headers.fields.items.len) return null;
            return headers.fields.items[col];
        }
        return null;
    }
};

/// Row in CSV document
pub const Row = struct {
    allocator: Allocator,
    fields: std.ArrayList([]const u8),

    pub fn init(allocator: Allocator) Row {
        return .{
            .allocator = allocator,
            .fields = std.ArrayList([]const u8).init(allocator),
        };
    }

    pub fn deinit(self: *Row) void {
        for (self.fields.items) |field| {
            self.allocator.free(field);
        }
        self.fields.deinit();
    }

    pub fn addField(self: *Row, field: []const u8) !void {
        const owned = try self.allocator.dupe(u8, field);
        try self.fields.append(owned);
    }
    
    pub fn fieldCount(self: *const Row) usize {
        return self.fields.items.len;
    }
};

/// Encoding types
pub const Encoding = enum {
    utf8,
    utf16_le,
    utf16_be,
    latin1,
    ascii,
    
    pub fn toString(self: Encoding) []const u8 {
        return switch (self) {
            .utf8 => "UTF-8",
            .utf16_le => "UTF-16 LE",
            .utf16_be => "UTF-16 BE",
            .latin1 => "Latin-1",
            .ascii => "ASCII",
        };
    }
};

/// Parser options
pub const ParseOptions = struct {
    delimiter: ?u8 = null, // Auto-detect if null
    quote_char: u8 = '"',
    escape_char: ?u8 = null, // Use quote_char if null (RFC 4180 standard)
    has_headers: bool = true,
    skip_empty_lines: bool = true,
    trim_fields: bool = false,
    max_field_size: usize = 10 * 1024 * 1024, // 10MB per field
    max_row_size: usize = 100 * 1024 * 1024, // 100MB per row
    strict_mode: bool = false, // Strict RFC 4180 compliance
};

/// Parser errors
pub const ParseError = error{
    InvalidQuotedField,
    UnclosedQuotedField,
    FieldTooLarge,
    RowTooLarge,
    InvalidEncoding,
    InvalidDelimiter,
};

// ============================================================================
// CSV Parser
// ============================================================================

/// CSV Parser (non-streaming)
pub const Parser = struct {
    allocator: Allocator,
    options: ParseOptions,

    pub fn init(allocator: Allocator, options: ParseOptions) Parser {
        return .{
            .allocator = allocator,
            .options = options,
        };
    }

    /// Parse CSV data
    pub fn parse(self: *Parser, data: []const u8) !CsvDocument {
        var doc = CsvDocument.init(self.allocator);
        errdefer doc.deinit();

        // Detect encoding
        const encoding_result = try detectEncodingWithBOM(data);
        doc.encoding = encoding_result.encoding;
        
        // Skip BOM if present
        const data_without_bom = data[encoding_result.bom_size..];
        
        // Convert to UTF-8 if needed
        const utf8_data = if (encoding_result.encoding == .utf8)
            data_without_bom
        else
            try convertToUtf8(self.allocator, data_without_bom, encoding_result.encoding);
        defer if (encoding_result.encoding != .utf8) self.allocator.free(utf8_data);

        // Detect delimiter if not provided
        const delimiter = self.options.delimiter orelse try detectDelimiter(utf8_data);
        doc.delimiter = delimiter;
        doc.quote_char = self.options.quote_char;

        // Parse line by line
        var line_iter = LineIterator.init(utf8_data);
        var line_num: usize = 0;
        var max_columns: usize = 0;

        while (try line_iter.next()) |line| {
            line_num += 1;

            // Skip empty lines if configured
            if (self.options.skip_empty_lines and isEmptyLine(line)) {
                continue;
            }

            // Parse row
            var row = try self.parseRow(line, delimiter);
            errdefer row.deinit();
            
            // Track maximum columns
            if (row.fieldCount() > max_columns) {
                max_columns = row.fieldCount();
            }

            // First non-empty row might be headers
            if (self.options.has_headers and doc.headers == null and doc.rows.items.len == 0) {
                doc.headers = row;
            } else {
                try doc.rows.append(row);
            }
        }

        doc.row_count = doc.rows.items.len;
        doc.column_count = max_columns;

        return doc;
    }

    /// Parse a single row (RFC 4180 compliant)
    fn parseRow(self: *Parser, line: []const u8, delimiter: u8) !Row {
        var row = Row.init(self.allocator);
        errdefer row.deinit();

        var in_quotes = false;
        var field_buffer = std.ArrayList(u8).init(self.allocator);
        defer field_buffer.deinit();
        
        var field_size: usize = 0;

        var i: usize = 0;
        while (i < line.len) : (i += 1) {
            const ch = line[i];
            
            // Check field size limit
            field_size += 1;
            if (field_size > self.options.max_field_size) {
                return ParseError.FieldTooLarge;
            }

            if (ch == self.options.quote_char) {
                if (in_quotes) {
                    // Check for escaped quote (double quote - RFC 4180)
                    if (i + 1 < line.len and line[i + 1] == self.options.quote_char) {
                        try field_buffer.append(self.options.quote_char);
                        i += 1; // Skip next quote
                    } else {
                        // End of quoted field
                        in_quotes = false;
                    }
                } else if (field_buffer.items.len == 0) {
                    // Start of quoted field (must be at field start for RFC 4180)
                    in_quotes = true;
                } else if (self.options.strict_mode) {
                    // In strict mode, quotes must be at field start
                    return ParseError.InvalidQuotedField;
                } else {
                    // In non-strict mode, treat as regular character
                    try field_buffer.append(ch);
                }
            } else if (ch == delimiter and !in_quotes) {
                // End of field
                const field = try self.finalizeField(field_buffer.items);
                try row.addField(field);
                field_buffer.clearRetainingCapacity();
                field_size = 0;
            } else {
                // Regular character
                try field_buffer.append(ch);
            }
        }
        
        // Check for unclosed quotes
        if (in_quotes and self.options.strict_mode) {
            return ParseError.UnclosedQuotedField;
        }

        // Add last field
        const field = try self.finalizeField(field_buffer.items);
        try row.addField(field);

        return row;
    }

    /// Finalize field (trim if configured)
    fn finalizeField(self: *Parser, field: []const u8) ![]const u8 {
        if (self.options.trim_fields) {
            return std.mem.trim(u8, field, &std.ascii.whitespace);
        }
        return field;
    }
};

// ============================================================================
// Streaming Parser
// ============================================================================

/// Streaming CSV parser for large files (O(1) memory)
pub const StreamingParser = struct {
    allocator: Allocator,
    options: ParseOptions,
    delimiter: u8,
    encoding: Encoding,
    
    pub fn init(allocator: Allocator, options: ParseOptions) StreamingParser {
        return .{
            .allocator = allocator,
            .options = options,
            .delimiter = options.delimiter orelse ',',
            .encoding = .utf8,
        };
    }
    
    /// Parse CSV with callback for each row
    pub fn parseWithCallback(
        self: *StreamingParser,
        reader: anytype,
        callback: *const fn(row: *Row, context: ?*anyopaque) anyerror!void,
        context: ?*anyopaque,
    ) !void {
        var buffer = std.ArrayList(u8).init(self.allocator);
        defer buffer.deinit();
        
        // Read first chunk to detect encoding and delimiter
        const initial_size = @min(4096, try reader.read(buffer.addManyAsSlice(4096)));
        buffer.items.len = initial_size;
        
        const encoding_result = try detectEncodingWithBOM(buffer.items);
        self.encoding = encoding_result.encoding;
        
        // Skip BOM
        const start_pos = encoding_result.bom_size;
        
        // Detect delimiter if not provided
        if (self.options.delimiter == null) {
            self.delimiter = try detectDelimiter(buffer.items[start_pos..]);
        }
        
        var line_buffer = std.ArrayList(u8).init(self.allocator);
        defer line_buffer.deinit();
        
        var parser = Parser.init(self.allocator, self.options);
        var pos = start_pos;
        var is_first_row = true;
        
        while (true) {
            // Find next line
            const line_end = std.mem.indexOfScalarPos(u8, buffer.items, pos, '\n');
            
            if (line_end) |end| {
                // Found complete line
                const line = buffer.items[pos..end];
                const trimmed = if (line.len > 0 and line[line.len - 1] == '\r')
                    line[0..line.len - 1]
                else
                    line;
                
                // Skip header if configured
                if (is_first_row and self.options.has_headers) {
                    is_first_row = false;
                    pos = end + 1;
                    continue;
                }
                
                // Parse and callback
                var row = try parser.parseRow(trimmed, self.delimiter);
                defer row.deinit();
                
                try callback(&row, context);
                
                pos = end + 1;
            } else {
                // Need more data
                const remaining = buffer.items[pos..];
                std.mem.copyForwards(u8, buffer.items[0..remaining.len], remaining);
                buffer.items.len = remaining.len;
                
                const bytes_read = try reader.read(buffer.addManyAsSlice(4096));
                if (bytes_read == 0) {
                    // EOF - process last line if any
                    if (buffer.items.len > 0) {
                        var row = try parser.parseRow(buffer.items, self.delimiter);
                        defer row.deinit();
                        try callback(&row, context);
                    }
                    break;
                }
                
                buffer.items.len += bytes_read;
                pos = 0;
            }
        }
    }
};

// ============================================================================
// Line Iterator
// ============================================================================

/// Iterator for lines (handles \n, \r\n, and multi-line quoted fields)
const LineIterator = struct {
    data: []const u8,
    pos: usize,
    
    pub fn init(data: []const u8) LineIterator {
        return .{
            .data = data,
            .pos = 0,
        };
    }
    
    pub fn next(self: *LineIterator) !?[]const u8 {
        if (self.pos >= self.data.len) return null;
        
        const start = self.pos;
        var in_quotes = false;
        
        while (self.pos < self.data.len) {
            const ch = self.data[self.pos];
            
            if (ch == '"') {
                in_quotes = !in_quotes;
            } else if (ch == '\n' and !in_quotes) {
                const end = self.pos;
                self.pos += 1;
                
                // Trim \r if present (Windows line ending)
                const line = if (end > start and self.data[end - 1] == '\r')
                    self.data[start..end - 1]
                else
                    self.data[start..end];
                
                return line;
            }
            
            self.pos += 1;
        }
        
        // Return remaining data as last line
        if (start < self.data.len) {
            const line = self.data[start..];
            self.pos = self.data.len;
            return line;
        }
        
        return null;
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

/// Detect CSV delimiter from data sample
fn detectDelimiter(data: []const u8) !u8 {
    const candidates = [_]u8{ ',', '\t', ';', '|', ' ' };
    var counts = [_]usize{0} ** 5;
    var line_counts = std.ArrayList(usize).init(std.heap.page_allocator);
    
    for (&line_counts) |*lc| {
        lc.* = std.ArrayList(usize).init(std.heap.page_allocator);
    }
    defer for (&line_counts) |*lc| lc.deinit();

    // Sample first few lines
    var line_iter = LineIterator.init(data);
    var lines_sampled: usize = 0;
    const max_lines_to_sample = 5;
    
    while (try line_iter.next()) |line| {
        if (lines_sampled >= max_lines_to_sample) break;
        
        var in_quotes = false;
        for (line) |ch| {
            if (ch == '"') {
                in_quotes = !in_quotes;
            } else if (!in_quotes) {
                for (candidates, 0..) |candidate, idx| {
                    if (ch == candidate) {
                        counts[idx] += 1;
                        if (line_counts[idx].items.len <= lines_sampled) {
                            try line_counts[idx].append(1);
                        } else {
                            line_counts[idx].items[lines_sampled] += 1;
                        }
                    }
                }
            }
        }
        
        lines_sampled += 1;
    }
    
    // Find most consistent delimiter (same count per line)
    var best_score: f64 = 0;
    var best_idx: usize = 0;
    
    for (counts, 0..) |count, idx| {
        if (count == 0) continue;
        
        // Calculate consistency (lower variance = better)
        const avg = @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(lines_sampled));
        var variance: f64 = 0;
        
        for (line_counts[idx].items) |lc| {
            const diff = @as(f64, @floatFromInt(lc)) - avg;
            variance += diff * diff;
        }
        variance /= @as(f64, @floatFromInt(lines_sampled));
        
        // Score: high count, low variance
        const score = avg / (1.0 + variance);
        
        if (score > best_score) {
            best_score = score;
            best_idx = idx;
        }
    }

    // Default to comma if no clear winner
    if (best_score == 0) {
        return ',';
    }

    return candidates[best_idx];
}

/// Detect encoding with BOM check
const EncodingResult = struct {
    encoding: Encoding,
    bom_size: usize,
};

fn detectEncodingWithBOM(data: []const u8) !EncodingResult {
    // Check for BOM (Byte Order Mark)
    if (data.len >= 3) {
        // UTF-8 BOM: EF BB BF
        if (data[0] == 0xEF and data[1] == 0xBB and data[2] == 0xBF) {
            return .{ .encoding = .utf8, .bom_size = 3 };
        }
    }

    if (data.len >= 2) {
        // UTF-16 LE BOM: FF FE
        if (data[0] == 0xFF and data[1] == 0xFE) {
            return .{ .encoding = .utf16_le, .bom_size = 2 };
        }
        // UTF-16 BE BOM: FE FF
        if (data[0] == 0xFE and data[1] == 0xFF) {
            return .{ .encoding = .utf16_be, .bom_size = 2 };
        }
    }

    // Check if valid UTF-8
    if (string.validateUtf8(data)) |_| {
        return .{ .encoding = .utf8, .bom_size = 0 };
    } else |_| {}

    // Check if ASCII (all bytes < 128)
    var is_ascii = true;
    for (data) |ch| {
        if (ch >= 128) {
            is_ascii = false;
            break;
        }
    }
    if (is_ascii) {
        return .{ .encoding = .ascii, .bom_size = 0 };
    }

    // Default to Latin-1 for other cases
    return .{ .encoding = .latin1, .bom_size = 0 };
}

/// Convert data to UTF-8
fn convertToUtf8(allocator: Allocator, data: []const u8, from_encoding: Encoding) ![]u8 {
    return switch (from_encoding) {
        .utf8 => try allocator.dupe(u8, data),
        .ascii => try allocator.dupe(u8, data), // ASCII is subset of UTF-8
        .latin1 => try convertLatin1ToUtf8(allocator, data),
        .utf16_le => try convertUtf16ToUtf8(allocator, data, false),
        .utf16_be => try convertUtf16ToUtf8(allocator, data, true),
    };
}

/// Convert Latin-1 to UTF-8
fn convertLatin1ToUtf8(allocator: Allocator, data: []const u8) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();
    
    for (data) |byte| {
        if (byte < 128) {
            try result.append(byte);
        } else {
            // Latin-1 bytes 128-255 map to Unicode U+0080-U+00FF
            try result.append(0xC0 | (byte >> 6));
            try result.append(0x80 | (byte & 0x3F));
        }
    }
    
    return result.toOwnedSlice();
}

/// Convert UTF-16 to UTF-8
fn convertUtf16ToUtf8(allocator: Allocator, data: []const u8, big_endian: bool) ![]u8 {
    var result = std.ArrayList(u8).init(allocator);
    errdefer result.deinit();
    
    var i: usize = 0;
    while (i + 1 < data.len) : (i += 2) {
        const c1 = if (big_endian)
            (@as(u16, data[i]) << 8) | data[i + 1]
        else
            (@as(u16, data[i + 1]) << 8) | data[i];
        
        const codepoint: u21 = if (c1 >= 0xD800 and c1 <= 0xDBFF) blk: {
            // High surrogate - need low surrogate
            if (i + 3 < data.len) {
                i += 2;
                const c2 = if (big_endian)
                    (@as(u16, data[i]) << 8) | data[i + 1]
                else
                    (@as(u16, data[i + 1]) << 8) | data[i];
                
                if (c2 >= 0xDC00 and c2 <= 0xDFFF) {
                    const high = c1 - 0xD800;
                    const low = c2 - 0xDC00;
                    break :blk @as(u21, 0x10000) + (@as(u21, high) << 10) + low;
                }
            }
            break :blk 0xFFFD; // Replacement character
        } else c1;
        
        var buf: [4]u8 = undefined;
        const len = try string.encodeUtf8(codepoint, &buf);
        try result.appendSlice(buf[0..len]);
    }
    
    return result.toOwnedSlice();
}

/// Check if line is empty (only whitespace)
fn isEmptyLine(line: []const u8) bool {
    for (line) |ch| {
        if (!std.ascii.isWhitespace(ch)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Export to DoclingDocument
// ============================================================================

/// Export CSV to DoclingDocument
pub fn toDoclingDocument(csv: *const CsvDocument, allocator: Allocator) !types.DoclingDocument {
    var doc = types.DoclingDocument.init(allocator);
    errdefer doc.deinit();

    // Create a single page for CSV (table representation)
    const page_meta = types.PageMetadata.init(1, 800, 600);
    var page = types.Page.init(allocator, page_meta);
    errdefer page.deinit();

    // Create table element
    var table_element = try types.Element.init(allocator, .Table);
    errdefer table_element.deinit();

    // Build table content (Markdown table format)
    var content = std.ArrayList(u8).init(allocator);
    defer content.deinit();

    // Add headers if present
    if (csv.headers) |headers| {
        try content.appendSlice("| ");
        for (headers.fields.items, 0..) |field, idx| {
            if (idx > 0) try content.appendSlice(" | ");
            try content.appendSlice(field);
        }
        try content.appendSlice(" |\n");
        
        // Add separator
        try content.appendSlice("|");
        for (0..headers.fields.items.len) |_| {
            try content.appendSlice(" --- |");
        }
        try content.appendSlice("\n");
    }

    // Add rows
    for (csv.rows.items) |row| {
        try content.appendSlice("| ");
        for (row.fields.items, 0..) |field, idx| {
            if (idx > 0) try content.appendSlice(" | ");
            try content.appendSlice(field);
        }
        try content.appendSlice(" |\n");
    }

    try table_element.setContent(content.items);
    try page.addElement(table_element);
    try doc.addPage(page);

    return doc;
}

// ============================================================================
// FFI Exports
// ============================================================================

/// Parse CSV from data
export fn nExtract_CSV_parse(data: [*]const u8, len: usize) ?*CsvDocument {
    const allocator = std.heap.c_allocator;
    const slice = data[0..len];

    var parser = Parser.init(allocator, .{});
    const csv = parser.parse(slice) catch return null;

    const doc = allocator.create(CsvDocument) catch return null;
    doc.* = csv;
    return doc;
}

/// Parse CSV with options
export fn nExtract_CSV_parseWithOptions(
    data: [*]const u8,
    len: usize,
    delimiter: u8,
    has_headers: bool,
) ?*CsvDocument {
    const allocator = std.heap.c_allocator;
    const slice = data[0..len];

    const options = ParseOptions{
        .delimiter = if (delimiter == 0) null else delimiter,
        .has_headers = has_headers,
    };

    var parser = Parser.init(allocator, options);
    const csv = parser.parse(slice) catch return null;

    const doc = allocator.create(CsvDocument) catch return null;
    doc.* = csv;
    return doc;
}

/// Destroy CSV document
export fn nExtract_CSV_destroy(doc: ?*CsvDocument) void {
    if (doc) |d| {
        var doc_copy = d.*;
        doc_copy.deinit();
        std.heap.c_allocator.destroy(d);
    }
}

/// Get row count
export fn nExtract_CSV_rowCount(doc: ?*const CsvDocument) usize {
    if (doc) |d| {
        return d.rowCount();
    }
    return 0;
}

/// Get column count
export fn nExtract_CSV_columnCount(doc: ?*const CsvDocument) usize {
    if (doc) |d| {
        return d.columnCount();
    }
    return 0;
}

/// Get cell value
export fn nExtract_CSV_getCell(
    doc: ?*const CsvDocument,
    row: usize,
    col: usize,
    out_len: *usize,
) ?[*]const u8 {
    if (doc) |d| {
        if (d.getCell(row, col)) |cell| {
            out_len.* = cell.len;
            return cell.ptr;
        }
    }
    out_len.* = 0;
    return null;
}

// ============================================================================
// Tests
// ============================================================================

test "CSV parser - simple data" {
    const allocator = std.testing.allocator;
    const data = "name,age,city\nAlice,30,NYC\nBob,25,LA\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
    try std.testing.expectEqual(@as(usize, 3), csv.columnCount());
    try std.testing.expectEqual(@as(u8, ','), csv.delimiter);
}

test "CSV parser - quoted fields with commas" {
    const allocator = std.testing.allocator;
    const data = "name,description\n\"Alice\",\"Hello, world!\"\n\"Bob\",\"Test, data\"\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
    try std.testing.expectEqualStrings("Hello, world!", csv.rows.items[0].fields.items[1]);
    try std.testing.expectEqualStrings("Test, data", csv.rows.items[1].fields.items[1]);
}

test "CSV parser - escaped quotes (RFC 4180)" {
    const allocator = std.testing.allocator;
    const data = "name,quote\n\"Alice\",\"She said \"\"Hello\"\"\"\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqualStrings("She said \"Hello\"", csv.rows.items[0].fields.items[1]);
}

test "CSV parser - multi-line quoted field" {
    const allocator = std.testing.allocator;
    const data = "name,description\n\"Alice\",\"Line 1\nLine 2\nLine 3\"\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 1), csv.rowCount());
    try std.testing.expect(std.mem.indexOf(u8, csv.rows.items[0].fields.items[1], "\n") != null);
}

test "CSV parser - tab delimiter detection" {
    const allocator = std.testing.allocator;
    const data = "name\tage\tcity\nAlice\t30\tNYC\nBob\t25\tLA\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(u8, '\t'), csv.delimiter);
    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
}

test "CSV parser - semicolon delimiter" {
    const allocator = std.testing.allocator;
    const data = "name;age;city\nAlice;30;NYC\nBob;25;LA\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(u8, ';'), csv.delimiter);
}

test "CSV parser - pipe delimiter" {
    const allocator = std.testing.allocator;
    const data = "name|age|city\nAlice|30|NYC\nBob|25|LA\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(u8, '|'), csv.delimiter);
}

test "CSV parser - encoding detection UTF-8" {
    const allocator = std.testing.allocator;
    const data = "name,age\nAlice,30\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(Encoding.utf8, csv.encoding);
}

test "CSV parser - empty lines" {
    const allocator = std.testing.allocator;
    const data = "name,age\n\nAlice,30\n\nBob,25\n";

    var parser = Parser.init(allocator, .{ .skip_empty_lines = true });
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
}

test "CSV parser - no headers" {
    const allocator = std.testing.allocator;
    const data = "Alice,30,NYC\nBob,25,LA\n";

    var parser = Parser.init(allocator, .{ .has_headers = false });
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
    try std.testing.expect(csv.headers == null);
}

test "CSV parser - trim fields" {
    const allocator = std.testing.allocator;
    const data = "name,age\n  Alice  ,  30  \n  Bob  ,  25  \n";

    var parser = Parser.init(allocator, .{ .trim_fields = true });
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqualStrings("Alice", csv.rows.items[0].fields.items[0]);
    try std.testing.expectEqualStrings("30", csv.rows.items[0].fields.items[1]);
}

test "CSV parser - Windows line endings" {
    const allocator = std.testing.allocator;
    const data = "name,age\r\nAlice,30\r\nBob,25\r\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 2), csv.rowCount());
}

test "CSV parser - getCell" {
    const allocator = std.testing.allocator;
    const data = "name,age\nAlice,30\nBob,25\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqualStrings("Alice", csv.getCell(0, 0).?);
    try std.testing.expectEqualStrings("30", csv.getCell(0, 1).?);
    try std.testing.expectEqualStrings("Bob", csv.getCell(1, 0).?);
    try std.testing.expect(csv.getCell(10, 0) == null);
}

test "CSV parser - getHeader" {
    const allocator = std.testing.allocator;
    const data = "name,age\nAlice,30\n";

    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data);
    defer csv.deinit();

    try std.testing.expectEqualStrings("name", csv.getHeader(0).?);
    try std.testing.expectEqualStrings("age", csv.getHeader(1).?);
    try std.testing.expect(csv.getHeader(10) == null);
}

test "CSV parser - large field" {
    const allocator = std.testing.allocator;
    
    var large_field = std.ArrayList(u8).init(allocator);
    defer large_field.deinit();
    
    // Create 1MB field
    for (0..1024*1024) |_| {
        try large_field.append('x');
    }
    
    var data = std.ArrayList(u8).init(allocator);
    defer data.deinit();
    
    try data.appendSlice("field1,field2\n");
    try data.appendSlice("a,\"");
    try data.appendSlice(large_field.items);
    try data.appendSlice("\"\n");
    
    var parser = Parser.init(allocator, .{});
    var csv = try parser.parse(data.items);
    defer csv.deinit();

    try std.testing.expectEqual(@as(usize, 1), csv.rowCount());
    try std.testing.expectEqual(@as(usize, 1024*1024), csv.getCell(0, 1).?.len);
}
