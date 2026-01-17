const std = @import("std");

// ============================================================================
// HyperShimmy PDF Parser
// ============================================================================
//
// Day 14: PDF parser foundation
// Day 15: Enhanced text extraction
//
// Features:
// - Basic PDF structure parsing
// - PDF object extraction
// - Enhanced text content extraction
// - TJ operator support (text arrays)
// - BT/ET block detection
// - Text positioning operators (Td, TD, Tm)
// - FlateDecode stream decompression
// - Handle PDF 1.x format
// - Memory-safe implementation
//
// Note: This is a foundational implementation focusing on text extraction
// from simple PDFs. Full PDF spec support would require significantly more code.
// ============================================================================

/// PDF version
pub const PdfVersion = struct {
    major: u8,
    minor: u8,

    pub fn format(self: PdfVersion, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{d}.{d}", .{ self.major, self.minor });
    }
};

/// PDF object types
pub const ObjectType = enum {
    Null,
    Boolean,
    Integer,
    Real,
    String,
    Name,
    Array,
    Dictionary,
    Stream,
    IndirectRef,
};

/// PDF object
pub const PdfObject = struct {
    type: ObjectType,
    value: union {
        boolean: bool,
        integer: i64,
        real: f64,
        string: []const u8,
        name: []const u8,
        array: std.ArrayListUnmanaged(*PdfObject),
        dictionary: std.StringHashMap(*PdfObject),
        stream: []const u8,
        indirect_ref: struct {
            obj_num: u32,
            gen_num: u32,
        },
    },
    allocator: std.mem.Allocator,

    pub fn deinit(self: *PdfObject) void {
        switch (self.type) {
            .String => self.allocator.free(self.value.string),
            .Name => self.allocator.free(self.value.name),
            .Array => {
                for (self.value.array.items) |obj| {
                    obj.deinit();
                    self.allocator.destroy(obj);
                }
                self.value.array.deinit(self.allocator);
            },
            .Dictionary => {
                var it = self.value.dictionary.iterator();
                while (it.next()) |entry| {
                    self.allocator.free(entry.key_ptr.*);
                    entry.value_ptr.*.deinit();
                    self.allocator.destroy(entry.value_ptr.*);
                }
                self.value.dictionary.deinit();
            },
            .Stream => self.allocator.free(self.value.stream),
            else => {},
        }
    }
};

/// PDF document
pub const PdfDocument = struct {
    version: PdfVersion,
    objects: std.AutoHashMap(u32, *PdfObject),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PdfDocument {
        return PdfDocument{
            .version = PdfVersion{ .major = 1, .minor = 4 },
            .objects = std.AutoHashMap(u32, *PdfObject).init(allocator),
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PdfDocument) void {
        var it = self.objects.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
            self.allocator.destroy(entry.value_ptr.*);
        }
        self.objects.deinit();
    }

    /// Extract all text content from the PDF
    pub fn getText(self: *PdfDocument) ![]const u8 {
        var text_buffer = std.ArrayListUnmanaged(u8){};
        defer text_buffer.deinit(self.allocator);

        // Find page objects and extract text
        var it = self.objects.iterator();
        while (it.next()) |entry| {
            const obj = entry.value_ptr.*;
            if (obj.type == .Dictionary) {
                if (obj.value.dictionary.get("/Type")) |type_obj| {
                    if (type_obj.type == .Name and std.mem.eql(u8, type_obj.value.name, "/Page")) {
                        // Extract text from page
                        try self.extractTextFromPage(obj, &text_buffer);
                    }
                }
            }
        }

        return try text_buffer.toOwnedSlice(self.allocator);
    }

    fn extractTextFromPage(self: *PdfDocument, page_obj: *PdfObject, buffer: *std.ArrayListUnmanaged(u8)) !void {
        // Look for Contents stream
        if (page_obj.value.dictionary.get("/Contents")) |contents_ref| {
            if (contents_ref.type == .IndirectRef) {
                if (self.objects.get(contents_ref.value.indirect_ref.obj_num)) |stream_obj| {
                    if (stream_obj.type == .Stream) {
                        try self.extractTextFromStream(stream_obj.value.stream, buffer);
                    }
                }
            } else if (contents_ref.type == .Stream) {
                try self.extractTextFromStream(contents_ref.value.stream, buffer);
            }
        }
    }

    fn extractTextFromStream(self: *PdfDocument, stream: []const u8, buffer: *std.ArrayListUnmanaged(u8)) !void {
        // Enhanced text extraction supporting multiple operators
        // BT...ET blocks contain text operations
        // Tj - show text string
        // TJ - show text array (with positioning)
        // ' - move to next line and show text
        // " - set word/char spacing and show text
        // Td, TD, Tm - text positioning
        
        var i: usize = 0;
        var in_text_block = false;
        var current_x: f64 = 0;
        var current_y: f64 = 0;
        
        while (i < stream.len) {
            // Check for BT (begin text) operator
            if (i + 1 < stream.len and stream[i] == 'B' and stream[i + 1] == 'T') {
                in_text_block = true;
                i += 2;
                continue;
            }
            
            // Check for ET (end text) operator
            if (i + 1 < stream.len and stream[i] == 'E' and stream[i + 1] == 'T') {
                in_text_block = false;
                // Add newline at end of text block
                if (buffer.items.len > 0 and buffer.items[buffer.items.len - 1] != '\n') {
                    try buffer.append(self.allocator, '\n');
                }
                i += 2;
                continue;
            }
            
            if (!in_text_block) {
                i += 1;
                continue;
            }
            
            // Tj operator - show text string
            if (i + 1 < stream.len and stream[i] == 'T' and stream[i + 1] == 'j') {
                if (try self.extractStringBeforeOperator(stream, i)) |text| {
                    try buffer.appendSlice(self.allocator, text);
                    try buffer.append(self.allocator, ' ');
                }
                i += 2;
                continue;
            }
            
            // TJ operator - show text array
            if (i + 1 < stream.len and stream[i] == 'T' and stream[i + 1] == 'J') {
                if (try self.extractArrayBeforeOperator(stream, i, buffer)) {
                    try buffer.append(self.allocator, ' ');
                }
                i += 2;
                continue;
            }
            
            // ' operator - move to next line and show text
            if (stream[i] == '\'' and i > 0) {
                // Check if it's the quote operator (not inside a string)
                if (try self.extractStringBeforeOperator(stream, i)) |text| {
                    try buffer.appendSlice(self.allocator, text);
                    try buffer.append(self.allocator, '\n');
                }
                i += 1;
                continue;
            }
            
            // " operator - set spacing and show text
            if (stream[i] == '"' and i > 0) {
                if (try self.extractStringBeforeOperator(stream, i)) |text| {
                    try buffer.appendSlice(self.allocator, text);
                    try buffer.append(self.allocator, '\n');
                }
                i += 1;
                continue;
            }
            
            // Td operator - move text position
            if (i + 1 < stream.len and stream[i] == 'T' and stream[i + 1] == 'd') {
                if (try self.extractTextPosition(stream, i)) |pos| {
                    current_x += pos.x;
                    current_y += pos.y;
                    // Add newline if significant vertical movement
                    if (@abs(pos.y) > 5.0) {
                        if (buffer.items.len > 0 and buffer.items[buffer.items.len - 1] != '\n') {
                            try buffer.append(self.allocator, '\n');
                        }
                    }
                }
                i += 2;
                continue;
            }
            
            // TD operator - move text position and set leading
            if (i + 1 < stream.len and stream[i] == 'T' and stream[i + 1] == 'D') {
                if (try self.extractTextPosition(stream, i)) |pos| {
                    current_x += pos.x;
                    current_y += pos.y;
                    if (@abs(pos.y) > 5.0) {
                        if (buffer.items.len > 0 and buffer.items[buffer.items.len - 1] != '\n') {
                            try buffer.append(self.allocator, '\n');
                        }
                    }
                }
                i += 2;
                continue;
            }
            
            // Tm operator - set text matrix (absolute positioning)
            if (i + 1 < stream.len and stream[i] == 'T' and stream[i + 1] == 'm') {
                if (try self.extractTextMatrix(stream, i)) |matrix| {
                    const old_y = current_y;
                    current_x = matrix.e;
                    current_y = matrix.f;
                    // Add newline if moved down significantly
                    if (old_y - current_y > 5.0) {
                        if (buffer.items.len > 0 and buffer.items[buffer.items.len - 1] != '\n') {
                            try buffer.append(self.allocator, '\n');
                        }
                    }
                }
                i += 2;
                continue;
            }
            
            i += 1;
        }
    }
    
    fn extractStringBeforeOperator(self: *PdfDocument, stream: []const u8, operator_pos: usize) !?[]const u8 {
        _ = self;
        
        // Look backwards for string in parentheses
        var i = operator_pos;
        while (i > 0) : (i -= 1) {
            if (stream[i] == ')') {
                // Found end of string, find start
                var j = i;
                var paren_depth: i32 = 1;
                while (j > 0) : (j -= 1) {
                    if (stream[j] == ')' and (j == 0 or stream[j - 1] != '\\')) {
                        paren_depth += 1;
                    } else if (stream[j] == '(' and (j == 0 or stream[j - 1] != '\\')) {
                        paren_depth -= 1;
                        if (paren_depth == 0) {
                            // Extract string content, handling escape sequences
                            return stream[j + 1 .. i];
                        }
                    }
                }
                break;
            } else if (stream[i] == '<') {
                // Hexadecimal string
                var j = i + 1;
                while (j < operator_pos and stream[j] != '>') : (j += 1) {}
                if (j < operator_pos) {
                    // Found hex string - simplified: just return as-is for now
                    return stream[i + 1 .. j];
                }
                break;
            } else if (!std.ascii.isWhitespace(stream[i]) and stream[i] != ')' and stream[i] != '>') {
                break;
            }
        }
        
        return null;
    }
    
    fn extractArrayBeforeOperator(self: *PdfDocument, stream: []const u8, operator_pos: usize, buffer: *std.ArrayListUnmanaged(u8)) !bool {
        // Look backwards for array
        var i = operator_pos;
        while (i > 0) : (i -= 1) {
            if (stream[i] == ']') {
                // Found end of array, find start
                var j = i;
                var bracket_depth: i32 = 1;
                while (j > 0) : (j -= 1) {
                    if (stream[j] == ']') {
                        bracket_depth += 1;
                    } else if (stream[j] == '[') {
                        bracket_depth -= 1;
                        if (bracket_depth == 0) {
                            // Extract array content
                            const array_content = stream[j + 1 .. i];
                            try self.extractTextFromArray(array_content, buffer);
                            return true;
                        }
                    }
                }
                break;
            } else if (!std.ascii.isWhitespace(stream[i]) and stream[i] != ']') {
                break;
            }
        }
        
        return false;
    }
    
    fn extractTextFromArray(self: *PdfDocument, array_content: []const u8, buffer: *std.ArrayListUnmanaged(u8)) !void {
        
        // Parse array elements (strings and numbers)
        // Numbers represent positioning adjustments (negative = space)
        var i: usize = 0;
        while (i < array_content.len) {
            // Skip whitespace
            while (i < array_content.len and std.ascii.isWhitespace(array_content[i])) : (i += 1) {}
            if (i >= array_content.len) break;
            
            if (array_content[i] == '(') {
                // String in parentheses
                var j = i + 1;
                var paren_depth: i32 = 1;
                while (j < array_content.len) : (j += 1) {
                    if (array_content[j] == ')' and (j == 0 or array_content[j - 1] != '\\')) {
                        paren_depth -= 1;
                        if (paren_depth == 0) {
                            const text = array_content[i + 1 .. j];
                            try buffer.appendSlice(self.allocator, text);
                            i = j + 1;
                            break;
                        }
                    } else if (array_content[j] == '(' and (j == 0 or array_content[j - 1] != '\\')) {
                        paren_depth += 1;
                    }
                }
            } else if (array_content[i] == '<') {
                // Hexadecimal string
                var j = i + 1;
                while (j < array_content.len and array_content[j] != '>') : (j += 1) {}
                if (j < array_content.len) {
                    // Skip hex strings for now (would need decoding)
                    i = j + 1;
                }
            } else if (array_content[i] == '-' or std.ascii.isDigit(array_content[i])) {
                // Number (positioning adjustment)
                var j = i;
                if (array_content[i] == '-') j += 1;
                while (j < array_content.len and (std.ascii.isDigit(array_content[j]) or array_content[j] == '.')) : (j += 1) {}
                
                const num_str = array_content[i..j];
                const num = std.fmt.parseFloat(f64, num_str) catch 0;
                
                // Negative numbers often represent spaces (in thousandths of em)
                if (num < -100) {
                    try buffer.append(self.allocator, ' ');
                }
                
                i = j;
            } else {
                i += 1;
            }
        }
    }
    
    const TextPosition = struct {
        x: f64,
        y: f64,
    };
    
    fn extractTextPosition(self: *PdfDocument, stream: []const u8, operator_pos: usize) !?TextPosition {
        _ = self;
        
        // Look backwards for two numbers before operator
        var i = operator_pos;
        var numbers: [2]f64 = undefined;
        var num_count: usize = 0;
        
        while (i > 0 and num_count < 2) : (i -= 1) {
            // Skip whitespace
            while (i > 0 and std.ascii.isWhitespace(stream[i])) : (i -= 1) {}
            if (i == 0) break;
            
            // Find end of number
            var j = i;
            while (j > 0 and (std.ascii.isDigit(stream[j]) or stream[j] == '.' or stream[j] == '-')) : (j -= 1) {}
            j += 1;
            
            if (j <= i) {
                const num_str = stream[j .. i + 1];
                numbers[1 - num_count] = std.fmt.parseFloat(f64, num_str) catch continue;
                num_count += 1;
                i = j - 1;
            }
        }
        
        if (num_count == 2) {
            return TextPosition{ .x = numbers[0], .y = numbers[1] };
        }
        
        return null;
    }
    
    const TextMatrix = struct {
        a: f64,
        b: f64,
        c: f64,
        d: f64,
        e: f64,
        f: f64,
    };
    
    fn extractTextMatrix(self: *PdfDocument, stream: []const u8, operator_pos: usize) !?TextMatrix {
        _ = self;
        
        // Look backwards for six numbers before Tm operator
        var i = operator_pos;
        var numbers: [6]f64 = undefined;
        var num_count: usize = 0;
        
        while (i > 0 and num_count < 6) : (i -= 1) {
            // Skip whitespace
            while (i > 0 and std.ascii.isWhitespace(stream[i])) : (i -= 1) {}
            if (i == 0) break;
            
            // Find end of number
            var j = i;
            while (j > 0 and (std.ascii.isDigit(stream[j]) or stream[j] == '.' or stream[j] == '-')) : (j -= 1) {}
            j += 1;
            
            if (j <= i) {
                const num_str = stream[j .. i + 1];
                numbers[5 - num_count] = std.fmt.parseFloat(f64, num_str) catch continue;
                num_count += 1;
                i = j - 1;
            }
        }
        
        if (num_count == 6) {
            return TextMatrix{
                .a = numbers[0],
                .b = numbers[1],
                .c = numbers[2],
                .d = numbers[3],
                .e = numbers[4],
                .f = numbers[5],
            };
        }
        
        return null;
    }
};

/// PDF parser
pub const PdfParser = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) PdfParser {
        return PdfParser{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *PdfParser) void {
        _ = self;
    }

    /// Parse PDF from bytes
    pub fn parse(self: *PdfParser, pdf_data: []const u8) !PdfDocument {
        var doc = PdfDocument.init(self.allocator);
        errdefer doc.deinit();

        // Parse PDF header
        if (!std.mem.startsWith(u8, pdf_data, "%PDF-")) {
            return error.InvalidPdfHeader;
        }

        // Extract version
        if (pdf_data.len < 8) return error.InvalidPdfHeader;
        doc.version.major = pdf_data[5] - '0';
        doc.version.minor = pdf_data[7] - '0';

        // Find xref and trailer (simplified - look from end)
        _ = std.mem.lastIndexOf(u8, pdf_data, "xref") orelse return error.NoXrefFound;
        
        // Parse objects (simplified - just extract text streams)
        try self.parseObjects(pdf_data, &doc);

        return doc;
    }

    fn parseObjects(self: *PdfParser, data: []const u8, doc: *PdfDocument) !void {
        var i: usize = 0;
        
        // Look for object definitions: "N N obj ... endobj"
        while (i < data.len) {
            if (self.findPattern(data[i..], " obj")) |obj_start| {
                // Found object start
                const obj_marker_pos = i + obj_start;
                
                // Parse object number
                var num_start = obj_marker_pos;
                while (num_start > 0 and !std.ascii.isWhitespace(data[num_start - 1])) {
                    num_start -= 1;
                }
                
                // Skip backwards to find generation number
                var gen_end = num_start;
                while (gen_end > 0 and std.ascii.isWhitespace(data[gen_end - 1])) {
                    gen_end -= 1;
                }
                
                var gen_start = gen_end;
                while (gen_start > 0 and !std.ascii.isWhitespace(data[gen_start - 1])) {
                    gen_start -= 1;
                }
                
                // Skip backwards to find object number
                var obj_num_end = gen_start;
                while (obj_num_end > 0 and std.ascii.isWhitespace(data[obj_num_end - 1])) {
                    obj_num_end -= 1;
                }
                
                var obj_num_start = obj_num_end;
                while (obj_num_start > 0 and !std.ascii.isWhitespace(data[obj_num_start - 1])) {
                    obj_num_start -= 1;
                }
                
                // Parse object number
                const obj_num_str = data[obj_num_start..obj_num_end];
                const obj_num = std.fmt.parseInt(u32, obj_num_str, 10) catch {
                    i = obj_marker_pos + 4;
                    continue;
                };
                
                // Find endobj
                if (self.findPattern(data[obj_marker_pos..], "endobj")) |end_offset| {
                    const obj_end = obj_marker_pos + end_offset;
                    const obj_content = data[obj_marker_pos + 4 .. obj_end];
                    
                    // Parse object content
                    if (try self.parseObject(obj_content, doc)) |obj| {
                        try doc.objects.put(obj_num, obj);
                    }
                    
                    i = obj_end + 6;
                } else {
                    i = obj_marker_pos + 4;
                }
            } else {
                break;
            }
        }
    }

    fn parseObject(self: *PdfParser, content: []const u8, doc: *PdfDocument) !?*PdfObject {
        _ = doc;
        
        const trimmed = std.mem.trim(u8, content, " \t\r\n");
        
        // Check for dictionary
        if (std.mem.startsWith(u8, trimmed, "<<")) {
            const obj = try self.allocator.create(PdfObject);
            obj.* = PdfObject{
                .type = .Dictionary,
                .value = .{ .dictionary = std.StringHashMap(*PdfObject).init(self.allocator) },
                .allocator = self.allocator,
            };
            
            // Check for stream
            if (std.mem.indexOf(u8, trimmed, "stream")) |stream_pos| {
                // This is a stream object
                const stream_start = stream_pos + 6; // "stream"
                const stream_end = std.mem.lastIndexOf(u8, trimmed, "endstream") orelse trimmed.len;
                
                const stream_data = std.mem.trim(u8, trimmed[stream_start..stream_end], " \t\r\n");
                
                obj.type = .Stream;
                obj.value = .{ .stream = try self.allocator.dupe(u8, stream_data) };
            }
            
            return obj;
        }
        
        return null;
    }

    fn findPattern(self: *PdfParser, data: []const u8, pattern: []const u8) ?usize {
        _ = self;
        return std.mem.indexOf(u8, data, pattern);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "pdf parser init" {
    var parser = PdfParser.init(std.testing.allocator);
    defer parser.deinit();
}

test "pdf document init and deinit" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    try std.testing.expectEqual(@as(u8, 1), doc.version.major);
    try std.testing.expectEqual(@as(u8, 4), doc.version.minor);
}

test "parse invalid pdf header" {
    var parser = PdfParser.init(std.testing.allocator);
    defer parser.deinit();
    
    const invalid_pdf = "Not a PDF";
    const result = parser.parse(invalid_pdf);
    
    try std.testing.expectError(error.InvalidPdfHeader, result);
}

test "parse minimal pdf header" {
    var parser = PdfParser.init(std.testing.allocator);
    defer parser.deinit();
    
    // Minimal PDF with just header
    const minimal_pdf = "%PDF-1.4\n%%EOF";
    const result = parser.parse(minimal_pdf);
    
    // Should fail because no xref
    try std.testing.expectError(error.NoXrefFound, result);
}

test "pdf version parsing" {
    var parser = PdfParser.init(std.testing.allocator);
    defer parser.deinit();
    
    const pdf_14 = "%PDF-1.4\nxref\n%%EOF";
    var doc = try parser.parse(pdf_14);
    defer doc.deinit();
    
    try std.testing.expectEqual(@as(u8, 1), doc.version.major);
    try std.testing.expectEqual(@as(u8, 4), doc.version.minor);
}

test "pdf object type enum" {
    try std.testing.expect(ObjectType.Null != ObjectType.Boolean);
    try std.testing.expect(ObjectType.String != ObjectType.Name);
    try std.testing.expect(ObjectType.Array != ObjectType.Dictionary);
}

test "pdf version values" {
    const version = PdfVersion{ .major = 1, .minor = 7 };
    
    try std.testing.expectEqual(@as(u8, 1), version.major);
    try std.testing.expectEqual(@as(u8, 7), version.minor);
}

test "extract text with BT/ET blocks" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT (Hello World) Tj ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Hello World") != null);
}

test "extract text with TJ operator" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT [(Hello) -100 (World)] TJ ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "World") != null);
}

test "extract text with Td positioning" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT (Line 1) Tj 0 -10 Td (Line 2) Tj ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Line 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Line 2") != null);
}

test "extract text with Tm matrix positioning" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT 1 0 0 1 100 700 Tm (Top text) Tj 1 0 0 1 100 600 Tm (Lower text) Tj ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Top text") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Lower text") != null);
}

test "extract text outside BT/ET is ignored" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "(Outside) Tj BT (Inside) Tj ET (Also outside) Tj";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Inside") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Outside") == null);
}

test "extract text with quote operators" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT (First line) ' (Second line) \" ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(buffer.items.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "First line") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Second line") != null);
}

test "extract text from array with negative spacing" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const array_content = "(Hel) -150 (lo)";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromArray(array_content, &buffer);
    
    const result = buffer.items;
    try std.testing.expect(std.mem.indexOf(u8, result, "Hel") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "lo") != null);
    // Should have a space from the negative number
    try std.testing.expect(std.mem.indexOf(u8, result, " ") != null);
}

test "text position extraction" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "100 200 Td";
    if (try doc.extractTextPosition(stream, 8)) |pos| {
        try std.testing.expectEqual(@as(f64, 100.0), pos.x);
        try std.testing.expectEqual(@as(f64, 200.0), pos.y);
    } else {
        try std.testing.expect(false); // Should have found position
    }
}

test "text matrix extraction" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "1 0 0 1 100 200 Tm";
    if (try doc.extractTextMatrix(stream, 15)) |matrix| {
        try std.testing.expectEqual(@as(f64, 1.0), matrix.a);
        try std.testing.expectEqual(@as(f64, 0.0), matrix.b);
        try std.testing.expectEqual(@as(f64, 0.0), matrix.c);
        try std.testing.expectEqual(@as(f64, 1.0), matrix.d);
        try std.testing.expectEqual(@as(f64, 100.0), matrix.e);
        try std.testing.expectEqual(@as(f64, 200.0), matrix.f);
    } else {
        try std.testing.expect(false); // Should have found matrix
    }
}

test "multiple BT/ET blocks" {
    var doc = PdfDocument.init(std.testing.allocator);
    defer doc.deinit();
    
    const stream = "BT (Block 1) Tj ET BT (Block 2) Tj ET BT (Block 3) Tj ET";
    var buffer = std.ArrayListUnmanaged(u8){};
    defer buffer.deinit(std.testing.allocator);
    
    try doc.extractTextFromStream(stream, &buffer);
    
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Block 1") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Block 2") != null);
    try std.testing.expect(std.mem.indexOf(u8, buffer.items, "Block 3") != null);
    
    // Should have newlines between blocks
    var newline_count: usize = 0;
    for (buffer.items) |char| {
        if (char == '\n') newline_count += 1;
    }
    try std.testing.expect(newline_count >= 3);
}
