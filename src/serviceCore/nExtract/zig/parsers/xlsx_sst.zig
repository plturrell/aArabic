// XLSX Shared String Table (SST) Parser
// Handles xl/sharedStrings.xml for string deduplication in Excel files

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;
const xml = @import("xml.zig");

/// Shared String Table - stores all unique strings in an XLSX workbook
pub const SharedStringTable = struct {
    strings: ArrayList(SharedString),
    count: usize,           // Total string occurrences (with duplicates)
    unique_count: usize,    // Number of unique strings
    allocator: Allocator,
    
    const Self = @This();
    
    pub fn init(allocator: Allocator) Self {
        return .{
            .strings = ArrayList(SharedString).init(allocator),
            .count = 0,
            .unique_count = 0,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Self) void {
        for (self.strings.items) |*str| {
            str.deinit();
        }
        self.strings.deinit();
    }
    
    /// Parse shared strings from XML data
    pub fn parseFromXml(allocator: Allocator, xml_data: []const u8) !Self {
        var sst = Self.init(allocator);
        errdefer sst.deinit();
        
        var doc = try xml.parseDocument(allocator, xml_data);
        defer doc.deinit();
        
        const root = doc.root orelse return error.InvalidSharedStrings;
        
        // Get count and uniqueCount attributes
        if (root.element.getAttribute("count")) |count_str| {
            sst.count = try std.fmt.parseInt(usize, count_str, 10);
        }
        if (root.element.getAttribute("uniqueCount")) |unique_count_str| {
            sst.unique_count = try std.fmt.parseInt(usize, unique_count_str, 10);
        }
        
        // Parse each <si> (string item) element
        for (root.element.children.items) |child| {
            if (child != .element) continue;
            const elem = child.element;
            
            if (std.mem.eql(u8, elem.name, "si")) {
                const shared_str = try parseStringItem(allocator, &elem);
                try sst.strings.append(shared_str);
            }
        }
        
        // Update unique count if not specified
        if (sst.unique_count == 0) {
            sst.unique_count = sst.strings.items.len;
        }
        
        return sst;
    }
    
    /// Get string at index
    pub fn getString(self: *const Self, index: usize) ?*const SharedString {
        if (index >= self.strings.items.len) return null;
        return &self.strings.items[index];
    }
    
    /// Get plain text at index (simplified access)
    pub fn getPlainText(self: *const Self, index: usize) ?[]const u8 {
        const str = self.getString(index) orelse return null;
        return str.getPlainText();
    }
};

/// A single shared string entry (can be simple text or rich text)
pub const SharedString = struct {
    /// Type of string content
    type: StringType,
    /// Simple text content (for simple strings)
    text: ?[]const u8,
    /// Rich text runs (for formatted strings)
    rich_text: ArrayList(RichTextRun),
    /// Phonetic properties (for Japanese/Chinese)
    phonetic: ?PhoneticProperties,
    allocator: Allocator,
    
    pub const StringType = enum {
        simple,      // Plain text
        rich_text,   // Formatted text with multiple runs
        phonetic,    // Text with phonetic annotations
    };
    
    pub fn init(allocator: Allocator) SharedString {
        return .{
            .type = .simple,
            .text = null,
            .rich_text = ArrayList(RichTextRun).init(allocator),
            .phonetic = null,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SharedString) void {
        if (self.text) |t| self.allocator.free(t);
        
        for (self.rich_text.items) |*run| {
            run.deinit();
        }
        self.rich_text.deinit();
        
        if (self.phonetic) |*p| {
            p.deinit(self.allocator);
        }
    }
    
    /// Get plain text representation (concatenates rich text if needed)
    pub fn getPlainText(self: *const SharedString) []const u8 {
        switch (self.type) {
            .simple => return self.text orelse "",
            .rich_text, .phonetic => {
                // For rich text, concatenate all runs
                // Note: In production, this should be cached
                if (self.rich_text.items.len == 0) return "";
                if (self.rich_text.items.len == 1) return self.rich_text.items[0].text;
                
                // Multiple runs - would need to concatenate (simplified for now)
                return self.rich_text.items[0].text;
            },
        }
    }
    
    /// Check if string has formatting
    pub fn hasFormatting(self: *const SharedString) bool {
        return self.type != .simple;
    }
};

/// Rich text run with formatting
pub const RichTextRun = struct {
    text: []const u8,
    properties: ?TextProperties,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, text: []const u8) !RichTextRun {
        return .{
            .text = try allocator.dupe(u8, text),
            .properties = null,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *RichTextRun) void {
        self.allocator.free(self.text);
        if (self.properties) |*props| {
            props.deinit(self.allocator);
        }
    }
};

/// Text formatting properties
pub const TextProperties = struct {
    bold: bool = false,
    italic: bool = false,
    underline: bool = false,
    strike: bool = false,
    font_name: ?[]const u8 = null,
    font_size: ?f32 = null,
    color: ?[]const u8 = null,
    
    pub fn deinit(self: *TextProperties, allocator: Allocator) void {
        if (self.font_name) |name| allocator.free(name);
        if (self.color) |c| allocator.free(c);
    }
};

/// Phonetic properties for Japanese/Chinese text
pub const PhoneticProperties = struct {
    font_name: ?[]const u8 = null,
    font_size: ?f32 = null,
    alignment: PhoneticAlignment = .left,
    type: PhoneticType = .full_width_katakana,
    
    pub const PhoneticAlignment = enum {
        left,
        center,
        distributed,
        no_control,
    };
    
    pub const PhoneticType = enum {
        half_width_katakana,
        full_width_katakana,
        hiragana,
        no_conversion,
    };
    
    pub fn deinit(self: *PhoneticProperties, allocator: Allocator) void {
        if (self.font_name) |name| allocator.free(name);
    }
};

/// Parse a string item (<si> element)
fn parseStringItem(allocator: Allocator, si_elem: *const xml.Element) !SharedString {
    var shared_str = SharedString.init(allocator);
    errdefer shared_str.deinit();
    
    // Check for simple text (<t> element)
    for (si_elem.children.items) |child| {
        if (child != .element) continue;
        const elem = child.element;
        
        if (std.mem.eql(u8, elem.name, "t")) {
            // Simple text
            const text = elem.getTextContent() orelse "";
            shared_str.text = try allocator.dupe(u8, text);
            shared_str.type = .simple;
            return shared_str;
        } else if (std.mem.eql(u8, elem.name, "r")) {
            // Rich text run
            const run = try parseRichTextRun(allocator, &elem);
            try shared_str.rich_text.append(run);
            shared_str.type = .rich_text;
        } else if (std.mem.eql(u8, elem.name, "rPh")) {
            // Phonetic run
            shared_str.type = .phonetic;
        } else if (std.mem.eql(u8, elem.name, "phoneticPr")) {
            // Phonetic properties
            shared_str.phonetic = try parsePhoneticProperties(allocator, &elem);
        }
    }
    
    return shared_str;
}

/// Parse a rich text run (<r> element)
fn parseRichTextRun(allocator: Allocator, r_elem: *const xml.Element) !RichTextRun {
    var text: []const u8 = "";
    var properties: ?TextProperties = null;
    
    for (r_elem.children.items) |child| {
        if (child != .element) continue;
        const elem = child.element;
        
        if (std.mem.eql(u8, elem.name, "t")) {
            // Text content
            text = elem.getTextContent() orelse "";
        } else if (std.mem.eql(u8, elem.name, "rPr")) {
            // Run properties (formatting)
            properties = try parseTextProperties(allocator, &elem);
        }
    }
    
    var run = try RichTextRun.init(allocator, text);
    run.properties = properties;
    return run;
}

/// Parse text properties (<rPr> element)
fn parseTextProperties(allocator: Allocator, rPr_elem: *const xml.Element) !TextProperties {
    var props = TextProperties{};
    
    for (rPr_elem.children.items) |child| {
        if (child != .element) continue;
        const elem = child.element;
        
        if (std.mem.eql(u8, elem.name, "b")) {
            props.bold = true;
        } else if (std.mem.eql(u8, elem.name, "i")) {
            props.italic = true;
        } else if (std.mem.eql(u8, elem.name, "u")) {
            props.underline = true;
        } else if (std.mem.eql(u8, elem.name, "strike")) {
            props.strike = true;
        } else if (std.mem.eql(u8, elem.name, "rFont")) {
            if (elem.getAttribute("val")) |val| {
                props.font_name = try allocator.dupe(u8, val);
            }
        } else if (std.mem.eql(u8, elem.name, "sz")) {
            if (elem.getAttribute("val")) |val| {
                props.font_size = try std.fmt.parseFloat(f32, val);
            }
        } else if (std.mem.eql(u8, elem.name, "color")) {
            if (elem.getAttribute("rgb")) |rgb| {
                props.color = try allocator.dupe(u8, rgb);
            }
        }
    }
    
    return props;
}

/// Parse phonetic properties (<phoneticPr> element)
fn parsePhoneticProperties(allocator: Allocator, phoneticPr_elem: *const xml.Element) !PhoneticProperties {
    var props = PhoneticProperties{};
    
    if (phoneticPr_elem.getAttribute("fontId")) |font_id| {
        // Font ID would need to be resolved from styles
        _ = font_id;
    }
    
    if (phoneticPr_elem.getAttribute("type")) |type_str| {
        if (std.mem.eql(u8, type_str, "halfwidthKatakana")) {
            props.type = .half_width_katakana;
        } else if (std.mem.eql(u8, type_str, "fullwidthKatakana")) {
            props.type = .full_width_katakana;
        } else if (std.mem.eql(u8, type_str, "Hiragana")) {
            props.type = .hiragana;
        } else if (std.mem.eql(u8, type_str, "noConversion")) {
            props.type = .no_conversion;
        }
    }
    
    if (phoneticPr_elem.getAttribute("alignment")) |align_str| {
        if (std.mem.eql(u8, align_str, "Left")) {
            props.alignment = .left;
        } else if (std.mem.eql(u8, align_str, "Center")) {
            props.alignment = .center;
        } else if (std.mem.eql(u8, align_str, "Distributed")) {
            props.alignment = .distributed;
        } else if (std.mem.eql(u8, align_str, "NoControl")) {
            props.alignment = .no_control;
        }
    }
    
    return props;
}

// Export functions for FFI
export fn nExtract_SST_parse(xml_data: [*:0]const u8) ?*SharedStringTable {
    const allocator = std.heap.c_allocator;
    const xml_slice = std.mem.span(xml_data);
    
    var sst = SharedStringTable.parseFromXml(allocator, xml_slice) catch return null;
    const sst_ptr = allocator.create(SharedStringTable) catch return null;
    sst_ptr.* = sst;
    
    return sst_ptr;
}

export fn nExtract_SST_destroy(sst: *SharedStringTable) void {
    const allocator = sst.allocator;
    sst.deinit();
    allocator.destroy(sst);
}

export fn nExtract_SST_getString(sst: *const SharedStringTable, index: usize) ?[*:0]const u8 {
    const text = sst.getPlainText(index) orelse return null;
    // Return as C string (already null-terminated in our storage)
    return @ptrCast(text.ptr);
}

export fn nExtract_SST_getCount(sst: *const SharedStringTable) usize {
    return sst.count;
}

export fn nExtract_SST_getUniqueCount(sst: *const SharedStringTable) usize {
    return sst.unique_count;
}

// Tests
test "SST - parse simple strings" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="3" uniqueCount="3">
        \\  <si><t>Hello</t></si>
        \\  <si><t>World</t></si>
        \\  <si><t>Test</t></si>
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    try std.testing.expectEqual(@as(usize, 3), sst.count);
    try std.testing.expectEqual(@as(usize, 3), sst.unique_count);
    try std.testing.expectEqual(@as(usize, 3), sst.strings.items.len);
    
    const str1 = sst.getPlainText(0).?;
    try std.testing.expect(std.mem.eql(u8, str1, "Hello"));
    
    const str2 = sst.getPlainText(1).?;
    try std.testing.expect(std.mem.eql(u8, str2, "World"));
}

test "SST - parse rich text" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
        \\  <si>
        \\    <r>
        \\      <t>Normal </t>
        \\    </r>
        \\    <r>
        \\      <rPr>
        \\        <b/>
        \\        <rFont val="Calibri"/>
        \\      </rPr>
        \\      <t>Bold</t>
        \\    </r>
        \\  </si>
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    try std.testing.expectEqual(@as(usize, 1), sst.strings.items.len);
    
    const str = &sst.strings.items[0];
    try std.testing.expectEqual(SharedString.StringType.rich_text, str.type);
    try std.testing.expectEqual(@as(usize, 2), str.rich_text.items.len);
    
    // Check first run (normal)
    const run1 = &str.rich_text.items[0];
    try std.testing.expect(std.mem.eql(u8, run1.text, "Normal "));
    try std.testing.expect(run1.properties == null);
    
    // Check second run (bold)
    const run2 = &str.rich_text.items[1];
    try std.testing.expect(std.mem.eql(u8, run2.text, "Bold"));
    try std.testing.expect(run2.properties != null);
    try std.testing.expect(run2.properties.?.bold);
}

test "SST - text properties parsing" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
        \\  <si>
        \\    <r>
        \\      <rPr>
        \\        <b/>
        \\        <i/>
        \\        <u/>
        \\        <strike/>
        \\        <rFont val="Arial"/>
        \\        <sz val="12"/>
        \\        <color rgb="FF0000"/>
        \\      </rPr>
        \\      <t>Formatted</t>
        \\    </r>
        \\  </si>
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    const str = &sst.strings.items[0];
    const run = &str.rich_text.items[0];
    const props = run.properties.?;
    
    try std.testing.expect(props.bold);
    try std.testing.expect(props.italic);
    try std.testing.expect(props.underline);
    try std.testing.expect(props.strike);
    try std.testing.expect(std.mem.eql(u8, props.font_name.?, "Arial"));
    try std.testing.expectEqual(@as(f32, 12.0), props.font_size.?);
    try std.testing.expect(std.mem.eql(u8, props.color.?, "FF0000"));
}

test "SST - whitespace preservation" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
        \\  <si><t xml:space="preserve">  Leading and trailing  </t></si>
        \\  <si><t>No preserve</t></si>
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    const str1 = sst.getPlainText(0).?;
    try std.testing.expect(std.mem.eql(u8, str1, "  Leading and trailing  "));
    
    const str2 = sst.getPlainText(1).?;
    try std.testing.expect(std.mem.eql(u8, str2, "No preserve"));
}

test "SST - empty string table" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="0" uniqueCount="0">
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    try std.testing.expectEqual(@as(usize, 0), sst.count);
    try std.testing.expectEqual(@as(usize, 0), sst.unique_count);
    try std.testing.expectEqual(@as(usize, 0), sst.strings.items.len);
}

test "SST - out of bounds access" {
    const allocator = std.testing.allocator;
    
    const xml_data =
        \\<?xml version="1.0" encoding="UTF-8"?>
        \\<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
        \\  <si><t>Only one</t></si>
        \\</sst>
    ;
    
    var sst = try SharedStringTable.parseFromXml(allocator, xml_data);
    defer sst.deinit();
    
    // Valid access
    try std.testing.expect(sst.getPlainText(0) != null);
    
    // Out of bounds
    try std.testing.expect(sst.getPlainText(1) == null);
    try std.testing.expect(sst.getPlainText(100) == null);
}
