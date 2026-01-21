const std = @import("std");
const testing = std.testing;
const ooxml = @import("../parsers/ooxml.zig");
const xlsx_sst = @import("../parsers/xlsx_sst.zig");
const office_styles = @import("../parsers/office_styles.zig");

// Test: Complex DOCX document with nested sections
test "complex DOCX with nested sections and headers" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Simulate complex DOCX structure
    var package = try ooxml.OOXMLPackage.create(allocator);
    defer package.deinit();

    // Add content types
    try package.content_types.put("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml", {});
    try package.content_types.put("application/vnd.openxmlformats-officedocument.wordprocessingml.header+xml", {});
    try package.content_types.put("application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml", {});

    // Add relationships
    var rel1 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"),
        .target = try allocator.dupe(u8, "word/document.xml"),
    };
    try package.relationships.append(rel1);

    var rel2 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId2"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/header"),
        .target = try allocator.dupe(u8, "word/header1.xml"),
    };
    try package.relationships.append(rel2);

    var rel3 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId3"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"),
        .target = try allocator.dupe(u8, "word/styles.xml"),
    };
    try package.relationships.append(rel3);

    // Test relationship lookup
    const doc_rel = package.getRelationshipByType("http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument");
    try testing.expect(doc_rel != null);
    try testing.expectEqualStrings("word/document.xml", doc_rel.?.target);

    const header_rel = package.getRelationshipByType("http://schemas.openxmlformats.org/officeDocument/2006/relationships/header");
    try testing.expect(header_rel != null);
    try testing.expectEqualStrings("word/header1.xml", header_rel.?.target);

    // Test multiple sections
    try testing.expect(package.relationships.items.len == 3);
}

// Test: Large XLSX spreadsheet with 10,000+ cells
test "large XLSX with many cells and shared strings" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create large shared string table
    var sst = try xlsx_sst.SharedStringTable.create(allocator);
    defer sst.deinit();

    // Add 10,000 unique strings
    var i: u32 = 0;
    while (i < 10000) : (i += 1) {
        var buf: [32]u8 = undefined;
        const str = try std.fmt.bufPrint(&buf, "Cell value {d}", .{i});
        _ = try sst.addString(try allocator.dupe(u8, str));
    }

    try testing.expectEqual(@as(usize, 10000), sst.strings.items.len);
    try testing.expectEqual(@as(u32, 10000), sst.count);

    // Test random access
    const str_0 = sst.getString(0);
    try testing.expect(str_0 != null);
    try testing.expectEqualStrings("Cell value 0", str_0.?.plain_text.?);

    const str_5000 = sst.getString(5000);
    try testing.expect(str_5000 != null);
    try testing.expectEqualStrings("Cell value 5000", str_5000.?.plain_text.?);

    const str_9999 = sst.getString(9999);
    try testing.expect(str_9999 != null);
    try testing.expectEqualStrings("Cell value 9999", str_9999.?.plain_text.?);

    // Test out of bounds
    const str_invalid = sst.getString(10000);
    try testing.expect(str_invalid == null);
}

// Test: XLSX with complex cell styles
test "XLSX with complex style inheritance" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Create style sheet
    var styles = try office_styles.StyleSheet.create(allocator);
    defer styles.deinit();

    // Add fonts
    var font1 = office_styles.FontStyle.create(allocator);
    font1.name = try allocator.dupe(u8, "Arial");
    font1.size = 12.0;
    font1.bold = true;
    _ = try styles.addFont(font1);

    var font2 = office_styles.FontStyle.create(allocator);
    font2.name = try allocator.dupe(u8, "Times New Roman");
    font2.size = 11.0;
    font2.italic = true;
    _ = try styles.addFont(font2);

    // Add fills
    var fill1 = office_styles.FillStyle.solid(office_styles.Color.rgb(0xFFFFFF00)); // Yellow
    _ = try styles.addFill(fill1);

    var fill2 = office_styles.FillStyle.solid(office_styles.Color.rgb(0xFF00FF00)); // Green
    _ = try styles.addFill(fill2);

    // Add borders
    var border1 = office_styles.BorderStyle{};
    border1.left = office_styles.Border{
        .style = .thin,
        .color = office_styles.Color.rgb(0xFF000000),
    };
    border1.right = border1.left;
    border1.top = border1.left;
    border1.bottom = border1.left;
    _ = try styles.addBorder(border1);

    // Add cell formats with different style combinations
    var xf1 = office_styles.CellFormat{
        .font_id = 0,
        .fill_id = 0,
        .border_id = 0,
        .number_format_id = 0,
        .alignment = null,
        .apply_font = true,
        .apply_fill = true,
        .apply_border = true,
        .apply_number_format = false,
        .apply_alignment = false,
    };
    _ = try styles.addCellFormat(xf1);

    var xf2 = office_styles.CellFormat{
        .font_id = 1,
        .fill_id = 1,
        .border_id = 0,
        .number_format_id = office_styles.NumberFormat.PERCENT_D2,
        .alignment = null,
        .apply_font = true,
        .apply_fill = true,
        .apply_border = false,
        .apply_number_format = true,
        .apply_alignment = false,
    };
    _ = try styles.addCellFormat(xf2);

    // Test style retrieval
    const retrieved_font = styles.getFont(0);
    try testing.expect(retrieved_font != null);
    try testing.expectEqualStrings("Arial", retrieved_font.?.name.?);
    try testing.expect(retrieved_font.?.bold);

    const retrieved_xf = styles.getCellFormat(1);
    try testing.expect(retrieved_xf != null);
    try testing.expectEqual(@as(u32, 1), retrieved_xf.?.font_id);
    try testing.expectEqual(@as(u32, 1), retrieved_xf.?.fill_id);
    try testing.expect(retrieved_xf.?.apply_number_format);
}

// Test: PPTX with nested shapes and groups
test "PPTX with nested shape groups" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var package = try ooxml.OOXMLPackage.create(allocator);
    defer package.deinit();

    // Simulate PPTX structure
    try package.content_types.put("application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml", {});
    try package.content_types.put("application/vnd.openxmlformats-officedocument.presentationml.slide+xml", {});
    try package.content_types.put("application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml", {});
    try package.content_types.put("application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml", {});

    // Add slide relationships
    var rel1 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"),
        .target = try allocator.dupe(u8, "ppt/slides/slide1.xml"),
    };
    try package.relationships.append(rel1);

    var rel2 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId2"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"),
        .target = try allocator.dupe(u8, "ppt/slides/slide2.xml"),
    };
    try package.relationships.append(rel2);

    var rel3 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId3"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster"),
        .target = try allocator.dupe(u8, "ppt/slideMasters/slideMaster1.xml"),
    };
    try package.relationships.append(rel3);

    // Test slide relationships
    try testing.expectEqual(@as(usize, 3), package.relationships.items.len);

    // Count slides
    var slide_count: u32 = 0;
    for (package.relationships.items) |rel| {
        if (std.mem.indexOf(u8, rel.type, "slide") != null and
            std.mem.indexOf(u8, rel.type, "slideMaster") == null)
        {
            slide_count += 1;
        }
    }
    try testing.expectEqual(@as(u32, 2), slide_count);
}

// Test: Malformed OOXML package recovery
test "malformed OOXML package with missing relationships" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var package = try ooxml.OOXMLPackage.create(allocator);
    defer package.deinit();

    // Add content types but NO relationships
    try package.content_types.put("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml", {});

    // Try to get main document relationship (should be null)
    const doc_rel = package.getRelationshipByType("http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument");
    try testing.expect(doc_rel == null);

    // Package should still be valid (just empty)
    try testing.expectEqual(@as(usize, 0), package.relationships.items.len);
    try testing.expectEqual(@as(usize, 1), package.content_types.count());
}

// Test: OOXML with image relationships
test "OOXML with image and media relationships" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var package = try ooxml.OOXMLPackage.create(allocator);
    defer package.deinit();

    // Add image content types
    try package.content_types.put("image/png", {});
    try package.content_types.put("image/jpeg", {});
    try package.content_types.put("video/mp4", {});

    // Add image relationships
    var img1 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId1"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
        .target = try allocator.dupe(u8, "word/media/image1.png"),
    };
    try package.relationships.append(img1);

    var img2 = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId2"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"),
        .target = try allocator.dupe(u8, "word/media/image2.jpg"),
    };
    try package.relationships.append(img2);

    var video = ooxml.Relationship{
        .id = try allocator.dupe(u8, "rId3"),
        .type = try allocator.dupe(u8, "http://schemas.openxmlformats.org/officeDocument/2006/relationships/video"),
        .target = try allocator.dupe(u8, "word/media/video1.mp4"),
    };
    try package.relationships.append(video);

    // Test image lookup
    const img_rel = package.getRelationshipById("rId1");
    try testing.expect(img_rel != null);
    try testing.expectEqualStrings("word/media/image1.png", img_rel.?.target);

    // Count media files
    var media_count: u32 = 0;
    for (package.relationships.items) |rel| {
        if (std.mem.indexOf(u8, rel.type, "image") != null or
            std.mem.indexOf(u8, rel.type, "video") != null)
        {
            media_count += 1;
        }
    }
    try testing.expectEqual(@as(u32, 3), media_count);
}

// Test: Rich text in shared strings with multiple runs
test "shared strings with complex rich text formatting" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var sst = try xlsx_sst.SharedStringTable.create(allocator);
    defer sst.deinit();

    // Create rich text with multiple runs
    var string1 = xlsx_sst.SharedString{
        .plain_text = null,
        .rich_text = std.ArrayList(xlsx_sst.RichTextRun).init(allocator),
        .phonetic = null,
        .allocator = allocator,
    };

    // Run 1: "Hello " (normal)
    var run1 = xlsx_sst.RichTextRun{
        .text = try allocator.dupe(u8, "Hello "),
        .font_name = null,
        .font_size = null,
        .bold = false,
        .italic = false,
        .underline = false,
        .color = null,
        .allocator = allocator,
    };
    try string1.rich_text.?.append(run1);

    // Run 2: "World" (bold, red)
    var run2 = xlsx_sst.RichTextRun{
        .text = try allocator.dupe(u8, "World"),
        .font_name = try allocator.dupe(u8, "Arial"),
        .font_size = 14.0,
        .bold = true,
        .italic = false,
        .underline = false,
        .color = 0xFFFF0000, // Red
        .allocator = allocator,
    };
    try string1.rich_text.?.append(run2);

    // Run 3: "!" (italic)
    var run3 = xlsx_sst.RichTextRun{
        .text = try allocator.dupe(u8, "!"),
        .font_name = null,
        .font_size = null,
        .bold = false,
        .italic = true,
        .underline = false,
        .color = null,
        .allocator = allocator,
    };
    try string1.rich_text.?.append(run3);

    _ = try sst.addString(string1);

    // Test retrieval
    const retrieved = sst.getString(0);
    try testing.expect(retrieved != null);
    try testing.expect(retrieved.?.rich_text != null);
    try testing.expectEqual(@as(usize, 3), retrieved.?.rich_text.?.items.len);

    // Test individual runs
    const r1 = &retrieved.?.rich_text.?.items[0];
    try testing.expectEqualStrings("Hello ", r1.text);
    try testing.expect(!r1.bold);

    const r2 = &retrieved.?.rich_text.?.items[1];
    try testing.expectEqualStrings("World", r2.text);
    try testing.expect(r2.bold);
    try testing.expectEqual(@as(?u32, 0xFFFF0000), r2.color);

    const r3 = &retrieved.?.rich_text.?.items[2];
    try testing.expectEqualStrings("!", r3.text);
    try testing.expect(r3.italic);
}

// Test: Number format resolution for various built-in formats
test "number format resolution for built-in formats" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    // Test various built-in formats
    const general_fmt = office_styles.NumberFormat.getBuiltInFormat(office_styles.NumberFormat.GENERAL);
    try testing.expect(general_fmt != null);
    try testing.expectEqualStrings("General", general_fmt.?);

    const percent_fmt = office_styles.NumberFormat.getBuiltInFormat(office_styles.NumberFormat.PERCENT_D2);
    try testing.expect(percent_fmt != null);
    try testing.expectEqualStrings("0.00%", percent_fmt.?);

    const date_fmt = office_styles.NumberFormat.getBuiltInFormat(office_styles.NumberFormat.DATE);
    try testing.expect(date_fmt != null);
    try testing.expectEqualStrings("mm-dd-yy", date_fmt.?);

    const currency_fmt = office_styles.NumberFormat.getBuiltInFormat(office_styles.NumberFormat.CURRENCY_D2);
    try testing.expect(currency_fmt != null);
    try testing.expectEqualStrings("$#,##0.00", currency_fmt.?);

    const scientific_fmt = office_styles.NumberFormat.getBuiltInFormat(11); // Scientific
    try testing.expect(scientific_fmt != null);
    try testing.expectEqualStrings("0.00E+00", scientific_fmt.?);

    const time_fmt = office_styles.NumberFormat.getBuiltInFormat(20); // h:mm
    try testing.expect(time_fmt != null);
    try testing.expectEqualStrings("h:mm", time_fmt.?);

    // Test invalid format ID
    const invalid_fmt = office_styles.NumberFormat.getBuiltInFormat(999);
    try testing.expect(invalid_fmt == null);
}

// Test: Theme color resolution with all theme colors
test "theme color resolution with complete theme" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var theme = office_styles.Theme.createDefault(allocator);
    defer theme.deinit();

    // Test all 12 theme colors
    const dark1 = theme.getThemeColor(0);
    try testing.expectEqual(@as(u32, 0xFF000000), dark1); // Black

    const light1 = theme.getThemeColor(1);
    try testing.expectEqual(@as(u32, 0xFFFFFFFF), light1); // White

    const dark2 = theme.getThemeColor(2);
    try testing.expectEqual(@as(u32, 0xFF1F497D), dark2); // Dark blue

    const light2 = theme.getThemeColor(3);
    try testing.expectEqual(@as(u32, 0xFFEEECE1), light2); // Light gray

    const accent1 = theme.getThemeColor(4);
    try testing.expectEqual(@as(u32, 0xFF4F81BD), accent1); // Blue

    const accent2 = theme.getThemeColor(5);
    try testing.expectEqual(@as(u32, 0xFFC0504D), accent2); // Red

    const accent3 = theme.getThemeColor(6);
    try testing.expectEqual(@as(u32, 0xFF9BBB59), accent3); // Green

    const accent4 = theme.getThemeColor(7);
    try testing.expectEqual(@as(u32, 0xFF8064A2), accent4); // Purple

    const accent5 = theme.getThemeColor(8);
    try testing.expectEqual(@as(u32, 0xFF4BACC6), accent5); // Aqua

    const accent6 = theme.getThemeColor(9);
    try testing.expectEqual(@as(u32, 0xFFF79646), accent6); // Orange

    const hyperlink = theme.getThemeColor(10);
    try testing.expectEqual(@as(u32, 0xFF0000FF), hyperlink); // Blue

    const followed_hyperlink = theme.getThemeColor(11);
    try testing.expectEqual(@as(u32, 0xFF800080), followed_hyperlink); // Purple

    // Test out of bounds
    const invalid = theme.getThemeColor(12);
    try testing.expectEqual(@as(u32, 0xFF000000), invalid); // Default to black
}

// Test: Border style with diagonal borders
test "border style with diagonal borders" {
    var border = office_styles.BorderStyle{};

    // Set all sides
    border.left = office_styles.Border{
        .style = .thin,
        .color = office_styles.Color.rgb(0xFF000000),
    };
    border.right = border.left;
    border.top = border.left;
    border.bottom = border.left;

    // Set diagonal
    border.diagonal = office_styles.Border{
        .style = .medium,
        .color = office_styles.Color.rgb(0xFFFF0000),
    };
    border.diagonal_up = true;
    border.diagonal_down = false;

    try testing.expect(border.hasBorders());
    try testing.expect(border.left != null);
    try testing.expect(border.diagonal != null);
    try testing.expect(border.diagonal_up);
    try testing.expect(!border.diagonal_down);
}

// Test: Cell alignment with all properties
test "cell alignment with complete properties" {
    var alignment = office_styles.Alignment{
        .horizontal = .center,
        .vertical = .middle,
        .text_rotation = 45,
        .wrap_text = true,
        .indent = 2,
        .shrink_to_fit = false,
        .reading_order = .left_to_right,
    };

    try testing.expectEqual(office_styles.HorizontalAlignment.center, alignment.horizontal);
    try testing.expectEqual(office_styles.VerticalAlignment.middle, alignment.vertical);
    try testing.expectEqual(@as(i16, 45), alignment.text_rotation);
    try testing.expect(alignment.wrap_text);
    try testing.expectEqual(@as(u8, 2), alignment.indent);
    try testing.expect(!alignment.shrink_to_fit);
    try testing.expectEqual(office_styles.ReadingOrder.left_to_right, alignment.reading_order);
}

// Test: Memory safety with large style sheet
test "memory safety with large style sheet" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const allocator = arena.allocator();

    var styles = try office_styles.StyleSheet.create(allocator);
    defer styles.deinit();

    // Add 1000 fonts
    var i: u32 = 0;
    while (i < 1000) : (i += 1) {
        var font = office_styles.FontStyle.create(allocator);
        var buf: [32]u8 = undefined;
        font.name = try allocator.dupe(u8, try std.fmt.bufPrint(&buf, "Font{d}", .{i}));
        font.size = @as(f32, @floatFromInt(i % 20 + 8));
        _ = try styles.addFont(font);
    }

    // Add 1000 fills
    i = 0;
    while (i < 1000) : (i += 1) {
        const color = @as(u32, i * 0x010101) | 0xFF000000;
        const fill = office_styles.FillStyle.solid(office_styles.Color.rgb(color));
        _ = try styles.addFill(fill);
    }

    // Verify counts
    try testing.expectEqual(@as(usize, 1000), styles.fonts.items.len);
    try testing.expectEqual(@as(usize, 1000), styles.fills.items.len);

    // Random access test
    const font_500 = styles.getFont(500);
    try testing.expect(font_500 != null);
    try testing.expectEqualStrings("Font500", font_500.?.name.?);
}
