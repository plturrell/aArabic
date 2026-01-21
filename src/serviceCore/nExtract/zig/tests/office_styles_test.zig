// office_styles_test.zig - Comprehensive tests for Office Style System
// Part of nExtract - Day 19 Implementation

const std = @import("std");
const styles = @import("../parsers/office_styles.zig");
const testing = std.testing;

test "FontStyle - complete font properties" {
    const allocator = testing.allocator;
    var font = styles.FontStyle.init(allocator);
    defer font.deinit();

    font.name = try allocator.dupe(u8, "Calibri");
    font.family = .swiss;
    font.size = 11.0;
    font.bold = true;
    font.italic = false;
    font.underline = .single;
    font.strike = false;
    font.color = styles.Color.fromRgb(0xFF0000FF);
    font.charset = 1;
    font.scheme = .minor;

    try testing.expectEqualStrings("Calibri", font.name.?);
    try testing.expect(font.family == .swiss);
    try testing.expect(font.size.? == 11.0);
    try testing.expect(font.bold);
    try testing.expect(!font.italic);
    try testing.expect(font.underline == .single);
    try testing.expect(!font.strike);
    try testing.expect(font.color.?.type == .rgb);
    try testing.expect(font.charset.? == 1);
    try testing.expect(font.scheme.? == .minor);
}

test "FontStyle - clone font" {
    const allocator = testing.allocator;
    var original = styles.FontStyle.init(allocator);
    defer original.deinit();

    original.name = try allocator.dupe(u8, "Arial");
    original.size = 12.0;
    original.bold = true;
    original.italic = true;

    var cloned = try original.clone(allocator);
    defer cloned.deinit();

    try testing.expectEqualStrings("Arial", cloned.name.?);
    try testing.expect(cloned.size.? == 12.0);
    try testing.expect(cloned.bold);
    try testing.expect(cloned.italic);
}

test "Color - RGB color" {
    const red = styles.Color.fromRgb(0xFFFF0000);
    try testing.expect(red.type == .rgb);
    try testing.expect(red.value.rgb == 0xFFFF0000);

    const rgb = red.toRgb(null);
    try testing.expect(rgb == 0xFFFF0000);
}

test "Color - Theme color" {
    const allocator = testing.allocator;
    var theme = styles.Theme.init(allocator);
    defer theme.deinit();

    const accent1 = styles.Color.fromTheme(4);
    try testing.expect(accent1.type == .theme);
    try testing.expect(accent1.value.theme == 4);

    const rgb = accent1.toRgb(&theme);
    try testing.expect(rgb == theme.colors.accent1);
}

test "Color - Indexed color" {
    const indexed = styles.Color.fromIndexed(2);
    try testing.expect(indexed.type == .indexed);
    try testing.expect(indexed.value.indexed == 2);

    const rgb = indexed.toRgb(null);
    try testing.expect(rgb == 0xFFFF0000); // Red (index 2)
}

test "Color - Auto color" {
    const auto_color = styles.Color.auto();
    try testing.expect(auto_color.type == .auto);

    const rgb = auto_color.toRgb(null);
    try testing.expect(rgb == 0xFF000000); // Black (default)
}

test "BorderStyle - all borders" {
    const allocator = testing.allocator;
    var border = styles.BorderStyle.init(allocator);
    defer border.deinit();

    border.left = styles.BorderStyle.Border{
        .style = .thin,
        .color = styles.Color.fromRgb(0xFF000000),
    };
    border.right = styles.BorderStyle.Border{
        .style = .medium,
        .color = styles.Color.fromRgb(0xFFFF0000),
    };
    border.top = styles.BorderStyle.Border{
        .style = .thick,
        .color = styles.Color.fromRgb(0xFF00FF00),
    };
    border.bottom = styles.BorderStyle.Border{
        .style = .dashed,
        .color = styles.Color.fromRgb(0xFF0000FF),
    };
    border.diagonal = styles.BorderStyle.Border{
        .style = .dotted,
        .color = styles.Color.fromRgb(0xFFFFFF00),
    };
    border.diagonal_up = true;
    border.diagonal_down = false;

    try testing.expect(border.hasBorders());
    try testing.expect(border.left.?.style == .thin);
    try testing.expect(border.right.?.style == .medium);
    try testing.expect(border.top.?.style == .thick);
    try testing.expect(border.bottom.?.style == .dashed);
    try testing.expect(border.diagonal.?.style == .dotted);
    try testing.expect(border.diagonal_up);
    try testing.expect(!border.diagonal_down);
}

test "BorderStyle - no borders" {
    const allocator = testing.allocator;
    var border = styles.BorderStyle.init(allocator);
    defer border.deinit();

    try testing.expect(!border.hasBorders());
}

test "FillStyle - solid fill" {
    const yellow = styles.Color.fromRgb(0xFFFFFF00);
    const fill = styles.FillStyle.solid(yellow);

    try testing.expect(fill.type == .pattern);
    try testing.expect(fill.pattern == .solid);
    try testing.expect(fill.foreground_color.?.type == .rgb);
    try testing.expect(fill.foreground_color.?.value.rgb == 0xFFFFFF00);
}

test "FillStyle - pattern fills" {
    const fg = styles.Color.fromRgb(0xFF000000);
    const bg = styles.Color.fromRgb(0xFFFFFFFF);

    const fill = styles.FillStyle{
        .type = .pattern,
        .foreground_color = fg,
        .background_color = bg,
        .pattern = .medium_gray,
    };

    try testing.expect(fill.type == .pattern);
    try testing.expect(fill.pattern == .medium_gray);
    try testing.expect(fill.foreground_color != null);
    try testing.expect(fill.background_color != null);
}

test "FillStyle - none" {
    const fill = styles.FillStyle.none();
    try testing.expect(fill.type == .none);
    try testing.expect(fill.pattern == .none);
}

test "Alignment - all properties" {
    const alignment = styles.Alignment{
        .horizontal = .center,
        .vertical = .center,
        .text_rotation = 45,
        .wrap_text = true,
        .indent = 2,
        .shrink_to_fit = false,
        .reading_order = .left_to_right,
    };

    try testing.expect(alignment.horizontal == .center);
    try testing.expect(alignment.vertical == .center);
    try testing.expect(alignment.text_rotation.? == 45);
    try testing.expect(alignment.wrap_text);
    try testing.expect(alignment.indent == 2);
    try testing.expect(!alignment.shrink_to_fit);
    try testing.expect(alignment.reading_order == .left_to_right);
}

test "NumberFormat - built-in formats" {
    try testing.expect(styles.NumberFormat.isBuiltIn(0));
    try testing.expect(styles.NumberFormat.isBuiltIn(163));
    try testing.expect(!styles.NumberFormat.isBuiltIn(164));

    const general = styles.NumberFormat.getBuiltInFormat(styles.NumberFormat.GENERAL);
    try testing.expect(general != null);
    try testing.expectEqualStrings("General", general.?);

    const percent = styles.NumberFormat.getBuiltInFormat(styles.NumberFormat.PERCENT_D2);
    try testing.expect(percent != null);
    try testing.expectEqualStrings("0.00%", percent.?);

    const date = styles.NumberFormat.getBuiltInFormat(styles.NumberFormat.DATE);
    try testing.expect(date != null);
    try testing.expectEqualStrings("mm-dd-yy", date.?);

    const currency = styles.NumberFormat.getBuiltInFormat(styles.NumberFormat.CURRENCY_D2);
    try testing.expect(currency != null);
}

test "NumberFormat - custom format" {
    const allocator = testing.allocator;

    var fmt = try styles.NumberFormat.init(allocator, 164, "#,##0.00");
    defer fmt.deinit();

    try testing.expect(fmt.id == 164);
    try testing.expect(fmt.code != null);
    try testing.expectEqualStrings("#,##0.00", fmt.code.?);
    try testing.expect(!styles.NumberFormat.isBuiltIn(164));
}

test "CellFormat - complete format" {
    var cell_fmt = styles.CellFormat{
        .font_id = 0,
        .fill_id = 1,
        .border_id = 2,
        .number_format_id = styles.NumberFormat.CURRENCY_D2,
        .alignment = styles.Alignment{
            .horizontal = .right,
            .vertical = .center,
        },
        .apply_font = true,
        .apply_fill = true,
        .apply_border = true,
        .apply_number_format = true,
        .apply_alignment = true,
        .apply_protection = false,
        .protection = .{
            .locked = true,
            .hidden = false,
        },
    };

    try testing.expect(cell_fmt.font_id == 0);
    try testing.expect(cell_fmt.fill_id == 1);
    try testing.expect(cell_fmt.border_id == 2);
    try testing.expect(cell_fmt.number_format_id == styles.NumberFormat.CURRENCY_D2);
    try testing.expect(cell_fmt.alignment.?.horizontal == .right);
    try testing.expect(cell_fmt.apply_font);
    try testing.expect(cell_fmt.apply_fill);
    try testing.expect(cell_fmt.apply_border);
    try testing.expect(cell_fmt.apply_number_format);
    try testing.expect(cell_fmt.apply_alignment);
    try testing.expect(cell_fmt.protection.locked);
    try testing.expect(!cell_fmt.protection.hidden);
}

test "Theme - default colors" {
    const allocator = testing.allocator;
    var theme = styles.Theme.init(allocator);
    defer theme.deinit();

    try testing.expect(theme.colors.dark1 == 0xFF000000);
    try testing.expect(theme.colors.light1 == 0xFFFFFFFF);
    try testing.expect(theme.colors.accent1 == 0xFF4F81BD);
    try testing.expect(theme.colors.hyperlink == 0xFF0000FF);
    try testing.expect(theme.colors.followed_hyperlink == 0xFF800080);
}

test "Theme - get theme color by index" {
    const allocator = testing.allocator;
    var theme = styles.Theme.init(allocator);
    defer theme.deinit();

    try testing.expect(theme.getThemeColor(0) == theme.colors.dark1);
    try testing.expect(theme.getThemeColor(1) == theme.colors.light1);
    try testing.expect(theme.getThemeColor(4) == theme.colors.accent1);
    try testing.expect(theme.getThemeColor(5) == theme.colors.accent2);
    try testing.expect(theme.getThemeColor(10) == theme.colors.hyperlink);
    try testing.expect(theme.getThemeColor(11) == theme.colors.followed_hyperlink);
    try testing.expect(theme.getThemeColor(99) == 0xFF000000); // Out of range returns black
}

test "StyleSheet - creation and destruction" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    try testing.expect(sheet.fonts.items.len == 0);
    try testing.expect(sheet.fills.items.len == 0);
    try testing.expect(sheet.borders.items.len == 0);
    try testing.expect(sheet.cell_formats.items.len == 0);
    try testing.expect(sheet.number_formats.items.len == 0);
}

test "StyleSheet - add and retrieve fonts" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    var font1 = styles.FontStyle.init(allocator);
    font1.name = try allocator.dupe(u8, "Arial");
    font1.size = 11.0;
    font1.bold = true;
    try sheet.fonts.append(allocator, font1);

    var font2 = styles.FontStyle.init(allocator);
    font2.name = try allocator.dupe(u8, "Calibri");
    font2.size = 12.0;
    font2.italic = true;
    try sheet.fonts.append(allocator, font2);

    try testing.expect(sheet.fonts.items.len == 2);

    const retrieved1 = sheet.getFont(0);
    try testing.expect(retrieved1 != null);
    try testing.expect(retrieved1.?.bold);

    const retrieved2 = sheet.getFont(1);
    try testing.expect(retrieved2 != null);
    try testing.expect(retrieved2.?.italic);

    const invalid = sheet.getFont(99);
    try testing.expect(invalid == null);
}

test "StyleSheet - add and retrieve fills" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    const fill1 = styles.FillStyle.solid(styles.Color.fromRgb(0xFFFFFF00));
    try sheet.fills.append(allocator, fill1);

    const fill2 = styles.FillStyle.none();
    try sheet.fills.append(allocator, fill2);

    try testing.expect(sheet.fills.items.len == 2);

    const retrieved1 = sheet.getFill(0);
    try testing.expect(retrieved1 != null);
    try testing.expect(retrieved1.?.pattern == .solid);

    const retrieved2 = sheet.getFill(1);
    try testing.expect(retrieved2 != null);
    try testing.expect(retrieved2.?.type == .none);
}

test "StyleSheet - add and retrieve borders" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    var border1 = styles.BorderStyle.init(allocator);
    border1.left = styles.BorderStyle.Border{
        .style = .thin,
        .color = styles.Color.fromRgb(0xFF000000),
    };
    try sheet.borders.append(allocator, border1);

    var border2 = styles.BorderStyle.init(allocator);
    border2.top = styles.BorderStyle.Border{
        .style = .thick,
        .color = styles.Color.fromRgb(0xFFFF0000),
    };
    try sheet.borders.append(allocator, border2);

    try testing.expect(sheet.borders.items.len == 2);

    const retrieved1 = sheet.getBorder(0);
    try testing.expect(retrieved1 != null);
    try testing.expect(retrieved1.?.left.?.style == .thin);

    const retrieved2 = sheet.getBorder(1);
    try testing.expect(retrieved2 != null);
    try testing.expect(retrieved2.?.top.?.style == .thick);
}

test "StyleSheet - add and retrieve cell formats" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    const fmt1 = styles.CellFormat{
        .font_id = 0,
        .apply_font = true,
    };
    try sheet.cell_formats.append(allocator, fmt1);

    const fmt2 = styles.CellFormat{
        .fill_id = 1,
        .apply_fill = true,
    };
    try sheet.cell_formats.append(allocator, fmt2);

    try testing.expect(sheet.cell_formats.items.len == 2);

    const retrieved1 = sheet.getCellFormat(0);
    try testing.expect(retrieved1 != null);
    try testing.expect(retrieved1.?.font_id == 0);
    try testing.expect(retrieved1.?.apply_font);

    const retrieved2 = sheet.getCellFormat(1);
    try testing.expect(retrieved2 != null);
    try testing.expect(retrieved2.?.fill_id == 1);
    try testing.expect(retrieved2.?.apply_fill);
}

test "StyleSheet - add and retrieve number formats" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    var fmt1 = try styles.NumberFormat.init(allocator, 164, "#,##0.00");
    try sheet.number_formats.append(allocator, fmt1);

    var fmt2 = try styles.NumberFormat.init(allocator, 165, "0.00%");
    try sheet.number_formats.append(allocator, fmt2);

    try testing.expect(sheet.number_formats.items.len == 2);

    const retrieved1 = sheet.getNumberFormat(164);
    try testing.expect(retrieved1 != null);
    try testing.expect(retrieved1.?.id == 164);

    const retrieved2 = sheet.getNumberFormat(165);
    try testing.expect(retrieved2 != null);
    try testing.expect(retrieved2.?.id == 165);

    const builtin = sheet.getNumberFormat(0); // Built-in format
    try testing.expect(builtin == null); // Should return null, use NumberFormat.getBuiltInFormat instead
}

test "StyleSheet - complete style system" {
    const allocator = testing.allocator;
    var sheet = styles.StyleSheet.init(allocator);
    defer sheet.deinit();

    // Add default font
    var font = styles.FontStyle.init(allocator);
    font.name = try allocator.dupe(u8, "Calibri");
    font.size = 11.0;
    try sheet.fonts.append(allocator, font);

    // Add default fill
    const fill = styles.FillStyle.none();
    try sheet.fills.append(allocator, fill);

    // Add default border
    var border = styles.BorderStyle.init(allocator);
    try sheet.borders.append(allocator, border);

    // Add cell format
    const cell_fmt = styles.CellFormat{
        .font_id = 0,
        .fill_id = 0,
        .border_id = 0,
        .number_format_id = 0,
        .apply_font = true,
        .apply_fill = true,
        .apply_border = true,
    };
    try sheet.cell_formats.append(allocator, cell_fmt);

    // Add theme
    sheet.theme = styles.Theme.init(allocator);

    try testing.expect(sheet.fonts.items.len == 1);
    try testing.expect(sheet.fills.items.len == 1);
    try testing.expect(sheet.borders.items.len == 1);
    try testing.expect(sheet.cell_formats.items.len == 1);
    try testing.expect(sheet.theme != null);
}

test "FFI - StyleSheet creation and destruction" {
    const sheet = styles.nExtract_StyleSheet_create();
    try testing.expect(sheet != null);

    const font_count = styles.nExtract_StyleSheet_getFontCount(sheet.?);
    try testing.expect(font_count == 0);

    const fill_count = styles.nExtract_StyleSheet_getFillCount(sheet.?);
    try testing.expect(fill_count == 0);

    const border_count = styles.nExtract_StyleSheet_getBorderCount(sheet.?);
    try testing.expect(border_count == 0);

    const fmt_count = styles.nExtract_StyleSheet_getCellFormatCount(sheet.?);
    try testing.expect(fmt_count == 0);

    styles.nExtract_StyleSheet_destroy(sheet.?);
}
