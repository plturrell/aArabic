// office_styles.zig - Office Format Style System Parser
// Implements font, color, border styles, number formatting, conditional formatting, and theme parsing
// Part of nExtract - Day 19 Implementation

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

// ============================================================================
// Font Styles
// ============================================================================

pub const FontStyle = struct {
    name: ?[]const u8 = null,
    family: FontFamily = .swiss,
    size: ?f32 = null,
    bold: bool = false,
    italic: bool = false,
    underline: UnderlineType = .none,
    strike: bool = false,
    color: ?Color = null,
    charset: ?u8 = null,
    scheme: ?FontScheme = null,
    allocator: Allocator,

    pub const FontFamily = enum {
        not_applicable,
        roman,
        swiss,
        modern,
        script,
        decorative,
    };

    pub const UnderlineType = enum {
        none,
        single,
        double,
        single_accounting,
        double_accounting,
    };

    pub const FontScheme = enum {
        none,
        major,
        minor,
    };

    pub fn init(allocator: Allocator) FontStyle {
        return FontStyle{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *FontStyle) void {
        if (self.name) |name| {
            self.allocator.free(name);
        }
    }

    pub fn clone(self: *const FontStyle, allocator: Allocator) !FontStyle {
        var new_style = FontStyle.init(allocator);
        if (self.name) |name| {
            new_style.name = try allocator.dupe(u8, name);
        }
        new_style.family = self.family;
        new_style.size = self.size;
        new_style.bold = self.bold;
        new_style.italic = self.italic;
        new_style.underline = self.underline;
        new_style.strike = self.strike;
        new_style.color = self.color;
        new_style.charset = self.charset;
        new_style.scheme = self.scheme;
        return new_style;
    }
};

// ============================================================================
// Color Support
// ============================================================================

pub const Color = struct {
    type: ColorType,
    value: ColorValue,

    pub const ColorType = enum {
        rgb,
        theme,
        indexed,
        auto,
    };

    pub const ColorValue = union {
        rgb: u32, // ARGB format (0xAARRGGBB)
        theme: u8, // Theme color index (0-11)
        indexed: u8, // Indexed color (0-63)
        auto: void,
    };

    pub fn fromRgb(rgb: u32) Color {
        return Color{
            .type = .rgb,
            .value = .{ .rgb = rgb },
        };
    }

    pub fn fromTheme(index: u8) Color {
        return Color{
            .type = .theme,
            .value = .{ .theme = index },
        };
    }

    pub fn fromIndexed(index: u8) Color {
        return Color{
            .type = .indexed,
            .value = .{ .indexed = index },
        };
    }

    pub fn auto() Color {
        return Color{
            .type = .auto,
            .value = .{ .auto = {} },
        };
    }

    pub fn toRgb(self: Color, theme: ?*const Theme) u32 {
        return switch (self.type) {
            .rgb => self.value.rgb,
            .theme => if (theme) |t| t.getThemeColor(self.value.theme) else 0xFF000000,
            .indexed => getIndexedColor(self.value.indexed),
            .auto => 0xFF000000, // Black
        };
    }

    fn getIndexedColor(index: u8) u32 {
        // Excel's default indexed color palette
        const indexed_colors = [_]u32{
            0xFF000000, 0xFFFFFFFF, 0xFFFF0000, 0xFF00FF00, 0xFF0000FF, 0xFFFFFF00,
            0xFFFF00FF, 0xFF00FFFF, 0xFF000000, 0xFFFFFFFF, 0xFFFF0000, 0xFF00FF00,
            0xFF0000FF, 0xFFFFFF00, 0xFFFF00FF, 0xFF00FFFF, 0xFF800000, 0xFF008000,
            0xFF000080, 0xFF808000, 0xFF800080, 0xFF008080, 0xFFC0C0C0, 0xFF808080,
            0xFF9999FF, 0xFF993366, 0xFFFFFFCC, 0xFFCCFFFF, 0xFF660066, 0xFFFF8080,
            0xFF0066CC, 0xFFCCCCFF, 0xFF000080, 0xFFFF00FF, 0xFFFFFF00, 0xFF00FFFF,
            0xFF800080, 0xFF800000, 0xFF008080, 0xFF0000FF, 0xFF00CCFF, 0xFFCCFFFF,
            0xFFCCFFCC, 0xFFFFFF99, 0xFF99CCFF, 0xFFFF99CC, 0xFFCC99FF, 0xFFFFCC99,
            0xFF3366FF, 0xFF33CCCC, 0xFF99CC00, 0xFFFFCC00, 0xFFFF9900, 0xFFFF6600,
            0xFF666699, 0xFF969696, 0xFF003366, 0xFF339966, 0xFF003300, 0xFF333300,
            0xFF993300, 0xFF993366, 0xFF333399, 0xFF333333,
        };
        if (index < indexed_colors.len) {
            return indexed_colors[index];
        }
        return 0xFF000000;
    }
};

// ============================================================================
// Border Styles
// ============================================================================

pub const BorderStyle = struct {
    left: ?Border = null,
    right: ?Border = null,
    top: ?Border = null,
    bottom: ?Border = null,
    diagonal: ?Border = null,
    diagonal_up: bool = false,
    diagonal_down: bool = false,
    allocator: Allocator,

    pub const Border = struct {
        style: LineStyle,
        color: ?Color = null,
    };

    pub const LineStyle = enum {
        none,
        thin,
        medium,
        dashed,
        dotted,
        thick,
        double,
        hair,
        medium_dashed,
        dash_dot,
        medium_dash_dot,
        dash_dot_dot,
        medium_dash_dot_dot,
        slant_dash_dot,
    };

    pub fn init(allocator: Allocator) BorderStyle {
        return BorderStyle{
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *BorderStyle) void {
        _ = self;
    }

    pub fn hasBorders(self: *const BorderStyle) bool {
        return self.left != null or self.right != null or
            self.top != null or self.bottom != null or self.diagonal != null;
    }
};

// ============================================================================
// Fill Styles
// ============================================================================

pub const FillStyle = struct {
    type: FillType,
    foreground_color: ?Color = null,
    background_color: ?Color = null,
    pattern: PatternType = .none,

    pub const FillType = enum {
        none,
        pattern,
        gradient,
    };

    pub const PatternType = enum {
        none,
        solid,
        medium_gray,
        dark_gray,
        light_gray,
        dark_horizontal,
        dark_vertical,
        dark_down,
        dark_up,
        dark_grid,
        dark_trellis,
        light_horizontal,
        light_vertical,
        light_down,
        light_up,
        light_grid,
        light_trellis,
        gray_125,
        gray_0625,
    };

    pub fn solid(color: Color) FillStyle {
        return FillStyle{
            .type = .pattern,
            .foreground_color = color,
            .pattern = .solid,
        };
    }

    pub fn none() FillStyle {
        return FillStyle{
            .type = .none,
            .pattern = .none,
        };
    }
};

// ============================================================================
// Alignment
// ============================================================================

pub const Alignment = struct {
    horizontal: HorizontalAlignment = .general,
    vertical: VerticalAlignment = .bottom,
    text_rotation: ?i8 = null, // -90 to 90, or 255 for vertical
    wrap_text: bool = false,
    indent: u8 = 0,
    shrink_to_fit: bool = false,
    reading_order: ReadingOrder = .context,

    pub const HorizontalAlignment = enum {
        general,
        left,
        center,
        right,
        fill,
        justify,
        center_continuous,
        distributed,
    };

    pub const VerticalAlignment = enum {
        top,
        center,
        bottom,
        justify,
        distributed,
    };

    pub const ReadingOrder = enum {
        context,
        left_to_right,
        right_to_left,
    };
};

// ============================================================================
// Number Formats
// ============================================================================

pub const NumberFormat = struct {
    id: u32,
    code: ?[]const u8 = null,
    allocator: Allocator,

    // Built-in format IDs (0-163 reserved by Excel)
    pub const GENERAL: u32 = 0;
    pub const NUMBER: u32 = 1; // 0
    pub const NUMBER_D2: u32 = 2; // 0.00
    pub const NUMBER_SEP: u32 = 3; // #,##0
    pub const NUMBER_SEP_D2: u32 = 4; // #,##0.00
    pub const CURRENCY: u32 = 5; // $#,##0_);($#,##0)
    pub const CURRENCY_D2: u32 = 7; // $#,##0.00_);($#,##0.00)
    pub const PERCENT: u32 = 9; // 0%
    pub const PERCENT_D2: u32 = 10; // 0.00%
    pub const SCIENTIFIC: u32 = 11; // 0.00E+00
    pub const FRACTION: u32 = 12; // # ?/?
    pub const FRACTION_D2: u32 = 13; // # ??/??
    pub const DATE: u32 = 14; // mm-dd-yy
    pub const DATE_D_MON_YY: u32 = 15; // d-mmm-yy
    pub const DATE_D_MON: u32 = 16; // d-mmm
    pub const DATE_MON_YY: u32 = 17; // mmm-yy
    pub const TIME: u32 = 18; // h:mm AM/PM
    pub const TIME_AMPM: u32 = 19; // h:mm:ss AM/PM
    pub const TIME_HMS: u32 = 20; // h:mm
    pub const TIME_HMSS: u32 = 21; // h:mm:ss
    pub const DATETIME: u32 = 22; // m/d/yy h:mm

    pub fn init(allocator: Allocator, id: u32, code: ?[]const u8) !NumberFormat {
        var fmt = NumberFormat{
            .id = id,
            .code = null,
            .allocator = allocator,
        };
        if (code) |c| {
            fmt.code = try allocator.dupe(u8, c);
        }
        return fmt;
    }

    pub fn deinit(self: *NumberFormat) void {
        if (self.code) |code| {
            self.allocator.free(code);
        }
    }

    pub fn isBuiltIn(id: u32) bool {
        return id <= 163;
    }

    pub fn getBuiltInFormat(id: u32) ?[]const u8 {
        return switch (id) {
            GENERAL => "General",
            NUMBER => "0",
            NUMBER_D2 => "0.00",
            NUMBER_SEP => "#,##0",
            NUMBER_SEP_D2 => "#,##0.00",
            PERCENT => "0%",
            PERCENT_D2 => "0.00%",
            SCIENTIFIC => "0.00E+00",
            DATE => "mm-dd-yy",
            DATE_D_MON_YY => "d-mmm-yy",
            DATE_D_MON => "d-mmm",
            DATE_MON_YY => "mmm-yy",
            TIME => "h:mm AM/PM",
            TIME_AMPM => "h:mm:ss AM/PM",
            TIME_HMS => "h:mm",
            TIME_HMSS => "h:mm:ss",
            DATETIME => "m/d/yy h:mm",
            else => null,
        };
    }
};

// ============================================================================
// Cell Format (CellXf)
// ============================================================================

pub const CellFormat = struct {
    font_id: ?u32 = null,
    fill_id: ?u32 = null,
    border_id: ?u32 = null,
    number_format_id: u32 = 0,
    alignment: ?Alignment = null,
    apply_font: bool = false,
    apply_fill: bool = false,
    apply_border: bool = false,
    apply_number_format: bool = false,
    apply_alignment: bool = false,
    apply_protection: bool = false,
    protection: Protection = .{},

    pub const Protection = struct {
        locked: bool = true,
        hidden: bool = false,
    };
};

// ============================================================================
// Theme
// ============================================================================

pub const Theme = struct {
    name: ?[]const u8 = null,
    colors: ThemeColors,
    allocator: Allocator,

    pub const ThemeColors = struct {
        // Standard theme colors (indices 0-11)
        dark1: u32 = 0xFF000000, // Background 1
        light1: u32 = 0xFFFFFFFF, // Text 1
        dark2: u32 = 0xFF1F497D, // Background 2
        light2: u32 = 0xFFEEECE1, // Text 2
        accent1: u32 = 0xFF4F81BD,
        accent2: u32 = 0xFFC0504D,
        accent3: u32 = 0xFF9BBB59,
        accent4: u32 = 0xFF8064A2,
        accent5: u32 = 0xFF4BACC6,
        accent6: u32 = 0xFFF79646,
        hyperlink: u32 = 0xFF0000FF,
        followed_hyperlink: u32 = 0xFF800080,
    };

    pub fn init(allocator: Allocator) Theme {
        return Theme{
            .allocator = allocator,
            .colors = ThemeColors{},
        };
    }

    pub fn deinit(self: *Theme) void {
        if (self.name) |name| {
            self.allocator.free(name);
        }
    }

    pub fn getThemeColor(self: *const Theme, index: u8) u32 {
        return switch (index) {
            0 => self.colors.dark1,
            1 => self.colors.light1,
            2 => self.colors.dark2,
            3 => self.colors.light2,
            4 => self.colors.accent1,
            5 => self.colors.accent2,
            6 => self.colors.accent3,
            7 => self.colors.accent4,
            8 => self.colors.accent5,
            9 => self.colors.accent6,
            10 => self.colors.hyperlink,
            11 => self.colors.followed_hyperlink,
            else => 0xFF000000,
        };
    }
};

// ============================================================================
// Conditional Formatting
// ============================================================================

pub const ConditionalFormat = struct {
    type: FormatType,
    priority: u32 = 0,
    stop_if_true: bool = false,
    rule: Rule,
    dxf_id: ?u32 = null, // Differential format ID
    allocator: Allocator,

    pub const FormatType = enum {
        cell_is,
        expression,
        color_scale,
        data_bar,
        icon_set,
        top10,
        unique_values,
        duplicate_values,
        contains_text,
        not_contains_text,
        begins_with,
        ends_with,
        contains_blanks,
        not_contains_blanks,
        contains_errors,
        not_contains_errors,
        time_period,
        above_average,
    };

    pub const Rule = union(FormatType) {
        cell_is: CellIsRule,
        expression: ExpressionRule,
        color_scale: ColorScaleRule,
        data_bar: DataBarRule,
        icon_set: IconSetRule,
        top10: Top10Rule,
        unique_values: void,
        duplicate_values: void,
        contains_text: TextRule,
        not_contains_text: TextRule,
        begins_with: TextRule,
        ends_with: TextRule,
        contains_blanks: void,
        not_contains_blanks: void,
        contains_errors: void,
        not_contains_errors: void,
        time_period: TimePeriodRule,
        above_average: AboveAverageRule,
    };

    pub const CellIsRule = struct {
        operator: Operator,
        formula: []const u8,

        pub const Operator = enum {
            less_than,
            less_than_or_equal,
            equal,
            not_equal,
            greater_than_or_equal,
            greater_than,
            between,
            not_between,
        };
    };

    pub const ExpressionRule = struct {
        formula: []const u8,
    };

    pub const ColorScaleRule = struct {
        min_value: ColorScaleValue,
        mid_value: ?ColorScaleValue,
        max_value: ColorScaleValue,
    };

    pub const ColorScaleValue = struct {
        type: ValueType,
        value: ?[]const u8,
        color: Color,

        pub const ValueType = enum {
            min,
            max,
            number,
            percent,
            formula,
            percentile,
        };
    };

    pub const DataBarRule = struct {
        min_value: DataBarValue,
        max_value: DataBarValue,
        color: Color,
        show_value: bool = true,
    };

    pub const DataBarValue = struct {
        type: ValueType,
        value: ?[]const u8,

        pub const ValueType = enum {
            min,
            max,
            number,
            percent,
            formula,
            percentile,
            automatic,
        };
    };

    pub const IconSetRule = struct {
        icon_set: IconSet,
        show_value: bool = true,
        reverse: bool = false,
        values: []IconSetValue,

        pub const IconSet = enum {
            three_arrows,
            three_arrows_gray,
            three_flags,
            three_traffic_lights,
            three_signs,
            three_symbols,
            three_symbols2,
            four_arrows,
            four_arrows_gray,
            four_red_to_black,
            four_rating,
            four_traffic_lights,
            five_arrows,
            five_arrows_gray,
            five_rating,
            five_quarters,
        };
    };

    pub const IconSetValue = struct {
        type: ValueType,
        value: ?[]const u8,
        greater_than_or_equal: bool = true,

        pub const ValueType = enum {
            number,
            percent,
            formula,
            percentile,
        };
    };

    pub const Top10Rule = struct {
        percent: bool = false,
        bottom: bool = false,
        rank: u32 = 10,
    };

    pub const TextRule = struct {
        text: []const u8,
    };

    pub const TimePeriodRule = struct {
        period: TimePeriod,

        pub const TimePeriod = enum {
            today,
            yesterday,
            tomorrow,
            last7_days,
            this_month,
            last_month,
            next_month,
            this_week,
            last_week,
            next_week,
        };
    };

    pub const AboveAverageRule = struct {
        above_average: bool = true,
        equal_average: bool = false,
        std_dev: ?u32 = null,
    };

    pub fn init(allocator: Allocator, format_type: FormatType) ConditionalFormat {
        return ConditionalFormat{
            .type = format_type,
            .rule = undefined, // Must be set by caller
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *ConditionalFormat) void {
        _ = self;
        // Cleanup allocated strings in rules if needed
    }
};

// ============================================================================
// Style Sheet Container
// ============================================================================

pub const StyleSheet = struct {
    fonts: ArrayList(FontStyle),
    fills: ArrayList(FillStyle),
    borders: ArrayList(BorderStyle),
    cell_formats: ArrayList(CellFormat),
    number_formats: ArrayList(NumberFormat),
    theme: ?Theme = null,
    allocator: Allocator,

    pub fn init(allocator: Allocator) StyleSheet {
        return StyleSheet{
            .fonts = ArrayList(FontStyle){},
            .fills = ArrayList(FillStyle){},
            .borders = ArrayList(BorderStyle){},
            .cell_formats = ArrayList(CellFormat){},
            .number_formats = ArrayList(NumberFormat){},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *StyleSheet) void {
        for (self.fonts.items) |*font| {
            font.deinit();
        }
        self.fonts.deinit(self.allocator);

        self.fills.deinit(self.allocator);

        for (self.borders.items) |*border| {
            border.deinit();
        }
        self.borders.deinit(self.allocator);

        self.cell_formats.deinit(self.allocator);

        for (self.number_formats.items) |*fmt| {
            fmt.deinit();
        }
        self.number_formats.deinit(self.allocator);

        if (self.theme) |*theme| {
            theme.deinit();
        }
    }

    pub fn getFont(self: *const StyleSheet, id: u32) ?*const FontStyle {
        if (id < self.fonts.items.len) {
            return &self.fonts.items[id];
        }
        return null;
    }

    pub fn getFill(self: *const StyleSheet, id: u32) ?*const FillStyle {
        if (id < self.fills.items.len) {
            return &self.fills.items[id];
        }
        return null;
    }

    pub fn getBorder(self: *const StyleSheet, id: u32) ?*const BorderStyle {
        if (id < self.borders.items.len) {
            return &self.borders.items[id];
        }
        return null;
    }

    pub fn getCellFormat(self: *const StyleSheet, id: u32) ?*const CellFormat {
        if (id < self.cell_formats.items.len) {
            return &self.cell_formats.items[id];
        }
        return null;
    }

    pub fn getNumberFormat(self: *const StyleSheet, id: u32) ?*const NumberFormat {
        // Check custom formats first
        for (self.number_formats.items) |*fmt| {
            if (fmt.id == id) {
                return fmt;
            }
        }
        // Return null for built-in formats (caller should use NumberFormat.getBuiltInFormat)
        return null;
    }
};

// ============================================================================
// FFI Exports
// ============================================================================

export fn nExtract_StyleSheet_create() ?*StyleSheet {
    const allocator = std.heap.c_allocator;
    const sheet = allocator.create(StyleSheet) catch return null;
    sheet.* = StyleSheet.init(allocator);
    return sheet;
}

export fn nExtract_StyleSheet_destroy(sheet: *StyleSheet) void {
    const allocator = sheet.allocator;
    sheet.deinit();
    allocator.destroy(sheet);
}

export fn nExtract_StyleSheet_getFontCount(sheet: *const StyleSheet) usize {
    return sheet.fonts.items.len;
}

export fn nExtract_StyleSheet_getFillCount(sheet: *const StyleSheet) usize {
    return sheet.fills.items.len;
}

export fn nExtract_StyleSheet_getBorderCount(sheet: *const StyleSheet) usize {
    return sheet.borders.items.len;
}

export fn nExtract_StyleSheet_getCellFormatCount(sheet: *const StyleSheet) usize {
    return sheet.cell_formats.items.len;
}

// ============================================================================
// Tests
// ============================================================================

test "Style - font style creation" {
    const allocator = std.testing.allocator;
    var font = FontStyle.init(allocator);
    defer font.deinit();

    font.bold = true;
    font.italic = true;
    font.size = 12.0;
    font.underline = .single;

    try std.testing.expect(font.bold);
    try std.testing.expect(font.italic);
    try std.testing.expect(font.size.? == 12.0);
    try std.testing.expect(font.underline == .single);
}

test "Style - color conversion" {
    const rgb_color = Color.fromRgb(0xFFFF0000); // Red
    try std.testing.expect(rgb_color.type == .rgb);
    try std.testing.expect(rgb_color.value.rgb == 0xFFFF0000);

    const theme_color = Color.fromTheme(4); // Accent1
    try std.testing.expect(theme_color.type == .theme);
    try std.testing.expect(theme_color.value.theme == 4);

    const theme = Theme.init(std.testing.allocator);
    const resolved = theme_color.toRgb(&theme);
    try std.testing.expect(resolved == theme.colors.accent1);
}

test "Style - border style" {
    const allocator = std.testing.allocator;
    var border = BorderStyle.init(allocator);
    defer border.deinit();

    border.left = BorderStyle.Border{
        .style = .thin,
        .color = Color.fromRgb(0xFF000000),
    };
    border.top = BorderStyle.Border{
        .style = .thick,
        .color = Color.fromRgb(0xFFFF0000),
    };

    try std.testing.expect(border.hasBorders());
    try std.testing.expect(border.left.?.style == .thin);
    try std.testing.expect(border.top.?.style == .thick);
}

test "Style - fill patterns" {
    const solid_fill = FillStyle.solid(Color.fromRgb(0xFFFFFF00));
    try std.testing.expect(solid_fill.type == .pattern);
    try std.testing.expect(solid_fill.pattern == .solid);
    try std.testing.expect(solid_fill.foreground_color.?.type == .rgb);

    const none_fill = FillStyle.none();
    try std.testing.expect(none_fill.type == .none);
    try std.testing.expect(none_fill.pattern == .none);
}

test "Style - number formats" {
    const allocator = std.testing.allocator;

    var fmt = try NumberFormat.init(allocator, 164, "#,##0.00");
    defer fmt.deinit();

    try std.testing.expect(fmt.id == 164);
    try std.testing.expect(!NumberFormat.isBuiltIn(164));

    const builtin = NumberFormat.getBuiltInFormat(NumberFormat.PERCENT_D2);
    try std.testing.expect(builtin != null);
    try std.testing.expectEqualStrings("0.00%", builtin.?);
}

test "Style - cell format" {
    const cell_fmt = CellFormat{
        .font_id = 0,
        .fill_id = 1,
        .border_id = 2,
        .number_format_id = NumberFormat.CURRENCY_D2,
        .apply_font = true,
        .apply_fill = true,
        .apply_border = true,
        .apply_number_format = true,
    };

    try std.testing.expect(cell_fmt.font_id == 0);
    try std.testing.expect(cell_fmt.fill_id == 1);
    try std.testing.expect(cell_fmt.border_id == 2);
    try std.testing.expect(cell_fmt.apply_font);
}

test "Style - style sheet" {
    const allocator = std.testing.allocator;
    var sheet = StyleSheet.init(allocator);
    defer sheet.deinit();

    // Add a font
    var font = FontStyle.init(allocator);
    font.name = try allocator.dupe(u8, "Arial");
    font.size = 11.0;
    font.bold = true;
    try sheet.fonts.append(allocator, font);

    // Add a fill
    const fill = FillStyle.solid(Color.fromRgb(0xFFFFFF00));
    try sheet.fills.append(allocator, fill);

    // Add a border
    var border = BorderStyle.init(allocator);
    border.left = BorderStyle.Border{
        .style = .thin,
        .color = Color.fromRgb(0xFF000000),
    };
    try sheet.borders.append(allocator, border);

    try std.testing.expect(sheet.fonts.items.len == 1);
    try std.testing.expect(sheet.fills.items.len == 1);
    try std.testing.expect(sheet.borders.items.len == 1);

    const retrieved_font = sheet.getFont(0);
    try std.testing.expect(retrieved_font != null);
    try std.testing.expect(retrieved_font.?.bold);
}

test "Style - theme colors" {
    const allocator = std.testing.allocator;
    var theme = Theme.init(allocator);
    defer theme.deinit();

    const accent1 = theme.getThemeColor(4);
    try std.testing.expect(accent1 == theme.colors.accent1);

    const hyperlink = theme.getThemeColor(10);
    try std.testing.expect(hyperlink == theme.colors.hyperlink);
}
