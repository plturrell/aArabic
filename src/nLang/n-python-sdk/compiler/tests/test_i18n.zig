// Mojo SDK - i18n Tests
// Comprehensive test suite for internationalization

const std = @import("std");
const i18n = @import("../i18n/i18n.zig");
const plurals = @import("../i18n/plurals.zig");
const locale_detect = @import("../i18n/locale_detect.zig");
const datetime = @import("../i18n/datetime.zig");

// ============================================================================
// Language Tests
// ============================================================================

test "i18n: language count" {
    // Verify we have 33 languages
    try std.testing.expectEqual(@as(usize, 33), i18n.ALL_LANGUAGES.len);
}

test "i18n: english is default" {
    try std.testing.expectEqualStrings("en", i18n.getCurrentLanguage().code);
}

test "i18n: set language by code" {
    // Save original
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("ar");
    try std.testing.expectEqualStrings("ar", i18n.getCurrentLanguage().code);
    try std.testing.expect(i18n.getCurrentLanguage().rtl);

    i18n.setLanguageByCode("zh");
    try std.testing.expectEqualStrings("zh", i18n.getCurrentLanguage().code);
    try std.testing.expect(!i18n.getCurrentLanguage().rtl);
}

test "i18n: RTL languages" {
    // Arabic
    const arabic = i18n.getLanguageByCode("ar");
    try std.testing.expect(arabic.rtl);

    // Hebrew
    const hebrew = i18n.getLanguageByCode("he");
    try std.testing.expect(hebrew.rtl);

    // Persian
    const persian = i18n.getLanguageByCode("fa");
    try std.testing.expect(persian.rtl);

    // Urdu
    const urdu = i18n.getLanguageByCode("ur");
    try std.testing.expect(urdu.rtl);

    // English is LTR
    const english = i18n.getLanguageByCode("en");
    try std.testing.expect(!english.rtl);
}

test "i18n: language native names" {
    try std.testing.expectEqualStrings("العربية", i18n.getLanguageByCode("ar").native_name);
    try std.testing.expectEqualStrings("中文", i18n.getLanguageByCode("zh").native_name);
    try std.testing.expectEqualStrings("日本語", i18n.getLanguageByCode("ja").native_name);
    try std.testing.expectEqualStrings("Русский", i18n.getLanguageByCode("ru").native_name);
    try std.testing.expectEqualStrings("עברית", i18n.getLanguageByCode("he").native_name);
}

test "i18n: unknown language defaults to english" {
    const lang = i18n.getLanguageByCode("xyz");
    try std.testing.expectEqualStrings("en", lang.code);
}

// ============================================================================
// Plural Tests
// ============================================================================

test "plurals: english (germanic)" {
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", 0));
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("en", 1));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", 2));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", 5));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", 100));
}

test "plurals: arabic (full 6 forms)" {
    try std.testing.expectEqual(plurals.PluralCategory.zero, plurals.getPluralCategory("ar", 0));
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("ar", 1));
    try std.testing.expectEqual(plurals.PluralCategory.two, plurals.getPluralCategory("ar", 2));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ar", 3));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ar", 10));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ar", 11));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ar", 99));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ar", 100));
}

test "plurals: russian (slavic)" {
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("ru", 1));
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("ru", 21));
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("ru", 31));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ru", 2));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ru", 3));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ru", 4));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ru", 22));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ru", 0));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ru", 5));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ru", 11));
    try std.testing.expectEqual(plurals.PluralCategory.many, plurals.getPluralCategory("ru", 12));
}

test "plurals: chinese (asian - no plurals)" {
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("zh", 0));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("zh", 1));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("zh", 100));
}

test "plurals: japanese (asian - no plurals)" {
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ja", 0));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ja", 1));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ja", 100));
}

test "plurals: french (0 and 1 are singular)" {
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("fr", 0));
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("fr", 1));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("fr", 2));
}

test "plurals: hebrew (one/two/other)" {
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("he", 1));
    try std.testing.expectEqual(plurals.PluralCategory.two, plurals.getPluralCategory("he", 2));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("he", 3));
}

test "plurals: romanian" {
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("ro", 1));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ro", 0));
    try std.testing.expectEqual(plurals.PluralCategory.few, plurals.getPluralCategory("ro", 19));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ro", 20));
}

test "plurals: plural forms selection" {
    const forms = plurals.PluralForms{
        .zero = "no items",
        .one = "one item",
        .two = "two items",
        .few = "{d} items",
        .many = "{d} items",
        .other = "{d} items",
    };

    try std.testing.expectEqualStrings("one item", forms.select("en", 1));
    try std.testing.expectEqualStrings("{d} items", forms.select("en", 5));
    try std.testing.expectEqualStrings("no items", forms.select("ar", 0));
    try std.testing.expectEqualStrings("two items", forms.select("ar", 2));
}

test "plurals: negative numbers" {
    // Should use absolute value
    try std.testing.expectEqual(plurals.PluralCategory.one, plurals.getPluralCategory("en", -1));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", -5));
}

// ============================================================================
// Locale Detection Tests
// ============================================================================

test "locale_detect: parse locale string" {
    var detector = locale_detect.LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("en_US.UTF-8");
    try std.testing.expectEqualStrings("en", result.language);
    try std.testing.expectEqualStrings("US", result.region);
    try std.testing.expectEqualStrings("UTF-8", result.encoding);
}

test "locale_detect: parse arabic locale" {
    var detector = locale_detect.LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("ar_SA.UTF-8");
    try std.testing.expectEqualStrings("ar", result.language);
    try std.testing.expectEqualStrings("SA", result.region);
}

test "locale_detect: parse minimal locale" {
    var detector = locale_detect.LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("fr");
    try std.testing.expectEqualStrings("fr", result.language);
    try std.testing.expectEqualStrings("", result.region);
}

test "locale_detect: parse locale without encoding" {
    var detector = locale_detect.LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("de_DE");
    try std.testing.expectEqualStrings("de", result.language);
    try std.testing.expectEqualStrings("DE", result.region);
    try std.testing.expectEqualStrings("UTF-8", result.encoding);
}

test "locale_detect: i18n config disabled check" {
    // Without env var set, should not be disabled
    try std.testing.expect(!locale_detect.I18nConfig.isDisabled());
}

test "locale_detect: i18n config rtl forced check" {
    // Without env var set, should not be forced
    try std.testing.expect(!locale_detect.I18nConfig.isRtlForced());
}

// ============================================================================
// Date/Time Formatting Tests
// ============================================================================

test "datetime: month names english" {
    const names = datetime.getMonthNames("en");
    try std.testing.expectEqualStrings("January", names.full[0]);
    try std.testing.expectEqualStrings("February", names.full[1]);
    try std.testing.expectEqualStrings("December", names.full[11]);
    try std.testing.expectEqualStrings("Jan", names.short[0]);
    try std.testing.expectEqualStrings("Dec", names.short[11]);
}

test "datetime: month names arabic" {
    const names = datetime.getMonthNames("ar");
    try std.testing.expectEqualStrings("يناير", names.full[0]);
    try std.testing.expectEqualStrings("فبراير", names.full[1]);
    try std.testing.expectEqualStrings("ديسمبر", names.full[11]);
}

test "datetime: month names japanese" {
    const names = datetime.getMonthNames("ja");
    try std.testing.expectEqualStrings("1月", names.full[0]);
    try std.testing.expectEqualStrings("12月", names.full[11]);
}

test "datetime: month names russian" {
    const names = datetime.getMonthNames("ru");
    try std.testing.expectEqualStrings("Январь", names.full[0]);
    try std.testing.expectEqualStrings("Декабрь", names.full[11]);
}

test "datetime: day names english" {
    const names = datetime.getDayNames("en");
    try std.testing.expectEqualStrings("Sunday", names.full[0]);
    try std.testing.expectEqualStrings("Monday", names.full[1]);
    try std.testing.expectEqualStrings("Saturday", names.full[6]);
    try std.testing.expectEqualStrings("Sun", names.short[0]);
    try std.testing.expectEqualStrings("Sat", names.short[6]);
}

test "datetime: day names arabic" {
    const names = datetime.getDayNames("ar");
    try std.testing.expectEqualStrings("الأحد", names.full[0]);
    try std.testing.expectEqualStrings("الإثنين", names.full[1]);
    try std.testing.expectEqualStrings("السبت", names.full[6]);
}

test "datetime: format date short" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const date = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 23,
        .hour = 14,
        .minute = 30,
        .second = 0,
    };

    const result = try formatter.formatDate(date, .short, &buf);
    try std.testing.expectEqualStrings("1/23/26", result);
}

test "datetime: format date iso" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const date = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 23,
        .hour = 0,
        .minute = 0,
        .second = 0,
    };

    const result = try formatter.formatDate(date, .iso, &buf);
    try std.testing.expectEqualStrings("2026-01-23", result);
}

test "datetime: format time 24h" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const date = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 23,
        .hour = 14,
        .minute = 30,
        .second = 45,
    };

    const result = try formatter.formatTime(date, .time_24h, &buf);
    try std.testing.expectEqualStrings("14:30:45", result);
}

test "datetime: format time 12h" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const date = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 23,
        .hour = 14,
        .minute = 30,
        .second = 0,
    };

    const result = try formatter.formatTime(date, .time_12h, &buf);
    try std.testing.expectEqualStrings("2:30 PM", result);
}

test "datetime: format time 12h morning" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const date = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 23,
        .hour = 9,
        .minute = 15,
        .second = 0,
    };

    const result = try formatter.formatTime(date, .time_12h, &buf);
    try std.testing.expectEqualStrings("9:15 AM", result);
}

test "datetime: relative time just now" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatRelative(0, &buf);
    try std.testing.expectEqualStrings("just now", result);
}

test "datetime: relative time minutes ago" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatRelative(-120, &buf);
    try std.testing.expectEqualStrings("2 minutes ago", result);
}

test "datetime: relative time hours ago" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatRelative(-7200, &buf);
    try std.testing.expectEqualStrings("2 hours ago", result);
}

test "datetime: relative time days ago" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatRelative(-172800, &buf);
    try std.testing.expectEqualStrings("2 days ago", result);
}

test "datetime: relative time arabic" {
    var formatter = datetime.DateTimeFormatter.init("ar");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatRelative(0, &buf);
    try std.testing.expectEqualStrings("الآن", result);
}

// ============================================================================
// Message Catalog Tests
// ============================================================================

test "message: get english message" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("en");
    const msg = i18n.getMessage(.syntax_error);
    try std.testing.expectEqualStrings("Syntax error", msg);
}

test "message: get arabic message" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("ar");
    const msg = i18n.getMessage(.syntax_error);
    try std.testing.expectEqualStrings("خطأ في الصياغة", msg);
}

test "message: get chinese message" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("zh");
    const msg = i18n.getMessage(.syntax_error);
    try std.testing.expectEqualStrings("语法错误", msg);
}

test "message: get japanese message" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("ja");
    const msg = i18n.getMessage(.syntax_error);
    try std.testing.expectEqualStrings("構文エラー", msg);
}

test "message: get russian message" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("ru");
    const msg = i18n.getMessage(.syntax_error);
    try std.testing.expectEqualStrings("Синтаксическая ошибка", msg);
}

// ============================================================================
// RTL Handling Tests
// ============================================================================

test "rtl: is current language rtl" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    i18n.setLanguageByCode("ar");
    try std.testing.expect(i18n.isCurrentLanguageRtl());

    i18n.setLanguageByCode("en");
    try std.testing.expect(!i18n.isCurrentLanguageRtl());
}

test "rtl: all rtl languages" {
    const rtl_codes = [_][]const u8{ "ar", "he", "fa", "ur" };

    for (rtl_codes) |code| {
        const lang = i18n.getLanguageByCode(code);
        try std.testing.expect(lang.rtl);
    }
}

test "rtl: non-rtl languages" {
    const ltr_codes = [_][]const u8{ "en", "zh", "ja", "ko", "fr", "de", "es", "ru" };

    for (ltr_codes) |code| {
        const lang = i18n.getLanguageByCode(code);
        try std.testing.expect(!lang.rtl);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

test "edge: empty string handling" {
    var detector = locale_detect.LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("");
    try std.testing.expectEqualStrings("", result.language);
}

test "edge: plural with zero" {
    try std.testing.expectEqual(plurals.PluralCategory.zero, plurals.getPluralCategory("ar", 0));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("en", 0));
}

test "edge: large numbers plural" {
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ar", 1000));
    try std.testing.expectEqual(plurals.PluralCategory.other, plurals.getPluralCategory("ru", 1000));
}

test "edge: datetime boundary hours" {
    var formatter = datetime.DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    // Midnight
    const midnight = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 1,
        .hour = 0,
        .minute = 0,
        .second = 0,
    };
    const result1 = try formatter.formatTime(midnight, .time_12h, &buf);
    try std.testing.expectEqualStrings("12:00 AM", result1);

    // Noon
    const noon = datetime.DateTime{
        .year = 2026,
        .month = 1,
        .day = 1,
        .hour = 12,
        .minute = 0,
        .second = 0,
    };
    const result2 = try formatter.formatTime(noon, .time_12h, &buf);
    try std.testing.expectEqualStrings("12:00 PM", result2);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "integration: full localization flow" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    // Test complete localization for Arabic
    i18n.setLanguageByCode("ar");
    try std.testing.expect(i18n.getCurrentLanguage().rtl);
    try std.testing.expectEqualStrings("خطأ في الصياغة", i18n.getMessage(.syntax_error));

    var formatter = datetime.DateTimeFormatter.init("ar");
    const month_names = datetime.getMonthNames("ar");
    try std.testing.expectEqualStrings("يناير", month_names.full[0]);

    // Verify plural works
    try std.testing.expectEqual(plurals.PluralCategory.two, plurals.getPluralCategory("ar", 2));
}

test "integration: language switching" {
    const original = i18n.getCurrentLanguage();
    defer i18n.setLanguage(original);

    const languages = [_][]const u8{ "en", "ar", "zh", "ja", "ru", "fr", "de" };

    for (languages) |code| {
        i18n.setLanguageByCode(code);
        try std.testing.expectEqualStrings(code, i18n.getCurrentLanguage().code);
        // Should always be able to get syntax_error message
        const msg = i18n.getMessage(.syntax_error);
        try std.testing.expect(msg.len > 0);
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

test "performance: message lookup" {
    const iterations: usize = 10000;
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        _ = i18n.getMessage(.syntax_error);
        _ = i18n.getMessage(.type_mismatch);
        _ = i18n.getMessage(.undefined_variable);
    }
    // If we get here without timeout, performance is acceptable
}

test "performance: plural calculation" {
    const iterations: usize = 10000;
    var i: usize = 0;
    while (i < iterations) : (i += 1) {
        _ = plurals.getPluralCategory("ar", @as(i64, @intCast(i % 1000)));
        _ = plurals.getPluralCategory("ru", @as(i64, @intCast(i % 1000)));
        _ = plurals.getPluralCategory("en", @as(i64, @intCast(i % 1000)));
    }
}
