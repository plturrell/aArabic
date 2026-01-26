// plurals.zig
// CLDR-based pluralization rules for Mojo SDK
// Supports 33 languages with correct plural forms

const std = @import("std");

// ============================================================================
// Plural Categories (CLDR)
// ============================================================================

pub const PluralCategory = enum {
    zero,
    one,
    two,
    few,
    many,
    other,
};

// ============================================================================
// Plural Rules by Language
// ============================================================================

pub const PluralRule = enum {
    // Rule 1: Only "other" (Chinese, Japanese, Korean, Vietnamese, Thai, Indonesian)
    asian,

    // Rule 2: "one" and "other" (English, German, Spanish, Italian, Portuguese, Dutch, etc.)
    germanic,

    // Rule 3: "one", "few", "many", "other" (Russian, Ukrainian, Polish, Czech)
    slavic,

    // Rule 4: "zero", "one", "two", "few", "many", "other" (Arabic)
    arabic,

    // Rule 5: "one", "two", "other" (Hebrew)
    hebrew,

    // Rule 6: "one", "few", "other" (Romanian)
    romanian,

    // Rule 7: "one", "other" with special cases (French - 0 and 1 are singular)
    french,

    pub fn forLanguage(lang_code: []const u8) PluralRule {
        // Asian languages (no plural forms)
        if (std.mem.eql(u8, lang_code, "zh") or
            std.mem.eql(u8, lang_code, "ja") or
            std.mem.eql(u8, lang_code, "ko") or
            std.mem.eql(u8, lang_code, "vi") or
            std.mem.eql(u8, lang_code, "th") or
            std.mem.eql(u8, lang_code, "id"))
        {
            return .asian;
        }

        // Slavic languages
        if (std.mem.eql(u8, lang_code, "ru") or
            std.mem.eql(u8, lang_code, "uk") or
            std.mem.eql(u8, lang_code, "pl") or
            std.mem.eql(u8, lang_code, "cs"))
        {
            return .slavic;
        }

        // Arabic
        if (std.mem.eql(u8, lang_code, "ar")) {
            return .arabic;
        }

        // Hebrew
        if (std.mem.eql(u8, lang_code, "he")) {
            return .hebrew;
        }

        // Romanian
        if (std.mem.eql(u8, lang_code, "ro")) {
            return .romanian;
        }

        // French
        if (std.mem.eql(u8, lang_code, "fr")) {
            return .french;
        }

        // Default: Germanic rule
        return .germanic;
    }
};

// ============================================================================
// Plural Resolution
// ============================================================================

pub fn getPluralCategory(lang_code: []const u8, n: i64) PluralCategory {
    const rule = PluralRule.forLanguage(lang_code);
    const abs_n = if (n < 0) -n else n;

    return switch (rule) {
        .asian => .other,

        .germanic => if (abs_n == 1) .one else .other,

        .slavic => blk: {
            const mod10 = @mod(abs_n, 10);
            const mod100 = @mod(abs_n, 100);

            if (mod10 == 1 and mod100 != 11) {
                break :blk .one;
            } else if (mod10 >= 2 and mod10 <= 4 and (mod100 < 12 or mod100 > 14)) {
                break :blk .few;
            } else if (mod10 == 0 or (mod10 >= 5 and mod10 <= 9) or (mod100 >= 11 and mod100 <= 14)) {
                break :blk .many;
            } else {
                break :blk .other;
            }
        },

        .arabic => blk: {
            if (abs_n == 0) break :blk .zero;
            if (abs_n == 1) break :blk .one;
            if (abs_n == 2) break :blk .two;

            const mod100 = @mod(abs_n, 100);
            if (mod100 >= 3 and mod100 <= 10) break :blk .few;
            if (mod100 >= 11 and mod100 <= 99) break :blk .many;
            break :blk .other;
        },

        .hebrew => blk: {
            if (abs_n == 1) break :blk .one;
            if (abs_n == 2) break :blk .two;
            break :blk .other;
        },

        .romanian => blk: {
            if (abs_n == 1) break :blk .one;
            const mod100 = @mod(abs_n, 100);
            if (abs_n == 0 or (mod100 >= 1 and mod100 <= 19)) break :blk .few;
            break :blk .other;
        },

        .french => if (abs_n == 0 or abs_n == 1) .one else .other,
    };
}

// ============================================================================
// Plural Message Selection
// ============================================================================

pub const PluralForms = struct {
    zero: ?[]const u8 = null,
    one: ?[]const u8 = null,
    two: ?[]const u8 = null,
    few: ?[]const u8 = null,
    many: ?[]const u8 = null,
    other: []const u8,

    pub fn select(self: PluralForms, lang_code: []const u8, n: i64) []const u8 {
        const category = getPluralCategory(lang_code, n);

        return switch (category) {
            .zero => self.zero orelse self.other,
            .one => self.one orelse self.other,
            .two => self.two orelse self.other,
            .few => self.few orelse self.other,
            .many => self.many orelse self.other,
            .other => self.other,
        };
    }
};

// ============================================================================
// Common Plural Messages
// ============================================================================

pub const PLURAL_ERRORS = struct {
    pub const en = PluralForms{
        .one = "{d} error",
        .other = "{d} errors",
    };

    pub const ar = PluralForms{
        .zero = "لا أخطاء",
        .one = "خطأ واحد",
        .two = "خطآن",
        .few = "{d} أخطاء",
        .many = "{d} خطأ",
        .other = "{d} خطأ",
    };

    pub const ru = PluralForms{
        .one = "{d} ошибка",
        .few = "{d} ошибки",
        .many = "{d} ошибок",
        .other = "{d} ошибок",
    };

    pub const fr = PluralForms{
        .one = "{d} erreur",
        .other = "{d} erreurs",
    };

    pub const de = PluralForms{
        .one = "{d} Fehler",
        .other = "{d} Fehler",
    };

    pub const es = PluralForms{
        .one = "{d} error",
        .other = "{d} errores",
    };

    pub const zh = PluralForms{
        .other = "{d} 个错误",
    };

    pub const ja = PluralForms{
        .other = "{d} 個のエラー",
    };
};

pub const PLURAL_WARNINGS = struct {
    pub const en = PluralForms{
        .one = "{d} warning",
        .other = "{d} warnings",
    };

    pub const ar = PluralForms{
        .zero = "لا تحذيرات",
        .one = "تحذير واحد",
        .two = "تحذيران",
        .few = "{d} تحذيرات",
        .many = "{d} تحذيراً",
        .other = "{d} تحذير",
    };

    pub const ru = PluralForms{
        .one = "{d} предупреждение",
        .few = "{d} предупреждения",
        .many = "{d} предупреждений",
        .other = "{d} предупреждений",
    };

    pub const fr = PluralForms{
        .one = "{d} avertissement",
        .other = "{d} avertissements",
    };

    pub const de = PluralForms{
        .one = "{d} Warnung",
        .other = "{d} Warnungen",
    };

    pub const es = PluralForms{
        .one = "{d} advertencia",
        .other = "{d} advertencias",
    };

    pub const zh = PluralForms{
        .other = "{d} 个警告",
    };

    pub const ja = PluralForms{
        .other = "{d} 個の警告",
    };
};

pub const PLURAL_FILES = struct {
    pub const en = PluralForms{
        .one = "{d} file",
        .other = "{d} files",
    };

    pub const ar = PluralForms{
        .zero = "لا ملفات",
        .one = "ملف واحد",
        .two = "ملفان",
        .few = "{d} ملفات",
        .many = "{d} ملفاً",
        .other = "{d} ملف",
    };

    pub const ru = PluralForms{
        .one = "{d} файл",
        .few = "{d} файла",
        .many = "{d} файлов",
        .other = "{d} файлов",
    };

    pub const fr = PluralForms{
        .one = "{d} fichier",
        .other = "{d} fichiers",
    };

    pub const de = PluralForms{
        .one = "{d} Datei",
        .other = "{d} Dateien",
    };

    pub const es = PluralForms{
        .one = "{d} archivo",
        .other = "{d} archivos",
    };

    pub const zh = PluralForms{
        .other = "{d} 个文件",
    };

    pub const ja = PluralForms{
        .other = "{d} 個のファイル",
    };
};

pub const PLURAL_TESTS = struct {
    pub const en = PluralForms{
        .one = "{d} test",
        .other = "{d} tests",
    };

    pub const ar = PluralForms{
        .zero = "لا اختبارات",
        .one = "اختبار واحد",
        .two = "اختباران",
        .few = "{d} اختبارات",
        .many = "{d} اختباراً",
        .other = "{d} اختبار",
    };

    pub const ru = PluralForms{
        .one = "{d} тест",
        .few = "{d} теста",
        .many = "{d} тестов",
        .other = "{d} тестов",
    };

    pub const fr = PluralForms{
        .one = "{d} test",
        .other = "{d} tests",
    };

    pub const de = PluralForms{
        .one = "{d} Test",
        .other = "{d} Tests",
    };

    pub const es = PluralForms{
        .one = "{d} prueba",
        .other = "{d} pruebas",
    };

    pub const zh = PluralForms{
        .other = "{d} 个测试",
    };

    pub const ja = PluralForms{
        .other = "{d} 個のテスト",
    };
};

// ============================================================================
// Helper Functions
// ============================================================================

pub fn formatPlural(comptime lang: []const u8, forms: PluralForms, n: i64, buf: []u8) []const u8 {
    const template = forms.select(lang, n);
    return std.fmt.bufPrint(buf, template, .{n}) catch template;
}

// ============================================================================
// Tests
// ============================================================================

test "plural english" {
    try std.testing.expectEqual(PluralCategory.one, getPluralCategory("en", 1));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("en", 0));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("en", 2));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("en", 5));
}

test "plural arabic" {
    try std.testing.expectEqual(PluralCategory.zero, getPluralCategory("ar", 0));
    try std.testing.expectEqual(PluralCategory.one, getPluralCategory("ar", 1));
    try std.testing.expectEqual(PluralCategory.two, getPluralCategory("ar", 2));
    try std.testing.expectEqual(PluralCategory.few, getPluralCategory("ar", 3));
    try std.testing.expectEqual(PluralCategory.few, getPluralCategory("ar", 10));
    try std.testing.expectEqual(PluralCategory.many, getPluralCategory("ar", 11));
    try std.testing.expectEqual(PluralCategory.many, getPluralCategory("ar", 99));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("ar", 100));
}

test "plural russian" {
    try std.testing.expectEqual(PluralCategory.one, getPluralCategory("ru", 1));
    try std.testing.expectEqual(PluralCategory.one, getPluralCategory("ru", 21));
    try std.testing.expectEqual(PluralCategory.few, getPluralCategory("ru", 2));
    try std.testing.expectEqual(PluralCategory.few, getPluralCategory("ru", 22));
    try std.testing.expectEqual(PluralCategory.many, getPluralCategory("ru", 5));
    try std.testing.expectEqual(PluralCategory.many, getPluralCategory("ru", 11));
}

test "plural chinese" {
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("zh", 0));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("zh", 1));
    try std.testing.expectEqual(PluralCategory.other, getPluralCategory("zh", 100));
}
