// locale module - Phase 1.10 Priority 8 - Locale Support
const std = @import("std");

// Locale categories
pub const LC_ALL: c_int = 0;
pub const LC_COLLATE: c_int = 1;
pub const LC_CTYPE: c_int = 2;
pub const LC_MONETARY: c_int = 3;
pub const LC_NUMERIC: c_int = 4;
pub const LC_TIME: c_int = 5;
pub const LC_MESSAGES: c_int = 6;

pub const lconv = extern struct {
    decimal_point: [*:0]u8,
    thousands_sep: [*:0]u8,
    grouping: [*:0]u8,
    int_curr_symbol: [*:0]u8,
    currency_symbol: [*:0]u8,
    mon_decimal_point: [*:0]u8,
    mon_thousands_sep: [*:0]u8,
    mon_grouping: [*:0]u8,
    positive_sign: [*:0]u8,
    negative_sign: [*:0]u8,
    int_frac_digits: u8,
    frac_digits: u8,
    p_cs_precedes: u8,
    p_sep_by_space: u8,
    n_cs_precedes: u8,
    n_sep_by_space: u8,
    p_sign_posn: u8,
    n_sign_posn: u8,
};

var default_lconv = lconv{
    .decimal_point = @constCast("."),
    .thousands_sep = @constCast(""),
    .grouping = @constCast(""),
    .int_curr_symbol = @constCast(""),
    .currency_symbol = @constCast(""),
    .mon_decimal_point = @constCast(""),
    .mon_thousands_sep = @constCast(""),
    .mon_grouping = @constCast(""),
    .positive_sign = @constCast(""),
    .negative_sign = @constCast(""),
    .int_frac_digits = 127,
    .frac_digits = 127,
    .p_cs_precedes = 127,
    .p_sep_by_space = 127,
    .n_cs_precedes = 127,
    .n_sep_by_space = 127,
    .p_sign_posn = 127,
    .n_sign_posn = 127,
};

pub export fn setlocale(category: c_int, locale: ?[*:0]const u8) ?[*:0]u8 {
    _ = category;
    if (locale) |loc| {
        const loc_str = std.mem.span(loc);
        if (loc_str.len == 0 or std.mem.eql(u8, loc_str, "C") or std.mem.eql(u8, loc_str, "POSIX")) {
            return @constCast("C");
        }
    }
    return @constCast("C");
}

pub export fn localeconv() *lconv {
    return &default_lconv;
}

// POSIX.1-2008 locale functions
pub const locale_t = ?*LocaleData;

// Category masks for newlocale
pub const LC_COLLATE_MASK: c_int = 1 << LC_COLLATE;
pub const LC_CTYPE_MASK: c_int = 1 << LC_CTYPE;
pub const LC_MONETARY_MASK: c_int = 1 << LC_MONETARY;
pub const LC_NUMERIC_MASK: c_int = 1 << LC_NUMERIC;
pub const LC_TIME_MASK: c_int = 1 << LC_TIME;
pub const LC_MESSAGES_MASK: c_int = 1 << LC_MESSAGES;
pub const LC_ALL_MASK: c_int = LC_COLLATE_MASK | LC_CTYPE_MASK | LC_MONETARY_MASK | LC_NUMERIC_MASK | LC_TIME_MASK | LC_MESSAGES_MASK;

// Special locale value for uselocale
pub const LC_GLOBAL_LOCALE: locale_t = @ptrFromInt(std.math.maxInt(usize));

const LocaleData = struct {
    names: [7][]const u8, // One per category + LC_ALL
    lconv_data: lconv,
};

var gpa = std.heap.GeneralPurposeAllocator(.{}){};

// Global locale state
var global_locale: LocaleData = .{
    .names = .{ "C", "C", "C", "C", "C", "C", "C" },
    .lconv_data = .{
        .decimal_point = @constCast("."),
        .thousands_sep = @constCast(""),
        .grouping = @constCast(""),
        .int_curr_symbol = @constCast(""),
        .currency_symbol = @constCast(""),
        .mon_decimal_point = @constCast(""),
        .mon_thousands_sep = @constCast(""),
        .mon_grouping = @constCast(""),
        .positive_sign = @constCast(""),
        .negative_sign = @constCast(""),
        .int_frac_digits = 127,
        .frac_digits = 127,
        .p_cs_precedes = 127,
        .p_sep_by_space = 127,
        .n_cs_precedes = 127,
        .n_sep_by_space = 127,
        .p_sign_posn = 127,
        .n_sign_posn = 127,
    },
};

// Thread-local current locale
threadlocal var current_locale: locale_t = null;

/// Create a new locale object
pub export fn newlocale(category_mask: c_int, locale: [*:0]const u8, base: locale_t) locale_t {
    const allocator = gpa.allocator();

    // Create new locale or clone base
    const new_loc = allocator.create(LocaleData) catch return null;

    if (base != null and base != LC_GLOBAL_LOCALE) {
        new_loc.* = base.?.*;
    } else {
        new_loc.* = global_locale;
    }

    // Parse locale name
    const loc_str = std.mem.span(locale);
    const is_c = loc_str.len == 0 or std.mem.eql(u8, loc_str, "C") or std.mem.eql(u8, loc_str, "POSIX");

    // Apply to requested categories
    if (is_c) {
        if ((category_mask & LC_COLLATE_MASK) != 0) new_loc.names[@intCast(LC_COLLATE)] = "C";
        if ((category_mask & LC_CTYPE_MASK) != 0) new_loc.names[@intCast(LC_CTYPE)] = "C";
        if ((category_mask & LC_MONETARY_MASK) != 0) new_loc.names[@intCast(LC_MONETARY)] = "C";
        if ((category_mask & LC_NUMERIC_MASK) != 0) new_loc.names[@intCast(LC_NUMERIC)] = "C";
        if ((category_mask & LC_TIME_MASK) != 0) new_loc.names[@intCast(LC_TIME)] = "C";
        if ((category_mask & LC_MESSAGES_MASK) != 0) new_loc.names[@intCast(LC_MESSAGES)] = "C";
    }

    return new_loc;
}

/// Duplicate a locale object
pub export fn duplocale(locobj: locale_t) locale_t {
    if (locobj == null) return null;
    if (locobj == LC_GLOBAL_LOCALE) {
        return newlocale(LC_ALL_MASK, "C", null);
    }

    const allocator = gpa.allocator();
    const new_loc = allocator.create(LocaleData) catch return null;
    new_loc.* = locobj.?.*;
    return new_loc;
}

/// Free a locale object
pub export fn freelocale(locobj: locale_t) void {
    if (locobj == null or locobj == LC_GLOBAL_LOCALE) return;
    const allocator = gpa.allocator();
    allocator.destroy(locobj.?);
}

/// Set and/or query thread-local locale
pub export fn uselocale(newloc: locale_t) locale_t {
    const old = current_locale orelse LC_GLOBAL_LOCALE;

    if (newloc != null) {
        if (newloc == LC_GLOBAL_LOCALE) {
            current_locale = null;
        } else {
            current_locale = newloc;
        }
    }

    return old;
}
