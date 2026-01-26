// datetime.zig
// Locale-aware date/time formatting for Mojo SDK
// Pure Zig implementation

const std = @import("std");
const i18n = @import("i18n.zig");

// ============================================================================
// DateTime Components
// ============================================================================

pub const DateTime = struct {
    year: i32,
    month: u8,   // 1-12
    day: u8,     // 1-31
    hour: u8,    // 0-23
    minute: u8,  // 0-59
    second: u8,  // 0-59
    millisecond: u16 = 0,

    pub fn now() DateTime {
        const timestamp = std.time.timestamp();
        return fromTimestamp(timestamp);
    }

    pub fn fromTimestamp(timestamp: i64) DateTime {
        const epoch_secs = @as(i64, timestamp);
        const SECS_PER_DAY = 86400;
        const DAYS_PER_400Y = 146097;
        const DAYS_PER_100Y = 36524;
        const DAYS_PER_4Y = 1461;

        var days = @divFloor(epoch_secs, SECS_PER_DAY);
        var rem = @mod(epoch_secs, SECS_PER_DAY);
        if (rem < 0) {
            rem += SECS_PER_DAY;
            days -= 1;
        }

        const hour: u8 = @intCast(@divFloor(rem, 3600));
        rem = @mod(rem, 3600);
        const minute: u8 = @intCast(@divFloor(rem, 60));
        const second: u8 = @intCast(@mod(rem, 60));

        // Days since 1970-01-01 to year/month/day
        days += 719468; // days from year 0 to 1970

        var era = @divFloor(days, DAYS_PER_400Y);
        const doe = @mod(days, DAYS_PER_400Y);
        const yoe = @divFloor(doe - @divFloor(doe, DAYS_PER_4Y - 1) + @divFloor(doe, DAYS_PER_100Y) - @divFloor(doe, DAYS_PER_400Y - 1), 365);
        const year: i32 = @intCast(yoe + era * 400);
        const doy = doe - (365 * yoe + @divFloor(yoe, 4) - @divFloor(yoe, 100));
        const mp = @divFloor(5 * doy + 2, 153);
        const day: u8 = @intCast(doy - @divFloor(153 * mp + 2, 5) + 1);
        const month: u8 = @intCast(if (mp < 10) mp + 3 else mp - 9);

        return DateTime{
            .year = if (month <= 2) year + 1 else year,
            .month = month,
            .day = day,
            .hour = hour,
            .minute = minute,
            .second = second,
        };
    }
};

// ============================================================================
// Locale-specific Date Formats
// ============================================================================

pub const DateFormat = enum {
    short,   // e.g., "1/5/24" or "٥/١/٢٠٢٤"
    medium,  // e.g., "Jan 5, 2024" or "٥ يناير ٢٠٢٤"
    long_,   // e.g., "January 5, 2024" or "٥ يناير ٢٠٢٤"
    full,    // e.g., "Friday, January 5, 2024"
    iso,     // "2024-01-05"
};

pub const TimeFormat = enum {
    short,   // e.g., "3:30 PM" or "١٥:٣٠"
    medium,  // e.g., "3:30:45 PM"
    long_,   // e.g., "3:30:45 PM UTC"
    iso,     // "15:30:45"
};

// ============================================================================
// Month Names by Language
// ============================================================================

pub const MonthNames = struct {
    short: [12][]const u8,
    full: [12][]const u8,
};

pub fn getMonthNames(lang_code: []const u8) MonthNames {
    if (std.mem.eql(u8, lang_code, "ar")) {
        return .{
            .short = .{ "ينا", "فبر", "مار", "أبر", "ماي", "يون", "يول", "أغس", "سبت", "أكت", "نوف", "ديس" },
            .full = .{ "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر" },
        };
    } else if (std.mem.eql(u8, lang_code, "fr")) {
        return .{
            .short = .{ "janv.", "févr.", "mars", "avr.", "mai", "juin", "juil.", "août", "sept.", "oct.", "nov.", "déc." },
            .full = .{ "janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre" },
        };
    } else if (std.mem.eql(u8, lang_code, "de")) {
        return .{
            .short = .{ "Jan.", "Feb.", "März", "Apr.", "Mai", "Juni", "Juli", "Aug.", "Sept.", "Okt.", "Nov.", "Dez." },
            .full = .{ "Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober", "November", "Dezember" },
        };
    } else if (std.mem.eql(u8, lang_code, "es")) {
        return .{
            .short = .{ "ene.", "feb.", "mar.", "abr.", "may.", "jun.", "jul.", "ago.", "sept.", "oct.", "nov.", "dic." },
            .full = .{ "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre" },
        };
    } else if (std.mem.eql(u8, lang_code, "zh")) {
        return .{
            .short = .{ "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月" },
            .full = .{ "一月", "二月", "三月", "四月", "五月", "六月", "七月", "八月", "九月", "十月", "十一月", "十二月" },
        };
    } else if (std.mem.eql(u8, lang_code, "ja")) {
        return .{
            .short = .{ "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月" },
            .full = .{ "1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月" },
        };
    } else if (std.mem.eql(u8, lang_code, "ru")) {
        return .{
            .short = .{ "янв.", "февр.", "март", "апр.", "май", "июнь", "июль", "авг.", "сент.", "окт.", "нояб.", "дек." },
            .full = .{ "январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь" },
        };
    }

    // English default
    return .{
        .short = .{ "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec" },
        .full = .{ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" },
    };
}

// ============================================================================
// Day Names by Language
// ============================================================================

pub const DayNames = struct {
    short: [7][]const u8,
    full: [7][]const u8,
};

pub fn getDayNames(lang_code: []const u8) DayNames {
    if (std.mem.eql(u8, lang_code, "ar")) {
        return .{
            .short = .{ "أحد", "إثن", "ثلا", "أرب", "خمي", "جمع", "سبت" },
            .full = .{ "الأحد", "الإثنين", "الثلاثاء", "الأربعاء", "الخميس", "الجمعة", "السبت" },
        };
    } else if (std.mem.eql(u8, lang_code, "fr")) {
        return .{
            .short = .{ "dim.", "lun.", "mar.", "mer.", "jeu.", "ven.", "sam." },
            .full = .{ "dimanche", "lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi" },
        };
    } else if (std.mem.eql(u8, lang_code, "de")) {
        return .{
            .short = .{ "So.", "Mo.", "Di.", "Mi.", "Do.", "Fr.", "Sa." },
            .full = .{ "Sonntag", "Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag" },
        };
    } else if (std.mem.eql(u8, lang_code, "es")) {
        return .{
            .short = .{ "dom.", "lun.", "mar.", "mié.", "jue.", "vie.", "sáb." },
            .full = .{ "domingo", "lunes", "martes", "miércoles", "jueves", "viernes", "sábado" },
        };
    } else if (std.mem.eql(u8, lang_code, "zh")) {
        return .{
            .short = .{ "日", "一", "二", "三", "四", "五", "六" },
            .full = .{ "星期日", "星期一", "星期二", "星期三", "星期四", "星期五", "星期六" },
        };
    } else if (std.mem.eql(u8, lang_code, "ja")) {
        return .{
            .short = .{ "日", "月", "火", "水", "木", "金", "土" },
            .full = .{ "日曜日", "月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日" },
        };
    } else if (std.mem.eql(u8, lang_code, "ru")) {
        return .{
            .short = .{ "вс", "пн", "вт", "ср", "чт", "пт", "сб" },
            .full = .{ "воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота" },
        };
    }

    // English default
    return .{
        .short = .{ "Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat" },
        .full = .{ "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday" },
    };
}

// ============================================================================
// Date/Time Formatter
// ============================================================================

pub const DateTimeFormatter = struct {
    lang_code: []const u8,
    use_24h: bool,
    use_arabic_numerals: bool,

    pub fn init(lang_code: []const u8) DateTimeFormatter {
        const use_24h = !std.mem.eql(u8, lang_code, "en");
        const use_arabic_numerals = std.mem.eql(u8, lang_code, "ar") or
            std.mem.eql(u8, lang_code, "fa") or
            std.mem.eql(u8, lang_code, "ur");

        return .{
            .lang_code = lang_code,
            .use_24h = use_24h,
            .use_arabic_numerals = use_arabic_numerals,
        };
    }

    pub fn formatDate(self: *const DateTimeFormatter, dt: DateTime, format: DateFormat, buf: []u8) ![]const u8 {
        return switch (format) {
            .iso => try std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}", .{ dt.year, dt.month, dt.day }),
            .short => try self.formatDateShort(dt, buf),
            .medium => try self.formatDateMedium(dt, buf),
            .long_ => try self.formatDateLong(dt, buf),
            .full => try self.formatDateFull(dt, buf),
        };
    }

    fn formatDateShort(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        _ = self;
        return try std.fmt.bufPrint(buf, "{d}/{d}/{d}", .{ dt.month, dt.day, @mod(dt.year, 100) });
    }

    fn formatDateMedium(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        const months = getMonthNames(self.lang_code);
        const month_name = months.short[dt.month - 1];
        return try std.fmt.bufPrint(buf, "{s} {d}, {d}", .{ month_name, dt.day, dt.year });
    }

    fn formatDateLong(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        const months = getMonthNames(self.lang_code);
        const month_name = months.full[dt.month - 1];
        return try std.fmt.bufPrint(buf, "{s} {d}, {d}", .{ month_name, dt.day, dt.year });
    }

    fn formatDateFull(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        const months = getMonthNames(self.lang_code);
        const days = getDayNames(self.lang_code);
        const dow = dayOfWeek(dt.year, dt.month, dt.day);
        return try std.fmt.bufPrint(buf, "{s}, {s} {d}, {d}", .{
            days.full[dow],
            months.full[dt.month - 1],
            dt.day,
            dt.year,
        });
    }

    pub fn formatTime(self: *const DateTimeFormatter, dt: DateTime, format: TimeFormat, buf: []u8) ![]const u8 {
        return switch (format) {
            .iso => try std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}", .{ dt.hour, dt.minute, dt.second }),
            .short => try self.formatTimeShort(dt, buf),
            .medium => try self.formatTimeMedium(dt, buf),
            .long_ => try self.formatTimeLong(dt, buf),
        };
    }

    fn formatTimeShort(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        if (self.use_24h) {
            return try std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}", .{ dt.hour, dt.minute });
        } else {
            const hour12 = if (dt.hour == 0) 12 else if (dt.hour > 12) dt.hour - 12 else dt.hour;
            const ampm: []const u8 = if (dt.hour < 12) "AM" else "PM";
            return try std.fmt.bufPrint(buf, "{d}:{d:0>2} {s}", .{ hour12, dt.minute, ampm });
        }
    }

    fn formatTimeMedium(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        if (self.use_24h) {
            return try std.fmt.bufPrint(buf, "{d:0>2}:{d:0>2}:{d:0>2}", .{ dt.hour, dt.minute, dt.second });
        } else {
            const hour12 = if (dt.hour == 0) 12 else if (dt.hour > 12) dt.hour - 12 else dt.hour;
            const ampm: []const u8 = if (dt.hour < 12) "AM" else "PM";
            return try std.fmt.bufPrint(buf, "{d}:{d:0>2}:{d:0>2} {s}", .{ hour12, dt.minute, dt.second, ampm });
        }
    }

    fn formatTimeLong(self: *const DateTimeFormatter, dt: DateTime, buf: []u8) ![]const u8 {
        const medium = try self.formatTimeMedium(dt, buf[0..32]);
        const rest = buf[medium.len..];
        _ = try std.fmt.bufPrint(rest, " UTC", .{});
        return buf[0 .. medium.len + 4];
    }

    pub fn formatDateTime(self: *const DateTimeFormatter, dt: DateTime, date_fmt: DateFormat, time_fmt: TimeFormat, buf: []u8) ![]const u8 {
        var date_buf: [64]u8 = undefined;
        var time_buf: [32]u8 = undefined;

        const date_str = try self.formatDate(dt, date_fmt, &date_buf);
        const time_str = try self.formatTime(dt, time_fmt, &time_buf);

        return try std.fmt.bufPrint(buf, "{s} {s}", .{ date_str, time_str });
    }
};

// ============================================================================
// Helper Functions
// ============================================================================

fn dayOfWeek(year: i32, month: u8, day: u8) u8 {
    // Zeller's congruence
    var y = year;
    var m = month;

    if (m < 3) {
        m += 12;
        y -= 1;
    }

    const k = @mod(y, 100);
    const j = @divFloor(y, 100);

    var h = day + @divFloor(13 * (m + 1), 5) + k + @divFloor(k, 4) + @divFloor(j, 4) - 2 * j;
    h = @mod(h, 7);

    // Convert to 0=Sunday
    return @intCast(@mod(h + 6, 7));
}

// ============================================================================
// Relative Time Formatting
// ============================================================================

pub const RelativeTime = struct {
    pub fn format(lang_code: []const u8, seconds: i64, buf: []u8) ![]const u8 {
        const abs_secs = if (seconds < 0) -seconds else seconds;
        const future = seconds > 0;

        if (abs_secs < 60) {
            return formatRelative(lang_code, abs_secs, "second", future, buf);
        } else if (abs_secs < 3600) {
            return formatRelative(lang_code, @divFloor(abs_secs, 60), "minute", future, buf);
        } else if (abs_secs < 86400) {
            return formatRelative(lang_code, @divFloor(abs_secs, 3600), "hour", future, buf);
        } else if (abs_secs < 2592000) {
            return formatRelative(lang_code, @divFloor(abs_secs, 86400), "day", future, buf);
        } else if (abs_secs < 31536000) {
            return formatRelative(lang_code, @divFloor(abs_secs, 2592000), "month", future, buf);
        } else {
            return formatRelative(lang_code, @divFloor(abs_secs, 31536000), "year", future, buf);
        }
    }

    fn formatRelative(lang_code: []const u8, value: i64, unit: []const u8, future: bool, buf: []u8) ![]const u8 {
        if (std.mem.eql(u8, lang_code, "ar")) {
            if (future) {
                return try std.fmt.bufPrint(buf, "بعد {d} {s}", .{ value, getArabicUnit(unit) });
            } else {
                return try std.fmt.bufPrint(buf, "منذ {d} {s}", .{ value, getArabicUnit(unit) });
            }
        }

        // English default
        const unit_str = if (value == 1) unit else try std.fmt.bufPrint(buf[0..16], "{s}s", .{unit});
        if (future) {
            return try std.fmt.bufPrint(buf, "in {d} {s}", .{ value, unit_str });
        } else {
            return try std.fmt.bufPrint(buf, "{d} {s} ago", .{ value, unit_str });
        }
    }

    fn getArabicUnit(unit: []const u8) []const u8 {
        if (std.mem.eql(u8, unit, "second")) return "ثانية";
        if (std.mem.eql(u8, unit, "minute")) return "دقيقة";
        if (std.mem.eql(u8, unit, "hour")) return "ساعة";
        if (std.mem.eql(u8, unit, "day")) return "يوم";
        if (std.mem.eql(u8, unit, "month")) return "شهر";
        if (std.mem.eql(u8, unit, "year")) return "سنة";
        return unit;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "datetime iso format" {
    const dt = DateTime{ .year = 2024, .month = 1, .day = 15, .hour = 14, .minute = 30, .second = 45 };
    const formatter = DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatDate(dt, .iso, &buf);
    try std.testing.expectEqualStrings("2024-01-15", result);
}

test "datetime time format" {
    const dt = DateTime{ .year = 2024, .month = 1, .day = 15, .hour = 14, .minute = 30, .second = 45 };
    const formatter = DateTimeFormatter.init("en");
    var buf: [64]u8 = undefined;

    const result = try formatter.formatTime(dt, .short, &buf);
    try std.testing.expectEqualStrings("2:30 PM", result);
}

test "day of week" {
    // January 15, 2024 is a Monday (1)
    try std.testing.expectEqual(@as(u8, 1), dayOfWeek(2024, 1, 15));
}
