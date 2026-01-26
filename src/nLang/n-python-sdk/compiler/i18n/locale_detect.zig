// locale_detect.zig
// System locale detection for Mojo SDK
// Pure Zig implementation using environment variables and system APIs

const std = @import("std");
const i18n = @import("i18n");
const builtin = @import("builtin");

// ============================================================================
// Locale Detection
// ============================================================================

pub const DetectedLocale = struct {
    language: []const u8,
    region: []const u8,
    encoding: []const u8,
    full: []const u8,
};

pub const LocaleDetector = struct {
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) LocaleDetector {
        return .{ .allocator = allocator };
    }

    /// Detect system locale
    pub fn detect(self: *LocaleDetector) DetectedLocale {
        // Try environment variables in order of precedence
        const env_vars = [_][]const u8{ "LC_ALL", "LC_MESSAGES", "LANG", "LANGUAGE" };

        for (env_vars) |env_var| {
            if (std.posix.getenv(env_var)) |value| {
                if (value.len > 0 and !std.mem.eql(u8, value, "C") and !std.mem.eql(u8, value, "POSIX")) {
                    return self.parseLocaleString(value);
                }
            }
        }

        // Platform-specific detection
        if (builtin.os.tag == .macos) {
            return self.detectMacOS();
        } else if (builtin.os.tag == .windows) {
            return self.detectWindows();
        }

        // Default to English
        return DetectedLocale{
            .language = "en",
            .region = "US",
            .encoding = "UTF-8",
            .full = "en_US.UTF-8",
        };
    }

    /// Parse POSIX locale string (e.g., "en_US.UTF-8")
    fn parseLocaleString(self: *LocaleDetector, locale_str: []const u8) DetectedLocale {
        _ = self;
        var language: []const u8 = "en";
        var region: []const u8 = "";
        var encoding: []const u8 = "UTF-8";

        // Split by '.' to separate encoding
        var base = locale_str;
        if (std.mem.indexOf(u8, locale_str, ".")) |dot_pos| {
            base = locale_str[0..dot_pos];
            encoding = locale_str[dot_pos + 1 ..];
        }

        // Split by '_' to separate language and region
        if (std.mem.indexOf(u8, base, "_")) |underscore_pos| {
            language = base[0..underscore_pos];
            region = base[underscore_pos + 1 ..];
        } else {
            language = base;
        }

        return DetectedLocale{
            .language = language,
            .region = region,
            .encoding = encoding,
            .full = locale_str,
        };
    }

    fn detectMacOS(self: *LocaleDetector) DetectedLocale {
        _ = self;
        // On macOS, try AppleLocale from defaults
        // For now, fall back to environment detection
        return DetectedLocale{
            .language = "en",
            .region = "US",
            .encoding = "UTF-8",
            .full = "en_US.UTF-8",
        };
    }

    fn detectWindows(self: *LocaleDetector) DetectedLocale {
        _ = self;
        // On Windows, would use GetUserDefaultUILanguage
        // For now, fall back to default
        return DetectedLocale{
            .language = "en",
            .region = "US",
            .encoding = "UTF-8",
            .full = "en_US.UTF-8",
        };
    }

    /// Get Language struct from detected locale
    pub fn getLanguage(self: *LocaleDetector) i18n.Language {
        const detected = self.detect();
        return i18n.getLanguageByCode(detected.language);
    }
};

// ============================================================================
// Environment-based Configuration
// ============================================================================

pub const I18nConfig = struct {
    /// Environment variable for language override
    pub const ENV_MOJO_LANG = "MOJO_LANG";

    /// Environment variable for forcing RTL
    pub const ENV_MOJO_RTL = "MOJO_RTL";

    /// Environment variable for disabling i18n
    pub const ENV_MOJO_NO_I18N = "MOJO_NO_I18N";

    /// Check if i18n is disabled
    pub fn isDisabled() bool {
        if (std.posix.getenv(ENV_MOJO_NO_I18N)) |value| {
            return std.mem.eql(u8, value, "1") or
                std.mem.eql(u8, value, "true") or
                std.mem.eql(u8, value, "yes");
        }
        return false;
    }

    /// Get language override from environment
    pub fn getLanguageOverride() ?[]const u8 {
        return std.posix.getenv(ENV_MOJO_LANG);
    }

    /// Check if RTL is forced
    pub fn isRtlForced() bool {
        if (std.posix.getenv(ENV_MOJO_RTL)) |value| {
            return std.mem.eql(u8, value, "1") or
                std.mem.eql(u8, value, "true") or
                std.mem.eql(u8, value, "yes");
        }
        return false;
    }
};

// ============================================================================
// Auto-initialization
// ============================================================================

var initialized = false;

pub fn autoInit(allocator: std.mem.Allocator) void {
    if (initialized) return;
    initialized = true;

    // Check for disabled i18n
    if (I18nConfig.isDisabled()) {
        i18n.setLanguage(i18n.LANG_ENGLISH);
        return;
    }

    // Check for language override
    if (I18nConfig.getLanguageOverride()) |lang_code| {
        i18n.setLanguageByCode(lang_code);
        return;
    }

    // Auto-detect from system
    var detector = LocaleDetector.init(allocator);
    const lang = detector.getLanguage();
    i18n.setLanguage(lang);
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Get detected language code
pub fn getSystemLanguage(allocator: std.mem.Allocator) []const u8 {
    var detector = LocaleDetector.init(allocator);
    const detected = detector.detect();
    return detected.language;
}

/// Check if system is RTL
pub fn isSystemRtl(allocator: std.mem.Allocator) bool {
    if (I18nConfig.isRtlForced()) return true;

    var detector = LocaleDetector.init(allocator);
    const lang = detector.getLanguage();
    return lang.rtl;
}

// ============================================================================
// Tests
// ============================================================================

test "locale parse" {
    var detector = LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("en_US.UTF-8");
    try std.testing.expectEqualStrings("en", result.language);
    try std.testing.expectEqualStrings("US", result.region);
    try std.testing.expectEqualStrings("UTF-8", result.encoding);
}

test "locale parse arabic" {
    var detector = LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("ar_SA.UTF-8");
    try std.testing.expectEqualStrings("ar", result.language);
    try std.testing.expectEqualStrings("SA", result.region);
}

test "locale parse minimal" {
    var detector = LocaleDetector.init(std.testing.allocator);

    const result = detector.parseLocaleString("fr");
    try std.testing.expectEqualStrings("fr", result.language);
}
