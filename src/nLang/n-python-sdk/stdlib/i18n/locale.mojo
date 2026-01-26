# locale.mojo
# Core locale and language interface for Mojo SDK
# Pure Mojo implementation - no Python dependencies

from collections import Dict, List

# ============================================================================
# Language Codes (ISO 639-1)
# ============================================================================

struct Language:
    """Language identifier with ISO 639-1 code"""
    var code: String
    var name: String
    var native_name: String
    var rtl: Bool  # Right-to-left writing direction
    var script: String  # Primary script (e.g., "Arab", "Latn")

    fn __init__(out self, code: String, name: String, native_name: String, rtl: Bool = False, script: String = "Latn"):
        self.code = code
        self.name = name
        self.native_name = native_name
        self.rtl = rtl
        self.script = script

    fn __init__(out self):
        self.code = "en"
        self.name = "English"
        self.native_name = "English"
        self.rtl = False
        self.script = "Latn"

    fn __eq__(self, other: Language) -> Bool:
        return self.code == other.code

    fn __ne__(self, other: Language) -> Bool:
        return self.code != other.code

    fn __str__(self) -> String:
        return self.code

    fn __copyinit__(out self, existing: Self):
        self.code = existing.code
        self.name = existing.name
        self.native_name = existing.native_name
        self.rtl = existing.rtl
        self.script = existing.script

# ============================================================================
# Pre-defined Languages
# ============================================================================

fn arabic() -> Language:
    return Language("ar", "Arabic", "العربية", True, "Arab")

fn english() -> Language:
    return Language("en", "English", "English", False, "Latn")

fn french() -> Language:
    return Language("fr", "French", "Français", False, "Latn")

fn german() -> Language:
    return Language("de", "German", "Deutsch", False, "Latn")

fn spanish() -> Language:
    return Language("es", "Spanish", "Español", False, "Latn")

fn chinese() -> Language:
    return Language("zh", "Chinese", "中文", False, "Hans")

fn japanese() -> Language:
    return Language("ja", "Japanese", "日本語", False, "Jpan")

fn korean() -> Language:
    return Language("ko", "Korean", "한국어", False, "Kore")

fn russian() -> Language:
    return Language("ru", "Russian", "Русский", False, "Cyrl")

fn hebrew() -> Language:
    return Language("he", "Hebrew", "עברית", True, "Hebr")

fn persian() -> Language:
    return Language("fa", "Persian", "فارسی", True, "Arab")

fn urdu() -> Language:
    return Language("ur", "Urdu", "اردو", True, "Arab")

fn hindi() -> Language:
    return Language("hi", "Hindi", "हिन्दी", False, "Deva")

fn portuguese() -> Language:
    return Language("pt", "Portuguese", "Português", False, "Latn")

fn italian() -> Language:
    return Language("it", "Italian", "Italiano", False, "Latn")

fn dutch() -> Language:
    return Language("nl", "Dutch", "Nederlands", False, "Latn")

fn turkish() -> Language:
    return Language("tr", "Turkish", "Türkçe", False, "Latn")

# ============================================================================
# Locale
# ============================================================================

struct Locale:
    """
    Locale combining language, region, and formatting preferences.
    Format: language_REGION (e.g., "en_US", "ar_SA")
    """
    var language: Language
    var region: String  # ISO 3166-1 alpha-2 (e.g., "US", "SA")
    var collation: String  # Collation variant
    var calendar: String  # Calendar system (e.g., "gregorian", "islamic")
    var numbering_system: String  # Numbering system (e.g., "latn", "arab")

    fn __init__(out self, language: Language, region: String = ""):
        self.language = language
        self.region = region
        self.collation = "standard"
        self.calendar = "gregorian"
        self.numbering_system = "latn"

    fn __init__(out self):
        self.language = english()
        self.region = "US"
        self.collation = "standard"
        self.calendar = "gregorian"
        self.numbering_system = "latn"

    fn __init__(out self, language: Language, region: String, numbering: String, calendar: String):
        self.language = language
        self.region = region
        self.collation = "standard"
        self.calendar = calendar
        self.numbering_system = numbering

    fn __copyinit__(out self, existing: Self):
        self.language = existing.language
        self.region = existing.region
        self.collation = existing.collation
        self.calendar = existing.calendar
        self.numbering_system = existing.numbering_system

    fn to_bcp47(self) -> String:
        """Convert to BCP 47 language tag (e.g., 'en-US', 'ar-SA')"""
        if len(self.region) > 0:
            return self.language.code + "-" + self.region
        return self.language.code

    fn to_posix(self) -> String:
        """Convert to POSIX locale format (e.g., 'en_US.UTF-8')"""
        if len(self.region) > 0:
            return self.language.code + "_" + self.region + ".UTF-8"
        return self.language.code + ".UTF-8"

    fn is_rtl(self) -> Bool:
        """Check if locale uses right-to-left writing direction"""
        return self.language.rtl

    fn __str__(self) -> String:
        return self.to_bcp47()

# ============================================================================
# Pre-defined Locales
# ============================================================================

fn locale_en_us() -> Locale:
    return Locale(english(), "US")

fn locale_ar_sa() -> Locale:
    return Locale(arabic(), "SA", "arab", "islamic")

fn locale_ar_eg() -> Locale:
    return Locale(arabic(), "EG", "arab", "gregorian")

fn locale_fr_fr() -> Locale:
    return Locale(french(), "FR")

fn locale_de_de() -> Locale:
    return Locale(german(), "DE")

fn locale_es_es() -> Locale:
    return Locale(spanish(), "ES")

fn locale_zh_cn() -> Locale:
    return Locale(chinese(), "CN", "latn", "gregorian")

fn locale_ja_jp() -> Locale:
    return Locale(japanese(), "JP")

fn locale_ko_kr() -> Locale:
    return Locale(korean(), "KR")

fn locale_ru_ru() -> Locale:
    return Locale(russian(), "RU")

fn locale_he_il() -> Locale:
    return Locale(hebrew(), "IL", "latn", "hebrew")

fn locale_fa_ir() -> Locale:
    return Locale(persian(), "IR", "arabext", "persian")

fn locale_ur_pk() -> Locale:
    return Locale(urdu(), "PK", "arab", "islamic")

fn locale_hi_in() -> Locale:
    return Locale(hindi(), "IN", "deva", "gregorian")

# ============================================================================
# Language Registry
# ============================================================================

struct LanguageRegistry:
    """Registry of all supported languages"""
    var languages: List[Language]

    fn __init__(out self):
        self.languages = List[Language]()
        # Register all supported languages
        self.languages.append(arabic())
        self.languages.append(english())
        self.languages.append(french())
        self.languages.append(german())
        self.languages.append(spanish())
        self.languages.append(chinese())
        self.languages.append(japanese())
        self.languages.append(korean())
        self.languages.append(russian())
        self.languages.append(hebrew())
        self.languages.append(persian())
        self.languages.append(urdu())
        self.languages.append(hindi())
        self.languages.append(portuguese())
        self.languages.append(italian())
        self.languages.append(dutch())
        self.languages.append(turkish())

    fn get_by_code(self, code: String) -> Language:
        """Get language by ISO 639-1 code"""
        for i in range(len(self.languages)):
            if self.languages[i].code == code:
                return self.languages[i]
        # Return English as fallback
        return english()

    fn get_rtl_languages(self) -> List[Language]:
        """Get all RTL languages"""
        var rtl = List[Language]()
        for i in range(len(self.languages)):
            if self.languages[i].rtl:
                rtl.append(self.languages[i])
        return rtl

    fn is_supported(self, code: String) -> Bool:
        """Check if language code is supported"""
        for i in range(len(self.languages)):
            if self.languages[i].code == code:
                return True
        return False

# ============================================================================
# Global Locale State
# ============================================================================

struct LocaleContext:
    """Thread-local locale context"""
    var current: Locale
    var fallback: Locale
    var initialized: Bool

    fn __init__(out self):
        self.current = locale_en_us()
        self.fallback = locale_en_us()
        self.initialized = False

    fn __init__(out self, locale: Locale):
        self.current = locale
        self.fallback = locale_en_us()
        self.initialized = True

    fn set(inout self, locale: Locale):
        """Set current locale"""
        self.current = locale
        self.initialized = True

    fn get(self) -> Locale:
        """Get current locale"""
        return self.current

    fn reset(inout self):
        """Reset to fallback locale"""
        self.current = self.fallback
        self.initialized = True

# ============================================================================
# Number Formatting (Pure Mojo)
# ============================================================================

struct NumberFormatter:
    """Locale-aware number formatting"""
    var locale: Locale

    fn __init__(out self, locale: Locale):
        self.locale = locale

    fn format_integer(self, value: Int) -> String:
        """Format integer with locale-specific digits"""
        var result = String(value)

        if self.locale.numbering_system == "arab":
            result = self._to_arabic_numerals(result)
        elif self.locale.numbering_system == "arabext":
            result = self._to_extended_arabic_numerals(result)
        elif self.locale.numbering_system == "deva":
            result = self._to_devanagari_numerals(result)

        return result

    fn format_with_separator(self, value: Int) -> String:
        """Format integer with thousands separator"""
        var abs_value = value if value >= 0 else -value
        var is_negative = value < 0

        # Get separator based on locale
        var separator = ","
        if self.locale.language.code in ["de", "fr", "es", "it", "nl", "pt", "ru"]:
            separator = "."
        elif self.locale.language.code == "ar":
            separator = "٬"

        # Build string with separators
        var digits = String(abs_value)
        var result = String("")
        var count = 0

        # Process from right to left
        var i = len(digits) - 1
        while i >= 0:
            if count > 0 and count % 3 == 0:
                result = separator + result
            result = digits[i] + result
            count += 1
            i -= 1

        if is_negative:
            result = "-" + result

        # Convert digits if needed
        if self.locale.numbering_system == "arab":
            result = self._to_arabic_numerals(result)

        return result

    fn _to_arabic_numerals(self, s: String) -> String:
        """Convert Western digits to Arabic-Indic numerals"""
        var result = s
        result = result.replace("0", "٠")
        result = result.replace("1", "١")
        result = result.replace("2", "٢")
        result = result.replace("3", "٣")
        result = result.replace("4", "٤")
        result = result.replace("5", "٥")
        result = result.replace("6", "٦")
        result = result.replace("7", "٧")
        result = result.replace("8", "٨")
        result = result.replace("9", "٩")
        return result

    fn _to_extended_arabic_numerals(self, s: String) -> String:
        """Convert Western digits to Extended Arabic-Indic numerals (Persian/Urdu)"""
        var result = s
        result = result.replace("0", "۰")
        result = result.replace("1", "۱")
        result = result.replace("2", "۲")
        result = result.replace("3", "۳")
        result = result.replace("4", "۴")
        result = result.replace("5", "۵")
        result = result.replace("6", "۶")
        result = result.replace("7", "۷")
        result = result.replace("8", "۸")
        result = result.replace("9", "۹")
        return result

    fn _to_devanagari_numerals(self, s: String) -> String:
        """Convert Western digits to Devanagari numerals (Hindi)"""
        var result = s
        result = result.replace("0", "०")
        result = result.replace("1", "१")
        result = result.replace("2", "२")
        result = result.replace("3", "३")
        result = result.replace("4", "४")
        result = result.replace("5", "५")
        result = result.replace("6", "६")
        result = result.replace("7", "७")
        result = result.replace("8", "८")
        result = result.replace("9", "९")
        return result

# ============================================================================
# Text Direction Utilities
# ============================================================================

struct TextDirection:
    """Utilities for bidirectional text handling"""

    @staticmethod
    fn wrap_rtl(text: String) -> String:
        """Wrap text with RTL Unicode markers"""
        # U+200F = Right-to-Left Mark
        # U+202B = Right-to-Left Embedding
        # U+202C = Pop Directional Formatting
        return "\u202B" + text + "\u202C"

    @staticmethod
    fn wrap_ltr(text: String) -> String:
        """Wrap text with LTR Unicode markers"""
        # U+200E = Left-to-Right Mark
        # U+202A = Left-to-Right Embedding
        # U+202C = Pop Directional Formatting
        return "\u202A" + text + "\u202C"

    @staticmethod
    fn isolate_rtl(text: String) -> String:
        """Isolate RTL text (recommended for mixed content)"""
        # U+2067 = Right-to-Left Isolate
        # U+2069 = Pop Directional Isolate
        return "\u2067" + text + "\u2069"

    @staticmethod
    fn isolate_ltr(text: String) -> String:
        """Isolate LTR text (recommended for mixed content)"""
        # U+2066 = Left-to-Right Isolate
        # U+2069 = Pop Directional Isolate
        return "\u2066" + text + "\u2069"

    @staticmethod
    fn is_rtl_char(c: String) -> Bool:
        """Check if character is in RTL Unicode range"""
        # Simplified check for Arabic and Hebrew ranges
        # Arabic: U+0600-U+06FF, U+0750-U+077F, U+08A0-U+08FF
        # Hebrew: U+0590-U+05FF
        if len(c) == 0:
            return False
        # Check first byte for common RTL ranges
        let first_byte = ord(c[0])
        return first_byte >= 0xD8 and first_byte <= 0xDF  # Simplified UTF-8 check
