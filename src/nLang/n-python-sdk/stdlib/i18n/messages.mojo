# messages.mojo
# Message catalog system for internationalization
# Pure Mojo implementation - no external dependencies

from collections import Dict, List
from .locale import Language, Locale, english, arabic

# ============================================================================
# Message ID Constants
# ============================================================================

# Compiler Error Messages (E001-E999)
alias E001_TYPE_MISMATCH = "E001"
alias E002_UNDEFINED_VARIABLE = "E002"
alias E003_UNDEFINED_FUNCTION = "E003"
alias E004_ARGUMENT_COUNT = "E004"
alias E005_RETURN_TYPE = "E005"
alias E006_BORROW_ERROR = "E006"
alias E007_LIFETIME_ERROR = "E007"
alias E008_SYNTAX_ERROR = "E008"
alias E009_IMPORT_ERROR = "E009"
alias E010_ATTRIBUTE_ERROR = "E010"

# Runtime Messages (R001-R999)
alias R001_NULL_POINTER = "R001"
alias R002_INDEX_OUT_OF_BOUNDS = "R002"
alias R003_DIVISION_BY_ZERO = "R003"
alias R004_STACK_OVERFLOW = "R004"
alias R005_OUT_OF_MEMORY = "R005"

# CLI Messages (C001-C999)
alias C001_USAGE = "C001"
alias C002_VERSION = "C002"
alias C003_HELP = "C003"
alias C004_BUILD_SUCCESS = "C004"
alias C005_BUILD_FAILED = "C005"
alias C006_TEST_PASSED = "C006"
alias C007_TEST_FAILED = "C007"
alias C008_FILE_NOT_FOUND = "C008"

# ============================================================================
# Message Entry
# ============================================================================

struct MessageEntry:
    """A single translated message"""
    var id: String
    var text: String
    var plural_text: String  # For pluralization
    var context: String  # Disambiguation context

    fn __init__(out self, id: String, text: String):
        self.id = id
        self.text = text
        self.plural_text = ""
        self.context = ""

    fn __init__(out self, id: String, text: String, plural: String):
        self.id = id
        self.text = text
        self.plural_text = plural
        self.context = ""

    fn __init__(out self, id: String, text: String, plural: String, context: String):
        self.id = id
        self.text = text
        self.plural_text = plural
        self.context = context

    fn __copyinit__(out self, existing: Self):
        self.id = existing.id
        self.text = existing.text
        self.plural_text = existing.plural_text
        self.context = existing.context

# ============================================================================
# Language Catalog
# ============================================================================

struct LanguageCatalog:
    """Message catalog for a single language"""
    var language: Language
    var messages: Dict[String, MessageEntry]

    fn __init__(out self, language: Language):
        self.language = language
        self.messages = Dict[String, MessageEntry]()

    fn add(inout self, id: String, text: String):
        """Add a message to the catalog"""
        self.messages[id] = MessageEntry(id, text)

    fn add_plural(inout self, id: String, singular: String, plural: String):
        """Add a message with plural form"""
        self.messages[id] = MessageEntry(id, singular, plural)

    fn get(self, id: String) -> String:
        """Get message text by ID"""
        if id in self.messages:
            return self.messages[id].text
        return id  # Return ID as fallback

    fn get_plural(self, id: String, count: Int) -> String:
        """Get singular or plural form based on count"""
        if id in self.messages:
            let entry = self.messages[id]
            if count == 1 or len(entry.plural_text) == 0:
                return entry.text
            return entry.plural_text
        return id

    fn has(self, id: String) -> Bool:
        """Check if message ID exists"""
        return id in self.messages

# ============================================================================
# Message Catalog Manager
# ============================================================================

struct MessageCatalog:
    """
    Multi-language message catalog manager.
    Provides translation lookup with fallback chain.
    """
    var catalogs: Dict[String, LanguageCatalog]
    var current_language: String
    var fallback_language: String

    fn __init__(out self):
        self.catalogs = Dict[String, LanguageCatalog]()
        self.current_language = "en"
        self.fallback_language = "en"
        self._init_builtin_catalogs()

    fn _init_builtin_catalogs(inout self):
        """Initialize built-in message catalogs"""
        # English catalog (default)
        var en = LanguageCatalog(english())
        self._populate_english(en)
        self.catalogs["en"] = en

        # Arabic catalog
        var ar = LanguageCatalog(arabic())
        self._populate_arabic(ar)
        self.catalogs["ar"] = ar

    fn _populate_english(inout self, inout catalog: LanguageCatalog):
        """Populate English messages"""
        # Compiler errors
        catalog.add(E001_TYPE_MISMATCH, "Type mismatch: expected '{0}', got '{1}'")
        catalog.add(E002_UNDEFINED_VARIABLE, "Undefined variable: '{0}'")
        catalog.add(E003_UNDEFINED_FUNCTION, "Undefined function: '{0}'")
        catalog.add(E004_ARGUMENT_COUNT, "Wrong number of arguments: expected {0}, got {1}")
        catalog.add(E005_RETURN_TYPE, "Return type mismatch: expected '{0}', got '{1}'")
        catalog.add(E006_BORROW_ERROR, "Cannot borrow '{0}' as mutable, already borrowed as immutable")
        catalog.add(E007_LIFETIME_ERROR, "Lifetime '{0}' does not live long enough")
        catalog.add(E008_SYNTAX_ERROR, "Syntax error: {0}")
        catalog.add(E009_IMPORT_ERROR, "Cannot import module: '{0}'")
        catalog.add(E010_ATTRIBUTE_ERROR, "Unknown attribute: '{0}'")

        # Runtime errors
        catalog.add(R001_NULL_POINTER, "Null pointer dereference")
        catalog.add(R002_INDEX_OUT_OF_BOUNDS, "Index {0} out of bounds for length {1}")
        catalog.add(R003_DIVISION_BY_ZERO, "Division by zero")
        catalog.add(R004_STACK_OVERFLOW, "Stack overflow")
        catalog.add(R005_OUT_OF_MEMORY, "Out of memory")

        # CLI messages
        catalog.add(C001_USAGE, "Usage: mojo <command> [options] [files]")
        catalog.add(C002_VERSION, "Mojo SDK version {0}")
        catalog.add(C003_HELP, "Use 'mojo help <command>' for more information")
        catalog.add(C004_BUILD_SUCCESS, "Build succeeded in {0}ms")
        catalog.add(C005_BUILD_FAILED, "Build failed with {0} error(s)")
        catalog.add(C006_TEST_PASSED, "{0} test(s) passed")
        catalog.add(C007_TEST_FAILED, "{0} test(s) failed")
        catalog.add(C008_FILE_NOT_FOUND, "File not found: '{0}'")

        # Plurals
        catalog.add_plural("error_count", "{0} error", "{0} errors")
        catalog.add_plural("warning_count", "{0} warning", "{0} warnings")
        catalog.add_plural("file_count", "{0} file", "{0} files")

    fn _populate_arabic(inout self, inout catalog: LanguageCatalog):
        """Populate Arabic messages"""
        # Compiler errors
        catalog.add(E001_TYPE_MISMATCH, "عدم تطابق النوع: متوقع '{0}'، حصلت على '{1}'")
        catalog.add(E002_UNDEFINED_VARIABLE, "متغير غير معرّف: '{0}'")
        catalog.add(E003_UNDEFINED_FUNCTION, "دالة غير معرّفة: '{0}'")
        catalog.add(E004_ARGUMENT_COUNT, "عدد خاطئ من المعاملات: متوقع {0}، حصلت على {1}")
        catalog.add(E005_RETURN_TYPE, "عدم تطابق نوع الإرجاع: متوقع '{0}'، حصلت على '{1}'")
        catalog.add(E006_BORROW_ERROR, "لا يمكن استعارة '{0}' كقابل للتعديل، مستعار بالفعل كغير قابل للتعديل")
        catalog.add(E007_LIFETIME_ERROR, "مدة الحياة '{0}' غير كافية")
        catalog.add(E008_SYNTAX_ERROR, "خطأ نحوي: {0}")
        catalog.add(E009_IMPORT_ERROR, "لا يمكن استيراد الوحدة: '{0}'")
        catalog.add(E010_ATTRIBUTE_ERROR, "سمة غير معروفة: '{0}'")

        # Runtime errors
        catalog.add(R001_NULL_POINTER, "إلغاء مرجع مؤشر فارغ")
        catalog.add(R002_INDEX_OUT_OF_BOUNDS, "الفهرس {0} خارج النطاق للطول {1}")
        catalog.add(R003_DIVISION_BY_ZERO, "القسمة على صفر")
        catalog.add(R004_STACK_OVERFLOW, "تجاوز سعة المكدس")
        catalog.add(R005_OUT_OF_MEMORY, "نفاد الذاكرة")

        # CLI messages
        catalog.add(C001_USAGE, "الاستخدام: mojo <أمر> [خيارات] [ملفات]")
        catalog.add(C002_VERSION, "إصدار Mojo SDK {0}")
        catalog.add(C003_HELP, "استخدم 'mojo help <أمر>' لمزيد من المعلومات")
        catalog.add(C004_BUILD_SUCCESS, "نجح البناء في {0} مللي ثانية")
        catalog.add(C005_BUILD_FAILED, "فشل البناء مع {0} خطأ(أخطاء)")
        catalog.add(C006_TEST_PASSED, "نجح {0} اختبار(اختبارات)")
        catalog.add(C007_TEST_FAILED, "فشل {0} اختبار(اختبارات)")
        catalog.add(C008_FILE_NOT_FOUND, "الملف غير موجود: '{0}'")

        # Plurals (Arabic has complex plural rules)
        catalog.add_plural("error_count", "خطأ واحد", "{0} أخطاء")
        catalog.add_plural("warning_count", "تحذير واحد", "{0} تحذيرات")
        catalog.add_plural("file_count", "ملف واحد", "{0} ملفات")

    fn set_language(inout self, lang_code: String):
        """Set current language"""
        self.current_language = lang_code

    fn get(self, id: String) -> String:
        """Get translated message"""
        # Try current language
        if self.current_language in self.catalogs:
            let catalog = self.catalogs[self.current_language]
            if catalog.has(id):
                return catalog.get(id)

        # Try fallback language
        if self.fallback_language in self.catalogs:
            let catalog = self.catalogs[self.fallback_language]
            if catalog.has(id):
                return catalog.get(id)

        # Return ID as last resort
        return id

    fn get_plural(self, id: String, count: Int) -> String:
        """Get translated message with plural handling"""
        if self.current_language in self.catalogs:
            let catalog = self.catalogs[self.current_language]
            if catalog.has(id):
                return catalog.get_plural(id, count)

        if self.fallback_language in self.catalogs:
            let catalog = self.catalogs[self.fallback_language]
            if catalog.has(id):
                return catalog.get_plural(id, count)

        return id

    fn format(self, id: String, *args: String) -> String:
        """Get translated message and format with arguments"""
        var template = self.get(id)
        return self._format_string(template, args)

    fn _format_string(self, template: String, args: VariadicList[String]) -> String:
        """Format string with positional arguments {0}, {1}, etc."""
        var result = template

        for i in range(len(args)):
            let placeholder = "{" + String(i) + "}"
            result = result.replace(placeholder, args[i])

        return result

    fn add_catalog(inout self, lang_code: String, catalog: LanguageCatalog):
        """Add a custom language catalog"""
        self.catalogs[lang_code] = catalog

    fn get_supported_languages(self) -> List[String]:
        """Get list of supported language codes"""
        var languages = List[String]()
        # Note: In real implementation, iterate over catalog keys
        languages.append("en")
        languages.append("ar")
        return languages

# ============================================================================
# Global Message Catalog Instance
# ============================================================================

var _global_catalog: MessageCatalog = MessageCatalog()

fn get_catalog() -> MessageCatalog:
    """Get the global message catalog"""
    return _global_catalog

fn set_language(lang_code: String):
    """Set the global language"""
    _global_catalog.set_language(lang_code)

fn tr(id: String) -> String:
    """Translate message by ID (shorthand)"""
    return _global_catalog.get(id)

fn tr_format(id: String, *args: String) -> String:
    """Translate and format message (shorthand)"""
    return _global_catalog.format(id, args)

fn tr_plural(id: String, count: Int) -> String:
    """Translate message with plural handling (shorthand)"""
    return _global_catalog.get_plural(id, count)

# ============================================================================
# Compile-Time Message Validation (via traits)
# ============================================================================

trait Translatable:
    """Trait for types that can provide translated representations"""
    fn to_translated_string(self, locale: Locale) -> String

trait LocalizedError:
    """Trait for errors that provide localized messages"""
    fn error_id(self) -> String
    fn localized_message(self, locale: Locale) -> String
