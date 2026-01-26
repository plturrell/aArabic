# __init__.mojo
# Mojo SDK Internationalization Module
# Pure Mojo/Zig implementation - no external dependencies

from .locale import (
    Language,
    Locale,
    LocaleContext,
    LanguageRegistry,
    NumberFormatter,
    TextDirection,
    # Language constructors
    arabic,
    english,
    french,
    german,
    spanish,
    chinese,
    japanese,
    korean,
    russian,
    hebrew,
    persian,
    urdu,
    hindi,
    portuguese,
    italian,
    dutch,
    turkish,
    # Locale constructors
    locale_en_us,
    locale_ar_sa,
    locale_ar_eg,
    locale_fr_fr,
    locale_de_de,
    locale_es_es,
    locale_zh_cn,
    locale_ja_jp,
    locale_ko_kr,
    locale_ru_ru,
    locale_he_il,
    locale_fa_ir,
    locale_ur_pk,
    locale_hi_in,
)

from .messages import (
    MessageCatalog,
    MessageEntry,
    LanguageCatalog,
    # Message ID constants
    E001_TYPE_MISMATCH,
    E002_UNDEFINED_VARIABLE,
    E003_UNDEFINED_FUNCTION,
    E004_ARGUMENT_COUNT,
    E005_RETURN_TYPE,
    E006_BORROW_ERROR,
    E007_LIFETIME_ERROR,
    E008_SYNTAX_ERROR,
    E009_IMPORT_ERROR,
    E010_ATTRIBUTE_ERROR,
    R001_NULL_POINTER,
    R002_INDEX_OUT_OF_BOUNDS,
    R003_DIVISION_BY_ZERO,
    R004_STACK_OVERFLOW,
    R005_OUT_OF_MEMORY,
    C001_USAGE,
    C002_VERSION,
    C003_HELP,
    C004_BUILD_SUCCESS,
    C005_BUILD_FAILED,
    C006_TEST_PASSED,
    C007_TEST_FAILED,
    C008_FILE_NOT_FOUND,
    # Global functions
    get_catalog,
    set_language,
    tr,
    tr_format,
    tr_plural,
    # Traits
    Translatable,
    LocalizedError,
)

# ============================================================================
# Module-level convenience functions
# ============================================================================

fn init_i18n(locale: Locale = locale_en_us()):
    """Initialize i18n system with specified locale"""
    set_language(locale.language.code)

fn get_current_language() -> Language:
    """Get the currently active language"""
    return get_catalog().catalogs[get_catalog().current_language].language

fn is_rtl() -> Bool:
    """Check if current language is right-to-left"""
    return get_current_language().rtl

fn format_number(value: Int) -> String:
    """Format number according to current locale"""
    let ctx = LocaleContext()
    let formatter = NumberFormatter(ctx.get())
    return formatter.format_integer(value)

fn format_number_with_sep(value: Int) -> String:
    """Format number with thousands separator"""
    let ctx = LocaleContext()
    let formatter = NumberFormatter(ctx.get())
    return formatter.format_with_separator(value)
