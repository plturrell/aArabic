# Mojo SDK Developer Guide - Internationalization (i18n)

**Version:** 1.0.0
**Status:** Production Ready
**Last Updated:** January 2026

---

## Overview

The Mojo SDK provides comprehensive internationalization (i18n) support at both the compiler and standard library levels. This enables:

- **33 languages** fully supported at the compiler level
- **55 languages** for runtime translation via TranslateGemma
- **CLDR-compliant pluralization** for linguistically correct plural forms
- **RTL (Right-to-Left) text support** for Arabic, Hebrew, Persian, and Urdu
- **Locale-aware formatting** for dates, times, and numbers

---

## Architecture

### Two-Layer i18n System

```
+----------------------+     +------------------------+
|   Mojo stdlib i18n   |     |   Zig compiler i18n    |
|  (Runtime support)   |     |  (Compile-time msgs)   |
+----------------------+     +------------------------+
         |                            |
         v                            v
+----------------------+     +------------------------+
|  TranslateGemma-27B  |     |  Compile-time catalog  |
|  (55 languages)      |     |  (33 languages)        |
+----------------------+     +------------------------+
```

### Compiler-Level i18n (Zig)

Located in `compiler/i18n/`:
- `i18n.zig` - Core message catalog and language definitions
- `plurals.zig` - CLDR-based pluralization rules
- `locale_detect.zig` - System locale detection
- `datetime.zig` - Locale-aware date/time formatting

### Standard Library i18n (Mojo)

Located in `stdlib/i18n/`:
- `locale.mojo` - Language and locale definitions
- `messages.mojo` - Message catalog system
- `translation.mojo` - TranslateGemma integration

---

## Quick Start

### Setting the Language

```mojo
from stdlib.i18n import set_current_language, get_current_language

# Set language by code
set_current_language("ar")  # Arabic
set_current_language("zh")  # Chinese
set_current_language("ru")  # Russian

# Get current language
let lang = get_current_language()
print("Current language:", lang.code)
```

### Translating Messages

```mojo
from stdlib.i18n.translation import translate, t

# Full translation
let result = translate("Hello, world!", "en", "ar")
print(result)  # مرحبا بالعالم!

# Shorthand with auto-detect source
let arabic = t("Compile error", "ar")
```

### Using Message Catalogs

```mojo
from stdlib.i18n.messages import get_catalog

var catalog = get_catalog()
catalog.set_language("ar")

let msg = catalog.get("E001")  # Gets Arabic translation
print(msg)  # خطأ في الصياغة
```

---

## Supported Languages

### Tier 1 Languages (Full Support)

| Code | Language | Native Name | RTL |
|------|----------|-------------|-----|
| en | English | English | No |
| ar | Arabic | العربية | Yes |
| zh | Chinese | 中文 | No |
| ja | Japanese | 日本語 | No |
| ko | Korean | 한국어 | No |
| es | Spanish | Español | No |
| fr | French | Français | No |
| de | German | Deutsch | No |
| ru | Russian | Русский | No |
| pt | Portuguese | Português | No |

### Tier 2 Languages (Compiler Messages)

| Code | Language | Native Name | RTL |
|------|----------|-------------|-----|
| he | Hebrew | עברית | Yes |
| fa | Persian | فارسی | Yes |
| ur | Urdu | اردو | Yes |
| hi | Hindi | हिन्दी | No |
| th | Thai | ไทย | No |
| vi | Vietnamese | Tiếng Việt | No |
| id | Indonesian | Bahasa Indonesia | No |
| tr | Turkish | Türkçe | No |
| pl | Polish | Polski | No |
| nl | Dutch | Nederlands | No |
| it | Italian | Italiano | No |
| cs | Czech | Čeština | No |
| sv | Swedish | Svenska | No |
| da | Danish | Dansk | No |
| fi | Finnish | Suomi | No |
| no | Norwegian | Norsk | No |
| uk | Ukrainian | Українська | No |
| ro | Romanian | Română | No |
| hu | Hungarian | Magyar | No |
| el | Greek | Ελληνικά | No |
| bg | Bulgarian | Български | No |
| hr | Croatian | Hrvatski | No |
| sk | Slovak | Slovenčina | No |

### Tier 3 Languages (TranslateGemma Only)

Additional 22 languages supported via TranslateGemma for runtime translation.

---

## Pluralization

### CLDR Plural Categories

The SDK implements CLDR-compliant plural rules with 6 categories:

| Category | Description | Example Languages |
|----------|-------------|-------------------|
| `zero` | Zero quantity | Arabic |
| `one` | Singular | English, German, Spanish |
| `two` | Dual | Arabic, Hebrew |
| `few` | Paucal (small) | Russian, Polish, Arabic |
| `many` | Large quantities | Russian, Arabic |
| `other` | Default/remaining | All languages |

### Plural Rules by Language Family

```zig
// Asian languages (no plurals)
// zh, ja, ko, vi, th, id
getPluralCategory("zh", 1)  // .other
getPluralCategory("zh", 100)  // .other

// Germanic languages (one/other)
// en, de, es, it, pt, nl
getPluralCategory("en", 1)  // .one
getPluralCategory("en", 2)  // .other

// Slavic languages (one/few/many/other)
// ru, uk, pl, cs
getPluralCategory("ru", 1)   // .one
getPluralCategory("ru", 2)   // .few
getPluralCategory("ru", 5)   // .many
getPluralCategory("ru", 21)  // .one

// Arabic (zero/one/two/few/many/other)
getPluralCategory("ar", 0)   // .zero
getPluralCategory("ar", 1)   // .one
getPluralCategory("ar", 2)   // .two
getPluralCategory("ar", 5)   // .few
getPluralCategory("ar", 15)  // .many
getPluralCategory("ar", 100) // .other
```

### Using Plural Forms

```zig
const plurals = @import("compiler/i18n/plurals.zig");

// Define plural forms
const error_forms = plurals.PluralForms{
    .zero = "no errors",
    .one = "1 error",
    .two = "2 errors",
    .few = "{d} errors",
    .many = "{d} errors",
    .other = "{d} errors",
};

// Arabic plural forms
const ar_error_forms = plurals.PluralForms{
    .zero = "لا أخطاء",
    .one = "خطأ واحد",
    .two = "خطآن",
    .few = "{d} أخطاء",
    .many = "{d} خطأ",
    .other = "{d} خطأ",
};

// Select correct form
const msg = error_forms.select("en", 5);  // "5 errors"
const ar_msg = ar_error_forms.select("ar", 2);  // "خطآن"
```

---

## RTL (Right-to-Left) Support

### RTL Languages

| Code | Language | Script |
|------|----------|--------|
| ar | Arabic | Arabic |
| he | Hebrew | Hebrew |
| fa | Persian | Arabic |
| ur | Urdu | Arabic |

### Checking RTL Status

```mojo
from stdlib.i18n.locale import Language, is_rtl

let lang = Language("ar", "Arabic", "العربية", True, "Arab")
if lang.rtl:
    # Apply RTL formatting
    pass

# Global check
if is_rtl():
    print("RTL mode enabled")
```

### RTL Text Direction

```mojo
from stdlib.i18n.locale import TextDirection

var direction = TextDirection()
direction.set_language("ar")

print(direction.get_dir())  # "rtl"
print(direction.get_align())  # "right"
```

---

## Date/Time Formatting

### Date Formats

```zig
const datetime = @import("compiler/i18n/datetime.zig");

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

// Different formats
formatter.formatDate(date, .short, &buf);  // "1/23/26"
formatter.formatDate(date, .medium, &buf); // "Jan 23, 2026"
formatter.formatDate(date, .long, &buf);   // "January 23, 2026"
formatter.formatDate(date, .iso, &buf);    // "2026-01-23"
```

### Localized Month/Day Names

```zig
// English
const en_months = datetime.getMonthNames("en");
en_months.full[0]  // "January"

// Arabic
const ar_months = datetime.getMonthNames("ar");
ar_months.full[0]  // "يناير"

// Japanese
const ja_months = datetime.getMonthNames("ja");
ja_months.full[0]  // "1月"
```

### Relative Time

```zig
// English
formatter.formatRelative(0, &buf);      // "just now"
formatter.formatRelative(-60, &buf);    // "1 minute ago"
formatter.formatRelative(-3600, &buf);  // "1 hour ago"
formatter.formatRelative(-86400, &buf); // "1 day ago"

// Arabic (with formatter.init("ar"))
formatter.formatRelative(-86400, &buf); // "منذ يوم"
```

---

## Locale Detection

### Automatic Detection

```zig
const locale_detect = @import("compiler/i18n/locale_detect.zig");

// Auto-initialize from system locale
locale_detect.autoInit(allocator);

// Manual detection
var detector = locale_detect.LocaleDetector.init(allocator);
const detected = detector.detect();

print("Language: {s}\n", .{detected.language});  // "en"
print("Region: {s}\n", .{detected.region});      // "US"
print("Encoding: {s}\n", .{detected.encoding});  // "UTF-8"
```

### Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `MOJO_LANG` | Override language | `MOJO_LANG=ar` |
| `MOJO_RTL` | Force RTL mode | `MOJO_RTL=1` |
| `MOJO_NO_I18N` | Disable i18n | `MOJO_NO_I18N=1` |
| `LANG` | System locale | `LANG=ar_SA.UTF-8` |
| `LC_ALL` | Override all locale | `LC_ALL=fr_FR.UTF-8` |

### Detection Priority

1. `MOJO_LANG` environment variable (highest)
2. `LC_ALL` environment variable
3. `LC_MESSAGES` environment variable
4. `LANG` environment variable
5. Platform-specific detection (macOS/Windows)
6. Default to English (lowest)

---

## Number Formatting

### Arabic-Indic Numerals

```mojo
from stdlib.i18n.locale import NumberFormatter

var formatter = NumberFormatter()
formatter.use_arabic_numerals = True

let num = formatter.format_number(12345)
print(num)  # ١٢٣٤٥

let decimal = formatter.format_decimal(3.14159)
print(decimal)  # ٣.١٤١٥٩
```

### Supported Numeral Systems

| Language | Numerals | Example |
|----------|----------|---------|
| Arabic | Arabic-Indic | ٠١٢٣٤٥٦٧٨٩ |
| Persian | Extended Arabic-Indic | ۰۱۲۳۴۵۶۷۸۹ |
| Hindi | Devanagari | ०१२३४५६७८९ |
| Thai | Thai | ๐๑๒๓๔๕๖๗๘๙ |
| Chinese | CJK | 〇一二三四五六七八九 |

---

## TranslateGemma Integration

### Overview

The SDK integrates TranslateGemma-27B-IT for high-quality runtime translation supporting 55 languages.

### Configuration

```mojo
from stdlib.i18n.translation import TranslateGemmaProvider

# Default local endpoint
var provider = TranslateGemmaProvider()

# Custom endpoint
var custom = TranslateGemmaProvider("http://custom:11435/v1/chat/completions")
```

### Translation API

```mojo
from stdlib.i18n.translation import TranslationRequest, TranslationManager

var manager = TranslationManager()

# Basic translation
let result = manager.translate("Hello, world!", "en", "ar")
if result.success:
    print(result.translated)  # مرحبا بالعالم!
    print("Confidence:", result.confidence)

# With auto-detect source
let text = manager.translate_to("Bonjour le monde", "en")
print(text)  # "Hello world"
```

### Caching

```mojo
# Translation results are cached
let stats = manager.get_cache_stats()
print("Cache size:", stats.0, "/", stats.1)

# Clear cache
manager.clear_cache()
```

### Localizable Strings

```mojo
from stdlib.i18n.translation import L, LocalizedString

# Create localizable string
let greeting = L("Welcome to Mojo!")

# Get translations
print(greeting.get("en"))  # "Welcome to Mojo!"
print(greeting.get("ar"))  # "مرحبا بموجو!"
print(greeting.get("zh"))  # "欢迎使用Mojo!"

# Preload translations
greeting.preload(["en", "ar", "zh", "ja", "ko"])
```

---

## Compiler Messages

### Message IDs

The compiler uses structured message IDs for all diagnostics:

| Prefix | Category | Example |
|--------|----------|---------|
| `E0xx` | Syntax Errors | E001: Syntax error |
| `E1xx` | Type Errors | E101: Type mismatch |
| `E2xx` | Name Errors | E201: Undefined variable |
| `W0xx` | Warnings | W001: Unused variable |
| `R0xx` | Runtime Errors | R001: Division by zero |
| `C0xx` | Compiler Messages | C001: Compiling |

### Localized Error Messages

```zig
const i18n = @import("compiler/i18n/i18n.zig");

// Set language
i18n.setLanguageByCode("ar");

// Get localized message
const msg = i18n.getMessage(.syntax_error);
// Returns: "خطأ في الصياغة"

// Format with arguments
const formatted = i18n.formatMessage(.type_mismatch, .{"Int", "String"});
// Returns: "عدم تطابق النوع: متوقع 'Int' ولكن وجد 'String'"
```

---

## Best Practices

### 1. Use Message IDs, Not Hardcoded Strings

```mojo
# Bad
print("Error: file not found")

# Good
print(catalog.get("E201"))
```

### 2. Design for Pluralization

```mojo
# Bad
print(str(count) + " file(s) processed")

# Good
let msg = pluralize("file", count)
print(msg)
```

### 3. Respect RTL Layout

```mojo
# Check RTL before rendering
if is_rtl():
    render_rtl_layout()
else:
    render_ltr_layout()
```

### 4. Use Locale-Aware Formatting

```mojo
# Bad
print(str(month) + "/" + str(day) + "/" + str(year))

# Good
print(format_date(date, current_locale()))
```

### 5. Cache Translations

```mojo
# Preload common translations at startup
manager.preload_translations(["en", "ar", "zh"])
```

---

## CLI Integration

### Language Selection

```bash
# Set via environment
export MOJO_LANG=ar
mojo build hello.mojo

# Compiler messages in Arabic
خطأ: لم يتم العثور على الملف 'hello.mojo'
```

### RTL Terminal Output

When RTL is enabled, the CLI automatically:
- Aligns text to the right
- Reverses bidirectional text segments
- Uses appropriate Unicode markers

---

## API Reference

### Compiler i18n (`compiler/i18n/`)

#### i18n.zig
- `Language` - Language definition struct
- `MessageId` - Compile-time message IDs
- `getMessage(id)` - Get localized message
- `setLanguage(lang)` - Set current language
- `setLanguageByCode(code)` - Set by ISO code

#### plurals.zig
- `PluralCategory` - Plural category enum
- `PluralRule` - Language family rules
- `PluralForms` - Plural form definitions
- `getPluralCategory(lang, n)` - Get category for count

#### locale_detect.zig
- `LocaleDetector` - System locale detection
- `I18nConfig` - Environment configuration
- `autoInit(alloc)` - Auto-initialize i18n

#### datetime.zig
- `DateTimeFormatter` - Locale-aware formatting
- `getMonthNames(lang)` - Localized month names
- `getDayNames(lang)` - Localized day names

### Standard Library (`stdlib/i18n/`)

#### locale.mojo
- `Language` - Language struct
- `Locale` - Full locale (language + region)
- `NumberFormatter` - Number formatting
- `TextDirection` - RTL/LTR handling

#### messages.mojo
- `MessageCatalog` - Message storage
- `get_catalog()` - Get global catalog
- `LanguageCatalog` - Per-language messages

#### translation.mojo
- `TranslationManager` - Translation orchestration
- `TranslateGemmaProvider` - TranslateGemma backend
- `translate(text, src, tgt)` - Translate text
- `L(text)` - Create localizable string

---

## Troubleshooting

### Common Issues

**Q: Translations not loading**
- Check `MOJO_LANG` environment variable
- Verify locale files exist
- Check TranslateGemma endpoint is running

**Q: Incorrect plural forms**
- Verify language code is correct
- Check CLDR plural rules for your language
- Ensure count is passed correctly

**Q: RTL not working**
- Set `MOJO_RTL=1` to force RTL
- Check terminal supports RTL
- Verify language is marked as RTL

**Q: Date format incorrect**
- Check locale region code
- Verify DateFormat enum value
- Ensure DateTimeFormatter is initialized with correct language

---

## Contributing

To add support for a new language:

1. Add language definition to `compiler/i18n/i18n.zig`
2. Add plural rules if needed in `plurals.zig`
3. Add month/day names to `datetime.zig`
4. Add message translations to message catalog
5. Add tests to `compiler/tests/test_i18n.zig`
6. Update documentation

---

**Previous:** [17 - Migration Guides](17-migration-guides.md)

---

*Mojo SDK Developer Guide v1.0.0*
*Last Updated: January 2026*
