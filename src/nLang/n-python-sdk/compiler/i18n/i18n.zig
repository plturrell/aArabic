// i18n.zig
// Internationalization support for Mojo SDK compiler
// Pure Zig implementation - no external dependencies

const std = @import("std");

// ============================================================================
// Language Definition
// ============================================================================

pub const Language = struct {
    code: []const u8,        // ISO 639-1 code (e.g., "en", "ar")
    name: []const u8,        // English name
    native_name: []const u8, // Name in native script
    rtl: bool,               // Right-to-left writing direction
    script: []const u8,      // Primary script (e.g., "Arab", "Latn")

    pub fn isRtl(self: Language) bool {
        return self.rtl;
    }
};

// Pre-defined languages
pub const LANG_ENGLISH = Language{
    .code = "en",
    .name = "English",
    .native_name = "English",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_ARABIC = Language{
    .code = "ar",
    .name = "Arabic",
    .native_name = "العربية",
    .rtl = true,
    .script = "Arab",
};

pub const LANG_FRENCH = Language{
    .code = "fr",
    .name = "French",
    .native_name = "Français",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_GERMAN = Language{
    .code = "de",
    .name = "German",
    .native_name = "Deutsch",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_SPANISH = Language{
    .code = "es",
    .name = "Spanish",
    .native_name = "Español",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_CHINESE = Language{
    .code = "zh",
    .name = "Chinese",
    .native_name = "中文",
    .rtl = false,
    .script = "Hans",
};

pub const LANG_JAPANESE = Language{
    .code = "ja",
    .name = "Japanese",
    .native_name = "日本語",
    .rtl = false,
    .script = "Jpan",
};

pub const LANG_RUSSIAN = Language{
    .code = "ru",
    .name = "Russian",
    .native_name = "Русский",
    .rtl = false,
    .script = "Cyrl",
};

pub const LANG_HEBREW = Language{
    .code = "he",
    .name = "Hebrew",
    .native_name = "עברית",
    .rtl = true,
    .script = "Hebr",
};

pub const LANG_PERSIAN = Language{
    .code = "fa",
    .name = "Persian",
    .native_name = "فارسی",
    .rtl = true,
    .script = "Arab",
};

pub const LANG_PORTUGUESE = Language{
    .code = "pt",
    .name = "Portuguese",
    .native_name = "Português",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_ITALIAN = Language{
    .code = "it",
    .name = "Italian",
    .native_name = "Italiano",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_DUTCH = Language{
    .code = "nl",
    .name = "Dutch",
    .native_name = "Nederlands",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_TURKISH = Language{
    .code = "tr",
    .name = "Turkish",
    .native_name = "Türkçe",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_KOREAN = Language{
    .code = "ko",
    .name = "Korean",
    .native_name = "한국어",
    .rtl = false,
    .script = "Kore",
};

pub const LANG_HINDI = Language{
    .code = "hi",
    .name = "Hindi",
    .native_name = "हिन्दी",
    .rtl = false,
    .script = "Deva",
};

pub const LANG_URDU = Language{
    .code = "ur",
    .name = "Urdu",
    .native_name = "اردو",
    .rtl = true,
    .script = "Arab",
};

pub const LANG_INDONESIAN = Language{
    .code = "id",
    .name = "Indonesian",
    .native_name = "Bahasa Indonesia",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_VIETNAMESE = Language{
    .code = "vi",
    .name = "Vietnamese",
    .native_name = "Tiếng Việt",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_THAI = Language{
    .code = "th",
    .name = "Thai",
    .native_name = "ภาษาไทย",
    .rtl = false,
    .script = "Thai",
};

pub const LANG_POLISH = Language{
    .code = "pl",
    .name = "Polish",
    .native_name = "Polski",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_UKRAINIAN = Language{
    .code = "uk",
    .name = "Ukrainian",
    .native_name = "Українська",
    .rtl = false,
    .script = "Cyrl",
};

pub const LANG_SWEDISH = Language{
    .code = "sv",
    .name = "Swedish",
    .native_name = "Svenska",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_CZECH = Language{
    .code = "cs",
    .name = "Czech",
    .native_name = "Čeština",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_GREEK = Language{
    .code = "el",
    .name = "Greek",
    .native_name = "Ελληνικά",
    .rtl = false,
    .script = "Grek",
};

pub const LANG_ROMANIAN = Language{
    .code = "ro",
    .name = "Romanian",
    .native_name = "Română",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_HUNGARIAN = Language{
    .code = "hu",
    .name = "Hungarian",
    .native_name = "Magyar",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_FINNISH = Language{
    .code = "fi",
    .name = "Finnish",
    .native_name = "Suomi",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_DANISH = Language{
    .code = "da",
    .name = "Danish",
    .native_name = "Dansk",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_NORWEGIAN = Language{
    .code = "no",
    .name = "Norwegian",
    .native_name = "Norsk",
    .rtl = false,
    .script = "Latn",
};

pub const LANG_BENGALI = Language{
    .code = "bn",
    .name = "Bengali",
    .native_name = "বাংলা",
    .rtl = false,
    .script = "Beng",
};

pub const LANG_TAMIL = Language{
    .code = "ta",
    .name = "Tamil",
    .native_name = "தமிழ்",
    .rtl = false,
    .script = "Taml",
};

pub const LANG_SWAHILI = Language{
    .code = "sw",
    .name = "Swahili",
    .native_name = "Kiswahili",
    .rtl = false,
    .script = "Latn",
};

// All supported languages array
pub const ALL_LANGUAGES = [_]Language{
    LANG_ENGLISH, LANG_ARABIC, LANG_FRENCH, LANG_GERMAN, LANG_SPANISH,
    LANG_CHINESE, LANG_JAPANESE, LANG_RUSSIAN, LANG_HEBREW, LANG_PERSIAN,
    LANG_PORTUGUESE, LANG_ITALIAN, LANG_DUTCH, LANG_TURKISH, LANG_KOREAN,
    LANG_HINDI, LANG_URDU, LANG_INDONESIAN, LANG_VIETNAMESE, LANG_THAI,
    LANG_POLISH, LANG_UKRAINIAN, LANG_SWEDISH, LANG_CZECH, LANG_GREEK,
    LANG_ROMANIAN, LANG_HUNGARIAN, LANG_FINNISH, LANG_DANISH, LANG_NORWEGIAN,
    LANG_BENGALI, LANG_TAMIL, LANG_SWAHILI,
};

// ============================================================================
// Message IDs
// ============================================================================

pub const MessageId = enum(u32) {
    // Compiler Errors (E001-E099)
    E001_TYPE_MISMATCH = 1,
    E002_UNDEFINED_VARIABLE = 2,
    E003_UNDEFINED_FUNCTION = 3,
    E004_ARGUMENT_COUNT = 4,
    E005_RETURN_TYPE = 5,
    E006_BORROW_ERROR = 6,
    E007_LIFETIME_ERROR = 7,
    E008_SYNTAX_ERROR = 8,
    E009_IMPORT_ERROR = 9,
    E010_ATTRIBUTE_ERROR = 10,

    // Runtime Errors (R001-R099)
    R001_NULL_POINTER = 101,
    R002_INDEX_OUT_OF_BOUNDS = 102,
    R003_DIVISION_BY_ZERO = 103,
    R004_STACK_OVERFLOW = 104,
    R005_OUT_OF_MEMORY = 105,

    // CLI Messages (C001-C099)
    C001_USAGE = 201,
    C002_VERSION = 202,
    C003_HELP = 203,
    C004_BUILD_SUCCESS = 204,
    C005_BUILD_FAILED = 205,
    C006_TEST_PASSED = 206,
    C007_TEST_FAILED = 207,
    C008_FILE_NOT_FOUND = 208,

    // LSP Messages (L001-L099)
    L001_HOVER_TYPE = 301,
    L002_COMPLETION_HINT = 302,
    L003_DIAGNOSTIC_ERROR = 303,
    L004_DIAGNOSTIC_WARNING = 304,

    pub fn toCode(self: MessageId) []const u8 {
        return switch (self) {
            .E001_TYPE_MISMATCH => "E001",
            .E002_UNDEFINED_VARIABLE => "E002",
            .E003_UNDEFINED_FUNCTION => "E003",
            .E004_ARGUMENT_COUNT => "E004",
            .E005_RETURN_TYPE => "E005",
            .E006_BORROW_ERROR => "E006",
            .E007_LIFETIME_ERROR => "E007",
            .E008_SYNTAX_ERROR => "E008",
            .E009_IMPORT_ERROR => "E009",
            .E010_ATTRIBUTE_ERROR => "E010",
            .R001_NULL_POINTER => "R001",
            .R002_INDEX_OUT_OF_BOUNDS => "R002",
            .R003_DIVISION_BY_ZERO => "R003",
            .R004_STACK_OVERFLOW => "R004",
            .R005_OUT_OF_MEMORY => "R005",
            .C001_USAGE => "C001",
            .C002_VERSION => "C002",
            .C003_HELP => "C003",
            .C004_BUILD_SUCCESS => "C004",
            .C005_BUILD_FAILED => "C005",
            .C006_TEST_PASSED => "C006",
            .C007_TEST_FAILED => "C007",
            .C008_FILE_NOT_FOUND => "C008",
            .L001_HOVER_TYPE => "L001",
            .L002_COMPLETION_HINT => "L002",
            .L003_DIAGNOSTIC_ERROR => "L003",
            .L004_DIAGNOSTIC_WARNING => "L004",
        };
    }
};

// ============================================================================
// Message Catalog (Compile-time generated)
// ============================================================================

pub const MessageEntry = struct {
    id: MessageId,
    en: []const u8,  // English (required)
    ar: []const u8,  // Arabic
    fr: []const u8,  // French
    de: []const u8,  // German
    es: []const u8,  // Spanish
    zh: []const u8,  // Chinese
    ja: []const u8,  // Japanese
    ru: []const u8,  // Russian
};

// Compile-time message catalog
pub const MESSAGE_CATALOG = [_]MessageEntry{
    // Compiler Errors
    .{
        .id = .E001_TYPE_MISMATCH,
        .en = "Type mismatch: expected '{s}', got '{s}'",
        .ar = "عدم تطابق النوع: متوقع '{s}'، حصلت على '{s}'",
        .fr = "Incompatibilité de type: attendu '{s}', obtenu '{s}'",
        .de = "Typfehler: erwartet '{s}', erhalten '{s}'",
        .es = "Tipo no coincide: esperado '{s}', obtenido '{s}'",
        .zh = "类型不匹配：期望 '{s}'，得到 '{s}'",
        .ja = "型の不一致：期待 '{s}'、取得 '{s}'",
        .ru = "Несоответствие типов: ожидалось '{s}', получено '{s}'",
    },
    .{
        .id = .E002_UNDEFINED_VARIABLE,
        .en = "Undefined variable: '{s}'",
        .ar = "متغير غير معرّف: '{s}'",
        .fr = "Variable non définie: '{s}'",
        .de = "Undefinierte Variable: '{s}'",
        .es = "Variable indefinida: '{s}'",
        .zh = "未定义的变量：'{s}'",
        .ja = "未定義の変数：'{s}'",
        .ru = "Неопределённая переменная: '{s}'",
    },
    .{
        .id = .E003_UNDEFINED_FUNCTION,
        .en = "Undefined function: '{s}'",
        .ar = "دالة غير معرّفة: '{s}'",
        .fr = "Fonction non définie: '{s}'",
        .de = "Undefinierte Funktion: '{s}'",
        .es = "Función indefinida: '{s}'",
        .zh = "未定义的函数：'{s}'",
        .ja = "未定義の関数：'{s}'",
        .ru = "Неопределённая функция: '{s}'",
    },
    .{
        .id = .E004_ARGUMENT_COUNT,
        .en = "Wrong number of arguments: expected {d}, got {d}",
        .ar = "عدد خاطئ من المعاملات: متوقع {d}، حصلت على {d}",
        .fr = "Nombre d'arguments incorrect: attendu {d}, obtenu {d}",
        .de = "Falsche Anzahl von Argumenten: erwartet {d}, erhalten {d}",
        .es = "Número incorrecto de argumentos: esperado {d}, obtenido {d}",
        .zh = "参数数量错误：期望 {d}，得到 {d}",
        .ja = "引数の数が間違っています：期待 {d}、取得 {d}",
        .ru = "Неверное число аргументов: ожидалось {d}, получено {d}",
    },
    .{
        .id = .E005_RETURN_TYPE,
        .en = "Return type mismatch: expected '{s}', got '{s}'",
        .ar = "عدم تطابق نوع الإرجاع: متوقع '{s}'، حصلت على '{s}'",
        .fr = "Type de retour incompatible: attendu '{s}', obtenu '{s}'",
        .de = "Rückgabetyp stimmt nicht überein: erwartet '{s}', erhalten '{s}'",
        .es = "Tipo de retorno no coincide: esperado '{s}', obtenido '{s}'",
        .zh = "返回类型不匹配：期望 '{s}'，得到 '{s}'",
        .ja = "戻り値の型が一致しません：期待 '{s}'、取得 '{s}'",
        .ru = "Несоответствие типа возврата: ожидалось '{s}', получено '{s}'",
    },
    .{
        .id = .E006_BORROW_ERROR,
        .en = "Cannot borrow '{s}' as mutable, already borrowed as immutable",
        .ar = "لا يمكن استعارة '{s}' كقابل للتعديل، مستعار بالفعل كغير قابل للتعديل",
        .fr = "Impossible d'emprunter '{s}' comme mutable, déjà emprunté comme immuable",
        .de = "Kann '{s}' nicht als veränderlich ausleihen, bereits als unveränderlich ausgeliehen",
        .es = "No se puede tomar prestado '{s}' como mutable, ya prestado como inmutable",
        .zh = "无法将 '{s}' 借用为可变，已借用为不可变",
        .ja = "'{s}' を可変として借用できません、すでに不変として借用されています",
        .ru = "Невозможно заимствовать '{s}' как изменяемое, уже заимствовано как неизменяемое",
    },
    .{
        .id = .E007_LIFETIME_ERROR,
        .en = "Lifetime '{s}' does not live long enough",
        .ar = "مدة الحياة '{s}' غير كافية",
        .fr = "La durée de vie '{s}' n'est pas assez longue",
        .de = "Lebenszeit '{s}' ist nicht lang genug",
        .es = "El tiempo de vida '{s}' no es suficientemente largo",
        .zh = "生命周期 '{s}' 不够长",
        .ja = "ライフタイム '{s}' は十分に長くありません",
        .ru = "Время жизни '{s}' недостаточно",
    },
    .{
        .id = .E008_SYNTAX_ERROR,
        .en = "Syntax error: {s}",
        .ar = "خطأ نحوي: {s}",
        .fr = "Erreur de syntaxe: {s}",
        .de = "Syntaxfehler: {s}",
        .es = "Error de sintaxis: {s}",
        .zh = "语法错误：{s}",
        .ja = "構文エラー：{s}",
        .ru = "Синтаксическая ошибка: {s}",
    },
    .{
        .id = .E009_IMPORT_ERROR,
        .en = "Cannot import module: '{s}'",
        .ar = "لا يمكن استيراد الوحدة: '{s}'",
        .fr = "Impossible d'importer le module: '{s}'",
        .de = "Modul kann nicht importiert werden: '{s}'",
        .es = "No se puede importar el módulo: '{s}'",
        .zh = "无法导入模块：'{s}'",
        .ja = "モジュールをインポートできません：'{s}'",
        .ru = "Невозможно импортировать модуль: '{s}'",
    },
    .{
        .id = .E010_ATTRIBUTE_ERROR,
        .en = "Unknown attribute: '{s}'",
        .ar = "سمة غير معروفة: '{s}'",
        .fr = "Attribut inconnu: '{s}'",
        .de = "Unbekanntes Attribut: '{s}'",
        .es = "Atributo desconocido: '{s}'",
        .zh = "未知属性：'{s}'",
        .ja = "不明な属性：'{s}'",
        .ru = "Неизвестный атрибут: '{s}'",
    },

    // Runtime Errors
    .{
        .id = .R001_NULL_POINTER,
        .en = "Null pointer dereference",
        .ar = "إلغاء مرجع مؤشر فارغ",
        .fr = "Déréférencement de pointeur nul",
        .de = "Null-Zeiger-Dereferenzierung",
        .es = "Desreferencia de puntero nulo",
        .zh = "空指针解引用",
        .ja = "ヌルポインタの参照解除",
        .ru = "Разыменование нулевого указателя",
    },
    .{
        .id = .R002_INDEX_OUT_OF_BOUNDS,
        .en = "Index {d} out of bounds for length {d}",
        .ar = "الفهرس {d} خارج النطاق للطول {d}",
        .fr = "Index {d} hors limites pour la longueur {d}",
        .de = "Index {d} außerhalb der Grenzen für Länge {d}",
        .es = "Índice {d} fuera de límites para longitud {d}",
        .zh = "索引 {d} 超出长度 {d} 的范围",
        .ja = "インデックス {d} が長さ {d} の範囲外です",
        .ru = "Индекс {d} выходит за пределы длины {d}",
    },
    .{
        .id = .R003_DIVISION_BY_ZERO,
        .en = "Division by zero",
        .ar = "القسمة على صفر",
        .fr = "Division par zéro",
        .de = "Division durch Null",
        .es = "División por cero",
        .zh = "除以零",
        .ja = "ゼロによる除算",
        .ru = "Деление на ноль",
    },
    .{
        .id = .R004_STACK_OVERFLOW,
        .en = "Stack overflow",
        .ar = "تجاوز سعة المكدس",
        .fr = "Débordement de pile",
        .de = "Stapelüberlauf",
        .es = "Desbordamiento de pila",
        .zh = "堆栈溢出",
        .ja = "スタックオーバーフロー",
        .ru = "Переполнение стека",
    },
    .{
        .id = .R005_OUT_OF_MEMORY,
        .en = "Out of memory",
        .ar = "نفاد الذاكرة",
        .fr = "Mémoire insuffisante",
        .de = "Speicher erschöpft",
        .es = "Sin memoria",
        .zh = "内存不足",
        .ja = "メモリ不足",
        .ru = "Недостаточно памяти",
    },

    // CLI Messages
    .{
        .id = .C001_USAGE,
        .en = "Usage: mojo <command> [options] [files]",
        .ar = "الاستخدام: mojo <أمر> [خيارات] [ملفات]",
        .fr = "Utilisation: mojo <commande> [options] [fichiers]",
        .de = "Verwendung: mojo <Befehl> [Optionen] [Dateien]",
        .es = "Uso: mojo <comando> [opciones] [archivos]",
        .zh = "用法：mojo <命令> [选项] [文件]",
        .ja = "使用法：mojo <コマンド> [オプション] [ファイル]",
        .ru = "Использование: mojo <команда> [параметры] [файлы]",
    },
    .{
        .id = .C002_VERSION,
        .en = "Mojo SDK version {s}",
        .ar = "إصدار Mojo SDK {s}",
        .fr = "Version du SDK Mojo {s}",
        .de = "Mojo SDK Version {s}",
        .es = "Versión del SDK Mojo {s}",
        .zh = "Mojo SDK 版本 {s}",
        .ja = "Mojo SDK バージョン {s}",
        .ru = "Mojo SDK версия {s}",
    },
    .{
        .id = .C003_HELP,
        .en = "Use 'mojo help <command>' for more information",
        .ar = "استخدم 'mojo help <أمر>' لمزيد من المعلومات",
        .fr = "Utilisez 'mojo help <commande>' pour plus d'informations",
        .de = "Verwenden Sie 'mojo help <Befehl>' für weitere Informationen",
        .es = "Use 'mojo help <comando>' para más información",
        .zh = "使用 'mojo help <命令>' 获取更多信息",
        .ja = "詳細は 'mojo help <コマンド>' を使用してください",
        .ru = "Используйте 'mojo help <команда>' для дополнительной информации",
    },
    .{
        .id = .C004_BUILD_SUCCESS,
        .en = "Build succeeded in {d}ms",
        .ar = "نجح البناء في {d} مللي ثانية",
        .fr = "Compilation réussie en {d}ms",
        .de = "Build erfolgreich in {d}ms",
        .es = "Compilación exitosa en {d}ms",
        .zh = "构建成功，耗时 {d}ms",
        .ja = "ビルド成功：{d}ms",
        .ru = "Сборка успешна за {d}мс",
    },
    .{
        .id = .C005_BUILD_FAILED,
        .en = "Build failed with {d} error(s)",
        .ar = "فشل البناء مع {d} خطأ(أخطاء)",
        .fr = "Échec de la compilation avec {d} erreur(s)",
        .de = "Build fehlgeschlagen mit {d} Fehler(n)",
        .es = "Compilación fallida con {d} error(es)",
        .zh = "构建失败，{d} 个错误",
        .ja = "ビルド失敗：{d} 個のエラー",
        .ru = "Сборка не удалась: {d} ошибок",
    },
    .{
        .id = .C006_TEST_PASSED,
        .en = "{d} test(s) passed",
        .ar = "نجح {d} اختبار(اختبارات)",
        .fr = "{d} test(s) réussi(s)",
        .de = "{d} Test(s) bestanden",
        .es = "{d} prueba(s) pasada(s)",
        .zh = "{d} 个测试通过",
        .ja = "{d} 個のテスト成功",
        .ru = "{d} тестов пройдено",
    },
    .{
        .id = .C007_TEST_FAILED,
        .en = "{d} test(s) failed",
        .ar = "فشل {d} اختبار(اختبارات)",
        .fr = "{d} test(s) échoué(s)",
        .de = "{d} Test(s) fehlgeschlagen",
        .es = "{d} prueba(s) fallida(s)",
        .zh = "{d} 个测试失败",
        .ja = "{d} 個のテスト失敗",
        .ru = "{d} тестов не пройдено",
    },
    .{
        .id = .C008_FILE_NOT_FOUND,
        .en = "File not found: '{s}'",
        .ar = "الملف غير موجود: '{s}'",
        .fr = "Fichier non trouvé: '{s}'",
        .de = "Datei nicht gefunden: '{s}'",
        .es = "Archivo no encontrado: '{s}'",
        .zh = "文件未找到：'{s}'",
        .ja = "ファイルが見つかりません：'{s}'",
        .ru = "Файл не найден: '{s}'",
    },

    // LSP Messages
    .{
        .id = .L001_HOVER_TYPE,
        .en = "Type: {s}",
        .ar = "النوع: {s}",
        .fr = "Type: {s}",
        .de = "Typ: {s}",
        .es = "Tipo: {s}",
        .zh = "类型：{s}",
        .ja = "型：{s}",
        .ru = "Тип: {s}",
    },
    .{
        .id = .L002_COMPLETION_HINT,
        .en = "Press Tab to complete",
        .ar = "اضغط Tab للإكمال",
        .fr = "Appuyez sur Tab pour compléter",
        .de = "Tab drücken zum Vervollständigen",
        .es = "Presione Tab para completar",
        .zh = "按 Tab 键完成",
        .ja = "Tab キーで補完",
        .ru = "Нажмите Tab для завершения",
    },
    .{
        .id = .L003_DIAGNOSTIC_ERROR,
        .en = "Error",
        .ar = "خطأ",
        .fr = "Erreur",
        .de = "Fehler",
        .es = "Error",
        .zh = "错误",
        .ja = "エラー",
        .ru = "Ошибка",
    },
    .{
        .id = .L004_DIAGNOSTIC_WARNING,
        .en = "Warning",
        .ar = "تحذير",
        .fr = "Avertissement",
        .de = "Warnung",
        .es = "Advertencia",
        .zh = "警告",
        .ja = "警告",
        .ru = "Предупреждение",
    },
};

// ============================================================================
// I18n Context
// ============================================================================

pub const I18nContext = struct {
    language: Language,
    fallback: Language,

    const Self = @This();

    pub fn init() Self {
        return Self{
            .language = LANG_ENGLISH,
            .fallback = LANG_ENGLISH,
        };
    }

    pub fn initWithLanguage(lang: Language) Self {
        return Self{
            .language = lang,
            .fallback = LANG_ENGLISH,
        };
    }

    pub fn setLanguage(self: *Self, lang: Language) void {
        self.language = lang;
    }

    pub fn getMessage(self: *const Self, id: MessageId) []const u8 {
        // Find message in catalog
        for (MESSAGE_CATALOG) |entry| {
            if (entry.id == id) {
                return self.getLocalizedText(entry);
            }
        }
        return id.toCode();
    }

    fn getLocalizedText(self: *const Self, entry: MessageEntry) []const u8 {
        // Try current language
        const text = self.getTextForLanguage(entry, self.language.code);
        if (text.len > 0) return text;

        // Try fallback
        const fallback_text = self.getTextForLanguage(entry, self.fallback.code);
        if (fallback_text.len > 0) return fallback_text;

        // Return English as last resort
        return entry.en;
    }

    fn getTextForLanguage(self: *const Self, entry: MessageEntry, code: []const u8) []const u8 {
        _ = self;
        if (std.mem.eql(u8, code, "en")) return entry.en;
        if (std.mem.eql(u8, code, "ar")) return entry.ar;
        if (std.mem.eql(u8, code, "fr")) return entry.fr;
        if (std.mem.eql(u8, code, "de")) return entry.de;
        if (std.mem.eql(u8, code, "es")) return entry.es;
        if (std.mem.eql(u8, code, "zh")) return entry.zh;
        if (std.mem.eql(u8, code, "ja")) return entry.ja;
        if (std.mem.eql(u8, code, "ru")) return entry.ru;
        return "";
    }

    pub fn isRtl(self: *const Self) bool {
        return self.language.rtl;
    }
};

// ============================================================================
// Global Instance
// ============================================================================

var global_context: I18nContext = I18nContext.init();

pub fn getContext() *I18nContext {
    return &global_context;
}

pub fn setLanguage(lang: Language) void {
    global_context.setLanguage(lang);
}

pub fn setLanguageByCode(code: []const u8) void {
    const lang = getLanguageByCode(code);
    global_context.setLanguage(lang);
}

pub fn getMessage(id: MessageId) []const u8 {
    return global_context.getMessage(id);
}

pub fn isRtl() bool {
    return global_context.isRtl();
}

pub fn getCurrentLanguage() Language {
    return global_context.language;
}

pub fn isCurrentLanguageRtl() bool {
    return global_context.language.rtl;
}

pub fn getLanguageByCode(code: []const u8) Language {
    if (std.mem.eql(u8, code, "ar")) return LANG_ARABIC;
    if (std.mem.eql(u8, code, "fr")) return LANG_FRENCH;
    if (std.mem.eql(u8, code, "de")) return LANG_GERMAN;
    if (std.mem.eql(u8, code, "es")) return LANG_SPANISH;
    if (std.mem.eql(u8, code, "zh")) return LANG_CHINESE;
    if (std.mem.eql(u8, code, "ja")) return LANG_JAPANESE;
    if (std.mem.eql(u8, code, "ru")) return LANG_RUSSIAN;
    if (std.mem.eql(u8, code, "he")) return LANG_HEBREW;
    if (std.mem.eql(u8, code, "fa")) return LANG_PERSIAN;
    return LANG_ENGLISH;
}

// ============================================================================
// Formatting Helpers
// ============================================================================

pub fn formatMessage(comptime id: MessageId, args: anytype) []const u8 {
    const template = getMessage(id);
    _ = args; // TODO: Implement proper formatting
    return template;
}

// ============================================================================
// Tests
// ============================================================================

test "i18n basic" {
    const ctx = I18nContext.init();
    const msg = ctx.getMessage(.E001_TYPE_MISMATCH);
    try std.testing.expect(msg.len > 0);
}

test "i18n arabic" {
    var ctx = I18nContext.initWithLanguage(LANG_ARABIC);
    const msg = ctx.getMessage(.E001_TYPE_MISMATCH);
    try std.testing.expect(std.mem.indexOf(u8, msg, "عدم") != null);
}

test "i18n rtl detection" {
    const en_ctx = I18nContext.initWithLanguage(LANG_ENGLISH);
    try std.testing.expect(!en_ctx.isRtl());

    const ar_ctx = I18nContext.initWithLanguage(LANG_ARABIC);
    try std.testing.expect(ar_ctx.isRtl());
}
