// CLI i18n Support
// Localized messages for Mojo CLI tool

const std = @import("std");
const i18n = @import("i18n");
const locale_detect = @import("locale_detect");

// ============================================================================
// CLI Message IDs
// ============================================================================

pub const CliMessageId = enum {
    // Usage messages
    usage_header,
    usage_commands,
    usage_footer,

    // Command names
    cmd_run,
    cmd_build,
    cmd_test,
    cmd_format,
    cmd_doc,
    cmd_repl,
    cmd_version,
    cmd_help,

    // Command descriptions
    desc_run,
    desc_build,
    desc_test,
    desc_format,
    desc_doc,
    desc_repl,
    desc_version,
    desc_help,

    // Options
    opt_output,
    opt_optimize,
    opt_release,
    opt_verbose,
    opt_filter,
    opt_write,
    opt_check,
    opt_recursive,

    // Status messages
    status_compiling,
    status_running,
    status_testing,
    status_formatting,
    status_generating_docs,
    status_complete,

    // Error messages
    err_unknown_command,
    err_file_not_found,
    err_compile_failed,
    err_no_input,
};

// ============================================================================
// CLI Messages
// ============================================================================

const CliMessage = struct {
    id: CliMessageId,
    en: []const u8,
    ar: []const u8,
    zh: []const u8,
    ja: []const u8,
    ru: []const u8,
    fr: []const u8,
    de: []const u8,
    es: []const u8,
};

const CLI_MESSAGES = [_]CliMessage{
    // Usage messages
    .{
        .id = .usage_header,
        .en = "Usage: mojo <command> [options]",
        .ar = "الاستخدام: mojo <أمر> [خيارات]",
        .zh = "用法: mojo <命令> [选项]",
        .ja = "使用法: mojo <コマンド> [オプション]",
        .ru = "Использование: mojo <команда> [опции]",
        .fr = "Usage: mojo <commande> [options]",
        .de = "Verwendung: mojo <Befehl> [Optionen]",
        .es = "Uso: mojo <comando> [opciones]",
    },
    .{
        .id = .usage_commands,
        .en = "Commands:",
        .ar = "الأوامر:",
        .zh = "命令:",
        .ja = "コマンド:",
        .ru = "Команды:",
        .fr = "Commandes:",
        .de = "Befehle:",
        .es = "Comandos:",
    },
    .{
        .id = .usage_footer,
        .en = "Use 'mojo <command> --help' for more information on a command.",
        .ar = "استخدم 'mojo <أمر> --help' لمزيد من المعلومات عن أمر ما.",
        .zh = "使用 'mojo <命令> --help' 获取命令的更多信息。",
        .ja = "コマンドの詳細は 'mojo <コマンド> --help' を使用してください。",
        .ru = "Используйте 'mojo <команда> --help' для получения информации о команде.",
        .fr = "Utilisez 'mojo <commande> --help' pour plus d'informations sur une commande.",
        .de = "Verwenden Sie 'mojo <Befehl> --help' für weitere Informationen zu einem Befehl.",
        .es = "Use 'mojo <comando> --help' para más información sobre un comando.",
    },

    // Command descriptions
    .{
        .id = .desc_run,
        .en = "JIT compile and execute a Mojo file",
        .ar = "ترجمة وتنفيذ ملف Mojo",
        .zh = "JIT编译并执行Mojo文件",
        .ja = "MojoファイルをJITコンパイルして実行",
        .ru = "JIT-компиляция и выполнение файла Mojo",
        .fr = "Compiler JIT et exécuter un fichier Mojo",
        .de = "JIT-Kompilierung und Ausführung einer Mojo-Datei",
        .es = "Compilar JIT y ejecutar un archivo Mojo",
    },
    .{
        .id = .desc_build,
        .en = "AOT compile to native binary",
        .ar = "ترجمة إلى ملف تنفيذي",
        .zh = "AOT编译为原生二进制文件",
        .ja = "ネイティブバイナリにAOTコンパイル",
        .ru = "AOT-компиляция в нативный бинарник",
        .fr = "Compiler AOT en binaire natif",
        .de = "AOT-Kompilierung zu nativer Binärdatei",
        .es = "Compilar AOT a binario nativo",
    },
    .{
        .id = .desc_test,
        .en = "Run test suite",
        .ar = "تشغيل مجموعة الاختبارات",
        .zh = "运行测试套件",
        .ja = "テストスイートを実行",
        .ru = "Запуск набора тестов",
        .fr = "Exécuter la suite de tests",
        .de = "Testsuite ausführen",
        .es = "Ejecutar suite de pruebas",
    },
    .{
        .id = .desc_format,
        .en = "Format Mojo source files",
        .ar = "تنسيق ملفات Mojo المصدرية",
        .zh = "格式化Mojo源文件",
        .ja = "Mojoソースファイルをフォーマット",
        .ru = "Форматирование исходных файлов Mojo",
        .fr = "Formater les fichiers source Mojo",
        .de = "Mojo-Quelldateien formatieren",
        .es = "Formatear archivos fuente Mojo",
    },
    .{
        .id = .desc_doc,
        .en = "Generate documentation",
        .ar = "إنشاء التوثيق",
        .zh = "生成文档",
        .ja = "ドキュメントを生成",
        .ru = "Генерация документации",
        .fr = "Générer la documentation",
        .de = "Dokumentation generieren",
        .es = "Generar documentación",
    },
    .{
        .id = .desc_repl,
        .en = "Start interactive REPL",
        .ar = "بدء الوضع التفاعلي",
        .zh = "启动交互式REPL",
        .ja = "対話型REPLを開始",
        .ru = "Запуск интерактивного REPL",
        .fr = "Démarrer le REPL interactif",
        .de = "Interaktive REPL starten",
        .es = "Iniciar REPL interactivo",
    },
    .{
        .id = .desc_version,
        .en = "Show version information",
        .ar = "إظهار معلومات الإصدار",
        .zh = "显示版本信息",
        .ja = "バージョン情報を表示",
        .ru = "Показать информацию о версии",
        .fr = "Afficher les informations de version",
        .de = "Versionsinformationen anzeigen",
        .es = "Mostrar información de versión",
    },
    .{
        .id = .desc_help,
        .en = "Show this help message",
        .ar = "إظهار رسالة المساعدة هذه",
        .zh = "显示此帮助信息",
        .ja = "このヘルプメッセージを表示",
        .ru = "Показать это сообщение справки",
        .fr = "Afficher ce message d'aide",
        .de = "Diese Hilfemeldung anzeigen",
        .es = "Mostrar este mensaje de ayuda",
    },

    // Status messages
    .{
        .id = .status_compiling,
        .en = "Compiling",
        .ar = "جاري الترجمة",
        .zh = "正在编译",
        .ja = "コンパイル中",
        .ru = "Компиляция",
        .fr = "Compilation",
        .de = "Kompilierung",
        .es = "Compilando",
    },
    .{
        .id = .status_running,
        .en = "Running",
        .ar = "جاري التشغيل",
        .zh = "正在运行",
        .ja = "実行中",
        .ru = "Выполнение",
        .fr = "Exécution",
        .de = "Ausführung",
        .es = "Ejecutando",
    },
    .{
        .id = .status_testing,
        .en = "Testing",
        .ar = "جاري الاختبار",
        .zh = "正在测试",
        .ja = "テスト中",
        .ru = "Тестирование",
        .fr = "Test",
        .de = "Testen",
        .es = "Probando",
    },
    .{
        .id = .status_formatting,
        .en = "Formatting",
        .ar = "جاري التنسيق",
        .zh = "正在格式化",
        .ja = "フォーマット中",
        .ru = "Форматирование",
        .fr = "Formatage",
        .de = "Formatierung",
        .es = "Formateando",
    },
    .{
        .id = .status_generating_docs,
        .en = "Generating documentation",
        .ar = "جاري إنشاء التوثيق",
        .zh = "正在生成文档",
        .ja = "ドキュメント生成中",
        .ru = "Генерация документации",
        .fr = "Génération de la documentation",
        .de = "Dokumentation wird generiert",
        .es = "Generando documentación",
    },
    .{
        .id = .status_complete,
        .en = "Complete",
        .ar = "اكتمل",
        .zh = "完成",
        .ja = "完了",
        .ru = "Завершено",
        .fr = "Terminé",
        .de = "Abgeschlossen",
        .es = "Completado",
    },

    // Error messages
    .{
        .id = .err_unknown_command,
        .en = "Unknown command",
        .ar = "أمر غير معروف",
        .zh = "未知命令",
        .ja = "不明なコマンド",
        .ru = "Неизвестная команда",
        .fr = "Commande inconnue",
        .de = "Unbekannter Befehl",
        .es = "Comando desconocido",
    },
    .{
        .id = .err_file_not_found,
        .en = "File not found",
        .ar = "الملف غير موجود",
        .zh = "文件未找到",
        .ja = "ファイルが見つかりません",
        .ru = "Файл не найден",
        .fr = "Fichier non trouvé",
        .de = "Datei nicht gefunden",
        .es = "Archivo no encontrado",
    },
    .{
        .id = .err_compile_failed,
        .en = "Compilation failed",
        .ar = "فشلت الترجمة",
        .zh = "编译失败",
        .ja = "コンパイル失敗",
        .ru = "Компиляция не удалась",
        .fr = "Échec de la compilation",
        .de = "Kompilierung fehlgeschlagen",
        .es = "Compilación fallida",
    },
    .{
        .id = .err_no_input,
        .en = "No input file specified",
        .ar = "لم يتم تحديد ملف إدخال",
        .zh = "未指定输入文件",
        .ja = "入力ファイルが指定されていません",
        .ru = "Входной файл не указан",
        .fr = "Aucun fichier d'entrée spécifié",
        .de = "Keine Eingabedatei angegeben",
        .es = "No se especificó archivo de entrada",
    },
};

// ============================================================================
// Message Lookup
// ============================================================================

pub fn getCliMessage(id: CliMessageId) []const u8 {
    const lang = i18n.getCurrentLanguage();

    for (CLI_MESSAGES) |msg| {
        if (msg.id == id) {
            if (std.mem.eql(u8, lang.code, "ar")) return msg.ar;
            if (std.mem.eql(u8, lang.code, "zh")) return msg.zh;
            if (std.mem.eql(u8, lang.code, "ja")) return msg.ja;
            if (std.mem.eql(u8, lang.code, "ru")) return msg.ru;
            if (std.mem.eql(u8, lang.code, "fr")) return msg.fr;
            if (std.mem.eql(u8, lang.code, "de")) return msg.de;
            if (std.mem.eql(u8, lang.code, "es")) return msg.es;
            return msg.en;
        }
    }
    return "";
}

// ============================================================================
// Initialization
// ============================================================================

pub fn initCli(allocator: std.mem.Allocator) void {
    locale_detect.autoInit(allocator);
}

// ============================================================================
// RTL Support
// ============================================================================

pub fn isRtl() bool {
    return i18n.isCurrentLanguageRtl();
}

pub fn getTextAlign() []const u8 {
    return if (isRtl()) "right" else "left";
}

// ============================================================================
// Formatted Output
// ============================================================================

pub fn printLocalizedUsage() void {
    const is_rtl = isRtl();

    if (is_rtl) {
        std.debug.print("\u{200F}", .{}); // RTL mark
    }

    std.debug.print("{s}\n\n", .{getCliMessage(.usage_header)});
    std.debug.print("{s}\n", .{getCliMessage(.usage_commands)});
    std.debug.print("  run      {s}\n", .{getCliMessage(.desc_run)});
    std.debug.print("  build    {s}\n", .{getCliMessage(.desc_build)});
    std.debug.print("  test     {s}\n", .{getCliMessage(.desc_test)});
    std.debug.print("  format   {s}\n", .{getCliMessage(.desc_format)});
    std.debug.print("  doc      {s}\n", .{getCliMessage(.desc_doc)});
    std.debug.print("  repl     {s}\n", .{getCliMessage(.desc_repl)});
    std.debug.print("  version  {s}\n", .{getCliMessage(.desc_version)});
    std.debug.print("  help     {s}\n", .{getCliMessage(.desc_help)});
    std.debug.print("\n{s}\n", .{getCliMessage(.usage_footer)});
}

pub fn printError(id: CliMessageId, detail: []const u8) void {
    const is_rtl = isRtl();

    if (is_rtl) {
        std.debug.print("\u{200F}", .{}); // RTL mark
    }

    if (detail.len > 0) {
        std.debug.print("{s}: {s}\n", .{ getCliMessage(id), detail });
    } else {
        std.debug.print("{s}\n", .{getCliMessage(id)});
    }
}

pub fn printStatus(id: CliMessageId, detail: []const u8) void {
    const is_rtl = isRtl();

    if (is_rtl) {
        std.debug.print("\u{200F}", .{}); // RTL mark
    }

    if (detail.len > 0) {
        std.debug.print("{s}: {s}\n", .{ getCliMessage(id), detail });
    } else {
        std.debug.print("{s}...\n", .{getCliMessage(id)});
    }
}

// ============================================================================
// Tests
// ============================================================================

test "cli_i18n: message lookup" {
    const msg = getCliMessage(.desc_run);
    try std.testing.expect(msg.len > 0);
}

test "cli_i18n: all messages exist" {
    const ids = [_]CliMessageId{
        .usage_header,
        .usage_commands,
        .desc_run,
        .desc_build,
        .status_compiling,
        .err_unknown_command,
    };

    for (ids) |id| {
        const msg = getCliMessage(id);
        try std.testing.expect(msg.len > 0);
    }
}
