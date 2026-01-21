const std = @import("std");
const scip_writer = @import("zig_scip_writer");

// ============================================================================
// Tree-sitter based SCIP indexer for data languages
// Supports: JSON, XML, YAML, TOML, SQL, GraphQL, Protobuf, Markdown, HTML, CSS
// ============================================================================

const Language = enum {
    json,
    xml,
    yaml,
    toml,
    sql,
    graphql,
    protobuf,
    thrift,
    markdown,
    html,
    css,
    scss,
    less,
    unknown,

    pub fn fromString(s: []const u8) Language {
        const lang_map = std.StaticStringMap(Language).initComptime(.{
            .{ "json", .json },
            .{ "xml", .xml },
            .{ "yaml", .yaml },
            .{ "yml", .yaml },
            .{ "toml", .toml },
            .{ "sql", .sql },
            .{ "graphql", .graphql },
            .{ "gql", .graphql },
            .{ "protobuf", .protobuf },
            .{ "proto", .protobuf },
            .{ "thrift", .thrift },
            .{ "markdown", .markdown },
            .{ "md", .markdown },
            .{ "html", .html },
            .{ "htm", .html },
            .{ "css", .css },
            .{ "scss", .scss },
            .{ "less", .less },
        });
        return lang_map.get(s) orelse .unknown;
    }

    pub fn fileExtensions(self: Language) []const []const u8 {
        return switch (self) {
            .json => &.{".json"},
            .xml => &.{ ".xml", ".xsd", ".xsl", ".svg" },
            .yaml => &.{ ".yaml", ".yml" },
            .toml => &.{".toml"},
            .sql => &.{ ".sql", ".ddl" },
            .graphql => &.{ ".graphql", ".gql" },
            .protobuf => &.{".proto"},
            .thrift => &.{".thrift"},
            .markdown => &.{ ".md", ".markdown" },
            .html => &.{ ".html", ".htm" },
            .css => &.{".css"},
            .scss => &.{".scss"},
            .less => &.{".less"},
            .unknown => &.{},
        };
    }
};

// Symbol extraction patterns for each language
const SymbolPattern = struct {
    start_marker: []const u8,
    end_marker: []const u8,
    symbol_kind: u32,
};

fn getPatterns(lang: Language) []const SymbolPattern {
    return switch (lang) {
        .json => &.{
            .{ .start_marker = "\"", .end_marker = "\":", .symbol_kind = 8 }, // Property
        },
        .xml => &.{
            .{ .start_marker = "<", .end_marker = " ", .symbol_kind = 5 }, // Class (element)
            .{ .start_marker = "<", .end_marker = ">", .symbol_kind = 5 },
        },
        .yaml => &.{
            .{ .start_marker = "", .end_marker = ":", .symbol_kind = 8 }, // Property (key)
        },
        .sql => &.{
            .{ .start_marker = "CREATE TABLE ", .end_marker = " ", .symbol_kind = 5 }, // Class
            .{ .start_marker = "CREATE TABLE ", .end_marker = "(", .symbol_kind = 5 },
            .{ .start_marker = "ALTER TABLE ", .end_marker = " ", .symbol_kind = 5 },
        },
        else => &.{},
    };
}

// ============================================================================
// File indexing
// ============================================================================

fn indexFileContent(
    allocator: std.mem.Allocator,
    lang: Language,
    content: []const u8,
    file_path: []const u8,
) !void {
    _ = allocator;

    // Begin document with language name
    const lang_name: [:0]const u8 = switch (lang) {
        .json => "json",
        .xml => "xml",
        .yaml => "yaml",
        .toml => "toml",
        .sql => "sql",
        .graphql => "graphql",
        .protobuf => "protobuf",
        .thrift => "thrift",
        .markdown => "markdown",
        .html => "html",
        .css => "css",
        .scss => "scss",
        .less => "less",
        .unknown => "unknown",
    };

    // Need null-terminated path
    var path_buf: [4096]u8 = undefined;
    const path_z = std.fmt.bufPrintZ(&path_buf, "{s}", .{file_path}) catch return;

    _ = scip_writer.scip_begin_document(lang_name.ptr, path_z.ptr);

    // Simple line-based symbol extraction
    var line_num: i32 = 0;
    var lines = std.mem.splitSequence(u8, content, "\n");
    
    while (lines.next()) |line| {
        // Extract symbols based on language patterns
        switch (lang) {
            .json => extractJsonSymbols(line, line_num),
            .yaml => extractYamlSymbols(line, line_num),
            .sql => extractSqlSymbols(line, line_num),
            .xml, .html => extractXmlSymbols(line, line_num),
            .toml => extractTomlSymbols(line, line_num),
            .css, .scss, .less => extractCssSymbols(line, line_num),
            .graphql => extractGraphqlSymbols(line, line_num),
            else => {},
        }
        line_num += 1;
    }

    // Write document
    _ = scip_writer.scip_write_document(lang_name.ptr, path_z.ptr);
}

fn extractJsonSymbols(line: []const u8, line_num: i32) void {
    // Find JSON keys: "key":
    var i: usize = 0;
    while (i < line.len) {
        if (line[i] == '"') {
            const start = i + 1;
            i += 1;
            while (i < line.len and line[i] != '"') : (i += 1) {}
            if (i < line.len) {
                const end = i;
                i += 1;
                // Check for colon after quote
                while (i < line.len and (line[i] == ' ' or line[i] == '\t')) : (i += 1) {}
                if (i < line.len and line[i] == ':') {
                    const key = line[start..end];
                    // Create null-terminated key
                    var key_buf: [256]u8 = undefined;
                    const key_z = std.fmt.bufPrintZ(&key_buf, "{s}", .{key}) catch continue;
                    _ = scip_writer.scip_add_occurrence(
                        line_num, @intCast(start), line_num, @intCast(end),
                        key_z.ptr, 1 // Definition
                    );
                    _ = scip_writer.scip_add_symbol_info(key_z.ptr, "JSON property", 8);
                }
            }
        }
        i += 1;
    }
}

fn extractYamlSymbols(line: []const u8, line_num: i32) void {
    // Find YAML keys: key:
    const trimmed = std.mem.trimLeft(u8, line, " \t");
    if (trimmed.len == 0 or trimmed[0] == '#') return;

    if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
        if (colon_pos > 0) {
            const key = trimmed[0..colon_pos];
            const start = line.len - trimmed.len;
            var key_buf: [256]u8 = undefined;
            const key_z = std.fmt.bufPrintZ(&key_buf, "{s}", .{key}) catch return;
            _ = scip_writer.scip_add_occurrence(
                line_num, @intCast(start), line_num, @intCast(start + colon_pos),
                key_z.ptr, 1
            );
            _ = scip_writer.scip_add_symbol_info(key_z.ptr, "YAML key", 8);
        }
    }
}

fn extractSqlSymbols(line: []const u8, line_num: i32) void {
    // Table operations
    const table_keywords = [_][]const u8{ "CREATE TABLE", "ALTER TABLE", "DROP TABLE", "INSERT INTO", "UPDATE", "DELETE FROM", "TRUNCATE TABLE" };
    for (table_keywords) |kw| {
        if (std.ascii.indexOfIgnoreCase(line, kw)) |pos| {
            extractSqlIdentifier(line, line_num, pos + kw.len, "SQL table", 5);
        }
    }
    // Views
    if (std.ascii.indexOfIgnoreCase(line, "CREATE VIEW")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 11, "SQL view", 5);
    }
    // Indexes
    if (std.ascii.indexOfIgnoreCase(line, "CREATE INDEX")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 12, "SQL index", 14);
    }
    if (std.ascii.indexOfIgnoreCase(line, "CREATE UNIQUE INDEX")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 19, "SQL unique index", 14);
    }
    // Procedures/Functions
    if (std.ascii.indexOfIgnoreCase(line, "CREATE PROCEDURE")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 16, "SQL procedure", 12);
    }
    if (std.ascii.indexOfIgnoreCase(line, "CREATE FUNCTION")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 15, "SQL function", 12);
    }
    // Triggers
    if (std.ascii.indexOfIgnoreCase(line, "CREATE TRIGGER")) |pos| {
        extractSqlIdentifier(line, line_num, pos + 14, "SQL trigger", 24);
    }
    // Column definitions (inside parentheses after CREATE TABLE)
    extractSqlColumns(line, line_num);
}

fn extractSqlIdentifier(line: []const u8, line_num: i32, after_kw: usize, kind_desc: [:0]const u8, kind: c_int) void {
    if (after_kw >= line.len) return;
    var start = after_kw;
    while (start < line.len and (line[start] == ' ' or line[start] == '\t')) : (start += 1) {}
    var end = start;
    while (end < line.len and line[end] != ' ' and line[end] != '(' and line[end] != ';' and line[end] != ',') : (end += 1) {}
    if (end > start) {
        const name = line[start..end];
        var name_buf: [256]u8 = undefined;
        const name_z = std.fmt.bufPrintZ(&name_buf, "{s}", .{name}) catch return;
        _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(end), name_z.ptr, 1);
        _ = scip_writer.scip_add_symbol_info(name_z.ptr, kind_desc, kind);
    }
}

fn extractSqlColumns(line: []const u8, line_num: i32) void {
    // Look for column definitions: column_name TYPE
    const types = [_][]const u8{ "INT", "INTEGER", "BIGINT", "SMALLINT", "VARCHAR", "CHAR", "TEXT", "DECIMAL", "NUMERIC", "FLOAT", "DOUBLE", "BOOLEAN", "DATE", "TIME", "TIMESTAMP", "BLOB", "CLOB", "UUID" };
    for (types) |typ| {
        var search_start: usize = 0;
        while (std.ascii.indexOfIgnoreCasePos(line, search_start, typ)) |pos| {
            if (pos > 0) {
                // Find the column name before the type
                var end = pos;
                while (end > 0 and (line[end - 1] == ' ' or line[end - 1] == '\t')) : (end -= 1) {}
                if (end > 0) {
                    var start = end;
                    while (start > 0 and line[start - 1] != ' ' and line[start - 1] != '\t' and line[start - 1] != '(' and line[start - 1] != ',') : (start -= 1) {}
                    if (end > start) {
                        const col_name = line[start..end];
                        // Skip SQL keywords
                        if (!isSqlKeyword(col_name)) {
                            var col_buf: [256]u8 = undefined;
                            const col_z = std.fmt.bufPrintZ(&col_buf, "{s}", .{col_name}) catch continue;
                            _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(end), col_z.ptr, 1);
                            _ = scip_writer.scip_add_symbol_info(col_z.ptr, "SQL column", 8);
                        }
                    }
                }
            }
            search_start = pos + typ.len;
        }
    }
}

fn isSqlKeyword(s: []const u8) bool {
    const keywords = [_][]const u8{ "PRIMARY", "FOREIGN", "KEY", "NOT", "NULL", "DEFAULT", "UNIQUE", "CHECK", "REFERENCES", "ON", "CASCADE", "SET", "CONSTRAINT", "ADD", "DROP", "ALTER", "IF", "EXISTS" };
    for (keywords) |kw| {
        if (std.ascii.eqlIgnoreCase(s, kw)) return true;
    }
    return false;
}

fn extractXmlSymbols(line: []const u8, line_num: i32) void {
    var i: usize = 0;
    while (i < line.len) {
        if (line[i] == '<' and i + 1 < line.len and line[i + 1] != '/' and line[i + 1] != '!' and line[i + 1] != '?') {
            const elem_start = i + 1;
            i += 1;
            while (i < line.len and line[i] != ' ' and line[i] != '>' and line[i] != '/') : (i += 1) {}
            const elem_end = i;
            if (elem_end > elem_start) {
                const element = line[elem_start..elem_end];
                var elem_buf: [256]u8 = undefined;
                const elem_z = std.fmt.bufPrintZ(&elem_buf, "{s}", .{element}) catch continue;
                _ = scip_writer.scip_add_occurrence(line_num, @intCast(elem_start), line_num, @intCast(elem_end), elem_z.ptr, 1);
                _ = scip_writer.scip_add_symbol_info(elem_z.ptr, "XML element", 5);
            }
            // Extract attributes: attr="value"
            while (i < line.len and line[i] != '>') {
                while (i < line.len and (line[i] == ' ' or line[i] == '\t')) : (i += 1) {}
                if (i < line.len and line[i] != '>' and line[i] != '/') {
                    const attr_start = i;
                    while (i < line.len and line[i] != '=' and line[i] != ' ' and line[i] != '>') : (i += 1) {}
                    if (i < line.len and line[i] == '=') {
                        const attr = line[attr_start..i];
                        var attr_buf: [256]u8 = undefined;
                        const attr_z = std.fmt.bufPrintZ(&attr_buf, "{s}", .{attr}) catch continue;
                        _ = scip_writer.scip_add_occurrence(line_num, @intCast(attr_start), line_num, @intCast(i), attr_z.ptr, 1);
                        _ = scip_writer.scip_add_symbol_info(attr_z.ptr, "XML attribute", 8);
                        // Skip the value
                        i += 1;
                        if (i < line.len and (line[i] == '"' or line[i] == '\'')) {
                            const quote = line[i];
                            i += 1;
                            while (i < line.len and line[i] != quote) : (i += 1) {}
                            if (i < line.len) i += 1;
                        }
                    }
                } else break;
            }
        }
        i += 1;
    }
}

fn extractTomlSymbols(line: []const u8, line_num: i32) void {
    const trimmed = std.mem.trimLeft(u8, line, " \t");
    if (trimmed.len == 0 or trimmed[0] == '#') return;
    // Section headers: [section] or [[array]]
    if (trimmed[0] == '[') {
        var end: usize = 1;
        if (trimmed.len > 1 and trimmed[1] == '[') end = 2;
        const is_array = end == 2;
        while (end < trimmed.len and trimmed[end] != ']') : (end += 1) {}
        if (end > (if (is_array) @as(usize, 2) else @as(usize, 1))) {
            const section = trimmed[(if (is_array) @as(usize, 2) else @as(usize, 1))..end];
            var sec_buf: [256]u8 = undefined;
            const sec_z = std.fmt.bufPrintZ(&sec_buf, "{s}", .{section}) catch return;
            const start = line.len - trimmed.len + (if (is_array) @as(usize, 2) else @as(usize, 1));
            _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + section.len), sec_z.ptr, 1);
            _ = scip_writer.scip_add_symbol_info(sec_z.ptr, if (is_array) "TOML array table" else "TOML table", 5);
        }
    } else if (std.mem.indexOf(u8, trimmed, "=")) |eq_pos| {
        // Key = value
        var key_end = eq_pos;
        while (key_end > 0 and (trimmed[key_end - 1] == ' ' or trimmed[key_end - 1] == '\t')) : (key_end -= 1) {}
        if (key_end > 0) {
            const key = trimmed[0..key_end];
            var key_buf: [256]u8 = undefined;
            const key_z = std.fmt.bufPrintZ(&key_buf, "{s}", .{key}) catch return;
            const start = line.len - trimmed.len;
            _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + key_end), key_z.ptr, 1);
            _ = scip_writer.scip_add_symbol_info(key_z.ptr, "TOML key", 8);
        }
    }
}

fn extractCssSymbols(line: []const u8, line_num: i32) void {
    const trimmed = std.mem.trimLeft(u8, line, " \t");
    if (trimmed.len == 0) return;
    // Selectors: .class, #id, element, @media, @keyframes
    if (trimmed[0] == '.' or trimmed[0] == '#' or trimmed[0] == '@') {
        var end: usize = 1;
        while (end < trimmed.len and trimmed[end] != ' ' and trimmed[end] != '{' and trimmed[end] != ',' and trimmed[end] != ':') : (end += 1) {}
        if (end > 1) {
            const selector = trimmed[0..end];
            var sel_buf: [256]u8 = undefined;
            const sel_z = std.fmt.bufPrintZ(&sel_buf, "{s}", .{selector}) catch return;
            const start = line.len - trimmed.len;
            const kind: c_int = if (trimmed[0] == '.') 5 else if (trimmed[0] == '#') 14 else 24;
            const desc: [:0]const u8 = if (trimmed[0] == '.') "CSS class" else if (trimmed[0] == '#') "CSS ID" else "CSS at-rule";
            _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + end), sel_z.ptr, 1);
            _ = scip_writer.scip_add_symbol_info(sel_z.ptr, desc, kind);
        }
    }
    // Properties: property: value;
    if (std.mem.indexOf(u8, trimmed, ":")) |colon_pos| {
        if (colon_pos > 0 and !std.mem.startsWith(u8, trimmed, "//") and !std.mem.startsWith(u8, trimmed, "/*")) {
            const prop = trimmed[0..colon_pos];
            if (!std.mem.startsWith(u8, prop, "http") and !std.mem.startsWith(u8, prop, "https")) {
                var prop_buf: [256]u8 = undefined;
                const prop_z = std.fmt.bufPrintZ(&prop_buf, "{s}", .{prop}) catch return;
                const start = line.len - trimmed.len;
                _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + colon_pos), prop_z.ptr, 1);
                _ = scip_writer.scip_add_symbol_info(prop_z.ptr, "CSS property", 8);
            }
        }
    }
}

fn extractGraphqlSymbols(line: []const u8, line_num: i32) void {
    const trimmed = std.mem.trimLeft(u8, line, " \t");
    if (trimmed.len == 0 or trimmed[0] == '#') return;
    // Type definitions
    const type_keywords = [_][]const u8{ "type ", "interface ", "enum ", "input ", "scalar ", "union " };
    for (type_keywords) |kw| {
        if (std.mem.startsWith(u8, trimmed, kw)) {
            var end = kw.len;
            while (end < trimmed.len and trimmed[end] != ' ' and trimmed[end] != '{' and trimmed[end] != '@') : (end += 1) {}
            if (end > kw.len) {
                const name = trimmed[kw.len..end];
                var name_buf: [256]u8 = undefined;
                const name_z = std.fmt.bufPrintZ(&name_buf, "{s}", .{name}) catch return;
                const start = line.len - trimmed.len + kw.len;
                const kind: c_int = if (std.mem.eql(u8, kw, "enum ")) 10 else 5;
                _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + name.len), name_z.ptr, 1);
                _ = scip_writer.scip_add_symbol_info(name_z.ptr, "GraphQL type", kind);
            }
            return;
        }
    }
    // Query/Mutation/Subscription
    const op_keywords = [_][]const u8{ "query ", "mutation ", "subscription ", "fragment " };
    for (op_keywords) |kw| {
        if (std.mem.startsWith(u8, trimmed, kw)) {
            var end = kw.len;
            while (end < trimmed.len and trimmed[end] != ' ' and trimmed[end] != '(' and trimmed[end] != '{') : (end += 1) {}
            if (end > kw.len) {
                const name = trimmed[kw.len..end];
                var name_buf: [256]u8 = undefined;
                const name_z = std.fmt.bufPrintZ(&name_buf, "{s}", .{name}) catch return;
                const start = line.len - trimmed.len + kw.len;
                _ = scip_writer.scip_add_occurrence(line_num, @intCast(start), line_num, @intCast(start + name.len), name_z.ptr, 1);
                _ = scip_writer.scip_add_symbol_info(name_z.ptr, "GraphQL operation", 12);
            }
            return;
        }
    }
}

// ============================================================================
// C ABI exports for Mojo integration
// ============================================================================

export fn treesitter_index_file(
    language: [*:0]const u8,
    file_path: [*:0]const u8,
    output_path: [*:0]const u8,
) callconv(.c) c_int {
    const lang_str = std.mem.span(language);
    const lang = Language.fromString(lang_str);
    if (lang == .unknown) return -1;

    const path_str = std.mem.span(file_path);
    const out_str = std.mem.span(output_path);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Read file
    const file = std.fs.cwd().openFile(path_str, .{}) catch return -2;
    defer file.close();
    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch return -3;
    defer allocator.free(content);

    // Initialize SCIP output
    if (scip_writer.scip_init(out_str.ptr) != 0) return -4;
    defer _ = scip_writer.scip_close();

    // Write metadata
    _ = scip_writer.scip_write_metadata("ncode-treesitter", "1.0.0", ".");

    // Index file
    indexFileContent(allocator, lang, content, path_str) catch return -5;

    return 0;
}

export fn treesitter_index_directory(
    language: [*:0]const u8,
    dir_path: [*:0]const u8,
    output_path: [*:0]const u8,
) callconv(.c) c_int {
    const lang_str = std.mem.span(language);
    const lang = Language.fromString(lang_str);
    if (lang == .unknown) return -1;

    const path_str = std.mem.span(dir_path);
    const out_str = std.mem.span(output_path);

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize SCIP output
    if (scip_writer.scip_init(out_str.ptr) != 0) return -4;
    defer _ = scip_writer.scip_close();

    // Write metadata
    _ = scip_writer.scip_write_metadata("ncode-treesitter", "1.0.0", path_str.ptr);

    // Walk directory
    var dir = std.fs.cwd().openDir(path_str, .{ .iterate = true }) catch return -2;
    defer dir.close();

    var walker = dir.walk(allocator) catch return -3;
    defer walker.deinit();

    const extensions = lang.fileExtensions();
    var count: c_int = 0;

    while (walker.next() catch null) |entry| {
        if (entry.kind != .file) continue;
        
        for (extensions) |ext| {
            if (std.mem.endsWith(u8, entry.path, ext)) {
                const full_path = std.fmt.allocPrint(allocator, "{s}/{s}", .{ path_str, entry.path }) catch continue;
                defer allocator.free(full_path);

                const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
                defer file.close();
                const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch continue;
                defer allocator.free(content);

                indexFileContent(allocator, lang, content, entry.path) catch continue;
                count += 1;
                break;
            }
        }
    }

    return count;
}

// ============================================================================
// CLI main function
// ============================================================================

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len < 4) {
        std.debug.print("Usage: ncode-treesitter index --language <lang> [--output <path>] <input>\n", .{});
        std.debug.print("\nSupported languages: json, xml, yaml, toml, sql, graphql, protobuf, markdown, html, css\n", .{});
        std.process.exit(1);
    }

    var language: ?[:0]const u8 = null;
    var output: [:0]const u8 = "index.scip";
    var input: ?[:0]const u8 = null;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--language") or std.mem.eql(u8, args[i], "-l")) {
            i += 1;
            if (i < args.len) language = args[i];
        } else if (std.mem.eql(u8, args[i], "--output") or std.mem.eql(u8, args[i], "-o")) {
            i += 1;
            if (i < args.len) output = args[i];
        } else if (!std.mem.startsWith(u8, args[i], "-")) {
            input = args[i];
        }
    }

    if (language == null or input == null) {
        std.debug.print("Error: --language and input path are required\n", .{});
        std.process.exit(1);
    }

    const lang = Language.fromString(language.?);
    if (lang == .unknown) {
        std.debug.print("Error: Unknown language '{s}'\n", .{language.?});
        std.process.exit(1);
    }

    // Check if input is file or directory
    const stat = std.fs.cwd().statFile(input.?) catch |err| {
        std.debug.print("Error: Cannot access '{s}': {}\n", .{ input.?, err });
        std.process.exit(1);
    };

    // Initialize output
    if (scip_writer.scip_init(output.ptr) != 0) {
        std.debug.print("Error: Cannot create output file '{s}'\n", .{output});
        std.process.exit(1);
    }
    defer _ = scip_writer.scip_close();
    _ = scip_writer.scip_write_metadata("ncode-treesitter", "1.0.0", input.?.ptr);

    if (stat.kind == .directory) {
        std.debug.print("Indexing directory '{s}' for {s} files...\n", .{ input.?, language.? });
        // Index directory
        var dir = try std.fs.cwd().openDir(input.?, .{ .iterate = true });
        defer dir.close();
        var walker = try dir.walk(allocator);
        defer walker.deinit();

        var count: usize = 0;
        while (try walker.next()) |entry| {
            if (entry.kind != .file) continue;
            for (lang.fileExtensions()) |ext| {
                if (std.mem.endsWith(u8, entry.path, ext)) {
                    const full_path = try std.fmt.allocPrint(allocator, "{s}/{s}", .{ input.?, entry.path });
                    defer allocator.free(full_path);
                    const file = std.fs.cwd().openFile(full_path, .{}) catch continue;
                    defer file.close();
                    const content = file.readToEndAlloc(allocator, 10 * 1024 * 1024) catch continue;
                    defer allocator.free(content);
                    try indexFileContent(allocator, lang, content, entry.path);
                    count += 1;
                    break;
                }
            }
        }
        std.debug.print("Indexed {d} files -> {s}\n", .{ count, output });
    } else {
        std.debug.print("Indexing file '{s}'...\n", .{input.?});
        const file = try std.fs.cwd().openFile(input.?, .{});
        defer file.close();
        const content = try file.readToEndAlloc(allocator, 10 * 1024 * 1024);
        defer allocator.free(content);
        try indexFileContent(allocator, lang, content, input.?);
        std.debug.print("Indexed 1 file -> {s}\n", .{output});
    }
}

