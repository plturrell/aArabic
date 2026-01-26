// Autocomplete Engine
// Day 78: Code completion for LSP

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Completion Types
// ============================================================================

/// LSP Completion Item Kind
pub const CompletionItemKind = enum(u8) {
    Text = 1,
    Method = 2,
    Function = 3,
    Constructor = 4,
    Field = 5,
    Variable = 6,
    Class = 7,
    Interface = 8,
    Module = 9,
    Property = 10,
    Unit = 11,
    Value = 12,
    Enum = 13,
    Keyword = 14,
    Snippet = 15,
    Color = 16,
    File = 17,
    Reference = 18,
    Folder = 19,
    EnumMember = 20,
    Constant = 21,
    Struct = 22,
    Event = 23,
    Operator = 24,
    TypeParameter = 25,
};

/// Position in document
pub const Position = struct {
    line: u32,
    character: u32,
};

/// Completion trigger kind
pub const CompletionTriggerKind = enum(u8) {
    Invoked = 1,         // Manually invoked (Ctrl+Space)
    TriggerCharacter = 2, // Triggered by character (., ::)
    TriggerForIncompleteCompletions = 3,
};

/// Completion context
pub const CompletionContext = struct {
    trigger_kind: CompletionTriggerKind,
    trigger_character: ?u8 = null,
};

// ============================================================================
// Completion Item
// ============================================================================

pub const CompletionItem = struct {
    label: []const u8,
    kind: CompletionItemKind,
    detail: ?[]const u8 = null,
    documentation: ?[]const u8 = null,
    insert_text: ?[]const u8 = null,
    sort_text: ?[]const u8 = null,
    filter_text: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, label: []const u8, kind: CompletionItemKind) !CompletionItem {
        return CompletionItem{
            .label = try allocator.dupe(u8, label),
            .kind = kind,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *CompletionItem) void {
        self.allocator.free(self.label);
        if (self.detail) |d| self.allocator.free(d);
        if (self.documentation) |d| self.allocator.free(d);
        if (self.insert_text) |t| self.allocator.free(t);
        if (self.sort_text) |t| self.allocator.free(t);
        if (self.filter_text) |t| self.allocator.free(t);
    }
    
    pub fn withDetail(self: *CompletionItem, detail: []const u8) !void {
        self.detail = try self.allocator.dupe(u8, detail);
    }
    
    pub fn withDocumentation(self: *CompletionItem, doc: []const u8) !void {
        self.documentation = try self.allocator.dupe(u8, doc);
    }
    
    pub fn withInsertText(self: *CompletionItem, text: []const u8) !void {
        self.insert_text = try self.allocator.dupe(u8, text);
    }
};

// ============================================================================
// Keyword Completions
// ============================================================================

const KEYWORDS = [_][]const u8{
    "fn",      "def",     "struct",  "class",   "enum",
    "if",      "else",    "for",     "while",   "return",
    "break",   "continue", "let",     "var",     "const",
    "import",  "from",    "as",      "pub",     "mut",
    "ref",     "self",    "true",    "false",   "null",
};

// ============================================================================
// Completion Provider
// ============================================================================

pub const CompletionProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CompletionProvider {
        return CompletionProvider{ .allocator = allocator };
    }
    
    /// Provide completions at position
    pub fn provideCompletions(
        self: *CompletionProvider,
        content: []const u8,
        position: Position,
        context: CompletionContext,
    ) !std.ArrayList(CompletionItem) {
        var items = std.ArrayList(CompletionItem){};
        
        // Check trigger character
        if (context.trigger_character) |trigger| {
            if (trigger == '.') {
                // Member access completion
                try self.provideMemberCompletions(&items, content, position);
                return items;
            } else if (trigger == ':') {
                // Namespace/module completion
                try self.provideNamespaceCompletions(&items, content, position);
                return items;
            }
        }
        
        // Get word at cursor
        const word = self.getWordAtPosition(content, position);
        
        // Keyword completions
        try self.provideKeywordCompletions(&items, word);
        
        // Symbol completions (functions, variables, types)
        try self.provideSymbolCompletions(&items, content, word);
        
        return items;
    }
    
    /// Get word at cursor position
    fn getWordAtPosition(self: *CompletionProvider, content: []const u8, position: Position) []const u8 {
        _ = self;
        
        // Find line start
        var line: u32 = 0;
        var line_start: usize = 0;
        var i: usize = 0;
        
        while (i < content.len) : (i += 1) {
            if (line == position.line) {
                break;
            }
            if (content[i] == '\n') {
                line += 1;
                line_start = i + 1;
            }
        }
        
        // Find word boundaries
        const cursor_pos = line_start + position.character;
        if (cursor_pos >= content.len) return "";
        
        var word_start = cursor_pos;
        while (word_start > line_start) : (word_start -= 1) {
            const ch = content[word_start - 1];
            if (!std.ascii.isAlphanumeric(ch) and ch != '_') break;
        }
        
        var word_end = cursor_pos;
        while (word_end < content.len) : (word_end += 1) {
            const ch = content[word_end];
            if (!std.ascii.isAlphanumeric(ch) and ch != '_') break;
        }
        
        return content[word_start..word_end];
    }
    
    /// Provide keyword completions
    fn provideKeywordCompletions(
        self: *CompletionProvider,
        items: *std.ArrayList(CompletionItem),
        prefix: []const u8,
    ) !void {
        for (KEYWORDS) |keyword| {
            if (prefix.len == 0 or std.mem.startsWith(u8, keyword, prefix)) {
                const item = try CompletionItem.init(self.allocator, keyword, .Keyword);
                try items.append(self.allocator, item);
            }
        }
    }
    
    /// Provide symbol completions (simple parsing)
    fn provideSymbolCompletions(
        self: *CompletionProvider,
        items: *std.ArrayList(CompletionItem),
        content: []const u8,
        prefix: []const u8,
    ) !void {
        var parser = SimpleParser.init(content);
        
        while (try parser.nextSymbol()) |symbol| {
            if (prefix.len == 0 or std.mem.startsWith(u8, symbol.name, prefix)) {
                var item = try CompletionItem.init(self.allocator, symbol.name, symbol.kind);
                if (symbol.detail) |detail| {
                    try item.withDetail(detail);
                }
                try items.append(self.allocator, item);
            }
        }
    }
    
    /// Provide member completions (struct.field)
    fn provideMemberCompletions(
        self: *CompletionProvider,
        items: *std.ArrayList(CompletionItem),
        content: []const u8,
        position: Position,
    ) !void {
        _ = content;
        _ = position;
        
        // Mock: Suggest common members
        const members = [_][]const u8{ "field", "method", "property" };
        for (members) |member| {
            const item = try CompletionItem.init(self.allocator, member, .Field);
            try items.append(self.allocator, item);
        }
    }
    
    /// Provide namespace completions (Module::item)
    fn provideNamespaceCompletions(
        self: *CompletionProvider,
        items: *std.ArrayList(CompletionItem),
        content: []const u8,
        position: Position,
    ) !void {
        _ = content;
        _ = position;
        
        // Mock: Suggest common items
        const ns_items = [_][]const u8{ "function", "type", "constant" };
        for (ns_items) |ns_item| {
            const item = try CompletionItem.init(self.allocator, ns_item, .Function);
            try items.append(self.allocator, item);
        }
    }
};

// ============================================================================
// Simple Parser for Symbol Extraction
// ============================================================================

const ParsedSymbol = struct {
    name: []const u8,
    kind: CompletionItemKind,
    detail: ?[]const u8 = null,
};

pub const SimpleParser = struct {
    content: []const u8,
    position: usize,
    
    pub fn init(content: []const u8) SimpleParser {
        return SimpleParser{
            .content = content,
            .position = 0,
        };
    }
    
    fn peek(self: *SimpleParser) ?u8 {
        if (self.position >= self.content.len) return null;
        return self.content[self.position];
    }
    
    fn advance(self: *SimpleParser) ?u8 {
        if (self.position >= self.content.len) return null;
        const ch = self.content[self.position];
        self.position += 1;
        return ch;
    }
    
    fn skipWhitespace(self: *SimpleParser) void {
        while (self.peek()) |ch| {
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                _ = self.advance();
            } else {
                break;
            }
        }
    }
    
    fn readIdentifier(self: *SimpleParser) ?[]const u8 {
        const start = self.position;
        while (self.peek()) |ch| {
            if (std.ascii.isAlphanumeric(ch) or ch == '_') {
                _ = self.advance();
            } else {
                break;
            }
        }
        if (self.position > start) {
            return self.content[start..self.position];
        }
        return null;
    }
    
    /// Parse next symbol
    pub fn nextSymbol(self: *SimpleParser) !?ParsedSymbol {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return ParsedSymbol{
                            .name = name,
                            .kind = .Function,
                            .detail = "function",
                        };
                    }
                } else if (std.mem.eql(u8, ident, "struct") or std.mem.eql(u8, ident, "class")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return ParsedSymbol{
                            .name = name,
                            .kind = .Struct,
                            .detail = "type",
                        };
                    }
                } else if (std.mem.eql(u8, ident, "let") or std.mem.eql(u8, ident, "var")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return ParsedSymbol{
                            .name = name,
                            .kind = .Variable,
                            .detail = if (std.mem.eql(u8, ident, "let")) "constant" else "variable",
                        };
                    }
                }
            } else {
                _ = self.advance();
            }
        }
        
        return null;
    }
};

// ============================================================================
// Completion Filtering
// ============================================================================

pub const CompletionFilter = struct {
    pub fn filterByPrefix(items: []const CompletionItem, prefix: []const u8) !std.ArrayList(CompletionItem) {
        var filtered = std.ArrayList(CompletionItem){};
        
        for (items) |item| {
            if (prefix.len == 0 or std.mem.startsWith(u8, item.label, prefix)) {
                // Note: This doesn't copy, just references
                try filtered.append(item.allocator, item);
            }
        }
        
        return filtered;
    }
    
    pub fn sortByLabel(items: []CompletionItem) void {
        std.mem.sort(CompletionItem, items, {}, compareLabel);
    }
    
    fn compareLabel(_: void, a: CompletionItem, b: CompletionItem) bool {
        return std.mem.order(u8, a.label, b.label) == .lt;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CompletionItemKind: values" {
    try std.testing.expectEqual(@as(u8, 3), @intFromEnum(CompletionItemKind.Function));
    try std.testing.expectEqual(@as(u8, 14), @intFromEnum(CompletionItemKind.Keyword));
    try std.testing.expectEqual(@as(u8, 22), @intFromEnum(CompletionItemKind.Struct));
}

test "CompletionItem: creation" {
    var item = try CompletionItem.init(std.testing.allocator, "myFunc", .Function);
    defer item.deinit();
    
    try std.testing.expectEqualStrings("myFunc", item.label);
    try std.testing.expectEqual(CompletionItemKind.Function, item.kind);
}

test "CompletionItem: with detail" {
    var item = try CompletionItem.init(std.testing.allocator, "myFunc", .Function);
    defer item.deinit();
    
    try item.withDetail("fn() -> i32");
    
    try std.testing.expectEqualStrings("fn() -> i32", item.detail.?);
}

test "CompletionItem: with documentation" {
    var item = try CompletionItem.init(std.testing.allocator, "myFunc", .Function);
    defer item.deinit();
    
    try item.withDocumentation("This is a function");
    
    try std.testing.expectEqualStrings("This is a function", item.documentation.?);
}

test "CompletionProvider: keyword completions" {
    var provider = CompletionProvider.init(std.testing.allocator);
    
    const content = "";
    const position = Position{ .line = 0, .character = 0 };
    const context = CompletionContext{ .trigger_kind = .Invoked };
    
    var items = try provider.provideCompletions(content, position, context);
    defer {
        for (items.items) |*item| {
            item.deinit();
        }
        items.deinit(std.testing.allocator);
    }
    
    // Should have keywords
    try std.testing.expect(items.items.len > 0);
}

test "CompletionProvider: filtered keywords" {
    var provider = CompletionProvider.init(std.testing.allocator);
    
    const content = "fu";
    const position = Position{ .line = 0, .character = 2 };
    const context = CompletionContext{ .trigger_kind = .Invoked };
    
    var items = try provider.provideCompletions(content, position, context);
    defer {
        for (items.items) |*item| {
            item.deinit();
        }
        items.deinit(std.testing.allocator);
    }
    
    // Should have "fn" at minimum
    var has_fn = false;
    for (items.items) |item| {
        if (std.mem.eql(u8, item.label, "fn")) {
            has_fn = true;
            break;
        }
    }
    try std.testing.expect(has_fn);
}

test "CompletionProvider: member access" {
    var provider = CompletionProvider.init(std.testing.allocator);
    
    const content = "obj.";
    const position = Position{ .line = 0, .character = 4 };
    const context = CompletionContext{
        .trigger_kind = .TriggerCharacter,
        .trigger_character = '.',
    };
    
    var items = try provider.provideCompletions(content, position, context);
    defer {
        for (items.items) |*item| {
            item.deinit();
        }
        items.deinit(std.testing.allocator);
    }
    
    // Should have member suggestions
    try std.testing.expect(items.items.len > 0);
}

test "CompletionProvider: namespace access" {
    var provider = CompletionProvider.init(std.testing.allocator);
    
    const content = "Module:";
    const position = Position{ .line = 0, .character = 7 };
    const context = CompletionContext{
        .trigger_kind = .TriggerCharacter,
        .trigger_character = ':',
    };
    
    var items = try provider.provideCompletions(content, position, context);
    defer {
        for (items.items) |*item| {
            item.deinit();
        }
        items.deinit(std.testing.allocator);
    }
    
    // Should have namespace suggestions
    try std.testing.expect(items.items.len > 0);
}

test "SimpleParser: parse function" {
    const content = "fn myFunc() {}";
    var parser = SimpleParser.init(content);
    
    const symbol = try parser.nextSymbol();
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("myFunc", symbol.?.name);
    try std.testing.expectEqual(CompletionItemKind.Function, symbol.?.kind);
}

test "SimpleParser: parse struct" {
    const content = "struct Point {}";
    var parser = SimpleParser.init(content);
    
    const symbol = try parser.nextSymbol();
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("Point", symbol.?.name);
    try std.testing.expectEqual(CompletionItemKind.Struct, symbol.?.kind);
}

test "SimpleParser: parse variable" {
    const content = "let x = 5";
    var parser = SimpleParser.init(content);
    
    const symbol = try parser.nextSymbol();
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("x", symbol.?.name);
    try std.testing.expectEqual(CompletionItemKind.Variable, symbol.?.kind);
}

test "CompletionProvider: symbol completions" {
    var provider = CompletionProvider.init(std.testing.allocator);
    
    const content = "fn helper() {}\nfn main() { h }";
    const position = Position{ .line = 1, .character = 13 };
    const context = CompletionContext{ .trigger_kind = .Invoked };
    
    var items = try provider.provideCompletions(content, position, context);
    defer {
        for (items.items) |*item| {
            item.deinit();
        }
        items.deinit(std.testing.allocator);
    }
    
    // Should include "helper" function
    var has_helper = false;
    for (items.items) |item| {
        if (std.mem.eql(u8, item.label, "helper")) {
            has_helper = true;
            break;
        }
    }
    try std.testing.expect(has_helper);
}
