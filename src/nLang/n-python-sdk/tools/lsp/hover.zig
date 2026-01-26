// Hover Information
// Day 82: Symbol information on hover

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Position & Range
// ============================================================================

pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
};

pub const Range = struct {
    start: Position,
    end: Position,
    
    pub fn init(start: Position, end: Position) Range {
        return Range{ .start = start, .end = end };
    }
};

// ============================================================================
// Markup Content
// ============================================================================

pub const MarkupKind = enum {
    PlainText,
    Markdown,
    
    pub fn toString(self: MarkupKind) []const u8 {
        return switch (self) {
            .PlainText => "plaintext",
            .Markdown => "markdown",
        };
    }
};

pub const MarkupContent = struct {
    kind: MarkupKind,
    value: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, kind: MarkupKind, value: []const u8) !MarkupContent {
        return MarkupContent{
            .kind = kind,
            .value = try allocator.dupe(u8, value),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MarkupContent) void {
        self.allocator.free(self.value);
    }
};

// ============================================================================
// Hover Result
// ============================================================================

pub const Hover = struct {
    contents: MarkupContent,
    range: ?Range = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, contents: MarkupContent, range: ?Range) Hover {
        return Hover{
            .contents = contents,
            .range = range,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Hover) void {
        self.contents.deinit();
    }
};

// ============================================================================
// Symbol Information
// ============================================================================

pub const SymbolKind = enum {
    Function,
    Struct,
    Variable,
    Method,
    Field,
    Constant,
};

pub const SymbolInfo = struct {
    name: []const u8,
    kind: SymbolKind,
    type_signature: ?[]const u8 = null,
    documentation: ?[]const u8 = null,
    
    pub fn init(name: []const u8, kind: SymbolKind) SymbolInfo {
        return SymbolInfo{
            .name = name,
            .kind = kind,
        };
    }
};

// ============================================================================
// Hover Provider
// ============================================================================

pub const HoverProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) HoverProvider {
        return HoverProvider{ .allocator = allocator };
    }
    
    /// Provide hover information at position
    pub fn provideHover(
        self: *HoverProvider,
        content: []const u8,
        position: Position,
    ) !?Hover {
        // Get symbol at cursor
        const symbol_name = self.getSymbolAtPosition(content, position);
        if (symbol_name.len == 0) return null;
        
        // Find symbol definition
        const symbol_info = try self.findSymbolInfo(content, symbol_name);
        if (symbol_info == null) return null;
        
        // Build hover content
        const hover_content = try self.buildHoverContent(symbol_info.?);
        
        // Calculate hover range
        const hover_range = self.getSymbolRange(content, position);
        
        return Hover.init(self.allocator, hover_content, hover_range);
    }
    
    /// Get symbol at position
    fn getSymbolAtPosition(self: *HoverProvider, content: []const u8, position: Position) []const u8 {
        _ = self;
        
        var line: u32 = 0;
        var line_start: usize = 0;
        var i: usize = 0;
        
        while (i < content.len) : (i += 1) {
            if (line == position.line) break;
            if (content[i] == '\n') {
                line += 1;
                line_start = i + 1;
            }
        }
        
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
    
    /// Get range of symbol at position
    fn getSymbolRange(self: *HoverProvider, content: []const u8, position: Position) ?Range {
        _ = self;
        
        var line: u32 = 0;
        var line_start: usize = 0;
        var i: usize = 0;
        
        while (i < content.len) : (i += 1) {
            if (line == position.line) break;
            if (content[i] == '\n') {
                line += 1;
                line_start = i + 1;
            }
        }
        
        const cursor_pos = line_start + position.character;
        if (cursor_pos >= content.len) return null;
        
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
        
        const start_char = @as(u32, @intCast(word_start - line_start));
        const end_char = @as(u32, @intCast(word_end - line_start));
        
        return Range.init(
            Position.init(position.line, start_char),
            Position.init(position.line, end_char),
        );
    }
    
    /// Find symbol information
    fn findSymbolInfo(self: *HoverProvider, content: []const u8, symbol_name: []const u8) !?SymbolInfo {
        _ = self;
        
        var parser = SimpleParser.init(content);
        while (try parser.nextSymbol()) |sym| {
            if (std.mem.eql(u8, sym.name, symbol_name)) {
                return sym;
            }
        }
        
        return null;
    }
    
    /// Build hover content
    fn buildHoverContent(self: *HoverProvider, info: SymbolInfo) !MarkupContent {
        var builder = MarkdownBuilder.init(self.allocator);
        defer builder.deinit();
        
        // Add signature
        const signature = try self.formatSignature(info);
        defer self.allocator.free(signature);
        try builder.addCodeBlock("mojo", signature);
        
        // Add separator
        try builder.addText("\n");
        
        // Add documentation
        if (info.documentation) |doc| {
            try builder.addText(doc);
        } else {
            try builder.addText("No documentation available");
        }
        
        const markdown = try builder.build();
        defer self.allocator.free(markdown);
        return try MarkupContent.init(self.allocator, .Markdown, markdown);
    }
    
    /// Format symbol signature
    fn formatSignature(self: *HoverProvider, info: SymbolInfo) ![]const u8 {
        return switch (info.kind) {
            .Function => if (info.type_signature) |sig|
                try std.fmt.allocPrint(self.allocator, "fn {s}{s}", .{ info.name, sig })
            else
                try std.fmt.allocPrint(self.allocator, "fn {s}()", .{info.name}),
            
            .Struct => try std.fmt.allocPrint(self.allocator, "struct {s}", .{info.name}),
            
            .Variable, .Constant => if (info.type_signature) |sig|
                try std.fmt.allocPrint(self.allocator, "{s}: {s}", .{ info.name, sig })
            else
                try std.fmt.allocPrint(self.allocator, "{s}", .{info.name}),
            
            else => try self.allocator.dupe(u8, info.name),
        };
    }
};

// ============================================================================
// Markdown Builder
// ============================================================================

pub const MarkdownBuilder = struct {
    content: std.ArrayList(u8),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MarkdownBuilder {
        return MarkdownBuilder{
            .content = std.ArrayList(u8){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MarkdownBuilder) void {
        self.content.deinit(self.allocator);
    }
    
    pub fn addText(self: *MarkdownBuilder, text: []const u8) !void {
        try self.content.appendSlice(self.allocator, text);
    }
    
    pub fn addCodeBlock(self: *MarkdownBuilder, language: []const u8, code: []const u8) !void {
        try self.content.appendSlice(self.allocator, "```");
        try self.content.appendSlice(self.allocator, language);
        try self.content.append(self.allocator, '\n');
        try self.content.appendSlice(self.allocator, code);
        try self.content.append(self.allocator, '\n');
        try self.content.appendSlice(self.allocator, "```");
    }
    
    pub fn build(self: *MarkdownBuilder) ![]const u8 {
        return try self.allocator.dupe(u8, self.content.items);
    }
};

// ============================================================================
// Simple Parser
// ============================================================================

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
    
    pub fn nextSymbol(self: *SimpleParser) !?SymbolInfo {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return SymbolInfo{
                            .name = name,
                            .kind = .Function,
                            .type_signature = "() -> void",
                            .documentation = "Function definition",
                        };
                    }
                } else if (std.mem.eql(u8, ident, "struct") or std.mem.eql(u8, ident, "class")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return SymbolInfo{
                            .name = name,
                            .kind = .Struct,
                            .documentation = "Type definition",
                        };
                    }
                } else if (std.mem.eql(u8, ident, "let") or std.mem.eql(u8, ident, "const")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return SymbolInfo{
                            .name = name,
                            .kind = .Constant,
                            .type_signature = "i32",
                            .documentation = "Constant value",
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
// Tests
// ============================================================================

test "MarkupKind: toString" {
    try std.testing.expectEqualStrings("plaintext", MarkupKind.PlainText.toString());
    try std.testing.expectEqualStrings("markdown", MarkupKind.Markdown.toString());
}

test "MarkupContent: creation" {
    var content = try MarkupContent.init(std.testing.allocator, .Markdown, "# Test");
    defer content.deinit();
    
    try std.testing.expectEqual(MarkupKind.Markdown, content.kind);
    try std.testing.expectEqualStrings("# Test", content.value);
}

test "MarkdownBuilder: build content" {
    var builder = MarkdownBuilder.init(std.testing.allocator);
    defer builder.deinit();
    
    try builder.addCodeBlock("mojo", "fn test()");
    try builder.addText("\nDescription");
    
    const result = try builder.build();
    defer std.testing.allocator.free(result);
    
    try std.testing.expect(std.mem.indexOf(u8, result, "```mojo") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "fn test()") != null);
}

test "SimpleParser: parse function" {
    const content = "fn myFunc() {}";
    var parser = SimpleParser.init(content);
    
    const sym = try parser.nextSymbol();
    try std.testing.expect(sym != null);
    try std.testing.expectEqualStrings("myFunc", sym.?.name);
    try std.testing.expectEqual(SymbolKind.Function, sym.?.kind);
}

test "SimpleParser: parse struct" {
    const content = "struct Point { x: i32 }";
    var parser = SimpleParser.init(content);
    
    const sym = try parser.nextSymbol();
    try std.testing.expect(sym != null);
    try std.testing.expectEqualStrings("Point", sym.?.name);
    try std.testing.expectEqual(SymbolKind.Struct, sym.?.kind);
}

test "HoverProvider: get symbol at position" {
    var provider = HoverProvider.init(std.testing.allocator);
    
    const content = "fn myFunction() {}";
    const position = Position.init(0, 5); // In "myFunction"
    
    const symbol = provider.getSymbolAtPosition(content, position);
    try std.testing.expectEqualStrings("myFunction", symbol);
}

test "HoverProvider: provide hover for function" {
    var provider = HoverProvider.init(std.testing.allocator);
    
    const content = "fn helper() {}\nlet x = helper()";
    const position = Position.init(1, 10); // On "helper" usage
    
    var hover = try provider.provideHover(content, position);
    try std.testing.expect(hover != null);
    defer if (hover) |*h| h.deinit();
    
    try std.testing.expectEqual(MarkupKind.Markdown, hover.?.contents.kind);
    try std.testing.expect(std.mem.indexOf(u8, hover.?.contents.value, "fn helper") != null);
}

test "HoverProvider: no hover for non-symbol" {
    var provider = HoverProvider.init(std.testing.allocator);
    
    const content = "fn test() {}";
    const position = Position.init(0, 0); // On "fn" keyword
    
    const hover = try provider.provideHover(content, position);
    try std.testing.expect(hover == null);
}
