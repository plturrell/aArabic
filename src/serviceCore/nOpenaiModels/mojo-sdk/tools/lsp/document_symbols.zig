// Document Symbols & Outline
// Day 76: Document outline and hierarchical navigation

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Symbol Types
// ============================================================================

/// LSP Symbol Kind (same as in symbol_index.zig for consistency)
pub const SymbolKind = enum(u8) {
    File = 1,
    Module = 2,
    Namespace = 3,
    Package = 4,
    Class = 5,
    Method = 6,
    Property = 7,
    Field = 8,
    Constructor = 9,
    Enum = 10,
    Interface = 11,
    Function = 12,
    Variable = 13,
    Constant = 14,
    String = 15,
    Number = 16,
    Boolean = 17,
    Array = 18,
    Object = 19,
    Key = 20,
    Null = 21,
    EnumMember = 22,
    Struct = 23,
    Event = 24,
    Operator = 25,
    TypeParameter = 26,
};

/// Position in a document
pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
};

/// Range in a document
pub const Range = struct {
    start: Position,
    end: Position,
    
    pub fn init(start: Position, end: Position) Range {
        return Range{ .start = start, .end = end };
    }
    
    pub fn contains(self: Range, other: Range) bool {
        // Check if self contains other
        if (self.start.line > other.start.line) return false;
        if (self.end.line < other.end.line) return false;
        if (self.start.line == other.start.line and self.start.character > other.start.character) return false;
        if (self.end.line == other.end.line and self.end.character < other.end.character) return false;
        return true;
    }
};

// ============================================================================
// Document Symbol
// ============================================================================

/// Document Symbol with hierarchy
pub const DocumentSymbol = struct {
    name: []const u8,
    detail: ?[]const u8 = null,
    kind: SymbolKind,
    range: Range,              // Full range including body
    selection_range: Range,    // Just the name
    children: std.ArrayList(*DocumentSymbol),
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        kind: SymbolKind,
        range: Range,
        selection_range: Range,
    ) !*DocumentSymbol {
        const name_copy = try allocator.dupe(u8, name);
        
        const symbol = try allocator.create(DocumentSymbol);
        symbol.* = DocumentSymbol{
            .name = name_copy,
            .kind = kind,
            .range = range,
            .selection_range = selection_range,
            .children = std.ArrayList(*DocumentSymbol){},
            .allocator = allocator,
        };
        
        return symbol;
    }
    
    pub fn deinit(self: *DocumentSymbol) void {
        self.allocator.free(self.name);
        if (self.detail) |detail| {
            self.allocator.free(detail);
        }
        
        // Recursively free children
        for (self.children.items) |child| {
            child.deinit();
            self.allocator.destroy(child);
        }
        self.children.deinit(self.allocator);
    }
    
    /// Add a child symbol
    pub fn addChild(self: *DocumentSymbol, child: *DocumentSymbol) !void {
        try self.children.append(self.allocator, child);
    }
    
    /// Set detail string
    pub fn withDetail(self: *DocumentSymbol, detail: []const u8) !void {
        self.detail = try self.allocator.dupe(u8, detail);
    }
    
    /// Get depth in hierarchy
    pub fn getDepth(self: *DocumentSymbol) usize {
        var max_child_depth: usize = 0;
        for (self.children.items) |child| {
            const child_depth = child.getDepth();
            if (child_depth > max_child_depth) {
                max_child_depth = child_depth;
            }
        }
        return max_child_depth + 1;
    }
    
    /// Count total symbols (including descendants)
    pub fn countSymbols(self: *DocumentSymbol) usize {
        var count: usize = 1; // Self
        for (self.children.items) |child| {
            count += child.countSymbols();
        }
        return count;
    }
};

// ============================================================================
// Document Symbol Provider
// ============================================================================

pub const DocumentSymbolProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DocumentSymbolProvider {
        return DocumentSymbolProvider{
            .allocator = allocator,
        };
    }
    
    /// Extract document symbols from content
    pub fn provideSymbols(self: *DocumentSymbolProvider, content: []const u8) !std.ArrayList(*DocumentSymbol) {
        var symbols = std.ArrayList(*DocumentSymbol){};
        var parser = SimpleParser.init(content);
        
        // Parse top-level symbols
        while (try parser.nextSymbol(self.allocator)) |symbol| {
            try symbols.append(self.allocator, symbol);
        }
        
        return symbols;
    }
    
    /// Build symbol hierarchy from flat list
    pub fn buildHierarchy(self: *DocumentSymbolProvider, flat_symbols: []const *DocumentSymbol) !std.ArrayList(*DocumentSymbol) {
        _ = self;
        
        var roots = std.ArrayList(*DocumentSymbol){};
        
        // Simple hierarchy: assume flat symbols are in source order
        // In real implementation, would analyze containment relationships
        for (flat_symbols) |symbol| {
            try roots.append(symbol.allocator, symbol);
        }
        
        return roots;
    }
};

// ============================================================================
// Simple Parser for Symbol Extraction
// ============================================================================

pub const SimpleParser = struct {
    content: []const u8,
    position: usize,
    line: u32,
    column: u32,
    
    pub fn init(content: []const u8) SimpleParser {
        return SimpleParser{
            .content = content,
            .position = 0,
            .line = 0,
            .column = 0,
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
        if (ch == '\n') {
            self.line += 1;
            self.column = 0;
        } else {
            self.column += 1;
        }
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
    pub fn nextSymbol(self: *SimpleParser, allocator: Allocator) !?*DocumentSymbol {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            const start_line = self.line;
            const start_col = self.column;
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        const name_end_line = self.line;
                        const name_end_col = self.column;
                        
                        // Simple: assume function ends at next 'fn' or EOF
                        const body_end_line = name_end_line + 5; // Mock body range
                        
                        const range = Range.init(
                            Position.init(start_line, start_col),
                            Position.init(body_end_line, 0),
                        );
                        
                        const selection_range = Range.init(
                            Position.init(start_line, start_col),
                            Position.init(name_end_line, name_end_col),
                        );
                        
                        return try DocumentSymbol.init(allocator, name, .Function, range, selection_range);
                    }
                } else if (std.mem.eql(u8, ident, "struct") or std.mem.eql(u8, ident, "class")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        const name_end_line = self.line;
                        const name_end_col = self.column;
                        
                        const body_end_line = name_end_line + 10; // Mock body range
                        
                        const range = Range.init(
                            Position.init(start_line, start_col),
                            Position.init(body_end_line, 0),
                        );
                        
                        const selection_range = Range.init(
                            Position.init(start_line, start_col),
                            Position.init(name_end_line, name_end_col),
                        );
                        
                        return try DocumentSymbol.init(allocator, name, .Struct, range, selection_range);
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

test "SymbolKind: enumeration" {
    try std.testing.expectEqual(@as(u8, 12), @intFromEnum(SymbolKind.Function));
    try std.testing.expectEqual(@as(u8, 23), @intFromEnum(SymbolKind.Struct));
    try std.testing.expectEqual(@as(u8, 13), @intFromEnum(SymbolKind.Variable));
}

test "Range: contains" {
    const outer = Range.init(Position.init(5, 0), Position.init(15, 0));
    const inner = Range.init(Position.init(7, 5), Position.init(10, 10));
    const outside = Range.init(Position.init(20, 0), Position.init(25, 0));
    
    try std.testing.expect(outer.contains(inner));
    try std.testing.expect(!outer.contains(outside));
}

test "DocumentSymbol: creation" {
    const range = Range.init(Position.init(10, 0), Position.init(20, 0));
    const selection_range = Range.init(Position.init(10, 0), Position.init(10, 10));
    
    const symbol = try DocumentSymbol.init(
        std.testing.allocator,
        "myFunction",
        .Function,
        range,
        selection_range,
    );
    defer {
        symbol.deinit();
        std.testing.allocator.destroy(symbol);
    }
    
    try std.testing.expectEqualStrings("myFunction", symbol.name);
    try std.testing.expectEqual(SymbolKind.Function, symbol.kind);
}

test "DocumentSymbol: hierarchy" {
    const parent_range = Range.init(Position.init(5, 0), Position.init(25, 0));
    const parent_sel = Range.init(Position.init(5, 0), Position.init(5, 10));
    
    const parent = try DocumentSymbol.init(
        std.testing.allocator,
        "MyClass",
        .Class,
        parent_range,
        parent_sel,
    );
    defer {
        parent.deinit();
        std.testing.allocator.destroy(parent);
    }
    
    const child_range = Range.init(Position.init(10, 2), Position.init(15, 2));
    const child_sel = Range.init(Position.init(10, 2), Position.init(10, 12));
    
    const child = try DocumentSymbol.init(
        std.testing.allocator,
        "method",
        .Method,
        child_range,
        child_sel,
    );
    
    try parent.addChild(child);
    
    try std.testing.expectEqual(@as(usize, 1), parent.children.items.len);
    try std.testing.expectEqualStrings("method", parent.children.items[0].name);
}

test "DocumentSymbol: depth calculation" {
    const root_range = Range.init(Position.init(0, 0), Position.init(50, 0));
    const root_sel = Range.init(Position.init(0, 0), Position.init(0, 10));
    
    const root = try DocumentSymbol.init(
        std.testing.allocator,
        "Module",
        .Module,
        root_range,
        root_sel,
    );
    defer {
        root.deinit();
        std.testing.allocator.destroy(root);
    }
    
    const child1_range = Range.init(Position.init(5, 0), Position.init(20, 0));
    const child1_sel = Range.init(Position.init(5, 0), Position.init(5, 10));
    const child1 = try DocumentSymbol.init(
        std.testing.allocator,
        "Class",
        .Class,
        child1_range,
        child1_sel,
    );
    try root.addChild(child1);
    
    const child2_range = Range.init(Position.init(10, 2), Position.init(15, 2));
    const child2_sel = Range.init(Position.init(10, 2), Position.init(10, 12));
    const child2 = try DocumentSymbol.init(
        std.testing.allocator,
        "method",
        .Method,
        child2_range,
        child2_sel,
    );
    try child1.addChild(child2);
    
    try std.testing.expectEqual(@as(usize, 3), root.getDepth());
}

test "DocumentSymbol: count symbols" {
    const parent_range = Range.init(Position.init(5, 0), Position.init(25, 0));
    const parent_sel = Range.init(Position.init(5, 0), Position.init(5, 10));
    
    const parent = try DocumentSymbol.init(
        std.testing.allocator,
        "MyClass",
        .Class,
        parent_range,
        parent_sel,
    );
    defer {
        parent.deinit();
        std.testing.allocator.destroy(parent);
    }
    
    const child1_range = Range.init(Position.init(10, 2), Position.init(15, 2));
    const child1_sel = Range.init(Position.init(10, 2), Position.init(10, 12));
    const child1 = try DocumentSymbol.init(
        std.testing.allocator,
        "method1",
        .Method,
        child1_range,
        child1_sel,
    );
    try parent.addChild(child1);
    
    const child2_range = Range.init(Position.init(17, 2), Position.init(22, 2));
    const child2_sel = Range.init(Position.init(17, 2), Position.init(17, 12));
    const child2 = try DocumentSymbol.init(
        std.testing.allocator,
        "method2",
        .Method,
        child2_range,
        child2_sel,
    );
    try parent.addChild(child2);
    
    try std.testing.expectEqual(@as(usize, 3), parent.countSymbols());
}

test "DocumentSymbolProvider: extract functions" {
    var provider = DocumentSymbolProvider.init(std.testing.allocator);
    
    const content = "fn main() { }\nfn helper() { }";
    var symbols = try provider.provideSymbols(content);
    defer {
        for (symbols.items) |sym| {
            sym.deinit();
            std.testing.allocator.destroy(sym);
        }
        symbols.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 2), symbols.items.len);
    try std.testing.expectEqualStrings("main", symbols.items[0].name);
    try std.testing.expectEqual(SymbolKind.Function, symbols.items[0].kind);
}

test "SimpleParser: parse struct" {
    const content = "struct Point { }";
    
    var parser = SimpleParser.init(content);
    const symbol = try parser.nextSymbol(std.testing.allocator);
    
    try std.testing.expect(symbol != null);
    defer {
        symbol.?.deinit();
        std.testing.allocator.destroy(symbol.?);
    }
    
    try std.testing.expectEqualStrings("Point", symbol.?.name);
    try std.testing.expectEqual(SymbolKind.Struct, symbol.?.kind);
}
