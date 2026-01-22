// Symbol Table Indexing
// Day 73: Workspace symbol indexing for code navigation

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Symbol Types
// ============================================================================

/// LSP Symbol Kind
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
    
    pub fn toString(self: SymbolKind) []const u8 {
        return switch (self) {
            .File => "File",
            .Module => "Module",
            .Namespace => "Namespace",
            .Package => "Package",
            .Class => "Class",
            .Method => "Method",
            .Property => "Property",
            .Field => "Field",
            .Constructor => "Constructor",
            .Enum => "Enum",
            .Interface => "Interface",
            .Function => "Function",
            .Variable => "Variable",
            .Constant => "Constant",
            .String => "String",
            .Number => "Number",
            .Boolean => "Boolean",
            .Array => "Array",
            .Object => "Object",
            .Key => "Key",
            .Null => "Null",
            .EnumMember => "EnumMember",
            .Struct => "Struct",
            .Event => "Event",
            .Operator => "Operator",
            .TypeParameter => "TypeParameter",
        };
    }
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
};

/// Location in a workspace
pub const Location = struct {
    uri: []const u8,
    range: Range,
    
    pub fn init(uri: []const u8, range: Range) Location {
        return Location{ .uri = uri, .range = range };
    }
};

// ============================================================================
// Symbol Information
// ============================================================================

/// Symbol Information
pub const SymbolInfo = struct {
    name: []const u8,
    kind: SymbolKind,
    location: Location,
    container_name: ?[]const u8 = null, // Parent symbol (e.g., class name for methods)
    detail: ?[]const u8 = null, // Additional info (e.g., type signature)
    allocator: Allocator,
    
    pub fn init(
        allocator: Allocator,
        name: []const u8,
        kind: SymbolKind,
        location: Location,
    ) !SymbolInfo {
        const name_copy = try allocator.dupe(u8, name);
        const uri_copy = try allocator.dupe(u8, location.uri);
        
        return SymbolInfo{
            .name = name_copy,
            .kind = kind,
            .location = Location.init(uri_copy, location.range),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SymbolInfo) void {
        self.allocator.free(self.name);
        self.allocator.free(self.location.uri);
        if (self.container_name) |container| {
            self.allocator.free(container);
        }
        if (self.detail) |detail| {
            self.allocator.free(detail);
        }
    }
    
    pub fn withContainer(self: SymbolInfo, container: []const u8) !SymbolInfo {
        var info = self;
        info.container_name = try self.allocator.dupe(u8, container);
        return info;
    }
    
    pub fn withDetail(self: SymbolInfo, detail: []const u8) !SymbolInfo {
        var info = self;
        info.detail = try self.allocator.dupe(u8, detail);
        return info;
    }
    
    /// Get fully qualified name (container.name)
    pub fn getQualifiedName(self: SymbolInfo, allocator: Allocator) ![]const u8 {
        if (self.container_name) |container| {
            return try std.fmt.allocPrint(allocator, "{s}.{s}", .{ container, self.name });
        }
        return try allocator.dupe(u8, self.name);
    }
};

// ============================================================================
// Symbol Reference
// ============================================================================

/// Symbol Reference (usage of a symbol)
pub const SymbolReference = struct {
    location: Location,
    is_definition: bool = false,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, location: Location, is_definition: bool) !SymbolReference {
        const uri_copy = try allocator.dupe(u8, location.uri);
        
        return SymbolReference{
            .location = Location.init(uri_copy, location.range),
            .is_definition = is_definition,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SymbolReference) void {
        self.allocator.free(self.location.uri);
    }
};

// ============================================================================
// Symbol Index
// ============================================================================

pub const SymbolIndex = struct {
    // Symbol name -> SymbolInfo
    symbols: std.StringHashMap(SymbolInfo),
    
    // URI -> list of symbols in that file
    file_symbols: std.StringHashMap(std.ArrayList([]const u8)),
    
    // Symbol name -> list of references
    references: std.StringHashMap(std.ArrayList(SymbolReference)),
    
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) SymbolIndex {
        return SymbolIndex{
            .symbols = std.StringHashMap(SymbolInfo).init(allocator),
            .file_symbols = std.StringHashMap(std.ArrayList([]const u8)).init(allocator),
            .references = std.StringHashMap(std.ArrayList(SymbolReference)).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SymbolIndex) void {
        // Clean up symbols
        var symbol_iter = self.symbols.iterator();
        while (symbol_iter.next()) |entry| {
            var info = entry.value_ptr;
            info.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.symbols.deinit();
        
        // Clean up file_symbols
        var file_iter = self.file_symbols.iterator();
        while (file_iter.next()) |entry| {
            for (entry.value_ptr.items) |name| {
                self.allocator.free(name);
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.file_symbols.deinit();
        
        // Clean up references
        var ref_iter = self.references.iterator();
        while (ref_iter.next()) |entry| {
            for (entry.value_ptr.items) |*ref| {
                ref.deinit();
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.references.deinit();
    }
    
    /// Add a symbol to the index
    pub fn addSymbol(self: *SymbolIndex, symbol: SymbolInfo) !void {
        const name_key = try self.allocator.dupe(u8, symbol.name);
        try self.symbols.put(name_key, symbol);
        
        // Track symbols per file
        const uri = symbol.location.uri;
        if (self.file_symbols.getPtr(uri)) |list| {
            const name_copy = try self.allocator.dupe(u8, symbol.name);
            try list.append(self.allocator, name_copy);
        } else {
            var list = try std.ArrayList([]const u8).initCapacity(self.allocator, 8);
            const name_copy = try self.allocator.dupe(u8, symbol.name);
            try list.append(self.allocator, name_copy);
            const uri_key = try self.allocator.dupe(u8, uri);
            try self.file_symbols.put(uri_key, list);
        }
    }
    
    /// Add a reference to a symbol
    pub fn addReference(self: *SymbolIndex, symbol_name: []const u8, reference: SymbolReference) !void {
        if (self.references.getPtr(symbol_name)) |list| {
            try list.append(self.allocator, reference);
        } else {
            var list = try std.ArrayList(SymbolReference).initCapacity(self.allocator, 4);
            try list.append(self.allocator, reference);
            const name_key = try self.allocator.dupe(u8, symbol_name);
            try self.references.put(name_key, list);
        }
    }
    
    /// Get a symbol by name
    pub fn getSymbol(self: *SymbolIndex, name: []const u8) ?*SymbolInfo {
        return self.symbols.getPtr(name);
    }
    
    /// Get all symbols in a file
    pub fn getFileSymbols(self: *SymbolIndex, uri: []const u8) ?[]const []const u8 {
        if (self.file_symbols.getPtr(uri)) |list| {
            return list.items;
        }
        return null;
    }
    
    /// Get all references to a symbol
    pub fn getReferences(self: *SymbolIndex, name: []const u8) ?[]const SymbolReference {
        if (self.references.getPtr(name)) |list| {
            return list.items;
        }
        return null;
    }
    
    /// Search symbols by name (prefix match)
    pub fn searchSymbols(self: *SymbolIndex, query: []const u8, results: *std.ArrayList(SymbolInfo)) !void {
        var iter = self.symbols.iterator();
        while (iter.next()) |entry| {
            if (std.mem.startsWith(u8, entry.value_ptr.name, query)) {
                // Note: This doesn't copy, just references
                try results.append(self.allocator, entry.value_ptr.*);
            }
        }
    }
    
    /// Remove all symbols from a file (for incremental updates)
    pub fn removeFileSymbols(self: *SymbolIndex, uri: []const u8) !void {
        if (self.file_symbols.fetchRemove(uri)) |kv| {
            // Remove each symbol
            for (kv.value.items) |name| {
                if (self.symbols.fetchRemove(name)) |symbol_entry| {
                    var info = symbol_entry.value;
                    info.deinit();
                    self.allocator.free(symbol_entry.key);
                }
                self.allocator.free(name);
            }
            var list = kv.value;
            list.deinit(self.allocator);
            self.allocator.free(kv.key);
        }
    }
    
    /// Get count of indexed symbols
    pub fn getSymbolCount(self: *SymbolIndex) usize {
        return self.symbols.count();
    }
    
    /// Get count of indexed files
    pub fn getFileCount(self: *SymbolIndex) usize {
        return self.file_symbols.count();
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
    
    /// Extract symbols from content
    pub fn extractSymbols(self: *SimpleParser, allocator: Allocator, uri: []const u8, symbols: *std.ArrayList(SymbolInfo)) !void {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            const start_line = self.line;
            const start_col = self.column;
            
            if (self.readIdentifier()) |ident| {
                // Simple heuristic: look for common keywords
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        const location = Location.init(
                            uri,
                            Range.init(
                                Position.init(start_line, start_col),
                                Position.init(self.line, self.column),
                            ),
                        );
                        const symbol = try SymbolInfo.init(allocator, name, .Function, location);
                        try symbols.append(allocator, symbol);
                    }
                } else if (std.mem.eql(u8, ident, "struct") or std.mem.eql(u8, ident, "class")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        const location = Location.init(
                            uri,
                            Range.init(
                                Position.init(start_line, start_col),
                                Position.init(self.line, self.column),
                            ),
                        );
                        const symbol = try SymbolInfo.init(allocator, name, .Struct, location);
                        try symbols.append(allocator, symbol);
                    }
                } else if (std.mem.eql(u8, ident, "var") or std.mem.eql(u8, ident, "let")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        const location = Location.init(
                            uri,
                            Range.init(
                                Position.init(start_line, start_col),
                                Position.init(self.line, self.column),
                            ),
                        );
                        const kind: SymbolKind = if (std.mem.eql(u8, ident, "let")) .Constant else .Variable;
                        const symbol = try SymbolInfo.init(allocator, name, kind, location);
                        try symbols.append(allocator, symbol);
                    }
                }
            } else {
                _ = self.advance();
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "SymbolKind: toString" {
    try std.testing.expectEqualStrings("Function", SymbolKind.Function.toString());
    try std.testing.expectEqualStrings("Struct", SymbolKind.Struct.toString());
    try std.testing.expectEqualStrings("Variable", SymbolKind.Variable.toString());
}

test "SymbolInfo: creation" {
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(10, 5), Position.init(10, 15)),
    );
    
    var symbol = try SymbolInfo.init(std.testing.allocator, "myFunction", .Function, location);
    defer symbol.deinit();
    
    try std.testing.expectEqualStrings("myFunction", symbol.name);
    try std.testing.expectEqual(SymbolKind.Function, symbol.kind);
}

test "SymbolInfo: qualified name" {
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(10, 5), Position.init(10, 15)),
    );
    
    var symbol = try SymbolInfo.init(std.testing.allocator, "method", .Method, location);
    defer symbol.deinit();
    
    symbol = try symbol.withContainer("MyClass");
    
    const qualified = try symbol.getQualifiedName(std.testing.allocator);
    defer std.testing.allocator.free(qualified);
    
    try std.testing.expectEqualStrings("MyClass.method", qualified);
}

test "SymbolReference: creation" {
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(5, 10), Position.init(5, 20)),
    );
    
    var reference = try SymbolReference.init(std.testing.allocator, location, true);
    defer reference.deinit();
    
    try std.testing.expect(reference.is_definition);
}

test "SymbolIndex: add and get symbol" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(10, 0), Position.init(10, 10)),
    );
    
    const symbol = try SymbolInfo.init(std.testing.allocator, "testFunc", .Function, location);
    try index.addSymbol(symbol);
    
    const retrieved = index.getSymbol("testFunc");
    try std.testing.expect(retrieved != null);
    try std.testing.expectEqualStrings("testFunc", retrieved.?.name);
}

test "SymbolIndex: file symbols tracking" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const uri = "file:///test.mojo";
    const location = Location.init(uri, Range.init(Position.init(0, 0), Position.init(0, 5)));
    
    const symbol1 = try SymbolInfo.init(std.testing.allocator, "func1", .Function, location);
    try index.addSymbol(symbol1);
    
    const symbol2 = try SymbolInfo.init(std.testing.allocator, "func2", .Function, location);
    try index.addSymbol(symbol2);
    
    const file_symbols = index.getFileSymbols(uri);
    try std.testing.expect(file_symbols != null);
    try std.testing.expectEqual(@as(usize, 2), file_symbols.?.len);
}

test "SymbolIndex: add references" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(5, 0), Position.init(5, 5)),
    );
    
    const reference = try SymbolReference.init(std.testing.allocator, location, false);
    try index.addReference("testFunc", reference);
    
    const refs = index.getReferences("testFunc");
    try std.testing.expect(refs != null);
    try std.testing.expectEqual(@as(usize, 1), refs.?.len);
}

test "SymbolIndex: search symbols" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(0, 0), Position.init(0, 5)),
    );
    
    const symbol1 = try SymbolInfo.init(std.testing.allocator, "testFunc1", .Function, location);
    try index.addSymbol(symbol1);
    
    const symbol2 = try SymbolInfo.init(std.testing.allocator, "testFunc2", .Function, location);
    try index.addSymbol(symbol2);
    
    const symbol3 = try SymbolInfo.init(std.testing.allocator, "otherFunc", .Function, location);
    try index.addSymbol(symbol3);
    
    var results = try std.ArrayList(SymbolInfo).initCapacity(std.testing.allocator, 8);
    defer results.deinit(std.testing.allocator);
    
    try index.searchSymbols("test", &results);
    
    try std.testing.expectEqual(@as(usize, 2), results.items.len);
}

test "SymbolIndex: remove file symbols" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const uri = "file:///test.mojo";
    const location = Location.init(uri, Range.init(Position.init(0, 0), Position.init(0, 5)));
    
    const symbol = try SymbolInfo.init(std.testing.allocator, "testFunc", .Function, location);
    try index.addSymbol(symbol);
    
    try std.testing.expectEqual(@as(usize, 1), index.getSymbolCount());
    
    try index.removeFileSymbols(uri);
    
    try std.testing.expectEqual(@as(usize, 0), index.getSymbolCount());
}

test "SymbolIndex: counts" {
    var index = SymbolIndex.init(std.testing.allocator);
    defer index.deinit();
    
    const location1 = Location.init("file:///test1.mojo", Range.init(Position.init(0, 0), Position.init(0, 5)));
    const location2 = Location.init("file:///test2.mojo", Range.init(Position.init(0, 0), Position.init(0, 5)));
    
    const symbol1 = try SymbolInfo.init(std.testing.allocator, "func1", .Function, location1);
    try index.addSymbol(symbol1);
    
    const symbol2 = try SymbolInfo.init(std.testing.allocator, "func2", .Function, location2);
    try index.addSymbol(symbol2);
    
    try std.testing.expectEqual(@as(usize, 2), index.getSymbolCount());
    try std.testing.expectEqual(@as(usize, 2), index.getFileCount());
}

test "SimpleParser: extract functions" {
    const content = "fn main() { }\nfn helper() { }";
    
    var parser = SimpleParser.init(content);
    var symbols = try std.ArrayList(SymbolInfo).initCapacity(std.testing.allocator, 8);
    defer {
        for (symbols.items) |*sym| {
            sym.deinit();
        }
        symbols.deinit(std.testing.allocator);
    }
    
    try parser.extractSymbols(std.testing.allocator, "file:///test.mojo", &symbols);
    
    try std.testing.expectEqual(@as(usize, 2), symbols.items.len);
    try std.testing.expectEqualStrings("main", symbols.items[0].name);
    try std.testing.expectEqualStrings("helper", symbols.items[1].name);
}

test "SimpleParser: extract structs and variables" {
    const content = "struct Point { }\nlet x = 5\nvar y = 10";
    
    var parser = SimpleParser.init(content);
    var symbols = try std.ArrayList(SymbolInfo).initCapacity(std.testing.allocator, 8);
    defer {
        for (symbols.items) |*sym| {
            sym.deinit();
        }
        symbols.deinit(std.testing.allocator);
    }
    
    try parser.extractSymbols(std.testing.allocator, "file:///test.mojo", &symbols);
    
    try std.testing.expectEqual(@as(usize, 3), symbols.items.len);
    try std.testing.expectEqual(SymbolKind.Struct, symbols.items[0].kind);
    try std.testing.expectEqual(SymbolKind.Constant, symbols.items[1].kind);
    try std.testing.expectEqual(SymbolKind.Variable, symbols.items[2].kind);
}
