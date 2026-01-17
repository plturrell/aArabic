// Go-to-Definition
// Day 80: Navigate to symbol definitions

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Location Types
// ============================================================================

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

/// Location (URI + Range)
pub const Location = struct {
    uri: []const u8,
    range: Range,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, uri: []const u8, range: Range) !Location {
        return Location{
            .uri = try allocator.dupe(u8, uri),
            .range = range,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Location) void {
        self.allocator.free(self.uri);
    }
};

// ============================================================================
// Symbol Definition
// ============================================================================

pub const SymbolKind = enum {
    Function,
    Struct,
    Variable,
    Method,
    Field,
    Module,
};

pub const SymbolDefinition = struct {
    name: []const u8,
    kind: SymbolKind,
    location: Location,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8, kind: SymbolKind, location: Location) !SymbolDefinition {
        return SymbolDefinition{
            .name = try allocator.dupe(u8, name),
            .kind = kind,
            .location = location,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *SymbolDefinition) void {
        self.allocator.free(self.name);
        self.location.deinit();
    }
};

// ============================================================================
// Definition Provider
// ============================================================================

pub const DefinitionProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DefinitionProvider {
        return DefinitionProvider{ .allocator = allocator };
    }
    
    /// Find definition of symbol at position
    pub fn findDefinition(
        self: *DefinitionProvider,
        uri: []const u8,
        content: []const u8,
        position: Position,
    ) !std.ArrayList(Location) {
        var locations = std.ArrayList(Location){};
        
        // Get word at cursor
        const word = self.getWordAtPosition(content, position);
        if (word.len == 0) return locations;
        
        // Search for definition in current file
        if (try self.findInFile(uri, content, word)) |location| {
            try locations.append(self.allocator, location);
        }
        
        return locations;
    }
    
    /// Find definition in file
    fn findInFile(
        self: *DefinitionProvider,
        uri: []const u8,
        content: []const u8,
        symbol_name: []const u8,
    ) !?Location {
        var parser = SimpleParser.init(content);
        
        while (try parser.nextDefinition()) |def| {
            if (std.mem.eql(u8, def.name, symbol_name)) {
                return try Location.init(
                    self.allocator,
                    uri,
                    Range.init(
                        Position.init(def.line, def.column),
                        Position.init(def.line, def.column + @as(u32, @intCast(def.name.len))),
                    ),
                );
            }
        }
        
        return null;
    }
    
    /// Get word at cursor position
    fn getWordAtPosition(self: *DefinitionProvider, content: []const u8, position: Position) []const u8 {
        _ = self;
        
        // Find line start
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
};

// ============================================================================
// Cross-file Definition Resolver
// ============================================================================

pub const CrossFileResolver = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CrossFileResolver {
        return CrossFileResolver{ .allocator = allocator };
    }
    
    /// Find definitions across multiple files
    pub fn findAcrossFiles(
        self: *CrossFileResolver,
        symbol_name: []const u8,
        file_contents: std.StringHashMap([]const u8),
    ) !std.ArrayList(Location) {
        var locations = std.ArrayList(Location){};
        
        var iter = file_contents.iterator();
        while (iter.next()) |entry| {
            const uri = entry.key_ptr.*;
            const content = entry.value_ptr.*;
            
            var parser = SimpleParser.init(content);
            while (try parser.nextDefinition()) |def| {
                if (std.mem.eql(u8, def.name, symbol_name)) {
                    const location = try Location.init(
                        self.allocator,
                        uri,
                        Range.init(
                            Position.init(def.line, def.column),
                            Position.init(def.line, def.column + @as(u32, @intCast(def.name.len))),
                        ),
                    );
                    try locations.append(self.allocator, location);
                }
            }
        }
        
        return locations;
    }
};

// ============================================================================
// Simple Parser for Definition Finding
// ============================================================================

const DefinitionInfo = struct {
    name: []const u8,
    line: u32,
    column: u32,
    kind: SymbolKind,
};

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
    
    /// Parse next definition
    pub fn nextDefinition(self: *SimpleParser) !?DefinitionInfo {
        while (self.position < self.content.len) {
            self.skipWhitespace();
            
            const def_line = self.line;
            const def_col = self.column;
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, "fn") or std.mem.eql(u8, ident, "def")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return DefinitionInfo{
                            .name = name,
                            .line = def_line,
                            .column = def_col,
                            .kind = .Function,
                        };
                    }
                } else if (std.mem.eql(u8, ident, "struct") or std.mem.eql(u8, ident, "class")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return DefinitionInfo{
                            .name = name,
                            .line = def_line,
                            .column = def_col,
                            .kind = .Struct,
                        };
                    }
                } else if (std.mem.eql(u8, ident, "let") or std.mem.eql(u8, ident, "var")) {
                    self.skipWhitespace();
                    if (self.readIdentifier()) |name| {
                        return DefinitionInfo{
                            .name = name,
                            .line = def_line,
                            .column = def_col,
                            .kind = .Variable,
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

test "Position: creation" {
    const pos = Position.init(10, 5);
    try std.testing.expectEqual(@as(u32, 10), pos.line);
    try std.testing.expectEqual(@as(u32, 5), pos.character);
}

test "Range: creation" {
    const range = Range.init(
        Position.init(5, 10),
        Position.init(5, 20),
    );
    try std.testing.expectEqual(@as(u32, 5), range.start.line);
    try std.testing.expectEqual(@as(u32, 10), range.start.character);
}

test "Location: creation" {
    var location = try Location.init(
        std.testing.allocator,
        "file:///test.mojo",
        Range.init(Position.init(5, 10), Position.init(5, 20)),
    );
    defer location.deinit();
    
    try std.testing.expectEqualStrings("file:///test.mojo", location.uri);
}

test "SimpleParser: parse function definition" {
    const content = "fn myFunc() {}";
    var parser = SimpleParser.init(content);
    
    const def = try parser.nextDefinition();
    try std.testing.expect(def != null);
    try std.testing.expectEqualStrings("myFunc", def.?.name);
    try std.testing.expectEqual(SymbolKind.Function, def.?.kind);
}

test "SimpleParser: parse struct definition" {
    const content = "struct Point { x: i32 }";
    var parser = SimpleParser.init(content);
    
    const def = try parser.nextDefinition();
    try std.testing.expect(def != null);
    try std.testing.expectEqualStrings("Point", def.?.name);
    try std.testing.expectEqual(SymbolKind.Struct, def.?.kind);
}

test "SimpleParser: parse variable definition" {
    const content = "let x = 42";
    var parser = SimpleParser.init(content);
    
    const def = try parser.nextDefinition();
    try std.testing.expect(def != null);
    try std.testing.expectEqualStrings("x", def.?.name);
    try std.testing.expectEqual(SymbolKind.Variable, def.?.kind);
}

test "DefinitionProvider: find function definition" {
    var provider = DefinitionProvider.init(std.testing.allocator);
    
    const content = "fn myFunc() {}\nlet x = myFunc()";
    const uri = "file:///test.mojo";
    const position = Position.init(1, 10); // On "myFunc" in line 2
    
    var locations = try provider.findDefinition(uri, content, position);
    defer {
        for (locations.items) |*loc| {
            loc.deinit();
        }
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 1), locations.items.len);
}

test "DefinitionProvider: no definition found" {
    var provider = DefinitionProvider.init(std.testing.allocator);
    
    const content = "fn myFunc() {}";
    const uri = "file:///test.mojo";
    const position = Position.init(0, 0); // On "fn" keyword
    
    var locations = try provider.findDefinition(uri, content, position);
    defer locations.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(@as(usize, 0), locations.items.len);
}

test "CrossFileResolver: find across files" {
    var resolver = CrossFileResolver.init(std.testing.allocator);
    
    var file_contents = std.StringHashMap([]const u8).init(std.testing.allocator);
    defer file_contents.deinit();
    
    try file_contents.put("file:///a.mojo", "fn helper() {}");
    try file_contents.put("file:///b.mojo", "fn helper() {}"); // Overload
    
    var locations = try resolver.findAcrossFiles("helper", file_contents);
    defer {
        for (locations.items) |*loc| {
            loc.deinit();
        }
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 2), locations.items.len);
}

test "DefinitionProvider: get word at position" {
    var provider = DefinitionProvider.init(std.testing.allocator);
    
    const content = "fn myFunction() {}";
    const position = Position.init(0, 5); // In "myFunction"
    
    const word = provider.getWordAtPosition(content, position);
    try std.testing.expectEqualStrings("myFunction", word);
}
