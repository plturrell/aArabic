// Find All References
// Day 81: Locate all symbol usages in workspace

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Location Types
// ============================================================================

/// Position in document
pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
};

/// Range in document
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
// Reference Context
// ============================================================================

/// Reference context (read vs write)
pub const ReferenceKind = enum {
    Read,    // Variable read, function call
    Write,   // Variable assignment
    Declaration, // Symbol definition
};

/// Reference with context
pub const Reference = struct {
    location: Location,
    kind: ReferenceKind,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, location: Location, kind: ReferenceKind) Reference {
        return Reference{
            .location = location,
            .kind = kind,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Reference) void {
        self.location.deinit();
    }
};

// ============================================================================
// Reference Options
// ============================================================================

pub const ReferenceOptions = struct {
    include_declaration: bool = true,
    
    pub fn init(include_declaration: bool) ReferenceOptions {
        return ReferenceOptions{ .include_declaration = include_declaration };
    }
};

// ============================================================================
// Reference Provider
// ============================================================================

pub const ReferenceProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ReferenceProvider {
        return ReferenceProvider{ .allocator = allocator };
    }
    
    /// Find all references to symbol at position
    pub fn findReferences(
        self: *ReferenceProvider,
        uri: []const u8,
        content: []const u8,
        position: Position,
        options: ReferenceOptions,
    ) !std.ArrayList(Location) {
        var locations = std.ArrayList(Location){};
        
        // Get symbol name at cursor
        const symbol_name = self.getSymbolAtPosition(content, position);
        if (symbol_name.len == 0) return locations;
        
        // Find all usages in file
        var parser = ReferenceParser.init(content);
        while (try parser.nextReference(symbol_name)) |ref_info| {
            // Skip declaration if not wanted
            if (!options.include_declaration and ref_info.is_definition) {
                continue;
            }
            
            const location = try Location.init(
                self.allocator,
                uri,
                Range.init(
                    Position.init(ref_info.line, ref_info.column),
                    Position.init(ref_info.line, ref_info.column + @as(u32, @intCast(symbol_name.len))),
                ),
            );
            try locations.append(self.allocator, location);
        }
        
        return locations;
    }
    
    /// Get symbol name at position
    fn getSymbolAtPosition(self: *ReferenceProvider, content: []const u8, position: Position) []const u8 {
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
// Workspace Reference Finder
// ============================================================================

pub const WorkspaceReferenceFinder = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceReferenceFinder {
        return WorkspaceReferenceFinder{ .allocator = allocator };
    }
    
    /// Find references across entire workspace
    pub fn findInWorkspace(
        self: *WorkspaceReferenceFinder,
        symbol_name: []const u8,
        workspace_files: std.StringHashMap([]const u8),
        options: ReferenceOptions,
    ) !std.ArrayList(Location) {
        var all_locations = std.ArrayList(Location){};
        
        var iter = workspace_files.iterator();
        while (iter.next()) |entry| {
            const uri = entry.key_ptr.*;
            const content = entry.value_ptr.*;
            
            var parser = ReferenceParser.init(content);
            while (try parser.nextReference(symbol_name)) |ref_info| {
                // Skip declaration if not wanted
                if (!options.include_declaration and ref_info.is_definition) {
                    continue;
                }
                
                const location = try Location.init(
                    self.allocator,
                    uri,
                    Range.init(
                        Position.init(ref_info.line, ref_info.column),
                        Position.init(ref_info.line, ref_info.column + @as(u32, @intCast(symbol_name.len))),
                    ),
                );
                try all_locations.append(self.allocator, location);
            }
        }
        
        return all_locations;
    }
};

// ============================================================================
// Reference Parser
// ============================================================================

const ReferenceInfo = struct {
    line: u32,
    column: u32,
    is_definition: bool,
};

pub const ReferenceParser = struct {
    content: []const u8,
    position: usize,
    line: u32,
    column: u32,
    
    pub fn init(content: []const u8) ReferenceParser {
        return ReferenceParser{
            .content = content,
            .position = 0,
            .line = 0,
            .column = 0,
        };
    }
    
    fn peek(self: *ReferenceParser) ?u8 {
        if (self.position >= self.content.len) return null;
        return self.content[self.position];
    }
    
    fn advance(self: *ReferenceParser) ?u8 {
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
    
    fn skipWhitespace(self: *ReferenceParser) void {
        while (self.peek()) |ch| {
            if (ch == ' ' or ch == '\t' or ch == '\n' or ch == '\r') {
                _ = self.advance();
            } else {
                break;
            }
        }
    }
    
    fn readIdentifier(self: *ReferenceParser) ?[]const u8 {
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
    
    fn peekPreviousWord(self: *ReferenceParser, start_pos: usize) ?[]const u8 {
        if (start_pos == 0) return null;
        
        var pos = start_pos - 1;
        // Skip whitespace backwards
        while (pos > 0 and (self.content[pos] == ' ' or self.content[pos] == '\t')) {
            pos -= 1;
        }
        
        // Read word backwards
        const word_end = pos + 1;
        while (pos > 0 and (std.ascii.isAlphanumeric(self.content[pos]) or self.content[pos] == '_')) {
            pos -= 1;
        }
        var word_start = pos;
        if (!std.ascii.isAlphanumeric(self.content[pos]) and self.content[pos] != '_') {
            word_start += 1;
        }
        
        if (word_start < word_end) {
            return self.content[word_start..word_end];
        }
        return null;
    }
    
    /// Find next reference to symbol
    pub fn nextReference(self: *ReferenceParser, target_symbol: []const u8) !?ReferenceInfo {
        while (self.position < self.content.len) {
            const ref_line = self.line;
            const ref_col = self.column;
            const before_pos = self.position;
            
            if (self.readIdentifier()) |ident| {
                if (std.mem.eql(u8, ident, target_symbol)) {
                    // Check if this is a definition
                    const prev_word = self.peekPreviousWord(before_pos);
                    const is_def = if (prev_word) |word|
                        std.mem.eql(u8, word, "fn") or
                        std.mem.eql(u8, word, "def") or
                        std.mem.eql(u8, word, "struct") or
                        std.mem.eql(u8, word, "class") or
                        std.mem.eql(u8, word, "let") or
                        std.mem.eql(u8, word, "var")
                    else
                        false;
                    
                    return ReferenceInfo{
                        .line = ref_line,
                        .column = ref_col,
                        .is_definition = is_def,
                    };
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
    const pos = Position.init(5, 10);
    try std.testing.expectEqual(@as(u32, 5), pos.line);
    try std.testing.expectEqual(@as(u32, 10), pos.character);
}

test "Location: creation" {
    var loc = try Location.init(
        std.testing.allocator,
        "file:///test.mojo",
        Range.init(Position.init(1, 2), Position.init(1, 10)),
    );
    defer loc.deinit();
    
    try std.testing.expectEqualStrings("file:///test.mojo", loc.uri);
}

test "ReferenceOptions: with declaration" {
    const opts = ReferenceOptions.init(true);
    try std.testing.expect(opts.include_declaration);
}

test "ReferenceOptions: without declaration" {
    const opts = ReferenceOptions.init(false);
    try std.testing.expect(!opts.include_declaration);
}

test "ReferenceParser: find usages" {
    const content = "fn myFunc() {}\nlet x = myFunc()\nlet y = myFunc()";
    var parser = ReferenceParser.init(content);
    
    var count: usize = 0;
    while (try parser.nextReference("myFunc")) |_| {
        count += 1;
    }
    
    try std.testing.expectEqual(@as(usize, 3), count); // 1 def + 2 usages
}

test "ReferenceParser: detect definition" {
    const content = "fn myFunc() {}";
    var parser = ReferenceParser.init(content);
    
    const ref = try parser.nextReference("myFunc");
    try std.testing.expect(ref != null);
    try std.testing.expect(ref.?.is_definition);
}

test "ReferenceParser: detect usage" {
    const content = "let x = myFunc()";
    var parser = ReferenceParser.init(content);
    
    const ref = try parser.nextReference("myFunc");
    try std.testing.expect(ref != null);
    try std.testing.expect(!ref.?.is_definition);
}

test "ReferenceProvider: find in file" {
    var provider = ReferenceProvider.init(std.testing.allocator);
    
    const content = "fn helper() {}\nlet x = helper()\nlet y = helper()";
    const uri = "file:///test.mojo";
    const position = Position.init(0, 5); // On definition
    const options = ReferenceOptions.init(true);
    
    var locations = try provider.findReferences(uri, content, position, options);
    defer {
        for (locations.items) |*loc| loc.deinit();
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 3), locations.items.len);
}

test "ReferenceProvider: exclude declaration" {
    var provider = ReferenceProvider.init(std.testing.allocator);
    
    const content = "fn helper() {}\nlet x = helper()";
    const uri = "file:///test.mojo";
    const position = Position.init(0, 5);
    const options = ReferenceOptions.init(false); // Exclude declaration
    
    var locations = try provider.findReferences(uri, content, position, options);
    defer {
        for (locations.items) |*loc| loc.deinit();
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 1), locations.items.len); // Only usage
}

test "WorkspaceReferenceFinder: find across files" {
    var finder = WorkspaceReferenceFinder.init(std.testing.allocator);
    
    var files = std.StringHashMap([]const u8).init(std.testing.allocator);
    defer files.deinit();
    
    try files.put("file:///a.mojo", "fn helper() {}");
    try files.put("file:///b.mojo", "let x = helper()");
    try files.put("file:///c.mojo", "let y = helper()");
    
    const options = ReferenceOptions.init(true);
    var locations = try finder.findInWorkspace("helper", files, options);
    defer {
        for (locations.items) |*loc| loc.deinit();
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 3), locations.items.len);
}
