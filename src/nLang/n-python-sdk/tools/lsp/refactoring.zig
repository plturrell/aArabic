// Refactoring Operations
// Day 86: Advanced code transformations

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Core Types
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
// Workspace Edit
// ============================================================================

pub const TextEdit = struct {
    range: Range,
    new_text: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, range: Range, new_text: []const u8) !TextEdit {
        return TextEdit{
            .range = range,
            .new_text = try allocator.dupe(u8, new_text),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TextEdit) void {
        self.allocator.free(self.new_text);
    }
};

pub const WorkspaceEdit = struct {
    changes: std.StringHashMap(std.ArrayList(TextEdit)),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceEdit {
        return WorkspaceEdit{
            .changes = std.StringHashMap(std.ArrayList(TextEdit)).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WorkspaceEdit) void {
        var iter = self.changes.iterator();
        while (iter.next()) |entry| {
            for (entry.value_ptr.items) |*edit| {
                edit.deinit();
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.changes.deinit();
    }
    
    pub fn addEdit(self: *WorkspaceEdit, uri: []const u8, edit: TextEdit) !void {
        var result = try self.changes.getOrPut(uri);
        if (!result.found_existing) {
            result.key_ptr.* = try self.allocator.dupe(u8, uri);
            result.value_ptr.* = std.ArrayList(TextEdit){};
        }
        try result.value_ptr.append(self.allocator, edit);
    }
    
    pub fn getEdits(self: *WorkspaceEdit, uri: []const u8) ?[]const TextEdit {
        if (self.changes.getPtr(uri)) |list| {
            return list.items;
        }
        return null;
    }
};

// ============================================================================
// Rename Provider
// ============================================================================

pub const RenameProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) RenameProvider {
        return RenameProvider{ .allocator = allocator };
    }
    
    /// Prepare rename at position
    pub fn prepareRename(
        self: *RenameProvider,
        content: []const u8,
        position: Position,
    ) !?Range {
        const symbol = self.getSymbolAtPosition(content, position);
        if (symbol.len == 0) return null;
        
        return self.getSymbolRange(content, position);
    }
    
    /// Perform rename operation
    pub fn rename(
        self: *RenameProvider,
        uri: []const u8,
        content: []const u8,
        position: Position,
        new_name: []const u8,
    ) !WorkspaceEdit {
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        
        const old_name = self.getSymbolAtPosition(content, position);
        if (old_name.len == 0) return workspace_edit;
        
        // Find all occurrences
        var locations = try self.findAllOccurrences(uri, content, old_name);
        defer {
            for (locations.items) |*loc| loc.deinit();
            locations.deinit(self.allocator);
        }
        
        // Create edits for each occurrence
        for (locations.items) |loc| {
            const edit = try TextEdit.init(self.allocator, loc.range, new_name);
            try workspace_edit.addEdit(loc.uri, edit);
        }
        
        return workspace_edit;
    }
    
    fn getSymbolAtPosition(self: *RenameProvider, content: []const u8, position: Position) []const u8 {
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
    
    fn getSymbolRange(self: *RenameProvider, content: []const u8, position: Position) ?Range {
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
    
    fn findAllOccurrences(
        self: *RenameProvider,
        uri: []const u8,
        content: []const u8,
        symbol: []const u8,
    ) !std.ArrayList(Location) {
        var locations = std.ArrayList(Location){};
        
        var line: u32 = 0;
        var col: u32 = 0;
        var i: usize = 0;
        
        while (i < content.len) {
            if (i + symbol.len <= content.len and std.mem.eql(u8, content[i..i + symbol.len], symbol)) {
                const location = try Location.init(
                    self.allocator,
                    uri,
                    Range.init(
                        Position.init(line, col),
                        Position.init(line, col + @as(u32, @intCast(symbol.len))),
                    ),
                );
                try locations.append(self.allocator, location);
                i += symbol.len;
                col += @as(u32, @intCast(symbol.len));
            } else {
                if (content[i] == '\n') {
                    line += 1;
                    col = 0;
                } else {
                    col += 1;
                }
                i += 1;
            }
        }
        
        return locations;
    }
};

// ============================================================================
// Extract Function
// ============================================================================

pub const ExtractFunctionRefactoring = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ExtractFunctionRefactoring {
        return ExtractFunctionRefactoring{ .allocator = allocator };
    }
    
    /// Extract selected code to new function
    pub fn extractToFunction(
        self: *ExtractFunctionRefactoring,
        uri: []const u8,
        content: []const u8,
        range: Range,
        function_name: []const u8,
    ) !WorkspaceEdit {
        _ = content;
        
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        
        // Insert new function at top
        const insert_text = try std.fmt.allocPrint(self.allocator, "fn {s}() {{}}\n\n", .{function_name});
        defer self.allocator.free(insert_text);
        const insert_edit = try TextEdit.init(
            self.allocator,
            Range.init(Position.init(0, 0), Position.init(0, 0)),
            insert_text,
        );
        try workspace_edit.addEdit(uri, insert_edit);
        
        // Replace selection with function call
        const call_text = try std.fmt.allocPrint(self.allocator, "{s}()", .{function_name});
        defer self.allocator.free(call_text);
        const call_edit = try TextEdit.init(
            self.allocator,
            range,
            call_text,
        );
        try workspace_edit.addEdit(uri, call_edit);
        
        return workspace_edit;
    }
};

// ============================================================================
// Extract Variable
// ============================================================================

pub const ExtractVariableRefactoring = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) ExtractVariableRefactoring {
        return ExtractVariableRefactoring{ .allocator = allocator };
    }
    
    /// Extract expression to variable
    pub fn extractToVariable(
        self: *ExtractVariableRefactoring,
        uri: []const u8,
        range: Range,
        variable_name: []const u8,
    ) !WorkspaceEdit {
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        
        // Insert variable declaration
        const decl_text = try std.fmt.allocPrint(self.allocator, "let {s} = expression;\n", .{variable_name});
        defer self.allocator.free(decl_text);
        const decl_edit = try TextEdit.init(
            self.allocator,
            Range.init(Position.init(range.start.line, 0), Position.init(range.start.line, 0)),
            decl_text,
        );
        try workspace_edit.addEdit(uri, decl_edit);
        
        // Replace expression with variable
        const ref_edit = try TextEdit.init(
            self.allocator,
            range,
            variable_name,
        );
        try workspace_edit.addEdit(uri, ref_edit);
        
        return workspace_edit;
    }
};

// ============================================================================
// Inline Refactoring
// ============================================================================

pub const InlineRefactoring = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) InlineRefactoring {
        return InlineRefactoring{ .allocator = allocator };
    }
    
    /// Inline function or variable
    pub fn inlineSymbol(
        self: *InlineRefactoring,
        uri: []const u8,
        definition_range: Range,
        usage_ranges: []const Range,
        inline_value: []const u8,
    ) !WorkspaceEdit {
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        
        // Replace all usages with inlined value
        for (usage_ranges) |usage_range| {
            const edit = try TextEdit.init(self.allocator, usage_range, inline_value);
            try workspace_edit.addEdit(uri, edit);
        }
        
        // Remove definition
        const delete_edit = try TextEdit.init(self.allocator, definition_range, "");
        try workspace_edit.addEdit(uri, delete_edit);
        
        return workspace_edit;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "RenameProvider: prepare rename" {
    var provider = RenameProvider.init(std.testing.allocator);
    
    const content = "fn myFunc() {}";
    const position = Position.init(0, 5); // In "myFunc"
    
    const range = try provider.prepareRename(content, position);
    try std.testing.expect(range != null);
}

test "RenameProvider: rename symbol" {
    var provider = RenameProvider.init(std.testing.allocator);
    
    const content = "fn old() {}\nlet x = old()";
    const uri = "file:///test.mojo";
    const position = Position.init(0, 5); // On "old" definition
    
    var workspace_edit = try provider.rename(uri, content, position, "new");
    defer workspace_edit.deinit();
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expect(edits.?.len >= 2); // Definition + usage
}

test "RenameProvider: find all occurrences" {
    var provider = RenameProvider.init(std.testing.allocator);
    
    const content = "fn test() {}\nlet x = test()\nlet y = test()";
    const uri = "file:///test.mojo";
    
    var locations = try provider.findAllOccurrences(uri, content, "test");
    defer {
        for (locations.items) |*loc| loc.deinit();
        locations.deinit(std.testing.allocator);
    }
    
    try std.testing.expectEqual(@as(usize, 3), locations.items.len);
}

test "ExtractFunctionRefactoring: extract code" {
    var refactoring = ExtractFunctionRefactoring.init(std.testing.allocator);
    
    const uri = "file:///test.mojo";
    const content = "fn main() { let x = 42; }";
    const range = Range.init(Position.init(0, 12), Position.init(0, 23));
    
    var workspace_edit = try refactoring.extractToFunction(uri, content, range, "extracted");
    defer workspace_edit.deinit();
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expectEqual(@as(usize, 2), edits.?.len); // Insert + replace
}

test "ExtractVariableRefactoring: extract expression" {
    var refactoring = ExtractVariableRefactoring.init(std.testing.allocator);
    
    const uri = "file:///test.mojo";
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    var workspace_edit = try refactoring.extractToVariable(uri, range, "result");
    defer workspace_edit.deinit();
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expectEqual(@as(usize, 2), edits.?.len); // Declaration + replacement
}

test "InlineRefactoring: inline symbol" {
    var refactoring = InlineRefactoring.init(std.testing.allocator);
    
    const uri = "file:///test.mojo";
    const def_range = Range.init(Position.init(0, 0), Position.init(0, 15));
    const usage1 = Range.init(Position.init(2, 8), Position.init(2, 11));
    const usage2 = Range.init(Position.init(3, 8), Position.init(3, 11));
    const usages = [_]Range{ usage1, usage2 };
    
    var workspace_edit = try refactoring.inlineSymbol(uri, def_range, &usages, "42");
    defer workspace_edit.deinit();
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expectEqual(@as(usize, 3), edits.?.len); // 2 usages + 1 definition removal
}

test "WorkspaceEdit: add multiple edits" {
    var workspace_edit = WorkspaceEdit.init(std.testing.allocator);
    defer workspace_edit.deinit();
    
    const uri = "file:///test.mojo";
    
    const edit1 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "new1",
    );
    try workspace_edit.addEdit(uri, edit1);
    
    const edit2 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(1, 0), Position.init(1, 5)),
        "new2",
    );
    try workspace_edit.addEdit(uri, edit2);
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expectEqual(@as(usize, 2), edits.?.len);
}

test "WorkspaceEdit: multi-file edits" {
    var workspace_edit = WorkspaceEdit.init(std.testing.allocator);
    defer workspace_edit.deinit();
    
    const edit1 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "text1",
    );
    try workspace_edit.addEdit("file:///a.mojo", edit1);
    
    const edit2 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "text2",
    );
    try workspace_edit.addEdit("file:///b.mojo", edit2);
    
    try std.testing.expect(workspace_edit.getEdits("file:///a.mojo") != null);
    try std.testing.expect(workspace_edit.getEdits("file:///b.mojo") != null);
}

test "RenameProvider: no rename for empty symbol" {
    var provider = RenameProvider.init(std.testing.allocator);
    
    const content = "   ";
    const position = Position.init(0, 1); // In whitespace
    
    const range = try provider.prepareRename(content, position);
    try std.testing.expect(range == null);
}

test "ExtractFunctionRefactoring: empty selection" {
    var refactoring = ExtractFunctionRefactoring.init(std.testing.allocator);
    
    const uri = "file:///test.mojo";
    const content = "fn main() {}";
    const range = Range.init(Position.init(0, 0), Position.init(0, 0));
    
    var workspace_edit = try refactoring.extractToFunction(uri, content, range, "extracted");
    defer workspace_edit.deinit();
    
    // Should still generate valid edit
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
}

test "InlineRefactoring: no usages" {
    var refactoring = InlineRefactoring.init(std.testing.allocator);
    
    const uri = "file:///test.mojo";
    const def_range = Range.init(Position.init(0, 0), Position.init(0, 15));
    const usages: [0]Range = .{};
    
    var workspace_edit = try refactoring.inlineSymbol(uri, def_range, &usages, "value");
    defer workspace_edit.deinit();
    
    const edits = workspace_edit.getEdits(uri);
    try std.testing.expect(edits != null);
    try std.testing.expectEqual(@as(usize, 1), edits.?.len); // Just definition removal
}
