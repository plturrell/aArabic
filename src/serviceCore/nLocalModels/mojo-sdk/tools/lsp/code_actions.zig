// Code Actions Infrastructure
// Day 85: Quick fixes, refactorings, and source actions

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

// ============================================================================
// Code Action Kinds
// ============================================================================

pub const CodeActionKind = enum {
    QuickFix,         // Fix errors/warnings
    Refactor,         // Refactoring operations
    RefactorExtract,  // Extract to function/variable
    RefactorInline,   // Inline function/variable
    RefactorRewrite,  // Rewrite code
    Source,           // Source-level actions
    SourceOrganizeImports, // Organize imports
    
    pub fn toString(self: CodeActionKind) []const u8 {
        return switch (self) {
            .QuickFix => "quickfix",
            .Refactor => "refactor",
            .RefactorExtract => "refactor.extract",
            .RefactorInline => "refactor.inline",
            .RefactorRewrite => "refactor.rewrite",
            .Source => "source",
            .SourceOrganizeImports => "source.organizeImports",
        };
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

pub const TextDocumentEdit = struct {
    uri: []const u8,
    edits: std.ArrayList(TextEdit),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, uri: []const u8) !TextDocumentEdit {
        return TextDocumentEdit{
            .uri = try allocator.dupe(u8, uri),
            .edits = std.ArrayList(TextEdit){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TextDocumentEdit) void {
        self.allocator.free(self.uri);
        for (self.edits.items) |*edit| {
            edit.deinit();
        }
        self.edits.deinit(self.allocator);
    }
    
    pub fn addEdit(self: *TextDocumentEdit, edit: TextEdit) !void {
        try self.edits.append(self.allocator, edit);
    }
};

pub const WorkspaceEdit = struct {
    document_changes: std.ArrayList(TextDocumentEdit),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) WorkspaceEdit {
        return WorkspaceEdit{
            .document_changes = std.ArrayList(TextDocumentEdit){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *WorkspaceEdit) void {
        for (self.document_changes.items) |*doc_edit| {
            doc_edit.deinit();
        }
        self.document_changes.deinit(self.allocator);
    }
    
    pub fn addDocumentEdit(self: *WorkspaceEdit, doc_edit: TextDocumentEdit) !void {
        try self.document_changes.append(self.allocator, doc_edit);
    }
};

// ============================================================================
// Code Action
// ============================================================================

pub const CodeAction = struct {
    title: []const u8,
    kind: CodeActionKind,
    edit: ?WorkspaceEdit = null,
    is_preferred: bool = false,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, title: []const u8, kind: CodeActionKind) !CodeAction {
        return CodeAction{
            .title = try allocator.dupe(u8, title),
            .kind = kind,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *CodeAction) void {
        self.allocator.free(self.title);
        if (self.edit) |*edit| {
            edit.deinit();
        }
    }
    
    pub fn withEdit(self: CodeAction, edit: WorkspaceEdit) CodeAction {
        var action = self;
        action.edit = edit;
        return action;
    }
    
    pub fn asPreferred(self: CodeAction) CodeAction {
        var action = self;
        action.is_preferred = true;
        return action;
    }
};

// ============================================================================
// Code Action Context
// ============================================================================

pub const DiagnosticInfo = struct {
    range: Range,
    message: []const u8,
    code: ?[]const u8 = null,
};

pub const CodeActionContext = struct {
    diagnostics: []const DiagnosticInfo,
    only: ?[]const CodeActionKind = null,
    
    pub fn init(diagnostics: []const DiagnosticInfo) CodeActionContext {
        return CodeActionContext{ .diagnostics = diagnostics };
    }
    
    pub fn withOnly(self: CodeActionContext, only: []const CodeActionKind) CodeActionContext {
        var ctx = self;
        ctx.only = only;
        return ctx;
    }
};

// ============================================================================
// Code Action Provider
// ============================================================================

pub const CodeActionProvider = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CodeActionProvider {
        return CodeActionProvider{ .allocator = allocator };
    }
    
    /// Provide code actions for a range
    pub fn provideCodeActions(
        self: *CodeActionProvider,
        uri: []const u8,
        content: []const u8,
        range: Range,
        context: CodeActionContext,
    ) !std.ArrayList(CodeAction) {
        var actions = std.ArrayList(CodeAction){};
        
        // Generate quick fixes for diagnostics
        for (context.diagnostics) |diag| {
            if (try self.generateQuickFix(uri, content, diag)) |action| {
                try actions.append(self.allocator, action);
            }
        }
        
        // Generate refactoring actions
        if (context.only == null or self.hasKind(context.only.?, .Refactor)) {
            if (try self.generateExtractFunction(uri, content, range)) |action| {
                try actions.append(self.allocator, action);
            }
        }
        
        // Generate source actions
        if (context.only == null or self.hasKind(context.only.?, .Source)) {
            if (try self.generateOrganizeImports(uri, content)) |action| {
                try actions.append(self.allocator, action);
            }
        }
        
        return actions;
    }
    
    /// Check if kind list contains a kind
    fn hasKind(self: *CodeActionProvider, kinds: []const CodeActionKind, target: CodeActionKind) bool {
        _ = self;
        for (kinds) |k| {
            if (k == target) return true;
        }
        return false;
    }
    
    /// Generate quick fix for diagnostic
    fn generateQuickFix(
        self: *CodeActionProvider,
        uri: []const u8,
        content: []const u8,
        diagnostic: DiagnosticInfo,
    ) !?CodeAction {
        _ = content;
        
        // Example: Fix unterminated string
        if (std.mem.indexOf(u8, diagnostic.message, "Unterminated") != null) {
            var action = try CodeAction.init(
                self.allocator,
                "Add closing quote",
                .QuickFix,
            );
            
            // Create workspace edit
            var workspace_edit = WorkspaceEdit.init(self.allocator);
            var doc_edit = try TextDocumentEdit.init(self.allocator, uri);
            
            const edit = try TextEdit.init(
                self.allocator,
                Range.init(diagnostic.range.end, diagnostic.range.end),
                "\"",
            );
            try doc_edit.addEdit(edit);
            try workspace_edit.addDocumentEdit(doc_edit);
            
            action.edit = workspace_edit;
            action.is_preferred = true;
            
            return action;
        }
        
        return null;
    }
    
    /// Generate extract function refactoring
    fn generateExtractFunction(
        self: *CodeActionProvider,
        uri: []const u8,
        content: []const u8,
        range: Range,
    ) !?CodeAction {
        _ = content;
        _ = range;
        
        var action = try CodeAction.init(
            self.allocator,
            "Extract to function",
            .RefactorExtract,
        );
        
        // Create workspace edit (simplified)
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        var doc_edit = try TextDocumentEdit.init(self.allocator, uri);
        
        const edit = try TextEdit.init(
            self.allocator,
            Range.init(Position.init(0, 0), Position.init(0, 0)),
            "fn extracted() {}\n",
        );
        try doc_edit.addEdit(edit);
        try workspace_edit.addDocumentEdit(doc_edit);
        
        action.edit = workspace_edit;
        
        return action;
    }
    
    /// Generate organize imports action
    fn generateOrganizeImports(
        self: *CodeActionProvider,
        uri: []const u8,
        content: []const u8,
    ) !?CodeAction {
        _ = content;
        
        var action = try CodeAction.init(
            self.allocator,
            "Organize imports",
            .SourceOrganizeImports,
        );
        
        // Create workspace edit
        var workspace_edit = WorkspaceEdit.init(self.allocator);
        var doc_edit = try TextDocumentEdit.init(self.allocator, uri);
        
        const edit = try TextEdit.init(
            self.allocator,
            Range.init(Position.init(0, 0), Position.init(0, 0)),
            "// Organized imports\n",
        );
        try doc_edit.addEdit(edit);
        try workspace_edit.addDocumentEdit(doc_edit);
        
        action.edit = workspace_edit;
        
        return action;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "CodeActionKind: toString" {
    try std.testing.expectEqualStrings("quickfix", CodeActionKind.QuickFix.toString());
    try std.testing.expectEqualStrings("refactor.extract", CodeActionKind.RefactorExtract.toString());
    try std.testing.expectEqualStrings("source.organizeImports", CodeActionKind.SourceOrganizeImports.toString());
}

test "TextEdit: creation" {
    var edit = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "new text",
    );
    defer edit.deinit();
    
    try std.testing.expectEqualStrings("new text", edit.new_text);
}

test "WorkspaceEdit: multiple edits" {
    var workspace_edit = WorkspaceEdit.init(std.testing.allocator);
    defer workspace_edit.deinit();
    
    var doc_edit = try TextDocumentEdit.init(std.testing.allocator, "file:///test.mojo");
    const edit1 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "text1",
    );
    try doc_edit.addEdit(edit1);
    
    try workspace_edit.addDocumentEdit(doc_edit);
    
    try std.testing.expectEqual(@as(usize, 1), workspace_edit.document_changes.items.len);
}

test "CodeAction: basic creation" {
    var action = try CodeAction.init(std.testing.allocator, "Fix error", .QuickFix);
    defer action.deinit();
    
    try std.testing.expectEqualStrings("Fix error", action.title);
    try std.testing.expectEqual(CodeActionKind.QuickFix, action.kind);
    try std.testing.expect(!action.is_preferred);
}

test "CodeAction: with preferred flag" {
    var action = try CodeAction.init(std.testing.allocator, "Fix error", .QuickFix);
    defer action.deinit();
    
    action.is_preferred = true;
    try std.testing.expect(action.is_preferred);
}

test "CodeActionContext: creation" {
    const diag = DiagnosticInfo{
        .range = Range.init(Position.init(0, 0), Position.init(0, 5)),
        .message = "Test error",
    };
    const diagnostics = [_]DiagnosticInfo{diag};
    
    const context = CodeActionContext.init(&diagnostics);
    try std.testing.expectEqual(@as(usize, 1), context.diagnostics.len);
}

test "CodeActionProvider: generate quick fix" {
    var provider = CodeActionProvider.init(std.testing.allocator);
    
    const diag = DiagnosticInfo{
        .range = Range.init(Position.init(0, 0), Position.init(0, 10)),
        .message = "Unterminated string",
    };
    const diagnostics = [_]DiagnosticInfo{diag};
    const context = CodeActionContext.init(&diagnostics);
    
    var actions = try provider.provideCodeActions(
        "file:///test.mojo",
        "let x = \"test",
        Range.init(Position.init(0, 0), Position.init(0, 10)),
        context,
    );
    defer {
        for (actions.items) |*action| {
            action.deinit();
        }
        actions.deinit(std.testing.allocator);
    }
    
    try std.testing.expect(actions.items.len > 0);
}

test "CodeActionProvider: generate refactoring" {
    var provider = CodeActionProvider.init(std.testing.allocator);
    
    const context = CodeActionContext.init(&[_]DiagnosticInfo{});
    
    var actions = try provider.provideCodeActions(
        "file:///test.mojo",
        "fn test() { let x = 42; }",
        Range.init(Position.init(0, 12), Position.init(0, 23)),
        context,
    );
    defer {
        for (actions.items) |*action| {
            action.deinit();
        }
        actions.deinit(std.testing.allocator);
    }
    
    try std.testing.expect(actions.items.len > 0);
}

test "CodeActionProvider: filter by kind" {
    var provider = CodeActionProvider.init(std.testing.allocator);
    
    const kinds = [_]CodeActionKind{.Refactor};
    const context = CodeActionContext.init(&[_]DiagnosticInfo{}).withOnly(&kinds);
    
    var actions = try provider.provideCodeActions(
        "file:///test.mojo",
        "fn test() {}",
        Range.init(Position.init(0, 0), Position.init(0, 10)),
        context,
    );
    defer {
        for (actions.items) |*action| {
            action.deinit();
        }
        actions.deinit(std.testing.allocator);
    }
    
    // Should only have refactoring actions
    for (actions.items) |action| {
        try std.testing.expect(action.kind == .RefactorExtract or action.kind == .Refactor);
    }
}

test "WorkspaceEdit: complex edit chain" {
    var workspace_edit = WorkspaceEdit.init(std.testing.allocator);
    defer workspace_edit.deinit();
    
    // First document
    var doc_edit1 = try TextDocumentEdit.init(std.testing.allocator, "file:///a.mojo");
    const edit1 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "new1",
    );
    try doc_edit1.addEdit(edit1);
    try workspace_edit.addDocumentEdit(doc_edit1);
    
    // Second document
    var doc_edit2 = try TextDocumentEdit.init(std.testing.allocator, "file:///b.mojo");
    const edit2 = try TextEdit.init(
        std.testing.allocator,
        Range.init(Position.init(1, 0), Position.init(1, 5)),
        "new2",
    );
    try doc_edit2.addEdit(edit2);
    try workspace_edit.addDocumentEdit(doc_edit2);
    
    try std.testing.expectEqual(@as(usize, 2), workspace_edit.document_changes.items.len);
}
