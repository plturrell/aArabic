// Diagnostics Engine
// Day 75: Real-time error reporting for LSP

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Diagnostic Types
// ============================================================================

/// Diagnostic Severity (LSP compliant)
pub const DiagnosticSeverity = enum(u8) {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
    
    pub fn toString(self: DiagnosticSeverity) []const u8 {
        return switch (self) {
            .Error => "Error",
            .Warning => "Warning",
            .Information => "Information",
            .Hint => "Hint",
        };
    }
};

/// Diagnostic Tag (LSP 3.15+)
pub const DiagnosticTag = enum(u8) {
    Unnecessary = 1,  // Code is unnecessary (e.g., unused variable)
    Deprecated = 2,   // Code is deprecated
    
    pub fn toString(self: DiagnosticTag) []const u8 {
        return switch (self) {
            .Unnecessary => "Unnecessary",
            .Deprecated => "Deprecated",
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

/// Location in a document
pub const Location = struct {
    uri: []const u8,
    range: Range,
    
    pub fn init(uri: []const u8, range: Range) Location {
        return Location{ .uri = uri, .range = range };
    }
};

/// Related Information for a diagnostic
pub const DiagnosticRelatedInformation = struct {
    location: Location,
    message: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, location: Location, message: []const u8) !DiagnosticRelatedInformation {
        const uri_copy = try allocator.dupe(u8, location.uri);
        const msg_copy = try allocator.dupe(u8, message);
        
        return DiagnosticRelatedInformation{
            .location = Location.init(uri_copy, location.range),
            .message = msg_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DiagnosticRelatedInformation) void {
        self.allocator.free(self.location.uri);
        self.allocator.free(self.message);
    }
};

/// Code Action Kind for quick fixes
pub const CodeActionKind = enum {
    QuickFix,
    Refactor,
    RefactorExtract,
    RefactorInline,
    RefactorRewrite,
    Source,
    SourceOrganizeImports,
    
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

/// Quick Fix suggestion
pub const QuickFix = struct {
    title: []const u8,
    kind: CodeActionKind,
    range: Range,
    new_text: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, title: []const u8, kind: CodeActionKind, range: Range, new_text: []const u8) !QuickFix {
        const title_copy = try allocator.dupe(u8, title);
        const text_copy = try allocator.dupe(u8, new_text);
        
        return QuickFix{
            .title = title_copy,
            .kind = kind,
            .range = range,
            .new_text = text_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *QuickFix) void {
        self.allocator.free(self.title);
        self.allocator.free(self.new_text);
    }
};

// ============================================================================
// Diagnostic
// ============================================================================

pub const Diagnostic = struct {
    range: Range,
    severity: DiagnosticSeverity,
    code: ?[]const u8 = null,
    source: ?[]const u8 = null,
    message: []const u8,
    tags: std.ArrayList(DiagnosticTag),
    related_information: std.ArrayList(DiagnosticRelatedInformation),
    quick_fixes: std.ArrayList(QuickFix),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, range: Range, severity: DiagnosticSeverity, message: []const u8) !Diagnostic {
        const msg_copy = try allocator.dupe(u8, message);
        
        return Diagnostic{
            .range = range,
            .severity = severity,
            .message = msg_copy,
            .tags = std.ArrayList(DiagnosticTag){},
            .related_information = std.ArrayList(DiagnosticRelatedInformation){},
            .quick_fixes = std.ArrayList(QuickFix){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Diagnostic) void {
        self.allocator.free(self.message);
        if (self.code) |code| {
            self.allocator.free(code);
        }
        if (self.source) |source| {
            self.allocator.free(source);
        }
        
        self.tags.deinit(self.allocator);
        
        for (self.related_information.items) |*info| {
            info.deinit();
        }
        self.related_information.deinit(self.allocator);
        
        for (self.quick_fixes.items) |*fix| {
            fix.deinit();
        }
        self.quick_fixes.deinit(self.allocator);
    }
    
    /// Set diagnostic code
    pub fn withCode(self: *Diagnostic, code: []const u8) !void {
        self.code = try self.allocator.dupe(u8, code);
    }
    
    /// Set diagnostic source
    pub fn withSource(self: *Diagnostic, source: []const u8) !void {
        self.source = try self.allocator.dupe(u8, source);
    }
    
    /// Add a tag
    pub fn addTag(self: *Diagnostic, tag: DiagnosticTag) !void {
        try self.tags.append(self.allocator, tag);
    }
    
    /// Add related information
    pub fn addRelatedInfo(self: *Diagnostic, info: DiagnosticRelatedInformation) !void {
        try self.related_information.append(self.allocator, info);
    }
    
    /// Add a quick fix
    pub fn addQuickFix(self: *Diagnostic, fix: QuickFix) !void {
        try self.quick_fixes.append(self.allocator, fix);
    }
};

// ============================================================================
// Diagnostics Collector
// ============================================================================

pub const DiagnosticsCollector = struct {
    // URI -> list of diagnostics
    diagnostics: std.StringHashMap(std.ArrayList(Diagnostic)),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DiagnosticsCollector {
        return DiagnosticsCollector{
            .diagnostics = std.StringHashMap(std.ArrayList(Diagnostic)).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DiagnosticsCollector) void {
        var iter = self.diagnostics.iterator();
        while (iter.next()) |entry| {
            for (entry.value_ptr.items) |*diag| {
                diag.deinit();
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.diagnostics.deinit();
    }
    
    /// Add a diagnostic for a file
    pub fn addDiagnostic(self: *DiagnosticsCollector, uri: []const u8, diagnostic: Diagnostic) !void {
        if (self.diagnostics.getPtr(uri)) |list| {
            try list.append(self.allocator, diagnostic);
        } else {
            var list = std.ArrayList(Diagnostic){};
            try list.append(self.allocator, diagnostic);
            const uri_key = try self.allocator.dupe(u8, uri);
            try self.diagnostics.put(uri_key, list);
        }
    }
    
    /// Get diagnostics for a file
    pub fn getDiagnostics(self: *DiagnosticsCollector, uri: []const u8) ?[]const Diagnostic {
        if (self.diagnostics.getPtr(uri)) |list| {
            return list.items;
        }
        return null;
    }
    
    /// Clear diagnostics for a file
    pub fn clearDiagnostics(self: *DiagnosticsCollector, uri: []const u8) !void {
        if (self.diagnostics.fetchRemove(uri)) |kv| {
            for (kv.value.items) |*diag| {
                diag.deinit();
            }
            kv.value.deinit(self.allocator);
            self.allocator.free(kv.key);
        }
    }
    
    /// Clear all diagnostics
    pub fn clearAll(self: *DiagnosticsCollector) void {
        var iter = self.diagnostics.iterator();
        while (iter.next()) |entry| {
            for (entry.value_ptr.items) |*diag| {
                diag.deinit();
            }
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.diagnostics.clearRetainingCapacity();
    }
    
    /// Get diagnostic count for a file
    pub fn getCount(self: *DiagnosticsCollector, uri: []const u8) usize {
        if (self.diagnostics.getPtr(uri)) |list| {
            return list.items.len;
        }
        return 0;
    }
    
    /// Get total diagnostic count
    pub fn getTotalCount(self: *DiagnosticsCollector) usize {
        var count: usize = 0;
        var iter = self.diagnostics.iterator();
        while (iter.next()) |entry| {
            count += entry.value_ptr.items.len;
        }
        return count;
    }
};

// ============================================================================
// Compiler Integration (Simulated)
// ============================================================================

pub const CompilerError = struct {
    line: u32,
    column: u32,
    length: u32,
    message: []const u8,
    error_code: []const u8,
    is_warning: bool = false,
};

pub const CompilerIntegration = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CompilerIntegration {
        return CompilerIntegration{
            .allocator = allocator,
        };
    }
    
    /// Analyze source code and collect diagnostics
    pub fn analyze(self: *CompilerIntegration, uri: []const u8, content: []const u8, collector: *DiagnosticsCollector) !void {
        // Simple analysis: look for common errors
        var line: u32 = 0;
        var column: u32 = 0;
        var in_string = false;
        
        for (content, 0..) |c, i| {
            if (c == '"' and (i == 0 or content[i - 1] != '\\')) {
                in_string = !in_string;
            }
            
            if (c == '\n') {
                if (in_string) {
                    // Unterminated string error
                    const range = Range.init(
                        Position.init(line, column),
                        Position.init(line, column + 1),
                    );
                    
                    var diag = try Diagnostic.init(
                        self.allocator,
                        range,
                        .Error,
                        "Unterminated string literal",
                    );
                    try diag.withCode("E001");
                    try diag.withSource("mojo-compiler");
                    
                    // Add quick fix
                    const fix = try QuickFix.init(
                        self.allocator,
                        "Add closing quote",
                        .QuickFix,
                        range,
                        "\"",
                    );
                    try diag.addQuickFix(fix);
                    
                    try collector.addDiagnostic(uri, diag);
                    in_string = false;
                }
                line += 1;
                column = 0;
            } else {
                column += 1;
            }
        }
    }
};

// ============================================================================
// Tests
// ============================================================================

test "DiagnosticSeverity: toString" {
    try std.testing.expectEqualStrings("Error", DiagnosticSeverity.Error.toString());
    try std.testing.expectEqualStrings("Warning", DiagnosticSeverity.Warning.toString());
    try std.testing.expectEqualStrings("Hint", DiagnosticSeverity.Hint.toString());
}

test "DiagnosticTag: toString" {
    try std.testing.expectEqualStrings("Unnecessary", DiagnosticTag.Unnecessary.toString());
    try std.testing.expectEqualStrings("Deprecated", DiagnosticTag.Deprecated.toString());
}

test "DiagnosticRelatedInformation: creation" {
    const location = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(5, 10), Position.init(5, 20)),
    );
    
    var info = try DiagnosticRelatedInformation.init(
        std.testing.allocator,
        location,
        "See related declaration here",
    );
    defer info.deinit();
    
    try std.testing.expectEqualStrings("See related declaration here", info.message);
}

test "QuickFix: creation" {
    const range = Range.init(Position.init(10, 5), Position.init(10, 15));
    
    var fix = try QuickFix.init(
        std.testing.allocator,
        "Remove unused variable",
        .QuickFix,
        range,
        "",
    );
    defer fix.deinit();
    
    try std.testing.expectEqualStrings("Remove unused variable", fix.title);
    try std.testing.expectEqual(CodeActionKind.QuickFix, fix.kind);
}

test "Diagnostic: creation and metadata" {
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    var diag = try Diagnostic.init(
        std.testing.allocator,
        range,
        .Error,
        "Undefined variable 'x'",
    );
    defer diag.deinit();
    
    try diag.withCode("E101");
    try diag.withSource("mojo-compiler");
    
    try std.testing.expectEqualStrings("Undefined variable 'x'", diag.message);
    try std.testing.expectEqual(DiagnosticSeverity.Error, diag.severity);
    try std.testing.expectEqualStrings("E101", diag.code.?);
    try std.testing.expectEqualStrings("mojo-compiler", diag.source.?);
}

test "Diagnostic: tags" {
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    var diag = try Diagnostic.init(
        std.testing.allocator,
        range,
        .Warning,
        "Unused variable 'x'",
    );
    defer diag.deinit();
    
    try diag.addTag(.Unnecessary);
    
    try std.testing.expectEqual(@as(usize, 1), diag.tags.items.len);
    try std.testing.expectEqual(DiagnosticTag.Unnecessary, diag.tags.items[0]);
}

test "Diagnostic: related information" {
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    var diag = try Diagnostic.init(
        std.testing.allocator,
        range,
        .Error,
        "Variable redeclared",
    );
    defer diag.deinit();
    
    const related_loc = Location.init(
        "file:///test.mojo",
        Range.init(Position.init(3, 5), Position.init(3, 10)),
    );
    
    const info = try DiagnosticRelatedInformation.init(
        std.testing.allocator,
        related_loc,
        "First declared here",
    );
    
    try diag.addRelatedInfo(info);
    
    try std.testing.expectEqual(@as(usize, 1), diag.related_information.items.len);
}

test "Diagnostic: quick fixes" {
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    var diag = try Diagnostic.init(
        std.testing.allocator,
        range,
        .Error,
        "Missing semicolon",
    );
    defer diag.deinit();
    
    const fix = try QuickFix.init(
        std.testing.allocator,
        "Add semicolon",
        .QuickFix,
        range,
        ";",
    );
    
    try diag.addQuickFix(fix);
    
    try std.testing.expectEqual(@as(usize, 1), diag.quick_fixes.items.len);
    try std.testing.expectEqualStrings("Add semicolon", diag.quick_fixes.items[0].title);
}

test "DiagnosticsCollector: add and get" {
    var collector = DiagnosticsCollector.init(std.testing.allocator);
    defer collector.deinit();
    
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    const diag = try Diagnostic.init(
        std.testing.allocator,
        range,
        .Error,
        "Test error",
    );
    
    try collector.addDiagnostic("file:///test.mojo", diag);
    
    const diagnostics = collector.getDiagnostics("file:///test.mojo");
    try std.testing.expect(diagnostics != null);
    try std.testing.expectEqual(@as(usize, 1), diagnostics.?.len);
}

test "DiagnosticsCollector: counts" {
    var collector = DiagnosticsCollector.init(std.testing.allocator);
    defer collector.deinit();
    
    const range = Range.init(Position.init(5, 10), Position.init(5, 20));
    
    const diag1 = try Diagnostic.init(std.testing.allocator, range, .Error, "Error 1");
    try collector.addDiagnostic("file:///test1.mojo", diag1);
    
    const diag2 = try Diagnostic.init(std.testing.allocator, range, .Warning, "Warning 1");
    try collector.addDiagnostic("file:///test1.mojo", diag2);
    
    const diag3 = try Diagnostic.init(std.testing.allocator, range, .Error, "Error 2");
    try collector.addDiagnostic("file:///test2.mojo", diag3);
    
    try std.testing.expectEqual(@as(usize, 2), collector.getCount("file:///test1.mojo"));
    try std.testing.expectEqual(@as(usize, 1), collector.getCount("file:///test2.mojo"));
    try std.testing.expectEqual(@as(usize, 3), collector.getTotalCount());
}

test "CompilerIntegration: analyze unterminated string" {
    var compiler = CompilerIntegration.init(std.testing.allocator);
    var collector = DiagnosticsCollector.init(std.testing.allocator);
    defer collector.deinit();
    
    const content = "let x = \"unterminated\n";
    
    try compiler.analyze("file:///test.mojo", content, &collector);
    
    const diagnostics = collector.getDiagnostics("file:///test.mojo");
    try std.testing.expect(diagnostics != null);
    try std.testing.expectEqual(@as(usize, 1), diagnostics.?.len);
    try std.testing.expectEqual(DiagnosticSeverity.Error, diagnostics.?[0].severity);
}
