// Type System Diagnostics
// Day 68: Beautiful error messages and edge case handling

const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const StringHashMap = std.StringHashMap;

// ============================================================================
// Diagnostic Severity
// ============================================================================

pub const DiagnosticSeverity = enum {
    Error,
    Warning,
    Info,
    Hint,
    
    pub fn toString(self: DiagnosticSeverity) []const u8 {
        return switch (self) {
            .Error => "error",
            .Warning => "warning",
            .Info => "info",
            .Hint => "hint",
        };
    }
};

// ============================================================================
// Source Location
// ============================================================================

pub const SourceLocation = struct {
    file: []const u8,
    line: u32,
    column: u32,
    length: u32,
    
    pub fn format(self: SourceLocation, allocator: Allocator) ![]const u8 {
        return try std.fmt.allocPrint(allocator, "{s}:{}:{}", .{ self.file, self.line, self.column });
    }
};

// ============================================================================
// Diagnostic Message
// ============================================================================

pub const Diagnostic = struct {
    severity: DiagnosticSeverity,
    code: []const u8,
    message: []const u8,
    location: SourceLocation,
    notes: ArrayList(Note),
    suggestions: ArrayList(Suggestion),
    
    pub const Note = struct {
        message: []const u8,
        location: ?SourceLocation,
        
        pub fn init(allocator: Allocator, message: []const u8, location: ?SourceLocation) !Note {
            return Note{
                .message = try allocator.dupe(u8, message),
                .location = location,
            };
        }
        
        pub fn deinit(self: *Note, allocator: Allocator) void {
            allocator.free(self.message);
        }
    };
    
    pub const Suggestion = struct {
        message: []const u8,
        replacement: ?[]const u8,
        
        pub fn init(allocator: Allocator, message: []const u8, replacement: ?[]const u8) !Suggestion {
            return Suggestion{
                .message = try allocator.dupe(u8, message),
                .replacement = if (replacement) |r| try allocator.dupe(u8, r) else null,
            };
        }
        
        pub fn deinit(self: *Suggestion, allocator: Allocator) void {
            allocator.free(self.message);
            if (self.replacement) |r| {
                allocator.free(r);
            }
        }
    };
    
    pub fn init(
        allocator: Allocator,
        severity: DiagnosticSeverity,
        code: []const u8,
        message: []const u8,
        location: SourceLocation,
    ) !Diagnostic {
        return Diagnostic{
            .severity = severity,
            .code = try allocator.dupe(u8, code),
            .message = try allocator.dupe(u8, message),
            .location = location,
            .notes = ArrayList(Note){},
            .suggestions = ArrayList(Suggestion){},
        };
    }
    
    pub fn deinit(self: *Diagnostic, allocator: Allocator) void {
        allocator.free(self.code);
        allocator.free(self.message);
        for (self.notes.items) |*note| {
            note.deinit(allocator);
        }
        self.notes.deinit(allocator);
        for (self.suggestions.items) |*suggestion| {
            suggestion.deinit(allocator);
        }
        self.suggestions.deinit(allocator);
    }
    
    pub fn addNote(self: *Diagnostic, allocator: Allocator, note: Note) !void {
        try self.notes.append(allocator, note);
    }
    
    pub fn addSuggestion(self: *Diagnostic, allocator: Allocator, suggestion: Suggestion) !void {
        try self.suggestions.append(allocator, suggestion);
    }
    
    pub fn format(self: *Diagnostic, allocator: Allocator) ![]const u8 {
        var output = ArrayList(u8){};
        
        // Main message
        const loc_str = try self.location.format(allocator);
        defer allocator.free(loc_str);
        
        try output.appendSlice(allocator, loc_str);
        try output.appendSlice(allocator, ": ");
        try output.appendSlice(allocator, self.severity.toString());
        try output.appendSlice(allocator, "[");
        try output.appendSlice(allocator, self.code);
        try output.appendSlice(allocator, "]: ");
        try output.appendSlice(allocator, self.message);
        try output.appendSlice(allocator, "\n");
        
        // Notes
        for (self.notes.items) |note| {
            try output.appendSlice(allocator, "  note: ");
            try output.appendSlice(allocator, note.message);
            try output.appendSlice(allocator, "\n");
        }
        
        // Suggestions
        for (self.suggestions.items) |suggestion| {
            try output.appendSlice(allocator, "  help: ");
            try output.appendSlice(allocator, suggestion.message);
            if (suggestion.replacement) |replacement| {
                try output.appendSlice(allocator, " `");
                try output.appendSlice(allocator, replacement);
                try output.appendSlice(allocator, "`");
            }
            try output.appendSlice(allocator, "\n");
        }
        
        return try output.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Diagnostic Builder
// ============================================================================

pub const DiagnosticBuilder = struct {
    allocator: Allocator,
    diagnostics: ArrayList(Diagnostic),
    
    pub fn init(allocator: Allocator) DiagnosticBuilder {
        return DiagnosticBuilder{
            .allocator = allocator,
            .diagnostics = ArrayList(Diagnostic){},
        };
    }
    
    pub fn deinit(self: *DiagnosticBuilder) void {
        for (self.diagnostics.items) |*diag| {
            diag.deinit(self.allocator);
        }
        self.diagnostics.deinit(self.allocator);
    }
    
    pub fn addDiagnostic(self: *DiagnosticBuilder, diagnostic: Diagnostic) !void {
        try self.diagnostics.append(self.allocator, diagnostic);
    }
    
    pub fn missingProtocol(
        self: *DiagnosticBuilder,
        protocol_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0001",
            try std.fmt.allocPrint(self.allocator, "protocol `{s}` not found", .{protocol_name}),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "protocol must be defined before use",
            null,
        );
        try diag.addNote(self.allocator, note);
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            "consider defining the protocol first",
            null,
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn missingMethod(
        self: *DiagnosticBuilder,
        method_name: []const u8,
        protocol_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0002",
            try std.fmt.allocPrint(
                self.allocator,
                "missing method `{s}` required by protocol `{s}`",
                .{ method_name, protocol_name },
            ),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            try std.fmt.allocPrint(
                self.allocator,
                "method `{s}` is required by the protocol definition",
                .{method_name},
            ),
            null,
        );
        try diag.addNote(self.allocator, note);
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            "add the missing method implementation",
            try std.fmt.allocPrint(self.allocator, "fn {s}(self) {{ ... }}", .{method_name}),
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn duplicateMethod(
        self: *DiagnosticBuilder,
        method_name: []const u8,
        location: SourceLocation,
        previous_location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0003",
            try std.fmt.allocPrint(self.allocator, "duplicate method `{s}`", .{method_name}),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "previous definition here",
            previous_location,
        );
        try diag.addNote(self.allocator, note);
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            "remove the duplicate method or rename it",
            null,
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn missingAssociatedType(
        self: *DiagnosticBuilder,
        type_name: []const u8,
        protocol_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0004",
            try std.fmt.allocPrint(
                self.allocator,
                "missing associated type `{s}` for protocol `{s}`",
                .{ type_name, protocol_name },
            ),
            location,
        );
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            "specify the associated type",
            try std.fmt.allocPrint(self.allocator, "type {s} = YourType", .{type_name}),
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn circularInheritance(
        self: *DiagnosticBuilder,
        protocol_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0005",
            try std.fmt.allocPrint(
                self.allocator,
                "circular protocol inheritance detected for `{s}`",
                .{protocol_name},
            ),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "protocol inheritance must form a directed acyclic graph (DAG)",
            null,
        );
        try diag.addNote(self.allocator, note);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn constraintNotSatisfied(
        self: *DiagnosticBuilder,
        type_name: []const u8,
        constraint: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0006",
            try std.fmt.allocPrint(
                self.allocator,
                "type `{s}` does not satisfy constraint `{s}`",
                .{ type_name, constraint },
            ),
            location,
        );
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            try std.fmt.allocPrint(
                self.allocator,
                "implement `{s}` for `{s}` to satisfy the constraint",
                .{ constraint, type_name },
            ),
            null,
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.addDiagnostic(diag);
    }
    
    pub fn formatAll(self: *DiagnosticBuilder, allocator: Allocator) ![]const u8 {
        var output = ArrayList(u8){};
        
        for (self.diagnostics.items) |*diag| {
            const formatted = try diag.format(allocator);
            defer allocator.free(formatted);
            try output.appendSlice(allocator, formatted);
            try output.appendSlice(allocator, "\n");
        }
        
        return try output.toOwnedSlice(allocator);
    }
};

// ============================================================================
// Edge Case Handler
// ============================================================================

pub const EdgeCaseHandler = struct {
    allocator: Allocator,
    builder: DiagnosticBuilder,
    
    pub fn init(allocator: Allocator) EdgeCaseHandler {
        return EdgeCaseHandler{
            .allocator = allocator,
            .builder = DiagnosticBuilder.init(allocator),
        };
    }
    
    pub fn deinit(self: *EdgeCaseHandler) void {
        self.builder.deinit();
    }
    
    /// Handle empty protocol
    pub fn handleEmptyProtocol(
        self: *EdgeCaseHandler,
        protocol_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Warning,
            "W0001",
            try std.fmt.allocPrint(self.allocator, "protocol `{s}` has no requirements", .{protocol_name}),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "consider adding method or property requirements",
            null,
        );
        try diag.addNote(self.allocator, note);
        
        try self.builder.addDiagnostic(diag);
    }
    
    /// Handle conflicting implementations
    pub fn handleConflictingImpls(
        self: *EdgeCaseHandler,
        protocol_name: []const u8,
        type_name: []const u8,
        location: SourceLocation,
        previous_location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0007",
            try std.fmt.allocPrint(
                self.allocator,
                "conflicting implementations of `{s}` for `{s}`",
                .{ protocol_name, type_name },
            ),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "previous implementation here",
            previous_location,
        );
        try diag.addNote(self.allocator, note);
        
        try self.builder.addDiagnostic(diag);
    }
    
    /// Handle orphan implementation
    pub fn handleOrphanImpl(
        self: *EdgeCaseHandler,
        protocol_name: []const u8,
        type_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0008",
            try std.fmt.allocPrint(
                self.allocator,
                "cannot implement external protocol `{s}` for external type `{s}`",
                .{ protocol_name, type_name },
            ),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "either the protocol or the type must be defined in the current crate",
            null,
        );
        try diag.addNote(self.allocator, note);
        
        try self.builder.addDiagnostic(diag);
    }
    
    /// Handle derive on non-struct
    pub fn handleDeriveOnNonStruct(
        self: *EdgeCaseHandler,
        type_name: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0009",
            try std.fmt.allocPrint(
                self.allocator,
                "cannot derive protocols on non-struct type `{s}`",
                .{type_name},
            ),
            location,
        );
        
        const note = try Diagnostic.Note.init(
            self.allocator,
            "derive macros only work on struct types",
            null,
        );
        try diag.addNote(self.allocator, note);
        
        try self.builder.addDiagnostic(diag);
    }
    
    /// Handle unresolvable constraint
    pub fn handleUnresolvableConstraint(
        self: *EdgeCaseHandler,
        constraint: []const u8,
        location: SourceLocation,
    ) !void {
        var diag = try Diagnostic.init(
            self.allocator,
            .Error,
            "E0010",
            try std.fmt.allocPrint(self.allocator, "cannot resolve constraint `{s}`", .{constraint}),
            location,
        );
        
        const suggestion = try Diagnostic.Suggestion.init(
            self.allocator,
            "check that all required protocols are implemented",
            null,
        );
        try diag.addSuggestion(self.allocator, suggestion);
        
        try self.builder.addDiagnostic(diag);
    }
    
    pub fn getDiagnostics(self: *EdgeCaseHandler) *DiagnosticBuilder {
        return &self.builder;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "diagnostic creation" {
    const allocator = std.testing.allocator;
    
    var diag = try Diagnostic.init(
        allocator,
        .Error,
        "E0001",
        "test error",
        .{ .file = "test.mojo", .line = 1, .column = 1, .length = 5 },
    );
    defer diag.deinit(allocator);
    
    try std.testing.expectEqual(DiagnosticSeverity.Error, diag.severity);
    try std.testing.expectEqualStrings("E0001", diag.code);
}

test "diagnostic with note" {
    const allocator = std.testing.allocator;
    
    var diag = try Diagnostic.init(
        allocator,
        .Error,
        "E0001",
        "test error",
        .{ .file = "test.mojo", .line = 1, .column = 1, .length = 5 },
    );
    defer diag.deinit(allocator);
    
    const note = try Diagnostic.Note.init(allocator, "additional context", null);
    try diag.addNote(allocator, note);
    
    try std.testing.expectEqual(@as(usize, 1), diag.notes.items.len);
}

test "diagnostic with suggestion" {
    const allocator = std.testing.allocator;
    
    var diag = try Diagnostic.init(
        allocator,
        .Error,
        "E0001",
        "test error",
        .{ .file = "test.mojo", .line = 1, .column = 1, .length = 5 },
    );
    defer diag.deinit(allocator);
    
    const suggestion = try Diagnostic.Suggestion.init(allocator, "try this", "fn foo() {}");
    try diag.addSuggestion(allocator, suggestion);
    
    try std.testing.expectEqual(@as(usize, 1), diag.suggestions.items.len);
}

test "diagnostic builder missing protocol" {
    const allocator = std.testing.allocator;
    
    var builder = DiagnosticBuilder.init(allocator);
    defer builder.deinit();
    
    try builder.missingProtocol("Drawable", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
        .length = 8,
    });
    
    try std.testing.expectEqual(@as(usize, 1), builder.diagnostics.items.len);
    try std.testing.expectEqualStrings("E0001", builder.diagnostics.items[0].code);
}

test "diagnostic builder missing method" {
    const allocator = std.testing.allocator;
    
    var builder = DiagnosticBuilder.init(allocator);
    defer builder.deinit();
    
    try builder.missingMethod("draw", "Drawable", .{
        .file = "test.mojo",
        .line = 10,
        .column = 5,
        .length = 10,
    });
    
    try std.testing.expectEqual(@as(usize, 1), builder.diagnostics.items.len);
    try std.testing.expect(builder.diagnostics.items[0].suggestions.items.len > 0);
}

test "diagnostic builder duplicate method" {
    const allocator = std.testing.allocator;
    
    var builder = DiagnosticBuilder.init(allocator);
    defer builder.deinit();
    
    try builder.duplicateMethod(
        "draw",
        .{ .file = "test.mojo", .line = 20, .column = 5, .length = 4 },
        .{ .file = "test.mojo", .line = 10, .column = 5, .length = 4 },
    );
    
    try std.testing.expectEqual(@as(usize, 1), builder.diagnostics.items.len);
    try std.testing.expectEqual(@as(usize, 1), builder.diagnostics.items[0].notes.items.len);
}

test "edge case handler empty protocol" {
    const allocator = std.testing.allocator;
    
    var handler = EdgeCaseHandler.init(allocator);
    defer handler.deinit();
    
    try handler.handleEmptyProtocol("Empty", .{
        .file = "test.mojo",
        .line = 1,
        .column = 1,
        .length = 5,
    });
    
    try std.testing.expectEqual(@as(usize, 1), handler.builder.diagnostics.items.len);
    try std.testing.expectEqual(DiagnosticSeverity.Warning, handler.builder.diagnostics.items[0].severity);
}

test "edge case handler conflicting impls" {
    const allocator = std.testing.allocator;
    
    var handler = EdgeCaseHandler.init(allocator);
    defer handler.deinit();
    
    try handler.handleConflictingImpls(
        "Drawable",
        "Circle",
        .{ .file = "test.mojo", .line = 20, .column = 1, .length = 10 },
        .{ .file = "test.mojo", .line = 10, .column = 1, .length = 10 },
    );
    
    try std.testing.expectEqual(@as(usize, 1), handler.builder.diagnostics.items.len);
}

test "diagnostic formatting" {
    const allocator = std.testing.allocator;
    
    var diag = try Diagnostic.init(
        allocator,
        .Error,
        "E0001",
        "test error message",
        .{ .file = "test.mojo", .line = 1, .column = 5, .length = 4 },
    );
    defer diag.deinit(allocator);
    
    const formatted = try diag.format(allocator);
    defer allocator.free(formatted);
    
    try std.testing.expect(formatted.len > 0);
    try std.testing.expect(std.mem.indexOf(u8, formatted, "error") != null);
}
