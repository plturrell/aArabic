// LSP Server Foundation
// Day 113: Basic LSP server implementation with document synchronization and diagnostics

const std = @import("std");
const json = std.json;
const jsonrpc = @import("jsonrpc.zig");
const Allocator = std.mem.Allocator;

// ============================================================================
// LSP Protocol Types
// ============================================================================

/// LSP Position (0-indexed line and character)
pub const Position = struct {
    line: u32,
    character: u32,
    
    pub fn init(line: u32, character: u32) Position {
        return Position{ .line = line, .character = character };
    }
    
    pub fn eql(self: Position, other: Position) bool {
        return self.line == other.line and self.character == other.character;
    }
};

/// LSP Range (start and end positions)
pub const Range = struct {
    start: Position,
    end: Position,
    
    pub fn init(start: Position, end: Position) Range {
        return Range{ .start = start, .end = end };
    }
    
    pub fn contains(self: Range, pos: Position) bool {
        if (pos.line < self.start.line or pos.line > self.end.line) {
            return false;
        }
        if (pos.line == self.start.line and pos.character < self.start.character) {
            return false;
        }
        if (pos.line == self.end.line and pos.character > self.end.character) {
            return false;
        }
        return true;
    }
};

/// LSP Location (URI + range)
pub const Location = struct {
    uri: []const u8,
    range: Range,
    
    pub fn init(uri: []const u8, range: Range) Location {
        return Location{ .uri = uri, .range = range };
    }
};

/// Diagnostic Severity
pub const DiagnosticSeverity = enum(u8) {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
    
    pub fn toInt(self: DiagnosticSeverity) u8 {
        return @intFromEnum(self);
    }
};

/// LSP Diagnostic
pub const Diagnostic = struct {
    range: Range,
    severity: ?DiagnosticSeverity = null,
    code: ?[]const u8 = null,
    source: ?[]const u8 = null,
    message: []const u8,
    
    pub fn init(range: Range, message: []const u8) Diagnostic {
        return Diagnostic{
            .range = range,
            .message = message,
        };
    }
    
    pub fn withSeverity(self: Diagnostic, severity: DiagnosticSeverity) Diagnostic {
        var d = self;
        d.severity = severity;
        return d;
    }
    
    pub fn withCode(self: Diagnostic, code: []const u8) Diagnostic {
        var d = self;
        d.code = code;
        return d;
    }
    
    pub fn withSource(self: Diagnostic, source: []const u8) Diagnostic {
        var d = self;
        d.source = source;
        return d;
    }
};

/// Text Document Identifier
pub const TextDocumentIdentifier = struct {
    uri: []const u8,
    
    pub fn init(uri: []const u8) TextDocumentIdentifier {
        return TextDocumentIdentifier{ .uri = uri };
    }
};

/// Versioned Text Document Identifier
pub const VersionedTextDocumentIdentifier = struct {
    uri: []const u8,
    version: i32,
    
    pub fn init(uri: []const u8, version: i32) VersionedTextDocumentIdentifier {
        return VersionedTextDocumentIdentifier{ .uri = uri, .version = version };
    }
};

// ============================================================================
// Text Document
// ============================================================================

pub const TextDocument = struct {
    uri: []const u8,
    language_id: []const u8,
    version: i32,
    content: []const u8,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, uri: []const u8, language_id: []const u8, version: i32, content: []const u8) !TextDocument {
        const uri_copy = try allocator.dupe(u8, uri);
        const lang_copy = try allocator.dupe(u8, language_id);
        const content_copy = try allocator.dupe(u8, content);
        
        return TextDocument{
            .uri = uri_copy,
            .language_id = lang_copy,
            .version = version,
            .content = content_copy,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TextDocument) void {
        self.allocator.free(self.uri);
        self.allocator.free(self.language_id);
        self.allocator.free(self.content);
    }
    
    pub fn updateContent(self: *TextDocument, new_content: []const u8, new_version: i32) !void {
        self.allocator.free(self.content);
        self.content = try self.allocator.dupe(u8, new_content);
        self.version = new_version;
    }
    
    pub fn getLineCount(self: TextDocument) usize {
        var count: usize = 1;
        for (self.content) |c| {
            if (c == '\n') count += 1;
        }
        return count;
    }
    
    pub fn getLine(self: TextDocument, line_number: usize) ?[]const u8 {
        var current_line: usize = 0;
        var line_start: usize = 0;
        
        for (self.content, 0..) |c, i| {
            if (current_line == line_number and c == '\n') {
                return self.content[line_start..i];
            }
            if (c == '\n') {
                current_line += 1;
                line_start = i + 1;
            }
        }
        
        if (current_line == line_number) {
            return self.content[line_start..];
        }
        
        return null;
    }
};

// ============================================================================
// Document Manager
// ============================================================================

pub const DocumentManager = struct {
    documents: std.StringHashMap(TextDocument),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DocumentManager {
        return DocumentManager{
            .documents = std.StringHashMap(TextDocument).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DocumentManager) void {
        var iter = self.documents.iterator();
        while (iter.next()) |entry| {
            var doc = entry.value_ptr;
            doc.deinit();
            self.allocator.free(entry.key_ptr.*);
        }
        self.documents.deinit();
    }
    
    pub fn open(self: *DocumentManager, uri: []const u8, language_id: []const u8, version: i32, content: []const u8) !void {
        const doc = try TextDocument.init(self.allocator, uri, language_id, version, content);
        const uri_copy = try self.allocator.dupe(u8, uri);
        try self.documents.put(uri_copy, doc);
    }
    
    pub fn close(self: *DocumentManager, uri: []const u8) !void {
        if (self.documents.fetchRemove(uri)) |kv| {
            var doc = kv.value;
            doc.deinit();
            self.allocator.free(kv.key);
        }
    }
    
    pub fn update(self: *DocumentManager, uri: []const u8, content: []const u8, version: i32) !void {
        if (self.documents.getPtr(uri)) |doc| {
            try doc.updateContent(content, version);
        } else {
            return error.DocumentNotFound;
        }
    }
    
    pub fn get(self: *DocumentManager, uri: []const u8) ?*TextDocument {
        return self.documents.getPtr(uri);
    }
    
    pub fn has(self: *DocumentManager, uri: []const u8) bool {
        return self.documents.contains(uri);
    }
};

// ============================================================================
// Diagnostics Manager
// ============================================================================

pub const DiagnosticsManager = struct {
    diagnostics: std.StringHashMap(std.ArrayList(Diagnostic)),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DiagnosticsManager {
        return DiagnosticsManager{
            .diagnostics = std.StringHashMap(std.ArrayList(Diagnostic)).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *DiagnosticsManager) void {
        var iter = self.diagnostics.iterator();
        while (iter.next()) |entry| {
            entry.value_ptr.deinit(self.allocator);
            self.allocator.free(entry.key_ptr.*);
        }
        self.diagnostics.deinit();
    }
    
    pub fn set(self: *DiagnosticsManager, uri: []const u8, diagnostics: []const Diagnostic) !void {
        // Remove old diagnostics
        if (self.diagnostics.fetchRemove(uri)) |kv| {
            var list = kv.value;
            list.deinit(self.allocator);
            self.allocator.free(kv.key);
        }
        
        // Add new diagnostics
        var list = try std.ArrayList(Diagnostic).initCapacity(self.allocator, diagnostics.len);
        try list.appendSlice(self.allocator, diagnostics);
        
        const uri_copy = try self.allocator.dupe(u8, uri);
        try self.diagnostics.put(uri_copy, list);
    }
    
    pub fn get(self: *DiagnosticsManager, uri: []const u8) ?[]const Diagnostic {
        if (self.diagnostics.getPtr(uri)) |list| {
            return list.items;
        }
        return null;
    }
    
    pub fn clear(self: *DiagnosticsManager, uri: []const u8) !void {
        if (self.diagnostics.fetchRemove(uri)) |kv| {
            var list = kv.value;
            list.deinit(self.allocator);
            self.allocator.free(kv.key);
        }
    }
};

// ============================================================================
// LSP Server
// ============================================================================

pub const ServerCapabilities = struct {
    text_document_sync: u8 = 1, // Full document sync
    diagnostic_provider: bool = true,
    completion_provider: bool = true,
    hover_provider: bool = true,
    signature_help_provider: bool = true,
    definition_provider: bool = true,
    references_provider: bool = true,
    document_symbol_provider: bool = true,
    
    pub fn init() ServerCapabilities {
        return ServerCapabilities{};
    }
};

pub const LspServer = struct {
    allocator: Allocator,
    document_manager: DocumentManager,
    diagnostics_manager: DiagnosticsManager,
    initialized: bool = false,
    
    pub fn init(allocator: Allocator) LspServer {
        return LspServer{
            .allocator = allocator,
            .document_manager = DocumentManager.init(allocator),
            .diagnostics_manager = DiagnosticsManager.init(allocator),
        };
    }
    
    pub fn deinit(self: *LspServer) void {
        self.document_manager.deinit();
        self.diagnostics_manager.deinit();
    }
    
    /// Handle initialize request
    pub fn handleInitialize(self: *LspServer) !json.Value {
        self.initialized = true;
        
        // Return server capabilities
        const capabilities = ServerCapabilities.init();
        
        // Build response object manually
        // In real implementation, would use proper JSON serialization
        _ = capabilities;
        
        return json.Value{ .null = {} };
    }
    
    /// Handle textDocument/didOpen notification
    pub fn handleDidOpen(self: *LspServer, uri: []const u8, language_id: []const u8, version: i32, text: []const u8) !void {
        try self.document_manager.open(uri, language_id, version, text);
        
        // Analyze document and generate diagnostics
        try self.analyzeDocument(uri);
    }
    
    /// Handle textDocument/didChange notification
    pub fn handleDidChange(self: *LspServer, uri: []const u8, version: i32, text: []const u8) !void {
        try self.document_manager.update(uri, text, version);
        
        // Re-analyze document
        try self.analyzeDocument(uri);
    }
    
    /// Handle textDocument/didClose notification
    pub fn handleDidClose(self: *LspServer, uri: []const u8) !void {
        try self.document_manager.close(uri);
        try self.diagnostics_manager.clear(uri);
    }
    
    /// Analyze document and generate diagnostics
    fn analyzeDocument(self: *LspServer, uri: []const u8) !void {
        const doc = self.document_manager.get(uri) orelse return error.DocumentNotFound;
        
        // Simple syntax checking - look for common errors
        var diagnostics = try std.ArrayList(Diagnostic).initCapacity(self.allocator, 8);
        defer diagnostics.deinit(self.allocator);
        
        // Example: Check for unterminated strings
        var line_number: u32 = 0;
        var char_pos: u32 = 0;
        var in_string = false;
        var string_start_line: u32 = 0;
        var string_start_char: u32 = 0;
        
        for (doc.content) |c| {
            if (c == '"' and !in_string) {
                in_string = true;
                string_start_line = line_number;
                string_start_char = char_pos;
            } else if (c == '"' and in_string) {
                in_string = false;
            } else if (c == '\n') {
                if (in_string) {
                    // Unterminated string
                    const range = Range.init(
                        Position.init(string_start_line, string_start_char),
                        Position.init(line_number, char_pos),
                    );
                    const diag = Diagnostic.init(range, "Unterminated string literal")
                        .withSeverity(.Error)
                        .withSource("mojo-lsp");
                    try diagnostics.append(self.allocator, diag);
                    in_string = false;
                }
                line_number += 1;
                char_pos = 0;
            } else {
                char_pos += 1;
            }
        }
        
        // Store diagnostics
        try self.diagnostics_manager.set(uri, diagnostics.items);
    }
    
    /// Get diagnostics for a document
    pub fn getDiagnostics(self: *LspServer, uri: []const u8) ?[]const Diagnostic {
        return self.diagnostics_manager.get(uri);
    }
};

// ============================================================================
// Tests
// ============================================================================

test "Position: equality" {
    const pos1 = Position.init(10, 5);
    const pos2 = Position.init(10, 5);
    const pos3 = Position.init(10, 6);
    
    try std.testing.expect(pos1.eql(pos2));
    try std.testing.expect(!pos1.eql(pos3));
}

test "Range: contains position" {
    const range = Range.init(
        Position.init(5, 0),
        Position.init(10, 20),
    );
    
    try std.testing.expect(range.contains(Position.init(7, 10)));
    try std.testing.expect(!range.contains(Position.init(4, 10)));
    try std.testing.expect(!range.contains(Position.init(11, 10)));
}

test "TextDocument: initialization" {
    var doc = try TextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "fn main():\n  print(\"Hello\")\n",
    );
    defer doc.deinit();
    
    try std.testing.expectEqualStrings("file:///test.mojo", doc.uri);
    try std.testing.expectEqual(@as(i32, 1), doc.version);
    try std.testing.expectEqual(@as(usize, 3), doc.getLineCount());
}

test "TextDocument: get line" {
    var doc = try TextDocument.init(
        std.testing.allocator,
        "file:///test.mojo",
        "mojo",
        1,
        "line 1\nline 2\nline 3",
    );
    defer doc.deinit();
    
    const line1 = doc.getLine(0).?;
    try std.testing.expectEqualStrings("line 1", line1);
    
    const line2 = doc.getLine(1).?;
    try std.testing.expectEqualStrings("line 2", line2);
}

test "DocumentManager: open and close" {
    var manager = DocumentManager.init(std.testing.allocator);
    defer manager.deinit();
    
    try manager.open("file:///test.mojo", "mojo", 1, "content");
    try std.testing.expect(manager.has("file:///test.mojo"));
    
    try manager.close("file:///test.mojo");
    try std.testing.expect(!manager.has("file:///test.mojo"));
}

test "DocumentManager: update content" {
    var manager = DocumentManager.init(std.testing.allocator);
    defer manager.deinit();
    
    try manager.open("file:///test.mojo", "mojo", 1, "old content");
    try manager.update("file:///test.mojo", "new content", 2);
    
    const doc = manager.get("file:///test.mojo").?;
    try std.testing.expectEqualStrings("new content", doc.content);
    try std.testing.expectEqual(@as(i32, 2), doc.version);
}

test "DiagnosticsManager: set and get" {
    var manager = DiagnosticsManager.init(std.testing.allocator);
    defer manager.deinit();
    
    const diag1 = Diagnostic.init(
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "Test error",
    ).withSeverity(.Error);
    
    const diagnostics = [_]Diagnostic{diag1};
    try manager.set("file:///test.mojo", &diagnostics);
    
    const retrieved = manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(usize, 1), retrieved.len);
    try std.testing.expectEqualStrings("Test error", retrieved[0].message);
}

test "LspServer: document lifecycle" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    // Open document
    try server.handleDidOpen("file:///test.mojo", "mojo", 1, "fn main():\n  pass\n");
    try std.testing.expect(server.document_manager.has("file:///test.mojo"));
    
    // Update document
    try server.handleDidChange("file:///test.mojo", 2, "fn main():\n  print(\"hi\")\n");
    const doc = server.document_manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(i32, 2), doc.version);
    
    // Close document
    try server.handleDidClose("file:///test.mojo");
    try std.testing.expect(!server.document_manager.has("file:///test.mojo"));
}

test "LspServer: diagnostics generation" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    // Open document with unterminated string
    try server.handleDidOpen("file:///test.mojo", "mojo", 1, "let x = \"unterminated\n");
    
    // Check diagnostics were generated
    const diagnostics = server.getDiagnostics("file:///test.mojo");
    try std.testing.expect(diagnostics != null);
    try std.testing.expect(diagnostics.?.len > 0);
}

test "ServerCapabilities: all features enabled" {
    const caps = ServerCapabilities.init();
    
    try std.testing.expect(caps.diagnostic_provider);
    try std.testing.expect(caps.completion_provider);
    try std.testing.expect(caps.hover_provider);
    try std.testing.expect(caps.signature_help_provider);
    try std.testing.expect(caps.definition_provider);
    try std.testing.expect(caps.references_provider);
    try std.testing.expect(caps.document_symbol_provider);
}

test "LspServer: initialization" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    try std.testing.expect(!server.initialized);
    
    _ = try server.handleInitialize();
    try std.testing.expect(server.initialized);
}

test "LspServer: multiple documents" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    // Open multiple documents
    try server.handleDidOpen("file:///doc1.mojo", "mojo", 1, "fn test1() {}");
    try server.handleDidOpen("file:///doc2.mojo", "mojo", 1, "fn test2() {}");
    try server.handleDidOpen("file:///doc3.mojo", "mojo", 1, "fn test3() {}");
    
    try std.testing.expect(server.document_manager.has("file:///doc1.mojo"));
    try std.testing.expect(server.document_manager.has("file:///doc2.mojo"));
    try std.testing.expect(server.document_manager.has("file:///doc3.mojo"));
    
    // Close one document
    try server.handleDidClose("file:///doc2.mojo");
    try std.testing.expect(!server.document_manager.has("file:///doc2.mojo"));
    try std.testing.expect(server.document_manager.has("file:///doc1.mojo"));
    try std.testing.expect(server.document_manager.has("file:///doc3.mojo"));
}

test "LspServer: document version tracking" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    // Open at version 1
    try server.handleDidOpen("file:///test.mojo", "mojo", 1, "v1");
    const doc1 = server.document_manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(i32, 1), doc1.version);
    
    // Update to version 2
    try server.handleDidChange("file:///test.mojo", 2, "v2");
    const doc2 = server.document_manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(i32, 2), doc2.version);
    
    // Update to version 3
    try server.handleDidChange("file:///test.mojo", 3, "v3");
    const doc3 = server.document_manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(i32, 3), doc3.version);
}

test "DiagnosticsManager: multiple diagnostics" {
    var manager = DiagnosticsManager.init(std.testing.allocator);
    defer manager.deinit();
    
    const diag1 = Diagnostic.init(
        Range.init(Position.init(0, 0), Position.init(0, 5)),
        "Error 1",
    ).withSeverity(.Error);
    
    const diag2 = Diagnostic.init(
        Range.init(Position.init(1, 0), Position.init(1, 5)),
        "Warning 1",
    ).withSeverity(.Warning);
    
    const diag3 = Diagnostic.init(
        Range.init(Position.init(2, 0), Position.init(2, 5)),
        "Hint 1",
    ).withSeverity(.Hint);
    
    const diagnostics = [_]Diagnostic{ diag1, diag2, diag3 };
    try manager.set("file:///test.mojo", &diagnostics);
    
    const retrieved = manager.get("file:///test.mojo").?;
    try std.testing.expectEqual(@as(usize, 3), retrieved.len);
}

test "LspServer: diagnostics cleared on close" {
    var server = LspServer.init(std.testing.allocator);
    defer server.deinit();
    
    // Open document with error
    try server.handleDidOpen("file:///test.mojo", "mojo", 1, "let x = \"unterminated\n");
    
    // Verify diagnostics exist
    const diag1 = server.getDiagnostics("file:///test.mojo");
    try std.testing.expect(diag1 != null);
    
    // Close document
    try server.handleDidClose("file:///test.mojo");
    
    // Verify diagnostics cleared
    const diag2 = server.getDiagnostics("file:///test.mojo");
    try std.testing.expect(diag2 == null);
}
