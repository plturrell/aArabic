// Completion Item Resolution
// Day 79: Detailed completion information with lazy loading

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Documentation Types
// ============================================================================

/// Documentation format
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

/// Markup Content (LSP MarkupContent)
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
// Type Information
// ============================================================================

/// Type signature information
pub const TypeSignature = struct {
    name: []const u8,
    parameters: std.ArrayList([]const u8),
    return_type: ?[]const u8 = null,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, name: []const u8) !TypeSignature {
        return TypeSignature{
            .name = try allocator.dupe(u8, name),
            .parameters = std.ArrayList([]const u8){},
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TypeSignature) void {
        self.allocator.free(self.name);
        for (self.parameters.items) |param| {
            self.allocator.free(param);
        }
        self.parameters.deinit(self.allocator);
        if (self.return_type) |ret| {
            self.allocator.free(ret);
        }
    }
    
    pub fn addParameter(self: *TypeSignature, param: []const u8) !void {
        const param_copy = try self.allocator.dupe(u8, param);
        try self.parameters.append(self.allocator, param_copy);
    }
    
    pub fn setReturnType(self: *TypeSignature, return_type: []const u8) !void {
        self.return_type = try self.allocator.dupe(u8, return_type);
    }
    
    /// Format as string: fn(param1, param2) -> ReturnType
    pub fn format(self: TypeSignature, allocator: Allocator) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(allocator);
        
        try buffer.appendSlice(allocator, self.name);
        try buffer.append(allocator, '(');
        
        for (self.parameters.items, 0..) |param, i| {
            if (i > 0) try buffer.appendSlice(allocator, ", ");
            try buffer.appendSlice(allocator, param);
        }
        
        try buffer.append(allocator, ')');
        
        if (self.return_type) |ret| {
            try buffer.appendSlice(allocator, " -> ");
            try buffer.appendSlice(allocator, ret);
        }
        
        return try allocator.dupe(u8, buffer.items);
    }
};

// ============================================================================
// Completion Resolver
// ============================================================================

pub const CompletionResolver = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) CompletionResolver {
        return CompletionResolver{ .allocator = allocator };
    }
    
    /// Resolve completion item with full details
    pub fn resolve(self: *CompletionResolver, label: []const u8, kind: u8) !ResolvedCompletion {
        // In real implementation, this would:
        // 1. Look up symbol in index
        // 2. Extract full documentation
        // 3. Get type signature from AST
        // 4. Format everything nicely
        
        return ResolvedCompletion{
            .label = try self.allocator.dupe(u8, label),
            .detail = try self.generateDetail(label, kind),
            .documentation = try self.generateDocumentation(label, kind),
            .signature = try self.generateSignature(label, kind),
            .allocator = self.allocator,
        };
    }
    
    /// Generate detail string
    fn generateDetail(self: *CompletionResolver, label: []const u8, kind: u8) ![]const u8 {
        _ = kind;
        return try std.fmt.allocPrint(self.allocator, "Detail for {s}", .{label});
    }
    
    /// Generate documentation
    fn generateDocumentation(self: *CompletionResolver, label: []const u8, kind: u8) !MarkupContent {
        _ = kind;
        
        const doc_text = try std.fmt.allocPrint(
            self.allocator,
            "## {s}\n\nDetailed documentation for this symbol.",
            .{label},
        );
        defer self.allocator.free(doc_text);
        
        return try MarkupContent.init(self.allocator, .Markdown, doc_text);
    }
    
    /// Generate type signature
    fn generateSignature(self: *CompletionResolver, label: []const u8, kind: u8) !?TypeSignature {
        // Only generate signatures for functions/methods
        if (kind != 3 and kind != 2) return null; // 3=Function, 2=Method
        
        var sig = try TypeSignature.init(self.allocator, label);
        try sig.addParameter("x: i32");
        try sig.addParameter("y: i32");
        try sig.setReturnType("i32");
        
        return sig;
    }
};

/// Resolved completion with full details
pub const ResolvedCompletion = struct {
    label: []const u8,
    detail: []const u8,
    documentation: MarkupContent,
    signature: ?TypeSignature,
    allocator: Allocator,
    
    pub fn deinit(self: *ResolvedCompletion) void {
        self.allocator.free(self.label);
        self.allocator.free(self.detail);
        self.documentation.deinit();
        if (self.signature) |*sig| {
            sig.deinit();
        }
    }
};

// ============================================================================
// Documentation Formatter
// ============================================================================

pub const DocumentationFormatter = struct {
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) DocumentationFormatter {
        return DocumentationFormatter{ .allocator = allocator };
    }
    
    /// Format documentation as Markdown
    pub fn formatMarkdown(self: *DocumentationFormatter, title: []const u8, description: []const u8, signature: ?[]const u8) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        
        // Title
        try buffer.appendSlice(self.allocator, "# ");
        try buffer.appendSlice(self.allocator, title);
        try buffer.append(self.allocator, '\n');
        try buffer.append(self.allocator, '\n');
        
        // Signature
        if (signature) |sig| {
            try buffer.appendSlice(self.allocator, "```mojo\n");
            try buffer.appendSlice(self.allocator, sig);
            try buffer.append(self.allocator, '\n');
            try buffer.appendSlice(self.allocator, "```\n\n");
        }
        
        // Description
        try buffer.appendSlice(self.allocator, description);
        
        return try self.allocator.dupe(u8, buffer.items);
    }
    
    /// Format documentation as plain text
    pub fn formatPlainText(self: *DocumentationFormatter, title: []const u8, description: []const u8, signature: ?[]const u8) ![]const u8 {
        var buffer = std.ArrayList(u8){};
        defer buffer.deinit(self.allocator);
        
        // Title
        try buffer.appendSlice(self.allocator, title);
        try buffer.append(self.allocator, '\n');
        
        // Signature
        if (signature) |sig| {
            try buffer.append(self.allocator, '\n');
            try buffer.appendSlice(self.allocator, sig);
            try buffer.append(self.allocator, '\n');
        }
        
        // Description
        try buffer.append(self.allocator, '\n');
        try buffer.appendSlice(self.allocator, description);
        
        return try self.allocator.dupe(u8, buffer.items);
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
    var content = try MarkupContent.init(std.testing.allocator, .Markdown, "# Test\n\nContent");
    defer content.deinit();
    
    try std.testing.expectEqual(MarkupKind.Markdown, content.kind);
    try std.testing.expectEqualStrings("# Test\n\nContent", content.value);
}

test "TypeSignature: format" {
    var sig = try TypeSignature.init(std.testing.allocator, "myFunction");
    defer sig.deinit();
    
    try sig.addParameter("x: i32");
    try sig.addParameter("y: String");
    try sig.setReturnType("bool");
    
    const formatted = try sig.format(std.testing.allocator);
    defer std.testing.allocator.free(formatted);
    
    try std.testing.expectEqualStrings("myFunction(x: i32, y: String) -> bool", formatted);
}

test "CompletionResolver: resolve item" {
    var resolver = CompletionResolver.init(std.testing.allocator);
    
    var resolved = try resolver.resolve("testFunc", 3); // 3 = Function
    defer resolved.deinit();
    
    try std.testing.expectEqualStrings("testFunc", resolved.label);
    try std.testing.expect(resolved.signature != null);
}

test "DocumentationFormatter: markdown" {
    var formatter = DocumentationFormatter.init(std.testing.allocator);
    
    const doc = try formatter.formatMarkdown(
        "myFunction",
        "This is a test function",
        "fn(x: i32) -> i32",
    );
    defer std.testing.allocator.free(doc);
    
    try std.testing.expect(std.mem.indexOf(u8, doc, "# myFunction") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "```mojo") != null);
}

test "DocumentationFormatter: plain text" {
    var formatter = DocumentationFormatter.init(std.testing.allocator);
    
    const doc = try formatter.formatPlainText(
        "myFunction",
        "This is a test function",
        "fn(x: i32) -> i32",
    );
    defer std.testing.allocator.free(doc);
    
    try std.testing.expect(std.mem.indexOf(u8, doc, "myFunction") != null);
    try std.testing.expect(std.mem.indexOf(u8, doc, "fn(x: i32) -> i32") != null);
}
