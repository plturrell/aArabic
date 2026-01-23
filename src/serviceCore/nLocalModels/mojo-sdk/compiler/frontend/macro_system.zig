// Advanced Macro System - Day 126
// Procedural macros with token stream manipulation

const std = @import("std");
const Allocator = std.mem.Allocator;
const ast = @import("ast.zig");
const lexer = @import("lexer.zig");

// ============================================================================
// Token Stream Types
// ============================================================================

pub const TokenTree = union(enum) {
    token: lexer.Token,
    delimited: Delimited,
    
    pub const Delimited = struct {
        delimiter: Delimiter,
        tokens: []TokenTree,
        
        pub const Delimiter = enum {
            paren,    // ( )
            brace,    // { }
            bracket,  // [ ]
        };
    };
};

pub const TokenStream = struct {
    tokens: std.ArrayList(TokenTree),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) TokenStream {
        return .{
            .tokens = std.ArrayList(TokenTree).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *TokenStream) void {
        self.tokens.deinit();
    }
    
    pub fn append(self: *TokenStream, token: TokenTree) !void {
        try self.tokens.append(token);
    }
    
    pub fn isEmpty(self: *const TokenStream) bool {
        return self.tokens.items.len == 0;
    }
    
    pub fn len(self: *const TokenStream) usize {
        return self.tokens.items.len;
    }
};

// ============================================================================
// Macro Definition
// ============================================================================

pub const MacroKind = enum {
    function_like,    // foo!(...)
    attribute,        // @foo
    derive,          // @derive(Trait)
    procedural,      // proc_macro!
};

pub const MacroDefinition = struct {
    name: []const u8,
    kind: MacroKind,
    handler: MacroHandler,
    
    pub const MacroHandler = *const fn (
        allocator: Allocator,
        input: TokenStream,
    ) anyerror!TokenStream;
};

// ============================================================================
// Macro Registry
// ============================================================================

pub const MacroRegistry = struct {
    macros: std.StringHashMap(MacroDefinition),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) MacroRegistry {
        return .{
            .macros = std.StringHashMap(MacroDefinition).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *MacroRegistry) void {
        self.macros.deinit();
    }
    
    pub fn register(self: *MacroRegistry, macro_def: MacroDefinition) !void {
        try self.macros.put(macro_def.name, macro_def);
    }
    
    pub fn get(self: *const MacroRegistry, name: []const u8) ?MacroDefinition {
        return self.macros.get(name);
    }
    
    pub fn exists(self: *const MacroRegistry, name: []const u8) bool {
        return self.macros.contains(name);
    }
};

// ============================================================================
// Macro Expansion
// ============================================================================

pub const MacroExpander = struct {
    registry: *MacroRegistry,
    allocator: Allocator,
    expansion_depth: usize,
    max_depth: usize,
    
    pub fn init(allocator: Allocator, registry: *MacroRegistry) MacroExpander {
        return .{
            .registry = registry,
            .allocator = allocator,
            .expansion_depth = 0,
            .max_depth = 32,  // Prevent infinite recursion
        };
    }
    
    pub fn expand(self: *MacroExpander, name: []const u8, input: TokenStream) !TokenStream {
        if (self.expansion_depth >= self.max_depth) {
            return error.MacroExpansionTooDeep;
        }
        
        const macro_def = self.registry.get(name) orelse {
            return error.MacroNotFound;
        };
        
        self.expansion_depth += 1;
        defer self.expansion_depth -= 1;
        
        return try macro_def.handler(self.allocator, input);
    }
    
    pub fn expandAll(self: *MacroExpander, tokens: TokenStream) !TokenStream {
        var result = TokenStream.init(self.allocator);
        
        for (tokens.tokens.items) |token| {
            // Check if token is macro invocation
            // If yes, expand it
            // If no, copy to result
            try result.append(token);
        }
        
        return result;
    }
};

// ============================================================================
// Built-in Macros
// ============================================================================

/// println! macro - prints with newline
/// Generates: print(format_args(...)); print("\n");
pub fn builtinPrintln(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Generate print function call token
    try result.append(.{ .token = .{
        .kind = .identifier,
        .source = "print",
        .line = 0,
        .column = 0,
    } });

    // Add opening paren
    try result.append(.{ .token = .{
        .kind = .lparen,
        .source = "(",
        .line = 0,
        .column = 0,
    } });

    // Copy input tokens (format arguments)
    for (input.tokens.items) |token| {
        try result.append(token);
    }

    // Add newline string literal
    try result.append(.{ .token = .{
        .kind = .string_literal,
        .source = "\"\\n\"",
        .line = 0,
        .column = 0,
    } });

    // Add closing paren
    try result.append(.{ .token = .{
        .kind = .rparen,
        .source = ")",
        .line = 0,
        .column = 0,
    } });

    return result;
}

/// vec! macro - creates vector from elements
/// Generates: { var v = List[T](); v.append(e1); v.append(e2); ... v }
pub fn builtinVec(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Generate block expression for vector creation
    try result.append(.{ .token = .{
        .kind = .lbrace,
        .source = "{",
        .line = 0,
        .column = 0,
    } });

    // var v = List()
    try result.append(.{ .token = .{ .kind = .keyword_var, .source = "var", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .identifier, .source = "_vec_temp", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .equal, .source = "=", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .identifier, .source = "List", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .lparen, .source = "(", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .rparen, .source = ")", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .semicolon, .source = ";", .line = 0, .column = 0 } });

    // For each element in input, generate: _vec_temp.append(elem);
    for (input.tokens.items) |token| {
        if (token == .token and token.token.kind == .comma) continue;

        try result.append(.{ .token = .{ .kind = .identifier, .source = "_vec_temp", .line = 0, .column = 0 } });
        try result.append(.{ .token = .{ .kind = .dot, .source = ".", .line = 0, .column = 0 } });
        try result.append(.{ .token = .{ .kind = .identifier, .source = "append", .line = 0, .column = 0 } });
        try result.append(.{ .token = .{ .kind = .lparen, .source = "(", .line = 0, .column = 0 } });
        try result.append(token);
        try result.append(.{ .token = .{ .kind = .rparen, .source = ")", .line = 0, .column = 0 } });
        try result.append(.{ .token = .{ .kind = .semicolon, .source = ";", .line = 0, .column = 0 } });
    }

    // Return the vector
    try result.append(.{ .token = .{ .kind = .identifier, .source = "_vec_temp", .line = 0, .column = 0 } });

    try result.append(.{ .token = .{
        .kind = .rbrace,
        .source = "}",
        .line = 0,
        .column = 0,
    } });

    return result;
}

/// assert! macro - runtime assertion
/// Generates: if not (condition) { raise Error("Assertion failed: " + msg) }
pub fn builtinAssert(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Generate: if not (...)
    try result.append(.{ .token = .{ .kind = .keyword_if, .source = "if", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .keyword_not, .source = "not", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .lparen, .source = "(", .line = 0, .column = 0 } });

    // Copy condition tokens
    for (input.tokens.items) |token| {
        try result.append(token);
    }

    try result.append(.{ .token = .{ .kind = .rparen, .source = ")", .line = 0, .column = 0 } });

    // Generate: { raise Error("Assertion failed") }
    try result.append(.{ .token = .{ .kind = .lbrace, .source = "{", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .keyword_raise, .source = "raise", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .identifier, .source = "Error", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .lparen, .source = "(", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .string_literal, .source = "\"Assertion failed\"", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .rparen, .source = ")", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .rbrace, .source = "}", .line = 0, .column = 0 } });

    return result;
}

/// format! macro - string formatting
/// Generates: String.format(format_str, args...)
pub fn builtinFormat(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Generate: String.format(...)
    try result.append(.{ .token = .{ .kind = .identifier, .source = "String", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .dot, .source = ".", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .identifier, .source = "format", .line = 0, .column = 0 } });
    try result.append(.{ .token = .{ .kind = .lparen, .source = "(", .line = 0, .column = 0 } });

    // Copy format string and arguments
    for (input.tokens.items) |token| {
        try result.append(token);
    }

    try result.append(.{ .token = .{ .kind = .rparen, .source = ")", .line = 0, .column = 0 } });

    return result;
}

/// include_str! macro - include file as string at compile time
/// Generates: string literal containing file contents
pub fn builtinIncludeStr(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Extract filename from input tokens
    var filename: []const u8 = "";
    for (input.tokens.items) |token| {
        if (token == .token and token.token.kind == .string_literal) {
            // Remove quotes from string literal
            const source = token.token.source;
            if (source.len >= 2) {
                filename = source[1 .. source.len - 1];
            }
            break;
        }
    }

    // Read file contents at compile time
    const file_contents = std.fs.cwd().readFileAlloc(allocator, filename, 1024 * 1024) catch |_| {
        // If file not found, generate error string
        try result.append(.{ .token = .{
            .kind = .string_literal,
            .source = "\"<file not found>\"",
            .line = 0,
            .column = 0,
        } });
        return result;
    };
    defer allocator.free(file_contents);

    // Generate string literal with escaped content
    const escaped = try std.fmt.allocPrint(allocator, "\"{s}\"", .{file_contents});
    defer allocator.free(escaped);

    try result.append(.{ .token = .{
        .kind = .string_literal,
        .source = escaped,
        .line = 0,
        .column = 0,
    } });

    return result;
}

/// stringify! macro - converts tokens to string
/// Generates: string literal of token text
pub fn builtinStringify(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Build string from all input tokens
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    try buffer.append('"');
    for (input.tokens.items) |token| {
        if (token == .token) {
            try buffer.appendSlice(token.token.source);
            try buffer.append(' ');
        }
    }
    try buffer.append('"');

    const stringified = try buffer.toOwnedSlice();

    try result.append(.{ .token = .{
        .kind = .string_literal,
        .source = stringified,
        .line = 0,
        .column = 0,
    } });

    return result;
}

/// concat! macro - concatenates literals at compile time
/// Generates: concatenated string literal
pub fn builtinConcat(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Concatenate all string literals
    var buffer = std.ArrayList(u8).init(allocator);
    defer buffer.deinit();

    for (input.tokens.items) |token| {
        if (token == .token and token.token.kind == .string_literal) {
            const source = token.token.source;
            // Strip quotes and append content
            if (source.len >= 2) {
                try buffer.appendSlice(source[1 .. source.len - 1]);
            }
        }
    }

    const concatenated = try std.fmt.allocPrint(allocator, "\"{s}\"", .{buffer.items});

    try result.append(.{ .token = .{
        .kind = .string_literal,
        .source = concatenated,
        .line = 0,
        .column = 0,
    } });

    return result;
}

/// env! macro - gets environment variable at compile time
/// Generates: string literal of env var value
pub fn builtinEnv(allocator: Allocator, input: TokenStream) !TokenStream {
    var result = TokenStream.init(allocator);

    // Extract env var name from input
    var env_name: []const u8 = "";
    for (input.tokens.items) |token| {
        if (token == .token and token.token.kind == .string_literal) {
            const source = token.token.source;
            if (source.len >= 2) {
                env_name = source[1 .. source.len - 1];
            }
            break;
        }
    }

    // Get environment variable at compile time
    const env_value = std.posix.getenv(env_name) orelse "";
    const value_str = try std.fmt.allocPrint(allocator, "\"{s}\"", .{env_value});

    try result.append(.{ .token = .{
        .kind = .string_literal,
        .source = value_str,
        .line = 0,
        .column = 0,
    } });

    return result;
}

// ============================================================================
// Macro Parser
// ============================================================================

pub const MacroParser = struct {
    tokens: []const lexer.Token,
    pos: usize,
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, tokens: []const lexer.Token) MacroParser {
        return .{
            .tokens = tokens,
            .pos = 0,
            .allocator = allocator,
        };
    }
    
    pub fn parseTokenStream(self: *MacroParser) !TokenStream {
        var stream = TokenStream.init(self.allocator);
        
        while (self.pos < self.tokens.len) {
            const token = self.tokens[self.pos];
            self.pos += 1;
            
            try stream.append(.{ .token = token });
        }
        
        return stream;
    }
    
    pub fn parseDelimited(self: *MacroParser, delimiter: TokenTree.Delimited.Delimiter) !TokenTree {
        var tokens = std.ArrayList(TokenTree).init(self.allocator);
        defer tokens.deinit();
        
        // Parse until matching closing delimiter
        while (self.pos < self.tokens.len) {
            const token = self.tokens[self.pos];
            
            // Check for closing delimiter
            const is_close = switch (delimiter) {
                .paren => token.type == .right_paren,
                .brace => token.type == .right_brace,
                .bracket => token.type == .right_bracket,
            };
            
            if (is_close) {
                self.pos += 1;
                break;
            }
            
            self.pos += 1;
            try tokens.append(.{ .token = token });
        }
        
        return .{
            .delimited = .{
                .delimiter = delimiter,
                .tokens = try tokens.toOwnedSlice(),
            },
        };
    }
};

// ============================================================================
// Quote/Unquote System
// ============================================================================

pub const QuoteBuilder = struct {
    tokens: std.ArrayList(TokenTree),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator) QuoteBuilder {
        return .{
            .tokens = std.ArrayList(TokenTree).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *QuoteBuilder) void {
        self.tokens.deinit();
    }
    
    pub fn addToken(self: *QuoteBuilder, token: lexer.Token) !void {
        try self.tokens.append(.{ .token = token });
    }
    
    pub fn addIdent(self: *QuoteBuilder, name: []const u8) !void {
        try self.addToken(.{
            .type = .identifier,
            .lexeme = name,
            .line = 0,
            .column = 0,
        });
    }
    
    pub fn addKeyword(self: *QuoteBuilder, keyword: lexer.TokenType) !void {
        try self.addToken(.{
            .type = keyword,
            .lexeme = "",
            .line = 0,
            .column = 0,
        });
    }
    
    pub fn unquote(self: *QuoteBuilder, stream: TokenStream) !void {
        for (stream.tokens.items) |token| {
            try self.tokens.append(token);
        }
    }
    
    pub fn build(self: *QuoteBuilder) !TokenStream {
        var stream = TokenStream.init(self.allocator);
        stream.tokens = try self.tokens.clone();
        return stream;
    }
};

// ============================================================================
// Procedural Macro Context
// ============================================================================

pub const ProcMacroContext = struct {
    allocator: Allocator,
    crate_name: []const u8,
    span_context: ?*SpanContext,
    
    pub const SpanContext = struct {
        file: []const u8,
        line: usize,
        column: usize,
    };
    
    pub fn init(allocator: Allocator) ProcMacroContext {
        return .{
            .allocator = allocator,
            .crate_name = "unknown",
            .span_context = null,
        };
    }
    
    pub fn error_with_span(self: *ProcMacroContext, message: []const u8) !void {
        if (self.span_context) |span| {
            std.debug.print("Error at {s}:{d}:{d}: {s}\n", 
                .{span.file, span.line, span.column, message});
        } else {
            std.debug.print("Error: {s}\n", .{message});
        }
        return error.MacroError;
    }
};

// ============================================================================
// Hygiene System
// ============================================================================

pub const HygieneContext = struct {
    scope_id: usize,
    parent_scope: ?*HygieneContext,
    bindings: std.StringHashMap(usize),
    allocator: Allocator,
    
    pub fn init(allocator: Allocator, scope_id: usize) HygieneContext {
        return .{
            .scope_id = scope_id,
            .parent_scope = null,
            .bindings = std.StringHashMap(usize).init(allocator),
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *HygieneContext) void {
        self.bindings.deinit();
    }
    
    pub fn addBinding(self: *HygieneContext, name: []const u8, id: usize) !void {
        try self.bindings.put(name, id);
    }
    
    pub fn lookup(self: *const HygieneContext, name: []const u8) ?usize {
        if (self.bindings.get(name)) |id| {
            return id;
        }
        
        if (self.parent_scope) |parent| {
            return parent.lookup(name);
        }
        
        return null;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "TokenStream creation" {
    const allocator = std.testing.allocator;
    var stream = TokenStream.init(allocator);
    defer stream.deinit();
    
    try std.testing.expect(stream.isEmpty());
    try std.testing.expectEqual(@as(usize, 0), stream.len());
}

test "MacroRegistry" {
    const allocator = std.testing.allocator;
    var registry = MacroRegistry.init(allocator);
    defer registry.deinit();
    
    const macro_def = MacroDefinition{
        .name = "test_macro",
        .kind = .function_like,
        .handler = builtinPrintln,
    };
    
    try registry.register(macro_def);
    try std.testing.expect(registry.exists("test_macro"));
}

test "MacroExpander initialization" {
    const allocator = std.testing.allocator;
    var registry = MacroRegistry.init(allocator);
    defer registry.deinit();
    
    var expander = MacroExpander.init(allocator, &registry);
    try std.testing.expectEqual(@as(usize, 0), expander.expansion_depth);
    try std.testing.expectEqual(@as(usize, 32), expander.max_depth);
}

test "QuoteBuilder" {
    const allocator = std.testing.allocator;
    var builder = QuoteBuilder.init(allocator);
    defer builder.deinit();
    
    try builder.addIdent("test");
    try std.testing.expectEqual(@as(usize, 1), builder.tokens.items.len);
}

test "HygieneContext" {
    const allocator = std.testing.allocator;
    var ctx = HygieneContext.init(allocator, 1);
    defer ctx.deinit();
    
    try ctx.addBinding("x", 42);
    const id = ctx.lookup("x");
    try std.testing.expect(id != null);
    try std.testing.expectEqual(@as(usize, 42), id.?);
}

test "ProcMacroContext" {
    const allocator = std.testing.allocator;
    const ctx = ProcMacroContext.init(allocator);
    try std.testing.expectEqualStrings("unknown", ctx.crate_name);
}
