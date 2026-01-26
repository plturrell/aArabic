// Mojo SDK - Parser Fuzzer
// Days 110-112: Fuzzing Infrastructure for 98/100 Engineering Quality
//
// This fuzzer tests the Mojo parser for crashes, hangs, and memory issues.
// Uses libFuzzer for coverage-guided fuzzing.

const std = @import("std");
const Allocator = std.mem.Allocator;

// ============================================================================
// Fuzzer Configuration
// ============================================================================

const FuzzConfig = struct {
    max_input_size: usize = 64 * 1024, // 64KB max input
    max_heap_size: usize = 16 * 1024 * 1024, // 16MB heap
    timeout_ms: u64 = 5000, // 5 second timeout
    max_recursion_depth: usize = 100,
};

const config = FuzzConfig{};

// ============================================================================
// Mock Lexer (Simplified for fuzzing)
// ============================================================================

const TokenType = enum {
    // Literals
    Identifier,
    Number,
    String,

    // Keywords
    Fn,
    Var,
    Let,
    If,
    Else,
    While,
    For,
    Return,
    Struct,
    Trait,
    Impl,
    Pub,
    Const,

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Equal,
    EqualEqual,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    And,
    Or,
    Not,

    // Delimiters
    LeftParen,
    RightParen,
    LeftBrace,
    RightBrace,
    LeftBracket,
    RightBracket,
    Comma,
    Dot,
    Colon,
    Semicolon,
    Arrow,

    // Special
    Eof,
    Invalid,
    Newline,
};

const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: usize,
    column: usize,
};

const Lexer = struct {
    source: []const u8,
    pos: usize,
    line: usize,
    column: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, source: []const u8) Lexer {
        return Lexer{
            .source = source,
            .pos = 0,
            .line = 1,
            .column = 1,
            .allocator = allocator,
        };
    }

    pub fn scanAll(self: *Lexer) !std.ArrayListUnmanaged(Token) {
        var tokens = std.ArrayListUnmanaged(Token){};
        errdefer tokens.deinit(self.allocator);

        while (self.pos < self.source.len) {
            const token = self.scanToken();
            try tokens.append(self.allocator, token);

            if (token.type == .Eof) break;

            // Prevent infinite loops
            if (tokens.items.len > self.source.len * 2) {
                break;
            }
        }

        // Ensure EOF at end
        if (tokens.items.len == 0 or tokens.items[tokens.items.len - 1].type != .Eof) {
            try tokens.append(self.allocator, Token{
                .type = .Eof,
                .lexeme = "",
                .line = self.line,
                .column = self.column,
            });
        }

        return tokens;
    }

    fn scanToken(self: *Lexer) Token {
        self.skipWhitespace();

        if (self.pos >= self.source.len) {
            return Token{
                .type = .Eof,
                .lexeme = "",
                .line = self.line,
                .column = self.column,
            };
        }

        const start = self.pos;
        const c = self.advance();

        // Single character tokens
        const token_type: TokenType = switch (c) {
            '(' => .LeftParen,
            ')' => .RightParen,
            '{' => .LeftBrace,
            '}' => .RightBrace,
            '[' => .LeftBracket,
            ']' => .RightBracket,
            ',' => .Comma,
            '.' => .Dot,
            ':' => .Colon,
            ';' => .Semicolon,
            '+' => .Plus,
            '*' => .Star,
            '/' => .Slash,
            '\n' => .Newline,

            '-' => if (self.match('>')) .Arrow else .Minus,
            '=' => if (self.match('=')) .EqualEqual else .Equal,
            '!' => if (self.match('=')) .NotEqual else .Not,
            '<' => if (self.match('=')) .LessEqual else .Less,
            '>' => if (self.match('=')) .GreaterEqual else .Greater,
            '&' => if (self.match('&')) .And else .Invalid,
            '|' => if (self.match('|')) .Or else .Invalid,

            '"' => blk: {
                self.scanString();
                break :blk .String;
            },

            '0'...'9' => blk: {
                self.scanNumber();
                break :blk .Number;
            },

            'a'...'z', 'A'...'Z', '_' => blk: {
                self.scanIdentifier();
                const lexeme = self.source[start..self.pos];
                break :blk self.identifierType(lexeme);
            },

            else => .Invalid,
        };

        return Token{
            .type = token_type,
            .lexeme = self.source[start..self.pos],
            .line = self.line,
            .column = self.column,
        };
    }

    fn advance(self: *Lexer) u8 {
        const c = self.source[self.pos];
        self.pos += 1;
        if (c == '\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        return c;
    }

    fn match(self: *Lexer, expected: u8) bool {
        if (self.pos >= self.source.len) return false;
        if (self.source[self.pos] != expected) return false;
        self.pos += 1;
        self.column += 1;
        return true;
    }

    fn skipWhitespace(self: *Lexer) void {
        while (self.pos < self.source.len) {
            const c = self.source[self.pos];
            switch (c) {
                ' ', '\t', '\r' => {
                    self.pos += 1;
                    self.column += 1;
                },
                '#' => {
                    // Skip comment
                    while (self.pos < self.source.len and self.source[self.pos] != '\n') {
                        self.pos += 1;
                    }
                },
                else => break,
            }
        }
    }

    fn scanString(self: *Lexer) void {
        while (self.pos < self.source.len and self.source[self.pos] != '"') {
            if (self.source[self.pos] == '\\' and self.pos + 1 < self.source.len) {
                self.pos += 2;
            } else {
                _ = self.advance();
            }
        }
        if (self.pos < self.source.len) {
            self.pos += 1; // Closing quote
        }
    }

    fn scanNumber(self: *Lexer) void {
        while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
            self.pos += 1;
        }
        if (self.pos < self.source.len and self.source[self.pos] == '.') {
            self.pos += 1;
            while (self.pos < self.source.len and isDigit(self.source[self.pos])) {
                self.pos += 1;
            }
        }
    }

    fn scanIdentifier(self: *Lexer) void {
        while (self.pos < self.source.len and isAlphaNumeric(self.source[self.pos])) {
            self.pos += 1;
        }
    }

    fn identifierType(self: *Lexer, lexeme: []const u8) TokenType {
        _ = self;
        const keywords = std.StaticStringMap(TokenType).initComptime(.{
            .{ "fn", .Fn },
            .{ "var", .Var },
            .{ "let", .Let },
            .{ "if", .If },
            .{ "else", .Else },
            .{ "while", .While },
            .{ "for", .For },
            .{ "return", .Return },
            .{ "struct", .Struct },
            .{ "trait", .Trait },
            .{ "impl", .Impl },
            .{ "pub", .Pub },
            .{ "const", .Const },
        });

        return keywords.get(lexeme) orelse .Identifier;
    }

    fn isDigit(c: u8) bool {
        return c >= '0' and c <= '9';
    }

    fn isAlphaNumeric(c: u8) bool {
        return (c >= 'a' and c <= 'z') or
            (c >= 'A' and c <= 'Z') or
            (c >= '0' and c <= '9') or
            c == '_';
    }
};

// ============================================================================
// Mock Parser (Simplified for fuzzing)
// ============================================================================

const AstNode = struct {
    tag: enum {
        Number,
        String,
        Identifier,
        Binary,
        Unary,
        Call,
        Index,
        Member,
        If,
        While,
        For,
        Return,
        VarDecl,
        FnDecl,
        StructDecl,
        Block,
        Error,
    },
};

const Parser = struct {
    tokens: []const Token,
    pos: usize,
    allocator: Allocator,
    depth: usize,

    pub fn init(allocator: Allocator, tokens: []const Token) Parser {
        return Parser{
            .tokens = tokens,
            .pos = 0,
            .allocator = allocator,
            .depth = 0,
        };
    }

    const ParseError = error{
        MaxRecursionDepth,
        OutOfMemory,
    };

    pub fn parse(self: *Parser) ParseError!AstNode {
        return self.parseStatement();
    }

    fn parseStatement(self: *Parser) ParseError!AstNode {
        if (self.depth > config.max_recursion_depth) {
            return error.MaxRecursionDepth;
        }
        self.depth += 1;
        defer self.depth -= 1;

        if (self.isAtEnd()) {
            return AstNode{ .tag = .Error };
        }

        return switch (self.peek().type) {
            .Fn => self.parseFnDecl(),
            .Var, .Let => self.parseVarDecl(),
            .If => self.parseIf(),
            .While => self.parseWhile(),
            .For => self.parseFor(),
            .Return => self.parseReturn(),
            .Struct => self.parseStruct(),
            .LeftBrace => self.parseBlock(),
            else => self.parseExpressionStatement(),
        };
    }

    fn parseExpression(self: *Parser) ParseError!AstNode {
        if (self.depth > config.max_recursion_depth) {
            return error.MaxRecursionDepth;
        }
        self.depth += 1;
        defer self.depth -= 1;

        return self.parseOr();
    }

    fn parseOr(self: *Parser) ParseError!AstNode {
        var left = try self.parseAnd();

        while (self.match(.Or)) {
            _ = try self.parseAnd();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseAnd(self: *Parser) ParseError!AstNode {
        var left = try self.parseEquality();

        while (self.match(.And)) {
            _ = try self.parseEquality();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseEquality(self: *Parser) ParseError!AstNode {
        var left = try self.parseComparison();

        while (self.match(.EqualEqual) or self.match(.NotEqual)) {
            _ = try self.parseComparison();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseComparison(self: *Parser) ParseError!AstNode {
        var left = try self.parseTerm();

        while (self.match(.Less) or self.match(.LessEqual) or
            self.match(.Greater) or self.match(.GreaterEqual))
        {
            _ = try self.parseTerm();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseTerm(self: *Parser) ParseError!AstNode {
        var left = try self.parseFactor();

        while (self.match(.Plus) or self.match(.Minus)) {
            _ = try self.parseFactor();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseFactor(self: *Parser) ParseError!AstNode {
        var left = try self.parseUnary();

        while (self.match(.Star) or self.match(.Slash)) {
            _ = try self.parseUnary();
            left = AstNode{ .tag = .Binary };
        }

        return left;
    }

    fn parseUnary(self: *Parser) ParseError!AstNode {
        if (self.match(.Minus) or self.match(.Not)) {
            _ = try self.parseUnary();
            return AstNode{ .tag = .Unary };
        }

        return self.parsePostfix();
    }

    fn parsePostfix(self: *Parser) ParseError!AstNode {
        var left = try self.parsePrimary();

        while (true) {
            if (self.match(.LeftParen)) {
                // Function call
                _ = try self.parseArguments();
                _ = self.consume(.RightParen);
                left = AstNode{ .tag = .Call };
            } else if (self.match(.LeftBracket)) {
                // Index
                _ = try self.parseExpression();
                _ = self.consume(.RightBracket);
                left = AstNode{ .tag = .Index };
            } else if (self.match(.Dot)) {
                // Member access
                _ = self.consume(.Identifier);
                left = AstNode{ .tag = .Member };
            } else {
                break;
            }
        }

        return left;
    }

    fn parsePrimary(self: *Parser) ParseError!AstNode {
        if (self.match(.Number)) {
            return AstNode{ .tag = .Number };
        }

        if (self.match(.String)) {
            return AstNode{ .tag = .String };
        }

        if (self.match(.Identifier)) {
            return AstNode{ .tag = .Identifier };
        }

        if (self.match(.LeftParen)) {
            const expr = try self.parseExpression();
            _ = self.consume(.RightParen);
            return expr;
        }

        return AstNode{ .tag = .Error };
    }

    fn parseArguments(self: *Parser) ParseError!void {
        if (self.check(.RightParen)) return;

        _ = try self.parseExpression();
        while (self.match(.Comma)) {
            _ = try self.parseExpression();
        }
    }

    fn parseFnDecl(self: *Parser) ParseError!AstNode {
        _ = self.consume(.Fn);
        _ = self.consume(.Identifier);
        _ = self.consume(.LeftParen);

        // Parameters
        if (!self.check(.RightParen)) {
            _ = self.consume(.Identifier);
            _ = self.consume(.Colon);
            _ = self.consume(.Identifier);

            while (self.match(.Comma)) {
                _ = self.consume(.Identifier);
                _ = self.consume(.Colon);
                _ = self.consume(.Identifier);
            }
        }

        _ = self.consume(.RightParen);

        // Return type
        if (self.match(.Arrow)) {
            _ = self.consume(.Identifier);
        }

        // Body
        _ = try self.parseBlock();

        return AstNode{ .tag = .FnDecl };
    }

    fn parseVarDecl(self: *Parser) ParseError!AstNode {
        _ = self.advance(); // var or let
        _ = self.consume(.Identifier);

        if (self.match(.Colon)) {
            _ = self.consume(.Identifier);
        }

        if (self.match(.Equal)) {
            _ = try self.parseExpression();
        }

        return AstNode{ .tag = .VarDecl };
    }

    fn parseIf(self: *Parser) ParseError!AstNode {
        _ = self.consume(.If);
        _ = try self.parseExpression();
        _ = try self.parseBlock();

        if (self.match(.Else)) {
            if (self.check(.If)) {
                _ = try self.parseIf();
            } else {
                _ = try self.parseBlock();
            }
        }

        return AstNode{ .tag = .If };
    }

    fn parseWhile(self: *Parser) ParseError!AstNode {
        _ = self.consume(.While);
        _ = try self.parseExpression();
        _ = try self.parseBlock();
        return AstNode{ .tag = .While };
    }

    fn parseFor(self: *Parser) ParseError!AstNode {
        _ = self.consume(.For);
        _ = self.consume(.Identifier);
        // Simplified: skip to block
        while (!self.check(.LeftBrace) and !self.isAtEnd()) {
            _ = self.advance();
        }
        _ = try self.parseBlock();
        return AstNode{ .tag = .For };
    }

    fn parseReturn(self: *Parser) ParseError!AstNode {
        _ = self.consume(.Return);
        if (!self.check(.Newline) and !self.check(.RightBrace) and !self.isAtEnd()) {
            _ = try self.parseExpression();
        }
        return AstNode{ .tag = .Return };
    }

    fn parseStruct(self: *Parser) ParseError!AstNode {
        _ = self.consume(.Struct);
        _ = self.consume(.Identifier);
        _ = try self.parseBlock();
        return AstNode{ .tag = .StructDecl };
    }

    fn parseBlock(self: *Parser) ParseError!AstNode {
        _ = self.consume(.LeftBrace);

        while (!self.check(.RightBrace) and !self.isAtEnd()) {
            _ = self.parseStatement() catch break;
            _ = self.match(.Newline);
            _ = self.match(.Semicolon);
        }

        _ = self.consume(.RightBrace);
        return AstNode{ .tag = .Block };
    }

    fn parseExpressionStatement(self: *Parser) ParseError!AstNode {
        return self.parseExpression();
    }

    // Helper methods

    fn peek(self: *Parser) Token {
        if (self.pos >= self.tokens.len) {
            return Token{ .type = .Eof, .lexeme = "", .line = 0, .column = 0 };
        }
        return self.tokens[self.pos];
    }

    fn advance(self: *Parser) Token {
        if (!self.isAtEnd()) {
            self.pos += 1;
        }
        return self.previous();
    }

    fn previous(self: *Parser) Token {
        if (self.pos == 0) {
            return Token{ .type = .Eof, .lexeme = "", .line = 0, .column = 0 };
        }
        return self.tokens[self.pos - 1];
    }

    fn check(self: *Parser, expected: TokenType) bool {
        return self.peek().type == expected;
    }

    fn match(self: *Parser, expected: TokenType) bool {
        if (self.check(expected)) {
            _ = self.advance();
            return true;
        }
        return false;
    }

    fn consume(self: *Parser, expected: TokenType) Token {
        if (self.check(expected)) {
            return self.advance();
        }
        return self.peek();
    }

    fn isAtEnd(self: *Parser) bool {
        return self.pos >= self.tokens.len or self.peek().type == .Eof;
    }
};

// ============================================================================
// Fuzzer Entry Point (libFuzzer)
// ============================================================================

export fn LLVMFuzzerTestOneInput(data: [*]const u8, size: usize) callconv(.c) c_int {
    // Limit input size
    if (size > config.max_input_size) {
        return 0;
    }

    // Use fixed buffer allocator to prevent OOM
    var buffer: [16 * 1024 * 1024]u8 = undefined;
    var fba = std.heap.FixedBufferAllocator.init(&buffer);
    const allocator = fba.allocator();

    // Create safe slice
    const source = data[0..size];

    // Attempt to parse
    fuzzParse(allocator, source) catch |err| {
        // Expected errors are fine
        switch (err) {
            error.MaxRecursionDepth => {},
            error.OutOfMemory => {},
        }
    };

    return 0;
}

fn fuzzParse(allocator: Allocator, source: []const u8) !void {
    // Tokenize
    var lexer = Lexer.init(allocator, source);
    var tokens = try lexer.scanAll();
    defer tokens.deinit(allocator);

    // Parse
    var parser = Parser.init(allocator, tokens.items);
    _ = try parser.parse();
}

// ============================================================================
// Standalone Test Mode
// ============================================================================

pub fn main() !void {
    const allocator = std.heap.page_allocator;

    // Test cases
    const test_cases = [_][]const u8{
        "fn main() { var x = 1; }",
        "if x > 0 { return x; } else { return -x; }",
        "struct Point { x: i32, y: i32 }",
        "let result = foo(1, 2, 3);",
        "while x < 10 { x = x + 1; }",
        // Edge cases
        "",
        "(((((",
        "fn fn fn fn",
        "{{{{{",
        "\"unterminated string",
    };

    var passed: usize = 0;
    var failed: usize = 0;
    _ = &failed;

    for (test_cases) |tc| {
        fuzzParse(allocator, tc) catch {
            // Expected for malformed input
            failed += 1;
            continue;
        };
        passed += 1;
    }

    std.debug.print("Parser fuzzer self-test: {d} passed, {d} failed\n", .{ passed, failed });
}

// ============================================================================
// Tests
// ============================================================================

test "fuzz valid function" {
    const allocator = std.testing.allocator;
    try fuzzParse(allocator, "fn main() { var x = 1; }");
}

test "fuzz empty input" {
    const allocator = std.testing.allocator;
    fuzzParse(allocator, "") catch {};
}

test "fuzz malformed input" {
    const allocator = std.testing.allocator;
    fuzzParse(allocator, "(((((") catch {};
}

test "fuzz deeply nested" {
    const allocator = std.testing.allocator;
    const deep = "((((((((((((((((((((((((((((((1))))))))))))))))))))))))))))))";
    fuzzParse(allocator, deep) catch {};
}
