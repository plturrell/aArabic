// Mojo SDK - Lexer (Tokenizer)
// Day 1: Foundation of the compiler pipeline
// Tokenizes Mojo source code into a stream of tokens

const std = @import("std");

// ============================================================================
// Token Types
// ============================================================================

pub const TokenType = enum {
    // Keywords
    fn_keyword,
    struct_keyword,
    var_keyword,
    let_keyword,
    if_keyword,
    else_keyword,
    for_keyword,
    while_keyword,
    return_keyword,
    in_keyword,
    import_keyword,
    from_keyword,
    as_keyword,
    alias_keyword,
    trait_keyword,
    impl_keyword,
    inout_keyword,
    owned_keyword,
    borrowed_keyword,
    ref_keyword,
    const_keyword,
    static_keyword,
    async_keyword,
    await_keyword,
    break_keyword,
    continue_keyword,
    pass_keyword,
    raise_keyword,
    try_keyword,
    except_keyword,
    finally_keyword,
    with_keyword,
    match_keyword,
    case_keyword,
    
    // Primitive types
    int_type,
    float_type,
    bool_type,
    string_type,
    
    // Operators
    plus,           // +
    minus,          // -
    star,           // *
    slash,          // /
    percent,        // %
    power,          // **
    ampersand,      // &
    pipe,           // |
    caret,          // ^
    tilde,          // ~
    left_shift,     // <<
    right_shift,    // >>
    
    // Comparison
    equal_equal,    // ==
    not_equal,      // !=
    less,           // <
    less_equal,     // <=
    greater,        // >
    greater_equal,  // >=
    
    // Assignment
    equal,          // =
    plus_equal,     // +=
    minus_equal,    // -=
    star_equal,     // *=
    slash_equal,    // /=
    
    // Logical
    and_keyword,    // and
    or_keyword,     // or
    not_keyword,    // not
    
    // Delimiters
    left_paren,     // (
    right_paren,    // )
    left_bracket,   // [
    right_bracket,  // ]
    left_brace,     // {
    right_brace,    // }
    comma,          // ,
    dot,            // .
    colon,          // :
    semicolon,      // ;
    arrow,          // ->
    double_colon,   // ::
    
    // Literals
    identifier,
    integer_literal,
    float_literal,
    string_literal,
    true_literal,
    false_literal,
    
    // Special
    newline,
    indent,
    dedent,
    eof,
    invalid,
};

// ============================================================================
// Token
// ============================================================================

pub const Token = struct {
    type: TokenType,
    lexeme: []const u8,
    line: usize,
    column: usize,
    
    pub fn init(token_type: TokenType, lexeme: []const u8, line: usize, column: usize) Token {
        return Token{
            .type = token_type,
            .lexeme = lexeme,
            .line = line,
            .column = column,
        };
    }
    
    pub fn format(self: Token, comptime fmt: []const u8, options: std.fmt.FormatOptions, writer: anytype) !void {
        _ = fmt;
        _ = options;
        try writer.print("{s}('{s}') at {}:{}", .{
            @tagName(self.type),
            self.lexeme,
            self.line,
            self.column,
        });
    }
};

// ============================================================================
// Lexer
// ============================================================================

pub const Lexer = struct {
    source: []const u8,
    start: usize = 0,
    current: usize = 0,
    line: usize = 1,
    column: usize = 1,
    allocator: std.mem.Allocator,
    indent_stack: std.ArrayList(usize),
    
    pub fn init(allocator: std.mem.Allocator, source: []const u8) !Lexer {
        var indent_stack = try std.ArrayList(usize).initCapacity(allocator, 16);
        try indent_stack.append(allocator, 0); // Base indentation level
        
        return Lexer{
            .source = source,
            .allocator = allocator,
            .indent_stack = indent_stack,
        };
    }
    
    pub fn deinit(self: *Lexer) void {
        self.indent_stack.deinit(self.allocator);
    }
    
    // ========================================================================
    // Main Scanning
    // ========================================================================
    
    pub fn scanToken(self: *Lexer) !Token {
        self.skipWhitespace();
        
        self.start = self.current;
        
        if (self.isAtEnd()) {
            // Emit remaining dedents
            if (self.indent_stack.items.len > 1) {
                _ = self.indent_stack.pop();
                return self.makeToken(.dedent);
            }
            return self.makeToken(.eof);
        }
        
        const c = self.advance();
        
        // Handle newlines and indentation
        if (c == '\n') {
            self.line += 1;
            self.column = 1;
            return self.makeToken(.newline);
        }
        
        // Identifiers and keywords
        if (isAlpha(c)) {
            return self.identifier();
        }
        
        // Numbers
        if (isDigit(c)) {
            return self.number();
        }
        
        // String literals
        if (c == '"' or c == '\'') {
            return self.string(c);
        }
        
        // Operators and delimiters
        return switch (c) {
            '(' => self.makeToken(.left_paren),
            ')' => self.makeToken(.right_paren),
            '[' => self.makeToken(.left_bracket),
            ']' => self.makeToken(.right_bracket),
            '{' => self.makeToken(.left_brace),
            '}' => self.makeToken(.right_brace),
            ',' => self.makeToken(.comma),
            '.' => self.makeToken(.dot),
            ';' => self.makeToken(.semicolon),
            '~' => self.makeToken(.tilde),
            '^' => self.makeToken(.caret),
            '%' => self.makeToken(.percent),
            '&' => self.makeToken(.ampersand),
            '|' => self.makeToken(.pipe),
            
            '+' => if (self.match('=')) self.makeToken(.plus_equal) else self.makeToken(.plus),
            '-' => if (self.match('>')) self.makeToken(.arrow) else if (self.match('=')) self.makeToken(.minus_equal) else self.makeToken(.minus),
            '*' => if (self.match('*')) self.makeToken(.power) else if (self.match('=')) self.makeToken(.star_equal) else self.makeToken(.star),
            '/' => if (self.match('=')) self.makeToken(.slash_equal) else self.makeToken(.slash),
            
            '=' => if (self.match('=')) self.makeToken(.equal_equal) else self.makeToken(.equal),
            '!' => if (self.match('=')) self.makeToken(.not_equal) else self.makeToken(.invalid),
            '<' => if (self.match('=')) self.makeToken(.less_equal) else if (self.match('<')) self.makeToken(.left_shift) else self.makeToken(.less),
            '>' => if (self.match('=')) self.makeToken(.greater_equal) else if (self.match('>')) self.makeToken(.right_shift) else self.makeToken(.greater),
            
            ':' => if (self.match(':')) self.makeToken(.double_colon) else self.makeToken(.colon),
            
            else => self.makeToken(.invalid),
        };
    }
    
    pub fn scanAll(self: *Lexer) !std.ArrayList(Token) {
        var tokens = try std.ArrayList(Token).initCapacity(self.allocator, 256);
        
        while (true) {
            const token = try self.scanToken();
            try tokens.append(self.allocator, token);
            if (token.type == .eof) break;
        }
        
        return tokens;
    }
    
    // ========================================================================
    // Token Creators
    // ========================================================================
    
    fn makeToken(self: *Lexer, token_type: TokenType) Token {
        const lexeme = self.source[self.start..self.current];
        const col = if (self.column >= lexeme.len) self.column - lexeme.len else 1;
        return Token.init(token_type, lexeme, self.line, col);
    }
    
    fn identifier(self: *Lexer) Token {
        while (isAlphaNumeric(self.peek())) {
            _ = self.advance();
        }
        
        const text = self.source[self.start..self.current];
        const token_type = self.identifierType(text);
        
        return self.makeToken(token_type);
    }
    
    fn identifierType(self: *Lexer, text: []const u8) TokenType {
        _ = self;
        
        // Keywords
        if (std.mem.eql(u8, text, "fn")) return .fn_keyword;
        if (std.mem.eql(u8, text, "struct")) return .struct_keyword;
        if (std.mem.eql(u8, text, "var")) return .var_keyword;
        if (std.mem.eql(u8, text, "let")) return .let_keyword;
        if (std.mem.eql(u8, text, "if")) return .if_keyword;
        if (std.mem.eql(u8, text, "else")) return .else_keyword;
        if (std.mem.eql(u8, text, "for")) return .for_keyword;
        if (std.mem.eql(u8, text, "while")) return .while_keyword;
        if (std.mem.eql(u8, text, "return")) return .return_keyword;
        if (std.mem.eql(u8, text, "in")) return .in_keyword;
        if (std.mem.eql(u8, text, "import")) return .import_keyword;
        if (std.mem.eql(u8, text, "from")) return .from_keyword;
        if (std.mem.eql(u8, text, "as")) return .as_keyword;
        if (std.mem.eql(u8, text, "alias")) return .alias_keyword;
        if (std.mem.eql(u8, text, "trait")) return .trait_keyword;
        if (std.mem.eql(u8, text, "impl")) return .impl_keyword;
        if (std.mem.eql(u8, text, "inout")) return .inout_keyword;
        if (std.mem.eql(u8, text, "owned")) return .owned_keyword;
        if (std.mem.eql(u8, text, "borrowed")) return .borrowed_keyword;
        if (std.mem.eql(u8, text, "ref")) return .ref_keyword;
        if (std.mem.eql(u8, text, "const")) return .const_keyword;
        if (std.mem.eql(u8, text, "static")) return .static_keyword;
        if (std.mem.eql(u8, text, "async")) return .async_keyword;
        if (std.mem.eql(u8, text, "await")) return .await_keyword;
        if (std.mem.eql(u8, text, "break")) return .break_keyword;
        if (std.mem.eql(u8, text, "continue")) return .continue_keyword;
        if (std.mem.eql(u8, text, "pass")) return .pass_keyword;
        if (std.mem.eql(u8, text, "raise")) return .raise_keyword;
        if (std.mem.eql(u8, text, "try")) return .try_keyword;
        if (std.mem.eql(u8, text, "except")) return .except_keyword;
        if (std.mem.eql(u8, text, "finally")) return .finally_keyword;
        if (std.mem.eql(u8, text, "with")) return .with_keyword;
        if (std.mem.eql(u8, text, "match")) return .match_keyword;
        if (std.mem.eql(u8, text, "case")) return .case_keyword;
        
        // Logical operators
        if (std.mem.eql(u8, text, "and")) return .and_keyword;
        if (std.mem.eql(u8, text, "or")) return .or_keyword;
        if (std.mem.eql(u8, text, "not")) return .not_keyword;
        
        // Types
        if (std.mem.eql(u8, text, "Int")) return .int_type;
        if (std.mem.eql(u8, text, "Float")) return .float_type;
        if (std.mem.eql(u8, text, "Bool")) return .bool_type;
        if (std.mem.eql(u8, text, "String")) return .string_type;
        
        // Literals
        if (std.mem.eql(u8, text, "true")) return .true_literal;
        if (std.mem.eql(u8, text, "false")) return .false_literal;
        
        return .identifier;
    }
    
    fn number(self: *Lexer) Token {
        while (isDigit(self.peek())) {
            _ = self.advance();
        }
        
        // Look for decimal point
        if (self.peek() == '.' and isDigit(self.peekNext())) {
            _ = self.advance(); // Consume '.'
            
            while (isDigit(self.peek())) {
                _ = self.advance();
            }
            
            return self.makeToken(.float_literal);
        }
        
        return self.makeToken(.integer_literal);
    }
    
    fn string(self: *Lexer, quote: u8) Token {
        while (self.peek() != quote and !self.isAtEnd()) {
            if (self.peek() == '\n') {
                self.line += 1;
                self.column = 0;
            }
            _ = self.advance();
        }
        
        if (self.isAtEnd()) {
            return self.makeToken(.invalid);
        }
        
        _ = self.advance(); // Closing quote
        return self.makeToken(.string_literal);
    }
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    fn isAtEnd(self: *Lexer) bool {
        return self.current >= self.source.len;
    }
    
    fn advance(self: *Lexer) u8 {
        const c = self.source[self.current];
        self.current += 1;
        self.column += 1;
        return c;
    }
    
    fn peek(self: *Lexer) u8 {
        if (self.isAtEnd()) return 0;
        return self.source[self.current];
    }
    
    fn peekNext(self: *Lexer) u8 {
        if (self.current + 1 >= self.source.len) return 0;
        return self.source[self.current + 1];
    }
    
    fn match(self: *Lexer, expected: u8) bool {
        if (self.isAtEnd()) return false;
        if (self.source[self.current] != expected) return false;
        
        self.current += 1;
        self.column += 1;
        return true;
    }
    
    fn skipWhitespace(self: *Lexer) void {
        while (true) {
            const c = self.peek();
            switch (c) {
                ' ', '\r', '\t' => {
                    _ = self.advance();
                },
                '#' => {
                    // Comment until end of line
                    while (self.peek() != '\n' and !self.isAtEnd()) {
                        _ = self.advance();
                    }
                },
                else => return,
            }
        }
    }
};

// ============================================================================
// Character Classification
// ============================================================================

fn isAlpha(c: u8) bool {
    return (c >= 'a' and c <= 'z') or
           (c >= 'A' and c <= 'Z') or
           c == '_';
}

fn isDigit(c: u8) bool {
    return c >= '0' and c <= '9';
}

fn isAlphaNumeric(c: u8) bool {
    return isAlpha(c) or isDigit(c);
}

// ============================================================================
// Tests
// ============================================================================

test "lexer keywords" {
    const source = "fn struct var let if else";
    var lexer = try Lexer.init(std.testing.allocator, source);
    defer lexer.deinit();
    
    const tokens = try lexer.scanAll();
    defer tokens.deinit();
    
    try std.testing.expectEqual(TokenType.fn_keyword, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.struct_keyword, tokens.items[1].type);
    try std.testing.expectEqual(TokenType.var_keyword, tokens.items[2].type);
    try std.testing.expectEqual(TokenType.let_keyword, tokens.items[3].type);
    try std.testing.expectEqual(TokenType.if_keyword, tokens.items[4].type);
    try std.testing.expectEqual(TokenType.else_keyword, tokens.items[5].type);
}

test "lexer operators" {
    const source = "+ - * / == != < >";
    var lexer = try Lexer.init(std.testing.allocator, source);
    defer lexer.deinit();
    
    const tokens = try lexer.scanAll();
    defer tokens.deinit();
    
    try std.testing.expectEqual(TokenType.plus, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.minus, tokens.items[1].type);
    try std.testing.expectEqual(TokenType.star, tokens.items[2].type);
    try std.testing.expectEqual(TokenType.slash, tokens.items[3].type);
    try std.testing.expectEqual(TokenType.equal_equal, tokens.items[4].type);
    try std.testing.expectEqual(TokenType.not_equal, tokens.items[5].type);
    try std.testing.expectEqual(TokenType.less, tokens.items[6].type);
    try std.testing.expectEqual(TokenType.greater, tokens.items[7].type);
}

test "lexer literals" {
    const source = "42 3.14 \"hello\" true false";
    var lexer = try Lexer.init(std.testing.allocator, source);
    defer lexer.deinit();
    
    const tokens = try lexer.scanAll();
    defer tokens.deinit();
    
    try std.testing.expectEqual(TokenType.integer_literal, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.float_literal, tokens.items[1].type);
    try std.testing.expectEqual(TokenType.string_literal, tokens.items[2].type);
    try std.testing.expectEqual(TokenType.true_literal, tokens.items[3].type);
    try std.testing.expectEqual(TokenType.false_literal, tokens.items[4].type);
}
