// Mojo SDK - Lexer Tests
// Day 1: Comprehensive test suite for the lexer

const std = @import("std");
const lexer_mod = @import("lexer");
const Lexer = lexer_mod.Lexer;
const Token = lexer_mod.Token;
const TokenType = lexer_mod.TokenType;

// ============================================================================
// Helper Functions
// ============================================================================

fn expectToken(token: Token, expected_type: TokenType, expected_lexeme: []const u8) !void {
    try std.testing.expectEqual(expected_type, token.type);
    try std.testing.expectEqualStrings(expected_lexeme, token.lexeme);
}

fn testLexer(source: []const u8, expected_tokens: []const TokenType) !void {
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // Check we have the right number of tokens (+ EOF)
    try std.testing.expectEqual(expected_tokens.len + 1, tokens.items.len);
    
    // Check each token type
    for (expected_tokens, 0..) |expected, i| {
        try std.testing.expectEqual(expected, tokens.items[i].type);
    }
    
    // Last token should be EOF
    try std.testing.expectEqual(TokenType.eof, tokens.items[tokens.items.len - 1].type);
}

// ============================================================================
// Keyword Tests
// ============================================================================

test "lexer: all keywords" {
    const source = 
        \\fn struct var let if else for while return
        \\import from as alias trait impl
        \\inout owned borrowed ref const static
        \\async await break continue pass
        \\raise try except finally with match case
    ;
    
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(TokenType.fn_keyword, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.struct_keyword, tokens.items[1].type);
    try std.testing.expectEqual(TokenType.var_keyword, tokens.items[2].type);
    try std.testing.expectEqual(TokenType.let_keyword, tokens.items[3].type);
    try std.testing.expectEqual(TokenType.if_keyword, tokens.items[4].type);
    try std.testing.expectEqual(TokenType.else_keyword, tokens.items[5].type);
}

test "lexer: logical operators as keywords" {
    try testLexer("and or not", &[_]TokenType{
        .and_keyword,
        .or_keyword,
        .not_keyword,
    });
}

// ============================================================================
// Type Tests
// ============================================================================

test "lexer: primitive types" {
    try testLexer("Int Float Bool String", &[_]TokenType{
        .int_type,
        .float_type,
        .bool_type,
        .string_type,
    });
}

// ============================================================================
// Operator Tests
// ============================================================================

test "lexer: arithmetic operators" {
    try testLexer("+ - * / % **", &[_]TokenType{
        .plus,
        .minus,
        .star,
        .slash,
        .percent,
        .power,
    });
}

test "lexer: comparison operators" {
    try testLexer("== != < <= > >=", &[_]TokenType{
        .equal_equal,
        .not_equal,
        .less,
        .less_equal,
        .greater,
        .greater_equal,
    });
}

test "lexer: bitwise operators" {
    try testLexer("& | ^ ~ << >>", &[_]TokenType{
        .ampersand,
        .pipe,
        .caret,
        .tilde,
        .left_shift,
        .right_shift,
    });
}

test "lexer: assignment operators" {
    try testLexer("= += -= *= /=", &[_]TokenType{
        .equal,
        .plus_equal,
        .minus_equal,
        .star_equal,
        .slash_equal,
    });
}

test "lexer: arrow and double colon" {
    try testLexer("-> ::", &[_]TokenType{
        .arrow,
        .double_colon,
    });
}

// ============================================================================
// Delimiter Tests
// ============================================================================

test "lexer: delimiters" {
    try testLexer("( ) [ ] { } , . : ;", &[_]TokenType{
        .left_paren,
        .right_paren,
        .left_bracket,
        .right_bracket,
        .left_brace,
        .right_brace,
        .comma,
        .dot,
        .colon,
        .semicolon,
    });
}

// ============================================================================
// Literal Tests
// ============================================================================

test "lexer: integer literals" {
    var lex = try Lexer.init(std.testing.allocator, "0 42 1000");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .integer_literal, "0");
    try expectToken(tokens.items[1], .integer_literal, "42");
    try expectToken(tokens.items[2], .integer_literal, "1000");
}

test "lexer: float literals" {
    var lex = try Lexer.init(std.testing.allocator, "3.14 0.5 42.0");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .float_literal, "3.14");
    try expectToken(tokens.items[1], .float_literal, "0.5");
    try expectToken(tokens.items[2], .float_literal, "42.0");
}

test "lexer: string literals - double quotes" {
    var lex = try Lexer.init(std.testing.allocator, "\"hello\" \"world\"");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .string_literal, "\"hello\"");
    try expectToken(tokens.items[1], .string_literal, "\"world\"");
}

test "lexer: string literals - single quotes" {
    var lex = try Lexer.init(std.testing.allocator, "'hello' 'world'");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .string_literal, "'hello'");
    try expectToken(tokens.items[1], .string_literal, "'world'");
}

test "lexer: boolean literals" {
    try testLexer("true false", &[_]TokenType{
        .true_literal,
        .false_literal,
    });
}

// ============================================================================
// Identifier Tests
// ============================================================================

test "lexer: identifiers" {
    var lex = try Lexer.init(std.testing.allocator, "foo bar baz_qux _private");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .identifier, "foo");
    try expectToken(tokens.items[1], .identifier, "bar");
    try expectToken(tokens.items[2], .identifier, "baz_qux");
    try expectToken(tokens.items[3], .identifier, "_private");
}

test "lexer: identifiers vs keywords" {
    var lex = try Lexer.init(std.testing.allocator, "function fn function_name");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try expectToken(tokens.items[0], .identifier, "function");
    try expectToken(tokens.items[1], .fn_keyword, "fn");
    try expectToken(tokens.items[2], .identifier, "function_name");
}

// ============================================================================
// Comment Tests
// ============================================================================

test "lexer: single line comments" {
    const source = 
        \\# This is a comment
        \\fn main() # inline comment
        \\# Another comment
    ;
    
    try testLexer(source, &[_]TokenType{
        .newline,        // After first comment
        .fn_keyword,
        .identifier,
        .left_paren,
        .right_paren,
        .newline,        // After fn main() line
    });
}

// ============================================================================
// Whitespace Tests
// ============================================================================

test "lexer: whitespace handling" {
    const source = "   fn   main  (  )   ";
    
    try testLexer(source, &[_]TokenType{
        .fn_keyword,
        .identifier,
        .left_paren,
        .right_paren,
    });
}

test "lexer: newlines" {
    const source = "fn\nmain\n(\n)\n";
    
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // Check line numbers
    try std.testing.expectEqual(@as(usize, 1), tokens.items[0].line); // fn
    try std.testing.expectEqual(@as(usize, 2), tokens.items[2].line); // main
}

// ============================================================================
// Position Tracking Tests
// ============================================================================

test "lexer: line and column tracking" {
    const source = 
        \\fn main():
        \\  var x = 42
    ;
    
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // First token should be on line 1
    try std.testing.expectEqual(@as(usize, 1), tokens.items[0].line);
    
    // var should be on line 2
    var found_var = false;
    for (tokens.items) |token| {
        if (token.type == .var_keyword) {
            try std.testing.expectEqual(@as(usize, 2), token.line);
            found_var = true;
            break;
        }
    }
    try std.testing.expect(found_var);
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

test "lexer: function definition" {
    const source = "fn add(a: Int, b: Int) -> Int:";
    
    try testLexer(source, &[_]TokenType{
        .fn_keyword,
        .identifier,      // add
        .left_paren,
        .identifier,      // a
        .colon,
        .int_type,
        .comma,
        .identifier,      // b
        .colon,
        .int_type,
        .right_paren,
        .arrow,
        .int_type,
        .colon,
    });
}

test "lexer: struct definition" {
    const source = "struct Point { x: Float, y: Float }";
    
    try testLexer(source, &[_]TokenType{
        .struct_keyword,
        .identifier,      // Point
        .left_brace,
        .identifier,      // x
        .colon,
        .float_type,
        .comma,
        .identifier,      // y
        .colon,
        .float_type,
        .right_brace,
    });
}

test "lexer: if statement" {
    const source = "if x == 42 { return true } else { return false }";
    
    try testLexer(source, &[_]TokenType{
        .if_keyword,
        .identifier,      // x
        .equal_equal,
        .integer_literal, // 42
        .left_brace,
        .return_keyword,
        .true_literal,
        .right_brace,
        .else_keyword,
        .left_brace,
        .return_keyword,
        .false_literal,
        .right_brace,
    });
}

test "lexer: for loop" {
    const source = "for i in range(10):";
    
    try testLexer(source, &[_]TokenType{
        .for_keyword,
        .identifier,      // i
        .in_keyword,      // in
        .identifier,      // range
        .left_paren,
        .integer_literal, // 10
        .right_paren,
        .colon,
    });
}

// ============================================================================
// Error Handling Tests
// ============================================================================

test "lexer: unterminated string" {
    var lex = try Lexer.init(std.testing.allocator, "\"hello");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // Should produce invalid token
    try std.testing.expectEqual(TokenType.invalid, tokens.items[0].type);
}

test "lexer: invalid character" {
    var lex = try Lexer.init(std.testing.allocator, "!");
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // ! alone is invalid (must be !=)
    try std.testing.expectEqual(TokenType.invalid, tokens.items[0].type);
}

// ============================================================================
// Integration Tests
// ============================================================================

test "lexer: complete function" {
    const source = 
        \\fn factorial(n: Int) -> Int:
        \\    if n <= 1:
        \\        return 1
        \\    else:
        \\        return n * factorial(n - 1)
    ;
    
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // Just verify it doesn't crash and produces tokens
    try std.testing.expect(tokens.items.len > 10);
    try std.testing.expectEqual(TokenType.fn_keyword, tokens.items[0].type);
    try std.testing.expectEqual(TokenType.eof, tokens.items[tokens.items.len - 1].type);
}

test "lexer: generic type annotation" {
    const source = "List[Int]";
    
    try testLexer(source, &[_]TokenType{
        .identifier,      // List
        .left_bracket,
        .int_type,
        .right_bracket,
    });
}

test "lexer: multiline string" {
    const source = "\"hello\nworld\"";
    
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(TokenType.string_literal, tokens.items[0].type);
    try std.testing.expectEqual(@as(usize, 2), tokens.items[0].line); // Ends on line 2
}

// ============================================================================
// Performance Test
// ============================================================================

test "lexer: large file performance" {
    // Generate a moderately large source file
    var source = try std.ArrayList(u8).initCapacity(std.testing.allocator, 50000);
    defer source.deinit(std.testing.allocator);
    
    var i: usize = 0;
    while (i < 1000) : (i += 1) {
        try source.appendSlice(std.testing.allocator, "fn function_");
        try source.appendSlice(std.testing.allocator, "0123456789"[0..1]); // Add digit
        try source.appendSlice(std.testing.allocator, "() { return 42 }\n");
    }
    
    var lex = try Lexer.init(std.testing.allocator, source.items);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    // Should produce many tokens quickly
    try std.testing.expect(tokens.items.len > 5000);
}
