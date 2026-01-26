// Mojo SDK - Parser Tests
// Day 2: Comprehensive test suite for the parser

const std = @import("std");
const parser_mod = @import("parser");
const lexer_mod = @import("lexer");
const ast_mod = @import("ast");

const Parser = parser_mod.Parser;
const Lexer = lexer_mod.Lexer;
const ast = ast_mod;

// ============================================================================
// Helper Functions
// ============================================================================

fn parseSource(source: []const u8) !ast.Expr {
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var p = Parser.init(std.testing.allocator, tokens.items);
    return try p.parseExpression();
}

fn parseAndCleanup(source: []const u8, comptime testFn: fn(ast.Expr) anyerror!void) !void {
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var p = Parser.init(std.testing.allocator, tokens.items);
    const expr = try p.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    try testFn(expr);
}

// ============================================================================
// Literal Tests
// ============================================================================

test "parser: integer literal" {
    try parseAndCleanup("42", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.LiteralValue.integer, std.meta.activeTag(expr.literal.value));
            try std.testing.expectEqual(@as(i64, 42), expr.literal.value.integer);
        }
    }.test_fn);
}

test "parser: float literal" {
    try parseAndCleanup("3.14", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.LiteralValue.float, std.meta.activeTag(expr.literal.value));
            try std.testing.expectApproxEqRel(@as(f64, 3.14), expr.literal.value.float, 0.001);
        }
    }.test_fn);
}

test "parser: string literal" {
    try parseAndCleanup("\"hello world\"", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.LiteralValue.string, std.meta.activeTag(expr.literal.value));
            try std.testing.expectEqualStrings("hello world", expr.literal.value.string);
        }
    }.test_fn);
}

test "parser: boolean literals" {
    try parseAndCleanup("true", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
            try std.testing.expectEqual(true, expr.literal.value.boolean);
        }
    }.test_fn);
    
    try parseAndCleanup("false", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
            try std.testing.expectEqual(false, expr.literal.value.boolean);
        }
    }.test_fn);
}

test "parser: identifier" {
    try parseAndCleanup("variable_name", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr));
            try std.testing.expectEqualStrings("variable_name", expr.identifier.name);
        }
    }.test_fn);
}

// ============================================================================
// Binary Expression Tests
// ============================================================================

test "parser: addition" {
    try parseAndCleanup("1 + 2", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.operator);
            try std.testing.expectEqual(@as(i64, 1), expr.binary.left.literal.value.integer);
            try std.testing.expectEqual(@as(i64, 2), expr.binary.right.literal.value.integer);
        }
    }.test_fn);
}

test "parser: subtraction" {
    try parseAndCleanup("10 - 5", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.subtract, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: multiplication" {
    try parseAndCleanup("3 * 4", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.multiply, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: division" {
    try parseAndCleanup("10 / 2", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.divide, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: modulo" {
    try parseAndCleanup("10 % 3", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.modulo, expr.binary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Comparison Tests
// ============================================================================

test "parser: equality" {
    try parseAndCleanup("x == 42", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.equal, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: inequality" {
    try parseAndCleanup("x != 42", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.not_equal, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: less than" {
    try parseAndCleanup("x < 10", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.less, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: greater than" {
    try parseAndCleanup("x > 10", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.greater, expr.binary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Logical Operator Tests
// ============================================================================

test "parser: logical and" {
    try parseAndCleanup("true and false", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.logical_and, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: logical or" {
    try parseAndCleanup("true or false", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.logical_or, expr.binary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Unary Expression Tests
// ============================================================================

test "parser: negation" {
    try parseAndCleanup("-42", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.unary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.UnaryOp.negate, expr.unary.operator);
            try std.testing.expectEqual(@as(i64, 42), expr.unary.operand.literal.value.integer);
        }
    }.test_fn);
}

test "parser: logical not" {
    try parseAndCleanup("not true", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.unary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.UnaryOp.logical_not, expr.unary.operator);
        }
    }.test_fn);
}

test "parser: bitwise not" {
    try parseAndCleanup("~42", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.unary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.UnaryOp.bitwise_not, expr.unary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Precedence Tests
// ============================================================================

test "parser: multiplication before addition" {
    try parseAndCleanup("1 + 2 * 3", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse as: 1 + (2 * 3)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.operator);
            
            // Left side should be 1
            try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr.binary.left.*));
            try std.testing.expectEqual(@as(i64, 1), expr.binary.left.literal.value.integer);
            
            // Right side should be (2 * 3)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.right.*));
            try std.testing.expectEqual(ast.BinaryOp.multiply, expr.binary.right.binary.operator);
        }
    }.test_fn);
}

test "parser: division before subtraction" {
    try parseAndCleanup("10 - 6 / 2", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse as: 10 - (6 / 2)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.subtract, expr.binary.operator);
            
            // Right side should be division
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.right.*));
            try std.testing.expectEqual(ast.BinaryOp.divide, expr.binary.right.binary.operator);
        }
    }.test_fn);
}

test "parser: comparison before logical" {
    try parseAndCleanup("x > 5 and y < 10", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse as: (x > 5) and (y < 10)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.logical_and, expr.binary.operator);
            
            // Both sides should be comparisons
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.left.*));
            try std.testing.expectEqual(ast.BinaryOp.greater, expr.binary.left.binary.operator);
            
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.right.*));
            try std.testing.expectEqual(ast.BinaryOp.less, expr.binary.right.binary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Grouping Tests
// ============================================================================

test "parser: parentheses override precedence" {
    try parseAndCleanup("(1 + 2) * 3", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse as: (1 + 2) * 3
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.multiply, expr.binary.operator);
            
            // Left side should be grouping with addition
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr.binary.left.*));
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.left.grouping.expression.*));
            try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.left.grouping.expression.binary.operator);
        }
    }.test_fn);
}

test "parser: nested parentheses" {
    try parseAndCleanup("((1 + 2) * 3)", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.grouping.expression.*));
        }
    }.test_fn);
}

// ============================================================================
// Postfix Expression Tests
// ============================================================================

test "parser: function call no args" {
    try parseAndCleanup("foo()", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqual(@as(usize, 0), expr.call.arguments.len);
            try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr.call.callee.*));
        }
    }.test_fn);
}

test "parser: function call with args" {
    try parseAndCleanup("add(1, 2)", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqual(@as(usize, 2), expr.call.arguments.len);
            try std.testing.expectEqual(@as(i64, 1), expr.call.arguments[0].literal.value.integer);
            try std.testing.expectEqual(@as(i64, 2), expr.call.arguments[1].literal.value.integer);
        }
    }.test_fn);
}

test "parser: array index" {
    try parseAndCleanup("arr[0]", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.index, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr.index.object.*));
            try std.testing.expectEqualStrings("arr", expr.index.object.identifier.name);
            try std.testing.expectEqual(@as(i64, 0), expr.index.index.literal.value.integer);
        }
    }.test_fn);
}

test "parser: member access" {
    try parseAndCleanup("obj.field", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr.member.object.*));
            try std.testing.expectEqualStrings("obj", expr.member.object.identifier.name);
            try std.testing.expectEqualStrings("field", expr.member.member);
        }
    }.test_fn);
}

test "parser: chained member access" {
    try parseAndCleanup("obj.field1.field2", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr));
            try std.testing.expectEqualStrings("field2", expr.member.member);
            
            // Object should be another member access
            try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr.member.object.*));
            try std.testing.expectEqualStrings("field1", expr.member.object.member.member);
        }
    }.test_fn);
}

test "parser: method call" {
    try parseAndCleanup("obj.method()", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr.call.callee.*));
        }
    }.test_fn);
}

test "parser: array index with expression" {
    try parseAndCleanup("arr[i + 1]", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.index, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.index.index.*));
            try std.testing.expectEqual(ast.BinaryOp.add, expr.index.index.binary.operator);
        }
    }.test_fn);
}

// ============================================================================
// Complex Expression Tests
// ============================================================================

test "parser: complex arithmetic" {
    try parseAndCleanup("1 + 2 * 3 - 4 / 2", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse as: (1 + (2 * 3)) - (4 / 2)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.subtract, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: nested function calls" {
    try parseAndCleanup("outer(inner(42))", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqualStrings("outer", expr.call.callee.identifier.name);
            try std.testing.expectEqual(@as(usize, 1), expr.call.arguments.len);
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr.call.arguments[0]));
        }
    }.test_fn);
}

test "parser: combined operators" {
    try parseAndCleanup("x > 5 and y < 10 or z == 0", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Should parse with correct precedence: ((x > 5) and (y < 10)) or (z == 0)
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.logical_or, expr.binary.operator);
        }
    }.test_fn);
}

test "parser: unary with binary" {
    try parseAndCleanup("-x + 5", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.operator);
            try std.testing.expectEqual(ast.Expr.unary, std.meta.activeTag(expr.binary.left.*));
            try std.testing.expectEqual(ast.UnaryOp.negate, expr.binary.left.unary.operator);
        }
    }.test_fn);
}

test "parser: not with comparison" {
    try parseAndCleanup("not (x == 5)", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.unary, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.UnaryOp.logical_not, expr.unary.operator);
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr.unary.operand.*));
        }
    }.test_fn);
}

// ============================================================================
// Edge Case Tests
// ============================================================================

test "parser: single identifier" {
    try parseAndCleanup("x", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr));
            try std.testing.expectEqualStrings("x", expr.identifier.name);
        }
    }.test_fn);
}

test "parser: deeply nested parentheses" {
    try parseAndCleanup("(((42)))", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr));
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr.grouping.expression.*));
            try std.testing.expectEqual(ast.Expr.grouping, std.meta.activeTag(expr.grouping.expression.grouping.expression.*));
        }
    }.test_fn);
}

test "parser: complex postfix chain" {
    try parseAndCleanup("obj.method(arg)[0].field", struct {
        fn test_fn(expr: ast.Expr) !void {
            // Final operation should be member access
            try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr));
            try std.testing.expectEqualStrings("field", expr.member.member);
            
            // Previous should be index
            try std.testing.expectEqual(ast.Expr.index, std.meta.activeTag(expr.member.object.*));
            
            // Previous should be call
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr.member.object.index.object.*));
        }
    }.test_fn);
}

test "parser: multiple function args" {
    try parseAndCleanup("func(1, 2, 3, 4)", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqual(@as(usize, 4), expr.call.arguments.len);
        }
    }.test_fn);
}

test "parser: expression as function arg" {
    try parseAndCleanup("func(1 + 2)", struct {
        fn test_fn(expr: ast.Expr) !void {
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(expr));
            try std.testing.expectEqual(@as(usize, 1), expr.call.arguments.len);
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.call.arguments[0]));
        }
    }.test_fn);
}

// ============================================================================
// Statement Tests (Day 3)
// ============================================================================

fn parseStmtAndCleanup(source: []const u8, comptime testFn: fn(ast.Stmt) anyerror!void) !void {
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var p = Parser.init(std.testing.allocator, tokens.items);
    const stmt = try p.parseStatement();
    defer stmt.deinit(std.testing.allocator);
    
    try testFn(stmt);
}

test "parser: var declaration without init" {
    try parseStmtAndCleanup("var x: Int", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.var_decl, std.meta.activeTag(stmt));
            try std.testing.expectEqualStrings("x", stmt.var_decl.name);
            try std.testing.expect(stmt.var_decl.type_annotation != null);
            try std.testing.expectEqualStrings("Int", stmt.var_decl.type_annotation.?.name);
            try std.testing.expect(stmt.var_decl.initializer == null);
        }
    }.test_fn);
}

test "parser: var declaration with init" {
    try parseStmtAndCleanup("var x = 42", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.var_decl, std.meta.activeTag(stmt));
            try std.testing.expectEqualStrings("x", stmt.var_decl.name);
            try std.testing.expect(stmt.var_decl.initializer != null);
            try std.testing.expectEqual(@as(i64, 42), stmt.var_decl.initializer.?.literal.value.integer);
        }
    }.test_fn);
}

test "parser: let declaration" {
    try parseStmtAndCleanup("let x = 42", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.let_decl, std.meta.activeTag(stmt));
            try std.testing.expectEqualStrings("x", stmt.let_decl.name);
            try std.testing.expectEqual(@as(i64, 42), stmt.let_decl.initializer.literal.value.integer);
        }
    }.test_fn);
}

test "parser: let with type annotation" {
    try parseStmtAndCleanup("let x: Int = 42", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.let_decl, std.meta.activeTag(stmt));
            try std.testing.expectEqualStrings("x", stmt.let_decl.name);
            try std.testing.expect(stmt.let_decl.type_annotation != null);
            try std.testing.expectEqualStrings("Int", stmt.let_decl.type_annotation.?.name);
        }
    }.test_fn);
}

test "parser: expression statement" {
    try parseStmtAndCleanup("42", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.expr, std.meta.activeTag(stmt));
            try std.testing.expectEqual(@as(i64, 42), stmt.expr.expression.literal.value.integer);
        }
    }.test_fn);
}

test "parser: return with value" {
    try parseStmtAndCleanup("return 42", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.return_stmt, std.meta.activeTag(stmt));
            try std.testing.expect(stmt.return_stmt.value != null);
            try std.testing.expectEqual(@as(i64, 42), stmt.return_stmt.value.?.literal.value.integer);
        }
    }.test_fn);
}

test "parser: return without value" {
    try parseStmtAndCleanup("return", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.return_stmt, std.meta.activeTag(stmt));
            try std.testing.expect(stmt.return_stmt.value == null);
        }
    }.test_fn);
}

test "parser: if statement" {
    try parseStmtAndCleanup("if x > 5:\n    return 1", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.if_stmt, std.meta.activeTag(stmt));
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(stmt.if_stmt.condition));
            try std.testing.expectEqual(ast.Stmt.return_stmt, std.meta.activeTag(stmt.if_stmt.then_branch.*));
            try std.testing.expect(stmt.if_stmt.else_branch == null);
        }
    }.test_fn);
}

test "parser: if-else statement" {
    try parseStmtAndCleanup("if x > 5:\n    return 1\nelse:\n    return 0", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.if_stmt, std.meta.activeTag(stmt));
            try std.testing.expect(stmt.if_stmt.else_branch != null);
            try std.testing.expectEqual(ast.Stmt.return_stmt, std.meta.activeTag(stmt.if_stmt.else_branch.?.*));
        }
    }.test_fn);
}

test "parser: while statement" {
    try parseStmtAndCleanup("while x < 10:\n    x = x + 1", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.while_stmt, std.meta.activeTag(stmt));
            try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(stmt.while_stmt.condition));
            try std.testing.expectEqual(ast.BinaryOp.less, stmt.while_stmt.condition.binary.operator);
        }
    }.test_fn);
}

test "parser: for statement" {
    try parseStmtAndCleanup("for i in range(10):\n    print(i)", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.for_stmt, std.meta.activeTag(stmt));
            try std.testing.expectEqualStrings("i", stmt.for_stmt.variable);
            try std.testing.expectEqual(ast.Expr.call, std.meta.activeTag(stmt.for_stmt.iterable));
        }
    }.test_fn);
}

test "parser: block statement" {
    try parseStmtAndCleanup("{\n    let x = 1\n    let y = 2\n}", struct {
        fn test_fn(stmt: ast.Stmt) !void {
            try std.testing.expectEqual(ast.Stmt.block, std.meta.activeTag(stmt));
            try std.testing.expectEqual(@as(usize, 2), stmt.block.statements.len);
            try std.testing.expectEqual(ast.Stmt.let_decl, std.meta.activeTag(stmt.block.statements[0]));
            try std.testing.expectEqual(ast.Stmt.let_decl, std.meta.activeTag(stmt.block.statements[1]));
        }
    }.test_fn);
}

// ============================================================================
// Declaration Tests (Day 4)
// ============================================================================

fn parseDeclAndCleanup(source: []const u8, comptime testFn: fn(ast.Decl) anyerror!void) !void {
    var lex = try Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var p = Parser.init(std.testing.allocator, tokens.items);
    const decl = try p.parseDeclaration();
    defer decl.deinit(std.testing.allocator);
    
    try testFn(decl);
}

test "parser: function with no params no return" {
    try parseDeclAndCleanup("fn main(): {\n    return\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.function, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("main", decl.function.name);
            try std.testing.expectEqual(@as(usize, 0), decl.function.parameters.len);
            try std.testing.expect(decl.function.return_type == null);
            try std.testing.expectEqual(@as(usize, 1), decl.function.body.statements.len);
        }
    }.test_fn);
}

test "parser: function with params" {
    try parseDeclAndCleanup("fn add(a: Int, b: Int): {\n    return a + b\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.function, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("add", decl.function.name);
            try std.testing.expectEqual(@as(usize, 2), decl.function.parameters.len);
            try std.testing.expectEqualStrings("a", decl.function.parameters[0].name);
            try std.testing.expectEqualStrings("Int", decl.function.parameters[0].type_annotation.name);
            try std.testing.expectEqualStrings("b", decl.function.parameters[1].name);
        }
    }.test_fn);
}

test "parser: function with return type" {
    try parseDeclAndCleanup("fn get_value() -> Int: {\n    return 42\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.function, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("get_value", decl.function.name);
            try std.testing.expect(decl.function.return_type != null);
            try std.testing.expectEqualStrings("Int", decl.function.return_type.?.name);
        }
    }.test_fn);
}

test "parser: function with ownership params" {
    try parseDeclAndCleanup("fn process(owned x: Int, borrowed y: Int, inout z: Int): {\n    return\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.function, std.meta.activeTag(decl));
            try std.testing.expectEqual(@as(usize, 3), decl.function.parameters.len);
            try std.testing.expectEqual(ast.Ownership.owned, decl.function.parameters[0].ownership);
            try std.testing.expectEqual(ast.Ownership.borrowed, decl.function.parameters[1].ownership);
            try std.testing.expectEqual(ast.Ownership.inout, decl.function.parameters[2].ownership);
        }
    }.test_fn);
}

test "parser: function with complex body" {
    try parseDeclAndCleanup("fn factorial(n: Int) -> Int: {\n    if n <= 1:\n        return 1\n    else:\n        return n\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.function, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("factorial", decl.function.name);
            try std.testing.expect(decl.function.body.statements.len > 0);
        }
    }.test_fn);
}

test "parser: struct with no fields" {
    try parseDeclAndCleanup("struct Empty: {\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.struct_decl, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("Empty", decl.struct_decl.name);
            try std.testing.expectEqual(@as(usize, 0), decl.struct_decl.fields.len);
        }
    }.test_fn);
}

test "parser: struct with fields" {
    try parseDeclAndCleanup("struct Point: {\n    x: Float\n    y: Float\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.struct_decl, std.meta.activeTag(decl));
            try std.testing.expectEqualStrings("Point", decl.struct_decl.name);
            try std.testing.expectEqual(@as(usize, 2), decl.struct_decl.fields.len);
            try std.testing.expectEqualStrings("x", decl.struct_decl.fields[0].name);
            try std.testing.expectEqualStrings("Float", decl.struct_decl.fields[0].type_annotation.name);
            try std.testing.expectEqualStrings("y", decl.struct_decl.fields[1].name);
        }
    }.test_fn);
}

test "parser: struct with multiple field types" {
    try parseDeclAndCleanup("struct Person: {\n    name: String\n    age: Int\n    height: Float\n    active: Bool\n}", struct {
        fn test_fn(decl: ast.Decl) !void {
            try std.testing.expectEqual(ast.Decl.struct_decl, std.meta.activeTag(decl));
            try std.testing.expectEqual(@as(usize, 4), decl.struct_decl.fields.len);
            try std.testing.expectEqualStrings("String", decl.struct_decl.fields[0].type_annotation.name);
            try std.testing.expectEqualStrings("Int", decl.struct_decl.fields[1].type_annotation.name);
            try std.testing.expectEqualStrings("Float", decl.struct_decl.fields[2].type_annotation.name);
            try std.testing.expectEqualStrings("Bool", decl.struct_decl.fields[3].type_annotation.name);
        }
    }.test_fn);
}
