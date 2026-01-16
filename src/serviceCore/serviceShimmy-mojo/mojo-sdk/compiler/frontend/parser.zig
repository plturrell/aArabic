// Mojo SDK - Parser
// Day 2: Recursive descent parser for Mojo language

const std = @import("std");
const lexer = @import("lexer");
const ast = @import("ast");

const Token = lexer.Token;
const TokenType = lexer.TokenType;

// ============================================================================
// Parser
// ============================================================================

pub const Parser = struct {
    tokens: []Token,
    current: usize = 0,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, tokens: []Token) Parser {
        return Parser{
            .tokens = tokens,
            .allocator = allocator,
        };
    }
    
    // ========================================================================
    // Helper Methods
    // ========================================================================
    
    fn peek(self: *Parser) Token {
        if (self.current < self.tokens.len) {
            return self.tokens[self.current];
        }
        return self.tokens[self.tokens.len - 1]; // Return EOF
    }
    
    fn advance(self: *Parser) Token {
        const token = self.peek();
        if (self.current < self.tokens.len) {
            self.current += 1;
        }
        return token;
    }
    
    fn check(self: *Parser, token_type: TokenType) bool {
        return self.peek().type == token_type;
    }
    
    fn match(self: *Parser, types: []const TokenType) bool {
        for (types) |t| {
            if (self.check(t)) {
                _ = self.advance();
                return true;
            }
        }
        return false;
    }
    
    fn consume(self: *Parser, token_type: TokenType, message: []const u8) !Token {
        if (self.check(token_type)) {
            return self.advance();
        }
        std.debug.print("Parse error at {s}: {s}\n", .{self.peek().lexeme, message});
        return error.ParseError;
    }
    
    fn skipNewlines(self: *Parser) void {
        while (self.check(.newline)) {
            _ = self.advance();
        }
    }
    
    // ========================================================================
    // Expression Parsing
    // ========================================================================
    
    pub fn parseExpression(self: *Parser) error{OutOfMemory, ParseError, UnexpectedToken, InvalidCharacter, Overflow}!ast.Expr {
        return try self.parseAssignment();
    }
    
    fn parseAssignment(self: *Parser) !ast.Expr {
        const expr = try self.parseLogicalOr();
        
        // Check for assignment operator
        if (self.match(&[_]TokenType{.equal})) {
            const equals_token = self.tokens[self.current - 1];
            const value = try self.parseAssignment(); // Right-associative
            
            // Validate that left side is an lvalue (identifier, index, or member)
            switch (expr) {
                .identifier, .index, .member => {
                    const target_ptr = try self.allocator.create(ast.Expr);
                    target_ptr.* = expr;
                    
                    const value_ptr = try self.allocator.create(ast.Expr);
                    value_ptr.* = value;
                    
                    return ast.Expr{
                        .assignment = ast.AssignmentExpr{
                            .target = target_ptr,
                            .value = value_ptr,
                            .token = equals_token,
                        },
                    };
                },
                else => {
                    std.debug.print("Invalid assignment target\n", .{});
                    return error.ParseError;
                },
            }
        }
        
        return expr;
    }
    
    fn parseLogicalOr(self: *Parser) !ast.Expr {
        var expr = try self.parseLogicalAnd();
        
        while (self.match(&[_]TokenType{.or_keyword})) {
            const operator_token = self.tokens[self.current - 1];
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseLogicalAnd();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = .logical_or,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseLogicalAnd(self: *Parser) !ast.Expr {
        var expr = try self.parseEquality();
        
        while (self.match(&[_]TokenType{.and_keyword})) {
            const operator_token = self.tokens[self.current - 1];
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseEquality();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = .logical_and,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseEquality(self: *Parser) !ast.Expr {
        var expr = try self.parseComparison();
        
        while (self.match(&[_]TokenType{.equal_equal, .not_equal})) {
            const operator_token = self.tokens[self.current - 1];
            const op: ast.BinaryOp = if (operator_token.type == .equal_equal) .equal else .not_equal;
            
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseComparison();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = op,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseComparison(self: *Parser) !ast.Expr {
        var expr = try self.parseTerm();
        
        while (self.match(&[_]TokenType{.less, .less_equal, .greater, .greater_equal})) {
            const operator_token = self.tokens[self.current - 1];
            const op: ast.BinaryOp = switch (operator_token.type) {
                .less => .less,
                .less_equal => .less_equal,
                .greater => .greater,
                .greater_equal => .greater_equal,
                else => unreachable,
            };
            
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseTerm();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = op,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseTerm(self: *Parser) !ast.Expr {
        var expr = try self.parseFactor();
        
        while (self.match(&[_]TokenType{.plus, .minus})) {
            const operator_token = self.tokens[self.current - 1];
            const op: ast.BinaryOp = if (operator_token.type == .plus) .add else .subtract;
            
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseFactor();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = op,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseFactor(self: *Parser) !ast.Expr {
        var expr = try self.parseUnary();
        
        while (self.match(&[_]TokenType{.star, .slash, .percent})) {
            const operator_token = self.tokens[self.current - 1];
            const op: ast.BinaryOp = switch (operator_token.type) {
                .star => .multiply,
                .slash => .divide,
                .percent => .modulo,
                else => unreachable,
            };
            
            const right_ptr = try self.allocator.create(ast.Expr);
            right_ptr.* = try self.parseUnary();
            
            const left_ptr = try self.allocator.create(ast.Expr);
            left_ptr.* = expr;
            
            expr = ast.Expr{
                .binary = ast.BinaryExpr{
                    .left = left_ptr,
                    .operator = op,
                    .operator_token = operator_token,
                    .right = right_ptr,
                },
            };
        }
        
        return expr;
    }
    
    fn parseUnary(self: *Parser) !ast.Expr {
        if (self.match(&[_]TokenType{.minus, .not_keyword, .tilde})) {
            const operator_token = self.tokens[self.current - 1];
            const op: ast.UnaryOp = switch (operator_token.type) {
                .minus => .negate,
                .not_keyword => .logical_not,
                .tilde => .bitwise_not,
                else => unreachable,
            };
            
            const operand_ptr = try self.allocator.create(ast.Expr);
            operand_ptr.* = try self.parseUnary();
            
            return ast.Expr{
                .unary = ast.UnaryExpr{
                    .operator = op,
                    .operator_token = operator_token,
                    .operand = operand_ptr,
                },
            };
        }
        
        return try self.parsePostfix();
    }
    
    fn parsePostfix(self: *Parser) !ast.Expr {
        var expr = try self.parsePrimary();
        
        while (true) {
            if (self.match(&[_]TokenType{.left_paren})) {
                // Function call
                const paren_token = self.tokens[self.current - 1];
                var arguments = try std.ArrayList(ast.Expr).initCapacity(self.allocator, 4);
                
                if (!self.check(.right_paren)) {
                    while (true) {
                        try arguments.append(self.allocator, try self.parseExpression());
                        if (!self.match(&[_]TokenType{.comma})) break;
                    }
                }
                
                _ = try self.consume(.right_paren, "Expected ')' after arguments");
                
                const callee_ptr = try self.allocator.create(ast.Expr);
                callee_ptr.* = expr;
                
                expr = ast.Expr{
                    .call = ast.CallExpr{
                        .callee = callee_ptr,
                        .arguments = try arguments.toOwnedSlice(self.allocator),
                        .token = paren_token,
                    },
                };
            } else if (self.match(&[_]TokenType{.left_bracket})) {
                // Index access
                const bracket_token = self.tokens[self.current - 1];
                const index_expr = try self.parseExpression();
                _ = try self.consume(.right_bracket, "Expected ']' after index");
                
                const object_ptr = try self.allocator.create(ast.Expr);
                object_ptr.* = expr;
                
                const index_ptr = try self.allocator.create(ast.Expr);
                index_ptr.* = index_expr;
                
                expr = ast.Expr{
                    .index = ast.IndexExpr{
                        .object = object_ptr,
                        .index = index_ptr,
                        .token = bracket_token,
                    },
                };
            } else if (self.match(&[_]TokenType{.dot})) {
                // Member access
                const dot_token = self.tokens[self.current - 1];
                const member_token = try self.consume(.identifier, "Expected member name after '.'");
                
                const object_ptr = try self.allocator.create(ast.Expr);
                object_ptr.* = expr;
                
                expr = ast.Expr{
                    .member = ast.MemberExpr{
                        .object = object_ptr,
                        .member = member_token.lexeme,
                        .token = dot_token,
                    },
                };
            } else {
                break;
            }
        }
        
        return expr;
    }
    
    fn parsePrimary(self: *Parser) !ast.Expr {
        const token = self.peek();
        
        switch (token.type) {
            .integer_literal => {
                const tok = self.advance();
                const value = try std.fmt.parseInt(i64, tok.lexeme, 10);
                return ast.Expr{
                    .literal = ast.LiteralExpr{
                        .value = ast.LiteralValue{ .integer = value },
                        .token = tok,
                    },
                };
            },
            
            .float_literal => {
                const tok = self.advance();
                const value = try std.fmt.parseFloat(f64, tok.lexeme);
                return ast.Expr{
                    .literal = ast.LiteralExpr{
                        .value = ast.LiteralValue{ .float = value },
                        .token = tok,
                    },
                };
            },
            
            .string_literal => {
                const tok = self.advance();
                const str_value = tok.lexeme[1..tok.lexeme.len-1];
                return ast.Expr{
                    .literal = ast.LiteralExpr{
                        .value = ast.LiteralValue{ .string = str_value },
                        .token = tok,
                    },
                };
            },
            
            .true_literal => {
                const tok = self.advance();
                return ast.Expr{
                    .literal = ast.LiteralExpr{
                        .value = ast.LiteralValue{ .boolean = true },
                        .token = tok,
                    },
                };
            },
            
            .false_literal => {
                const tok = self.advance();
                return ast.Expr{
                    .literal = ast.LiteralExpr{
                        .value = ast.LiteralValue{ .boolean = false },
                        .token = tok,
                    },
                };
            },
            
            .identifier => {
                const tok = self.advance();
                return ast.Expr{
                    .identifier = ast.IdentifierExpr{
                        .name = tok.lexeme,
                        .token = tok,
                    },
                };
            },
            
            .left_paren => {
                _ = self.advance();
                const expr = try self.parseExpression();
                _ = try self.consume(.right_paren, "Expected ')' after expression");
                
                const expr_ptr = try self.allocator.create(ast.Expr);
                expr_ptr.* = expr;
                
                return ast.Expr{
                    .grouping = ast.GroupingExpr{
                        .expression = expr_ptr,
                    },
                };
            },
            
            else => {
                std.debug.print("Unexpected token: {s}\n", .{@tagName(token.type)});
                return error.UnexpectedToken;
            },
        }
    }
    
    // ========================================================================
    // Statement Parsing (Day 3)
    // ========================================================================
    
    pub fn parseStatement(self: *Parser) error{OutOfMemory, ParseError, UnexpectedToken, InvalidCharacter, Overflow}!ast.Stmt {
        self.skipNewlines();
        
        // Variable declarations
        if (self.match(&[_]TokenType{.var_keyword})) {
            return try self.parseVarDecl();
        }
        if (self.match(&[_]TokenType{.let_keyword})) {
            return try self.parseLetDecl();
        }
        
        // Control flow
        if (self.match(&[_]TokenType{.if_keyword})) {
            return try self.parseIfStmt();
        }
        if (self.match(&[_]TokenType{.while_keyword})) {
            return try self.parseWhileStmt();
        }
        if (self.match(&[_]TokenType{.for_keyword})) {
            return try self.parseForStmt();
        }
        if (self.match(&[_]TokenType{.return_keyword})) {
            return try self.parseReturnStmt();
        }
        
        // Block
        if (self.check(.left_brace)) {
            return try self.parseBlockStmt();
        }
        
        // Expression statement
        return try self.parseExprStmt();
    }
    
    fn parseVarDecl(self: *Parser) !ast.Stmt {
        const name_token = try self.consume(.identifier, "Expected variable name");
        
        var type_annotation: ?ast.TypeRef = null;
        if (self.match(&[_]TokenType{.colon})) {
            const type_token = self.advance();
            if (type_token.type != .identifier and 
                type_token.type != .int_type and 
                type_token.type != .float_type and 
                type_token.type != .bool_type and 
                type_token.type != .string_type) {
                return error.ParseError;
            }
            type_annotation = ast.TypeRef.init(type_token.lexeme, type_token);
        }
        
        var initializer: ?ast.Expr = null;
        if (self.match(&[_]TokenType{.equal})) {
            initializer = try self.parseExpression();
        }
        
        self.skipNewlines();
        
        return ast.Stmt{
            .var_decl = ast.VarDeclStmt{
                .name = name_token.lexeme,
                .name_token = name_token,
                .type_annotation = type_annotation,
                .initializer = initializer,
            },
        };
    }
    
    fn parseLetDecl(self: *Parser) !ast.Stmt {
        const name_token = try self.consume(.identifier, "Expected variable name");
        
        var type_annotation: ?ast.TypeRef = null;
        if (self.match(&[_]TokenType{.colon})) {
            const type_token = self.advance();
            if (type_token.type != .identifier and 
                type_token.type != .int_type and 
                type_token.type != .float_type and 
                type_token.type != .bool_type and 
                type_token.type != .string_type) {
                return error.ParseError;
            }
            type_annotation = ast.TypeRef.init(type_token.lexeme, type_token);
        }
        
        _ = try self.consume(.equal, "let declaration requires initializer");
        const initializer = try self.parseExpression();
        
        self.skipNewlines();
        
        return ast.Stmt{
            .let_decl = ast.LetDeclStmt{
                .name = name_token.lexeme,
                .name_token = name_token,
                .type_annotation = type_annotation,
                .initializer = initializer,
            },
        };
    }
    
    fn parseIfStmt(self: *Parser) !ast.Stmt {
        const if_token = self.tokens[self.current - 1];
        
        const condition = try self.parseExpression();
        _ = try self.consume(.colon, "Expected ':' after if condition");
        self.skipNewlines();
        
        const then_ptr = try self.allocator.create(ast.Stmt);
        then_ptr.* = try self.parseStatement();
        
        var else_ptr: ?*ast.Stmt = null;
        if (self.match(&[_]TokenType{.else_keyword})) {
            _ = try self.consume(.colon, "Expected ':' after else");
            self.skipNewlines();
            else_ptr = try self.allocator.create(ast.Stmt);
            else_ptr.?.* = try self.parseStatement();
        }
        
        return ast.Stmt{
            .if_stmt = ast.IfStmt{
                .condition = condition,
                .then_branch = then_ptr,
                .else_branch = else_ptr,
                .token = if_token,
            },
        };
    }
    
    fn parseWhileStmt(self: *Parser) !ast.Stmt {
        const while_token = self.tokens[self.current - 1];
        
        const condition = try self.parseExpression();
        _ = try self.consume(.colon, "Expected ':' after while condition");
        self.skipNewlines();
        
        const body_ptr = try self.allocator.create(ast.Stmt);
        body_ptr.* = try self.parseStatement();
        
        return ast.Stmt{
            .while_stmt = ast.WhileStmt{
                .condition = condition,
                .body = body_ptr,
                .token = while_token,
            },
        };
    }
    
    fn parseForStmt(self: *Parser) !ast.Stmt {
        const for_token = self.tokens[self.current - 1];
        
        const var_token = try self.consume(.identifier, "Expected variable name in for loop");
        _ = try self.consume(.in_keyword, "Expected 'in' after for variable");
        
        const iterable = try self.parseExpression();
        _ = try self.consume(.colon, "Expected ':' after for iterable");
        self.skipNewlines();
        
        const body_ptr = try self.allocator.create(ast.Stmt);
        body_ptr.* = try self.parseStatement();
        
        return ast.Stmt{
            .for_stmt = ast.ForStmt{
                .variable = var_token.lexeme,
                .variable_token = var_token,
                .iterable = iterable,
                .body = body_ptr,
                .token = for_token,
            },
        };
    }
    
    fn parseReturnStmt(self: *Parser) !ast.Stmt {
        const return_token = self.tokens[self.current - 1];
        
        var value: ?ast.Expr = null;
        if (!self.check(.newline) and !self.check(.eof)) {
            value = try self.parseExpression();
        }
        
        self.skipNewlines();
        
        return ast.Stmt{
            .return_stmt = ast.ReturnStmt{
                .value = value,
                .token = return_token,
            },
        };
    }
    
    fn parseBlockStmt(self: *Parser) !ast.Stmt {
        const brace_token = try self.consume(.left_brace, "Expected '{'");
        self.skipNewlines();
        
        var statements = try std.ArrayList(ast.Stmt).initCapacity(self.allocator, 8);
        
        while (!self.check(.right_brace) and !self.check(.eof)) {
            try statements.append(self.allocator, try self.parseStatement());
            self.skipNewlines();
        }
        
        _ = try self.consume(.right_brace, "Expected '}' after block");
        
        return ast.Stmt{
            .block = ast.BlockStmt{
                .statements = try statements.toOwnedSlice(self.allocator),
                .token = brace_token,
            },
        };
    }
    
    fn parseExprStmt(self: *Parser) !ast.Stmt {
        const expression = try self.parseExpression();
        self.skipNewlines();
        
        return ast.Stmt{
            .expr = ast.ExprStmt{
                .expression = expression,
            },
        };
    }
    
    // ========================================================================
    // Declaration Parsing (Day 4)
    // ========================================================================
    
    pub fn parseDeclaration(self: *Parser) error{OutOfMemory, ParseError, UnexpectedToken, InvalidCharacter, Overflow}!ast.Decl {
        self.skipNewlines();
        
        if (self.match(&[_]TokenType{.fn_keyword})) {
            return try self.parseFunctionDecl();
        }
        
        if (self.match(&[_]TokenType{.struct_keyword})) {
            return try self.parseStructDecl();
        }
        
        return error.ParseError;
    }
    
    fn parseFunctionDecl(self: *Parser) !ast.Decl {
        const name_token = try self.consume(.identifier, "Expected function name");
        
        // Parse parameters
        _ = try self.consume(.left_paren, "Expected '(' after function name");
        var parameters = try std.ArrayList(ast.Parameter).initCapacity(self.allocator, 4);
        
        if (!self.check(.right_paren)) {
            while (true) {
                // Parse ownership keyword (optional)
                var ownership = ast.Ownership.default;
                if (self.match(&[_]TokenType{.owned_keyword})) {
                    ownership = .owned;
                } else if (self.match(&[_]TokenType{.borrowed_keyword})) {
                    ownership = .borrowed;
                } else if (self.match(&[_]TokenType{.inout_keyword})) {
                    ownership = .inout;
                }
                
                const param_name_token = try self.consume(.identifier, "Expected parameter name");
                _ = try self.consume(.colon, "Expected ':' after parameter name");
                
                const type_token = self.advance();
                if (type_token.type != .identifier and 
                    type_token.type != .int_type and 
                    type_token.type != .float_type and 
                    type_token.type != .bool_type and 
                    type_token.type != .string_type) {
                    return error.ParseError;
                }
                
                try parameters.append(self.allocator, ast.Parameter{
                    .name = param_name_token.lexeme,
                    .name_token = param_name_token,
                    .type_annotation = ast.TypeRef.init(type_token.lexeme, type_token),
                    .ownership = ownership,
                });
                
                if (!self.match(&[_]TokenType{.comma})) break;
            }
        }
        
        _ = try self.consume(.right_paren, "Expected ')' after parameters");
        
        // Parse return type (optional)
        var return_type: ?ast.TypeRef = null;
        if (self.match(&[_]TokenType{.arrow})) {
            const type_token = self.advance();
            if (type_token.type != .identifier and 
                type_token.type != .int_type and 
                type_token.type != .float_type and 
                type_token.type != .bool_type and 
                type_token.type != .string_type) {
                return error.ParseError;
            }
            return_type = ast.TypeRef.init(type_token.lexeme, type_token);
        }
        
        _ = try self.consume(.colon, "Expected ':' before function body");
        self.skipNewlines();
        
        // Parse body
        const body_stmt = try self.parseBlockStmt();
        
        return ast.Decl{
            .function = ast.FunctionDecl{
                .name = name_token.lexeme,
                .name_token = name_token,
                .parameters = try parameters.toOwnedSlice(self.allocator),
                .return_type = return_type,
                .body = body_stmt.block,
            },
        };
    }
    
    fn parseStructDecl(self: *Parser) !ast.Decl {
        const name_token = try self.consume(.identifier, "Expected struct name");
        _ = try self.consume(.colon, "Expected ':' after struct name");
        self.skipNewlines();
        _ = try self.consume(.left_brace, "Expected '{' to start struct body");
        self.skipNewlines();
        
        var fields = try std.ArrayList(ast.FieldDecl).initCapacity(self.allocator, 4);
        
        while (!self.check(.right_brace) and !self.check(.eof)) {
            const field_name_token = try self.consume(.identifier, "Expected field name");
            _ = try self.consume(.colon, "Expected ':' after field name");
            
            const type_token = self.advance();
            if (type_token.type != .identifier and 
                type_token.type != .int_type and 
                type_token.type != .float_type and 
                type_token.type != .bool_type and 
                type_token.type != .string_type) {
                return error.ParseError;
            }
            
            try fields.append(self.allocator, ast.FieldDecl{
                .name = field_name_token.lexeme,
                .name_token = field_name_token,
                .type_annotation = ast.TypeRef.init(type_token.lexeme, type_token),
            });
            
            self.skipNewlines();
        }
        
        _ = try self.consume(.right_brace, "Expected '}' after struct body");
        
        return ast.Decl{
            .struct_decl = ast.StructDecl{
                .name = name_token.lexeme,
                .name_token = name_token,
                .fields = try fields.toOwnedSlice(self.allocator),
            },
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "parser: simple integer" {
    const source = "42";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    const expr = try parser.parseExpression();
    
    try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr));
    try std.testing.expectEqual(@as(i64, 42), expr.literal.value.integer);
}

test "parser: binary addition" {
    const source = "1 + 2";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    const expr = try parser.parseExpression();
    
    try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.operator);
}

test "parser: operator precedence" {
    const source = "1 + 2 * 3";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    const expr = try parser.parseExpression();
    
    // Should parse as: 1 + (2 * 3)
    try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.BinaryOp.add, expr.binary.operator);
    try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.binary.right.*));
    try std.testing.expectEqual(ast.BinaryOp.multiply, expr.binary.right.binary.operator);
}

test "parser: simple assignment" {
    const source = "x = 42";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    var expr = try parser.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.Expr.identifier, std.meta.activeTag(expr.assignment.target.*));
    try std.testing.expectEqualStrings("x", expr.assignment.target.identifier.name);
    try std.testing.expectEqual(ast.Expr.literal, std.meta.activeTag(expr.assignment.value.*));
    try std.testing.expectEqual(@as(i64, 42), expr.assignment.value.literal.value.integer);
}

test "parser: assignment to member" {
    const source = "obj.field = 10";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    var expr = try parser.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.Expr.member, std.meta.activeTag(expr.assignment.target.*));
    try std.testing.expectEqualStrings("field", expr.assignment.target.member.member);
}

test "parser: assignment to index" {
    const source = "arr[0] = 5";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    var expr = try parser.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.Expr.index, std.meta.activeTag(expr.assignment.target.*));
    try std.testing.expectEqualStrings("arr", expr.assignment.target.index.object.identifier.name);
}

test "parser: chained assignment" {
    const source = "x = y = 1";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    var expr = try parser.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    // Should parse as: x = (y = 1) - right associative
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr));
    try std.testing.expectEqualStrings("x", expr.assignment.target.identifier.name);
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr.assignment.value.*));
    try std.testing.expectEqualStrings("y", expr.assignment.value.assignment.target.identifier.name);
}

test "parser: assignment with expression" {
    const source = "x = 1 + 2";
    var lex = try lexer.Lexer.init(std.testing.allocator, source);
    defer lex.deinit();
    
    var tokens = try lex.scanAll();
    defer tokens.deinit(std.testing.allocator);
    
    var parser = Parser.init(std.testing.allocator, tokens.items);
    var expr = try parser.parseExpression();
    defer expr.deinit(std.testing.allocator);
    
    try std.testing.expectEqual(ast.Expr.assignment, std.meta.activeTag(expr));
    try std.testing.expectEqual(ast.Expr.binary, std.meta.activeTag(expr.assignment.value.*));
    try std.testing.expectEqual(ast.BinaryOp.add, expr.assignment.value.binary.operator);
}
