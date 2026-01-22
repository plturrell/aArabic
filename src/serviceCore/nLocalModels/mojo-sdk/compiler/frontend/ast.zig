// Mojo SDK - Abstract Syntax Tree (AST)
// Day 2: AST node definitions for the Mojo language

const std = @import("std");
const Token = @import("lexer").Token;

// ============================================================================
// AST Node Types
// ============================================================================

/// Base type for all AST nodes
pub const NodeType = enum {
    // Expressions
    binary_expr,
    unary_expr,
    literal_expr,
    identifier_expr,
    call_expr,
    index_expr,
    member_expr,
    grouping_expr,
    assignment_expr,
    
    // Statements
    expr_stmt,
    var_decl_stmt,
    let_decl_stmt,
    if_stmt,
    while_stmt,
    for_stmt,
    return_stmt,
    block_stmt,
    
    // Declarations
    function_decl,
    struct_decl,
    trait_decl,
    impl_decl,
    
    // Types
    type_ref,
    generic_type,
};

// ============================================================================
// Type References
// ============================================================================

pub const TypeRef = struct {
    name: []const u8,
    generic_args: ?[]TypeRef = null,
    token: Token,
    
    pub fn init(name: []const u8, token: Token) TypeRef {
        return TypeRef{
            .name = name,
            .token = token,
        };
    }
};

// ============================================================================
// Expressions
// ============================================================================

pub const BinaryOp = enum {
    // Arithmetic
    add,        // +
    subtract,   // -
    multiply,   // *
    divide,     // /
    modulo,     // %
    power,      // **
    
    // Comparison
    equal,      // ==
    not_equal,  // !=
    less,       // <
    less_equal, // <=
    greater,    // >
    greater_equal, // >=
    
    // Logical
    logical_and, // and
    logical_or,  // or
    
    // Bitwise
    bitwise_and, // &
    bitwise_or,  // |
    bitwise_xor, // ^
    left_shift,  // <<
    right_shift, // >>
};

pub const UnaryOp = enum {
    negate,     // -
    logical_not, // not
    bitwise_not, // ~
};

pub const Expr = union(enum) {
    binary: BinaryExpr,
    unary: UnaryExpr,
    literal: LiteralExpr,
    identifier: IdentifierExpr,
    call: CallExpr,
    index: IndexExpr,
    member: MemberExpr,
    grouping: GroupingExpr,
    assignment: AssignmentExpr,
    
    pub fn getToken(self: Expr) Token {
        return switch (self) {
            .binary => |e| e.operator_token,
            .unary => |e| e.operator_token,
            .literal => |e| e.token,
            .identifier => |e| e.token,
            .call => |e| e.callee.*.getToken(),
            .index => |e| e.object.*.getToken(),
            .member => |e| e.object.*.getToken(),
            .grouping => |e| e.expression.*.getToken(),
            .assignment => |e| e.target.*.getToken(),
        };
    }
    
    /// Recursively free all memory allocated for this expression
    pub fn deinit(self: Expr, allocator: std.mem.Allocator) void {
        switch (self) {
            .binary => |e| {
                e.left.deinit(allocator);
                allocator.destroy(e.left);
                e.right.deinit(allocator);
                allocator.destroy(e.right);
            },
            .unary => |e| {
                e.operand.deinit(allocator);
                allocator.destroy(e.operand);
            },
            .call => |e| {
                e.callee.deinit(allocator);
                allocator.destroy(e.callee);
                for (e.arguments) |arg| {
                    arg.deinit(allocator);
                }
                allocator.free(e.arguments);
            },
            .index => |e| {
                e.object.deinit(allocator);
                allocator.destroy(e.object);
                e.index.deinit(allocator);
                allocator.destroy(e.index);
            },
            .member => |e| {
                e.object.deinit(allocator);
                allocator.destroy(e.object);
            },
            .grouping => |e| {
                e.expression.deinit(allocator);
                allocator.destroy(e.expression);
            },
            .assignment => |e| {
                e.target.deinit(allocator);
                allocator.destroy(e.target);
                e.value.deinit(allocator);
                allocator.destroy(e.value);
            },
            .literal, .identifier => {
                // No allocated memory to free
            },
        }
    }
};

pub const BinaryExpr = struct {
    left: *Expr,
    operator: BinaryOp,
    operator_token: Token,
    right: *Expr,
};

pub const UnaryExpr = struct {
    operator: UnaryOp,
    operator_token: Token,
    operand: *Expr,
};

pub const LiteralExpr = struct {
    value: LiteralValue,
    token: Token,
};

pub const LiteralValue = union(enum) {
    integer: i64,
    float: f64,
    string: []const u8,
    boolean: bool,
    nil,
};

pub const IdentifierExpr = struct {
    name: []const u8,
    token: Token,
};

pub const CallExpr = struct {
    callee: *Expr,
    arguments: []Expr,
    token: Token, // The '(' token
};

pub const IndexExpr = struct {
    object: *Expr,
    index: *Expr,
    token: Token, // The '[' token
};

pub const MemberExpr = struct {
    object: *Expr,
    member: []const u8,
    token: Token, // The '.' token
};

pub const GroupingExpr = struct {
    expression: *Expr,
};

pub const AssignmentExpr = struct {
    target: *Expr, // Left-hand side (must be lvalue)
    value: *Expr,  // Right-hand side
    token: Token,  // The '=' token
};

// ============================================================================
// Statements
// ============================================================================

pub const Stmt = union(enum) {
    expr: ExprStmt,
    var_decl: VarDeclStmt,
    let_decl: LetDeclStmt,
    if_stmt: IfStmt,
    while_stmt: WhileStmt,
    for_stmt: ForStmt,
    return_stmt: ReturnStmt,
    block: BlockStmt,
    
    /// Recursively free all memory allocated for this statement
    pub fn deinit(self: Stmt, allocator: std.mem.Allocator) void {
        switch (self) {
            .expr => |s| {
                s.expression.deinit(allocator);
            },
            .var_decl => |s| {
                if (s.initializer) |init| {
                    init.deinit(allocator);
                }
            },
            .let_decl => |s| {
                s.initializer.deinit(allocator);
            },
            .if_stmt => |s| {
                s.condition.deinit(allocator);
                s.then_branch.deinit(allocator);
                allocator.destroy(s.then_branch);
                if (s.else_branch) |else_branch| {
                    else_branch.deinit(allocator);
                    allocator.destroy(else_branch);
                }
            },
            .while_stmt => |s| {
                s.condition.deinit(allocator);
                s.body.deinit(allocator);
                allocator.destroy(s.body);
            },
            .for_stmt => |s| {
                s.iterable.deinit(allocator);
                s.body.deinit(allocator);
                allocator.destroy(s.body);
            },
            .return_stmt => |s| {
                if (s.value) |val| {
                    val.deinit(allocator);
                }
            },
            .block => |s| {
                for (s.statements) |stmt| {
                    stmt.deinit(allocator);
                }
                allocator.free(s.statements);
            },
        }
    }
};

pub const ExprStmt = struct {
    expression: Expr,
};

pub const VarDeclStmt = struct {
    name: []const u8,
    name_token: Token,
    type_annotation: ?TypeRef,
    initializer: ?Expr,
};

pub const LetDeclStmt = struct {
    name: []const u8,
    name_token: Token,
    type_annotation: ?TypeRef,
    initializer: Expr,
};

pub const IfStmt = struct {
    condition: Expr,
    then_branch: *Stmt,
    else_branch: ?*Stmt,
    token: Token,
};

pub const WhileStmt = struct {
    condition: Expr,
    body: *Stmt,
    token: Token,
};

pub const ForStmt = struct {
    variable: []const u8,
    variable_token: Token,
    iterable: Expr,
    body: *Stmt,
    token: Token,
};

pub const ReturnStmt = struct {
    value: ?Expr,
    token: Token,
};

pub const BlockStmt = struct {
    statements: []Stmt,
    token: Token,
};

// ============================================================================
// Declarations
// ============================================================================

pub const Decl = union(enum) {
    function: FunctionDecl,
    struct_decl: StructDecl,
    trait_decl: TraitDecl,
    impl_decl: ImplDecl,
    
    /// Recursively free all memory allocated for this declaration
    pub fn deinit(self: Decl, allocator: std.mem.Allocator) void {
        switch (self) {
            .function => |f| {
                allocator.free(f.parameters);
                for (f.body.statements) |stmt| {
                    stmt.deinit(allocator);
                }
                allocator.free(f.body.statements);
            },
            .struct_decl => |s| {
                allocator.free(s.fields);
            },
            .trait_decl => |t| {
                for (t.methods) |method| {
                    allocator.free(method.parameters);
                    for (method.body.statements) |stmt| {
                        stmt.deinit(allocator);
                    }
                    allocator.free(method.body.statements);
                }
                allocator.free(t.methods);
            },
            .impl_decl => |i| {
                for (i.methods) |method| {
                    allocator.free(method.parameters);
                    for (method.body.statements) |stmt| {
                        stmt.deinit(allocator);
                    }
                    allocator.free(method.body.statements);
                }
                allocator.free(i.methods);
            },
        }
    }
};

pub const Parameter = struct {
    name: []const u8,
    name_token: Token,
    type_annotation: TypeRef,
    ownership: Ownership,
};

pub const Ownership = enum {
    owned,
    borrowed,
    inout,
    default,
};

pub const FunctionDecl = struct {
    name: []const u8,
    name_token: Token,
    parameters: []Parameter,
    return_type: ?TypeRef,
    body: BlockStmt,
};

pub const FieldDecl = struct {
    name: []const u8,
    name_token: Token,
    type_annotation: TypeRef,
};

pub const StructDecl = struct {
    name: []const u8,
    name_token: Token,
    fields: []FieldDecl,
};

pub const TraitDecl = struct {
    name: []const u8,
    name_token: Token,
    methods: []FunctionDecl,
};

pub const ImplDecl = struct {
    trait_name: []const u8,
    type_name: []const u8,
    methods: []FunctionDecl,
    token: Token,
};

// ============================================================================
// Program (Root Node)
// ============================================================================

pub const Program = struct {
    declarations: []Decl,
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, declarations: []Decl) Program {
        return Program{
            .declarations = declarations,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Program) void {
        // Free all allocations
        self.allocator.free(self.declarations);
    }
};

// ============================================================================
// AST Visitor Pattern (for traversal)
// ============================================================================

pub const Visitor = struct {
    visitExpr: *const fn (*Visitor, Expr) anyerror!void,
    visitStmt: *const fn (*Visitor, Stmt) anyerror!void,
    visitDecl: *const fn (*Visitor, Decl) anyerror!void,
};
