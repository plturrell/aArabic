// Mojo SDK - Semantic Analyzer
// Day 6: Type checking and name resolution

const std = @import("std");
const ast = @import("ast");
const symbol_table = @import("symbol_table");
const borrow_integration = @import("borrow_integration.zig");
const borrow_checker_mod = @import("borrow_checker.zig");
const lifetimes = @import("lifetimes.zig");

const SymbolTable = symbol_table.SymbolTable;
const Symbol = symbol_table.Symbol;
const SymbolKind = symbol_table.SymbolKind;
const IntegratedBorrowChecker = borrow_integration.IntegratedBorrowChecker;
const Borrow = borrow_checker_mod.Borrow;
const Move = IntegratedBorrowChecker.Move;

// ============================================================================
// Semantic Error Types
// ============================================================================

pub const SemanticError = struct {
    message: []const u8,
    line: usize,
    column: usize,
    
    pub fn init(message: []const u8, line: usize, column: usize) SemanticError {
        return SemanticError{
            .message = message,
            .line = line,
            .column = column,
        };
    }
};

// ============================================================================
// Semantic Analyzer
// ============================================================================

pub const SemanticAnalyzer = struct {
    allocator: std.mem.Allocator,
    sym_table: *SymbolTable,
    errors: std.ArrayList(SemanticError),
    current_function_return_type: ?[]const u8,
    
    pub fn init(allocator: std.mem.Allocator, sym_table: *SymbolTable) !SemanticAnalyzer {
        const errors = try std.ArrayList(SemanticError).initCapacity(allocator, 16);
        
        return SemanticAnalyzer{
            .allocator = allocator,
            .sym_table = sym_table,
            .errors = errors,
            .current_function_return_type = null,
        };
    }
    
    pub fn deinit(self: *SemanticAnalyzer) void {
        // symbol_table is owned externally
        self.errors.deinit(self.allocator);
    }
    
    pub fn hasErrors(self: *SemanticAnalyzer) bool {
        return self.errors.items.len > 0;
    }
    
    fn addError(self: *SemanticAnalyzer, message: []const u8, line: usize) !void {
        try self.errors.append(self.allocator, SemanticError.init(message, line, 0));
    }
    
    // ========================================================================
    // Declaration Analysis
    // ========================================================================
    
    pub fn analyzeDeclaration(self: *SemanticAnalyzer, decl: ast.Decl) !void {
        switch (decl) {
            .function => |f| try self.analyzeFunctionDecl(f),
            .struct_decl => |s| try self.analyzeStructDecl(s),
            .trait_decl => |t| try self.analyzeTraitDecl(t),
            .impl_decl => |i| try self.analyzeImplDecl(i),
        }
    }
    
    fn analyzeFunctionDecl(self: *SemanticAnalyzer, func: ast.FunctionDecl) !void {
        // Check if function name already exists
        if (self.sym_table.lookupLocal(func.name)) |_| {
            try self.addError("Function already defined", func.name_token.line);
            return;
        }
        
        // Register function in sym_table
        const return_type_name = if (func.return_type) |rt| rt.name else null;
        try self.sym_table.define(Symbol.init(
            func.name,
            .function,
            return_type_name,
            false,
            self.sym_table.current_scope.level,
        ));
        
        // Enter function scope
        try self.sym_table.enterScope();
        defer self.sym_table.exitScope() catch {};
        
        // Set current function return type for return statement validation
        self.current_function_return_type = return_type_name;
        defer self.current_function_return_type = null;
        
        // Register parameters
        for (func.parameters) |param| {
            // Check parameter type exists
            if (!self.sym_table.isTypeDefined(param.type_annotation.name)) {
                try self.addError("Unknown type", param.type_annotation.token.line);
            }
            
            try self.sym_table.define(Symbol.init(
                param.name,
                .parameter,
                param.type_annotation.name,
                false,
                self.sym_table.current_scope.level,
            ));
        }
        
        // Analyze function body
        try self.analyzeBlockStmt(func.body);
    }
    
    fn analyzeStructDecl(self: *SemanticAnalyzer, struct_decl: ast.StructDecl) !void {
        // Check if struct name already exists
        if (self.sym_table.lookupLocal(struct_decl.name)) |_| {
            try self.addError("Struct already defined", struct_decl.name_token.line);
            return;
        }
        
        // Register struct as a type
        try self.sym_table.define(Symbol.init(
            struct_decl.name,
            .struct_type,
            null,
            false,
            self.sym_table.current_scope.level,
        ));
        
        // Check field types exist
        for (struct_decl.fields) |field| {
            if (!self.sym_table.isTypeDefined(field.type_annotation.name)) {
                try self.addError("Unknown type in field", field.name_token.line);
            }
        }
    }
    
    fn analyzeTraitDecl(self: *SemanticAnalyzer, trait_decl: ast.TraitDecl) !void {
        // Register trait as a type
        try self.sym_table.define(Symbol.init(
            trait_decl.name,
            .trait_type,
            null,
            false,
            self.sym_table.current_scope.level,
        ));
    }
    
    fn analyzeImplDecl(self: *SemanticAnalyzer, impl_decl: ast.ImplDecl) !void {
        // Check trait and type exist
        if (!self.sym_table.isTypeDefined(impl_decl.trait_name)) {
            try self.addError("Unknown trait", impl_decl.token.line);
        }
        if (!self.sym_table.isTypeDefined(impl_decl.type_name)) {
            try self.addError("Unknown type", impl_decl.token.line);
        }
        
        // Analyze methods
        for (impl_decl.methods) |method| {
            try self.analyzeFunctionDecl(method);
        }
    }
    
    // ========================================================================
    // Statement Analysis
    // ========================================================================
    
    pub fn analyzeStatement(self: *SemanticAnalyzer, stmt: ast.Stmt) anyerror!void {
        switch (stmt) {
            .expr => |s| { _ = try self.analyzeExpr(s.expression, .Read); },
            .var_decl => |s| try self.analyzeVarDecl(s),
            .let_decl => |s| try self.analyzeLetDecl(s),
            .if_stmt => |s| try self.analyzeIfStmt(s),
            .while_stmt => |s| try self.analyzeWhileStmt(s),
            .for_stmt => |s| try self.analyzeForStmt(s),
            .return_stmt => |s| try self.analyzeReturnStmt(s),
            .block => |s| try self.analyzeBlockStmt(s),
        }
    }
    
    fn analyzeVarDecl(self: *SemanticAnalyzer, var_decl: ast.VarDeclStmt) !void {
        // Check if variable already exists in current scope
        if (self.sym_table.lookupLocal(var_decl.name)) |_| {
            try self.addError("Variable already defined", var_decl.name_token.line);
            return;
        }
        
        // Check type annotation if present
        var type_name: ?[]const u8 = null;
        if (var_decl.type_annotation) |type_ref| {
            if (!self.sym_table.isTypeDefined(type_ref.name)) {
                try self.addError("Unknown type", type_ref.token.line);
            }
            type_name = type_ref.name;
        }
        
        // Analyze initializer if present
        if (var_decl.initializer) |initializer| {
            const init_type = try self.analyzeExpr(initializer, .Read);
            
            // Type check if we have both type annotation and initializer type
            if (type_name != null and init_type != null) {
                if (!std.mem.eql(u8, type_name.?, init_type.?)) {
                    try self.addError("Type mismatch in initialization", var_decl.name_token.line);
                }
            }
            
            // Infer type from initializer if no annotation
            if (type_name == null) {
                type_name = init_type;
            }
        }
        
        // Register variable
        try self.sym_table.define(Symbol.init(
            var_decl.name,
            .variable,
            type_name,
            true, // var is mutable
            self.sym_table.current_scope.level,
        ));
    }
    
    fn analyzeLetDecl(self: *SemanticAnalyzer, let_decl: ast.LetDeclStmt) !void {
        // Check if variable already exists in current scope
        if (self.sym_table.lookupLocal(let_decl.name)) |_| {
            try self.addError("Variable already defined", let_decl.name_token.line);
            return;
        }
        
        // Analyze initializer
        const init_type = try self.analyzeExpr(let_decl.initializer, .Read);
        
        // Determine type
        var type_name: ?[]const u8 = null;
        if (let_decl.type_annotation) |type_ref| {
            if (!self.sym_table.isTypeDefined(type_ref.name)) {
                try self.addError("Unknown type", type_ref.token.line);
            }
            type_name = type_ref.name;
            
            // Type check
            if (init_type != null and !std.mem.eql(u8, type_name.?, init_type.?)) {
                try self.addError("Type mismatch in initialization", let_decl.name_token.line);
            }
        } else {
            type_name = init_type;
        }
        
        // Register variable
        try self.sym_table.define(Symbol.init(
            let_decl.name,
            .variable,
            type_name,
            false, // let is immutable
            self.sym_table.current_scope.level,
        ));
    }
    
    fn analyzeIfStmt(self: *SemanticAnalyzer, if_stmt: ast.IfStmt) !void {
        // Analyze condition
        const cond_type = try self.analyzeExpr(if_stmt.condition, .Read);
        if (cond_type) |t| {
            if (!std.mem.eql(u8, t, "Bool")) {
                try self.addError("If condition must be Bool", if_stmt.token.line);
            }
        }
        
        // Analyze branches
        try self.analyzeStatement(if_stmt.then_branch.*);
        if (if_stmt.else_branch) |else_branch| {
            try self.analyzeStatement(else_branch.*);
        }
    }
    
    fn analyzeWhileStmt(self: *SemanticAnalyzer, while_stmt: ast.WhileStmt) !void {
        // Analyze condition
        const cond_type = try self.analyzeExpr(while_stmt.condition, .Read);
        if (cond_type) |t| {
            if (!std.mem.eql(u8, t, "Bool")) {
                try self.addError("While condition must be Bool", while_stmt.token.line);
            }
        }
        
        // Analyze body
        try self.analyzeStatement(while_stmt.body.*);
    }
    
    fn analyzeForStmt(self: *SemanticAnalyzer, for_stmt: ast.ForStmt) !void {
        // Enter new scope for loop variable
        try self.sym_table.enterScope();
        defer self.sym_table.exitScope() catch {};
        
        // Register loop variable (type inference from iterable)
        try self.sym_table.define(Symbol.init(
            for_stmt.variable,
            .variable,
            null, // Type would be inferred from iterable
            false,
            self.sym_table.current_scope.level,
        ));
        
        // Analyze iterable
        _ = try self.analyzeExpr(for_stmt.iterable, .Read);
        
        // Analyze body
        try self.analyzeStatement(for_stmt.body.*);
    }
    
    fn analyzeReturnStmt(self: *SemanticAnalyzer, return_stmt: ast.ReturnStmt) !void {
        if (return_stmt.value) |val| {
            const return_type = try self.analyzeExpr(val, .Read);
            
            // Check against function return type
            if (self.current_function_return_type) |expected| {
                if (return_type) |actual| {
                    if (!std.mem.eql(u8, expected, actual)) {
                        try self.addError("Return type mismatch", return_stmt.token.line);
                    }
                }
            }
        } else {
            // Returning void
            if (self.current_function_return_type) |_| {
                try self.addError("Function must return a value", return_stmt.token.line);
            }
        }
    }
    
    fn analyzeBlockStmt(self: *SemanticAnalyzer, block: ast.BlockStmt) !void {
        try self.sym_table.enterScope();
        defer self.sym_table.exitScope() catch {};
        
        for (block.statements) |stmt| {
            try self.analyzeStatement(stmt);
        }
    }
    
    // ========================================================================
    // Expression Analysis (returns inferred type)
    // ========================================================================
    
    pub const AccessMode = enum {
        Read,
        Write,
    };
    
    fn analyzeExpr(self: *SemanticAnalyzer, expr: ast.Expr, mode: AccessMode) anyerror!?[]const u8 {
        return switch (expr) {
            .literal => |lit| self.analyzeLiteral(lit),
            .identifier => |id| try self.analyzeIdentifier(id, mode),
            .binary => |bin| try self.analyzeBinary(bin),
            .unary => |un| try self.analyzeUnary(un),
            .call => |call| try self.analyzeCall(call),
            .index => |idx| try self.analyzeIndex(idx, mode),
            .member => |mem| try self.analyzeMember(mem, mode),
            .grouping => |grp| try self.analyzeExpr(grp.expression.*, mode),
            .assignment => |assign| try self.analyzeAssignment(assign),
        };
    }
    
    fn analyzeLiteral(self: *SemanticAnalyzer, literal: ast.LiteralExpr) ?[]const u8 {
        _ = self;
        return switch (literal.value) {
            .integer => "Int",
            .float => "Float",
            .string => "String",
            .boolean => "Bool",
            .nil => null,
        };
    }
    
    fn analyzeIdentifier(self: *SemanticAnalyzer, identifier: ast.IdentifierExpr, mode: AccessMode) !?[]const u8 {
        if (self.sym_table.lookup(identifier.name)) |symbol| {
            // Check mutability for writes
            if (mode == .Write and !symbol.is_mutable) {
                try self.addError("Cannot assign to immutable variable", identifier.token.line);
            }
            return symbol.type_name;
        }
        
        try self.addError("Undefined variable", identifier.token.line);
        return null;
    }
    
    fn analyzeBinary(self: *SemanticAnalyzer, binary: ast.BinaryExpr) !?[]const u8 {
        const left_type = try self.analyzeExpr(binary.left.*, .Read);
        const right_type = try self.analyzeExpr(binary.right.*, .Read);
        
        // Type checking for binary operators
        return switch (binary.operator) {
            .add, .subtract, .multiply, .divide, .modulo, .power => {
                // Arithmetic operators
                if (left_type != null and right_type != null) {
                    if (std.mem.eql(u8, left_type.?, right_type.?)) {
                        return left_type;
                    }
                    try self.addError("Type mismatch in arithmetic", binary.operator_token.line);
                }
                return left_type;
            },
            .equal, .not_equal, .less, .less_equal, .greater, .greater_equal => {
                // Comparison operators return Bool
                return "Bool";
            },
            .logical_and, .logical_or => {
                // Logical operators expect Bool
                if (left_type) |lt| {
                    if (!std.mem.eql(u8, lt, "Bool")) {
                        try self.addError("Logical operator requires Bool", binary.operator_token.line);
                    }
                }
                return "Bool";
            },
            else => left_type,
        };
    }
    
    fn analyzeUnary(self: *SemanticAnalyzer, unary: ast.UnaryExpr) !?[]const u8 {
        const operand_type = try self.analyzeExpr(unary.operand.*, .Read);
        
        return switch (unary.operator) {
            .negate => operand_type, // Returns same type as operand
            .logical_not => "Bool",
            .bitwise_not => operand_type,
        };
    }
    
    fn analyzeCall(self: *SemanticAnalyzer, call: ast.CallExpr) !?[]const u8 {
        // Get callee type
        const callee_type = try self.analyzeExpr(call.callee.*, .Read);
        
        // Analyze arguments
        for (call.arguments) |arg| {
            _ = try self.analyzeExpr(arg, .Read);
        }
        
        // For now, return callee type (would need function signature info for better checking)
        return callee_type;
    }
    
    fn analyzeIndex(self: *SemanticAnalyzer, index: ast.IndexExpr, mode: AccessMode) !?[]const u8 {
        _ = try self.analyzeExpr(index.object.*, mode);
        const index_type = try self.analyzeExpr(index.index.*, .Read);
        
        // Check index is Int
        if (index_type) |it| {
            if (!std.mem.eql(u8, it, "Int")) {
                try self.addError("Index must be Int", index.token.line);
            }
        }
        
        // Would need to extract element type from container
        return null;
    }
    
    fn analyzeMember(self: *SemanticAnalyzer, member: ast.MemberExpr, mode: AccessMode) !?[]const u8 {
        _ = try self.analyzeExpr(member.object.*, mode);
        
        // Would need struct/type info to resolve member type
        return null;
    }
    
    fn analyzeAssignment(self: *SemanticAnalyzer, assignment: ast.AssignmentExpr) !?[]const u8 {
        // Analyze target as lvalue (Write mode)
        const target_type = try self.analyzeExpr(assignment.target.*, .Write);
        
        // Analyze value as rvalue (Read mode)
        const value_type = try self.analyzeExpr(assignment.value.*, .Read);
        
        // Type check: target and value must be compatible
        if (target_type != null and value_type != null) {
            if (!std.mem.eql(u8, target_type.?, value_type.?)) {
                try self.addError("Type mismatch in assignment", assignment.token.line);
            }
        }
        
        // Assignment expression returns the type of the assigned value
        return value_type;
    }
};

// ============================================================================
// Tests
// ============================================================================

test "semantic: variable definition" {
    var analyzer = try SemanticAnalyzer.init(std.testing.allocator);
    defer analyzer.deinit();
    
    const var_decl = ast.VarDeclStmt{
        .name = "x",
        .name_token = .{ .type = .identifier, .lexeme = "x", .line = 1, .column = 0 },
        .type_annotation = ast.TypeRef.init("Int", .{ .type = .int_type, .lexeme = "Int", .line = 1, .column = 0 }),
        .initializer = null,
    };
    
    try analyzer.analyzeVarDecl(var_decl);
    
    try std.testing.expect(!analyzer.hasErrors());
    const symbol = analyzer.sym_table.lookup("x");
    try std.testing.expect(symbol != null);
    try std.testing.expectEqualStrings("Int", symbol.?.type_name.?);
}

test "semantic: undefined type error" {
    var analyzer = try SemanticAnalyzer.init(std.testing.allocator);
    defer analyzer.deinit();
    
    const var_decl = ast.VarDeclStmt{
        .name = "x",
        .name_token = .{ .type = .identifier, .lexeme = "x", .line = 1, .column = 0 },
        .type_annotation = ast.TypeRef.init("UnknownType", .{ .type = .identifier, .lexeme = "UnknownType", .line = 1, .column = 0 }),
        .initializer = null,
    };
    
    try analyzer.analyzeVarDecl(var_decl);
    
    try std.testing.expect(analyzer.hasErrors());
}

test "semantic: duplicate variable error" {
    var analyzer = try SemanticAnalyzer.init(std.testing.allocator);
    defer analyzer.deinit();
    
    const var_decl1 = ast.VarDeclStmt{
        .name = "x",
        .name_token = .{ .type = .identifier, .lexeme = "x", .line = 1, .column = 0 },
        .type_annotation = ast.TypeRef.init("Int", .{ .type = .int_type, .lexeme = "Int", .line = 1, .column = 0 }),
        .initializer = null,
    };
    
    try analyzer.analyzeVarDecl(var_decl1);
    try std.testing.expect(!analyzer.hasErrors());
    
    // Try to define again
    try analyzer.analyzeVarDecl(var_decl1);
    try std.testing.expect(analyzer.hasErrors());
}
