// Mojo SDK - IR Builder
// Day 8: AST to IR transformation

const std = @import("std");
const ast = @import("ast");
const ir = @import("ir");
const symbol_table = @import("symbol_table");

const Type = ir.Type;
const Value = ir.Value;
const Instruction = ir.Instruction;
const BasicBlock = ir.BasicBlock;
const Function = ir.Function;
const Module = ir.Module;

// ============================================================================
// IR Builder - Converts AST to IR
// ============================================================================

pub const IRBuilder = struct {
    allocator: std.mem.Allocator,
    module: Module,
    current_function: ?*Function,
    current_block: ?*BasicBlock,
    value_map: std.StringHashMap(Value),
    
    pub fn init(allocator: std.mem.Allocator, module_name: []const u8) IRBuilder {
        return IRBuilder{
            .allocator = allocator,
            .module = Module.init(allocator, module_name),
            .current_function = null,
            .current_block = null,
            .value_map = std.StringHashMap(Value).init(allocator),
        };
    }
    
    pub fn deinit(self: *IRBuilder) void {
        self.module.deinit();
        self.value_map.deinit();
    }
    
    // ========================================================================
    // Type Mapping
    // ========================================================================
    
    fn mapType(type_name: []const u8) Type {
        if (std.mem.eql(u8, type_name, "Int")) return .i64;
        if (std.mem.eql(u8, type_name, "Float")) return .f64;
        if (std.mem.eql(u8, type_name, "Bool")) return .bool_type;
        if (std.mem.eql(u8, type_name, "String")) return .ptr; // String is a pointer
        return .i64; // Default
    }
    
    // ========================================================================
    // Declaration Generation
    // ========================================================================
    
    pub fn generateDeclaration(self: *IRBuilder, decl: ast.Decl) !void {
        switch (decl) {
            .function => |f| try self.generateFunction(f),
            else => {}, // Structs, traits handled separately
        }
    }
    
    fn generateFunction(self: *IRBuilder, func_decl: ast.FunctionDecl) !void {
        // Map return type
        const return_type = if (func_decl.return_type) |rt|
            mapType(rt.name)
        else
            .void_type;
        
        // Create parameters
        var params = try std.ArrayList(Function.Parameter).initCapacity(self.allocator, 4);
        defer params.deinit(self.allocator);
        
        for (func_decl.parameters, 0..) |param, i| {
            const param_type = mapType(param.type_annotation.name);
            const param_reg = Value.Register{
                .id = i,
                .type = param_type,
                .name = param.name,
            };
            
            try params.append(self.allocator, Function.Parameter{
                .name = param.name,
                .type = param_type,
                .register = param_reg,
            });
            
            // Map parameter name to register
            try self.value_map.put(param.name, .{ .register = param_reg });
        }
        
        // Create function
        var func = try Function.init(
            self.allocator,
            func_decl.name,
            return_type,
            params.items,
        );
        
        // Set current function and block
        self.current_function = &func;
        self.current_block = func.entry_block;
        
        // Generate function body
        try self.generateBlock(func_decl.body);
        
        // Add function to module
        try self.module.addFunction(func);
        
        // Reset context
        self.current_function = null;
        self.current_block = null;
        self.value_map.clearRetainingCapacity();
    }
    
    // ========================================================================
    // Statement Generation
    // ========================================================================
    
    fn generateBlock(self: *IRBuilder, block: ast.BlockStmt) !void {
        for (block.statements) |stmt| {
            try self.generateStatement(stmt);
        }
    }
    
    fn generateStatement(self: *IRBuilder, stmt: ast.Stmt) anyerror!void {
        switch (stmt) {
            .var_decl => |s| try self.generateVarDecl(s),
            .let_decl => |s| try self.generateLetDecl(s),
            .return_stmt => |s| try self.generateReturn(s),
            .if_stmt => |s| try self.generateIf(s),
            .while_stmt => |s| try self.generateWhile(s),
            .expr => |s| _ = try self.generateExpr(s.expression),
            .block => |s| try self.generateBlock(s),
            else => {},
        }
    }
    
    fn generateVarDecl(self: *IRBuilder, var_decl: ast.VarDeclStmt) !void {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        // Allocate stack space
        const var_type = if (var_decl.type_annotation) |ta|
            mapType(ta.name)
        else
            .i64; // Default type
        
        const alloca_reg = func.allocateRegister(.ptr, var_decl.name);
        const alloca_inst = Instruction{
            .alloca = .{
                .result = alloca_reg,
                .type = var_type,
                .name = var_decl.name,
            },
        };
        try block.addInstruction(self.allocator, alloca_inst);
        
        // Store initial value if present
        if (var_decl.initializer) |initializer| {
            const init_value = try self.generateExpr(initializer);
            const store_inst = Instruction{
                .store = .{
                    .value = init_value,
                    .ptr = .{ .register = alloca_reg },
                },
            };
            try block.addInstruction(self.allocator, store_inst);
        }
        
        // Map variable name to pointer
        try self.value_map.put(var_decl.name, .{ .register = alloca_reg });
    }
    
    fn generateLetDecl(self: *IRBuilder, let_decl: ast.LetDeclStmt) !void {
        _ = self.current_function.?;
        _ = self.current_block.?;
        
        // For let, we can optimize by using the value directly (immutable)
        const init_value = try self.generateExpr(let_decl.initializer);
        
        // Map variable name to value
        try self.value_map.put(let_decl.name, init_value);
    }
    
    fn generateReturn(self: *IRBuilder, return_stmt: ast.ReturnStmt) !void {
        const block = self.current_block.?;
        
        const ret_value = if (return_stmt.value) |val|
            try self.generateExpr(val)
        else
            null;
        
        const ret_inst = Instruction{
            .ret = .{ .value = ret_value },
        };
        try block.addInstruction(self.allocator, ret_inst);
    }
    
    fn generateIf(self: *IRBuilder, if_stmt: ast.IfStmt) !void {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        // Generate condition
        const cond_value = try self.generateExpr(if_stmt.condition);
        
        // Create blocks
        const then_block = try func.createBasicBlock("then");
        const else_block = if (if_stmt.else_branch != null)
            try func.createBasicBlock("else")
        else
            try func.createBasicBlock("merge");
        const merge_block = try func.createBasicBlock("merge");
        
        // Conditional branch
        const cond_br = Instruction{
            .cond_br = .{
                .condition = cond_value,
                .true_block = then_block,
                .false_block = if (if_stmt.else_branch != null) else_block else merge_block,
            },
        };
        try block.addInstruction(self.allocator, cond_br);
        
        // Generate then branch
        self.current_block = then_block;
        try self.generateStatement(if_stmt.then_branch.*);
        const br_merge = Instruction{ .br = .{ .target = merge_block } };
        try then_block.addInstruction(self.allocator, br_merge);
        
        // Generate else branch if present
        if (if_stmt.else_branch) |else_branch| {
            self.current_block = else_block;
            try self.generateStatement(else_branch.*);
            const br_merge2 = Instruction{ .br = .{ .target = merge_block } };
            try else_block.addInstruction(self.allocator, br_merge2);
        }
        
        // Continue in merge block
        self.current_block = merge_block;
    }
    
    fn generateWhile(self: *IRBuilder, while_stmt: ast.WhileStmt) !void {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        // Create blocks
        const cond_block = try func.createBasicBlock("while_cond");
        const body_block = try func.createBasicBlock("while_body");
        const exit_block = try func.createBasicBlock("while_exit");
        
        // Branch to condition
        const br_cond = Instruction{ .br = .{ .target = cond_block } };
        try block.addInstruction(self.allocator, br_cond);
        
        // Generate condition
        self.current_block = cond_block;
        const cond_value = try self.generateExpr(while_stmt.condition);
        const cond_br = Instruction{
            .cond_br = .{
                .condition = cond_value,
                .true_block = body_block,
                .false_block = exit_block,
            },
        };
        try cond_block.addInstruction(self.allocator, cond_br);
        
        // Generate body
        self.current_block = body_block;
        try self.generateStatement(while_stmt.body.*);
        const br_back = Instruction{ .br = .{ .target = cond_block } };
        try body_block.addInstruction(self.allocator, br_back);
        
        // Continue in exit block
        self.current_block = exit_block;
    }
    
    // ========================================================================
    // Expression Generation
    // ========================================================================
    
    fn generateExpr(self: *IRBuilder, expr: ast.Expr) anyerror!Value {
        return switch (expr) {
            .literal => |lit| self.generateLiteral(lit),
            .identifier => |id| try self.generateIdentifier(id),
            .binary => |bin| try self.generateBinary(bin),
            .unary => |un| try self.generateUnary(un),
            .call => |call| try self.generateCall(call),
            .grouping => |grp| try self.generateExpr(grp.expression.*),
            else => Value{ .constant = .{ .value = 0, .type = .i64 } },
        };
    }
    
    fn generateLiteral(self: *IRBuilder, literal: ast.LiteralExpr) Value {
        _ = self;
        return switch (literal.value) {
            .integer => |i| Value{ .constant = .{ .value = i, .type = .i64 } },
            .float => |_| Value{ .constant = .{ .value = 0, .type = .f64 } }, // Simplified
            .boolean => |b| Value{ .constant = .{ .value = if (b) 1 else 0, .type = .bool_type } },
            .string => Value{ .constant = .{ .value = 0, .type = .ptr } }, // Simplified
            .nil => Value{ .constant = .{ .value = 0, .type = .i64 } },
        };
    }
    
    fn generateIdentifier(self: *IRBuilder, identifier: ast.IdentifierExpr) !Value {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        // Look up variable
        if (self.value_map.get(identifier.name)) |val| {
            // If it's a pointer (var), load it
            if (val.register.type == .ptr) {
                const load_reg = func.allocateRegister(.i64, null);
                const load_inst = Instruction{
                    .load = .{
                        .result = load_reg,
                        .ptr = val,
                    },
                };
                try block.addInstruction(self.allocator, load_inst);
                return Value{ .register = load_reg };
            }
            return val;
        }
        
        return Value{ .constant = .{ .value = 0, .type = .i64 } }; // Error case
    }
    
    fn generateBinary(self: *IRBuilder, binary: ast.BinaryExpr) !Value {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        const left = try self.generateExpr(binary.left.*);
        const right = try self.generateExpr(binary.right.*);
        
        // Determine result type
        const result_type = switch (left) {
            .register => |r| r.type,
            .constant => |c| c.type,
        };
        
        const result_reg = func.allocateRegister(result_type, null);
        
        // Generate instruction based on operator
        const inst = switch (binary.operator) {
            .add => Instruction{ .add = .{ .result = result_reg, .lhs = left, .rhs = right } },
            .subtract => Instruction{ .sub = .{ .result = result_reg, .lhs = left, .rhs = right } },
            .multiply => Instruction{ .mul = .{ .result = result_reg, .lhs = left, .rhs = right } },
            .divide => Instruction{ .div = .{ .result = result_reg, .lhs = left, .rhs = right } },
            .modulo => Instruction{ .mod = .{ .result = result_reg, .lhs = left, .rhs = right } },
            .equal => blk: {
                const cmp_reg = func.allocateRegister(.bool_type, null);
                break :blk Instruction{ .eq = .{ .result = cmp_reg, .lhs = left, .rhs = right } };
            },
            .less => blk: {
                const cmp_reg = func.allocateRegister(.bool_type, null);
                break :blk Instruction{ .lt = .{ .result = cmp_reg, .lhs = left, .rhs = right } };
            },
            else => Instruction{ .add = .{ .result = result_reg, .lhs = left, .rhs = right } },
        };
        
        try block.addInstruction(self.allocator, inst);
        
        return Value{ .register = result_reg };
    }
    
    fn generateUnary(self: *IRBuilder, unary: ast.UnaryExpr) !Value {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        const operand = try self.generateExpr(unary.operand.*);
        
        const operand_type = switch (operand) {
            .register => |r| r.type,
            .constant => |c| c.type,
        };
        
        const result_reg = func.allocateRegister(operand_type, null);
        
        const inst = switch (unary.operator) {
            .negate => blk: {
                // Negate: 0 - operand
                const zero = Value{ .constant = .{ .value = 0, .type = operand_type } };
                break :blk Instruction{ .sub = .{ .result = result_reg, .lhs = zero, .rhs = operand } };
            },
            .logical_not => Instruction{ .not_op = .{ .result = result_reg, .operand = operand } },
            else => Instruction{ .not_op = .{ .result = result_reg, .operand = operand } },
        };
        
        try block.addInstruction(self.allocator, inst);
        
        return Value{ .register = result_reg };
    }
    
    fn generateCall(self: *IRBuilder, call: ast.CallExpr) !Value {
        const func = self.current_function.?;
        const block = self.current_block.?;
        
        // Get function name from callee
        const func_name = switch (call.callee.*) {
            .identifier => |id| id.name,
            else => "unknown",
        };
        
        // Generate arguments
        var args = try std.ArrayList(Value).initCapacity(self.allocator, 4);
        defer args.deinit(self.allocator);
        
        for (call.arguments) |arg| {
            const arg_value = try self.generateExpr(arg);
            try args.append(self.allocator, arg_value);
        }
        
        // Assume function returns i64 (would need type info)
        const result_reg = func.allocateRegister(.i64, null);
        
        const call_inst = Instruction{
            .call = .{
                .result = result_reg,
                .function = func_name,
                .args = args.items,
            },
        };
        
        try block.addInstruction(self.allocator, call_inst);
        
        return Value{ .register = result_reg };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ir_builder: simple function" {
    var builder = IRBuilder.init(std.testing.allocator, "test");
    defer builder.deinit();
    
    // Create AST: fn add() -> Int { return 42; }
    const return_stmt = ast.Stmt{
        .return_stmt = .{
            .token = .{ .type = .return_keyword, .lexeme = "return", .line = 1, .column = 0 },
            .value = ast.Expr{
                .literal = .{
                    .value = .{ .integer = 42 },
                    .token = .{ .type = .integer_literal, .lexeme = "42", .line = 1, .column = 7 },
                },
            },
        },
    };
    
    var stmts = [_]ast.Stmt{return_stmt};
    const body = ast.BlockStmt{
        .statements = &stmts,
        .token = .{ .type = .left_brace, .lexeme = "{", .line = 1, .column = 0 },
    };
    
    const func_decl = ast.FunctionDecl{
        .name = "add",
        .name_token = .{ .type = .identifier, .lexeme = "add", .line = 1, .column = 3 },
        .parameters = &[_]ast.Parameter{},
        .return_type = ast.TypeRef.init("Int", .{ .type = .int_type, .lexeme = "Int", .line = 1, .column = 0 }),
        .body = body,
    };
    
    try builder.generateFunction(func_decl);
    
    try std.testing.expectEqual(@as(usize, 1), builder.module.functions.items.len);
    try std.testing.expectEqualStrings("add", builder.module.functions.items[0].name);
}

test "ir_builder: binary expression" {
    var builder = IRBuilder.init(std.testing.allocator, "test");
    defer builder.deinit();
    
    // Create AST: fn compute() -> Int { return 5 + 3; }
    var left_expr = ast.Expr{
        .literal = .{
            .value = .{ .integer = 5 },
            .token = .{ .type = .integer_literal, .lexeme = "5", .line = 1, .column = 0 },
        },
    };
    var right_expr = ast.Expr{
        .literal = .{
            .value = .{ .integer = 3 },
            .token = .{ .type = .integer_literal, .lexeme = "3", .line = 1, .column = 4 },
        },
    };
    const add_expr = ast.Expr{
        .binary = .{
            .left = &left_expr,
            .operator = .add,
            .operator_token = .{ .type = .plus, .lexeme = "+", .line = 1, .column = 2 },
            .right = &right_expr,
        },
    };
    
    const return_stmt = ast.Stmt{
        .return_stmt = .{
            .token = .{ .type = .return_keyword, .lexeme = "return", .line = 1, .column = 0 },
            .value = add_expr,
        },
    };
    
    var stmts2 = [_]ast.Stmt{return_stmt};
    const body = ast.BlockStmt{
        .statements = &stmts2,
        .token = .{ .type = .left_brace, .lexeme = "{", .line = 1, .column = 0 },
    };
    
    const func_decl = ast.FunctionDecl{
        .name = "compute",
        .name_token = .{ .type = .identifier, .lexeme = "compute", .line = 1, .column = 3 },
        .parameters = &[_]ast.Parameter{},
        .return_type = ast.TypeRef.init("Int", .{ .type = .int_type, .lexeme = "Int", .line = 1, .column = 0 }),
        .body = body,
    };
    
    try builder.generateFunction(func_decl);
    
    const func = &builder.module.functions.items[0];
    // Should have: add instruction + return instruction
    try std.testing.expect(func.entry_block.instructions.items.len >= 2);
}

test "ir_builder: variable declaration" {
    var builder = IRBuilder.init(std.testing.allocator, "test");
    defer builder.deinit();
    
    // Create AST: fn test() { var x: Int = 10; }
    const var_decl = ast.Stmt{
        .var_decl = .{
            .name = "x",
            .name_token = .{ .type = .identifier, .lexeme = "x", .line = 1, .column = 4 },
            .type_annotation = ast.TypeRef.init("Int", .{ .type = .int_type, .lexeme = "Int", .line = 1, .column = 7 }),
            .initializer = ast.Expr{
                .literal = .{
                    .value = .{ .integer = 10 },
                    .token = .{ .type = .integer_literal, .lexeme = "10", .line = 1, .column = 13 },
                },
            },
        },
    };
    
    var stmts3 = [_]ast.Stmt{var_decl};
    const body = ast.BlockStmt{
        .statements = &stmts3,
        .token = .{ .type = .left_brace, .lexeme = "{", .line = 1, .column = 0 },
    };
    
    const func_decl = ast.FunctionDecl{
        .name = "test",
        .name_token = .{ .type = .identifier, .lexeme = "test", .line = 1, .column = 3 },
        .parameters = &[_]ast.Parameter{},
        .return_type = null,
        .body = body,
    };
    
    try builder.generateFunction(func_decl);
    
    const func = &builder.module.functions.items[0];
    // Should have: alloca + store instructions
    try std.testing.expect(func.entry_block.instructions.items.len >= 2);
}
