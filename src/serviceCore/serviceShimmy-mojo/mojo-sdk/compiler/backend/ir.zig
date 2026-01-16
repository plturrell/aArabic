// Mojo SDK - Intermediate Representation
// Day 7: IR for code generation and optimization

const std = @import("std");

// ============================================================================
// IR Types
// ============================================================================

pub const Type = enum {
    i32,    // 32-bit integer
    i64,    // 64-bit integer
    f32,    // 32-bit float
    f64,    // 64-bit float
    bool_type,
    void_type,
    ptr,    // Pointer type
    
    pub fn toString(self: Type) []const u8 {
        return switch (self) {
            .i32 => "i32",
            .i64 => "i64",
            .f32 => "f32",
            .f64 => "f64",
            .bool_type => "i1",
            .void_type => "void",
            .ptr => "ptr",
        };
    }
};

// ============================================================================
// IR Values
// ============================================================================

pub const Value = union(enum) {
    register: Register,
    constant: Constant,
    
    pub const Register = struct {
        id: usize,
        type: Type,
        name: ?[]const u8,
    };
    
    pub const Constant = struct {
        value: i64,
        type: Type,
    };
};

// ============================================================================
// IR Instructions
// ============================================================================

pub const Instruction = union(enum) {
    // Arithmetic
    add: BinaryOp,
    sub: BinaryOp,
    mul: BinaryOp,
    div: BinaryOp,
    mod: BinaryOp,
    
    // Comparison
    eq: CompareOp,
    ne: CompareOp,
    lt: CompareOp,
    le: CompareOp,
    gt: CompareOp,
    ge: CompareOp,
    
    // Logical
    and_op: BinaryOp,
    or_op: BinaryOp,
    not_op: UnaryOp,
    
    // Memory
    alloca: AllocaOp,
    load: LoadOp,
    store: StoreOp,
    
    // Control flow
    br: BranchOp,
    cond_br: CondBranchOp,
    ret: ReturnOp,
    call: CallOp,
    
    // Phi node (for SSA form)
    phi: PhiOp,
    
    pub const BinaryOp = struct {
        result: Value.Register,
        lhs: Value,
        rhs: Value,
    };
    
    pub const CompareOp = struct {
        result: Value.Register,
        lhs: Value,
        rhs: Value,
    };
    
    pub const UnaryOp = struct {
        result: Value.Register,
        operand: Value,
    };
    
    pub const AllocaOp = struct {
        result: Value.Register,
        type: Type,
        name: ?[]const u8,
    };
    
    pub const LoadOp = struct {
        result: Value.Register,
        ptr: Value,
    };
    
    pub const StoreOp = struct {
        value: Value,
        ptr: Value,
    };
    
    pub const BranchOp = struct {
        target: *BasicBlock,
    };
    
    pub const CondBranchOp = struct {
        condition: Value,
        true_block: *BasicBlock,
        false_block: *BasicBlock,
    };
    
    pub const ReturnOp = struct {
        value: ?Value,
    };
    
    pub const CallOp = struct {
        result: ?Value.Register,
        function: []const u8,
        args: []Value,
    };
    
    pub const PhiOp = struct {
        result: Value.Register,
        incoming: []PhiIncoming,
    };
    
    pub const PhiIncoming = struct {
        value: Value,
        block: *BasicBlock,
    };
};

// ============================================================================
// Basic Block
// ============================================================================

pub const BasicBlock = struct {
    label: []const u8,
    instructions: std.ArrayList(Instruction),
    predecessors: std.ArrayList(*BasicBlock),
    successors: std.ArrayList(*BasicBlock),
    
    pub fn init(allocator: std.mem.Allocator, label: []const u8) !BasicBlock {
        return BasicBlock{
            .label = label,
            .instructions = try std.ArrayList(Instruction).initCapacity(allocator, 8),
            .predecessors = try std.ArrayList(*BasicBlock).initCapacity(allocator, 2),
            .successors = try std.ArrayList(*BasicBlock).initCapacity(allocator, 2),
        };
    }
    
    pub fn deinit(self: *BasicBlock, allocator: std.mem.Allocator) void {
        self.instructions.deinit(allocator);
        self.predecessors.deinit(allocator);
        self.successors.deinit(allocator);
    }
    
    pub fn addInstruction(self: *BasicBlock, allocator: std.mem.Allocator, inst: Instruction) !void {
        try self.instructions.append(allocator, inst);
    }
    
    pub fn addPredecessor(self: *BasicBlock, allocator: std.mem.Allocator, pred: *BasicBlock) !void {
        try self.predecessors.append(allocator, pred);
    }
    
    pub fn addSuccessor(self: *BasicBlock, allocator: std.mem.Allocator, succ: *BasicBlock) !void {
        try self.successors.append(allocator, succ);
    }
};

// ============================================================================
// Function
// ============================================================================

pub const Function = struct {
    name: []const u8,
    return_type: Type,
    parameters: []Parameter,
    blocks: std.ArrayList(*BasicBlock),
    entry_block: *BasicBlock,
    next_register: usize,
    allocator: std.mem.Allocator,
    
    pub const Parameter = struct {
        name: []const u8,
        type: Type,
        register: Value.Register,
    };
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8, return_type: Type, parameters: []Parameter) !Function {
        var blocks = try std.ArrayList(*BasicBlock).initCapacity(allocator, 4);
        
        // Create entry block
        const entry = try allocator.create(BasicBlock);
        entry.* = try BasicBlock.init(allocator, "entry");
        try blocks.append(allocator, entry);
        
        return Function{
            .name = name,
            .return_type = return_type,
            .parameters = parameters,
            .blocks = blocks,
            .entry_block = entry,
            .next_register = parameters.len,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Function) void {
        for (self.blocks.items) |block| {
            block.deinit(self.allocator);
            self.allocator.destroy(block);
        }
        self.blocks.deinit(self.allocator);
    }
    
    pub fn createBasicBlock(self: *Function, label: []const u8) !*BasicBlock {
        const block = try self.allocator.create(BasicBlock);
        block.* = try BasicBlock.init(self.allocator, label);
        try self.blocks.append(self.allocator, block);
        return block;
    }
    
    pub fn allocateRegister(self: *Function, reg_type: Type, name: ?[]const u8) Value.Register {
        const reg = Value.Register{
            .id = self.next_register,
            .type = reg_type,
            .name = name,
        };
        self.next_register += 1;
        return reg;
    }
};

// ============================================================================
// Module
// ============================================================================

pub const Module = struct {
    name: []const u8,
    functions: std.ArrayList(Function),
    allocator: std.mem.Allocator,
    
    pub fn init(allocator: std.mem.Allocator, name: []const u8) Module {
        const functions = std.ArrayList(Function).initCapacity(allocator, 4) catch unreachable;
        
        return Module{
            .name = name,
            .functions = functions,
            .allocator = allocator,
        };
    }
    
    pub fn deinit(self: *Module) void {
        for (self.functions.items) |*func| {
            func.deinit();
        }
        self.functions.deinit(self.allocator);
    }
    
    pub fn addFunction(self: *Module, func: Function) !void {
        try self.functions.append(self.allocator, func);
    }
    
    // Print module in human-readable format
    pub fn print(self: *Module, writer: anytype) !void {
        try writer.print("; Module: {s}\n\n", .{self.name});
        
        for (self.functions.items) |func| {
            try writer.print("define {s} @{s}(", .{ func.return_type.toString(), func.name });
            
            for (func.parameters, 0..) |param, i| {
                if (i > 0) try writer.writeAll(", ");
                try writer.print("{s} %{s}", .{ param.type.toString(), param.name });
            }
            
            try writer.writeAll(") {\n");
            
            for (func.blocks.items) |block| {
                try writer.print("{s}:\n", .{block.label});
                
                for (block.instructions.items) |inst| {
                    try writer.writeAll("  ");
                    try printInstruction(inst, writer);
                    try writer.writeAll("\n");
                }
            }
            
            try writer.writeAll("}\n\n");
        }
    }
    
    fn printInstruction(inst: Instruction, writer: anytype) !void {
        switch (inst) {
            .add => |op| try writer.print("%{} = add {s} {}, {}", .{ op.result.id, op.result.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .sub => |op| try writer.print("%{} = sub {s} {}, {}", .{ op.result.id, op.result.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .mul => |op| try writer.print("%{} = mul {s} {}, {}", .{ op.result.id, op.result.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .div => |op| try writer.print("%{} = div {s} {}, {}", .{ op.result.id, op.result.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .eq => |op| try writer.print("%{} = icmp eq {s} {}, {}", .{ op.result.id, op.lhs.register.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .lt => |op| try writer.print("%{} = icmp lt {s} {}, {}", .{ op.result.id, op.lhs.register.type.toString(), formatValue(op.lhs), formatValue(op.rhs) }),
            .alloca => |op| {
                if (op.name) |name| {
                    try writer.print("%{s} = alloca {s}", .{ name, op.type.toString() });
                } else {
                    try writer.print("%{} = alloca {s}", .{ op.result.id, op.type.toString() });
                }
            },
            .load => |op| try writer.print("%{} = load {s}, ptr {}", .{ op.result.id, op.result.type.toString(), formatValue(op.ptr) }),
            .store => |op| try writer.print("store {s} {}, ptr {}", .{ getValueType(op.value).toString(), formatValue(op.value), formatValue(op.ptr) }),
            .ret => |op| {
                if (op.value) |val| {
                    try writer.print("ret {s} {}", .{ getValueType(val).toString(), formatValue(val) });
                } else {
                    try writer.writeAll("ret void");
                }
            },
            .br => |op| try writer.print("br label %{s}", .{op.target.label}),
            .cond_br => |op| try writer.print("br i1 {}, label %{s}, label %{s}", .{ formatValue(op.condition), op.true_block.label, op.false_block.label }),
            .call => |op| {
                if (op.result) |res| {
                    try writer.print("%{} = call {s} @{s}(", .{ res.id, res.type.toString(), op.function });
                } else {
                    try writer.print("call void @{s}(", .{op.function});
                }
                for (op.args, 0..) |arg, i| {
                    if (i > 0) try writer.writeAll(", ");
                    try writer.print("{s} {}", .{ getValueType(arg).toString(), formatValue(arg) });
                }
                try writer.writeAll(")");
            },
            else => try writer.writeAll("<instruction>"),
        }
    }
    
    fn formatValue(val: Value) std.fmt.Formatter(formatValueFn) {
        return .{ .data = val };
    }
    
    fn formatValueFn(val: Value, comptime _: []const u8, _: std.fmt.FormatOptions, writer: anytype) !void {
        switch (val) {
            .register => |reg| {
                if (reg.name) |name| {
                    try writer.print("%{s}", .{name});
                } else {
                    try writer.print("%{}", .{reg.id});
                }
            },
            .constant => |c| try writer.print("{}", .{c.value}),
        }
    }
    
    fn getValueType(val: Value) Type {
        return switch (val) {
            .register => |reg| reg.type,
            .constant => |c| c.type,
        };
    }
};

// ============================================================================
// Tests
// ============================================================================

test "ir: create module and function" {
    var module = Module.init(std.testing.allocator, "test_module");
    defer module.deinit();
    
    const params = [_]Function.Parameter{};
    const func = try Function.init(std.testing.allocator, "main", .i32, &params);
    
    try module.addFunction(func);
    
    try std.testing.expectEqual(@as(usize, 1), module.functions.items.len);
    try std.testing.expectEqualStrings("main", module.functions.items[0].name);
}

test "ir: add instructions to basic block" {
    const params = [_]Function.Parameter{};
    var func = try Function.init(std.testing.allocator, "test", .void_type, &params);
    defer func.deinit();
    
    const r0 = func.allocateRegister(.i32, null);
    
    const add_inst = Instruction{
        .add = .{
            .result = r0,
            .lhs = .{ .constant = .{ .value = 5, .type = .i32 } },
            .rhs = .{ .constant = .{ .value = 3, .type = .i32 } },
        },
    };
    
    try func.entry_block.addInstruction(std.testing.allocator, add_inst);
    
    try std.testing.expectEqual(@as(usize, 1), func.entry_block.instructions.items.len);
}

test "ir: create multiple basic blocks" {
    const params = [_]Function.Parameter{};
    var func = try Function.init(std.testing.allocator, "test", .void_type, &params);
    defer func.deinit();
    
    const then_block = try func.createBasicBlock("then");
    const else_block = try func.createBasicBlock("else");
    
    try std.testing.expectEqual(@as(usize, 3), func.blocks.items.len); // entry + then + else
    try std.testing.expectEqualStrings("then", then_block.label);
    try std.testing.expectEqualStrings("else", else_block.label);
}
